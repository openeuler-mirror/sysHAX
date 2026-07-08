"""
Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

sysHAX is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
    http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
See the Mulan PSL v2 for more details.
Created: 2025-05-23
Desc:sysHAX 调度决策模块
"""

import time
import asyncio
from typing import Any
from src.core.monitor import SystemMonitor
from src.core.router import RequestRouter
from src.core.runner import Runner
from src.core.metrics import MetricsService, SSE_DONE_EVENT
from src.utils.config import SyshaxConfig, Worker
from src.utils.logger import Logger

SCHEDULE_DICT: dict[int, Any] = {
    100: "gpu_running_num为0，优先向GPU发任务",
    101: "CPU分配的运行中请求数{cpu_allocated}，超过最大并发量{cpu_max}，优先向GPU发任务",
    102: {
        "message": "{reason_detail}",
        "reasons": {
            "GPU_LOW_THROUGHPUT": "GPU、CPU暂时无法检测到吞吐量，动态向二者发送请求，本次向GPU发送请求",
            "GPU_HIGHER_TP": "GPU平均吞吐量{gpu_tp:.2f}tokens/s，高于CPU平均吞吐量{cpu_tp:.2f}tokens/s，优先向GPU发任务",
        },
    },
    200: "gpu_running_num为{gpu_running_num}，cpu_running_num为0，优先向CPU发任务",
    201: {
        "message": "{reason_detail}",
        "reasons": {
            "CPU_LOW_THROUGHPUT": "GPU、CPU暂时无法检测到吞吐量，动态向二者发送请求，本次向CPU发送请求",
        },
    },
    202: "CPU平均吞吐量{cpu_tp:.2f}tokens/s，高于GPU平均吞吐量{gpu_tp:.2f}tokens/s，优先向CPU发任务",
}

REASON_GPU_LOW_THROUGHPUT = "GPU_LOW_THROUGHPUT"
REASON_CPU_LOW_THROUGHPUT = "CPU_LOW_THROUGHPUT"
REASON_GPU_HIGHER_TP = "GPU_HIGHER_TP"

class Scheduler:
    """
    调度决策类，根据系统指标决定在何处执行解码任务

    职责：
    1. 根据SystemMonitor提供的指标做出调度决策
    2. 提供设备和token限制的决策
    """

    def __init__(self,
                 system_monitor: SystemMonitor,
                 runner: Runner,
                 metrics_service: MetricsService,
                 syshax_config: SyshaxConfig) -> None:
        """
        初始化调度决策器
        """
        self.system_monitor: SystemMonitor = system_monitor
        self.runner: Runner = runner
        self.metrics_service: MetricsService = metrics_service
        self.syshax_config: SyshaxConfig = syshax_config

        # 每个实例(worker)允许的最大在途请求数
        self.per_worker_max_batch = 256

        # 全量 workers（含故障节点）
        self._all_gpu_workers: list[Worker] = syshax_config.gpu_workers
        self._all_cpu_workers: list[Worker] = syshax_config.cpu_workers

        # 活跃调度池：仅包含健康节点，由 on_worker_down/on_worker_up 动态维护
        self._active_gpu_workers: set[Worker] = set(self._all_gpu_workers)
        self._active_cpu_workers: set[Worker] = set(self._all_cpu_workers)

        self.waiting: asyncio.Queue = asyncio.Queue()

        # per-worker 在途请求计数(唯一事实来源)；覆盖全量 workers，防止故障节点在途计数丢失
        self._worker_running: dict[Worker, int] = {
            worker: 0 for worker in [*self._all_gpu_workers, *self._all_cpu_workers]
        }
        # least-loaded 打平时用的 round-robin 游标
        self._rr_cursor: dict[str, int] = {"GPU": 0, "CPU": 0}
        self.gpu_scheduled_running_num: int = 0

        self._running_tasks: set[asyncio.Task] = set()

        # 请求字段路由规则引擎（无规则时退回纯指标决策）
        self._router = RequestRouter(syshax_config.routing_rules)

    @property
    def gpu_workers(self) -> list[Worker]:
        """当前活跃的 GPU Worker 列表（健康节点）"""
        return list(self._active_gpu_workers)

    @property
    def cpu_workers(self) -> list[Worker]:
        """当前活跃的 CPU Worker 列表（健康节点）"""
        return list(self._active_cpu_workers)

    @property
    def gpu_running_num(self) -> int:
        """GPU 池在途请求总数（含故障节点，防止计数不一致）"""
        return sum(self._worker_running[w] for w in self._all_gpu_workers)

    @property
    def cpu_running_num(self) -> int:
        """CPU 池在途请求总数（含故障节点，防止计数不一致）"""
        return sum(self._worker_running[w] for w in self._all_cpu_workers)

    def _device_workers(self, device: str) -> list[Worker]:
        """返回当前活跃的设备 Worker 列表（供调度使用）"""
        return self.gpu_workers if device == "GPU" else self.cpu_workers

    def on_worker_down(self, worker: Worker) -> None:
        """
        心跳探活回调：节点故障，从活跃调度池摘除。
        在途任务不受影响，_worker_running 计数正常维护直至任务结束。
        """
        removed = False
        if worker.device == "GPU" and worker in self._active_gpu_workers:
            self._active_gpu_workers.discard(worker)
            removed = True
        elif worker.device == "CPU" and worker in self._active_cpu_workers:
            self._active_cpu_workers.discard(worker)
            removed = True
        if removed:
            active_gpu = len(self._active_gpu_workers)
            active_cpu = len(self._active_cpu_workers)
            Logger.warning(
                f"[调度池] 节点已摘除: {worker.label}，"
                f"剩余活跃 GPU={active_gpu}，CPU={active_cpu}"
            )
        else:
            Logger.debug(f"[调度池] on_worker_down 调用但节点不在活跃池中: {worker.label}")

    def on_worker_up(self, worker: Worker) -> None:
        """
        心跳探活回调：节点恢复，重新加入活跃调度池。
        """
        if worker.device == "GPU" and worker in self._all_gpu_workers:
            self._active_gpu_workers.add(worker)
            Logger.info(
                f"[调度池] 节点重新加入: {worker.label}，"
                f"当前活跃 GPU={len(self._active_gpu_workers)}，CPU={len(self._active_cpu_workers)}"
            )
        elif worker.device == "CPU" and worker in self._all_cpu_workers:
            self._active_cpu_workers.add(worker)
            Logger.info(
                f"[调度池] 节点重新加入: {worker.label}，"
                f"当前活跃 GPU={len(self._active_gpu_workers)}，CPU={len(self._active_cpu_workers)}"
            )
        else:
            Logger.warning(f"[调度池] on_worker_up 收到未知节点: {worker.label}")

    def _apply_route_hint(self, hint: str) -> dict:
        """
        软强制路由建议：目标设备有容量时采纳，否则退回指标层决策。

        Args:
            hint: 规则建议的设备，"GPU" 或 "CPU"

        Returns:
            最终决策 dict，格式与 _make_decision() 一致
        """
        if self._device_has_capacity(hint):
            return {"device": hint, "token_limit": 0}
        # 建议设备无容量，软降级到指标决策
        Logger.debug(
            f"[路由规则] 建议设备 {hint} 当前无容量，退回指标层决策"
        )
        return self._make_decision()

    def _device_has_capacity(self, device: str) -> bool:
        """设备池内是否存在未达到 per_worker_max_batch 的实例"""
        return any(self._worker_running[w] < self.per_worker_max_batch for w in self._device_workers(device))

    def _select_worker(self, device: str) -> Worker:
        """在设备池内挑选 least-loaded 的可用实例，负载相同时按 round-robin 打平"""
        workers = [w for w in self._device_workers(device)
                   if self._worker_running[w] < self.per_worker_max_batch]
        min_load = min(self._worker_running[w] for w in workers)
        candidates = [w for w in workers if self._worker_running[w] == min_load]
        chosen = candidates[self._rr_cursor[device] % len(candidates)]
        self._rr_cursor[device] += 1
        return chosen

    async def submit_task(self, data: dict[str, Any]) -> asyncio.Queue:
        output_queue = asyncio.Queue()
        task_data = {
            "input": data,
            "output_queue": output_queue,
            "create_time": time.time()
        }
        await self.waiting.put(task_data)
        return output_queue

    def has_unfinshed_tasks(self) -> bool:
        return self.waiting.qsize() > 0

    def has_running_tasks(self) -> bool:
        return self.cpu_running_num > 0 or self.gpu_running_num > 0

    async def scheduler(self) -> dict[str, int]:
        scheduled = {"GPU": 0, "CPU": 0, "skipped": 0}
        while not self.waiting.empty():
            if not self._device_has_capacity("GPU") and not self._device_has_capacity("CPU"):
                break
            try:
                task_data = self.waiting.get_nowait()
            except asyncio.QueueEmpty:
                break

            decision = {"device": "GPU", "token_limit": 0}
            # ① 接续任务（含 num_decode_tokens）：强制 CPU，跳过规则层
            if "num_decode_tokens" in task_data["input"]:
                decision["device"] = "CPU"
                Logger.debug("任务包含num_decode_tokens，强制调度到CPU")
            else:
                # ② 规则层：按请求字段匹配路由规则
                route_hint = self._router.match(task_data["input"])
                if route_hint is not None:
                    decision = self._apply_route_hint(route_hint)
                else:
                    # ③ 指标层：现有吞吐量/负载决策（保持不变）
                    decision = self._make_decision()

                # ④ auto_pd_offload（现有逻辑不变）
                if self.syshax_config.auto_pd_offload and decision["device"] == "CPU":
                    # 不含有num_decode_tokens字段，说明是完整任务，首先会进行prefill任务
                    # CPU侧不适合执行prefill任务，当开启auto_pd_offload会自动进行PD解耦
                    task_data["input"]["num_decode_tokens"] = 1
                    decision["device"] = "GPU"
                    self.gpu_scheduled_running_num += 1

            device = decision["device"]
            if device == "GPU" and self._device_has_capacity("GPU"):
                worker = self._select_worker("GPU")
                scheduled["GPU"] += 1
                self._worker_running[worker] += 1
                self.metrics_service.set_gpu_running_num(self.gpu_running_num)
                task = asyncio.create_task(self._execute_task(worker, task_data))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)
                Logger.debug(f"任务分配到 {worker.label} 执行")
            elif device == "CPU" and self._device_has_capacity("CPU"):
                Logger.debug("自动开启CPU侧prefill任务的num_decode_tokens=1以启用部分解码卸载")
                worker = self._select_worker("CPU")
                scheduled["CPU"] += 1
                self._worker_running[worker] += 1
                self.metrics_service.set_cpu_running_num(self.cpu_running_num)
                task = asyncio.create_task(self._execute_task(worker, task_data))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)
                Logger.debug(f"任务分配到 {worker.label} 执行")
            else:
                self.waiting.put_nowait(task_data)
                scheduled["skipped"] += 1
                Logger.debug(f"任务暂无可用资源，继续等待")
                break

        self.metrics_service.set_waiting_num(self.waiting.qsize())
        return scheduled

    async def _execute_task(self, worker: Worker, task_data: dict[str, Any]) -> None:
        device = worker.device
        request = task_data["input"]
        output_queue = task_data["output_queue"]
        is_stream = request.get("stream", False)

        # 用于传出接续任务
        resubmit_task_data = {"data": None}
        try:
            async for chunk in self.runner.task_handler(worker=worker, data=request, resubmit_task_data=resubmit_task_data):
                await output_queue.put(chunk)

            if resubmit_task_data["data"] is not None:
                resubmit_task = {
                    "input": resubmit_task_data["data"],
                    "output_queue": output_queue,
                    "create_time": time.time()
                }
                await self.waiting.put(resubmit_task)
                Logger.debug(f"接续任务已加入调度队列: {resubmit_task_data['data'].get('request_id_inference')}")
            else:
                if is_stream:
                    await output_queue.put(None)
                else:
                    await output_queue.put(b"[DONE]")

        except Exception as e:
            Logger.error(f"{worker.label}任务执行失败: {e}", exc_info=True)
            if is_stream:
                await output_queue.put(b'data: {"error": "internal_error"}\n\n')
                await output_queue.put(SSE_DONE_EVENT)
                await output_queue.put(None)
            else:
                await output_queue.put(b"[DONE]")
        finally:
            self._worker_running[worker] -= 1
            if device == "GPU":
                self.metrics_service.set_gpu_running_num(self.gpu_running_num)
                if "num_decode_tokens" in request and request["num_decode_tokens"] != 0:
                    self.gpu_scheduled_running_num -= 1
            elif device == "CPU":
                self.metrics_service.set_cpu_running_num(self.cpu_running_num)

    def _format_schedule_message(self, code: int, **context: Any) -> str:
        """根据调度码渲染日志消息"""
        entry = SCHEDULE_DICT.get(code)
        if entry is None:
            template = "调度码{code}未定义"
        elif isinstance(entry, dict):
            template = entry.get("message", "调度码{code}未定义")
        else:
            template = entry
        try:
            return template.format(**context, code=code)
        except KeyError as exc:
            missing_key = exc.args[0]
            Logger.warning(f"调度消息缺少参数: {missing_key}, code={code}, context={context}")
            return template

    def _get_reason_detail(self, code: int, reason_key: str, **context: Any) -> str:
        """从调度字典中获取reason_detail模板"""
        entry = SCHEDULE_DICT.get(code)
        if isinstance(entry, dict):
            template = entry.get("reasons", {}).get(reason_key)
            if template:
                try:
                    return template.format(**context)
                except KeyError as exc:
                    missing_key = exc.args[0]
                    Logger.warning(
                        f"reason_detail缺少参数: {missing_key}, code={code}, reason_key={reason_key}, context={context}"
                    )
                    return template
        Logger.warning(f"未找到reason_detail: code={code}, reason_key={reason_key}")
        return ""

    def _make_decision(self) -> dict:
        """
        做出调度决策，返回设备类型和token限制

        Returns:
            decision: {
                "device": 设备类型,None表示系统繁忙
                "token_limit": token限制,0表示不限制
            }
        """
        CPU_MAX_BATCH_SIZE = self.syshax_config.cpu_max_batch_size
        # 是否将任务转移到CPU
        msg_code = None
        context: dict[str, Any] = {}
        use_cpu = False
        gpu_decode_throughout_per_batch = (
            self.metrics_service.gpu_decode_throughout / self.metrics_service.gpu_running_num
            if self.metrics_service.gpu_running_num > 0 else 0)
        cpu_decode_throughout_per_batch = (
            self.metrics_service.cpu_decode_throughout / self.metrics_service.cpu_running_num
            if self.metrics_service.cpu_running_num > 0 else 0)
        if self.gpu_running_num == 0:
            use_cpu = False
            msg_code = 100
        elif self.cpu_running_num + self.gpu_scheduled_running_num == 0:
            use_cpu = True
            msg_code = 200
            context = {"gpu_running_num": self.gpu_running_num}
        elif self.cpu_running_num + self.gpu_scheduled_running_num >= CPU_MAX_BATCH_SIZE:
            use_cpu = False
            msg_code = 101
            context = {
                "cpu_allocated": self.cpu_running_num + self.gpu_scheduled_running_num,
                "cpu_max": CPU_MAX_BATCH_SIZE
            }

        if msg_code is None:
            if gpu_decode_throughout_per_batch < 0.1 and cpu_decode_throughout_per_batch < 0.1:
                if (self.gpu_running_num - self.gpu_scheduled_running_num <
                    self.cpu_running_num + self.gpu_scheduled_running_num):
                    use_cpu = False
                    msg_code = 102
                    context = {
                        "reason_detail": self._get_reason_detail(102, REASON_GPU_LOW_THROUGHPUT)
                    }
                else:
                    use_cpu = True
                    msg_code = 201
                    context = {
                        "reason_detail": self._get_reason_detail(201, REASON_CPU_LOW_THROUGHPUT)
                    }
            elif gpu_decode_throughout_per_batch >= cpu_decode_throughout_per_batch:
                use_cpu = False
                msg_code = 102
                context = {
                    "reason_detail": self._get_reason_detail(
                        102,
                        REASON_GPU_HIGHER_TP,
                        gpu_tp=gpu_decode_throughout_per_batch,
                        cpu_tp=cpu_decode_throughout_per_batch
                    )
                }
            else:
                use_cpu = True
                msg_code = 202
                context = {
                    "gpu_tp": gpu_decode_throughout_per_batch,
                    "cpu_tp": cpu_decode_throughout_per_batch
                }

        decision = {"device": "CPU" if use_cpu else "GPU", "token_limit": 0}
        log_msg = self._format_schedule_message(msg_code or -1, **context)
        Logger.debug(f"\033[1;32m{log_msg} (code={msg_code}), 调度决策: {decision}\033[0m")
        return decision

    async def cancel_all_tasks(self):
        """取消所有正在运行的任务"""
        if not self._running_tasks:
            return
        Logger.info(f"正在取消 {len(self._running_tasks)} 个运行中的任务...")
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._running_tasks, return_exceptions=True),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            Logger.warning("部分任务未能在 2 秒内取消")
        self._running_tasks.clear()
