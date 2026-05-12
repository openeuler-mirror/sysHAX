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
from src.core.runner import Runner
from src.core.metrics import MetricsService, SSE_DONE_EVENT
from src.utils.config import SyshaxConfig
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

        self.cpu_max_batch = 256
        self.gpu_max_batch = 256
        self.waiting : asyncio.Queue = asyncio.Queue()
        self.cpu_running_num: int = 0
        self.gpu_running_num: int = 0
        self.gpu_scheduled_running_num: int = 0

        self._running_tasks: set[asyncio.Task] = set()

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
            if self.gpu_running_num >= self.gpu_max_batch and \
               self.cpu_running_num >= self.cpu_max_batch:
                break
            try:
                task_data = self.waiting.get_nowait()
            except asyncio.QueueEmpty:
                break

            decision = self._make_decision()
            # 动态调度当前暂时只能接续调度到CPU侧
            if "num_decode_tokens" in task_data["input"]:
                decision["device"] = "CPU"
                Logger.debug("任务包含num_decode_tokens，强制调度到CPU")
            else:
                if self.syshax_config.auto_pd_offload and decision["device"] == "CPU":
                    # 不含有num_decode_tokens字段，说明是完整任务，首先会进行prefill任务
                    # CPU侧不适合执行prefill任务，当开启auto_pd_offload会自动进行PD解耦
                    task_data["input"]["num_decode_tokens"] = 1
                    decision["device"] = "GPU"
                    self.gpu_scheduled_running_num += 1
            if decision["device"] == "GPU" and self.gpu_running_num < self.gpu_max_batch:
                scheduled["GPU"] += 1
                self.gpu_running_num += 1
                self.metrics_service.set_gpu_running_num(self.gpu_running_num)
                task = asyncio.create_task(self._execute_task(decision["device"], task_data))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)
                Logger.debug(f"任务分配到GPU执行")
            elif decision["device"] == "CPU" and self.cpu_running_num < self.cpu_max_batch:
                Logger.debug("自动开启CPU侧prefill任务的num_decode_tokens=1以启用部分解码卸载")
                scheduled["CPU"] += 1
                self.cpu_running_num += 1
                self.metrics_service.set_cpu_running_num(self.cpu_running_num)
                task = asyncio.create_task(self._execute_task(decision["device"], task_data))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)
                Logger.debug(f"任务分配到CPU执行")
            else:
                self.waiting.put_nowait(task_data)
                scheduled["skipped"] += 1
                Logger.debug(f"任务暂无可用资源，继续等待")
                break

        self.metrics_service.set_waiting_num(self.waiting.qsize())
        return scheduled

    async def _execute_task(self, device: str, task_data: dict[str, Any]) -> None:
        request = task_data["input"]
        output_queue = task_data["output_queue"]
        is_stream = request.get("stream", False)

        # 用于传出接续任务
        resubmit_task_data = {"data": None}
        try:
            async for chunk in self.runner.task_handler(device=device, data=request, resubmit_task_data=resubmit_task_data):
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
            Logger.error(f"{device}任务执行失败: {e}", exc_info=True)
            if is_stream:
                await output_queue.put(b'data: {"error": "internal_error"}\n\n')
                await output_queue.put(SSE_DONE_EVENT)
                await output_queue.put(None)
            else:
                await output_queue.put(b"[DONE]")
        finally:
            if device == "GPU":
                self.gpu_running_num -= 1
                self.metrics_service.set_gpu_running_num(self.gpu_running_num)
                if "num_decode_tokens" in request and request["num_decode_tokens"] != 0:
                    self.gpu_scheduled_running_num -= 1
            elif device == "CPU":
                self.cpu_running_num -= 1
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
