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
from src.core.metrics import MetricsService
from src.utils.config import (
    CPU_MAX_BATCH_SIZE
)
from src.utils.logger import Logger

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
                 metrics_service: MetricsService) -> None:
        """
        初始化调度决策器
        """
        self.system_monitor: SystemMonitor = system_monitor
        self.runner: Runner = runner
        self.metrics_service: MetricsService = metrics_service
    
        self.cpu_max_batch = 256
        self.gpu_max_batch = 256
        self.waiting : asyncio.Queue = asyncio.Queue()
        self.cpu_running_num: int = 0
        self.gpu_running_num: int = 0

    async def submit_task(self, data: dict[str, Any]) -> None:
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

    def scheduler(self) -> dict[str, int]:
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

            if decision["device"] == "GPU" and self.gpu_running_num < self.gpu_max_batch:
                scheduled["GPU"] += 1
                asyncio.create_task(self._execute_task(decision["device"], task_data))
                Logger.debug(f"任务分配到GPU执行")
            elif decision["device"] == "CPU" and self.cpu_running_num < self.cpu_max_batch:
                scheduled["CPU"] += 1
                asyncio.create_task(self._execute_task(decision["device"], task_data))
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

        if device == "GPU":
            self.gpu_running_num += 1
            self.metrics_service.set_gpu_running_num(self.gpu_running_num)
        elif device == "CPU":
            self.cpu_running_num += 1
            self.metrics_service.set_cpu_running_num(self.cpu_running_num)
        try:
            async for chunk in self.runner.task_handler(device=device, data=request):
                await output_queue.put(chunk)
        except Exception as e:
            Logger.error(f"{device}任务执行失败: {e}", exc_info=True)
            await output_queue.put(b"[DONE]")
        finally:
            if device == "GPU":
                self.gpu_running_num -= 1
                self.metrics_service.set_gpu_running_num(self.gpu_running_num)
            elif device == "CPU":
                self.cpu_running_num -= 1
                self.metrics_service.set_cpu_running_num(self.cpu_running_num)

    def _make_decision(self) -> dict:

        """
        做出调度决策，返回设备类型和token限制

        Returns:
            decision: {
                "device": 设备类型,None表示系统繁忙
                "token_limit": token限制,0表示不限制
            }
        """
        # 更新系统指标
        try:
            self.system_monitor.get_gpu_monitor()
            self.system_monitor.get_cpu_monitor()
        except Exception as e:
            Logger.error(f"更新系统指标失败: {e}", exc_info=True)

        # 是否将任务转移到CPU
        log_msg = ""
        use_cpu = False
        gpu_decode_throughout_per_batch = self.metrics_service.gpu_decode_throughout / self.metrics_service.gpu_num_running if self.metrics_service.gpu_num_running > 0 else 0
        cpu_decode_throughout_per_batch = self.metrics_service.cpu_decode_throughout / self.metrics_service.cpu_num_running if self.metrics_service.cpu_num_running > 0 else 0
        if self.metrics_service.gpu_num_running == 0:
            use_cpu = False
            log_msg = "gpu_num_running为0，优先向GPU发任务"
        elif self.metrics_service.cpu_num_running == 0:
            use_cpu = True
            log_msg = f"gpu_num_running为{self.metrics_service.gpu_num_running}，gpu_num_running为0，优先向CPU发任务"
        elif gpu_decode_throughout_per_batch > cpu_decode_throughout_per_batch:
            use_cpu = False
            log_msg = f"GPU平均吞吐量{gpu_decode_throughout_per_batch:.2f}tokens/s，高于CPU平均吞吐量{cpu_decode_throughout_per_batch:.2f}tokens/s，优先向GPU发任务"
        elif self.metrics_service.cpu_num_running > CPU_MAX_BATCH_SIZE:
            use_cpu = False
            log_msg = f"CPU运行中请求数{self.metrics_service.cpu_num_running}，超过最大并发量{CPU_MAX_BATCH_SIZE}，优先向GPU发任务"
        else:
            use_cpu = True
            log_msg = f"CPU平均吞吐量{cpu_decode_throughout_per_batch:.2f}tokens/s，高于GPU平均吞吐量{gpu_decode_throughout_per_batch:.2f}tokens/s，优先向CPU发任务"        
        
        decision = {"device": "CPU" if use_cpu else "GPU", "token_limit": 0}
        Logger.debug(f"\033[1;32m{log_msg}\033[0m")
        Logger.debug(f"\033[1;32m调度决策: {decision}\033[0m")
        return decision

