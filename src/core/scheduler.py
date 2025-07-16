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

from src.core.monitor import SystemMonitor
from src.utils.config import (
    GPU_KV_CACHE_THRESHOLD,
    GPU_THROUGHPUT_LOWER_BOUND,
    GPU_MAX_BATCH_SIZE,
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

    def __init__(self, system_monitor: SystemMonitor) -> None:
        """
        初始化调度决策器

        Args:
            system_monitor: 系统监控器实例

        """
        self.system_monitor: SystemMonitor = system_monitor

    async def scheduler(self) -> dict:
        """
        做出调度决策，返回设备类型和token限制

        Returns:
            decision: {
                "device": 设备类型,None表示系统繁忙
                "token_limit": token限制,0表示不限制
            }

        """
        # 更新系统指标
        await self.system_monitor.update_metrics()

        gpu_cache_usage = self.system_monitor.gpu_metrics.gpu_cache_usage * 100  # 转换为百分比
        gpu_throughput = self.system_monitor.gpu_metrics.decode_throughout # tokens/s
        cpu_throughput = self.system_monitor.cpu_metrics.decode_throughout # tokens/s
        gpu_running = self.system_monitor.gpu_metrics.num_running
        gpu_waiting = self.system_monitor.gpu_metrics.num_waiting
        gpu_swapped = self.system_monitor.gpu_metrics.num_swapped
        cpu_running = self.system_monitor.cpu_metrics.num_running

        # 是否将任务转移到CPU
        log_msg = ""
        use_cpu = False
        if GPU_THROUGHPUT_LOWER_BOUND is not None and gpu_throughput < GPU_THROUGHPUT_LOWER_BOUND and \
          gpu_throughput > 0.1:    # 校验0.1的目的是防止未进行推理时的误调度
            use_cpu = True
            log_msg = f"GPU吞吐量为{gpu_throughput:.2f}tokens/s，低于{GPU_THROUGHPUT_LOWER_BOUND:.2f}tokens/s，"
        elif GPU_KV_CACHE_THRESHOLD is not None and gpu_cache_usage > GPU_KV_CACHE_THRESHOLD:
            use_cpu = True
            log_msg = f"GPU kvcache使用率为{gpu_cache_usage:.2f}%，超过{GPU_KV_CACHE_THRESHOLD:.2f}%"
        elif GPU_MAX_BATCH_SIZE is not None and gpu_running >= GPU_MAX_BATCH_SIZE:
            use_cpu = True
            log_msg = f"GPU达到最大并发量{GPU_MAX_BATCH_SIZE}%，"
        elif gpu_swapped > 0:
            use_cpu = True
            log_msg = f"GPU侧资源不足，"

        # 如果GPU可用，则继续在GPU上执行
        if not use_cpu:
            decision = {"device": "GPU", "token_limit": 0}
            Logger.info(f"\033[1;32m调度决策: {decision}\033[0m")
            return decision

        # CPU侧调度逻辑
        if cpu_running >= CPU_MAX_BATCH_SIZE:
            decision = {"device": None, "token_limit": 0}
            log_msg += "CPU达到最大并发量，停止接收新任务。"
        else:
            Logger.info_console(f"cpu_throughput: {cpu_throughput}")
            if int(cpu_throughput) > 0:
                self.token_limit = int(cpu_throughput)
            else:
                from src.utils.config import DEFAULT_TOKEN_LIMIT
                self.token_limit = DEFAULT_TOKEN_LIMIT
            decision = {"device": "CPU", "token_limit": self.token_limit}
            log_msg += "执行PD分离。"
            from src.utils.config import DEFAULT_TOKEN_LIMIT
            Logger.info_console(f"DEFAULT_TOKEN_LIMIT: {DEFAULT_TOKEN_LIMIT}")

        Logger.info(f"\033[1;32m{log_msg}\033[0m")
        Logger.info(f"\033[1;32m调度决策: {decision}\033[0m")
        return decision

