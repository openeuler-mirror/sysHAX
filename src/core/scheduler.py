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
    CPU_THROUGHPUT_THRESHOLD,
    GPU_CACHE_THRESHOLD,
    TOKEN_LIMIT_MAX,
    TOKEN_LIMIT_MIN,
    TOKEN_LIMIT_MULTIPLIER,
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

        # 获取GPU和CPU指标
        gpu_metrics = self.system_monitor.get_gpu_metrics()
        cpu_metrics = self.system_monitor.get_cpu_metrics()

        # 获取GPU缓存使用率和CPU吞吐量
        gpu_cache_usage = gpu_metrics["gpu_cache_usage"] * 100  # 转换为百分比
        cpu_throughput = cpu_metrics["generation_throughput"]  # tokens/s

        # 默认决策：使用GPU，无token限制
        decision = {"device": "GPU", "token_limit": 0}

        # 检查GPU缓存使用率是否超过阈值
        if gpu_cache_usage > GPU_CACHE_THRESHOLD:
            # 如果CPU吞吐量小于阈值，停止接收新任务
            if cpu_throughput < CPU_THROUGHPUT_THRESHOLD:
                decision["device"] = None  # None表示系统繁忙
                Logger.info("系统繁忙，停止接收新任务")
            else:
                # 否则将任务传输给CPU，设置token限制
                decision["device"] = "CPU"
                token_limit = self._calculate_token_limit(cpu_throughput)
                decision["token_limit"] = token_limit
                Logger.info(
                    f"GPU缓存使用率过高({gpu_cache_usage:.1f}%)，任务转移到CPU，token限制: {token_limit}",
                )

        Logger.info(f"调度决策: {decision}")
        return decision

    @staticmethod
    def _calculate_token_limit(cpu_throughput: float) -> int:
        """计算token限制"""
        token = round(cpu_throughput * TOKEN_LIMIT_MULTIPLIER / 10) * 10
        return min(max(token, TOKEN_LIMIT_MIN), TOKEN_LIMIT_MAX)
