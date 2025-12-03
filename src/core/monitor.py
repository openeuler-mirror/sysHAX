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
Desc:sysHAX 资源监控模块
"""

import re
import httpx
from typing import Pattern, Callable, Any

from src.utils.logger import Logger
from src.core.metrics import MetricsService
from src.utils.config import SyshaxConfig

# Prometheus指标正则匹配模式
# 资源使用指标
RE_GPU_CACHE = re.compile(
    r"vllm:gpu_cache_usage_perc{[^}]*}\s+([\d.]+)",
)  # GPU KV缓存使用率：值域0-1，1表示100%使用
RE_CPU_CACHE = re.compile(
    r"vllm:cpu_cache_usage_perc{[^}]*}\s+([\d.]+)",
)  # CPU KV缓存使用率：值域0-1，1表示100%使用
RE_RUNNING_REQS = re.compile(
    r"vllm:num_requests_running{[^}]*}\s+(\d+)",
)  # 运行中请求数：当前在GPU上执行的请求数量
RE_WAITING_REQS = re.compile(
    r"vllm:num_requests_waiting{[^}]*}\s+(\d+)",
)  # 等待中请求数：等待GPU资源的请求数量
RE_SWAPPED_REQS = re.compile(
    r"vllm:num_requests_swapped{[^}]*}\s+(\d+)",
)  # 已交换请求数：从GPU交换到CPU内存的请求数量

class ResourceMonitor:
    """
    资源监控类，解析单个vLLM服务的Prometheus指标

    职责：
    1. 从指定URL获取单个服务的指标
    2. 解析指标并提供简单的接口访问这些指标
    """

    def __init__(self, metrics_url: str) -> None:
        """初始化资源监控器"""
        self.metrics_url = metrics_url
        self._client = httpx.AsyncClient()

    async def close(self) -> None:
        """关闭异步 HTTP 客户端"""
        await self._client.aclose()

    async def update_metrics(self) -> dict[str, float | int]:
        """异步获取并解析 Prometheus 指标"""
        try:
            monitor_data = {
                "gpu_cache_usage": 0.0,   # GPU KV缓存使用率，百分比
                "cpu_cache_usage": 0.0,   # CPU KV缓存使用率，百分比
                "num_running": 0,         # 运行中请求数
                "num_waiting": 0,         # 等待中请求数
                "num_swapped": 0,         # 已交换请求数
            }
            # 发起异步 HTTP 请求获取指标
            response = await self._client.get(self.metrics_url, timeout=3.0)
            if response.status_code != httpx.codes.OK:
                Logger.warning(f"获取指标失败: HTTP {response.status_code}")
                return monitor_data
            monitor_text = response.text
            await response.aclose()
            # 解析指标文本
            monitor_data["gpu_cache_usage"] = self._parse_metrics(monitor_text, RE_GPU_CACHE, float)
            monitor_data["cpu_cache_usage"] = self._parse_metrics(monitor_text, RE_CPU_CACHE, float)
            monitor_data["num_running"] = self._parse_metrics(monitor_text, RE_RUNNING_REQS, int)
            monitor_data["num_waiting"] = self._parse_metrics(monitor_text, RE_WAITING_REQS, int)
            monitor_data["num_swapped"] = self._parse_metrics(monitor_text, RE_SWAPPED_REQS, int)
            return monitor_data
        except httpx.TimeoutException as e:
            Logger.warning(f"获取指标超时: {e}")
            return monitor_data
        except httpx.HTTPStatusError as e:
            Logger.error(f"Monitor错误: {e}")
            return monitor_data
        except Exception as e:
            Logger.error(f"Monitor监控失败: {e}", exc_info=True)
            return monitor_data

    def _parse_metrics(self, metrics_text: str, pattern: Pattern, converter: Callable[[str], Any]) -> Any:
        match = pattern.search(metrics_text)
        if match:
            return converter(match.group(1))
        return converter("0")

class SystemMonitor:
    """
    系统监控类，同时监控GPU和CPU服务
    """

    def __init__(self, metrics_service: MetricsService, syshax_config: SyshaxConfig) -> None:
        """初始化系统监控器：根据配置拼接 metrics URL"""
        self.config = syshax_config
        # 构建 GPU/CPU metrics URL
        self.gpu_monitor = ResourceMonitor(f"http://{syshax_config.gpu_host}:{syshax_config.gpu_port}/metrics")
        self.cpu_monitor = ResourceMonitor(f"http://{syshax_config.cpu_host}:{syshax_config.cpu_port}/metrics")
        self.metrics_service = metrics_service

    async def get_gpu_monitor(self) -> None:
        """异步更新并记录 GPU 指标"""
        monitor_data = await self.gpu_monitor.update_metrics()
        self.metrics_service.set_gpu_cache_usage(monitor_data["gpu_cache_usage"])


    async def get_cpu_monitor(self) -> None:
        """异步更新并记录 CPU 指标"""
        monitor_data = await self.cpu_monitor.update_metrics()
        self.metrics_service.set_cpu_cache_usage(monitor_data["cpu_cache_usage"])
