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
from re import Pattern
import time
from typing import Callable
import httpx

from src.utils.config import CPU_HOST, CPU_PORT, GPU_HOST, GPU_PORT, MONITOR_INTERVAL
from src.utils.logger import Logger
from dataclasses import dataclass

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

@dataclass
class MetricsData:
    gpu_cache_usage: float = 0.0
    cpu_cache_usage: float = 0.0
    num_running: int = 0
    num_waiting: int = 0
    num_swapped: int = 0
    prefill_throughout: float = 0.0
    decode_throughout: float = 0.0

class ResourceMonitor:
    """
    资源监控类，解析单个vLLM服务的Prometheus指标

    职责：
    1. 从指定URL获取单个服务的指标
    2. 解析指标并提供简单的接口访问这些指标
    """

    def __init__(self, metrics_url: str, service_name: str = "Unknown") -> None:
        """
        初始化资源监控器

        Args:
            metrics_url: 指标URL
            service_name: 服务名称，用于日志

        """
        self.metrics_url = metrics_url
        self.service_name = service_name
        self.update_interval = MONITOR_INTERVAL

        # 配置日志信息
        Logger.info(
            f"初始化{service_name}监控：{metrics_url}, 更新间隔={self.update_interval}秒",
        )

        self.last_update_time = 0.0
        self.metrics_data = MetricsData()

        # cumulative stats for throughput aggregation
        self._cum_prefill_tokens = 0.0
        self._cum_prefill_time_ns = 0
        self._cum_decode_tokens = 0.0
        self._cum_decode_time_ns = 0

    def update_metrics(self, force: bool = False) -> bool:
        """
        更新指标，只在需要时获取

        Args:
            force: 强制刷新，忽略时间间隔限制

        Returns:
            更新是否成功

        """
        try:
            current_time = time.time()
            # 如果不是强制刷新且上次更新是在更新间隔内，直接返回缓存的结果
            if (not force and current_time - self.last_update_time < self.update_interval):
                return True

            # 发起HTTP请求获取指标
            with httpx.Client() as client:
                response = client.get(self.metrics_url, timeout=3.0)

                if response.status_code != httpx.codes.OK:
                    Logger.warning(f"获取指标失败: HTTP {response.status_code}")
                    return False

                metrics_text = response.text
                self.last_update_time = current_time

                self.metrics_data.gpu_cache_usage = self._parse_metrics(metrics_text, RE_GPU_CACHE, float)
                self.metrics_data.cpu_cache_usage = self._parse_metrics(metrics_text, RE_CPU_CACHE, float)
                self.metrics_data.num_running = self._parse_metrics(metrics_text, RE_RUNNING_REQS, int)
                self.metrics_data.num_waiting = self._parse_metrics(metrics_text, RE_WAITING_REQS, int)
                self.metrics_data.num_swapped = self._parse_metrics(metrics_text, RE_SWAPPED_REQS, int)
                
                # 根据累积的统计数据计算并重置吞吐量指标
                if self._cum_prefill_time_ns > 0:
                    self.metrics_data.prefill_throughout = self._cum_prefill_tokens / (self._cum_prefill_time_ns / 1e9)
                else:
                    self.metrics_data.prefill_throughout = 0.0
                self._cum_prefill_tokens = 0.0
                self._cum_prefill_time_ns = 0

                if self._cum_decode_time_ns > 0:
                    self.metrics_data.decode_throughout = self._cum_decode_tokens / (self._cum_decode_time_ns / 1e9)
                else:
                    self.metrics_data.decode_throughout = 0.0
                self._cum_decode_tokens = 0.0
                self._cum_decode_time_ns = 0
                return True
        except httpx.TimeoutException as e:
            Logger.warning(f"获取指标超时: {e}")
            return False
        except httpx.HTTPStatusError as e:
            Logger.warning(f"HTTP错误: {e}")
            return False

    def _parse_metrics(self, metrics_text: str, pattern: Pattern, converter) -> None:
        match = pattern.search(metrics_text)
        if match:
            return converter(match.group(1))
        return converter("0")

    def get_metrics(self) -> dict:
        """
        获取当前指标

        Returns:
            指标数据字典

        """
        return {
            # 资源使用
            "gpu_cache_usage": self.metrics_data.gpu_cache_usage,
            "cpu_cache_usage": self.metrics_data.cpu_cache_usage,
            "num_running": self.metrics_data.num_running,
            "num_waiting": self.metrics_data.num_waiting,
            "num_swapped": self.metrics_data.num_swapped,
            # 吞吐量
            "prefill_throughout": self.metrics_data.prefill_throughout,
            "decode_throughout": self.metrics_data.decode_throughout,
        }
    
    def set_prefill_throughout(self, prefill_throughout: float) -> None:
        self.metrics_data.prefill_throughout = prefill_throughout
    
    def set_decode_throughout(self, decode_throughout: float) -> None:
        self.metrics_data.decode_throughout = decode_throughout

    def add_prefill_stats(self, tokens: float, time_ns: int) -> None:
        """Accumulate prefill tokens and time, update throughput."""
        self._cum_prefill_tokens += tokens
        self._cum_prefill_time_ns += time_ns
        if self._cum_prefill_time_ns > 0:
            self.metrics_data.prefill_throughout = self._cum_prefill_tokens / (self._cum_prefill_time_ns / 1e9)
        else:
            self.metrics_data.prefill_throughout = 0.0

    def add_decode_stats(self, tokens: float, time_ns: int) -> None:
        """Accumulate decode tokens and time, update throughput."""
        self._cum_decode_tokens += tokens
        self._cum_decode_time_ns += time_ns
        if self._cum_decode_time_ns > 0:
            self.metrics_data.decode_throughout = self._cum_decode_tokens / (self._cum_decode_time_ns / 1e9)
        else:
            self.metrics_data.decode_throughout = 0.0


class SystemMonitor:
    """
    系统监控类，同时监控GPU和CPU服务

    职责：
    1. 管理GPU和CPU服务的ResourceMonitor实例
    2. 提供统一的接口获取所有指标
    """

    def __init__(self) -> None:
        """初始化系统监控器：根据配置拼接 metrics URL"""
        # 构建 GPU/CPU metrics URL
        gpu_metrics_url = f"http://{GPU_HOST}:{GPU_PORT}/metrics"
        cpu_metrics_url = f"http://{CPU_HOST}:{CPU_PORT}/metrics"
        self.gpu_monitor = ResourceMonitor(gpu_metrics_url, service_name="GPU")
        self.cpu_monitor = ResourceMonitor(cpu_metrics_url, service_name="CPU")
        self.last_update_time = 0.0
        Logger.info("系统监控器初始化完成")

    def update_metrics(self, *, force: bool = False) -> tuple[bool, bool]:
        """
        同时更新GPU和CPU指标

        Args:
            force: 强制刷新，忽略时间间隔限制

        Returns:
            元组 (GPU更新成功, CPU更新成功)

        """
        gpu_success = self.gpu_monitor.update_metrics(force=force)
        cpu_success = self.cpu_monitor.update_metrics(force=force)
        Logger.info("SystemMonitor.update_metrics OK")

        if gpu_success or cpu_success:
            self.last_update_time = time.time()

        return gpu_success, cpu_success

    @property
    def gpu_metrics(self) -> MetricsData:
        """获取GPU服务指标"""
        return self.gpu_monitor.metrics_data

    @property
    def cpu_metrics(self) -> MetricsData:
        """获取CPU服务指标"""
        return self.cpu_monitor.metrics_data

    def get_metrics(self) -> dict:
        """获取所有系统指标"""
        return {
            "gpu": self.gpu_monitor.get_metrics(),
            "cpu": self.cpu_monitor.get_metrics(),
            "last_update": self.last_update_time,
        }
