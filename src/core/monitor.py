"""资源监控模块

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
import httpx
import re
import time
from typing import Dict, Tuple

from src.utils.config import GPU_METRICS_URL, CPU_METRICS_URL
from src.utils.logger import Logger

# Prometheus指标正则匹配模式
# 吞吐量指标 - vLLM原生吞吐量指标，tokens/s
RE_GEN_THROUGHPUT = re.compile(r'vllm:avg_generation_throughput_toks_per_s{[^}]*}\s+([\d.]+)')  # 生成阶段吞吐量：每秒生成多少新token
RE_PROMPT_THROUGHPUT = re.compile(r'vllm:avg_prompt_throughput_toks_per_s{[^}]*}\s+([\d.]+)')  # 输入处理阶段吞吐量：每秒处理多少输入token

# 资源使用指标
RE_GPU_CACHE = re.compile(r'vllm:gpu_cache_usage_perc{[^}]*}\s+([\d.]+)')  # GPU KV缓存使用率：值域0-1，1表示100%使用
RE_CPU_CACHE = re.compile(r'vllm:cpu_cache_usage_perc{[^}]*}\s+([\d.]+)')  # CPU KV缓存使用率：值域0-1，1表示100%使用
RE_RUNNING_REQS = re.compile(r'vllm:num_requests_running{[^}]*}\s+(\d+)')  # 运行中请求数：当前在GPU上执行的请求数量
RE_WAITING_REQS = re.compile(r'vllm:num_requests_waiting{[^}]*}\s+(\d+)')  # 等待中请求数：等待GPU资源的请求数量
RE_SWAPPED_REQS = re.compile(r'vllm:num_requests_swapped{[^}]*}\s+(\d+)')  # 已交换请求数：从GPU交换到CPU内存的请求数量

# Token计数指标
RE_PREFILL_TOKENS = re.compile(r'vllm:prompt_tokens_total{[^}]*}\s+([\d.]+)')  # Prefill token累计总数：处理过的输入token总数，用于性能测试
RE_DECODE_TOKENS = re.compile(r'vllm:generation_tokens_total{[^}]*}\s+([\d.]+)')  # Decode token累计总数：生成的token总数，用于性能测试

class ResourceMonitor:
    """资源监控类，解析单个vLLM服务的Prometheus指标
    
    职责：
    1. 从指定URL获取单个服务的指标
    2. 解析指标并提供简单的接口访问这些指标
    """
    
    def __init__(self, metrics_url: str, service_name: str = "Unknown"):
        """初始化资源监控器
        
        Args:
            metrics_url: 指标URL
            service_name: 服务名称，用于日志
        """
        self.metrics_url = metrics_url
        self.service_name = service_name
        self.update_interval = 1.0
            
        # 配置日志信息
        Logger.info(f"初始化{service_name}监控：{metrics_url}, 更新间隔={self.update_interval}秒")
        
        # 指标缓存
        self.last_update_time = 0.0
        
        # 初始指标值
        self.metrics = {
            'gpu_cache_usage': 0.0,       # GPU设备上的KV缓存使用率
            'cpu_cache_usage': 0.0,       # CPU设备上的KV缓存使用率
            'num_running': 0,             # 运行中的请求数量
            'num_waiting': 0,             # 等待处理的请求数量
            'num_swapped': 0,             # 已交换到CPU内存的请求数量
            
            # 吞吐量指标 (GPU和CPU有不同特性)
            'generation_throughput': 0.0, # 生成阶段吞吐量，GPU通常远高于CPU
            'prompt_throughput': 0.0,     # 处理输入阶段吞吐量，GPU优势较大
            
            # token计数 (累计值)
            'prefill_tokens': 0.0,        # 已处理的输入token总数
            'decode_tokens': 0.0,         # 已生成的token总数
        }
        
        # 定义正则表达式到指标的映射
        self.metric_patterns = {
            # 浮点数指标
            'gpu_cache_usage': (RE_GPU_CACHE, float),
            'cpu_cache_usage': (RE_CPU_CACHE, float),
            'generation_throughput': (RE_GEN_THROUGHPUT, float),
            'prompt_throughput': (RE_PROMPT_THROUGHPUT, float),
            'prefill_tokens': (RE_PREFILL_TOKENS, float),
            'decode_tokens': (RE_DECODE_TOKENS, float),
            
            # 整数指标
            'num_running': (RE_RUNNING_REQS, int),
            'num_waiting': (RE_WAITING_REQS, int),
            'num_swapped': (RE_SWAPPED_REQS, int),
        }
    
    async def update_metrics(self) -> bool:
        """更新指标，只在需要时获取
        
        Returns:
            更新是否成功
        """
        try:
            current_time = time.time()
            # 如果上次更新是在更新间隔内，直接返回缓存的结果
            if current_time - self.last_update_time < self.update_interval:
                return True
            
            # 发起HTTP请求获取指标
            async with httpx.AsyncClient() as client:
                response = await client.get(self.metrics_url, timeout=3.0)
                
                if response.status_code != 200:
                    Logger.warning(f"获取指标失败: HTTP {response.status_code}")
                    return False
                
                metrics_text = response.text
                self.last_update_time = current_time
                
                # 解析指标
                self._parse_metrics(metrics_text)
                return True
        
        except Exception as e:
            Logger.error(f"更新指标出错: {e}", exc_info=True)
            return False
    
    def _parse_metrics(self, metrics_text: str):
        """解析Prometheus格式的指标文本"""
        # 使用定义好的映射遍历处理每个指标
        for metric_name, (pattern, converter) in self.metric_patterns.items():
            match = pattern.search(metrics_text)
            if match:
                self.metrics[metric_name] = converter(match.group(1))
    
    def get_metrics(self) -> Dict:
        """获取当前指标
        
        Returns:
            指标数据字典
        """
        return {
            # 资源使用
            'gpu_cache_usage': self.metrics['gpu_cache_usage'],
            'cpu_cache_usage': self.metrics['cpu_cache_usage'],
            'num_running': self.metrics['num_running'],
            'num_waiting': self.metrics['num_waiting'], 
            'num_swapped': self.metrics['num_swapped'],
            
            # token统计
            'prefill_tokens': self.metrics['prefill_tokens'],
            'decode_tokens': self.metrics['decode_tokens'],
            
            # 吞吐量
            'prompt_throughput': self.metrics['prompt_throughput'],
            'generation_throughput': self.metrics['generation_throughput']
        }


class SystemMonitor:
    """系统监控类，同时监控GPU和CPU服务
    
    职责：
    1. 管理GPU和CPU服务的ResourceMonitor实例
    2. 提供统一的接口获取所有指标
    """
    
    def __init__(self, gpu_metrics_url: str = GPU_METRICS_URL, cpu_metrics_url: str = CPU_METRICS_URL):
        """初始化系统监控器"""
        self.gpu_monitor = ResourceMonitor(gpu_metrics_url, service_name="GPU")
        self.cpu_monitor = ResourceMonitor(cpu_metrics_url, service_name="CPU")
        self.last_update_time = 0.0
        
        Logger.info("系统监控器初始化完成")
    
    async def update_metrics(self) -> Tuple[bool, bool]:
        """同时更新GPU和CPU指标
        
        Returns:
            元组 (GPU更新成功, CPU更新成功)
        """
        gpu_success = await self.gpu_monitor.update_metrics()
        cpu_success = await self.cpu_monitor.update_metrics()
        
        if gpu_success or cpu_success:
            self.last_update_time = time.time()
            
        return gpu_success, cpu_success
    
    def get_gpu_metrics(self) -> Dict:
        """获取GPU服务指标"""
        return self.gpu_monitor.get_metrics()
    
    def get_cpu_metrics(self) -> Dict:
        """获取CPU服务指标"""
        return self.cpu_monitor.get_metrics()
    
    def get_all_metrics(self) -> Dict:
        """获取所有系统指标"""
        return {
            "gpu": self.get_gpu_metrics(),
            "cpu": self.get_cpu_metrics(),
            "last_update": self.last_update_time
        }
    