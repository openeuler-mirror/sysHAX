import httpx
import re
import time
from typing import Dict, Tuple

from config import GPU_METRICS_URL, CPU_METRICS_URL
from logger import Logger

log = Logger

# Prometheus指标正则匹配模式
# GPU KV缓存使用率 - 百分比值，表示GPU上KV缓存的使用程度，1.0表示100%使用
RE_KV_CACHE = re.compile(r'vllm:gpu_cache_usage_perc{[^}]*}\s+([\d.]+)')
# CPU KV缓存使用率 - 百分比值，表示CPU上KV缓存的使用程度，1.0表示100%使用
RE_CPU_CACHE = re.compile(r'vllm:cpu_cache_usage_perc{[^}]*}\s+([\d.]+)')
# 运行中请求数 - 整数值，表示当前在GPU上运行的请求数量
RE_RUNNING_REQS = re.compile(r'vllm:num_requests_running{[^}]*}\s+(\d+)')
# 等待中请求数 - 整数值，表示当前在队列中等待处理的请求数量
RE_WAITING_REQS = re.compile(r'vllm:num_requests_waiting{[^}]*}\s+(\d+)')
# 已交换请求数 - 整数值，表示当前被交换到CPU内存的请求数量
RE_SWAPPED_REQS = re.compile(r'vllm:num_requests_swapped{[^}]*}\s+(\d+)')
# Prefill处理Token总数 - 浮点值，表示处理的prefill token累计总数（用于计算吞吐量）
RE_PREFILL_TOKENS = re.compile(r'vllm:prompt_tokens_total{[^}]*}\s+([\d.]+)')
# Decode Token总数 - 浮点值，表示解码的token累计总数（用于计算吞吐量）
RE_DECODE_TOKENS = re.compile(r'vllm:generation_tokens_total{[^}]*}\s+([\d.]+)')


class ResourceMonitor:
    """资源监控类，解析单个vLLM服务的Prometheus指标
    
    职责：
    1. 从指定URL获取单个服务的指标
    2. 解析指标并提供简单的接口访问这些指标
    """
    
    def __init__(self, metrics_url: str, is_cpu_service: bool = False):
        """初始化资源监控器
        
        Args:
            metrics_url: 指标URL
            is_cpu_service: 是否为CPU服务，影响某些指标的默认值
        """
        self.metrics_url = metrics_url
        self.is_cpu_service = is_cpu_service
        
        # 指标缓存
        self.last_update_time = 0
        self.update_interval = 1.0  # 秒
        
        # 初始指标值
        self.metrics = {
            'gpu_cache_usage': 1.0 if is_cpu_service else 0.0,  # CPU服务默认GPU使用率为100%
            'cpu_cache_usage': 0.0,
            'num_running': 0,
            'num_waiting': 0,
            'num_swapped': 0,
            'prefill_tokens': 0,  # CPU服务没有prefill能力，默认为0
            'decode_tokens': 0,
            'last_prefill_tokens': 0,
            'last_decode_tokens': 0
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
                    log.warning(f"获取指标失败: HTTP {response.status_code}")
                    return False
                
                metrics_text = response.text
                self.last_update_time = current_time
                
                # 保存前一次的token计数用于计算吞吐量
                self.metrics['last_prefill_tokens'] = self.metrics['prefill_tokens']
                self.metrics['last_decode_tokens'] = self.metrics['decode_tokens']
                
                # 解析指标
                self._parse_metrics(metrics_text)
                return True
        
        except Exception as e:
            log.error(f"更新指标出错: {e}", exc_info=True)
            return False
    
    def _parse_metrics(self, metrics_text: str):
        """解析Prometheus格式的指标文本"""
        # KV Cache使用率 - 如果是CPU服务，保持GPU缓存使用率为100%
        if not self.is_cpu_service:
            kv_cache_match = RE_KV_CACHE.search(metrics_text)
            if kv_cache_match:
                self.metrics['gpu_cache_usage'] = float(kv_cache_match.group(1))
        
        # CPU Cache使用率
        cpu_cache_match = RE_CPU_CACHE.search(metrics_text)
        if cpu_cache_match:
            self.metrics['cpu_cache_usage'] = float(cpu_cache_match.group(1))
        
        # 运行中请求数
        running_match = RE_RUNNING_REQS.search(metrics_text)
        if running_match:
            self.metrics['num_running'] = int(running_match.group(1))
        
        # 等待中请求数
        waiting_match = RE_WAITING_REQS.search(metrics_text)
        if waiting_match:
            self.metrics['num_waiting'] = int(waiting_match.group(1))
        
        # 已swap请求数
        swapped_match = RE_SWAPPED_REQS.search(metrics_text)
        if swapped_match:
            self.metrics['num_swapped'] = int(swapped_match.group(1))
        
        # Token计数 - 只有GPU服务才解析prefill tokens
        if not self.is_cpu_service:
            prefill_tokens_match = RE_PREFILL_TOKENS.search(metrics_text)
            if prefill_tokens_match:
                self.metrics['prefill_tokens'] = float(prefill_tokens_match.group(1))
        
        # 解码token计数
        decode_tokens_match = RE_DECODE_TOKENS.search(metrics_text)
        if decode_tokens_match:
            self.metrics['decode_tokens'] = float(decode_tokens_match.group(1))
    
    def get_throughput(self) -> Tuple[float, float]:
        """获取当前吞吐量（tokens/s）
        
        Returns:
            元组 (prefill吞吐量, decode吞吐量)
        """
        elapsed = time.time() - self.last_update_time
        if elapsed <= 0:
            return 0.0, 0.0
            
        prefill_diff = self.metrics['prefill_tokens'] - self.metrics['last_prefill_tokens']
        decode_diff = self.metrics['decode_tokens'] - self.metrics['last_decode_tokens']
        
        prefill_throughput = prefill_diff / elapsed
        decode_throughput = decode_diff / elapsed
        
        return prefill_throughput, decode_throughput
    
    def get_metrics(self) -> Dict:
        """获取当前指标
        
        Returns:
            指标数据字典
        """
        prefill_throughput, decode_throughput = self.get_throughput()
        
        return {
            'gpu_cache_usage': self.metrics['gpu_cache_usage'],
            'cpu_cache_usage': self.metrics['cpu_cache_usage'],
            'num_running': self.metrics['num_running'],
            'num_waiting': self.metrics['num_waiting'], 
            'num_swapped': self.metrics['num_swapped'],
            'prefill_tokens': self.metrics['prefill_tokens'],
            'decode_tokens': self.metrics['decode_tokens'],
            'prefill_throughput': prefill_throughput,
            'decode_throughput': decode_throughput
        }


class SystemMonitor:
    """系统监控类，同时监控GPU和CPU服务
    
    职责：
    1. 管理GPU和CPU服务的ResourceMonitor实例
    2. 提供统一的接口获取所有指标
    3. 不做任何决策逻辑
    """
    
    def __init__(self, gpu_metrics_url: str = GPU_METRICS_URL, cpu_metrics_url: str = CPU_METRICS_URL):
        """初始化系统监控器
        
        Args:
            gpu_metrics_url: GPU服务指标URL
            cpu_metrics_url: CPU服务指标URL
        """
        self.gpu_monitor = ResourceMonitor(gpu_metrics_url, is_cpu_service=False)
        self.cpu_monitor = ResourceMonitor(cpu_metrics_url, is_cpu_service=True)
        self.last_update_time = 0
    
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