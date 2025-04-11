"""资源监控模块

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
import httpx
import re
import time
from typing import Dict, Tuple, Optional

from src.utils.config import GPU_METRICS_URL, CPU_METRICS_URL
from src.utils.logger import Logger

log = Logger

# Prometheus指标正则匹配模式
# 吞吐量指标 - vLLM原生吞吐量指标，tokens/s
RE_GEN_THROUGHPUT = re.compile(r'vllm:avg_generation_throughput_toks_per_s{[^}]*}\s+([\d.]+)')
RE_PROMPT_THROUGHPUT = re.compile(r'vllm:avg_prompt_throughput_toks_per_s{[^}]*}\s+([\d.]+)')
RE_TOTAL_THROUGHPUT = re.compile(r'vllm:avg_total_throughput_toks_per_s{[^}]*}\s+([\d.]+)')

# 资源使用指标
RE_GPU_CACHE = re.compile(r'vllm:gpu_cache_usage_perc{[^}]*}\s+([\d.]+)')  # GPU KV缓存使用率
RE_CPU_CACHE = re.compile(r'vllm:cpu_cache_usage_perc{[^}]*}\s+([\d.]+)')  # CPU KV缓存使用率
RE_RUNNING_REQS = re.compile(r'vllm:num_requests_running{[^}]*}\s+(\d+)')  # 运行中请求数
RE_WAITING_REQS = re.compile(r'vllm:num_requests_waiting{[^}]*}\s+(\d+)')  # 等待中请求数
RE_SWAPPED_REQS = re.compile(r'vllm:num_requests_swapped{[^}]*}\s+(\d+)')  # 已交换请求数

# Token计数指标
RE_PREFILL_TOKENS = re.compile(r'vllm:prompt_tokens_total{[^}]*}\s+([\d.]+)')  # Prefill token累计总数
RE_DECODE_TOKENS = re.compile(r'vllm:generation_tokens_total{[^}]*}\s+([\d.]+)')  # Decode token累计总数

# 时间指标
RE_TIME_PER_TOKEN = re.compile(r'vllm:time_per_output_token_seconds_sum{[^}]*}\s+([\d.]+)')  # 总token生成时间
RE_TIME_PER_TOKEN_COUNT = re.compile(r'vllm:time_per_output_token_seconds_count{[^}]*}\s+([\d.]+)')  # token计数

# 监控系统配置参数 - 只包含影响监控行为的参数，不影响指标值
CPU_MONITOR_CONFIG = {
    'update_interval': 2.0,                 # CPU指标更新间隔 (秒)
    'throughput_decay_window': 30.0,        # 吞吐量衰减窗口 (秒)
    'status_active_threshold': 30.0,        # 活跃状态阈值 (秒)
    'default_gpu_cache_usage': 1.0,         # CPU服务默认的GPU缓存使用率
}

GPU_MONITOR_CONFIG = {
    'update_interval': 1.0,                 # GPU指标更新间隔 (秒)
    'throughput_decay_window': 15.0,        # 吞吐量衰减窗口 (秒)
    'status_active_threshold': 15.0,        # 活跃状态阈值 (秒)
    'default_gpu_cache_usage': 0.0,         # GPU服务默认的GPU缓存使用率
}

class ResourceMonitor:
    """资源监控类，解析单个vLLM服务的Prometheus指标
    
    职责：
    1. 从指定URL获取单个服务的指标
    2. 解析指标并提供简单的接口访问这些指标
    """
    
    def __init__(self, metrics_url: str, is_cpu_service: bool = False, config_override: Optional[Dict] = None):
        """初始化资源监控器
        
        Args:
            metrics_url: 指标URL
            is_cpu_service: 是否为CPU服务
            config_override: 覆盖默认配置的参数字典
        """
        self.metrics_url = metrics_url
        self.is_cpu_service = is_cpu_service
        
        # 根据服务类型选择默认配置，并应用覆盖配置
        default_config = CPU_MONITOR_CONFIG if is_cpu_service else GPU_MONITOR_CONFIG
        self.config = default_config.copy()
        if config_override:
            self.config.update(config_override)
            
        # 配置日志信息
        service_type = "CPU" if is_cpu_service else "GPU"
        log.info(f"初始化{service_type}服务监控：{metrics_url}, 更新间隔={self.config['update_interval']}秒")
        
        # 指标缓存
        self.last_update_time = 0.0
        self.update_interval = self.config['update_interval']
        
        # 初始指标值
        self.metrics = {
            # 资源使用指标
            'gpu_cache_usage': self.config['default_gpu_cache_usage'],
            'cpu_cache_usage': 0.0,
            'num_running': 0,
            'num_waiting': 0,
            'num_swapped': 0,
            
            # 吞吐量指标 - 直接来自vLLM
            'generation_throughput': 0.0,  # 生成阶段吞吐量
            'prompt_throughput': 0.0,      # 处理输入阶段吞吐量
            'total_throughput': 0.0,       # 总吞吐量
            
            # token计数
            'prefill_tokens': 0.0,
            'decode_tokens': 0.0,
            
            # 平均token生成时间
            'time_per_token': 0.0,
            'time_per_token_count': 0,
            
            # 服务状态
            'is_active': False,  # 服务是否有活跃请求
            'last_active_time': 0.0,  # 最后活跃时间
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
                
                # 解析指标
                self._parse_metrics(metrics_text)
                
                # 更新服务活跃状态
                was_active = self.metrics['is_active']
                self.metrics['is_active'] = self.metrics['num_running'] > 0
                
                # 如果服务状态从活跃变为不活跃，记录最后活跃时间
                if was_active and not self.metrics['is_active']:
                    self.metrics['last_active_time'] = current_time
                    log.debug(f"服务状态变更：活跃 -> 不活跃，记录时间 {current_time}")
                
                return True
        
        except Exception as e:
            log.error(f"更新指标出错: {e}", exc_info=True)
            return False
    
    def _parse_metrics(self, metrics_text: str):
        """解析Prometheus格式的指标文本"""
        # 解析资源使用指标
        # 如果是CPU服务，保持GPU缓存使用率为默认值
        if not self.is_cpu_service:
            gpu_cache_match = RE_GPU_CACHE.search(metrics_text)
            if gpu_cache_match:
                self.metrics['gpu_cache_usage'] = float(gpu_cache_match.group(1))
        
        # CPU Cache使用率
        cpu_cache_match = RE_CPU_CACHE.search(metrics_text)
        if cpu_cache_match:
            self.metrics['cpu_cache_usage'] = float(cpu_cache_match.group(1))
        
        # 请求状态指标
        running_match = RE_RUNNING_REQS.search(metrics_text)
        if running_match:
            self.metrics['num_running'] = int(running_match.group(1))
        
        waiting_match = RE_WAITING_REQS.search(metrics_text)
        if waiting_match:
            self.metrics['num_waiting'] = int(waiting_match.group(1))
        
        swapped_match = RE_SWAPPED_REQS.search(metrics_text)
        if swapped_match:
            self.metrics['num_swapped'] = int(swapped_match.group(1))
        
        # 解析吞吐量指标 - 使用原始值，不应用任何上限
        gen_throughput_match = RE_GEN_THROUGHPUT.search(metrics_text)
        if gen_throughput_match:
            self.metrics['generation_throughput'] = float(gen_throughput_match.group(1))
            
        prompt_throughput_match = RE_PROMPT_THROUGHPUT.search(metrics_text)
        if prompt_throughput_match:
            self.metrics['prompt_throughput'] = float(prompt_throughput_match.group(1))
            
        total_throughput_match = RE_TOTAL_THROUGHPUT.search(metrics_text)
        if total_throughput_match:
            self.metrics['total_throughput'] = float(total_throughput_match.group(1))
        
        # Token计数指标
        prefill_tokens_match = RE_PREFILL_TOKENS.search(metrics_text)
        if prefill_tokens_match:
            self.metrics['prefill_tokens'] = float(prefill_tokens_match.group(1))
        
        decode_tokens_match = RE_DECODE_TOKENS.search(metrics_text)
        if decode_tokens_match:
            self.metrics['decode_tokens'] = float(decode_tokens_match.group(1))
            
        # 解析token生成时间
        time_per_token_match = RE_TIME_PER_TOKEN.search(metrics_text)
        time_per_token_count_match = RE_TIME_PER_TOKEN_COUNT.search(metrics_text)
        
        if time_per_token_match and time_per_token_count_match:
            time_sum = float(time_per_token_match.group(1))
            count = float(time_per_token_count_match.group(1))
            
            if count > 0:
                # 原始值，不应用最小值限制
                self.metrics['time_per_token'] = time_sum / count
                self.metrics['time_per_token_count'] = count
    
    def get_throughput(self) -> Tuple[float, float]:
        """获取当前吞吐量（tokens/s）
        
        Returns:
            元组 (prefill吞吐量, decode吞吐量)
        """
        # 如果服务有活跃请求，直接使用vLLM原生吞吐量指标
        if self.metrics['is_active']:
            return self.metrics['prompt_throughput'], self.metrics['generation_throughput']
        
        # 如果服务没有活跃请求，但最近活跃过，返回适当衰减的吞吐量
        decay_window = self.config['throughput_decay_window']
        time_since_active = time.time() - self.metrics['last_active_time']
        
        if self.metrics['last_active_time'] > 0 and time_since_active < decay_window:
            # 应用衰减因子（仅影响显示，不修改原始值）
            decay_factor = max(0.0, 1.0 - (time_since_active / decay_window))
            
            return (
                self.metrics['prompt_throughput'] * decay_factor,
                self.metrics['generation_throughput'] * decay_factor
            )
        
        # 服务长时间不活跃，返回基于最近平均token生成时间的估计值
        elif self.metrics['time_per_token'] > 0:
            # 使用原始token生成时间，不应用任何限制
            estimated_throughput = 1.0 / self.metrics['time_per_token']
            return 0.0, estimated_throughput
            
        # 没有任何数据可用
        else:
            return 0.0, 0.0
    
    def get_metrics(self) -> Dict:
        """获取当前指标
        
        Returns:
            指标数据字典
        """
        prefill_throughput, decode_throughput = self.get_throughput()
        
        service_type = "CPU" if self.is_cpu_service else "GPU"
        if decode_throughput > 0:
            log.debug(f"{service_type}服务吞吐量: {decode_throughput:.2f} tokens/s, 原始值: {self.metrics['generation_throughput']:.2f} tokens/s")
        
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
            'prefill_throughput': prefill_throughput,
            'decode_throughput': decode_throughput,
            'raw_generation_throughput': self.metrics['generation_throughput'],
            'raw_prompt_throughput': self.metrics['prompt_throughput'],
            'raw_total_throughput': self.metrics['total_throughput'],
            
            # 服务状态
            'is_active': self.metrics['is_active'],
            'last_active_time': self.metrics['last_active_time'],
            'time_per_token': self.metrics['time_per_token']
        }


class SystemMonitor:
    """系统监控类，同时监控GPU和CPU服务
    
    职责：
    1. 管理GPU和CPU服务的ResourceMonitor实例
    2. 提供统一的接口获取所有指标
    3. 不做任何决策逻辑
    """
    
    def __init__(self, gpu_metrics_url: str = GPU_METRICS_URL, cpu_metrics_url: str = CPU_METRICS_URL, 
                 gpu_config: Optional[Dict] = None, cpu_config: Optional[Dict] = None):
        """初始化系统监控器
        
        Args:
            gpu_metrics_url: GPU服务指标URL
            cpu_metrics_url: CPU服务指标URL
            gpu_config: GPU服务监控配置
            cpu_config: CPU服务监控配置
        """
        self.gpu_monitor = ResourceMonitor(gpu_metrics_url, is_cpu_service=False, config_override=gpu_config)
        self.cpu_monitor = ResourceMonitor(cpu_metrics_url, is_cpu_service=True, config_override=cpu_config)
        self.last_update_time = 0.0
        
        log.info("系统监控器初始化完成")
    
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
    
    def get_service_status(self) -> Dict[str, str]:
        """获取服务状态概述
        
        Returns:
            包含GPU和CPU服务状态描述的字典
        """
        gpu_metrics = self.get_gpu_metrics()
        cpu_metrics = self.get_cpu_metrics()
        
        # 基于指标确定服务状态
        def get_status(metrics, is_cpu=False):
            threshold = CPU_MONITOR_CONFIG['status_active_threshold'] if is_cpu else GPU_MONITOR_CONFIG['status_active_threshold']
            
            if metrics['num_running'] > 0:
                return "运行中"
            elif metrics['num_waiting'] > 0:
                return "等待中"
            elif metrics['is_active']:
                return "活跃"
            elif time.time() - metrics.get('last_active_time', 0) < threshold:
                return "刚刚完成"
            else:
                return "空闲"
        
        return {
            "gpu": get_status(gpu_metrics),
            "cpu": get_status(cpu_metrics, is_cpu=True)
        } 