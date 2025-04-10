# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, Dict

# 服务URL配置
PREFILL_URL = "http://172.168.178.64:7004/v1/completions"  # GPU服务
DECODE_URL = "http://172.168.178.64:7005/v1/completions"   # CPU服务
GPU_METRICS_URL = "http://172.168.178.64:7004/metrics"     # GPU服务指标
CPU_METRICS_URL = "http://172.168.178.64:7005/metrics"     # CPU服务指标

# 请求队列配置
MAX_QUEUE_SIZE = 100
REQUEST_TIMEOUT = 300  # 5分钟超时

# 调度参数默认值
DEFAULT_SCHEDULING_PARAMS = {
    # 负载阈值
    "high_load_threshold": 0.8,   # 高负载阈值
    "medium_load_threshold": 0.6, # 中等负载阈值
    "low_load_threshold": 0.4,    # 低负载阈值
    
    # 资源监控配置
    "kv_cache_threshold": 0.5,    # KV Cache使用率阈值
    "running_reqs_threshold": 8,  # 运行中请求数阈值
    "waiting_reqs_threshold": 4,  # 等待中请求数阈值
    
    # 系统繁忙判断阈值
    "system_busy_threshold": 0.9,  # 当GPU和CPU负载都超过这个值时，系统判定为繁忙
}


@dataclass
class ServicePerformance:
    """服务性能指标数据类"""
    avg_latency: float  # 平均延迟，单位毫秒
    throughput: float   # 吞吐量，单位tokens/s
    success_rate: float # 成功率，0-1之间


@dataclass
class RequestState:
    """请求状态数据类"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    created_at: float
    status: str  # pending, processing, completed, failed
    assigned_service: Optional[str] = None  # gpu, cpu, or None
    response: Optional[Dict] = None
    error: Optional[str] = None


class CompletionRequest(BaseModel):
    """完成请求模型"""
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    request_id: Optional[str] = None
    prefill_then_swapout: bool = False      # 是否在prefill后立即交换出
    continue_decoding: Optional[str] = None # 继续解码的指示
    active_token: Optional[int] = None      # 用于继续解码的活动token
    gpu_preference: float = 0.5             # GPU偏好度 (0-1, 越高越偏向GPU)
