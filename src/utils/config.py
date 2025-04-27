"""配置管理模块
负责加载和提供系统配置

Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """加载配置
    从配置文件config/config.yaml加载
    """
    config = {}
    
    # 从配置文件加载
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = yaml_config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    else:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    return config

# 加载配置
CONFIG = load_config()

# 服务URL配置
PREFILL_URL = CONFIG["services"]["gpu"]["url"]
DECODE_URL = CONFIG["services"]["cpu"]["url"]
GPU_METRICS_URL = CONFIG["services"]["gpu"]["metrics_url"]
CPU_METRICS_URL = CONFIG["services"]["cpu"]["metrics_url"]
SYSHAX_HOST = CONFIG["services"]["conductor"]["host"]
SYSHAX_PORT = CONFIG["services"]["conductor"]["port"]

# 请求队列配置
MAX_QUEUE_SIZE = CONFIG["system"]["max_queue_size"]
REQUEST_TIMEOUT = CONFIG["system"]["request_timeout"]

# 模型配置
DEFAULT_MODEL = CONFIG["models"]["default"]
DEFAULT_MAX_TOKENS = CONFIG["models"]["params"]["max_tokens"]
DEFAULT_TEMPERATURE = CONFIG["models"]["params"]["temperature"]
DEFAULT_TEST_PROMPT = CONFIG["models"]["params"]["test_prompt"]
DEFAULT_TEST_TOKENS = CONFIG["models"]["params"]["test_tokens"]

# 调度决策器配置
GPU_CACHE_THRESHOLD = CONFIG["decider"]["gpu_cache_threshold"]
CPU_THROUGHPUT_THRESHOLD = CONFIG["decider"]["cpu_throughput_threshold"]
TOKEN_LIMIT_MULTIPLIER = CONFIG["decider"]["token_limit_multiplier"]
TOKEN_LIMIT_MIN = CONFIG["decider"]["token_limit_min"]
TOKEN_LIMIT_MAX = CONFIG["decider"]["token_limit_max"]

# 监控配置
MONITOR_INTERVAL = CONFIG["monitor"]["interval"]

@dataclass
class ServicePerformance:
    """服务性能指标数据类
    
    用于存储GPU或CPU服务的性能指标，包括延迟和吞吐量。
    ServicePerformance本身不区分GPU或CPU，区分是在使用时通过创建不同实例实现的。
    """
    avg_latency: float  # 平均延迟，单位毫秒
    throughput: float   # 吞吐量，单位tokens/s
