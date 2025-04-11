"""配置管理模块
负责加载和提供系统配置

Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

from .default_config import DEFAULT_CONFIG

def load_config() -> Dict[str, Any]:
    """加载配置
    加载顺序:
    1. 默认配置 (default_config.py)
    2. 配置文件 (config/config.yaml)
    3. 环境变量 (APP_*)
    """
    config = DEFAULT_CONFIG.copy()
    
    # 1. 从配置文件加载
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    _deep_update_dict(config, yaml_config)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    # 2. 从环境变量加载覆盖
    for env_key, env_value in os.environ.items():
        if env_key.startswith("APP_"):
            parts = env_key[4:].lower().split("_")
            if len(parts) < 2:
                continue
                
            current = config
            for part in parts[:-1]:
                if part not in current:
                    break
                current = current[part]
            else:
                last_part = parts[-1]
                if last_part in current:
                    original_value = current[last_part]
                    # 类型转换
                    try:
                        if isinstance(original_value, bool):
                            current[last_part] = env_value.lower() in ("1", "true", "yes")
                        elif isinstance(original_value, int):
                            current[last_part] = int(env_value)
                        elif isinstance(original_value, float):
                            current[last_part] = float(env_value)
                        else:
                            current[last_part] = env_value
                    except ValueError:
                        current[last_part] = env_value
    
    return config

def _deep_update_dict(destination: Dict, source: Dict) -> None:
    """深度更新字典"""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            _deep_update_dict(destination[key], value)
        else:
            destination[key] = value

# 加载配置
CONFIG = load_config()

# 服务URL配置
PREFILL_URL = CONFIG["services"]["gpu"]["url"]
DECODE_URL = CONFIG["services"]["cpu"]["url"]
GPU_METRICS_URL = CONFIG["services"]["gpu"]["metrics_url"]
CPU_METRICS_URL = CONFIG["services"]["cpu"]["metrics_url"]

# 请求队列配置
MAX_QUEUE_SIZE = CONFIG["system"]["max_queue_size"]
REQUEST_TIMEOUT = CONFIG["system"]["request_timeout"]

# 模型配置
DEFAULT_MODEL = CONFIG["models"]["default"]
DEFAULT_MAX_TOKENS = CONFIG["models"]["params"]["max_tokens"]
DEFAULT_TEMPERATURE = CONFIG["models"]["params"]["temperature"]
DEFAULT_TEST_PROMPT = CONFIG["models"]["params"]["test_prompt"]
DEFAULT_TEST_TOKENS = CONFIG["models"]["params"]["test_tokens"]

@dataclass
class ServicePerformance:
    """服务性能指标数据类
    
    用于存储GPU或CPU服务的性能指标，包括延迟和吞吐量。
    ServicePerformance本身不区分GPU或CPU，区分是在使用时通过创建不同实例实现的。
    """
    avg_latency: float  # 平均延迟，单位毫秒
    throughput: float   # 吞吐量，单位tokens/s
