"""配置管理模块
负责加载和提供系统配置

Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any
from pydantic import BaseModel

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

@dataclass
class RequestState:
    """请求状态数据类
    
    用于在系统内部跟踪和管理请求的生命周期。
    使用标准库的@dataclass装饰器，而不是Pydantic的BaseModel，因为：
    1. 这是内部使用的数据结构，不需要输入验证
    2. dataclass更轻量高效，适合频繁创建和修改的内部状态对象
    3. 系统内部数据类型已知，不需要Pydantic的验证和转换功能
    """
    request_id: str          # 请求唯一标识符
    prompt: str              # 用户输入的提示文本
    max_tokens: int          # 最大生成token数
    temperature: float       # 生成随机性参数
    created_at: float        # 创建时间戳
    status: str              # 状态: pending, processing, completed, failed
    assigned_service: Optional[str] = None  # 分配的服务: gpu, cpu, or None
    response: Optional[Dict] = None         # 完成后的响应数据
    error: Optional[str] = None             # 如果失败，存储错误信息

class CompletionRequest(BaseModel):
    """完成请求模型
    
    用于API接收和验证用户的文本生成请求。
    继承自Pydantic的BaseModel，提供以下功能：
    1. 自动数据验证和类型转换
    2. 默认值处理
    3. 与FastAPI集成，用于请求验证和API文档生成
    4. 数据序列化与反序列化
    """
    model: str = DEFAULT_MODEL               # 使用的模型名称
    prompt: str                              # 输入提示文本
    max_tokens: int = DEFAULT_MAX_TOKENS     # 生成的最大token数
    temperature: float = DEFAULT_TEMPERATURE # 控制生成随机性
    request_id: Optional[str] = None         # 可选的请求ID
    prefill_then_swapout: bool = False       # 是否在prefill后立即交换出
    continue_decoding: Optional[str] = None  # 继续解码的指示
    active_token: Optional[int] = None       # 继续解码时的活动token位置
