"""
配置管理模块

负责加载和提供系统配置

Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import Logger


def load_config() -> dict[str, Any]:
    """
    加载配置

    从配置文件config/config.yaml加载
    """
    config = {}

    # 从配置文件加载
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = yaml_config
        except (OSError, yaml.YAMLError) as e:
            Logger.error(f"加载配置文件失败: {e}")
    else:
        msg = f"配置文件不存在: {config_path}"
        raise FileNotFoundError(msg)

    return config

# 加载配置
CONFIG = load_config()

# 服务URL配置
GPU_URL = CONFIG["services"]["gpu"]["url"]
CPU_URL = CONFIG["services"]["cpu"]["url"]
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
    """
    服务性能指标数据类

    用于存储GPU或CPU服务的性能指标，包括延迟和吞吐量。
    ServicePerformance本身不区分GPU或CPU，区分是在使用时通过创建不同实例实现的。
    """

    avg_latency: float  # 平均延迟，单位毫秒
    throughput: float  # 吞吐量，单位tokens/s


@dataclass
class DecodeData:
    """
    Decode 阶段的数据格式

    保存用户的原始请求数据和Decode阶段的特有参数。
    """

    # 用户的原始请求数据
    data: dict[str, Any]

    # Decode相关参数
    continue_decoding: str = ""
    active_token: int = 0
    completion_id: str = ""

    # 性能和调度信息
    step: int = 0
    latency: float = 0.0
    throughput: float = 0.0
    device: str = "CPU"
    token_limit: int = 0

    def get_request_data(self) -> dict[str, Any]:
        """获取用于API请求的数据"""
        request_data = self.data.copy()

        # 添加decode特有参数
        if self.continue_decoding:
            request_data["continue_decoding"] = self.continue_decoding

        if self.active_token:
            request_data["active_token"] = self.active_token

        # 添加调度相关参数
        if self.device:
            request_data["device"] = self.device

        if self.token_limit > 0:
            request_data["token_limit"] = self.token_limit

        return request_data
