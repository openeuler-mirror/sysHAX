"""
配置管理模块

负责加载和提供系统配置

Copyright (c) 2025 Huawei Technologies Co., Ltd.
sysHAX is licensed under Mulan PSL v2. See LICENSE for details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import Logger


def load_config() -> dict[str, Any]:
    """
    加载配置

    优先读取 config/config.yaml，若不存在则读取 config/config.example.yaml，均不存在则报错
    """
    base = Path(__file__).parent.parent.parent / "config"
    primary = base / "config.yaml"
    try:
        with primary.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except (OSError, yaml.YAMLError, FileNotFoundError):
        Logger.warning("请配置 config/config.yaml 文件")
        fallback = base / "config.example.yaml"
        assert fallback.exists(), "未检测到 config/config.example.yaml 文件"
        with fallback.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}


# 加载配置
CONFIG = load_config()

# 常量赋值，如配置存在
try:
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
except KeyError:
    # 部分配置缺失时跳过
    pass


@dataclass
class ServicePerformance:
    """
    服务性能指标数据类

    用于存储GPU或CPU服务的性能指标，包括延迟和吞吐量。
    ServicePerformance本身不区分GPU或CPU，区分是在使用时通过创建不同实例实现的。
    """

    avg_latency: float  # 平均延迟，单位毫秒
    throughput: float  # 吞吐量，单位tokens/s
