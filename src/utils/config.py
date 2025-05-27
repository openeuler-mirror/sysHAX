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
Desc:sysHAX 配置管理模块
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
    except (OSError, FileNotFoundError):
        # 配置文件不存在或无法打开，尝试使用示例配置
        Logger.warning("请配置 config/config.yaml 文件，使用示例配置启动")
        fallback = base / "config.example.yaml"
        if not fallback.exists():
            # 示例配置也不存在，抛出文件未找到错误
            raise FileNotFoundError("未检测到 config/config.example.yaml 文件")
        # 加载示例配置
        with fallback.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except yaml.YAMLError as e:
        # YAML 解析错误
        raise yaml.YAMLError(f"配置文件解析失败: {e}")


# 加载配置
CONFIG = load_config()

# 常量赋值，如配置存在
try:
    # 服务主机和端口配置（供 workflow 拼接 URL）
    GPU_HOST = CONFIG["services"]["gpu"]["host"]
    GPU_PORT = CONFIG["services"]["gpu"]["port"]
    CPU_HOST = CONFIG["services"]["cpu"]["host"]
    CPU_PORT = CONFIG["services"]["cpu"]["port"]
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
