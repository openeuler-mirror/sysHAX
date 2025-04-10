# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
默认配置值模块
包含系统所有可配置项的默认值，作为配置的基础
这些是开发环境的安全默认值，生产环境应通过config.yaml或环境变量覆盖
"""

# 默认配置字典
DEFAULT_CONFIG = {
    # 服务配置 - 默认使用本地回环地址，适合本地开发
    "services": {
        "gpu": {
            "url": "http://localhost:7004/v1/completions",  # 本地GPU服务地址
            "metrics_url": "http://localhost:7004/metrics", # 本地GPU服务指标地址
        },
        "cpu": {
            "url": "http://localhost:7005/v1/completions",  # 本地CPU服务地址
            "metrics_url": "http://localhost:7005/metrics", # 本地CPU服务指标地址
        }
    },
    
    # 模型配置
    "models": {
        "default": "ds1.5b",  # 默认使用的模型
        "params": {
            "max_tokens": 100,      # 默认生成的最大token数
            "temperature": 0.7,     # 默认生成随机性参数
            "test_prompt": "这是一个标准测试语句。",  # 性能测试使用的提示语
            "test_tokens": 20       # 性能测试生成的token数量
        }
    },
    
    # 系统配置
    "system": {
        "max_queue_size": 100,  # 请求队列大小
        "request_timeout": 300,  # 请求超时时间(秒)
    }
} 