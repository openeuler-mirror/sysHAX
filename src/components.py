"""组件容器模块
提供一个简单的容器类，用于存储和访问系统核心组件实例。

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
class Components:
    """系统组件容器"""
    
    monitor = None  # 系统监控器实例
    decider = None  # 调度决策器实例
    performance_tester = None  # 性能测试器实例
    adaptive_decoder = None  # 自适应解码器实例
