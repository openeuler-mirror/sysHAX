"""调度决策模块

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""
from typing import Dict

from src.core.monitor import SystemMonitor
from src.utils.logger import Logger

log = Logger

class SchedulingDecider:
    """调度决策类，根据系统指标决定在何处执行解码任务
    
    职责：
    1. 根据SystemMonitor提供的指标做出调度决策
    2. 提供设备和token限制的决策
    """
    
    def __init__(self, system_monitor: SystemMonitor):
        """初始化调度决策器
        
        Args:
            system_monitor: 系统监控器实例
        """
        self.system_monitor: SystemMonitor = system_monitor
    
    async def make_scheduling_decision(self) -> Dict:
        """做出调度决策，返回设备类型和token限制
        
        Returns:
            result: {
                "device": 设备类型,None表示系统繁忙
                "token_limit": token限制,0表示不限制
            }
        """
        # 更新系统指标
        await self.system_monitor.update_metrics()
        
        result = {"device": "GPU", "token_limit": 0}
        return result