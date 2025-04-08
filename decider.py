from typing import Dict, Tuple

from monitor import SystemMonitor
from config import DEFAULT_SCHEDULING_PARAMS
from logger import Logger

log = Logger

class SchedulingDecider:
    """调度决策类，根据系统指标决定在何处执行解码任务
    
    职责：
    1. 根据SystemMonitor提供的指标做出调度决策
    2. 维护调度相关参数
    3. 提供设备和token限制的决策
    """
    
    def __init__(self, system_monitor: SystemMonitor):
        """初始化调度决策器
        
        Args:
            system_monitor: 系统监控器实例
        """
        self.system_monitor: SystemMonitor = system_monitor
        
        # 从配置中加载默认参数
        # 调度阈值配置
        self.kv_cache_threshold = DEFAULT_SCHEDULING_PARAMS["kv_cache_threshold"]
        self.running_reqs_threshold = DEFAULT_SCHEDULING_PARAMS["running_reqs_threshold"]
        self.waiting_reqs_threshold = DEFAULT_SCHEDULING_PARAMS["waiting_reqs_threshold"]
        
        # 负载阈值
        self.high_load_threshold = DEFAULT_SCHEDULING_PARAMS["high_load_threshold"]
        self.medium_load_threshold = DEFAULT_SCHEDULING_PARAMS["medium_load_threshold"]
        self.low_load_threshold = DEFAULT_SCHEDULING_PARAMS["low_load_threshold"]
        
        # 系统繁忙判断阈值
        self.system_busy_threshold = DEFAULT_SCHEDULING_PARAMS["system_busy_threshold"]
        
        log.info("初始化调度决策器: 使用简化版本，目前默认返回GPU")
    
    def update_parameters(self, 
                          kv_cache_threshold: float = None, 
                          running_reqs_threshold: int = None,
                          waiting_reqs_threshold: int = None,
                          load_thresholds: Dict[str, float] = None,
                          system_busy_threshold: float = None):
        """更新调度参数
        
        Args:
            kv_cache_threshold: KV缓存使用率阈值
            running_reqs_threshold: 运行中请求数阈值
            waiting_reqs_threshold: 等待中请求数阈值
            load_thresholds: 负载阈值字典，包含'high'、'medium'、'low'键
            system_busy_threshold: 系统繁忙阈值
        """
        if kv_cache_threshold is not None:
            self.kv_cache_threshold = kv_cache_threshold
            
        if running_reqs_threshold is not None:
            self.running_reqs_threshold = running_reqs_threshold
            
        if waiting_reqs_threshold is not None:
            self.waiting_reqs_threshold = waiting_reqs_threshold
            
        if load_thresholds is not None:
            self.high_load_threshold = load_thresholds.get("high", self.high_load_threshold)
            self.medium_load_threshold = load_thresholds.get("medium", self.medium_load_threshold)
            self.low_load_threshold = load_thresholds.get("low", self.low_load_threshold)
            
        if system_busy_threshold is not None:
            self.system_busy_threshold = system_busy_threshold
        
        log.info(f"已更新调度参数: GPU高负载={self.high_load_threshold}, " +
                 f"系统繁忙阈值={self.system_busy_threshold}")
    
    async def make_scheduling_decision(self) -> Tuple[str, int]:
        """做出调度决策，返回设备类型和token限制
        
        Returns:
            Tuple[str, int]: 
                - 第一个值为设备类型，"GPU"或"CPU"，None表示系统繁忙
                - 第二个值为token限制，0表示不限制
        """
        # 更新系统指标
        await self.system_monitor.update_metrics()
        
        # 目前简单实现，直接返回GPU且不限制token
        return ("GPU", 0) 