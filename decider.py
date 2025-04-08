from typing import Dict, List, Tuple

from monitor import SystemMonitor
from config import DEFAULT_SCHEDULING_PARAMS
from logger import Logger

log = Logger

class SchedulingDecider:
    """调度决策类，根据系统指标决定是否使用GPU进行decode
    
    职责：
    1. 根据SystemMonitor提供的指标做出调度决策
    2. 维护调度相关参数
    3. 提供可配置的决策逻辑
    """
    
    def __init__(self, system_monitor: SystemMonitor):
        """初始化调度决策器
        
        Args:
            system_monitor: 系统监控器实例
        """
        self.system_monitor = system_monitor
        
        # 从配置中加载默认参数
        # 调度阈值配置 - 仍然使用KV缓存阈值和请求数阈值计算GPU负载
        self.kv_cache_threshold = DEFAULT_SCHEDULING_PARAMS["kv_cache_threshold"]
        self.running_reqs_threshold = DEFAULT_SCHEDULING_PARAMS["running_reqs_threshold"]
        self.waiting_reqs_threshold = DEFAULT_SCHEDULING_PARAMS["waiting_reqs_threshold"]
        
        # 负载阈值 - 使用high_load_threshold判断GPU是否繁忙
        self.high_load_threshold = DEFAULT_SCHEDULING_PARAMS["high_load_threshold"]
        self.medium_load_threshold = DEFAULT_SCHEDULING_PARAMS["medium_load_threshold"]
        self.low_load_threshold = DEFAULT_SCHEDULING_PARAMS["low_load_threshold"]
        
        # 系统繁忙判断阈值
        self.system_busy_threshold = DEFAULT_SCHEDULING_PARAMS["system_busy_threshold"]
        
        # 历史负载状态 (用于平滑决策)
        self.gpu_load_history: List[float] = []
        self.cpu_load_history: List[float] = []
        self.history_size = 5
        
        log.info("初始化调度决策器，使用新的调度策略：只在GPU繁忙且CPU空闲时使用CPU")
    
    def update_parameters(self, 
                          kv_cache_threshold: float = None, 
                          running_reqs_threshold: int = None,
                          waiting_reqs_threshold: int = None,
                          load_thresholds: Dict[str, float] = None,
                          system_busy_threshold: float = None):
        """更新调度参数"""
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
    
    def calculate_gpu_load_score(self) -> float:
        """计算GPU负载分数（0-1之间，越高表示负载越大）"""
        gpu_metrics = self.system_monitor.get_gpu_metrics()
        
        # 将各个指标归一化并加权
        kv_cache_factor = min(1.0, gpu_metrics['gpu_cache_usage'] / self.kv_cache_threshold)
        running_reqs_factor = min(1.0, gpu_metrics['num_running'] / self.running_reqs_threshold)
        waiting_reqs_factor = min(1.0, gpu_metrics['num_waiting'] / self.waiting_reqs_threshold)
        
        # 综合负载分数（可以根据实际情况调整权重）
        load_score = 0.5 * kv_cache_factor + 0.3 * running_reqs_factor + 0.2 * waiting_reqs_factor
        
        # 更新历史记录
        self.gpu_load_history.append(load_score)
        if len(self.gpu_load_history) > self.history_size:
            self.gpu_load_history.pop(0)
            
        return load_score
    
    def get_smooth_gpu_load_score(self) -> float:
        """获取平滑后的GPU负载分数
        
        使用指数加权移动平均方法对历史负载进行平滑处理，
        减少瞬时负载波动对调度决策的影响。
        
        Returns:
            float: 0.0-1.0之间的平滑负载分数，越高表示负载越大
        """
        if not self.gpu_load_history:
            return 0.0
        
        # 使用指数加权移动平均
        weights = [0.6 ** i for i in range(len(self.gpu_load_history))]
        weights.reverse()  # 最近的数据权重更大
        
        weighted_sum = sum(w * score for w, score in zip(weights, self.gpu_load_history))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def is_cpu_completely_idle(self) -> bool:
        """检查CPU是否完全空闲（无运行中或等待中的请求）
        
        Returns:
            bool: 如果CPU完全空闲返回True，否则返回False
        """
        cpu_metrics = self.system_monitor.get_cpu_metrics()
        return cpu_metrics['num_running'] == 0 and cpu_metrics['num_waiting'] == 0
    
    def is_gpu_busy(self) -> bool:
        """检查GPU是否繁忙（负载超过high_load_threshold）
        
        Returns:
            bool: 如果GPU繁忙返回True，否则返回False
        """
        gpu_load = self.get_smooth_gpu_load_score()
        return gpu_load >= self.high_load_threshold
    
    async def check_gpu_status(self, test_mode = False) -> Tuple[bool, float]:
        """检查GPU是否繁忙，决定是否使用CPU
        
        Args:
            test_mode: 测试模式，如果为True则强制返回GPU繁忙
            
        Returns:
            Tuple[bool, float]: (是否应使用CPU, GPU负载分数)
                - 第一个值为True表示应使用CPU (即GPU繁忙且CPU空闲)
                - 第二个值为GPU负载分数，用于记录和分析
        """
        # 测试模式：强制返回GPU繁忙
        if test_mode:
            log.info("测试模式：强制将GPU状态设为繁忙")
            return True, 1.0
            
        # 更新系统指标
        await self.system_monitor.update_metrics()
        
        # 计算GPU负载
        self.calculate_gpu_load_score()
        gpu_load = self.get_smooth_gpu_load_score()
        
        # 判断GPU是否繁忙
        gpu_busy = self.is_gpu_busy()
        
        # 判断CPU是否空闲
        cpu_idle = self.is_cpu_completely_idle()
        
        # 只有当GPU繁忙且CPU空闲时才使用CPU
        use_cpu = gpu_busy and cpu_idle
        
        if use_cpu:
            log.debug(f"调度决策: GPU繁忙(负载={gpu_load:.2f})且CPU空闲，使用CPU")
        else:
            if gpu_busy:
                log.debug(f"调度决策: GPU繁忙(负载={gpu_load:.2f})但CPU不空闲，使用GPU")
            else:
                log.debug(f"调度决策: GPU空闲(负载={gpu_load:.2f})，使用GPU")
        
        return use_cpu, gpu_load
    
    async def make_scheduling_decision(self, prompt: str = "") -> Tuple[bool, bool]:
        """做出完整的调度决策
        
        新策略实现:
        1. 只有当GPU繁忙且CPU完全空闲时才使用CPU
        2. 其他情况都使用GPU
        
        Args:
            prompt: 请求的prompt文本
            
        Returns:
            Tuple[bool, bool]：第一个值表示是否使用GPU，第二个值表示系统是否繁忙
            - (True, False): 使用GPU，系统不繁忙
            - (False, False): 使用CPU，系统不繁忙
            - (X, True): 系统繁忙，无法处理请求
        """
        # 先更新指标
        await self.system_monitor.update_metrics()
        
        # 计算GPU负载并检查是否繁忙
        self.calculate_gpu_load_score()
        gpu_busy = self.is_gpu_busy()
        
        # 检查CPU是否完全空闲
        cpu_idle = self.is_cpu_completely_idle()
        
        # 新策略：当GPU繁忙且CPU完全空闲时，使用CPU
        if gpu_busy and cpu_idle:
            log.info(f"GPU繁忙(负载>{self.high_load_threshold})且CPU完全空闲，分配任务给CPU")
            return False, False  # 使用CPU且不繁忙
        
        # 检查系统是否处于极度繁忙状态
        gpu_load = self.get_smooth_gpu_load_score()
        if gpu_load > self.system_busy_threshold:
            log.warning(f"GPU负载过高: {gpu_load:.2f}，系统繁忙")
            return True, True  # 系统繁忙，但方向是GPU
        
        # 默认情况：使用GPU
        log.info("使用GPU进行decode" + 
               (", GPU负载正常" if not gpu_busy else ", 虽然GPU繁忙但CPU不空闲"))
        return True, False 