"""基准测试模块

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
from typing import Dict, Tuple, Optional
import time

from src.workflow import AdaptiveDecoder
from src.core.decider import SchedulingDecider
from src.core.monitor import SystemMonitor
from src.utils.logger import Logger
from src.utils.config import ServicePerformance
from src.utils.config import PREFILL_URL, DECODE_URL
from src.utils.config import DEFAULT_TEST_PROMPT, DEFAULT_TEST_TOKENS, DEFAULT_MODEL

class PerformanceTester:
    """性能测试工具，在启动时测量GPU和CPU服务的性能"""
    
    def __init__(self, system_monitor: SystemMonitor, decoder: AdaptiveDecoder):
        self.gpu_service_url = PREFILL_URL
        self.cpu_service_url = DECODE_URL
        
        # 使用传入的系统监控器实例
        self.system_monitor = system_monitor
        
        # 创建自适应解码器，用于性能测试
        self.adaptive_decoder = decoder
        
        # 测试参数 - 从配置中获取
        self.test_prompt = DEFAULT_TEST_PROMPT
        self.test_tokens = DEFAULT_TEST_TOKENS
        self.model_name = DEFAULT_MODEL
        
        # 性能测试结果
        self.gpu_performance = None
        self.cpu_performance = None
        # 性能比率 (GPU/CPU)
        self.performance_ratio = 1.0
    
    async def run_benchmarks(self):
        """运行简单的基准测试，快速测量CPU和GPU性能差异
        
        这是一个备用方法，实现与用户提供的代码类似的功能
        """
        Logger.info("开始基准测试...")
        
        try:
            # 如果没有设置adaptive_decoder，则创建临时调度决策器
            if self.adaptive_decoder is None:
                temp_decider = SchedulingDecider(system_monitor=self.system_monitor)
                self.adaptive_decoder = AdaptiveDecoder(
                    system_monitor=self.system_monitor,
                    scheduling_decider=temp_decider
                )
                Logger.info("为性能测试创建临时自适应解码器")
            
            # 0. 对测试数据进行prefill，准备decode数据
            decode_data = await self._prepare_test_data()
            if not decode_data:
                Logger.error("准备测试数据失败，无法继续进行基准测试")
                return
            
            # 1. 在decode测试前更新一次指标，获取基准值，强制刷新
            await self.system_monitor.update_metrics(force=True)
            Logger.info("-"*60)

            # 2. 测试CPU性能
            cpu_results = await self._test_performance(decode_data, device="CPU")
            
            if cpu_results:
                cpu_latency, final_cpu_throughput = cpu_results
            else:
                Logger.warning("CPU性能测试失败，使用默认值")
                cpu_latency, final_cpu_throughput = 2000.0, 1.0
            Logger.info("-"*60)

            # 3. 测试GPU性能
            decode_data["max_tokens"] = decode_data["max_tokens"] * 10
            gpu_results = await self._test_performance(decode_data, device="GPU")
            
            if gpu_results:
                gpu_latency, final_gpu_throughput = gpu_results
            else:
                Logger.warning("GPU性能测试失败，使用默认值")
                gpu_latency, final_gpu_throughput = 1000.0, 10.0
            Logger.info("-"*60)
            # 4. 保存测试结果并计算性能比率
            self._save_performance_results(
                gpu_latency, final_gpu_throughput, 
                cpu_latency, final_cpu_throughput)
            
            result_msg = f"测试结果: \n" + \
                f"GPU耗时={gpu_latency:.2f}ms, GPU吞吐量={final_gpu_throughput:.2f}tokens/s\n" + \
                f"CPU耗时={cpu_latency:.2f}ms, CPU吞吐量={final_cpu_throughput:.2f}tokens/s\n" + \
                f"GPU/CPU性能比={self.performance_ratio:.2f}x"
            
            Logger.info(result_msg)
            
        except Exception as e:
            Logger.error(f"基准测试失败: {str(e)}", exc_info=True)
    
    async def _prepare_test_data(self) -> Dict:
        """准备测试数据，在GPU上执行prefill
        
        Returns:
            Dict: 包含decode所需参数的字典
        """
        try:
            Logger.info("准备prefill测试数据")
            
            # 在GPU上执行prefill
            prefill_data = {
                "model": self.model_name,
                "prompt": self.test_prompt,
                "max_tokens": 2,
                "temperature": 0,  # 基准测试固定使用temperature=0
                "prefill_then_swapout": True
            }
            Logger.info(f"执行prefill请求，prompt长度={len(self.test_prompt)}")

            prefill_result = await self.adaptive_decoder.prefill_request(prefill_data)
            
            if prefill_result.get("status") == "busy":
                Logger.warning("系统繁忙，无法执行基准测试")
                return {}  # 返回空字典表示测试无法进行
                
            completion_id = prefill_result["completion_id"]
            active_token = prefill_result["active_token"]
            Logger.info(f"prefill完成，获取到completion_id={completion_id}, active_token={active_token}")
            
            # 准备decode数据
            decode_data = {
                "model": self.model_name,
                "prompt": self.test_prompt,
                "max_tokens": self.test_tokens,
                "temperature": 0,  # 基准测试固定使用temperature=0
                "continue_decoding": f"{completion_id}-0",
                "active_token": active_token
            }
            
            Logger.info(f"准备好decode数据，model={self.model_name}, max_tokens={self.test_tokens}")
            
            return decode_data
        except Exception as e:
            Logger.error(f"准备测试数据失败: {e}", exc_info=True)
            return {}
    
    async def _test_performance(self, decode_data: Dict, device: str) -> Optional[Tuple[float, float]]:
        """测试硬件吞吐量"""
        try:
            Logger.info(f"测试{device} decode性能")
            
            # 简化参数记录
            if "prompt" in decode_data:
                prompt_length = len(decode_data["prompt"])
                Logger.info(f"{device} decode测试: prompt长度={prompt_length}, max_tokens={decode_data.get('max_tokens', 0)}")
            
            # 执行GPU解码并测量时间
            start_time = time.time()
            if device == "GPU":
                decode_result = await self.adaptive_decoder.decode_on_gpu(decode_data)
            else:
                decode_result = await self.adaptive_decoder.decode_on_cpu(decode_data)
            decode_time = time.time() - start_time

            # 计算生成时间（毫秒）
            latency = decode_time * 1000
            
            # 计算实际吞吐量（统一用 usage.completion_tokens）
            actual_throughput = 0.0
            usage = decode_result["response"].get("usage", {})
            actual_tokens = usage.get("completion_tokens", 0)
            if decode_time > 0 and actual_tokens > 0:
                actual_throughput = actual_tokens / decode_time
                Logger.info(f"{device}实际吞吐量: {actual_throughput:.2f}tokens/s")
            
            await self.system_monitor.update_metrics(force=True)
            updated_metrics = self.system_monitor.get_all_metrics()
            throughput = updated_metrics[device.lower()]['generation_throughput']
            final_throughput = throughput
            
            # 记录原生指标与实际计算值的差异（用于验证指标准确性）
            if throughput > 0 and actual_throughput > 0:
                diff_percent = abs(throughput - actual_throughput) / actual_throughput * 100
                Logger.info(f"{device}吞吐量对比: " + \
                            f"vLLM指标={throughput:.2f}tokens/s, " + \
                            f"实际计算={actual_throughput:.2f}tokens/s, " + \
                            f"差异={diff_percent:.1f}%")
                
                # 如果差异过大，记录警告
                if diff_percent > 30:
                    Logger.warning(f"{device}吞吐量指标与实际计算值差异较大({diff_percent:.1f}%)，可能影响决策准确性")
            
            # 如果仍然无法获取吞吐量，返回None让外部使用默认值
            if final_throughput <= 0:
                Logger.warning(f"无法获取{device}吞吐量指标")
                return None
            
            if device == "GPU":
                latency = latency / 10

            Logger.info(f"{device}性能结果: 解码耗时={latency:.2f}ms, 吞吐量={final_throughput:.2f}tokens/s")
            return latency, final_throughput
            
        except Exception as e:
            Logger.error(f"测试{device}性能失败: {e}", exc_info=True)
            return None
    
    def _save_performance_results(self, 
        gpu_latency: float, gpu_throughput: float, 
        cpu_latency: float, cpu_throughput: float
    ):
        """保存性能测试结果并计算性能比率
        
        性能比率计算公式:
        1. 吞吐量比率 = GPU吞吐量 / CPU吞吐量
           (衡量GPU每秒处理的token数量与CPU相比的倍数)
        2. 延迟比率 = CPU延迟 / GPU延迟
           (衡量CPU响应时间与GPU相比的倍数)
        3. 综合性能比率 = (吞吐量比率 + 延迟比率) / 2
           (取两个比率的平均值，综合考虑速度和响应时间)
        
        比率越高，表示GPU相对于CPU的性能优势越大。
        
        Args:
            gpu_latency: GPU服务延迟(毫秒)
            gpu_throughput: GPU服务吞吐量(tokens/秒)
            cpu_latency: CPU服务延迟(毫秒)
            cpu_throughput: CPU服务吞吐量(tokens/秒)
        """
        # 保存性能测试结果
        self.gpu_performance = ServicePerformance(gpu_latency, gpu_throughput)
        self.cpu_performance = ServicePerformance(cpu_latency, cpu_throughput)
        
        # 计算性能比率
        if self.cpu_performance.throughput > 0:
            throughput_ratio = self.gpu_performance.throughput / self.cpu_performance.throughput
            latency_ratio = self.cpu_performance.avg_latency / self.gpu_performance.avg_latency
            # 综合考虑吞吐量和延迟
            self.performance_ratio = (throughput_ratio + latency_ratio) / 2
            Logger.info(f"GPU/CPU性能比: {self.performance_ratio:.2f}x")
        else:
            self.performance_ratio = 10.0  # 默认值
            Logger.info(f"使用默认性能比率: {self.performance_ratio}")
    
    def get_performance_summary(self) -> Dict:
        """获取性能测试结果摘要"""
        if not self.gpu_performance or not self.cpu_performance:
            return {"status": "未完成测试"}
            
        return {
            "gpu": {
                "latency_ms": self.gpu_performance.avg_latency,
                "throughput": self.gpu_performance.throughput
            },
            "cpu": {
                "latency_ms": self.cpu_performance.avg_latency,
                "throughput": self.cpu_performance.throughput
            },
            "performance_ratio": self.performance_ratio
        }
