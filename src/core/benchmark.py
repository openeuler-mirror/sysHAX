"""基准测试模块

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""
import asyncio
from typing import Dict, Tuple, Optional
import time

from src.workflow import AdaptiveDecoder
from src.core.decider import SchedulingDecider
from src.core.monitor import SystemMonitor
from src.utils.logger import Logger
from src.utils.config import PREFILL_URL, DECODE_URL, GPU_METRICS_URL, CPU_METRICS_URL, ServicePerformance

log = Logger

class PerformanceTester:
    """性能测试工具，在启动时测量GPU和CPU服务的性能"""
    
    def __init__(self, system_monitor=None):
        self.gpu_service_url = PREFILL_URL
        self.cpu_service_url = DECODE_URL
        
        # 使用传入的系统监控器实例，或创建一个新的
        self.system_monitor = system_monitor or SystemMonitor(GPU_METRICS_URL, CPU_METRICS_URL)
        
        # 创建自适应解码器，用于性能测试
        self.adaptive_decoder = None
        
        # 测试参数
        self.test_prompt = "这是一个标准测试语句。"  # 更新测试提示
        self.test_tokens = 20   # 减少生成的token数量，缩短测试时间
        
        # 性能测试结果
        self.gpu_performance = None
        self.cpu_performance = None
        # 性能比率 (GPU/CPU)
        self.performance_ratio = 1.0
    
    def set_adaptive_decoder(self, decoder: AdaptiveDecoder):
        """设置自适应解码器实例
        
        Args:
            decoder: 自适应解码器实例
        """
        self.adaptive_decoder = decoder
    
    async def run_benchmarks(self):
        """运行简单的基准测试，快速测量CPU和GPU性能差异
        
        这是一个备用方法，实现与用户提供的代码类似的功能
        """
        log.info("开始基准测试...")
        
        try:
            # 如果没有设置adaptive_decoder，则创建临时调度决策器
            if self.adaptive_decoder is None:
                temp_decider = SchedulingDecider(system_monitor=self.system_monitor)
                self.adaptive_decoder = AdaptiveDecoder(
                    system_monitor=self.system_monitor,
                    scheduling_decider=temp_decider
                )
                log.info("为性能测试创建临时自适应解码器")
            
            # 0. 在测试前更新一次指标，获取基准值
            await self.system_monitor.update_metrics()
            
            # 1. 准备测试数据
            decode_data = await self._prepare_test_data(self.test_prompt)
            
            # 2. 测试GPU性能
            gpu_results = await self._test_gpu_performance(decode_data)
            
            if not gpu_results:
                log.warning("GPU性能测试失败，使用默认值")
                gpu_latency, final_gpu_throughput = 1000.0, 10.0
            else:
                gpu_latency, final_gpu_throughput = gpu_results
            
            # 3. 测试CPU性能
            cpu_results = await self._test_cpu_performance(decode_data)
            
            if not cpu_results:
                log.warning("CPU性能测试失败，使用默认值")
                cpu_latency, final_cpu_throughput = 2000.0, 1.0
            else:
                cpu_latency, final_cpu_throughput = cpu_results
            
            # 4. 保存测试结果并计算性能比率
            self._save_performance_results(
                gpu_latency, final_gpu_throughput, 
                cpu_latency, final_cpu_throughput)
            
            result_msg = f"测试结果: \n" + \
                f"GPU延迟={gpu_latency:.2f}ms, GPU吞吐量={final_gpu_throughput:.2f}tokens/s\n" + \
                f"CPU延迟={cpu_latency:.2f}ms, CPU吞吐量={final_cpu_throughput:.2f}tokens/s\n" + \
                f"GPU/CPU性能比={self.performance_ratio:.2f}x"
            
            log.info(result_msg)
            
        except Exception as e:
            log.error(f"基准测试失败: {str(e)}", exc_info=True)
    
    async def _prepare_test_data(self, prompt: str) -> Dict:
        """准备测试数据，在GPU上执行prefill
        
        Args:
            prompt: 测试用的prompt文本
            
        Returns:
            Dict: 包含decode所需参数的字典
        """
        try:
            log.info("准备prefill测试数据")
            
            # 在GPU上执行prefill
            prefill_data = {
                "model": "ds1.5b",
                "prompt": prompt,
                "max_tokens": 2,
                "temperature": 0,
                "prefill_then_swapout": True
            }
            log.info(f"执行prefill请求，prompt长度={len(prompt)}")

            prefill_result = await self.adaptive_decoder.prefill_request(prefill_data)
            completion_id = prefill_result["completion_id"]
            active_token = prefill_result["active_token"]
            log.info(f"prefill完成，获取到completion_id={completion_id}, active_token={active_token}")
            
            # 准备decode数据
            decode_data = {
                "model": "ds1.5b",
                "prompt": prompt,
                "max_tokens": self.test_tokens,
                "temperature": 0,
                "continue_decoding": f"{completion_id}-0",
                "active_token": active_token
            }
            
            log.info(f"准备好decode数据，max_tokens={self.test_tokens}")
            
            return decode_data
        except Exception as e:
            log.error(f"准备测试数据失败: {e}", exc_info=True)
            raise
    
    async def _test_gpu_performance(self, decode_data: Dict) -> Optional[Tuple[float, float]]:
        """测试GPU性能"""
        try:
            log.info("测试GPU decode性能")
            
            # 记录decode_data的详细信息
            decode_data_copy = decode_data.copy()
            if "prompt" in decode_data_copy:
                prompt_length = len(decode_data_copy["prompt"])
                decode_data_copy["prompt"] = f"[长度: {prompt_length}]"
            log.info(f"GPU decode测试使用的参数: {decode_data_copy}")
            
            # 在性能测试前先刷新一次指标，确保获取最新值
            await self.system_monitor.update_metrics()
            fresh_gpu_metrics = self.system_monitor.get_gpu_metrics()
            fresh_start_tokens = fresh_gpu_metrics['decode_tokens']
            
            # 使用time.time()直接测量开始和结束时间
            start_time = time.time()
            
            # 执行GPU解码
            gpu_decode_result = await self.adaptive_decoder.decode_on_gpu(decode_data)
            
            # 记录结束时间
            end_time = time.time()
            decode_time = end_time - start_time
            
            # 计算延迟（毫秒）
            gpu_latency = decode_time * 1000
            
            # 获取GPU token指标 (优先使用)
            await asyncio.sleep(2)  # 等待指标更新
            await self.system_monitor.update_metrics()
            updated_gpu_metrics = self.system_monitor.get_gpu_metrics()
            gpu_decode_tokens_diff = updated_gpu_metrics['decode_tokens'] - fresh_start_tokens
            
            # 如果差异非常小，可能指标还没更新，再多等待一下
            if gpu_decode_tokens_diff < 5 and self.test_tokens > 10:
                log.info(f"指标差异较小({gpu_decode_tokens_diff})，等待更新...")
                await asyncio.sleep(3)
                await self.system_monitor.update_metrics()
                updated_gpu_metrics = self.system_monitor.get_gpu_metrics()
                gpu_decode_tokens_diff = updated_gpu_metrics['decode_tokens'] - fresh_start_tokens
            
            # 计算基于指标的吞吐量
            metrics_throughput = 0
            if decode_time > 0 and gpu_decode_tokens_diff > 0:
                metrics_throughput = gpu_decode_tokens_diff / decode_time
                log.info(f"指标显示生成了{gpu_decode_tokens_diff}个tokens, 指标吞吐量={metrics_throughput:.2f}tokens/s")
            
            # 同时记录文本长度信息作为参考
            if (gpu_decode_result["response"].get("choices") and 
                gpu_decode_result["response"]["choices"][0].get("text")):
                generated_text = gpu_decode_result["response"]["choices"][0]["text"]
                # 仅作为参考信息记录
                log.info(f"生成的文本长度: {len(generated_text)}字符, 文本内容: {generated_text[:50]}")
            
            # 使用指标数据作为最终吞吐量计算依据
            if gpu_decode_tokens_diff > 0:
                final_gpu_throughput = metrics_throughput
                log.info(f"使用指标计算的吞吐量: {final_gpu_throughput:.2f}tokens/s")
            else:
                # 备用方案：使用预期的token数
                estimated_tokens = self.test_tokens
                time_based_throughput = estimated_tokens / decode_time if decode_time > 0 else 0
                final_gpu_throughput = time_based_throughput
                log.info(f"无法获取指标数据，使用预期token数({estimated_tokens})计算吞吐量: {final_gpu_throughput:.2f}tokens/s")
            
            log.info(f"GPU性能: 延迟={gpu_latency:.2f}ms, 吞吐量={final_gpu_throughput:.2f}tokens/s")
            
            return gpu_latency, final_gpu_throughput
        except Exception as e:
            log.error(f"测试GPU性能失败: {e}", exc_info=True)
            return None
    
    async def _test_cpu_performance(self, decode_data: Dict) -> Optional[Tuple[float, float]]:
        """测试CPU性能"""
        try:
            log.info("测试CPU decode性能")
            
            # 记录decode_data的详细信息
            decode_data_copy = decode_data.copy()
            if "prompt" in decode_data_copy:
                prompt_length = len(decode_data_copy["prompt"])
                decode_data_copy["prompt"] = f"[长度: {prompt_length}]"
            log.info(f"CPU decode测试使用的参数: {decode_data_copy}")
            
            # 在性能测试前先刷新一次指标，确保获取最新值
            await self.system_monitor.update_metrics()
            fresh_cpu_metrics = self.system_monitor.get_cpu_metrics()
            fresh_start_tokens = fresh_cpu_metrics['decode_tokens']
            
            # 使用time.time()直接测量开始和结束时间
            start_time = time.time()
            
            # 执行CPU解码 - 注意：CPU解码可能使用了时间限制
            cpu_decode_result = await self.adaptive_decoder.decode_on_cpu(decode_data)
            
            # 记录结束时间
            end_time = time.time()
            decode_time = end_time - start_time
            
            # 计算延迟（毫秒）
            cpu_latency = decode_time * 1000
            
            # 获取CPU token指标 (优先使用)
            await asyncio.sleep(2)  # 等待指标更新
            await self.system_monitor.update_metrics()
            updated_cpu_metrics = self.system_monitor.get_cpu_metrics()
            cpu_decode_tokens_diff = updated_cpu_metrics['decode_tokens'] - fresh_start_tokens
            
            # 如果差异非常小，可能指标还没更新，再多等待一下
            if cpu_decode_tokens_diff < 5 and self.test_tokens > 10:
                log.info(f"指标差异较小({cpu_decode_tokens_diff})，等待更新...")
                await asyncio.sleep(3)
                await self.system_monitor.update_metrics()
                updated_cpu_metrics = self.system_monitor.get_cpu_metrics()
                cpu_decode_tokens_diff = updated_cpu_metrics['decode_tokens'] - fresh_start_tokens
            
            # 计算基于指标的吞吐量
            metrics_throughput = 0
            if decode_time > 0 and cpu_decode_tokens_diff > 0:
                metrics_throughput = cpu_decode_tokens_diff / decode_time
                log.info(f"指标显示生成了{cpu_decode_tokens_diff}个tokens, 指标吞吐量={metrics_throughput:.2f}tokens/s")
            
            # 同时记录文本长度信息作为参考
            if (cpu_decode_result["response"].get("choices") and 
                cpu_decode_result["response"]["choices"][0].get("text")):
                generated_text = cpu_decode_result["response"]["choices"][0]["text"]
                # 仅作为参考信息记录
                log.info(f"生成的文本长度: {len(generated_text)}字符, 文本内容: {generated_text[:50]}")
            
            # 使用指标数据作为最终吞吐量计算依据
            if cpu_decode_tokens_diff > 0:
                final_cpu_throughput = metrics_throughput
                log.info(f"使用指标计算的吞吐量: {final_cpu_throughput:.2f}tokens/s")
            else:
                # 备用方案：使用默认的5 tokens/s估算
                estimated_throughput = 5.0  # 基于测试，CPU的真实吞吐量约为5 tokens/s
                log.info(f"无法获取指标数据，使用默认CPU吞吐量: {estimated_throughput:.2f}tokens/s")
                final_cpu_throughput = estimated_throughput
            
            log.info(f"CPU性能: 延迟={cpu_latency:.2f}ms, 吞吐量={final_cpu_throughput:.2f}tokens/s")
            
            return cpu_latency, final_cpu_throughput
        except Exception as e:
            log.error(f"测试CPU性能失败: {e}", exc_info=True)
            return None
    
    def _compare_throughput_methods(self, service_type: str, time_throughput: float, metrics_throughput: float) -> float:
        """比较两种吞吐量计算方法，并返回最终使用的值"""
        if time_throughput > 0 and metrics_throughput > 0:
            diff_percent = abs(time_throughput - metrics_throughput) / time_throughput * 100
            if diff_percent > 50:
                log.warning(
                    f"{service_type}吞吐量计算差异较大: 时间计算={time_throughput:.2f}tokens/s, " + 
                    f"指标计算={metrics_throughput:.2f}tokens/s, 差异={diff_percent:.2f}%"
                )
                log.info(f"使用时间计算的吞吐量: {time_throughput:.2f}tokens/s")
                return time_throughput
        
        # 正常情况下使用指标计算的吞吐量，如果有的话
        return metrics_throughput if metrics_throughput > 0 else time_throughput
    
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
        self.gpu_performance = ServicePerformance(gpu_latency, gpu_throughput, 1.0)
        self.cpu_performance = ServicePerformance(cpu_latency, cpu_throughput, 1.0)
        
        # 计算性能比率
        if self.cpu_performance.throughput > 0:
            throughput_ratio = self.gpu_performance.throughput / self.cpu_performance.throughput
            latency_ratio = self.cpu_performance.avg_latency / self.gpu_performance.avg_latency
            # 综合考虑吞吐量和延迟
            self.performance_ratio = (throughput_ratio + latency_ratio) / 2
            log.info(f"GPU/CPU性能比: {self.performance_ratio:.2f}x")
        else:
            self.performance_ratio = 10.0  # 默认值
            log.info(f"使用默认性能比率: {self.performance_ratio}")
    
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
