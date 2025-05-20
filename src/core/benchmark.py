"""基准测试模块

# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
from typing import Dict

from src.workflow import AdaptiveDecoder
from src.core.monitor import SystemMonitor
from src.utils.logger import Logger
from src.utils.config import ServicePerformance
from src.utils.config import DEFAULT_TEST_PROMPT, DEFAULT_TEST_TOKENS, DEFAULT_MODEL

class PerformanceTester:
    """性能测试工具，在启动时测量GPU和CPU服务的性能"""
    
    def __init__(self, system_monitor: SystemMonitor, decoder: AdaptiveDecoder):
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
        """运行基准测试：GPU prefill+CPU decode 测量性能差异"""
        Logger.info("开始基准测试...")
        try:
            # 0. 准备 decode 数据并获取原始 max_tokens
            test_data = await self._prepare_test_data()
            max_tokens = test_data.get("max_tokens", self.test_tokens)

            # 1. GPU 测试（default_request）
            test_data["max_tokens"] = max_tokens * 10
            gpu_res = await self.adaptive_decoder.default_request(test_data)
            gpu_time = gpu_res.get("decode_time", 0.0)
            gpu_tp = await self._get_throughput("gpu", gpu_res)
            Logger.info(f"GPU性能测试: 生成token={gpu_res.get('response', {}).get('usage', {}).get('completion_tokens', 0)}个, 耗时={gpu_time:.3f}s, metrics吞吐量={gpu_tp:.2f}tokens/s")

            # 2. CPU 测试（prefill_request + decode_request）
            test_data["max_tokens"] = max_tokens
            prefill = await self.adaptive_decoder.prefill_request(test_data)
            completion_id = prefill.get("completion_id")
            decision = {"device": "CPU", "token_limit": max_tokens + 1}
            cpu_res = await self.adaptive_decoder.decode_request(test_data, completion_id, decision)
            cpu_time = cpu_res.get("decode_time", 0.0)
            cpu_tp = await self._get_throughput("cpu", cpu_res)
            Logger.info(f"CPU性能测试: 生成token={cpu_res.get('total_tokens_generated', 0)}个, 耗时={cpu_time:.3f}s, metrics吞吐量={cpu_tp:.2f}tokens/s")

            # 保存结果并计算比率
            self._save_performance_results(gpu_time * 1000, gpu_tp, cpu_time * 1000, cpu_tp)
            summary = self.get_performance_summary()
            Logger.info(f"基准测试完成: GPU/CPU 性能比={summary.get('performance_ratio', 0):.2f}x")
        except Exception as e:
            Logger.error(f"基准测试失败: {str(e)}", exc_info=True)
    
    async def _prepare_test_data(self) -> Dict:
        """准备测试数据，在GPU上执行prefill
        
        Returns:
            Dict: 包含decode所需参数的字典
        """
        try:
            Logger.info("准备prefill测试数据")
            
            # 准备decode数据 (使用 prompt 字段)
            test_data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": self.test_prompt}],
                "stream": False,
                "n": 1,
                "max_tokens": self.test_tokens,
                "temperature": 0  # 基准测试固定使用temperature=0
            }
            
            Logger.info(f"准备好decode数据，model={self.model_name}, max_tokens={self.test_tokens}")
            
            return test_data
        except Exception as e:
            Logger.error(f"准备测试数据失败: {e}", exc_info=True)
            return {}
        
    async def _get_throughput(self, device: str, res: Dict) -> float:
        """获取指定设备的吞吐量"""
        # 从原生metrics获取吞吐量
        await self.system_monitor.update_metrics(force=True)
        updated_metrics = self.system_monitor.get_all_metrics()
        native_throughput = updated_metrics[device.lower()]['generation_throughput']
        
        # 通过生成tokens/s计算吞吐量
        if device == "gpu":
            generation_tokens = res.get("response", {}).get("usage", {}).get("completion_tokens", 0)
            generation_time = res.get("decode_time", 0.0)
            calculate_throughput = generation_tokens / generation_time
        else:
            generation_tokens = res.get("total_tokens_generated", 0)
            generation_time = res.get("decode_time", 0.0)
            calculate_throughput = generation_tokens / generation_time

        print(f"native_throughput: {native_throughput}, calculate_throughput: {calculate_throughput}")
        # 记录原生指标与实际计算值的差异（用于验证指标准确性）
        if native_throughput > 0 and calculate_throughput > 0:
            diff_percent = abs(native_throughput - calculate_throughput) / calculate_throughput * 100
            Logger.info(f"{device}吞吐量对比: " + \
                        f"vLLM指标={native_throughput:.2f}tokens/s, " + \
                        f"实际计算={calculate_throughput:.2f}tokens/s, " + \
                        f"差异={diff_percent:.1f}%")
            
            # 如果差异过大，记录警告
            if diff_percent > 30:
                Logger.warning(f"{device}吞吐量指标与实际计算值差异较大({diff_percent:.1f}%)，可能影响决策准确性，使用计算值")
                return calculate_throughput

        return native_throughput
    
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
