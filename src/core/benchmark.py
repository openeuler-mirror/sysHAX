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
Desc:sysHAX 基准测试模块
"""

import asyncio

from src.core.monitor import SystemMonitor
from src.utils.config import DEFAULT_MODEL, DEFAULT_TEST_PROMPT, DEFAULT_TEST_TOKENS, ServicePerformance
from src.utils.logger import Logger
from src.workflow import AdaptiveDecoder


class PerformanceTester:
    """性能测试工具，在启动时测量GPU和CPU服务的性能"""

    def __init__(self, system_monitor: SystemMonitor, decoder: AdaptiveDecoder) -> None:
        """
        初始化性能测试工具

        Args:
            system_monitor: 系统监控器实例
            decoder: 自适应解码器实例

        """
        # 使用传入的系统监控器实例
        self.system_monitor = system_monitor

        # 创建自适应解码器，用于性能测试
        self.adaptive_decoder = decoder

        # 测试参数 - 从配置中获取
        self.test_prompt = DEFAULT_TEST_PROMPT
        self.test_tokens = DEFAULT_TEST_TOKENS
        self.model_name = DEFAULT_MODEL
        self.diff_percent_threshold = 30

        # 性能测试结果
        self.gpu_performance = None
        self.cpu_performance = None

        # 性能比
        self.performance_ratio = 1.0

    async def run_benchmarks(self) -> None:
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
            await asyncio.sleep(3)
            gpu_tp = await self._check_throughput("gpu", gpu_res)
            Logger.info(f"GPU性能测试: 耗时={gpu_time:.3f}s, 吞吐量={gpu_tp:.2f}tokens/s")

            # 2. CPU 测试（prefill_request + decode_request）
            test_data["max_tokens"] = max_tokens
            prefill = await self.adaptive_decoder.prefill_request(test_data)
            completion_id = prefill.get("completion_id")
            assert completion_id is not None

            decision = {"device": "CPU", "token_limit": max_tokens + 1}
            cpu_res = await self.adaptive_decoder.decode_request(test_data, completion_id, decision)
            cpu_time = cpu_res.get("decode_time", 0.0)
            await asyncio.sleep(3)
            cpu_tp = await self._check_throughput("cpu", cpu_res)
            Logger.info(f"CPU性能测试: 耗时={cpu_time:.3f}s, 吞吐量={cpu_tp:.2f}tokens/s")

            # 保存结果并计算比率
            self._save_performance_results(gpu_time * 1000, gpu_tp, cpu_time * 1000, cpu_tp)
            summary = self.get_performance_summary()
            Logger.info(f"基准测试完成: GPU/CPU 性能比={summary.get('performance_ratio', 0):.2f}x")
        except (ValueError, KeyError) as e:
            Logger.error(f"基准测试失败: {e!s}")

    async def _prepare_test_data(self) -> dict:
        """
        准备测试数据，在GPU上执行prefill

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
                "temperature": 0,  # 基准测试固定使用temperature=0
            }

            Logger.info(f"准备好decode数据，model={self.model_name}, max_tokens={self.test_tokens}")
        except (KeyError, TypeError) as e:
            Logger.error(f"准备测试数据失败: {e}", exc_info=True)
            return {}
        else:
            return test_data

    async def _get_throughput_metrics(self, device: str) -> float:
        """从监控器获取指定设备的吞吐量指标"""
        await self.system_monitor.update_metrics(force=True)
        metrics = self.system_monitor.get_gpu_metrics() if device == "gpu" else self.system_monitor.get_cpu_metrics()
        return metrics.get("generation_throughput", 0.0)

    async def _check_throughput(self, device: str, res: dict) -> float:
        """校验并返回吞吐量，比较监控数据与计算值"""
        native_throughput = await self._get_throughput_metrics(device)
        # 计算值
        if device == "gpu":
            tokens = res.get("response", {}).get("usage", {}).get("completion_tokens", 0)
            t = res.get("decode_time", 0.0)
        else:
            tokens = res.get("total_tokens_generated", 0)
            t = res.get("decode_time", 0.0)
        calculate_throughput = tokens / t if t > 0 else 0.0

        # 如果原生吞吐量不可用，直接返回计算值
        if native_throughput <= 0:
            return calculate_throughput

        # 原生与计算值都有效时，校验误差
        if calculate_throughput > 0:
            diff = abs(native_throughput - calculate_throughput) / calculate_throughput * 100
            Logger.info(
                f"{device}吞吐量对比: 原生={native_throughput:.2f}, 计算={calculate_throughput:.2f}, 差异={diff:.1f}%",
            )
            if diff > self.diff_percent_threshold:
                Logger.warning(f"{device}吞吐量差异较大({diff:.1f}%)，使用计算值")
                return calculate_throughput
        return native_throughput

    def _save_performance_results(
        self,
        gpu_latency: float,
        gpu_throughput: float,
        cpu_latency: float,
        cpu_throughput: float,
    ) -> None:
        """
        保存性能测试结果并计算性能比率

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

    def get_performance_summary(self) -> dict:
        """获取性能测试结果摘要"""
        if not self.gpu_performance or not self.cpu_performance:
            return {"status": "未完成测试"}

        return {
            "gpu": {"latency_ms": self.gpu_performance.avg_latency, "throughput": self.gpu_performance.throughput},
            "cpu": {"latency_ms": self.cpu_performance.avg_latency, "throughput": self.cpu_performance.throughput},
            "performance_ratio": self.performance_ratio,
        }
