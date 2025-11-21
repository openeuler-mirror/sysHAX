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
Created: 2025-09-19
Desc:sysHAX 指标管理模块
"""

import time
import json
import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Coroutine

from src.utils.logger import Logger

SSE_DONE_EVENT = b"data: [DONE]\n\n"

@dataclass
class MetricsData:
    gpu_num_running: int = 0        # 正在运行的任务数
    cpu_num_running: int = 0        # 正在运行的任务数
    num_waiting: int = 0            # 等待中的任务数
    gpu_decode_throughout: float = 0.0      # 解码吞吐量，tokens/s
    cpu_decode_throughout: float = 0.0      # 解码吞吐量，tokens/s
    total_decode_throughout: float = 0.0    # 总解码吞吐量
    gpu_cache_usage: float = 0.0    # GPU缓存使用率，百分比
    cpu_cache_usage: float = 0.0    # CPU缓存使用率，百分比


class MetricsService:
    """
    服务指标类，包含资源使用和吞吐量
    """
    def __init__(self) -> None:
        self.metrics_data = MetricsData()

        self._elapsed = 1               # 统计间隔，单位秒
        self._print_interval = 5.0      # 打印间隔，单位秒
        self._gpu_token_num = 0
        self._cpu_token_num = 0
        self._reporter_task: asyncio.Task | None = None

    async def _report_metrics_periodically(self):
        """每 self._elapsed 秒统计吞吐量，每 self._print_interval 秒打印一次"""
        print_counter = 0
        total_elapsed = 0.0
        steps_per_print = int(self._print_interval / self._elapsed)
        if steps_per_print <= 0:
            steps_per_print = 1

        while True:
            try:
                await asyncio.sleep(self._elapsed)
                total_elapsed += self._elapsed

                self.metrics_data.gpu_decode_throughout = self._gpu_token_num / self._elapsed
                self.metrics_data.cpu_decode_throughout = self._cpu_token_num / self._elapsed
                self.metrics_data.total_decode_throughout = \
                    self.metrics_data.gpu_decode_throughout + self.metrics_data.cpu_decode_throughout
                self._gpu_token_num = 0
                self._cpu_token_num = 0

                print_counter += 1
                if print_counter >= steps_per_print:

                    Logger.info(
                        f"gpu Running: {self.metrics_data.gpu_num_running} reqs, "
                        f"cpu Running: {self.metrics_data.cpu_num_running} reqs, "
                        f"Pending: {self.metrics_data.num_waiting} reqs, "
                        f"Avg gpu generation throughput: {self.metrics_data.gpu_decode_throughout:.1f} tokens/s, "
                        f"Avg cpu generation throughput: {self.metrics_data.cpu_decode_throughout:.1f} tokens/s, "
                        f"Avg generation throughput: {self.metrics_data.total_decode_throughout:.1f} tokens/s"
                    )
                    print_counter = 0

            except asyncio.CancelledError:
                Logger.debug("[MetricsService] 吞吐量定时器已停止")
                break
            except Exception as e:
                Logger.error(f"[MetricsService] 定时器异常: {e}")

    async def stop(self):
        """关闭定时器"""
        if self._reporter_task:
            self._reporter_task.cancel()
            try:
                await self._reporter_task
            except asyncio.CancelledError:
                pass

    async def start(self):
        """启动定时器"""
        if self._reporter_task is not None and not self._reporter_task.done():
            return
        else:
            self._reporter_task = asyncio.create_task(self._report_metrics_periodically())

    async def stream_with_metrics(self, generator: AsyncGenerator[bytes, None], device: str) -> AsyncGenerator[bytes, None]:
        start_time = time.time_ns()
        first_token_time = None
        tokens = 0
        done_chunk = None
        async for chunk in generator:
            try:
                chunk_str = chunk.decode('utf-8').removeprefix("data: ").strip()
            except UnicodeDecodeError:
                chunk_str = ""
            if chunk_str == "[DONE]":
                done_chunk = chunk if chunk.startswith(b"data:") else SSE_DONE_EVENT
                continue
            if device == "GPU":
                self._gpu_token_num += 1
            else:
                self._cpu_token_num += 1

            try:
                chunk_json = json.loads(chunk_str)
                delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") is not None:
                    tokens += 1
                    if first_token_time is None:
                        first_token_time = time.time_ns()
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass                                # 如果解析失败，不计数，但依然透传原始 chunk
            yield chunk                             # 原样透传

        # 计算并返回 metrics
        time_used = time.time_ns() - start_time
        ttfb = (first_token_time - start_time) / 1e9 if first_token_time else 0.0

        metrics = {
            "TTFB": f"{round(ttfb, 3)}s",
            "time_used": f"{round(time_used / 1e9, 3)}s",
            "tokens": tokens,
            "throughput": f"{round(tokens / (time_used / 1e9), 3) if time_used > 0 else 0} tokens/s",
        }

        # SSE 格式返回 metrics 事件
        metrics_event = f"data: {json.dumps({'metrics': metrics})}\n\n"
        yield metrics_event.encode('utf-8')
        yield done_chunk if done_chunk is not None else SSE_DONE_EVENT

    async def normal_with_metrics(self, coro: Coroutine[any, any, dict], device: str) -> dict:
        start_time = time.time_ns()
        result = await coro
        time_used = time.time_ns() - start_time
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        metrics = {
            "time_used": f"{round(time_used / 1e9, 3)}s",
            "tokens": tokens,
            "throughput": f"{round(tokens / (time_used / 1e9), 3) if time_used > 0 else 0}tokens/s",
        }
        result["metrics"] = metrics
        return result

    def set_cpu_running_num(self, num: int) -> None:
        self.metrics_data.cpu_num_running = num

    def set_gpu_running_num(self, num: int) -> None:
        self.metrics_data.gpu_num_running = num

    def set_waiting_num(self, num: int) -> None:
        self.metrics_data.num_waiting = num

    def set_gpu_cache_usage(self, usage: float) -> None:
        self.metrics_data.gpu_cache_usage = usage

    def set_cpu_cache_usage(self, usage: float) -> None:
        self.metrics_data.cpu_cache_usage = usage

    @property
    def gpu_num_running(self) -> int:
        return self.metrics_data.gpu_num_running

    @property
    def cpu_num_running(self) -> int:
        return self.metrics_data.cpu_num_running

    @property
    def num_waiting(self) -> int:
        return self.metrics_data.num_waiting

    @property
    def gpu_decode_throughout(self) -> float:
        return self.metrics_data.gpu_decode_throughout

    @property
    def cpu_decode_throughout(self) -> float:
        return self.metrics_data.cpu_decode_throughout

    @property
    def total_decode_throughout(self) -> float:
        return self.metrics_data.total_decode_throughout

    @property
    def gpu_cache_usage(self) -> float:
        return self.metrics_data.gpu_cache_usage

    @property
    def cpu_cache_usage(self) -> float:
        return self.metrics_data.cpu_cache_usage

