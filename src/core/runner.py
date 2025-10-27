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
Created: 2025-09-22
Desc:sysHAX 任务执行模块
"""

from typing import AsyncGenerator
import httpx
import json
from collections import deque
import asyncio

from src.core.metrics import MetricsService
from src.utils.config import SyshaxConfig
from src.utils.logger import Logger

class Runner:
    def __init__(self, metrics_service: MetricsService, syshax_config: SyshaxConfig) -> None:
        self.metrics_service: MetricsService = metrics_service
        self.syshax_config: SyshaxConfig = syshax_config
        # 拼接 /v1/chat/completions 服务地址
        self.v1_chat_gpu = f"http://{syshax_config.gpu_host}:{syshax_config.gpu_port}/v1/chat/completions"
        self.v1_chat_cpu = f"http://{syshax_config.cpu_host}:{syshax_config.cpu_port}/v1/chat/completions"
        self.v1_chat = f"http://{syshax_config.syshax_host}:{syshax_config.syshax_port}/v1/chat/completions"

    async def task_handler(self, device: str, data: dict[str, any], resubmit_task_data: dict):
        is_stream = data.get("stream", False)
        recent_chunks = deque(maxlen=3)
        try:
            if is_stream:
                gen = self.default_request_stream(device, data)
                async for chunk in self.metrics_service.stream_with_metrics(gen, device=device):
                    recent_chunks.append(chunk)
                    yield chunk
            else:
                Logger.warning("非流式请求暂时无法实时计算吞吐量，下面的统计数据有误，建议使用流式输出......")
                coro = self.default_request(device, data)
                result = await self.metrics_service.normal_with_metrics(coro, device=device)
                json_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
                recent_chunks.append(json_bytes)
                yield json_bytes
                yield b"\n\n"
        finally:
            pass
        id, finish_reason = self._extract_and_check_finish_reason(recent_chunks)
        if finish_reason is None:
            Logger.debug(f"任务{id}未找到包含 finish_reason 的事件")
        elif finish_reason == "scheduled":
            Logger.info(f"任务{id} finish_reason 为 {finish_reason}, 被接续推理")
            if resubmit_task_data is not None:
                resubmit_task_data["data"] = await self._create_resubmit_task(data, id)

    async def default_request_stream(
        self,
        device: str,
        data: dict[str, any]
    ) -> AsyncGenerator[bytes, None]:
        """执行流式的 decode 请求，输出流式 chunk。"""
        service_url = self.v1_chat_gpu if device == "GPU" else self.v1_chat_cpu
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                        "POST",
                        service_url,
                        headers={"Content-Type": "application/json"},
                        json=data,
                        timeout=self.syshax_config.request_timeout,
                    ) as response:

                        if response.status_code != 200:
                            Logger.error(f"{device} 请求失败: {response.status_code}")
                            yield b'data: {"error": "service_unavailable"}\n\n'
                            return

                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
            except Exception as e:
                Logger.error(f"{device} 流式请求异常: {e}", exc_info=True)
                yield b'data: {"error": "internal_error"}\n\n'

    async def default_request(self, device: str, data: dict[str, any]) -> dict[str, any]:
        """执行非流式的请求，等待完整响应后返回 JSON 字典。"""
        service_url = self.v1_chat_gpu if device == "GPU" else self.v1_chat_cpu
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    service_url,
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=self.syshax_config.request_timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                Logger.error(f"{device} 请求失败: {e.response.status_code}, 响应: {e.response.text}")
                raise
            except Exception as e:
                Logger.error(f"{device} 请求异常: {e}", exc_info=True)
                raise

    def _parse_sse_buffer(self, raw_data: bytes) -> dict | None:
        """
        逆向扫描SSE流，定位首个含"finish_reason"字段的事件并解析。
        """
        if not raw_data:
            return None

        data_lines = []
        for line in raw_data.splitlines():
            line = line.strip()
            if line.startswith(b"data:"):
                data_lines.append(line[len("data:"):])  # 移除"data:"前缀
            else:
                data_lines.append(line)

        if not data_lines:
            return None

        try:
            combined_data = b"\n".join(data_lines).decode("utf-8")
            event_dict = json.loads(combined_data)
            return event_dict
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    
    def _extract_and_check_finish_reason(self, chunks: list[bytes]) -> str | None:
        """
        从 chunks 列表中尝试提取 SSE 事件，检查是否有 finish_reason == 'scheduled'
        :param chunks: 字节块列表，按时间顺序
        :param context: 日志上下文，如 "GPU stream" 或 "CPU non-stream"
        """
        if not chunks:
            return

        id = None
        finish_reason = None
        for chunk in chunks:
            chunk_dict = self._parse_sse_buffer(chunk)
            try:
                if 'finish_reason' in chunk_dict['choices'][0]:
                    finish_reason = chunk_dict.get("choices", [{}])[0].get("finish_reason", None)
                if 'id' in chunk_dict:
                    id = chunk_dict.get("id", None)
                if id is not None and finish_reason is not None:
                    return id, finish_reason
            except Exception as e:
                Logger.debug(f"解析SSE事件失败: {e}, chunk: {chunk}")
                continue
        return id, finish_reason

    async def _create_resubmit_task(self,
                                data: dict[str, any],
                                request_id_inference: str) -> None:
        """
        将任务重新提交给自己执行，适用于 finish_reason == 'scheduled' 的情况
        """
        new_data = data.copy()
        new_data["request_id_inference"] = request_id_inference
        new_data["max_tokens"] = data.get("max_tokens")
        new_data["num_decode_tokens"] = new_data["max_tokens"]
        
        return new_data

