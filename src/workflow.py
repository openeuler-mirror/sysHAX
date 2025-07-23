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
Desc: sysHAX /v1/chat/completions 接口适配
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, NoReturn

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.core.monitor import SystemMonitor
    from src.core.scheduler import Scheduler

from src.utils.config import CPU_HOST, CPU_PORT, GPU_HOST, GPU_PORT, MONITOR_INTERVAL
import asyncio

from src.utils.logger import Logger


class AdaptiveDecoderError(Exception):
    """自适应解码器相关异常"""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        初始化自适应解码器异常

        Args:
            message: 错误信息
            cause: 可选的原始异常

        """
        super().__init__(message)
        self.cause = cause


def _raise_error(message: str, cause: Exception | None = None) -> NoReturn:
    """记录日志并抛出 AdaptiveDecoderError"""
    Logger.error(message)
    raise AdaptiveDecoderError(message, cause)


class AdaptiveDecoder:
    """自适应解码器，负责在GPU和CPU之间动态切换解码任务"""

    def __init__(self, system_monitor: SystemMonitor, scheduler: Scheduler) -> None:
        """初始化自适应解码器"""
        self.system_monitor: SystemMonitor = system_monitor
        self.scheduler: Scheduler = scheduler

        # 拼接 /v1/chat/completions 服务地址
        self.v1_chat_gpu = f"http://{GPU_HOST}:{GPU_PORT}/v1/chat/completions"
        self.v1_chat_cpu = f"http://{CPU_HOST}:{CPU_PORT}/v1/chat/completions"

    # ===== 主流程接口 =====
    async def chat_completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        处理 /v1/chat/completions 接口的主流程。

        流程说明：
        1. 首先调用 self.scheduler.scheduler() 获取推理调度决策（decision）。
           - decision 字典中包含 device 字段，指示推理阶段应使用的设备（如 GPU 或 CPU）。
        2. 根据决策分支处理：
           - 如果 decision["device"] == "GPU"：
               * 走 GPU 全流程，直接调用 self.default_request(data) 完成推理。
           - 否则（如 PD 分离场景）：
               * 先在 GPU 上执行 prefill 阶段，调用 self.prefill_request(data.copy())，获取 completion_id。
               * 再在 CPU 上执行 decode 阶段，调用 self.decode_request(data.copy(), completion_id, decision)。
        3. 返回最终推理结果 response。
        """
        try:
            decision = self.scheduler.scheduler()
            if decision.get("device") == "GPU":
                # GPU 全流程
                response = await self.default_request(data)
                if response.get("choices") and response["choices"][0]["finish_reason"] == "scheduled":
                    completion_id = response["id"]
                    decision = self.scheduler.scheduler()
                    decision["device"] = "CPU"
                    response = await self.decode_request(data.copy(), completion_id, decision)
            else:
                # PD 分离
                prefill = await self.prefill_request(data.copy())
                completion_id = prefill["completion_id"]
                response = await self.decode_request(data.copy(), completion_id, decision)
            return response
        except AdaptiveDecoderError as e:
            # 处理默认请求失败，记录并返回错误信息给客户端
            Logger.info_console(f"请求失败: {e}")
            return {"error": str(e)}

    async def chat_completion_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """处理 /v1/chat/completions 接口的流式请求。"""
        try:
            decision = self.scheduler.scheduler()
            if decision.get("device") == "GPU":
                raw_data = bytearray()
                # GPU 全流程流式，直接透传所有 SSE chunk
                async for chunk in self.default_request_stream(data):
                    yield chunk
                    raw_data.extend(chunk)
                last_event = self._parse_sse_buffer(raw_data)
                if(self._check_scheduled_status(last_event)):
                    completion_id = last_event["id"]
                    decision = self.scheduler.scheduler()
                    decision["device"] = "CPU"
                    async for chunk in self.decode_request_stream(data.copy(), completion_id, decision):
                        yield chunk
            else:
                # PD 分离流式解码，委托给通用方法
                prefill = await self.prefill_request(data.copy())
                completion_id = prefill["completion_id"]
                # 使用初始调度决策作为参数，交由decode_request_stream内部调度
                async for chunk in self.decode_request_stream(data.copy(), completion_id, decision):
                    yield chunk
        except AdaptiveDecoderError as e:
            # 捕获默认请求或解码流式失败，将错误作为 SSE 事件返回并结束生成器
            Logger.info_console(f"请求失败: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    # ===== 核心请求 =====
    async def default_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """执行默认：向GPU服务发送完整请求，返回完整响应"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.v1_chat_gpu,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=300,
            )
        if response.status_code != httpx.codes.OK:
            _raise_error(
                f"默认请求失败: HTTP {response.status_code}, 响应: {response.text}",
            )
        return response.json()

    async def default_request_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """执行默认：向GPU服务发送流式请求，返回字节流生成器"""
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                self.v1_chat_gpu,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=300,
            ) as response,
        ):
            if response.status_code != httpx.codes.OK:
                # 读取完整响应内容以记录错误
                text = await response.aread()
                _raise_error(
                    f"默认流式请求失败: HTTP {response.status_code}, 响应: {text.decode()}",
                )
            # 异步迭代并返回原始字节块
            async for chunk in response.aiter_bytes():
                yield chunk

    # ===== prefill 请求 =====
    async def prefill_request(self, data: dict[str, Any]) -> dict:
        """
        向GPU服务发送prefill请求，强制关闭流式，返回completion_id用于继承kv缓存。

        Args:
            data: 包含model和prompt等参数的字典

        Returns:
            Dict: 包含completion_id和相关信息的字典

        """
        start_time = time.time_ns()
        prefill_data = data.copy()
        # Prefill 强制关闭流式
        prefill_data["stream"] = False
        # 设置num_decode_tokens=2（通过data字典）
        prefill_data["num_decode_tokens"] = 2

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.v1_chat_gpu,
                    headers={"Content-Type": "application/json"},
                    json=prefill_data,
                    timeout=300,
                )

                if response.status_code != httpx.codes.OK:
                    _raise_error(f"Prefill请求失败: HTTP {response.status_code}, 响应: {response.text}")

                prefill_time = time.time_ns() - start_time
                prefill_response = response.json()
                usage = prefill_response.get("usage", {})
                tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                self.system_monitor.gpu_monitor.add_prefill_stats(tokens, prefill_time)

                completion_id = prefill_response.get("id")
                if not completion_id:
                    _raise_error("Prefill响应缺少completion ID")
                Logger.info(
                    f"Prefill完成: 耗时={prefill_time / 1e9:.3f}秒, completion_id={completion_id}",
                )
            except (httpx.RequestError, ValueError) as e:
                _raise_error(f"Prefill请求异常: {e!s}", e)
            else:
                return {
                    "completion_id": completion_id,
                    "prefill_response": prefill_response,
                }

    # ===== decode 请求 =====
    async def decode_request(
        self,
        decode_data: dict[str, Any],
        completion_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        """
        执行非流式的 decode 请求。

        该方法在CPU上以非流式方式执行解码，根据调度器决策循环进行解码步骤，直到生成结束（finish_reason不为"scheduled"）。
        每一步会根据上一次的生成结果和剩余token数动态调整解码参数，最终返回解码的完整响应结果。
        """
        start_time = time.time_ns()
        last_step_res: dict[str, Any] = {}
        max_tokens = decode_data.get("max_tokens")
        if max_tokens is None:
            Logger.warning("max_tokens not found in request which will using default value 10")
            max_tokens = 10
        last_generated_text = ""
        finish_reason = "scheduled"
        curr_decision = decision

        while finish_reason == "scheduled":
            device = curr_decision.get("device", "GPU")
            token_limit = curr_decision.get("token_limit", 0)
            if device == "GPU" or token_limit == 0:
                device, token_limit = "GPU", max_tokens + 1

            request_data = decode_data.copy()
            request_data["request_id_inference"] = completion_id
            request_data["num_decode_tokens"] = token_limit
            request_data["generated_text"] = last_generated_text

            step_res = await self._execute_decode_step(request_data, device)
            assert step_res is not None, "decode结果为空"

            completion_id = step_res.get("request_id", "")
            total_token_count = step_res.get("new_token_count", 0)
            finish_reason = step_res.get("finish_reason") or ""
            last_generated_text = step_res.get("generated_text", "")

            last_step_res = step_res.get("response", {})

            if finish_reason == "scheduled":
                curr_decision = await self.scheduler.scheduler()

        decode_time = time.time_ns() - start_time
        Logger.info(
            f"PD 分离解码完成: 耗时={decode_time / 1e9:.3f}秒, "
            f"共生成{total_token_count}个tokens, "
            f"finish_reason={finish_reason}",
        )
        return last_step_res

    async def decode_request_stream(
        self,
        decode_data: dict[str, Any],
        completion_id: str,
        decision: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        """执行流式的 decode 请求，输出流式 chunk。"""
        start_time = time.time_ns()
        max_tokens = decode_data.get("max_tokens")
        if max_tokens is None:
            Logger.warning("max_tokens not found in request which will using default value 10")
            max_tokens = 10
        last_generated_text = ""
        finish_reason = "scheduled"
        curr_decision = decision
        total_token_count = 0
        while finish_reason == "scheduled":
            segment_start = time.time_ns()
            # 调度决策
            device = curr_decision.get("device", "GPU")
            token_limit = curr_decision.get("token_limit", 0)
            if device == "GPU" or token_limit == 0:
                device, token_limit = "GPU", max_tokens + 1
            if token_limit <= 0:
                break

            # 构造请求
            request_data = decode_data.copy()
            request_data["request_id_inference"] = completion_id
            request_data["num_decode_tokens"] = token_limit
            request_data["generated_text"] = last_generated_text
            request_data["stream"] = True

            token_count = 0
            generated_text_acc = ""
            # 单步流式请求并处理每个 SSE chunk
            async for chunk in self._step_stream(request_data, device):
                # 过滤默认的 DONE 事件
                filtered = chunk.replace(b"data: [DONE]\n\n", b"")
                if not filtered:
                    continue
                # 输出给客户端
                yield filtered
                # 解析 JSON 以提取字段
                data_str = filtered.decode().removeprefix("data: ").strip()
                obj = json.loads(data_str)
                # 递增 token 计数
                token_count += 1
                # 累加生成文本
                delta = obj.get("choices", [{}])[0].get("delta", {})
                generated_text_acc += delta.get("content", "")
                # 更新 request_id 和 finish_reason
                completion_id = obj.get("id", completion_id)
                finish_reason = obj.get("choices", [{}])[0].get("finish_reason") or finish_reason

            # 更新状态，为下一轮做准备
            last_generated_text = generated_text_acc
            total_token_count += token_count
            decode_data["max_tokens"] = max_tokens
            if finish_reason == "scheduled":
                curr_decision = self.scheduler.scheduler()
            # accumulate per-segment decode stats for aggregated throughput
            segment_time = time.time_ns() - segment_start
            self.system_monitor.gpu_monitor.add_decode_stats(token_count, segment_time) if device == "GPU" else \
                self.system_monitor.cpu_monitor.add_decode_stats(token_count, segment_time)
        decode_time = time.time_ns() - start_time
        Logger.info(
            f"PD 分离解码完成: 耗时={decode_time / 1e9:.3f}秒, 共生成{total_token_count}个tokens, finish_reason={finish_reason}",
        )

    # ===== 私有解码步骤 =====
    async def _execute_decode_step(
        self,
        decode_data: dict[str, Any],
        device_type: str,
    ) -> dict[str, Any]:
        """执行单步 decode 请求：根据 device_type 发送请求，返回解码结果、生成文本和 token 数"""
        service_url = self.v1_chat_gpu if device_type == "GPU" else self.v1_chat_cpu
        async with httpx.AsyncClient() as client:
            response = await client.post(
                service_url,
                headers={"Content-Type": "application/json"},
                json=decode_data,
                timeout=300,
            )
        if response.status_code != httpx.codes.OK:
            _raise_error(
                f"{device_type} 解码请求失败: HTTP {response.status_code}, 响应: {response.text}",
            )
        resp_json = response.json()
        # 提取请求id
        request_id = resp_json.get("id", "")
        # 提取生成文本
        generated_text = ""
        finish_reason = None
        if resp_json.get("choices") and resp_json["choices"][0].get("message"):
            generated_text = resp_json["choices"][0]["message"].get("content", "")
            finish_reason = resp_json["choices"][0].get("finish_reason")
        # 提取新生成 token 数
        new_token_count = resp_json.get("usage", {}).get("completion_tokens", 0)
        return {
            "request_id": request_id,
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "new_token_count": new_token_count,
            "response": resp_json,
        }

    async def _step_stream(
        self,
        request_data: dict[str, Any],
        device_type: str,
    ) -> AsyncGenerator[bytes, None]:
        """单步 decode 流式请求，只产出原始字节块，调用者解析完整响应"""
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                self.v1_chat_gpu if device_type == "GPU" else self.v1_chat_cpu,
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=300,
            ) as response,
        ):
            if response.status_code != httpx.codes.OK:
                text = await response.aread()
                _raise_error(f"{device_type} 解码流式失败: HTTP {response.status_code}, {text.decode()}")
            async for chunk in response.aiter_bytes():
                yield chunk

    # ===== 强制PD分离解码主接口 =====
    async def pd_disagg_completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """非流式处理 /v1/chat/pd_disagg，强制执行：GPU prefill + CPU decode"""
        assert data.get("max_tokens") is not None, "max_tokens 不能为空"
        decision = {"device": "CPU", "token_limit": data.get("max_tokens") + 1}
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        return await self.decode_request(data.copy(), completion_id, decision)

    async def pd_disagg_completion_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """流式处理 /v1/chat/pd_disagg，强制执行：GPU prefill + CPU decode"""
        assert data.get("max_tokens") is not None, "max_tokens 不能为空"
        decision = {"device": "CPU", "token_limit": data.get("max_tokens") + 1}
        # PD 分离流式解码
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        # 逐块输出
        async for chunk in self.decode_request_stream(data.copy(), completion_id, decision):
            yield chunk

    def _parse_sse_buffer(self, raw_data: bytes) -> dict | None:
        """
        逆向扫描SSE流，定位首个含"finish_reason"字段的事件并解析。
        """
        if not raw_data:
            return None

        end_index = len(raw_data)
        event_count = 0
        max_events_to_scan = 10  # 最大逆向扫描事件数
        
        while event_count < max_events_to_scan and end_index > 0:
            start_index = raw_data.rfind(b"\n\n", 0, end_index - 1)
            event_block = raw_data[start_index + 2: end_index] \
                if start_index != -1 else raw_data[:end_index]

            data_lines = []
            for line in event_block.splitlines():
                line = line.strip()
                if line.startswith(b"data:"):
                    data_lines.append(line[len("data:"):])  # 移除"data:"前缀

            if not data_lines:
                end_index = start_index
                event_count += 1
                continue

            try:
                combined_data = b"\n".join(data_lines).decode("utf-8")
                event_dict = json.loads(combined_data)
                if "choices" in event_dict and "finish_reason" in event_dict["choices"][0]:
                    return event_dict
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

            end_index = start_index
            event_count += 1

        Logger.warning("未找到含'finish_reason'字段的有效事件")
        return None

    def _check_scheduled_status(self, event_data: Any) -> bool:
        """检查并处理scheduled状态"""
        if not event_data:
            return
        try:
            if (
                isinstance(event_data, dict) and
                "choices" in event_data and
                isinstance(event_data["choices"], list) and
                len(event_data["choices"]) > 0 and
                event_data["choices"][0].get("finish_reason") == "scheduled"
            ):
                return True
            else:
                return False
        except Exception as e:
            return False
