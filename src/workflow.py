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

from src.utils.config import CPU_HOST, CPU_PORT, DEFAULT_MAX_TOKENS, GPU_HOST, GPU_PORT
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

        self.default_max_tokens = DEFAULT_MAX_TOKENS

    # ===== 主流程接口 =====
    async def chat_completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """处理 /v1/chat/completions，动态调度"""
        decision = await self.scheduler.scheduler()
        if decision.get("device") == "GPU":
            # GPU 全流程
            response = await self.default_request(data)
        else:
            # PD 分离
            prefill = await self.prefill_request(data.copy())
            completion_id = prefill["completion_id"]
            response = await self.decode_request(data.copy(), completion_id, decision)
        return response

    async def chat_completion_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """流式处理 /v1/chat/completions，支持 PD 分离和 GPU 全流程"""
        start_time = time.time()

        # 第一次调度
        decision = await self.scheduler.scheduler()
        if decision.get("device") == "GPU":
            # GPU 全流程流式，直接透传所有 SSE chunk
            async for chunk in self.default_request_stream(data):
                yield chunk
        else:
            # PD 分离流式解码，委托给通用方法
            prefill = await self.prefill_request(data.copy())
            completion_id = prefill["completion_id"]
            # 使用初始调度决策作为参数，交由decode_request_stream内部调度
            async for chunk in self.decode_request_stream(data.copy(), completion_id, decision):
                yield chunk

        Logger.info(f"流式处理完成: 耗时={time.time() - start_time:.3f}秒")

    # ===== 核心请求 =====
    async def default_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """直接执行默认的 GPU 全流程请求"""
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
        """执行默认 GPU 全流程的流式请求，返回字节流生成器"""
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
        执行prefill请求

        Args:
            data: 包含model和prompt等参数的字典

        Returns:
            Dict: 包含completion_id和相关信息的字典

        """
        start_time = time.time()
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

                prefill_time = time.time() - start_time
                prefill_response = response.json()

                completion_id = prefill_response.get("id")
                if not completion_id:
                    _raise_error("Prefill响应缺少completion ID")
                Logger.info(
                    f"Prefill完成: 耗时={prefill_time:.3f}秒, completion_id={completion_id}",
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
        """执行 PD 分离非流式解码。"""
        start_time = time.time()
        last_step_res: dict[str, Any] = {}
        max_tokens = decode_data.get("max_tokens", self.default_max_tokens)
        total_tokens_generated = 0
        last_generated_text = ""
        finish_reason = "scheduled"
        curr_decision = decision

        # 当前版本仅支持CPU完整decode
        curr_decision = {"device": "CPU", "token_limit": max_tokens + 1}

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
            new_token_count = step_res.get("new_token_count", 0)
            finish_reason = step_res.get("finish_reason") or ""
            last_generated_text = step_res.get("generated_text", "")

            last_step_res = step_res.get("response", {})
            total_tokens_generated += new_token_count
            max_tokens -= new_token_count
            decode_data["max_tokens"] = max_tokens

            if finish_reason == "scheduled":
                curr_decision = await self.scheduler.scheduler()

        decode_time = time.time() - start_time
        Logger.info(
            f"PD 分离解码完成: 耗时={decode_time:.3f}秒, "
            f"生成{total_tokens_generated}个tokens, "
            f"finish_reason={finish_reason}",
        )
        return last_step_res

    async def decode_request_stream(
        self,
        decode_data: dict[str, Any],
        completion_id: str,
        decision: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        """执行 PD 分离流式解码，仅输出流式 chunk。"""
        start_time = time.time()
        max_tokens = decode_data.get("max_tokens", self.default_max_tokens)
        last_generated_text = ""
        finish_reason = "scheduled"
        curr_decision = decision

        # 当前版本仅支持CPU完整decode
        curr_decision = {"device": "CPU", "token_limit": max_tokens + 1}

        while finish_reason == "scheduled":
            # 调度决策
            device = curr_decision.get("device", "GPU")
            token_limit = curr_decision.get("token_limit", 0)
            if device == "GPU" or token_limit == 0:
                device, token_limit = "GPU", max_tokens + 1

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
            max_tokens -= token_count
            decode_data["max_tokens"] = max_tokens
            if finish_reason == "scheduled":
                curr_decision = await self.scheduler.scheduler()

        decode_time = time.time() - start_time
        Logger.info(
            f"PD 分离解码完成: 耗时={decode_time:.3f}秒, 生成{token_count}个tokens, finish_reason={finish_reason}",
        )

    # ===== 私有解码步骤 =====
    async def _execute_decode_step(
        self,
        decode_data: dict[str, Any],
        device_type: str,
    ) -> dict[str, Any]:
        """执行单步解码：根据 device_type 发送请求，返回解码结果、生成文本和 token 数"""
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
        """单步流式请求，只产出原始字节块，调用者解析完整响应"""
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
    async def hybrid_inference_completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """处理 /v1/test/decode_sequence，强制执行：GPU prefill + CPU decode"""
        assert data.get("max_tokens") is not None, "max_tokens 不能为空"
        decision = {"device": "CPU", "token_limit": data.get("max_tokens") + 1}
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        return await self.decode_request(data.copy(), completion_id, decision)

    async def hybrid_inference_completion_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """流式处理 /v1/test/decode_sequence，强制执行：GPU prefill + CPU decode"""
        assert data.get("max_tokens") is not None, "max_tokens 不能为空"
        decision = {"device": "CPU", "token_limit": data.get("max_tokens") + 1}
        # PD 分离流式解码
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        # 逐块输出
        async for chunk in self.decode_request_stream(data.copy(), completion_id, decision):
            yield chunk
