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
        # 调度决策
        decision = await self.scheduler.scheduler()
        if decision.get("device") == "GPU":
            # GPU 全流程
            res = await self.default_request(data)
            response = res["response"]
            decode_time = res["decode_time"]
            service_used = [res.get("service_type", "GPU")]
        else:
            # PD 分离
            prefill = await self.prefill_request(data.copy())
            completion_id = prefill["completion_id"]
            decode = await self.decode_request(data.copy(), completion_id, decision)
            response = decode["result"]["response"]
            decode_time = decode["decode_time"]
            service_used = [step.get("service_type") for step in decode.get("decode_results", [])]
        # 构建结果
        if "usage" not in response:
            response["usage"] = {}
        response["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
        }
        return response

    async def chat_completion_stream(self, data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """流式处理 /v1/chat/completions，支持 PD 分离和 GPU 全流程"""
        # 调度决策
        decision = await self.scheduler.scheduler()
        # GPU 全流程流式
        if decision.get("device") == "GPU":
            async for chunk in self.default_request_stream(data):
                yield chunk
            return
        # PD 分离流式
        # 1. Prefill 请求
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        # 2. 动态解码流式
        max_tokens = data.get("max_tokens", self.default_max_tokens)
        last_generated = ""
        finish_reason = "scheduled"
        curr_decision = decision
        remaining_max_tokens = max_tokens
        while finish_reason == "scheduled" and remaining_max_tokens > 0:
            device = curr_decision.get("device", "GPU")
            token_limit = curr_decision.get("token_limit", 0)
            if device == "GPU" or token_limit == 0:
                device = "GPU"
                token_limit = remaining_max_tokens + 1
            actual_limit = min(token_limit, remaining_max_tokens)
            # 构造流式解码请求数据
            decode_data = data.copy()
            decode_data["stream"] = True
            decode_data["request_id_inference"] = completion_id
            decode_data["generated_text"] = last_generated
            decode_data["num_decode_tokens"] = actual_limit
            # 执行流式解码
            async for chunk in self._execute_decode_step_stream(decode_data, device):
                yield chunk
            # 当前实现只执行一次解码循环，更多循环需解析 finish_reason 并更新 curr_decision
            break

    # ===== 核心请求 =====
    async def default_request(self, data: dict[str, Any]) -> dict:
        """直接执行默认的 GPU 全流程请求"""
        start_time = time.time()
        # 直接将原始请求转发到 GPU 服务
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.v1_chat_gpu,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=300,
            )
        decode_time = time.time() - start_time
        if response.status_code != httpx.codes.OK:
            _raise_error(
                f"默认请求失败: HTTP {response.status_code}, 响应: {response.text}",
            )
        json_response = response.json()
        return {
            "response": json_response,
            "decode_time": decode_time,
            "service_type": "GPU",
        }

    # 添加流式默认请求方法
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

    async def prefill_request(self, data: dict[str, Any]) -> dict:
        """
        执行prefill请求

        Args:
            data: 包含model和prompt等参数的字典

        Returns:
            Dict: 包含completion_id和相关信息的字典

        """
        start_time = time.time()

        # 创建 PrefillData 对象，避免修改原始 data
        prefill_data = data.copy()
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
                    _raise_error(
                        f"Prefill请求失败: HTTP {response.status_code}, 响应: {response.text}",
                    )

                prefill_time = time.time() - start_time
                prefill_response = response.json()

                # 提取必要信息
                completion_id = prefill_response.get("id")
                if not completion_id:
                    _raise_error("Prefill响应缺少completion ID")
                else:
                    Logger.info(
                        f"Prefill完成: 耗时={prefill_time:.3f}秒, completion_id={completion_id}",
                    )
                    # 仅返回生成的 request ID
                    return {
                        "completion_id": completion_id,
                        "prefill_time": prefill_time,
                        "prefill_response": prefill_response,
                    }

            except (httpx.RequestError, ValueError) as e:
                _raise_error(f"Prefill请求异常: {e!s}", e)

    async def decode_request(
        self,
        decode_data: dict[str, Any],
        completion_id: str,
        decision: dict[str, Any],
    ) -> dict:
        """执行解码请求，实现自适应解码，直到 finish_reason 不为 'scheduled'"""
        start_time = time.time()
        Logger.info(
            f"开始动态解码请求：max_tokens={decode_data.get('max_tokens', self.default_max_tokens)}",
        )

        decode_results = []
        max_tokens = decode_data.get("max_tokens", self.default_max_tokens)
        total_tokens_generated = 0
        last_generated_text = ""
        finish_reason = "scheduled"
        curr_decision = decision

        while finish_reason == "scheduled":
            # 使用传入的或最新的决策
            device = curr_decision.get("device", "GPU")
            token_limit = curr_decision.get("token_limit", 0)
            if device == "GPU" or token_limit == 0:
                device = "GPU"
                token_limit = max_tokens + 1
            Logger.info(
                f"调度决策：使用{device}解码，token限制={token_limit or '不限'}",
            )

            # 构造本轮请求数据
            request_data = decode_data.copy()
            request_data["request_id_inference"] = completion_id
            request_data["num_decode_tokens"] = token_limit
            request_data["generated_text"] = last_generated_text

            try:
                step_res = await self._execute_decode_step(request_data, device)
            except (AdaptiveDecoderError, httpx.RequestError, ValueError) as e:
                Logger.error(f"{device} 解码请求异常: {e!s}", exc_info=True)
                if not decode_results:
                    _raise_error(f"首次解码失败，无法继续: {e!s}", e)
                break

            completion_id = step_res.get("request_id", "")
            new_token_count = step_res.get("new_token_count", 0)
            finish_reason = step_res.get("finish_reason")
            decode_results.append(step_res)
            last_generated_text = step_res.get("generated_text", "")
            Logger.info(
                f"{device} 解码完成: finish_reason={finish_reason}, 本次生成tokens={new_token_count}",
            )
            # 更新剩余 max_tokens 以供下一轮
            total_tokens_generated += new_token_count
            max_tokens -= new_token_count
            decode_data["max_tokens"] = max_tokens

            # 下一轮使用新的调度决策
            if finish_reason == "scheduled":
                curr_decision = await self.scheduler.scheduler()

        decode_time = time.time() - start_time
        return {
            "result": decode_results[-1] if decode_results else {},
            "decode_results": decode_results,
            "total_tokens_generated": total_tokens_generated,
            "step_count": len(decode_results),
            "finish_reason": finish_reason,
            "decode_time": decode_time,
        }

    # ===== 私有解码步骤 =====
    async def _execute_decode_step(
        self,
        decode_data: dict[str, Any],
        device_type: str,
    ) -> dict[str, Any]:
        """执行单步解码：根据 device_type 发送请求，返回解码结果、生成文本和 token 数"""
        service_url = self.v1_chat_gpu if device_type == "GPU" else self.v1_chat_cpu
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                service_url,
                headers={"Content-Type": "application/json"},
                json=decode_data,
                timeout=300,
            )
        decode_time = time.time() - start_time
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
        Logger.info(
            f"{device_type} 解码完成: 耗时={decode_time:.3f}秒, 生成{new_token_count}个tokens",
        )
        return {
            "request_id": request_id,
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "new_token_count": new_token_count,
            "service_type": device_type,
            "decode_time": decode_time,
            "response": resp_json,
        }

    # 添加单步解码流式请求方法
    async def _execute_decode_step_stream(
        self,
        decode_data: dict[str, Any],
        device_type: str,
    ) -> AsyncGenerator[bytes, None]:
        """单步解码的流式请求"""
        service_url = self.v1_chat_gpu if device_type == "GPU" else self.v1_chat_cpu
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                service_url,
                headers={"Content-Type": "application/json"},
                json=decode_data,
                timeout=300,
            ) as response,
        ):
            if response.status_code != httpx.codes.OK:
                text = await response.aread()
                _raise_error(
                    f"{device_type} 解码流式请求失败: HTTP {response.status_code}, 响应: {text.decode()}",
                )
            async for chunk in response.aiter_bytes():
                yield chunk

    # ===== PD分离解码测试函数 =====
    async def test_completion_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        """处理 /v1/test/decode_sequence，根据sequence字段顺序进行PD分离"""
        sequence = data.get("sequence", [("GPU", 0)])

        # 如果序列为空或第一个是GPU且token_limit为0，使用默认请求
        if not sequence or (sequence[0][0] == "GPU" and sequence[0][1] == 0):
            return await self.default_request(data)

        # 执行 prefill
        prefill = await self.prefill_request(data.copy())
        completion_id = prefill["completion_id"]
        max_tokens = data.get("max_tokens", self.default_max_tokens)

        # 按序列顺序执行解码
        decode_results = []
        total_tokens_generated = 0
        last_generated_text = ""
        finish_reason = "scheduled"
        remaining_max_tokens = max_tokens

        start_time = time.time()

        for device, token_limit in sequence:
            if finish_reason != "scheduled":
                break

            # 如果token_limit为0，使用剩余的所有tokens，否则确保不超过剩余的max_tokens
            actual_token_limit = (
                remaining_max_tokens + 1 if token_limit == 0 else min(token_limit, remaining_max_tokens)
            )

            if actual_token_limit <= 0:
                break

            Logger.info(f"序列解码: 使用{device}解码，token限制={actual_token_limit}")

            # 构造解码请求数据
            decode_data = data.copy()
            decode_data["request_id_inference"] = completion_id
            decode_data["num_decode_tokens"] = actual_token_limit
            decode_data["generated_text"] = last_generated_text
            decode_data["max_tokens"] = actual_token_limit

            try:
                step_res = await self._execute_decode_step(decode_data, device)
                completion_id = step_res.get("request_id", completion_id)
                new_token_count = step_res.get("new_token_count", 0)
                finish_reason = step_res.get("finish_reason", "scheduled")
                last_generated_text = step_res.get("generated_text", last_generated_text)

                decode_results.append(step_res)
                total_tokens_generated += new_token_count
                remaining_max_tokens -= new_token_count

                Logger.info(f"{device} 解码完成: finish_reason={finish_reason}, 生成tokens={new_token_count}")

                # 如果已经完成或没有剩余tokens，退出循环
                if finish_reason != "scheduled" or remaining_max_tokens <= 0:
                    break

            except (AdaptiveDecoderError, httpx.RequestError, ValueError) as e:
                Logger.error(f"{device} 解码请求异常: {e!s}", exc_info=True)
                if not decode_results:
                    _raise_error(f"首次解码失败，无法继续: {e!s}", e)
                break

        decode_time = time.time() - start_time

        # 构建最终响应
        if decode_results:
            response = decode_results[-1]["response"]
            service_used = [step.get("service_type") for step in decode_results]
        else:
            response = {}
            service_used = []

        if "usage" not in response:
            response["usage"] = {}

        response["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
            "total_tokens_generated": total_tokens_generated,
            "sequence_steps": len(decode_results),
        }

        return response
