"""
API路由模块

负责处理HTTP请求，并调用自适应解码器进行处理

Copyright (c) 2025 Huawei Technologies Co., Ltd.
sysHAX is licensed under Mulan PSL v2. See LICENSE for details.
"""

from __future__ import annotations

import json
import time
from typing import Any, NoReturn, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.utils.config import GPU_URL
from src.utils.logger import Logger
from src.workflow import AdaptiveDecoder, AdaptiveDecoderError

# 创建路由器
router = APIRouter()


def raise_http_exception(status_code: int, detail: str) -> NoReturn:
    """抛出HTTP异常，返回指定状态码和错误信息"""
    raise HTTPException(status_code=status_code, detail=detail)


async def _gpu_completion(data: dict[str, Any]) -> JSONResponse:
    async with httpx.AsyncClient() as client:
        gpu_response = await client.post(
            GPU_URL,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=300,
        )
    if gpu_response.status_code != httpx.codes.OK:
        Logger.error(f"GPU服务请求失败: HTTP {gpu_response.status_code}")
        raise_http_exception(gpu_response.status_code, "GPU服务错误")
    return JSONResponse(content=gpu_response.json())


async def _cpu_completion(
    data: dict[str, Any],
    adaptive_decoder: AdaptiveDecoder,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], float, list[str], str]:
    Logger.info("启动PD分离")
    try:
        prefil_data = data.copy()
        prefill_result = await adaptive_decoder.prefill_request(prefil_data)
    except httpx.RequestError as e:
        Logger.error(f"Prefill 请求失败: {e!s}")
        raise_http_exception(500, "Prefill 请求失败")
    completion_id = prefill_result["completion_id"]
    decode_data = data.copy()
    decode_result = await adaptive_decoder.decode_request(decode_data, completion_id, decision)
    response = decode_result["result"]["response"]
    decode_time = decode_result["decode_time"]
    service_used = [step.get("service_type") for step in decode_result.get("decode_results", [])]
    return response, decode_time, service_used, completion_id


@router.post("/v1/completions", response_model=None)
async def completions(request: Request) -> Union[JSONResponse, dict[str, Any]]:  # noqa: UP007
    """
    处理完成请求

    实现分离前缀填充和自适应解码流程：
    1. 发送prefill请求到GPU服务，设置prefill_then_swapout=True
    2. 根据资源情况和请求特征，执行自适应解码
    """
    adaptive_decoder = request.app.state.adaptive_decoder
    if adaptive_decoder is None:
        raise HTTPException(status_code=500, detail="自适应解码器未初始化")

    try:
        # 获取请求数据
        data: dict[str, Any] = await request.json()
        prompt = data.get("prompt", "")

        # 记录请求ID
        Logger.info(f"收到请求，prompt: {prompt}")

        # 1. 调度决策
        # 直接使用应用状态中单独创建的 Scheduler
        decision: dict[str, Any] = await request.app.state.scheduler.scheduler()
        # GPU 全流程
        if decision.get("device") == "GPU":
            return await _gpu_completion(data)

        # CPU 全流程
        if decision.get("device") == "CPU":
            result, decode_time, service_used, completion_id = await _cpu_completion(data, adaptive_decoder, decision)
        else:
            # 系统繁忙
            return JSONResponse(content={"error": "系统繁忙，请稍后再试"}, status_code=503)

        # 记录完成日志
        Logger.info(f"请求 {completion_id} 完成，decode耗时={decode_time:.3f}秒，服务序列={service_used}")

        # 添加性能指标
        if "usage" not in result:
            result["usage"] = {}

        result["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
        }

    except json.JSONDecodeError:
        raise_http_exception(400, "无效的JSON")
    except AdaptiveDecoderError as e:
        Logger.error(f"自适应解码器异常: {e!s}", exc_info=True)
        raise_http_exception(500, f"解码器内部错误: {e!s}")
    except (httpx.RequestError, TypeError, KeyError) as e:
        Logger.error(f"处理请求出错: {e!s}", exc_info=True)
        raise_http_exception(500, f"内部服务器错误: {e!s}")
    else:
        return result


# 添加一个新的端点，用于测试自定义解码序列（主要用于测试）
@router.post("/v1/test/decode_sequence", response_model=None)
async def test_decode_sequence(request: Request) -> Union[JSONResponse, dict[str, Any]]:  # noqa: UP007
    """
    执行解码测试：GPU prefill + CPU decode

    请求格式：
    {
        "model": 模型名称,
        "prompt": "测试提示词",
        "max_tokens": 50
    }
    """
    adaptive_decoder = request.app.state.adaptive_decoder
    if adaptive_decoder is None:
        raise HTTPException(status_code=500, detail="自适应解码器未初始化")

    try:
        # 获取请求数据
        data: dict = await request.json()
        prompt = data.get("prompt", "")
        Logger.info(f"收到测试解码请求: prompt长度={len(prompt)}")

        # 强制执行: GPU prefill + CPU decode
        prefill_result = await adaptive_decoder.prefill_request(data)
        completion_id = prefill_result["completion_id"]
        # CPU decode via decode_request, 初始 token_limit = max_tokens+1
        max_tokens = data.get("max_tokens", adaptive_decoder.default_max_tokens)
        initial_decision = {"device": "CPU", "token_limit": max_tokens + 1}
        decode_data = data.copy()
        decode_result = await adaptive_decoder.decode_request(decode_data, completion_id, initial_decision)
        # 提取最后一次 decode 的原始 response
        result = decode_result["result"]["response"]
        decode_time = decode_result["decode_time"]
        service_used = [step.get("service_type") for step in decode_result.get("decode_results", [])]

        # 添加性能指标
        if "usage" not in result:
            result["usage"] = {}

        result["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
        }

        Logger.info(f"解码序列测试完成: 服务序列={service_used}, 耗时={decode_time:.3f}秒")

    except json.JSONDecodeError:
        raise_http_exception(400, "无效的JSON")
    except AdaptiveDecoderError as e:
        Logger.error(f"自适应解码器异常: {e!s}", exc_info=True)
        raise_http_exception(500, f"解码器内部错误: {e!s}")
    except (httpx.RequestError, TypeError, KeyError) as e:
        Logger.error(f"解码序列测试出错: {e!s}", exc_info=True)
        raise_http_exception(500, f"内部服务器错误: {e!s}")
    else:
        return result


@router.get("/metrics", response_model=None)
async def get_metrics(request: Request) -> Union[JSONResponse, dict[str, Any]]:  # noqa: UP007
    """返回当前的资源指标"""
    system_monitor = request.app.state.monitor
    if system_monitor is None:
        raise HTTPException(status_code=500, detail="系统监控器未初始化")

    await system_monitor.update_metrics()

    # 获取GPU和CPU指标
    gpu_metrics = system_monitor.get_gpu_metrics()
    cpu_metrics = system_monitor.get_cpu_metrics()

    Logger.info(f"GPU指标: {gpu_metrics}")
    Logger.info(f"CPU指标: {cpu_metrics}")

    # 格式化数值以提高可读性
    return JSONResponse(
        content={
            "gpu": {
                "cache_usage": f"{gpu_metrics['gpu_cache_usage']:.2f}",
                "num_running": gpu_metrics["num_running"],
                "num_waiting": gpu_metrics["num_waiting"],
                "num_swapped": gpu_metrics["num_swapped"],
                "prompt_throughput": f"{gpu_metrics['prompt_throughput']:.2f} tokens/s",
                "generation_throughput": f"{gpu_metrics['generation_throughput']:.2f} tokens/s",
            },
            "cpu": {
                "cache_usage": f"{cpu_metrics['cpu_cache_usage']:.2f}",
                "num_running": cpu_metrics["num_running"],
                "num_waiting": cpu_metrics["num_waiting"],
                "num_swapped": cpu_metrics["num_swapped"],
                "prompt_throughput": f"{cpu_metrics['prompt_throughput']:.2f} tokens/s",
                "generation_throughput": f"{cpu_metrics['generation_throughput']:.2f} tokens/s",
            },
            "system": {
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(system_monitor.last_update_time)),
            },
        },
        # 设置缩进使JSON输出格式化
        media_type="application/json",
    )
