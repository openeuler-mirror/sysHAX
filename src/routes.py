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
Desc:sysHAX API路由模块
"""

from __future__ import annotations

import json
import time
from typing import Any, NoReturn, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.utils.config import GPU_HOST, GPU_PORT
from src.utils.logger import Logger
from src.workflow import AdaptiveDecoderError

# 创建路由器
router = APIRouter()


def raise_http_exception(status_code: int, detail: str) -> NoReturn:
    """抛出HTTP异常，返回指定状态码和错误信息"""
    raise HTTPException(status_code=status_code, detail=detail)


@router.post("/v1/chat/completions", response_model=None)
async def completions(request: Request) -> Any:
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
        data: dict[str, Any] = await request.json()
        # 支持流式 PD 分离和 GPU 全流程
        if data.get("stream"):
            return StreamingResponse(
                adaptive_decoder.chat_completion_stream(data),
                media_type="text/event-stream",
            )
        return await adaptive_decoder.chat_completion(data)
    except json.JSONDecodeError:
        raise_http_exception(400, "无效请求")
    except AdaptiveDecoderError as e:
        Logger.error(f"自适应解码器异常: {e!s}", exc_info=True)
        raise_http_exception(500, f"解码器内部错误: {e!s}")
    except (httpx.RequestError, ValueError, KeyError, AttributeError) as e:
        Logger.error(f"处理请求出错: {e!s}", exc_info=True)
        raise_http_exception(500, f"内部服务器错误: {e!s}")


# 测试接口
@router.post("/v1/test/decode_sequence", response_model=None)
async def test_decode_sequence(request: Request) -> Any:
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
        data: dict[str, Any] = await request.json()
        # 支持流式 PD 分离和 GPU 全流程
        if data.get("stream"):
            return StreamingResponse(
                adaptive_decoder.test_completion_stream(data),
                media_type="text/event-stream",
            )
        return await adaptive_decoder.test_completion(data)
    except json.JSONDecodeError:
        raise_http_exception(400, "无效请求")
    except AdaptiveDecoderError as e:
        Logger.error(f"自适应解码器异常: {e!s}", exc_info=True)
        raise_http_exception(500, f"解码器内部错误: {e!s}")
    except (httpx.RequestError, ValueError, KeyError, AttributeError) as e:
        Logger.error(f"处理请求出错: {e!s}", exc_info=True)
        raise_http_exception(500, f"内部服务器错误: {e!s}")


@router.get("/metrics", response_model=None)
async def get_metrics(request: Request) -> Union[JSONResponse, dict[str, Any]]:
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


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def fallback_to_gpu(request: Request, full_path: str) -> Response:
    """Fallback: 未识别接口时转发给 GPU 服务"""
    url = f"http://{GPU_HOST}:{GPU_PORT}/{full_path}"
    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        async with httpx.AsyncClient() as client:
            resp = await client.request(request.method, url, headers=headers, content=body, timeout=300)
    except httpx.RequestError as e:
        Logger.error(f"转发到 GPU 服务失败: {e!s}")
        raise HTTPException(status_code=502, detail="GPU 服务不可用") from e
    if resp.status_code == httpx.codes.NOT_FOUND:
        raise HTTPException(status_code=httpx.codes.NOT_FOUND, detail="接口不存在")
    return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers)
