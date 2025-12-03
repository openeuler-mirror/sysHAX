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
import httpx
import asyncio
from typing import Any, NoReturn
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.utils.logger import Logger

# 创建路由器
router = APIRouter()

def raise_http_exception(status_code: int, detail: str) -> NoReturn:
    """抛出HTTP异常，返回指定状态码和错误信息"""
    raise HTTPException(status_code=status_code, detail=detail)


@router.post("/v1/chat/completions", response_model=None)
async def completions(request: Request) -> StreamingResponse:
    scheduler = request.app.state.scheduler
    if scheduler is None:
        raise HTTPException(status_code=500, detail="自适应解码器未初始化")

    try:
        data: dict[str, Any] = await request.json()
        is_stream = data.get("stream", False)
        output_queue = await scheduler.submit_task(data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效JSON")

    if is_stream:
        async def event_generator():
            try:
                while True:
                    chunk = await output_queue.get()
                    if chunk is None:  # 流结束或出错
                        break
                    yield chunk
            except asyncio.CancelledError:
                Logger.info("客户端断开连接，停止生成")
                raise

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    else:
        final_chunk = await output_queue.get()
        return Response(
            content=final_chunk,
            media_type="application/json"
        )

@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def fallback_to_gpu(request: Request, full_path: str) -> Response:
    """Fallback: 未识别接口时转发给 GPU 服务"""
    gpu_host = request.app.state.config.gpu_host
    gpu_port = request.app.state.config.gpu_port
    REQUEST_TIMEOUT = request.app.state.config.request_timeout
    url = f"http://{gpu_host}:{gpu_port}/{full_path}"
    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        async with httpx.AsyncClient() as client:
            resp = await client.request(request.method, url, headers=headers, content=body, timeout=REQUEST_TIMEOUT)
    except httpx.TimeoutException as e:
        Logger.error(f"转发到 GPU 服务超时: {e!s}", exc_info=True)
        raise HTTPException(status_code=504, detail="GPU 服务请求超时") from e
    except httpx.RequestError as e:
        Logger.error(f"转发到 GPU 服务失败: {e!s}")
        raise HTTPException(status_code=502, detail="GPU 服务不可用") from e
    if resp.status_code == httpx.codes.NOT_FOUND:
        raise HTTPException(status_code=httpx.codes.NOT_FOUND, detail="接口不存在")
    return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers)
