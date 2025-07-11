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
Desc:sysHAX 主程序
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from src import routes
from src.core.benchmark import PerformanceTester
from src.core.monitor import SystemMonitor
from src.core.scheduler import Scheduler
from src.utils.config import SYSHAX_HOST, SYSHAX_PORT
from src.utils.logger import Logger
from src.workflow import AdaptiveDecoder, AdaptiveDecoderError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """管理应用生命周期：初始化监控、调度器、解码器和性能测试，并在关闭时清理资源"""
    Path("logs").mkdir(parents=True, exist_ok=True)
    app.state.monitor = SystemMonitor()
    app.state.scheduler = Scheduler(system_monitor=app.state.monitor)
    app.state.adaptive_decoder = AdaptiveDecoder(
        system_monitor=app.state.monitor, scheduler=app.state.scheduler,
    )
    app.state.performance_tester = PerformanceTester(
        system_monitor=app.state.monitor,
        decoder=app.state.adaptive_decoder,
    )
    Logger.info("开始执行性能基准测试...")
    try:
        await app.state.performance_tester.run_benchmarks()
        performance_summary = app.state.performance_tester.get_performance_summary()
        Logger.info(
            f"性能测试完成，性能比: {performance_summary.get('performance_ratio', 0):.2f}x",
        )
    except (AdaptiveDecoderError, httpx.RequestError) as e:
        Logger.error(f"性能测试失败: {e!s}", exc_info=True)
    Logger.info("应用启动完成，API接口已就绪")
    try:
        yield
    finally:
        Logger.info("应用关闭，正在清理资源...")


# 将 FastAPI 实例化并使用 lifespan 管理生命周期
app = FastAPI(lifespan=lifespan)

# 注册路由
app.include_router(routes.router)


def run() -> None:
    """主程序入口"""
    # 从配置读取主机和端口
    host = SYSHAX_HOST
    port = SYSHAX_PORT

    # 启动FastAPI应用
    uvicorn.run("main:app", host=host, port=port)


if __name__ == "__main__":
    run()
