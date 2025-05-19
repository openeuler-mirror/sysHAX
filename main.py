# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.utils.config import GPU_METRICS_URL, CPU_METRICS_URL, SYSHAX_HOST, SYSHAX_PORT
from src.core.monitor import SystemMonitor
from src.core.scheduler import Scheduler
from src.core.benchmark import PerformanceTester
from src.workflow import AdaptiveDecoder
from src.utils.logger import Logger
import src.routes as routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs('logs', exist_ok=True)
    app.state.monitor = SystemMonitor(
        gpu_metrics_url=GPU_METRICS_URL,
        cpu_metrics_url=CPU_METRICS_URL
    )
    app.state.scheduler = Scheduler(system_monitor=app.state.monitor)
    app.state.adaptive_decoder = AdaptiveDecoder(
        system_monitor=app.state.monitor,
        scheduler=app.state.scheduler
    )
    app.state.performance_tester = PerformanceTester(
        system_monitor=app.state.monitor,
        decoder=app.state.adaptive_decoder
    )
    Logger.info("开始执行性能基准测试...")
    try:
        await app.state.performance_tester.run_benchmarks()
        performance_summary = app.state.performance_tester.get_performance_summary()
        Logger.info(f"性能测试完成，性能比: {performance_summary.get('performance_ratio', 0):.2f}x")
    except Exception as e:
        Logger.error(f"性能测试失败: {str(e)}", exc_info=True)
    Logger.info("应用启动完成，API接口已就绪")
    try:
        yield
    finally:
        Logger.info("应用关闭，正在清理资源...")

# 将 FastAPI 实例化并使用 lifespan 管理生命周期
app = FastAPI(lifespan=lifespan)

# 注册路由
app.include_router(routes.router)

def run():
    """主程序入口"""
    # 从配置读取主机和端口
    host = SYSHAX_HOST
    port = SYSHAX_PORT
    
    # 启动FastAPI应用
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port
    )

if __name__ == "__main__":
    run() 