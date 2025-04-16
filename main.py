# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.utils.config import GPU_METRICS_URL, CPU_METRICS_URL, SYSHAX_HOST, SYSHAX_PORT
from src.core.monitor import SystemMonitor
from src.core.decider import SchedulingDecider
from src.core.benchmark import PerformanceTester
from src.workflow import AdaptiveDecoder
from src.utils.logger import Logger
from src.components import Components
import src.routes as routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理，处理启动和关闭事件

    启动时：
    1. 创建系统监控器，用于跟踪GPU和CPU服务的性能和负载
    2. 创建调度决策器，用于决定请求应该在GPU还是CPU上执行
    3. 创建自适应解码器，用于处理自适应解码逻辑
    4. 创建性能测试器，用于测量GPU和CPU服务的基准性能
    5. 执行性能测试，然后根据结果更新调度参数
    
    关闭时：
    1. 清理资源
    """
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 创建系统监控器实例
    Components.monitor = SystemMonitor(
        gpu_metrics_url=GPU_METRICS_URL,
        cpu_metrics_url=CPU_METRICS_URL
    )
    
    # 创建调度决策器实例
    Components.decider = SchedulingDecider(
        system_monitor=Components.monitor
    )
    
    # 创建自适应解码器实例
    Components.adaptive_decoder = AdaptiveDecoder(
        system_monitor=Components.monitor,
        scheduling_decider=Components.decider
    )
    
    # 创建性能测试器实例
    Components.performance_tester = PerformanceTester(
        system_monitor=Components.monitor,
        decoder=Components.adaptive_decoder
    )
    
    Logger.info("系统组件初始化完成")
    
    # 执行性能基准测试
    Logger.info("开始执行性能基准测试...")
    try:
        # 执行测试
        await Components.performance_tester.run_benchmarks()
        
        # 获取性能测试结果摘要
        performance_summary = Components.performance_tester.get_performance_summary()
        Logger.info(f"性能测试完成，性能比: {performance_summary.get('performance_ratio', 0):.2f}x")
        
    except Exception as e:
        Logger.error(f"性能测试失败: {str(e)}", exc_info=True)
    
    Logger.info("应用启动完成，API接口已就绪")
    yield  # 应用运行中
    
    # 应用关闭时的清理操作
    Logger.info("应用关闭，正在清理资源...")

# 创建FastAPI应用
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