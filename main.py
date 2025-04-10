# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import os
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import GPU_METRICS_URL, CPU_METRICS_URL, DEFAULT_SCHEDULING_PARAMS
from src.core.monitor import SystemMonitor
from src.core.decider import SchedulingDecider
from src.core.benchmark import PerformanceTester
from src.workflow import AdaptiveDecoder
from src.utils.logger import Logger
import src.routes as routes

log = Logger

# 全局变量
# 系统监控器实例
system_monitor_instance: Optional[SystemMonitor] = None
# 调度决策器实例
scheduling_decider_instance: Optional[SchedulingDecider] = None
# 性能测试器实例
performance_tester_instance: Optional[PerformanceTester] = None
# 自适应解码器实例
adaptive_decoder_instance: Optional[AdaptiveDecoder] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理，处理启动和关闭事件

    启动时：
    1. 创建系统监控器，用于跟踪GPU和CPU服务的性能和负载
    2. 创建调度决策器，用于决定请求应该在GPU还是CPU上执行
    3. 创建性能测试器，用于测量GPU和CPU服务的基准性能
    4. 创建自适应解码器，用于处理自适应解码逻辑
    5. 执行性能测试，然后根据结果更新调度参数
    
    关闭时：
    1. 清理资源
    """
    global system_monitor_instance, scheduling_decider_instance, performance_tester_instance, adaptive_decoder_instance
    
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 初始化系统组件
    system_monitor_instance = SystemMonitor(
        gpu_metrics_url=GPU_METRICS_URL,
        cpu_metrics_url=CPU_METRICS_URL
    )
    
    scheduling_decider_instance = SchedulingDecider(
        system_monitor=system_monitor_instance
    )
    
    # 创建自适应解码器实例
    adaptive_decoder_instance = AdaptiveDecoder(
        system_monitor=system_monitor_instance,
        scheduling_decider=scheduling_decider_instance
    )
    
    performance_tester_instance = PerformanceTester(
        system_monitor=system_monitor_instance,
    )
    
    # 设置performance_tester的自适应解码器实例
    performance_tester_instance.set_adaptive_decoder(adaptive_decoder_instance)
    
    log.info("系统组件初始化完成")
    
    # 将实例注入到routes模块
    routes.system_monitor = system_monitor_instance
    routes.scheduling_decider = scheduling_decider_instance
    routes.performance_tester = performance_tester_instance
    routes.adaptive_decoder = adaptive_decoder_instance
    
    # 检查是否跳过基准测试（仅通过环境变量设置）
    skip_benchmark = os.environ.get("SKIP_BENCHMARK", "").lower() in ("1", "true", "yes")
    
    if skip_benchmark:
        log.info("SKIP_BENCHMARK设置，跳过基准测试")
        # 使用默认配置
        scheduling_decider_instance.update_parameters(
            system_busy_threshold=DEFAULT_SCHEDULING_PARAMS["system_busy_threshold"]
        )
    # 执行性能基准测试
    elif performance_tester_instance is not None:
        log.info("开始执行性能基准测试...")
        try:
            # 执行测试，直接调用异步方法
            await performance_tester_instance.run_benchmarks()
            
            # 获取性能测试结果摘要
            performance_summary = performance_tester_instance.get_performance_summary()
            log.info(f"性能测试完成，性能比: {performance_summary.get('performance_ratio', 0):.2f}x")
            
        except Exception as e:
            error_msg = f"性能测试失败: {str(e)}"
            log.error(error_msg)
            import traceback
            traceback.print_exc()  # 打印详细堆栈跟踪
            
            
        # 确保调度参数正确配置
        scheduling_decider_instance.update_parameters(
            system_busy_threshold=DEFAULT_SCHEDULING_PARAMS["system_busy_threshold"]
        )
    
    log.info("应用启动完成，API接口已就绪")
    yield  # 应用运行中
    
    # 应用关闭时的清理操作
    log.info("应用关闭，正在清理资源...")

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(routes.router)

def run():
    """主程序入口"""
    # 启动FastAPI应用，不使用热重载功能
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=7011
    )

if __name__ == "__main__":
    run() 