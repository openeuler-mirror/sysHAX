# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import httpx
import json
import time
from typing import Optional, Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from src.workflow import AdaptiveDecoder
from src.core.monitor import SystemMonitor
from src.core.decider import SchedulingDecider
from src.core.benchmark import PerformanceTester
from src.utils.logger import Logger

log = Logger

# 全局变量引用 - 这些变量会在应用启动时由main.py注入
system_monitor: Optional[SystemMonitor] = None  # 系统监控器实例
scheduling_decider: Optional[SchedulingDecider] = None  # 调度决策器实例
performance_tester: Optional[PerformanceTester] = None  # 性能测试器实例
adaptive_decoder: Optional[AdaptiveDecoder] = None  # 自适应解码器实例

# 创建路由器
router = APIRouter()


@router.post("/v1/completions")
async def completions(request: Request):
    """处理完成请求
    实现分离前缀填充和自适应解码流程：
    1. 发送prefill请求到GPU服务，设置prefill_then_swapout=True
    2. 根据资源情况和请求特征，执行自适应解码
    """
    # 确保系统组件已初始化
    if adaptive_decoder is None:
        raise HTTPException(status_code=500, detail="自适应解码器未初始化")
        
    try:
        # 获取请求数据
        data: Dict = await request.json()
        prompt = data.get("prompt", "")
        
        # 记录请求ID和时间戳
        request_id = f"req_{int(time.time())}_{id(request)}"
        log.info(f"收到请求 {request_id}: prompt长度={len(prompt)}")
        
        try:
            # 第一步：执行prefill请求
            prefill_result = await adaptive_decoder.prefill_request(data)
            completion_id = prefill_result["completion_id"]
            active_token = prefill_result["active_token"]
            
            # 第二步：执行decode请求
            decode_data = data.copy()
            decode_data["continue_decoding"] = f"{completion_id}-0"
            decode_data["active_token"] = active_token
            
            decode_result = await adaptive_decoder.decode_request(decode_data)
            
            # 提取结果
            result = decode_result["response"]
            decode_time = decode_result["decode_time"]
            service_used = decode_result["service_type"]
                
        except httpx.RequestError as e:
            # 请求错误处理
            log.error(f"请求 {request_id} 解码失败: {str(e)}")
            return {
                "id": f"error_{request_id}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": data.get("model", "unknown"),
                "choices": [{
                    "text": "处理请求时发生错误，请稍后再试。",
                    "index": 0,
                    "finish_reason": "error"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt.split())
                }
            }
        
        # 记录完成日志
        log.info(f"请求 {request_id} 完成，decode耗时={decode_time:.3f}秒，服务={service_used}")
        
        # 添加性能指标
        if "usage" not in result:
            result["usage"] = {}

        result["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
        }
        
        return result
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的JSON")
    except Exception as e:
        log.error(f"处理请求出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


# 添加一个新的端点，用于测试自定义解码序列（主要用于测试）
@router.post("/v1/test/decode_sequence")
async def test_decode_sequence(request: Request):
    """执行自定义解码序列测试
    
    请求格式：
    {
        "model": 模型名称,
        "prompt": "测试提示词",
        "max_tokens": 50,
        "sequence": [
            ["CPU", 10],   // [设备类型, token限制]
            ["GPU", null]  // null表示不限制token数量
        ]
    }
    """
    # 确保系统组件已初始化
    if adaptive_decoder is None:
        raise HTTPException(status_code=500, detail="自适应解码器未初始化")
        
    try:
        # 获取请求数据
        data: Dict = await request.json()
        prompt = data.get("prompt", "")
        
        # 获取解码序列，默认为CPU到GPU的切换
        sequence_data = data.get("sequence", [["CPU", 10], ["GPU", None]])
        
        # 转换为预期的格式
        sequence = [(item[0], item[1]) for item in sequence_data]
        
        # 记录请求
        sequence_desc = ", ".join([f"{s[0]}({s[1] if s[1] is not None else '无限制'})" for s in sequence])
        log.info(f"收到解码序列测试请求: 序列=[{sequence_desc}], prompt长度={len(prompt)}")
        
        # 第一步：执行prefill请求
        prefill_result = await adaptive_decoder.prefill_request(data)
        completion_id = prefill_result["completion_id"]
        active_token = prefill_result["active_token"]
        
        # 设置decode请求参数
        decode_data = data.copy()
        decode_data["continue_decoding"] = f"{completion_id}-0"
        decode_data["active_token"] = active_token
        
        # 执行测试解码序列
        decode_result = await adaptive_decoder.test_decode_sequence(
            decode_data, 
            sequence
        )
        
        # 提取结果
        result = decode_result["response"]
        decode_time = decode_result["decode_time"]
        service_used = decode_result["service_type"]
        
        # 添加性能指标
        if "usage" not in result:
            result["usage"] = {}
            
        result["conductor_metrics"] = {
            "decode_time_ms": int(decode_time * 1000),
            "service_used": service_used,
        }
        
        log.info(f"解码序列测试完成: 服务={service_used}, 耗时={decode_time:.3f}秒")
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的JSON")
    except Exception as e:
        log.error(f"解码序列测试出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.get("/metrics")
async def get_metrics():
    """返回当前的资源指标"""
    if system_monitor is None:
        raise HTTPException(status_code=500, detail="系统监控器未初始化")
        
    await system_monitor.update_metrics()
    
    # 获取GPU和CPU指标
    gpu_metrics = system_monitor.get_gpu_metrics()
    cpu_metrics = system_monitor.get_cpu_metrics()
    
    log.info(f"GPU指标: {gpu_metrics}")
    log.info(f"CPU指标: {cpu_metrics}")
    
    # 格式化数值以提高可读性
    return JSONResponse(
        content={
            "gpu": {
                "cache_usage": f"{gpu_metrics['gpu_cache_usage']:.2f}",
                "num_running": gpu_metrics['num_running'],
                "num_waiting": gpu_metrics['num_waiting'],
                "num_swapped": gpu_metrics['num_swapped'],
                "prefill_throughput": f"{gpu_metrics['prefill_throughput']:.2f} tokens/s",
                "decode_throughput": f"{gpu_metrics['decode_throughput']:.2f} tokens/s"
            },
            "cpu": {
                "cache_usage": f"{cpu_metrics['cpu_cache_usage']:.2f}",
                "num_running": cpu_metrics['num_running'],
                "num_waiting": cpu_metrics['num_waiting'],
                "num_swapped": cpu_metrics['num_swapped'],
                "prefill_throughput": f"{cpu_metrics['prefill_throughput']:.2f} tokens/s", 
                "decode_throughput": f"{cpu_metrics['decode_throughput']:.2f} tokens/s"
            },
            "system": {
                "last_update": system_monitor.last_update_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        },
        # 设置缩进使JSON输出格式化
        media_type="application/json"
    )
