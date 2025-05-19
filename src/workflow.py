# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import httpx
from typing import Dict, Any
import time

from src.core.monitor import SystemMonitor
from src.core.scheduler import Scheduler
from src.utils.config import GPU_URL, CPU_URL, DEFAULT_MAX_TOKENS
from src.utils.logger import Logger

class AdaptiveDecoder:
    """自适应解码器，负责在GPU和CPU之间动态切换解码任务"""
    
    def __init__(self, system_monitor, scheduler):
        """初始化自适应解码器"""
        self.system_monitor: SystemMonitor = system_monitor
        self.scheduler: Scheduler = scheduler
        self.prefill_url = GPU_URL
        self.decode_url = CPU_URL
        self.default_max_tokens = DEFAULT_MAX_TOKENS

    # ===== 核心对外API =====
    async def default_request(self, data: Dict[str, Any]) -> Dict:
        """直接执行默认的 GPU 全流程请求"""
        start_time = time.time()
        # 直接将原始请求转发到 GPU 服务
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.prefill_url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=300
            )
        decode_time = time.time() - start_time
        if response.status_code != 200:
            Logger.error(f"默认请求失败: HTTP {response.status_code}, 响应: {response.text}")
            raise Exception(f"默认请求失败: HTTP {response.status_code}")
        json_response = response.json()
        return {
            "response": json_response,
            "decode_time": decode_time,
            "service_type": "GPU"
        }
    
    async def prefill_request(self, data: Dict[str, Any]) -> Dict:
        """执行prefill请求
        
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
                    self.prefill_url,
                    headers={"Content-Type": "application/json"},
                    json=prefill_data,
                    timeout=300
                )
                
                if response.status_code != 200:
                    Logger.error(f"Prefill请求失败: HTTP {response.status_code}, 响应: {response.text}")
                    raise Exception(f"Prefill请求失败: HTTP {response.status_code}")
                
                prefill_time = time.time() - start_time
                prefill_response = response.json()
                
                # 提取必要信息
                completion_id = prefill_response.get("id")
                if not completion_id:
                    raise Exception("Prefill响应缺少completion ID")
                
                Logger.info(f"Prefill完成: 耗时={prefill_time:.3f}秒, completion_id={completion_id}")
                
                # 仅返回生成的 request ID
                return {
                    "completion_id": completion_id,
                    "prefill_time": prefill_time,
                    "prefill_response": prefill_response,
                }
            
            except Exception as e:
                Logger.error(f"Prefill请求异常: {str(e)}", exc_info=True)
                raise
    
    async def decode_request(self, decode_data: Dict[str, Any], completion_id: str, decision: Dict[str, Any]) -> Dict:
        """执行解码请求，实现自适应解码，直到 finish_reason 不为 'scheduled'"""
        start_time = time.time()
        Logger.info(f"开始动态解码请求：max_tokens={decode_data.get('max_tokens', self.default_max_tokens)}")

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
            Logger.info(f"调度决策：使用{device}解码，token限制={token_limit or '不限'}")

            # 构造本轮请求数据
            request_data = decode_data.copy()
            request_data["request_id_inference"] = completion_id
            request_data["num_decode_tokens"] = token_limit
            request_data["generated_text"] = last_generated_text

            try:
                step_res = await self._execute_decode_step(request_data, device)
            except Exception as e:
                Logger.error(f"{device} 解码请求异常: {e}", exc_info=True)
                if not decode_results:
                    raise ValueError(f"首次解码失败，无法继续: {str(e)}")
                break

            completion_id = step_res.get("request_id", "")
            total_tokens_generated += step_res.get("new_token_count", 0)
            finish_reason = step_res.get("finish_reason")
            decode_results.append(step_res)
            last_generated_text = step_res.get("generated_text", "")
            Logger.info(f"{device} 解码完成: finish_reason={finish_reason}, 共生成tokens={total_tokens_generated}")
            
            # 下一轮使用新的调度决策
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
    
    # ===== 解码器基础实现 =====
    async def _execute_decode_step(self, decode_data: Dict[str, Any], device_type: str) -> Dict[str, Any]:
        """执行单步解码：根据 device_type 发送请求，返回解码结果、生成文本和 token 数"""
        service_url = self.prefill_url if device_type == "GPU" else self.decode_url
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                service_url,
                headers={"Content-Type": "application/json"},
                json=decode_data,
                timeout=300
            )
        decode_time = time.time() - start_time
        if response.status_code != 200:
            Logger.error(f"{device_type} 解码请求失败: HTTP {response.status_code}, 响应: {response.text}")
            raise Exception(f"{device_type} 解码请求失败: HTTP {response.status_code}")
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
        Logger.info(f"{device_type} 解码完成: 耗时={decode_time:.3f}秒, 生成{new_token_count}个tokens")
        return {
            "request_id": request_id,
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "new_token_count": new_token_count,
            "service_type": device_type,
            "decode_time": decode_time,
            "response": resp_json,
        }
