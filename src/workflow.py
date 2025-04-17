# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import httpx
from typing import Dict, List, Tuple, Optional
import time
import copy

from src.core.monitor import SystemMonitor
from src.core.decider import SchedulingDecider
from src.utils.config import GPU_URL, CPU_URL
from src.utils.logger import Logger

class AdaptiveDecoder:
    """自适应解码器，负责在GPU和CPU之间动态切换解码任务"""
    
    def __init__(self, system_monitor, scheduling_decider):
        """初始化自适应解码器"""
        self.system_monitor: SystemMonitor = system_monitor
        self.scheduling_decider: SchedulingDecider = scheduling_decider
        self.prefill_url = GPU_URL
        self.decode_url = CPU_URL
    
    # ===== 核心对外API =====
    
    async def prefill_request(self, data: Dict) -> Dict:
        """执行prefill请求
        
        Args:
            data: 包含model和prompt等参数的字典
            
        Returns:
            Dict: 包含completion_id和相关信息的字典
        """
        start_time = time.time()
        
        # 在prefill前调用决策器判断是否接受新请求
        decision = await self.scheduling_decider.make_scheduling_decision()

        # 如果决策器返回的设备类型不是GPU，则拒绝接收新的prefill请求
        if decision["device"] != 'GPU':
            Logger.warning("系统繁忙，拒绝接收新的prefill请求")
            return {"error": "系统繁忙，请稍后再试", "status": "busy"}
        
        # 准备prefill数据
        prefill_data = data.copy()
        prefill_data["max_tokens"] = 2
        prefill_data["prefill_then_swapout"] = True
        
        prompt_length = len(prefill_data.get("prompt", ""))
        Logger.info(f"执行prefill请求: prompt长度={prompt_length}")
        
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
                
                if (not prefill_response.get("choices") or 
                    not prefill_response["choices"][0].get("token_ids") or 
                    len(prefill_response["choices"][0]["token_ids"]) == 0):
                    raise Exception("Prefill响应缺少token IDs")
                
                active_token = prefill_response["choices"][0]["token_ids"][0]
                
                Logger.info(f"Prefill完成: 耗时={prefill_time:.3f}秒, completion_id={completion_id}")
                
                return {
                    "response": prefill_response,
                    "completion_id": completion_id,
                    "active_token": active_token,
                    "prefill_time": prefill_time
                }
            except Exception as e:
                Logger.error(f"Prefill请求异常: {str(e)}", exc_info=True)
                raise
    
    async def decode_request(self, decode_data: Dict) -> Dict:
        """执行解码请求，实现自适应解码
        
        Args:
            decode_data: 解码参数
            
        Returns:
            Dict: 解码结果
        """
        start_time = time.time()
        Logger.info(f"开始动态解码请求：max_tokens={decode_data.get('max_tokens', 100)}")
        
        # 保存原始decode_data，用于后续接力解码
        original_decode_data = decode_data.copy()
        current_decode_data = decode_data.copy()
        
        # 解码结果列表
        decode_results = []
        
        # 记录总共生成的token数和累积文本
        total_tokens_generated = 0
        accumulated_text = ""
        max_tokens = decode_data.get("max_tokens", 100)
        
        # 解码步骤计数
        step_count = 0
        
        while True:
            step_count += 1
            Logger.info(f"---------- 解码步骤 {step_count} 开始 ----------")
            
            try:
                # 调用调度决策器获取下一步解码信息
                decision = await self.scheduling_decider.make_scheduling_decision()
                device_type = decision["device"]
                token_limit = decision["token_limit"]
                
                # 即使决策器返回None，也要继续解码（因为prefill已经完成）
                if device_type is None:
                    device_type = "GPU"  # 系统繁忙时默认使用GPU继续解码
                    Logger.info("系统资源紧张，解码阶段继续使用GPU（不中断已开始的任务）")
                
                # 处理token限制
                remaining_tokens = max(1, max_tokens - total_tokens_generated)
                if token_limit > 0:
                    token_limit = min(token_limit, remaining_tokens)
                else:
                    # token_limit为0时，使用剩余token数作为限制
                    token_limit = remaining_tokens
                
                Logger.info(f"调度决策：使用{device_type}解码，token限制={token_limit}")
                
                # 执行解码步骤
                step_result = await self._execute_decode_step(
                    current_decode_data,
                    device_type,
                    token_limit
                )
                
                if not step_result["success"]:
                    Logger.error(f"解码步骤失败: {step_result.get('error', '未知错误')}")
                    break
                
                result = step_result["result"]
                decode_results.append(result)
                
                # 处理文本和token
                text_info = self._extract_text_tokens(
                    result,
                    step_count,
                    original_decode_data,
                    accumulated_text,
                    current_decode_data
                )
                
                new_token_count = text_info["new_token_count"]
                accumulated_text = text_info["accumulated_text"]
                total_tokens_generated += new_token_count
                
                Logger.info(f"新生成token数: {new_token_count}, 总计: {total_tokens_generated}/{max_tokens}")
                
                # 检查生成是否结束
                token_info = {
                    "new_count": new_token_count,
                    "total": total_tokens_generated,
                    "max": max(100, decode_data.get("max_tokens", 100)),
                    "is_first_step": step_count == 1
                }
                if self._is_generation_complete(result, token_info):
                    break
                
                # 准备下一步的解码数据
                Logger.info(f"---------- 准备接力数据 ----------")
                current_decode_data = self._prepare_continuation_for_relay(
                    original_decode_data,
                    result,
                    current_decode_data
                )
                
            except Exception as e:
                Logger.error(f"解码步骤{step_count}异常: {str(e)}", exc_info=True)
                if not decode_results:
                    raise ValueError(f"首次解码失败，无法继续: {str(e)}")
                break
            
            Logger.info(f"---------- 解码步骤 {step_count} 完成 ----------")
        
        return self._prepare_final_result(
            decode_results,
            accumulated_text,
            total_tokens_generated,
            step_count,
            start_time
        )
    
    # ===== 解码器基础实现 =====
    
    async def decode_on_gpu(self, decode_data: Dict, token_limit: int = None) -> Dict:
        """在GPU上执行解码"""
        service_type = "GPU"
        service_url = self.prefill_url
        
        # 处理token限制: token_limit为0或None时不限制
        if token_limit is not None and token_limit > 0:
            original_max_tokens = decode_data.get("max_tokens", 100)
            # 临时修改max_tokens
            decode_data["max_tokens"] = min(original_max_tokens, token_limit)
            result = await self.perform_decode(decode_data, service_url, service_type)
            # 恢复原始值
            decode_data["max_tokens"] = original_max_tokens
            return result
        return await self.perform_decode(decode_data, service_url, service_type)


    async def decode_on_cpu(self, decode_data: Dict, token_limit: int = None) -> Dict:
        """在CPU上执行解码"""
        service_type = "CPU"
        service_url = self.decode_url
        
        # 处理token限制: token_limit为0或None时不限制
        if token_limit is not None and token_limit > 0:
            original_max_tokens = decode_data.get("max_tokens", 100)
            # 临时修改max_tokens
            decode_data["max_tokens"] = min(original_max_tokens, token_limit)
            result = await self.perform_decode(decode_data, service_url, service_type)
            # 恢复原始值
            decode_data["max_tokens"] = original_max_tokens
            return result
        return await self.perform_decode(decode_data, service_url, service_type)

    async def perform_decode(self, decode_data: Dict, service_url: str, service_type: str) -> Dict:
        """执行解码请求"""
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    service_url,
                    headers={"Content-Type": "application/json"},
                    json=decode_data,
                    timeout=300
                )
                
                if response.status_code != 200:
                    Logger.error(f"{service_type} Decode请求失败: HTTP {response.status_code}, 响应: {response.text}")
                    raise Exception(f"{service_type} Decode请求失败: HTTP {response.status_code}")
                
                decode_time = time.time() - start_time
                response_json = response.json()
                
                generated_text = ""
                token_count = 0
                
                if (response_json.get("choices") and 
                    response_json["choices"][0].get("text")):
                    generated_text = response_json["choices"][0]["text"]
                
                # 获取token数量 (优先从usage获取)
                if response_json.get("usage") and "completion_tokens" in response_json["usage"]:
                    token_count = response_json["usage"]["completion_tokens"]
                elif (response_json.get("choices") and 
                      response_json["choices"][0].get("token_ids")):
                    token_count = len(response_json["choices"][0]["token_ids"])
                elif generated_text:
                    token_count = int(len(generated_text) / 1.5)
                
                Logger.info(f"{service_type}解码完成: 耗时={decode_time:.3f}秒, 生成{token_count}个tokens")
                
                return {
                    "response": response_json,
                    "service_type": service_type,
                    "decode_time": decode_time,
                    "decode_url": service_url,
                    "actual_token_count": token_count,
                    "generated_text": generated_text
                }
            except Exception as e:
                Logger.error(f"{service_type}解码失败: {str(e)}", exc_info=True)
                raise

    # ===== 辅助工具函数 =====

    async def _execute_decode_step(self, current_decode_data: Dict, device_type: str, token_limit: int) -> Dict:
        """执行单个解码步骤
        
        Args:
            current_decode_data: 当前解码数据
            device_type: 设备类型，如"GPU"或"CPU"
            token_limit: 最大允许的token数量，0表示不限制
            
        Returns:
            Dict: 包含解码结果和步骤信息的字典
        """
        # 精简返回结果结构
        step_result = {
            "success": True,
            "service_type": device_type,
            "token_limit": token_limit
        }
        
        try:
            if device_type == "GPU":
                result = await self.decode_on_gpu(current_decode_data, token_limit)
            else:
                result = await self.decode_on_cpu(current_decode_data, token_limit)
            
            step_result["result"] = result
            return step_result
        except Exception as e:
            Logger.error(f"{device_type}解码失败: {str(e)}", exc_info=True)
            step_result["success"] = False
            step_result["error"] = str(e)
            return step_result
    
    def _extract_text_tokens(self, result: Dict, step_count: int, original_data: Dict, 
                            accumulated_text: str, current_decode_data: Dict) -> Dict:
        """从解码结果中提取文本和计算token数量
        
        Args:
            result: 解码结果
            step_count: 当前步骤数
            original_data: 原始解码数据
            accumulated_text: 当前累积的文本
            current_decode_data: 当前解码数据
            
        Returns:
            Dict: 包含处理后的文本信息，包括新文本、token数量等
        """
        # 直接使用更简洁的方式提取文本
        current_text = result.get("generated_text", "")
        if not current_text and result["response"].get("choices"):
            current_text = result["response"]["choices"][0].get("text", "")
        
        text_info = {
            "new_token_count": 0,
            "current_text": current_text,
            "accumulated_text": accumulated_text
        }
        
        # 获取token数量
        token_count = result.get("actual_token_count", 0)
        if token_count == 0 and result["response"].get("usage") and "completion_tokens" in result["response"]["usage"]:
            token_count = result["response"]["usage"]["completion_tokens"]
        
        if step_count == 1:
            # 第一次解码
            text_info["new_token_count"] = token_count
            text_info["accumulated_text"] = current_text
        else:
            # 处理接力后的文本
            if current_decode_data.get("echo") == True:
                prompt_length = len(original_data.get("prompt", ""))
                prev_text_length = len(accumulated_text)
                
                if len(current_text) >= prompt_length + prev_text_length:
                    new_text = current_text[prompt_length + prev_text_length:]
                    Logger.info(f"新生成的文本长度: {len(new_text)}字符")
                    
                    text_info["accumulated_text"] = accumulated_text + new_text
                    text_info["new_token_count"] = token_count or int(len(new_text) / 1.5)
                else:
                    Logger.warning(f"无法从echo文本中提取新内容，可能接力失败")
                    text_info["new_token_count"] = token_count
                    text_info["accumulated_text"] = current_text
            else:
                text_info["new_token_count"] = token_count
                text_info["accumulated_text"] = current_text
        
        # 确保新token数量不为负数
        text_info["new_token_count"] = max(0, text_info["new_token_count"])
        
        return text_info

    def _is_generation_complete(self, result: Dict, token_info: Dict) -> bool:
        """判断文本生成是否已完成
        
        Args:
            result: 当前解码结果，包含response等数据
            token_info: 包含token相关信息的字典，必须包含以下键:
                - new_count: 新生成的token数量
                - total: 已生成的总token数量
                - max: 最大允许的token数量
                - is_first_step: 是否是第一步解码
            
        Returns:
            bool: 是否完成生成
        """
        # 提取token信息
        new_token_count = token_info.get("new_count", 0)
        total_tokens_generated = token_info.get("total", 0)
        max_tokens = token_info.get("max", 100)
        is_first_step = token_info.get("is_first_step", False)
        
        is_complete = False
        reason = ""
        
        # 检查finish_reason
        if (result["response"].get("choices") and 
            result["response"]["choices"][0].get("finish_reason")):
            finish_reason = result["response"]["choices"][0]["finish_reason"]
            
            if finish_reason in ["stop", "content_filter", "function_call", "tool_calls"]:
                is_complete = True
                reason = f"自然结束: {finish_reason}"
            elif finish_reason == "length":
                if total_tokens_generated >= max_tokens:
                    is_complete = True
                    reason = f"达到max_tokens限制({max_tokens})"
                elif new_token_count == 0:
                    is_complete = True
                    reason = "无新token生成"
        
        # 无新token生成且非首次解码
        if not is_complete and new_token_count == 0 and not is_first_step:
            is_complete = True
            reason = "无新token生成"
        
        # 达到最大tokens限制
        if not is_complete and total_tokens_generated >= max_tokens:
            is_complete = True
            reason = f"达到max_tokens限制({max_tokens})"

        if is_complete:
            Logger.info(f"生成完成: {reason}")

        return is_complete

    def _prepare_continuation_for_relay(self, original_data: Dict, last_result: Dict, current_data: Dict) -> Dict:
        """为解码接力准备继续解码的数据
        
        实现文本接力：将前一个设备生成的文本添加到下一个设备的prompt中，确保连续性。
        
        Args:
            original_data: 最初的解码请求数据
            last_result: 最近一次解码的结果
            current_data: 当前的解码数据
            
        Returns:
            Dict: 更新后的解码数据，用于下一个设备接力解码
        """
        # 修改current_data而不是创建新的字典
        relay_data = current_data.copy()
        
        # 获取上一次结果中的信息
        response = last_result["response"]
        
        try:
            # 提取生成的文本
            generated_text = last_result.get("generated_text", "")
            if not generated_text and response.get("choices") and response["choices"][0].get("text"):
                generated_text = response["choices"][0]["text"]
                
            if generated_text:
                # 将已生成文本添加到prompt
                original_prompt = original_data.get("prompt", "")
                relay_data["prompt"] = original_prompt + generated_text
            
            # 计算token数量和更新max_tokens
            token_count = 0
            if response.get("usage") and "completion_tokens" in response["usage"]:
                token_count = response["usage"]["completion_tokens"]
                
                # 更新max_tokens
                if "max_tokens" in original_data:
                    original_max_tokens = original_data.get("max_tokens", 100)
                    remaining_tokens = max(1, original_max_tokens - token_count)
                    relay_data["max_tokens"] = remaining_tokens
            
            # 传递其他必要参数
            for key in ["stop", "temperature", "top_p"]:
                if key in current_data and key not in relay_data:
                    relay_data[key] = current_data[key]
            
            # 设置echo=true获取完整文本
            relay_data["echo"] = True
            
            # 移除不需要的参数
            for key in ["continue_decoding", "active_token"]:
                relay_data.pop(key, None)
            
        except Exception as e:
            Logger.warning(f"准备接力数据出错: {str(e)}", exc_info=True)
            # 简单接力备用方案
            try:
                if (response.get("choices") and response["choices"][0].get("text")):
                    generated_text = response["choices"][0]["text"]
                    relay_data["prompt"] = original_data.get("prompt", "") + generated_text
                    relay_data["echo"] = True
            except Exception as e2:
                Logger.error(f"简单接力也失败: {str(e2)}")
        
        return relay_data
    
    def _merge_decode_results(self, results: List[Dict]) -> Dict:
        """合并多个解码结果
        
        在解码继承的正确逻辑下，我们只需要使用最后一次解码的结果，
        因为它已经包含了整个生成过程的完整结果
        
        Args:
            results: 解码结果列表
            
        Returns:
            Dict: 合并后的结果
        """
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        final_result = copy.deepcopy(results[-1])
        
        try:
            service_types = [r["service_type"] for r in results]
            service_used = "混合" if len(set(service_types)) > 1 else service_types[0]
            
            decode_time = sum(r.get("decode_time", 0) for r in results)
            final_result["decode_time"] = decode_time
            
            # 记录服务类型变化
            service_changes = []
            for i in range(1, len(service_types)):
                if service_types[i] != service_types[i-1]:
                    service_changes.append(f"{service_types[i-1]}->{service_types[i]}")
            
            if service_changes:
                service_used += f" (切换: {', '.join(service_changes)})"
            
            final_result["service_type"] = service_used
        except Exception as e:
            Logger.warning(f"更新性能指标出错: {str(e)}")
            if "service_type" not in final_result:
                final_result["service_type"] = "未知"
            if "decode_time" not in final_result:
                final_result["decode_time"] = 0.0
        
        return final_result
    
    def _prepare_final_result(self, decode_results: List[Dict], accumulated_text: str, 
                             total_tokens_generated: int, step_count: int, start_time: float) -> Dict:
        """准备最终的解码结果
        
        Args:
            decode_results: 解码结果列表
            accumulated_text: 累积的文本
            total_tokens_generated: 生成的token总数
            step_count: 总步骤数
            start_time: 开始时间
            
        Returns:
            Dict: 最终的解码结果
        """
        # 合并所有结果
        final_result = self._merge_decode_results(decode_results)
        
        # 将最终文本放入结果
        if (final_result.get("response") and final_result["response"].get("choices") and 
            len(final_result["response"]["choices"]) > 0):
            final_result["response"]["choices"][0]["text"] = accumulated_text
        
        # 添加性能指标
        total_time = time.time() - start_time
        final_result["total_tokens_generated"] = total_tokens_generated
        final_result["total_decode_steps"] = step_count
        final_result["total_decode_time"] = total_time
        
        Logger.info(f"========== 动态解码完成 ==========")
        Logger.info(f"总耗时: {total_time:.3f}秒, 总tokens: {total_tokens_generated}, 步骤: {step_count}")
        
        return final_result

    # ===== 测试函数 =====
    
    async def test_decode_sequence(self, decode_data: Dict, sequence: List[Tuple[str, Optional[int]]]) -> Dict:
        """测试指定解码序列，用于测试不同设备间的切换
        
        Args:
            decode_data: 解码参数
            sequence: 解码服务序列，每项为元组(服务类型, token限制)，例如：[("CPU", 10), ("GPU", None)]
            
        Returns:
            Dict: 最终的解码结果
        """
        start_time = time.time()
        Logger.info(f"开始测试解码序列，共{len(sequence)}个阶段")
        
        results = []
        current_decode_data = decode_data.copy()
        original_decode_data = decode_data.copy()  # 保存原始数据，用于接力
        total_tokens_generated = 0
        accumulated_text = ""
        
        for idx, (service_type, token_limit) in enumerate(sequence):
            step_count = idx + 1
            Logger.info(f"---------- 测试阶段 {step_count} 开始 ----------")
            Logger.info(f"使用{service_type}解码，token限制={token_limit if token_limit is not None else '不限制'}")
            
            try:
                # 使用统一的_execute_decode_step方法执行解码
                step_result = await self._execute_decode_step(
                    current_decode_data,
                    service_type,
                    token_limit
                )
                
                if not step_result["success"]:
                    Logger.error(f"解码步骤失败: {step_result.get('error', '未知错误')}")
                    break
                
                result = step_result["result"]
                results.append(result)
                
                # 处理文本和token
                text_info = self._extract_text_tokens(
                    result,
                    step_count,
                    original_decode_data,
                    accumulated_text,
                    current_decode_data
                )
                
                # 更新token和文本数据
                new_token_count = text_info["new_token_count"]
                accumulated_text = text_info["accumulated_text"]
                total_tokens_generated += new_token_count
                
                Logger.info(f"新生成token数: {new_token_count}, 总计: {total_tokens_generated}")
                
                # 检查生成是否结束
                token_info = {
                    "new_count": new_token_count,
                    "total": total_tokens_generated,
                    "max": max(100, decode_data.get("max_tokens", 100)),
                    "is_first_step": idx == 0
                }
                if self._is_generation_complete(result, token_info):
                    Logger.info("生成已完成，提前结束测试序列")
                    break
                
                # 如果不是最后一段解码，准备下一段的数据
                if idx < len(sequence) - 1:
                    Logger.info(f"---------- 准备接力数据 ----------")
                    current_decode_data = self._prepare_continuation_for_relay(
                        original_decode_data,  # 使用原始数据而不是第一个参数
                        result, 
                        current_decode_data
                    )
            
            except Exception as e:
                Logger.error(f"解码阶段{step_count}异常: {str(e)}", exc_info=True)
                if not results:
                    raise ValueError(f"首次解码失败，无法继续: {str(e)}")
                break
            
            Logger.info(f"---------- 测试阶段 {step_count} 完成 ----------")
        
        # 准备最终结果
        final_result = self._prepare_final_result(
            results,
            accumulated_text,
            total_tokens_generated,
            len(results),
            start_time
        )
        
        Logger.info(f"测试解码序列完成，总共执行了{len(results)}个阶段，生成了{total_tokens_generated}个tokens")
        
        return final_result