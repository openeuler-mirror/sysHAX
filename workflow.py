import httpx
from typing import Dict, List, Tuple
import time
import copy

from config import PREFILL_URL, DECODE_URL
from logger import Logger

# 使用自定义日志管理器
log = Logger

class AdaptiveDecoder:
    """自适应解码器，处理自适应解码逻辑
    
    负责在GPU和CPU之间动态切换解码任务，根据系统状态选择合适的解码设备。
    """
    
    def __init__(self, system_monitor, scheduling_decider):
        """初始化自适应解码器
        
        Args:
            system_monitor: 系统监控器实例
            scheduling_decider: 调度决策器实例
        """
        self.system_monitor = system_monitor
        self.scheduling_decider = scheduling_decider
        self.prefill_url = PREFILL_URL
        self.decode_url = DECODE_URL
    
    # ===== 核心对外API =====
    
    async def prefill_request(self, data: Dict) -> Dict:
        """执行prefill请求
        
        Args:
            data: 包含model和prompt等参数的字典
            
        Returns:
            Dict: 包含completion_id、active_token和prefill_time的字典
        """
        start_time = time.time()
        
        # 准备prefill数据
        prefill_data = data.copy()
        prefill_data["max_tokens"] = 2  # 只生成少量token
        prefill_data["prefill_then_swapout"] = True  # 开启prefill_then_swapout
        
        # 记录请求信息
        prompt_length = len(prefill_data.get("prompt", ""))
        log.info(f"执行prefill请求: prompt长度={prompt_length}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.prefill_url,
                    headers={"Content-Type": "application/json"},
                    json=prefill_data,
                    timeout=300
                )
                
                if response.status_code != 200:
                    log.error(f"Prefill请求失败: HTTP {response.status_code}, 响应: {response.text}")
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
                
                log.info(f"Prefill完成: 耗时={prefill_time:.3f}秒, completion_id={completion_id}")
                
                # 返回prefill信息
                return {
                    "response": prefill_response,
                    "completion_id": completion_id,
                    "active_token": active_token,
                    "prefill_time": prefill_time
                }
            except Exception as e:
                log.error(f"Prefill请求异常: {str(e)}", exc_info=True)
                raise
    
    async def decode_request(self, decode_data: Dict) -> Dict:
        """执行解码请求，实现自适应解码
        
        根据系统状态动态调度解码任务，能够在CPU和GPU之间进行智能切换。
        
        Args:
            decode_data: 解码参数
            
        Returns:
            Dict: 解码结果
        """
        start_time = time.time()
        log.info(f"开始动态解码请求：max_tokens={decode_data.get('max_tokens', 100)}")
        
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
            log.info(f"---------- 解码步骤 {step_count} 开始 ----------")
            
            try:
                # 调用调度决策器获取下一步解码信息
                # 使用新的接口形式，直接获取device_type和token_limit
                device_type, token_limit = ("GPU", None)# await self.scheduling_decider.make_scheduling_decision()
                
                # 如果系统繁忙（device_type为None），中止解码
                if device_type is None:
                    log.warning("系统负载过高，终止解码")
                    break
                
                # 确保不超过剩余的max_tokens限制
                remaining_tokens = max(1, max_tokens - total_tokens_generated)
                token_limit = min(token_limit, remaining_tokens)
                
                log.info(f"调度决策：使用{device_type}解码，token限制={token_limit}，剩余tokens={remaining_tokens}")
                
                # 执行解码步骤
                step_result = await self._execute_decode_step(
                    current_decode_data,
                    device_type,
                    token_limit
                )
                
                # 检查是否成功
                if not step_result["success"]:
                    log.error(f"解码步骤失败: {step_result.get('error', '未知错误')}")
                    break
                
                # 获取结果
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
                
                # 更新token和文本数据
                new_token_count = text_info["new_token_count"]
                accumulated_text = text_info["accumulated_text"]
                total_tokens_generated += new_token_count
                
                log.info(f"新生成token数: {new_token_count}, 总计: {total_tokens_generated}/{max_tokens}")
                
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
                log.info(f"---------- 准备接力数据 ----------")
                current_decode_data = self._prepare_continuation_for_relay(
                    original_decode_data,
                    result,
                    current_decode_data
                )
                log.info(f"接力准备完成，继续下一步解码")
                
            except Exception as e:
                log.error(f"解码步骤{step_count}异常: {str(e)}", exc_info=True)
                if not decode_results:
                    raise ValueError(f"首次解码失败，无法继续: {str(e)}")
                break
            
            log.info(f"---------- 解码步骤 {step_count} 完成 ----------")
        
        # 准备最终结果
        return self._prepare_final_result(
            decode_results,
            accumulated_text,
            total_tokens_generated,
            step_count,
            start_time
        )
    
    # ===== 解码器基础实现 =====
    
    async def decode_on_gpu(self, decode_data: Dict, token_limit: int = None) -> Dict:
        """在GPU上执行解码
        
        Args:
            decode_data: 解码参数
            token_limit: token数量限制，可选
            
        Returns:
            Dict: 解码结果
        """
        service_type = "GPU"
        service_url = self.prefill_url  # GPU服务的URL
        
        # 如果指定了token限制，应用限制
        if token_limit is not None:
            decode_data_copy = decode_data.copy()
            original_max_tokens = decode_data_copy.get("max_tokens", 100)
            decode_data_copy["max_tokens"] = min(original_max_tokens, token_limit)
            return await self.perform_decode(decode_data_copy, service_url, service_type)
        return await self.perform_decode(decode_data, service_url, service_type)


    async def decode_on_cpu(self, decode_data: Dict, token_limit: int = None) -> Dict:
        """在CPU上执行解码
        
        Args:
            decode_data: 解码参数
            token_limit: token数量限制，可选
            
        Returns:
            Dict: 解码结果
        """
        service_type = "CPU"
        service_url = self.decode_url  # CPU服务的URL
        
        # 如果指定了token限制，应用限制
        if token_limit is not None:
            decode_data_copy = decode_data.copy()
            original_max_tokens = decode_data_copy.get("max_tokens", 100)
            decode_data_copy["max_tokens"] = min(original_max_tokens, token_limit)
            return await self.perform_decode(decode_data_copy, service_url, service_type)
        return await self.perform_decode(decode_data, service_url, service_type)

    async def perform_decode(self, decode_data: Dict, service_url: str, service_type: str) -> Dict:
        """执行decode请求
        
        Args:
            decode_data: 解码参数
            service_url: 服务URL
            service_type: 服务类型（GPU或CPU）
            
        Returns:
            Dict: 包含response、service_type、decode_time和decode_url的解码结果
        """
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
                    log.error(f"{service_type} Decode请求失败: HTTP {response.status_code}, 响应: {response.text}")
                    raise Exception(f"{service_type} Decode请求失败: HTTP {response.status_code}")
                
                decode_time = time.time() - start_time
                response_json = response.json()
                
                # 记录生成的文本内容
                generated_text = ""
                log_text = "无文本"
                token_count = 0
                
                # 从文本获取内容
                if (response_json.get("choices") and 
                    response_json["choices"][0].get("text")):
                    generated_text = response_json["choices"][0]["text"]
                    # 只记录前100个字符，避免日志过长
                    log_text = generated_text[:100] + ("..." if len(generated_text) > 100 else "")
                
                # 从usage获取token数量（最准确的方式）
                if response_json.get("usage") and "completion_tokens" in response_json["usage"]:
                    token_count = response_json["usage"]["completion_tokens"]
                # 备选：从token_ids获取
                elif (response_json.get("choices") and 
                      response_json["choices"][0].get("token_ids")):
                    token_count = len(response_json["choices"][0]["token_ids"])
                # 如果都没有，尝试从文本估算
                elif generated_text:
                    token_count = int(len(generated_text) / 1.5)
                
                # 记录解码完成信息
                log.info(f"{service_type}解码完成: 耗时={decode_time:.3f}秒, 生成{token_count}个tokens")
                
                # 返回结果
                return {
                    "response": response_json,
                    "service_type": service_type,
                    "decode_time": decode_time,
                    "decode_url": service_url,
                    "actual_token_count": token_count,  # 添加实际生成的token数量
                    "generated_text": generated_text    # 添加生成的文本
                }
            except Exception as e:
                log.error(f"{service_type}解码失败: {str(e)}", exc_info=True)
                raise

    # ===== 辅助工具函数 =====

    async def _execute_decode_step(self, current_decode_data: Dict, use_gpu: bool, token_limit: int) -> Dict:
        """执行单个解码步骤
        
        Args:
            current_decode_data: 当前解码数据
            use_gpu: 是否使用GPU
            token_limit: 最大允许的token数量
            
        Returns:
            Dict: 包含解码结果和步骤信息的字典
        """
        # 准备返回结果
        step_result = {
            "success": True,
            "system_busy": False,
            "result": None,
            "service_type": "GPU" if use_gpu else "CPU",
            "token_limit": token_limit
        }
        
        try:
            # 执行解码
            if use_gpu:
                result = await self.decode_on_gpu(current_decode_data, token_limit)
            else:
                result = await self.decode_on_cpu(current_decode_data, token_limit)
            
            step_result["result"] = result
            return step_result
        except Exception as e:
            log.error(f"{step_result['service_type']}解码失败: {str(e)}", exc_info=True)
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
        # 提取生成的文本
        current_text = result.get("generated_text", "")
        if not current_text and result["response"].get("choices") and result["response"]["choices"][0].get("text"):
            current_text = result["response"]["choices"][0]["text"]
        
        # 处理返回结果
        text_info = {
            "new_token_count": 0,
            "current_text": current_text,
            "accumulated_text": accumulated_text
        }
        
        # 计算新生成的token数量
        if step_count == 1:
            # 第一次解码，直接使用生成的所有token
            text_info["new_token_count"] = result.get("actual_token_count", 0)
            if text_info["new_token_count"] == 0 and result["response"].get("usage") and "completion_tokens" in result["response"]["usage"]:
                text_info["new_token_count"] = result["response"]["usage"]["completion_tokens"]
            text_info["accumulated_text"] = current_text
        else:
            # 处理接力后的文本
            if current_decode_data.get("echo") == True:
                # 对于echo=true的响应，需要提取新生成的部分
                prompt_length = len(original_data.get("prompt", ""))
                prev_text_length = len(accumulated_text)
                
                # 检查是否包含完整的前文
                if len(current_text) >= prompt_length + prev_text_length:
                    # 提取新生成的内容
                    new_text = current_text[prompt_length + prev_text_length:]
                    log.info(f"新生成的文本: '{new_text}'")
                    
                    # 更新累积文本
                    text_info["accumulated_text"] = accumulated_text + new_text
                    
                    # 计算新token数量
                    if result["response"].get("usage") and "completion_tokens" in result["response"]["usage"]:
                        text_info["new_token_count"] = result["response"]["usage"]["completion_tokens"]
                    else:
                        text_info["new_token_count"] = int(len(new_text) / 1.5)
                else:
                    log.warning(f"无法从echo文本中提取新内容，可能接力失败")
                    text_info["new_token_count"] = result.get("actual_token_count", 0)
                    text_info["accumulated_text"] = current_text
            else:
                # 非echo响应
                text_info["new_token_count"] = result.get("actual_token_count", 0)
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
        
        # 初始化为未完成
        is_complete = False
        reason = ""
        
        # 检查结束原因
        if (result["response"].get("choices") and 
            result["response"]["choices"][0].get("finish_reason")):
            finish_reason = result["response"]["choices"][0]["finish_reason"]
            
            if finish_reason in ["stop", "content_filter", "function_call", "tool_calls"]:
                # 这些原因意味着生成已自然结束
                is_complete = True
                reason = f"自然结束，原因: {finish_reason}"
            elif finish_reason == "length":
                # 达到了当前步骤的token限制，但可能没有达到总体限制
                if total_tokens_generated >= max_tokens:
                    is_complete = True
                    reason = f"达到总体max_tokens限制({max_tokens})"
                elif new_token_count == 0:
                    is_complete = True
                    reason = "达到token限制且无新token生成"
                else:
                    reason = f"当前步骤达到token限制，需要继续生成"
        
        # 如果没有新token生成，也认为生成结束（非首次解码）
        if not is_complete and new_token_count == 0 and not is_first_step:
            is_complete = True
            reason = "没有新token生成"
        
        # 如果达到最大tokens限制
        if not is_complete and total_tokens_generated >= max_tokens:
            is_complete = True
            reason = f"达到总体max_tokens限制({max_tokens})"

        if is_complete:
            log.info(f"生成已完成或达到max_tokens限制，结束解码，原因: {reason}")

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
        # 创建新的解码数据
        new_decode_data = original_data.copy()
        
        # 获取上一次结果中的信息
        response = last_result["response"]
        
        try:
            # 1. 提取生成的文本并添加到prompt
            generated_text = ""
            if last_result.get("generated_text"):
                generated_text = last_result["generated_text"]
            elif (response.get("choices") and response["choices"][0].get("text")):
                generated_text = response["choices"][0]["text"]
                
            if generated_text:
                # 将已生成的文本添加到prompt，创建上下文连续性
                original_prompt = original_data.get("prompt", "")
                new_prompt = original_prompt + generated_text
                new_decode_data["prompt"] = new_prompt
                log.info(f"更新prompt: 添加上一阶段生成的文本")
            
            # 2. 计算token数量
            token_count = 0
            
            # 从usage获取token数量（最准确的方式）
            if response.get("usage") and "completion_tokens" in response["usage"]:
                token_count = response["usage"]["completion_tokens"]
                log.info(f"已生成token数: {token_count} (从usage获取)")
            
            # 3. 更新max_tokens
            if "max_tokens" in original_data:
                original_max_tokens = original_data.get("max_tokens", 100)
                remaining_tokens = max(1, original_max_tokens - token_count)
                new_decode_data["max_tokens"] = remaining_tokens
                log.info(f"更新接力max_tokens: {remaining_tokens} (原始:{original_max_tokens}, 已生成:{token_count})")
                
            # 4. 传递其他必要参数
            for key in ["stop", "temperature", "top_p"]:
                if key in current_data:
                    new_decode_data[key] = current_data[key]
            
            # 5. 设置echo=true以获取完整文本
            new_decode_data["echo"] = True
            log.info(f"设置echo=true以获取完整文本")
            
            # 6. 移除不需要的参数
            for key in ["continue_decoding", "active_token"]:
                if key in new_decode_data:
                    del new_decode_data[key]
            
        except Exception as e:
            log.warning(f"准备接力解码数据时出错: {str(e)}", exc_info=True)
            # 出错时的简单接力
            try:
                if (response.get("choices") and response["choices"][0].get("text")):
                    generated_text = response["choices"][0]["text"]
                    new_decode_data["prompt"] = original_data.get("prompt", "") + generated_text
                    new_decode_data["echo"] = True
                    log.info(f"出错后的简单接力: 将生成的文本添加到prompt")
            except Exception as e2:
                log.error(f"简单接力也失败: {str(e2)}")
        
        return new_decode_data 
    
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
        
        # 如果只有一个结果，直接返回
        if len(results) == 1:
            return results[0]
        
        # 使用最后一次解码的结果作为最终结果
        final_result = copy.deepcopy(results[-1])
        
        # 添加性能指标
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
            log.warning(f"更新性能指标时出错: {str(e)}")
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
        
        # 将最终累积的文本放入结果
        if (final_result.get("response") and final_result["response"].get("choices") and 
            len(final_result["response"]["choices"]) > 0):
            final_result["response"]["choices"][0]["text"] = accumulated_text
        
        # 添加总体性能指标
        total_time = time.time() - start_time
        final_result["total_tokens_generated"] = total_tokens_generated
        final_result["total_decode_steps"] = step_count
        final_result["total_decode_time"] = total_time
        
        log.info(f"========== 动态解码完成 ==========")
        log.info(f"总耗时: {total_time:.3f}秒")
        log.info(f"总生成tokens: {total_tokens_generated}个")
        log.info(f"总解码步骤: {step_count}步")
        
        return final_result

    # ===== 测试函数 =====
    
    async def test_decode_sequence(self, decode_data: Dict, sequence: List[Tuple[str, int]]) -> Dict:
        """测试指定解码序列，用于测试不同设备间的切换
        
        实现解码接力：将前一个设备生成的文本添加到下一个设备的prompt中，确保连续性。
        
        Args:
            decode_data: 解码参数
            sequence: 解码服务序列，每项为元组(服务类型, token限制)，例如：[("CPU", 10), ("GPU", None)]
            
        Returns:
            Dict: 最终的解码结果
        """
        start_time = time.time()
        results = []
        current_decode_data = decode_data.copy()
        
        # 记录总共生成的token数和累积文本
        total_tokens_generated = 0
        accumulated_text = ""
        
        for idx, (service_type, token_limit) in enumerate(sequence):
            # 阶段开始日志
            log.info(f"---------- 第{idx+1}段解码开始 ----------")
            log.info(f"在{service_type}上执行解码, token限制={token_limit if token_limit else '无限制'}")
            log.info(f"当前累积文本: '{accumulated_text}'")
            
            try:
                # 根据服务类型选择解码方法
                if service_type.upper() == "CPU":
                    result = await self.decode_on_cpu(current_decode_data, token_limit)
                elif service_type.upper() == "GPU":
                    result = await self.decode_on_gpu(current_decode_data, token_limit)
                else:
                    raise ValueError(f"不支持的服务类型: {service_type}")
                
                results.append(result)
                
                # 处理文本和token
                text_info = self._extract_text_tokens(
                    result,
                    idx + 1,  # 步骤数从1开始
                    decode_data,  # 原始解码数据
                    accumulated_text,
                    current_decode_data
                )
                
                # 更新token和文本数据
                new_token_count = text_info["new_token_count"]
                accumulated_text = text_info["accumulated_text"]
                total_tokens_generated += new_token_count
                
                log.info(f"新生成token数: {new_token_count}")
                log.info(f"总计生成token数: {total_tokens_generated}")
                
                # 检查生成是否结束
                token_info = {
                    "new_count": new_token_count,
                    "total": total_tokens_generated,
                    "max": max(100, decode_data.get("max_tokens", 100)),
                    "is_first_step": idx == 0
                }
                if self._is_generation_complete(result, token_info):
                    break
                
                # 如果不是最后一段解码，准备下一段的数据
                if idx < len(sequence) - 1:
                    log.info(f"---------- 准备接力数据 ----------")
                    # 准备接力数据
                    current_decode_data = self._prepare_continuation_for_relay(
                        decode_data, 
                        result, 
                        current_decode_data
                    )
                    log.info(f"接力准备完成, max_tokens={current_decode_data.get('max_tokens', 'N/A')}")
            
            except Exception as e:
                log.error(f"解码异常: {str(e)}", exc_info=True)
                if not results:
                    raise ValueError(f"{service_type}解码阶段失败，无法继续")
                break
            
            # 阶段结束日志
            log.info(f"---------- 第{idx+1}段解码完成 ----------")
        
        # 准备最终结果
        final_result = self._prepare_final_result(
            results,
            accumulated_text,
            total_tokens_generated,
            len(results),  # 步骤数等于结果数
            start_time
        )
        
        # 额外添加累积文本（与_prepare_final_result保持兼容）
        final_result["accumulated_text"] = accumulated_text
        
        return final_result
        
