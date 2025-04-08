#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

def print_json(obj):
    """格式化打印JSON对象"""
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def test_service(service_name, url, prompt="中国是一个", max_tokens=10):
    """测试特定服务并返回详细信息"""
    print(f"\n{'='*50}\n测试{service_name}服务 ({url})\n{'='*50}")
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "ds1.5b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0
    }
    
    print(f"请求数据: {data}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        request_time = time.time() - start_time
        
        print(f"HTTP状态码: {response.status_code}")
        print(f"请求耗时: {request_time:.3f}秒")
        
        if response.status_code == 200:
            result = response.json()
            
            # 提取关键信息
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("text", "")
                token_ids = result["choices"][0].get("token_ids", [])
                
                print(f"\n生成的文本: {text}")
                print(f"文本长度: {len(text)} 字符")
                
                if token_ids:
                    print(f"token_ids数量: {len(token_ids)}")
                    print(f"token_ids(前10个): {token_ids[:10]}")
                else:
                    print("响应中不包含token_ids")
            
            # 打印完整响应
            print("\n完整响应:")
            print_json(result)
            
            return result
        else:
            print(f"请求失败: {response.text}")
            return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def main():
    # 测试参数
    prompt = "中国是一个"
    max_tokens = 10
    
    # 测试GPU服务 (7004)
    gpu_url = "http://172.168.178.64:7004/v1/completions"
    gpu_result = test_service("GPU", gpu_url, prompt, max_tokens)
    
    # 测试CPU服务 (7005)
    cpu_url = "http://172.168.178.64:7005/v1/completions"
    cpu_result = test_service("CPU", cpu_url, prompt, max_tokens)
    
    # 比较结果结构
    if gpu_result and cpu_result:
        print("\n\n比较GPU和CPU响应结构的差异:")
        gpu_keys = set(gpu_result.keys())
        cpu_keys = set(cpu_result.keys())
        
        # 找出两者共有和独有的keys
        common_keys = gpu_keys & cpu_keys
        gpu_only_keys = gpu_keys - cpu_keys
        cpu_only_keys = cpu_keys - gpu_keys
        
        print(f"共有的顶级字段: {sorted(list(common_keys))}")
        if gpu_only_keys:
            print(f"仅GPU有的顶级字段: {sorted(list(gpu_only_keys))}")
        if cpu_only_keys:
            print(f"仅CPU有的顶级字段: {sorted(list(cpu_only_keys))}")
        
        # 比较choices结构
        if "choices" in gpu_result and "choices" in cpu_result:
            if gpu_result["choices"] and cpu_result["choices"]:
                gpu_choice_keys = set(gpu_result["choices"][0].keys())
                cpu_choice_keys = set(cpu_result["choices"][0].keys())
                
                common_choice_keys = gpu_choice_keys & cpu_choice_keys
                gpu_only_choice_keys = gpu_choice_keys - cpu_choice_keys
                cpu_only_choice_keys = cpu_choice_keys - gpu_only_choice_keys
                
                print(f"\n共有的choices字段: {sorted(list(common_choice_keys))}")
                if gpu_only_choice_keys:
                    print(f"仅GPU有的choices字段: {sorted(list(gpu_only_choice_keys))}")
                if cpu_only_choice_keys:
                    print(f"仅CPU有的choices字段: {sorted(list(cpu_only_choice_keys))}")

if __name__ == "__main__":
    main()
