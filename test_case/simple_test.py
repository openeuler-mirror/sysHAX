import requests
import time

# 强制使用CPU参与解码
print("-"*100)
print("使用GPU+CPU混合模式解码")
url = "http://172.168.178.64:7011/v1/test/decode_sequence"
headers = {"Content-Type": "application/json"}
data = {
    "model": "ds1.5b",
    "prompt": "中国是一个",
    "max_tokens": 100,
    "temperature": 0,
    "sequence": [
        ["GPU", 50],
        ["CPU", None]
    ]
}

start_time = time.time()
response = requests.post(url, headers=headers, json=data)
request_time = time.time() - start_time
result = response.json()
cpu_gpu_text = result["choices"][0]["text"]
print(f"生成的文本: {cpu_gpu_text}")
print(f"文本长度: {len(cpu_gpu_text)} 字符")
print(f"请求耗时: {request_time:.2f} 秒")
print(f"平均生成速度: {len(cpu_gpu_text) / request_time:.2f} 字符/秒")

time.sleep(5)

# 纯GPU解码
print("-"*100)
print("纯CPU解码")
url = "http://172.168.178.64:7011/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "ds1.5b",
    "prompt": "中国是一个",
    "max_tokens": 100,
    "temperature": 0
}

start_time = time.time()
response = requests.post(url, headers=headers, json=data)
request_time = time.time() - start_time
result = response.json()
gpu_text = result["choices"][0]["text"]
print(f"生成的文本: {gpu_text}")
print(f"文本长度: {len(gpu_text)} 字符")
print(f"请求耗时: {request_time:.2f} 秒")
print(f"平均生成速度: {len(gpu_text) / request_time:.2f} 字符/秒")
print("-"*100)
