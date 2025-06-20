import json
import requests

url = "http://localhost:11434/api/generate"

# 提供json格式调用数据
data = {
    "model": "qwen3:1.7b",
    "prompt": "为什么天空是蓝色的？",  # /nothink表示不使用思考链
    "stream": False,  # 是否开启流式输出
}

response = requests.post(url, json=data)

if response.status_code == 200:  # 请求成功
    print(json.loads(response.text)['response'])
else:
    print(f"请求响应: {response.text}")