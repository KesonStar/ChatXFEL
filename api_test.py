import requests
import json

url = "http://10.15.102.186:9000/api/chat"

# ====== 2. 请求体 ======
data = {
    "model": "qwen3:30b-a3b-instruct-2507-q8_0",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "stream": False
}

# ====== 3. 发送请求 ======
try:
    response = requests.post(
        url,
        json=data,
        timeout=600  
    )
    response.raise_for_status()

    res_json = response.json()
    print("\n=== 模型响应 ===\n")
    print(json.dumps(res_json, indent=2, ensure_ascii=False))

except Exception as e:
    print("请求失败：", e)