from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()   # 加载环境变量

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

print("🚀 正在调用大模型...")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一名友好的AI助教。"},
        {"role": "user", "content": "你好，你是谁?"}
    ],
    stream=False
)

print(response)