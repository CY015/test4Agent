from openai import OpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
import os

load_dotenv()   # 加载环境变量

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# def firstTest():
#     print("🚀 正在调用大模型...")
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "你是一名友好的AI助教。"},
#             {"role": "user", "content": "你好，你是谁?"}
#         ],
#         stream=False
#     )
#     print(response)

# 2.定义工具，通过注释描述工具
@tool
def getWeather(loc:str)->str:
    """获取指定城市的天气。"""
    return f'Current weather in {loc} is sunny'

agent = create_agent(
    model="deepseek-chat",
    # model="deepseek-v4-flash",  # 模型名称
    tools=[getWeather]  # 工具集
)
print("🚀 正在调用大模型...")
response = agent.invoke(
    {"messages":[{"role":"user", "content":"广州今天的天气怎么样？"}]}
)
print(response)

# if __name__ == "__main__":
#     # firstTest()
#     pass