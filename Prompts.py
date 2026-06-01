from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# 定义一个类，用来封装模型要输出的数据
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str
# agent = create_agent(
#     model='deepseek-chat',
#     system_prompt='像鲁迅一样说话.'
# )

# for token, metadata in agent.stream(
#     {'messages':[HumanMessage(content="你的兴趣爱好是什么？")]},
#     stream_mode='messages'
# ):
#     print(token.content, end="", flush=True)

# 身份(Identity): 描述AI的职责、沟通风格和总体目标。
# 说明(Instructions): 请指导模型如何生成所需的响应。它应该遵循哪些规则？模型应该做什么，以及模型绝对不能做什么？
# 示例(Examples): 提供可能的输入示例，以及模型期望的输出。
# 信息(Context): 向模型提供生成响应所需的任何额外信息，例如RAG的额外知识库数据，或您认为特别相关的任何其他数据。

system_prompt = """
# 身份
- 你是一个科幻作家，根据用户的要求创建一个太空之都。

# 指令
- 请务必以JSON格式输出，不要加任何markdown样式。

# 示例：
user: 月球的首都是什么？
assistant:
{
    "name": "月华城（Lunara）",
    "location": "位于月球正面赤道附近的静海基地遗址之上，依托巨大的穹顶与地下网络建成",
    "vibe": "冷冽、高效、革新",
    "economy": "氦-3能源开采、量子通信枢纽、尖端生物圈农业"
}
"""
agent = create_agent(
    model='deepseek-chat',
    system_prompt=system_prompt,
    response_format=CapitalInfo # 设置结构化输出的格式
)
response = agent.invoke(
    {"messages": [HumanMessage(content="月球的首都是什么?")]}
)
city = response['structured_response']
print(f"{city.name}位于{city.location}，是一座{city.vibe}的城市，其主要产业包括{city.economy}。")
# for token, metadata in agent.stream(
#     {"messages": [HumanMessage(content="金星的首都是什么")]},
#     stream_mode="messages"
# ):
#     print(token.content, end="", flush=True)