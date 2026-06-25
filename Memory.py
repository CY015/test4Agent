from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 创建智能体时指定checkpointer，LangChain会自动帮我们管理历史会话记忆
agent = create_agent(
    "deepseek-chat",
    checkpointer=InMemorySaver()    # 基于内存
)

# 设定thread_id，作为会话标识
config = {"configurable": {"thread_id": "thread_1"}}

# 第一次调用，告知AI我的信息
response = agent.invoke(
    {"messages": [HumanMessage(content="你好，我叫虎哥，我最喜欢冰冰。")]},
    config # 调用时添加thread_id，区分不同会话
)

print(response)

# 第二次调用，询问我的信息，这次带上thread_id，唤起记忆
response = agent.invoke(
    {"messages": [HumanMessage(content="我最喜欢的是什么？")]},
    config # 调用时添加thread_id
)

print(response)