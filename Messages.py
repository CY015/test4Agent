# from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
# model = init_chat_model(model='deepseek-chat')
agent = create_agent(model='deepseek-chat')
response = agent.invoke({
    'messages':[
        HumanMessage(content='你好，我是Ghost'),
        AIMessage(content='你好，Ghost，很高兴认识你'),
        HumanMessage(content='我的名字是什么')
    ]
})

# 查看完整的消息历史
for msg in response['messages']:
    msg.pretty_print()  # 这会以更友好的格式打印每条消息

# 获取模型对“我的名字是什么”的最终回答
print(response['messages'][-1].content)