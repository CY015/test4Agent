from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model='deepseek-chat')
# print(type(model))  # <class 'langchain_deepseek.chat_models.ChatDeepSeek'>

# # 调用invoke方法(阻塞式调用，一次性返回)
# response = model.invoke("月亮的首都是哪里？")
# # 查看响应结果
# print(response)

# # 通过stream方法实现流式访问
# stream = model.stream("月亮的首都是哪里？")
# # stream调用返回的结果是一个generator，方便我们循环获取结果
# print(type(stream))
# # 遍历stream结果，实时打印AI的回复
# for chunk in stream:
#     print(chunk.content, end="", flush=True)

agent = create_agent(model=model)

response = agent.invoke({"messages":[{"role":"user", "content":"广州今天的天气怎么样？"}]})
print("阻塞式调用：",response)

print("流式调用：")
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "广州今天的天气怎么样？"}]},
    stream_mode="messages"
):
    if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token
