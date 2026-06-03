from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# 自定义工具
@tool
def getWeather(loc:str) -> str:
    # 自定义工具要有一段描述(通过注释描述工具)
    """
    Get the weather in a given location.
    Args:
        loc: city name or coordinates
    """
    return f'{loc} weather is cool'

# 使用网页搜素引擎
tool1 = TavilySearch(
    max_results=5,
    topic='general'
)

# 绑定工具
agent = create_agent(
    model='deepseek-chat',
    tools=[tool1],
    system_prompt='你是一个智能助手，使用工具来解决用户问题。'
)

# 调用Agent
# response = agent.invoke(
#     {"messages": [HumanMessage(content="广州今天天气如何?")]},
# )
#
# for message in response['messages']:
#     message.pretty_print()

for chunk in agent.stream(
        {'messages':[HumanMessage(content="广州接下来5天天气如何?")]},
    stream_mode="updates"
):
    for step,data in chunk.items():
        print(f'step:{step}')
        print(f"content:{data['messages'][-1].content_blocks}")

# res = tool1.invoke("广州今天天气怎么样？")
# print(res)