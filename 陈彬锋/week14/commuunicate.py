"""
通过langchain实现特定主题聊天系统，支持多轮对话。
"""
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv(find_dotenv())
    # 创建模型
    model = ChatOpenAI(
        model="glm-z1-airx",
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['API_KEY'],
        temperature=0.7
    )

    # 创建prompt(提示词)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个精通python语言ai应用开发的专家。请使用{lang}回答所有问题。"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 创建解析器parser
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 定义储存消息的字典
    store = {}


    # 定义函数：根据sessionId获取聊天历史
    def get_chat_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]  # 检索 （langchain负责维护内部聊天历史信息）


    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=get_chat_history,
        input_messages_key="messages"  # input_messages_key 指明用户消息使用key
    )

    # 初始化用户session_id
    session_id = "session_1"

    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "lang": "汉语"
            },
            config={'configurable': {'session_id': session_id}})
        print(f"AI RESPONSE: {response}")
