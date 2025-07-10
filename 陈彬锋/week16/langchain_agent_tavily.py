"""
1.使用langchain_agent实现pdf和TAVILY工具组合动态调用的实现。
"""
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.tools.retriever import create_retriever_tool

# 设置API密钥
load_dotenv(find_dotenv())
api_key = os.environ['ZHIPU_API_KEY']
base_url = os.environ['ZHIPU_BASE_URL']
tavily_key = os.environ['TAVILY_API_KEY']


# 1. 加载并处理PDF文档
def load_and_process_pdf(pdf_path):
    """加载PDF文档并创建向量数据库"""
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # 创建向量存储
    embeddings = ZhipuAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model="embedding-3"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


# 2. 创建工具
def create_tools(pdf_path):
    """创建PDF检索工具和Tavily搜索工具"""
    # PDF检索工具
    vectorstore = load_and_process_pdf(pdf_path)

    # Tavily搜索工具
    search = TavilySearchResults(max_results=3, api_key=tavily_key)

    # 检索
    pdf_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(retriever=pdf_retriever,
                                           name="pdf retriever",
                                           description="书籍《Welcome to the Era of Experience》内容简介：当前技术结合适当算法已具备推动语言学习或科学突破的能力，并认为AI社区的努力将加速创新，推动AI向超人类智能发展")
    tools = [search, retriever_tool]
    return tools


# 3. 创建代理
def create_agent(pdf_path):
    """创建带有PDF和Tavily工具的代理"""
    # 创建LLM模型
    model = ChatOpenAI(
        model="glm-z1-airx",
        base_url=base_url,
        api_key=api_key,
        temperature=0.7
    )

    # 创建工具
    tools = create_tools(pdf_path)

    # 提示词模板
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 创建代理
    agent = create_tool_calling_agent(model, tools, prompt)

    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


# 4. 主函数
def main():
    # 替换为你的PDF文件路径
    pdf_path = "The Era of Experience Paper.pdf"

    # 创建代理
    agent = create_agent(pdf_path)

    while True:
        query = input("\n请输入查询：")
        response = agent.invoke({"input": query})
        print(f"\n回答：{response['output']}")
        if query == "exit" or query == "quit" or query == "退出":
            break

if __name__ == "__main__":
    main()
