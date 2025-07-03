"""
1. 根据课堂RAG示例，完成外部文档导入并进行RAG检索的过程。
外部PDF文档：https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
# 使用 langchain_community.document_loaders.PDFMinerLoader 加载 PDF 文件。
docs = PDFMinerLoader(path).load()
"""
import os
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from langchain_chroma import Chroma

load_dotenv(find_dotenv())
api_key = os.environ['ZHIPU_API_KEY']
base_url = os.environ['BASE_URL']

if __name__ == '__main__':
    llm = ChatOpenAI(
        model="glm-z1-airx",
        base_url=base_url,
        api_key=api_key,
        temperature=0.7
    )

    # 文档存储和向量化
    embeddings = ZhipuAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model="embedding-3"
    )
    if not os.path.exists("faiss_store"):
        # 加载文件
        loader_docs = PDFMinerLoader(
            "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
        ).load()
        # 文档拆分
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ""],
            chunk_size=1000,
            chunk_overlap=50)
        splits = text_splitter.split_documents(loader_docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        # 保存向量数据库
        vectorstore.save_local("faiss_store")
        print("保存向量数据库成功！")
    else:
        vectorstore = FAISS.load_local(
            "faiss_store",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("加载向量数据库成功！")

    # 检索
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # 提示词
    client = Client(api_key=os.environ["TVLY_API_KEY"])
    prompt = client.pull_prompt("rlm/rag-prompt",include_model=True)
    # prompt = hub.pull("rlm/rag-prompt")
    # 带有占位符的prompt template

    def format_docs(loader_docs):
        return "\n\n".join([doc.page_content for doc in loader_docs])

    # 链
    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    user_input = "这篇文章讲的是什么"
    # 调用
    rag_retriever = chain.invoke(user_input)
    print(len(rag_retriever))
    print(rag_retriever)
