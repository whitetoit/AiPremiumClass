"""
借助langchain实现图书管理系统开发扩展，通过图书简介为借阅读者提供咨询
"""
from dotenv import load_dotenv, find_dotenv
import os
from typing import List, Dict

# LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage

# 加载环境变量
load_dotenv(find_dotenv())


class SmartLibraryAssistant:
    def __init__(self):
        # 增强图书数据库
        self.books_database = [
            {"id": 1, "title": "三体", "author": "刘慈欣", "genre": "科幻", "available": True,
             "description": "讲述了地球人类文明和三体文明的信息交流、生死搏杀及两个文明在宇宙中的兴衰历程。"},
            {"id": 2, "title": "平凡的世界", "author": "路遥", "genre": "文学", "available": False,
             "description": "通过复杂的矛盾纠葛，刻画了社会各阶层普通人的形象，人生的自尊、自强与自信，奋斗与拼搏，痛苦与欢乐。"},
            {"id": 3, "title": "Python编程：从入门到实践", "author": "Eric Matthes", "genre": "技术", "available": True,
             "description": "针对所有层次的Python读者而作的Python入门书，从基本概念到完整项目开发，帮助读者快速掌握编程技能。"},
            {"id": 4, "title": "人类简史", "author": "尤瓦尔·赫拉利", "genre": "历史", "available": True,
             "description": "讲述了人类从石器时代至21世纪的演化与发展史，并将人类历史分为四个阶段：认知革命、农业革命、人类的融合统一与科学革命。"},
            {"id": 5, "title": "活着", "author": "余华", "genre": "文学", "available": True,
             "description": "讲述了在大时代背景下，徐福贵的人生和家庭不断经受着苦难，到了最后所有亲人都先后离他而去，仅剩下年老的他和一头老牛相依为命。"},
            {"id": 6, "title": "机器学习实战", "author": "Peter Harrington", "genre": "技术", "available": False,
             "description": "通过实例讲解机器学习核心算法，涵盖分类、聚类、预测分析等多个领域，适合有一定编程基础的读者。"},
            {"id": 7, "title": "时间简史", "author": "史蒂芬·霍金", "genre": "科普", "available": True,
             "description": "探索了宇宙的起源、发展、结构和最终命运，介绍了黑洞、时间箭头等概念，是理解现代物理学的经典之作。"}
        ]

        # 用户借阅记录
        self.user_records = {}

        # 初始化LangChain对话系统
        self.init_langchain_system()

    def init_langchain_system(self):
        """初始化LangChain对话系统"""
        # 创建模型
        self.model = ChatOpenAI(
            model="glm-z1-airx",
            base_url=os.environ['BASE_URL'],
            api_key=os.environ['API_KEY'],
            temperature=0.7
        )

        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是一位专业的图书管理员AI助手，负责管理图书馆的日常运营。请根据当前图书状态和用户需求提供专业服务。\n"
                "请使用{lang}回答所有问题。\n\n"
                "当前图书状态:\n"
                "-------------------------\n"
                "{books_status}\n"
                "-------------------------\n\n"
                "用户借阅历史:\n"
                "-------------------------\n"
                "{user_history}\n"
                "-------------------------\n\n"
                "职责：\n"
                "1. 处理图书借阅和归还\n"
                "2. 根据用户喜好推荐图书\n"
                "3. 解答图书相关咨询\n"
                "4. 提供图书内容咨询服务\n\n"
                "约束条件:\n"
                "- 回复长度控制在100字以内\n"
                "- 使用中文口语化表达\n"
                "- 对于无法识别的请求，礼貌询问更多信息"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 创建解析器
        parser = StrOutputParser()

        # 创建处理链
        self.chain = (
                self.prompt
                | self.model
                | parser
        )

        # 定义存储消息的字典
        self.store = {}

        # 定义函数：根据sessionId获取聊天历史
        def get_chat_history(session_id):
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]

        # 添加历史管理
        self.with_msg_hist = RunnableWithMessageHistory(
            self.chain,
            get_session_history=get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def get_books_status(self):
        """获取图书状态信息（字符串格式）"""
        available_books = [f"{book['title']} (ID:{book['id']})" for book in self.books_database if book['available']]
        borrowed_books = [f"{book['title']} (ID:{book['id']})" for book in self.books_database if not book['available']]

        return (
            f"可借阅图书: {', '.join(available_books) if available_books else '无'}\n"
            f"已借出图书: {', '.join(borrowed_books) if borrowed_books else '无'}"
        )

    def get_user_history(self, user_id):
        """获取用户借阅历史（字符串格式）"""
        if user_id in self.user_records and self.user_records[user_id]:
            return ", ".join([f"{book['title']} (ID:{book['id']})" for book in self.user_records[user_id]])
        return "暂无借阅记录"

    def borrow_book(self, user_id, book_id):
        """借阅图书"""
        book = next((b for b in self.books_database if b['id'] == book_id), None)
        if book and book['available']:
            book['available'] = False
            if user_id not in self.user_records:
                self.user_records[user_id] = []
            self.user_records[user_id].append(book)
            return True, book['title']
        return False, None

    def return_book(self, user_id, book_id):
        """归还图书"""
        book = next((b for b in self.books_database if b['id'] == book_id), None)
        if book and not book['available']:
            book['available'] = True
            if user_id in self.user_records:
                self.user_records[user_id] = [b for b in self.user_records[user_id] if b['id'] != book_id]
            return True, book['title']
        return False, None

    def find_books_by_query(self, query: str) -> List[Dict]:
        """基于关键词的简单图书检索"""
        # 小写查询以进行不区分大小写的匹配
        query = query.lower()

        # 查找匹配的图书
        results = []
        for book in self.books_database:
            # 检查书名、作者、类型或描述中是否包含查询词
            if (query in book['title'].lower() or
                    query in book['author'].lower() or
                    query in book['genre'].lower() or
                    query in book['description'].lower()):
                results.append(book)

        # 如果没有完全匹配，尝试部分匹配
        if not results:
            for book in self.books_database:
                if any(term in book['description'].lower() for term in query.split()):
                    results.append(book)

        return results[:3]  # 最多返回3本书

    def process_request(self, user_id: str, user_input: str, session_id: str = "library_session") -> str:
        """处理用户请求"""
        # 获取图书状态和用户历史
        books_status = self.get_books_status()
        user_history = self.get_user_history(user_id)

        # 调用LangChain对话系统
        response = self.with_msg_hist.invoke(
            {
                "input": user_input,
                "lang": "汉语",
                "books_status": books_status,
                "user_history": user_history
            },
            config={'configurable': {'session_id': session_id}}
        )

        # 检查是否需要执行借阅/归还操作
        if "借阅" in user_input or "借书" in user_input:
            book_id = self.extract_book_id(user_input)
            if book_id:
                success, book_title = self.borrow_book(user_id, book_id)
                if success:
                    return f"《{book_title}》借阅成功！请于30天内归还。"
                return "借阅失败，该书可能已被借出或不存在"

        elif "归还" in user_input or "还书" in user_input:
            book_id = self.extract_book_id(user_input)
            if book_id:
                success, book_title = self.return_book(user_id, book_id)
                if success:
                    return f"《{book_title}》归还成功！感谢您的使用。"
                return "归还失败，请检查图书ID是否正确"

        # 处理查询请求
        elif "查询" in user_input or "找书" in user_input or "推荐" in user_input:
            # 查找相关图书
            relevant_books = self.find_books_by_query(user_input)

            if relevant_books:
                books_info = "\n".join([
                    f"《{book['title']}》(ID:{book['id']}) - {book['author']} "
                    f"[{'可借阅' if book['available'] else '已借出'}]"
                    for book in relevant_books
                ])
                return f"找到以下相关图书:\n{books_info}\n\n{response}"

        return response

    def extract_book_id(self, text: str) -> int:
        """从文本中提取图书ID"""
        # 简单实现：查找类似"ID:123"的模式
        import re
        match = re.search(r'ID:?(\d+)', text)
        if match:
            return int(match.group(1))

        # 尝试从书名中提取ID
        for book in self.books_database:
            if book['title'] in text:
                return book['id']

        return None


# 主程序
if __name__ == "__main__":
    # 创建图书助手
    assistant = SmartLibraryAssistant()

    # 用户ID和会话ID
    user_id = "user_1001"
    session_id = "library_session"

    print("=" * 60)
    print("智能图书管理系统已启动")
    print("支持以下操作:")
    print("- 查询图书信息 (例: 我想了解科幻类书籍)")
    print("- 借阅图书 (例: 我想借阅ID:1的图书)")
    print("- 归还图书 (例: 我要归还ID:4的图书)")
    print("- 图书推荐 (例: 推荐一本关于Python的书籍)")
    print("输入'退出'结束程序")
    print("=" * 60)

    while True:
        try:
            # 用户输入
            user_input = input("\n用户: ")

            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用图书管理系统，再见！")
                break

            # 处理请求
            response = assistant.process_request(user_id, user_input, session_id)
            print("\n系统:", response)

        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue