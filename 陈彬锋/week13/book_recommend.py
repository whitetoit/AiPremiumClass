from zhipuai import ZhipuAI
import json

api_key = "36d7d32e72fc4e2498be47589aca3ca5.RsFlLAtDCiBTzYRZ"
client = ZhipuAI(api_key=api_key)


class SmartLibraryAssistant:
    def __init__(self):

        # 模拟图书数据库
        self.books_database = [
            {"id": 1, "title": "三体", "author": "刘慈欣", "genre": "科幻", "available": True},
            {"id": 2, "title": "平凡的世界", "author": "路遥", "genre": "文学", "available": False},
            {"id": 3, "title": "Python编程：从入门到实践", "author": "Eric Matthes", "genre": "技术", "available": True},
            {"id": 4, "title": "人类简史", "author": "尤瓦尔·赫拉利", "genre": "历史", "available": True},
            {"id": 5, "title": "活着", "author": "余华", "genre": "文学", "available": True},
            {"id": 6, "title": "机器学习实战", "author": "Peter Harrington", "genre": "技术", "available": False},
            {"id": 7, "title": "时间简史", "author": "史蒂芬·霍金", "genre": "科普", "available": True}
        ]

        # 用户借阅记录
        self.user_records = {}

    def get_books_status(self):
        """获取图书状态信息"""
        return json.dumps({
            "available_books": [book for book in self.books_database if book['available']],
            "borrowed_books": [book for book in self.books_database if not book['available']]
        }, ensure_ascii=False)

    def get_user_history(self, user_id):
        """获取用户借阅历史"""
        return self.user_records.get(user_id, [])

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

    def generate_prompt(self, user_id, user_input):
        """生成智能提示词"""
        books_status = self.get_books_status()
        user_history = self.get_user_history(user_id)

        # 核心提示词设计
        prompt = f"""
        你是一位专业的图书管理员AI助手，负责管理图书馆的日常运营。请根据当前图书状态和用户需求提供专业服务。

        # 角色设定
        身份：智能图书管理员
        职责：
        1. 处理图书借阅和归还
        2. 根据用户喜好推荐图书
        3. 解答图书相关咨询

        # 当前图书状态
        {books_status}

        # 用户借阅历史（用户ID: {user_id}）
        {json.dumps(user_history, ensure_ascii=False) if user_history else "暂无借阅记录"}

        # 任务指令
        1. 分析用户输入，识别意图（借阅/归还/推荐/咨询）
        2. 对于借阅/归还请求：明确书名或图书ID
        3. 对于推荐请求：基于用户历史偏好推荐3本相关书籍
        4. 保持专业友好的服务态度

        # 约束条件
        - 回复长度控制在100字以内
        - 使用中文口语化表达
        - 对于无法识别的请求，礼貌询问更多信息

        # 用户输入
        {user_input}

        # 输出格式要求
        {{ 
            "intent": "借阅/归还/推荐/咨询",
            "book_id": "仅当借阅或归还时需要",
            "response": "给用户的自然语言回复"
        }}
        """
        return prompt

    def process_request(self, user_id, user_input):
        """处理用户请求"""
        # 生成提示词
        prompt = self.generate_prompt(user_id, user_input)

        try:
            # 调用大模型API
            response = client.chat.completions.create(
                model="glm-z1-airx",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            # 解析响应
            result = response.choices[0].message.content
            if response.choices[0].message.content.__contains__("</think>"):
                result = json.loads(response.choices[0].message.content.split("</think>")[1])
                intent = result.get("intent", "")
                book_id = result.get("book_id", "")
                ai_response = result.get("response", "")

                # 执行实际借阅/归还操作
                if intent == "借阅" and book_id:
                    success, book_title = self.borrow_book(user_id, int(book_id))
                    if success:
                        return f"《{book_title}》借阅成功！请于30天内归还。"
                    return "借阅失败，该书可能已被借出或不存在"

                elif intent == "归还" and book_id:
                    success, book_title = self.return_book(user_id, int(book_id))
                    if success:
                        return f"《{book_title}》归还成功！感谢您的使用。"
                    return "归还失败，请检查图书ID是否正确"
            else:
                ai_response = result

            return ai_response

        except Exception as e:
            return f"请求处理出错: {str(e)}"


# 测试示例
if __name__ == "__main__":
    assistant = SmartLibraryAssistant()

    # 模拟用户交互
    user_id = "1001"

    # 场景1: 借阅图书
    print("用户: 我想借阅《三体》")
    print("AI:", assistant.process_request(user_id, "我想借阅《三体》"))
    print()

    # 场景2: 查看借阅历史后推荐
    print("用户: 根据我的历史推荐一些书")
    print("AI:", assistant.process_request(user_id, "根据我的历史推荐一些书"))
    print()

    # 场景3: 归还图书
    print("用户: 我要归还ID为1的书")
    print("AI:", assistant.process_request(user_id, "我要归还ID为1的书"))
    print()

    # 场景4: 咨询图书
    print("用户: 有没有好的技术类书籍推荐？")
    print("AI:", assistant.process_request(user_id, "有没有好的技术类书籍推荐？"))
