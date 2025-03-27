import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm as tqdm

def deal_data(filename):
    book_comments = {}; # 多本书的所有评论集
    with open(filename,"r") as f:
        dic_comments = csv.DictReader(f,delimiter="\t");
        for item in dic_comments:
            book_name = item["book"]
            book_comment = item["body"]
            cut_comment = jieba.lcut(book_comment)
            if book_name == "": continue;

            book_comments[book_name] = book_comments.get(book_name,[])
            book_comments[book_name].extend(cut_comment)
        return book_comments;

if __name__ == "__main__":
    #加载停用词列表
    stop_words = [line.strip() for line in open("stopwords.txt","r", encoding="utf-8")]
    #加载图书评论信息
    book_comments = deal_data("douban_comment_fixed.txt")
    print(len(book_comments))
    # 定义书名收集器
    book_names = []
    # 定义评论文本收集器 
    book_comms = []
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    #构建tf-idf特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
    #计算图书之间的余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)
    #输入要推荐的图书名称
    book_list = list(book_comments. keys())
    print(book_list)
    book_name = input("请输入图书名称:")
    book_idx = book_names.index(book_name)#获取图书索引
    #获取与输入图书最相似的图书
    recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]#输出推荐的图书
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》\t 相似度:{similarity_matrix[book_idx][idx]:.4f}")
        print()
