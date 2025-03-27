import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 训练模型
print("正在训练Word2Vec模型...")
model = fasttext.train_unsupervised('hlm.txt', model='skipgram')
model.save_model('word2vec_model.bin')

# 定义辅助函数
def get_word_vector(word):
    return model.get_word_vector(word)

def calculate_similarity(word1, word2):
    vec1 = get_word_vector(word1).reshape(1, -1)
    vec2 = get_word_vector(word2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def find_most_similar(target_word, topn=3):
    nearest_neighbors = model.get_nearest_neighbors(target_word, k=topn)
    print(f"\n与'{target_word}'最相似的{topn}个词:")
    for similarity, word in nearest_neighbors:
        print(f"{word}: {similarity:.4f}")

# 计算词汇相似度
words = ['宝玉', '黛玉', '宝钗', '王熙凤', '探春','庚辰','李纨','史湘云','贾元春']

for word1 in words:
    similarities = [calculate_similarity(word1, word2) for word2 in words]
    print(f"{word1.ljust(10)}\t" + "\t".join(f"{sim:.3f}" for sim in similarities))

# 查找相似词
find_most_similar('迎春')
find_most_similar('妙玉')
