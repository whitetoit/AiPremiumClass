import jieba
import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt',dim=100,epoch=50,lr=0.05)
print(model.predict("which baking dish is best to bake a banana bread?"))

print(model.predict("Which plastic wrap is okay for oven use?"))

print(model.predict("Why did my meringue deflate and go soft?"))
