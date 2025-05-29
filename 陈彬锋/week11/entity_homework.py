from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
import torch

ckpt = 'google-bert/bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForTokenClassification.from_pretrained(ckpt, num_labels=7)

ds = load_dataset("doushabao4766/msra_ner_k_V3")
ds
# ds["train"]
# ds["validation"]

#实体映射数据集词典准备
entites = ["O"] + list(
    {"movie", "name", "game", "address", "position", "company", "scene", "book", "organization", "government"})
tags = ["O"]
for entity in entites[1:]:
    tags.append("B-" + entity.upper())
    tags.append("I-" + entity.upper())

print(tags)

entity_index = {entity: i for i, entity in enumerate(entites)}

entity_index

def entity_tags_proc(item):
    return {'ent_tag': item["ner_tags"]}
#使用自定义回调函数处理数据集记录
ds1 = ds.map(entity_tags_proc)


for i,row in enumerate(ds1["train"]):
    if 2== i:
        print("".join(row["tokens"]))
        print(row["ent_tag"])
        break


# token_index = tokenizer.encode("2000年2月add",add_special_tokens=True)
token_index = tokenizer.encode("双方确定了今后发展中美关系的指导方针",add_special_tokens=False)
print(token_index)
tokens = tokenizer.decode(token_index)
print(tokens)


# 初始化文本转换成模型需要的token_index,生成和token_index对齐的label
def data_input_proc(item):
    # 输入文本转换成模型输入token索引
    input_data =  tokenizer("".join(item["tokens"]),truncation=True,add_special_tokens=False,max_length=512)
    # 初始化保存所有修正后的label索引列表
    adjust_labels = []
    for k in range(len(input_data["input_ids"])):
        # 记录每条token对应的word_ids
        word_ids = input_data.word_ids(k)
        # 每个批次的ner_tag长度和token长度对齐
        tags = item["ent_tag"][k]
        # 当前批次修正的label索引列表
        adjust_label_ids =[]
        i, pre_wid = -1,-1
        j = []
        for wid in word_ids:
            if(wid != pre_wid):
                # token 对应索引位置+1
                i += 1
                pre_wid = wid
            j.append(i)
            adjust_label_ids.append(tags[i])
            # print(adjust_label_ids)
        # print(j)
        adjust_labels.append(adjust_label_ids)
    input_data['labels'] = adjust_labels
    return input_data

ds2 = ds1.map(data_input_proc,batched=True) # batched 每次传入自定义方法样本数量多个

# 记录转成torch
ds2.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
# 记录生成新的torch
# new_dataset2 = with_format("torch",columns=["input_ids","token_type_ids","attention_mask","labels"])

for item in ds2["train"]:
    print(item)
    break

args = TrainingArguments(
    output_dir="ner_train",  # 模型训练工作目录（tensorboard,临时模型存盘文件日志
    num_train_epochs=3,  # 训练轮次
    save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
    per_device_train_batch_size=32,  # 训练批次
    report_to="tensorboard",  # 结果输出到tensorboard
    eval_strategy="epoch"  # 验证策略, no,steps,epoch
)


def compute_metric(result):
    seqeval = evaluate.load("seqeval")
    logits, labels = result
    predictions = np.argmax(logits, axis=2)

    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [tags[l] for l in label if l != -100]
        for label in labels
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return results


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

trainer = Trainer(
    model,
    args,
    train_dataset=ds2["train"],
    eval_dataset=ds2["test"],
    data_collator=data_collator,
    compute_metrics=compute_metric
)

trainer.train()


def extract_entities(sentence, model, tokenizer, tags):
    # 确保模型在正确的设备上
    device = model.device

    # 1. 预处理输入
    tokens = list(sentence)  # 将句子拆分为字符列表
    encoding = tokenizer(tokens,
                         is_split_into_words=True,  # 指示输入已分词
                         truncation=True,
                         max_length=512,
                         return_tensors="pt",
                         return_offsets_mapping=True)  # 获取字符偏移映射

    # 2. 获取 word_ids（在转换为设备前）
    word_ids = encoding.word_ids(batch_index=0)  # 获取第一个序列的word_ids

    # 3. 准备模型输入
    inputs = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "token_type_ids": encoding["token_type_ids"].to(device) if "token_type_ids" in encoding else None
    }

    # 4. 模型预测
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 5. 获取预测标签 (移回CPU处理)
    predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    # 6. 对齐原始字符
    char_predictions = []
    prev_word_id = None
    for i, word_id in enumerate(word_ids):
        # 跳过特殊token ([CLS], [SEP], padding)
        if word_id is None:
            continue

        # 每个字符只取第一个token的预测结果
        if word_id != prev_word_id:
            char_predictions.append(predictions[i])
            prev_word_id = word_id

    # 7. 提取实体
    entities = []
    current_entity = None

    for idx, tag_idx in enumerate(char_predictions):
        tag = tags[tag_idx]

        # 处理实体开始 (B-XXX)
        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)

            entity_type = tag.split("-")[1]
            current_entity = {
                "entity": entity_type,
                "content": tokens[idx],
                "start": idx,
                "end": idx
            }

        # 处理实体延续 (I-XXX)
        elif tag.startswith("I-") and current_entity is not None:
            entity_type = tag.split("-")[1]
            # 确保I标签与当前实体类型匹配
            if entity_type == current_entity["entity"]:
                current_entity["content"] += tokens[idx]
                current_entity["end"] = idx
            else:
                # 类型不匹配，结束当前实体
                entities.append(current_entity)
                current_entity = None

        # 处理非实体 (O)
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    # 添加最后一个实体
    if current_entity is not None:
        entities.append(current_entity)

    # 8. 格式化输出
    formatted_entities = []
    for entity in entities:
        # 只保留需要的字段
        formatted_entities.append({
            "entity": entity["entity"],
            "content": entity["content"]
        })

    return formatted_entities


# 确保模型在评估模式
model.eval()

# 如果模型在GPU上，确保它也在GPU上
if torch.cuda.is_available():
    model = model.to("cuda")

# 测试句子
sentence = "双方确定了今后发展中美关系的指导方针。"

# 抽取实体
entities = extract_entities(sentence, model, tokenizer, tags)
print(entities)