# %%writefile train_model_fine_turning_tips.py
import os
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, pipeline
import torch
import evaluate
from datasets import load_dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# 清理分布式环境
def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    # 数据集加载
    ds = load_dataset('nlhappy/CLUE-NER')

    # 实体类别定义
    entites = ['O'] + list({
        'movie', 'name', 'game', 'address', 'position',
        'company', 'scene', 'book', 'organization', 'government'
    })

    # 标签生成 (BIO格式)
    tags = ['O']
    for entity in entites[1:]:
        tags.append(f'B-{entity.upper()}')
        tags.append(f'I-{entity.upper()}')

    # 标签映射
    tag2id = {tag: i for i, tag in enumerate(tags)}
    id2tag = {i: tag for i, tag in enumerate(tags)}

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    def entity_tags_proc(item):
        # 初始化全O标签
        text = item['text']
        tags = ['O'] * len(text)

        # 遍历实体列表，添加标签
        for ent in item['ents']:
            # 添加边界检查
            start = max(0, min(ent['indices'][0], len(text) - 1))
            end = max(0, min(ent['indices'][-1] + 1, len(text)))
            entity_type = ent['label'].upper()

            # BIO标注
            tags[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                tags[i] = f'I-{entity_type}'

        # 转换为ID
        tag_ids = [tag2id[tag] for tag in tags]
        return {'tags': tag_ids, 'text': text}

    # 处理数据集
    ds = ds.map(entity_tags_proc)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=128,
            is_split_into_words=False,
            return_offsets_mapping=True
        )

        # 获取偏移映射
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        # 对齐标签
        labels = []
        for i, offsets in enumerate(offset_mapping):
            # 获取原始标签
            original_tags = examples["tags"][i]
            # 创建新标签列表，初始化为-100（忽略）
            new_tags = [-100] * len(offsets)

            # 对齐标签
            for j, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:  # 特殊token
                    continue
                if start == end:  # 填充token
                    continue
                # 修复标签对齐逻辑
                if start < len(original_tags):
                    new_tags[j] = original_tags[start]

            labels.append(new_tags)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 应用处理函数
    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)

    # 分布式训练设置
    local_rank = rank
    torch.cuda.set_device(local_rank)

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=len(tags),
        id2label=id2tag,
        label2id=tag2id
    )
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 训练参数
    args = TrainingArguments(
        output_dir="ner_train",  # 模型训练工作目录(tensorboard,临时模型存盘文件,日志)
        num_train_epochs=3,  # 训练epoch
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        per_device_train_batch_size=32,  # 训练批次
        per_device_eval_batch_size=32,
        report_to='tensorboard',  # 训练输出记录
        eval_strategy="epoch",
        local_rank=local_rank,  # 当前进程,RANK
        fp16=True,  # 使用混合精度
        lr_scheduler_type='linear',  # 动态学习率
        warmup_steps=100,  # 预热步数
        ddp_find_unused_parameters=False  # 优化DDP性能
    )

    # 评估指标
    seqeval_metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # 移除忽略的索引（-100）
        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval_metric.compute(
            predictions=true_predictions,
            references=true_labels
        )

        # 返回主要指标
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # 数据整理器
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 训练模型
    trainer.train()

    # 保存模型（只在主进程保存）
    if rank == 0:
        trainer.save_model("./ner_model")
        tokenizer.save_pretrained("./ner_model")
        print("模型保存完成!")

    cleanup()


def run_inference():
    """单独运行推理的函数"""
    # 训练完成后加载模型进行推理
    torch.cuda.empty_cache()
    device = 0 if torch.cuda.is_available() else -1

    # 创建NER pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model="./ner_model",
        tokenizer='google-bert/bert-base-chinese',  # 直接使用基础分词器
        aggregation_strategy="simple",
        device=device
    )

    # 测试文本
    test_text = "张艺谋导演了电影《英雄》，该片由李连杰主演。"

    # 进行预测
    results = ner_pipeline(test_text)

    # 打印结果
    print("\n实体识别结果:")
    for entity in results:
        print(f"{entity['word']} -> {entity['entity_group']} (置信度: {entity['score']:.2f})")


def main():
    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 个GPU进行训练")

    # 运行分布式训练
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

    # 训练完成后运行推理
    run_inference()


if __name__ == '__main__':
    main()

# !python train_model_fine_turning_tips.py