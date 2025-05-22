import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


# 数据预处理
class CommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment = str(self.data.iloc[idx]['评价内容(content)'])
        label = int(self.data.iloc[idx]['评分（总分5分）(score)']) - 1  # 转换为0-4
        return comment, label


# 模型定义
class BertTextClassification(nn.Module):
    def __init__(self, model_name, freeze_bert=False):
        super().__init__()
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config

        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 5)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(pooled_output)


# 训练函数
def train_model(freeze_bert=False, experiment_name='default'):
    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5 if freeze_bert else 5e-5
    LOG_DIR = f'logs/{experiment_name}'

    # 修改后的数据加载部分
    df = pd.read_excel('/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx')

    # 使用更安全的变量名
    filter_condition = df['评价内容(content)'] != "此用户未填写评价内容"
    df = df[filter_condition]

    # 分割训练集和验证集
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(labels)
        }

    train_loader = DataLoader(CommentDataset(train_df, tokenizer),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(CommentDataset(val_df, tokenizer),
                            batch_size=BATCH_SIZE,
                            collate_fn=collate_fn)

    # 模型初始化
    model = BertTextClassification('uer/roberta-base-finetuned-dianping-chinese', freeze_bert)
    model.to(device)

    # 优化器和损失函数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard记录
    writer = SummaryWriter(LOG_DIR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]'):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Val]'):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 记录指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        writer.add_scalars('Loss', {
            'train': avg_train_loss,
            'val': avg_val_loss
        }, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'/kaggle/working/best_model_{experiment_name}.pth')

        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f}')

    writer.close()
    return model


# 对比实验
print("Training with frozen BERT...")
train_model(freeze_bert=True, experiment_name='frozen_bert')

print("\nTraining with unfrozen BERT...")
train_model(freeze_bert=False, experiment_name='unfrozen_bert')


# 预测示例
def predict(text, model_path='best_model_unfrozen_bert.pth'):
    tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
    model = BertTextClassification('uer/roberta-base-finetuned-dianping-chinese')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    inputs = tokenizer(text,
                       padding=True,
                       truncation=True,
                       max_length=512,
                       return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(prob, dim=1).item() + 1  # 转换回1-5分

    return pred, prob.cpu().numpy()[0]


# 使用示例
sample_text = "手机质量很好，运行速度快，拍照效果出色"
pred_score, probabilities = predict(sample_text)
print(f"预测评分: {pred_score}星")
print("各分数概率:", probabilities)