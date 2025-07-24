"""
1. 使用nano-gpt5.0训练文本语料，提升内容生成可读性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 配置参数优化
block_size = 256  # 增加上下文长度
batch_size = 64  # 增加批大小
max_iter = 10000  # 增加训练轮次
learn_rate = 3e-4  # 调整学习率
n_embd = 256  # 增加嵌入维度
n_head = 8  # 增加注意力头数
n_layer = 8  # 增加Transformer层数
dropout = 0.2  # 适度增加dropout
temperature = 0.8  # 生成温度参数
top_k = 40  # top-k采样参数
weight_decay = 0.01  # 权重衰减

# 其他配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
eval_iters = 200


# 数据加载函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# 模型组件优化
class FeedForward(nn.Module):
    """添加残差连接和层归一化"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """添加注意力掩码和dropout"""

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        # 创建因果掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力实现"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """添加残差连接前的dropout"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力子层
        attn_out = self.sa(self.ln1(x))
        x = x + self.dropout(attn_out)
        # 前馈子层
        ffwd_out = self.ffwd(self.ln2(x))
        x = x + self.dropout(ffwd_out)
        return x


class BingramLanguageModel(nn.Module):
    """添加最终层dropout和权重绑定"""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        # 权重绑定: 输入和输出共享相同的嵌入矩阵
        self.head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 生成位置索引
        pos = torch.arange(0, T, device=device).unsqueeze(0)

        # 嵌入层
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Transformer块
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # 计算损失
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 截取最后block_size个token
            idx_cond = idx[:, -block_size:]
            # 前向计算
            logits, _ = self(idx_cond)
            # 获取最后一个时间步的logits
            logits = logits[:, -1, :]

            # 应用温度控制
            logits = logits / temperature

            # top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 转换为概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            # 添加到序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == '__main__':
    # 数据预处理（保持不变）
    with open('input.txt') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    # 初始化模型
    model = BingramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    model.to(device)

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learn_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=500,
        verbose=True
    )


    # 损失估计函数（保持不变）
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    # 训练循环
    best_val_loss = float('inf')
    for iter in range(max_iter):
        # 定期评估
        if iter % eval_interval == 0:
            losses = estimate_loss()
            val_loss = losses['val']
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}")

            # 学习率调度
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')

        # 训练步骤
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 文本生成
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=1000,
        temperature=temperature,
        top_k=top_k
    )
    print("\nGenerated text:\n", decode(generated[0].tolist()))
