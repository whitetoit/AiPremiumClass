import torch
import torch.nn as nn
from torch.nn import functional as F


def get_batch(split):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data
    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    """单头self-attention"""

    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, input_x):
        B, T, C = input_x.shape
        key = self.key(input_x)
        query = self.query(input_x)
        value = self.value(input_x)

        wei = query @ key.transpose(-2, -1) * C ** -0.5
        T = wei.shape[-1]
        tril = torch.tril(torch.ones(T, T, device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)

        out = wei @ value
        return out


class BingramLanguageModel(nn.Module):

    def __init__(self, block_size, vocab_size, n_embd):
        super().__init__()
        self.block_size = block_size  # 保存block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置编码
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 单头attention
        self.sa_head = Head(n_embd)
        # 大模型前向运算
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # 安全的位置索引
        pos = torch.arange(T, device=device)
        # 确保位置索引在有效范围内
        if T > self.block_size:
            pos = pos % self.block_size  # 循环使用位置编码

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (B,T)数组对应着当前的输入内容[1,1]
        for _ in range(max_new_tokens):
            # 截断输入到block_size长度
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':

    # 模型训练数据集
    block_size = 8
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    n_embd = 32
    eval_interval = 500
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('hlm.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 字典、编码器(函数)、解码器(函数)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}  # str_to_index
    itos = {i: ch for i, ch in enumerate(chars)}  # index_to_str

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 文本转换token index
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]

    model = BingramLanguageModel(block_size, vocab_size, n_embd)  # 模型训练
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)


    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step: {iter} | train loss: {losses['train']} | val loss: {losses['val']}")
        xb, yb = get_batch('train')  # 推理计算损失
        logtis, loss = model(xb, yb)  # backward
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

    # 模型生成
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
