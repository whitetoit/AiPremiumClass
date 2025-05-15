import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import math
import os


# --------------- 模型定义 ---------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        token_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :token_len])


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size,
                 dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_forward,
            dropout=dropout,
            batch_first=True
        )
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        self.predict = nn.Linear(d_model, dec_voc_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        outs = self.transformer(
            src=enc_emb, tgt=dec_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=enc_pad_mask,
            tgt_key_padding_mask=dec_pad_mask
        )
        return self.predict(outs)

    def encode(self, enc_inp):
        return self.transformer.encoder(self.pos_encoding(self.enc_emb(enc_inp)))


# --------------- 数据预处理 ---------------
def build_vocab(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as f:
        in_lines = [line.strip().split() for line in f]
    with open(out_path, 'r', encoding='utf-8') as f:
        out_lines = [line.strip().split() for line in f]

    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    all_chars = list(set(ch for line in in_lines + out_lines for ch in line if ch not in special_tokens))
    all_chars = sorted(all_chars)
    vocab = special_tokens + all_chars
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx, vocab


class CoupletDataset(Dataset):
    def __init__(self, enc_data, dec_data, word2idx):
        self.enc_data = [[word2idx.get(ch, word2idx['<unk>']) for ch in line] for line in enc_data]
        self.dec_data = [[word2idx.get(ch, word2idx['<unk>']) for ch in line] for line in dec_data]
        self.word2idx = word2idx

    def __len__(self):
        return len(self.enc_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.enc_data[idx]),
            torch.tensor([self.word2idx['<s>']] + self.dec_data[idx] + [self.word2idx['</s>']])
        )


def collate_fn(batch):
    enc_batch = [item[0] for item in batch]
    dec_batch = [item[1] for item in batch]
    enc_padded = pad_sequence(enc_batch, padding_value=0, batch_first=True)
    dec_padded = pad_sequence(dec_batch, padding_value=0, batch_first=True)
    return enc_padded, dec_padded[:, :-1], dec_padded[:, 1:]


epochs = 100
LR = 0.0001
batch_size = 32
d_model = 256
nhead = 8
dim_forward = 1024
num_enc_layers = 3
num_dec_layers = 3
dropout = 0.1

# --------------- 训练配置 ---------------
if __name__ == '__main__':
    # 加载数据
    in_path, out_path = 'in.txt', 'out.txt'
    word2idx, vocab = build_vocab(in_path, out_path)
    with open(in_path, 'r', encoding='utf-8') as f:
        enc_data = [line.strip().split() for line in f]
    with open(out_path, 'r', encoding='utf-8') as f:
        dec_data = [line.strip().split() for line in f]

    # 数据集
    dataset = CoupletDataset(enc_data, dec_data, word2idx)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    # 模型参数
    config = {
        'd_model': d_model,
        'nhead': nhead,
        'num_enc_layers': num_enc_layers,
        'num_dec_layers': num_dec_layers,
        'dim_forward': dim_forward,
        'dropout': dropout,
        'enc_voc_size': len(vocab),
        'dec_voc_size': len(vocab)
    }
    model = Seq2SeqTransformer(**config)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 训练循环
    os.makedirs('couplet_model', exist_ok=True)
    for epoch in range(100):
        model.train()
        total_loss = 0
        for enc, dec_in, dec_out in dataloader:
            enc, dec_in, dec_out = enc.to(device), dec_in.to(device), dec_out.to(device)
            optimizer.zero_grad()

            # 掩码生成
            seq_len = dec_in.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
            enc_pad_mask = (enc == 0)
            dec_pad_mask = (dec_in == 0)

            # 前向传播
            outputs = model(enc, dec_in, tgt_mask, enc_pad_mask, dec_pad_mask)
            loss = criterion(outputs.reshape(-1, len(vocab)), dec_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 保存模型
        if epoch == epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'config': config
            }, 'couplet_final_model.pth')
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}')
