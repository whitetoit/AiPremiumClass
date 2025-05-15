import torch
from homework_train import Seq2SeqTransformer


class CoupletGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.vocab = checkpoint['vocab']
        self.word2idx = checkpoint['word2idx']
        self.model = Seq2SeqTransformer(**checkpoint['config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate(self, input_text, max_len=50):
        # 预处理输入
        input_tokens = input_text.strip().split()
        if not input_tokens:
            return "输入不能为空"
        indices = [self.word2idx.get(tok, self.word2idx['<unk>']) for tok in input_tokens]
        enc_input = torch.tensor([indices], dtype=torch.long, device=self.device)

        # 编码
        with torch.no_grad():
            memory = self.model.encode(enc_input)

        # 解码生成
        dec_input = torch.tensor([[self.word2idx['<s>']]], device=self.device)
        generated = []
        for _ in range(max_len):
            seq_len = dec_input.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
            with torch.no_grad():
                output = self.model(enc_input, dec_input, tgt_mask, None, None)
                next_token = output.argmax(dim=-1)[:, -1]
            generated.append(next_token.item())
            if next_token == self.word2idx['</s>']:
                break
            dec_input = torch.cat([dec_input, next_token.unsqueeze(1)], dim=1)

        # 后处理
        result = [self.vocab[idx] for idx in generated if self.vocab[idx] not in ['<s>', '</s>', '<pad>']]
        return ''.join(result)


if __name__ == '__main__':
    # 初始化生成器（选择最新模型）
    # import glob

    # model_files = glob.glob('couplet_model/epoch_*.pth')
    # latest_model = sorted(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    latest_model = 'couplet_final_model.pth'
    generator = CoupletGenerator(latest_model)

    # 交互测试
    print("输入上联（空格分隔），输入 '退出' 结束")
    while True:
        text = input("\n上联：").strip()
        if text.lower() in ['退出', 'exit']:
            break
        output = generator.generate(text)
        print(f"下联：{output}")
