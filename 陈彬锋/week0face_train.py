import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 数据加载和预处理
def load_olivetti_data():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data.reshape(-1, 64, 64)  # (400, 64, 64)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return (
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)/255.0,  # 添加通道维度
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)/255.0,
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

# 自定义数据集
class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# RNN模型基类
class FaceRNN(nn.Module):
    def __init__(self, rnn_type, hidden_size=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # 序列处理：将图像视为高度序列（64时间步，每个时间步64特征）
        self.rnn = self._create_rnn(
            input_size=64,  # 每行的像素数作为特征维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1),
            40  # Olivetti数据集的40个类别
        )
        
    def _create_rnn(self, input_size, hidden_size, num_layers, bidirectional):
        if self.rnn_type == "lstm":
            return nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional
            )
        elif self.rnn_type == "gru":
            return nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional
            )
        else:  # 普通RNN
            return nn.RNN(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional
            )
    
    def forward(self, x):
        # 输入形状: (batch, channel, height, width)
        # 转换为序列: (batch, seq_len, input_size)
        x = x.squeeze(1).permute(0, 2, 1)  # (batch, 64, 64)
        
        # RNN处理
        if isinstance(self.rnn, nn.LSTM):
            output, (h_n, c_n) = self.rnn(x)
        else:
            output, h_n = self.rnn(x)
        
        # 获取最后一个时间步的输出
        if self.bidirectional:
            output = torch.cat([output[:, -1, :self.rnn.hidden_size], 
                              output[:, 0, self.rnn.hidden_size:]], dim=1)
        else:
            output = output[:, -1, :]
            
        return self.fc(output)

# 训练函数
def train(model, device, loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss, correct = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    
    # 记录到TensorBoard
    writer.add_scalar("Train Loss", avg_loss, epoch)
    writer.add_scalar("Train Acc", accuracy, epoch)
    return avg_loss, accuracy

# 测试函数
def test(model, device, loader, criterion, epoch, writer):
    model.eval()
    total_loss, correct = 0, 0
    all_images, all_targets, all_preds = [], [], []
    
    with torch.no_grad():
        for data, target in loader:
            all_images.append(data.cpu())
            all_targets.append(target.cpu())
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            all_preds.append(pred.cpu())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    
    # 记录到TensorBoard
    writer.add_scalar("Test Loss", avg_loss, epoch)
    writer.add_scalar("Test Acc", accuracy, epoch)
    return (avg_loss, accuracy, 
            torch.cat(all_images), 
            torch.cat(all_targets), 
            torch.cat(all_preds))

# 可视化函数
def plot_results(images, true_labels, pred_labels, epoch, num_samples=15):
    plt.figure(figsize=(15, 10))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i+1)
        img = images[idx].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        title_color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        plt.title(f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}", 
                 color=title_color, fontsize=8)
        plt.axis('off')
    
    plt.suptitle(f"Epoch {epoch} Prediction Results", fontsize=14)
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "hidden_size": 128,
        "num_layers": 2,
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.001
    }
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_olivetti_data()
    train_dataset = FaceDataset(X_train, y_train)
    test_dataset = FaceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 对比模型配置
    models = [
        ("RNN", "rnn", False),
        ("LSTM", "lstm", False),
        ("GRU", "gru", False)
    ]
    
    # 训练所有模型
    for model_name, rnn_type, bidirectional in models:
        print(f"\n{'='*30}\nTraining {model_name}...\n{'='*30}")
        writer = SummaryWriter(f"runs/{model_name}")
        
        # 初始化模型
        model = FaceRNN(
            rnn_type=rnn_type,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            bidirectional=bidirectional
        ).to(device)
        
        optimizer = Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        for epoch in range(1, config["epochs"]+1):
            train_loss, train_acc = train(
                model, device, train_loader, optimizer, criterion, epoch, writer)
            
            test_loss, test_acc, test_images, test_targets, test_preds = test(
                model, device, test_loader, criterion, epoch, writer)
            
            print(f"Epoch {epoch:02d}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
            # 每5个epoch可视化结果
            if epoch % 5 == 0 or epoch == config["epochs"]:
                plot_results(
                    test_images, 
                    test_targets.numpy(), 
                    test_preds.numpy(),
                    epoch
                )
        
        writer.close()

if __name__ == "__main__":
    main()
