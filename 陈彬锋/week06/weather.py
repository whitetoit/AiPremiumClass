import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

#数据预处理
class WeatherDataProcessor:
    def __init__(self, file_path, sequence_length=14, predict_steps=5):
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        
        # 加载和处理数据
        raw_data = pd.read_csv(file_path, parse_dates=['Date'])
        raw_data = raw_data.sort_values('Date')
        raw_data['MaxTemp'] = raw_data['MaxTemp'].interpolate()
        
        # 特征工程
        self.data = raw_data[['Date', 'MaxTemp']].set_index('Date')
        
        # 数据归一化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaled_data = self.scaler.fit_transform(self.data.values)
    
    def create_sequences(self):
        #创建时间序列样本
        xs, ys = [], []
        for i in range(len(self.scaled_data)-self.sequence_length-self.predict_steps):
            x = self.scaled_data[i:(i+self.sequence_length)]
            y = self.scaled_data[i+self.sequence_length : i+self.sequence_length+self.predict_steps, 0]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

# 数据集类
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#RNN模型
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_steps=5):
        super().__init__()
        self.output_steps = output_steps
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_steps)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 训练函数
def train_model(model, train_loader, val_loader, scaler, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    writer = SummaryWriter(f"runs/{config['model_name']}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_loss += criterion(model(X_val), y_val).item()
        
        # 记录指标
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_{config['model_name']}.pth")
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    writer.close()
    return model

def plot_forecast_series(series, actuals, predictions, num_samples=15):
    """时间序列预测可视化（修复版）"""
    r, c = 3, 5
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(20, 12))
    
    for i in range(min(num_samples, r*c)):
        row = i // c
        col = i % c
        ax = axes[row][col]
        
        # 获取历史数据长度
        hist_length = len(series[i])
        
        # 绘制历史数据（不包含最后一个点的连线到预测）
        ax.plot(range(hist_length), series[i], 'b.-', label='History')
        
        # 绘制实际值（从历史数据末尾开始）
        ax.plot(range(hist_length, hist_length + len(actuals[i])), 
                actuals[i], 
                'gx-', markersize=8, label='Actual')
        
        # 绘制预测值（从历史数据末尾开始）
        ax.plot(range(hist_length, hist_length + len(predictions[i])),
                predictions[i],
                'r.--', markersize=8, label='Predicted')
        
        ax.grid(True)
        ax.set_title(f'Sample {i+1}', fontsize=10)
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Temperature Forecast Results', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    # 配置参数
    config = {
        'sequence_length': 14,
        'predict_steps': 5,
        'batch_size': 32,
        'hidden_size': 64,
        'num_layers': 2,
        'lr': 0.001,
        'epochs': 15,
        'model_name': 'WeatherLSTM'
    }
    
    # 数据准备
    processor = WeatherDataProcessor('weather_history.csv', 
                                    config['sequence_length'],
                                    config['predict_steps'])
    X, y = processor.create_sequences()
    
    # 数据集划分
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    # 创建数据加载器
    dataloader = {
        'train': DataLoader(WeatherDataset(X_train, y_train), shuffle=True, batch_size=config['batch_size']),
        'val': DataLoader(WeatherDataset(X_val, y_val), batch_size=config['batch_size']),
        'test': DataLoader(WeatherDataset(X_test, y_test), batch_size=config['batch_size'])
    }
    
    # 初始化模型
    model = WeatherRNN(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_steps=config['predict_steps']
    )
    
    # 训练模型
    trained_model = train_model(model, dataloader['train'], dataloader['val'], processor.scaler, config)
    
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.load_state_dict(torch.load(f"best_{config['model_name']}.pth"))
    
    # 获取测试数据
    test_series = processor.scaler.inverse_transform(
        np.concatenate([X_test[:, :, 0], y_test], axis=1)
    )[:, :config['sequence_length']]  # 历史数据部分
    
    # 预测
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader['test']:
            pred = trained_model(X_batch.to(device)).cpu().numpy()
            predictions.append(pred)
    predictions = processor.scaler.inverse_transform(np.concatenate(predictions))
    
    # 可视化
    plot_forecast_series(
        series=test_series[:15],  # 显示前15个样本
        actuals=processor.scaler.inverse_transform(y_test)[:15],
        predictions=predictions[:15]
    )
    
    # 评估指标
    rmse = np.sqrt(mean_squared_error(
        processor.scaler.inverse_transform(y_test), 
        predictions
    ))
    print(f"Test RMSE: {rmse:.2f}°C")

if __name__ == "__main__":
    main()
