import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class MichiganPowerDataset(Dataset):
    def __init__(self, data, lookback=288, predict_size=48):  # 24小时=288个5分钟, 4小时=48个5分钟
        # 确保数据按时间排序
        self.data = data.sort_values('datetime').reset_index(drop=True)
        
        # 时间特征
        self.data['hour'] = pd.to_datetime(self.data['datetime']).dt.hour
        self.data['day_of_week'] = pd.to_datetime(self.data['datetime']).dt.dayofweek
        
        # 事件类型编码
        self.event_encoder = LabelEncoder()
        self.data['event_type_encoded'] = self.event_encoder.fit_transform(self.data['event_type'])
        
        # 归一化功率和调节幅度
        self.power_scaler = MinMaxScaler()
        self.change_scaler = MinMaxScaler()
        
        # 归一化功率数据
        power_data = self.data['power'].values.reshape(-1, 1)
        self.normalized_power = self.power_scaler.fit_transform(power_data)
        
        # 归一化调节幅度数据
        change_data = np.column_stack((self.data['up_change'], self.data['down_change']))
        self.normalized_changes = self.change_scaler.fit_transform(change_data)
        
        self.lookback = lookback
        self.predict_size = predict_size
        
    def __len__(self):
        return len(self.data) - self.lookback - self.predict_size + 1
    
    def __getitem__(self, idx):
        # 基础特征：功率
        x_power = self.normalized_power[idx:idx+self.lookback]
        
        # 时间特征
        x_hour = self.data['hour'].values[idx:idx+self.lookback]
        x_day = self.data['day_of_week'].values[idx:idx+self.lookback]
        
        # 事件特征
        x_event = self.data['event_type_encoded'].values[idx:idx+self.lookback]
        x_up_change = self.normalized_changes[idx:idx+self.lookback, 0]
        x_down_change = self.normalized_changes[idx:idx+self.lookback, 1]
        
        # 组合所有特征
        x = np.column_stack((
            x_power,
            x_hour/24.0,  # 归一化时间
            x_day/7.0,    # 归一化星期
            x_event/len(self.event_encoder.classes_),  # 归一化事件类型
            x_up_change,
            x_down_change
        ))
        
        # 获取目标序列
        y = self.normalized_power[idx+self.lookback:idx+self.lookback+self.predict_size]
        
        return torch.FloatTensor(x), torch.FloatTensor(y.reshape(-1))

    def inverse_transform_power(self, normalized_power):
        return self.power_scaler.inverse_transform(normalized_power.reshape(-1, 1))

class PowerPredictor(nn.Module):
    def __init__(self, input_features=6, hidden_size=128, num_layers=2, output_size=48):
        super(PowerPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_features, 
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2是因为双向LSTM
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        out = self.relu(out)
        out = self.dropout(out)
        predictions = self.fc2(out)
        return predictions

def train_model(model, train_loader, val_loader, num_epochs=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    # 使用trange替代range来显示epoch进度
    epoch_pbar = trange(num_epochs, desc='Training')
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0
        # 添加训练批次进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', 
                         leave=False, position=1)
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # 更新训练进度条的描述
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        # 添加验证批次进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation', 
                       leave=False, position=1)
        
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                
                # 更新验证进度条的描述
                val_pbar.set_postfix({'batch_loss': f'{val_loss/len(val_loader):.4f}'})
        
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        
        # 更新epoch进度条的描述
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型和早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_power_predictor.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered")
                break
    
    print("\n训练完成！")
    return model

def visualize_prediction(true_values, predicted_values, title="功率预测对比"):
    """
    可视化预测结果和真实值的对比
    
    Args:
        true_values: 真实值数组
        predicted_values: 预测值数组
        title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(true_values)) * 5  # 转换为分钟
    
    plt.plot(x_axis, true_values, label='真实值', marker='o', markersize=3)
    plt.plot(x_axis, predicted_values, label='预测值', marker='s', markersize=3)
    
    plt.title(title)
    plt.xlabel('预测时间（分钟）')
    plt.ylabel('功率 (kW)')
    plt.grid(True)
    plt.legend()
    
    # 添加RMSE和MAE指标
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    mae = np.mean(np.abs(true_values - predicted_values))
    plt.text(0.02, 0.98, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    print("正在加载数据...")
    # 读取数据
    df = pd.read_csv('/home/suruixian/Documents/Graduation_Thesis/explore/michigan_power_events.csv')
    
    print("正在准备数据集...")
    # 创建数据集
    dataset = MichiganPowerDataset(df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print("正在初始化模型...")
    # 初始化模型
    model = PowerPredictor()
    
    print("开始训练...")
    # 训练模型
    model = train_model(model, train_loader, val_loader)
    
    print("开始预测...")
    # 预测示例
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        # 获取一个有效的样本进行预测
        test_x, test_y = dataset[0]  # 获取输入和真实值
        if test_x.dim() == 2:
            test_x = test_x.unsqueeze(0)
        test_x = test_x.to(device)
        
        print(f"输入张量形状: {test_x.shape}")
        
        prediction = model(test_x)
        
        # 转换回实际功率值
        prediction_power = dataset.inverse_transform_power(prediction.cpu().numpy()).flatten()
        true_power = dataset.inverse_transform_power(test_y.numpy().reshape(-1, 1)).flatten()
        
        print("\n未来4小时的预测功率值:")
        for i, power in enumerate(prediction_power):
            print(f"{i*5}分钟后: {float(power):.2f} kW")
        
        # 可视化预测结果
        visualize_prediction(true_power, prediction_power)

if __name__ == "__main__":
    main() 