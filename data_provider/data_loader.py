import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Seq2Seq_DR_Dataset(Dataset):
    def __init__(self, data_path, flag='train', time_window=288, output_size=48):  # 24小时=288个5分钟, 4小时=48个5分钟
        
        # 读取数据
        data = pd.read_csv(data_path)
        
        # 数据集切分
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        # 计算切分点
        num_samples = len(data)
        train_split = int(num_samples * train_ratio)
        val_split = int(num_samples * (train_ratio + val_ratio))
        
        # 根据flag选择相应数据段
        if flag == 'train':
            self.data = data[:train_split]
        elif flag == 'val':
            self.data = data[train_split:val_split]
        elif flag == 'test':  # test
            self.data = data[val_split:]
        else: 
            self.data = data
        
        # 确保数据按时间排序
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # 时间特征
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour
        self.data['day_of_week'] = pd.to_datetime(self.data['timestamp']).dt.dayofweek
        
        # 事件类型编码
        self.event_encoder = LabelEncoder()
        self.data['event_type_encoded'] = self.event_encoder.fit_transform(self.data['Event_Type'])

        # Event_Type,Event_Up Change,Event_Down Change
        
        # 归一化功率和调节幅度
        self.power_scaler = MinMaxScaler()
        self.change_scaler = MinMaxScaler()
        
        # 归一化功率数据
        power_data = self.data['FanPower'].values.reshape(-1, 1)
        self.normalized_power = self.power_scaler.fit_transform(power_data)
        
        # 归一化调节幅度数据
        change_data = np.column_stack((self.data['Event_Up Change'], self.data['Event_Down Change']))
        self.normalized_changes = self.change_scaler.fit_transform(change_data)
        
        self.time_window = time_window
        self.output_size = output_size
    
    def __len__(self):
        return len(self.data) - self.time_window - self.output_size + 1
    
    def __getitem__(self, idx):

        encoder_begin = idx
        encoder_end = idx + self.time_window
        decoder_begin = idx + self.time_window
        decoder_end = idx + self.time_window + self.output_size

        # 基础特征：功率
        x_power = self.normalized_power[encoder_begin:encoder_end]
        
        # 时间特征
        x_encoder_hour = self.data['hour'].values[encoder_begin:encoder_end]
        x_encoder_day = self.data['day_of_week'].values[encoder_begin:encoder_end]
        
        # 事件特征
        x_encoder_event = self.data['event_type_encoded'].values[encoder_begin:encoder_end]
        x_encoder_up_change = self.normalized_changes[encoder_begin:encoder_end, 0]
        x_encoder_down_change = self.normalized_changes[encoder_begin:encoder_end, 1]
        
        # 组合所有特征
        encoder_x = np.column_stack((
            x_power,
            x_encoder_hour/24.0,  # 归一化时间
            x_encoder_day/7.0,    # 归一化星期
            x_encoder_event/len(self.event_encoder.classes_),  # 归一化事件类型
            x_encoder_up_change,
            x_encoder_down_change
        ))

        # 时间特征
        x_decoder_hour = self.data['hour'].values[decoder_begin:decoder_end]
        x_decoder_day = self.data['day_of_week'].values[decoder_begin:decoder_end]
        
        # 事件特征
        x_decoder_event = self.data['event_type_encoded'].values[decoder_begin:decoder_end]
        x_decoder_up_change = self.normalized_changes[decoder_begin:decoder_end, 0]
        x_decoder_down_change = self.normalized_changes[decoder_begin:decoder_end, 1]

        x_zero = np.zeros_like(x_decoder_hour)

        decoder_x = np.column_stack((
            x_zero,
            x_decoder_hour/24.0,  # 归一化时间
            x_decoder_day/7.0,    # 归一化星期
            x_decoder_event/len(self.event_encoder.classes_),  # 归一化事件类型
            x_decoder_up_change,
            x_decoder_down_change
        ))

        # 将encoder和decoder序列拼接在一起
        encoder_x = torch.FloatTensor(encoder_x)  # [time_window, input_size]
        decoder_x = torch.FloatTensor(decoder_x)  # [output_size, input_size]
        
        # 获取目标序列
        y = self.normalized_power[decoder_begin:decoder_end]
        
        return torch.FloatTensor(encoder_x), torch.FloatTensor(decoder_x), torch.FloatTensor(y.reshape(-1))

    def inverse_transform_power(self, normalized_power):
        return self.power_scaler.inverse_transform(normalized_power.reshape(-1, 1))