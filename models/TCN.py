import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # 基本参数设置
        self.time_window = args.time_window
        self.output_size = args.output_size
        
        layers = []
        num_channels = args.num_channels
        kernel_size = args.kernel_size
        
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = args.input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=args.dropout_rate
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        self.projection = nn.Linear(num_channels[-1], args.output_size)
        
    def forward(self, x_enc, x_dec):
        # x_enc: [Batch, time_window, input_size]
        x = x_enc.transpose(1, 2)  # TCN expects [Batch, Channel, Length]
        
        # TCN 前向传播
        output = self.tcn(x)
        output = output.transpose(1, 2)  # 转回 [Batch, Length, Channel]
        
        # 只使用最后一个时间步的输出进行预测
        last_hidden = output[:, -1, :]
        
        # 生成预测序列
        predictions = self.projection(last_hidden)
        
        return predictions 