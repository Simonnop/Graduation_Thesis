import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

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
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # 从args中获取参数
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_channels = args.num_channels
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout_rate

        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(self.kernel_size-1) * dilation_size,
                                   dropout=self.dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(self.num_channels[-1], self.output_size)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]  # 取最后一个时间步
        return self.linear(x) 