import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # 基本参数设置
        self.time_window = args.time_window
        self.output_size = args.output_size
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout_rate
        )
        
        # 输出层
        self.projection = nn.Linear(args.hidden_size, args.output_size)
    
    def forward(self, x_enc, x_dec):
        # x_enc: [Batch, time_window, input_size]
        
        # GRU 前向传播
        output, _ = self.gru(x_enc)
        
        # 只使用最后一个时间步的输出进行预测
        last_hidden = output[:, -1, :]
        
        # 生成预测序列
        predictions = self.projection(last_hidden)
        
        return predictions 