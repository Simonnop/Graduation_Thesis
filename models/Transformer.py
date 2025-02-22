import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # 基本参数设置
        self.time_window = args.time_window
        self.output_size = args.output_size
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(args.hidden_size)
        
        # 输入投影
        self.input_projection = nn.Linear(args.input_size, args.hidden_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_layers
        )
        
        # 输出层
        self.projection = nn.Linear(args.hidden_size, args.output_size)
        
    def forward(self, x_enc, x_dec):
        # x_enc: [Batch, time_window, input_size]
        
        # 输入投影
        x = self.input_projection(x_enc)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码器
        output = self.transformer_encoder(x)
        
        # 只使用最后一个时间步的输出进行预测
        last_hidden = output[:, -1, :]
        
        # 生成预测序列
        predictions = self.projection(last_hidden)
        
        return predictions 