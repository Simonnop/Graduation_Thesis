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
        self.input_size = args.input_size
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_encoder_layers = args.num_encoder_layers
        self.num_decoder_layers = args.num_decoder_layers
        self.dim_feedforward = args.dim_feedforward
        self.output_size = args.output_size
        self.dropout = args.dropout_rate
        
        # 输入映射层
        self.encoder_input_proj = nn.Linear(self.input_size, self.d_model)
        self.decoder_input_proj = nn.Linear(self.input_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        # Transformer层
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 输出映射层
        self.output_proj = nn.Linear(self.d_model, 1)
    
    def forward(self, src, tgt):
        # src: [batch_size, src_seq_len, input_size]
        # tgt: [batch_size, tgt_seq_len, input_size]
        
        # 输入映射
        src = self.encoder_input_proj(src)  # [batch_size, src_seq_len, d_model]
        tgt = self.decoder_input_proj(tgt)  # [batch_size, tgt_seq_len, d_model]
        
        # 位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # 创建掩码
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        # Transformer前向传播
        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask
        )  # [batch_size, tgt_seq_len, d_model]
        
        # 输出映射
        out = self.output_proj(out)  # [batch_size, tgt_seq_len, 1]
        out = out.squeeze(-1)  # [batch_size, tgt_seq_len]
        
        return out 