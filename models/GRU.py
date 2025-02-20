import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.dropout = args.dropout_rate
        
        # 为decoder输入添加额外的线性层
        self.decoder_proj = nn.Linear(5, self.input_size)  # 5 -> 6
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                         batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, 1)  # 每个时间步预测一个值
    
    def forward(self, batch_x_encoder, batch_x_decoder):
        # batch_x_encoder: [batch_size, encoder_seq_len, input_size]
        # batch_x_decoder: [batch_size, decoder_seq_len, input_size-1]
        
        # 1. 使用encoder数据获取隐藏状态
        h0 = torch.zeros(self.num_layers, batch_x_encoder.size(0), self.hidden_size).to(batch_x_encoder.device)
        _, hidden = self.gru(batch_x_encoder, h0)
        
        # 2. 将decoder输入投影到正确的维度
        decoder_input = self.decoder_proj(batch_x_decoder)
        
        # 3. 使用encoder的隐藏状态和处理后的decoder数据进行预测
        decoder_output, _ = self.gru(decoder_input, hidden)
        
        # 4. 对每个时间步进行预测
        predictions = self.fc(decoder_output)  # [batch_size, decoder_seq_len, 1]
        predictions = predictions.squeeze(-1)  # [batch_size, decoder_seq_len]
        
        return predictions 