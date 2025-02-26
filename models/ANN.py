import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # 基本参数设置
        self.encoder_input_dim = args.input_size
        self.decoder_input_dim = args.input_size
        self.hidden_dim = args.hidden_size
        self.output_dim = args.output_size
        
        # 时间窗口设置
        self.time_window = args.time_window
        
        # 构建多层编码器网络
        encoder_layers = []
        input_size = self.encoder_input_dim * self.time_window
        for _ in range(args.num_layers):
            encoder_layers.extend([
                nn.Linear(input_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = self.hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.decoder_input_dim * self.output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x_enc, x_dec):
        """
        参数:
            x_enc: 编码器输入 [batch_size, time_window, encoder_input_dim]
            x_dec: 解码器输入 [batch_size, pred_window, decoder_input_dim]
        """
        # x_enc shape: [batch_size, seq_len, encoder_input_dim]
        # x_dec shape: [batch_size, pred_len, decoder_input_dim]
        
        batch_size = x_enc.shape[0]
        
        # 展平编码器输入
        x_enc_flat = x_enc.reshape(batch_size, -1)        
        # 通过编码器
        enc_output = self.encoder(x_enc_flat)
        
        # 展平解码器输入
        x_dec_flat = x_dec.reshape(batch_size, -1)
        # 连接编码器输出和解码器输入
        dec_input = torch.cat([enc_output, x_dec_flat], dim=1)
        
        # 通过解码器得到预测结果
        output = self.decoder(dec_input)
        
        return output 