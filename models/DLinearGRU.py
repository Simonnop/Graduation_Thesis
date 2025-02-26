import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.time_window
        self.pred_len = args.output_size
        self.individual = args.individual
        
        # 趋势分解
        self.decomp = SeriesDecomp(kernel_size=args.decomp_kernel)
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # 输入投影
        self.input_projection = nn.Linear(args.input_size * 3, args.hidden_size)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout_rate if args.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.projection = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, batch_x_encoder, batch_x_decoder):
        # batch_x_encoder: [Batch, Input length, Channel]
        x = batch_x_encoder
        
        # 趋势分解
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                        dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                     dtype=trend_init.dtype).to(trend_init.device)
            
            for i in range(seasonal_init.size(1)):
                seasonal_output[:, i, :] = self.Linear_Seasonal[0](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[0](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = torch.cat((seasonal_output, trend_output), dim=1)
        hidden_state = x.permute(0, 2, 1)[:,:,:]  # to [Batch, Output length, Channel]
        
        # x_enc: [Batch, time_window, input_size]
        x = torch.cat((hidden_state, batch_x_decoder), dim=2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # GRU处理
        output, _ = self.gru(x)
        
        # 只使用最后一个时间步的输出进行预测
        last_hidden = output[:, -1, :]
        
        # 生成预测序列
        predictions = self.projection(last_hidden)
        
        return predictions

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

def moving_avg(kernel_size):
    """
    移动平均
    """
    return nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2) 