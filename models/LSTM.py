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
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                           batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out 