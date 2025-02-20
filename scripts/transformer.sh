#!/bin/bash

# 设置模型参数
model_name="Transformer"
data="Huron"

# 网络结构参数
input_size=6      # 输入特征维度
output_size=48    # 输出维度
time_window=288   # 时间窗口大小

# Transformer特有参数
d_model=32        # 模型维度(必须能被nhead整除)
nhead=4           # 注意力头数
num_encoder_layers=2  # 编码器层数
num_decoder_layers=2  # 解码器层数
dim_feedforward=128   # 前馈网络维度

# 训练参数
learning_rate=0.001
train_epochs=100
batch_size=32
patience=10

# 运行训练
python run.py \
  --is_training 1 \
  --model $model_name \
  --data $data \
  --input_size $input_size \
  --output_size $output_size \
  --time_window $time_window \
  --d_model $d_model \
  --nhead $nhead \
  --num_encoder_layers $num_encoder_layers \
  --num_decoder_layers $num_decoder_layers \
  --dim_feedforward $dim_feedforward \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --loss "MSE"

# 运行测试
python run.py \
  --is_training 0 \
  --model $model_name \
  --data $data \
  --input_size $input_size \
  --output_size $output_size \
  --time_window $time_window \
  --d_model $d_model \
  --nhead $nhead \
  --num_encoder_layers $num_encoder_layers \
  --num_decoder_layers $num_decoder_layers \
  --dim_feedforward $dim_feedforward \
  --batch_size $batch_size \
  --loss "MSE"

echo "脚本运行完成！" 