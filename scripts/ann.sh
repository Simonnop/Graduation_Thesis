#!/bin/bash

# 设置模型参数
model_name="ANN"
data="Huron"

# 网络结构参数
input_size=6      # 必需参数：输入特征维度
hidden_size=256   # 必需参数：隐藏层大小
output_size=48    # 必需参数：输出维度
num_hidden_layers=2  # 必需参数：隐藏层数量
time_window=288  # 新增参数：时间窗口大小

# 训练参数
learning_rate=0.001
train_epochs=100
batch_size=32
patience=10

# # 运行训练
# python run.py \
#   --is_training 1 \
#   --model $model_name \
#   --data $data \
#   --input_size $input_size \
#   --hidden_size $hidden_size \
#   --output_size $output_size \
#   --num_hidden_layers $num_hidden_layers \
#   --time_window $time_window \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --batch_size $batch_size \
#   --patience $patience \
#   --loss "MSE"

# 运行测试
python run.py \
  --is_training 1 \
  --model $model_name \
  --data $data \
  --input_size $input_size \
  --hidden_size $hidden_size \
  --output_size $output_size \
  --num_hidden_layers $num_hidden_layers \
  --time_window $time_window \
  --batch_size $batch_size \
  --loss "MSE"

echo "脚本运行完成！"