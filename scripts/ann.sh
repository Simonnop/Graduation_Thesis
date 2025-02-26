#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# 设置模型参数
model_name="ANN"
data="Huron"

# 网络结构参数
input_size=6      # 必需参数：输入特征维度
hidden_size=1024   # 必需参数：隐藏层大小
output_size=48    # 必需参数：输出维度
num_layers=4  # 必需参数：隐藏层数量
time_window=288  # 新增参数：时间窗口大小

# 训练参数
learning_rate=0.005
train_epochs=30
batch_size=32
patience=3
weight=5.0

# 运行训练
python run.py \
  --is_training 1 \
  --model $model_name \
  --data $data \
  --input_size $input_size \
  --hidden_size $hidden_size \
  --output_size $output_size \
  --num_layers $num_layers \
  --time_window $time_window \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --weight $weight \
  --loss "Weighted" \
  --use_gpu True

# 运行测试
python run.py \
  --is_training 0 \
  --model $model_name \
  --data $data \
  --input_size $input_size \
  --hidden_size $hidden_size \
  --output_size $output_size \
  --num_layers $num_layers \
  --time_window $time_window \
  --batch_size $batch_size \
  --weight $weight \
  --loss "Weighted" \
  --use_gpu True

echo "脚本运行完成！"