#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 设置模型参数
model_name="DLinearGRU"
data="Huron"

# 网络结构参数
input_size=6      # 必需参数：输入特征维度
hidden_size=512   # 必需参数：隐藏层大小
output_size=48    # 必需参数：输出维度
time_window=288   # 时间窗口大小
individual=True   # DLinear特有参数：是否独立建模
decomp_kernel=25  # DLinear特有参数：分解核大小
num_layers=2      # GRU特有参数：GRU层数
dropout=0.1       # GRU特有参数：dropout率

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
  --use_gpu True \
  --input_size $input_size \
  --hidden_size $hidden_size \
  --output_size $output_size \
  --time_window $time_window \
  --individual $individual \
  --decomp_kernel $decomp_kernel \
  --num_layers $num_layers \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --weight $weight \
  --loss "Weighted"

# 运行测试
python run.py \
  --is_training 0 \
  --model $model_name \
  --data $data \
  --use_gpu True \
  --input_size $input_size \
  --hidden_size $hidden_size \
  --output_size $output_size \
  --time_window $time_window \
  --individual $individual \
  --decomp_kernel $decomp_kernel \
  --num_layers $num_layers \
  --dropout $dropout \
  --batch_size $batch_size \
  --weight $weight \
  --loss "Weighted"

echo "脚本运行完成！" 