#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 基础参数设置
model_name="TCN"
data="Huron"
input_size=6
output_size=48
time_window=288
train_epochs=100
patience=10

# 网格搜索参数
hidden_sizes=(32 64 128)
kernel_sizes=(2 3 4)
learning_rates=(0.01 0.005 0.001)
num_channels_options=("16 32 64" "32 64 128" "64 128 256" "128 256 512")

# 创建日志目录
log_dir="logs/grid_search/TCN"
mkdir -p $log_dir

# 开始网格搜索
for hidden_size in "${hidden_sizes[@]}"; do
  for kernel_size in "${kernel_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for num_channels in "${num_channels_options[@]}"; do
        
        # 构建实验名称
        exp_name="${model_name}_h${hidden_size}_k${kernel_size}_lr${learning_rate}_ch${num_channels// /_}"
        log_file="${log_dir}/${exp_name}.log"
        
        echo "开始训练: $exp_name"
        echo "参数配置:" > $log_file
        echo "hidden_size: $hidden_size" >> $log_file
        echo "kernel_size: $kernel_size" >> $log_file
        echo "learning_rate: $learning_rate" >> $log_file
        echo "num_channels: $num_channels" >> $log_file
        echo "------------------------" >> $log_file
        
        # 运行训练
        python run.py \
          --is_training 1 \
          --model $model_name \
          --data $data \
          --input_size $input_size \
          --hidden_size $hidden_size \
          --output_size $output_size \
          --time_window $time_window \
          --kernel_size $kernel_size \
          --num_channels $num_channels \
          --learning_rate $learning_rate \
          --train_epochs $train_epochs \
          --batch_size 32 \
          --patience $patience \
          --loss "MSE" \
          --use_gpu True >> $log_file 2>&1

        # 运行测试
        python run.py \
          --is_training 0 \
          --model $model_name \
          --data $data \
          --input_size $input_size \
          --hidden_size $hidden_size \
          --output_size $output_size \
          --time_window $time_window \
          --kernel_size $kernel_size \
          --num_channels $num_channels \
          --batch_size 32 \
          --loss "MSE" \
          --use_gpu True >> $log_file 2>&1
          
        echo "完成实验: $exp_name"
        echo "结果已保存到: $log_file"
        echo "------------------------"
      done
    done
  done
done

echo "网格搜索完成！"

# 分析结果
echo "最佳验证损失的配置："
grep -r "Vali Loss:" $log_dir/* | sort -n -k4 | head -n 5 