#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate py310_torch23

echo "开始测试所有模型..."

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 数据集列表
datasets=("Huron" "Victoria")
# datasets=("Victoria")

# 模型列表
models=("ANN" "LSTM" "GRU" "TCN" "DLinear" "DLinearGRU" "Transformer")

# 通用参数
input_size=6
output_size=48
time_window=288
batch_size=32
weight=5.0

# 遍历所有数据集
for data in "${datasets[@]}"; do
    echo "===== 测试数据集: $data ====="
    
    # 遍历所有模型
    for model in "${models[@]}"; do
        echo "----- 测试模型: $model -----"
        
        # 根据模型类型设置特定参数
        case $model in
            "ANN")
                hidden_size=1024
                num_layers=4
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
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
                ;;
                
            "LSTM" | "GRU")
                hidden_size=512
                num_layers=2
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
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
                ;;
                
            "TCN")
                hidden_size=512
                kernel_size=3
                num_channels="64 128 256"
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
                  --data $data \
                  --input_size $input_size \
                  --hidden_size $hidden_size \
                  --output_size $output_size \
                  --kernel_size $kernel_size \
                  --num_channels $num_channels \
                  --time_window $time_window \
                  --batch_size $batch_size \
                  --weight $weight \
                  --loss "Weighted" \
                  --use_gpu True
                ;;
                
            "DLinear")
                hidden_size=512
                individual=True
                decomp_kernel=25
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
                  --data $data \
                  --input_size $input_size \
                  --hidden_size $hidden_size \
                  --output_size $output_size \
                  --time_window $time_window \
                  --individual $individual \
                  --decomp_kernel $decomp_kernel \
                  --batch_size $batch_size \
                  --weight $weight \
                  --loss "Weighted" \
                  --use_gpu True
                ;;
                
            "DLinearGRU")
                hidden_size=512
                individual=True
                decomp_kernel=25
                num_layers=2
                dropout=0.1
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
                  --data $data \
                  --input_size $input_size \
                  --hidden_size $hidden_size \
                  --output_size $output_size \
                  --time_window $time_window \
                  --individual $individual \
                  --decomp_kernel $decomp_kernel \
                  --num_layers $num_layers \
                  --dropout_rate $dropout \
                  --batch_size $batch_size \
                  --weight $weight \
                  --loss "Weighted" \
                  --use_gpu True
                ;;
                
            "Transformer")
                hidden_size=256
                num_layers=2
                n_heads=8
                d_ff=1024
                
                # 运行测试
                python run.py \
                  --is_training 0 \
                  --model $model \
                  --data $data \
                  --input_size $input_size \
                  --hidden_size $hidden_size \
                  --output_size $output_size \
                  --time_window $time_window \
                  --num_layers $num_layers \
                  --n_heads $n_heads \
                  --d_ff $d_ff \
                  --batch_size $batch_size \
                  --weight $weight \
                  --loss "Weighted" \
                  --use_gpu True
                ;;
        esac
        
        echo "----- $model 测试完成 -----"
    done
    
    echo "===== $data 数据集所有模型测试完成 ====="
done

echo "所有模型测试完成！" 