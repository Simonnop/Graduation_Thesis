#!/bin/bash

echo "开始训练所有模型..."

# 训练参数
learning_rate=0.005
train_epochs=30
batch_size=32
patience=3
weight=5.0

# 运行所有模型
bash scripts/ann.sh
bash scripts/lstm.sh
bash scripts/gru.sh
bash scripts/tcn.sh
bash scripts/dlinear.sh
bash scripts/dgru.sh
bash scripts/transformer.sh
# bash scripts/dlinearformer.sh

echo "所有模型训练完成！"
