import argparse
import os
import torch
from exp.exp_forecast import Exp_Forecast

import random
import numpy as np

import os

# 切换工作路径至当前
os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="ANN",
        help="model name, options: [ANN, DLinear, LSTM, GRU, TCN, Transformer]",
    )

    # 数据参数
    parser.add_argument("--data", type=str, required=True, default="ETTm1", help="dataset type")
    parser.add_argument("--features", type=str, default="M",
                      help="forecasting task, options:[M, S, MS]")
    
    # 通用模型参数
    parser.add_argument("--time_window", type=int, default=96, help="input sequence length")
    parser.add_argument("--input_size", type=int, required=True, help="input feature size")
    parser.add_argument("--hidden_size", type=int, default=512, help="hidden layer dimension")
    parser.add_argument("--output_size", type=int, required=True, help="output size")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout rate")

    # DLinear 特有参数
    parser.add_argument("--individual", type=bool, default=True, help="DLinear: individual modeling for each dimension")
    parser.add_argument("--decomp_kernel", type=int, default=25, help="DLinear: decomposition kernel size")

    # LSTM/GRU 特有参数
    parser.add_argument("--num_layers", type=int, default=2, help="number of LSTM/GRU layers")

    # TCN 特有参数
    parser.add_argument("--kernel_size", type=int, default=3, help="TCN kernel size")
    parser.add_argument("--num_channels", nargs='+', type=int, default=[64, 128, 256], 
                        help="TCN channel numbers")

    # Transformer 特有参数
    parser.add_argument("--d_model", type=int, default=512, help="Transformer model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Transformer head number")
    parser.add_argument("--d_ff", type=int, default=2048, help="Transformer feedforward dimension")

    # 优化器参数
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--weight", type=float, default=5.0, help="weight for loss function")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print("Args in experiment:")
    setting = f"{args.data}/{args.model}"
    print("参数设置:")
    print(f"是否训练: {args.is_training}")
    print(f"模型名称: {args.model}")
    print(f"数据集类型: {args.data}")
    print(f"时间窗口大小: {args.time_window}")
    print(f"输入特征大小: {args.input_size}")
    print(f"隐藏层大小: {args.hidden_size}")
    print(f"输出特征大小: {args.output_size}")
    print(f"dropout比率: {args.dropout_rate}")

    # 根据模型类型打印特有参数
    if args.model == "DLinear" or args.model == "DLinearformer":
        print(f"是否独立建模: {args.individual}")
        print(f"分解核大小: {args.decomp_kernel}")
    elif args.model in ["LSTM", "GRU"]:
        print(f"循环层数: {args.num_layers}")
    elif args.model == "TCN":
        print(f"卷积核大小: {args.kernel_size}")
        print(f"通道数: {args.num_channels}")
    elif args.model == "Transformer" or args.model == "DLinearformer":
        print(f"模型维度: {args.d_model}")
        print(f"注意力头数: {args.n_heads}")
        print(f"前馈网络维度: {args.d_ff}")

    print(f"训练轮数: {args.train_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"损失函数: {args.loss}")

    
    Exp = Exp_Forecast
    exp = Exp(args)  # set experiments

    if args.is_training:
        # setting record of experiments

        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        setting = f"{args.data}/{args.model}"
        checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        # 检查模型是否已训练过
        if not os.path.exists(checkpoint_path):
            print(f">>>>>>>模型 {setting} 尚未训练，自动开始训练过程<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # 临时修改训练状态
            args.is_training = 1
            exp.train(setting)
            # 恢复测试状态
            args.is_training = 0
            # 重新初始化实验对象
            exp = Exp(args)
        
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()