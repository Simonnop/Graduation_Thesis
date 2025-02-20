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

    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="ANN",
        help="model name, options: [...]",
    )

    # 任务描述
    parser.add_argument("--des", type=str, default="test", help="exp description")

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )

    # path
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # # model define
    parser.add_argument("--time_window", type=int, required=False, default=1, help="时间窗口大小")
    parser.add_argument("--input_size", type=int, required=True, help="输入特征的大小")
    parser.add_argument("--hidden_size", type=int, required=False, help="隐藏层的大小")
    parser.add_argument("--output_size", type=int, required=True, help="输出特征的大小")
    parser.add_argument("--num_class", type=int, required=False, default=2, help="分类的类别数")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout比率")
    parser.add_argument(
        "--num_hidden_layers", type=int, required=False, help="隐藏层的数量"
    )

    # TCN 特有参数
    parser.add_argument("--kernel_size", type=int, default=2, help="TCN的卷积核大小")
    parser.add_argument("--num_channels", type=list, default=[64,128], help="TCN每层通道数")

    # LSTM/GRU 特有参数
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM/GRU的层数")

    # DLinear 特有参数
    parser.add_argument("--individual", type=bool, default=True, help="是否对每个特征单独建模")
    parser.add_argument("--decomp_kernel", type=int, default=25, help="分解的kernel大小")

    # Transformer 特有参数
    parser.add_argument("--d_model", type=int, default=512, help="Transformer的特征维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="编码器层数")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="解码器层数")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="前馈网络维度")

    # optimization
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print("Args in experiment:")
    setting = f"{args.data}/{args.model}"
    print("参数设置:")
    print(f"是否训练: {args.is_training}")
    print(f"模型名称: {args.model}")
    print(f"描述: {args.des}")
    print(f"数据集类型: {args.data}")
    print(f"时间窗口大小: {args.time_window}")
    print(f"输入特征大小: {args.input_size}")
    print(f"隐藏层大小: {args.hidden_size}")
    print(f"输出特征大小: {args.output_size}")
    print(f"dropout比率: {args.dropout_rate}")
    print(f"隐藏层数量: {args.num_hidden_layers}")
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
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()