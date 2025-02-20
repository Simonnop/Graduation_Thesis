import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, path + '/' + 'checkpoint.pth')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, attacked_preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    num = 48

    if torch.is_tensor(true):
        true = true.cpu().detach().numpy()
    if preds is not None and torch.is_tensor(preds):
        preds = preds.cpu().detach().numpy()
    if attacked_preds is not None and torch.is_tensor(attacked_preds):
        attacked_preds = attacked_preds.cpu().detach().numpy()
    
    
    plt.figure(figsize=(18, 18))  # 增大图像尺寸

    # 绘制前48个点
    plt.subplot(3, 1, 1)
    for i in range(num):
        if preds is not None and attacked_preds is not None:
            orig_error = abs(preds[i] - true[i])
            attack_error = abs(attacked_preds[i] - true[i])
            if attack_error < orig_error:
                plt.axvline(x=i, color='red', alpha=0.15, linestyle='-')
            else:
                plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
        else:
            plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
    
    plt.plot(true[:num], 'o-', color='green', label='Ground Truth', linewidth=2)  # 绿色
    if preds is not None:
        plt.plot(preds[:num], 'o-', color='blue', label='Prediction', linewidth=2)  # 蓝色
    if attacked_preds is not None:
        plt.plot(attacked_preds[:num], 'o-', color='orange', label='Attacked Prediction', linewidth=2, linestyle='--')  # 橙色
    plt.title('前48个点')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 绘制中间48个点
    plt.subplot(3, 1, 2)
    for i in range(num):
        if preds is not None and attacked_preds is not None:
            orig_error = abs(preds[len(true)//2 - num//2 + i] - true[len(true)//2 - num//2 + i])
            attack_error = abs(attacked_preds[len(true)//2 - num//2 + i] - true[len(true)//2 - num//2 + i])
            if attack_error < orig_error:
                plt.axvline(x=i, color='red', alpha=0.15, linestyle='-')
            else:
                plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
        else:
            plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
    
    plt.plot(true[len(true)//2 - num//2:len(true)//2 + num//2], 'o-', color='green', label='Ground Truth', linewidth=2)  # 绿色
    if preds is not None:
        plt.plot(preds[len(true)//2 - num//2:len(true)//2 + num//2], 'o-', color='blue', label='Prediction', linewidth=2)  # 蓝色
    if attacked_preds is not None:
        plt.plot(attacked_preds[len(true)//2 - num//2:len(true)//2 + num//2], 'o-', color='orange', label='Attacked Prediction', linewidth=2, linestyle='--')  # 橙色
    plt.title('中间48个点')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 绘制末尾48个点
    plt.subplot(3, 1, 3)
    for i in range(num):
        if preds is not None and attacked_preds is not None:
            orig_error = abs(preds[-num + i] - true[-num + i])
            attack_error = abs(attacked_preds[-num + i] - true[-num + i])
            if attack_error < orig_error:
                plt.axvline(x=i, color='red', alpha=0.15, linestyle='-')
            else:
                plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
        else:
            plt.axvline(x=i, color='gray', alpha=0.15, linestyle=':')
    
    plt.plot(true[-num:], 'o-', color='green', label='Ground Truth', linewidth=2)  # 绿色
    if preds is not None:
        plt.plot(preds[-num:], 'o-', color='blue', label='Prediction', linewidth=2)  # 蓝色
    if attacked_preds is not None:
        plt.plot(attacked_preds[-num:], 'o-', color='orange', label='Attacked Prediction', linewidth=2, linestyle='--')  # 橙色
    plt.title('末尾48个点')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplots_adjust(hspace=0.1)  # 调整子图之间的垂直间距
    plt.savefig(name, bbox_inches='tight', dpi=300)  # 增大分辨率


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)