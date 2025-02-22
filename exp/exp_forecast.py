from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from .utils.tools import EarlyStopping, adjust_learning_rate, visual
from .utils.losses import mape_loss, mase_loss, smape_loss, calcu_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x_encoder, batch_x_decoder, batch_y) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x_encoder = batch_x_encoder.float().to(self.device)
                batch_x_decoder = batch_x_decoder.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # # 打印输入数据的形状
                # print("编码器输入形状:", batch_x_encoder.shape)
                # print("解码器输入形状:", batch_x_decoder.shape)

                outputs = self.model(batch_x_encoder, batch_x_decoder)

                # loss_value = criterion(None,None,batch_y,outputs,torch.ones_like(outputs))
                # 打印预测值和真实值的形状
                # print("预测值形状:", outputs.shape)
                # print("真实值形状:", batch_y.shape)
                loss_value = criterion(batch_y, outputs)

                loss = loss_value
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model = torch.load(best_model_path)

        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        criterion = self._select_criterion(self.args.loss)
        vali_loss = []
        with torch.no_grad():
            # decoder input
            for i, (batch_x_encoder, batch_x_decoder, batch_y) in enumerate(vali_loader):
                
                batch_x_encoder = batch_x_encoder.float().to(self.device)
                batch_x_decoder = batch_x_decoder.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x_encoder, batch_x_decoder)

                loss_value = criterion(batch_y, outputs)
                # loss_value = criterion(None,None,batch_y,outputs,torch.ones_like(outputs))

                loss = loss_value
                vali_loss.append(loss.item())

            vali_loss = np.average(vali_loss)

        self.model.train()
        return vali_loss

    def test(self, setting, test=0):
        test_set, test_loader = self._get_data(flag='test')  # 获取数据集对象以访问归一化参数
        criterion = self._select_criterion(self.args.loss)

        if test:
            print('loading model')
            self.model = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        vali_loss = []
        self.model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            # decoder input
            for i, (batch_x_encoder, batch_x_decoder, batch_y) in enumerate(test_loader):
                
                batch_x_encoder = batch_x_encoder.float().to(self.device)
                batch_x_decoder = batch_x_decoder.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x_encoder, batch_x_decoder)

                # loss_value = criterion(None,None,batch_y,outputs,torch.ones_like(outputs))
                loss_value = criterion(batch_y, outputs)

                vali_loss.append(loss_value.cpu().numpy())
                
                predictions.append(outputs.cpu().numpy())
                true_values.append(batch_y.cpu().numpy())

            vali_loss = np.average(vali_loss)
    
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)

        # 反归一化处理
        predictions = test_set.inverse_transform_power(predictions)
        true_values = test_set.inverse_transform_power(true_values)

        # 使用calcu_loss计算各项指标
        metrics = calcu_loss(torch.from_numpy(predictions), torch.from_numpy(true_values))
        
        # 保存结果到CSV
        results_path = f'./results/metrics_{self.args.data}.csv'
        results_dict = {
            'model': self.args.model,
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'MAPE': metrics['mape'],
            'MSPE': metrics['mspe']
        }
        
        df_new = pandas.DataFrame([results_dict])
        if os.path.exists(results_path):
            df_existing = pandas.read_csv(results_path)
            # 检查模型是否已存在
            if self.args.model in df_existing['model'].values:
                # 更新已存在模型的指标值
                model_idx = df_existing.index[df_existing['model'] == self.args.model][0]
                df_existing.iloc[model_idx] = df_new.iloc[0]
                df = df_existing
            else:
                # 如果模型不存在，则追加新行
                df = pandas.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_csv(results_path, index=False)

        print('---------------------------------')
        print("Test Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")
        print('---------------------------------')

        np.save(os.path.join(folder_path, 'predictions.npy'), predictions)
        np.save(os.path.join(folder_path, 'true_values.npy'), true_values)

        print("预测结果路径:", os.path.join(folder_path, 'predictions.npy'))
        print("真实值路径:", os.path.join(folder_path, 'true_values.npy'))
        print("指标结果保存在:", results_path)
        
        return