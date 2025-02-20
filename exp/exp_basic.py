
import os
import torch
from models import ANN, DLinear, GRU, LSTM, TCN, Transformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args

        # TODO 添加模型
        self.model_dict = {
            'ANN': ANN,
            'DLinear': DLinear,
            'GRU': GRU,
            'LSTM': LSTM,
            'TCN': TCN,
            'Transformer': Transformer
        }
        
        self.device = self._acquire_device()
        self.model = self._build_model()
        try:
            self.model = self.model.to(self.device)
        except:
            pass

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            # print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
