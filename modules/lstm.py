# coding: utf-8
# Author: Yuxiang Zeng

import torch
from modules.pred_layer import Predictor
import pickle

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super(LSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM
        self.trans = torch.nn.Linear(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.predictor = Predictor(self.hidden_size, self.hidden_size, self.args.num_preds)

    def forward(self, x):
        x = x.to(torch.float32)  # 确保输入是float32类型
        x = self.trans(x)
        output, (hn, cn) = self.lstm(x)
        last_output = output[:, -1, :]
        y = self.predictor(last_output)
        return y

