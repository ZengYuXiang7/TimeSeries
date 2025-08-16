# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.dft import DFT
from layers.revin import RevIN

class TimeSeriesModel(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(TimeSeriesModel, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)

        # self.
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x, x_mark):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        y = self.predict_linear(x)

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        return y