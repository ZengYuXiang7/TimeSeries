# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange
from torch import nn

from layers.revin import RevIN

#以下是做法一：d_model = 32
class Linear(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.d_model = config.d_model

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # 中间线性层，对时间维度进行映射（注意是对 seq_len 映射到更大维度）
        # 输入维度：seq_len → 输出维度：d_model + seq_len
        self.middle_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        # 用于将中间表示直接预测为 pred_len 步（这层在 forward 中未使用）
        self.predict_linear = nn.Linear(self.seq_len + self.pred_len, self.pred_len)

        # 对特征维度进行非线性特征提升（例如 21 → 50）
        self.up_feature_linear = nn.Linear(self.config.input_size, self.d_model)

        # 再降回原特征维度（例如 50 → 21）
        self.down_feature_linear = nn.Linear(self.d_model, self.config.input_size)

    def forward(self, x, x_mark):
        # x: [B, L, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # rearrange to [B, D, L] to apply Linear on seq_len dimension
        x = rearrange(x, 'B L D -> B D L')
        y = self.middle_linear(x)  # [B D d_model]
        y = self.predict_linear(y)  # [B, D, P]
        y = rearrange(y, 'B D P -> B P D')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        # shape = [B, P, D]
        return y


