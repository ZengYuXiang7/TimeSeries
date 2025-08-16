# coding: utf-8
# Author: mkw
# Date: 2025-06-06 11:58
# Description: Linear5（增加 Dropout 和 LayerNorm）

import torch
from einops import rearrange
from torch import nn

from layers.revin import RevIN


class Linear5(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear5, self).__init__()
        self.config = config
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.patch_num = config.patch_num
        self.input_size = config.input_size
        self.patch_len = config.seq_len // config.patch_num

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # 增加 Dropout 和激活函数
        self.linear_up = nn.Sequential(
            nn.Linear(self.input_size, self.d_model),
            nn.GELU()
            # nn.Dropout(p=0.1)
        )

        # 增加 LayerNorm
        self.linear_back = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.input_size)
        )

        self.linear_final = torch.nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_mark=None):
        # x: [Bs, seq_len, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        # seq_len = patch_num * patch_len
        x = rearrange(x, 'Bs (patch_num patch_len) D -> Bs patch_num patch_len D', patch_num=self.patch_num)
        x = self.linear_up(x)  # [ Bs, patch_num, patch_len d_model ]
        x = rearrange(x, 'Bs patch_num patch_len d_model -> Bs (patch_num patch_len) d_model')
        x = self.linear_back(x)

        x = rearrange(x, 'Bs seq_len D -> Bs D seq_len')
        x = self.linear_final(x)

        x = rearrange(x, 'Bs D pred_len -> Bs pred_len D')

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
