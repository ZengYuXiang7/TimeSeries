# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange

from layers.revin import RevIN


class Linear3(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear3, self).__init__()
        self.config = config
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.d_model = config.d_model

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.seq_len, self.seq_len + self.d_model * 2),  # 第一层：扩张隐藏维度
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(self.seq_len + self.d_model * 2),
            torch.nn.Linear(self.seq_len + self.d_model * 2, self.d_model),  # 第二层：压缩回原始隐藏维度
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(self.d_model),  # 层归一化
            torch.nn.Linear(self.d_model, self.pred_len)  # 第三层：输出层
        )

    def forward(self, x, x_mark):
        # x: [B, L, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = rearrange(x, 'bs seq_len d_model -> bs d_model seq_len')
        y = self.model(x)
        y = rearrange(y, 'bs d_model pred_len -> bs pred_len d_model')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        # shape = [bs, pred_len, d=21]
        return y
