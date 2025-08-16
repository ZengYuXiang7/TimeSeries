# coding: utf-8
# Author: mkw
# Date: 2025-06-08 19:00
# Description: DFTGetSeasonal

import torch
import torch.nn as nn


class DFT(nn.Module):
    def __init__(self, top_k):
        super(DFT, self).__init__()
        self.top_k = top_k

    def forward(self, x, x_mark=None):
        # x: [B, L, D]
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0  # 去除直流分量
        topk_vals, _ = torch.topk(freq, self.top_k, dim=1)
        threshold = topk_vals[:, -1:, :]  # [B, 1, D]
        xf = torch.where(freq >= threshold, xf, torch.zeros_like(xf))
        seasonal = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        return seasonal  # [B, L, D]


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x, x_mark=None):
        # x: [B, L, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # [B, D, L]
        x = x.permute(0, 2, 1)  # [B, L, D]
        return x


class DFTDecomModel(nn.Module):
    def __init__(self, config):
        super(DFTDecomModel, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.top_k = config.top_k
        self.kernel_size = config.kernel_size
        self.individual = config.individual
        self.channels = config.input_size  # 即 D

        self.dft = DFT(top_k=self.top_k)
        self.trend = moving_avg(kernel_size=self.kernel_size, stride=1)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_mark=None):  # x: [B, L, D]
        seasonal = self.dft(x)
        trend = self.trend(x)

        seasonal = seasonal.permute(0, 2, 1)  # [B, D, L]
        trend = trend.permute(0, 2, 1)        # [B, D, L]

        if self.individual:
            seasonal_out = torch.zeros([x.size(0), self.channels, self.pred_len], device=x.device)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.channels):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)
            trend_out = self.Linear_Trend(trend)

        output = seasonal_out + trend_out  # [B, D, pred_len]
        return output.permute(0, 2, 1)     # [B, pred_len, D]
