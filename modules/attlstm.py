# coding : utf-8
# Author : yuxiang Zeng


import torch

from modules.pred_layer import Predictor

import torch


class AttnLSTMTimeRefiner(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_windows = args.num_windows
        self._time_linear = torch.nn.Linear(args.dimension, args.dimension)
        self.lstm = torch.nn.LSTM(args.dimension, args.dimension, batch_first=False)
        self.attn = torch.nn.Sequential(torch.nn.Linear(2 * args.dimension, 1), torch.nn.Tanh())
        self.rainbow = torch.arange(-self.num_windows + 1, 1).reshape(1, -1).to(args.device)

    def forward(self, time_embeds):
        outputs, (hs, _) = self.lstm(time_embeds)
        hss = hs.repeat(time_embeds.size(0), 1, 1)
        combined_feats = torch.cat([outputs, hss], dim=-1)
        # print("Combined features shape:", combined_feats.shape)
        attn = self.attn(combined_feats)
        attn = torch.softmax(attn, dim=1)  # 使用 softmax 计算注意力权重
        # print("Attention shape:", attn.shape)
        time_embeds = torch.sum(outputs * attn, dim=1)  # 按时间步聚合
        # print("Time embeddings after attention shape:", time_embeds.shape)
        time_embeds = self._time_linear(time_embeds)
        # print("Final output shape:", time_embeds.shape)
        return time_embeds

class AttLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #
        self.trans = torch.nn.Linear(input_size, hidden_size)
        self.time_encoder = AttnLSTMTimeRefiner(args)
        self.predictor = Predictor(self.hidden_size, self.hidden_size, self.args.num_preds)

    def forward(self, x):
        x = x.to(torch.float32)  # 确保输入是float32类型
        x = self.trans(x)
        output = self.time_encoder(x)  # 获取LSTM输出和隐状态
        y = self.predictor(output)  # 只使用最后一个时间步的输出进行预测
        return y

