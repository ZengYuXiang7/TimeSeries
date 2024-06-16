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
        self.attn = torch.nn.Sequential(torch.nn.Linear(args.dimension, 1), torch.nn.Tanh())
        self.rainbow = torch.arange(-self.num_windows + 1, 1).reshape(1, -1).to(args.device)

    def attention_layer(self, lstm_output):
        attention_scores = self.attn(lstm_output).squeeze(-1)  # [bs, window]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)  # [bs, window]
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # [bs, hidden_dim]
        return context_vector, attention_weights

    def forward(self, time_embeds):
        outputs, (hs, _) = self.lstm(time_embeds)
        time_embeds, _ = self.attention_layer(outputs)
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

