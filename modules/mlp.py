# coding : utf-8
# Author : yuxiang Zeng
import torch
from torch.nn import *

from modules.pred_layer import Predictor


class MLPTimeRefiner(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.window = args.window
        self._time_linear = Linear(args.dimension, args.dimension)
        self.context = Sequential(
            Linear(self.window * args.dimension, args.dimension),
            LayerNorm(args.dimension),
            ReLU(),
            Linear(args.dimension, args.dimension)
        )

        self.rainbow = torch.arange(-self.window + 1, 1).reshape(1, -1).to(args.device)
        self.attn = Sequential(Linear(2 * args.dimension, 1), Tanh())

    def forward(self, time_embeds):
        bs, window, dimension = time_embeds.size()

        # attention score
        query = time_embeds.reshape(bs, -1)
        local_context = self.context(query)
        local_context = local_context.unsqueeze(1).repeat(1, window, 1)
        query = torch.cat([time_embeds, local_context], dim=-1)
        attn_score = self.attn(query)

        # Attention Weighting
        time_embeds = torch.sum(time_embeds * attn_score, dim=1)

        # Post Linear
        return self._time_linear(time_embeds)

    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.args.window)
        tids += self.rainbow - 1
        tids = tids.relu()
        return tids


class MLP(Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #
        self.trans = torch.nn.Linear(input_size, hidden_size)
        self.time_encoder = MLPTimeRefiner(args)
        self.predictor = Predictor(self.hidden_size, self.hidden_size, 1)
    
    def forward(self, x):
        x = x.to(torch.float32)  # 确保输入是float32类型
        x = self.trans(x)
        output = self.time_encoder(x)  # 获取LSTM输出和隐状态
        y = self.predictor(output)  # 只使用最后一个时间步的输出进行预测
        return y