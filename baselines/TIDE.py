# file: baselines/TIDE.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.revin import RevIN


# ===================================================================================
#  子模块: IEBlock
# ===================================================================================
class IEBlock(nn.Module):
    """
    TIDE模型的基础交互与提取模块
    """

    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()
        # 1. 初始化维度参数
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        # 2. 初始化网络层
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )
        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        # 1. 特征维度变换
        x_permuted_for_spatial = x.permute(0, 2, 1)
        x_proj = self.spatial_proj(x_permuted_for_spatial)

        # 2. 节点/时间维度信息交互
        x_permuted_for_channel = x_proj.permute(0, 2, 1)
        channel_interaction = self.channel_proj(x_permuted_for_channel)
        x_interact = x_permuted_for_channel + channel_interaction

        # 3. 输出映射
        x_interact = x_interact.permute(0, 2, 1)
        x_out = self.output_proj(x_interact)

        return x_out.permute(0, 2, 1)


# ===================================================================================
#  主模型: TIDE
# ===================================================================================
class TIDE(nn.Module):
    """
    TIDE: Time-series Dense Encoder, adapted for your framework.
    """

    def __init__(self, config):
        super(TIDE, self).__init__()
        # 1. 初始化模型参数
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.input_size = config.input_size
        self.d_model = config.d_model
        self.chunk_size = config.chunk_size

        # 2. 计算padding长度和分块数量
        if self.seq_len % self.chunk_size != 0:
            self.padding_len = self.chunk_size - self.seq_len % self.chunk_size
        else:
            self.padding_len = 0
        self.padded_seq_len = self.seq_len + self.padding_len
        self.num_chunks = self.padded_seq_len // self.chunk_size

        # 3. 初始化RevIN归一化层
        self.revin_layer = RevIN(num_features=self.input_size, affine=False, subtract_last=False)

        # 4. 构建模型主要网络层
        self._build()

    def _build(self):
        # 定义高速公路(AR)分支、两个采样分支处理层(layer_1, layer_2)和最终融合层(layer_3)
        self.ar = nn.Linear(self.padded_seq_len, self.pred_len)
        self.layer_1 = IEBlock(self.chunk_size, self.d_model // 4, self.d_model // 4, self.num_chunks)
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        self.layer_2 = IEBlock(self.chunk_size, self.d_model // 4, self.d_model // 4, self.num_chunks)
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)
        self.layer_3 = IEBlock(self.d_model // 2, self.d_model // 2, self.pred_len, self.input_size)

    def forward(self, x, x_mark=None):
        B, _, N = x.shape

        # 1. 对输入数据进行归一化和Padding
        x = self.revin_layer(x, 'norm')
        if self.padding_len > 0:
            x_padded = F.pad(x, (0, 0, 0, self.padding_len))
        else:
            x_padded = x

        # 2. 计算高速公路分支(简单线性预测)
        highway = self.ar(x_padded.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. 连续采样分支，提取短期局部特征
        x1 = x_padded.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1).reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # 4. 间隔采样分支，提取长期周期性特征
        x2 = x_padded.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2).reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        # 5. 融合双分支特征并进行最终预测
        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1).permute(0, 2, 1)
        out = self.layer_3(x3)
        out = out.permute(0, 2, 1)

        # 6. 将复杂模型预测与简单线性预测相加
        final_out = out.permute(0, 2, 1) + highway

        # 7. 对预测结果进行反归一化
        final_out = self.revin_layer(final_out, 'denorm')

        return final_out