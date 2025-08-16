# coding : utf-8
# Author : Yuxiang Zeng
import torch
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.revin import RevIN
from layers.att.external_attention import ExternalAttention
from layers.att.groupquery_attention import GroupQueryAttention
from layers.att.multilatent_attention import MLA
from layers.att.multiquery_attention import MultiQueryAttentionBatched
from layers.feedforward.ffn import FeedForward
from layers.feedforward.moe import MoE
from layers.att.self_attention import Attention
from layers.feedforward.smoe import SparseMoE
from einops import rearrange

def get_norm(d_model, method):
    if method == 'batch':
        return torch.nn.BatchNorm1d(d_model)
    elif method == 'layer':
        return torch.nn.LayerNorm(d_model)
    elif method == 'rms':
        return torch.nn.RMSNorm(d_model)
    return None


def get_ffn(d_model, method):
    if method == 'ffn':
        return FeedForward(d_model, d_ff=d_model * 2, dropout=0.10)
    elif method == 'moe':
        return MoE(d_model=d_model, d_ff=d_model, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2, loss_coef=0.001)
    elif method == 'smoe':
        return SparseMoE(d_model=d_model, d_ff=d_model, num_experts=8, noisy_gating=True, num_k=2, loss_coef=0.001)
    return None


def get_att(d_model, num_heads, method):
    if method == 'self':
        return Attention(d_model, num_heads, dropout=0.10)
    elif method == 'external':
        return ExternalAttention(d_model, S=d_model*2)
    elif method == 'mla':
        return MLA(d_model, S=d_model*2)
    elif method == 'gqa':
        return GroupQueryAttention(d_model, S=d_model*2)
    elif method == 'mqa':
        return MultiQueryAttentionBatched(d_model, S=d_model*2)
    return None


class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_layers, norm_method='rms', ffn_method='ffn', att_method='self'):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        get_norm(d_model, norm_method),
                        get_att(d_model, num_heads, att_method),
                        get_norm(d_model, norm_method),
                        get_ffn(d_model, ffn_method)
                    ]
                )
            )

    def toeplitz_penalty(self, weights):
        # weights: [bsz * num_heads, tgt_len, src_len]
        shifted = weights[..., 1:, 1:]
        original = weights[..., :-1, :-1]
        return ((shifted - original) ** 2).mean()

    def forward(self, x, x_mark=None):
        toeplitz_loss = 0.0
        for norm1, attn, norm2, ff in self.layers:
            att_out, attn_weights = attn(norm1(x), weight = True)  # 获取 attention 输出和权重
            toeplitz_loss += self.toeplitz_penalty(attn_weights)  # 正则化项
            x = att_out + x
            x = ff(norm2(x)) + x
        return x, toeplitz_loss  # 返回主输出和 Toeplitz 正则项
    


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)

        self.projection = torch.nn.Linear(enc_in, config.d_model, bias=True)
        # self.position_embedding = PositionEncoding(d_model=self.d_model, max_len=config.seq_len, method='bert')
        # self.fund_embedding = torch.nn.Embedding(999999, config.d_model)
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.feature_linear = torch.nn.Linear(7, config.d_model, bias=True)
        self.encoder1 = Transformer(
            self.d_model,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )
        self.encoder2 = Transformer(
            self.d_model,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )
        # self.encoder1 = torch.nn.Linear(config.d_model, config.d_model, bias=True)
        # self.encoder2 = torch.nn.Linear(config.d_model, config.d_model, bias=True)

        self.decoder = torch.nn.Linear(config.d_model, 3)

    def forward(self, x, x_mark, x_fund, x_features):
        if self.fft:
            x = self.seasonality_and_trend_decompose(x)
        
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # x_enc = self.projection(x)
        x_enc = self.projection(x) + self.feature_linear(x_features)
        # x_enc = self.fund_embedding(x_fund)
        x_enc = self.predict_linear(x_enc.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        bs, pred_len, channels, dim = x_enc.shape
        x_enc = x_enc.permute(2, 0, 1, 3).reshape(channels, bs * pred_len, dim)
        # x_enc = self.encoder1(x_enc)
        x_enc, loss1 = self.encoder1(x_enc)
        x_enc = x_enc.reshape(channels, bs, pred_len, dim).permute(1, 2, 0, 3)
        x_enc = x_enc.permute(1, 0, 2, 3).reshape(pred_len, bs * channels, dim)
        # x_enc = self.encoder2(x_enc)
        x_enc, loss2 = self.encoder2(x_enc)
        self.toeplitz_loss = loss1 + loss2
        x_enc = x_enc.reshape(pred_len, bs, channels, dim).permute(1, 0, 2, 3)

        x_enc = self.decoder(x_enc)
        y = x_enc[:, -self.pred_len:, :]

        if self.revin:
            y = y * stdev[:, 0, :, :].unsqueeze(1).repeat(1, self.pred_len, 1, 1)
            y = y + means[:, 0, :, :].unsqueeze(1).repeat(1, self.pred_len, 1, 1)
        return y
