# coding : utf-8
# Author : yuxiang Zeng
import torch

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
        return MoE(d_model=d_model, d_ff=d_model, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2,
                   loss_coef=0.001)
    elif method == 'smoe':
        return SparseMoE(d_model=d_model, d_ff=d_model, num_experts=8, noisy_gating=True, num_k=2, loss_coef=0.001)
    return None


def get_att(d_model, num_heads, method):
    if method == 'self':
        return Attention(d_model, num_heads, dropout=0.10)
    elif method == 'external':
        return ExternalAttention(d_model, S=d_model * 2)
    elif method == 'mla':
        return MLA(d_model, S=d_model * 2)
    elif method == 'gqa':
        return GroupQueryAttention(d_model, S=d_model * 2)
    elif method == 'mqa':
        return MultiQueryAttentionBatched(d_model, S=d_model * 2)
    return None


class Transformer2(torch.nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, seq_len, pred_len, norm_method='layer',
                 ffn_method='ffn', att_method='self'):
        super().__init__()
        self.pred_len = pred_len
        self.input_projection = torch.nn.Linear(input_size, d_model)
        self.input_seq_projection = torch.nn.Linear(seq_len, seq_len + pred_len)
        self.seq_to_pred_Linear = torch.nn.Linear(seq_len, pred_len)  # FFN层后接Linear
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
        self.norm = get_norm(d_model, norm_method)
        self.output_projection = torch.nn.Linear(d_model, input_size)  # 特征还原原始维度

    def forward(self, x, x_mark=None):
        # [B, L, d_model]
        x = self.input_projection(x)
        # [B, d_model, L] -> [B, d_model, seq + pred]
        x = self.input_seq_projection(x.permute(0, 2, 1))
        # [B, seq + pred, d_model]
        x = x.permute(0, 2, 1)

        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        x = self.output_projection(self.norm(x))
        x = x[:, -self.pred_len:, :]
        return x
