# coding : utf-8
# Author : yuxiang Zeng
import math
import torch

from layers.att.external_attention import ExternalAttention
from layers.att.groupquery_attention import GroupQueryAttention
from layers.att.multilatent_attention import MLA
from layers.att.multiquery_attention import MultiQueryAttentionBatched
from layers.feedforward.ffn import FeedForward
from layers.feedforward.moe import MoE
from layers.feedforward.smoe import SparseMoE
from einops import rearrange

from layers.revin import RevIN


class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.10):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

    def forward(self, x, attn_mask=None, weight=False):
        out, weights = self.att(x, x, x, attn_mask=attn_mask)
        return (out, weights) if weight else out


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

class TokenEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
    
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FixedEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = torch.nn.Embedding(c_in, d_model)
        self.emb.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    

class TemporalEmbedding(torch.nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else torch.nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            # self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        #  + hour_x + minute_x
        return month_x + day_x + weekday_x + hour_x
    
    

class TimeFeatureEmbedding(torch.nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = torch.nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
    
class DataEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model, match_mode, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.match_mode = match_mode

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = torch.nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x_out = self.value_embedding(x) + self.position_embedding(x)
        else:
            # a 
            # a b
            # a c
            # a b c
            if self.match_mode == 'a':
                x_out = self.value_embedding(x)
            elif self.match_mode == 'ab':
                x_out = self.value_embedding(x) + self.position_embedding(x)
            elif self.match_mode == 'ac':
                x_out = self.value_embedding(x) + self.temporal_embedding(x_mark)
            elif self.match_mode == 'abc':
                x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
            else:
                raise ValueError(f"Unknown ablation mode: {self.match_mode}")

        return self.dropout(x_out)


def generate_causal_window_mask(seq_len, win_size, device):
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)  # 上三角屏蔽未来
    for i in range(seq_len):
        left = max(0, i - win_size + 1)
        mask[i, :left] = True  # 屏蔽太早的历史
    return mask.masked_fill(mask, float('-inf'))  #  保持为 (T, T)

# class Transformer(torch.nn.Module):
#     def __init__(self, input_size, d_model, revin, num_heads, num_layers,
#                  seq_len, pred_len, match_mode, win_size, patch_len, device,
#                  norm_method='layer', ffn_method='ffn', att_method='self',):
#         super().__init__()
#         self.revin = revin
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.match_mode = match_mode
#         self.win_size = win_size
#         self.device = device
#         self.total_len = seq_len + pred_len
#         self.patch_len = patch_len
#         assert self.total_len % patch_len == 0, "total_len must be divisible by patch_len"
#         self.num_patches = self.total_len // patch_len

#         if self.revin:
#             self.revin_layer = RevIN(num_features=input_size, affine=False, subtract_last=False)

#         self.enc_embedding = DataEmbedding(input_size, d_model, self.match_mode)

#         # 1. 映射到 seq_len + pred_len 长度
#         self.predict_linear = torch.nn.Linear(seq_len, self.total_len)

#         # 2. 为每个 patch 分配一个独立的 Transformer block
#         self.patch_transformers = torch.nn.ModuleList([
#             torch.nn.Sequential(*[
#                 torch.nn.Sequential(
#                     get_norm(d_model, norm_method),
#                     get_att(d_model, num_heads, att_method),
#                     get_norm(d_model, norm_method),
#                     get_ffn(d_model, ffn_method)
#                 ) for _ in range(num_layers)
#             ]) for _ in range(self.num_patches)
#         ])


#         # self.patch_inter_transformer = torch.nn.Sequential(*[
#         #     torch.nn.Sequential(
#         #         get_norm(d_model, norm_method),
#         #         get_att(d_model, num_heads, att_method),
#         #         get_norm(d_model, norm_method),
#         #         get_ffn(d_model, ffn_method)
#         #     ) for _ in range(num_layers)
#         # ])


#         self.norm = get_norm(d_model, norm_method)
        
#         self.projection = torch.nn.Linear(d_model, input_size)


#     def forward(self, x, x_mark=None):
#         if self.revin:
#             means = x.mean(1, keepdim=True).detach()
#             x = x - means
#             stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x /= stdev

#         # [B, L, C] → [B, L, d_model]
#         x = self.enc_embedding(x, x_mark)

#         # [B, L, d_model] → [B, d_model, L] → project to total_len
#         x = rearrange(x, 'b l d -> b d l')  # for predict_linear
#         x = self.predict_linear(x)         # [B, d_model, total_len]
#         x = rearrange(x, 'b d l -> b l d')  # back to [B, total_len, d_model]

#         # Split into patches: [B, total_len, d_model] → [B, num_patches, patch_len, d_model]
#         x = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)

#         # Pass each patch into its own Transformer block
#         outs = []
#         for i in range(self.num_patches):
#             patch = x[:, i]  # [B, patch_len, d_model]
#             out = patch
#             for block in self.patch_transformers[i]:
#                 norm1, attn, norm2, ff = block
#                 out = attn(norm1(out)) + out
#                 out = ff(norm2(out)) + out
#             out = self.norm(out)  # optional
#             outs.append(out)

#         # Patch 内处理后拼接
#         x = torch.cat(outs, dim=1)  # [B, total_len, d_model]

#         # # Patch 表征 pooling + patch 间 Transformer
#         # x_patch_tokens = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)
#         # x_patch_tokens = x_patch_tokens.mean(dim=2)  # [B, num_patches, d_model]
#         # x_patch_tokens = self.patch_inter_transformer(x_patch_tokens)  # [B, num_patches, d_model]

#         # # 融合 patch 间上下文
#         # x_patch_tokens = x_patch_tokens.unsqueeze(2).repeat(1, 1, self.patch_len, 1)  # [B, np, pl, d]
#         # x = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)
#         # x = x + x_patch_tokens  # 加残差
#         # x = rearrange(x, 'b np pl d -> b (np pl) d')  # [B, total_len, d_model]


#         # Project back to input_size
#         y = self.projection(x)  # [B, total_len, input_size]

#         if self.revin:
#             y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)
#             y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)

#         return y[:, -self.pred_len:, :]

class Transformer(torch.nn.Module):
    def __init__(self, input_size, d_model, revin, num_heads, num_layers,
                 seq_len, pred_len, match_mode, win_size, patch_len, device,
                 norm_method='layer', ffn_method='ffn', att_method='self',
                 # === [SeqComp] 新增：序列互补开关与数量 ===
                 use_seqcomp: bool = True,
                 k_seqcomp: int = 3):
        super().__init__()
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.match_mode = match_mode
        self.win_size = win_size
        self.device = device
        self.total_len = seq_len + pred_len
        self.patch_len = patch_len
        assert self.total_len % patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // patch_len

        # === [SeqComp] 保存配置 ===
        self.use_seqcomp = use_seqcomp
        self.k_seqcomp = k_seqcomp

        if self.revin:
            self.revin_layer = RevIN(num_features=input_size, affine=False, subtract_last=False)

        self.enc_embedding = DataEmbedding(input_size, d_model, self.match_mode)

        # 1) 先把序列长度线性映射到 seq_len + pred_len（保持 d_model 维度不变）
        self.predict_linear = torch.nn.Linear(seq_len, self.total_len)

        # 2) 每个 patch 一个独立的 Transformer 堆栈（保持你的设计不变）
        self.patch_transformers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    get_norm(d_model, norm_method),
                    get_att(d_model, num_heads, att_method),
                    get_norm(d_model, norm_method),
                    get_ffn(d_model, ffn_method)
                ) for _ in range(num_layers)
            ]) for _ in range(self.num_patches)
        ])

        self.norm = get_norm(d_model, norm_method)
        self.projection = torch.nn.Linear(d_model, input_size)

        # === [SeqComp] 关键：可学习补充 token（共享于所有 patch；位于 embedding 空间 d_model）===
        if self.use_seqcomp and self.k_seqcomp > 0:
            # 形状 [K, d_model]；初始化小随机值；补充 token 不加位置编码
            self.seqcomp_tokens = torch.nn.Parameter(
                torch.randn(self.k_seqcomp, d_model) * 0.02
            )

    # === [SeqComp] 多样化损失：鼓励 K 个补充 token 彼此正交（log-volume / SVD）===
    def _diversification_loss(self, S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        S: (K, d_model) —— K 个补充 token
        返回：标量，多样化损失（越小越好；训练时 +λ）
        """
        S = torch.nn.functional.normalize(S, p=2, dim=1)  # 行 L2 归一化
        try:
            _, s, _ = torch.linalg.svd(S, full_matrices=False)  # s: 奇异值
        except RuntimeError:
            _, s, _ = torch.linalg.svd(S.cpu(), full_matrices=False)
            s = s.to(S.device)
        # -sum(log(sigma_i+eps)) 等价最大化体积（鼓励“正交/多样”）
        return -(torch.log(s + eps).sum())

    def forward(self, x, x_mark=None):
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # [B, L, C] → [B, L, d_model]
        x = self.enc_embedding(x, x_mark)

        # [B, L, d] → [B, d, L] → 线性到 total_len → 回到 [B, total_len, d]
        x = rearrange(x, 'b l d -> b d l')
        x = self.predict_linear(x)                 # [B, d_model, total_len]
        x = rearrange(x, 'b d l -> b l d')         # [B, total_len, d_model]

        # 切成 patch：[B, total_len, d] → [B, num_patches, patch_len, d]
        x = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)

        # === [SeqComp] 初始化 dcs_loss（不开启则为 0）===
        dcs_loss = x.new_zeros([])

        # 逐 patch 处理（在进入每个 patch 的 Transformer 前拼接 K 个补充 token）
        outs = []
        for i in range(self.num_patches):
            patch = x[:, i]              # [B, patch_len, d_model]
            out = patch                  # 仅原始 patch token（已具备位置/内容嵌入）

            # === [SeqComp] 关键：为该 patch 拼接 K 个补充 token（不加位置编码）===
            if self.use_seqcomp and self.k_seqcomp > 0:
                comp = self.seqcomp_tokens.unsqueeze(0).expand(out.size(0), -1, -1)  # [B, K, d]
                out_in = torch.cat([out, comp], dim=1)  # [B, patch_len + K, d]
            else:
                out_in = out

            # === 在扩展后的长度上做自注意力（原始 token ↔ 补充 token 充分交互）===
            z = out_in
            for block in self.patch_transformers[i]:
                norm1, attn, norm2, ff = block
                z = attn(norm1(z)) + z
                z = ff(norm2(z)) + z

            # === [SeqComp] 关键：仅取回前 patch_len 个原始 token 的输出，丢弃补充 token 的直接输出 ===
            if self.use_seqcomp and self.k_seqcomp > 0:
                z = z[:, :self.patch_len, :]

            z = self.norm(z)  # 可选
            outs.append(z)

        # 拼回时间维度：[B, total_len, d_model]
        x = torch.cat(outs, dim=1)

        # 映射回输入通道
        y = self.projection(x)  # [B, total_len, input_size]

        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)

        # === [SeqComp] 计算并返回多样化损失（仅在开启时）===
        if self.use_seqcomp and self.k_seqcomp > 0:
            dcs_loss = self._diversification_loss(self.seqcomp_tokens)

        # 仅返回预测段；训练时使用：loss = mse + λ * dcs_loss
        return y[:, -self.pred_len:, :], dcs_loss
