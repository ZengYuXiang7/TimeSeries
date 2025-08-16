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
from layers.att.self_attention import Attention
from layers.feedforward.smoe import SparseMoE
from einops import rearrange
import layers.diffusion.gaussian_diffusion as gd
from layers.diffusion.DNN import DNN
from layers.revin import RevIN


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
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
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
            x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x_out)


class Transformer2(torch.nn.Module):
    def __init__(self, input_size, d_model, revin, num_heads, num_layers, seq_len, pred_len, diffusion=False, noise_scale=1, noise_steps=100,
                 norm_method='layer', ffn_method='ffn', att_method='self'):
        super().__init__()
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        if self.revin:
            self.revin_layer = RevIN(num_features=input_size, affine=False, subtract_last=False)
        self.enc_embedding = DataEmbedding(input_size, d_model)
        self.predict_linear = torch.nn.Linear(seq_len, seq_len + pred_len)
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
        self.projection = torch.nn.Linear(d_model, input_size)
        self.diffusion_loss = 0

        if diffusion:
            self.diffusion = gd.GaussianDiffusion(
                mean_type=gd.ModelMeanType.EPSILON,
                noise_schedule='linear-var',
                noise_scale=noise_scale,   # 何向南的做法就是把这里改成0.01
                noise_min=0.0001,
                noise_max=0.02,
                steps=noise_steps,
                device=torch.device('cuda')
            )
            self.reverse = DNN(in_dims=[input_size * seq_len, input_size * seq_len], out_dims=[input_size * seq_len, input_size * seq_len], emb_size=d_model).to(torch.device('cuda'))
        else:
            self.diffusion = None
            self.reverse = None

    def diffusion_forward(self, y):
        # 在这里加扩散去噪，使得x变成一个新的x
        if self.diffusion is not None:
            if self.training:
                raw_shape = y.shape
                y = y.reshape(y.shape[0], -1)
                diff_output = self.diffusion.training_losses(self.reverse, y, False)
                y = diff_output["pred_xstart"]
                y = y.reshape(raw_shape)
                self.diffusion_loss = diff_output["loss"].mean()
            else:
                raw_shape = y.shape
                y = y.reshape(y.shape[0], -1)
                y = self.diffusion.p_sample(self.reverse, y, 5, False)
                y = y.reshape(raw_shape)
        return y
    
    def forward(self, x, x_mark=None,timesteps=None):
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            
        x = self.diffusion_forward(x)
            
        x = self.enc_embedding(x, x_mark)  # 调整形状为 [B, L, d_model]
        x = rearrange(x, 'bs seq_len d_model -> bs d_model seq_len')
        x = self.predict_linear(x)
        x = rearrange(x, 'bs d_model seq_pred_len -> bs seq_pred_len d_model')
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        x = self.norm(x)
        
        y = self.projection(x)[:, -self.pred_len:, :]
        
        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        
        return y

    
    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt