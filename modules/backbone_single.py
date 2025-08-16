# coding : utf-8
# Author : Yuxiang Zeng
import torch
import math 
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.revin import RevIN
from layers.transformer import Transformer
import einops


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
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x_out = self.value_embedding(x) + self.position_embedding(x)
        else:
            # x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
            x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x_out)
    

class SingleModel(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(SingleModel, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)


        self.projection = DataEmbedding(enc_in, self.d_model)
        # self.projection = torch.nn.Linear(enc_in, config.d_model, bias=True)
        self.position_embedding = PositionEncoding(d_model=self.d_model, max_len=config.seq_len, method='bert')
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.encoder1 = Transformer(
            config.seq_len + config.pred_len,
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


    def forward(self, x, x_mark, x_fund):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x_enc = self.projection(x)
        # x_enc += self.position_embedding(x_enc)

        # [bs, seq_len, dim]
        x_enc = einops.rearrange(x_enc, 'bs seq_len dim -> bs dim seq_len')
        x_enc = self.predict_linear(x_enc)

        # [bs, dim, seq_len + pred_len]
        x_enc = self.encoder1(x_enc)
        x_enc = einops.rearrange(x_enc, 'bs dim seq_pred_len -> bs seq_pred_len dim')
        x_enc = self.encoder2(x_enc)

        # [bs, seq_len + pred_len, 3]
        x_enc = self.decoder(x_enc)

        # [bs, pred_len, 3]
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)
        
        if self.revin:
            y = self.revin_layer(y, 'denorm')

        return y
