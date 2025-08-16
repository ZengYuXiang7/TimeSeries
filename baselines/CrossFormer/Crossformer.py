import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from baselines.CrossFormer.cross_encoder import Encoder
from baselines.CrossFormer.cross_decoder import Decoder
from baselines.CrossFormer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from baselines.CrossFormer.cross_embed import DSW_embedding

from math import ceil

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.baseline = baseline
        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        self.fc = nn.Linear(data_dim, 1)


    def forward(self, x_enc, x_mark):
        x_enc = x_enc.unsqueeze(-1)
        batch_size = x_enc.shape[0]
        if self.in_len_add != 0:
            x_enc = torch.cat((x_enc[:, :1, :].expand(-1, self.in_len_add, -1), x_enc), dim = 1)
        x_enc = self.enc_value_embedding(x_enc)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out = self.encoder(x_enc)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        predict_y = predict_y[:, :self.out_len, :]
        predict_y = self.fc(predict_y).reshape(-1, self.out_len)
        return predict_y