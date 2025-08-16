# coding: utf-8
# Author: mkw
# Date: 2025-06-04 17:40
# Description: Transformer

import torch
import torch.nn as nn
from layers.transformer_library.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.transformer_library.SelfAttention_Family import FullAttention, AttentionLayer
from layers.transformer_library.Embed import DataEmbedding


class TransformerLibrary(nn.Module):

    def __init__(self, configs):
        super(TransformerLibrary, self).__init__()
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding(configs.input_size, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder

        self.dec_embedding = DataEmbedding(configs.input_size, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


    def forward(self, x_enc, x_mark):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(enc_out, enc_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
