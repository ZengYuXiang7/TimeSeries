import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(WordEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class BertEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(BertEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_len+1, embedding_dim=d_model)

    def forward(self, x):
        pos_idx = torch.arange(0, x.size(1))
        pos_idx = pos_idx.view(1, -1)
        pos_idx= pos_idx.repeat(x.size(0), 1).long()
        # print(pos_idx)
        pos_idx = pos_idx.to(x.device)
        return self.embedding(pos_idx)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, method='bert'):
        super(PositionEncoding, self).__init__()
        if method == 'bert':
            self.pos_encoding = BertEmbedding(d_model, max_len)
        elif method == 'word':
            self.pos_encoding = WordEmbedding(d_model, max_len)

    def forward(self, x):
        return self.pos_encoding(x)  # 修正错误

# 测试代码
if __name__ == "__main__":
    batch_size, seq_len, d_model = 5, 30, 16
    # 1. 测试 WordEmbedding 方法
    pos_enc_word = PositionEncoding(d_model=d_model, max_len=seq_len, method='word')
    pos_input_word = torch.randn(batch_size, seq_len, d_model)
    out_word = pos_input_word + pos_enc_word(pos_input_word)
    print("WordEmbedding Positional Encoding Output Shape:", out_word.shape)

    # 2. 测试 BertEmbedding 方法
    pos_enc_bert = PositionEncoding(d_model=d_model, max_len=seq_len, method='bert')
    pos_input_word = torch.randn(batch_size, seq_len, d_model)
    out_bert = pos_input_word + pos_enc_bert(pos_input_word)  # 直接输出位置编码
    print("BertEmbedding Positional Encoding Output Shape:", out_bert.shape)