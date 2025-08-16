# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn


class MultiQueryAttentionBatched(nn.Module):
    def __init__(self, d_model, num_heads, k_dim, v_dim):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.k = k_dim
        self.v = v_dim

        # 初始化投影参数
        self.P_q = nn.Parameter(torch.randn(num_heads, d_model, k_dim))  # [h, d, k]
        self.P_k = nn.Parameter(torch.randn(d_model, k_dim))  # [d, k]
        self.P_v = nn.Parameter(torch.randn(d_model, v_dim))  # [d, v]
        self.P_o = nn.Parameter(torch.randn(num_heads, d_model, v_dim))  # [h, d, v]

    def forward(self, X, M, mask):
        b, n, d = X.shape
        m = M.shape[1]

        # 生成查询、键、值
        Q = torch.einsum('bnd,hdk->bhnk', X, self.P_q)  # [b, h, n, k]
        K = torch.einsum('bmd,dk->bmk', M, self.P_k)  # [b, m, k]
        V = torch.einsum('bmd,dv->bmv', M, self.P_v)  # [b, m, v]

        # 计算注意力分数
        logits = torch.einsum('bhnk,bmk->bhnm', Q, K)  # [b, h, n, m]
        logits += mask  # 应用掩码

        # 计算注意力权重
        weights = torch.softmax(logits, dim=-1)  # [b, h, n, m]

        # 计算输出
        O = torch.einsum('bhnm,bmv->bhnv', weights, V)  # [b, h, n, v]
        Y = torch.einsum('bhnv,hdv->bnd', O, self.P_o)  # [b, n, d]

        return Y


class MultiQuerySelfAttentionIncremental(nn.Module):
    def __init__(self, d_model, num_heads, k_dim, v_dim):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.k = k_dim
        self.v = v_dim

        # 初始化投影参数
        self.P_q = nn.Parameter(torch.randn(num_heads, d_model, k_dim))  # [h, d, k]
        self.P_k = nn.Parameter(torch.randn(d_model, k_dim))  # [d, k]
        self.P_v = nn.Parameter(torch.randn(d_model, v_dim))  # [d, v]
        self.P_o = nn.Parameter(torch.randn(num_heads, d_model, v_dim))  # [h, d, v]

    def forward(self, x, prev_K, prev_V):
        b, d = x.shape

        # 生成当前步的查询、键、值
        q = torch.einsum('bd,hdk->bhk', x, self.P_q)  # [b, h, k]
        new_k = torch.einsum('bd,dk->bk', x, self.P_k)  # [b, k]
        new_v = torch.einsum('bd,dv->bv', x, self.P_v)  # [b, v]

        # 拼接记忆
        new_K = torch.cat([prev_K, new_k.unsqueeze(1)], dim=1)  # [b, m+1, k]
        new_V = torch.cat([prev_V, new_v.unsqueeze(1)], dim=1)  # [b, m+1, v]

        # 计算注意力分数
        logits = torch.einsum('bhk,bmk->bhm', q, new_K)  # [b, h, m+1]

        # 计算注意力权重
        weights = torch.softmax(logits, dim=-1)  # [b, h, m+1]

        # 计算输出
        O = torch.einsum('bhm,bmv->bhv', weights, new_V)  # [b, h, v]
        Y = torch.einsum('bhv,hdv->bd', O, self.P_o)  # [b, d]

        return Y,new_K,new_V


# 形状验证
if __name__ == '__main__':
    # 批量处理验证
    d_model = 512
    h = 8
    k = 64
    v = 64
    batch_size = 2 # 批次
    seq_len = 5 # 输入句子长度
    mem_len = 10 # 上下文长度

    # 批量处理测试
    mqa_batched = MultiQueryAttentionBatched(d_model, h, k, v)
    X = torch.randn(batch_size, seq_len, d_model)
    M = torch.randn(batch_size, mem_len, d_model)
    mask = torch.randn(batch_size, h, seq_len, mem_len)
    Y = mqa_batched(X, M, mask)
    print("批处理 output shape:", Y.shape)  # 预期 [b, n, d]
    print("")
    print("----------------------")

    # 增量推理验证
    mqa_incr = MultiQuerySelfAttentionIncremental(d_model, h, k, v)
    x = torch.randn(batch_size, d_model)
    prev_K = torch.randn(batch_size, 0, k)  # 初始空记忆
    prev_V = torch.randn(batch_size, 0, v)


    for step in range(3):
        Y, prev_K, prev_V = mqa_incr(x, prev_K, prev_V)
        print(f"Step {step + 1}:")
        print("增量 Output shape:", Y.shape)  # 预期 [b, n]


