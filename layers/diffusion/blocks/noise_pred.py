import torch
import torch.nn as nn


def vocab_embedding(vocabs, len_seq, device):
    # 词嵌入
    vocab_to_idx = {"cos": 0, "sin": 1}
    indices = [vocab_to_idx[word] for word in vocabs]
    # 根据索引生成全0或全1的列表
    expanded_indices = [torch.full((len_seq,), idx, dtype=torch.long, device=device) for idx in indices]
    # 将列表转换为tensor
    indices_tensor = torch.stack(expanded_indices)
    return indices_tensor

class NoisePredictor(torch.nn.Module):
    def __init__(self, len_seq, vocab_size, device, drop_out=0.15):
        super().__init__()
        self.len_seq = len_seq
        self.hidden_dim = len_seq
        self.drop_out = drop_out
        self.device = device
        self.norm = nn.LayerNorm(self.len_seq * 3)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.len_seq),
            nn.SiLU(),
            nn.Linear(self.len_seq, self.len_seq)
        )

        self.layer = nn.Sequential(
            nn.Linear(self.len_seq * 3, self.len_seq * 3),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 3, self.len_seq * 3),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 3, self.len_seq * 3)
        )

        self.net = nn.Sequential(
            nn.Linear(self.len_seq * 3, self.len_seq * 3),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 3, self.len_seq * 2),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 2, self.len_seq)
        )

    def forward(self, x, vocab, t):
        t = t.unsqueeze(-1).float()
        time_emb = self.time_mlp(t)
        x = torch.cat([x, vocab, time_emb], dim=1)
        x = self.layer(x) + x
        x = self.norm(x)
        x = self.layer(x) + x
        x = self.norm(x)
        # 预测噪声
        noise_pred = self.net(x)  # [batch_size, seq_length]
        return noise_pred


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, len_seq = 1, 32
    model = NoisePredictor(len_seq, vocab_size=2, device=device).to(device)
    x = torch.randn(batch_size, len_seq, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    vocab = vocab_embedding(['cos'], len_seq, device)
    output = model(x, vocab, t)
    print("Output shape:", output.shape)
