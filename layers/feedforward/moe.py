# coding : utf-8
# Author : yuxiang Zeng
import torch

class Expert(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(Expert, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)

class Gating(torch.nn.Module):
    def __init__(self, d_model, num_router_experts, num_k, loss_coef):
        super(Gating, self).__init__()
        self.num_k = num_k
        self.loss_coef = loss_coef
        self.gates = torch.nn.Linear(d_model, num_router_experts)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.gates(x))
        topk_values, topk_indices = torch.topk(weights, self.num_k, dim=-1)  # 取 top-k
        gated_output = torch.zeros_like(weights).scatter_(-1, topk_indices, topk_values) # 取前k个权重
        return gated_output.permute(2, 0, 1), weights.permute(2, 0, 1)  # [num_expets, bs, seq_len]


# DeepSeek MoE ：细粒度专家 + 共享专家
class MoE(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_m, num_router_experts, num_share_experts, num_k, loss_coef):
        super(MoE, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_m = num_m
        self.num_router_experts = num_router_experts
        self.num_share_experts = num_share_experts
        self.num_k = num_k
        self.loss_coef = loss_coef

        self.shared_experts = torch.nn.ModuleList(
            Expert(self.d_model, self.d_ff) for _ in range(self.num_share_experts)
        )

        # N' = mN - ks
        self.num_router_experts = num_router_experts * num_m - self.num_share_experts
        self.router_experts = torch.nn.ModuleList(
            Expert(self.d_model, self.d_ff // num_m) for _ in range(self.num_router_experts)
        )
        # K' = mk - ks
        self.router_num_k = num_m * num_k - num_share_experts
        self.router_gates = Gating(self.d_model, self.num_router_experts, self.router_num_k, self.loss_coef)

    def forward(self, x):
        x = self.__checkinput(x)
        bs, seq_len, d_model = x.shape

        if self.num_share_experts > 0:
            shared_enc = sum(self.shared_experts[i](x) for i in range(self.num_share_experts))
        else:
            shared_enc = torch.zeros_like(x)

        router_enc = []
        router_weights, raw_weights = self.router_gates(x)
        self.aux_loss = self.get_aux_loss(router_weights, raw_weights)

        for i in range(self.num_router_experts):
            now_out = self.router_experts[i](x)
            now_weights = router_weights[i].unsqueeze(-1).expand(-1, -1, d_model)  # 变成 [batch_size, seq_len, 1]，广播成系数 [batch_size, seq_len, d_model]
            now_enc = now_out * now_weights
            router_enc.append(now_enc)
        router_enc = torch.stack(router_enc).sum(0)

        final_enc = shared_enc + router_enc + x
        final_enc = self.__checkoutput(final_enc)

        return final_enc

    def get_aux_loss(self, router_weights, raw_weights):
        # shape = [num_experts, bs, seq]
        N_prime = self.num_router_experts  # N' = mN - Ks
        K_prime = self.router_num_k  # K' = mK - Ks
        T = raw_weights.shape[-1]  # Token length
        p = torch.sum(raw_weights, dim=-1) / T                          # \sum_i^T [num_experts, bs]
        f = N_prime / (K_prime * T) * torch.sum(router_weights, dim=-1) # \sum_i^T  [num_experts, bs]
        aux_loss = torch.sum(self.loss_coef * f * p)
        return aux_loss

    def __checkinput(self, x):
        if len(x.shape) == 2:
            self.flag = 1
            return x.unsqueeze(1)
        elif len(x.shape) == 3:
            self.flag = 2
            return x
        elif len(x.shape) == 4:
            self.flag = 3
            self.bs, self.seq_len, self.channels, self.dim = x.shape
            return x.reshape(x.shape[0], x.shape[1], -1)
        else:
            raise ValueError

    def __checkoutput(self, x):
        if self.flag == 1:
            return x.squeeze(1)
        elif self.flag == 2:
            return x
        elif self.flag == 3:
            return x.reshape(self.bs, self.seq_len, self.channels, self.dim)
        else:
            raise ValueError

if __name__ == '__main__':
    inputs = torch.randn(1, 2, 50)
    expert = MoE(50, 50, 1, 8, 1, 3, 0.01)
    output = expert(inputs)
    print(output.size())

    inputs = torch.randn(1, 2, 50)
    expert = MoE(50, 50, 1, 8, 0, 3, 0.01)
    output = expert(inputs)
    print(output.size())

    inputs = torch.randn(1, 50)
    expert = MoE(50, 50, 1, 8, 1, 3, 0.01)
    output = expert(inputs)
    print(output.size())