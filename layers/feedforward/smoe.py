# 导入 PyTorch 库
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import numpy as np

# 稀疏调度器：用于将输入分发给多个专家，并将其输出加权合并
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """初始化稀疏调度器"""

        self._gates = gates  # gates 张量表示每个样本与各专家的权重
        self._num_experts = num_experts

        # 找出 gates 中非零元素的位置（即需要被分配的样本）
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # 提取专家索引（列）
        _, self._expert_index = sorted_experts.split(1, dim=1)

        # 提取每个专家所对应的样本索引（行）
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        # 每个专家获得的样本数量
        self._part_sizes = (gates > 0).sum(0).tolist()

        # 根据 batch_index 取出对应的 gate 值（用于合并输出时加权）
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """将输入张量分发给各专家
        Args:
          inp: 输入张量，形状为 [batch_size, input_dim]
        Returns:
          每个专家对应的输入张量列表
        """
        # 按照 batch_index 提取输入数据
        inp_exp = inp[self._batch_index].squeeze(1)

        # 根据每个专家处理样本数量进行切分
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """合并各个专家的输出，按 gates 权重加权求和
        Args:
          expert_out: 专家输出列表，每个为 [expert_batch_size_i, output_dim]
          multiply_by_gates: 是否乘以 gate 权重
        Returns:
          合并后的输出张量，形状为 [batch_size, output_dim]
        """
        # 拼接所有专家输出
        stitched = torch.cat(expert_out, 0)

        # 若开启加权，则乘以非零 gates 值
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        # 初始化全零张量用于累加合并结果
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)

        # 将专家输出按照样本索引相加合并
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """返回每个专家对应的非零 gate 值列表"""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

# 定义一个两层的前馈神经网络（MLP）
class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

# 混合专家网络模块（Mixture of Experts）
class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, noisy_gating=True, num_k=4, loss_coef=1e-3):
        super(SparseMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_k = num_k  # 每个样本最多激活 num_k 个专家
        self.loss_coef = loss_coef

        # 初始化专家网络
        self.experts = nn.ModuleList([MLP(self.d_model, self.d_ff) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)

        # self.w_gate = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=False), nn.ReLU(),
        #                                       nn.Linear(self.d_ff, num_experts, bias=False))
        # self.w_noise = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=False), nn.ReLU(),
        #                                       nn.Linear(self.d_ff, num_experts, bias=False))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.num_k <= self.num_experts)

    def cv_squared(self, x):
        """计算变异系数的平方，用于衡量负载均衡
        Args:
          x: 输入张量
        Returns:
          标量张量
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """计算每个专家的实际负载（处理样本数）"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """计算 clean_values 中每个值出现在 top-num_k 的概率"""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)

        # 展平后根据位置索引获取阈值
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.num_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)

        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        # 标准正态分布下计算概率
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """实现带噪声的 top-num_k gate 选择
        Returns:
          gates: 每个样本对应的专家权重
          load: 每个专家的负载
        """
        # 计算 gating logits

        # if len(x.shape) == 3:
        #     x = rearrange(x, 'b l n -> (b n) l 1')
        #     x = self.start_linear(x).squeeze(-1)

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # top-num_k + 1 选择，便于计算概率
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.num_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.num_k]
        top_k_indices = top_indices[:, :self.num_k]

        # 归一化 top-num_k logits 作为 gates
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        # 将 gates 按照 top-num_k 位置 scatter 回原始矩阵
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 计算每个专家的负载
        if self.noisy_gating and self.num_k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        """MoE 前向传播过程
        Returns:
          y: MoE 输出
          loss: 负载均衡损失项
        """
        # 获取 gates 和负载
        gates, load = self.noisy_top_k_gating(x, self.training)

        # 计算重要性和负载的变异系数损失
        importance = gates.sum(0)
        # print(gates)
        # print(gates.shape, importance.shape, load.shape)
        # exit()
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        # 构建 dispatcher 并分发输入
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        # 每个专家分别处理输入
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        # 合并各专家输出
        y = dispatcher.combine(expert_outputs)
        return y, loss

if __name__ == '__main__':
    inputs = torch.randn(32, 50)
    # d_model, output_size, num_experts, d_ff, noisy_gating=True, num_k=4
    expert = SparseMoE(50, 50, 8, True, 3, 0.001)
    output = expert(inputs)
    print(output.shape)