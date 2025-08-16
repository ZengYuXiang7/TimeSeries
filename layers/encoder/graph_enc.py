# coding : utf-8
# Author : Yuxiang Zeng
import torch
import dgl
from dgl.nn.pytorch import *
import numpy as np

from layers.feedforward.ffn import FeedForward
from layers.feedforward.moe import MoE
from baselines.Transformer import get_norm


def get_norm(d_model, method):
    if method == 'batch':
        return torch.nn.BatchNorm1d(d_model)
    elif method == 'layer':
        return torch.nn.LayerNorm(d_model)
    elif method == 'rms':
        return torch.nn.RMSNorm(d_model)


def get_ffn(d_model, method):
    if method == 'ffn':
        return FeedForward(d_model, d_ff=d_model * 2, dropout=0.10)
    elif method == 'moe':
        return MoE(d_model=d_model, d_ff=d_model, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2,
                   loss_coef=0.001)


def get_gcn(d_model, method):
    if method == 'gcn':
        return GraphConv(d_model, d_model)
    elif method == 'graphsage':
        return SAGEConv(d_model, d_model, aggregator_type='gcn')
    elif method == 'gat':
        return GATConv(d_model, d_model, 4, 0)
    elif method == 'gin':
        return GINConv(GINMLP(d_model, d_model), 'sum')


class GnnFamily(torch.nn.Module):
    def __init__(self, d_model, order, gcn_method, norm_method, ffn_method):
        super(GnnFamily, self).__init__()
        self.order = order
        self.gcn_method = gcn_method
        self.layers = torch.nn.ModuleList([])
        for _ in range(order):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        get_norm(d_model, norm_method),
                        get_gcn(d_model, gcn_method),
                        get_norm(d_model, norm_method),
                        get_ffn(d_model, ffn_method)
                    ]
                )
            )

    def forward(self, graph, x):
        graph, x = graph, x
        # print(x.shape)
        for norm1, gcn, norm2, ff in self.layers:
            x = gcn(graph, norm1(x)) if not self.gcn_method == 'gat' else gcn(graph, norm1(x)).mean(dim=1) + x
            x = ff(norm2(x)) + x

        batch_sizes = torch.as_tensor(
            graph.batch_num_nodes()
        ).to(x.device)  # 每个图的节点数

        first_nodes_idx = torch.cumsum(
            torch.cat(
                [torch.tensor([0]).to(x.device), batch_sizes[:-1]]
            ),
            dim=0
        )  # 使用torch.cat来连接Tensor

        x = x[first_nodes_idx]
        return x


class GINMLP(torch.nn.Module):
    def __init__(self, in_feats, out_feats=64, layer_nums=3):
        super(GINMLP, self).__init__()
        self.linear_layers = torch.nn.ModuleList()
        for each in range(layer_nums):
            if each == 0:
                in_features = in_feats
            else:
                in_features = out_feats
            self.linear_layers.append(torch.nn.Linear(in_features=in_features, out_features=out_feats))
        self.activate = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm1d(out_feats)
        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, x):
        x1 = x
        for mod in self.linear_layers:
            x1 = mod(x1)
            x1 = self.activate(x1)
        x2 = self.batchnorm(x1)
        x3 = self.dropout(x2)
        return x3


if __name__ == '__main__':
    # 设置超参数
    num_nodes = 5  # 5 个节点
    num_edges = 6  # 假设有 6 条边
    rank = 64  # 特征维度
    max_flow_length = 5  # 假设最大流长度为 5
    num_classes = 5  # 假设分类任务有 2 类

    # 构建节点特征，随机生成节点特征 图
    feats = torch.randn(num_nodes, rank)  # 5 个节点，每个节点有 1 维特征
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 4]])  # 连接节点的边
    src, dst = torch.tensor(edges[:, 0]), torch.tensor(edges[:, 1])
    graph = dgl.graph((src, dst))
    graph.ndata['feats'] = feats  # 将节点特征添加到图中
    # 创建模型实例
    graph_encoder = 'gcn'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(d_model=rank, order=3, gcn_method=graph_encoder, norm_method='layer', ffn_method='ffn')
    output = model(graph, feats)
    print(output)

    graph_encoder = 'gat'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(d_model=rank, order=3, gcn_method=graph_encoder, norm_method='layer', ffn_method='ffn')
    output = model(graph, feats)
    print(output)

    graph_encoder = 'gin'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(d_model=rank, order=3, gcn_method=graph_encoder, norm_method='layer', ffn_method='ffn')
    output = model(graph, feats)
    print(output)