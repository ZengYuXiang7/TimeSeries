# coding : utf-8
# Author : Yuxiang Zeng

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
    pairwise_linear_similarity,
)

from layers.metric.soft_dtw_cuda import SoftDTW


class PairwiseLoss(nn.Module):
    def __init__(self, method: str = 'cosine', reduction: str = 'mean', **kwargs):
        super().__init__()
        self.method = method.lower()
        self.reduction = reduction
        self.kwargs = kwargs
        self.func = self._get_func()

    def _get_func(self):
        if self.method == 'cosine':
            return lambda x, y: 1 - pairwise_cosine_similarity(x, y)
        elif self.method == 'euclidean':
            return pairwise_euclidean_distance
        elif self.method == 'manhattan':
            return pairwise_manhattan_distance
        elif self.method == 'minkowski':
            return lambda x, y: pairwise_minkowski_distance(x, y, **self.kwargs)
        elif self.method == 'linear':
            return lambda x, y: -pairwise_linear_similarity(x, y)
        elif self.method == 'kl':
            return lambda x, y: F.kl_div(
                torch.log(torch.clamp(y / y.sum(dim=-1, keepdim=True), min=1e-10)),
                torch.clamp(x / x.sum(dim=-1, keepdim=True), min=1e-10),
                reduction='sum'
            )
        elif self.method == 'dtw':
            return SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)
        elif self.method == 'mahalanobis':
            return self._mahalanobis_distance
        elif self.method == 'none':
            return lambda x, y: 0
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _mahalanobis_distance(self, x, y):
        """
        Computes Mahalanobis distance between batched x and y:
        D(x, y) = sqrt((x - y)^T Σ^{-1} (x - y))
        Assumes: x, y shape [B, D]; cov_inv shape [D, D]
        """
        diff = x - y
        cov_inv = self.kwargs.get('cov_inv', None)
        if cov_inv is None:
            # 自动估计协方差逆
            stacked = torch.cat([x, y], dim=0)
            cov = torch.cov(stacked.T)
            cov_inv = torch.linalg.pinv(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))

        # batch-wise Mahalanobis: (diff @ cov_inv) * diff → sum over last dim
        left = torch.matmul(diff, cov_inv)
        dists = torch.sum(left * diff, dim=1).sqrt()
        return dists

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.method != 'dtw':
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)

        if len(x.shape) == 2 and self.method == 'dtw':
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)

        if self.method == 'none':
            return 0.0

        loss = self.func(x, y)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Example for each method
if __name__ == "__main__":
    bs = 32
    dim = 64

    x = torch.randn(bs, 3, dim)
    y = torch.randn(bs, 3, dim)

    print("\n--- Cosine Similarity ---")
    calc_cos = PairwiseLoss(method='cosine')
    print(calc_cos(x, y))
    print("\n--- Euclidean Distance ---")
    calc_euc = PairwiseLoss(method='euclidean')
    print(calc_euc(x, y))

    print("\n--- Manhattan Distance ---")
    calc_man = PairwiseLoss(method='manhattan')
    print(calc_man(x, y))

    print("\n--- Linear Similarity ---")
    calc_lin = PairwiseLoss(method='linear')
    print(calc_lin(x, y))

    print("\n--- Minkowski Distance (p=3) ---")
    calc_mink = PairwiseLoss(method='minkowski', exponent=3)
    print(calc_mink(x, y))

    print("\n--- Kullback-Leibler Divergence ---")
    calc_kl = PairwiseLoss(method='kl')
    print(calc_kl(x, y))

    print("\n--- Mahalanobis Distance ---")
    calc_maha = PairwiseLoss(method='mahalanobis')
    print(calc_maha(x, y))

    print("\n--- Dynamic Time Warping ---")
    calc_dtw = PairwiseLoss(method='dtw')
    print(calc_dtw(x, y))