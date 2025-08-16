# coding : utf-8
# Author : Yuxiang Zeng
import torch 
import torch.nn.functional as F

# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, inputs, pred, label, config):
    loss = model.loss_function(pred, label)
    
    if getattr(config, 'constraint', False):
        loss += model.distance(pred, label) * 1e-4                             # 添加Consine损失
        loss += model.model.toeplitz_loss * 1e-2                               # 添加 Toeplitz 正则项的损失
        # loss += torch.abs(torch.sum(pred) - torch.sum(label)) * 1e-3         # 添加 Sigcomm 数量和约束
        # 构建差分约束，不宜超过历史最大最小涨跌
        hist = inputs[0]                        # [bs, seq_len, channels, 3]
        # Step 1: 历史相邻差分
        hist_diff = hist[:, 1:, :, :] - hist[:, :-1, :, :]     # [bs, seq_len - 1, channels, 3]
        max_hist_gain = hist_diff.max(dim=1).values            # [bs, channels, 3]
        max_hist_drop = hist_diff.min(dim=1).values            # [bs, channels, 3]
        # Step 2: 预测相邻差分
        pred_diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]     # [bs, pred_len - 1, channels, 3]
        # Step 3: 约束惩罚项
        underflow = F.relu(max_hist_drop.unsqueeze(1) - pred_diff)  # 跌太狠
        overflow = F.relu(pred_diff - max_hist_gain.unsqueeze(1))   # 涨太猛
        constraint_penalty = (underflow + overflow).mean()
        loss += constraint_penalty * 1e-3

    # if getattr(config, 'model', 'transformer2'):
    #     # loss = loss * (1 - config.lamda) + config.lamda * model.model.diffusion_loss
    #     loss = loss  + config.lamda * model.model.diffusion_loss
    if getattr(config, 'model', None) == 'transformer2':
        loss = loss + config.lamda * model.model.diffusion_loss
    return loss
