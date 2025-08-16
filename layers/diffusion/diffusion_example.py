# 这是一个使用条件扩散模型（Classifier-Free Guidance）在MNIST数据集上进行图像生成的完整训练与采样脚本

# 导入标准库
import os
from typing import Dict, Tuple
from tqdm import tqdm

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入图像处理和可视化工具
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# ============================ 模型模块 ============================

# 残差卷积块，类似ResNet的结构
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414  # 缩放因子防止爆炸
        else:
            return self.conv2(self.conv1(x))

# 下采样模块
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, x):
        return self.model(x)

# 上采样模块
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

# 嵌入向量生成模块，用于时间和条件嵌入
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))

# 带条件输入的Unet结构，结合时间步与标签条件信息
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # 生成条件 one-hot
        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        context_mask = context_mask[:, None].repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # 将 0/1 翻转并赋符号
        c = c * context_mask

        # 条件嵌入和时间嵌入
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        return self.out(torch.cat((up3, x), 1))

# 计算扩散相关的时间步参数表
# 包括beta序列、alpha、sqrt(alpha_bar)等
# 用于训练与反向采样过程

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

# DDPM 训练/采样核心类，融合了 ContextUnet 与 CFG 技术
class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0, 10).to(device).repeat(n_sample // 10)
        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.
        x_i_store = []

        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device).repeat(n_sample, 1, 1, 1)
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1, eps2 = eps[:n_sample], eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        return x_i, np.array(x_i_store)

# ============================ 训练函数 ============================

def train_mnist():
    n_epoch = 20
    batch_size = 256
    n_T = 400
    device = "cuda:0"
    n_classes = 10
    n_feat = 128
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs10/'
    os.makedirs(save_dir, exist_ok=True)
    ws_test = [0.0, 0.5, 2.0]  # CFG 权重

    ddpm = DDPM(ContextUnet(1, n_feat, n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        ddpm.train()
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x, c = x.to(device), c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # 采样和可视化
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            for w in ws_test:
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(n_sample // n_classes):
                        idx = (c == k).nonzero(as_tuple=True)[0][j] if (c == k).any() else 0
                        x_real[k + j * n_classes] = x[idx]
                x_all = torch.cat([x_gen, x_real])
                save_image(make_grid(x_all * -1 + 1, nrow=10), f"{save_dir}image_ep{ep}_w{w}.png")

                if ep % 5 == 0 or ep == n_epoch - 1:
                    fig, axs = plt.subplots(n_sample // n_classes, n_classes, sharex=True, sharey=True, figsize=(8, 3))

                    def animate_diff(i, x_gen_store):
                        for row in range(n_sample // n_classes):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].imshow(-x_gen_store[i, row * n_classes + col, 0], cmap='gray',
                                                     vmin=-x_gen_store[i].min(), vmax=-x_gen_store[i].max())
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False,
                                        repeat=True, frames=x_gen_store.shape[0])
                    ani.save(f"{save_dir}gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))

        if save_model and ep == n_epoch - 1:
            torch.save(ddpm.state_dict(), f"{save_dir}model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()
