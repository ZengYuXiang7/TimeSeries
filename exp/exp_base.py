# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import contextlib
import torch
from time import time

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
from torch.cuda.amp import autocast

class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)  # ✅ 初始化 GradScaler

    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if config.classification else 'max', factor=0.5,
                                                                    patience=config.patience // 1.5, threshold=0.0)

    # def train_one_epoch(self, dataModule):
    #     loss = None
    #     self.train()
    #     torch.set_grad_enabled(True)
    #     t1 = time()

    #     for train_batch in dataModule.train_loader:
    #         all_item = [item.to(self.config.device) for item in train_batch]
    #         inputs, label = all_item[:-1], all_item[-1]

    #         self.optimizer.zero_grad()

    #         if self.config.use_amp:
    #             with torch.amp.autocast(device_type=self.config.device):
    #                 pred = self.forward(*inputs)
    #                 loss = compute_loss(self, inputs, pred, label, self.config)

    #             self.scaler.scale(loss).backward()
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         else:
    #             pred = self.forward(*inputs)
    #             loss = compute_loss(self, inputs, pred, label, self.config)
    #             loss.backward()
    #             self.optimizer.step()

    #     t2 = time()
    #     self.eval()
    #     torch.set_grad_enabled(False)
    #     return loss, t2 - t1

    def train_one_epoch(self, dataModule):
        last_total_loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time()

        lambda_dcs = getattr(self.config, "lambda_dcs", 0.1)  # 可在配置里改

        for train_batch in dataModule.train_loader:
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]

            self.optimizer.zero_grad()

            if getattr(self.config, "use_amp", False):
                # ✅ 注意：autocast 的 device_type 通常应为 'cuda' 或 'cpu'，
                # 若 self.config.device 是 'cuda:0' 之类，需传 'cuda'
                dev_type = "cuda" if "cuda" in str(self.config.device) else "cpu"
                with torch.amp.autocast(device_type=dev_type):
                    outputs = self.forward(*inputs)

                    if isinstance(outputs, tuple):
                        y_pred, dcs_loss = outputs
                    else:
                        y_pred, dcs_loss = outputs, 0.0

                    task_loss = compute_loss(self, inputs, y_pred, label, self.config)
                    total_loss = task_loss + lambda_dcs * dcs_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.forward(*inputs)

                if isinstance(outputs, tuple):
                    y_pred, dcs_loss = outputs
                else:
                    y_pred, dcs_loss = outputs, 0.0

                task_loss = compute_loss(self, inputs, y_pred, label, self.config)
                total_loss = task_loss + lambda_dcs * dcs_loss

                total_loss.backward()
                self.optimizer.step()

            last_total_loss = total_loss.detach()

        t2 = time()
        self.eval()
        torch.set_grad_enabled(False)
        return last_total_loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        self.eval()
        torch.set_grad_enabled(False)
        dataloader = (
            dataModule.valid_loader
            if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0
            else dataModule.test_loader
        )
        preds, reals, val_loss = [], [], 0.

        context = (
            torch.amp.autocast(device_type=self.config.device)
            if self.config.use_amp else
            contextlib.nullcontext()
        )

        with context:
            for batch in dataloader:
                all_item = [item.to(self.config.device) for item in batch]
                inputs, label = all_item[:-1], all_item[-1]

                outputs = self.forward(*inputs)
                # ✅ 解包：兼容 tuple / tensor
                if isinstance(outputs, tuple):
                    pred, _ = outputs   # 预测张量, dcs_loss（评估时不用）
                else:
                    pred = outputs

                if mode == 'valid':
                    val_loss += compute_loss(self, inputs, pred, label, self.config)

                if self.config.classification:
                    pred = torch.max(pred, 1)[1]

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        if self.config.dataset != 'weather':
            reals, preds = (
                dataModule.y_scaler.inverse_transform(reals),
                dataModule.y_scaler.inverse_transform(preds)
            )

        if mode == 'valid':
            self.scheduler.step(val_loss)

        return ErrorMetrics(reals, preds, self.config)


    # def evaluate_one_epoch(self, dataModule, mode='valid'):
    #     self.eval()
    #     torch.set_grad_enabled(False)
    #     dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
    #     preds, reals, val_loss = [], [], 0.

    #     context = (
    #         torch.amp.autocast(device_type=self.config.device)
    #         if self.config.use_amp else
    #         contextlib.nullcontext()
    #     )

    #     with context:
    #         for batch in dataloader:
    #             all_item = [item.to(self.config.device) for item in batch]
    #             inputs, label = all_item[:-1], all_item[-1]
    #             pred = self.forward(*inputs)

    #             if mode == 'valid':
    #                 val_loss += compute_loss(self, inputs, pred, label, self.config)

    #             if self.config.classification:
    #                 pred = torch.max(pred, 1)[1]

    #             reals.append(label)
    #             preds.append(pred)

    #     reals = torch.cat(reals, dim=0)
    #     preds = torch.cat(preds, dim=0)

    #     if self.config.dataset != 'weather':
    #         reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)

    #     if mode == 'valid':
    #         self.scheduler.step(val_loss)

    #     return ErrorMetrics(reals, preds, self.config)