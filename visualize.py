# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from tqdm import *
from modules.attlstm import AttLstm
from modules.lstm import LSTM
from modules.mlp import MLP
from modules.rnn import RNN
from data import experiment, DataModule
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import set_settings, set_seed
global log, args

torch.set_default_dtype(torch.float32)

class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.dimension
        if args.model == 'rnn':
            self.model = RNN(self.input_size, args.dimension, 1, args)

        elif args.model == 'lstm':
            self.model = LSTM(self.input_size, args.dimension, 1, args)

        elif args.model == 'mlp':
            self.model = MLP(self.input_size, args.dimension, 1, args)

        elif args.model == 'attlstm':
            self.model = AttLstm(self.input_size, args.dimension, 1, args)

        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, windows):
        y = self.model.forward(windows)
        return y

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if args.classification else 'max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader):
            windows, value = train_Batch
            windows, value = windows.to(self.args.device), value.to(self.args.device)
            pred = self.forward(windows)
            # print(pred.shape, value.shape)
            loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        val_loss = 0.
        preds = []
        reals = []
        dataloader = dataModule.valid_loader if mode == 'valid' else dataModule.test_loader
        for batch in tqdm(dataloader):
            windows, value = batch
            windows, value = windows.to(self.args.device), value.to(self.args.device)
            pred = self.forward(windows)
            if mode == 'valid':
                val_loss += self.loss_function(pred, value)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        if mode == 'valid':
            self.scheduler.step(val_loss)
        metrics_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return metrics_error


def visualize_predictions(reals, preds, args):
    plt.figure(dpi=700)
    plt.plot(reals, label='Actual Values')
    plt.plot(preds, label='Predicted Values')
    plt.title(f"Predicted vs Actual Values ({args.model})")
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f'./results/predictions_{args.model}.png')
    plt.show()


def visualize(log, args):
    # Set seed
    set_seed(args.seed + 0)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)

    # Load the best model parameters
    model_path = f'./checkpoints/{args.model}_{args.seed}.pt'
    model.load_state_dict(torch.load(model_path))

    preds = []
    reals = []
    for batch in tqdm(datamodule.test_loader):
        windows, value = batch
        windows, value = windows.to(model.args.device), value.to(model.args.device)
        pred = model.forward(windows)
        preds.append(pred)
        reals.append(value)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)

    # Calculate metrics
    metrics_error = ErrorMetrics(reals * datamodule.max_value, preds * datamodule.max_value, model.args)
    log(f'MAE={metrics_error["MAE"]:.4f} RMSE={metrics_error["RMSE"]:.4f} NMAE={metrics_error["NMAE"]:.4f} NRMSE={metrics_error["NRMSE"]:.4f}')

    # Visualize predictions vs actual values
    visualize_predictions(reals.cpu().detach().numpy(), preds.cpu().detach().numpy(), args)


if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Density : {args.density}"
    log_filename = f'w{args.num_windows}_p{args.num_preds}_r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    args.log = log
    log(str(args.__dict__))
    visualize(log, args)
