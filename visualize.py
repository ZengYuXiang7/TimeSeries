# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from tqdm import *

from data import experiment, DataModule
from train_model import Model
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.plotter import MetricsPlotter
from utils.utils import set_settings, set_seed
from utils.utils import makedir
global log, args

torch.set_default_dtype(torch.float32)



def visualize_predictions(reals, preds, args):
    plt.figure(dpi=500)
    plt.plot(reals, label='Actual Values', alpha=0.6)
    plt.plot(preds, label='Predicted Values', alpha=0.6)
    plt.title(f"Predicted vs Actual Values ({args.model})")
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f'./results/predictions_{args.model}.png')
    plt.show()


if __name__ == '__main__':
    args = get_config()
    set_settings(args)

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Density : {args.density}"
    log_filename = f'w{args.num_windows}_p{args.num_preds}_r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

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

