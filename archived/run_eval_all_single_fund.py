# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
import pickle

from data_provider.data_loader import DataModule
from exp.exp_model import Model
from run_train import get_experiment_name
from exp.exp_metrics import ErrorMetrics
from utils.utils import set_seed
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
torch.set_default_dtype(torch.float32)


def RunOnce(config, runId, log):
    set_seed(config.seed + runId)
    datamodule = DataModule(config)
    model = Model(datamodule, config)
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
    sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.setup_optimizer(config)
    results = model.evaluate_one_epoch(datamodule, 'test')
    if not config.classification:
        log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} time={sum_time:.1f} s ')
    else:
        log(f'Ac={results["AC"]:.4f} Pr={results["PR"]:.4f} Rc={results["RC"]:.4f} F1={results["F1"]:.4f} time={sum_time:.1f} s ')
    results['train_time'] = sum_time
    config.record = False
    mode = 'test'
    model.setup_optimizer(config)
    model.eval()
    torch.set_grad_enabled(False)
    dataloader = datamodule.valid_loader if mode == 'valid' and len(
        datamodule.valid_loader.dataset) != 0 else datamodule.test_loader
    preds, reals = [], []
    for batch in (dataloader):
        all_item = [item.to(model.config.device) for item in batch]
        inputs, label = all_item[:-1], all_item[-1]
        pred = model.forward(*inputs)
        if model.config.classification:
            pred = torch.max(pred, 1)[1]
        reals.append(label)
        preds.append(pred)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)
    reals, preds = datamodule.scaler.inverse_transform(reals), datamodule.scaler.inverse_transform(preds)
    return reals, preds


def run(config):
    all_dataset = os.listdir('./checkpoints/ours')
    r64_models = [name for name in all_dataset if '_R64_' in name]
    preds, reals = [], []
    for idx in range(len(r64_models)):
        config.idx = idx
        set_settings(config)
        log_filename, exper_detail = get_experiment_name(config)
        plotter = MetricsPlotter(log_filename, config)
        log = Logger(log_filename, exper_detail, plotter, config, False)
        real, pred = RunOnce(config, 0, log)
        preds.append(pred)
        reals.append(real)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)
    results = ErrorMetrics(reals, preds, config)
    print(preds.shape, reals.shape)
    print(results)
    log.save_in_log(results)
    return results


if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config()
    run(config)
