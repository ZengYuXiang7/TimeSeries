# coding : utf-8
# Author : Yuxiang Zeng
import numpy as np
import torch
import collections
from data_provider.data_loader import DataModule
from exp.exp_train import RunOnce
from exp.exp_model import Model
import utils.model_efficiency
import utils.utils
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

def get_experiment_name(config):
    log_filename = f'Model_{config.model}_Dataset_{config.dataset}_R{config.rank}'
    if config.dataset == 'financial':
        if config.multi_dataset:
            log_filename = f'Model_{config.model}_Dataset_{config.dataset}_Multi_{config.idx}_pred_{config.pred_len}'
        else:
            log_filename = f'Model_{config.model}_Dataset_{config.dataset}_R{config.rank}_F{config.idx}_pred_{config.pred_len}'
    exper_detail = (
         f"Dataset : {config.dataset.upper()}, "
         f"Model : {config.model}, "
         f"Density : {config.density:.3f}, "
         f"d_model : {config.d_model}, "
         f"Seq_len : {config.seq_len}, "
         f"Pred_len : {config.pred_len}, "
    )
    try:
        exper_detail += f"noise_steps : {config.noise_steps}, "
        exper_detail += f"noise_scale : {config.noise_scale}, "
        exper_detail += f"lamda : {config.lamda}"
        exper_detail += f"Idx : {config.idx}, "
    except AttributeError:
        pass

    return log_filename, exper_detail


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(config.rounds):
        utils.utils.set_seed(config.seed + runId)
        datamodule = DataModule(config)
        model = Model(config)
        log.plotter.reset_round()
        results = RunOnce(config, runId, model, datamodule, log)
        for key in results:
            metrics[key].append(results[key])
        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)
    log(f'Train_length : {len(datamodule.train_loader.dataset)} Valid_length : {len(datamodule.valid_loader.dataset)} Test_length : {len(datamodule.test_loader.dataset)}')

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    try:
        flops, params, inference_time = utils.model_efficiency.get_efficiency(datamodule, Model(config), config)
        log(f"Flops: {flops:.0f}")
        log(f"Params: {params:.0f}")
        log(f"Inference time: {inference_time:.2f} ms")
    except Exception as e:
        log('Skip the efficiency calculation')

    log.save_in_log(metrics)

    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    log.end_the_experiment(model)
    return metrics


def run(config):
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    set_settings(config)
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_config import get_config
    # config = get_config('FinancialConfig')
    # config = get_config('TransformerConfig')
    config = get_config('Transformer2Config')
    run(config)
