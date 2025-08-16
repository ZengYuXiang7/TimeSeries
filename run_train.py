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
    # === 构建 exper_detail 字典（基础字段）===
    detail_fields = {
        'Dataset': config.dataset,
        'Model': config.model,
        'Density': f"{config.density:.3f}",
        'd_model': config.d_model,
        'Rank': config.rank,
    }

    # === 动态添加字段（只有在 config 中存在才加入）===
    optional_fields = ['seq_len', 'pred_len', 'constraint', 'noise_steps', 'noise_scale', 'lamda', 'idx']
    for field in optional_fields:
        if hasattr(config, field):
            key = field.replace('_', ' ').title().replace(' ', '_')  # e.g. seq_len -> Seq_Len
            detail_fields[key] = getattr(config, field)

    # === 构建字符串 ===
    exper_detail = ', '.join(f"{k} : {v}" for k, v in detail_fields.items())
    filename = '_'.join(f"{k.replace('_', '')}{v}" for k, v in detail_fields.items())
    return filename, exper_detail


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
    filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(filename, config)
    log = Logger(filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_config import get_config
    # config = get_config('FinancialConfig')
    config = get_config('TransformerConfig')
    # config = get_config('Transformer2Config')
    run(config)
