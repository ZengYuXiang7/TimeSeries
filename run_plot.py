# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import *
import numpy as np

from data_provider.data_loader import DataModule
from data_provider.data_getitem import TensorDataset
from exp.exp_model import Model
from run_train import get_experiment_name
from utils.utils import set_seed
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
torch.set_default_dtype(torch.float32)


def data_to_dataloader(data_input, label):
    data_set = TensorDataset(data_input, label, 'pred', config)
    bs = 512
    flag = 'test'
    pred_dataloader = DataLoader(
        data_set,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: data_set.custom_collate_fn(batch, config),
    )
    return pred_dataloader, flag

def save_figure(inputs, label, pred, cnt, code_idx, config):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6), dpi=300)
    file_root = f'./figs/{config.model}/constraint_{config.constraint}/{code_idx}'
    os.makedirs(file_root, exist_ok=True)

    input_seq = inputs.reshape(-1)
    real_seq = label.reshape(-1)
    pred_seq = pred.reshape(-1)

    input_len = len(input_seq)
    future_len = len(pred_seq)

    input_time = np.arange(input_len)  # 原始输入时间
    future_time = np.arange(input_len - 1, input_len + future_len)  # 从前一个时间点衔接

    # 构建衔接后的 real/pred 序列：前接一个 input 的最后值
    real_seq_plot = np.concatenate([[input_seq[-1]], real_seq])
    pred_seq_plot = np.concatenate([[input_seq[-1]], pred_seq])

    # 画图
    plt.plot(input_time, input_seq, label='History Adj Nav', linestyle='-.', marker='s', markersize=3)
    plt.plot(future_time, real_seq_plot, label='Future Adj Nav', linestyle='--', marker='o', markersize=3)
    plt.plot(future_time, pred_seq_plot, label='Future Predicted Adj Nav', linestyle='-', marker='x', markersize=3)

    plt.legend()
    plt.title(f'Prediction vs Real - Sample {cnt}')
    plt.xlabel('Time Index')
    plt.ylabel('Value' if not config.classification else 'Class Label')
    plt.grid(True)
    plt.savefig(f'{file_root}/{cnt}.jpg')
    plt.close()
    # print(f"Figure {cnt} has done!")


def predict(model, data_input, label, scaler, config):
    dataloader, flag = data_to_dataloader(data_input, label)
    
    model.setup_optimizer(config)
    cnt = 0
    print(len(dataloader.dataset))
    for batch in (dataloader):
        all_item = [item.to(config.device) for item in batch]
        inputs, label = all_item[:-1], all_item[-1]
        pred = model.forward(*inputs)

        pred_value = pred.detach().cpu().numpy()
        real_value = label.detach().cpu().numpy()

        history_value = scaler.inverse_transform(inputs[0])[:, :, :, -1]
        pred_value = scaler.inverse_transform(pred_value)[:, :, :, -1]
        real_value = scaler.inverse_transform(real_value)[:, :, :, -1]

        print(history_value.shape, real_value.shape, pred_value.shape)
        for k in range(history_value.shape[-1]):
            now_idx = cnt
            for i in trange(history_value.shape[0]):
                now_idx += 1
                save_figure(history_value[i, :, k], real_value[i, :, k], pred_value[i, :, k], now_idx, k, config)

        cnt += label.shape[0]

    return True



def predict_7_30_60_90(all_model, all_dataloader, all_y_scaler, config):
    all_history = np.zeros((len(all_dataloader[-1].dataset), all_dataloader[-1].dataset.seq_len, all_dataloader[-1].dataset.y.shape[1]))
    all_reals = np.zeros((len(all_dataloader[-1].dataset), all_dataloader[-1].dataset.pred_len, all_dataloader[-1].dataset.y.shape[1]))
    all_preds = np.zeros((len(all_dataloader[-1].dataset), all_dataloader[-1].dataset.pred_len, all_dataloader[-1].dataset.y.shape[1]))
    for i in range(len(all_model) - 1, 0, -1):
        # print(len(all_dataloader[i].dataset))
        for batch in (all_dataloader[i]):
            all_item = [item.to(config.device) for item in batch]
            inputs, label = all_item[:-1], all_item[-1]
            pred = all_model[i].forward(*inputs)
            pred_value = pred.detach().cpu().numpy()
            real_value = label.detach().cpu().numpy()
            history_value = all_y_scaler[i].inverse_transform(inputs[0])[:, :, :, -1]
            pred_value = all_y_scaler[i].inverse_transform(pred_value)[:, :, :, -1]
            real_value = all_y_scaler[i].inverse_transform(real_value)[:, :, :, -1]
            # print(history_value.shape, real_value.shape, pred_value.shape)
            all_history[:len(all_dataloader[-1].dataset), :all_dataloader[i].dataset.seq_len, :all_dataloader[i].dataset.y.shape[1]] = history_value[:len(all_dataloader[-1].dataset)]
            all_reals[:len(all_dataloader[-1].dataset), :all_dataloader[i].dataset.pred_len, :all_dataloader[i].dataset.y.shape[1]] = real_value[:len(all_dataloader[-1].dataset)]
            all_preds[:len(all_dataloader[-1].dataset), :all_dataloader[i].dataset.pred_len, :all_dataloader[i].dataset.y.shape[1]] = pred_value[:len(all_dataloader[-1].dataset)]

    # 已经获取了90长度的内容了
    # [bs, pred_len, 131]
    print(all_reals.shape)
    for k in range(all_reals.shape[-1]):
        now_idx = 0
        for i in trange(all_reals.shape[0]):
            save_figure(all_history[i, :, k], all_reals[i, :, k], all_preds[i, :, k], now_idx, k, config)
            now_idx += 1
        if k >= 20:
            break
    return True


def RunOnce(config, runId):
    set_seed(config.seed + runId)
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    all_model = []
    all_test_x = []
    all_test_y = []
    all_y_scaler = []
    all_dataloader = []
    all_scnerios = [[36, 7], [36, 30], [36, 60], [36, 90]]
    for seq_len, pred_len in all_scnerios:
        config.seq_len = seq_len
        config.pred_len = pred_len
        filename, exper_detail = get_experiment_name(config)
        log.filename = filename
        datamodule = DataModule(config)
        model = Model(config)
        model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
        print(model_path)
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
            results['train_time'] = sum_time
        except Exception as e:
            # raise e
            pass
        model.setup_optimizer(config)
        results = model.evaluate_one_epoch(datamodule, 'test')
        log(f'NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Acc_10={results["Acc_10"]:.4f} DTW={results["DTW"]:.4f}')
        all_model.append(model)
        dataloader, flag = data_to_dataloader(datamodule.test_set.x, datamodule.test_set.y)
        all_dataloader.append(dataloader)
        all_y_scaler.append(datamodule.y_scaler)
        config.record = False
    # results = predict(model, datamodule.test_set.x, datamodule.test_set.y, datamodule.y_scaler, config)
    results = predict_7_30_60_90(all_model, all_dataloader, all_y_scaler, config)
    return results


def run(config):
    set_settings(config)
    metrics = RunOnce(config, 0)
    return metrics

if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config('FinancialConfig')
    run(config)
