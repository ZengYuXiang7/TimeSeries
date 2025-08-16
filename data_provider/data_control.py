# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

from data_provider.generate_financial import generate_data
from data_provider.get_financial import *
from data_provider.get_ts import get_ts
from data_provider.data_getitem import TensorDataset, TimeSeriesDataset

def load_data(config):
    if config.dataset == 'financial':
        if config.multi_dataset:
            x, y, x_scaler, y_scaler = multi_dataset(config)
        else:
            x, y, x_scaler, y_scaler = single_dataset(config)
    elif config.dataset == 'weather':
        x, y, x_scaler, y_scaler = get_ts(config.dataset, config)
    return x, y, x_scaler, y_scaler

def get_dataset(train_x, train_y, valid_x, valid_y, test_x, test_y, config):
    if config.dataset == 'financial':
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )
    else:
        return (
            TimeSeriesDataset(train_x, train_y, 'train', config),
            TimeSeriesDataset(valid_x, valid_y, 'valid', config),
            TimeSeriesDataset(test_x, test_y, 'test', config)
        )
    