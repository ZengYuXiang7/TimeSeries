# coding : utf-8
# Author : Yuxiang Zeng
import numpy as np
from b_data_control import TensorDataset
from modules.load_data.get_financial import get_financial_data, multi_dataset
from modules.load_data.get_lottery import get_lottery
from modules.load_data.get_ts import get_ts
from utils.data_dataloader import get_dataloaders
from utils.data_spliter import get_split_dataset
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
from utils.exp_config import get_config


def load_data(config):
    if config.dataset == 'financial':
        all_x, all_y, scaler = get_financial_data('2020-07-13', '2025-03-8', config.idx, config)
    elif config.dataset == 'weather':
        all_x, all_y, scaler = get_ts(config.dataset, config)
    elif config.dataset == 'lottery':
        all_x, all_y, scaler = get_lottery(config.dataset, config)
    return all_x, all_y, scaler


# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.x, self.y, self.scaler = load_data(config)
        if config.debug:
            self.x, self.y = self.x[:300], self.y[:300]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)

        if config.dataset == 'financial' and config.multi_dataset:
            self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.scaler = multi_dataset(config)
            self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        else:
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)


    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )


if __name__ == '__main__':
    config = get_config()
    set_settings(config)
    config.experiment = True

    # logger plotter
    exper_detail = f"Dataset : {config.dataset.upper()}, Model : {config.model}, Train_size : {config.train_size}"
    log_filename = f'{config.train_size}_r{config.rank}'
    log = Logger(log_filename, exper_detail, config)
    plotter = MetricsPlotter(log_filename, config)
    config.log = log
    log(str(config.__dict__))

    datamodule = DataModule(config)
    for train_batch in datamodule.train_loader:
        all_item = [item.to(config.device) for item in train_batch]
        inputs, label = all_item[:-1], all_item[-1]
        print(inputs, label.shape)
        # break
    print('Done!')
