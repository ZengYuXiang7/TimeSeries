# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from data_provider.generate_financial import get_all_fund_list, generate_data, process_date_columns, process_fund, query_fund_data
from data_provider.data_scaler import get_scaler


def get_group_idx(group_index, config):
    with open(f'./datasets/func_code_to_label_{config.n_clusters}_balanced.pkl', 'rb') as f:
    # with open(f'./datasets/func_code_to_label_{config.n_clusters}.pkl', 'rb') as f:
        data = pickle.load(f)
    all_func_code = []
    for i in range(len(data)):
        if int(data[i][1]) == group_index:
            all_func_code.append(data[i][0])
    return all_func_code


def multi_dataset(config):
    now_fund_code = get_group_idx(config.idx, config)
    # now_fund_code = get_benchmark_code()
    # now_fund_code = get_group_idx(config.idx)
    min_length = 1e9
    all_data = []
    fund_dict = query_fund_data(now_fund_code, config.start_date, config.end_date)
    for fund_code, value in fund_dict.items():
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            min_length = min(len(value), min_length)
            value = process_date_columns(value)
            all_data.append(value)
        except Exception as e:
            print(e)
            process_fund(0, fund_code, config.start_date, config.end_date)

    print(f"Num_code = {len(now_fund_code)} Min_length = {min_length}")
    raw_data = []
    for df in all_data:
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            # df = process_fund(0, fund_code, config.start_date, config.end_date)
            raw_data.append(df[-min_length:])
        except Exception as e:
            print(e)
    data = np.stack(raw_data, axis=0)
    data = data.transpose(1, 0, 2)
    x = data[:, :, :].copy()  # 所有特征除
    y = data[:, :, -3:].copy()  # 只取最后3列作为目标变量

    x[:, :, 4:] = x[:, :, 4:].astype(np.float32)
    x_scaler = get_scaler(x[:, :, 4:], config, 'minmax')
    x[:, :, 4:] = x_scaler.transform(x[:, :, 4:])

    y_scaler = get_scaler(y, config, 'minmax')
    y = y_scaler.transform(y)
    return x, y, x_scaler, y_scaler


def single_dataset(config):
    # now_fund_code = get_group_idx(config.idx)
    # 2025年07月04日20:06:51，这个代码数据库暂时有点问题
    all_fund_code = get_all_fund_list()
    now_fund_code = all_fund_code[config.idx]
    fund_dict = query_fund_data([now_fund_code], config.start_date, config.end_date)
    data = process_date_columns(fund_dict[now_fund_code])
    if len(data) < 827: 
        exit()
    x, y = data[:, :], data[:, -3:]

    x[:, 4:] = x[:, 4:].astype(np.float32)
    x_scaler = get_scaler(x[:, 4:], config, 'minmax')
    x[:, 4:] = x_scaler.transform(x[:, 4:])

    y_scaler = get_scaler(y, config, 'minmax')
    y = y_scaler.transform(y)
    return x, y, x_scaler, y_scaler
