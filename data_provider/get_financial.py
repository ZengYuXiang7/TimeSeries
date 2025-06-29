# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from data_provider.generate_financial import get_all_fund_list, generate_data, process_fund
from data_provider.data_scaler import get_scaler


# 为了对齐实验，现在加上this one  20250513 15时47分
def get_benchmark_code():
    with open('./datasets/benchmark.pkl', 'rb') as f:
        group = pickle.load(f)
        group = group['stock-270000']
    return group


def get_group_idx(group_index):
    # './results/func_code_to_label_{n_clusters}.pkl'
    with open('./results/func_code_to_label_40_balanced.pkl', 'rb') as f:
        data = pickle.load(f)
    all_func_code = []
    for i in range(len(data)):
        if int(data[i][1]) == group_index:
            all_func_code.append(data[i][0])
    return all_func_code


def multi_dataset(config):
    # now_fund_code = get_benchmark_code()
    now_fund_code = get_group_idx(config.idx)
    min_length = 1e9
    all_data = []
    for fund_code in tqdm(now_fund_code, desc='SQL'):
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            df = process_fund(0, fund_code, config.start_date, config.end_date)
            print(f"{fund_code} -- len = {len(df)}, now min_length = {min_length}")
            min_length = min(len(df), min_length)
            all_data.append(df)
        except Exception as e:
            print(e)
            process_fund(0, fund_code, config.start_date, config.end_date)

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
    x, y = data[:, :, :], data[:, :, -3:]

    x[:, :, -3:] = x[:, :, -3:].astype(np.float32)
    x_scaler = get_scaler(x[:, :, -3:], config, 'stander')
    x[:, :, -3:] = x_scaler.transform(x[:, :, -3:])

    y_scaler = get_scaler(y, config, 'stander')
    y = y_scaler.transform(y)
    return x, y, x_scaler, y_scaler


def get_financial_data(start_date, end_date, idx, config):
    # fund_code = get_all_fund_list()[idx]
    # 为了对齐实验，现在加上this one  20250513 15时47分
    fund_code = get_benchmark_code()[idx]
    try:
        data = get_data(start_date, end_date, fund_code)
        # data = process_fund(0, fund_code, config.start_date, config.end_date)
    except Exception as e:
        print(e)
        generate_data(start_date, end_date)
        data = get_data(start_date, end_date, fund_code)

    data = data.astype(np.float32)

    x, y = data, data[:, -1].astype(np.float32)
    scaler = None

    if not config.multi_dataset:
        for i in range(len(input_keys)):
            x[:, -i] = x[:, -i].astype(np.float32)
            scaler = get_scaler(x[:, -i], config)
            x[:, -i] = scaler.transform(x[:, -i])

        scaler = get_scaler(y, config)
        y = scaler.transform(y)

    y = y.astype(np.float32)
    # 构建
    X_window, y_window = x, y
    # X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    print(X_window.shape, y_window.shape)
    return X_window, y_window, scaler





