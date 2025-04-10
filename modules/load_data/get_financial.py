# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text
from utils.data_scaler import get_scaler
from modules.load_data.create_window_dataset import create_window_dataset

pred_value = 'nav'
# pred_value = 'accnav'

def get_data(code_idx):
    address = os.listdir(f'./datasets/financial/{pred_value}/')
    with open(f'./datasets/financial/{pred_value}/{address[code_idx]}', 'rb') as f:
        data = pickle.load(f)
    print(f'./datasets/financial/{pred_value}/{address[code_idx]}')
    return data

def generate_data(start_date, end_date, code_idx):
    # 数据库配置
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)

    engine = create_engine(DB_URI)
    DATA_JSON_PATH = './datasets/data.json'

    # 加载分类数据
    with open(DATA_JSON_PATH) as f:
        group_map = json.load(f)

    # 遍历所有股票类别
    for group_idx, (group_name, group_list) in enumerate(group_map.items()):
        for group_data in group_list:
            if group_name == 'stock1' and group_data['industry_code'] in ['270000', '480000']:
                code_list = group_data['code_list']
                try:
                    sql = text(f"SELECT fund_code, date, {pred_value} FROM b_fund_nav_details_new WHERE fund_code IN :codes AND date BETWEEN :start AND :end ORDER BY date")
                    df = pd.read_sql_query(
                        sql.bindparams(codes=tuple(code_list), start=start_date, end=end_date),
                        engine
                    )
                except Exception as e:
                    print(f"数据库查询失败: {str(e)}")
                    raise e

                # 为每组的股票做映射，获得分组可以去其他地方做
                df = df.to_numpy()
                unique_values = np.unique(df[:, 0])
                index_mapping = {value: idx for idx, value in enumerate(unique_values)}
                values = [[] for _ in range(len(unique_values))]
                dates = [[] for _ in range(len(unique_values))]
                for i in range(len(df)):
                    idx = index_mapping[df[i][0]]
                    dates[idx].append([df[i][1].year, df[i][1].month, df[i][1].day, df[i][1].weekday()])
                    values[idx].append(float(df[i][2]))

                os.makedirs(f'./datasets/financial/{pred_value}/', exist_ok=True)
                for key, value in index_mapping.items():
                    now_data = np.concatenate([np.array([key] * len(dates[value])).reshape(-1, 1), np.array(dates[value]), np.array(values[value]).reshape(-1, 1)], axis=1)
                    with open(f'./datasets/financial/{pred_value}/{key}.pkl', 'wb') as f:
                        pickle.dump(now_data, f)
                        print(f'./datasets/financial/{pred_value}/{key}.pkl 存储完毕')
    data = get_data(code_idx)
    return data




def get_financial_data(start_date, end_date, idx, config):
    try:
        data = get_data(idx)
    except Exception as e:
        data = generate_data(start_date, end_date, idx)
    data = data.astype(np.float32)

    x, y = data, data[:, -1].astype(np.float32)
    # x, y = data[:, -1], data[:, -1].astype(np.float32)
    scaler = None
    if not config.multi_dataset:
        scaler = get_scaler(y, config)
        y = scaler.transform(y)
        # x = scaler.transform(x)
        x[:, -1] = x[:, -1].astype(np.float32)
        temp = x[:, -1].astype(np.float32)
        x[:, -1] = (temp - scaler.y_mean) / scaler.y_std

    # x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    # X_window, y_window = filter_jump_sequences(X_window, y_window, threshold=0.5, mode='absolute')
    return X_window, y_window, scaler

def filter_jump_sequences(X_window, y_window, threshold=0.3, mode='absolute'):
    """
    X_window: [n, seq_len, d] numpy array，其中最后一个维度是 value
    y_window: [n, ...]，标签
    返回：
        - 过滤后的 X_window
        - 对应的 y_window
        - 保留的索引 idx（相对于原始）
    """
    values = X_window[:, :, -1]  # 提取 value 部分
    diff = values[:, 1:] - values[:, :-1]

    if mode == 'absolute':
        mask = np.any(np.abs(diff) > threshold, axis=1)
    elif mode == 'relative':
        prev = np.clip(values[:, :-1], 1e-5, None)
        rel_diff = np.abs(diff / prev)
        mask = np.any(rel_diff > threshold, axis=1)
    else:
        raise ValueError("mode must be 'absolute' or 'relative'")
    idx = np.where(~mask)[0]
    X_window, y_window = X_window[idx], y_window[idx]
    return X_window, y_window


def normorlize(data, value_mean, value_std):
    # print(data.shape)
    data[:, :, -1] = data[:, :, -1].astype(np.float32)
    temp = data[:, :, -1].astype(np.float32)
    data[:, :, -1] = (temp - value_mean) / value_std
    return data


def multi_dataset(config):
    all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y = [], [], [], [], [], []
    for i in range(10):
        config.idx = i
        config.multi_dataset = False
        datamodule = data_center.DataModule(config)
        if len(datamodule.train_set.x) == 0 or len(datamodule.y) <= config.seq_len:
            continue
        all_train_x.append(datamodule.train_set.x)
        all_train_y.append(datamodule.train_set.y)

        all_valid_x.append(datamodule.valid_set.x)
        all_valid_y.append(datamodule.valid_set.y)

        all_test_x.append(datamodule.test_set.x)
        all_test_y.append(datamodule.test_set.y)
        del datamodule

    all_train_x = np.concatenate(all_train_x, axis=0)
    all_train_y = np.concatenate(all_train_y, axis=0)

    all_valid_x = np.concatenate(all_valid_x, axis=0)
    all_valid_y = np.concatenate(all_valid_y, axis=0)

    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    for i in range(len(all_train_x)):
        if all_train_x[i][0][1] > 2025:
            print(all_train_x[i][0])
            # exit()

    y_mean = np.mean(all_train_y)
    y_std = np.std(all_train_y)

    all_train_y = (all_train_y - y_mean) / y_std
    all_valid_y = (all_valid_y - y_mean) / y_std
    all_test_y = (all_test_y - y_mean) / y_std

    all_train_x = normorlize(all_train_x, y_mean, y_std)
    all_valid_x = normorlize(all_valid_x, y_mean, y_std)
    all_test_x = normorlize(all_test_x, y_mean, y_std)

    scaler = get_scaler(all_train_y, config)
    scaler.y_mean, scaler.y_std = y_mean, y_std
    print(all_train_x.shape, all_train_y.shape)
    return all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y, scaler


