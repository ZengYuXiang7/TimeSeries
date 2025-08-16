import torch
from data_provider.data_loader import DataModule
from exp.exp_model import Model
from data_provider.generate_financial import process_date_columns, query_fund_data
from data_provider.get_financial import get_group_idx
from run_train import get_experiment_name
from utils.exp_config import get_config
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import os 
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter

def drop_sql_temp(tabel_name):
    try:
        # 读取数据库连接字符串
        with open('./datasets/sql_token.pkl', 'rb') as f:
            DB_URI = pickle.load(f)

        # 创建数据库引擎
        engine = create_engine(DB_URI)

        # 执行 DROP TABLE 操作
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {tabel_name}"))
            print(f"✅ 表 {tabel_name} 已成功删除。")

    except Exception as e:
        print(f"❌ 删除表时出错: {e}")

def get_start_date(end_date: str, window_size: int) -> str:
    """
    给定结束日期和历史窗口长度，返回窗口开始日期（字符串格式）。

    参数：
    - end_date (str): 结束日期，格式 'YYYY-MM-DD'
    - window_size (int): 历史窗口长度（天数）

    返回：
    - start_date (str): 开始日期，格式 'YYYY-MM-DD'
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_size)
    return start_dt.strftime("%Y-%m-%d")

def check_bad_model(all_scnerios, log, config):
    all_acc = []
    try:
        for seq_len, pred_len in all_scnerios:
            config.seq_len = seq_len
            config.pred_len = pred_len
            filename, exper_detail = get_experiment_name(config)
            log.filename = filename
            with open(f'./results/metrics/{log.filename}.pkl', 'rb') as f:
                metrics = pickle.load(f)
                Acc = metrics['Acc_10']
                all_acc.append(Acc)
    except FileNotFoundError as e:
        print(f"❌ {e}，可能是模型未收敛。")
        return True
    
    return np.mean(all_acc) < 0.5

def get_history_data(get_group_idx, current_date, config):
    all_history_input = []
    start_date = get_start_date(current_date, window_size=2000)
    fund_dict = query_fund_data(get_group_idx, start_date, current_date)
    min_len = 1e9
    for key, value in fund_dict.items():
        min_len = min(len(value), min_len)

    for key, value in fund_dict.items():
        df = process_date_columns(value)
        df = df[-min_len:, :]
        all_history_input.append(df)
    data = all_history_input
    return data

def check_input(all_history_input, config):
    data = np.stack(all_history_input, axis=0)
    data = data.transpose(1, 0, 2)
    # 只取符合模型的历史天数
    data = data[-config.seq_len:, :, :]
    return data

def get_pretrained_model(log, config):
    model = Model(config)
    runId = 0
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
    print(model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    return model 


def apply_delta_with_hist_constraints(hist, pred):
    """
    hist: shape [seq_len, n]
    pred: shape [T_pred, n]
    """
    pred_clipped = pred.copy().T  # shape: [n, T_pred]
    hist = hist.copy().T          # shape: [n, seq_len]
    N, T_pred = pred_clipped.shape

    # Step 1: 计算历史最大涨跌
    hist_diff = hist[:, 1:] - hist[:, :-1]
    max_gain = np.max(hist_diff, axis=1)  # [n]
    max_drop = np.min(hist_diff, axis=1)  # [n]

    # print("基金历史最大涨跌幅：")
    # for i in range(N):
        # print(f"基金{i}: max_gain = {max_gain[i]:.4f}, max_drop = {max_drop[i]:.4f}")

    # Step 2: 应用递推约束
    for i in range(N):
        for t in range(1, T_pred):
            prev = pred_clipped[i, t - 1]
            curr = pred_clipped[i, t]
            delta = curr - prev

            if delta > max_gain[i]:
                low = prev
                high = prev + max_gain[i]
                if high < low:
                    low, high = high, low
                new_val = np.random.uniform(low, high)
                print(f"[基金{i}] 第{t}步: 涨幅超限 (Δ={delta:.4f} > {max_gain[i]:.4f})，原值={curr:.4f} → 新值={new_val:.4f}")
                pred_clipped[i, t] = new_val

            elif delta < max_drop[i]:
                low = prev + max_drop[i]  # 注意是 prev + max_drop（max_drop 是负数）
                high = prev
                if high < low:
                    low, high = high, low
                new_val = np.random.uniform(low, high)
                print(f"[基金{i}] 第{t}步: 跌幅超限 (Δ={delta:.4f} < {max_drop[i]:.4f})，原值={curr:.4f} → 新值={new_val:.4f}")
                pred_clipped[i, t] = new_val

    return pred_clipped.T  # shape: [T_pred, n]

def predict_torch_model(model, history_input, x_scaler, y_scaler, config):
    # 因为我加了时间戳特征
    x = history_input
    x[:, :, 4:] = x_scaler.transform(x[:, :, 4:])
    x = history_input[:, :, -3:]
    x_fund = history_input[:, :, 0]
    x_mark = history_input[:, :, 1:4] 
    x_features = history_input[:, :, 4:-3]
    # unsqueeze 代表 batch size = 1
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    x_features = torch.from_numpy(x_features.astype(np.float32)).unsqueeze(0)
    pred_value = model(x, x_mark, x_fund, x_features).squeeze(0).detach().numpy()
    pred_value = y_scaler.inverse_transform(pred_value)
    
    pred_value = pred_value[:, :, -1]
    pred_value = np.abs(pred_value)
    
    return pred_value

def get_final_pred(all_scnerios, group_fund_code, current_date, log, config):

    history_input = get_history_data(group_fund_code, current_date, config)
    print(f"📈 历史数据已获取。列表长度: {len(history_input)}")
    all_pred = np.zeros((90, len(history_input)))

    prev_len = 0
    for seq_len, pred_len in all_scnerios:
        config.seq_len = seq_len
        config.pred_len = pred_len
        filename, exper_detail = get_experiment_name(config)
        datamodule = DataModule(config)
        x_scaler, y_scaler = datamodule.x_scaler, datamodule.y_scaler
        del datamodule
        log.filename = filename
        cleaned_input = check_input(history_input, config)
        print(f"🧹 清洗后的输入数据维度: {cleaned_input.shape}")  # 应为 [seq_len, group_num, feature_dim]

        model = get_pretrained_model(log, config)
        print("🤖 模型加载完成。")

        pred_value = predict_torch_model(model, cleaned_input, x_scaler, y_scaler, config)
        print(f"📉 预测结果维度: {pred_value.shape}")
        
        start_idx = prev_len
        end_idx = config.pred_len
        prev_len = config.pred_len
        all_pred[start_idx:end_idx, :] = pred_value[start_idx:end_idx, :]
        print(f"📊 预测结果已存入 all_pred，从 {start_idx} 到 {end_idx}。")

    hist = np.stack(history_input, axis=0).transpose(1, 0, 2)[:, :, -1]
    pred_value = apply_delta_with_hist_constraints(hist, pred_value)
    return pred_value, cleaned_input

def get_sql_format_data(pred_value, cleaned_input):
    from datetime import datetime
    now_df = []
    cleaned_input = cleaned_input[0, :, :]
    for j in range(pred_value.shape[1]):
        idx = config.idx
        fund_code = cleaned_input[j][0]
        forcast_date = current_date
        pred = '{"pre": [' + ', '.join(f'{item:.6f}' for item in pred_value[:, j]) + ']}'
        model_version = 'v2025'
        create_date = update_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_df.append([idx, fund_code, forcast_date, pred, model_version, create_date, update_date])
        # break
    # now_df
    now_df = np.array(now_df)
    now_df = pd.DataFrame(now_df, columns=['id', 'fund_code', 'forecast_date', 'pre_data', 'model_version', 'create_time', 'update_time'])
    return now_df



def insert_pred_to_sql(df, table_name):
    try:
        # 1. First save to CSV in results/csv directory
        os.makedirs('./results/csv', exist_ok=True)  # Create directory if it doesn't exist
        csv_path = f'./results/csv/{table_name}_predictions.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ 数据成功保存到CSV文件: {csv_path}")
        
        # 2. Then proceed with SQL insertion as before
        # 读取数据库连接字符串
        with open('./datasets/sql_token.pkl', 'rb') as f:
            DB_URI = pickle.load(f)
        # 创建数据库引擎
        engine = create_engine(DB_URI)
        # 插入数据
        df.to_sql(
            name=table_name,   # 表名
            con=engine,                   # 数据库连接
            if_exists='append',           # 追加到已有表中
            index=False                   # 不插入索引列
        )
        print(f"✅ 数据成功写入数据库{table_name}。")
        
    except FileNotFoundError:
        print("❌ 无法找到 sql_token.pkl 文件。请检查路径是否正确。")
    except SQLAlchemyError as e:
        print(f"❌ 数据库插入失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


# [128, 16, 33, 3])
def start_server(current_date, table_name = 'temp_sql'):
    drop_sql_temp(table_name)

    print(f"\n📅 当前预测日期: {current_date}")
    print(f"➡️ 输入序列长度: {config.seq_len}, 预测长度: {config.pred_len}")
    
    with open('./datasets/func_code_to_label_160_balanced.pkl', 'rb') as f:
        data = np.array(pickle.load(f))
        df = data[:, 1].astype(np.float32)
    group_num = int(df.max() + 1)
    all_scnerios = [[36, 7], [36, 30], [36, 60], [36, 90]]

    for i in range(group_num):
        # 27
        try:
            log_filename, exper_detail = get_experiment_name(config)
            plotter = MetricsPlotter(log_filename, config)
            log = Logger(log_filename, exper_detail, plotter, config, show_params=False)

            config.idx = i
            group_fund_code = get_group_idx(i, config)
            print(f"📊 获取基金组共 {len(group_fund_code)} 个基金列表中")

            if check_bad_model(all_scnerios, log, config):
                print(f"❗️ 模型效果不佳，跳过基金组 {i} 的预测。")
                continue

            print(f"🔍 正在处理基金组 {i}，部分基金代码: {group_fund_code[:10]}")
            pred_value, cleaned_input = get_final_pred(all_scnerios, group_fund_code, current_date, log, config)

            pred_value_sql = get_sql_format_data(pred_value, cleaned_input)
            print(f"🧾 预测结果已转为 DataFrame，准备写入数据库。表格 shape: {pred_value_sql.shape}")
            print(pred_value_sql.head(5))  # 打印前两行以核验内容结构
            
            insert_pred_to_sql(pred_value_sql, table_name)
        except Exception as e:
            raise e
            print(e)
            continue
    return pred_value_sql



if __name__ == '__main__':
    config = get_config('FinancialConfig')
    print("✅ 配置加载完成。")
    current_date = datetime.now().strftime('%Y-%m-%d')
    pred_value = start_server(current_date)