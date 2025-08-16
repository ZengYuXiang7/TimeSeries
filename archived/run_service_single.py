import torch
from exp.exp_model import Model
from data_provider.generate_financial import get_all_fund_list, process_date_columns, query_fund_data
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

def get_history_data(now_code, current_date, config):
    start_date = get_start_date(current_date, window_size=64)
    data = query_fund_data([now_code], start_date, current_date)[now_code]
    data = np.array(data)
    return data

def check_input(single_history_input, config):
    data = single_history_input[-config.seq_len:, :]
    return data

def get_pretrained_model(log, config):
    model = Model(config)
    runId = 0
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
    print(model_path)
    # model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    return model 


def constrain_nav_prediction(predictions, bar=0.05, scale=0.9):
    """
    检测单位净值预测中是否存在超过bar的相邻涨跌幅，
    如果是，则整条基金的净值序列按相对首日值重新缩放（温和调整）

    参数：
    - predictions: np.ndarray [7, 64]，表示64支基金7天的预测单位净值
    - bar: float，单位净值日涨跌幅上限（如0.05表示5%）
    - scale: float，检测异常后，使用的趋势缩放系数（如0.9）

    返回：
    - adjusted: np.ndarray [7, 64]，处理后的单位净值预测
    - mask: np.ndarray [64]，表示哪些基金被缩放（True为缩放）
    """
    adjusted = predictions.copy()
    mask = np.zeros(predictions.shape[1], dtype=bool)
    for fund_idx in range(predictions.shape[1]):
        nav_series = predictions[:, fund_idx]
        # 计算相邻涨跌幅
        returns = nav_series[1:] / nav_series[:-1] - 1
        if np.any(np.abs(returns) > bar):
            # 以首日为锚点，重构温和曲线
            base = nav_series[0]
            relative_change = (nav_series - base) / base
            softened = base * (1 + relative_change * scale)
            adjusted[:, fund_idx] = softened
            mask[fund_idx] = True
    return adjusted, mask

def predict_torch_model(model, history_input, config):
    # 因为我加了时间戳特征
    x = history_input[:, -3:]
    # unsqueeze 代表 batch size = 1
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    pred_value = model(x, None, None).squeeze(0).detach().numpy()
    # 因为模型改成了多变量预测多变量，按照预测结果的最后一个变量作为预测值
    pred_value = pred_value[:, -1]
    pred_value = np.abs(pred_value)
    pred_value = pred_value[:config.pred_len]
    # pred_value, _ = constrain_nav_prediction(pred_value, bar=1.0, scale=0.1)
    return pred_value

def get_sql_format_data(pred_value, cleaned_input):
    from datetime import datetime
    now_df = []
    cleaned_input = cleaned_input[0, :]
    idx = config.idx
    fund_code = cleaned_input[0]
    forcast_date = current_date
    pred = '{"pre": [' + ', '.join(f'{item:.6f}' for item in pred_value[:]) + ']}'
    model_version = 'v2025'
    create_date = update_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now_df.append([idx, fund_code, forcast_date, pred, model_version, create_date, update_date])
    now_df = np.array(now_df)
    now_df = pd.DataFrame(now_df, columns=['id', 'fund_code', 'forecast_date', 'pre_data', 'model_version',
       'create_time', 'update_time'])
    return now_df

def insert_pred_to_sql(df, table_name):
    try:
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
    if config.drop:
        drop_sql_temp(table_name)
    print("✅ 配置加载完成。")

    print(f"\n📅 当前预测日期: {current_date}")
    print(f"➡️ 输入序列长度: {config.seq_len}, 预测长度: {config.pred_len}")
    
    all_code_list = get_all_fund_list()
    for i in range(len(all_code_list)):
        # 27
        try:
            log_filename, exper_detail = get_experiment_name(config)
            plotter = MetricsPlotter(log_filename, config)
            log = Logger(log_filename, exper_detail, plotter, config, show_params=False)
            config.idx = i
            print(f"📊 获取基金组第 {i} 个基金，基金号为 {all_code_list[i]}")

            history_input = get_history_data(all_code_list[i], current_date, config)
            if len(history_input) < 827: 
                print('❗跳过，此基金长度过短，无法训练得到有效模型')
                continue
            print(f"📈 历史数据已获取。列表长度: {len(history_input)}")

            cleaned_input = check_input(history_input, config)
            print(f"🧹 清洗后的输入数据维度: {cleaned_input.shape}")  # 应为 [seq_len, group_num, feature_dim]

            model = get_pretrained_model(log, config)
            print("🤖 模型加载完成。")

            pred_value = predict_torch_model(model, cleaned_input, config)
            print(f"📉 预测结果维度: {pred_value.shape}")

            pred_value_sql = get_sql_format_data(pred_value, cleaned_input)
            print(f"🧾 预测结果已转为 DataFrame，准备写入数据库。表格 shape: {pred_value_sql.shape}")
            print(pred_value_sql.head(2))  # 打印前两行以核验内容结构

            insert_pred_to_sql(pred_value_sql, table_name)
        except Exception as e:
            raise e
            print(e)
            continue

    return pred_value_sql

if __name__ == '__main__':
    config = get_config('FinancialConfig')
    # current_date = '2025-4-15'
    current_date = datetime.now().strftime('%Y-%m-%d')
    pred_value = start_server(current_date)