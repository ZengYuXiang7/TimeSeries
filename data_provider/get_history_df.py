from datetime import datetime, timedelta
import pickle 
import numpy as np
from sqlalchemy import create_engine, text
import pandas as pd 
# from run_service import get_history_data
import empyrical

current_date = datetime.now().strftime('%Y-%m-%d')
DB_URI = 'mysql+pymysql://root:qilai123@123.57.74.222:3306/fund'

def get_start_date(end_date: str, window_size: int) -> str:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_size)
    return start_dt.strftime("%Y-%m-%d")


def query_fund_data(fund, start_date, end_date):
    engine = create_engine(DB_URI)
    """查询数据库中某支基金的净值数据
        SELECT fund_code, date, accnav, adj_nav, nav
    """
    sql = text("""
        SELECT fund_code, date, nav, accnav, adj_nav
        FROM b_fund_nav_details_new
        WHERE fund_code IN :codes
          AND date BETWEEN :start AND :end
        ORDER BY date
    """)
    try:
        df = pd.read_sql_query(
            sql.bindparams(codes=tuple(fund), start=start_date, end=end_date),
            engine
        )
        fund_dict = {code: df_group.reset_index(drop=True)
                     for code, df_group in df.groupby("fund_code")}
        return fund_dict
    except Exception as e:
        print(f"[{fund}] 数据库查询失败: {str(e)}")
        return pd.DataFrame()
    

def process_date_columns(df):
    """处理基金数据，返回包含统计指标的 numpy 数组"""
    df = df.copy()
    # 构造 datetime 列用于排序和时间索引
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    
    # 按时间排序
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # === 计算指标 ===
    daily_return = df['nav'].pct_change()
    cumulative = empyrical.cum_returns(daily_return, starting_value=0)
    annual_volatility = empyrical.annual_volatility(daily_return)
    stability = empyrical.stability_of_timeseries(daily_return)
    monthly_return = empyrical.aggregate_returns(daily_return, "monthly")
    monthwin = (monthly_return > 0).sum() / len(monthly_return)
    daywin = (daily_return > 0).sum() / len(daily_return)
    max_drawdown = empyrical.max_drawdown(daily_return)

    # 添加到 df 中
    df['daily_return'] = daily_return
    df['cumulative'] = cumulative
    df['annual_volatility'] = annual_volatility
    df['stability'] = stability
    df['monthwin'] = monthwin
    df['winning_day'] = daywin
    df['maxDrawdown'] = max_drawdown

    # 选择输出列
    df.reset_index(inplace=True)
    df = df[['fund_code', 'month', 'day', 'weekday',
             'daily_return', 'cumulative', 'annual_volatility',
             'stability', 'monthwin', 'winning_day', 'maxDrawdown', 'accnav', 'adj_nav', 'nav',]]
    df = df.fillna(0)  # 填充缺失值为0
    # print(df[:5])
    df = df.to_numpy() 
    return df


        
def get_history_data(get_group_idx, current_date):
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
    
    data = np.stack(all_history_input, axis=0)
    data = data.transpose(1, 0, 2)
    
    # 只取符合模型的历史天数
    data = data[-90:, :, :]
    data = data[:, :, -3]
    data = data.T
    history_data = data[:, :60]
    future_data = data[:, 60:]
    return history_data, future_data


if __name__ == '__main__':
    with open(f'func_code_to_label_160_balanced.pkl', 'rb') as f:
        data = pickle.load(f)
    all_func_code = []
    for i in range(len(data)):
        if int(data[i][1]) == 1:
            all_func_code.append(data[i][0])
    history_data, future_data = get_history_data(all_func_code, current_date)
    print(history_data.shape, future_data.shape)