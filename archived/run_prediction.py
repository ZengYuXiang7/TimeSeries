# coding : utf-8
# Author : Yuxiang Zeng
import pickle 
import pandas as pd
from sqlalchemy import create_engine, text

from data_provider.generate_financial import query_fund_data
from data_provider.get_financial import get_group_idx
from datetime import datetime, timedelta

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

if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config('FinancialConfig')

    # 读取数据库连接字符串
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)

    # 创建数据库引擎
    engine = create_engine(DB_URI)

    # SQL 查询语句
    sql = text("""
        SELECT fund_code, forecast_date, pre_data, model_version, create_time, update_time
        FROM b_fund_forecast_new
        WHERE fund_code IN :codes
        ORDER BY forecast_date
    """)

    # # 执行查询，传入参数（注意 tuple 中只有一个元素时加逗号）
    # df = pd.read_sql_query(
    #     sql.bindparams(codes=tuple(['005626'])),  # 或 codes=('005626',)
    #     engine
    # )
    # df.to_sql('my_table', engine, if_exists='replace', index=False)

    now_fund_code = get_group_idx(27)
    # 执行查询，传入参数（注意 tuple 中只有一个元素时加逗号）
    df = query_fund_data(fund, start_date, end_date)
    # df = pd.read_sql_query(
        # sql.bindparams(codes=tuple([now_fund_code])),  # 或 codes=('005626',)
        # engine
    # )

    print(df)