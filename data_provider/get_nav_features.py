# 导入所需模块
from datetime import datetime
import mysql
from mysql.connector import Error
import pandas as pd
import numpy as np
import empyrical
import warnings

# 忽略特定的 SQLAlchemy 警告
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# 数据库连接配置
DB_CONFIG = {
    'host': '123.57.74.222',
    'user': 'data_user',
    'password': 'DataUser123',
    'database': 'ai_data'
}

# 创建数据库连接
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# 执行 SQL 查询并返回 DataFrame
def get_df_from_sql(query: str):
    connection = get_db_connection()
    if not connection:
        raise Exception("Database connection failed")
    try:
        return pd.read_sql(query, connection)
    except Error as e:
        raise Exception(f"Database error: {e}")
    finally:
        if connection.is_connected():
            connection.close()

# 将日期转换为季度字符串（如 "2020Q1"）
def date_to_quarter(date_obj):
    quarter = (date_obj.month - 1) // 3 + 1
    return f"{date_obj.year}Q{quarter}"

# 获取并整合某支基金的时间序列数据及宏观经济数据
def get_df(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    # 将开始和结束日期转为 datetime.date 类型
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # 获取起止时间对应的季度
    start_quarter = date_to_quarter(start_date)
    end_quarter = date_to_quarter(end_date)

    # 查询美元指数数据
    usdind_df = get_df_from_sql(f"SELECT * FROM usdind WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' ORDER BY trade_date asc")

    # 查询 GDP 数据
    gdp_df = get_df_from_sql(f"SELECT * FROM gdp WHERE quarter BETWEEN '{start_quarter}' AND '{end_quarter}' ORDER BY quarter asc")

    # 查询 SHIBOR 数据
    shibor_df = get_df_from_sql(f"SELECT * FROM shibor WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")

    # 查询基金净值数据
    net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")

    # 合并美元指数（按日期匹配）
    merged_df = pd.merge(net_value_df, usdind_df, left_on='date', right_on='trade_date', how='left')

    # 添加季度信息用于合并 GDP
    merged_df['quarter'] = merged_df['date'].apply(date_to_quarter)

    # 合并 GDP（按季度匹配）
    merged_df = pd.merge(merged_df, gdp_df, on='quarter', how='left')

    # 合并 SHIBOR（按日期匹配）
    final_df = pd.merge(merged_df, shibor_df, left_on='date', right_on='date', how='left')

    # 删除不必要的字段
    final_df.drop(columns=['trade_date', 'quarter', 'id'], inplace=True, errors='ignore')

    # 将日期列设置为索引
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df.set_index("date", inplace=True)

    # 计算每日收益率（相邻净值变化率）
    daily_return = pd.DataFrame({
        "date": pd.to_datetime(net_value_df["date"]),
        "daily_return": net_value_df["adj_nav"].pct_change(),
    })
    daily_return.set_index("date", inplace=True)

    # 累计收益率（从起始日累积）
    cum_returns = empyrical.cum_returns(daily_return)
    final_df["cumulative"] = cum_returns

    # 年化波动率（标准差 × sqrt(年交易日数)）
    annual_volatility = empyrical.annual_volatility(daily_return)
    final_df["annual_volatility"] = annual_volatility[0]

    # 收益序列的稳定性（R² 拟合度）
    stability = empyrical.stability_of_timeseries(daily_return)
    final_df["stability"] = stability

    # 月胜率（上涨月份 / 总月份）
    monthly_return = empyrical.aggregate_returns(daily_return, "monthly")
    monthwin = (monthly_return > 0).sum() / len(monthly_return)
    final_df["monthwin"] = monthwin.item()

    # 日胜率（上涨交易日数 / 总交易日数）
    daywin = (daily_return > 0).sum() / len(daily_return)
    final_df["winning_day"] = daywin.item()

    # 最大回撤（peak 到 trough 的最大跌幅）
    max_drawdown = empyrical.max_drawdown(daily_return)
    final_df["maxDrawdown"] = max_drawdown.item()

    final_df = final_df[[
        'cumulative',
        'annual_volatility',
        'stability',
        'monthwin',
        'winning_day',
        'maxDrawdown'
    ]]
    return final_df

# 获取数据并以日期作为索引返回
def get_df_date_as_index(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    return get_df(fund_code, start_date, end_date)

# 获取数据并将日期作为普通列返回
def get_df_date_as_colum(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    df = get_df(fund_code, start_date, end_date)
    df.reset_index(inplace=True)
    return df

# 主函数入口（用于单独运行本脚本调试）
if __name__ == "__main__":
    """
    from get_df import get_df_date_as_index, get_df_date_as_colum
    """
    # 打印某支基金的完整 DataFrame（以日期为索引）
    print(get_df_date_as_index("000061", "2025-06-01", "2025-7-13"))

    # 打印以日期为列的格式（可注释上面一行测试）
    # print(get_df_date_as_colum("000001", "2020-01-01", "2025-06-30"))