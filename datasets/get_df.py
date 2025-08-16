
from datetime import datetime
import mysql
from mysql.connector import Error
import pandas as pd
import empyrical
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

DB_CONFIG = {
    'host': '123.57.74.222',
    'user': 'data_user',
    'password': 'DataUser123',
    'database': 'ai_data'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

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

def date_to_quarter(date_obj):
    quarter = (date_obj.month - 1) // 3 + 1
    return f"{date_obj.year}Q{quarter}"

def get_df(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    start_quarter = date_to_quarter(start_date)
    end_quarter = date_to_quarter(end_date)
    # print(start_date, end_date, start_quarter, end_quarter)

    type_df = get_df_from_sql(f"SELECT qilai_fund_type, qilai_fund_invest_type FROM fund_info WHERE fund_code = '{fund_code}'")
    net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")
    if type_df.empty:
        print("no such fund")
        return pd.DataFrame()
    elif net_value_df.empty:
        print("no net value data in database")
        return pd.DataFrame()

    usdind_df = get_df_from_sql(f"SELECT * FROM usdind WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' ORDER BY trade_date asc")
    gdp_df = get_df_from_sql(f"SELECT * FROM gdp WHERE quarter BETWEEN '{start_quarter}' AND '{end_quarter}' ORDER BY quarter asc")
    shibor_df = get_df_from_sql(f"SELECT * FROM shibor WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")
    # print(usdind_df, gdp_df, shibor_df, net_value_df)
    
    net_value_df['fund_type'] = type_df['qilai_fund_type'].iloc[0] + "-" + type_df['qilai_fund_invest_type'].iloc[0]
    merged_df = pd.merge(net_value_df, usdind_df, left_on='date', right_on='trade_date', how='left')
    merged_df['quarter'] = merged_df['date'].apply(date_to_quarter)
    merged_df = pd.merge(merged_df, gdp_df, on='quarter', how='left')
    final_df = pd.merge(merged_df, shibor_df, left_on='date', right_on='date', how='left')
    final_df.drop(columns=['trade_date', 'quarter', 'id'], inplace=True, errors='ignore')
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df.set_index("date", inplace=True)

    # # 总收益率（基金建立日期 到 end_date）
    # init_net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' ORDER BY date asc LIMIT 1")
    # init_net_value = 1.0000 if init_net_value_df.empty else init_net_value_df["adj_nav"][0]
    # print(init_net_value)

    # # 阶段收益率（start_date 到 end_date）
    # init_net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc LIMIT 1")
    # init_net_value = 1.0000 if init_net_value_df.empty else init_net_value_df["adj_nav"][0]
    # print(init_net_value)

    # final_df["cumulative"] = (final_df["adj_nav"] - init_net_value) / init_net_value
    # print(final_df)

    daily_return = pd.DataFrame({
        "date": pd.to_datetime(net_value_df["date"]),
        "daily_return": net_value_df["adj_nav"].pct_change()
    })
    daily_return.set_index("date", inplace=True)
    # print("daily_return", daily_return)

    cum_returns = empyrical.cum_returns(daily_return)
    # print("cum_returns", cum_returns)
    final_df["cumulative"] = cum_returns

    annual_volatility = empyrical.annual_volatility(daily_return)
    # print("annual_volatility", annual_volatility)
    final_df["annual_volatility"] = annual_volatility[0]

    stability = empyrical.stability_of_timeseries(daily_return)
    # print("stability", stability)
    final_df["stability"] = stability

    monthly_return = empyrical.aggregate_returns(daily_return, "monthly")
    monthwin = (monthly_return > 0).sum() / len(monthly_return)
    # print("monthwin", monthwin)
    final_df["monthwin"] = monthwin.item()

    daywin = (daily_return > 0).sum() / len(daily_return)
    # print("daywin", daywin)
    final_df["winning_day"] = daywin.item()

    max_drawdown = empyrical.max_drawdown(daily_return)
    # print("max_drawdown", max_drawdown)
    final_df["maxDrawdown"] = max_drawdown.item()

    return final_df

def get_df_date_as_index(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    return get_df(fund_code, start_date, end_date)

def get_df_date_as_colum(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    df = get_df(fund_code, start_date, end_date)
    df.reset_index(inplace=True)
    return df


def faster_get_df(fund_code: str, connection, start_date: str = datetime.now(), end_date: str = datetime.now()):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    start_quarter = date_to_quarter(start_date)
    end_quarter = date_to_quarter(end_date)
    # print(start_date, end_date, start_quarter, end_quarter)

    type_df = pd.read_sql(f"SELECT qilai_fund_type, qilai_fund_invest_type FROM fund_info WHERE fund_code = '{fund_code}'", connection)
    net_value_df = pd.read_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc", connection)
    net_value_df["date"] = pd.to_datetime(net_value_df["date"])
    if type_df.empty:
        print("no such fund")
        return pd.DataFrame()
    elif net_value_df.empty:
        print("no net value data in database")
        return pd.DataFrame()

    usdind_df = pd.read_sql(f"SELECT * FROM usdind WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' ORDER BY trade_date asc", connection)
    usdind_df["trade_date"] = pd.to_datetime(usdind_df["trade_date"])
    gdp_df = pd.read_sql(f"SELECT * FROM gdp WHERE quarter BETWEEN '{start_quarter}' AND '{end_quarter}' ORDER BY quarter asc", connection)
    shibor_df = pd.read_sql(f"SELECT * FROM shibor WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc", connection)
    shibor_df["date"] = pd.to_datetime(shibor_df["date"])
    VIX_df = pd.read_sql(f"SELECT * FROM VIX_index WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc", connection)
    VIX_df["date"] = pd.to_datetime(VIX_df["date"])
    # print(usdind_df, gdp_df, shibor_df, net_value_df)
    
    net_value_df['fund_type'] = type_df['qilai_fund_type'].iloc[0] + "-" + type_df['qilai_fund_invest_type'].iloc[0]
    net_value_df.set_index("date", inplace=True)

    # # 总收益率（基金建立日期 到 end_date）
    # init_net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' ORDER BY date asc LIMIT 1")
    # init_net_value = 1.0000 if init_net_value_df.empty else init_net_value_df["adj_nav"][0]
    # print(init_net_value)

    # # 阶段收益率（start_date 到 end_date）
    # init_net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc LIMIT 1")
    # init_net_value = 1.0000 if init_net_value_df.empty else init_net_value_df["adj_nav"][0]
    # print(init_net_value)

    # final_df["cumulative"] = (final_df["adj_nav"] - init_net_value) / init_net_value
    # print(final_df)

    daily_return = net_value_df.copy()
    daily_return['daily_return'] = daily_return['adj_nav'].pct_change()
    daily_return.drop(columns=['fund_code', 'fund_type', 'id', 'adj_nav'], inplace=True, errors='ignore')
    # print("daily_return", daily_return)

    cum_returns = empyrical.cum_returns(daily_return)
    # print("cum_returns", cum_returns)
    net_value_df["cumulative"] = cum_returns

    annual_volatility = empyrical.annual_volatility(daily_return)
    # print("annual_volatility", annual_volatility)
    net_value_df["annual_volatility"] = annual_volatility[0]

    stability = empyrical.stability_of_timeseries(daily_return)
    # print("stability", stability)
    net_value_df["stability"] = stability

    monthly_return = empyrical.aggregate_returns(daily_return, "monthly")
    monthwin = (monthly_return > 0).sum() / len(monthly_return)
    # print("monthwin", monthwin)
    net_value_df["monthwin"] = monthwin.item()

    daywin = (daily_return > 0).sum() / len(daily_return)
    # print("daywin", daywin)
    net_value_df["winning_day"] = daywin.item()

    max_drawdown = empyrical.max_drawdown(daily_return)
    # print("max_drawdown", max_drawdown)
    net_value_df["maxDrawdown"] = max_drawdown.item()

    net_value_df.reset_index(inplace=True)
    merged_df = pd.merge(net_value_df, usdind_df, left_on='date', right_on='trade_date', how='inner')
    merged_df['quarter'] = merged_df['date'].apply(date_to_quarter)
    merged_df = pd.merge(merged_df, gdp_df, on='quarter', how='inner')
    merged_df = pd.merge(merged_df, VIX_df, left_on='date', right_on='date', how='inner')
    final_df = pd.merge(merged_df, shibor_df, left_on='date', right_on='date', how='inner')
    final_df.drop(columns=['trade_date', 'quarter', 'id'], inplace=True, errors='ignore')
    return final_df

def faster_get_df_date_as_colum(fund_codes: list, start_date: str = datetime.now(), end_date: str = datetime.now()):
    connection = get_db_connection()
    
    res = pd.DataFrame()

    if not connection:
        raise Exception("Database connection failed")
    try:
        for fund_code in fund_codes:
            df = faster_get_df(fund_code, connection, start_date, end_date)
            # df.reset_index(inplace=True)
            df = df.reindex(columns=['fund_code', 'fund_type', 'date', 'adj_nav', 'usdind_close', 'gdp', 'gdp_yoy', 'pi_yoy', 'si_yoy', 'ti_yoy', '1w', '2w', '1m', '3m', '6m', '9m', '1y', 'cumulative', 'annual_volatility', 'stability', 'monthwin', 'winning_day', 'maxDrawdown', "VIX_index"])
            res = pd.concat([res, df], ignore_index=True)
    except Error as e:
        raise Exception(f"Database error: {e}")
    finally:
        if connection.is_connected():
            connection.close()
    return res
