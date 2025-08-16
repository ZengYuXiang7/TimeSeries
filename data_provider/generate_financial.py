import pandas as pd
from sqlalchemy import create_engine, text
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import empyrical


def get_all_fund_list():
    # 数据库配置
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)
    engine = create_engine(DB_URI)

    create_date = '2023-7-13'
    date_list = create_date.split('-')
    base_date = str(int(date_list[0]) - 1) + '-' + date_list[1] + '-' + date_list[2]  # 筛选成立1年以上的基金
    first_date = str(int(date_list[0]) - 2) + '-' + date_list[1] + '-' + date_list[2]  # 筛选成立1-2年的基金
    second_date = str(int(date_list[0]) - 3) + '-' + date_list[1] + '-' + date_list[2]  # 筛选成立2-3年的基金
    # print(base_date, first_date, second_date)
    sql = f"""
        SELECT fund_code, fund_name, market, survival_status, tu_fund_type, establish_time, tu_invest_type
        FROM b_fund_list
        WHERE establish_time < '{base_date}'
          AND fund_code IN (
            SELECT fund_code
            FROM b_fund_nav
            WHERE date = (
              SELECT MAX(date) FROM b_fund_nav
            )
              AND sub_status NOT LIKE '%%暂停申购%%'
              AND red_status NOT LIKE '%%封闭期%%'
          )
        """
    df = pd.read_sql_query(sql, engine)

    # 剔除定开、货币型基金
    df = df[df['survival_status'] != 'D']
    df = df[~df['fund_name'].str.contains('定开')]
    df = df[~(df['tu_fund_type'] == '货币市场型')]
    df = df.drop(['fund_name'], axis=1)
    df = df.drop(['survival_status'], axis=1)

    # 按成立时间划分，3类：1-2年，2-3年，3年以上
    df['establish_time'] = pd.to_datetime(df['establish_time'], format='%Y-%m-%d')
    df.loc[df['establish_time'] > first_date, 'establish_type'] = 1
    df.loc[(df['establish_type'] != 1) & (df['establish_time'] > second_date), 'establish_type'] = 2
    df.loc[(df['establish_type'] != 1) & (df['establish_type'] != 2), 'establish_type'] = 3
    df = df.drop(['establish_time'], axis=1)

    # 划分基金类型，五类：stock/bond/index(O or E)/other/mix
    df.loc[(df['tu_invest_type'] == '被动指数型') & (df['market'] == 'O'), 'tu_fund_type'] = 'index_O'
    df.loc[(df['tu_invest_type'] == '被动指数型') & (df['market'] == 'E'), 'tu_fund_type'] = 'index_E'
    df.loc[df['tu_fund_type'] == '股票型', 'tu_fund_type'] = 'stock'
    df.loc[df['tu_fund_type'] == '债券型', 'tu_fund_type'] = 'bond'
    df.loc[df['tu_fund_type'] == '混合型', 'tu_fund_type'] = 'mix'
    df.loc[(df['tu_fund_type'] != 'stock') & (df['tu_fund_type'] != 'bond') &
           (df['tu_fund_type'] != 'mix') & (df['tu_fund_type'] != 'index_E') &
           (df['tu_fund_type'] != 'index_O'), 'tu_fund_type'] = 'other'

    df = df.drop(['tu_invest_type'], axis=1)
    df = df.drop(['market'], axis=1)
    code_list = df['fund_code']

    # 2025年07月04日20:10:32，暂时用这里
    with open('./datasets/all_code_list.pkl', 'rb') as f:
        code_list = pickle.load(f)
    return code_list


def query_fund_data(fund, start_date, end_date):
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)
    engine = create_engine(DB_URI)
    """查询数据库中某支基金的净值数据
        SELECT fund_code, date, nav, accnav, adj_nav
    """
    sql = text("""
        SELECT fund_code, date, accnav, adj_nav, nav
        FROM b_fund_nav_details_new
        WHERE fund_code IN :codes
          AND date BETWEEN :start AND :end
        ORDER BY date
    """)
    try:
        df = pd.read_sql_query(
            sql.bindparams(codes=tuple([fund]), start=start_date, end=end_date),
            engine
        )
        return df
    except Exception as e:
        print(f"[{fund}] 数据库查询失败: {str(e)}")
        return pd.DataFrame()
    

def query_fund_data(fund, start_date, end_date):
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)
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


def save_fund_data(df, fund, index, start_date, end_date):
    """保存为本地 pickle 文件"""
    dir_name = 'S' + (start_date + '_E' + end_date).replace('-', '')
    filepath = f'./datasets/financial/{dir_name}/{fund}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)
        print(f'{index}: {filepath} 存储完毕')


# process_fund(0, fund_code, config.start_date, config.end_date）
def process_fund(index, fund, start_date, end_date):
    df = query_fund_data(fund, start_date, end_date)
    if df.empty:
        return
    df = process_date_columns(df)
    # save_fund_data(df, fund, index, start_date, end_date)
    return df


def generate_data(start_date, end_date):
    # start_date, end_date = '2020-07-13', '2025-03-08'  # 注意日期格式统一
    code_list = get_all_fund_list()
    dir_name = 'S' + (start_date + '_E' + end_date).replace('-', '')
    os.makedirs(f'./datasets/financial/{dir_name}', exist_ok=True)
    print(f'共需处理基金数量：{len(code_list)}')

    # 线性处理方式
    for i, fund in enumerate(code_list):
        df = query_fund_data([fund], start_date, end_date)[fund]
        if df.empty:
            continue
        df = process_date_columns(df)
        save_fund_data(df, fund, i, start_date, end_date)

    # 多线程处理
    # print(f'共需处理基金数量：{len(code_list)}')
    # from concurrent.futures import ThreadPoolExecutor

    # # 获取 CPU 核心数，并计算线程数（取至少为1）
    # cpu_count = os.cpu_count()
    # max_workers = max(1, cpu_count // 4)
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [
    #         executor.submit(process_fund, i, fund, start_date, end_date)
    #         for i, fund in enumerate(code_list)
    #     ]
    #     for future in as_completed(futures):
    #         # 可加入异常处理反馈
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"线程执行出错: {e}")
    engine.dispose()
    return True


if __name__ == '__main__':
    generate_data('2020-07-13', '2025-06-28')
