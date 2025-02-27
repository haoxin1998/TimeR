import pandas as pd
from datetime import datetime
import numpy as np
def get_historical_future_data(csv_path, window_size, significant_figures=2, month=1):
    """
    读取CSV文件并返回指定时间点前后的OT数据
    
    参数:
    csv_path (str): CSV文件路径
    window_size (int): 需要返回的数据长度
    significant_figures (int): 保留的有效数字位数，默认为2
    month (int): 控制future数据范围的月数，默认为1
        - month=1: future范围截至2023年11月底
        - month=3: future范围截至2024年1月底
        - month=6: future范围截至2024年4月底
    
    返回:
    tuple: (historical_data, future_data)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    date_column = None
    for col in ['end_date']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        raise ValueError(f"找不到日期列，现有列: {df.columns.tolist()}")
    
    # 转换日期列为datetime格式
    df[date_column] = pd.to_datetime(df[date_column])
    
    # 设置历史数据的分界点
    cutoff_date = pd.Timestamp('2023-10-31')
    
    # 根据month参数设置future数据的结束日期
    if month == 1:
        future_end_date = pd.Timestamp('2023-11-30')
    elif month == 3:
        future_end_date = pd.Timestamp('2024-01-31')
    elif month == 6:
        future_end_date = pd.Timestamp('2024-04-30')
    else:
        raise ValueError("month参数必须为1、3或6")
    
    # 获取历史数据
    historical_data = df[df['end_date'] <= cutoff_date].sort_values('end_date', ascending=False)
    historical_data = historical_data.head(window_size)
    historical_data['OT'] = historical_data['OT'].apply(lambda x: float(f'%.{significant_figures}g' % x))
    historical_dict = dict(zip(historical_data['end_date'].dt.strftime('%Y-%m-%d'), 
                             historical_data['OT']))
    
    # 获取future数据（添加结束日期限制）
    future_data = df[(df['end_date'] > cutoff_date) & (df['end_date'] <= future_end_date)].sort_values('end_date')
    future_data = future_data.head(window_size)
    future_data['OT'] = future_data['OT'].apply(lambda x: float(f'%.{significant_figures}g' % x))
    future_dict = dict(zip(future_data['end_date'].dt.strftime('%Y-%m-%d'), 
                          future_data['OT']))
    
    return historical_dict, future_dict
def get_recent_historical_data(historical_dict, k):
    """
    从历史数据字典中获取最近的k个数据点并格式化为字符串
    
    参数:
    historical_dict (dict): 包含历史数据的字典 {date: OT}
    k (int): 需要返回的最近数据点数量
    
    返回:
    str: 格式化后的字符串，包含最近k个数据点，按时间升序排列（由远及近）
    """
    # 将日期字符串转换为datetime对象进行排序
    sorted_items = sorted(
        historical_dict.items(),
        key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'),
        reverse=False  # 改为升序排列，较早的日期在前
    )[-k:]  # 取最后k个（最近的k个数据点）
    
    # 将排序后的数据转换为字符串格式
    formatted_data = [f"{date}: {value}" for date, value in sorted_items]
    
    # 用分号连接所有字符串
    return "; ".join(formatted_data)
def get_historical_future_data_v0(csv_path, window_size, significant_figures=2):
    """
    读取CSV文件并返回指定时间点前后的OT数据
    
    参数:
    csv_path (str): CSV文件路径
    window_size (int): 需要返回的数据长度
    significant_figures (int): 保留的有效数字位数，默认为2
    
    返回:
    tuple: (historical_data, future_data)
        - historical_data: 字典，包含历史数据 {date: OT}
        - future_data: 字典，包含未来数据 {date: OT}
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    date_column = None
    for col in ['end_date']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        raise ValueError(f"找不到日期列，现有列: {df.columns.tolist()}")
    
    # 转换日期列为datetime格式
    df[date_column] = pd.to_datetime(df[date_column])

    
    # 设置分界点
    cutoff_date = pd.Timestamp('2023-10-31')
    
    # 获取历史数据
    historical_data = df[df['end_date'] <= cutoff_date].sort_values('end_date', ascending=False)
    historical_data = historical_data.head(window_size)
    # 保留指定有效数字
    historical_data['OT'] = historical_data['OT'].apply(lambda x: float(f'%.{significant_figures}g' % x))
    historical_dict = dict(zip(historical_data['end_date'].dt.strftime('%Y-%m-%d'), 
                             historical_data['OT']))
    
    # 获取未来数据
    future_data = df[df['end_date'] > cutoff_date].sort_values('end_date')
    future_data = future_data.head(window_size)
    # 保留指定有效数字
    future_data['OT'] = future_data['OT'].apply(lambda x: float(f'%.{significant_figures}g' % x))
    future_dict = dict(zip(future_data['end_date'].dt.strftime('%Y-%m-%d'), 
                          future_data['OT']))
    
    return historical_dict, future_dict

def get_historical_text_data(csv_path, k):
    """
    从文本数据中获取历史数据
    
    参数:
    csv_path (str): 搜索CSV文件的路径
    window_size (int): 窗口大小
    k (int): 需要返回的样本数量
    
    返回:
    dict: 包含历史文本数据的字典 {date: fact}
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 转换日期列为datetime格式
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    # 计算日期中点
    df['mid_date'] = df['start_date'] + (df['end_date'] - df['start_date']) / 2
    
    # 使用固定的截止日期：2023年10月31日
    cutoff_date = pd.Timestamp('2023-10-31')
    
    # 筛选日期早于等于cutoff_date的数据
    historical_data = df[df['end_date'] <= cutoff_date].copy()
    
    # 按日期排序并获取最近的k个样本
    historical_data = historical_data.sort_values('end_date', ascending=False).head(k)
    
    # 创建返回字典，只包含fact不为NA的数据
    result_dict = {}
    for _, row in historical_data.iterrows():
        if pd.notna(row['fact']) and row['fact'] != 'NA':
            result_dict[row['mid_date'].strftime('%Y-%m-%d')] = row['fact']
    
    return result_dict