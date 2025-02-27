
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
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
def format_dict_to_string(data_dict, ascending=True):
    """
    将日期-文本字典转换为有序字符串，使用分号分隔
    
    参数:
    data_dict (dict): 包含日期和文本的字典 {date: text}
    ascending (bool): True表示从远到近排序，False表示从近到远排序
    
    返回:
    str: 格式化后的字符串，所有条目用分号分隔
    """
    # ... existing code ...
    sorted_items = sorted(
        data_dict.items(),
        key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'),
        reverse=not ascending
    )
    
    # 构建格式化的字符串列表并用分号连接
    formatted_strings = [f"{date}: {text}" for date, text in sorted_items]
    return "; ".join(formatted_strings)
def format_historical_data(historical_dict):
    """
    将历史数据格式化为中文字符串，自动判断时间单位
    
    参数:
    historical_dict (dict): 包含历史数据的字典 {date: OT}
    
    返回:
    str: 格式化后的中文字符串
    """
    # 将日期字符串转换为datetime对象进行排序
    dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in historical_dict.keys()])
    
    # 获取开始和结束日期
    start_date = dates[0]
    end_date = dates[-1]
    
    # 计算相邻数据点之间的最小时间间隔
    time_diffs = []
    for i in range(1, len(dates)):
        diff = dates[i] - dates[i-1]
        time_diffs.append(diff.days)
    min_diff = min(time_diffs) if time_diffs else 0
    
    # 根据最小时间间隔确定时间单位
    if min_diff >= 360:  # about a year
        time_unit = "year"
        start_str = f"{start_date.year}"
        end_str = f"{end_date.year}"
    elif min_diff >= 28:  # about a month
        time_unit = "month"
        start_str = f"{start_date.strftime('%B %Y')}"
        end_str = f"{end_date.strftime('%B %Y')}"
    elif min_diff >= 7:  # about a week
        time_unit = "week"
        start_week = int(start_date.strftime('%W'))
        end_week = int(end_date.strftime('%W'))
        start_str = f"Week {start_week}, {start_date.year}"
        end_str = f"Week {end_week}, {end_date.year}"
    else:  # days
        time_unit = "day"
        start_str = start_date.strftime('%B %d, %Y')
        end_str = end_date.strftime('%B %d, %Y')
    
    # 按时间顺序获取数据值并转换为字符串
    values = [str(historical_dict[date.strftime('%Y-%m-%d')]) 
             for date in dates]
    
    # 组合最终字符串
    result = f"Historical data from {start_str} to {end_str}, with {time_unit}ly intervals ({min_diff} {time_unit}s): {', '.join(values)}"

    
    return result
# def visualize_predictions(historical, future, prediction, prediction_with_context=None):
#     """
#     可视化历史数据、实际未来数据和预测数据
    
#     参数:
#     historical (dict): 历史数据字典
#     future (dict): 实际未来数据字典
#     prediction (dict): 预测数据字典
#     prediction_with_context (dict): 使用上下文的预测数据字典
#     """
#     # 转换数据为列表
#     hist_dates = [datetime.strptime(d, '%Y-%m-%d') for d in historical.keys()]
#     hist_values = list(historical.values())
    
#     future_dates = [datetime.strptime(d, '%Y-%m-%d') for d in future.keys()]
#     future_values = list(future.values())
    
#     pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction.keys()]
#     pred_values = list(prediction.values())
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(hist_dates, hist_values, 'b-', label='Historical Data')
#     plt.plot(future_dates, future_values, 'g-', label='Actual Future Data', linestyle='-')  # 实线

#     plt.plot(pred_dates, pred_values, 'r--', label='Base Prediction')

#     if prediction_with_context:
#         pred_context_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction_with_context.keys()]
#         pred_context_values = list(prediction_with_context.values())
#         plt.plot(pred_context_dates, pred_context_values, 'y--', label='Context Prediction')

#     # 添加竖线标记LLM的训练截止日期
#     cutoff_date = datetime.strptime('2023-10-31', '%Y-%m-%d')
#     plt.axvline(x=cutoff_date, color='k', linestyle='--')

#     # 在横轴上标注日期和说明
#     plt.annotate('LLMs Cut off', 
#                  xy=(cutoff_date, plt.ylim()[0]), 
#                  xytext=(0, -30), 
#                  textcoords='offset points', 
#                  ha='center', 
#                  va='top', 
#                  fontsize=9, 
#                  color='k',
#                  arrowprops=dict(arrowstyle='-', color='k'))

#     # 设置自定义的x轴刻度
#     ax = plt.gca()
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.gcf().autofmt_xdate()

#     # 获取所有日期并设置刻度标签
#     all_dates = hist_dates + future_dates + pred_dates
#     all_dates = sorted(set(all_dates))  # 去重并排序
#     # 去掉2024-01-01
#     all_dates = [date for date in all_dates if date != datetime.strptime('2024-01-01', '%Y-%m-%d')]
#     # 确保最后一个时间点显示
#     if future_dates:
#         all_dates.append(future_dates[-1])
#     ax.set_xticks(all_dates)

#     plt.title('Time Series Prediction Visualization')
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
def calculate_metrics_v3(historical, future, predictions_dict):
    """
    计算多个预测方法的评估指标
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    predictions_dict (dict): 包含多个预测方法结果的字典 {method_name: prediction_dict}
    
    返回:
    dict: {method_name: {'nmse': value, 'nmae': value}}
    """
    def calc_single_metrics(pred_dict):
        if not pred_dict or not future:
            return {'nmse': float('inf'), 'nmae': float('inf')}
            
        hist_values = np.array(list(historical.values()))
        hist_mean = np.mean(hist_values)
        hist_std = np.std(hist_values) or 1.0
            
        common_dates = sorted(set(future.keys()) & set(pred_dict.keys()))
        
        if not common_dates:
            return {'nmse': float('inf'), 'nmae': float('inf')}
        
        true_values = np.array([future[date] for date in common_dates])
        pred_values = np.array([pred_dict[date] for date in common_dates])
        
        if np.any(np.isnan(true_values)) or np.any(np.isnan(pred_values)):
            return {'nmse': float('inf'), 'nmae': float('inf')}
        
        true_norm = (true_values - hist_mean) / hist_std
        pred_norm = (pred_values - hist_mean) / hist_std
        
        mse = np.mean((true_norm - pred_norm) ** 2)
        mae = np.mean(np.abs(true_norm - pred_norm))
        
        return {'nmse': mse, 'nmae': mae}
    
    metrics = {}
    for method_name, prediction in predictions_dict.items():
        metrics[method_name] = calc_single_metrics(prediction)
    
    return metrics
def calculate_metrics(historical, future, predictions_dict):
    """
    计算多个预测方法的评估指标
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    predictions_dict (dict): 包含多个预测方法结果的字典 {method_name: prediction_dict}
    
    返回:
    dict: {method_name: {'nmse': value, 'nmae': value}}
    """
    metrics = {}
    
    # 计算历史数据的统计值
    hist_values = np.array(list(historical.values()))
    hist_mean = np.mean(hist_values)
    hist_std = np.std(hist_values) if np.std(hist_values) != 0 else 1.0
    
    # 将future转换为有序序列
    future_dates = sorted(future.keys())
    future_seq = np.array([future[date] for date in future_dates])
    
    for method_name, prediction in predictions_dict.items():
        try:
            # 将prediction转换为与future相同长度的序列
            pred_seq = np.zeros(len(future_seq))
            pred_dates = sorted(prediction.keys())
            
            if not pred_dates:
                # 如果没有预测值，用历史均值填充
                pred_seq.fill(hist_mean)
            else:
                # 获取最后一个预测值
                last_pred_value = prediction[pred_dates[-1]]
                
                for i, date in enumerate(future_dates):
                    if date in prediction:
                        pred_seq[i] = prediction[date]
                    else:
                        # 使用最后一个预测值填充
                        pred_seq[i] = last_pred_value
            
            # 归一化处理
            future_norm = (future_seq - hist_mean) / hist_std
            pred_norm = (pred_seq - hist_mean) / hist_std
            
            # 计算指标
            nmse = np.mean((future_norm - pred_norm) ** 2)
            nmae = np.mean(np.abs(future_norm - pred_norm))
            
            metrics[method_name] = {
                'nmse': float(nmse),
                'nmae': float(nmae)
            }
            
        except Exception as e:
            print(f"计算{method_name}指标时出错: {str(e)}")
            metrics[method_name] = {
                'nmse': float('inf'),
                'nmae': float('inf')
            }
    
    return metrics
# def calculate_metrics(historical, future, predictions_dict):
#     """
#     计算多个预测方法的评估指标。首先处理prediction的长度问题，如果跟future长度不同，则使用最后一个值或者hist均值填充到一样的长度，然后就把两个序列无关具体日期，然后对future和pred使用历史均值和标准差归一化，每个位置的mse再取平均值
#     """
#     metrics = {}
    
#     # 计算历史数据的统计值
#     hist_values = np.array(list(historical.values()))
#     hist_mean = np.mean(hist_values)
#     hist_std = np.std(hist_values) if np.std(hist_values) != 0 else 1.0
    
#     for method_name, prediction in predictions_dict.items():
#         try:
#             # 获取共同的日期
#             common_dates = sorted(set(future.keys()) & set(prediction.keys()))
            
#             if not common_dates:
#                 metrics[method_name] = {
#                     'nmse': float('inf'),
#                     'nmae': float('inf')
#                 }
#                 continue
                
#             # 获取真实值和预测值
#             true_values = np.array([future[date] for date in common_dates])
#             pred_values = np.array([prediction[date] for date in common_dates])
            
#             # 标准化
#             true_norm = (true_values - hist_mean) / hist_std
#             pred_norm = (pred_values - hist_mean) / hist_std
            
#             # 计算指标
#             nmse = np.mean((true_norm - pred_norm) ** 2)
#             nmae = np.mean(np.abs(true_norm - pred_norm))
            
#             metrics[method_name] = {
#                 'nmse': float(nmse),  # 确保转换为Python float
#                 'nmae': float(nmae)
#             }
            
#         except Exception as e:
#             print(f"计算{method_name}指标时出错: {str(e)}")
#             metrics[method_name] = {
#                 'nmse': float('inf'),
#                 'nmae': float('inf')
#             }
    
#     return metrics
def visualize_predictions(historical, future, predictions_dict, title=None):
    """
    可视化多个预测方法的结果
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    predictions_dict (dict): 包含多个预测方法结果的字典 {method_name: prediction_dict}
    title (str): 可选的图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制历史数据
    hist_dates = [datetime.strptime(d, '%Y-%m-%d') for d in historical.keys()]
    hist_values = list(historical.values())
    plt.plot(hist_dates, hist_values, 'b-', label='Historical Data')
    
    # 绘制实际未来数据
    future_dates = [datetime.strptime(d, '%Y-%m-%d') for d in future.keys()]
    future_values = list(future.values())
    plt.plot(future_dates, future_values, 'g-', label='Actual Future Data')
    
    # 绘制各种预测结果
    colors = ['r', 'y', 'm', 'c', 'k']  # 为不同预测方法准备不同颜色
    for (method_name, prediction), color in zip(predictions_dict.items(), colors):
        pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction.keys()]
        pred_values = list(prediction.values())
        plt.plot(pred_dates, pred_values, f'{color}--', label=f'{method_name}')
    
    # 添加LLM截止日期标记
    cutoff_date = datetime.strptime('2023-10-31', '%Y-%m-%d')
    plt.axvline(x=cutoff_date, color='k', linestyle='--')
    plt.annotate('LLMs Cut-off Date', 
                 xy=(cutoff_date, plt.ylim()[0]),
                 xytext=(0, -40),
                 textcoords='offset points',
                 ha='center',
                 va='top',
                 fontsize=9,
                 color='k',
                 arrowprops=dict(arrowstyle='-', color='k'))
    
    # 设置x轴格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # 设置标题和标签
    if title:
        plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def calculate_metrics_v0(historical, future, prediction, prediction_with_context=None):
    """
    计算预测误差指标
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    prediction (dict): 预测数据字典
    prediction_with_context (dict): 使用上下文的预测数据字典
    
    返回:
    tuple: (base_metrics, context_metrics) 每个metrics包含(normalized_mse, normalized_mae)
    """
    def calc_single_metrics(pred_dict):
        # 计算历史数据的均值和标准差
        hist_values = np.array(list(historical.values()))
        hist_mean = np.mean(hist_values)
        hist_std = np.std(hist_values)
        
        # 获取共同的日期
        common_dates = sorted(set(future.keys()) & set(pred_dict.keys()))
        
        if not common_dates:
            return None, None
        
        # 提取并归一化实际值和预测值
        true_values = np.array([future[date] for date in common_dates])
        pred_values = np.array([pred_dict[date] for date in common_dates])
        
        # 归一化
        true_norm = (true_values - hist_mean) / hist_std
        pred_norm = (pred_values - hist_mean) / hist_std
        
        # 计算MSE和MAE
        mse = np.mean((true_norm - pred_norm) ** 2)
        mae = np.mean(np.abs(true_norm - pred_norm))
        
        return mse, mae
    
    base_metrics = calc_single_metrics(prediction)
    context_metrics = None
    if prediction_with_context:
        context_metrics = calc_single_metrics(prediction_with_context)
    
    return base_metrics, context_metrics
def visualize_predictions_V0(historical, future, prediction, prediction_with_context=None):
    """
    可视化历史数据、实际未来数据和预测数据
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    prediction (dict): 预测数据字典
    prediction_with_context (dict): 使用上下文的预测数据字典
    """
    # 转换数据为列表
    hist_dates = [datetime.strptime(d, '%Y-%m-%d') for d in historical.keys()]
    hist_values = list(historical.values())
    
    future_dates = [datetime.strptime(d, '%Y-%m-%d') for d in future.keys()]
    future_values = list(future.values())
    
    pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction.keys()]
    pred_values = list(prediction.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(hist_dates, hist_values, 'b-', label='Historical Data')
    plt.plot(future_dates, future_values, 'g-', label='Actual Future Data', linestyle='-')  # 实线

    plt.plot(pred_dates, pred_values, 'r--', label='Base Prediction')

    if prediction_with_context:
        pred_context_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction_with_context.keys()]
        pred_context_values = list(prediction_with_context.values())
        plt.plot(pred_context_dates, pred_context_values, 'y--', label='Context Prediction')

    # 添加竖线标记LLM的训练截止日期
    cutoff_date = datetime.strptime('2023-10', '%Y-%m-%d')
    plt.axvline(x=cutoff_date, color='k', linestyle='--')

    # 在横轴上标注日期和说明
    plt.annotate('LLMs Cut off', 
                 xy=(cutoff_date, plt.ylim()[0]), 
                 xytext=(0, -30), 
                 textcoords='offset points', 
                 ha='center', 
                 va='top', 
                 fontsize=9, 
                 color='k',
                 arrowprops=dict(arrowstyle='-', color='k'))

    # 设置自定义的x轴刻度
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    # 获取所有日期并设置刻度标签
    all_dates = hist_dates + future_dates + pred_dates
    all_dates = sorted(set(all_dates))  # 去重并排序
    # 去掉2024-01-01
    all_dates = [date for date in all_dates if date != datetime.strptime('2024-01-01', '%Y-%m-%d')]
    # 确保最后一个时间点显示
    if future_dates:
        all_dates.append(future_dates[-1])
    ax.set_xticks(all_dates)

    plt.title('Time Series Prediction Visualization')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def visualize_predictions_V1(historical, future, prediction, prediction_with_context=None):
    """
    可视化历史数据、实际未来数据和预测数据
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    prediction (dict): 预测数据字典
    prediction_with_context (dict): 使用上下文的预测数据字典
    """
    # 转换数据为列表
    hist_dates = [datetime.strptime(d, '%Y-%m-%d') for d in historical.keys()]
    hist_values = list(historical.values())
    
    future_dates = [datetime.strptime(d, '%Y-%m-%d') for d in future.keys()]
    future_values = list(future.values())
    
    pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction.keys()]
    pred_values = list(prediction.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(hist_dates, hist_values, 'b-', label='Historical Data')
    plt.plot(future_dates, future_values, 'g-', label='Actual Future Data', linestyle='-')  # 实线

    plt.plot(pred_dates, pred_values, 'r--', label='Base Prediction')

    if prediction_with_context:
        pred_context_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction_with_context.keys()]
        pred_context_values = list(prediction_with_context.values())
        plt.plot(pred_context_dates, pred_context_values, 'y--', label='Context Prediction')

    # 添加竖线标记LLM的训练截止日期
    cutoff_date = datetime.strptime('2023-10-31', '%Y-%m-%d')
    plt.axvline(x=cutoff_date, color='k', linestyle='--')

    # 在横轴上标注日期和说明
    plt.annotate('LLMs Cut off', 
                 xy=(cutoff_date, plt.ylim()[0]), 
                 xytext=(0, -30), 
                 textcoords='offset points', 
                 ha='center', 
                 va='top', 
                 fontsize=9, 
                 color='k',
                 arrowprops=dict(arrowstyle='-', color='k'))

    # 设置自定义的x轴刻度
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()

    # 获取所有日期并设置刻度标签
    all_dates = hist_dates + future_dates + pred_dates
    all_dates = sorted(set(all_dates))  # 去重并排序
    # 去掉2024-01-01
    all_dates = [date for date in all_dates if date != datetime.strptime('2024-01-01', '%Y-%m-%d')]
    # 确保最后一个时间点显示
    if future_dates:
        all_dates.append(future_dates[-1])
    ax.set_xticks(all_dates)

    plt.title('Time Series Prediction Visualization')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def visualize_predictions_V2(historical, future, prediction, prediction_with_context=None):
    """
    可视化历史数据、实际未来数据和预测数据
    
    参数:
    historical (dict): 历史数据字典
    future (dict): 实际未来数据字典
    prediction (dict): 预测数据字典
    prediction_with_context (dict): 使用上下文的预测数据字典
    """
    # 转换数据为列表
    hist_dates = [datetime.strptime(d, '%Y-%m-%d') for d in historical.keys()]
    hist_values = list(historical.values())
    
    future_dates = [datetime.strptime(d, '%Y-%m-%d') for d in future.keys()]
    future_values = list(future.values())
    
    pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction.keys()]
    pred_values = list(prediction.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(hist_dates, hist_values, 'b-', label='Historical Data')
    plt.plot(future_dates, future_values, 'g-', label='Actual Future Data', linestyle='-')  # 实线

    plt.plot(pred_dates, pred_values, 'r--', label='Unimodal Prediction')

    if prediction_with_context:
        pred_context_dates = [datetime.strptime(d, '%Y-%m-%d') for d in prediction_with_context.keys()]
        pred_context_values = list(prediction_with_context.values())
        plt.plot(pred_context_dates, pred_context_values, 'y--', label='Contextual Prediction')

    # 添加竖线标记LLM的训练截止日期
    cutoff_date = datetime.strptime('2023-10-31', '%Y-%m-%d')
    plt.axvline(x=cutoff_date, color='k', linestyle='--')

    # 在横轴上标注日期和说明
    plt.annotate('LLMs Cut-off Date', 
                 xy=(cutoff_date, plt.ylim()[0]), 
                 xytext=(0, -40), 
                 textcoords='offset points', 
                 ha='center', 
                 va='top', 
                 fontsize=9, 
                 color='k',
                 arrowprops=dict(arrowstyle='-', color='k'))

    # 设置自定义的x轴刻度
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    #plt.gcf().autofmt_xdate()

    # 获取所有日期并设置刻度标签
    all_dates = hist_dates + future_dates + pred_dates
    all_dates = sorted(set(all_dates))  # 去重并排序
    # 只保留每年的第一天
    year_start_dates = [date for date in all_dates if date.month == 1 and date.day == 1]
    ax.set_xticks(year_start_dates)

    #plt.title('Visualization of Predictio')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def calculate_metrics_v1(historical, future, prediction, prediction_with_context=None):
    """
    计算预测误差指标，增加了错误处理和边界条件检查
    """
    def calc_single_metrics(pred_dict):
        # 检查输入是否有效
        if not pred_dict or not future:
            return float('inf'), float('inf')
            
        # 计算历史数据的均值和标准差
        hist_values = np.array(list(historical.values()))
        hist_mean = np.mean(hist_values)
        hist_std = np.std(hist_values)
        
        # 防止除零错误
        if hist_std == 0:
            hist_std = 1.0
            
        # 获取共同的日期
        common_dates = sorted(set(future.keys()) & set(pred_dict.keys()))
        
        if not common_dates:
            return float('inf'), float('inf')
        
        # 提取并归一化实际值和预测值
        true_values = np.array([future[date] for date in common_dates])
        pred_values = np.array([pred_dict[date] for date in common_dates])
        
        # 检查数值是否有效
        if np.any(np.isnan(true_values)) or np.any(np.isnan(pred_values)):
            return float('inf'), float('inf')
        
        # 归一化
        true_norm = (true_values - hist_mean) / hist_std
        pred_norm = (pred_values - hist_mean) / hist_std
        
        # 计算MSE和MAE
        mse = np.mean((true_norm - pred_norm) ** 2)
        mae = np.mean(np.abs(true_norm - pred_norm))
        
        return mse, mae
    
    # 计算基础预测的指标
    base_metrics = calc_single_metrics(prediction)
    
    # 计算上下文预测的指标（如果有）
    context_metrics = None
    if prediction_with_context:
        context_metrics = calc_single_metrics(prediction_with_context)
    
    return base_metrics, context_metrics
import os
from datetime import datetime
import json
def save_experiment_results_with_repeats(target_name, future_month, results, llm_family, save_dir="experiment_results",Multi=False):
    """
    保存多次重复实验的结果到JSON文件
    
    参数:
    target_name (str): 目标名称
    future_month (int): 预测月数
    results (dict): 包含所有实验结果的字典，包括historical, future, predictions等
    llm_family (str): 使用的语言模型系列名称
    save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if Multi==1:
        filename = f"{target_name.replace(' ', '_')}_{future_month}m_{llm_family}_Multi.json"
    else:       
        filename = f"{target_name.replace(' ', '_')}_{future_month}m_{llm_family}_Uni.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    return filepath

# def calculate_metrics_statistics(all_metrics):
#     """
#     计算多次重复实验的统计指标
    
#     参数:
#     all_metrics (list): 包含多次重复实验评估指标的列表
    
#     返回:
#     dict: 包含每个方法的均值和标准差的字典
#     """
#     stats = {}
    
#     # 确保有实验结果
#     if not all_metrics:
#         return stats
        
#     # 获取所有预测方法
#     methods = all_metrics[0].keys()
    
#     for method in methods:
#         # 收集该方法的所有nmse和nmae值
#         nmse_values = []
#         nmae_values = []
        
#         for metrics in all_metrics:
#             if method in metrics:
#                 nmse_values.append(metrics[method]['nmse'])
#                 nmae_values.append(metrics[method]['nmae'])
        
#         # 转换为numpy数组以便计算
#         nmse_array = np.array(nmse_values)
#         nmae_array = np.array(nmae_values)
        
#         # 计算均值和标准差
#         stats[method] = {
#             'nmse_mean': float(np.mean(nmse_array)),
#             'nmse_std': float(np.std(nmse_array)),
#             'nmae_mean': float(np.mean(nmae_array)),
#             'nmae_std': float(np.std(nmae_array))
#         }
    
#     return stats
def calculate_metrics_statistics(all_metrics):
    """
    计算多次重复实验的统计指标
    
    参数:
    all_metrics (dict): 格式为 {'repeat_0': {...}, 'repeat_1': {...}, ...}
    
    返回:
    dict: 包含每个方法的均值和标准差
    """
    if not all_metrics:
        return {}
    
    # 获取第一次重复实验的结果来确定所有预测方法
    first_repeat = next(iter(all_metrics.values()))
    methods = first_repeat.keys()
    
    stats = {}
    for method in methods:
        # 收集该方法的所有nmse和nmae值
        nmse_values = []
        nmae_values = []
        
        for repeat_metrics in all_metrics.values():
            if method in repeat_metrics:
                nmse_values.append(repeat_metrics[method]['nmse'])
                nmae_values.append(repeat_metrics[method]['nmae'])
        
        # 计算均值和标准差
        nmse_values = np.array(nmse_values)
        nmae_values = np.array(nmae_values)
        
        stats[method] = {
            'nmse_mean': np.mean(nmse_values),
            'nmse_std': np.std(nmse_values),
            'nmae_mean': np.mean(nmae_values),
            'nmae_std': np.std(nmae_values)
        }
    
    return stats