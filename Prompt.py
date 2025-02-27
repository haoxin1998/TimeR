import requests
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import statistics
def get_expert_prediction(future, target_name, api_key, model="openai/o3-mini", prompt=None, input_recent_text=None):
    """
    使用requests调用OpenRouter API获取专家预测
    
    参数:
    input_recent_text (str): 历史文本数据
    future (dict): 未来时间范围的字典 {date: value}
    target_name (str): 目标领域名称
    api_key (str): OpenRouter API密钥
    model (str): 使用的模型名称
    
    返回:
    str: API的预测响应
    """
    # 为不同模型设置对应的超参数
    model_params = {
        "openai/o1-mini-2024-09-12": {
            "top_p": 1,
            "temperature": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0
        },
        "google/gemini-2.0-flash-thinking-exp:free": {
            "top_p": 1,
            "temperature": 0.8,
            "repetition_penalty": 1
        },
        "deepseek/deepseek-r1": {
            "top_p": 1,
            "temperature": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0
        },
        "openai/gpt-4o-2024-05-13": {
            "top_p": 1,
            "temperature": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0
        },
        "google/gemini-2.0-flash-exp:free": {
            "top_p": 1,
            "temperature": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0
        },
        "deepseek/deepseek-chat": {
            "top_p": 1,
            "temperature": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0
        }
    }
    
    # 获取未来时间范围
    future_dates = sorted(future.keys())
    start_date = future_dates[0]
    end_date = future_dates[-1]
    
    # 构建prompt
    if prompt is None:
        prompt = f"""As an expert in {target_name}, please provide a rough prediction of the trends 
    from {start_date} to {end_date}.

    Please format your response by breaking it down by month, like this:
    - [Month 1]: [Your prediction]
    - [Month 2]: [Your prediction]
    - ...

    Recent related news and information:
    {input_recent_text}"""

    # 获取当前模型的超参数
    params = model_params.get(model, {
        "top_p": 1,
        "temperature": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "repetition_penalty": 1,
        "top_k": 0
    })
    
    # 构建请求数据
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    # 添加模型特定的参数
    data.update(params)
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/your-username/your-repo",  # 替换为你的网站URL
                "X-Title": "MM-TSFlib"  # 替换为你的站点名称
            },
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API调用失败: HTTP {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"API调用出错: {str(e)}"
def get_prediction_from_historical(historical, future, target_name, api_key, model="openai/gpt-4o-2024-05-13", 
                                 use_context=False, context_prediction=None):
    """
    根据历史数据获取LLM对未来的预测
    
    参数:
    historical (dict): 历史数据字典 {date: value}
    future (dict): 未来时间范围字典 {date: value}
    target_name (str): 预测目标名称
    api_key (str): API密钥
    model (str): 模型名称
    use_context (bool): 是否使用上下文预测作为参考
    context_prediction (str): 上下文预测结果
    
    返回:
    dict: 预测结果字典 {date: predicted_value}
    """
    # 格式化历史数据
    historical_str = "; ".join([f"{date}: {value}" for date, value in sorted(historical.items())])
    future_dates = sorted(future.keys())
    
    context_str = ""
    if use_context and context_prediction:
        context_str = f"""
Reference Prediction:
{context_prediction}

Please consider the trends from the above reference prediction, but don't rely on it completely. 
Your prediction should be primarily based on historical data while incorporating insights from the reference prediction.
"""
    
    prompt = f"""As an expert in {target_name}, please predict the trends from {future_dates[0]} to {future_dates[-1]} 
based on the following historical data.

Historical data (in chronological order):
{historical_str}{context_str}

Please provide your analysis if needed, but ensure your final predictions are enclosed between ## markers and strictly follow this format:
##
2024-01-01: 123.45
2024-02-01: 124.56
##
"""
    
    try:
        response = get_expert_prediction(prompt=prompt, future=future, target_name=target_name, api_key=api_key, model=model)
        
        # 解析预测结果
        predictions = {}
        for line in response.strip().split('\n'):
            if ':' in line:
                date, value = line.split(':')
                date = date.strip()
                try:
                    value = float(value.strip())
                    if date in future:  # 只保留future中存在的日期
                        predictions[date] = value
                except ValueError:
                    continue
                    
        return predictions
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        return {}
# ... existing code ...

def get_unimodal_prediction(historical, future, target_name, api_key, model_ID, method="naive"):
    """
    获取单模态预测结果
    
    参数:
    method: 可选 "naive", "cot", "self_consistency", "self_correction", "lrm"
    """
    LRM_list=["openai/o1-mini-2024-09-12","google/gemini-2.0-flash-thinking-exp-1219:free","deepseek/deepseek-r1"]
    LLM_list=['openai/gpt-4o-2024-05-13',"google/gemini-2.0-flash-001","deepseek/deepseek-chat"]
    
    historical_str = "; ".join([f"{date}: {value}" for date, value in sorted(historical.items())])
    future_dates = sorted(future.keys())
    
    base_prompt = f"""As an expert in {target_name}, predict the trends from {future_dates[0]} to {future_dates[-1]} based on the historical data.

Historical data (chronological order):
{historical_str}

Please enclose your final predictions between [PRED_START] and [PRED_END] markers exactly like this:
[PRED_START]
2024-01-01: 123.45
2024-02-01: 124.56
[PRED_END]"""
    model=LLM_list[model_ID]
    if method == "naive":
        prompt = base_prompt
        return get_single_prediction(prompt, future, target_name, api_key, model)
    
    elif method == "cot":
        prompt = base_prompt + "\n\nLet's approach this step by step:\n1. Analyze historical trends\n2. Identify patterns\n3. Make predictions"
        return get_single_prediction(prompt, future, target_name, api_key, model)
    
    elif method == "self_consistency":
        predictions = []
        for _ in range(3):
            pred = get_single_prediction(base_prompt, future, target_name, api_key, model)
            predictions.append(pred)
        return average_predictions(predictions)
    
    elif method == "self_correction":
        current_pred = get_single_prediction(base_prompt, future, target_name, api_key, model)
        for _ in range(2):
            correction_prompt = f"{base_prompt}\n\nPrevious prediction:\n{format_prediction(current_pred)}\n\nPlease review and improve the prediction."
            current_pred = get_single_prediction(correction_prompt, future, target_name, api_key, model)
        return current_pred
    
    elif method == "lrm":
        model=LRM_list[model_ID]
        prompt = base_prompt
        return get_single_prediction(prompt, future, target_name, api_key, model)

def get_multimodal_prediction(historical, future, target_name, context_text, api_key, model_ID, method="naive"):
    """
    获取多模态预测结果
    
    参数:
    method: 可选 "naive", "cot", "self_consistency", "self_correction", "lrm"
    """
    historical_str = "; ".join([f"{date}: {value}" for date, value in sorted(historical.items())])
    future_dates = sorted(future.keys())
    
    base_prompt = f"""As an expert in {target_name}, predict the trends from {future_dates[0]} to {future_dates[-1]} 
based on both historical data and contextual information.

Historical data (chronological order):
{historical_str}

Contextual information:
{context_text}

Please enclose your final predictions between [PRED_START] and [PRED_END] markers exactly like this:
[PRED_START]
2024-01-01: 123.45
2024-02-01: 124.56
[PRED_END]"""
    LRM_list=["openai/o1-mini-2024-09-12","google/gemini-2.0-flash-thinking-exp-1219:free","deepseek/deepseek-r1"]
    LLM_list=['openai/gpt-4o-2024-05-13',"google/gemini-2.0-flash-001","deepseek/deepseek-chat"]
    model=LLM_list[model_ID]
    if method == "naive":
        prompt = base_prompt
        return get_single_prediction(prompt, future, target_name, api_key, model)
    
    elif method == "cot":
        prompt = base_prompt + "\n\nLet's approach this step by step:\n1. Analyze historical trends\n2. Consider context impact\n3. Make predictions"
        return get_single_prediction(prompt, future, target_name, api_key, model)
    
    elif method == "self_consistency":
        predictions = []
        for _ in range(3):
            pred = get_single_prediction(base_prompt, future, target_name, api_key, model)
            predictions.append(pred)
        return average_predictions(predictions)
    
    elif method == "self_correction":
        current_pred = get_single_prediction(base_prompt, future, target_name, api_key, model)
        for _ in range(2):
            correction_prompt = f"{base_prompt}\n\nPrevious prediction:\n{format_prediction(current_pred)}\n\nPlease review and improve the prediction."
            current_pred = get_single_prediction(correction_prompt, future, target_name, api_key, model)
        return current_pred
    
    elif method == "lrm":
        model=LRM_list[model_ID]
        prompt = base_prompt
        return get_single_prediction(prompt, future, target_name, api_key, model)

# 辅助函数
def get_single_prediction(prompt, future, target_name, api_key, model):
    """获取单次预测结果"""
    response = get_expert_prediction(prompt=prompt, future=future, target_name=target_name, api_key=api_key, model=model)
    #print(response)
    return parse_prediction_response(response, future)

# def average_predictions(predictions_list):
#     """对多次预测结果取平均"""
#     if not predictions_list:
#         return {}
    
#     all_dates = set()
#     for pred in predictions_list:
#         all_dates.update(pred.keys())
    
#     averaged_pred = {}
#     for date in all_dates:
#         values = [pred.get(date, 0) for pred in predictions_list]
#         averaged_pred[date] = sum(values) / len(values)
    
#     return averaged_pred

def format_prediction(prediction_dict):
    """格式化预测结果为字符串"""
    return "\n".join([f"{date}: {value}" for date, value in sorted(prediction_dict.items())])

# def parse_prediction_response(response, future):
#     """解析API响应为预测字典"""
#     predictions = {}
#     for line in response.strip().split('\n'):
#         if ':' in line:
#             date, value = line.split(':')
#             date = date.strip()
#             try:
#                 value = float(value.strip())
#                 if date in future:
#                     predictions[date] = value
#             except ValueError:
#                 continue
#     return predictions
# def parse_prediction_response(response, future):
#     """解析API响应为预测字典"""
#     predictions = {}
#     if not response or not isinstance(response, str):
#         return predictions
        
#     try:
#         for line in response.strip().split('\n'):
#             if ':' in line:
#                 date, value = line.split(':')
#                 date = date.strip()
#                 try:
#                     value = float(value.strip())
#                     if date in future:
#                         predictions[date] = value
#                 except (ValueError, TypeError):
#                     continue
#     except Exception as e:
#         print(f"Error parsing prediction response: {e}")
        
#     return predictions

def average_predictions(predictions_list):
    """对多次预测结果取中位数"""
    if not predictions_list:
        return {}
    
    all_dates = set()
    for pred in predictions_list:
        if isinstance(pred, dict):  # 添加类型检查
            all_dates.update(pred.keys())
    
    if not all_dates:  # 如果没有有效日期
        return {}
        
    averaged_pred = {}
    for date in all_dates:
        values = [pred.get(date, 0) for pred in predictions_list if isinstance(pred, dict)]
        if values:  # 只在有有效值时计算平均
            #averaged_pred[date] = sum(values) / len(values)
            averaged_pred[date] = statistics.median(values)
    
    return averaged_pred

def parse_prediction_response(response, future):
    """解析API响应为预测字典"""
    predictions = {}
    if not response or not isinstance(response, str):
        return predictions
        
    try:
        # 提取 ## 标记之间的内容
        start_marker = "[PRED_START]"
        end_marker = "[PRED_END]"
        
        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)
        if start_idx == -1 or end_idx == -1:
            return predictions
            
        # 获取标记之间的内容
        prediction_text = response[start_idx + len(start_marker):end_idx].strip()

        
        # 解析预测值
        for line in prediction_text.split('\n'):
            line = line.strip()
            if ':' in line:
                date, value = line.split(':')
                date = date.strip()
                try:
                    value = float(value.strip())
                    if date in future:
                        predictions[date] = value
                except (ValueError, TypeError):
                    continue
                    
    except Exception as e:
        print(f"Error parsing prediction response: {str(e)}")
        
    return predictions