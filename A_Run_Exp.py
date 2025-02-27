from Get_Data import get_historical_future_data,get_historical_text_data,get_recent_historical_data
from Prompt import get_expert_prediction,get_prediction_from_historical
import numpy as np  
from Get_Data import get_historical_future_data, get_historical_text_data
import argparse

from Tool import (
    format_dict_to_string,
    format_historical_data,
    calculate_metrics,
    visualize_predictions,
    save_experiment_results_with_repeats,
    calculate_metrics_statistics
)
from Prompt import (
    get_expert_prediction,
    get_unimodal_prediction,
    get_multimodal_prediction
)
import json
import os
from datetime import datetime
LLM_Family=["GPT","Gemini","DeepSeek"]

def load_experiment_results(filepath):
    """
    从JSON文件加载实验结果
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results
import time  # 添加time模块导入

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
data_csv_list=[
    "./data/Algriculture/US_RetailBroilerComposite_Month.csv",
    "./data/Climate/US_precipitation_month.csv",
    "./data/Economy/US_TradeBalance_Month.csv",
    "./data/Energy/US_GasolinePrice_Week.csv",
    "./data/Public_Health/US_FLURATIO_Week.csv",
    "./data/Security/US_FEMAGrant_Month.csv",
    "./data/SocialGood/Unadj_UnemploymentRate_ALL_processed.csv",
    "./data/Traffic/US_VMT_Month.csv"

]
search_csv_list=[
    "./textual/Agriculture/Agriculture_search.csv",
    "./textual/Climate/Climate_search.csv",
    "./textual/Economy/Economy_search.csv",
    "./textual/Energy/Energy_search.csv",
    "./textual/Health_US/Health_US_search.csv",
    "./textual/Security/Security_search.csv",
    "./textual/SocialGood/SocialGood_search.csv",
    "./textual/Traffic/Traffic_search.csv"
]
target_name_list=[
"the US Retail Broiler Composit",
"the US Drought Level",
"the US International Trade Balance",
"the US Gasoline Prices",
"the US Influenza Patients Proportion",
"the US Disaster and Emergency Grant",
"the US Unemployment Rate",
"the US Travel Volume"
]
LRM_list=["openai/o1-mini-2024-09-12","google/gemini-2.0-flash-thinking-exp-1219:free","deepseek/deepseek-r1"]
LLM_list=['openai/gpt-4o-2024-05-13',"google/gemini-2.0-flash-001","deepseek/deepseek-chat"]

#API设置
openrouter_api_key = ""##Use your own API key from openrouter
#读取数值数据
def run_experiment_with_repeats(
    Data_ID, 
    LLM_ID, 
    config={
        'repeat_times': 3,
        'window_size': 96,
        'significant_figures': 5,
        'future_month': 6,
        'k': 96,
        'unimodal_methods': ["naive"],  # 可选: ["naive", "cot", "self_consistency", "self_correction", "lrm"]
        'multimodal_methods': ["naive"], # 可选: ["naive", "cot", "self_consistency", "self_correction", "lrm"]
        'run_unimodal': True,
        'run_multimodal': True
    }
):
    """
    对单个实验设置进行多次重复，并返回所有预测结果和统计指标
    
    参数:
    - Data_ID: 数据集ID
    - LLM_ID: 语言模型ID
    - config: 实验配置字典，包含所有可配置参数
    """
    target_name = target_name_list[Data_ID]
    data_csv_path = data_csv_list[Data_ID]
    search_csv_path = search_csv_list[Data_ID]
    
    # 存储所有重复实验的结果
    all_predictions = {f"repeat_{i}": {} for i in range(config['repeat_times'])}
    all_metrics = {f"repeat_{i}": {} for i in range(config['repeat_times'])}
    
    for repeat_i in range(config['repeat_times']):
        print(f"\nRepeat {repeat_i + 1}/{config['repeat_times']}")
        
        # 获取数据
        historical, future = get_historical_future_data(
            data_csv_path, 
            config['window_size'], 
            config['significant_figures'], 
            config['future_month']
        )
        text_data = get_historical_text_data(search_csv_path, config['k'])
        input_recent_text = format_dict_to_string(text_data, ascending=True)
        
        predictions = {}
        
        # 单模态预测
        if config['run_unimodal']==True:
            for method in config['unimodal_methods']:
                predictions[f"Unimodal_{method}"] = get_unimodal_prediction(
                    historical=historical,
                    future=future,
                    target_name=target_name,
                    api_key=openrouter_api_key,
                    model_ID=LLM_ID,
                    method=method
                )
                # 在每次实验结束后，如果是Gemini模型则暂停60秒
                if LLM_ID == 1:  # Gemini
                    print("使用Gemini模型，暂停60秒...")
                    time.sleep(60)
        # 多模态预测
        if config['run_multimodal']==1:
            for method in config['multimodal_methods']:
                predictions[f"Multimodal_{method}"] = get_multimodal_prediction(
                    historical=historical,
                    future=future,
                    target_name=target_name,
                    context_text=input_recent_text,
                    api_key=openrouter_api_key,
                    model_ID=LLM_ID,
                    method=method
                )                                                                                                                       
                # 在每次实验结束后，如果是Gemini模型则暂停60秒
                if LLM_ID == 1:  # Gemini
                    print("使用Gemini模型，暂停60秒...")
                    time.sleep(60)
        # 计算当次实验的指标
        metrics = calculate_metrics(historical, future, predictions)
        
        # 保存当次实验结果
        all_predictions[f"repeat_{repeat_i}"] = predictions
        all_metrics[f"repeat_{repeat_i}"] = metrics
    
    # 计算统计指标
    stats = calculate_metrics_statistics(all_metrics)
    
    return {
        "historical": historical,
        "future": future,
        "all_predictions": all_predictions,
        "all_metrics": all_metrics,
        "statistics": stats,
        "config": config  # 保存实验配置
    }
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter configuration for running experiments')
    
    # Add command-line arguments
    parser.add_argument('--future_months', type=int, nargs='+', default=[6],
                        help='List of future months to predict, e.g., --future_months 6 12')
    
    parser.add_argument('--data_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='List of dataset IDs, e.g., --data_ids 0-Agriculture, 1-Climate, 2-Economy, 3-Energy, 4-Health, 5-Security, 6-SocialGood, 7-Traffic')
    
    parser.add_argument('--llm_ids', type=int, nargs='+', default=[2],
                        help='List of language model IDs, e.g., --llm_ids 0-OpenAI, 1-Gemini, 2-DeepSeek')
    
    parser.add_argument('--multi_options', type=int, nargs='+', default=[0, 1],
                        help='Modal options, 0 for unimodal, 1 for multimodal, e.g., --multi_options 0 1')
    
    parser.add_argument('--repeat_times', type=int, default=3,
                        help='Number of times to repeat the experiment')
    
    parser.add_argument('--lookback_window_size', type=int, default=96,
                        help='Lookback Window size')
    
    parser.add_argument('--significant_figures', type=int, default=5,
                        help='Number of significant figures')
    
    parser.add_argument('--unimodal_reasoning', type=str, nargs='+', 
                        default=["naive", "cot", "self_consistency", "self_correction", "lrm"],
                        help='Unimodal Reasoning Strategies: sytem1-naive; sytem1 with test-time reasoning:cot, self_consistency, self_correction;sytem2-lrm')
    
    parser.add_argument('--multimodal_reasoning', type=str, nargs='+',
                        default=["naive", "cot", "self_consistency", "self_correction", "lrm"],
                        help='Multimodal Reasoning Strategies')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Build experiment configuration
    experiment_config = {
        'repeat_times': args.repeat_times,
        'window_size': args.lookback_window_size,
        'significant_figures': args.significant_figures,
        'k': args.lookback_window_size,
        'unimodal_reasoning': args.unimodal_reasoning,
        'multimodal_reasoning': args.multimodal_reasoning,
        'run_unimodal': True,
        'run_multimodal': True
    }

    # Run experiments using command-line parameters
    for future_month in args.future_months:
        for Data_ID in args.data_ids:
            for LLM_ID in args.llm_ids:
                for Multi in args.multi_options:
                    # Check experiment configuration
                    target_name = target_name_list[Data_ID]
                    print(f"\nChecking experiment configuration:")
                    print(f"Dataset: {target_name}")
                    print(f"LLM: {LLM_Family[LLM_ID]}")
                    print(f"Mode: {'Multimodal' if Multi == 1 else 'Unimodal'}")

                    # Pre-check if the file exists
                    test_filename = f"{target_name.replace(' ', '_')}_{future_month}m_{LLM_Family[LLM_ID]}_{'Multi' if Multi == 1 else 'Uni'}.json"
                    test_filepath = os.path.join("experiment_results", test_filename)

                    if os.path.exists(test_filepath):
                        print(f"Experiment results already exist, skipping this configuration")
                        continue

                    print("Starting new experiment...")
                    # Run repeated experiments
                    results = run_experiment_with_repeats(
                        Data_ID=Data_ID,
                        LLM_ID=LLM_ID,
                        repeat_times=experiment_config['repeat_times'],
                        window_size=experiment_config['window_size'],
                        significant_figures=experiment_config['significant_figures'],
                        future_month=future_month,
                        k=experiment_config['k'],
                        unimodal_methods=experiment_config['unimodal_methods'],
                        multimodal_methods=experiment_config['multimodal_methods'],
                        run_unimodal=experiment_config['run_unimodal'],
                        run_multimodal=Multi
                    )

                    # Print statistical results
                    print("\n=== Statistical Results ===")
                    for method, stats in results['statistics'].items():
                        print(f"\n{method}:")
                        print(f"NMSE: {stats['nmse_mean']:.4f} ± {stats['nmse_std']:.4f}")
                        print(f"NMAE: {stats['nmae_mean']:.4f} ± {stats['nmae_std']:.4f}")
                        print("-" * 30)

                    # Save experiment results
                    save_experiment_results_with_repeats(
                        target_name=target_name_list[Data_ID],
                        future_month=future_month,
                        results=results,
                        llm_family=LLM_Family[LLM_ID],
                        Multi=Multi
                    )

if __name__ == "__main__":
    main()