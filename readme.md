# Rec4Time: Time Series Forecasting with Large Language Models

Rec4Time is a framework for evaluating the performance of large language models (LLMs) on time series forecasting tasks. It supports both unimodal and multimodal prediction approaches with various prompting techniques.

## Features

- Support for multiple LLM providers (OpenAI, Google Gemini, DeepSeek)
- Unimodal and multimodal prediction capabilities
- Various prompting strategies:
  - Naive prompting
  - Chain-of-thought (CoT)
  - Self-consistency
  - Self-correction
  - Language reasoning models (LRM)
- Comprehensive evaluation metrics (NMSE, NMAE)
- Visualization tools for prediction analysis

## Project Structure

- **TimeThinking**: Contains 1.5K filtered reasoning-annotated time series forecasting samples.
- **Exp_Log**: Records all experimental results from benchmarking, including the complete outputs of various models and performance metrics.



## Usage

1. Set up your API keys in `A_EXP.py`:
2. Run experiments with different configurations:
3. Analyze results in the experiment_results directory

## Supported Datasets

The framework includes several time series datasets:
- US Disaster and Emergency Grant
- US Drought Level
- And more economic indicators

## Evaluation

Results are saved in JSON format with detailed metrics for each prediction method, including:
- NMSE (Normalized Mean Squared Error)
- NMAE (Normalized Mean Absolute Error)
- Statistical summaries across multiple runs

## Visualization

Use the visualization tools in `Tool.py` to plot historical data, actual future values, and model predictions:

## Requirements

- Python 3.10+
- OpenRouter API access for LLM providers
- Required packages: pandas, numpy, matplotlib, requests, json, statistics

## License

This project is licensed under the [CC-BY](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International) license. This license allows users to share and adapt your dataset as long as they give credit to you.