# Rec4Time: Time Series Forecasting with LLMs

Rec4Time is a framework for time series forecasting using Large Language Models (LLMs) and Large Reasoning Models (LRMs). It supports various reasoning strategies and both unimodal and multimodal approaches to time series prediction.

## Features

- Support for multiple LLM providers (OpenAI, Google Gemini, DeepSeek)
- Multiple reasoning strategies (naive, chain-of-thought, self-consistency, self-correction)
- Unimodal and multimodal prediction approaches
- Comprehensive evaluation metrics
- Experiment tracking and result visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Rec4Time.git
cd Rec4Time
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Open `A_Demo.py` or `A_Run_Exp.py` and add your OpenRouter API key:
   ```python
   openrouter_api_key = "your_openrouter_api_key"  # Replace with your API key
   ```

## Quick Start

### Running a Demo

To run a simple demo with default parameters:

```bash
python A_Demo.py
```

### Running Experiments

For more control over experiment parameters, use the `A_Run_Exp.py` script:

```bash
python A_Run_Exp.py --future_months 6 --data_ids 0 1 2 --llm_ids 0 --multimodal 1
```

## Command-line Arguments

`A_Run_Exp.py` supports the following command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--future_months` | List of future months to predict | `[6]` |
| `--data_ids` | List of dataset IDs to use | `[0]` |
| `--llm_ids` | List of LLM IDs to use | `[0]` |
| `--multimodal` | Whether to run multimodal experiments (0=No, 1=Yes) | `[0, 1]` |
| `--repeat_times` | Number of times to repeat each experiment | `3` |
| `--lookback_window_size` | Size of historical data window | `96` |
| `--significant_figures` | Number of significant figures for data | `5` |
| `--k_text` | Number of text samples to use | `10` |
| `--unimodal_methods` | Unimodal reasoning strategies | `["naive", "cot", "self_consistency", "self_correction", "lrm"]` |
| `--multimodal_methods` | Multimodal reasoning strategies | `["naive", "cot", "self_consistency", "self_correction", "lrm"]` |

## Dataset IDs

The following datasets are available:

| ID | Dataset Name |
|----|--------------|
| 0 | the US Disaster and Emergency Grant |
| 1 | the US Drought Level |
| 2 | the US Inflation Rate |
| 3 | the US Interest Rate |
| 4 | the US Retail Sales |
| 5 | the US Stock Market Index |
| 6 | the US Unemployment Rate |
| 7 | the US Travel Volume |

## LLM IDs

The following sytem 1 and system 2 models are supported:

| ID | Models |
|----|--------------|
| 0 | OpenAI (GPT-4o & o1-mini) |
| 1 | Google (Gemini-2.0-flash-thinking & Gemini-2.0-flash-001) |
| 2 | DeepSeek (deepseek-v3 & deepseek-r1) |

## Reasoning Strategies

- `naive`: Direct system 1 generation
- `cot`: Chain-of-thought reasoning
- `self_consistency`: Multiple predictions with averaging
- `self_correction`: Self-correction of initial predictions
- `lrm`: Using System 2 Models, also known as Large Reasoning Models

## Experiment Results

Experiment results are saved in the `experiment_results` directory in JSON format. 

## Collecting Results

To show experiment results, use the Jupyter notebook `A_DrawLatex.ipynb`. It will show the results in the form of LaTeX tables.



### Custom Data

To use your own data, prepare CSV files with the following format:
- Time series data: Date column and value column
- Text data: Date column and fact/news column

Place your data files in the `data` directory.

### Custom Models

To use different LLM providers, modify the `LLM_list` and `LRM_list` variables in `A_Demo.py` or `A_Run_Exp.py`:

```python
LRM_list = ["openai/o1-mini-2024-09-12", "google/gemini-2.0-flash-thinking-exp-1219:free", "deepseek/deepseek-r1"]
LLM_list = ['openai/gpt-4o-2024-05-13', "google/gemini-2.0-flash-001", "deepseek/deepseek-chat"]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
