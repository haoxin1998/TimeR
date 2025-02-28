<p align="center">
  <img src="https://github.com/AdityaLab/OpenTimeR/blob/main/Picture/Logo-2-Rec4TS.jpg" alt="Rec4TS-Logo" style="width: 15%; display: block; margin: auto;">
</p>

<h1 align="center">🔥 Rec4TS: Evaluating System 1 vs. 2 Reasoning Approaches for Zero-Shot Time-Series Forecasting 🔥</h1>
<p align="center">
  <a href="https://arxiv.org/abs/xxxx"><img src="https://img.shields.io/badge/arXiv-2405.01535-b31b1b.svg" alt="arXiv"></a>
</p>


REC4TS is the first benchmark that evaluates the effectiveness of reasoning strategies for zero-shot time series forecasting (TSF) tasks.

Specifically, REC4TS try to answer two research questions:

-**RQ1: Can zero-shot TSF benefit from enhanced reasoning ability?

-**RQ2: What kind of reasoning strategies does zero-shot TSF need?”

REC4TS covers three cognitive systems: Direct Sytem 1 (e.g. gpt-4o) ; the test-time enhanced System 1 (e.g., gpt-4o with Chain-of-Thought ) and System 2 (e.g. o1-mini).

*Since reasoning strategies for foundational time-series models have not yet been studied and are difficult to implement directly, we have to reuse foundational language models to explore effective TSF
reasoning strategies. We envision that our benchmark and insights offer promising potential for future research on understanding and designing effective reasoning strategies for zero-shot TSF.
<div align="center">
    <img src="https://github.com/AdityaLab/OpenTimeR/blob/main/Picture/reasoning_system_1.jpg" width="800">
</div>

## Key  Insights

0. Good News: reasoning is helpful for zero-shot TSF!
1. Self-consistency is currently the most effective reasoning strategy
2. Group-relative policy optimization enbaled DeepSeek-R1 is the only effective System 2 reasoning strategy
3. Multimodal TSF benefits more from reasoning strategies than unimodal TSF
4. The TimeT-hinking dataset:  containing reasoning trajectories of multiple advanced LLMs
5. A new and simple test-time scaling on foundation time-series models：based on self-consistency reasoning strategies and inspired by our insights
   <div align="center">
    <img src="https://github.com/AdityaLab/OpenTimeR/blob/main/Picture/Overall_Result_1.png" width="500">
</div>

## Additional Toolkits

- **TimeThinking**: Contains 1.5K filtered reasoning-annotated time series forecasting samples.
- **Exp_Log**: Records all experimental results from benchmarking, including the complete outputs of various models and performance metrics.

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
| `--llm_ids` | List of LLM IDs to use 0-OpenAI, 1-Gemini, 2-DeepSeek | `[0]` |
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
| 0 | Retail Broiler Composit|
| 1 | Drought Level |
| 2 | International Trade Balance |
| 3 | Gasoline Prices |
| 4 | Influenza Patients Proportio  |
| 5 | Disaster and Emergency Grant|
| 6 | Unemployment Rate |
| 7 | Travel Volume |

## LLM IDs

The following sytem 1 and system 2 models are supported:

| ID | Models |
|----|--------------|
| 0 | OpenAI (System 1: GPT-4o & Sytem 2: o1-mini) |
| 1 | Google (System 1:  Gemini-2.0-flash & System 2:Gemini-2.0-flash-thinking) |
| 2 | DeepSeek (System 1: DeepSeek-v3 & System 2: DeepSeek-R1) |

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


This project is licensed under the [CC-BY](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International) license. This license allows users to share and adapt your dataset as long as they give credit to you.
