# TwinVoice

TwinVoice 是一个对话系统评测框架，支持多维度的对话能力评估。

## 项目结构

```
TwinVoice/
├── dataset/                    # 评测数据集
│   ├── dimension_2/           
│   │   └── data.jsonl         
│   └── dimension_3/           
│       ├── choices.jsonl      # 多选题数据
│       └── profiles.jsonl     # 角色资料
├── twinvoice/                 # 主代码包
│   ├── api_config.py         # API配置
│   ├── discriminative/       # 判别式评测
│   │   └── dimension_3/     # 维度3评测模块
│   │       └── evaluate.py  # 评测主脚本
│   └── generative/          # 生成式评测
│       └── dimension_3/     # 维度3生成评测
│           ├── gen_step1.py # 生成步骤
│           └── judge_step2.py # 评判步骤
└── result/                   # 评测结果
    ├── discriminative/       # 判别式评测结果
    │   └── dimension_3/     # 维度3评测结果
    │       ├── results.jsonl       # 评测详细结果
    │       ├── wrong_cases.jsonl   # 错误案例
    │       └── capability_report.csv # 能力维度分析
    └── generative/          # 生成式评测结果
        └── dimension_3/     # 维度3生成评测结果
```

## 使用方法

### 1. API配置

1. 复制示例配置文件：
```bash
cp twinvoice/api_config.template.py twinvoice/api_config.py
```

2. 编辑 `twinvoice/api_config.py` 配置API：
```python
# Digital Twin API配置（用于生成和判别式任务）
twin_base_url = 'http://localhost:8005/v1'  # 本地模型服务地址
twin_api_key = 'EMPTY'

# LLM-as-a-Judge API配置（用于生成式评测的评判）
judge_base_url = 'https://svip.xty.app/v1'
judge_api_key = 'your-judge-api-key-here'  # 替换为评判API密钥
```

### 2. 启动本地模型服务

```bash
# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model/Qwen2.5-14B-Instruct \
    --port 8005 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1
```

### 3. Dimension 3评测Pipeline

#### 3.1 判别式评测（Multiple Choice）

```bash
# 基础评测（使用默认路径）
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model Qwen2.5-14B-Instruct \
    --sample 5  # 可选：测试时使用小样本

# 完整评测（指定数据路径）
python -m twinvoice.discriminative.dimension_3.evaluate \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model Qwen2.5-14B-Instruct \
    --report result/discriminative/dimension_3/results.jsonl

# 带能力维度分析的评测
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model Qwen2.5-14B-Instruct \
    --annotations annotated.jsonl \
    --cap-report-csv result/discriminative/dimension_3/capability_report.csv
```

#### 3.2 生成式评测（Generation + Judge）

生成式评测分为两个步骤：

1. **Step 1: 生成回复**
```bash
# 使用Digital Twin生成回复
python -m twinvoice.generative.dimension_3.gen_step1 \
    --input dataset/dimension_3/choices.jsonl \
    --profile dataset/dimension_3/profiles.jsonl \
    --gen_model Qwen2.5-14B-Instruct \
    --out_dir result/generative/dimension_3 \
    --workers 8
```

2. **Step 2: 评判生成质量**
```bash
# 使用Judge模型评判生成质量
python -m twinvoice.generative.dimension_3.judge_step2 \
    --input result/generative/dimension_3/step1_generations_*.jsonl \
    --judge_model gpt-5-chat \
    --workers 8
```

### 主要参数说明

#### 判别式评测参数
- `choices_jsonl`: 多选题数据文件（默认：dataset/dimension_3/choices.jsonl）
- `profile_json`: 角色资料文件（默认：dataset/dimension_3/profiles.jsonl）
- `--model`: 使用的模型名称
- `--sample`: 采样数量（可选）
- `--report`: 评测结果文件（默认：result/discriminative/dimension_3/results.jsonl）
- `--wrong-report`: 错误案例文件（默认：result/discriminative/dimension_3/wrong_cases.jsonl）
- `--temperature`: 采样温度（默认：0.0）
- `--history-max`: 历史对话最大条数（默认：30）
- `--annotations`: 能力标注文件（可选）
- `--cap-report-csv`: 能力维度报告（默认：result/discriminative/dimension_3/capability_report.csv）

#### 生成式评测参数

Step 1 (gen_step1.py):
- `--input`: 输入数据文件（choices.jsonl）
- `--profile`: 角色资料文件（profiles.jsonl）
- `--gen_model`: 生成模型名称
- `--out_dir`: 输出目录
- `--workers`: 并行工作进程数（默认：8）

Step 2 (judge_step2.py):
- `--input`: Step 1生成的结果文件
- `--judge_model`: 评判模型名称（默认：gpt-5-chat）
- `--workers`: 并行工作进程数（默认：8）

### 评测能力维度

- Opinion_Consistency（观点一致性）
- Memory_Recall（记忆回溯）
- Logical_Reasoning（逻辑推理）
- Lexical_Fidelity（词汇忠实度）
- Persona_Tone（人物语气）
- Syntactic_Style（句法风格）

### 注意事项

1. 确保在项目根目录下运行评测命令
2. 运行前确保已正确配置API密钥
3. 本地模型评测需要启动vLLM服务（端口8005）
4. 评测过程支持优雅退出（Ctrl+C）
5. 建议先使用小样本测试配置是否正确
6. 不要提交 `api_config.py` 到版本控制系统

## 输出说明

### 1. 判别式评测输出
- 总体准确率
- 错误案例分析（最多显示20条）
- 详细结果保存在report文件中
- 能力维度报告（如果提供annotations）

### 2. 生成式评测输出

Step 1输出：
- 生成的回复内容
- 生成状态（Success/Failed）
- 详细结果保存在step1_generations_*.jsonl中

Step 2输出：
- 映射准确率（Acc.Gen）
- 生成质量得分（1-5分）
- 归一化得分（0-1分）
- 详细分析报告（包含三个维度：观点一致性、逻辑事实忠实度、风格相似度）