# TwinVoice 本地运行说明

此文档记录个人使用的具体命令，不要上传到git仓库。

## 环境准备

### 1. 启动vLLM服务
```bash
# 在服务器上启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model /common/users/mg1998/models/Qwen2.5-14B-Instruct \
    --port 8005 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1
```

### 2. API配置
```python
# twinvoice/api_config.py 内容：

# Digital Twin API配置（本地vLLM服务）
twin_base_url = 'http://localhost:8005/v1'
twin_api_key = 'EMPTY'

# LLM-as-a-Judge API配置
judge_base_url = 'https://svip.xty.app/v1'
judge_api_key = 'sk-XOG1ygxM4TgeG8cx7b79DcB74aE344Ab93AcB2Cc868f1dB1'
```

## Dimension 2评测命令

### 判别式评测（User Style Matching）

```bash
# 测试运行（5个样本）
python twinvoice/discriminative/dimension_2/evaluate.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --sample 5 \
    --report result/discriminative/dimension_2/results_test.jsonl \
    --temperature 0.0 \
    --history-max 30

# 完整评测
python twinvoice/discriminative/dimension_2/evaluate.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --report result/discriminative/dimension_2/results.jsonl \
    --wrong-report result/discriminative/dimension_2/wrong_cases.jsonl \
    --temperature 0.0 \
    --history-max 30
```

## Dimension 2评测命令

### 1. 判别式评测（User Style Matching）

```bash
# 测试运行（5个样本）
python twinvoice/discriminative/dimension_2/evaluate.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --sample 5 \
    --report result/discriminative/dimension_2/results_test.jsonl \
    --temperature 0.0 \
    --history-max 30

# 完整评测
python twinvoice/discriminative/dimension_2/evaluate.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --report result/discriminative/dimension_2/results.jsonl \
    --wrong-report result/discriminative/dimension_2/wrong_cases.jsonl \
    --temperature 0.0 \
    --history-max 30
```

### 2. 生成式评测

#### Step 1: 生成回复
```bash
# 测试运行（5个样本）
python twinvoice/generative/Dimension_2/gen_step1.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --gen_model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --out_dir result/generative/dimension_2 \
    --workers 8 \
    --sample 5 \
    --temperature 0.0

# 完整运行
python twinvoice/generative/Dimension_2/gen_step1.py \
    --input dataset/dimension_2/conversation_data.jsonl \
    --gen_model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --out_dir result/generative/dimension_2 \
    --workers 8 \
    --temperature 0.0
```

#### Step 2: 评判生成质量
```bash
# 测试运行的评判
python twinvoice/generative/Dimension_2/judge_step2.py \
    --input result/generative/dimension_2/step1_generations_Qwen2.5-14B-Instruct.jsonl \
    --judge_model gpt-5-chat \
    --workers 8 \
    --temperature 0.0

# 完整运行的评判
python twinvoice/generative/Dimension_2/judge_step2.py \
    --input result/generative/dimension_2/step1_generations_Qwen2.5-14B-Instruct.jsonl \
    --judge_model gpt-5-chat \
    --workers 8 \
    --temperature 0.0
```

## Dimension 3评测命令

### 1. 判别式评测（Multiple Choice）

```bash
# 测试运行（5个样本）
python twinvoice/discriminative/dimension_3/evaluate.py \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --sample 5 \
    --report result/discriminative/dimension_3/results_test.jsonl \
    --temperature 0.0

# 完整评测
python twinvoice/discriminative/dimension_3/evaluate.py \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --report result/discriminative/dimension_3/results.jsonl \
    --temperature 0.0
```

### 2. 生成式评测

#### Step 1: 生成回复
```bash
# 测试运行（使用--sample参数）
python twinvoice/generative/Dimension_3/gen_step1.py \
    --input dataset/dimension_3/choices.jsonl \
    --profile dataset/dimension_3/profiles.jsonl \
    --gen_model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --out_dir result/generative/dimension_3 \
    --workers 8 \
    --sample 5

# 完整运行
python twinvoice/generative/Dimension_3/gen_step1.py \
    --input dataset/dimension_3/choices.jsonl \
    --profile dataset/dimension_3/profiles.jsonl \
    --gen_model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' \
    --out_dir result/generative/dimension_3 \
    --workers 8
```

#### Step 2: 评判生成质量
```bash
# 测试运行的评判
python twinvoice/generative/Dimension_3/judge_step2.py \
    --input result/generative/dimension_3/step1_generations_test_*.jsonl \
    --judge_model gpt-5-chat \
    --workers 8

# 完整运行的评判
python twinvoice/generative/Dimension_3/judge_step2.py \
    --input result/generative/dimension_3/step1_generations_*.jsonl \
    --judge_model gpt-5-chat \
    --workers 8
```

## 注意事项

1. 确保vLLM服务已启动且端口可访问
2. 使用绝对路径指定模型位置：`/common/users/mg1998/models/Qwen2.5-14B-Instruct`
3. 测试时使用`--sample 5`参数
4. 生成式评测的两个步骤需要按顺序执行
5. 评判步骤使用的是SVIP的API，需要确保API key可用

## 常用目录

- 模型路径：`/common/users/mg1998/models/Qwen2.5-14B-Instruct`
- 数据目录：
  - Dimension 2: `dataset/dimension_2/`
  - Dimension 3: `dataset/dimension_3/`
- 结果目录：
  - Dimension 2:
    - 判别式：`result/discriminative/dimension_2/`
    - 生成式：`result/generative/dimension_2/`
  - Dimension 3:
    - 判别式：`result/discriminative/dimension_3/`
    - 生成式：`result/generative/dimension_3/`

## 调试技巧

1. 先用`--sample 5`测试配置是否正确
2. 检查vLLM服务日志确认请求是否正常
3. 评测过程可以用Ctrl+C优雅退出
4. 生成的结果文件都是JSONL格式，可以用文本编辑器查看
