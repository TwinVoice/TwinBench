# TwinVoice

TwinVoice is a multi-dimensional benchmark towards digital twins that supports comprehensive assessment of conversational and persona simulation capabilities of Large Language Models.

## Evaluation Dimensions

TwinVoice evaluates persona simulation across three complementary dimensions:

1. **Dimension 1: Social Persona** — Public-facing identity via social media interactions
2. **Dimension 2: Interpersonal Persona** — Private, relational identity via multi-session dialogues
3. **Dimension 3: Narrative Persona** — Role-based expression in fictional scenarios

## Capability Dimensions

TwinVoice evaluates six fundamental capabilities across two main categories:

- **Mindset Coherence**
  - *Opinion_Consistency*: Alignment with previously stated opinions
  - *Memory_Recall*: Accurate retrieval of persona-specific context
  - *Logical_Reasoning*: Coherent reasoning patterns
- **Linguistic Expression**
  - *Lexical_Fidelity*: Use of signature words and phrases
  - *Persona_Tone*: Emotional tone and attitude consistency
  - *Syntactic_Style*: Sentence structure and formatting patterns

## Project Structure

```text
TwinVoice/
├── dataset/                    # Evaluation datasets
│   ├── dimension_1/            # Social Persona (public social interactions)
│   │   └── data.jsonl
│   ├── dimension_2/            # Interpersonal Persona (private dialogues)
│   │   └── conversation_data.jsonl
│   └── dimension_3/            # Narrative Persona (role-based expression)
│       ├── choices.jsonl       # Multiple choice data
│       └── profiles.jsonl      # Character profiles
├── code/                       # Standalone Evaluation Scripts
│   ├── 1_generate_and_judgewithranking.py   # Generative eval: LLM-as-Judge (ranking)
│   ├── 2_judge_with_scoring.py              # Generative eval: LLM-as-Judge (scoring)
│   ├── discriminative_evaluation.py         # Discriminative eval (multiple-choice)
│   └── objective_evaluation.py              # Objective metrics (BLEU, METEOR, BERT-Score)
├── twinvoice/                  # Main package
│   ├── api_config.py           # API configuration
│   ├── discriminative/         # Discriminative evaluation
│   │   ├── dimension_1/
│   │   │   └── evaluate.py
│   │   ├── dimension_2/
│   │   │   └── evaluate.py
│   │   └── dimension_3/
│   │       └── evaluate.py
│   └── generative/             # Generative evaluation
│       ├── dimension_1/
│       │   ├── gen_step1.py
│       │   └── judge_step2.py
│       ├── dimension_2/
│       │   ├── gen_step1.py
│       │   └── judge_step2.py
│       └── dimension_3/
│           ├── gen_step1.py
│           └── judge_step2.py
└── result/                     # Evaluation results
    ├── discriminative/         # Discriminative results
    │   ├── dimension_1/
    │   │   ├── results.jsonl
    │   │   ├── wrong_cases.jsonl
    │   │   └── capability_report.csv
    │   ├── dimension_2/
    │   │   ├── results.jsonl
    │   │   ├── wrong_cases.jsonl
    │   │   └── capability_report.csv
    │   └── dimension_3/
    │       ├── results.jsonl
    │       ├── wrong_cases.jsonl
    │       └── capability_report.csv
    └── generative/             # Generative results
        ├── dimension_1/
        ├── dimension_2/
        └── dimension_3/
```

## Usage Guide

### 1. API Configuration

1. Copy the example configuration file:
```bash
cp twinvoice/api_config.template.py twinvoice/api_config.py
# or cp twinvoice/api_config.example.py twinvoice/api_config.py
```

2. Edit `twinvoice/api_config.py` to configure APIs:
```python
# Digital Twin API configuration (for generation and discriminative tasks)
twin_base_url = 'http://localhost:8005/v1'  # Local model service address
twin_api_key = 'EMPTY'

# LLM-as-a-Judge API configuration (for generative evaluation judgment)
judge_base_url = '[https://api.your-endpoint.com/v1](https://api.your-endpoint.com/v1)'
judge_api_key = 'your-judge-api-key-here'  # Replace with your judge API key
```

### 2. Start Local Model Service

For local model evaluation, start the vLLM service:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model/Qwen2.5-14B-Instruct \
    --port 8005 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1
```

### 3. Dimension 1 (Social Persona) Evaluation Pipeline

Evaluates public-facing identity via social media interactions.

```bash
# Basic evaluation (default paths)
python -m twinvoice.discriminative.dimension_1.evaluate --model gpt-4o-mini

# Complete evaluation (specify data path)
python -m twinvoice.discriminative.dimension_1.evaluate \
    dataset/dimension_1/data.jsonl \
    --model gpt-4o-mini \
    --report result/discriminative/dimension_1/results.jsonl

# Evaluation with capability analysis
python -m twinvoice.discriminative.dimension_1.evaluate \
    --model gpt-4o-mini \
    --annotations annotated.jsonl \
    --cap-report-csv result/discriminative/dimension_1/capability_report.csv
```

### 4. Dimension 2 (Interpersonal Persona) Evaluation Pipeline

Evaluates the model's ability to maintain consistent user style in private, multi-session dialogues.

#### 4.1 Discriminative Evaluation (User Style Matching)

```bash
# Evaluation with error analysis
python -m twinvoice.discriminative.dimension_2.evaluate \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model MODEL_PATH \
    --report result/discriminative/dimension_2/results.jsonl \
    --wrong-report result/discriminative/dimension_2/wrong_cases.jsonl \
    --temperature 0.0 \
    --history-max 30
```

#### 4.2 Generative Evaluation

**Step 1: Generate Responses**
```bash
python -m twinvoice.generative.dimension_2.gen_step1 \
    --input dataset/dimension_2/conversation_data.jsonl \
    --gen_model MODEL_PATH \
    --out_dir result/generative/dimension_2 \
    --workers 8 \
    --temperature 0.0
```

**Step 2: Judge Generation Quality**
```bash
python -m twinvoice.generative.dimension_2.judge_step2 \
    --input result/generative/dimension_2/step1_generations_*.jsonl \
    --judge_model JUDGE_MODEL \
    --workers 8 \
    --temperature 0.0
```

### 5. Dimension 3 (Narrative Persona) Evaluation Pipeline

Evaluates role-based expression in fictional or defined persona scenarios.

#### 5.1 Discriminative Evaluation (Multiple Choice)

```bash
# Complete evaluation with specified data paths and capability analysis
python -m twinvoice.discriminative.dimension_3.evaluate \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model MODEL_PATH \
    --report result/discriminative/dimension_3/results.jsonl \
    --annotations annotated.jsonl \
    --cap-report-csv result/discriminative/dimension_3/capability_report.csv
```

#### 5.2 Generative Evaluation (Generation + Judge)

**Step 1: Generate Responses**
```bash
python -m twinvoice.generative.dimension_3.gen_step1 \
    --input dataset/dimension_3/choices.jsonl \
    --profile dataset/dimension_3/profiles.jsonl \
    --gen_model MODEL_PATH \
    --out_dir result/generative/dimension_3 \
    --workers 8
```

**Step 2: Judge Generation Quality**
```bash
python -m twinvoice.generative.dimension_3.judge_step2 \
    --input result/generative/dimension_3/step1_generations_*.jsonl \
    --judge_model JUDGE_MODEL \
    --workers 8
```

## Parameter Description

### Discriminative Evaluation Parameters
- `data_file` / `--input`: Input data file path specific to the dimension.
- `profile_json`: Character profile file (Dimension 3).
- `--model`: Evaluation model path or name (e.g., `gpt-4o-mini` or local path).
- `--report`: Results save path (default depends on dimension).
- `--wrong-report`: Error cases save path.
- `--temperature`: Sampling temperature (default: `0.0`).
- `--history-max`: Maximum dialogue history length (default: `30`).
- `--sample`: Sample size for quick testing (optional).
- `--annotations`: Capability annotation file for fine-grained analysis.
- `--cap-report-csv`: Capability dimension report output path.

### Generative Evaluation Parameters
- `--gen_model`: Generation model path or name.
- `--judge_model`: Judge model name (e.g., `gpt-4o`).
- `--out_dir`: Output directory for generated responses.
- `--workers`: Number of parallel workers (default: `8`).

## Output Description & Evaluation Results Structure

### 1. Discriminative Evaluation Output
Results are organized within `result/discriminative/dimension_X/`:
- `results.jsonl`: Complete evaluation results with model predictions and ground truth. Includes overall accuracy.
- `wrong_cases.jsonl`: Detailed analysis of incorrect predictions (up to 20 cases displayed in console).
- `capability_report.csv`: Performance breakdown across the six capabilities (sample count per capability, accuracy, statistical tables).

### 2. Generative Evaluation Output
Results are organized within `result/generative/dimension_X/`:
- **Step 1 Output (`step1_generations_*.jsonl`)**: Generated response content, generation status (Success/Failed), generation statistics, and real-time console progress.
- **Step 2 Output (`Score_and_Analysis_*.jsonl`)**: Mapping accuracy (Acc.Gen), generation quality score (1-5 scale), normalized score (0-1 scale), score distribution, and detailed analysis report across Opinion Consistency, Logical Factual Fidelity, and Stylistic Similarity.

## Important Notes

1. Ensure to run evaluation commands from the project root directory.
2. Make sure API keys are properly configured before running. Do not commit `api_config.py` to version control.
3. Local model evaluation requires running vLLM service (port 8005).
4. Evaluation process supports graceful exit (`Ctrl+C`).
5. Recommended to test configuration with small samples (`--sample 5`) first.
6. Do not commit `api_config.py` to version control system

## Output Description

### 1. Discriminative Evaluation Output
- Overall accuracy
- Error case analysis (up to 20 cases)
- Detailed results saved in report file
- Capability dimension report (if annotations provided)

### 2. Generative Evaluation Output

Step 1 output:
- Generated response content
- Generation status (Success/Failed)
- Detailed results saved in step1_generations_*.jsonl

Step 2 output:
- Mapping accuracy (Acc.Gen)
- Generation quality score (1-5 scale)
- Normalized score (0-1 scale)
- Detailed analysis report (including three dimensions: opinion consistency, logical factual fidelity, style similarity)
