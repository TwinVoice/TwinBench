# TwinVoice

TwinVoice is a persona simulation benchmark towards digital twins that supports multi-dimensional assessment of conversational capabilities.

## Project Structure

```
TwinVoice/
├── dataset/                    # Evaluation datasets
│   ├── dimension_2/           
│   │   └── conversation_data.jsonl  # Conversation history data
│   └── dimension_3/           
│       ├── choices.jsonl      # Multiple choice data
│       └── profiles.jsonl     # Character profiles
├── twinvoice/                 # Main package
│   ├── api_config.py         # API configuration
│   ├── discriminative/       # Discriminative evaluation
│   │   ├── dimension_2/     # Dimension 2 module
│   │   │   └── evaluate.py  # Main evaluation script
│   │   └── dimension_3/     # Dimension 3 module
│   │       └── evaluate.py  # Main evaluation script
│   └── generative/          # Generative evaluation
│       ├── dimension_2/     # Dimension 2 generation
│       │   ├── gen_step1.py # Generation step
│       │   └── judge_step2.py # Judgment step
│       └── dimension_3/     # Dimension 3 generation
│           ├── gen_step1.py # Generation step
│           └── judge_step2.py # Judgment step
└── result/                   # Evaluation results
    ├── discriminative/       # Discriminative results
    │   ├── dimension_2/     # Dimension 2 results
    │   │   ├── results.jsonl     # Detailed results
    │   │   └── wrong_cases.jsonl # Error cases
    │   └── dimension_3/     # Dimension 3 results
    │       ├── results.jsonl       # Detailed results
    │       ├── wrong_cases.jsonl   # Error cases
    │       └── capability_report.csv # Capability analysis
    └── generative/          # Generative results
        ├── dimension_2/     # Dimension 2 generation results
        └── dimension_3/     # Dimension 3 generation results
```

## Usage Guide

### 1. API Configuration

1. Copy the example configuration file:
```bash
cp twinvoice/api_config.template.py twinvoice/api_config.py
```

2. Edit `twinvoice/api_config.py` to configure APIs:
```python
# Digital Twin API configuration (for generation and discriminative tasks)
twin_base_url = 'http://localhost:8005/v1'  # Local model service address
twin_api_key = 'EMPTY'

# LLM-as-a-Judge API configuration (for generative evaluation judgment)
judge_base_url = 'https://svip.xty.app/v1'
judge_api_key = 'your-judge-api-key-here'  # Replace with your judge API key
```

### 2. Start Local Model Service

```bash
# Start vLLM service
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model/Qwen2.5-14B-Instruct \
    --port 8005 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1
```

### 3. Dimension 2 Evaluation Pipeline

Dimension 2 evaluation includes both discriminative and generative assessments.

#### 3.1 Discriminative Evaluation (User Style Matching)

Evaluates the model's ability to maintain consistent user style in conversations.

```bash
# Basic evaluation
python -m twinvoice.discriminative.dimension_2.evaluate \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model MODEL_PATH \
    --report result/discriminative/dimension_2/results.jsonl

# Evaluation with error analysis
python -m twinvoice.discriminative.dimension_2.evaluate \
    --input dataset/dimension_2/conversation_data.jsonl \
    --model MODEL_PATH \
    --report result/discriminative/dimension_2/results.jsonl \
    --wrong-report result/discriminative/dimension_2/wrong_cases.jsonl \
    --temperature 0.0 \
    --history-max 30
```

**Parameters:**
- `--input`: Input data file path
- `--model`: Evaluation model path or name
- `--report`: Results save path
- `--wrong-report`: Error cases save path (optional)
- `--temperature`: Sampling temperature, default 0.0
- `--history-max`: Maximum dialogue history length, default 30
- `--sample`: Sample size for quick testing (optional)

**Output:**
- Overall accuracy
- Error case analysis (if specified)
- Real-time console progress
- Detailed evaluation summary (summary.txt)

#### 3.2 Generative Evaluation

Generative evaluation consists of two steps: response generation and quality assessment.

##### Step 1: Generate Responses (gen_step1.py)

Uses Digital Twin model to generate user-style-consistent responses.

```bash
python -m twinvoice.generative.Dimension_2.gen_step1 \
    --input dataset/dimension_2/conversation_data.jsonl \
    --gen_model MODEL_PATH \
    --out_dir result/generative/dimension_2 \
    --workers 8 \
    --temperature 0.0
```

**Parameters:**
- `--input`: Input data file path
- `--gen_model`: Generation model path or name
- `--out_dir`: Output directory
- `--workers`: Number of parallel workers, default 8
- `--temperature`: Sampling temperature, default 0.0
- `--sample`: Sample size for quick testing (optional)

**Output:**
- Generated responses (step1_generations_*.jsonl)
- Generation statistics
- Real-time console progress
- Detailed generation summary (generation_summary.txt)

##### Step 2: Judge Generation Quality (judge_step2.py)

Uses LLM-as-a-Judge model to assess the quality of generated responses.

```bash
python -m twinvoice.generative.Dimension_2.judge_step2 \
    --input result/generative/dimension_2/step1_generations_*.jsonl \
    --judge_model JUDGE_MODEL \
    --workers 8 \
    --temperature 0.0
```

**Parameters:**
- `--input`: Step 1 generation result file
- `--judge_model`: Judge model name
- `--workers`: Number of parallel workers, default 8
- `--temperature`: Sampling temperature, default 0.0

**Output:**
- Score results (Score_and_Analysis_*.jsonl)
- Score statistics and distribution
- Real-time console progress
- Detailed judgment summary (judge_summary.txt)

### 4. Dimension 3 Evaluation Pipeline

#### 4.1 Discriminative Evaluation (Multiple Choice)

```bash
# Basic evaluation (using default paths)
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model MODEL_PATH \
    --sample 5  # Optional: use small sample for testing

# Complete evaluation (specify data paths)
python -m twinvoice.discriminative.dimension_3.evaluate \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model MODEL_PATH \
    --report result/discriminative/dimension_3/results.jsonl

# Evaluation with capability analysis
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model MODEL_PATH \
    --annotations annotated.jsonl \
    --cap-report-csv result/discriminative/dimension_3/capability_report.csv
```

#### 4.2 Generative Evaluation (Generation + Judge)

The generative evaluation consists of two steps:

1. **Step 1: Generate Responses**
```bash
# Use Digital Twin to generate responses
python -m twinvoice.generative.dimension_3.gen_step1 \
    --input dataset/dimension_3/choices.jsonl \
    --profile dataset/dimension_3/profiles.jsonl \
    --gen_model MODEL_PATH \
    --out_dir result/generative/dimension_3 \
    --workers 8
```

2. **Step 2: Judge Generation Quality**
```bash
# Use Judge model to assess generation quality
python -m twinvoice.generative.dimension_3.judge_step2 \
    --input result/generative/dimension_3/step1_generations_*.jsonl \
    --judge_model JUDGE_MODEL \
    --workers 8
```

### Parameter Description

#### Discriminative Evaluation Parameters
- `choices_jsonl`: Multiple choice data file (default: dataset/dimension_3/choices.jsonl)
- `profile_json`: Character profile file (default: dataset/dimension_3/profiles.jsonl)
- `--model`: Model name to use
- `--sample`: Sample size (optional)
- `--report`: Evaluation result file (default: result/discriminative/dimension_3/results.jsonl)
- `--wrong-report`: Error case file (default: result/discriminative/dimension_3/wrong_cases.jsonl)
- `--temperature`: Sampling temperature (default: 0.0)
- `--history-max`: Maximum dialogue history length (default: 30)
- `--annotations`: Capability annotation file (optional)
- `--cap-report-csv`: Capability dimension report (default: result/discriminative/dimension_3/capability_report.csv)

#### Generative Evaluation Parameters

Step 1 (gen_step1.py):
- `--input`: Input data file (choices.jsonl)
- `--profile`: Character profile file (profiles.jsonl)
- `--gen_model`: Generation model name
- `--out_dir`: Output directory
- `--workers`: Number of parallel workers (default: 8)

Step 2 (judge_step2.py):
- `--input`: Step 1 generation result file
- `--judge_model`: Judge model name (default: gpt-5-chat)
- `--workers`: Number of parallel workers (default: 8)

### Evaluation Capabilities

- Opinion_Consistency
- Memory_Recall
- Logical_Reasoning
- Lexical_Fidelity
- Persona_Tone
- Syntactic_Style

### Important Notes

1. Ensure to run evaluation commands from the project root directory
2. Make sure API keys are properly configured before running
3. Local model evaluation requires running vLLM service (port 8005)
4. Evaluation process supports graceful exit (Ctrl+C)
5. Recommended to test configuration with small samples first
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
