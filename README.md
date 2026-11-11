# TwinBench

<p align="center">
  <img src="Figs/twinbench_hero.png" alt="TwinBench: benchmarking role-playing fidelity in LLMs" width="100%">
</p>

<p align="center">
  <b>A benchmark for evaluating role-playing fidelity, persona simulation, and self-voice consistency in large language models.</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.25536"><img src="https://img.shields.io/badge/arXiv-2510.25536-b31b1b.svg" alt="arXiv"></a>
  <a href="https://twinvoice.github.io"><img src="https://img.shields.io/badge/Project-Page-2a78d6.svg" alt="Project page"></a>
  <a href="https://huggingface.co/spaces/bangdedadi/TwinVoice-Leaderboard"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Leaderboard-Space-ffd21e.svg" alt="Leaderboard"></a>
  <a href="https://huggingface.co/datasets/bangdedadi/TwinVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-TwinVoice-ffd21e.svg" alt="Dataset"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> |
  <a href="#role-play-model-leaderboard">Role-play leaderboard</a> |
  <a href="#starter-panel">Starter panel</a> |
  <a href="#full-benchmark-results">Full results</a> |
  <a href="#what-twinbench-measures">What it measures</a> |
  <a href="#evaluation-modes">Evaluation modes</a> |
  <a href="#citation">Citation</a>
</p>

TwinBench is the evaluation benchmark released with **TwinVoice (ACL 2026)**.
It tests whether a model can preserve a target persona's beliefs, memory,
reasoning, tone, lexical habits, and syntactic style when acting as that persona.

Unlike general chat benchmarks, TwinBench asks a sharper question:

> Can the model stay in character when the correct answer depends on a person's
> history, profile, relationships, and voice?

## Highlights

- **Role-playing evaluation for LLMs**: measure whether a model can act as a
  specific speaker, not only produce helpful generic answers.
- **Three persona settings**: social-media voice, private dialogue style, and
  narrative character role-play.
- **Two evaluation families**: low-cost discriminative evaluation and richer
  generative evaluation with LLM-as-a-judge.
- **Low-token quick eval**: run a small smoke test with one command before
  launching full experiments.
- **OpenAI-compatible endpoints**: works with OpenAI-style APIs, local vLLM
  servers, and compatible model gateways.
- **Model notes included**: see [`docs/model_notes.md`](docs/model_notes.md)
  for role-play model families and provider-specific tips.

## Role-Play Model Leaderboard

TwinBench is designed for a dedicated **role-playing model leaderboard**: models
should be evaluated on how well they preserve a persona's voice, memory,
relationships, behavior, and style across social, dialogue, and character
settings.

The public leaderboard should include both widely used general chat models and
role-play / creative-writing-specialized models. Candidate families include:

| Family | Examples to evaluate |
| --- | --- |
| Dedicated RP finetunes | Euryale / Stheno, Magnum, Rocinante, MythoMax / Mythalion, Noromaid, Pygmalion |
| Gemma writing/RP models | Gemma 4 / Gemma 3 / Gemma 2 instruct models, Gemma-The-Writer, Gemma-Ataraxy, Gemma RP Writer-style finetunes |
| Llama / Mistral RP variants | Llama creative-writing finetunes, Mistral Nemo writing/RP finetunes, Nemotron chat variants |
| Strong chat baselines | Claude, GPT, Gemini, DeepSeek, Qwen, GLM, Hy3, and other reproducible OpenAI-compatible models |

The current table below is a **general-model starter panel**, not the final
role-play-specialist leaderboard. It is included so users can verify the
pipeline immediately while the broader RP model panel is being expanded. See
[`docs/model_notes.md`](docs/model_notes.md) for the growing role-play model
candidate pool.

## Starter Panel

TwinBench ships with a compact six-model starter panel for quickly comparing
role-playing fidelity across all three persona settings. The table below is a
lightweight `small` preset smoke run with 50 examples per dimension; use it as
a reproducibility check and preview, not as a final paper-scale leaderboard.

| Rank | Model | D1 Social | D2 Dialogue | D3 Character | Macro Avg |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `claude-sonnet-4-6` | 62.0 | 80.0 | 96.0 | 79.3 |
| 2 | `gemini-3.5-flash-nothinking` | 50.0 | 84.0 | 100.0 | 78.0 |
| 3 | `gemini-3-flash-preview-nothinking` | 58.0 | 76.0 | 100.0 | 78.0 |
| 4 | `deepseek-v4-flash-nothinking` | 50.0 | 62.0 | 94.0 | 68.7 |
| 5 | `gpt-5.2-chat-latest` | 50.0 | 70.0 | 82.0 | 67.3 |
| 6 | `deepseek-v4-pro` | 50.0 | 48.0 | 98.0 | 65.3 |

Scores are percentages. Full counts, parse-failure notes, and reproduction
details are in [`docs/benchmark_results.md`](docs/benchmark_results.md).

## Full-Benchmark Results

Results on the **full** benchmark: 5,687 examples, split 2,000 / 2,500 / 1,187
across D1 / D2 / D3. Unlike the starter panel above, these are full-scale runs.
`gpt-6-astra` and `claude-opus-5-5` were added in November 2026; the other rows
are the results reported in the paper.

Discriminative accuracy (%):

| Rank | Model | D1 Social | D2 Dialogue | D3 Character | Macro Avg |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `gpt-6-astra` | **69.8** | **86.8** | **98.3** | **85.0** |
| 2 | Claude-Sonnet-4 | 53.9 | 84.4 | 90.2 | 76.2 |
| 3 | `claude-opus-5-5` | 55.7 | 74.8 | 96.9 | 75.8 |
| 4 | GPT-5-Chat | 46.9 | 77.4 | 89.4 | 71.2 |
| 5 | DeepSeek-V3 | 42.6 | 70.0 | 81.0 | 64.5 |
| 6 | GPT-OSS-20B | 39.1 | 63.3 | 43.9 | 48.8 |
| 7 | Qwen2.5-14B | 36.2 | 49.6 | 60.5 | 48.8 |
| 8 | GPT-3.5-Turbo | 34.9 | 41.2 | 66.3 | 47.5 |
| 9 | GPT-4o-mini | 35.3 | 39.2 | 63.1 | 45.9 |

Generative evaluation of the two new models, judged by `gpt-5.5`. Acc.(Gen) is
in percent and Score(Gen) is on a 1-5 scale:

| Model | D1 Acc.(Gen) | D1 Score | D2 Score | D3 Acc.(Gen) | D3 Score | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | 46.9 | 2.83 | 1.74 | 65.5 | 2.17 | 2.25 |
| `claude-opus-5-5` | 38.9 | 2.72 | 1.64 | 61.2 | 2.08 | 2.14 |

Even the strongest model scores close to 2 out of 5 when it has to write the
reply itself, so speaking in a persona's own voice remains unsolved. Counts and
run settings are in
[`docs/benchmark_results.md`](docs/benchmark_results.md#full-benchmark-results-november-2026);
every metric of every model is on the
[leaderboard](https://huggingface.co/spaces/bangdedadi/TwinVoice-Leaderboard).

### Data version v1.1

v1.1 removes rows whose reference reply also appears in the history or context
shown to the model, and scores each distinct question once: rows that share the
same stimulus and reference reply are averaged. v1.0, the data used in the
paper, stays the default and is unchanged. Scores of the two versions are not
directly comparable.

| Model | D1 Social | D2 Dialogue | D3 Character | Macro Avg | Avg Score(Gen) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | 69.6 | 70.3 | 98.6 | 79.5 | 2.18 |
| `claude-opus-5-5` | 55.4 | 57.3 | 97.5 | 70.0 | 2.08 |

```bash
python -m twinvoice.evaluate --dimension all --preset full --model MODEL_NAME --data-version v1.1
```

## Quickstart

### 1. Install

```bash
git clone https://github.com/TwinVoice/TwinBench.git
cd TwinBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure an OpenAI-compatible endpoint

For OpenAI or an API gateway:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

For a local vLLM server:

```bash
export TWINVOICE_TWIN_BASE_URL="http://localhost:8005/v1"
export TWINVOICE_TWIN_API_KEY="EMPTY"
```

PowerShell equivalent:

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

API keys are read from environment variables by `twinvoice/api_config.py`.
Do not put secrets into source files or README files.

### 3. Run the lowest-cost smoke test

Dimension 3 is the friendliest public demo: narrative role-play with character
profiles and multiple-choice voice matching.

```bash
python -m twinvoice.evaluate --dimension 3 --preset tiny --model gpt-4o-mini
```

The same entrypoint also supports Dimension 1 and Dimension 2:

```bash
python -m twinvoice.evaluate --dimension 1 --preset tiny --model gpt-4o-mini
python -m twinvoice.evaluate --dimension 2 --preset tiny --model gpt-4o-mini
```

This runs a 5-sample discriminative evaluation, clips long profile/context
fields, and writes outputs to `result/quick_eval/`.

Use a larger but still lightweight run with:

```bash
python -m twinvoice.evaluate --dimension 3 --preset small --model gpt-4o-mini
```

Run the recommended six-model starter panel with one command:

```bash
python -m twinvoice.evaluate --dimension all --preset small --models starter
```

### Optional: no-token local smoke test

To verify installation without spending API tokens, run the mock server in one
terminal:

```bash
python tools/mock_openai_server.py --port 8765
```

Then run the quick eval in another terminal:

```bash
export TWINVOICE_TWIN_BASE_URL="http://127.0.0.1:8765/v1"
export TWINVOICE_TWIN_API_KEY="EMPTY"
python -m twinvoice.evaluate --dimension 3 --preset tiny --model mock-model
```

The mock server always returns deterministic toy answers. It tests plumbing, not
model quality.

## What TwinBench Measures

TwinBench evaluates role-playing fidelity across three complementary persona
settings:

| Dimension | Persona setting | Core question | Dataset |
| --- | --- | --- | --- |
| 1 | Social Persona | Can the model match a user's public social-media voice? | `dataset/dimension_1/` |
| 2 | Interpersonal Persona | Can the model identify a user's private dialogue style from history? | `dataset/dimension_2/` |
| 3 | Narrative Persona | Can the model speak as a fictional or defined character in context? | `dataset/dimension_3/` |

The benchmark reports capability-level behavior across:

| Category | Capabilities |
| --- | --- |
| Mindset Coherence | `Opinion_Consistency`, `Memory_Recall`, `Logical_Reasoning` |
| Linguistic Expression | `Lexical_Fidelity`, `Persona_Tone`, `Syntactic_Style` |

<p align="center">
  <img src="Figs/twinvoice_main.png" alt="TwinVoice benchmark overview" width="92%">
</p>

## Evaluation Modes

### Discriminative evaluation

Recommended for quick runs and model comparison. The model sees persona context
and four candidate replies, then selects the reply most likely written by the
target persona.

```bash
python -m twinvoice.evaluate --dimension 3 --preset tiny --model MODEL_NAME
```

Advanced Dimension 3 command:

```bash
python -m twinvoice.discriminative.dimension_3.evaluate \
  dataset/dimension_3/choices.jsonl \
  dataset/dimension_3/profiles.jsonl \
  --model MODEL_NAME \
  --sample 20 \
  --history-max 8 \
  --profile-max-chars 900 \
  --context-max-chars 1500 \
  --choice-max-chars 400 \
  --report result/discriminative/dimension_3/results.jsonl
```

Advanced Dimension 2 command:

```bash
python -m twinvoice.discriminative.dimension_2.evaluate \
  --input dataset/dimension_2/conversation_data.jsonl \
  --model MODEL_NAME \
  --sample 20 \
  --history-max 8 \
  --context-max-chars 1500 \
  --choice-max-chars 400 \
  --report result/discriminative/dimension_2/results.jsonl \
  --wrong-report result/discriminative/dimension_2/wrong_cases.jsonl
```

### Generative evaluation

For deeper analysis, first ask the model to generate a persona-consistent reply,
then use an LLM judge to map or score the generated reply against the reference.
This mode costs more tokens but gives richer failure analysis.

The judge scripts read their endpoint from `TWINVOICE_JUDGE_BASE_URL` and
`TWINVOICE_JUDGE_API_KEY`, falling back to `OPENAI_BASE_URL` and
`OPENAI_API_KEY`. Set them when the judge is served from a different endpoint
than the model under test.

Dimension 1 generation with the ranking judge (Acc.(Gen)), then similarity
scoring (Score(Gen)):

```bash
python -m twinvoice.generative.Dimension_1.gen_step1 \
  --gen_model MODEL_NAME \
  --judge_model JUDGE_MODEL \
  --out_dir result/generative/dimension_1 \
  --sample 20 \
  --workers 4

python -m twinvoice.generative.Dimension_1.judge_step2 \
  --input result/generative/dimension_1/step1_generations_lmut-MODEL_NAME_judge-JUDGE_MODEL_range-1-20.jsonl \
  --judge_model JUDGE_MODEL \
  --workers 4
```

Dimension 3 generation:

```bash
python -m twinvoice.generative.Dimension_3.gen_step1 \
  --input dataset/dimension_3/choices.jsonl \
  --profile dataset/dimension_3/profiles.jsonl \
  --gen_model MODEL_NAME \
  --out_dir result/generative/dimension_3 \
  --sample 20 \
  --workers 4 \
  --history-max 8
```

Reasoning models return empty content when the completion budget is used up by
reasoning. Add `--max-tokens 1024` to the generation command for such models,
and `--map-max-tokens 512` to the judging command when the judge is one.

Dimension 3 judging:

```bash
python -m twinvoice.generative.Dimension_3.judge_step2 \
  --input result/generative/dimension_3/step1_generations_lmut-MODEL_NAME_choices.jsonl \
  --judge_model JUDGE_MODEL \
  --workers 4
```

Dimension 2 generation and judging:

```bash
python -m twinvoice.generative.Dimension_2.gen_step1 \
  --input dataset/dimension_2/conversation_data.jsonl \
  --gen_model MODEL_NAME \
  --out_dir result/generative/dimension_2 \
  --sample 20 \
  --workers 4 \
  --temperature 0.0

python -m twinvoice.generative.Dimension_2.judge_step2 \
  --input result/generative/dimension_2/step1_generations_MODEL_NAME.jsonl \
  --judge_model JUDGE_MODEL \
  --workers 4 \
  --temperature 0.0
```

## Low-Token Presets

The one-command interface defaults to discriminative evaluation because it needs
only one model call per sample and no judge call.

| Preset | Samples | History items | Text clipping | Use case |
| --- | ---: | ---: | --- | --- |
| `tiny` | 5 | 4 | aggressive | endpoint smoke test |
| `small` | 50 | 8 | moderate | cheap model comparison |
| `full` | all | 30 | none | paper-scale evaluation |

Override any preset field when needed:

```bash
python -m twinvoice.evaluate \
  --dimension 3 \
  --preset tiny \
  --sample 10 \
  --history-max 6 \
  --profile-max-chars 700 \
  --model MODEL_NAME
```

## Project Structure

```text
TwinBench/
  dataset/
    dimension_1/                 # Social persona data
    dimension_2/                 # Interpersonal dialogue data
    dimension_3/                 # Narrative persona choices and profiles
  Figs/                          # README and paper figures
  result/                        # Released result examples
  twinvoice/
    api_config.py                # Safe environment-variable config
    evaluate.py                  # One-command low-token evaluation
    discriminative/
      dimension_1/
      dimension_2/
      dimension_3/
    generative/
      Dimension_1/
      Dimension_2/
      Dimension_3/
```

## Data Note

Some social and interpersonal data contains user-generated text from online
settings. It may include offensive, biased, or otherwise uncomfortable language.
The benchmark preserves such text for persona/style evaluation; users should
apply appropriate filtering and safety review before demos or downstream use.

## Security Note

Never commit local API keys, private runbooks, or `.env` files. If a key is ever
committed to a public repository, revoke the key immediately and purge it from
Git history before announcing the release.

## Citation

If TwinBench or TwinVoice helps your work, please cite the paper:

```bibtex
@article{du2025twinvoice,
  title={TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation},
  author={Du, Bangde and Guo, Minghao and He, Songming and Ye, Ziyi and Zhu, Xi and Su, Weihang and Zhu, Shuqi and Zhou, Yujia and Zhang, Yongfeng and Ai, Qingyao and others},
  journal={arXiv preprint arXiv:2510.25536},
  year={2025}
}
```
