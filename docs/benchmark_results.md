# TwinBench General Starter Panel Results

This page reports a lightweight discriminative smoke run for the recommended
six-model general starter panel. These numbers are useful for checking the
evaluation pipeline and for previewing model behavior; they are not a
paper-scale leaderboard and not the final role-play-specialist leaderboard.
Full-benchmark runs for two frontier models are reported
[at the end of this page](#full-benchmark-results-november-2026).

Run configuration:

- Date: 2026-06-27
- Mode: discriminative multiple-choice evaluation
- Preset: `small`
- Sample size: 50 examples per dimension
- Seed: 42
- Temperature: 0.0
- Dimension 1: social-media persona matching
- Dimension 2: private dialogue style matching
- Dimension 3: narrative character role-play matching

Reproduce with:

```bash
python -m twinvoice.evaluate --dimension all --preset small --models starter
```

| Rank | Model | D1 Social | D2 Dialogue | D3 Character | Macro Avg | Total | Parse Fail |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `claude-sonnet-4-6` | 62.0% (31/50) | 80.0% (40/50) | 96.0% (48/50) | 79.3% | 119/150 | 0 |
| 2 | `gemini-3.5-flash-nothinking` | 50.0% (25/50) | 84.0% (42/50) | 100.0% (50/50) | 78.0% | 117/150 | 0 |
| 3 | `gemini-3-flash-preview-nothinking` | 58.0% (29/50) | 76.0% (38/50) | 100.0% (50/50) | 78.0% | 117/150 | 0 |
| 4 | `deepseek-v4-flash-nothinking` | 50.0% (25/50) | 62.0% (31/50) | 94.0% (47/50) | 68.7% | 103/150 | 0 |
| 5 | `gpt-5.2-chat-latest` | 50.0% (25/50) | 70.0% (35/50) | 82.0% (41/50) | 67.3% | 101/150 | 6 |
| 6 | `deepseek-v4-pro` | 50.0% (25/50) | 48.0% (24/50) | 98.0% (49/50) | 65.3% | 98/150 | 0 |

Interpretation notes:

- D3 is the easiest starter smoke test and is the best public demo path.
- D1 and D2 are noisier, style-heavy settings and should be evaluated with
  larger samples before making strong claims.
- Parse failures are counted as incorrect whenever they appear in the result
  file. `gpt-5.2-chat-latest` uses a 256-token completion budget by default.
- `deepseek-v4-pro` is run with reasoning disabled through
  `reasoning_effort: "none"`.

## Full-Benchmark Results (November 2026)

Full-scale runs for `gpt-6-astra` and `claude-opus-5-5`. Every example of the
three dimensions is covered (2,000 / 2,500 / 1,187, 5,687 in total), and no
example is missing from any stage.

Run configuration:

- Date: 2026-11-11
- Discriminative mode: preset `full`, seed 42, temperature 0.0
- Completion budget, discriminative mode: 512 tokens for `gpt-6-astra`, which
  reasons implicitly and returns empty content under a 128-token budget (the
  CLI now applies this to `gpt-6*` models by default); 128 tokens for
  `claude-opus-5-5`
- Generative mode: the released per-example generation and judging functions,
  with Dimension 3 history limited to 30 items (the script default)
- Judge: `gpt-5.5` with `reasoning_effort: "none"`
- Completion budget, Dimension 3 generation: the 256 tokens of the released
  script for `claude-opus-5-5`, raised to 1,024 for `gpt-6-astra`

Reproduce the discriminative runs with:

```bash
python -m twinvoice.evaluate --dimension all --preset full --model gpt-6-astra
python -m twinvoice.evaluate --dimension all --preset full --model claude-opus-5-5
```

### Discriminative

| Model | D1 Social | D2 Dialogue | D3 Character | Macro Avg | Total | Parse Fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | 69.80% (1396/2000) | 86.80% (2170/2500) | 98.32% (1167/1187) | 84.97% | 4733/5687 | 0 |
| `claude-opus-5-5` | 55.65% (1113/2000) | 74.76% (1869/2500) | 96.88% (1150/1187) | 75.76% | 4132/5687 | 1 |

### Generative

Acc.(Gen) for D1 is the share of examples where the judge picks the model's
reply over the distractors. For D3 it is the share where the judge maps the
generated reply to the gold option. Score(Gen) is the mean similarity score on
a 1-5 scale; Avg Score is the unweighted mean over the three dimensions.

| Model | D1 Acc.(Gen) | D1 Score | D2 Score | D3 Acc.(Gen) | D3 Score | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | 46.85% (937/2000) | 2.83 | 1.74 | 65.54% (778/1187) | 2.17 | 2.25 |
| `claude-opus-5-5` | 38.95% (768/1972) | 2.72 | 1.64 | 61.25% (727/1187) | 2.08 | 2.14 |

Notes:

- The judge of these runs is `gpt-5.5`; the paper results were judged by
  GPT-5-Chat.
- The released Dimension 2 scripts report Score(Gen) only, so no D2 Acc.(Gen)
  is listed.
- `gpt-6-astra` returned a reply for every example and declined none.
- `claude-opus-5-5` declined 183 of the 2,500 D2 examples (7.3%), citing
  hateful content in the message history. Each of them received a score of 1.
  Without them its D2 score is 1.69.
- For `claude-opus-5-5` on D1, 25 of the 2,000 generations failed: 13 were
  refusals and 12 did not return the required JSON. The judge scored them as
  they are, so they are included in the D1 score above (2.74 without them).
  Following the released scripts, they are left out of the Acc.(Gen)
  denominator, together with 3 judge outputs that could not be parsed.
  Counting all 28 as incorrect gives 38.40%.
- Parse failures in the discriminative mode are counted as incorrect.
