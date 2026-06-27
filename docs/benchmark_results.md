# TwinBench General Starter Panel Results

This page reports a lightweight discriminative smoke run for the recommended
six-model general starter panel. These numbers are useful for checking the
evaluation pipeline and for previewing model behavior; they are not a
paper-scale leaderboard and not the final role-play-specialist leaderboard.

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
