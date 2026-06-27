# Model Notes for TwinBench

TwinBench can evaluate any OpenAI-compatible chat model. For public comparison,
prefer models that are easy for other researchers to reproduce, plus a small set
of role-play-oriented or role-play-adjacent models.

## Role-Play Leaderboard Candidate Pool

The public TwinBench leaderboard should not be limited to general chat models.
It should include models that are actually used for character chat,
role-playing, and creative writing, plus strong general baselines.

### Dedicated RP and Character-Chat Finetunes

Common search terms and model families:

- Euryale / Stheno style Llama finetunes
- Rocinante
- Magnum
- MythoMax / Mythalion
- Noromaid
- Pygmalion
- Kimiko / creative-writing Llama variants
- Mistral Nemo creative-writing or RP finetunes

### Gemma Writing and RP Candidates

Gemma models should be part of the role-play leaderboard rather than treated as
an afterthought. Candidate families include:

- Gemma 4 instruct variants, such as `gemma-4-26b-a4b-it` or larger variants
- Gemma 3 instruct variants, such as `gemma-3-27b-it`
- Gemma 2 instruct variants, such as `gemma-2-27b-it` and `gemma-2-9b-it`
- Gemma-The-Writer style creative-writing finetunes
- Gemma-Ataraxy / Gemma SPPO / Gemma RP Writer style finetunes

### Strong Open and General Baselines

These are useful for anchoring the leaderboard against widely available chat
models:

- Llama 3.1 / 3.3 / 4 chat or instruction variants
- Mistral Large and Mistral Nemo variants
- Nemotron chat variants
- Qwen 2.5 / Qwen 3 / Qwen 3.5 / Qwen 3.7 chat variants
- DeepSeek V3 / V4 variants
- GLM 4.5 / GLM 5 variants
- Hy3 preview-style models
- Claude, GPT, and Gemini chat models

## Strong General Models Worth Testing

If dedicated RP models are unavailable, test strong chat models and recent
creative-writing-friendly models:

- `gpt-5.2-chat-latest`
- `gpt-5-chat-latest`
- `claude-sonnet-4-6`
- `deepseek-v4-flash-nothinking`
- `deepseek-v4-pro`
- `gemini-3-flash-preview-nothinking`
- `gemini-3.5-flash-nothinking`
- `deepseek-v3.2`
- `glm-5`
- `hy3-preview`

## Recommended Starter Panel

For a compact general-model TwinBench comparison, start with these six models.
This is a smoke-test panel, not the final RP-specialist leaderboard:

| Model | Note |
| --- | --- |
| `claude-sonnet-4-6` | Strong general chat and writing baseline. |
| `deepseek-v4-pro` | Run with reasoning disabled through `reasoning_effort: "none"`. |
| `deepseek-v4-flash-nothinking` | Low-cost DeepSeek no-thinking baseline. |
| `gemini-3.5-flash-nothinking` | Fast Gemini no-thinking baseline. |
| `gemini-3-flash-preview-nothinking` | Preview Gemini no-thinking baseline. |
| `gpt-5.2-chat-latest` | Use a slightly larger completion budget; TwinBench defaults to 256 output tokens for this family. |

Run the full three-dimension panel with:

```bash
python -m twinvoice.evaluate --dimension all --preset small --models starter
```

Run a custom role-play panel by passing comma-separated model IDs exposed by
your OpenAI-compatible provider:

```bash
python -m twinvoice.evaluate \
  --dimension all \
  --preset small \
  --models gemma-3-27b-it,gemma-4-26b-a4b-it,llama-3.1-nemotron-ultra-253b-v1
```

## External Discovery References

Useful places to discover new role-play or creative-writing candidates:

- [OpenRouter Roleplay collection](https://openrouter.ai/collections/roleplay)
- [EQ-Bench Creative Writing leaderboard](https://eqbench.com/creative_writing.html)
- Hugging Face searches for `roleplay`, `creative writing`, `Gemma writer`,
  `Gemma RP`, `Euryale`, `Magnum`, `Rocinante`, `MythoMax`, and `Pygmalion`

## Provider Notes

- `deepseek-v4-pro` can produce hidden or visible thinking on some gateways.
  TwinBench disables reasoning for this model by default through
  `reasoning_effort: "none"`.
- Override thinking-off matching with:

```bash
export TWINVOICE_THINKING_OFF_MODELS="deepseek-v4-pro"
```

- Add provider-specific request fields with:

```bash
export TWINVOICE_EXTRA_BODY_JSON='{"reasoning_effort":"none"}'
```

- Some reasoning models may return empty content when the completion budget is
  too low. The one-command CLI automatically uses `--max-tokens 256` for
  `gpt-5.2*` models; other models default to 128.
