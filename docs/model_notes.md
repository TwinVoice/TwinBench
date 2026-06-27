# Model Notes for TwinBench

TwinBench can evaluate any OpenAI-compatible chat model. For public comparison,
prefer models that are easy for other researchers to reproduce, plus a small set
of role-play-oriented or role-play-adjacent models.

## Common Role-Play Model Families to Check

Dedicated role-play models commonly discussed in the open-source RP community
include:

- Euryale / Stheno style Llama finetunes
- Rocinante
- Magnum
- MythoMax / Mythalion
- Noromaid
- Pygmalion
- Mistral Nemo / Nemotron creative-writing or RP finetunes

These names are useful search terms when checking a provider's model roster.
Not every API gateway exposes them.

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

For a compact public-facing TwinBench comparison, start with these six models:

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
