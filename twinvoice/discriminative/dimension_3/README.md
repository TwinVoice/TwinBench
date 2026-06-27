# Dimension 3 Discriminative Evaluation

This module evaluates narrative persona role-play with multiple-choice voice
matching. The model receives a speaker profile, prior utterance history, scene
context, and four candidate utterances, then selects the most plausible target
speaker reply.

Recommended public quickstart:

```bash
python -m twinvoice.evaluate --dimension 3 --preset tiny --model MODEL_NAME
```

Advanced command:

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

Configuration is read from environment variables via `twinvoice/api_config.py`.
Do not put API keys into source files.
