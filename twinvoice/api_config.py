"""Runtime configuration for TwinBench.

Configuration is intentionally read from environment variables so that API keys
never need to be written into source files.
"""

from __future__ import annotations

import os
import json


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


# OpenAI-compatible endpoint used for the model being evaluated.
twin_base_url = _env(
    "TWINVOICE_TWIN_BASE_URL",
    "TWINBENCH_TWIN_BASE_URL",
    "OPENAI_BASE_URL",
    default="http://localhost:8005/v1",
)
twin_api_key = _env(
    "TWINVOICE_TWIN_API_KEY",
    "TWINBENCH_TWIN_API_KEY",
    "OPENAI_API_KEY",
    default="EMPTY",
)

# OpenAI-compatible endpoint used by LLM-as-a-judge scripts.
judge_base_url = _env(
    "TWINVOICE_JUDGE_BASE_URL",
    "TWINBENCH_JUDGE_BASE_URL",
    "OPENAI_BASE_URL",
    default="https://api.openai.com/v1",
)
judge_api_key = _env(
    "TWINVOICE_JUDGE_API_KEY",
    "TWINBENCH_JUDGE_API_KEY",
    "OPENAI_API_KEY",
    default="",
)

# Backward compatibility for older scripts.
api_key = twin_api_key
base_url = twin_base_url


def _csv_env(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _json_env(name: str) -> dict:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def model_chat_extra_body(model_name: str | None = None) -> dict:
    """Return provider-specific extra request fields for chat calls."""
    body = _json_env("TWINVOICE_EXTRA_BODY_JSON")
    model_key = (model_name or "").strip().lower()
    thinking_off_models = _csv_env("TWINVOICE_THINKING_OFF_MODELS", "deepseek-v4-pro")
    if model_key in thinking_off_models:
        body.setdefault("reasoning_effort", "none")
    return body
