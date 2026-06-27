"""One-command, low-token TwinBench evaluation entrypoint."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from twinvoice.discriminative.dimension_1.evaluate import evaluate_mcq as evaluate_dim1
from twinvoice.discriminative.dimension_2.evaluate import evaluate_mcq as evaluate_dim2
from twinvoice.discriminative.dimension_3.evaluate import evaluate_mcq_only as evaluate_dim3


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "sample": 5,
        "history_max": 4,
        "context_max_chars": 900,
        "profile_max_chars": 500,
        "choice_max_chars": 240,
        "history_item_max_chars": 300,
        "max_tokens": 128,
    },
    "small": {
        "sample": 50,
        "history_max": 8,
        "context_max_chars": 1500,
        "profile_max_chars": 900,
        "choice_max_chars": 400,
        "history_item_max_chars": 500,
        "max_tokens": 128,
    },
    "full": {
        "sample": None,
        "history_max": 30,
        "context_max_chars": None,
        "profile_max_chars": None,
        "choice_max_chars": None,
        "history_item_max_chars": None,
        "max_tokens": 128,
    },
}

STARTER_PANEL = [
    "claude-sonnet-4-6",
    "deepseek-v4-pro",
    "deepseek-v4-flash-nothinking",
    "gemini-3.5-flash-nothinking",
    "gemini-3-flash-preview-nothinking",
    "gpt-5.2-chat-latest",
]


def _preset_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    return PRESETS[args.preset][name] if value is None else value


def _default_model() -> str:
    return os.getenv("TWINVOICE_MODEL") or os.getenv("TWINBENCH_MODEL") or "gpt-4o-mini"


def _effective_max_tokens(args: argparse.Namespace, model: str) -> int:
    if args.max_tokens is not None:
        return args.max_tokens
    if "gpt-5.2" in model.lower():
        return 256
    return PRESETS[args.preset]["max_tokens"]


def _parse_models(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    if args.models and args.model:
        parser.error("Use either --model or --models, not both.")
    if not args.models:
        return [args.model or _default_model()]

    panel_name = args.models.strip().lower()
    if panel_name in {"starter", "starter-panel", "recommended"}:
        return STARTER_PANEL

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        parser.error("--models must be a comma-separated model list or 'starter'.")
    return models


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def _run_one(args: argparse.Namespace, model: str) -> None:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    sample = _preset_value(args, "sample")
    history_max = _preset_value(args, "history_max")
    context_max_chars = _preset_value(args, "context_max_chars")
    profile_max_chars = _preset_value(args, "profile_max_chars")
    choice_max_chars = _preset_value(args, "choice_max_chars")
    history_item_max_chars = _preset_value(args, "history_item_max_chars")
    max_tokens = _effective_max_tokens(args, model)
    report_model = _safe_name(model)

    if args.dimension == "1":
        input_path = args.input or "dataset/dimension_1/pchatbot_pccd_top_2000.jsonl"
        report = report_dir / f"dimension1_{report_model}_{args.preset}.jsonl"
        wrong_report = report_dir / f"dimension1_{report_model}_{args.preset}_wrong.jsonl"
        evaluate_dim1(
            input_path=input_path,
            model=model,
            sample_n=sample,
            report_path=str(report),
            wrong_report_path=str(wrong_report),
            temperature=args.temperature,
            history_max=history_max,
            seed=args.seed,
            context_max_chars=context_max_chars,
            choice_max_chars=choice_max_chars,
            history_item_max_chars=history_item_max_chars,
            max_tokens=max_tokens,
        )
        return

    if args.dimension == "2":
        input_path = args.input or "dataset/dimension_2/conversation_data.jsonl"
        report = report_dir / f"dimension2_{report_model}_{args.preset}.jsonl"
        wrong_report = report_dir / f"dimension2_{report_model}_{args.preset}_wrong.jsonl"
        evaluate_dim2(
            input_path=input_path,
            model=model,
            sample_n=sample,
            report_path=str(report),
            wrong_report_path=str(wrong_report),
            temperature=args.temperature,
            history_max=history_max,
            seed=args.seed,
            context_max_chars=context_max_chars,
            choice_max_chars=choice_max_chars,
            history_item_max_chars=history_item_max_chars,
            max_tokens=max_tokens,
        )
        return

    choices = args.input or "dataset/dimension_3/choices.jsonl"
    profiles = args.profiles or "dataset/dimension_3/profiles.jsonl"
    report = report_dir / f"dimension3_{report_model}_{args.preset}.jsonl"
    wrong_report = report_dir / f"dimension3_{report_model}_{args.preset}_wrong.jsonl"
    evaluate_dim3(
        choices_path=choices,
        profile_path=profiles,
        model=model,
        sample_n=sample,
        report_path=str(report),
        wrong_report_path=str(wrong_report),
        temperature=args.temperature,
        history_max=history_max,
        seed=args.seed,
        context_max_chars=context_max_chars,
        profile_max_chars=profile_max_chars,
        choice_max_chars=choice_max_chars,
        max_tokens=max_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a low-token TwinBench discriminative evaluation."
    )
    parser.add_argument(
        "--dimension",
        choices=["1", "2", "3", "all"],
        default="3",
        help="Evaluation dimension. Use 'all' for the full three-dimension panel.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tiny",
        help="Token/cost preset. tiny is best for smoke tests.",
    )
    parser.add_argument("--model", help="Model name to evaluate.")
    parser.add_argument(
        "--models",
        help="Comma-separated model names, or 'starter' for the recommended six-model panel.",
    )
    parser.add_argument("--sample", type=int, help="Override preset sample size.")
    parser.add_argument("--history-max", type=int, help="Override preset history length.")
    parser.add_argument("--context-max-chars", type=int, help="Clip context text.")
    parser.add_argument("--profile-max-chars", type=int, help="Clip Dimension 3 persona fields.")
    parser.add_argument("--choice-max-chars", type=int, help="Clip candidate choices.")
    parser.add_argument("--history-item-max-chars", type=int, help="Clip each history item.")
    parser.add_argument("--max-tokens", type=int, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", default="result/quick_eval")
    parser.add_argument("--input", help="Override input JSONL path.")
    parser.add_argument("--profiles", help="Override Dimension 3 profile JSON path.")
    args = parser.parse_args()

    models = _parse_models(parser, args)
    dimensions = ["1", "2", "3"] if args.dimension == "all" else [args.dimension]
    total_runs = len(dimensions) * len(models)
    run_index = 0
    for dimension in dimensions:
        run_args = argparse.Namespace(**vars(args))
        run_args.dimension = dimension
        for model in models:
            run_index += 1
            if total_runs > 1:
                print(
                    f"\n===== TwinBench run {run_index}/{total_runs}: "
                    f"dimension {dimension}, model {model} ====="
                )
            _run_one(run_args, model)


if __name__ == "__main__":
    main()
