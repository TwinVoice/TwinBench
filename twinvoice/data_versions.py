"""Benchmark data versions and item-averaged scoring.

v1.0 is the data released with the paper (``dataset/dimension_*``).

v1.1 is defined by ``dataset/v1.1/index.json``. For each dimension it lists the
rows of the v1.0 file that are kept and an ``item_id`` for every kept row:

* a row is dropped when its reference reply also appears in the history or
  context shown to the model;
* rows that share the same stimulus and the same reference reply get the same
  ``item_id``.

v1.1 scores are item-averaged: the metric is averaged over the rows of each
item first, then over items, so every distinct question counts once.

Build the v1.1 files:   python -m twinvoice.data_versions build
Score a result file:    python -m twinvoice.data_versions score --input FILE --field correct
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
V11_DIR = REPO_ROOT / "dataset" / "v1.1"
VERSIONS = ("v1.0", "v1.1")

V10_FILES = {
    1: "dataset/dimension_1/pchatbot_pccd_top_2000.jsonl",
    2: "dataset/dimension_2/conversation_data.jsonl",
    3: "dataset/dimension_3/choices.jsonl",
}


def v11_path(dimension: int) -> Path:
    return V11_DIR / f"dimension_{dimension}.jsonl"


def build_v11(force: bool = False) -> dict[int, Path]:
    """Write the v1.1 JSONL files from the v1.0 files and the index. Returns their paths."""
    index = json.loads((V11_DIR / "index.json").read_text(encoding="utf-8"))
    paths = {}
    for dimension, source in V10_FILES.items():
        out = v11_path(dimension)
        paths[dimension] = out
        if out.exists() and not force:
            continue
        spec = index[f"dimension_{dimension}"]
        with open(REPO_ROOT / source, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        if len(lines) != spec["source_rows"]:
            raise ValueError(
                f"{source} has {len(lines)} rows, but the v1.1 index was built from {spec['source_rows']}."
            )
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for row_number, item_id in zip(spec["rows"], spec["item_ids"]):
                row = json.loads(lines[row_number])
                row["item_id"] = item_id
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paths


def input_path(dimension: int, version: str) -> str:
    """Default input file of a dimension for a data version (builds v1.1 on first use)."""
    if version == "v1.1":
        return str(build_v11()[dimension])
    return V10_FILES[dimension]


def item_average(pairs: Iterable[tuple[Any, float]]) -> tuple[float, int] | None:
    """Mean over items of the per-item mean. ``pairs`` yields (item_id, value)."""
    by_item: dict[Any, list[float]] = defaultdict(list)
    for item_id, value in pairs:
        by_item[item_id].append(float(value))
    if not by_item:
        return None
    return sum(sum(v) / len(v) for v in by_item.values()) / len(by_item), len(by_item)


def item_accuracy(records: Iterable[dict], field: str = "correct") -> tuple[float, int] | None:
    """Item-averaged accuracy in percent, or None when the records carry no ``item_id``."""
    pairs = [(r["item_id"], bool(r.get(field))) for r in records if r.get("item_id") is not None]
    result = item_average(pairs)
    if result is None:
        return None
    return result[0] * 100, result[1]


def _field(record: dict, dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def score_file(path: str, field: str) -> tuple[float, int, int] | None:
    """Item-averaged mean of ``field`` in a JSONL result file. Returns (mean, items, rows)."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            value = _field(record, field)
            if record.get("item_id") is None or value is None:
                continue
            try:
                pairs.append((record["item_id"], float(value)))
            except (TypeError, ValueError):
                continue
    result = item_average(pairs)
    if result is None:
        return None
    return result[0], result[1], len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="TwinBench data versions and item-averaged scoring.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Write the v1.1 JSONL files under dataset/v1.1/.")
    build.add_argument("--force", action="store_true", help="Rebuild files that already exist.")
    score = sub.add_parser("score", help="Item-averaged mean of a field in a result file.")
    score.add_argument("--input", required=True, help="Result JSONL whose rows carry item_id.")
    score.add_argument(
        "--field",
        default="correct",
        help="Field to average, e.g. correct, acc_gen, score_gen_1to5, is_lmut_chosen, "
             "judge_result.final_score.",
    )
    args = parser.parse_args()

    if args.command == "build":
        for dimension, path in build_v11(force=args.force).items():
            print(f"Dimension {dimension}: {path}")
        return

    result = score_file(args.input, args.field)
    if result is None:
        print("No rows with both item_id and the requested field were found.")
        return
    mean, items, rows = result
    print(f"Item-averaged {args.field}: {mean:.4f} ({items} items, {rows} rows)")


if __name__ == "__main__":
    main()
