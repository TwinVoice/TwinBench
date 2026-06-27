# -*- coding: utf-8 -*-
"""Discriminative evaluation for TwinBench Dimension 1.

Dimension 1 evaluates social-media persona matching: given a user's reply
history and four candidate replies to a new post, the model picks the reply most
likely written by the same user.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

from openai import OpenAI

from twinvoice.api_config import twin_api_key, twin_base_url, model_chat_extra_body


EVAL_PROMPT = """You are given a social-media user's reply history and 4 candidate replies to a new anchor post. Only one candidate reply was actually written by this user.

Your task is to choose the candidate most likely written by the same user, based on writing style, tone, interests, phrasing habits, and how the user usually reacts.

User's Historical Replies:
{history}

Current Anchor Post:
"{anchor}"

Candidate Replies:
A. {a}
B. {b}
C. {c}
D. {d}

Return ONLY strict JSON in this format, with no prose before or after it:
{{"choice": "A"}}
"""


def get_client() -> OpenAI:
    return OpenAI(base_url=twin_base_url, api_key=twin_api_key)


client = get_client()

JSON_ONLY_SYSTEM = (
    "You are a strict JSON-only multiple-choice classifier. "
    "Think silently. Do not explain. Return exactly one JSON object like {\"choice\":\"A\"}."
)


def parse_choice(resp: str) -> str | None:
    try:
        obj = json.loads(resp)
        choice = str(obj.get("choice", "")).strip().upper()
        return choice if choice in {"A", "B", "C", "D"} else None
    except Exception:
        pass
    match = re.search(r'"?choice"?\s*[:=]\s*"?([ABCD])"?', str(resp), re.I)
    return match.group(1).upper() if match else None


def letter_to_index(letter: str | None) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter or "", -1)


def clip_text(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def save_results(results: list[dict], output_path: str) -> None:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_section(title: str, char: str = "=") -> None:
    line = char * 50
    print(f"\n{line}")
    print(title)
    print(f"{line}\n")


def create_chat_completion_with_retries(**kwargs):
    last_error = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_error


def evaluate_mcq(
    input_path: str,
    model: str,
    sample_n: int | None = None,
    report_path: str | None = None,
    wrong_report_path: str | None = None,
    temperature: float = 0.0,
    history_max: int | None = 30,
    seed: int | None = None,
    context_max_chars: int | None = None,
    choice_max_chars: int | None = None,
    history_item_max_chars: int | None = 500,
    max_tokens: int = 128,
) -> list[dict]:
    print_section("Evaluation Configuration", "-")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"History Max: {'all' if not history_max else history_max}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Sample Size: {'all' if not sample_n else sample_n}")
    if seed is not None:
        print(f"Seed: {seed}")

    print_section("Loading Data", "-")
    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Found {len(rows)} social-persona entries")

    if sample_n and sample_n < len(rows):
        if seed is not None:
            random.seed(seed)
        random.shuffle(rows)
        rows = rows[:sample_n]
        print(f"Sampled {sample_n} entries for evaluation")

    total = correct = 0
    results: list[dict] = []
    wrongs: list[dict] = []

    print_section("Starting Evaluation", "-")
    print(f"Processing {len(rows)} entries...\n")

    for idx, row in enumerate(rows, 1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(rows)} entries processed")

        raw_history = row.get("history") or []
        if history_max:
            raw_history = raw_history[-history_max:]
        history = [clip_text(item, history_item_max_chars) for item in raw_history]
        anchor = clip_text(row.get("anchor_post", ""), context_max_chars)
        choices = [clip_text(choice, choice_max_chars) for choice in row.get("choices", [])]
        answer_idx = row.get("answer_idx")

        if not all([anchor, choices, isinstance(answer_idx, int), len(choices) == 4]):
            continue
        if not 0 <= answer_idx < 4:
            continue

        prompt = EVAL_PROMPT.format(
            history="\n".join(history),
            anchor=anchor,
            a=choices[0],
            b=choices[1],
            c=choices[2],
            d=choices[3],
        )

        try:
            resp = create_chat_completion_with_retries(
                model=model,
                messages=[
                    {"role": "system", "content": JSON_ONLY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=model_chat_extra_body(model),
            )
            response_text = (resp.choices[0].message.content or "").strip()
        except KeyboardInterrupt:
            print("safe exit")
            break
        except Exception as exc:
            print(f"API Error: {exc}")
            time.sleep(1)
            continue

        choice = parse_choice(response_text)
        pred_idx = letter_to_index(choice)
        ok = pred_idx == answer_idx
        total += 1
        correct += int(ok)

        rec = {
            "user_id": row.get("user_id"),
            "line_idx": row.get("line_idx"),
            "original_line_idx": row.get("original_line_idx"),
            "anchor_post": anchor,
            "choices": choices,
            "history_count": len(history),
            "predicted_choice": choice,
            "predicted_index": pred_idx,
            "answer_index": answer_idx,
            "correct": ok,
            "parse_status": "ok" if choice else "failed",
            "response_text": response_text,
        }
        results.append(rec)
        if not ok:
            wrongs.append(rec)

    print_section("Evaluation Results", "=")
    accuracy = correct / total * 100 if total else 0.0
    print(f"Total Items Evaluated: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    if wrongs:
        print(f"\nIncorrect Answers: {len(wrongs)} / {total}")
        print("\nTop 20 Wrong Cases:")
        for wrong in wrongs[:20]:
            print(
                f"- user={wrong['user_id']} | picked={wrong['predicted_choice']} "
                f"| answer_idx={wrong['answer_index']}"
            )

    if report_path:
        print(f"\nSaving results to {report_path}...")
        save_results(results, report_path)
        print("Results saved successfully")

    if wrong_report_path and wrongs:
        print(f"\nSaving wrong cases to {wrong_report_path}...")
        save_results(wrongs, wrong_report_path)
        print("Wrong cases saved successfully")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Dimension 1 social-media persona matching."
    )
    parser.add_argument(
        "--input",
        default="dataset/dimension_1/pchatbot_pccd_top_2000.jsonl",
        help="Path to Dimension 1 JSONL data.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to evaluate.")
    parser.add_argument("--sample", type=int, help="Number of samples to evaluate.")
    parser.add_argument(
        "--report",
        default="result/discriminative/dimension_1/results.jsonl",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "--wrong-report",
        default="result/discriminative/dimension_1/wrong_cases.jsonl",
        help="Path to save wrong cases.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--history-max", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--context-max-chars", type=int)
    parser.add_argument("--choice-max-chars", type=int)
    parser.add_argument("--history-item-max-chars", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    evaluate_mcq(
        input_path=args.input,
        model=args.model,
        sample_n=args.sample,
        report_path=args.report,
        wrong_report_path=args.wrong_report,
        temperature=args.temperature,
        history_max=args.history_max,
        seed=args.seed,
        context_max_chars=args.context_max_chars,
        choice_max_chars=args.choice_max_chars,
        history_item_max_chars=args.history_item_max_chars,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
