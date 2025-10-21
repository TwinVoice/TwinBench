# -*- coding: utf-8 -*-
import json, re, signal, os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from openai import OpenAI
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from twinvoice.api_config import twin_base_url, twin_api_key

# ================== API Configuration ==================
# Discriminative task only uses Digital Twin API
def get_client(model_name=None):
    """Return Digital Twin API client configuration"""
    return OpenAI(
        base_url=twin_base_url,
        api_key=twin_api_key
    )

# Default client uses Digital Twin API
client = get_client()

# ================== Prompt Template ==================
EVAL_PROMPT = """You are given a user's reply history and 4 candidate replies to a context message. Only one of the replies was actually written by this user. The other three were written by different users replying to the same context message.

Your task is to choose the most likely reply written by the same user, based on writing style, tone, and expression habits. Focus on how the user typically speaks, their phrasing, and how they respond emotionally or humorously.

User's Historical Conversations:
{history}

Current Anchor Post:
"{anchor}"

Candidate Replies:
A. {a}
B. {b}
C. {c}
D. {d}

Please respond in the following JSON format:
{{"choice": "A"}}  (A|B|C|D)
"""

# ================== Utilities ==================
def parse_choice(resp):
    """Parse model response to get choice A/B/C/D"""
    try:
        obj = json.loads(resp)
        c = obj.get("choice","").upper()
        return c if c in {"A","B","C","D"} else None
    except Exception:
        pass
    m = re.search(r'"?choice"?\s*[:=]\s*"?([ABCD])"?', str(resp), re.I)
    return m.group(1).upper() if m else None

def letter_to_index(letter):
    """Convert letter choice to index"""
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(letter, -1)

def save_results(results, output_path):
    """Save results to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def print_section(title, char="="):
    """Print a section title with decorative lines"""
    line = char * 50
    print(f"\n{line}")
    print(f"{title}")
    print(f"{line}\n")

# ================== Evaluation ==================
def evaluate_mcq(input_path, model, sample_n=None, report_path=None, wrong_report_path=None,
                temperature=0.0, history_max=30):
    """
    Read conversation_data.jsonl, evaluate model's ability to identify user's reply.
    Returns DataFrame and can write results to report_path(JSONL).
    """
    print_section(f"Evaluation Configuration", "-")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"History Max: {history_max}")
    print(f"Sample Size: {'all' if not sample_n else sample_n}")
    
    print_section("Loading Data", "-")
    print("Loading conversations...")
    with open(input_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"Found {len(rows)} conversation entries")
    
    if sample_n and sample_n < len(rows):
        import random
        random.shuffle(rows)
        rows = rows[:sample_n]
        print(f"Sampled {sample_n} entries for evaluation")

    total = correct = 0
    results, wrongs = [], []
    
    print_section("Starting Evaluation", "-")
    print(f"Processing {len(rows)} entries...\n")
    
    for idx, row in enumerate(rows, 1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(rows)} entries processed")

        # Extract data
        history = row.get("history", [])[-history_max:]  # Use last N history items
        context = row.get("context", "").strip()
        choices = row.get("choices", [])
        answer_idx = row.get("answer_idx")
        
        # Validate data
        if not all([context, choices, isinstance(answer_idx, int), 
                   0 <= answer_idx < 4, len(choices) == 4]):
            continue

        # Build prompt
        prompt = EVAL_PROMPT.format(
            history="\n".join(history),
            anchor=context,
            a=choices[0],
            b=choices[1],
            c=choices[2],
            d=choices[3]
        )

        # Get model response
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            resp_text = resp.choices[0].message.content.strip()
        except KeyboardInterrupt:
            print("safe exit"); break
        except Exception as e:
            print(f"API Error: {e}")
            continue

        # Parse response
        choice = parse_choice(resp_text)
        if not choice:
            continue
            
        pred_idx = letter_to_index(choice)
        ok = (pred_idx == answer_idx)
        total += 1
        if ok: correct += 1

        # Record result
        rec = {
            "user_id": row.get("user_id"),
            "context": context,
            "choices": choices,
            "history_count": len(history),
            "predicted_choice": choice,
            "predicted_index": pred_idx,
            "answer_index": answer_idx,
            "correct": ok
        }
        results.append(rec)
        if not ok:
            wrongs.append(rec)

    print_section("Evaluation Results", "=")
    
    acc = (correct/total*100) if total else 0.0
    print(f"Total Items Evaluated: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    
    if wrongs:
        print(f"\nIncorrect Answers: {len(wrongs)} / {total}")
        print("\nTop 20 Wrong Cases:")
        for w in wrongs[:20]:
            print(f"- user={w['user_id']} | picked={w['predicted_choice']} | answer_idx={w['answer_index']}")

    if report_path:
        print(f"\nSaving results to {report_path}...")
        save_results(results, report_path)
        print("Results saved successfully")

    if wrong_report_path and wrongs:
        print(f"\nSaving wrong cases to {wrong_report_path}...")
        save_results(wrongs, wrong_report_path)
        print("Wrong cases saved successfully")

    # Save a summary report
    summary_path = os.path.join(os.path.dirname(report_path), "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Dimension 2 Evaluation Summary ===\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Data: {os.path.basename(input_path)}\n")
        f.write(f"Sample Size: {'all' if not sample_n else sample_n}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"History Max: {history_max}\n\n")
        
        f.write("=== Results ===\n\n")
        f.write(f"Total Items: {total}\n")
        f.write(f"Correct Answers: {correct}\n")
        f.write(f"Accuracy: {acc:.2f}%\n\n")
        
        if wrongs:
            f.write("=== Error Analysis ===\n\n")
            f.write(f"Wrong Cases: {len(wrongs)}\n")
            f.write(f"Error Rate: {(len(wrongs)/total*100):.2f}%\n\n")
            
            # Add some example wrong cases
            f.write("Example Wrong Cases (up to 5):\n")
            for w in wrongs[:5]:
                f.write(f"- User: {w['user_id']}\n")
                f.write(f"  Context: {w['context'][:100]}...\n")
                f.write(f"  Predicted: {w['predicted_choice']} (idx={w['predicted_index']})\n")
                f.write(f"  Correct: idx={w['answer_index']}\n")
                f.write("\n")
        
        f.write("\nFull results saved in:\n")
        f.write(f"- {os.path.basename(report_path)}\n")
        if wrong_report_path:
            f.write(f"- {os.path.basename(wrong_report_path)}\n")
    
    print(f"\n💡 Summary report saved to: {summary_path}")
    return results

def main():
    import argparse
    
    print_section("Dimension 2 Evaluation Start", "=")
    
    ap = argparse.ArgumentParser(description="Evaluate dialogue system using conversation history")
    ap.add_argument("--input", default="dataset/dimension_2/conversation_data.jsonl",
                   help="Path to conversation data file")
    ap.add_argument("--model", default="Qwen2.5-14B-Instruct",
                   help="Model name to use for evaluation")
    ap.add_argument("--sample", type=int,
                   help="Number of samples to evaluate (optional)")
    ap.add_argument("--report", default="result/discriminative/dimension_2/results.jsonl",
                   help="Path to save evaluation results (JSONL format)")
    ap.add_argument("--wrong-report", default="result/discriminative/dimension_2/wrong_cases.jsonl",
                   help="Path to save wrong cases")
    ap.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature")
    ap.add_argument("--history-max", type=int, default=30,
                   help="Maximum number of history items to include")
    
    args = ap.parse_args()
    
    evaluate_mcq(
        input_path=args.input,
        model=args.model,
        sample_n=args.sample,
        report_path=args.report,
        wrong_report_path=args.wrong_report,
        temperature=args.temperature,
        history_max=args.history_max
    )

if __name__ == "__main__":
    main()