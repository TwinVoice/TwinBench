# -*- coding: utf-8 -*-
import json, argparse, random, re, signal, os
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from twinvoice.api_config import twin_base_url, twin_api_key

# --- API configuration ---
# Discriminative task only uses Digital Twin API
def get_client(model_name=None):
    """Return Digital Twin API client configuration"""
    return OpenAI(
        base_url=twin_base_url,
        api_key=twin_api_key
    )

# Default client uses Digital Twin API
client = get_client()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def chat(model, prompt, *, stream=False, verbose=False, json_mode=True, temperature=0.2, timeout=30):
    # Use Digital Twin API client
    current_client = get_client(model)
    
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "timeout": timeout,
    }
    
    # Digital Twin models generally don't support json_mode, use regex parsing instead
    # Can be manually enabled if the model supports json_mode
    # if json_mode:
    #     kwargs["response_format"] = {"type": "json_object"}

    if stream:
        words = []
        for chunk in current_client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta
            if delta.content:
                if verbose:
                    print(delta.content, end="", flush=True)
                words.append(delta.content)
        if verbose:
            print()
        return "".join(words)
    else:
        resp = current_client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

STOP = False
def _handle_sigint(signum, frame):
    global STOP
    STOP = True
    print("safe exit")

# =============== Prompt (with history) ===============
EVAL_PROMPT = """You are a careful reading-comprehension evaluator.
You are given ONLY:
- speaker name,
- a short persona/profile summary for that speaker (if available),
- the preceding narration (context) BEFORE someone speaks,
- the speaker's prior utterance history (chronological, up to BEFORE the given chunk),
- four multiple-choice options (A–D), each a possible utterance.

Task:
Pick the SINGLE best option (A/B/C/D) that most plausibly fits the context AND the speaker's persona and prior utterances.
Return ONLY strict JSON: {{"choice":"A"}} (A|B|C|D). Do not include any extra keys or text.

Speaker: {speaker}

Persona summary (may be partial):
- Traits: {traits}
- Goals: {goals}
- Details: {details}

Speaker's prior utterance history (up to BEFORE chunk {chunk_id}, may be empty):
{history}

Context (narration BEFORE the speech):
\"\"\"{context}\"\"\"

Options:
A) {optA}
B) {optB}
C) {optC}
D) {optD}
"""

# =============== Profile/History Utilities ===============
def load_profiles(path):
    data = json.load(open(path, encoding="utf-8"))
    profiles = {}
    for _, obj in (data.get("characterList") or {}).items():
        if not isinstance(obj, dict):
            continue
        name_canonical = obj.get("NameCanonical")
        variants = obj.get("NameVariants") or []
        if not name_canonical:
            continue
        if isinstance(variants, str):
            variants = [variants]
        keys = set([name_canonical] + variants)
        keys |= {k.lower() for k in keys}
        for k in keys:
            profiles[k] = obj
    return data, profiles

def best_match(name, keys):
    if name in keys: return name
    low = name.lower()
    if low in keys: return low
    import difflib
    m = difflib.get_close_matches(low, [k for k in keys if k.islower()], n=1, cutoff=0.82)
    return m[0] if m else None

def get_canonical_name(name, profiles_map):
    mk = best_match(name, set(profiles_map.keys()))
    if not mk: return name
    obj = profiles_map.get(mk) or {}
    return obj.get("NameCanonical") or name

def persona_fields(p):
    if not p: return ("none","none","none")
    t = p.get("Personality Traits") or "none"
    g = p.get("Motivation and Goals") or "none"
    d = p.get("Additional Details") or "none"
    if isinstance(t, list): t = ", ".join(t[:8]) or "none"
    if isinstance(g, list): g = ", ".join(g[:8]) or "none"
    if isinstance(d, dict):
        d = "; ".join(f"{k}:{','.join(v) if isinstance(v,list) else v}" for k,v in list(d.items())[:6]) or "none"
    elif isinstance(d, list):
        d = ", ".join(d[:8]) or "none"
    elif not isinstance(d, str):
        d = "none"
    return (t,g,d)

def parse_choice(resp):
    try:
        obj = json.loads(resp)
        c = obj.get("choice","").upper()
        return c if c in {"A","B","C","D"} else None
    except Exception:
        pass
    m = re.search(r'"?choice"?\s*[:=]\s*"?([ABCD])"?', str(resp), re.I)
    return m.group(1).upper() if m else None

def build_prompt(spk, traits, goals, details, ctx, opts, chunk_id, history_text):
    return EVAL_PROMPT.format(
        speaker=spk, traits=traits, goals=goals, details=details,
        context=ctx, optA=opts.get("A",""), optB=opts.get("B",""),
        optC=opts.get("C",""), optD=opts.get("D",""),
        chunk_id=chunk_id or "UNKNOWN",
        history=history_text or "(none)"
    )

def safe_chunk_key(cid):
    try: return (int(cid), str(cid))
    except: return (10**12, str(cid))

def load_histories_from_profile(profile_data):
    cl = profile_data.get("characterList") or {}
    histories = defaultdict(list)
    for _, obj in cl.items():
        if not isinstance(obj, dict): continue
        can = obj.get("NameCanonical")
        if not can: continue
        hs = obj.get("UtteranceHistory") or []
        if not isinstance(hs, list): continue
        for e in hs:
            if not isinstance(e, dict): continue
            cid = e.get("chunk_id")
            utt = (e.get("utterance") or "").strip()
            ctx  = (e.get("context") or "").strip()
            if not cid or not utt: continue
            histories[can].append({"chunk_id": cid, "utterance": utt, "context": ctx})
    for k in histories:
        histories[k].sort(key=lambda x: safe_chunk_key(x["chunk_id"]))
    return histories

def format_history(items, max_items=12):
    if not items: return ""
    tail = items[-max_items:] if max_items else items
    buf = []
    for it in tail:
        cid = it.get("chunk_id","")
        txt = it.get("utterance","")
        buf.append(f"- [chunk {cid}] {txt}")
    return "\n".join(buf)

# =============== Evaluation (Model Response) ===============
def evaluate_mcq_only(choices_path, profile_path, model, sample_n=None, report_path=None, wrong_report_path=None,
                      temperature=0.0, history_max=12):
    """
    Read choices.jsonl, combine with profile.json's history and character settings,
    construct Prompt to let the model choose one from A/B/C/D.
    Returns DataFrame, and can write results to report_path(JSONL) for later --reuse-report.
    """
    global STOP
    
    print_section(f"Evaluation Configuration", "-")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"History Max: {history_max}")
    print(f"Sample Size: {'all' if not sample_n else sample_n}")
    
    print_section("Loading Data", "-")
    print("Loading profiles...")
    profile_data, profiles_map = load_profiles(profile_path)
    histories_map = load_histories_from_profile(profile_data)
    print(f"Found {len(profiles_map)} profile entries")

    print("\nLoading choices...")
    rows = [json.loads(l) for l in open(choices_path,encoding="utf-8") if l.strip()]
    print(f"Found {len(rows)} choice entries")
    
    if sample_n and sample_n < len(rows):
        random.shuffle(rows); rows = rows[:sample_n]
        print(f"Sampled {sample_n} entries for evaluation")

    total = correct = 0
    results, wrongs = [], []

    print_section("Starting Evaluation", "-")
    print(f"Processing {len(rows)} entries...\n")
    
    for idx,row in enumerate(rows,1):
        if STOP: break
        mcq = row.get("mcq") or {}
        opts = mcq.get("options") or {}
        ans = (mcq.get("answer") or "").upper()
        ctx = row.get("context") or ""
        spk = row.get("speaker") or ""
        chunk_id = row.get("chunk_id")
        
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(rows)} entries processed")

        if not ctx or not opts or ans not in {"A","B","C","D"}:
            continue

        match = best_match(spk, set(profiles_map.keys()))
        prof_obj = profiles_map.get(match)
        traits,goals,details = persona_fields(prof_obj)

        canonical_spk = get_canonical_name(spk, profiles_map)
        hist_all = histories_map.get(canonical_spk, [])
        if chunk_id:
            hist_slice = [h for h in hist_all if safe_chunk_key(h["chunk_id"]) < safe_chunk_key(chunk_id)]
        else:
            hist_slice = hist_all
        history_text = format_history(hist_slice, max_items=history_max)

        prompt = build_prompt(spk, traits, goals, details, ctx, opts, chunk_id, history_text)
        try:
            resp = chat(model, prompt, json_mode=True, temperature=temperature)
        except KeyboardInterrupt:
            print("safe exit"); break

        choice = parse_choice(resp if isinstance(resp,str) else json.dumps(resp))
        choice_text = opts.get(choice or "", "")
        answer_text = opts.get(ans, "")
        ok = (choice == ans)
        total += 1
        if ok: correct += 1

        rec = {
            "chunk_id": str(chunk_id) if chunk_id is not None else None,  # Convert to str
            "speaker": spk,
            "choice": choice,
            "choice_text": choice_text,
            "answer": ans,
            "answer_text": answer_text,
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
            print(f"- chunk {w['chunk_id']} | speaker={w['speaker']} | picked={w['choice']} | answer={w['answer']}")

    if report_path:
        print(f"\nSaving results to {report_path}...")
        with open(report_path,"w",encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print("Results saved successfully")

    return pd.DataFrame(results)

# =============== Load Reused Reports/Annotations ===============
def load_eval_report(path):
    rows = [json.loads(l) for l in open(path,encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if "chunk_id" in df.columns:
        df["chunk_id"] = df["chunk_id"].astype(str)
    return df

CAP_LIST = [
    "Opinion_Consistency","Memory_Recall","Logical_Reasoning",
    "Lexical_Fidelity","Persona_Tone","Syntactic_Style"
]

def _flatten_ann_row(obj):
    ae = obj.get("all_evaluations") or {}
    row = {"chunk_id": str(obj.get("chunk_id")) if obj.get("chunk_id") is not None else None}
    for c in CAP_LIST:
        v = ae.get(c) or {}
        row[c] = bool(v.get("label")) if isinstance(v, dict) else False
    return row

def load_annotations(path):
    rows = [json.loads(l) for l in open(path,encoding="utf-8") if l.strip()]
    flat = [_flatten_ann_row(r) for r in rows]
    df = pd.DataFrame(flat)
    if "chunk_id" in df.columns:
        df["chunk_id"] = df["chunk_id"].astype(str)
    return df

# =============== Capability-based Accuracy Report ===============
def compute_required_cap_accuracy(eval_df: pd.DataFrame, ann_df: pd.DataFrame):
    """
    Merge eval with annotations, compute accuracy for each capability column (subset where True).
    Returns DataFrame: [capability, n, acc(%)]
    """
    M = eval_df.merge(ann_df, on="chunk_id", how="inner").copy()
    rows = []
    for cap in CAP_LIST:
        sub = M[M[cap] == True]
        n = len(sub)
        acc = (sub["correct"].mean()*100.0) if n else float("nan")
        rows.append({"capability": cap, "n": n, "acc(%)": round(acc, 2) if pd.notna(acc) else None})
    return pd.DataFrame(rows).sort_values(["n","acc(%)"], ascending=[False, False])

# =============== CLI ===============
def print_section(title, char="="):
    """Print a section title with decorative lines"""
    line = char * 50
    print(f"\n{line}")
    print(f"{title}")
    print(f"{line}\n")

def main():
    signal.signal(signal.SIGINT,_handle_sigint)
    print_section("Dimension 3 Evaluation Start", "=")
    
    ap = argparse.ArgumentParser(description="Evaluate dialogue system using multiple-choice questions")
    ap.add_argument("choices_jsonl", default="dataset/dimension_3/choices.jsonl", help="Path to choices data file")
    ap.add_argument("profile_json", default="dataset/dimension_3/profiles.jsonl", help="Path to profile data file")

    # Step 1: Evaluation
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name to use for evaluation")
    ap.add_argument("--sample", type=int, help="Number of samples to evaluate (optional)")
    ap.add_argument("--report", default="result/discriminative/dimension_3/results.jsonl", help="Path to save evaluation results (JSONL format)")
    ap.add_argument("--wrong-report", default="result/discriminative/dimension_3/wrong_cases.jsonl", help="Path to save wrong cases")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--history-max", type=int, default=30, help="Maximum number of history items to include")
    ap.add_argument("--cap-report-csv", default="result/discriminative/dimension_3/capability_report.csv", help="Path to save capability analysis report (CSV format)")

    # Step 2: Reuse + Capability statistics (required-cap only)
    ap.add_argument("--reuse-report", action="store_true", help="Don't re-evaluate, reuse JSONL from --report")
    ap.add_argument("--annotations", help="Capability annotation JSONL (annotated_by_*.jsonl)")

    args = ap.parse_args()

    # Select eval_df source
    if args.reuse_report:
        if not args.report:
            raise ValueError("When using --reuse-report, you must also provide --report <path-to-jsonl>.")
        eval_df = load_eval_report(args.report)
        print(f"[Info] Reused existing report: {args.report} ({len(eval_df)} rows)")
    else:
        eval_df = evaluate_mcq_only(
            choices_path=args.choices_jsonl,
            profile_path=args.profile_json,
            model=args.model,
            sample_n=args.sample,
            report_path=args.report,
            wrong_report_path=args.wrong_report,
            temperature=args.temperature,
            history_max=args.history_max
        )
        print(f"[Info] Fresh evaluation rows: {len(eval_df)}")

    # Can end here if only doing evaluation
    if not args.annotations:
        return

    # Load capability annotations and compute statistics
    try:
        print_section("Loading Annotations", "-")
        print(f"Loading annotations from {args.annotations}...")
        ann_df = load_annotations(args.annotations)
        print(f"Found {len(ann_df)} annotation entries")
        
        cap_required_df = compute_required_cap_accuracy(eval_df, ann_df)
    except FileNotFoundError:
        print(f"\nWarning: Annotation file '{args.annotations}' not found.")
        print("Skipping capability analysis...")
        cap_required_df = pd.DataFrame()

    if not cap_required_df.empty:
        print("\n=== Accuracy by REQUIRED capability (label==True) ===")
        print(cap_required_df.to_string(index=False))
    else:
        print("\n[Warn] Empty capability-required report")

    if args.cap_report_csv and not cap_required_df.empty:
        cap_required_df.to_csv(args.cap_report_csv, index=False)
        print(f"[Save] {args.cap_report_csv}")

    # Save and print summary
    summary_path = os.path.join(os.path.dirname(args.report), "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Write configuration section
        f.write("=== Dimension 3 Discriminative Evaluation Summary ===\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Sample Size: {'all' if not args.sample else args.sample}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"History Max: {args.history_max}\n\n")
        
        # Write results section
        f.write("=== Results ===\n\n")
        total_samples = len(eval_df)
        correct_samples = eval_df["correct"].sum()
        accuracy = (correct_samples/total_samples*100) if total_samples else 0
        
        f.write(f"Total Items: {total_samples}\n")
        f.write(f"Correct Answers: {correct_samples}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        
        # Write capability analysis if available
        if not cap_required_df.empty:
            f.write("=== Capability Analysis ===\n\n")
            f.write(cap_required_df.to_string())
            f.write("\n\n")
        
        # Write file locations
        f.write("\nFull results saved in:\n")
        f.write(f"- {os.path.basename(args.report)}\n")
        if args.wrong_report:
            f.write(f"- {os.path.basename(args.wrong_report)}\n")
        if args.cap_report_csv and not cap_required_df.empty:
            f.write(f"- {os.path.basename(args.cap_report_csv)}\n")
    
    # Print summary to console
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    print(f"📈 Total Samples: {total_samples}")
    print(f"✅ Correct: {correct_samples}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    
    if not cap_required_df.empty:
        print("\n📋 Capability Analysis:")
        # Print top 3 and bottom 3 capabilities by accuracy
        sorted_caps = cap_required_df.sort_values("acc(%)", ascending=False)
        print("\nTop 3 capabilities:")
        for _, row in sorted_caps.head(3).iterrows():
            print(f"  - {row['capability']}: {row['acc(%)']:.2f}% (n={row['n']})")
        print("\nBottom 3 capabilities:")
        for _, row in sorted_caps.tail(3).iterrows():
            print(f"  - {row['capability']}: {row['acc(%)']:.2f}% (n={row['n']})")
    
    print("\n💾 Results saved to:")
    print(f"  - Summary: {os.path.basename(summary_path)}")
    print(f"  - Details: {os.path.basename(args.report)}")
    print("="*50)

if __name__=="__main__":
    main()