# -*- coding: utf-8 -*-
import json, os, re, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from twinvoice.api_config import judge_base_url, judge_api_key

# ================== API Configuration ==================
# Judge step only uses LLM-as-a-Judge API

# Default judge model (can be overridden with --judge_model)
DEFAULT_JUDGE_MODEL = "gpt-5-chat"

# Concurrency and retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5

# ================== Prompt Templates ==================
MAP_PROMPT_TEMPLATE = """You are a strict classifier. Output ONLY a single letter (A/B/C/D).
Choose the option that best matches the style, tone, vocabulary, and stance of the Generated Reply.

[Options]
A. {A}
B. {B}
C. {C}
D. {D}

[Generated Reply]
{pred}

Output exactly one letter: A, B, C, or D.
"""

SCORE_PROMPT_TEMPLATE = """You are a meticulous and objective evaluator for a digital twin benchmark. Your task is to assess how well a 'Generated Reply' replicates a 'Ground Truth Reply' for a given social media post.

The 'Ground Truth Reply' is the single, undisputed gold standard. Your entire evaluation must be based on comparing the 'Generated Reply' against it.

The evaluation rests on three key pillars:
1.  **Opinion Consistency**: Does the 'Generated Reply' express the exact same core opinion, stance, and sentiment as the 'Ground Truth'?
2.  **Logical & Factual Fidelity**: Is the 'Generated Reply' based on the same reasoning and facts as the 'Ground Truth'? It must not introduce new, unsupported information or follow a different logical path.
3.  **Stylistic Similarity**: How closely does the 'Generated Reply' match the 'Ground Truth' in terms of writing style?
    *   **Lexical**: Use of similar vocabulary, slang, or emojis.
    *   **Tone**: Capturing the same tone (e.g., humorous, sarcastic, empathetic, proud).
    *   **Syntactic**: Similarity in sentence structure, length, and degree of formality.

---
**SCORING RUBRIC (1-5 Scale):**

-   **5: Perfect Replication**: The 'Generated Reply' is a perfect match across all three pillars (Opinion, Logic/Factual, Style). It feels like a natural, alternative expression from the same user. A perfect substitute for the ground truth.

-   **4: High Fidelity**: The Opinion and Logic/Factual pillars are perfectly matched. There are only minor, subtle differences in the Style pillar (e.g., a missing catchphrase, a slightly more formal tone), but the reply is still an excellent imitation.

-   **3: Core Alignment, Detail Loss**: The core Opinion is consistent, but there's a noticeable loss of detail in the Logic or Style pillars. For example, the tone is flattened, or unique phrasing is lost. The reply captures the 'what' but not the 'how'. It feels more like a summary than a replication.

-   **2: Partial Relevance, Major Deviation**: There is a major failure in at least one of the three pillars. For instance, the opinion is distorted (e.g., strong support becomes neutral), the logic is completely different, or the style is entirely mismatched.

-   **1: Irrelevant or Contradictory**: The 'Generated Reply' has almost nothing in common with the 'Ground Truth' or expresses a contradictory opinion. A total failure of replication.

---
**YOUR TASK:**
You will be provided with the original post, the ground truth reply, and the generated reply. All user-generated content is in Chinese, but your analysis and final JSON output must be in English. You MUST respond ONLY with a JSON object in the following format. Do not include any other text or explanations.

```json
{{
  "analysis": {{
    "opinion_consistency": {{
      "is_consistent": true,
      "justification": "A brief justification for the consistency of the opinion."
    }},
    "logical_factual_fidelity": {{
      "is_faithful": true,
      "justification": "A brief justification for the fidelity of the logic and facts."
    }},
    "stylistic_similarity": {{
      "similarity_level": "High/Medium/Low",
      "justification": "A brief justification for the level of stylistic similarity."
    }}
  }},
  "final_score": "An integer score from 1 to 5",
  "final_justification": "A concise, overall justification for the final score, synthesizing the three pillars."
}}
Now, evaluate the following case:

Context Message:
"{context}"

Ground Truth Reply:
"{ground_truth_reply}"

Generated Reply to Evaluate:
"{lmut_reply}"
"""

# ================== API Client Management ==================
def get_client():
    """Return LLM-as-a-Judge API client configuration"""
    return OpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key
    )

# Default client uses Judge API
client = get_client()

def call_llm(model, messages, temperature=0.0, response_format=None, max_tokens=None):
    """Call LLM with Judge API
    
    Args:
        model: Model name
        messages: Chat messages
        temperature: Sampling temperature
        response_format: Response format (e.g., {"type": "json_object"})
        max_tokens: Maximum tokens to generate
    
    Returns:
        Model response text or error message
    """
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            if response_format is not None:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Warn] API call failed {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f'{{"error":"API_ERROR: {str(e)}"}}'

def parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text or "")
        if m:
            try: return json.loads(m.group(1))
            except: pass
        m = re.search(r'\{[\s\S]*?\}', text or "")
        if m:
            try: return json.loads(m.group(0))
            except: return {"error":"JSON_PARSE_FAILED","raw_text":text}
    return {"error":"JSON_PARSE_FAILED","raw_text":text}

# ================== Judger1: Map A/B/C/D → Acc.(Gen) ==================
def judger1_map(sample, judge_model):
    if not sample.get("lmut_reply") or "API_ERROR" in str(sample.get("lmut_reply")):
        sample["step2a_status"] = "Skipped"
        return sample

    # 兼容两种结构
    opts = None
    gold_letter = None
    mcq = sample.get("mcq") or {}
    if isinstance(mcq.get("options"), dict) and mcq.get("answer") in mcq["options"]:
        opts = mcq["options"]; gold_letter = mcq["answer"]
    elif isinstance(sample.get("choices"), list) and isinstance(sample.get("answer_idx"), int):
        ch = sample["choices"]; idx = sample["answer_idx"]
        if len(ch) == 4 and 0 <= idx < 4:
            opts = {"A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3]}
            gold_letter = "ABCD"[idx]

    if not (opts and gold_letter in "ABCD"):
        sample["step2a_status"] = "Failed"
        return sample

    prompt = MAP_PROMPT_TEMPLATE.format(
        A=opts["A"], B=opts["B"], C=opts["C"], D=opts["D"],
        pred=sample["lmut_reply"]
    )
    resp = call_llm(judge_model, [{"role":"user","content":prompt}], temperature=0.0, max_tokens=10)

    guess = (resp or "").strip().upper()
    if guess not in "ABCD":
        m = re.search(r'\b([ABCD])\b', guess)
        guess = m.group(1) if m else None

    sample["mapped_choice"] = guess
    sample["gold_letter"] = gold_letter
    sample["acc_gen"] = int(guess == gold_letter) if (guess and gold_letter) else None
    sample["judge_map_raw"] = resp
    sample["step2a_status"] = "Success" if guess else "Failed"
    return sample

# ================== Judger2: Score 1-5 & Normalize 0-1 ==================
def judger2_score(sample, judge_model):
    if not sample.get("lmut_reply") or "API_ERROR" in str(sample.get("lmut_reply")):
        sample["step2b_status"] = "Skipped"
        return sample

    # 取 ground truth 文本
    gold = None
    mcq = sample.get("mcq") or {}
    if isinstance(mcq.get("options"), dict) and mcq.get("answer") in mcq["options"]:
        gold = mcq["options"][mcq["answer"]]
    elif isinstance(sample.get("choices"), list) and isinstance(sample.get("answer_idx"), int):
        ch = sample["choices"]; idx = sample["answer_idx"]
        if len(ch) == 4 and 0 <= idx < 4:
            gold = ch[idx]
    if not gold:
        sample["step2b_status"] = "Failed"
        return sample

    prompt = SCORE_PROMPT_TEMPLATE.format(
        context=sample.get("context",""),
        ground_truth_reply=gold,
        lmut_reply=sample["lmut_reply"]
    )
    msgs = [
        {"role":"system","content":"You are a helpful assistant designed to output JSON."},
        {"role":"user","content":prompt}
    ]
    resp = call_llm(judge_model, msgs, temperature=0.0, response_format={"type":"json_object"})
    obj = parse_json(resp)

    score = obj.get("final_score")
    try: score = int(score)
    except: score = None

    norm = None if score is None else round((score - 1) / 4.0, 4)  # 1..5 -> 0..1

    sample["judge_status"]   = "Success" if score is not None else "Failed"
    sample["judge_result"]   = obj
    sample["score_gen_1to5"] = score
    sample["score_gen_norm"] = norm
    sample["step2b_status"]  = "Success" if score is not None else "Failed"
    return sample

# ================== Main Process ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to step1_generations_*.jsonl")
    ap.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = in_path.parent
    base = in_path.with_suffix("").name
    out_path = out_dir / f"Score_and_Map_on_{base}_by_{args.judge_model}.jsonl"

    # 读取
    samples = []
    with in_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip(): continue
            obj = json.loads(line); obj.setdefault("line_idx", i)
            samples.append(obj)
    print(f"[Load] {len(samples)} samples from {in_path}")

    # Judger1 并发
    mapped = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judger1_map, s, args.judge_model) for s in samples]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Judger1 Mapping"):
            mapped.append(fu.result())

    # Judger2 并发
    final_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judger2_score, s, args.judge_model) for s in mapped]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Judger2 Scoring"):
            final_results.append(fu.result())

    final_results.sort(key=lambda x: x.get("line_idx", 0))
    with out_path.open("w", encoding="utf-8") as fout:
        for r in final_results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总
    acc_list = [r.get("acc_gen") for r in final_results if isinstance(r.get("acc_gen"), int)]
    s15_list = [r.get("score_gen_1to5") for r in final_results if isinstance(r.get("score_gen_1to5"), int)]
    sn_list  = [r.get("score_gen_norm") for r in final_results if isinstance(r.get("score_gen_norm"), float)]

    def avg(xs): return sum(xs)/len(xs) if xs else None
    acc_pct = (sum(acc_list)/len(acc_list)*100) if acc_list else None
    s15_avg = avg(s15_list); sn_avg = avg(sn_list)

    print("\n" + "="*60)
    print("--- FINAL EVALUATION REPORT ---")
    print("="*60)
    print(f"Samples: {len(final_results)}")
    print(f"Acc.(Gen): {acc_pct:.2f}%  (n={len(acc_list)})" if acc_pct is not None else "Acc.(Gen): N/A")
    print(f"Score(Gen) 1–5: {s15_avg:.2f}  (n={len(s15_list)})" if s15_avg is not None else "Score(Gen) 1–5: N/A")
    print(f"Score(Gen) 0–1: {sn_avg:.3f}  (n={len(sn_list)})" if sn_avg is not None else "Score(Gen) 0–1: N/A")
    print("="*60)
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()