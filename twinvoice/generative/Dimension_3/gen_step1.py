# -*- coding: utf-8 -*-
import json, os, re, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from twinvoice.api_config import twin_base_url, twin_api_key

# ================== API Configuration ==================
# Generative task only uses Digital Twin API

# Concurrency and retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5

# ================== Generation Prompt (with persona + history) ==================
GEN_PROMPT = """You are the digital twin of the TARGET speaker in a literary dialogue dataset.

Your job: write ONE new reply that this TARGET would plausibly say **in the exact context below**, referencing to their historical voice and habits, but must pay attention to the current context.

TARGET speaker: {speaker}

Persona summary (may be partial):
- Traits: {traits}
- Goals: {goals}
- Details: {details}

Scene context (preceding narration & situation, NOT the speaker’s own words):
\"\"\"{context}\"\"\"

Consider the history utterances of this character. Style anchors from the TARGET's prior utterances (chronological, up to BEFORE chunk {chunk_id}):
{history_block}

Hard requirements (STRICT):
1) Language & Era: match the character's tone/era considering their dialogue history.
2) Persona Fit: keep the TARGET’s formality, cadence, and turns of phrase.
3) Scene Consistency: must be logically possible given the context; introduce no new facts/characters/locations.
4) Length & Shape: one spoken line only (no stage directions); prefer 8–28 words unless a very short assent/command is natural.
5) No Copying: do NOT copy any exact sentence from the dataset.


Output ONLY strict JSON:
{{ "generated_content": "<the single line>" }}
"""

# ================== Profile/History Utilities ==================
def load_profiles(path: str):
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

def best_match(name: str, keys: set):
    if not name: return None
    if name in keys: return name
    low = name.lower()
    if low in keys: return low
    import difflib
    m = difflib.get_close_matches(low, [k for k in keys if isinstance(k,str) and k.islower()], n=1, cutoff=0.82)
    return m[0] if m else None

def get_canonical_name(name: str, profiles_map: Dict[str, Any]):
    mk = best_match(name, set(profiles_map.keys()))
    if not mk: return name
    obj = profiles_map.get(mk) or {}
    return obj.get("NameCanonical") or name

def persona_fields(p: Optional[Dict[str, Any]]):
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

def safe_chunk_key(cid):
    try: return (int(cid), str(cid))
    except: return (10**12, str(cid))

def load_histories_from_profile(profile_data: Dict[str, Any]):
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
            ctx = (e.get("context") or "").strip()
            if not cid or not utt: continue
            histories[can].append({"chunk_id": cid, "utterance": utt, "context": ctx})
    for k in histories:
        histories[k].sort(key=lambda x: safe_chunk_key(x["chunk_id"]))
    return histories

def format_history(items: List[Dict[str, Any]], max_items=30):
    if not items: return "(none)"
    tail = items[-max_items:] if max_items else items
    buf = []
    for it in tail:
        buf.append(f"- [chunk {it.get('chunk_id','')}] {it.get('utterance','')}")
    return "\n".join(buf)

# ================== API Client Management ==================
def get_client(model_name=None):
    """Return Digital Twin API client configuration"""
    return OpenAI(
        base_url=twin_base_url,
        api_key=twin_api_key
    )

# Default client uses Digital Twin API
client = get_client()

def call_llm(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 256):
    """Call LLM with Digital Twin API
    
    Args:
        model: Model name
        messages: Chat messages
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Model response text or error message
    """
    # Use Digital Twin API client
    current_client = get_client(model)
    
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 90
            }
            
            # Digital Twin models generally don't support json_mode, use regex parsing instead
            # Can be manually enabled if the model supports json_mode
            # kwargs["response_format"] = {"type": "json_object"}
            
            resp = current_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Warn] API call failed {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"API_ERROR: {str(e)}"

def extract_generated_content(text: str):
    """Extract generated content from model response
    
    Args:
        text: Model response text
    
    Returns:
        Extracted content or None
    """
    # Digital Twin models may not return JSON format, use lenient parsing
    text = text.strip()
    
    # 1) If text is wrapped in quotes, extract directly
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].strip()
    
    # 2) Try direct JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "generated_content" in obj:
            return (obj["generated_content"] or "").strip()
    except Exception:
        pass
    
    # 3) If contains generated_content field, try to extract with regex
    m = re.search(r'"generated_content"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    
    # 4) Wrapped in ```json
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text or "", flags=re.I)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "generated_content" in obj:
                return (obj["generated_content"] or "").strip()
        except Exception:
            pass
    
    # 5) Lenient extraction
    m = re.search(r'\{[^{}]*"generated_content"\s*:\s*"([\s\S]*?)"\s*[^{}]*\}', text or "")
    if m:
        return (m.group(1) or "").strip()
    
    # 6) Otherwise, return the whole text
    return text.strip()

# ================== Single Sample Generation ==================
def generate_for_sample(sample: Dict[str, Any],
                        gen_model: str,
                        profiles_map: Dict[str, Any],
                        histories_map: Dict[str, List[Dict[str,Any]]],
                        history_max: int):
    spk = sample.get("speaker") or sample.get("speaker_match_key") or "Unknown Speaker"
    ctx = sample.get("context") or ""
    chunk_id = sample.get("chunk_id")

    match = best_match(spk, set(profiles_map.keys()))
    prof_obj = profiles_map.get(match)
    traits, goals, details = persona_fields(prof_obj)

    canonical_spk = get_canonical_name(spk, profiles_map)
    hist_all = histories_map.get(canonical_spk, [])
    if chunk_id:
        hist_all = [h for h in hist_all if safe_chunk_key(h["chunk_id"]) < safe_chunk_key(chunk_id)]
    history_block = format_history(hist_all, max_items=history_max)

    prompt = GEN_PROMPT.format(
        speaker=spk,
        traits=traits, goals=goals, details=details,
        context=ctx,
        history_block=history_block,
        chunk_id=chunk_id or "UNKNOWN"
    )

    resp = call_llm(gen_model, [{"role":"user","content":prompt}], temperature=0.2, max_tokens=256)

    sample = dict(sample)
    if isinstance(resp, str) and resp.startswith("API_ERROR"):
        sample["lmut_reply"] = resp
        sample["step1_status"] = "Failed"
        return sample

    # Extract generated content
    gen = extract_generated_content(resp or "")
    if not gen:
        sample["lmut_reply"] = f"PARSE_ERROR: {resp}"
        sample["step1_status"] = "Failed"
        return sample

    sample["lmut_reply"] = gen.strip().strip('"').strip()
    sample["step1_status"] = "Success"
    return sample

# ================== Main Process ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="choices.jsonl (contains speaker/context/chunk_id)")
    ap.add_argument("--profile", required=True, help="profile.json (contains persona + UtteranceHistory)")
    ap.add_argument("--gen_model", required=True, help="Generation model name")
    ap.add_argument("--out_dir", default="evaluation_results", help="Output directory")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    ap.add_argument("--history-max", type=int, default=30, help="Maximum history items")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    os.makedirs(args.out_dir, exist_ok=True)
    base = in_path.with_suffix("").name
    
    # Extract short identifier from model path/name
    model_id = args.gen_model
    if "/" in model_id:
        # If it's a path, take the last part
        model_id = model_id.split("/")[-1]
    
    out_path = Path(args.out_dir) / f"step1_generations_lmut-{model_id}_{base}.jsonl"

    profile_data, profiles_map = load_profiles(args.profile)
    histories_map = load_histories_from_profile(profile_data)

    samples = []
    with in_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            obj = json.loads(line); obj.setdefault("line_idx", i)
            samples.append(obj)
    print(f"[Load] {len(samples)} samples from {in_path}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(generate_for_sample, s, args.gen_model, profiles_map, histories_map, args.history_max)
                for s in samples]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Step1 Generating"):
            results.append(fu.result())

    results.sort(key=lambda x: x.get("line_idx", 0))
    with out_path.open("w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r.get("step1_status") == "Success")
    failed = len(results) - ok
    
    # Save summary
    summary_path = os.path.join(out_dir, "generation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Configuration
        f.write("=== Dimension 3 Generation Summary ===\n\n")
        f.write(f"Model: {args.gen_model}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"History Max: {args.history_max}\n\n")
        
        # Results
        f.write("=== Results ===\n\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Success: {ok}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success Rate: {(ok/len(results)*100):.2f}%\n\n")
        
        # Example generations
        f.write("=== Example Generations ===\n\n")
        success_cases = [r for r in results if r.get("step1_status") == "Success"][:3]
        for i, case in enumerate(success_cases, 1):
            f.write(f"Example {i}:\n")
            f.write(f"Context: {case.get('context', '')[:100]}...\n")
            f.write(f"Generated: {case.get('lmut_reply', '')[:100]}...\n\n")
        
        # File locations
        f.write("\nFull results saved in:\n")
        f.write(f"- {os.path.basename(out_path)}\n")
    
    # Print console summary
    print("\n" + "="*60)
    print("📝 GENERATION SUMMARY")
    print("="*60)
    print(f"📊 Total Samples: {len(results)}")
    print(f"✅ Success: {ok}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(ok/len(results)*100):.2f}%")
    
    # Print example generations
    if success_cases:
        print("\n📋 Example Generations:")
        for i, case in enumerate(success_cases, 1):
            print(f"\nExample {i}:")
            print(f"Context: {case.get('context', '')[:50]}...")
            print(f"Generated: {case.get('lmut_reply', '')[:50]}...")
    
    print("\n💾 Results saved to:")
    print(f"  - Summary: {os.path.basename(summary_path)}")
    print(f"  - Details: {os.path.basename(out_path)}")
    print("="*60)

if __name__ == "__main__":
    main()