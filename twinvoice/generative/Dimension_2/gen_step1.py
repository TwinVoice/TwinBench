# -*- coding: utf-8 -*-
import json, os, re, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from twinvoice.api_config import twin_base_url, twin_api_key

# ================== API Configuration ==================
# Generative task only uses Digital Twin API
def get_client(model_name=None):
    """Return Digital Twin API client configuration"""
    return OpenAI(
        base_url=twin_base_url,
        api_key=twin_api_key
    )

# Default client uses Digital Twin API
client = get_client()

# ================== Generation Prompt ==================
GEN_PROMPT = """You are acting as a digital twin of a specific messaging app user.
Your task is to analyze the user's messaging history to understand their personality, tone, vocabulary, and style.
Different provided text (history, context, message) may use different language. You must analyze and respond in the same language as the provided text.

Here is the user's messaging history:
(Note: "Context" is a message by someone else, and "UserReply" is the user's own reply to it.)
---
{history_text}
---

Now, you must imitate this user's persona perfectly and write a new reply to the following message.

Please include your response in the following JSON format:
{{"generated_content": "your reply text here"}}

Message to reply to:
"{context}"
"""

# ================== Utilities ==================
def parse_generated_content(text: str):
    """Extract generated content from model response"""
    # Try direct JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "generated_content" in obj:
            return (obj["generated_content"] or "").strip()
    except Exception:
        pass
    
    # Try regex extraction
    m = re.search(r'"generated_content"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    
    # Try JSON block extraction
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text or "", flags=re.I)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "generated_content" in obj:
                return (obj["generated_content"] or "").strip()
        except Exception:
            pass
    
    # Return the whole text as fallback
    return text.strip()

# ================== Generation ==================
def generate_reply(sample, model, temperature=0.0):
    """Generate reply for a single sample"""
    # Build history text
    history_text = "\n".join(sample.get('history', []))
    
    # Build prompt
    prompt = GEN_PROMPT.format(
        history_text=history_text,
        context=sample.get('context', '')
    )
    
    # Call model
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        resp_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return {
            "line_idx": sample.get("line_idx"),
            "context": sample.get("context"),
            "choices": sample.get("choices"),
            "answer_idx": sample.get("answer_idx"),
            "history": sample.get("history"),
            "lmut_reply": f"API_ERROR: {str(e)}",
            "step1_status": "Failed"
        }
    
    # Parse response
    gen = parse_generated_content(resp_text)
    if not gen:
        return {
            "line_idx": sample.get("line_idx"),
            "context": sample.get("context"),
            "choices": sample.get("choices"),
            "answer_idx": sample.get("answer_idx"),
            "history": sample.get("history"),
            "lmut_reply": f"PARSE_ERROR: {resp_text}",
            "step1_status": "Failed"
        }
    
    # Return result
    return {
        "line_idx": sample.get("line_idx"),
        "context": sample.get("context"),
        "choices": sample.get("choices"),
        "answer_idx": sample.get("answer_idx"),
        "history": sample.get("history"),
        "lmut_reply": gen,
        "step1_status": "Success"
    }

def main():
    parser = argparse.ArgumentParser(description="Generate replies for conversation data")
    parser.add_argument("--input", default="dataset/dimension_2/conversation_data.jsonl",
                       help="Path to conversation data file")
    parser.add_argument("--gen_model", required=True,
                       help="Generation model name")
    parser.add_argument("--out_dir", default="result/generative/dimension_2",
                       help="Output directory")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--sample", type=int,
                       help="Number of samples to process (optional)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]
        for i, s in enumerate(samples):
            s['line_idx'] = i + 1
    print(f"Loaded {len(samples)} samples")
    
    # Sample if requested
    if args.sample and args.sample < len(samples):
        import random
        random.shuffle(samples)
        samples = samples[:args.sample]
        print(f"Sampled {args.sample} samples for testing")
    
    # Generate replies
    print("\nStarting generation...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(generate_reply, s, args.gen_model, args.temperature)
            for s in samples
        ]
        for future in tqdm(as_completed(futures), total=len(samples), desc="Generating"):
            results.append(future.result())
    
    # Sort results by line index
    results.sort(key=lambda x: x.get('line_idx', 0))
    
    # Save results
    out_path = os.path.join(args.out_dir, f"step1_generations_{os.path.basename(args.gen_model)}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Print statistics
    success = sum(1 for r in results if r['step1_status'] == 'Success')
    failed = len(results) - success

        print("\n" + "="*60)
    print("📝 GENERATION SUMMARY")
        print("="*60)
    print(f"📊 Total Samples: {len(results)}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(success/len(results)*100):.2f}%")
    
    # Print example generations
    success_cases = [r for r in results if r['step1_status'] == 'Success'][:3]
    if success_cases:
        print("\n📋 Example Generations:")
        for i, case in enumerate(success_cases, 1):
            print(f"\nExample {i}:")
            print(f"Context: {case['context'][:50]}...")
            print(f"Generated: {case['lmut_reply'][:50]}...")
    
    print("\n💾 Results saved to:")
    print(f"  - {out_path}")
        print("="*60)

if __name__ == "__main__":
    main()