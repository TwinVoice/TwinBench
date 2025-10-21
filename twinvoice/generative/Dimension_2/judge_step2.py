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
def get_client():
    """Return LLM-as-a-Judge API client configuration"""
    return OpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key
    )

# Default client uses Judge API
client = get_client()

# ================== Prompt Template ==================
JUDGE_PROMPT = """You are a meticulous and objective evaluator for a digital twin benchmark. Your task is to assess how well a 'Generated Reply' replicates a 'Ground Truth Reply' for a given interpersonal messaging interaction.

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
You will be provided with the original context message, the ground truth reply, and the generated reply. User-generated content may be in different languages, but your analysis and final JSON output must be in English. You MUST respond ONLY with a JSON object in the following format. Do not include any other text or explanations.

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

# ================== Utilities ==================
def parse_json(text: str):
    """Parse JSON from model response"""
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON block
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text or "", flags=re.I)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Try to extract any JSON-like object
        m = re.search(r'\{[\s\S]*?\}', text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"error": "JSON_PARSE_FAILED", "raw_text": text}
    return {"error": "JSON_PARSE_FAILED", "raw_text": text}

# ================== Judge Function ==================
def judge_generation(sample, judge_model, temperature=0.0):
    """Judge a single generation"""
    # Skip if no valid generation
    if not sample.get('lmut_reply') or "API_ERROR" in str(sample.get('lmut_reply')):
        sample['judge_status'] = 'Skipped'
        return sample
    
    # Get ground truth
    ground_truth = sample['choices'][sample['answer_idx']]
    
    # Build prompt
    prompt = JUDGE_PROMPT.format(
        context=sample.get('context', ''),
        ground_truth_reply=ground_truth,
        lmut_reply=sample['lmut_reply']
    )
    
    # Call model
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        resp_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        sample['judge_status'] = 'Failed'
        sample['judge_result'] = {"error": f"API_ERROR: {str(e)}"}
        return sample
    
    # Parse response
    result = parse_json(resp_text)
    if "error" in result:
        sample['judge_status'] = 'Failed'
        sample['judge_result'] = result
    else:
        sample['judge_status'] = 'Success'
        sample['judge_result'] = result
        # Extract score for convenience
        try:
            sample['score_gen_1to5'] = int(result['final_score'])
            sample['score_gen_norm'] = (sample['score_gen_1to5'] - 1) / 4.0  # 1..5 -> 0..1
        except (KeyError, ValueError, TypeError):
            sample['score_gen_1to5'] = None
            sample['score_gen_norm'] = None
    
    return sample

def main():
    parser = argparse.ArgumentParser(description="Judge generated replies")
    parser.add_argument("--input", required=True,
                       help="Path to generation results file")
    parser.add_argument("--judge_model", default="gpt-5-chat",
                       help="Judge model name")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load generations
    print(f"\nLoading generations from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} samples")
    
    # Judge generations
    print("\nStarting judgement...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(judge_generation, s, args.judge_model, args.temperature)
            for s in samples
        ]
        for future in tqdm(as_completed(futures), total=len(samples), desc="Judging"):
            results.append(future.result())
    
    # Sort results
    results.sort(key=lambda x: x.get('line_idx', 0))
    
    # Save results
    out_dir = os.path.dirname(args.input)
    out_path = os.path.join(out_dir, f"Score_and_Analysis_by_{os.path.basename(args.judge_model)}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Calculate statistics
    success = [r for r in results if r['judge_status'] == 'Success']
    scores = [r['score_gen_1to5'] for r in success if r.get('score_gen_1to5') is not None]
    norm_scores = [r['score_gen_norm'] for r in success if r.get('score_gen_norm') is not None]
    
    # Print summary
    print("\n" + "="*60)
    print("🔍 JUDGE EVALUATION SUMMARY")
    print("="*60)
    print(f"📊 Total Samples: {len(results)}")
    print(f"✅ Success: {len(success)}")
    print(f"❌ Failed/Skipped: {len(results) - len(success)}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        avg_norm = sum(norm_scores) / len(norm_scores)
        print(f"\n📈 Score Statistics:")
        print(f"  Score(1-5): {avg_score:.2f}")
        print(f"  Score(0-1): {avg_norm:.3f}")
        
        # Score distribution
        from collections import Counter
        dist = Counter(scores)
        print("\n📊 Score Distribution:")
        for score in sorted(dist.keys()):
            count = dist[score]
            percentage = count/len(scores)*100
            print(f"  Score {score}: {count} samples ({percentage:.1f}%)")
    
    print("\n💾 Results saved to:")
    print(f"  - {out_path}")
    print("="*60)

if __name__ == "__main__":
    main()