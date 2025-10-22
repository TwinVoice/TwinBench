# -*- coding: utf-8 -*-
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
import re

from openai import OpenAI
from tqdm import tqdm

# ======================================================================================
# --- 1. Configuration Section (Please modify your parameters here) ---
# ======================================================================================

# --- API and Model Configuration ---
# Your API Base URL and Key
BASE_URL = "https://api.example.com/"  # Example, please replace with your API address
API_KEY = "YOUR_API_KEY_HERE"  # 【IMPORTANT】Please replace with your real API key

# Define the "judge" model for scoring
JUDGE_MODEL = "gpt-5-chat"  # Use gpt-5-chat according to your decision

# --- Data and Execution Configuration ---
## Need to modify to adapt to different tracks
# 【IMPORTANT】Input file path: This is the file containing replies to be evaluated (lmut_reply)
# For example: "evaluation_results/step1_generations_lmut-gpt-4o-mini_judge-gpt-5-chat_range-1-2000.jsonl"
INPUT_PATH = "evaluation_results/step1_generations_lmut-gpt-oss-20b_judge-gpt-5-chat_range-1-2000.jsonl"

# Output filename will be automatically generated based on input filename and Judge model, no need to set manually

# --- Performance and Robustness Configuration ---
# Number of parallel worker threads (adjust according to your API rate limit and local performance)
NUM_WORKERS = 10
# Number of retries after API call failure
MAX_RETRIES = 3
# Wait time before each retry (seconds)
RETRY_DELAY = 5

## Need to modify to adapt to different tracks
# Scoring Prompt Template - May need to adjust evaluation criteria or context description for different tracks
JUDGE_PROMPT_TEMPLATE = """You are a meticulous and objective evaluator for a digital twin benchmark. Your task is to assess how well a 'Generated Reply' replicates a 'Ground Truth Reply' for a given social media post.

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

Original Post:
"{anchor_post}"

Ground Truth Reply:
"{ground_truth_reply}"

Generated Reply to Evaluate:
"{lmut_reply}"
"""

# ======================================================================================
# --- 3. Core Functionality Functions ---
# ======================================================================================

# Initialize OpenAI client
# Ensure API key is provided, otherwise an error will be thrown
if "sk-xxxxxxxx" in API_KEY or API_KEY == "":
    raise ValueError("Please replace API_KEY with your real API key in the configuration section (Part 1) of the code.")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_llm_api_with_retry(model, messages, temperature=0.0):
    """
    Call LLM API with integrated retry logic.
    For evaluation tasks, use lower temperature to get more stable and consistent output.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}, # Enable JSON mode to ensure output format
                timeout=90  # Evaluation tasks may be more complex, increase timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Warning] API call attempt {attempt + 1}/{MAX_RETRIES} failed. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f'{{"error": "API_ERROR: All {MAX_RETRIES} retries failed. Last error: {e}"}}'


def parse_json_from_response(text):
    """
    Robustly parse JSON object from LLM's response.
    Since JSON mode is enabled, it should theoretically be directly parsable. This function serves as a backup and check.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [Warning] JSON parsing failed. Original text: {text}")
        # Try to extract using regex from text that may be accidentally wrapped
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"error": "JSON_PARSE_FAILED", "raw_text": text}
        return {"error": "JSON_PARSE_FAILED", "raw_text": text}


def perform_judgement(sample):
    """
    Call Judge model to score a single sample.
    """
    # Check if the reply to be evaluated exists and is valid
    if not sample.get('lmut_reply') or "API_ERROR" in sample.get('lmut_reply'):
        sample['judge_status'] = 'Skipped (No valid lmut_reply)'
        return sample

    ## Need to modify to adapt to different tracks
    # 1. Prepare data - Data field access logic
    ground_truth_reply = sample['choices'][sample['answer_idx']]
    
    # 2. Construct judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        anchor_post=sample['anchor_post'],
        ground_truth_reply=ground_truth_reply,
        lmut_reply=sample['lmut_reply']
    )
    
    # 3. Call API and parse results
    messages = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}]
    
    judge_response_text = call_llm_api_with_retry(JUDGE_MODEL, messages)
    
    # 4. Parse and record judgment results
    parsed_json = parse_json_from_response(judge_response_text)
    
    if "error" in parsed_json:
        sample['judge_status'] = 'Failed'
        sample['judge_result'] = parsed_json
    else:
        sample['judge_status'] = 'Success'
        sample['judge_result'] = parsed_json
        
    return sample

# ======================================================================================
# --- 4. Main Execution Logic ---
# ======================================================================================

if __name__ == "__main__":
    # --- File and Directory Setup ---
    if "YOUR_GENERATED_REPLIES_FILE.jsonl" in INPUT_PATH:
        raise ValueError("Please set INPUT_PATH to your real input file path in the configuration section (Part 1) of the code.")

    output_dir = os.path.dirname(INPUT_PATH) # Output directory is the same as input directory
    base_filename = os.path.basename(INPUT_PATH).replace('.jsonl', '')
    
    ## Need to modify to adapt to different tracks
    # Dynamically generate output filename - May need to include track information
    output_filename = f"Score_judgements_on_{base_filename}_by_{JUDGE_MODEL}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    # --- Data Loading ---
    print(f"Reading samples from file: {INPUT_PATH}...")
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data_samples = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Fatal error: Input file not found at path {INPUT_PATH}.")
        exit()

    if not data_samples:
        print("No data samples found in the specified file. Exiting program.")
        exit()
    
    print(f"Successfully loaded {len(data_samples)} samples.")

    # --- Start Evaluation ---
    print("\n" + "="*50)
    print("--- Starting LLM-as-Judge Scoring ---")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Number of Parallel Threads: {NUM_WORKERS}")
    print("="*50)

    final_results = []
    # Use thread pool to execute evaluation tasks concurrently
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_sample = {executor.submit(perform_judgement, sample): sample for sample in data_samples}
        for future in tqdm(as_completed(future_to_sample), total=len(data_samples), desc="Scoring"):
            try:
                result = future.result()
                final_results.append(result)
            except Exception as exc:
                print(f"A task generated an exception: {exc}")
                
    # --- Save Results ---
    final_results.sort(key=lambda x: x['line_idx'])
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in final_results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # --- Results Statistics and Report ---
    successful_judgements = [res for res in final_results if res.get('judge_status') == 'Success']
    failed_judgements = len(final_results) - len(successful_judgements)
    
    print(f"\nEvaluation completed.")
    print(f"✅ Successfully evaluated: {len(successful_judgements)} entries")
    print(f"❌ Failed/Skipped: {failed_judgements} entries")
    print(f"📝 Detailed scoring results saved to: {output_path}")

    if successful_judgements:
        # Extract all valid scores
        scores = []
        for res in successful_judgements:
            try:
                # Try to extract 'final_score' from 'judge_result'
                score_val = res.get('judge_result', {}).get('final_score')
                if score_val is not None and str(score_val).isdigit():
                    scores.append(int(score_val))
            except (ValueError, TypeError):
                # Skip if score format is incorrect
                continue
        
        if scores:
            total_valid_scores = len(scores)
            average_score = sum(scores) / total_valid_scores
            
            # Calculate distribution of each score
            score_counts = {i: 0 for i in range(1, 6)}
            for score in scores:
                if score in score_counts:
                    score_counts[score] += 1
            
            print("\n" + "="*60)
            print("---           Final Score Summary           ---")
            print("="*60)
            print(f"  - Total Valid Scores: {total_valid_scores}")
            print(f"  - Average Score: {average_score:.2f}")
            print("-" * 60)
            print("  --- Score Distribution ---")
            for score, count in sorted(score_counts.items()):
                percentage = (count / total_valid_scores) * 100 if total_valid_scores > 0 else 0
                bar = "█" * int(percentage / 2) # Create a simple text bar chart
                print(f"    Score {score}: {count:>5} entries ({percentage:5.1f}%)  {bar}")
            print("="*60)
        else:
            print("\nNo valid scores extracted from successful evaluations, cannot calculate average and distribution.")
    else:
        print("\nNo successful evaluations, cannot generate summary report.")
