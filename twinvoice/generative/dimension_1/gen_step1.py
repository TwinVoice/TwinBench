# -*- coding: utf-8 -*-
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from openai import OpenAI
from tqdm import tqdm

from api_config import api_key

# ======================================================================================
# --- 1. Configuration Section (Please modify your parameters here) ---
# ======================================================================================

# --- API and Model Configuration ---
# Your API Base URL and Key
BASE_URL = "https://api.example.com/"  # Example, please replace with your API address
API_KEY = api_key         # Example, please replace with your API key

# Define the models to use
# LMUT_MODEL: The "attacker" model used to generate imitative replies
#LMUT_MODEL = "qwen2.5-14b-instruct"
#LMUT_MODEL = "gpt-3.5-turbo"
#LMUT_MODEL ="llama3.1-8b"
#LMUT_MODEL ="deepseek-v3"
#LMUT_MODEL ="claude-sonnet-4-20250514"
LMUT_MODEL ="gpt-oss-20b"
#LMUT_MODEL ="gpt-5-chat"
# JUDGE_MODEL: The "judge" model used for reference comparison (recommend using the most capable model)
#JUDGE_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-5-chat"

# --- Data and Execution Configuration ---
##Need to modify to adapt to different tracks
# Input file path
INPUT_PATH = "../data/pchatbot_dataset/merged/pchatbot_pccd_top_2000.jsonl"

# Which portion of the file to process (line numbers start from 1)
# For example, DATA_RANGE_START = 1, DATA_RANGE_END = 1000 will process lines 1 to 1000
DATA_RANGE_START = 1
DATA_RANGE_END = 2000 # Strongly recommend testing with a small range first (e.g., 1-5) to verify the complete workflow

# --- Performance and Robustness Configuration ---
# Number of parallel worker threads (adjust based on your API rate limits and local performance)
NUM_WORKERS = 10 
# Number of retries after API call failure
MAX_RETRIES = 3
# Wait time before each retry (seconds)
RETRY_DELAY = 5

# ======================================================================================
# --- 2. Prompt Templates (Designed according to our final approach) ---
# ======================================================================================

##Need to modify to adapt to different tracks
# Step 1: LMUT (attacker) prompt for generating imitative replies
LMUT_PROMPT_TEMPLATE = """You are acting as a digital twin of a specific social media user.
Your task is to analyze the user's posting history to understand their personality, tone, vocabulary, and style.
All provided text (history, post) is in Chinese. You must analyze and respond in Chinese.

Here is the user's posting history:
(Note: "AnchorPost" is a post by someone else, and "UserReply" is the user's own reply to it.)
---
{history_text}
---

Now, you must imitate this user's persona perfectly and write a new reply to the following post.

Please include your response in the following JSON format:
{{"generated_content": "your reply text here"}}

You may include thinking process or other content, but make sure to include the JSON format with the generated_content field.

Post to reply to:
"{anchor_post}"
"""

##Need to modify to adapt to different tracks
# Step 2: Judge LLM (judge) prompt for reference comparison
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of writing style. Your task is to compare several candidate replies against a known "Reference Reply" written by a specific user.

Your goal is to identify which candidate is the most similar to the reference in terms of **style, tone, vocabulary, sentiment, and topic**.

This is the **Reference Reply** (the ground truth written by the user):
---
{ground_truth_reply}
---

These are the **Candidate Replies**:
{candidate_replies_text}

Now, determine which single candidate is the closest match to the Reference Reply.
You MUST respond ONLY with a JSON object in the following format. Do not include any other text.
The reasoning should be concise, limited to 2-3 sentences.

```json
{{
  "choice": "The letter of the best matching candidate (e.g., 'A', 'B', 'C', or 'D')",
  "reasoning": "A brief explanation for your choice, focusing on the stylistic similarities."
}}
"""

#### **Part 3: Core Function Implementation (API calls and parsing)**

# ======================================================================================
# --- 3. Core Functions ---
# ======================================================================================

# Initialize OpenAI client
# Ensure API key is provided, otherwise throw an error
if API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError("Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key.")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_llm_api_with_retry(model, messages, temperature=0.0, max_tokens=None):
    """
    Call LLM API with integrated retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,  # Pass parameter to API call
                timeout=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Warning] API call failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"API_ERROR: All {MAX_RETRIES} retries failed. Last error: {e}"


def parse_json_from_response(text):
    """
    Robustly extract JSON object from LLM response.
    """
    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract content wrapped in ```json ...```
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass # If regex extraction also fails, continue
        # As a last resort, use more lenient regex to find the first complete JSON object
        match = re.search(r'\{[\s\S]*?\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

def extract_generated_content(text):
    """
    Extract the content of the generated_content field from LLM response.
    """
    # Try to parse the entire text as JSON directly
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict) and "generated_content" in json_obj:
            return json_obj["generated_content"]
    except json.JSONDecodeError:
        pass
    
    # Try to extract content wrapped in ```json ...``` using regex
    match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            json_obj = json.loads(match.group(1))
            if isinstance(json_obj, dict) and "generated_content" in json_obj:
                return json_obj["generated_content"]
        except json.JSONDecodeError:
            pass
    
    # Use more lenient regex to find any JSON object containing generated_content
    json_pattern = r'\{[^{}]*"generated_content"\s*:\s*"([^"]*)"[^{}]*\}'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    
    # Search for more complex JSON structures (handling nested quotes, etc.)
    matches = re.finditer(r'\{[\s\S]*?\}', text)
    for match in matches:
        try:
            json_obj = json.loads(match.group(0))
            if isinstance(json_obj, dict) and "generated_content" in json_obj:
                return json_obj["generated_content"]
        except json.JSONDecodeError:
            continue
    
    return None

def step1_generate_reply(sample):
    """
    (Step 1) Call LMUT model to generate imitative reply for a single sample.
    """
    ##Need to modify to adapt to different tracks
    # Data field access and history text construction logic
    history_text = "\n".join(sample['history'])
    prompt = LMUT_PROMPT_TEMPLATE.format(
        history_text=history_text,
        anchor_post=sample['anchor_post']
    )
    
    messages = [{"role": "user", "content": prompt}]
    lmut_response = call_llm_api_with_retry(LMUT_MODEL, messages, temperature=0.0, max_tokens=8000)
    
    # New: Extract generated_content from response
    if "API_ERROR" in lmut_response:
        sample['lmut_reply'] = lmut_response
        sample['step1_status'] = 'Failed'
    else:
        # Try to extract JSON-formatted generated_content from response
        extracted_content = extract_generated_content(lmut_response)
        if extracted_content:
            sample['lmut_reply'] = extracted_content
            sample['step1_status'] = 'Success'
        else:
            # If extraction fails, record original response and mark as failed
            sample['lmut_reply'] = f"PARSE_ERROR: Could not extract generated_content from: {lmut_response}"
            sample['step1_status'] = 'Failed'
        
    return sample


def step2_judge_similarity(sample):
    """
    (Step 2) Call Judge model for reference comparison on a single sample.
    """
    # Check if previous step succeeded
    if sample.get('step1_status') != 'Success':
        sample['step2_status'] = 'Skipped'
        return sample

    ##Need to modify to adapt to different tracks
    # 1. Prepare data - data field access logic
    ground_truth_reply = sample['choices'][sample['answer_idx']]
    distractor_replies = [choice for i, choice in enumerate(sample['choices']) if i != sample['answer_idx']]
    
    # 2. Create candidate list and shuffle randomly
    #    Each element is a tuple (reply text, identity)
    candidates = [(sample['lmut_reply'], 'LMUT')] + [(d, 'DISTRACTOR') for d in distractor_replies]
    random.shuffle(candidates)
    
    # 3. Record shuffled order and LMUT's new position
    candidate_map = {}
    lmut_new_letter = ''
    candidate_replies_text = []
    for i, (text, identity) in enumerate(candidates):
        letter = chr(ord('A') + i)
        candidate_map[letter] = {'text': text, 'identity': identity}
        candidate_replies_text.append(f"{letter}. {text}")
        if identity == 'LMUT':
            lmut_new_letter = letter

    # 4. Construct judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth_reply=ground_truth_reply,
        candidate_replies_text="\n".join(candidate_replies_text)
    )
    
    # 5. Call API and parse result
    messages = [{"role": "user", "content": prompt}]
    judge_response_text = call_llm_api_with_retry(JUDGE_MODEL, messages, temperature=0.0)
    
    # 6. Record judgment result
    sample['judge_prompt'] = prompt # Save prompt for debugging
    sample['shuffled_candidates'] = candidate_map
    
    if "API_ERROR" in judge_response_text:
        sample['step2_status'] = 'Failed'
        sample['judge_raw_response'] = judge_response_text
        return sample

    parsed_json = parse_json_from_response(judge_response_text)
    
    if parsed_json and 'choice' in parsed_json and 'reasoning' in parsed_json:
        judge_choice_letter = parsed_json['choice'].upper()
        
        sample['step2_status'] = 'Success'
        sample['judge_choice_letter'] = judge_choice_letter
        sample['judge_reasoning'] = parsed_json['reasoning']
        
        # Check if judge's choice is valid
        if judge_choice_letter in candidate_map:
            chosen_identity = candidate_map[judge_choice_letter]['identity']
            sample['judge_choice_identity'] = chosen_identity
            sample['is_lmut_chosen'] = (chosen_identity == 'LMUT')
        else:
            sample['judge_choice_identity'] = 'Invalid Choice'
            sample['is_lmut_chosen'] = False
    else:
        sample['step2_status'] = 'Failed'
        sample['judge_raw_response'] = judge_response_text
        sample['is_lmut_chosen'] = False

    return sample

# ======================================================================================
# --- 4. Main Execution Logic ---
# ======================================================================================

if __name__ == "__main__":
    # --- File and directory setup ---
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    ##Need to modify to adapt to different tracks
    # Dynamically generate filename based on model and data range - may need to include track information
    file_suffix = f"lmut-{LMUT_MODEL}_judge-{JUDGE_MODEL}_range-{DATA_RANGE_START}-{DATA_RANGE_END}.jsonl"
    step1_output_path = os.path.join(output_dir, f"step1_generations_{file_suffix}")
    step2_output_path = os.path.join(output_dir, f"step2_judgements_{file_suffix}")

    # --- Data loading ---
    if DATA_RANGE_START <= 0 or DATA_RANGE_END < DATA_RANGE_START:
        raise ValueError("Invalid DATA_RANGE settings.")
    
    start_zero_based = DATA_RANGE_START - 1
    num_to_read = DATA_RANGE_END - DATA_RANGE_START + 1

    print(f"Reading samples from line {DATA_RANGE_START} to {DATA_RANGE_END} from {INPUT_PATH}...")
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            # islice for efficiently reading specific line range from file
            lines_iterator = islice(f, start_zero_based, DATA_RANGE_END)
            data_samples = [json.loads(line) for line in lines_iterator]
    except FileNotFoundError:
        print(f"FATAL: Input file not found at {INPUT_PATH}")
        exit()

    if not data_samples:
        print("No data samples found in the specified range. Exiting.")
        exit()
    
    print(f"Successfully loaded {len(data_samples)} samples.")

    # --- STEP 1: Generate Imitative Replies (LMUT as Attacker) ---
    print("\n" + "="*50)
    print("--- STEP 1: GENERATING IMITATIVE REPLIES ---")
    print(f"Attacker Model (LMUT): {LMUT_MODEL}")
    print("="*50)

    step1_results = []
    # Execute concurrently using thread pool
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_sample = {executor.submit(step1_generate_reply, sample): sample for sample in data_samples}
        for future in tqdm(as_completed(future_to_sample), total=len(data_samples), desc="Step 1: Generating"):
            step1_results.append(future.result())

    # Save intermediate file
    with open(step1_output_path, 'w', encoding='utf-8') as f_out:
        for item in step1_results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    successful_step1 = [res for res in step1_results if res.get('step1_status') == 'Success']
    print(f"\nStep 1 finished. {len(successful_step1)}/{len(step1_results)} replies generated successfully.")
    print(f"📝 Intermediate results saved to: {step1_output_path}")

    if not successful_step1:
        print("\nNo samples were successfully processed in Step 1. Exiting.")
        exit()

    # --- STEP 2: Perform Reference Comparison Judgment (Judge LLM as Evaluator) ---
    print("\n" + "="*50)
    print("--- STEP 2: JUDGING SIMILARITY (REFERENCE COMPARISON) ---")
    print(f"Judge Model: {JUDGE_MODEL}")
    print("="*50)

    step2_results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_sample = {executor.submit(step2_judge_similarity, sample): sample for sample in successful_step1}
        for future in tqdm(as_completed(future_to_sample), total=len(successful_step1), desc="Step 2: Judging"):
            step2_results.append(future.result())

    # Save final evaluation file
    with open(step2_output_path, 'w', encoding='utf-8') as f_out:
        for item in step2_results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    successful_step2 = [res for res in step2_results if res.get('step2_status') == 'Success']
    print(f"\nStep 2 finished. {len(successful_step2)}/{len(step2_results)} samples judged successfully.")
    print(f"📝 Final evaluation results saved to: {step2_output_path}")

    # --- Final Results Calculation and Report ---
    if successful_step2:
        lmut_chosen_count = sum(1 for res in successful_step2 if res.get('is_lmut_chosen') is True)
        total_valid_judgements = len(successful_step2)
        
        accuracy = (lmut_chosen_count / total_valid_judgements) * 100 if total_valid_judgements > 0 else 0

        print("\n" + "="*60)
        print("--- FINAL EVALUATION REPORT ---")
        print("="*60)
        print(f"Total Samples Judged: {total_valid_judgements}")
        print(f"Times Judge Chose LMUT's Reply: {lmut_chosen_count}")
        print("-" * 60)
        print(f"🎯 Final Accuracy (LMUT Chosen Rate): {accuracy:.2f}%")
        print("="*60)
        print("\nThis accuracy score represents the percentage of times the Judge LLM identified")
        print("the LMUT-generated reply as the most similar to the ground truth,")
        print("when compared against three original distractor replies.")
    else:
        print("\nNo valid judgements were made in Step 2. Cannot calculate final accuracy.")
