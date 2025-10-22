import json
import re
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from api_config import api_key

# API and model configuration
#API_KEY = "your_api_key_here" 
BASE_URL = "https://api.example.com/"  # Example, please replace with your API address
#MODEL = "gpt-3.5-turbo"
#MODEL_NAME = "gpt-3.5-turbo"
#MODEL = "qwen2.5-14b-instruct"
#MODEL_NAME = "qwen2.5-14b-instruct"
#MODEL = "gpt-4o"
#MODEL_NAME = "gpt-4o"
#MODEL = "llama3.1-8b"
#MODEL_NAME = "llama3.1-8b"
#MODEL = "gpt-4o-mini"
#MODEL_NAME = "gpt-4o-mini"
#MODEL = "gemini-2.0-flash"
#MODEL_NAME = "gemini-2.0-flash"
#MODEL = "deepseek-v3"
#MODEL_NAME = "deepseek-v3"
#MODEL = "gpt-5-chat"
#MODEL_NAME = "gpt-5-chat"
#MODEL = "gemini-2.5-pro"
#MODEL_NAME = "gemini-2.5-pro"
#MODEL = "claude-sonnet-4-20250514"
#MODEL_NAME = "claude-sonnet-4-20250514"
#MODEL = "gpt-oss-20b"
#MODEL_NAME = "gpt-oss-20b"
#MODEL = "gemini-1.5-flash-latest"
#MODEL_NAME = "gemini-1.5-flash-latest"
MODEL = "llama3.1-70b"
MODEL_NAME = "llama3.1-70b"

# Set the data range to process. Note: Index starts from 1 for readability.
# For example, [1, 100] means processing from the 1st to the 100th data entry.
DATA_RANGE_START = 1001 # NEW: Starting line number of the data range (inclusive)
DATA_RANGE_END = 1200  # NEW: Ending line number of the data range (inclusive)

# [Requirement 2] Set the number of parallel worker threads
NUM_WORKERS = 10

# Data and execution configuration
DATA_SPLIT = "merged"
#INPUT_PATH = f"../data/pchatbot_dataset/{DATA_SPLIT}/pchatbot_benchmark.filtered_by_choice_length_20.jsonl"
INPUT_PATH = f"../data/pchatbot_dataset/{DATA_SPLIT}/pchatbot_pccd_top_2000.jsonl"
# MODIFIED: Updated output filename to include data range information
OUTPUT_PATH = f"pchatbot_pccd1_twin_{MODEL_NAME}_{DATA_SPLIT}_samples={DATA_RANGE_START}-{DATA_RANGE_END}_20.jsonl" # MODIFIED

# Original Prompt template (third-party analyst perspective)
prompt_template_analyst = """You are given a user's comment history and 4 candidate tweets. 
Your task is to identify which candidate is most likely written by the same user, based on writing style, tone, and themes.

User's Comment History:
{history}

Candidate Comments:
A. {a}
B. {b}
C. {c}
D. {d}

You must respond in the following format:

```json
{{
  "predicted_comment": "A",
  "reasoning": "Explain why this comment was chosen."
}}
```"""

# New Prompt template (digital twin perspective)
prompt_template_digital_twin = """Your task is to act as a specific social media user, becoming their digital twin.
Note: All provided text (history, post, choices) is in Chinese. You must analyze the user's style directly within the Chinese language context.

Based on the user's reply history, think and respond with their mindset, tone, and style.

Your reply history:
(Note: "AnchorPost" is another user's post, and "UserReply" is your own reply.)
{history}

Now, you see a new post:
"{anchor}"

Below are 4 candidate replies. Which one is most likely something you would say?

A. {a}
B. {b}
C. {c}
D. {d}

Please respond in the following JSON format. In the "reasoning" field, use the first-person perspective ("I") to explain your choice.

```json
{{
  "predicted_comment": "A",
  "reasoning": "Explain, from my perspective as the user, why I would choose this option."
}}
```"""

# Switch here to decide which template to use for this run
prompt_template = prompt_template_digital_twin
#prompt_template = prompt_template_analyst

client = OpenAI(
     base_url=BASE_URL, 
     api_key=api_key,
)

def process_sample(sample):
    """
    Function to process a single data sample: send request to LLM and parse the result.
    This function will be called in parallel.
    """
    anchor_post = sample["anchor_post"]
    choices = sample["choices"]
    label = sample["answer_idx"]
    history = sample["history"]

    # Fill in the prompt
    filled_prompt = prompt_template.format(
        anchor=anchor_post.strip(),
        history="\n".join(history[-30:]),
        a=choices[0],
        b=choices[1],
        c=choices[2],
        d=choices[3]
    )

    predicted_index = -1
    reasoning = ""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": filled_prompt}],
        )
        response_text = response.choices[0].message.content.strip()
        
        # Robustly extract JSON part from response text
        match = re.search(r'\{[\s\S]*?\}', response_text)
        if match:
            json_part = match.group(0)
            parsed = json.loads(json_part)
            predicted_letter = parsed.get("predicted_comment", "").strip().upper()
            reasoning = parsed.get("reasoning", "").strip()
            if predicted_letter in ["A", "B", "C", "D"]:
                predicted_index = ["A", "B", "C", "D"].index(predicted_letter)
        else:
            # If JSON not found, log the error
            reasoning = f"LLM did not return a valid JSON object. Response: {response_text}"

    except Exception as e:
        # Log API call or other exceptions
        reasoning = f"LLM error: {str(e)}"
        # Retry logic can be added here, but for simplicity, we only log the error
        time.sleep(1) # Wait briefly on error to avoid frequent request failures

    # Return a dictionary containing all information for subsequent processing
    return {
        "line_idx": sample.get("line_idx", -1), # Use .get() to avoid issues with old data lacking line_idx key
        "user_id": sample["user_id"],
        "anchor_post": anchor_post,
        "choices": choices,
        "ground_truth_index": label,
        "predicted_index": predicted_index,
        "correct": int(predicted_index == label),
        "reasoning": reasoning
    }

if __name__ == "__main__":
    # --- MODIFIED: Modified file reading logic to support data range ---
    # Validate input parameters
    if DATA_RANGE_START <= 0 or DATA_RANGE_END < DATA_RANGE_START:
        raise ValueError("Invalid DATA_RANGE settings. START must be > 0 and END must be >= START.")
    
    # Calculate the number of lines to skip and the number of lines to read
    start_zero_based = DATA_RANGE_START - 1 # islice uses 0-based indexing
    num_to_read = DATA_RANGE_END - DATA_RANGE_START + 1

    print(f"Reading samples from line {DATA_RANGE_START} to {DATA_RANGE_END} from {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        # Use islice to efficiently skip to the starting position and read specified number of lines
        lines_iterator = islice(f, start_zero_based, DATA_RANGE_END)
        lines = list(lines_iterator)
        
        # Check if enough lines were read
        if len(lines) < num_to_read:
            print(f"Warning: Requested range up to {DATA_RANGE_END}, but file only contains {start_zero_based + len(lines)} lines.")
        
        data_samples = []
        for i, line in enumerate(lines):
            sample = json.loads(line)
            # Add original line number to each sample for traceability
            sample['line_idx'] = start_zero_based + i + 1 
            data_samples.append(sample)
    # --- END MODIFIED ---
    
    if not data_samples:
        print("No data samples found in the specified range. Exiting.")
    else:
        print(f"Successfully loaded {len(data_samples)} samples. Starting parallel processing with {NUM_WORKERS} workers...")
        
        analysis_results = []
        # Create thread pool and execute tasks
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results_iterator = executor.map(process_sample, data_samples)
            analysis_results = list(tqdm(results_iterator, total=len(data_samples), desc="Running LLM History Match"))

        # Separate labels and predictions from results
        labels = [res["ground_truth_index"] for res in analysis_results]
        preds = [res["predicted_index"] for res in analysis_results]
        
        # Filter out failed predictions (-1)
        valid_indices = [i for i, p in enumerate(preds) if p != -1]
        valid_labels = [labels[i] for i in valid_indices]
        valid_preds = [preds[i] for i in valid_indices]
        
        # Save output
        print(f"Saving output to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
            for item in analysis_results:
                # Use sort_keys=False to maintain original order, indent=None and separators to save space
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Accuracy statistics
        if valid_labels:
            acc = accuracy_score(valid_labels, valid_preds)
            print(f"\n📊 Prompt LLM Match Accuracy: {acc:.4f} on {len(valid_labels)} valid samples (out of {len(data_samples)} total)")
        else:
            print("\nNo valid predictions were made.")
            
        print(f"📝 Output saved to: {OUTPUT_PATH}")
