import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

def get_single_token_id(tokenizer, word):
    """
    Get the token ID for a single word. 
    Checks both formatted with and without leading space.
    Returns None if the word splits into multiple tokens.
    """
    # Try with leading space (common for sentence piece / word piece in middle of sent)
    # The mask is preceded by a space in "... result is [MASK]."
    
    # Check directly
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    
    # Check with space
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
        
    return None

def benchmark():
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("\nLoading model and tokenizer...")
    try:
        model = AutoModelForMaskedLM.from_pretrained("../finetunes/boolq")
        tokenizer = AutoTokenizer.from_pretrained("../finetunes/boolq")
    except Exception as e:
        print(f"Error loading model from 'model' directory: {e}")
        # Fallback to current directory if script is run from inside
        if os.path.exists("../model"):
             model = AutoModelForMaskedLM.from_pretrained("model")
             tokenizer = AutoTokenizer.from_pretrained("model")
        else:
             raise e

    model.to(device)
    model.eval()

    print(f"✓ Model loaded: {model.num_parameters():,} parameters")

    true_words = ["true", "True", "correct", "Correct", "valid", "Valid", "yes", "Yes"]
    false_words = ["false", "False", "incorrect", "Incorrect", "invalid", "Invalid", "no", "No"]
    said_true = 0 # number of times model said True
    total = 0    # total examples

    true_ids = []
    false_ids = []

    print("\nMapping target words to token IDs:")
    for w in true_words:
        tid = get_single_token_id(tokenizer, w)
        if tid is not None:
            if tid not in true_ids:
                true_ids.append(tid)
                print(f"  '{w}' -> {tid} ({tokenizer.decode([tid])})")
    
    for w in false_words:
        tid = get_single_token_id(tokenizer, w)
        if tid is not None:
             if tid not in false_ids:
                false_ids.append(tid)
                print(f"  '{w}' -> {tid} ({tokenizer.decode([tid])})")
    
    if not true_ids or not false_ids:
        print("Warning: Could not find single token IDs for some true/false words.")
        
    true_ids_tensor = torch.tensor(true_ids, device=device)
    false_ids_tensor = torch.tensor(false_ids, device=device)

    # Load BoolQ
    print("\nLoading BoolQ dataset (validation split)...")
    try:
        dataset = load_dataset("google/boolq", split="validation")
    except Exception as e:
        print(f"Failed to load google/boolq: {e}")
        dataset = load_dataset("super_glue", "boolq", split="validation")

    print(f"Loaded {len(dataset)} examples.")

    correct_count = 0
    total_count = 0
    skipped_count = 0
    
    # Statistics Tracking
    said_true = 0
    total_conf_diff = 0.0 # Sum of abs(prob_true - prob_false)
    total_conf_true = 0.0 # Sum of prob_true
    total_conf_false = 0.0 # Sum of prob_false

    print("\nStarting benchmark...")
    pbar = tqdm(dataset)
    
    for example in pbar:
        passage = example['passage']
        question = example['question']
        ground_truth = example['answer'] # boolean

        # Construct prompt matching inf.py style
        # "The question "is saline..." can be answered either 'true' or 'false', and the answer is [MASK]."
        
        prompt_suffix = f'\nThe question "{question}" can be answered either \'true\' or \'false\', and the answer is [MASK].\n'
        
        # Simple prompt construction
        prompt = f"{passage}{prompt_suffix}"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Check if mask token is present
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id not in inputs.input_ids:
            # If rejected, try to shorten passage
            suffix_len = len(prompt_suffix)
            allowed_passage_len = (512 * 3) - suffix_len 
            
            if len(passage) > allowed_passage_len:
                truncated_passage = passage[:allowed_passage_len]
                prompt = f"{truncated_passage}{prompt_suffix}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if mask_token_id not in inputs.input_ids:
                # Still failed, skip
                skipped_count += 1
                continue

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Find mask position
        mask_pos = (inputs['input_ids'][0] == mask_token_id).nonzero()[0].item()
        
        # Get logits
        logits = outputs.logits[0, mask_pos] # [Vocab]
        probs = torch.softmax(logits, dim=0)
        
        # Calculate scores
        # Sum probabilities for synonyms of True and synonyms of False
        p_true = probs[true_ids_tensor].sum().item()
        p_false = probs[false_ids_tensor].sum().item()
        
        # Track confidence
        diff = abs(p_true - p_false)
        total_conf_diff += diff
        total_conf_true += p_true
        total_conf_false += p_false
        
        prediction = p_true > p_false

        if prediction:
            said_true += 1
        
        if prediction == ground_truth:
            correct_count += 1
        
        total_count += 1
        
        current_acc = correct_count / total_count
        pbar.set_description(f"Acc: {current_acc:.2%} ({correct_count}/{total_count})")

    final_acc = correct_count / total_count if total_count > 0 else 0
    avg_conf_diff = total_conf_diff / total_count if total_count > 0 else 0
    avg_prob_true = total_conf_true / total_count if total_count > 0 else 0
    avg_prob_false = total_conf_false / total_count if total_count > 0 else 0

    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Total Examples: {len(dataset)}")
    print(f"Processed:      {total_count}")
    print(f"Skipped:        {skipped_count}")
    print(f"Correct:        {correct_count}")
    print(f"Accuracy:       {final_acc:.2%}")
    print("-" * 30)
    print(f"Model said 'True':  {said_true} times ({(said_true/total_count*100) if total_count>0 else 0:.2f}%)")
    print("-" * 30)
    print("Confidence Stats (Avg):")
    print(f"  Avg Prob(True):   {avg_prob_true:.4f}")
    print(f"  Avg Prob(False):  {avg_prob_false:.4f}")
    print(f"  Avg Margin (Diff): {avg_conf_diff:.4f}")
    print("="*30)

if __name__ == "__main__":
    benchmark()
