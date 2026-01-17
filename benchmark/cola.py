import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
import os

def get_single_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1: return ids[0]
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1: return ids[0]
    return None

def benchmark():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load Model (Robust Path)
    model_paths = ["../model", "model", "/Users/28zhany/coberta/model"]
    model = None
    for p in model_paths:
        if os.path.exists(p):
            try:
                print(f"Loading '{p}'...")
                model = AutoModelForMaskedLM.from_pretrained(p)
                tokenizer = AutoTokenizer.from_pretrained(p)
                break
            except: continue
    
    if model is None: raise OSError("Model not found.")
    model.to(device)
    model.eval()

    # Define Verbalizers for CoLA
    # 1 -> Acceptable
    # 0 -> Unacceptable
    
    # We stick to simple ones first
    valid_words = ["correct", "yes", "valid", "acceptable"]
    invalid_words = ["incorrect", "no", "invalid", "unacceptable"]

    valid_ids = []
    invalid_ids = []

    print("Mapping targets...")
    for w in valid_words:
        tid = get_single_token_id(tokenizer, w)
        if tid is not None: valid_ids.append(tid)
    
    for w in invalid_words:
        tid = get_single_token_id(tokenizer, w)
        if tid is not None: invalid_ids.append(tid)

    valid_ids_tensor = torch.tensor(valid_ids, device=device)
    invalid_ids_tensor = torch.tensor(invalid_ids, device=device)

    print("Loading GLUE CoLA...")
    dataset = load_dataset("glue", "cola", split="validation")
    
    refs = []
    preds = []
    
    print("Benchmarking...")
    pbar = tqdm(dataset)
    for example in pbar:
        sentence = example['sentence']
        label = example['label'] # 0 or 1
        
        # Prompt: "The sentence: '{sentence}' is [MASK]."
        # Target: correct / incorrect
        prompt = f"The sentence: \"{sentence}\" is [MASK]."
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        mask_token_id = tokenizer.mask_token_id
        
        if mask_token_id not in inputs.input_ids: 
            continue
            
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        mask_pos = (inputs['input_ids'][0] == mask_token_id).nonzero()[0].item()
        logits = outputs.logits[0, mask_pos]
        probs = torch.softmax(logits, dim=0)
        
        p_valid = probs[valid_ids_tensor].sum().item()
        p_invalid = probs[invalid_ids_tensor].sum().item()
        
        # Predict 1 if valid > invalid
        prediction = 1 if p_valid > p_invalid else 0
        
        preds.append(prediction)
        refs.append(label)
        
    mcc = matthews_corrcoef(refs, preds)
    acc = sum([1 for p, r in zip(preds, refs) if p == r]) / len(refs)

    print("\n" + "="*30)
    print("RESULTS (CoLA)")
    print("="*30)
    print(f"Accuracy: {acc:.2%}")
    print(f"MCC:      {mcc:.4f}")
    print("="*30)
    print("(Note: MCC is the main metric for CoLA. Random = 0, Perfect = 1)")

if __name__ == "__main__":
    benchmark()
