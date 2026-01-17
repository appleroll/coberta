import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

# Configuration
OUTPUT_DIR = "boolq"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 6
SEED = 42

# Focal Loss Parameters
GAMMA = 1 # Standard value for Focal Loss (focuses on hard examples)

class FocalLossTrainer(Trainer):
    def __init__(self, *args, true_token_id=None, false_token_id=None, alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        # Alpha is the class balance weight tensor
        self.alpha = alpha
        print(f"Initialized FocalLossTrainer.")
        print(f"gamma={GAMMA}")
        if alpha is not None:
             print(f"Class Weights (alpha): {alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Calculate standard Cross Entropy (unreduced)
        # We use ignore_index=-100 to mask out padding/non-masked tokens
        ce_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        ce_loss = ce_loss_fct(logits, labels)
        
        # Get probabilities of these specific tokens
        pt = torch.exp(-ce_loss) # prob of correct class
        
        # Focal Loss Formula: (1 - pt)^gamma * log(pt)
        # Note: ce_loss is -log(pt)
        focal_term = (1 - pt) ** GAMMA
        focal_loss = focal_term * ce_loss
        
        # Apply Class Balancing Weights (Alpha)
        if self.alpha is not None:
            # Create a weight map for the active batch
            # Default weight = 1.0
            batch_weights = torch.ones_like(labels, dtype=torch.float)
            
            if self.false_token_id is not None:
                # Apply weight for FALSE class
                # alpha[0] corresponds to False usually if we ordered [False, True]
                # Let's assume alpha is passed as scalar for False class relative to True
                # self.alpha is the weight for the FALSE class
                batch_weights = torch.where(labels == self.false_token_id, batch_weights * self.alpha, batch_weights)
            
            focal_loss = focal_loss * batch_weights

        # Average over valid tokens only (non -100)
        active_mask = labels != -100
        if active_mask.sum() > 0:
            final_loss = focal_loss[active_mask].mean()
        else:
            final_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return (final_loss, outputs) if return_outputs else final_loss

def main():
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Silicon) acceleration")
    else:
        print("⚠ MPS not available, using CPU")

    print("Loading model...")
    # Determine correct path whether running from root or finetunes/ folder
    if os.path.exists("model"):
        model_path = "model"
    elif os.path.exists("../model"):
        model_path = "../model"
    else:
        model_path = "/Users/28zhany/coberta/model"

    print(f"Loading from: {model_path}")
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3. Load Dataset
    print("Loading BoolQ training set...")
    dataset = load_dataset("google/boolq", split="train")
    
    # 4. Define Prompt Format
    def get_token_id(word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1: return ids[0]
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1: return ids[0]
        return None

    TRUE_TOKEN = "true" 
    FALSE_TOKEN = "false"
    
    true_id = get_token_id(TRUE_TOKEN)
    false_id = get_token_id(FALSE_TOKEN)
    
    if true_id is None or false_id is None or true_id == false_id:
        print("Lowercase targets failed/ambiguous. Trying capitalized...")
        TRUE_TOKEN = "True"
        FALSE_TOKEN = "False"
        true_id = get_token_id(TRUE_TOKEN)
        false_id = get_token_id(FALSE_TOKEN)

    if true_id == false_id:
         raise ValueError(f"True ({true_id}) and False ({false_id}) map to the same token!")

    print(f"Training Targets: True='{TRUE_TOKEN}' ({true_id}), False='{FALSE_TOKEN}' ({false_id})")

    def preprocess_function(examples):
        passages = examples['passage']
        questions = examples['question']
        answers = examples['answer']
        
        inputs = []
        labels = []
        
        for p, q, a in zip(passages, questions, answers):
            prompt_suffix = f'\nThe question "{q}" can be answered either \'true\' or \'false\', and the answer is [MASK].\n'
            full_text = f"{p}{prompt_suffix}"
            
            encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding.input_ids[0]
            label_ids = torch.full_like(input_ids, -100)
            mask_token_id = tokenizer.mask_token_id
            
            mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0]
                target_id = true_id if a else false_id
                label_ids[mask_idx] = target_id
                inputs.append(input_ids)
                labels.append(label_ids)
        
        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": [torch.ones_like(ids) for ids in inputs]
        }

    print("Preprocessing data...")
    # Map first to get stats if we want precise calculation, 
    # but map is expensive. Let's do a quick pass or use known stats.
    # BoolQ Training Set Stats:
    # Total: 9427
    # True: 5874 (~62.3%)
    # False: 3553 (~37.7%)
    
    # Calculate Precise Class Weights
    n_total = 9427
    n_true = 5874
    n_false = 3553
    
    # Weight = Total / (n_classes * count)
    # But relative to True=1.0:
    # W_True_Raw = 9427 / (2 * 5874) = 0.80
    # W_False_Raw = 9427 / (2 * 3553) = 1.32
    # Ratio W_False / W_True = 1.32 / 0.80 = 1.65
    # however, this noobus "thinks that since the dataset is unbalanced,
    # just swing it towards true dawg"
    
    CALCULATED_WEIGHT_FALSE = 1.4
    print(f"Calculated Imbalance Ratio: 1.0 (True) : {CALCULATED_WEIGHT_FALSE:.2f} (False)")
    
    processed_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=1000, 
        remove_columns=dataset.column_names,
        desc="Formatting Prompts"
    )
    
    train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # 5. Training Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=50,
        load_best_model_at_end=True,
        seed=SEED,
        fp16=False, 
    )

    # USE FOCAL LOSS TRAINER
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        true_token_id=true_id,
        false_token_id=false_id,
        alpha=CALCULATED_WEIGHT_FALSE # Pass the calculated ratio
    )

    print("\nStarting Training with Focal Loss...")
    trainer.train()

    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
