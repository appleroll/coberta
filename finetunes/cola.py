import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

# Configuration
OUTPUT_DIR = "cola_ft"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5 # CoLA is small (~8.5k), more epochs needed
SEED = 42

class CoLATrainer(Trainer):
    def __init__(self, *args, valid_token_id=None, invalid_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_token_id = valid_token_id
        self.invalid_token_id = invalid_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Standard Cross Entropy
        loss_fct = nn.CrossEntropyLoss() 
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def main():
    if torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Using {device}")

    # Load Model
    path = "../model" if os.path.exists("../model") else "model"
    if not os.path.exists(path): path = "/Users/28zhany/coberta/model"
    
    print(f"Loading {path}...")
    model = AutoModelForMaskedLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    print("Loading GLUE CoLA...")
    dataset = load_dataset("glue", "cola", split="train")

    # Define targets for [MASK]
    # Sentence is "correct" (1) or "incorrect" (0) in grammar
    VALID_TOKEN = "correct"
    INVALID_TOKEN = "incorrect"
    
    valid_id = tokenizer.convert_tokens_to_ids(VALID_TOKEN)
    invalid_id = tokenizer.convert_tokens_to_ids(INVALID_TOKEN)
    mask_token_id = tokenizer.mask_token_id

    print(f"Targets: 1->'{VALID_TOKEN}' ({valid_id}), 0->'{INVALID_TOKEN}' ({invalid_id})")

    def preprocess(examples):
        sentences = examples['sentence']
        labels = examples['label'] # 0 or 1
        
        inputs = []
        target_labels = []

        for s, l in zip(sentences, labels):
            prompt = f"The sentence: \"{s}\" is [MASK]."
            
            enc = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
            
            input_ids = enc.input_ids[0]
            label_ids = torch.full_like(input_ids, -100)
            
            mask_idxs = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_idxs) > 0:
                mask_pos = mask_idxs[0]
                # Label Logic: 1 -> Valid, 0 -> Invalid
                label_ids[mask_pos] = valid_id if l == 1 else invalid_id
                
                inputs.append(input_ids)
                target_labels.append(label_ids)
        
        return {
            "input_ids": inputs,
            "labels": target_labels,
            "attention_mask": [torch.ones_like(ids) for ids in inputs]
        }

    encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    # Split
    split = encoded_dataset.train_test_split(test_size=0.1, seed=SEED)
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        seed=SEED,
        fp16=False
    )

    trainer = CoLATrainer(
        model=model,
        args=args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        valid_token_id=valid_id,
        invalid_token_id=invalid_id
    )

    print("Training...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
