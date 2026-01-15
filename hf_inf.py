import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

print("Loading model and tokenizer...")
model = AutoModelForMaskedLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

print(f"✓ Model loaded: {model.num_parameters():,} parameters")
print(f"✓ Tokenizer vocab: {tokenizer.vocab_size}")

# Test
prompt = "I think this is the beginning of a beautiful [MASK]."
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Get [MASK] position
mask_token_id = tokenizer.mask_token_id
mask_pos = (inputs.input_ids == mask_token_id)[0].nonzero()[0].item()

# Get predictions
mask_logits = outputs.logits[0, mask_pos]
probs = torch.softmax(mask_logits, dim=0)
top_5 = torch.topk(probs, 5)

print("\nTop 5 predictions:")
for i in range(5):
    token_id = top_5.indices[i].item()
    token = tokenizer.decode([token_id])
    prob = top_5.values[i].item() * 100
    print(f"{i+1}. {token} ({prob:.2f}%)")