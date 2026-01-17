import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

print("Loading model and tokenizer...")
model = AutoModelForMaskedLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

print(f"✓ Model loaded: {model.num_parameters():,} parameters")
print(f"✓ Tokenizer vocab: {tokenizer.vocab_size}")

# Test
prompt = """
Saline, also known as saline solution, is a mixture of sodium chloride in water and has a number of uses in medicine. Applied to the affected area it is used to clean wounds, help remove contact lenses, and help with dry eyes. By injection into a vein it is used to treat dehydration such as from gastroenteritis and diabetic ketoacidosis. It is also used to dilute other medications to be given by injection.
The question "is saline and sodium chloride the same thing" can be answered either 'true' or 'false', and the answer is [MASK].
"""
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