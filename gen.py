import mlx.core as mx
import yaml
from transformers import PreTrainedTokenizerFast
from model import RobertaConfig, RobertaForMaskedLM

# Load model
with open("config.yaml", "r") as f:
    config = RobertaConfig.from_dict(yaml.safe_load(f)["model_config"])

model = RobertaForMaskedLM(config)
model.load_weights("latest_model/model_epoch_8.npz")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "tokenizer",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

def fill_mask(prompt, top_k=5):
    """Simple mask filling with top predictions."""
    # Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = mx.array([input_ids])
    
    # Find mask
    mask_pos = [i for i, t in enumerate(input_ids) if t == tokenizer.mask_token_id]
    if not mask_pos:
        return f"No mask in: {prompt}"
    
    # Forward pass
    logits = model(input_tensor)[0]  # [seq_len, vocab_size]
    
    results = []
    for pos in mask_pos:
        mask_logits = logits[pos]
        probs = mx.softmax(mask_logits)
        top_indices = mx.argsort(mask_logits)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            token = tokenizer.decode([idx.item()])
            prob = probs[idx].item() * 100
            predictions.append((token.strip(), prob))
        
        results.append(predictions)
    
    return results

# Test it
prompt = "Effective communication requires [MASK] and listening."
predictions = fill_mask(prompt, top_k=3)
print(f"Prompt: {prompt}")
for i, (token, prob) in enumerate(predictions[0]):
    print(f"  {i+1}. '{token}' ({prob:.1f}%)")
