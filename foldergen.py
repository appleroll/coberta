from transformers import RobertaConfig
import json, subprocess

config = RobertaConfig(
    vocab_size=35000,
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=512,
    type_vocab_size=1,
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=1,
    tie_word_embeddings=False
)

config.save_pretrained("model")
print("[INFO] Saved config.")

# copy all tokenizer/* files to model/
subprocess.run(["cp", "-r", "tokenizer/.", "model/"])
print("[INFO] Moved tokenizer files.")

# copy latest checkpoint in checkpoints/ to model/
name = input("[Q] What is the latest epoch? (\"final\" for final model): ").strip()

if name == "final":
    src = "checkpoints/model_final.safetensors"
else:
    src = f"checkpoints/model_epoch_{name}.safetensors"
subprocess.run(["cp", src, "model/model.safetensors"])
print(f"[INFO] Moved {src} as model.safetensors")
print("[INFO] You are ready to rock and roll!")