import os
import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from model import RobertaConfig, RobertaForMaskedLM
from prepare_data import prepare_data

# Constants for Custom Tokenizer
# These must match tokenizer/tokenizer.json
MASK_TOKEN_ID = 4
PAD_TOKEN_ID = 0
VOCAB_SIZE = 35000

mx.set_default_device(mx.gpu)

print(f"Running on: {mx.default_device()}")

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npy")
    valid_path = os.path.join(data_dir, "valid.npy")
    
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        print("Data not found, running data preparation...")
        # Create a dummy args object
        class Args:
            data_dir = "data" # Default
            tokenizer = "roberta-base"
        
        args = Args()
        args.data_dir = data_dir
        prepare_data(args)
        
    train_data = np.load(train_path)
    valid_data = np.load(valid_path)
    return train_data, valid_data

def create_mask(input_ids, mask_prob=0.15):
    # MLX random mask generation
    # input_ids: [B, L]
    B, L = input_ids.shape
    
    # Create a probability matrix
    prob_matrix = mx.random.uniform(shape=(B, L))
    
    # Do not mask padding
    # Convert numpy padding check to MLX if needed, or just assume input_list are mx arrays
    # But for efficiency, we might do this on the batch before sending to compiled step
    
    # Mask where prob < mask_prob and token != PAD
    mask = prob_matrix < mask_prob
    pad_mask = input_ids != PAD_TOKEN_ID
    
    # Final mask
    mask = mask & pad_mask
    
    # Create labels: -100 where not masked, original id where masked
    labels = mx.where(mask, input_ids, -100)
    
    # Replace masked input with MASK_TOKEN_ID
    masked_input = mx.where(mask, MASK_TOKEN_ID, input_ids)
    
    return masked_input, labels

def loss_fn(model, X, y):
    logits = model(X)
    # y is [B, L], logits is [B, L, V]
    
    # MLX's cross_entropy can take indices [B, L] and logits [B, L, V]
    # We use reduction='none' to get a loss map [B, L]
    losses = nn.losses.cross_entropy(logits, y, reduction='none')
    
    # Create mask for valid tokens (ignoring -100)
    # y is int32, -100 is int with value -100
    mask = y != -100
    
    # Zero out losses for ignored tokens
    # Note: MLX cross_entropy might return NaN or something for out of bounds (-100), 
    # but usually if we mask the RESULT it works if the op didn't crash.
    # However, -100 is invalid index for cross_entropy lookup usually. 
    # To be safe, we replace -100 in y with 0 (or any valid index) before loss, calculate loss, then mask.
    
    safe_y = mx.where(mask, y, 0)
    losses = nn.losses.cross_entropy(logits, safe_y, reduction='none')
    
    # Apply mask
    losses = losses * mask
    
    # Mean of valid losses
    # sum(losses) / sum(mask)
    num_valid = mask.sum()
    # Avoid division by zero
    loss = losses.sum() / (num_valid + 1e-9)
    
    return loss

def train(args):
    full_config = load_config()
    model_config_dict = full_config.get("model_config", {})
    train_config = full_config.get("training_config", {})

    # Overwrite args with config if present
    epochs = train_config.get("epochs", args.epochs)
    batch_size = train_config.get("batch_size", args.batch_size)
    lr = float(train_config.get("lr", args.lr))
    log_interval = train_config.get("log_interval", args.log_interval)
    save_interval = train_config.get("save_interval", args.save_interval)
    seed = train_config.get("seed", args.seed)

    mx.random.seed(seed)
    np.random.seed(seed)
    
    print("Loading data...")
    train_np, valid_np = load_data(args.data_dir)
    print(f"Train size: {len(train_np)}, Valid size: {len(valid_np)}")
    
    # Config
    if model_config_dict:
        print("Using model config from config.yaml")
        config = RobertaConfig.from_dict(model_config_dict)
    else:
        print("Using default model config")
        config = RobertaConfig(
            vocab_size=VOCAB_SIZE,
            num_hidden_layers=args.num_layers,
            hidden_size=768,
            num_attention_heads=12,
            max_position_embeddings=514
        )
    
    model = RobertaForMaskedLM(config)
    mx.eval(model.parameters())
    
    print(f"Model parameters: {sum(x.size for x in tree_flatten(model.parameters())) / 1e6:.2f}M")
    
    # LR Schedule: Warmup for 2 epochs + Cosine Decay
    num_batches = len(train_np) // batch_size
    total_steps = num_batches * epochs
    warmup_steps = num_batches * 2  # 2 epochs warmup as requested
    
    if total_steps > warmup_steps:
        warmup = optim.linear_schedule(0, lr, steps=warmup_steps)
        decay = optim.cosine_decay(lr, decay_steps=total_steps - warmup_steps)
        lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        print(f"Using Linear Warmup ({warmup_steps} steps) + Cosine Decay")
    else:
        # Fallback if total epochs <= 2, just warmup
        lr_schedule = optim.linear_schedule(0, lr, steps=total_steps)
        print(f"Total steps ({total_steps}) <= Warmup ({warmup_steps}). Using only linear warmup.")

    optimizer = optim.AdamW(learning_rate=lr_schedule)
    
    @mx.compile
    def step(model_params, optimizer_state, X, y):
        def training_loss_fn(params):
             model.update(params)
             return loss_fn(model, X, y)
        
        loss, grads = mx.value_and_grad(training_loss_fn)(model_params)
        
        optimizer.state = optimizer_state
        optimizer.update(model, grads)
        
        return loss, model.parameters(), optimizer.state

    # Training Loop
    # num_batches calculated above
    print(f"Starting training for {epochs} epochs ({num_batches} batches/epoch)")
    
    for epoch in range(epochs):
        tic = time.time()
        
        # Shuffle indices
        indices = np.arange(len(train_np))
        np.random.shuffle(indices)
        
        total_loss = 0
        
        for i in range(num_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch_np = train_np[batch_idx]
            
            # Convert to MLX array (int32 needed for -100 labels)
            batch_mx = mx.array(batch_np, dtype=mx.int32)
            
            # Masking
            input_ids, labels = create_mask(batch_mx)
            
            # Step
            loss, new_params, new_opt_state = step(model.parameters(), optimizer.state, input_ids, labels)
            model.update(new_params)
            optimizer.state = new_opt_state
            mx.eval(loss, new_params, new_opt_state)
            
            total_loss += loss.item()
            
            if i % log_interval == 0 and i > 0:
                print(f"Epoch {epoch+1} | Batch {i}/{num_batches} | Loss: {loss.item():.4f}")
        
        toc = time.time()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} done in {toc - tic:.2f}s | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            model.save_weights(os.path.join(args.output_dir, f"model_epoch_{epoch+1}.safetensors"))

    print("Training complete.")
    model.save_weights(os.path.join(args.output_dir, "model_final.safetensors"))

def tree_flatten(tree):
    if isinstance(tree, (list, tuple)):
        return [leaf for node in tree for leaf in tree_flatten(node)]
    elif isinstance(tree, dict):
        return [leaf for node in tree.values() for leaf in tree_flatten(node)]
    else:
        return [tree] if hasattr(tree, "size") else []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16) # Smaller batch size for laptop training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_layers", type=int, default=6) # DistilRoBERTa sized for speed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
