"""
Simple training script for Standard Attention LLM

Usage:
    python train.py
    python train.py --max_steps 500 --batch_size 8
"""

import argparse
import math
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from model import create_model
from data.loader import quick_dataset
from utils.helpers import set_seed


def create_dataloaders(config: TrainingConfig):
    """Create train and validation dataloaders"""
    print(f"\n📦 Loading dataset...")
    
    dataset, tokenizer = quick_dataset(
        preset="cosmopedia",
        seq_length=config.max_seq_len,
        num_samples=config.num_samples,
    )
    
    # Split into train/val (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"✓ Vocab size: {tokenizer.vocab_size}")
    
    return train_loader, val_loader, tokenizer


def train(model_config: ModelConfig, train_config: TrainingConfig):
    """Main training loop"""
    
    # Setup
    set_seed(train_config.seed)
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on {device}")
    
    # Data
    train_loader, val_loader, tokenizer = create_dataloaders(train_config)
    model_config.vocab_size = tokenizer.vocab_size
    
    # Model
    model = create_model(model_config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas,
    )
    
    # LR scheduler with warmup + cosine decay
    def lr_lambda(step):
        if step < train_config.warmup_steps:
            return step / train_config.warmup_steps
        progress = (step - train_config.warmup_steps) / (train_config.max_steps - train_config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Checkpoints dir
    ckpt_dir = Path(train_config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"\n🏋️ Starting training for {train_config.max_steps} steps...\n")
    
    model.train()
    step = 0
    train_iter = iter(train_loader)
    running_loss = 0.0
    start_time = time.time()
    
    while step < train_config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        step += 1
        
        # Logging
        if step % train_config.log_interval == 0:
            avg_loss = running_loss / train_config.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (train_config.log_interval * train_config.batch_size * train_config.max_seq_len) / elapsed
            lr = scheduler.get_last_lr()[0]
            
            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            
            running_loss = 0.0
            start_time = time.time()
        
        # Evaluation
        if step % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, train_config.eval_batches)
            print(f"         | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
            model.train()
        
        # Checkpointing
        if step % train_config.save_interval == 0:
            ckpt_path = ckpt_dir / f"model_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
            }, ckpt_path)
            print(f"         | Saved checkpoint: {ckpt_path}")
    
    # Final save
    final_path = ckpt_dir / "model_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": model_config,
    }, final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")
    
    return model


def evaluate(model, val_loader, device, max_batches=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            if max_batches and num_batches >= max_batches:
                break
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Standard Attention LLM")
    
    # Model args
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=12)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=50000)
    
    args = parser.parse_args()
    
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_samples,
    )
    
    train(model_config, train_config)


if __name__ == "__main__":
    main()
