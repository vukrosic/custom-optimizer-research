"""
LLM Gradient Rank Experiment

Trains the GPT model with both Adam and Muon, tracking gradient rank dynamics.
This tests the hypothesis that orthonormalization preserves gradient information.
"""

import argparse
import math
import time
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from config import ModelConfig, TrainingConfig
from model import create_model
from muon import Muon, zeropower_via_newtonschulz5


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_rank(matrix):
    """Compute effective rank via entropy of normalized singular values."""
    S = torch.linalg.svdvals(matrix.float())
    S = S / (S.sum() + 1e-10)
    entropy = -(S * torch.log(S + 1e-10)).sum()
    return torch.exp(entropy).item()


def top_k_ratio(matrix, k=1):
    """Fraction of energy in top-k singular values."""
    S = torch.linalg.svdvals(matrix.float())
    return (S[:k].sum() / S.sum()).item()


class SimpleDataset(torch.utils.data.Dataset):
    """Quick dataset for testing."""
    
    def __init__(self, seq_len=256, num_samples=5000):
        print(f"📦 Loading SmolLM dataset ({num_samples} samples)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            streaming=True,
        )
        
        all_tokens = []
        for example in dataset:
            if len(all_tokens) >= num_samples * seq_len:
                break
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=True)
            all_tokens.extend(tokens)
        
        total_tokens = (len(all_tokens) // seq_len) * seq_len
        all_tokens = all_tokens[:total_tokens]
        self.data = torch.tensor(all_tokens).view(-1, seq_len)
        
        print(f"✓ Loaded {len(self.data)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {"input_ids": tokens, "labels": tokens.clone()}


def train_with_tracking(optimizer_name, max_steps=200, track_interval=10, seed=42):
    """Train LLM and track gradient rank metrics."""
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training with {optimizer_name.upper()} on {device}")
    
    # Small config for quick experiment
    model_config = ModelConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=256,
    )
    
    # Data
    dataset = SimpleDataset(seq_len=256, num_samples=5000)
    model_config.vocab_size = dataset.tokenizer.vocab_size
    
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Model
    model = create_model(model_config).to(device)
    
    # Optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    elif optimizer_name == 'muon':
        # Muon for 2D params, Adam for rest
        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if p.ndim == 2 and 'embedding' not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        optimizer = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)
        muon_optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
    
    # Metrics storage
    metrics = defaultdict(list)
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    
    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        if optimizer_name == 'adam':
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            muon_optimizer.zero_grad()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        
        # Track gradient metrics BEFORE optimizer step
        if step % track_interval == 0:
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.ndim == 2:
                    grad = param.grad.detach().float()
                    
                    # Skip if too small
                    if min(grad.shape) < 4:
                        continue
                    
                    try:
                        eff_rank = effective_rank(grad)
                        top1 = top_k_ratio(grad, k=1)
                        
                        # Simplified name
                        short_name = name.replace('layers.', 'L').replace('.weight', '')
                        metrics[f'{short_name}_rank'].append(eff_rank)
                        metrics[f'{short_name}_top1'].append(top1)
                        
                        # For Muon, also track after NS
                        if optimizer_name == 'muon' and 'embedding' not in name:
                            grad_ns = zeropower_via_newtonschulz5(grad.unsqueeze(0)).squeeze(0)
                            eff_rank_ns = effective_rank(grad_ns)
                            metrics[f'{short_name}_rank_ns'].append(eff_rank_ns)
                    except:
                        pass  # Skip problematic gradients
        
        metrics['loss'].append(loss.item())
        metrics['step'].append(step)
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if optimizer_name == 'muon':
            muon_optimizer.step()
        
        if step % 20 == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")
    
    return metrics


def plot_llm_results(adam_metrics, muon_metrics, save_path='llm_gradient_rank.png'):
    """Plot LLM gradient rank comparison."""
    
    # Find common layer keys
    adam_rank_keys = [k for k in adam_metrics.keys() if '_rank' in k and '_ns' not in k]
    muon_rank_keys = [k for k in muon_metrics.keys() if '_rank' in k and '_ns' not in k]
    common_keys = sorted(set(adam_rank_keys) & set(muon_rank_keys))[:6]  # Limit to 6 layers
    
    if not common_keys:
        print("No common layers found!")
        return
    
    n_layers = len(common_keys)
    fig, axes = plt.subplots(2, min(3, n_layers), figsize=(15, 8))
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    for i, key in enumerate(common_keys[:6]):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        if key in adam_metrics and key in muon_metrics:
            ax.plot(adam_metrics[key], label='Adam', alpha=0.7)
            ax.plot(muon_metrics[key], label='Muon (before NS)', alpha=0.7)
            
            # After NS for Muon
            ns_key = key.replace('_rank', '_rank_ns')
            if ns_key in muon_metrics:
                ax.plot(muon_metrics[ns_key], label='Muon (after NS)', 
                       alpha=0.7, linestyle='--')
        
        layer_name = key.replace('_rank', '')
        ax.set_title(layer_name)
        ax.set_xlabel('Tracking Step')
        ax.set_ylabel('Effective Rank')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('LLM Gradient Rank Dynamics: Adam vs Muon', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved plot to {save_path}")
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    
    # Smooth the loss for plotting
    def smooth(data, window=10):
        return [sum(data[max(0,i-window):i+1])/min(i+1, window) for i in range(len(data))]
    
    plt.plot(smooth(adam_metrics['loss']), label='Adam', alpha=0.8)
    plt.plot(smooth(muon_metrics['loss']), label='Muon', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss (smoothed)')
    plt.title('LLM Training Loss: Adam vs Muon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('llm_loss_comparison.png', dpi=150)
    print(f"✓ Saved loss plot to llm_loss_comparison.png")
    plt.close()


def print_llm_summary(adam_metrics, muon_metrics):
    """Print summary of gradient rank changes."""
    print("\n" + "="*60)
    print("SUMMARY: LLM Gradient Rank Dynamics")
    print("="*60)
    
    # Find rank keys
    rank_keys = [k for k in adam_metrics.keys() if '_rank' in k and '_ns' not in k]
    
    for key in sorted(rank_keys)[:5]:  # Top 5 layers
        if key in adam_metrics and key in muon_metrics:
            adam_vals = adam_metrics[key]
            muon_vals = muon_metrics[key]
            
            if len(adam_vals) > 2 and len(muon_vals) > 2:
                adam_start = sum(adam_vals[:2]) / 2
                adam_end = sum(adam_vals[-2:]) / 2
                muon_start = sum(muon_vals[:2]) / 2
                muon_end = sum(muon_vals[-2:]) / 2
                
                layer = key.replace('_rank', '')
                print(f"\n{layer}:")
                print(f"  Adam:  {adam_start:.1f} → {adam_end:.1f} (Δ = {adam_end-adam_start:+.1f})")
                print(f"  Muon:  {muon_start:.1f} → {muon_end:.1f} (Δ = {muon_end-muon_start:+.1f})")
    
    # Final loss comparison
    adam_final = sum(adam_metrics['loss'][-20:]) / 20
    muon_final = sum(muon_metrics['loss'][-20:]) / 20
    
    print("\n" + "-"*60)
    print(f"Final Loss (avg last 20 steps):")
    print(f"  Adam: {adam_final:.4f}")
    print(f"  Muon: {muon_final:.4f}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--track_interval', type=int, default=10)
    args = parser.parse_args()
    
    print("="*60)
    print("LLM Gradient Rank Dynamics Experiment")
    print("="*60)
    
    print("\n[1/2] Training with Adam...")
    adam_metrics = train_with_tracking('adam', max_steps=args.max_steps, 
                                        track_interval=args.track_interval)
    
    print("\n[2/2] Training with Muon...")
    muon_metrics = train_with_tracking('muon', max_steps=args.max_steps,
                                        track_interval=args.track_interval)
    
    print("\n📊 Generating plots...")
    plot_llm_results(adam_metrics, muon_metrics)
    
    print_llm_summary(adam_metrics, muon_metrics)
