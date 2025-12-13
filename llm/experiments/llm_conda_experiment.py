"""
LLM Conda Optimizer Experiment

Tests the Conda package optimizers (Conda, SOAP, Muon) on GPT language modeling.
Compares gradient dynamics, training loss, and matrix rank behavior.
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

from llm.configs.training_config import ModelConfig, TrainingConfig
from llm.models.model import create_model

# Import Conda optimizers
from optimizers.conda import Conda, SOAP
from optimizers.conda.muon_moonlight import Muon as CondaMuon


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


def get_param_groups(model, optimizer_name):
    """
    Separate parameters for different optimizers.
    Returns (muon_params, adamw_params) for Conda/SOAP/Muon.
    """
    muon_params = []
    adamw_params = []
    
    for name, p in model.named_parameters():
        if p.ndim == 2 and 'embedding' not in name and 'lm_head' not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    
    return muon_params, adamw_params


def train_with_optimizer(optimizer_name, max_steps=200, track_interval=10, seed=42):
    """Train LLM with specified optimizer and track metrics."""
    
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
    
    # Get parameter groups
    muon_params, adamw_params = get_param_groups(model, optimizer_name)
    
    # Setup optimizer based on type
    if optimizer_name == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        optimizers = [optimizer]
    
    elif optimizer_name == 'conda':
        # Conda optimizer for 2D params with projection
        conda_param_groups = [
            {
                'params': muon_params,
                'update_proj_gap': 10,
                'scale': 1.0,
                'proj_type': 'std',
                'dim': 2,
            }
        ]
        optimizer = Conda(
            conda_param_groups,
            lr=3e-4,
            weight_decay=0.1,
            no_deprecation_warning=True
        )
        adamw_opt = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.1)
        optimizers = [optimizer, adamw_opt]
    
    elif optimizer_name == 'soap':
        # SOAP optimizer - second-order Adam with preconditioning
        optimizer = SOAP(
            muon_params,
            lr=1e-3,
            betas=(0.95, 0.95),
            weight_decay=0.01,
            precondition_frequency=10,
        )
        adamw_opt = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.1)
        optimizers = [optimizer, adamw_opt]
    
    elif optimizer_name == 'conda_muon':
        # Muon from Conda package (moonlight implementation)
        optimizer = CondaMuon(
            lr=0.02,
            wd=0.1,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=5,
        )
        optimizers = [optimizer]
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
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
        
        # Zero gradients
        for opt in optimizers:
            opt.zero_grad()
        
        # Forward
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        
        # Track gradient metrics BEFORE optimizer step
        if step % track_interval == 0:
            grad_ranks = []
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.ndim == 2:
                    grad = param.grad.detach().float()
                    if min(grad.shape) < 4:
                        continue
                    try:
                        eff_rank = effective_rank(grad)
                        grad_ranks.append(eff_rank)
                        
                        short_name = name.replace('layers.', 'L').replace('.weight', '')
                        metrics[f'{short_name}_rank'].append(eff_rank)
                    except:
                        pass
            
            if grad_ranks:
                metrics['avg_rank'].append(sum(grad_ranks) / len(grad_ranks))
        
        metrics['loss'].append(loss.item())
        metrics['step'].append(step)
        
        # Optimizer step with gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
        
        if step % 20 == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")
    
    return metrics


def plot_conda_results(results, save_path='llm_conda_comparison.png'):
    """Plot comparison of all Conda optimizers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Smooth function
    def smooth(data, window=10):
        return [sum(data[max(0,i-window):i+1])/min(i+1, window) for i in range(len(data))]
    
    colors = {
        'adam': '#2196F3',
        'conda': '#4CAF50', 
        'soap': '#FF9800',
        'conda_muon': '#E91E63',
    }
    
    # Plot 1: Loss comparison
    ax = axes[0, 0]
    for name, metrics in results.items():
        ax.plot(smooth(metrics['loss']), label=name.upper(), 
                color=colors.get(name, 'gray'), alpha=0.8, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Training Loss: Conda Optimizers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average gradient rank
    ax = axes[0, 1]
    for name, metrics in results.items():
        if 'avg_rank' in metrics:
            ax.plot(metrics['avg_rank'], label=name.upper(),
                   color=colors.get(name, 'gray'), alpha=0.8, linewidth=2)
    ax.set_xlabel('Tracking Step')
    ax.set_ylabel('Average Effective Rank')
    ax.set_title('Gradient Rank Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: First attention layer rank
    ax = axes[1, 0]
    key_patterns = ['L0.attn.q_proj', 'L0.attn.k_proj', 'L0.attention']
    for name, metrics in results.items():
        for key in metrics.keys():
            if any(pattern in key for pattern in key_patterns) and '_rank' in key:
                ax.plot(metrics[key], label=f"{name.upper()}: {key.replace('_rank', '')}",
                       alpha=0.7)
                break
    ax.set_xlabel('Tracking Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('First Attention Layer Gradient Rank')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final loss comparison (bar chart)
    ax = axes[1, 1]
    final_losses = {}
    for name, metrics in results.items():
        if len(metrics['loss']) >= 20:
            final_losses[name] = sum(metrics['loss'][-20:]) / 20
    
    if final_losses:
        bars = ax.bar(final_losses.keys(), final_losses.values(),
                      color=[colors.get(k, 'gray') for k in final_losses.keys()])
        ax.set_ylabel('Final Loss (avg last 20 steps)')
        ax.set_title('Final Training Loss Comparison')
        for bar, val in zip(bars, final_losses.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('LLM Optimizer Comparison: Conda Package', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved plot to {save_path}")
    plt.close()


def print_summary(results):
    """Print summary of experiment results."""
    print("\n" + "="*60)
    print("SUMMARY: LLM Conda Optimizer Experiment")
    print("="*60)
    
    for name, metrics in results.items():
        if len(metrics['loss']) >= 20:
            final_loss = sum(metrics['loss'][-20:]) / 20
            initial_loss = sum(metrics['loss'][:10]) / min(10, len(metrics['loss']))
            
            avg_rank_start = None
            avg_rank_end = None
            if 'avg_rank' in metrics and len(metrics['avg_rank']) >= 2:
                avg_rank_start = metrics['avg_rank'][0]
                avg_rank_end = metrics['avg_rank'][-1]
            
            print(f"\n{name.upper()}:")
            print(f"  Loss: {initial_loss:.4f} → {final_loss:.4f} (Δ = {final_loss - initial_loss:+.4f})")
            if avg_rank_start and avg_rank_end:
                print(f"  Avg Rank: {avg_rank_start:.1f} → {avg_rank_end:.1f} (Δ = {avg_rank_end - avg_rank_start:+.1f})")
    
    # Determine best optimizer
    final_losses = {name: sum(metrics['loss'][-20:]) / 20 
                   for name, metrics in results.items() if len(metrics['loss']) >= 20}
    if final_losses:
        best = min(final_losses, key=final_losses.get)
        print(f"\n🏆 Best optimizer: {best.upper()} (loss: {final_losses[best]:.4f})")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Conda Optimizer Experiment')
    parser.add_argument('--max_steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--track_interval', type=int, default=10, help='Interval for tracking metrics')
    parser.add_argument('--optimizers', nargs='+', 
                        default=['adam', 'conda', 'soap', 'conda_muon'],
                        help='Which optimizers to test')
    args = parser.parse_args()
    
    print("="*60)
    print("LLM Conda Optimizer Experiment")
    print("="*60)
    
    results = {}
    
    for i, opt_name in enumerate(args.optimizers):
        print(f"\n[{i+1}/{len(args.optimizers)}] Training with {opt_name.upper()}...")
        try:
            metrics = train_with_optimizer(
                opt_name,
                max_steps=args.max_steps,
                track_interval=args.track_interval
            )
            results[opt_name] = metrics
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")
            continue
    
    if results:
        print("\n📊 Generating plots...")
        plot_conda_results(results)
        print_summary(results)
    else:
        print("\n❌ No results to plot - all optimizers failed!")
