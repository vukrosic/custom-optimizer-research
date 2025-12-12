"""
LLM Final Comparison: Effective Rank Analysis with Optimal Learning Rates

Compares effective rank evolution across three optimizers:
1. AdamW (lr=3e-4) - Standard optimizer baseline
2. Muon (3 NS, lr=0.02) - Original Muon baseline
3. Muon (5 NS, lr=0.010) - Optimal LR for higher NS iterations

This experiment validates that with proper LR scaling, 5 NS iterations
can outperform 3 NS iterations while maintaining better gradient rank properties.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimizers.muon import Muon
from llm.common import create_dataloaders, effective_rank, zeropower_via_newtonschulz
from llm.configs.training_config import ModelConfig
from llm.models.model import create_model


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_avg_rank(model, after_ns_steps=None):
    """Compute average effective rank across all 2D gradient matrices."""
    ranks = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.ndim == 2:
            grad = param.grad.detach().float()
            
            if min(grad.shape) < 4:
                continue
                
            try:
                if after_ns_steps is not None:
                    # Apply Newton-Schulz and measure rank after
                    grad_ns = zeropower_via_newtonschulz(grad.unsqueeze(0), steps=after_ns_steps).squeeze(0)
                    rank = effective_rank(grad_ns)
                else:
                    rank = effective_rank(grad)
                ranks.append(rank)
            except:
                pass
    
    return np.mean(ranks) if ranks else 0.0


def train_step(model, optimizer, batch, device, muon_optimizer=None, track_metrics=True, ns_steps=5, step=0):
    """Single training step with metric tracking."""
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    if muon_optimizer:
        muon_optimizer.zero_grad()
    
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    # Track metrics before optimizer step
    metrics = {}
    if track_metrics and step % 10 == 0:
        # Average rank before NS
        avg_rank = compute_avg_rank(model, after_ns_steps=None)
        metrics['avg_rank'] = avg_rank
        
        # Average rank after NS (with specified steps)
        if muon_optimizer:
            avg_rank_ns = compute_avg_rank(model, after_ns_steps=ns_steps)
            metrics['avg_rank_after_ns'] = avg_rank_ns
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    if muon_optimizer:
        muon_optimizer.step()
    
    metrics['loss'] = loss.item()
    return metrics


def evaluate(model, val_loader, device, max_batches=10):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            num_batches += 1
    
    return {'loss': total_loss / num_batches if num_batches > 0 else 0}


def train_model(config, train_loader, val_loader, device):
    """Train a model with given configuration."""
    
    set_seed(42)
    
    model_config = ModelConfig()
    model = create_model(model_config, device)
    
    optimizer_name = config['optimizer']
    ns_steps = config.get('ns_steps', 5)
    muon_lr = config.get('muon_lr', 0.02)
    
    print(f"\n{'='*70}")
    print(f"Training: {config['name']}")
    if optimizer_name == 'muon':
        print(f"  NS steps: {ns_steps}, Muon LR: {muon_lr:.4f}")
    print(f"{'='*70}")
    
    # Setup optimizers
    muon_optimizer = None
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    elif optimizer_name == 'muon':
        # Muon for 2D params, AdamW for rest
        muon_params = []
        adam_params = []
        
        for name, p in model.named_parameters():
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        optimizer = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)
        muon_optimizer = Muon(muon_params, lr=muon_lr, momentum=0.95, ns_steps=ns_steps)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'avg_rank': [],
        'avg_rank_after_ns': []
    }
    
    max_steps = config['max_steps']
    eval_interval = 20
    
    step = 0
    epoch = 0
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Train
            train_metrics = train_step(
                model, optimizer, batch, device,
                muon_optimizer=muon_optimizer,
                track_metrics=True,
                ns_steps=ns_steps,
                step=step
            )
            
            # Store metrics
            if 'avg_rank' in train_metrics:
                history['avg_rank'].append(train_metrics['avg_rank'])
                history['avg_rank_after_ns'].append(train_metrics.get('avg_rank_after_ns', 0.0))
            
            # Evaluate periodically
            if step % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device)
                history['train_loss'].append(train_metrics['loss'])
                history['val_loss'].append(val_metrics['loss'])
                
                print(f"Step {step:4d}/{max_steps} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Avg Rank: {train_metrics.get('avg_rank', 0):.1f}", end="")
                
                if train_metrics.get('avg_rank_after_ns', 0) > 0:
                    print(f" | Rank after NS: {train_metrics['avg_rank_after_ns']:.1f}")
                else:
                    print()
            
            step += 1
        
        epoch += 1
    
    return history


def plot_comparison(results, save_prefix='llm_final_comparison'):
    """Plot comprehensive comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {
        'adamw': '#1f77b4',
        'muon_ns3': '#ff7f0e',
        'muon_ns5_optimal': '#2ca02c'
    }
    
    labels = {
        'adamw': 'AdamW',
        'muon_ns3': 'Muon (3 NS, lr=0.020)',
        'muon_ns5_optimal': 'Muon (5 NS, lr=0.010)'
    }
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=labels[name], color=colors[name], linewidth=2.5)
    ax.set_xlabel('Checkpoint', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['val_loss'], label=labels[name], color=colors[name], linewidth=2.5, marker='o', markersize=5)
    ax.set_xlabel('Checkpoint', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Rank (Before NS for all)
    ax = axes[1, 0]
    for name, history in results.items():
        if len(history['avg_rank']) > 0 and history['avg_rank'][0] > 0:
            ax.plot(history['avg_rank'], label=labels[name], color=colors[name], linewidth=2.5)
    ax.set_xlabel('Checkpoint', fontsize=11)
    ax.set_ylabel('Average Effective Rank', fontsize=11)
    ax.set_title('Gradient Rank (Before NS)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Rank After NS (Muon only)
    ax = axes[1, 1]
    for name, history in results.items():
        if 'muon' in name and len(history['avg_rank_after_ns']) > 0 and sum(history['avg_rank_after_ns']) > 0:
            ax.plot(history['avg_rank_after_ns'], label=labels[name], color=colors[name], linewidth=2.5, marker='s', markersize=5)
    ax.set_xlabel('Checkpoint', fontsize=11)
    ax.set_ylabel('Average Effective Rank', fontsize=11)
    ax.set_title('Gradient Rank (After NS)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LLM Final Comparison: AdamW vs Muon with Optimal LR Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'llm/figures/{save_prefix}.png'
    os.makedirs('llm/figures', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_path}")
    plt.close()


def print_summary(results):
    """Print comprehensive summary."""
    
    print("\n" + "="*90)
    print("FINAL COMPARISON: Effective Rank Analysis with Optimal Learning Rates (LLM)")
    print("="*90)
    
    print("\n📊 Final Results:")
    print("-"*90)
    print(f"{'Configuration':<30} | {'Val Loss':>10} | {'Train Loss':>12} | {'Rank (Before)':>14} | {'Rank (After NS)':>15}")
    print("-"*90)
    
    for name, history in results.items():
        val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else 0
        train_loss = history['train_loss'][-1] if len(history['train_loss']) > 0 else 0
        avg_rank = history['avg_rank'][-1] if len(history['avg_rank']) > 0 else 0
        avg_rank_ns = history['avg_rank_after_ns'][-1] if len(history['avg_rank_after_ns']) > 0 and history['avg_rank_after_ns'][-1] > 0 else None
        
        label = {
            'adamw': 'AdamW',
            'muon_ns3': 'Muon (3 NS, lr=0.020)',
            'muon_ns5_optimal': 'Muon (5 NS, lr=0.010)'
        }[name]
        
        rank_ns_str = f"{avg_rank_ns:.1f}" if avg_rank_ns else "N/A"
        rank_str = f"{avg_rank:.1f}" if avg_rank > 0 else "N/A"
        
        print(f"{label:<30} | {val_loss:>10.4f} | {train_loss:>12.4f} | {rank_str:>14} | {rank_ns_str:>15}")
    
    print("-"*90)
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description='LLM Final Comparison with Optimal LRs')
    parser.add_argument('--max_steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📦 Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        seq_len=args.seq_len,
        num_samples=5000,
        batch_size=args.batch_size
    )
    
    # Configurations to test
    configs = [
        {'name': 'adamw', 'optimizer': 'adamw', 'max_steps': args.max_steps},
        {'name': 'muon_ns3', 'optimizer': 'muon', 'ns_steps': 3, 'muon_lr': 0.02, 'max_steps': args.max_steps},
        {'name': 'muon_ns5_optimal', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.010, 'max_steps': args.max_steps},
    ]
    
    results = {}
    
    for config in configs:
        history = train_model(config, train_loader, val_loader, device)
        results[config['name']] = history
    
    # Plotting
    print("\n📊 Generating plots...")
    plot_comparison(results, save_prefix='llm_final_comparison')
    
    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
