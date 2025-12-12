"""
Optimizer Comparison Experiment

Systematically compare all optimizers on LLM:
- AdamW (baseline)
- Muon (spectral normalized updates)
- Oblique (unit-norm columns)
- Grassmannian (subspace optimization)
- L1-Stiefel (sparse orthogonal)
- Block-Stiefel (for MHA-like structures)

Tracks: Loss, accuracy, effective rank, convergence speed, wall-clock time.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm.common.data import set_seed, get_llm_loaders, get_device
from llm.common.models import LLMNet
from llm.common.metrics import effective_rank

from optimizers.muon import Muon
from optimizers.oblique import ObliqueOptimizer
from optimizers.grassmannian import GrassmannianOptimizer
from optimizers.l1_stiefel import L1StiefelOptimizer


def create_optimizer(model, optimizer_name, lr=1e-3):
    """Create optimizer by name."""
    
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01), None
    
    elif optimizer_name == 'muon':
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        base_opt = torch.optim.AdamW(adam_params, lr=lr, weight_decay=0.01) if adam_params else None
        muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
        return base_opt, muon_opt
    
    elif optimizer_name == 'oblique':
        base_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        return ObliqueOptimizer(model.parameters(), base_opt), None
    
    elif optimizer_name == 'grassmannian':
        base_opt = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)
        return GrassmannianOptimizer(model.parameters(), base_opt, nuclear_weight=0.001), None
    
    elif optimizer_name == 'l1_stiefel':
        base_opt = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)
        return L1StiefelOptimizer(model.parameters(), base_opt, l1_weight=0.001), None
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def compute_avg_rank(model):
    """Compute average effective rank of gradients."""
    ranks = []
    for p in model.parameters():
        if p.grad is not None and p.ndim == 2 and min(p.shape) >= 4:
            try:
                r = effective_rank(p.grad.detach())
                if r > 0:
                    ranks.append(r)
            except:
                pass
    return np.mean(ranks) if ranks else 0.0


def train_with_optimizer(optimizer_name, train_loader, test_loader, device, n_epochs=10):
    """Train model with given optimizer and track metrics."""
    
    set_seed(42)
    model = LLMNet(hidden_sizes=[512, 256, 128]).to(device)
    
    optimizer, aux_optimizer = create_optimizer(model, optimizer_name)
    
    history = {
        'train_loss': [],
        'test_acc': [],
        'avg_rank': [],
        'epoch_time': [],
    }
    
    print(f"\n{'='*50}")
    print(f"Training with: {optimizer_name.upper()}")
    print(f"{'='*50}")
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0.0
        epoch_ranks = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            if optimizer:
                optimizer.zero_grad()
            if aux_optimizer:
                aux_optimizer.zero_grad()
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Track rank every 20 batches
            if batch_idx % 20 == 0:
                epoch_ranks.append(compute_avg_rank(model))
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if optimizer:
                optimizer.step()
            if aux_optimizer:
                aux_optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_acc = 100. * correct / len(test_loader.dataset)
        avg_loss = epoch_loss / len(train_loader)
        avg_rank = np.mean(epoch_ranks) if epoch_ranks else 0.0
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['avg_rank'].append(avg_rank)
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Rank: {avg_rank:.1f} | "
              f"Time: {epoch_time:.1f}s")
    
    return history


def plot_comparison(results, save_path='optimizer_comparison.png'):
    """Plot comparison of all optimizers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for (name, history), color in zip(results.items(), colors):
        ax.plot(history['train_loss'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for (name, history), color in zip(results.items(), colors):
        ax.plot(history['test_acc'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effective Rank
    ax = axes[1, 0]
    for (name, history), color in zip(results.items(), colors):
        ax.plot(history['avg_rank'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Effective Rank')
    ax.set_title('Gradient Rank Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics bar chart
    ax = axes[1, 1]
    names = list(results.keys())
    final_accs = [results[n]['test_acc'][-1] for n in names]
    x = np.arange(len(names))
    
    bars = ax.bar(x, final_accs, color=colors[:len(names)])
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Final Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Optimizer Comparison on LLM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_path}")
    plt.close()


def print_summary(results):
    """Print summary table."""
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY: Optimizer Comparison")
    print("="*70)
    
    print(f"\n{'Optimizer':<15} | {'Final Loss':>12} | {'Final Acc':>10} | {'Avg Rank':>10} | {'Time/Epoch':>12}")
    print("-"*70)
    
    for name, history in results.items():
        loss = history['train_loss'][-1]
        acc = history['test_acc'][-1]
        rank = np.mean(history['avg_rank'])
        time_per_epoch = np.mean(history['epoch_time'])
        
        print(f"{name:<15} | {loss:>12.4f} | {acc:>9.2f}% | {rank:>10.1f} | {time_per_epoch:>10.2f}s")
    
    print("-"*70)
    
    # Best optimizer
    best = max(results.items(), key=lambda x: x[1]['test_acc'][-1])
    print(f"\n🏆 Best Accuracy: {best[0]} ({best[1]['test_acc'][-1]:.2f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Optimizer Comparison Experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--optimizers', type=str, nargs='+',
                       default=['adamw', 'muon', 'oblique', 'grassmannian'],
                       help='Optimizers to compare')
    args = parser.parse_args()
    
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    train_loader, test_loader = get_llm_loaders(batch_size=args.batch_size)
    print(f"📦 Loaded LLM: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    
    results = {}
    for opt_name in args.optimizers:
        try:
            history = train_with_optimizer(
                opt_name, train_loader, test_loader, device, n_epochs=args.epochs
            )
            results[opt_name] = history
        except Exception as e:
            print(f"⚠️ Error with {opt_name}: {e}")
    
    if results:
        plot_comparison(results, save_path='../figures/optimizer_comparison.png')
        print_summary(results)


if __name__ == '__main__':
    main()
