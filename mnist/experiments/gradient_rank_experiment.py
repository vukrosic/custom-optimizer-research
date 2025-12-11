"""
Experiment 1: Gradient Rank Dynamics Over Training

Hypothesis: Orthonormal updates preserve the "information capacity" of gradients
better than raw gradients, preventing collapse into low-rank subspaces.

This script trains a small model with both Adam and Muon, tracking the 
effective rank of gradients throughout training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('..')

from muon import Muon, zeropower_via_newtonschulz5


# Simple 2-layer network for clean experiments
class SimpleNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def effective_rank(matrix):
    """
    Compute effective rank using entropy of normalized singular values.
    
    effective_rank = exp(H(σ)) where H is entropy of normalized singular values
    """
    # SVD
    S = torch.linalg.svdvals(matrix.float())
    
    # Normalize to probability distribution
    S = S / (S.sum() + 1e-10)
    
    # Entropy
    entropy = -(S * torch.log(S + 1e-10)).sum()
    
    return torch.exp(entropy).item()


def top_k_ratio(matrix, k=1):
    """Fraction of energy in top-k singular values."""
    S = torch.linalg.svdvals(matrix.float())
    return (S[:k].sum() / S.sum()).item()


def run_experiment(optimizer_name, num_steps=500, seed=42):
    """Run training and collect gradient rank metrics."""
    
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SimpleNet().to(device)
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == 'muon':
        optimizer = Muon(model.parameters(), lr=0.02)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Metrics storage
    metrics = defaultdict(list)
    
    # Synthetic regression task
    X_data = torch.randn(1000, 64, device=device)
    Y_data = torch.randn(1000, 32, device=device)
    
    for step in range(num_steps):
        # Random batch
        idx = torch.randint(0, len(X_data), (64,))
        x, y = X_data[idx], Y_data[idx]
        
        # Forward + backward
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        
        # Collect gradient metrics BEFORE optimizer step
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.ndim == 2:
                grad = param.grad.detach()
                
                # Effective rank of raw gradient
                eff_rank = effective_rank(grad)
                metrics[f'{name}_effective_rank'].append(eff_rank)
                
                # Top-1 singular value ratio
                top1 = top_k_ratio(grad, k=1)
                metrics[f'{name}_top1_ratio'].append(top1)
                
                # If Muon, also track after Newton-Schulz
                if optimizer_name == 'muon':
                    grad_ns = zeropower_via_newtonschulz5(grad.unsqueeze(0)).squeeze(0)
                    eff_rank_ns = effective_rank(grad_ns)
                    metrics[f'{name}_effective_rank_after_ns'].append(eff_rank_ns)
        
        # Track loss
        metrics['loss'].append(loss.item())
        
        # Optimizer step
        optimizer.step()
        
        if step % 100 == 0:
            print(f"[{optimizer_name}] Step {step}: loss = {loss.item():.4f}")
    
    return metrics


def plot_results(adam_metrics, muon_metrics):
    """Plot comparison of gradient rank dynamics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    layers = ['fc1.weight', 'fc2.weight', 'fc3.weight']
    
    # Row 1: Effective rank comparison
    for i, layer in enumerate(layers):
        ax = axes[0, i]
        key = f'{layer}_effective_rank'
        
        ax.plot(adam_metrics[key], label='Adam', alpha=0.7)
        ax.plot(muon_metrics[key], label='Muon (before NS)', alpha=0.7)
        
        if f'{layer}_effective_rank_after_ns' in muon_metrics:
            ax.plot(muon_metrics[f'{layer}_effective_rank_after_ns'], 
                   label='Muon (after NS)', alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Effective Rank')
        ax.set_title(f'{layer}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Top-1 ratio (measure of rank collapse)
    for i, layer in enumerate(layers):
        ax = axes[1, i]
        key = f'{layer}_top1_ratio'
        
        ax.plot(adam_metrics[key], label='Adam', alpha=0.7)
        ax.plot(muon_metrics[key], label='Muon', alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Top-1 SV Ratio')
        ax.set_title(f'{layer} - Rank Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Rank Dynamics: Adam vs Muon', fontsize=14)
    plt.tight_layout()
    plt.savefig('gradient_rank_comparison.png', dpi=150)
    plt.show()
    
    # Also plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(adam_metrics['loss'], label='Adam', alpha=0.7)
    plt.plot(muon_metrics['loss'], label='Muon', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss: Adam vs Muon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_comparison.png', dpi=150)
    plt.show()


def print_summary(adam_metrics, muon_metrics):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY: Gradient Rank Dynamics")
    print("="*60)
    
    layers = ['fc1.weight', 'fc2.weight', 'fc3.weight']
    
    for layer in layers:
        key = f'{layer}_effective_rank'
        
        adam_start = sum(adam_metrics[key][:10]) / 10
        adam_end = sum(adam_metrics[key][-10:]) / 10
        muon_start = sum(muon_metrics[key][:10]) / 10
        muon_end = sum(muon_metrics[key][-10:]) / 10
        
        print(f"\n{layer}:")
        print(f"  Adam:  rank {adam_start:.2f} → {adam_end:.2f} (Δ = {adam_end - adam_start:+.2f})")
        print(f"  Muon:  rank {muon_start:.2f} → {muon_end:.2f} (Δ = {muon_end - muon_start:+.2f})")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("- If Adam's effective rank decreases → rank collapse happening")
    print("- If Muon maintains higher rank → supports hypothesis")
    print("- After NS should always be ~max rank (fully orthonormal)")
    print("="*60)


if __name__ == '__main__':
    print("Running Gradient Rank Dynamics Experiment")
    print("="*50)
    
    print("\n[1/2] Training with Adam...")
    adam_metrics = run_experiment('adam', num_steps=500)
    
    print("\n[2/2] Training with Muon...")
    muon_metrics = run_experiment('muon', num_steps=500)
    
    print("\nGenerating plots...")
    plot_results(adam_metrics, muon_metrics)
    
    print_summary(adam_metrics, muon_metrics)
