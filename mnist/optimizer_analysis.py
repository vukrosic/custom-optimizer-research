#!/usr/bin/env python
"""
Comprehensive Optimizer Analysis for MNIST

Analyzes how different optimizers transform weight matrices, tracking:
- Singular value spectrum evolution
- Effective rank (Shannon entropy-based)
- Condition number
- Orthogonality measure
- Spectral norm
- Gradient statistics

Usage:
    python mnist/optimizer_analysis.py
"""

import sys
import os
import json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Output directory
OUTPUT_DIR = "mnist/results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for each optimizer
COLORS = {
    'AdamW': '#2ecc71',
    'SGD': '#3498db', 
    'Muon': '#e74c3c',
    'Oblique': '#9b59b6',
    'L1-Stiefel': '#f39c12'
}


def get_data(batch_size=256):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size)


class MNISTNet(nn.Module):
    """Simple MLP for MNIST."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 256, bias=False)
        self.fc3 = nn.Linear(256, 10, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_weight_matrices(self):
        """Return dict of weight matrices."""
        return {
            'fc1': self.fc1.weight.data,
            'fc2': self.fc2.weight.data,
            'fc3': self.fc3.weight.data
        }


def compute_matrix_metrics(W):
    """Compute comprehensive metrics for a weight matrix."""
    if W.ndim != 2:
        return None
    
    W = W.float()
    
    # SVD
    try:
        S = torch.linalg.svdvals(W)
    except:
        return None
    
    # Effective rank (Shannon entropy)
    S_norm = S / (S.sum() + 1e-10)
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
    eff_rank = torch.exp(entropy).item()
    
    # Condition number
    cond_num = (S[0] / (S[-1] + 1e-10)).item()
    
    # Spectral norm
    spectral_norm = S[0].item()
    
    # Orthogonality: ||W^T W - I||_F / sqrt(n)
    m, n = W.shape
    if m >= n:
        WtW = W.T @ W
        I = torch.eye(n, device=W.device)
        orth_error = torch.norm(WtW - I).item() / np.sqrt(n)
    else:
        WWt = W @ W.T
        I = torch.eye(m, device=W.device)
        orth_error = torch.norm(WWt - I).item() / np.sqrt(m)
    
    # Top singular values
    top_svs = S[:min(10, len(S))].cpu().numpy().tolist()
    
    return {
        'effective_rank': eff_rank,
        'condition_number': min(cond_num, 1e6),  # Cap for plotting
        'spectral_norm': spectral_norm,
        'orthogonality_error': orth_error,
        'frobenius_norm': torch.norm(W).item(),
        'top_singular_values': top_svs
    }


def compute_gradient_metrics(model):
    """Compute gradient metrics for all layers."""
    metrics = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.ndim == 2:
            G = param.grad.float()
            try:
                S = torch.linalg.svdvals(G)
                S_norm = S / (S.sum() + 1e-10)
                entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
                eff_rank = torch.exp(entropy).item()
                
                metrics[name] = {
                    'norm': torch.norm(G).item(),
                    'effective_rank': eff_rank
                }
            except:
                pass
    return metrics


def create_optimizer(model, name, device):
    """Create optimizer by name."""
    params = list(model.parameters())
    matrix_params = [p for p in params if p.ndim >= 2 and p.requires_grad]
    
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=1e-3)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)
    elif name == 'Muon':
        from optimizers.muon import Muon
        return Muon(matrix_params, lr=0.02)
    elif name == 'Oblique':
        from optimizers.oblique import ObliqueOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return ObliqueOptimizer(params, base_opt, radius=1.0)
    elif name == 'L1-Stiefel':
        from optimizers.l1_stiefel import L1StiefelOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return L1StiefelOptimizer(params, base_opt, l1_weight=0.0001, ns_steps=3)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_with_tracking(opt_name, train_loader, test_loader, device, epochs=5, track_interval=50):
    """Train model and track all metrics."""
    print(f"\n{'='*50}")
    print(f"Training with: {opt_name}")
    print(f"{'='*50}")
    
    model = MNISTNet().to(device)
    optimizer = create_optimizer(model, opt_name, device)
    
    metrics = {
        'losses': [],
        'accuracies': [],
        'matrix_metrics': defaultdict(list),  # per layer
        'gradient_metrics': defaultdict(list),
        'steps': []
    }
    
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            
            # Track metrics periodically
            if step % track_interval == 0:
                metrics['steps'].append(step)
                metrics['losses'].append(loss.item())
                
                # Matrix metrics
                for layer_name, W in model.get_weight_matrices().items():
                    m = compute_matrix_metrics(W)
                    if m:
                        metrics['matrix_metrics'][layer_name].append(m)
                
                # Gradient metrics
                grad_m = compute_gradient_metrics(model)
                for layer_name, gm in grad_m.items():
                    metrics['gradient_metrics'][layer_name].append(gm)
            
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        metrics['accuracies'].append(acc)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    metrics['final_loss'] = epoch_loss / len(train_loader)
    metrics['final_acc'] = acc
    
    return metrics


def plot_training_curves(all_metrics):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax = axes[0]
    for opt_name, metrics in all_metrics.items():
        ax.plot(metrics['steps'], metrics['losses'], 
                label=opt_name, color=COLORS[opt_name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Accuracy
    ax = axes[1]
    epochs = list(range(1, len(next(iter(all_metrics.values()))['accuracies']) + 1))
    for opt_name, metrics in all_metrics.items():
        ax.plot(epochs, metrics['accuracies'],
                label=f"{opt_name} ({metrics['final_acc']:.1f}%)", 
                color=COLORS[opt_name], linewidth=2, marker='o', markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training_curves.png")


def plot_matrix_analysis(all_metrics):
    """Plot matrix transformation analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    layer = 'fc2'  # Focus on middle layer
    
    metrics_to_plot = [
        ('effective_rank', 'Effective Rank', axes[0, 0]),
        ('condition_number', 'Condition Number', axes[0, 1]),
        ('orthogonality_error', 'Orthogonality Error', axes[0, 2]),
        ('spectral_norm', 'Spectral Norm', axes[1, 0]),
        ('frobenius_norm', 'Frobenius Norm', axes[1, 1]),
    ]
    
    for metric_key, title, ax in metrics_to_plot:
        for opt_name, metrics in all_metrics.items():
            if layer in metrics['matrix_metrics']:
                values = [m[metric_key] for m in metrics['matrix_metrics'][layer]]
                ax.plot(metrics['steps'][:len(values)], values,
                       label=opt_name, color=COLORS[opt_name], linewidth=2, alpha=0.8)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} (fc2 layer)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if metric_key == 'condition_number':
            ax.set_yscale('log')
    
    # Final comparison bar chart
    ax = axes[1, 2]
    x = np.arange(len(all_metrics))
    width = 0.35
    
    final_ranks = [all_metrics[opt]['matrix_metrics'][layer][-1]['effective_rank'] 
                   for opt in all_metrics]
    final_orth = [all_metrics[opt]['matrix_metrics'][layer][-1]['orthogonality_error'] 
                  for opt in all_metrics]
    
    ax.bar(x - width/2, final_ranks, width, label='Effective Rank', color='steelblue')
    ax.bar(x + width/2, final_orth, width, label='Orth Error', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics.keys(), rotation=45, ha='right')
    ax.set_title('Final Matrix Properties', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/matrix_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved matrix_analysis.png")


def plot_singular_values(all_metrics):
    """Plot singular value spectra."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, layer in enumerate(['fc1', 'fc2', 'fc3']):
        ax = axes[idx]
        
        for opt_name, metrics in all_metrics.items():
            if layer in metrics['matrix_metrics'] and metrics['matrix_metrics'][layer]:
                # Get final singular values
                svs = metrics['matrix_metrics'][layer][-1]['top_singular_values']
                ax.plot(range(1, len(svs)+1), svs, 
                       label=opt_name, color=COLORS[opt_name], 
                       linewidth=2, marker='o', markersize=5)
        
        ax.set_xlabel('Singular Value Index', fontsize=11)
        ax.set_ylabel('Singular Value', fontsize=11)
        ax.set_title(f'{layer} - Final Singular Value Spectrum', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/singular_values.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved singular_values.png")


def plot_gradient_analysis(all_metrics):
    """Plot gradient analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layer = 'fc2.weight'
    
    # Gradient norms
    ax = axes[0]
    for opt_name, metrics in all_metrics.items():
        if layer in metrics['gradient_metrics']:
            norms = [m['norm'] for m in metrics['gradient_metrics'][layer]]
            ax.plot(metrics['steps'][:len(norms)], norms,
                   label=opt_name, color=COLORS[opt_name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm (fc2)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Gradient effective rank
    ax = axes[1]
    for opt_name, metrics in all_metrics.items():
        if layer in metrics['gradient_metrics']:
            ranks = [m['effective_rank'] for m in metrics['gradient_metrics'][layer]]
            ax.plot(metrics['steps'][:len(ranks)], ranks,
                   label=opt_name, color=COLORS[opt_name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Gradient Effective Rank (fc2)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gradient_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved gradient_analysis.png")


def plot_final_comparison(all_metrics):
    """Plot final comparison summary."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    opt_names = list(all_metrics.keys())
    x = np.arange(len(opt_names))
    
    # Accuracy
    ax = axes[0]
    accs = [all_metrics[opt]['final_acc'] for opt in opt_names]
    bars = ax.bar(x, accs, color=[COLORS[opt] for opt in opt_names])
    ax.set_xticks(x)
    ax.set_xticklabels(opt_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Final Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(90, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Final Loss
    ax = axes[1]
    losses = [all_metrics[opt]['final_loss'] for opt in opt_names]
    bars = ax.bar(x, losses, color=[COLORS[opt] for opt in opt_names])
    ax.set_xticks(x)
    ax.set_xticklabels(opt_names, rotation=45, ha='right')
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    
    # Final Effective Rank
    ax = axes[2]
    ranks = [all_metrics[opt]['matrix_metrics']['fc2'][-1]['effective_rank'] for opt in opt_names]
    bars = ax.bar(x, ranks, color=[COLORS[opt] for opt in opt_names])
    ax.set_xticks(x)
    ax.set_xticklabels(opt_names, rotation=45, ha='right')
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Final Weight Effective Rank (fc2)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved final_comparison.png")


def save_metrics(all_metrics):
    """Save all metrics to JSON."""
    # Convert to JSON-serializable format
    output = {}
    for opt_name, metrics in all_metrics.items():
        output[opt_name] = {
            'final_loss': metrics['final_loss'],
            'final_acc': metrics['final_acc'],
            'accuracies': metrics['accuracies'],
            'steps': metrics['steps'],
            'losses': metrics['losses'],
        }
    
    with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved metrics.json")


def main():
    print("="*60)
    print("MNIST Optimizer Analysis")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    train_loader, test_loader = get_data(batch_size=256)
    
    optimizers = ['AdamW', 'SGD', 'Muon', 'Oblique', 'L1-Stiefel']
    epochs = 5
    
    all_metrics = {}
    
    for opt_name in optimizers:
        try:
            metrics = train_with_tracking(opt_name, train_loader, test_loader, 
                                          device, epochs=epochs, track_interval=50)
            all_metrics[opt_name] = metrics
        except Exception as e:
            print(f"ERROR with {opt_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    plot_training_curves(all_metrics)
    plot_matrix_analysis(all_metrics)
    plot_singular_values(all_metrics)
    plot_gradient_analysis(all_metrics)
    plot_final_comparison(all_metrics)
    save_metrics(all_metrics)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}/")
    
    # Print summary
    print("\nFinal Results:")
    for opt_name in sorted(all_metrics.keys(), key=lambda x: -all_metrics[x]['final_acc']):
        m = all_metrics[opt_name]
        print(f"  {opt_name:12s}: Acc={m['final_acc']:.2f}%, Loss={m['final_loss']:.4f}")


if __name__ == '__main__':
    main()
