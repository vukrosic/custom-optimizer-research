"""
MNIST Experiment: AdamW vs Muon with Newton-Schulz Iteration Analysis

This experiment:
1. Compares AdamW vs Muon optimizers on MNIST
2. For Muon, compares the effect of 3 vs 5 Newton-Schulz iterations
3. Tracks average effective rank (Shannon entropy-based) and loss

Key metrics:
- Average effective rank across all 2D weight layers
- Training/validation loss
- Accuracy
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')

from muon import Muon


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.compile
def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This is a copy of the original function but with configurable steps.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.half()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def effective_rank(matrix):
    """Compute effective rank via entropy of normalized singular values (Shannon entropy)."""
    S = torch.linalg.svdvals(matrix.float())
    S = S / (S.sum() + 1e-10)  # Normalize to probability distribution
    entropy = -(S * torch.log(S + 1e-10)).sum()  # Shannon entropy
    return torch.exp(entropy).item()  # Effective rank


class MNISTNet(nn.Module):
    """Simple MLP for MNIST with 2D weight matrices for Muon optimization."""
    
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = 784  # 28x28
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        
        layers.append(nn.Linear(prev_size, 10))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


def get_mnist_loaders(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def compute_avg_rank(model, after_ns_steps=None):
    """Compute average effective rank across all 2D gradient matrices.
    
    Args:
        model: The model with gradients computed
        after_ns_steps: If not None, compute rank after applying NS with this many steps
        
    Returns:
        Average effective rank across all 2D weight gradients
    """
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


def train_epoch(model, optimizer, train_loader, device, muon_optimizer=None, track_metrics=True, ns_steps=5):
    """Train for one epoch and track metrics."""
    model.train()
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    avg_ranks = []
    avg_ranks_after_ns = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        if muon_optimizer:
            muon_optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Track metrics before optimizer step
        if track_metrics and batch_idx % 20 == 0:
            # Average rank before NS
            avg_rank = compute_avg_rank(model, after_ns_steps=None)
            avg_ranks.append(avg_rank)
            
            # Average rank after NS (with specified steps)
            if muon_optimizer:
                avg_rank_ns = compute_avg_rank(model, after_ns_steps=ns_steps)
                avg_ranks_after_ns.append(avg_rank_ns)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if muon_optimizer:
            muon_optimizer.step()
        
        epoch_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return {
        'loss': epoch_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        'avg_rank': np.mean(avg_ranks) if avg_ranks else 0.0,
        'avg_rank_after_ns': np.mean(avg_ranks_after_ns) if avg_ranks_after_ns else 0.0
    }


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {'loss': test_loss, 'accuracy': accuracy}


def train_model(config, train_loader, test_loader, device):
    """Train a model with given configuration."""
    
    set_seed(42)
    
    model = MNISTNet(hidden_sizes=[512, 256, 128]).to(device)
    
    optimizer_name = config['optimizer']
    ns_steps = config.get('ns_steps', 5)
    
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name.upper()}", end="")
    if optimizer_name == 'muon':
        print(f" (NS steps: {ns_steps})")
    else:
        print()
    print(f"{'='*60}")
    
    # Setup optimizers
    muon_optimizer = None
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    elif optimizer_name == 'muon':
        # Muon for 2D params, AdamW for rest (biases, etc.)
        muon_params = []
        adam_params = []
        
        for name, p in model.named_parameters():
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        optimizer = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.01)
        muon_optimizer = Muon(muon_params, lr=0.02, momentum=0.95, ns_steps=ns_steps)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'avg_rank': [],
        'avg_rank_after_ns': []
    }
    
    n_epochs = config['epochs']
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(
            model, optimizer, train_loader, device,
            muon_optimizer=muon_optimizer,
            track_metrics=True,
            ns_steps=ns_steps
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['avg_rank'].append(train_metrics['avg_rank'])
        history['avg_rank_after_ns'].append(train_metrics['avg_rank_after_ns'])
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Test Acc: {test_metrics['accuracy']:.2f}% | "
              f"Avg Rank: {train_metrics['avg_rank']:.1f}", end="")
        
        if train_metrics['avg_rank_after_ns'] > 0:
            print(f" | Rank after NS: {train_metrics['avg_rank_after_ns']:.1f}")
        else:
            print()
    
    return history


def plot_comparison(results, save_prefix='mnist_comparison'):
    """Plot comparison of all configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'adamw': '#1f77b4',
        'muon_ns3': '#ff7f0e',
        'muon_ns5': '#2ca02c'
    }
    
    labels = {
        'adamw': 'AdamW',
        'muon_ns3': 'Muon (3 NS iters)',
        'muon_ns5': 'Muon (5 NS iters)'
    }
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=labels[name], color=colors[name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=labels[name], color=colors[name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average Rank (Before NS) - Compare Muon 3 vs 5
    ax = axes[1, 0]
    for name, history in results.items():
        if 'muon' in name:
            ax.plot(history['avg_rank'], label=f"{labels[name]} (before NS)", 
                   color=colors[name], linewidth=2, linestyle='-')
            ax.plot(history['avg_rank_after_ns'], label=f"{labels[name]} (after NS)", 
                   color=colors[name], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Effective Rank (Shannon)')
    ax.set_title('Muon: Avg Rank Comparison (3 vs 5 NS iterations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined view - Rank vs Loss relationship
    ax = axes[1, 1]
    for name, history in results.items():
        if 'muon' in name:
            rank_data = history['avg_rank_after_ns'] if sum(history['avg_rank_after_ns']) > 0 else history['avg_rank']
            ax.scatter(rank_data, history['train_loss'], 
                      label=labels[name], color=colors[name], alpha=0.7, s=50)
    ax.set_xlabel('Average Effective Rank (after NS)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Rank vs Loss Relationship (Muon variants)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('MNIST Experiment: AdamW vs Muon (3 vs 5 NS iterations)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_prefix}.png")
    plt.close()
    
    # Additional plot: Muon NS comparison only
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rank comparison
    ax = axes[0]
    muon_3_ranks = results['muon_ns3']['avg_rank_after_ns']
    muon_5_ranks = results['muon_ns5']['avg_rank_after_ns']
    epochs = range(1, len(muon_3_ranks) + 1)
    
    ax.plot(epochs, muon_3_ranks, label='3 NS iterations', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, muon_5_ranks, label='5 NS iterations', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    ax.fill_between(epochs, muon_3_ranks, muon_5_ranks, alpha=0.2, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Effective Rank (Shannon)')
    ax.set_title('Muon: Avg Rank After Newton-Schulz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss comparison
    ax = axes[1]
    muon_3_loss = results['muon_ns3']['train_loss']
    muon_5_loss = results['muon_ns5']['train_loss']
    
    ax.plot(epochs, muon_3_loss, label='3 NS iterations', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, muon_5_loss, label='5 NS iterations', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    ax.fill_between(epochs, muon_3_loss, muon_5_loss, alpha=0.2, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Muon: Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Muon: Effect of Newton-Schulz Iterations (3 vs 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_ns_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved NS comparison plot to {save_prefix}_ns_comparison.png")
    plt.close()


def print_summary(results):
    """Print summary of results."""
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY: MNIST - AdamW vs Muon (3 vs 5 NS iterations)")
    print("="*70)
    
    print("\n📊 Final Results (Last Epoch):")
    print("-"*70)
    print(f"{'Config':<20} | {'Train Loss':>12} | {'Test Acc':>10} | {'Avg Rank':>12} | {'Rank (NS)':>12}")
    print("-"*70)
    
    for name, history in results.items():
        train_loss = history['train_loss'][-1]
        test_acc = history['test_acc'][-1]
        avg_rank = history['avg_rank'][-1]
        avg_rank_ns = history['avg_rank_after_ns'][-1] if history['avg_rank_after_ns'][-1] > 0 else "N/A"
        
        label = {
            'adamw': 'AdamW',
            'muon_ns3': 'Muon (3 NS)',
            'muon_ns5': 'Muon (5 NS)'
        }[name]
        
        rank_ns_str = f"{avg_rank_ns:.1f}" if isinstance(avg_rank_ns, float) else avg_rank_ns
        print(f"{label:<20} | {train_loss:>12.4f} | {test_acc:>9.2f}% | {avg_rank:>12.1f} | {rank_ns_str:>12}")
    
    print("-"*70)
    
    # Compare Muon variants
    print("\n🔬 Muon NS Comparison:")
    muon_3 = results['muon_ns3']
    muon_5 = results['muon_ns5']
    
    rank_diff = np.mean(muon_5['avg_rank_after_ns']) - np.mean(muon_3['avg_rank_after_ns'])
    loss_diff = np.mean(muon_3['train_loss']) - np.mean(muon_5['train_loss'])
    
    print(f"  • Average Rank (5 NS) - Average Rank (3 NS): {rank_diff:+.2f}")
    print(f"  • Loss improvement (3 NS → 5 NS): {loss_diff:+.4f}")
    
    # Best configuration
    print("\n🏆 Best Configuration:")
    best_acc = max(results.items(), key=lambda x: x[1]['test_acc'][-1])
    config_labels = {'adamw': 'AdamW', 'muon_ns3': 'Muon (3 NS)', 'muon_ns5': 'Muon (5 NS)'}
    print(f"  • Highest Test Accuracy: {config_labels[best_acc[0]]} ({best_acc[1]['test_acc'][-1]:.2f}%)")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='MNIST: AdamW vs Muon with NS iteration analysis')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📦 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"✓ Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Configurations to test
    configs = [
        {'name': 'adamw', 'optimizer': 'adamw', 'epochs': args.epochs},
        {'name': 'muon_ns3', 'optimizer': 'muon', 'ns_steps': 3, 'epochs': args.epochs},
        {'name': 'muon_ns5', 'optimizer': 'muon', 'ns_steps': 5, 'epochs': args.epochs},
    ]
    
    results = {}
    
    for config in configs:
        history = train_model(config, train_loader, test_loader, device)
        results[config['name']] = history
    
    # Plotting
    print("\n📊 Generating plots...")
    plot_comparison(results, save_prefix='mnist_comparison')
    
    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
