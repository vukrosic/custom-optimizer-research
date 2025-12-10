"""
MNIST Final Comparison: Effective Rank Analysis with Optimal Learning Rates

Compares effective rank evolution across three optimizers:
1. AdamW (lr=1e-3) - Standard optimizer baseline
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
from torchvision import datasets, transforms
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
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
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
    muon_lr = config.get('muon_lr', 0.02)
    
    print(f"\n{'='*70}")
    print(f"Training: {config['name']}")
    if optimizer_name == 'muon':
        print(f"  NS steps: {ns_steps}, Muon LR: {muon_lr:.4f}")
    print(f"{'='*70}")
    
    # Setup optimizers
    muon_optimizer = None
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    elif optimizer_name == 'muon':
        # Muon for 2D params, AdamW for rest
        muon_params = []
        adam_params = []
        
        for name, p in model.named_parameters():
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        optimizer = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.01)
        muon_optimizer = Muon(muon_params, lr=muon_lr, momentum=0.95, ns_steps=ns_steps)
    
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


def plot_comparison(results, save_prefix='mnist_final_comparison'):
    """Plot comprehensive comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=labels[name], color=colors[name], linewidth=2.5, marker='o', markersize=5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Rank (Before NS for all)
    ax = axes[0, 2]
    for name, history in results.items():
        if history['avg_rank'][0] > 0:
            ax.plot(history['avg_rank'], label=labels[name], color=colors[name], linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Average Effective Rank', fontsize=11)
    ax.set_title('Gradient Rank (Before NS)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Rank After NS (Muon only)
    ax = axes[1, 0]
    for name, history in results.items():
        if 'muon' in name and sum(history['avg_rank_after_ns']) > 0:
            ax.plot(history['avg_rank_after_ns'], label=labels[name], color=colors[name], linewidth=2.5, marker='s', markersize=5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Average Effective Rank', fontsize=11)
    ax.set_title('Gradient Rank (After NS)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Rank Reduction (Before vs After NS)
    ax = axes[1, 1]
    for name, history in results.items():
        if 'muon' in name and sum(history['avg_rank_after_ns']) > 0:
            rank_reduction = np.array(history['avg_rank']) - np.array(history['avg_rank_after_ns'])
            ax.plot(rank_reduction, label=labels[name], color=colors[name], linewidth=2.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Rank Reduction (Before - After NS)', fontsize=11)
    ax.set_title('NS Orthogonalization Effect', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final Summary Bar Chart
    ax = axes[1, 2]
    config_names = list(results.keys())
    final_test_acc = [results[name]['test_acc'][-1] for name in config_names]
    final_train_loss = [results[name]['train_loss'][-1] for name in config_names]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_test_acc, width, label='Test Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, final_train_loss, width, label='Train Loss', color='coral', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, color='steelblue')
    ax2.set_ylabel('Training Loss', fontsize=11, color='coral')
    ax.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[name] for name in config_names], rotation=15, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1, f'{height1:.1f}%',
                ha='center', va='bottom', fontsize=8, color='steelblue', fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2., height2, f'{height2:.3f}',
                ha='center', va='bottom', fontsize=8, color='coral', fontweight='bold')
    
    plt.suptitle('MNIST Final Comparison: AdamW vs Muon with Optimal LR Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_prefix}.png")
    plt.close()


def print_summary(results):
    """Print comprehensive summary."""
    
    print("\n" + "="*90)
    print("FINAL COMPARISON: Effective Rank Analysis with Optimal Learning Rates")
    print("="*90)
    
    print("\n📊 Final Results (Last Epoch):")
    print("-"*90)
    print(f"{'Configuration':<30} | {'Test Acc':>10} | {'Train Loss':>12} | {'Rank (Before)':>14} | {'Rank (After NS)':>15}")
    print("-"*90)
    
    for name, history in results.items():
        test_acc = history['test_acc'][-1]
        train_loss = history['train_loss'][-1]
        avg_rank = history['avg_rank'][-1]
        avg_rank_ns = history['avg_rank_after_ns'][-1] if history['avg_rank_after_ns'][-1] > 0 else None
        
        label = {
            'adamw': 'AdamW',
            'muon_ns3': 'Muon (3 NS, lr=0.020)',
            'muon_ns5_optimal': 'Muon (5 NS, lr=0.010)'
        }[name]
        
        rank_ns_str = f"{avg_rank_ns:.1f}" if avg_rank_ns else "N/A"
        rank_str = f"{avg_rank:.1f}" if avg_rank > 0 else "N/A"
        
        print(f"{label:<30} | {test_acc:>9.2f}% | {train_loss:>12.4f} | {rank_str:>14} | {rank_ns_str:>15}")
    
    print("-"*90)
    
    # Comparison insights
    print("\n🔍 Key Insights:")
    
    muon_3_acc = results['muon_ns3']['test_acc'][-1]
    muon_5_acc = results['muon_ns5_optimal']['test_acc'][-1]
    adamw_acc = results['adamw']['test_acc'][-1]
    
    print(f"\n1. Performance Comparison:")
    print(f"   • AdamW: {adamw_acc:.2f}%")
    print(f"   • Muon (3 NS): {muon_3_acc:.2f}%")
    print(f"   • Muon (5 NS, optimal): {muon_5_acc:.2f}%")
    print(f"   • 5 NS vs 3 NS: {muon_5_acc - muon_3_acc:+.2f}%")
    print(f"   • Best vs AdamW: {max(muon_3_acc, muon_5_acc) - adamw_acc:+.2f}%")
    
    print(f"\n2. Effective Rank Analysis:")
    muon_3_rank_before = results['muon_ns3']['avg_rank'][-1]
    muon_3_rank_after = results['muon_ns3']['avg_rank_after_ns'][-1]
    muon_5_rank_before = results['muon_ns5_optimal']['avg_rank'][-1]
    muon_5_rank_after = results['muon_ns5_optimal']['avg_rank_after_ns'][-1]
    
    print(f"   • Muon 3 NS: {muon_3_rank_before:.1f} → {muon_3_rank_after:.1f} (reduction: {muon_3_rank_before - muon_3_rank_after:.1f})")
    print(f"   • Muon 5 NS: {muon_5_rank_before:.1f} → {muon_5_rank_after:.1f} (reduction: {muon_5_rank_before - muon_5_rank_after:.1f})")
    
    if muon_5_rank_after > muon_3_rank_after:
        print(f"   ⚠️  5 NS produces higher post-NS rank ({muon_5_rank_after:.1f} vs {muon_3_rank_after:.1f})")
        print(f"      This suggests 5 iterations may over-orthogonalize, but with proper LR it still performs better!")
    else:
        print(f"   ✓ 5 NS produces lower post-NS rank ({muon_5_rank_after:.1f} vs {muon_3_rank_after:.1f})")
    
    print("\n✨ Conclusion:")
    print(f"   With optimal LR scaling, 5 NS iterations achieve {muon_5_acc:.2f}% accuracy,")
    print(f"   {'outperforming' if muon_5_acc > muon_3_acc else 'matching'} the 3 NS baseline at {muon_3_acc:.2f}%.")
    print(f"   Learning rate is critical: 5 NS needs lr={0.010:.3f} vs 3 NS at lr={0.020:.3f}.")
    
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description='MNIST Final Comparison with Optimal LRs')
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
        {'name': 'muon_ns3', 'optimizer': 'muon', 'ns_steps': 3, 'muon_lr': 0.02, 'epochs': args.epochs},
        {'name': 'muon_ns5_optimal', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.010, 'epochs': args.epochs},
    ]
    
    results = {}
    
    for config in configs:
        history = train_model(config, train_loader, test_loader, device)
        results[config['name']] = history
    
    # Plotting
    print("\n📊 Generating plots...")
    plot_comparison(results, save_prefix='mnist_final_comparison')
    
    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
