"""
MNIST Diagnostic Experiment: Learning Rate Scaling for Newton-Schulz Iterations

This experiment tests the hypothesis that more NS iterations require lower learning rates
to prevent overshooting and maintain optimization stability.

Test configurations:
1. Muon (3 NS, lr=0.02) - baseline
2. Muon (5 NS, lr=0.02) - current poor performer
3. Muon (5 NS, lr=0.014) - scaled by sqrt(3/5)
4. Muon (5 NS, lr=0.012) - scaled by 3/5
5. Muon (5 NS, lr=0.010) - conservative scaling
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

from optimizers.muon import Muon


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_epoch(model, optimizer, train_loader, device, muon_optimizer=None):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        if muon_optimizer:
            muon_optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
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
        'accuracy': 100. * correct / total
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
    print(f"Config: {config['name']}")
    print(f"  NS steps: {ns_steps}, Muon LR: {muon_lr:.4f}")
    print(f"{'='*70}")
    
    # Setup optimizers
    muon_optimizer = None
    
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
        'test_acc': []
    }
    
    n_epochs = config['epochs']
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(model, optimizer, train_loader, device, muon_optimizer=muon_optimizer)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Test Acc: {test_metrics['accuracy']:.2f}%")
    
    return history


def plot_comparison(results, save_prefix='mnist_lr_scaling'):
    """Plot comparison of all configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define colors and labels
    colors = {
        'muon_ns3_lr020': '#1f77b4',
        'muon_ns5_lr020': '#d62728',
        'muon_ns5_lr014': '#ff7f0e',
        'muon_ns5_lr012': '#2ca02c',
        'muon_ns5_lr010': '#9467bd'
    }
    
    labels = {
        'muon_ns3_lr020': 'Muon (3 NS, lr=0.020)',
        'muon_ns5_lr020': 'Muon (5 NS, lr=0.020)',
        'muon_ns5_lr014': 'Muon (5 NS, lr=0.014)',
        'muon_ns5_lr012': 'Muon (5 NS, lr=0.012)',
        'muon_ns5_lr010': 'Muon (5 NS, lr=0.010)'
    }
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=labels[name], color=colors[name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=labels[name], color=colors[name], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Test Accuracy vs Learning Rate
    ax = axes[1, 0]
    lrs = []
    final_accs = []
    config_names = []
    for name, history in results.items():
        if 'ns5' in name:
            lr = float(name.split('lr')[1]) / 1000  # Extract LR from name
            lrs.append(lr)
            final_accs.append(history['test_acc'][-1])
            config_names.append(labels[name])
    
    # Add baseline
    baseline_name = [k for k in results.keys() if 'ns3' in k][0]
    baseline_acc = results[baseline_name]['test_acc'][-1]
    
    ax.plot(lrs, final_accs, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='5 NS iterations')
    ax.axhline(y=baseline_acc, color='#1f77b4', linestyle='--', linewidth=2, label=f'3 NS baseline ({baseline_acc:.2f}%)')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Learning Rate vs Final Test Accuracy (5 NS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Training Loss vs Learning Rate
    ax = axes[1, 1]
    final_losses = []
    for name, history in results.items():
        if 'ns5' in name:
            final_losses.append(history['train_loss'][-1])
    
    baseline_loss = results[baseline_name]['train_loss'][-1]
    
    ax.plot(lrs, final_losses, 's-', color='#d62728', linewidth=2, markersize=8, label='5 NS iterations')
    ax.axhline(y=baseline_loss, color='#1f77b4', linestyle='--', linewidth=2, label=f'3 NS baseline ({baseline_loss:.4f})')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Learning Rate vs Final Training Loss (5 NS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('MNIST LR Scaling Diagnostic: Effect on Newton-Schulz Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_prefix}.png")
    plt.close()


def print_summary(results):
    """Print summary of results."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC EXPERIMENT SUMMARY: Learning Rate Scaling for NS Iterations")
    print("="*80)
    
    print("\n📊 Final Results (Last Epoch):")
    print("-"*80)
    print(f"{'Configuration':<35} | {'Train Loss':>12} | {'Test Acc':>10} | {'Improvement':>12}")
    print("-"*80)
    
    # Get baseline
    baseline_name = [k for k in results.keys() if 'ns3' in k][0]
    baseline_acc = results[baseline_name]['test_acc'][-1]
    
    for name, history in results.items():
        train_loss = history['train_loss'][-1]
        test_acc = history['test_acc'][-1]
        improvement = test_acc - baseline_acc
        
        label = {
            'muon_ns3_lr020': 'Muon (3 NS, lr=0.020) [baseline]',
            'muon_ns5_lr020': 'Muon (5 NS, lr=0.020)',
            'muon_ns5_lr014': 'Muon (5 NS, lr=0.014)',
            'muon_ns5_lr012': 'Muon (5 NS, lr=0.012)',
            'muon_ns5_lr010': 'Muon (5 NS, lr=0.010)'
        }[name]
        
        improvement_str = f"{improvement:+.2f}%" if 'baseline' not in label else 'baseline'
        print(f"{label:<35} | {train_loss:>12.4f} | {test_acc:>9.2f}% | {improvement_str:>12}")
    
    print("-"*80)
    
    # Find best configuration
    print("\n🏆 Best Configuration (5 NS iterations):")
    ns5_configs = {k: v for k, v in results.items() if 'ns5' in k}
    best_config = max(ns5_configs.items(), key=lambda x: x[1]['test_acc'][-1])
    
    label_map = {
        'muon_ns5_lr020': 'lr=0.020',
        'muon_ns5_lr014': 'lr=0.014',
        'muon_ns5_lr012': 'lr=0.012',
        'muon_ns5_lr010': 'lr=0.010'
    }
    
    print(f"  • Best LR: {label_map[best_config[0]]}")
    print(f"  • Test Accuracy: {best_config[1]['test_acc'][-1]:.2f}%")
    print(f"  • Training Loss: {best_config[1]['train_loss'][-1]:.4f}")
    print(f"  • vs Baseline (3 NS): {best_config[1]['test_acc'][-1] - baseline_acc:+.2f}%")
    
    # Conclusion
    print("\n📝 Conclusion:")
    if best_config[1]['test_acc'][-1] >= baseline_acc - 0.5:
        print(f"  ✓ Learning rate scaling WORKS! With proper LR tuning, 5 NS iterations can")
        print(f"    match or exceed 3 NS iteration performance.")
        print(f"  • Recommended LR for 5 NS: {label_map[best_config[0]]}")
    else:
        print(f"  ✗ Learning rate scaling alone is NOT sufficient. Need to investigate other factors.")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='MNIST LR Scaling Diagnostic Experiment')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📦 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"✓ Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Test configurations
    configs = [
        {'name': 'muon_ns3_lr020', 'optimizer': 'muon', 'ns_steps': 3, 'muon_lr': 0.02, 'epochs': args.epochs},
        {'name': 'muon_ns5_lr020', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.02, 'epochs': args.epochs},
        {'name': 'muon_ns5_lr014', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.014, 'epochs': args.epochs},
        {'name': 'muon_ns5_lr012', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.012, 'epochs': args.epochs},
        {'name': 'muon_ns5_lr010', 'optimizer': 'muon', 'ns_steps': 5, 'muon_lr': 0.010, 'epochs': args.epochs},
    ]
    
    results = {}
    
    for config in configs:
        history = train_model(config, train_loader, test_loader, device)
        results[config['name']] = history
    
    # Plotting
    print("\n📊 Generating plots...")
    plot_comparison(results, save_prefix='mnist_lr_scaling')
    
    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
