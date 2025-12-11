"""
Modular Learning Rate Scaling Experiment

Based on the modular-manifolds concept: "Scalar coefficients sᵢ budget 
the learning rates across layers" when composing modules.

Tests various LR scaling strategies:
1. Depth-scaled LR (layers closer to output get different LR)
2. Gradient norm-aware LR (scale based on gradient magnitude)
3. Lipschitz-aware LR (scale based on spectral norm)
4. Adaptive scaling (track norms, adaptively scale)

Goal: Understand if per-layer LR budgeting improves training.
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


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DepthAwareMNISTNet(nn.Module):
    """MLP with depth-tracking for LR scaling experiments."""
    
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = 784
        
        for h in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, h))
            prev_size = h
        
        self.layers.append(nn.Linear(prev_size, 10))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
        
        return self.layers[-1](x)
    
    def get_layer_depths(self):
        """Return depth info for each layer (0 = input, higher = closer to output)."""
        depths = {}
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            depths[f'layers.{i}.weight'] = i / (n_layers - 1)  # Normalized 0-1
            depths[f'layers.{i}.bias'] = i / (n_layers - 1)
        return depths


class ModularLROptimizer:
    """Optimizer wrapper that applies per-layer LR scaling."""
    
    def __init__(self, model, base_lr=1e-3, scaling_strategy='uniform', 
                 depth_scale=1.0, momentum=0.9, weight_decay=0.01):
        self.model = model
        self.base_lr = base_lr
        self.scaling_strategy = scaling_strategy
        self.depth_scale = depth_scale
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Get depth info
        self.depths = model.get_layer_depths()
        
        # Create param groups with per-layer LR
        self.param_groups = self._create_param_groups()
        
        # Momentum buffers
        self.state = {p: {'momentum': torch.zeros_like(p)} 
                      for p in model.parameters()}
        
        # Track gradient norms for adaptive scaling
        self.grad_norm_ema = {}
        self.ema_beta = 0.99
        
    def _create_param_groups(self):
        """Create parameter groups with scaled learning rates."""
        groups = []
        
        for name, param in self.model.named_parameters():
            depth = self.depths.get(name, 0.5)
            
            if self.scaling_strategy == 'uniform':
                lr = self.base_lr
            elif self.scaling_strategy == 'depth_linear':
                # Deeper layers get higher LR
                lr = self.base_lr * (1 + self.depth_scale * depth)
            elif self.scaling_strategy == 'depth_inverse':
                # Deeper layers get lower LR (traditional approach)
                lr = self.base_lr * (1 - 0.5 * depth)
            elif self.scaling_strategy == 'depth_exponential':
                # Exponential scaling
                lr = self.base_lr * (2 ** (self.depth_scale * depth))
            else:
                lr = self.base_lr
            
            groups.append({
                'name': name,
                'params': [param],
                'lr': lr,
                'depth': depth,
            })
        
        return groups
    
    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Take optimization step with per-layer LR."""
        
        for group in self.param_groups:
            param = group['params'][0]
            name = group['name']
            
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Adaptive scaling: adjust LR based on gradient norm EMA
            if self.scaling_strategy == 'adaptive':
                grad_norm = grad.norm().item()
                
                if name not in self.grad_norm_ema:
                    self.grad_norm_ema[name] = grad_norm
                else:
                    self.grad_norm_ema[name] = (
                        self.ema_beta * self.grad_norm_ema[name] + 
                        (1 - self.ema_beta) * grad_norm
                    )
                
                # Scale LR inversely with gradient norm
                norm_scale = 1.0 / (self.grad_norm_ema[name] + 1e-7)
                lr = self.base_lr * min(norm_scale, 10.0)  # Cap scaling
            else:
                lr = group['lr']
            
            # Momentum
            state = self.state[param]
            state['momentum'] = self.momentum * state['momentum'] + grad
            
            # Update
            param.data.add_(state['momentum'], alpha=-lr)
    
    def get_lr_schedule(self):
        """Return current LR for each layer."""
        return {g['name']: g['lr'] for g in self.param_groups}


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


def train_with_strategy(strategy, train_loader, test_loader, device, 
                       n_epochs=10, base_lr=1e-3, depth_scale=1.0):
    """Train model with given LR scaling strategy."""
    
    set_seed(42)
    
    model = DepthAwareMNISTNet(hidden_sizes=[512, 256, 128]).to(device)
    optimizer = ModularLROptimizer(
        model, 
        base_lr=base_lr, 
        scaling_strategy=strategy,
        depth_scale=depth_scale
    )
    
    print(f"\n{'='*60}")
    print(f"Training with strategy: {strategy}")
    print(f"{'='*60}")
    
    # Print LR schedule
    lr_schedule = optimizer.get_lr_schedule()
    print("Layer LRs:")
    for name, lr in lr_schedule.items():
        if 'weight' in name:
            print(f"  {name}: {lr:.6f}")
    
    history = {
        'train_loss': [],
        'test_acc': [],
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
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
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    return history


def run_experiment(n_epochs=10, batch_size=128):
    """Run modular LR scaling experiment comparing strategies."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    
    strategies = [
        ('uniform', 1.0),
        ('depth_linear', 0.5),
        ('depth_linear', 1.0),
        ('depth_inverse', 1.0),
        ('adaptive', 1.0),
    ]
    
    results = {}
    
    for strategy, depth_scale in strategies:
        key = f"{strategy}" if depth_scale == 1.0 else f"{strategy}_{depth_scale}"
        history = train_with_strategy(
            strategy, train_loader, test_loader, device,
            n_epochs=n_epochs,
            depth_scale=depth_scale
        )
        results[key] = history
    
    return results


def plot_results(results, save_prefix='modular_lr'):
    """Plot comparison of LR scaling strategies."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # Plot 1: Training Loss
    ax = axes[0]
    for (name, history), color in zip(results.items(), colors):
        ax.plot(history['train_loss'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss by LR Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[1]
    for (name, history), color in zip(results.items(), colors):
        ax.plot(history['test_acc'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy by LR Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Modular Learning Rate Scaling Experiment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {save_prefix}.png")
    plt.close()


def print_summary(results):
    """Print summary of results."""
    
    print("\n" + "="*70)
    print("MODULAR LR SCALING SUMMARY")
    print("="*70)
    
    print(f"\n{'Strategy':<25} | {'Final Loss':<12} | {'Final Acc':<12}")
    print("-"*55)
    
    best_acc = 0
    best_strategy = None
    
    for name, history in results.items():
        final_loss = history['train_loss'][-1]
        final_acc = history['test_acc'][-1]
        print(f"{name:<25} | {final_loss:<12.4f} | {final_acc:<11.2f}%")
        
        if final_acc > best_acc:
            best_acc = final_acc
            best_strategy = name
    
    print("-"*55)
    print(f"\n🏆 Best Strategy: {best_strategy} ({best_acc:.2f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Modular LR Scaling Experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    results = run_experiment(n_epochs=args.epochs, batch_size=args.batch_size)
    plot_results(results)
    print_summary(results)


if __name__ == '__main__':
    main()
