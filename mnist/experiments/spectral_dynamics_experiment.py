"""
Spectral Dynamics Experiment: Full SVD Tracking During Training

This experiment goes beyond effective rank to track the FULL singular value 
distribution of gradients throughout training. This helps understand:
- HOW gradients change (not just effective rank)
- Spectral collapse patterns
- Difference between early and late training dynamics

Works with both MNIST and LLM models.
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
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimizers.muon import Muon


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MNISTNet(nn.Module):
    """Simple MLP for MNIST."""
    
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_size = 784
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        
        layers.append(nn.Linear(prev_size, 10))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
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


def compute_spectral_metrics(matrix):
    """Compute comprehensive spectral metrics for a matrix."""
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return None
    
    try:
        S = torch.linalg.svdvals(matrix.float())
        S_normalized = S / (S.sum() + 1e-10)
        
        # Effective rank (entropy-based)
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()
        
        # Condition number
        condition_number = (S[0] / (S[-1] + 1e-10)).item()
        
        # Top-k concentration
        top1_ratio = (S[0] / S.sum()).item()
        top5_ratio = (S[:5].sum() / S.sum()).item() if len(S) >= 5 else 1.0
        top10_ratio = (S[:10].sum() / S.sum()).item() if len(S) >= 10 else 1.0
        
        # Frobenius norm
        frobenius = matrix.norm().item()
        
        # Full spectrum (downsample if too large)
        max_vals = 50
        if len(S) > max_vals:
            indices = torch.linspace(0, len(S)-1, max_vals).long()
            spectrum = S[indices].cpu().numpy()
        else:
            spectrum = S.cpu().numpy()
        
        return {
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'top1_ratio': top1_ratio,
            'top5_ratio': top5_ratio,
            'top10_ratio': top10_ratio,
            'frobenius_norm': frobenius,
            'spectrum': spectrum,
            'max_singular': S[0].item(),
            'min_singular': S[-1].item(),
        }
    except Exception as e:
        print(f"Error computing spectral metrics: {e}")
        return None


def collect_gradient_spectra(model, layer_names=None):
    """Collect spectral metrics for all 2D gradient matrices."""
    spectra = {}
    
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.ndim != 2:
            continue
        if min(param.grad.shape) < 4:
            continue
        
        # Filter by layer names if specified
        if layer_names is not None:
            if not any(ln in name for ln in layer_names):
                continue
        
        grad = param.grad.detach().float()
        metrics = compute_spectral_metrics(grad)
        
        if metrics is not None:
            spectra[name] = metrics
    
    return spectra


def train_and_track(model, optimizer, train_loader, device, n_epochs=10, 
                    track_interval=50, muon_optimizer=None):
    """Train model while tracking spectral dynamics."""
    
    history = {
        'loss': [],
        'accuracy': [],
        'spectra': [],  # List of {layer_name: metrics} per tracking step
        'epochs': [],
        'steps': [],
    }
    
    step = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            if muon_optimizer:
                muon_optimizer.zero_grad()
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Track spectral metrics
            if step % track_interval == 0:
                spectra = collect_gradient_spectra(model)
                history['spectra'].append(spectra)
                history['epochs'].append(epoch)
                history['steps'].append(step)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if muon_optimizer:
                muon_optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    
    return history


def plot_spectral_dynamics(history, optimizer_name, save_prefix='spectral'):
    """Plot comprehensive spectral dynamics visualization."""
    
    if not history['spectra']:
        print("No spectral data to plot")
        return
    
    # Get layer names from first entry
    layer_names = list(history['spectra'][0].keys())
    n_layers = min(len(layer_names), 4)  # Plot up to 4 layers
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(history['spectra'])))
    
    # Plot 1: Effective Rank over training
    ax = axes[0, 0]
    for layer_name in layer_names[:n_layers]:
        ranks = [s[layer_name]['effective_rank'] for s in history['spectra'] if layer_name in s]
        steps = history['steps'][:len(ranks)]
        ax.plot(steps, ranks, label=layer_name.split('.')[-2] if '.' in layer_name else layer_name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Condition Number over training
    ax = axes[0, 1]
    for layer_name in layer_names[:n_layers]:
        cond = [s[layer_name]['condition_number'] for s in history['spectra'] if layer_name in s]
        steps = history['steps'][:len(cond)]
        ax.semilogy(steps, cond, label=layer_name.split('.')[-2] if '.' in layer_name else layer_name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_title('Condition Number Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Top-k Concentration
    ax = axes[1, 0]
    first_layer = layer_names[0]
    top1 = [s[first_layer]['top1_ratio'] for s in history['spectra'] if first_layer in s]
    top5 = [s[first_layer]['top5_ratio'] for s in history['spectra'] if first_layer in s]
    top10 = [s[first_layer]['top10_ratio'] for s in history['spectra'] if first_layer in s]
    steps = history['steps'][:len(top1)]
    
    ax.plot(steps, top1, label='Top-1', linewidth=2)
    ax.plot(steps, top5, label='Top-5', linewidth=2)
    ax.plot(steps, top10, label='Top-10', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction of Total Singular Value Mass')
    ax.set_title(f'Singular Value Concentration ({first_layer.split(".")[-2]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Spectrum Evolution (heatmap-style)
    ax = axes[1, 1]
    first_layer = layer_names[0]
    spectra_matrix = np.array([s[first_layer]['spectrum'] for s in history['spectra'] if first_layer in s])
    
    # Normalize each row for visualization
    spectra_matrix_norm = spectra_matrix / (spectra_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    im = ax.imshow(spectra_matrix_norm.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Training Step Index')
    ax.set_ylabel('Singular Value Index')
    ax.set_title(f'Spectrum Evolution ({first_layer.split(".")[-2]})')
    plt.colorbar(im, ax=ax, label='Normalized σ')
    
    plt.suptitle(f'Spectral Dynamics Analysis - {optimizer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{optimizer_name.lower()}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_prefix}_{optimizer_name.lower()}.png")
    plt.close()


def run_experiment(optimizer_name='adamw', n_epochs=10, batch_size=128):
    """Run spectral dynamics experiment for a given optimizer."""
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    
    # Create model
    model = MNISTNet(hidden_sizes=[512, 256, 128]).to(device)
    
    # Setup optimizer
    muon_optimizer = None
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    elif optimizer_name == 'muon':
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        
        optimizer = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.01)
        muon_optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"\n{'='*60}")
    print(f"Running Spectral Dynamics Experiment: {optimizer_name.upper()}")
    print(f"{'='*60}")
    
    # Train and track
    history = train_and_track(
        model, optimizer, train_loader, device,
        n_epochs=n_epochs,
        track_interval=50,
        muon_optimizer=muon_optimizer
    )
    
    # Plot results
    plot_spectral_dynamics(history, optimizer_name)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Spectral Dynamics Experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='both', 
                        choices=['adamw', 'muon', 'both'], help='Optimizer to test')
    args = parser.parse_args()
    
    results = {}
    
    if args.optimizer in ['adamw', 'both']:
        results['adamw'] = run_experiment('adamw', args.epochs, args.batch_size)
    
    if args.optimizer in ['muon', 'both']:
        results['muon'] = run_experiment('muon', args.epochs, args.batch_size)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for name, history in results.items():
        if history['spectra']:
            first_layer = list(history['spectra'][0].keys())[0]
            initial_rank = history['spectra'][0][first_layer]['effective_rank']
            final_rank = history['spectra'][-1][first_layer]['effective_rank']
            print(f"\n{name.upper()}:")
            print(f"  First layer effective rank: {initial_rank:.1f} → {final_rank:.1f}")
            print(f"  Final loss: {history['loss'][-1]:.4f}")
            print(f"  Final accuracy: {history['accuracy'][-1]:.2f}%")


if __name__ == '__main__':
    main()
