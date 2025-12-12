"""
LLM Spectral Dynamics Experiment: Full SVD Tracking During Training

Tracks spectral properties of gradients throughout LLM training.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimizers.muon import Muon
from llm.common import create_dataloaders, compute_spectral_metrics
from llm.configs.training_config import ModelConfig
from llm.models.model import create_model


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_gradient_spectra(model, layer_names=None):
    """Collect spectral metrics for all 2D gradient matrices."""
    spectra = {}
    
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.ndim != 2:
            continue
        if min(param.grad.shape) < 4:
            continue
        
        if layer_names is not None:
            if not any(ln in name for ln in layer_names):
                continue
        
        grad = param.grad.detach().float()
        metrics = compute_spectral_metrics(grad)
        
        if metrics is not None:
            spectra[name] = metrics
    
    return spectra


def train_and_track(model, optimizer, train_loader, device, max_steps=200, 
                    track_interval=10, muon_optimizer=None):
    """Train model while tracking spectral dynamics."""
    
    history = {
        'loss': [],
        'spectra': [],
        'steps': [],
    }
    
    model.train()
    step = 0
    
    for batch in train_loader:
        if step >= max_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        if muon_optimizer:
            muon_optimizer.zero_grad()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()
        
        # Track spectral metrics
        if step % track_interval == 0:
            spectra = collect_gradient_spectra(model)
            history['spectra'].append(spectra)
            history['steps'].append(step)
            history['loss'].append(loss.item())
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if muon_optimizer:
            muon_optimizer.step()
        
        step += 1
    
    return history


def plot_spectral_dynamics(history, optimizer_name, save_prefix=' spectral'):
    """Plot spectral dynamics visualization."""
    
    if not history['spectra']:
        print("No spectral data to plot")
        return
    
    layer_names = list(history['spectra'][0].keys())
    n_layers = min(len(layer_names), 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
    ax.set_title(f'Singular Value Concentration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    ax = axes[1, 1]
    ax.plot(history['steps'], history['loss'], linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Spectral Dynamics Analysis - {optimizer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'llm/figures/{save_prefix}_{optimizer_name.lower()}.png'
    os.makedirs('llm/figures', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LLM Spectral Dynamics Experiment')
    parser.add_argument('--max_steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--optimizer', type=str, default='muon', choices=['adamw', 'muon'])
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    train_loader, _ = create_dataloaders(seq_len=256, num_samples=5000, batch_size=16)
    
    # Create model
    model_config = ModelConfig()
    model = create_model(model_config, device)
    
    # Setup optimizer
    muon_optimizer = None
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    else:  # muon
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        
        optimizer = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)
        muon_optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
    
    print(f"\n{'='*60}")
    print(f"Running Spectral Dynamics Experiment: {args.optimizer.upper()}")
    print(f"{'='*60}")
    
    # Train and track
    history = train_and_track(
        model, optimizer, train_loader, device,
        max_steps=args.max_steps,
        track_interval=10,
        muon_optimizer=muon_optimizer
    )
    
    # Plot results
    plot_spectral_dynamics(history, args.optimizer)


if __name__ == '__main__':
    main()
