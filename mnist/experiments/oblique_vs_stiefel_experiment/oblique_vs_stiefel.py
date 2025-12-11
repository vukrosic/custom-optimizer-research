"""
Oblique vs Stiefel Matrix Dynamics Experiment

This experiment specifically focuses on the geometric differences between:
1. Oblique Optimizer (Unit-norm columns)
2. Stiefel Optimizer (Orthonormal columns)

We track:
- Orthogonality Error ||W^T W - I||_F
- Off-Diagonal Mass (correlation between columns)
- Cosine Similarity Distribution
- Sparsity
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

# Add project root to path
# Go up from: mnist/experiments/oblique_vs_stiefel_experiment/oblique_vs_stiefel.py
# To: root/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from optimizers.oblique import ObliqueOptimizer
from optimizers.l1_stiefel import StiefelOptimizer
from optimizers.muon import Muon
from mnist.common.models import MNISTNet

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Use shared data folder in mnist/data
    data_dir = os.path.join(project_root, 'mnist', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def compute_geometric_metrics(matrix):
    """Compute geometric properties of the weight matrix."""
    if matrix.ndim != 2:
        return None
    
    W = matrix.detach().float()
    rows, cols = W.shape
    
    # Analyze W^T W (Gram matrix)
    # Ensure we are analyzing correlations between columns (features)
    # MNISTNet linear layers are (out_features, in_features) -> (rows, cols)
    # So columns of W are the rows of the standard matrix notation w_i
    # We want to check properties of rows (the incoming feature vectors) in PyTorch convention
    # PyTorch Linear: y = x A^T + b.  Weight shape is (out, in).
    # We want to check columns of the weight matrix?
    # Oblique/Stiefel optimizers treat the last dimension as the one to normalize? [CHECK THIS]
    
    # Checking Oblique implementation:
    # norms = W.norm(dim=0) -> Normalizes columns.
    # So W is treated as (m, n) where columns are normalized.
    
    if rows < cols:
        # If fat matrix, we often normalize rows?
        # Let's stick to the implementation's convention: dim=0 is normalized.
        gram = W.T @ W
    else:
        # Tall matrix
        gram = W.T @ W

    # 1. Orthogonality Error (how far from I)
    I = torch.eye(gram.shape[0], device=gram.device)
    ortho_error = torch.norm(gram - I, p='fro').item() / gram.numel()
    
    # 2. Diagonal vs Off-Diagonal
    diag = torch.diag(gram)
    diag_mean = diag.mean().item()
    
    # Off-diagonal mask
    mask = ~torch.eye(gram.shape[0], dtype=bool, device=gram.device)
    off_diag_mean = gram[mask].abs().mean().item()
    
    # 3. Sparsity (fraction < 0.01)
    sparsity = (W.abs() < 0.01).float().mean().item()
    
    return {
        'ortho_error': ortho_error,
        'diag_mean': diag_mean,
        'off_diag_mean': off_diag_mean,
        'sparsity': sparsity,
        'gram_matrix': gram.cpu().numpy() if gram.shape[0] <= 256 else None # Only save small grams
    }

class MultiOptimizer:
    """Wrapper to handle multiple optimizers (Muon for 2D, AdamW for others)."""
    def __init__(self, optimizers):
        self.optimizers = optimizers
    
    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def step(self):
        for opt in self.optimizers.values():
            opt.step()
    
    @property
    def param_groups(self):
        # Return param groups from all optimizers
        groups = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups

def normalize_initialization_scale(model, original_scale):
    """Normalize weight scales after projection to match original initialization scale.
    
    This ensures fair comparison by making all optimizers start from similar
    output scales, isolating the effect of learning dynamics rather than initialization.
    """
    # Get a sample input to measure output scale
    sample_input = torch.randn(1, 784).to(next(model.parameters()).device)
    
    # Measure current output scale of first layer (after projection)
    with torch.no_grad():
        first_layer = model.network[0]
        output = sample_input @ first_layer.weight.T
        if first_layer.bias is not None:
            output = output + first_layer.bias
        current_scale = output.abs().mean().item()
        
        # Scale weights to match original scale
        if current_scale > 1e-6:
            scale_factor = original_scale / current_scale
            first_layer.weight.data *= scale_factor
            if first_layer.bias is not None:
                first_layer.bias.data *= scale_factor

def train_network(optimizer_name, train_loader, test_loader, device, epochs=5):
    set_seed(42)
    model = MNISTNet(hidden_sizes=[256, 128]).to(device) # Smaller width for cleaner visualization
    
    # Measure original output scale before any projection (for fair comparison)
    sample_input = torch.randn(1, 784).to(device)
    with torch.no_grad():
        first_layer = model.network[0]
        original_output = sample_input @ first_layer.weight.T
        if first_layer.bias is not None:
            original_output = original_output + first_layer.bias
        original_scale = original_output.abs().mean().item()
    
    # Separate 2D matrices from other parameters
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.ndim < 2]
    
    if optimizer_name == 'oblique':
        # Muon for 2D matrices, wrapped with Oblique
        muon_opt = Muon(matrix_params, lr=0.02, momentum=0.95)
        oblique_opt = ObliqueOptimizer(matrix_params, muon_opt)
        # AdamW for other parameters
        adamw_opt = torch.optim.AdamW(other_params, lr=1e-3) if other_params else None
        
        optimizers = {'oblique': oblique_opt}
        if adamw_opt:
            optimizers['adamw'] = adamw_opt
        optimizer = MultiOptimizer(optimizers)
        # Normalize initialization scale to match baseline (fair comparison)
        normalize_initialization_scale(model, original_scale)
        
    elif optimizer_name == 'stiefel':
        # Muon for 2D matrices, wrapped with Stiefel
        muon_opt = Muon(matrix_params, lr=0.02, momentum=0.95)
        stiefel_opt = StiefelOptimizer(matrix_params, muon_opt)
        # AdamW for other parameters
        adamw_opt = torch.optim.AdamW(other_params, lr=1e-3) if other_params else None
        
        optimizers = {'stiefel': stiefel_opt}
        if adamw_opt:
            optimizers['adamw'] = adamw_opt
        optimizer = MultiOptimizer(optimizers)
        # Normalize initialization scale to match baseline (fair comparison)
        normalize_initialization_scale(model, original_scale)
    
    elif optimizer_name == 'muon':
        # Muon for 2D matrices (baseline, no manifold constraint)
        muon_opt = Muon(matrix_params, lr=0.02, momentum=0.95)
        # AdamW for other parameters
        adamw_opt = torch.optim.AdamW(other_params, lr=1e-3) if other_params else None
        
        optimizers = {'muon': muon_opt}
        if adamw_opt:
            optimizers['adamw'] = adamw_opt
        optimizer = MultiOptimizer(optimizers)
    
    history = {
        'loss': [],
        'acc': [],
        'ortho_error': [],
        'off_diag_mean': [],
        'sparsity': [],
        'final_gram': None
    }
    
    print(f"\nTraining {optimizer_name.upper()}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track matrix metrics every 50 batches
            if batch_idx % 50 == 0:
                # Analyze first hidden layer (fc1)
                metrics = compute_geometric_metrics(model.network[0].weight)
                history['ortho_error'].append(metrics['ortho_error'])
                history['off_diag_mean'].append(metrics['off_diag_mean'])
                history['sparsity'].append(metrics['sparsity'])
                
        avg_loss = train_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Off-Diag {history['off_diag_mean'][-1]:.4f} | Sparsity {history['sparsity'][-1]:.4f}")
        
    # Save final Gram matrix of first layer for histogram
    final_metrics = compute_geometric_metrics(model.network[0].weight)
    history['final_gram'] = final_metrics['gram_matrix']
    
    return history

def plot_comparison(oblique_hist, stiefel_hist, muon_hist):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Off-Diagonal Correlation (The main difference)
    ax = axes[0, 0]
    ax.plot(oblique_hist['off_diag_mean'], label='Oblique', color='blue')
    ax.plot(stiefel_hist['off_diag_mean'], label='Stiefel', color='green')
    ax.plot(muon_hist['off_diag_mean'], label='Muon (baseline)', color='red', linestyle='--')
    ax.set_title("Off-Diagonal Correlation (Mean Abs Value)")
    ax.set_ylabel("Mean |<w_i, w_j>|")
    ax.set_xlabel("Step (x50)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sparsity
    ax = axes[0, 1]
    ax.plot(oblique_hist['sparsity'], label='Oblique', color='blue')
    ax.plot(stiefel_hist['sparsity'], label='Stiefel', color='green')
    ax.plot(muon_hist['sparsity'], label='Muon (baseline)', color='red', linestyle='--')
    ax.set_title("Sparsity (Fraction of weights < 0.01)")
    ax.set_ylabel("Sparsity Fraction")
    ax.set_xlabel("Step (x50)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Orthogonality Error
    ax = axes[1, 0]
    ax.plot(oblique_hist['ortho_error'], label='Oblique', color='blue')
    ax.plot(stiefel_hist['ortho_error'], label='Stiefel', color='green')
    ax.plot(muon_hist['ortho_error'], label='Muon (baseline)', color='red', linestyle='--')
    ax.set_title("Orthogonality Error ||W^T W - I||")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Cosine Similarity Histogram
    ax = axes[1, 1]
    
    def get_off_diag_values(gram):
        if gram is None: return []
        mask = ~np.eye(gram.shape[0], dtype=bool)
        return gram[mask].flatten()
    
    obl_vals = get_off_diag_values(oblique_hist['final_gram'])
    stf_vals = get_off_diag_values(stiefel_hist['final_gram'])
    muon_vals = get_off_diag_values(muon_hist['final_gram'])
    
    ax.hist(obl_vals, bins=50, alpha=0.5, label='Oblique', color='blue', density=True)
    ax.hist(stf_vals, bins=50, alpha=0.5, label='Stiefel', color='green', density=True)
    ax.hist(muon_vals, bins=50, alpha=0.5, label='Muon (baseline)', color='red', density=True, histtype='step')
    ax.set_title("Distribution of Column Correlations (Cosine Sim)")
    ax.set_xlabel("Cosine Similarity")
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'oblique_vs_stiefel_matrix_dynamics.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders()
    
    muon_res = train_network('muon', train_loader, test_loader, device)
    oblique_res = train_network('oblique', train_loader, test_loader, device)
    stiefel_res = train_network('stiefel', train_loader, test_loader, device)
    
    plot_comparison(oblique_res, stiefel_res, muon_res)

if __name__ == "__main__":
    main()
