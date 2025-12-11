"""
Newton-Schulz Transformation Analysis Experiment

Deep analysis of what Newton-Schulz orthogonalization does to gradients:
- Angular change between G and NS(G)
- Information preservation: ||G - NS(G)||_F / ||G||_F
- Effect of varying NS steps (1, 2, 3, 5, 10)
- Convergence to orthogonality

This helps understand the core mechanism of Muon optimizer.
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


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()

    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X


def effective_rank(matrix):
    """Compute effective rank via entropy of normalized singular values."""
    S = torch.linalg.svdvals(matrix.float())
    S = S / (S.sum() + 1e-10)
    entropy = -(S * torch.log(S + 1e-10)).sum()
    return torch.exp(entropy).item()


def analyze_ns_transformation(G, ns_steps_list=[1, 2, 3, 5, 10]):
    """Analyze the effect of Newton-Schulz on a gradient matrix."""
    
    results = {}
    G_float = G.float()
    
    # Original properties
    G_norm = G_float.norm().item()
    G_rank = effective_rank(G_float)
    
    # SVD of original
    U, S, Vh = torch.linalg.svd(G_float, full_matrices=False)
    
    results['original'] = {
        'frobenius_norm': G_norm,
        'effective_rank': G_rank,
        'singular_values': S.cpu().numpy(),
        'condition_number': (S[0] / (S[-1] + 1e-10)).item(),
    }
    
    for ns_steps in ns_steps_list:
        G_ns = zeropower_via_newtonschulz(G_float.unsqueeze(0), steps=ns_steps).squeeze(0)
        
        # Properties after NS
        G_ns_norm = G_ns.norm().item()
        G_ns_rank = effective_rank(G_ns)
        
        # SVD after NS
        U_ns, S_ns, Vh_ns = torch.linalg.svd(G_ns, full_matrices=False)
        
        # Angular change (cosine similarity of flattened matrices)
        cos_sim = (G_float.flatten() @ G_ns.flatten()) / (G_norm * G_ns_norm + 1e-10)
        angle_degrees = torch.acos(cos_sim.clamp(-1, 1)).item() * 180 / np.pi
        
        # Information loss (relative Frobenius distance)
        info_loss = (G_float - G_ns).norm().item() / (G_norm + 1e-10)
        
        # Orthogonality check: ||G_ns @ G_ns.T - I||
        if G_ns.size(0) <= G_ns.size(1):
            orth_error = (G_ns @ G_ns.T - torch.eye(G_ns.size(0), device=G_ns.device)).norm().item()
        else:
            orth_error = (G_ns.T @ G_ns - torch.eye(G_ns.size(1), device=G_ns.device)).norm().item()
        
        results[f'ns_{ns_steps}'] = {
            'frobenius_norm': G_ns_norm,
            'effective_rank': G_ns_rank,
            'singular_values': S_ns.cpu().numpy(),
            'condition_number': (S_ns[0] / (S_ns[-1] + 1e-10)).item(),
            'angle_degrees': angle_degrees,
            'relative_info_loss': info_loss,
            'orthogonality_error': orth_error,
        }
    
    return results


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader


def run_ns_analysis_experiment(n_epochs=5, batch_size=128, track_interval=100):
    """Run NS transformation analysis throughout training."""
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    train_loader = get_mnist_loaders(batch_size=batch_size)
    
    # Create model
    model = MNISTNet(hidden_sizes=[512, 256, 128]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n" + "="*60)
    print("Newton-Schulz Transformation Analysis")
    print("="*60)
    
    # Track NS effects over training
    ns_steps_to_test = [1, 2, 3, 5, 10]
    history = {
        'steps': [],
        'epochs': [],
        'layer_analyses': [],  # List of {layer_name: analysis_results}
    }
    
    step = 0
    
    for epoch in range(n_epochs):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Analyze NS transformation
            if step % track_interval == 0:
                layer_analysis = {}
                
                for name, param in model.named_parameters():
                    if param.grad is None or param.grad.ndim != 2:
                        continue
                    if min(param.grad.shape) < 4:
                        continue
                    
                    grad = param.grad.detach()
                    analysis = analyze_ns_transformation(grad, ns_steps_to_test)
                    layer_analysis[name] = analysis
                
                history['steps'].append(step)
                history['epochs'].append(epoch)
                history['layer_analyses'].append(layer_analysis)
                
                # Print quick summary
                if layer_analysis:
                    first_layer = list(layer_analysis.keys())[0]
                    ns5 = layer_analysis[first_layer]['ns_5']
                    print(f"Step {step:4d} | Angle(G,NS5): {ns5['angle_degrees']:.1f}° | "
                          f"Info loss: {ns5['relative_info_loss']:.3f} | "
                          f"Orth error: {ns5['orthogonality_error']:.4f}")
            
            optimizer.step()
            step += 1
        
        print(f"Epoch {epoch+1}/{n_epochs} completed")
    
    return history, ns_steps_to_test


def plot_ns_analysis(history, ns_steps_list, save_prefix='ns_analysis'):
    """Plot NS transformation analysis results."""
    
    if not history['layer_analyses']:
        print("No analysis data to plot")
        return
    
    layer_names = list(history['layer_analyses'][0].keys())
    first_layer = layer_names[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Angular change vs NS steps (at different training stages)
    ax = axes[0, 0]
    stages = [0, len(history['layer_analyses'])//2, -1]
    stage_labels = ['Early', 'Mid', 'Late']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for stage_idx, (stage, label, color) in enumerate(zip(stages, stage_labels, colors)):
        angles = []
        for ns_steps in ns_steps_list:
            angle = history['layer_analyses'][stage][first_layer][f'ns_{ns_steps}']['angle_degrees']
            angles.append(angle)
        ax.plot(ns_steps_list, angles, marker='o', label=f'{label} training', color=color, linewidth=2)
    
    ax.set_xlabel('Newton-Schulz Steps')
    ax.set_ylabel('Angular Change (degrees)')
    ax.set_title('Angular Change between G and NS(G)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Information loss vs NS steps
    ax = axes[0, 1]
    for stage_idx, (stage, label, color) in enumerate(zip(stages, stage_labels, colors)):
        info_loss = []
        for ns_steps in ns_steps_list:
            loss = history['layer_analyses'][stage][first_layer][f'ns_{ns_steps}']['relative_info_loss']
            info_loss.append(loss)
        ax.plot(ns_steps_list, info_loss, marker='s', label=f'{label} training', color=color, linewidth=2)
    
    ax.set_xlabel('Newton-Schulz Steps')
    ax.set_ylabel('Relative Information Loss')
    ax.set_title('||G - NS(G)||_F / ||G||_F')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Orthogonality error over training (for NS=5)
    ax = axes[1, 0]
    orth_errors = [history['layer_analyses'][i][first_layer]['ns_5']['orthogonality_error'] 
                   for i in range(len(history['layer_analyses']))]
    ax.plot(history['steps'], orth_errors, linewidth=2, color='#d62728')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Orthogonality Error')
    ax.set_title('||G_ns @ G_ns.T - I|| (NS steps=5)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Singular value distribution before/after NS
    ax = axes[1, 1]
    mid_stage = len(history['layer_analyses']) // 2
    analysis = history['layer_analyses'][mid_stage][first_layer]
    
    sv_original = analysis['original']['singular_values'][:20]  # Top 20
    sv_ns5 = analysis['ns_5']['singular_values'][:20]
    
    x = np.arange(len(sv_original))
    width = 0.35
    ax.bar(x - width/2, sv_original / sv_original.max(), width, label='Original G', alpha=0.8)
    ax.bar(x + width/2, sv_ns5 / sv_ns5.max(), width, label='After NS(5)', alpha=0.8)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized Singular Value')
    ax.set_title('Singular Value Distribution (Mid Training)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Newton-Schulz Transformation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {save_prefix}.png")
    plt.close()


def print_summary(history, ns_steps_list):
    """Print summary of NS analysis."""
    
    print("\n" + "="*70)
    print("NS TRANSFORMATION SUMMARY")
    print("="*70)
    
    if not history['layer_analyses']:
        return
    
    layer_names = list(history['layer_analyses'][0].keys())
    
    print(f"\nAnalyzed {len(layer_names)} layers over {len(history['steps'])} tracking points")
    print(f"NS steps tested: {ns_steps_list}")
    
    # Summary for first layer
    first_layer = layer_names[0]
    print(f"\n📊 Summary for {first_layer}:")
    
    # Early vs Late comparison
    early = history['layer_analyses'][0][first_layer]
    late = history['layer_analyses'][-1][first_layer]
    
    print(f"\n{'Metric':<30} | {'Early':<15} | {'Late':<15}")
    print("-"*65)
    
    for ns_steps in [3, 5]:
        key = f'ns_{ns_steps}'
        print(f"Angle (NS={ns_steps})" + " "*17 + f"| {early[key]['angle_degrees']:<15.1f} | {late[key]['angle_degrees']:<15.1f}")
        print(f"Info Loss (NS={ns_steps})" + " "*13 + f"| {early[key]['relative_info_loss']:<15.3f} | {late[key]['relative_info_loss']:<15.3f}")
        print(f"Orth Error (NS={ns_steps})" + " "*12 + f"| {early[key]['orthogonality_error']:<15.4f} | {late[key]['orthogonality_error']:<15.4f}")
        print()
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Newton-Schulz Transformation Analysis')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--track_interval', type=int, default=100, help='Steps between tracking')
    args = parser.parse_args()
    
    history, ns_steps_list = run_ns_analysis_experiment(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        track_interval=args.track_interval
    )
    
    plot_ns_analysis(history, ns_steps_list)
    print_summary(history, ns_steps_list)


if __name__ == '__main__':
    main()
