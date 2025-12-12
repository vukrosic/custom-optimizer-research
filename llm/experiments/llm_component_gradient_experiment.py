"""
Per-Component Gradient Analysis Experiment

Analyze how different network components have different gradient characteristics
and may benefit from different optimizers. Based on modular-manifolds concept:

- Embeddings: Norm stability needed → Sphere constraint candidate
- Attention QKV: Structured gradients → Muon/Stiefel candidate
- FFN layers: Near full rank → Standard Muon
- Output/LM Head: Classification → AdamW

Works with both LLM MLP (layer types) and LLM (component types).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization."""
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
    """Compute effective rank via entropy."""
    S = torch.linalg.svdvals(matrix.float())
    S = S / (S.sum() + 1e-10)
    entropy = -(S * torch.log(S + 1e-10)).sum()
    return torch.exp(entropy).item()


def compute_component_metrics(grad, ns_steps=5):
    """Compute comprehensive metrics for a gradient matrix."""
    if grad.ndim != 2 or min(grad.shape) < 2:
        return None
    
    try:
        grad_float = grad.float()
        
        # Basic metrics
        S = torch.linalg.svdvals(grad_float)
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()
        
        # After NS
        grad_ns = zeropower_via_newtonschulz(grad_float.unsqueeze(0), steps=ns_steps).squeeze(0)
        S_ns = torch.linalg.svdvals(grad_ns)
        S_ns_norm = S_ns / (S_ns.sum() + 1e-10)
        entropy_ns = -(S_ns_norm * torch.log(S_ns_norm + 1e-10)).sum()
        eff_rank_ns = torch.exp(entropy_ns).item()
        
        # NS benefit (rank increase ratio)
        ns_benefit = eff_rank_ns / (eff_rank + 1e-10)
        
        return {
            'effective_rank': eff_rank,
            'effective_rank_ns': eff_rank_ns,
            'ns_benefit_ratio': ns_benefit,
            'max_rank': min(grad.shape),
            'rank_utilization': eff_rank / min(grad.shape),
            'condition_number': (S[0] / (S[-1] + 1e-10)).item(),
            'frobenius_norm': grad_float.norm().item(),
            'spectral_norm': S[0].item(),
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None


class ComponentLLMNet(nn.Module):
    """MLP for LLM with named component groups."""
    
    def __init__(self):
        super().__init__()
        
        # Component 1: Input projection (like embeddings)
        self.input_proj = nn.Linear(784, 512)
        
        # Component 2: Hidden layers (like attention/FFN)
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 128)
        
        # Component 3: Output (like LM head)
        self.output = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.input_proj(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output(x)
    
    def get_component_params(self):
        """Return params grouped by component type."""
        return {
            'input': [self.input_proj.weight],
            'hidden': [self.hidden1.weight, self.hidden2.weight],
            'output': [self.output.weight],
        }


def categorize_layer(name):
    """Categorize layer by name for LLM-style models."""
    name_lower = name.lower()
    
    if 'embed' in name_lower:
        return 'embedding'
    elif 'qkv' in name_lower or 'query' in name_lower or 'key' in name_lower or 'value' in name_lower:
        return 'attention_qkv'
    elif 'out_proj' in name_lower or 'o_proj' in name_lower:
        return 'attention_out'
    elif 'w1' in name_lower or 'w2' in name_lower or 'w3' in name_lower or 'ffn' in name_lower or 'mlp' in name_lower:
        return 'ffn'
    elif 'lm_head' in name_lower or 'output' in name_lower:
        return 'output'
    elif 'input' in name_lower:
        return 'input'
    elif 'hidden' in name_lower:
        return 'hidden'
    else:
        return 'other'


def get_llm_loaders(batch_size=128):
    """Load LLM dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.LLM('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader


def run_component_analysis(n_epochs=5, batch_size=128, track_interval=50):
    """Run per-component gradient analysis."""
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    train_loader = get_llm_loaders(batch_size=batch_size)
    
    # Create model
    model = ComponentLLMNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n" + "="*60)
    print("Per-Component Gradient Analysis")
    print("="*60)
    
    # Track metrics by component
    history = {
        'steps': [],
        'epochs': [],
        'component_metrics': [],  # List of {component_type: metrics}
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
            
            # Analyze per-component
            if step % track_interval == 0:
                component_metrics = {}
                
                for name, param in model.named_parameters():
                    if param.grad is None or param.grad.ndim != 2:
                        continue
                    
                    component_type = categorize_layer(name)
                    metrics = compute_component_metrics(param.grad.detach())
                    
                    if metrics is not None:
                        if component_type not in component_metrics:
                            component_metrics[component_type] = []
                        component_metrics[component_type].append(metrics)
                
                # Average metrics per component
                avg_metrics = {}
                for comp_type, metrics_list in component_metrics.items():
                    avg_metrics[comp_type] = {
                        key: np.mean([m[key] for m in metrics_list])
                        for key in metrics_list[0].keys()
                    }
                
                history['steps'].append(step)
                history['epochs'].append(epoch)
                history['component_metrics'].append(avg_metrics)
                
                # Print summary
                print(f"Step {step:4d} | ", end="")
                for comp_type in sorted(avg_metrics.keys()):
                    m = avg_metrics[comp_type]
                    print(f"{comp_type}: R={m['effective_rank']:.1f} NS↑={m['ns_benefit_ratio']:.2f} | ", end="")
                print()
            
            optimizer.step()
            step += 1
        
        print(f"Epoch {epoch+1}/{n_epochs} completed")
    
    return history


def plot_component_analysis(history, save_prefix='component_analysis'):
    """Plot per-component gradient analysis."""
    
    if not history['component_metrics']:
        print("No data to plot")
        return
    
    # Get component types
    component_types = list(history['component_metrics'][0].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(component_types)))
    color_map = {ct: colors[i] for i, ct in enumerate(sorted(component_types))}
    
    # Plot 1: Effective Rank by Component
    ax = axes[0, 0]
    for comp_type in sorted(component_types):
        ranks = [m[comp_type]['effective_rank'] for m in history['component_metrics'] if comp_type in m]
        steps = history['steps'][:len(ranks)]
        ax.plot(steps, ranks, label=comp_type, color=color_map[comp_type], linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank by Component Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NS Benefit Ratio by Component
    ax = axes[0, 1]
    for comp_type in sorted(component_types):
        ns_benefit = [m[comp_type]['ns_benefit_ratio'] for m in history['component_metrics'] if comp_type in m]
        steps = history['steps'][:len(ns_benefit)]
        ax.plot(steps, ns_benefit, label=comp_type, color=color_map[comp_type], linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('NS Benefit (Rank After / Rank Before)')
    ax.set_title('Newton-Schulz Benefit by Component')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Rank Utilization (% of max possible rank)
    ax = axes[1, 0]
    for comp_type in sorted(component_types):
        util = [m[comp_type]['rank_utilization'] * 100 for m in history['component_metrics'] if comp_type in m]
        steps = history['steps'][:len(util)]
        ax.plot(steps, util, label=comp_type, color=color_map[comp_type], linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Rank Utilization (%)')
    ax.set_title('Rank Utilization by Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average metrics comparison (bar chart)
    ax = axes[1, 1]
    
    # Get final metrics
    final_metrics = history['component_metrics'][-1]
    
    comp_labels = sorted(final_metrics.keys())
    x = np.arange(len(comp_labels))
    width = 0.25
    
    rank_vals = [final_metrics[c]['effective_rank'] for c in comp_labels]
    ns_benefit_vals = [final_metrics[c]['ns_benefit_ratio'] for c in comp_labels]
    
    # Normalize for visualization
    rank_norm = np.array(rank_vals) / max(rank_vals)
    
    ax.bar(x - width/2, rank_norm, width, label='Norm. Rank', color='#1f77b4')
    ax.bar(x + width/2, ns_benefit_vals, width, label='NS Benefit', color='#ff7f0e')
    
    ax.set_xlabel('Component Type')
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics by Component')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Per-Component Gradient Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {save_prefix}.png")
    plt.close()


def print_recommendations(history):
    """Print optimizer recommendations based on analysis."""
    
    print("\n" + "="*70)
    print("OPTIMIZER RECOMMENDATIONS BY COMPONENT")
    print("="*70)
    
    if not history['component_metrics']:
        return
    
    final_metrics = history['component_metrics'][-1]
    
    print(f"\n{'Component':<15} | {'Rank':<8} | {'NS Benefit':<12} | {'Recommendation':<25}")
    print("-"*70)
    
    for comp_type in sorted(final_metrics.keys()):
        m = final_metrics[comp_type]
        rank = m['effective_rank']
        ns_benefit = m['ns_benefit_ratio']
        
        # Recommendation logic
        if ns_benefit > 3.0:
            recommendation = "Muon (high NS benefit)"
        elif ns_benefit > 1.5:
            recommendation = "Muon (moderate benefit)"
        elif m['rank_utilization'] < 0.3:
            recommendation = "Muon + Stiefel"
        else:
            recommendation = "AdamW (stable)"
        
        print(f"{comp_type:<15} | {rank:<8.1f} | {ns_benefit:<12.2f} | {recommendation:<25}")
    
    print("-"*70)
    print("\n📌 Key Insights:")
    print("  • Components with high NS benefit (>3x) benefit most from Muon")
    print("  • Low rank utilization suggests Stiefel constraint may help")
    print("  • Components already at high rank can use standard AdamW")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Per-Component Gradient Analysis')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--track_interval', type=int, default=50, help='Steps between tracking')
    args = parser.parse_args()
    
    history = run_component_analysis(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        track_interval=args.track_interval
    )
    
    plot_component_analysis(history)
    print_recommendations(history)


if __name__ == '__main__':
    main()
