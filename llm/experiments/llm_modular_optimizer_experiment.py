"""
LLM Modular Optimizer Experiment: Different Optimizers for Different Model Parts

This experiment implements the "modular manifolds" concept from the research:
- First layer (input projection): Muon with Sphere constraint (hyperspherical embeddings)
- Hidden layers: Muon with optional Stiefel manifold constraint  
- Output layer: Standard AdamW

Key idea: Different neural network components benefit from different geometric constraints
and optimization algorithms.

Configurations tested:
1. baseline_adamw: All AdamW (control)
2. baseline_muon: Muon for 2D params, AdamW for biases (standard Muon setup)
3. modular_sphere: First layer with sphere constraint, rest with Muon
4. modular_stiefel: Hidden layers with Stiefel constraint, output with AdamW
5. full_modular: All components with their optimal constraints
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from optimizers.muon import Muon
from optimizers.manifold_constraints import (
    SphereOptimizer, 
    StiefelOptimizer,
    sphere_project,
    stiefel_project_newton_schulz
)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModularLLMNet(nn.Module):
    """MLP for LLM with named layers for modular optimization."""
    
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super().__init__()
        
        # First layer (input projection) - candidates for sphere constraint
        self.input_layer = nn.Linear(784, hidden_sizes[0])
        self.input_act = nn.ReLU()
        
        # Hidden layers - candidates for Muon/Stiefel
        self.hidden_layers = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidden_acts.append(nn.ReLU())
        
        # Output layer - usually AdamW
        self.output_layer = nn.Linear(hidden_sizes[-1], 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.input_act(self.input_layer(x))
        
        for layer, act in zip(self.hidden_layers, self.hidden_acts):
            x = act(layer(x))
        
        return self.output_layer(x)
    
    def get_input_params(self):
        """Get input layer parameters."""
        return list(self.input_layer.parameters())
    
    def get_hidden_params(self):
        """Get hidden layer parameters."""
        params = []
        for layer in self.hidden_layers:
            params.extend(list(layer.parameters()))
        return params
    
    def get_output_params(self):
        """Get output layer parameters."""
        return list(self.output_layer.parameters())


# Using create_dataloaders from llm.common


def effective_rank(matrix):
    """Compute effective rank via entropy of normalized singular values."""
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return 0.0
    S = torch.linalg.svdvals(matrix.float())
    S = S / (S.sum() + 1e-10)
    entropy = -(S * torch.log(S + 1e-10)).sum()
    return torch.exp(entropy).item()


def compute_layer_metrics(model):
    """Compute metrics for each layer type."""
    metrics = {
        'input_rank': 0.0,
        'hidden_rank': 0.0,
        'output_rank': 0.0
    }
    
    # Input layer
    if model.input_layer.weight.grad is not None:
        metrics['input_rank'] = effective_rank(model.input_layer.weight.grad)
    
    # Hidden layers
    hidden_ranks = []
    for layer in model.hidden_layers:
        if layer.weight.grad is not None:
            hidden_ranks.append(effective_rank(layer.weight.grad))
    if hidden_ranks:
        metrics['hidden_rank'] = np.mean(hidden_ranks)
    
    # Output layer
    if model.output_layer.weight.grad is not None:
        metrics['output_rank'] = effective_rank(model.output_layer.weight.grad)
    
    return metrics


class ModularOptimizerWrapper:
    """Wrapper to manage multiple optimizers for different model parts."""
    
    def __init__(self, optimizers_dict, constraints_dict=None):
        """
        Args:
            optimizers_dict: Dict of {name: optimizer}
            constraints_dict: Dict of {name: constraint_fn} for post-step projection
        """
        self.optimizers = optimizers_dict
        self.constraints = constraints_dict or {}
    
    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def step(self):
        for name, opt in self.optimizers.items():
            opt.step()
            
            # Apply constraints if any
            if name in self.constraints:
                self.constraints[name]()


def create_modular_optimizer(model, config):
    """Create optimizer setup based on configuration."""
    
    config_name = config['name']
    
    if config_name == 'baseline_adamw':
        # All AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer, None
    
    elif config_name == 'baseline_muon':
        # Standard Muon setup: Muon for 2D, AdamW for 1D
        muon_params = []
        adam_params = []
        
        for p in model.parameters():
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.01)
        muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
        
        return ModularOptimizerWrapper({
            'adam': adam_opt,
            'muon': muon_opt
        }), None
    
    elif config_name == 'modular_sphere':
        # Input layer: AdamW + Sphere constraint
        # Hidden + Output: Muon
        
        input_weight = model.input_layer.weight
        input_bias = model.input_layer.bias
        
        hidden_muon_params = []
        hidden_adam_params = []
        for p in list(model.get_hidden_params()) + list(model.get_output_params()):
            if p.ndim == 2:
                hidden_muon_params.append(p)
            else:
                hidden_adam_params.append(p)
        
        input_adam = torch.optim.AdamW([input_weight, input_bias], lr=1e-3, weight_decay=0.01)
        hidden_adam = torch.optim.AdamW(hidden_adam_params, lr=1e-3, weight_decay=0.01)
        hidden_muon = Muon(hidden_muon_params, lr=0.02, momentum=0.95)
        
        # Sphere constraint for input layer
        def sphere_constraint():
            with torch.no_grad():
                # Project each row of input weights to unit sphere
                input_weight.data = sphere_project(input_weight.data, radius=1.0)
        
        # Initialize on sphere
        with torch.no_grad():
            input_weight.data = sphere_project(input_weight.data, radius=1.0)
        
        return ModularOptimizerWrapper(
            {'input_adam': input_adam, 'hidden_adam': hidden_adam, 'hidden_muon': hidden_muon},
            {'input_adam': sphere_constraint}
        ), 'sphere'
    
    elif config_name == 'modular_stiefel':
        # Input: Standard Muon
        # Hidden: Muon + Stiefel constraint
        # Output: AdamW
        
        input_params_2d = [p for p in model.get_input_params() if p.ndim == 2]
        input_params_1d = [p for p in model.get_input_params() if p.ndim == 1]
        
        hidden_params_2d = [p for p in model.get_hidden_params() if p.ndim == 2]
        hidden_params_1d = [p for p in model.get_hidden_params() if p.ndim == 1]
        
        output_params = model.get_output_params()
        
        input_muon = Muon(input_params_2d, lr=0.02, momentum=0.95) if input_params_2d else None
        input_adam = torch.optim.AdamW(input_params_1d, lr=1e-3, weight_decay=0.01) if input_params_1d else None
        
        hidden_muon = Muon(hidden_params_2d, lr=0.02, momentum=0.95) if hidden_params_2d else None
        hidden_adam = torch.optim.AdamW(hidden_params_1d, lr=1e-3, weight_decay=0.01) if hidden_params_1d else None
        
        output_adam = torch.optim.AdamW(output_params, lr=1e-3, weight_decay=0.01)
        
        # Stiefel constraint for hidden layers
        def stiefel_constraint():
            with torch.no_grad():
                for p in hidden_params_2d:
                    p.data = stiefel_project_newton_schulz(p.data, steps=5)
        
        # Initialize hidden weights on Stiefel manifold
        with torch.no_grad():
            for p in hidden_params_2d:
                p.data = stiefel_project_newton_schulz(p.data, steps=5)
        
        optimizers = {}
        if input_muon: optimizers['input_muon'] = input_muon
        if input_adam: optimizers['input_adam'] = input_adam
        if hidden_muon: optimizers['hidden_muon'] = hidden_muon
        if hidden_adam: optimizers['hidden_adam'] = hidden_adam
        optimizers['output_adam'] = output_adam
        
        return ModularOptimizerWrapper(
            optimizers,
            {'hidden_muon': stiefel_constraint} if hidden_muon else {}
        ), 'stiefel'
    
    elif config_name == 'full_modular':
        # Input: AdamW + Sphere constraint (hyperspherical embeddings)
        # Hidden: Muon + Stiefel constraint
        # Output: AdamW (unconstrained for final classification)
        
        input_weight = model.input_layer.weight
        input_bias = model.input_layer.bias
        
        hidden_params_2d = [p for p in model.get_hidden_params() if p.ndim == 2]
        hidden_params_1d = [p for p in model.get_hidden_params() if p.ndim == 1]
        
        output_params = model.get_output_params()
        
        input_adam = torch.optim.AdamW([input_weight, input_bias], lr=1e-3, weight_decay=0.01)
        hidden_muon = Muon(hidden_params_2d, lr=0.02, momentum=0.95) if hidden_params_2d else None
        hidden_adam = torch.optim.AdamW(hidden_params_1d, lr=1e-3, weight_decay=0.01) if hidden_params_1d else None
        output_adam = torch.optim.AdamW(output_params, lr=1e-3, weight_decay=0.01)
        
        def apply_all_constraints():
            with torch.no_grad():
                # Sphere for input
                input_weight.data = sphere_project(input_weight.data, radius=1.0)
                # Stiefel for hidden
                for p in hidden_params_2d:
                    p.data = stiefel_project_newton_schulz(p.data, steps=5)
        
        # Initialize
        with torch.no_grad():
            input_weight.data = sphere_project(input_weight.data, radius=1.0)
            for p in hidden_params_2d:
                p.data = stiefel_project_newton_schulz(p.data, steps=5)
        
        optimizers = {'input_adam': input_adam, 'output_adam': output_adam}
        if hidden_muon: optimizers['hidden_muon'] = hidden_muon
        if hidden_adam: optimizers['hidden_adam'] = hidden_adam
        
        # Apply constraints after hidden_muon step (main optimizer for hidden layers)
        constraint_key = 'hidden_muon' if hidden_muon else 'input_adam'
        
        return ModularOptimizerWrapper(
            optimizers,
            {constraint_key: apply_all_constraints}
        ), 'full'
    
    else:
        raise ValueError(f"Unknown config: {config_name}")


def train_epoch(model, optimizer, train_loader, device, track_metrics=True):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    layer_metrics_list = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Track layer-wise metrics
        if track_metrics and batch_idx % 50 == 0:
            metrics = compute_layer_metrics(model)
            layer_metrics_list.append(metrics)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if hasattr(optimizer, 'step'):
            optimizer.step()
        
        epoch_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    # Average layer metrics
    avg_layer_metrics = {}
    if layer_metrics_list:
        for key in layer_metrics_list[0].keys():
            avg_layer_metrics[key] = np.mean([m[key] for m in layer_metrics_list])
    
    return {
        'loss': epoch_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        **avg_layer_metrics
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
    
    return {
        'loss': test_loss / len(test_loader.dataset),
        'accuracy': 100. * correct / len(test_loader.dataset)
    }


def train_model(config, train_loader, test_loader, device, n_epochs=10):
    """Train a model with given configuration."""
    
    set_seed(42)
    
    model = ModularLLMNet(hidden_sizes=[512, 256, 128]).to(device)
    optimizer, constraint_type = create_modular_optimizer(model, config)
    
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    if constraint_type:
        print(f"Constraint type: {constraint_type}")
    print(f"{'='*60}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'input_rank': [], 'hidden_rank': [], 'output_rank': []
    }
    
    for epoch in range(n_epochs):
        train_metrics = train_epoch(model, optimizer, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['input_rank'].append(train_metrics.get('input_rank', 0))
        history['hidden_rank'].append(train_metrics.get('hidden_rank', 0))
        history['output_rank'].append(train_metrics.get('output_rank', 0))
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train: {train_metrics['accuracy']:.2f}% | "
              f"Test: {test_metrics['accuracy']:.2f}% | "
              f"Ranks: I={train_metrics.get('input_rank', 0):.1f} "
              f"H={train_metrics.get('hidden_rank', 0):.1f} "
              f"O={train_metrics.get('output_rank', 0):.1f}")
    
    return history


def plot_results(results, save_prefix='llm_modular'):
    """Plot comparison of all configurations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {
        'baseline_adamw': '#1f77b4',
        'baseline_muon': '#ff7f0e',
        'modular_sphere': '#2ca02c',
        'modular_stiefel': '#d62728',
        'full_modular': '#9467bd'
    }
    
    labels = {
        'baseline_adamw': 'AdamW (baseline)',
        'baseline_muon': 'Muon (baseline)',
        'modular_sphere': 'Modular (Sphere input)',
        'modular_stiefel': 'Modular (Stiefel hidden)',
        'full_modular': 'Full Modular'
    }
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=labels.get(name, name), 
                color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=labels.get(name, name), 
                color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Input Layer Rank
    ax = axes[0, 2]
    for name, history in results.items():
        if any(history['input_rank']):
            ax.plot(history['input_rank'], label=labels.get(name, name), 
                    color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Input Layer Gradient Rank')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Hidden Layer Rank
    ax = axes[1, 0]
    for name, history in results.items():
        if any(history['hidden_rank']):
            ax.plot(history['hidden_rank'], label=labels.get(name, name), 
                    color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Hidden Layer Gradient Rank')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Output Layer Rank
    ax = axes[1, 1]
    for name, history in results.items():
        if any(history['output_rank']):
            ax.plot(history['output_rank'], label=labels.get(name, name), 
                    color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Output Layer Gradient Rank')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final Accuracy Bar Chart
    ax = axes[1, 2]
    names = list(results.keys())
    final_accs = [results[n]['test_acc'][-1] for n in names]
    bars = ax.barh([labels.get(n, n) for n in names], final_accs, 
                   color=[colors.get(n, 'gray') for n in names])
    ax.set_xlabel('Final Test Accuracy (%)')
    ax.set_title('Final Accuracy Comparison')
    ax.set_xlim(min(final_accs) - 1, 100)
    
    # Add value labels
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontsize=9)
    
    plt.suptitle('LLM Modular Optimizer Experiment:\nDifferent Optimizers for Different Model Parts', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_prefix}_comparison.png")
    plt.close()


def print_summary(results):
    """Print summary table."""
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY: LLM Modular Optimizer")
    print("Different optimizers and constraints for different model parts")
    print("="*80)
    
    print("\n📊 Final Results:")
    print("-"*80)
    print(f"{'Configuration':<25} | {'Train Loss':>12} | {'Test Acc':>10} | {'Input Rank':>11} | {'Hidden Rank':>12}")
    print("-"*80)
    
    for name, history in results.items():
        train_loss = history['train_loss'][-1]
        test_acc = history['test_acc'][-1]
        input_rank = history['input_rank'][-1] if history['input_rank'][-1] else 'N/A'
        hidden_rank = history['hidden_rank'][-1] if history['hidden_rank'][-1] else 'N/A'
        
        input_str = f"{input_rank:.1f}" if isinstance(input_rank, float) else input_rank
        hidden_str = f"{hidden_rank:.1f}" if isinstance(hidden_rank, float) else hidden_rank
        
        print(f"{name:<25} | {train_loss:>12.4f} | {test_acc:>9.2f}% | {input_str:>11} | {hidden_str:>12}")
    
    print("-"*80)
    
    # Best configuration
    best = max(results.items(), key=lambda x: x[1]['test_acc'][-1])
    print(f"\n🏆 Best Configuration: {best[0]} ({best[1]['test_acc'][-1]:.2f}% test accuracy)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='LLM Modular Optimizer Experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--configs', nargs='+', default=None, 
                        help='Specific configs to run (default: all)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    print("📦 Loading LLM dataset...")
    train_loader, test_loader = get_llm_loaders(batch_size=args.batch_size)
    print(f"✓ Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # All configurations
    all_configs = [
        {'name': 'baseline_adamw'},
        {'name': 'baseline_muon'},
        {'name': 'modular_sphere'},
        {'name': 'modular_stiefel'},
        {'name': 'full_modular'},
    ]
    
    # Filter if specific configs requested
    if args.configs:
        all_configs = [c for c in all_configs if c['name'] in args.configs]
    
    results = {}
    
    for config in all_configs:
        history = train_model(config, train_loader, test_loader, device, n_epochs=args.epochs)
        results[config['name']] = history
    
    # Plotting
    print("\n📊 Generating plots...")
    plot_results(results, save_prefix='llm_modular')
    
    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
