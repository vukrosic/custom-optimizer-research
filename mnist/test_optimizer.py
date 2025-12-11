#!/usr/bin/env python
"""
Standalone optimizer test script for MNIST.

Run individual optimizers:
    python mnist/test_optimizer.py --optimizer adamw
    python mnist/test_optimizer.py --optimizer muon
    python mnist/test_optimizer.py --optimizer oblique
    python mnist/test_optimizer.py --optimizer grassmannian
    python mnist/test_optimizer.py --optimizer l1_stiefel
    python mnist/test_optimizer.py --optimizer all

Results saved to mnist/results/optimizer_comparison/
"""

import argparse
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data(batch_size=256):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size)


class MNISTNet(nn.Module):
    """Simple MLP for MNIST - compatible with all optimizers."""
    def __init__(self):
        super().__init__()
        # Using square-ish matrices for Grassmannian/Stiefel compatibility
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 256, bias=False)  # Square for Grassmannian
        self.fc3 = nn.Linear(256, 10, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_optimizer(model, name, device):
    """Create optimizer by name."""
    params = list(model.parameters())
    matrix_params = [p for p in params if p.ndim >= 2 and p.requires_grad]
    
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=1e-3)
    
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)
    
    elif name == 'muon':
        from optimizers.muon import Muon
        return Muon(matrix_params, lr=0.02)
    
    elif name == 'oblique':
        from optimizers.oblique import ObliqueOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return ObliqueOptimizer(params, base_opt, radius=1.0)
    
    elif name == 'grassmannian':
        from optimizers.grassmannian import GrassmannianOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return GrassmannianOptimizer(params, base_opt, nuclear_weight=0.0, retract_every=10)
    
    elif name == 'l1_stiefel':
        from optimizers.l1_stiefel import L1StiefelOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return L1StiefelOptimizer(params, base_opt, l1_weight=0.0001, ns_steps=3)
    
    elif name == 'sl_muon':
        from optimizers.sl_muon import SLMuonOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return SLMuonOptimizer(params, base_opt, ns_steps=3)
    
    elif name == 'symplectic':
        from optimizers.symplectic import SymplecticMuonOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return SymplecticMuonOptimizer(params, base_opt, cayley_steps=3)
    
    elif name == 'doubly_stochastic':
        from optimizers.doubly_stochastic import DoublyStochasticOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return DoublyStochasticOptimizer(params, base_opt, sinkhorn_iterations=10, temperature=1.0)
    
    elif name == 'block_stiefel':
        from optimizers.block_stiefel import BlockStiefelOptimizer
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        return BlockStiefelOptimizer(params, base_opt, num_heads=4, ns_steps=3)
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_epoch(model, optimizer, train_loader, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def test_optimizer(name, epochs=2, batch_size=256, verbose=True):
    """Test a single optimizer on MNIST."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Testing: {name.upper()}")
        print(f"Device: {device}")
        print(f"{'='*50}")
    
    train_loader, test_loader = get_data(batch_size)
    model = MNISTNet().to(device)
    
    try:
        optimizer = create_optimizer(model, name, device)
    except Exception as e:
        print(f"ERROR creating optimizer {name}: {e}")
        return {'name': name, 'error': str(e), 'success': False}
    
    results = {
        'name': name,
        'epochs': [],
        'success': True,
        'error': None
    }
    
    try:
        for epoch in range(epochs):
            loss = train_epoch(model, optimizer, train_loader, device)
            acc = evaluate(model, test_loader, device)
            results['epochs'].append({'epoch': epoch + 1, 'loss': loss, 'acc': acc})
            if verbose:
                print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.2f}%")
        
        results['final_loss'] = results['epochs'][-1]['loss']
        results['final_acc'] = results['epochs'][-1]['acc']
        
    except Exception as e:
        print(f"ERROR during training with {name}: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        results['success'] = False
    
    return results


def save_results(results, output_dir='mnist/results/optimizer_comparison'):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{results['name']}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Test optimizers on MNIST')
    parser.add_argument('--optimizer', '-o', type=str, default='all',
                        choices=['adamw', 'sgd', 'muon', 'oblique', 'grassmannian', 'l1_stiefel',
                                 'sl_muon', 'symplectic', 'doubly_stochastic', 'block_stiefel', 'all'],
                        help='Optimizer to test')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Batch size')
    args = parser.parse_args()
    
    if args.optimizer == 'all':
        optimizers = ['adamw', 'sgd', 'muon', 'oblique', 'grassmannian', 'l1_stiefel',
                      'sl_muon', 'symplectic', 'doubly_stochastic', 'block_stiefel']
    else:
        optimizers = [args.optimizer]
    
    all_results = {}
    
    for opt_name in optimizers:
        result = test_optimizer(opt_name, epochs=args.epochs, batch_size=args.batch_size)
        all_results[opt_name] = result
        save_results(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, result in sorted(all_results.items(), key=lambda x: -x[1].get('final_acc', 0)):
        if result['success']:
            print(f"{name:15s}: Loss={result['final_loss']:.4f}, Accuracy={result['final_acc']:.2f}%")
        else:
            print(f"{name:15s}: FAILED - {result['error']}")


if __name__ == '__main__':
    main()
