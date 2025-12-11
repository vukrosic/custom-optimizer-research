"""
Common MNIST models for experiments.
"""

import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """Simple MLP for MNIST with configurable hidden layers."""
    
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
        x = x.view(x.size(0), -1)
        return self.network(x)


class ComponentMNISTNet(nn.Module):
    """MLP for MNIST with named component groups (for per-component analysis)."""
    
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
        return {
            f'layers.{i}': i 
            for i in range(len(self.layers))
        }
