"""
Model utilities for LLM experiments.
"""

import torch
import torch.nn as nn
from llm.configs.training_config import ModelConfig
from llm.models.model import GPTModel


def create_model(config=None, device='cuda'):
    """Create and return a GPT model.
    
    Args:
        config: ModelConfig instance (default: creates new one)
        device: Device to place model on
        
    Returns:
        GPTModel instance
    """
    if config is None:
        config = ModelConfig()
    
    model = GPTModel(config)
    model = model.to(device)
    
    print(f"✓ Created GPTModel with {model.num_parameters():,} parameters")
    print(f"  - {config.num_layers} layers, {config.num_heads} heads, {config.hidden_size} hidden size")
    
    return model


def get_2d_params(model):
    """Get all 2D parameters suitable for Muon optimization.
    
    Args:
        model: GPTModel instance
        
    Returns:
        List of 2D parameters
    """
    params_2d = []
    for name, param in model.named_parameters():
        if param.ndim == 2:
            params_2d.append(param)
    return params_2d


def get_param_groups(model):
    """Get parameter groups for modular optimization.
    
    Args:
        model: GPTModel instance
        
    Returns:
        Dictionary with 'embeddings', 'attention', 'ffn', 'output' parameter lists
    """
    groups = {
        'embeddings': [],
        'attention_qkv': [],
        'attention_out': [],
        'ffn': [],
        'output': [],
        'norms': [],
    }
    
    for name, param in model.named_parameters():
        if 'token_embedding' in name:
            groups['embeddings'].append(param)
        elif 'qkv' in name:
            groups['attention_qkv'].append(param)
        elif 'out_proj' in name:
            groups['attention_out'].append(param)
        elif 'w1' in name or 'w2' in name or 'w3' in name:
            groups['ffn'].append(param)
        elif 'lm_head' in name:
            groups['output'].append(param)
        elif 'norm' in name or 'weight' in name:
            groups['norms'].append(param)
    
    return groups


class ModularGPTModel:
    """Wrapper for GPT model that provides easy access to parameter groups."""
    
    def __init__(self, config=None, device='cuda'):
        """Initialize modular GPT model.
        
        Args:
            config: ModelConfig instance
            device: Device to place model on
        """
        self.model = create_model(config, device)
        self.config = config or ModelConfig()
        self.device = device
    
    def get_embedding_params(self):
        """Get embedding layer parameters."""
        return [p for n, p in self.model.named_parameters() if 'token_embedding' in n]
    
    def get_attention_params(self):
        """Get attention layer parameters."""
        return [p for n, p in self.model.named_parameters() 
                if 'qkv' in n or 'out_proj' in n]
    
    def get_ffn_params(self):
        """Get FFN layer parameters."""
        return [p for n, p in self.model.named_parameters() 
                if 'w1' in n or 'w2' in n or 'w3' in n]
    
    def get_output_params(self):
        """Get output layer parameters."""
        return [p for n, p in self.model.named_parameters() if 'lm_head' in n]
    
    def get_transformer_params(self):
        """Get all transformer block parameters (excluding embeddings and output)."""
        return [p for n, p in self.model.named_parameters() 
                if 'layers' in n]
    
    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)
    
    def parameters(self):
        """Get all parameters."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get all named parameters."""
        return self.model.named_parameters()
    
    def train(self):
        """Set to training mode."""
        self.model.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.model.eval()
    
    def to(self, device):
        """Move to device."""
        self.model = self.model.to(device)
        self.device = device
        return self
