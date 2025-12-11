"""
Block-Diagonal Stiefel Optimizer for Multi-Head Attention.

Standard orthogonalization on a full d_model × d_model matrix destroys the
head structure by mixing information between heads that should be separate.

This optimizer enforces orthonormality WITHIN each attention head, but allows
heads to be correlated with each other. This matches the actual architecture
of Transformers.

Manifold: Product of Stiefel manifolds (block-diagonal structure)
Norm: Block-max spectral norm
"""

import torch
import torch.nn as nn
from typing import List, Optional


def stiefel_project_newton_schulz(W: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Project a matrix onto Stiefel manifold using Newton-Schulz iteration.
    
    From the Muon optimizer - makes all singular values equal to 1.
    """
    if W.ndim != 2:
        return W
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W.float()
    
    transposed = False
    if X.size(0) < X.size(1):
        X = X.T
        transposed = True
    
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.T
    
    return X


def block_stiefel_project(W: torch.Tensor, num_heads: int, ns_steps: int = 5) -> torch.Tensor:
    """Project to block-diagonal Stiefel manifold.
    
    Divides the matrix into num_heads blocks along the column dimension,
    and orthonormalizes each block independently.
    
    Args:
        W: Weight matrix (d_out x d_in)
        num_heads: Number of attention heads
        ns_steps: Newton-Schulz iterations per block
        
    Returns:
        Block-orthonormalized matrix
    """
    if W.ndim != 2:
        return W
    
    d_out, d_in = W.shape
    head_dim = d_in // num_heads
    
    if d_in % num_heads != 0:
        # Fall back to full Stiefel if not divisible
        return stiefel_project_newton_schulz(W, ns_steps)
    
    # Process each head block independently
    result = torch.zeros_like(W)
    
    for h in range(num_heads):
        start = h * head_dim
        end = (h + 1) * head_dim
        
        block = W[:, start:end]
        block_orth = stiefel_project_newton_schulz(block, ns_steps)
        result[:, start:end] = block_orth
    
    return result


class BlockStiefelOptimizer:
    """Optimizer with block-diagonal Stiefel constraint for multi-head attention.
    
    For QKV projection matrices in Transformers, we want:
    - Orthonormality WITHIN each head (decorrelated features per head)
    - Heads can be correlated with each other (preserves head specialization)
    
    This is the "Block-Diagonal Stiefel Muon" from the Modular Manifolds concept.
    
    Usage:
        # For a model with 8 attention heads
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        block_opt = BlockStiefelOptimizer(
            params, base_opt, 
            num_heads=8,
            attention_param_names=['q_proj', 'k_proj', 'v_proj']
        )
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 num_heads: int = 8,
                 attention_param_names: Optional[List[str]] = None,
                 ns_steps: int = 5):
        """
        Args:
            params: Model parameters
            base_optimizer: Underlying optimizer
            num_heads: Number of attention heads
            attention_param_names: Substrings to identify attention params
            ns_steps: Newton-Schulz iterations
        """
        self.base_optimizer = base_optimizer
        self.num_heads = num_heads
        self.ns_steps = ns_steps
        
        # Default attention parameter names
        if attention_param_names is None:
            attention_param_names = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'qkv']
        self.attention_param_names = attention_param_names
        
        # Collect attention 2D parameters
        self.attention_params = []
        self.other_params = []
        
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    # Check if this is an attention parameter
                    name = getattr(p, '_param_name', '')
                    is_attention = any(n in name.lower() for n in self.attention_param_names)
                    
                    if is_attention:
                        self.attention_params.append(p)
                    else:
                        self.other_params.append(p)
        
        # Initialize attention params on block-Stiefel manifold
        for p in self.attention_params:
            with torch.no_grad():
                p.data = block_stiefel_project(p.data, self.num_heads, self.ns_steps)
    
    def register_param_names(self, model: nn.Module):
        """Register parameter names for identification.
        
        Call this before training to enable name-based param identification.
        """
        for name, param in model.named_parameters():
            param._param_name = name
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step with block-Stiefel constraint."""
        # Take base optimizer step
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Project attention params back to block-Stiefel manifold
        for p in self.attention_params:
            p.data = block_stiefel_project(p.data, self.num_heads, self.ns_steps)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


class BlockMaxSpectralNorm:
    """Compute block-max spectral norm for multi-head matrices.
    
    Instead of the spectral norm of the full matrix, we take the maximum
    spectral norm across all head blocks. This better reflects the per-head
    dynamics in attention.
    """
    
    @staticmethod
    def compute(W: torch.Tensor, num_heads: int) -> float:
        """Compute max spectral norm across blocks."""
        if W.ndim != 2:
            return W.norm().item()
        
        d_out, d_in = W.shape
        head_dim = d_in // num_heads
        
        if d_in % num_heads != 0:
            return torch.linalg.svdvals(W.float())[0].item()
        
        max_spectral = 0.0
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = W[:, start:end]
            spectral = torch.linalg.svdvals(block.float())[0].item()
            max_spectral = max(max_spectral, spectral)
        
        return max_spectral
