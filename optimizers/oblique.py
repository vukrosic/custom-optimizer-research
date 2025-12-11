"""
Oblique Manifold Optimizer for Embeddings.

The Oblique manifold constrains each column of the matrix to be unit length,
but allows columns to be correlated (unlike Stiefel which also enforces orthogonality).

This is ideal for:
- Embedding tables where we want normalized vectors
- Attention Key/Query projections (stable dot products)
- Any case where we want to control norm but allow learned correlations

Manifold: Oblique(m, n) = {W ∈ R^{m×n} | ||w_i||_2 = 1 for all columns i}
"""

import torch
import torch.nn as nn
from typing import Optional


def oblique_project(W: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Project matrix columns onto the hypersphere (oblique manifold).
    
    Each column is normalized to have unit L2 norm (or specified radius).
    
    Args:
        W: 2D tensor (m x n)
        radius: Target norm for each column
        
    Returns:
        Projected tensor with unit-norm columns
    """
    if W.ndim != 2:
        return W
    
    # Normalize each column
    norms = W.norm(dim=0, keepdim=True)
    return W * (radius / (norms + 1e-8))


class ObliqueOptimizer:
    """Optimizer that constrains weight columns to unit norm (Oblique manifold).
    
    Unlike Stiefel (which requires orthonormal columns), Oblique only requires
    each column to have unit norm. Columns can be correlated with each other.
    
    This is useful for embeddings where we want:
    - Stable norms (no explosion/collapse)
    - Learned semantic similarities (correlations between embeddings)
    
    Usage:
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        oblique_opt = ObliqueOptimizer(params, base_opt, radius=1.0)
        
        loss.backward()
        oblique_opt.step()  # Projects to oblique manifold after step
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, radius: float = 1.0):
        """
        Args:
            params: Parameters to optimize (for reference)
            base_optimizer: Underlying optimizer (Adam, SGD, etc.)
            radius: Target column norm (default 1.0 for unit sphere)
        """
        self.base_optimizer = base_optimizer
        self.radius = radius
        
        # Collect 2D parameters
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # Initialize on oblique manifold
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = oblique_project(p.data, self.radius)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step and project back to oblique manifold."""
        # Take base optimizer step
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Project columns back to unit sphere
        for p in self.matrix_params:
            p.data = oblique_project(p.data, self.radius)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


class ObliqueConstraint(nn.Module):
    """Riemannian gradient projection for Oblique manifold (alternative approach).
    
    Instead of projecting weights after each step, this projects the gradient
    to be tangent to the manifold before the step.
    
    The tangent space at W is: T_W = {V | <V_i, W_i> = 0 for all columns i}
    
    This is more principled but requires modifying the backward pass.
    """
    
    @staticmethod
    def project_gradient(W: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient to tangent space of oblique manifold.
        
        For each column, remove the component parallel to W.
        """
        if W.ndim != 2 or grad.ndim != 2:
            return grad
        
        # For each column, remove radial component
        # tangent = grad - (grad · w) * w  for unit vector w
        dot_products = (grad * W).sum(dim=0, keepdim=True)
        norms_sq = (W * W).sum(dim=0, keepdim=True) + 1e-8
        
        return grad - (dot_products / norms_sq) * W
