"""
SL-Muon: Special Linear Group Optimizer.

The Special Linear Group SL(n) is the set of matrices with determinant = 1.
This guarantees volume preservation - useful for:
- Normalizing Flows (Jacobian determinant = 1 by construction)
- Invertible neural networks
- Deep generative models

Manifold: SL(n) = {W ∈ R^{n×n} | det(W) = 1}
Norm: Spectral norm (like Muon)
"""

import torch
import torch.nn as nn
from typing import Optional


def sl_project(W: torch.Tensor) -> torch.Tensor:
    """Project a square matrix to SL(n) (det = 1).
    
    We scale the matrix so that its determinant becomes 1:
        W_sl = W / |det(W)|^{1/n}
    
    For numerical stability with large matrices, we use the log determinant.
    
    Args:
        W: Square matrix
        
    Returns:
        Matrix with determinant = 1
    """
    if W.ndim != 2 or W.size(0) != W.size(1):
        return W
    
    n = W.size(0)
    
    try:
        # Use slogdet for numerical stability
        sign, logabsdet = torch.linalg.slogdet(W.float())
        
        if sign.item() < 0:
            # If determinant is negative, flip sign of first column
            W = W.clone()
            W[:, 0] = -W[:, 0]
            sign, logabsdet = torch.linalg.slogdet(W.float())
        
        # Scale factor: |det|^{-1/n}
        scale = torch.exp(-logabsdet / n)
        return (W * scale).to(W.dtype)
    except:
        return W


def sl_newton_schulz(W: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Apply Newton-Schulz orthogonalization followed by SL projection.
    
    This gives us a matrix that is both near-orthogonal AND has det = 1.
    """
    if W.ndim != 2 or W.size(0) != W.size(1):
        return W
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W.float()
    
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    # Project to SL(n)
    X = sl_project(X)
    
    return X.to(W.dtype)


class SLMuonOptimizer:
    """Muon-style optimizer constrained to SL(n) manifold.
    
    Combines Muon's spectral normalization with volume preservation.
    Useful for invertible networks where we want det(W) = 1.
    
    Usage:
        base_opt = torch.optim.AdamW(params, lr=1e-3)
        sl_opt = SLMuonOptimizer(params, base_opt)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 ns_steps: int = 5):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            ns_steps: Newton-Schulz iterations
        """
        self.base_optimizer = base_optimizer
        self.ns_steps = ns_steps
        
        # Only collect square 2D parameters
        self.square_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim == 2 and p.size(0) == p.size(1):
                    self.square_params.append(p)
        
        # Initialize on SL(n)
        for p in self.square_params:
            with torch.no_grad():
                p.data = sl_project(p.data)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step and project to SL(n)."""
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Project to SL(n) with Newton-Schulz
        for p in self.square_params:
            p.data = sl_newton_schulz(p.data, self.ns_steps)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
