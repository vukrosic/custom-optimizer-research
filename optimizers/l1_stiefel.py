"""
L1-Stiefel Descent: Sparse Orthogonal Optimizer.

Combines Stiefel manifold constraint (orthonormal columns) with L1 norm
on the updates to encourage sparse weight changes.

Useful for:
- Interpretability (only a few weights change per step)
- Communication-efficient distributed training
- Finding sparse orthogonal solutions

Manifold: Stiefel (W^T W = I)
Norm: L1 (Manhattan distance) on tangent vectors
"""

import torch
import torch.nn as nn
from typing import Optional


def stiefel_project_newton_schulz(W: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Project to Stiefel manifold using Newton-Schulz iteration."""
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
    
    return X.to(W.dtype)


def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft thresholding (proximal operator for L1 norm).
    
    This is the key operation that encourages sparsity.
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def project_to_stiefel_tangent(W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Project V to tangent space of Stiefel manifold at W.
    
    The tangent space at W is: T_W = {V | W^T V + V^T W = 0}
    
    Projection: V_tangent = V - W @ sym(W^T @ V)
    where sym(A) = (A + A^T) / 2
    """
    if W.ndim != 2 or V.ndim != 2:
        return V
    
    WtV = W.T @ V
    sym = (WtV + WtV.T) / 2
    return V - W @ sym


class L1StiefelOptimizer:
    """Optimizer with Stiefel constraint and L1-sparse updates.
    
    After computing the gradient, we:
    1. Project gradient to Stiefel tangent space
    2. Apply soft thresholding (L1 proximal operator)
    3. Take the step
    4. Retract back to Stiefel manifold
    
    This encourages sparse updates while maintaining orthonormality.
    
    Usage:
        base_opt = torch.optim.SGD(params, lr=0.01)
        l1_opt = L1StiefelOptimizer(params, base_opt, l1_weight=0.001)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 l1_weight: float = 0.001,
                 ns_steps: int = 5):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            l1_weight: Weight for L1 regularization (threshold for soft thresholding)
            ns_steps: Newton-Schulz iterations for retraction
        """
        self.base_optimizer = base_optimizer
        self.l1_weight = l1_weight
        self.ns_steps = ns_steps
        
        # Collect 2D parameters
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # Initialize on Stiefel manifold
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = stiefel_project_newton_schulz(p.data, self.ns_steps)
    
    def sparsify_gradients(self):
        """Apply L1 sparsification to gradients in tangent space."""
        for p in self.matrix_params:
            if p.grad is None:
                continue
            
            # Project gradient to tangent space
            grad_tangent = project_to_stiefel_tangent(p.data, p.grad)
            
            # Apply soft thresholding
            sparse_grad = soft_threshold(grad_tangent, self.l1_weight)
            
            # Replace gradient
            p.grad.data = sparse_grad
    
    def count_nonzero_grad_fraction(self) -> float:
        """Count fraction of non-zero gradient elements (for monitoring sparsity)."""
        total = 0
        nonzero = 0
        for p in self.matrix_params:
            if p.grad is not None:
                total += p.grad.numel()
                nonzero += (p.grad.abs() > 1e-8).sum().item()
        return nonzero / total if total > 0 else 0.0
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take sparse step on Stiefel manifold."""
        # Sparsify gradients
        self.sparsify_gradients()
        
        # Take base optimizer step
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Retract to Stiefel manifold
        for p in self.matrix_params:
            p.data = stiefel_project_newton_schulz(p.data, self.ns_steps)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
