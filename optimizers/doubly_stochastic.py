"""
Doubly Stochastic Optimizer for Permutation Learning.

The Birkhoff Polytope is the set of doubly stochastic matrices:
- All rows sum to 1
- All columns sum to 1  
- All entries are non-negative

This is the convex hull of permutation matrices, useful for:
- Graph matching / point cloud registration
- Hard attention mechanisms
- Learning discrete assignments in a differentiable way

Manifold: Birkhoff Polytope (approximate - relaxation of permutation matrices)
Projection: Sinkhorn algorithm
"""

import torch
import torch.nn as nn
from typing import Optional


def sinkhorn_normalize(M: torch.Tensor, iterations: int = 10, 
                       temperature: float = 1.0) -> torch.Tensor:
    """Project matrix to doubly stochastic using Sinkhorn algorithm.
    
    Alternates between row and column normalization until convergence.
    This is the standard way to project to the Birkhoff polytope.
    
    Args:
        M: 2D tensor (can have negative values - will be exponentiated)
        iterations: Number of Sinkhorn iterations
        temperature: Lower = closer to hard permutation
        
    Returns:
        Doubly stochastic matrix
    """
    if M.ndim != 2:
        return M
    
    # Ensure square for now (can be generalized)
    n = min(M.shape)
    M = M[:n, :n]
    
    # Apply exponential kernel (soft assignments)
    log_M = M.float() / temperature
    M_exp = torch.exp(log_M - log_M.max())  # Numerical stability
    
    # Sinkhorn iterations
    for _ in range(iterations):
        # Row normalization
        M_exp = M_exp / (M_exp.sum(dim=1, keepdim=True) + 1e-10)
        # Column normalization
        M_exp = M_exp / (M_exp.sum(dim=0, keepdim=True) + 1e-10)
    
    return M_exp.to(M.dtype)


def is_doubly_stochastic(M: torch.Tensor, tol: float = 1e-5) -> bool:
    """Check if M is doubly stochastic."""
    if M.ndim != 2:
        return False
    
    # Check non-negative
    if (M < -tol).any():
        return False
    
    # Check row sums
    row_sums = M.sum(dim=1)
    if (row_sums - 1).abs().max() > tol:
        return False
    
    # Check column sums
    col_sums = M.sum(dim=0)
    if (col_sums - 1).abs().max() > tol:
        return False
    
    return True


def hungarian_round(M: torch.Tensor) -> torch.Tensor:
    """Round doubly stochastic to nearest permutation (for inference).
    
    Uses greedy matching as approximation to Hungarian algorithm.
    """
    if M.ndim != 2:
        return M
    
    n = M.size(0)
    result = torch.zeros_like(M)
    
    # Greedy matching
    M_copy = M.clone()
    for _ in range(n):
        # Find maximum entry
        flat_idx = M_copy.argmax()
        i, j = flat_idx // n, flat_idx % n
        
        # Assign
        result[i, j] = 1.0
        
        # Remove row and column from consideration
        M_copy[i, :] = -float('inf')
        M_copy[:, j] = -float('inf')
    
    return result


class DoublyStochasticOptimizer:
    """Optimizer that constrains matrices to be doubly stochastic.
    
    After each optimization step, projects weights back to the Birkhoff
    polytope using the Sinkhorn algorithm.
    
    Useful for learning soft permutations / discrete assignments.
    
    Usage:
        base_opt = torch.optim.Adam(params, lr=1e-2)
        ds_opt = DoublyStochasticOptimizer(params, base_opt, temperature=0.1)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 sinkhorn_iterations: int = 10,
                 temperature: float = 1.0):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            sinkhorn_iterations: Sinkhorn normalization iterations
            temperature: Softmax temperature (lower = harder assignments)
        """
        self.base_optimizer = base_optimizer
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        
        # Collect square 2D parameters
        self.stochastic_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim == 2 and p.size(0) == p.size(1):
                    self.stochastic_params.append(p)
        
        # Initialize as doubly stochastic (uniform for now)
        for p in self.stochastic_params:
            with torch.no_grad():
                n = p.size(0)
                p.data = torch.ones_like(p.data) / n
    
    def anneal_temperature(self, factor: float = 0.99):
        """Gradually decrease temperature for harder assignments."""
        self.temperature = max(0.01, self.temperature * factor)
    
    def get_hard_assignments(self) -> dict:
        """Get hard (discrete) permutations from current soft assignments."""
        assignments = {}
        for i, p in enumerate(self.stochastic_params):
            assignments[f'param_{i}'] = hungarian_round(p.data)
        return assignments
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step and project to Birkhoff polytope."""
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Project to doubly stochastic
        for p in self.stochastic_params:
            p.data = sinkhorn_normalize(
                p.data, 
                iterations=self.sinkhorn_iterations,
                temperature=self.temperature
            )
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


def sinkhorn_distance(X: torch.Tensor, Y: torch.Tensor, 
                      cost_fn='l2', iterations: int = 100,
                      reg: float = 0.1) -> torch.Tensor:
    """Compute Sinkhorn distance between two point clouds.
    
    This is the entropy-regularized optimal transport distance.
    Can be used as a loss function for learning soft correspondences.
    
    Args:
        X: First point cloud (n x d)
        Y: Second point cloud (m x d)
        cost_fn: Cost function ('l2' or 'cosine')
        iterations: Sinkhorn iterations
        reg: Entropy regularization
        
    Returns:
        Sinkhorn distance (scalar)
    """
    if cost_fn == 'l2':
        C = torch.cdist(X, Y, p=2)
    elif cost_fn == 'cosine':
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        C = 1 - X_norm @ Y_norm.T
    else:
        raise ValueError(f"Unknown cost function: {cost_fn}")
    
    n, m = C.shape
    
    # Initialize
    u = torch.ones(n, device=C.device) / n
    v = torch.ones(m, device=C.device) / m
    
    K = torch.exp(-C / reg)
    
    # Sinkhorn iterations
    for _ in range(iterations):
        u = 1.0 / (K @ v + 1e-10) / n
        v = 1.0 / (K.T @ u + 1e-10) / m
    
    # Transport plan
    P = torch.diag(u) @ K @ torch.diag(v)
    
    # Distance
    return (P * C).sum()
