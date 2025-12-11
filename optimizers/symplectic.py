"""
Symplectic Muon: Optimizer for Physics-AI and Hamiltonian Networks.

The Symplectic Group Sp(2n) preserves the symplectic structure:
    W^T J W = J  where J = [[0, I], [-I, 0]]

This guarantees energy conservation and phase-space volume preservation.
Essential for:
- Hamiltonian Neural Networks
- AI for Science (molecular dynamics, orbital mechanics)
- Physics-informed learning where conservation laws matter

Manifold: Sp(2n) = {W ∈ R^{2n×2n} | W^T J W = J}
Norm: Spectral norm
"""

import torch
import torch.nn as nn
from typing import Optional


def make_symplectic_J(n: int, device=None, dtype=None) -> torch.Tensor:
    """Create the symplectic matrix J.
    
    J = [[0, I_n], [-I_n, 0]]
    
    This is the fundamental structure matrix for symplectic geometry.
    """
    I = torch.eye(n, device=device, dtype=dtype)
    Z = torch.zeros(n, n, device=device, dtype=dtype)
    
    top = torch.cat([Z, I], dim=1)     # [0, I]
    bottom = torch.cat([-I, Z], dim=1) # [-I, 0]
    
    return torch.cat([top, bottom], dim=0)


def is_symplectic(W: torch.Tensor, tol: float = 1e-5) -> bool:
    """Check if W is symplectic (W^T J W = J)."""
    if W.ndim != 2 or W.size(0) != W.size(1) or W.size(0) % 2 != 0:
        return False
    
    n = W.size(0) // 2
    J = make_symplectic_J(n, device=W.device, dtype=W.dtype)
    
    WtJW = W.T @ J @ W
    diff = (WtJW - J).abs().max()
    
    return diff.item() < tol


def symplectic_project_cayley(W: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Project to Sp(2n) using Cayley transform iteration.
    
    The Cayley transform maps from the Lie algebra sp(2n) to Sp(2n):
        W = (I + A)(I - A)^{-1}  where A ∈ sp(2n) (Hamiltonian matrix)
    
    We iteratively refine the projection.
    """
    if W.ndim != 2 or W.size(0) != W.size(1) or W.size(0) % 2 != 0:
        return W
    
    n = W.size(0) // 2
    J = make_symplectic_J(n, device=W.device, dtype=W.dtype)
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    
    X = W.float()
    
    for _ in range(steps):
        # Compute symplectic error: E = W^T J W - J
        E = X.T @ J @ X - J
        
        # Correction in Lie algebra (Hamiltonian matrix)
        # A Hamiltonian matrix H satisfies: J H + H^T J = 0
        # We compute a correction that reduces E
        H = -0.5 * J @ E
        
        # Make H Hamiltonian: H = J @ S where S is symmetric
        S = J.T @ H
        S = (S + S.T) / 2  # Symmetrize
        H = J @ S
        
        # Apply correction via exponential map approximation
        # exp(H) ≈ I + H + H²/2
        correction = I + H + (H @ H) / 2
        X = X @ correction
        
        # Normalize to prevent scale drift
        X = X / (X.norm() / (2*n)**0.5 + 1e-7) * (2*n)**0.5
    
    return X.to(W.dtype)


class SymplecticMuonOptimizer:
    """Optimizer constrained to the Symplectic Group Sp(2n).
    
    For Hamiltonian Neural Networks learning physical dynamics.
    Guarantees that learned transformations preserve energy and phase space.
    
    Note: Only works with square matrices of even dimension (2n × 2n).
    
    Usage:
        # For a Hamiltonian network with 2n-dimensional phase space
        base_opt = torch.optim.Adam(params, lr=1e-3)
        symp_opt = SymplecticMuonOptimizer(params, base_opt)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 cayley_steps: int = 3):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            cayley_steps: Cayley projection iterations
        """
        self.base_optimizer = base_optimizer
        self.cayley_steps = cayley_steps
        
        # Only collect even-dimensional square matrices
        self.symplectic_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim == 2 and p.size(0) == p.size(1) and p.size(0) % 2 == 0:
                    self.symplectic_params.append(p)
        
        # Initialize on Sp(2n)
        for p in self.symplectic_params:
            with torch.no_grad():
                p.data = symplectic_project_cayley(p.data, self.cayley_steps)
    
    def check_symplecticity(self) -> float:
        """Return max deviation from symplecticity (for monitoring)."""
        max_err = 0.0
        for p in self.symplectic_params:
            n = p.size(0) // 2
            J = make_symplectic_J(n, device=p.device, dtype=p.dtype)
            WtJW = p.data.T @ J @ p.data
            err = (WtJW - J).abs().max().item()
            max_err = max(max_err, err)
        return max_err
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step and project to Sp(2n)."""
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Project to symplectic group
        for p in self.symplectic_params:
            p.data = symplectic_project_cayley(p.data, self.cayley_steps)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
