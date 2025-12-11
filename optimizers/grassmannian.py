"""
Low-Rank Grassmannian Descent Optimizer.

The Grassmannian manifold Gr(k, n) treats matrices as representations of subspaces.
Two matrices W and WR (where R is a rotation) are considered equivalent because
they span the same subspace.

This is "Manifold LoRA" - instead of optimizing a dense matrix, we optimize the
subspace directly. This allows the network to rotate its internal representation
freely without wasting optimization energy on redundant rotation updates.

Manifold: Gr(k, n) = {col(W) | W ∈ R^{n×k}, rank(W) = k}
Norm: Nuclear norm (sum of singular values) encourages low rank
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def grassmann_project(W: torch.Tensor) -> torch.Tensor:
    """Project to Grassmannian by orthonormalizing columns.
    
    The Grassmannian quotient is represented by an orthonormal basis.
    We use QR decomposition to get this canonical representative.
    
    Args:
        W: 2D tensor
        
    Returns:
        Orthonormalized tensor representing the same subspace (SAME SHAPE as input)
    """
    if W.ndim != 2:
        return W
    
    m, n = W.shape
    
    # Use QR decomposition to get orthonormal representative
    try:
        # For Grassmannian on columns, we want orthonormal columns
        # QR gives Q with orthonormal columns, but may change shape
        # We need to preserve the original shape
        if m >= n:
            # Tall/square: QR directly, take first n columns
            Q, R = torch.linalg.qr(W.float(), mode='reduced')
            # Ensure consistent sign (canonical form)
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1
            Q = Q * signs.unsqueeze(0)
            return Q.to(W.dtype)
        else:
            # Wide matrix: normalize rows instead to preserve shape
            # Project to unit norm rows
            norms = W.norm(dim=1, keepdim=True)
            return W / (norms + 1e-8)
    except:
        return W



def grassmann_log(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """Compute logarithmic map from W1 to W2 on Grassmannian.
    
    This gives the tangent vector at W1 pointing towards W2.
    """
    if W1.ndim != 2 or W2.ndim != 2:
        return W2 - W1
    
    # Project W2 - W1 to tangent space at W1
    W1_orth = grassmann_project(W1)
    diff = W2 - W1
    
    # Remove component in column space of W1
    proj = W1_orth @ (W1_orth.T @ diff)
    tangent = diff - proj
    
    return tangent


class GrassmannianOptimizer:
    """Optimizer on the Grassmannian manifold.
    
    Optimizes subspaces rather than specific matrices. Useful for:
    - Low-rank adaptation (LoRA-style)
    - Feature subspace learning
    - Avoiding redundant rotation optimization
    
    The optimizer takes steps in the tangent space and retracts back to manifold.
    Uses nuclear norm regularization to encourage low-rank solutions.
    
    Usage:
        base_opt = torch.optim.SGD(params, lr=0.01)
        grass_opt = GrassmannianOptimizer(params, base_opt, nuclear_weight=0.01)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, 
                 nuclear_weight: float = 0.0,
                 retract_every: int = 1):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            nuclear_weight: Weight for nuclear norm regularization (encourages low rank)
            retract_every: How often to project back to manifold (1 = every step)
        """
        self.base_optimizer = base_optimizer
        self.nuclear_weight = nuclear_weight
        self.retract_every = retract_every
        self.step_count = 0
        
        # Collect 2D parameters
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # Initialize on Grassmannian
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = grassmann_project(p.data)
    
    def add_nuclear_grad(self):
        """Add nuclear norm gradient to encourage low rank.
        
        ∂||W||_* / ∂W = U @ V^T  (from SVD: W = U @ S @ V^T)
        """
        if self.nuclear_weight <= 0:
            return
        
        for p in self.matrix_params:
            if p.grad is None:
                continue
            try:
                U, S, Vh = torch.linalg.svd(p.data.float(), full_matrices=False)
                nuclear_grad = U @ Vh
                p.grad.data += self.nuclear_weight * nuclear_grad.to(p.grad.dtype)
            except:
                pass
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimizer step on Grassmannian."""
        self.step_count += 1
        
        # Add nuclear norm regularization gradient
        self.add_nuclear_grad()
        
        # Project gradients to tangent space
        for p in self.matrix_params:
            if p.grad is None:
                continue
            # Project gradient: tangent = grad - W @ (W^T @ grad)
            proj = p.data @ (p.data.T @ p.grad)
            p.grad.data = p.grad.data - proj
        
        # Take base optimizer step
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Retract to manifold if needed
        if self.step_count % self.retract_every == 0:
            for p in self.matrix_params:
                p.data = grassmann_project(p.data)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
