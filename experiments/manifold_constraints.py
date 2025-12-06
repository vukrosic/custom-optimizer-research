"""
Manifold Constraints for Neural Network Parameters.

Implements the core manifold constraints from the Modular Manifolds article:
- Hypersphere: vectors constrained to unit norm (for embeddings)
- Stiefel manifold: W^T W = I (orthonormal columns, all singular values = 1)
- Spectral constraint: largest singular value = 1
"""

import torch
import torch.nn as nn
import math


def stiefel_project_newton_schulz(W: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Project a matrix onto the Stiefel manifold using Newton-Schulz iteration.
    
    The Stiefel manifold St(m,n) = {W ∈ R^{m×n} | W^T W = I_n}
    is the set of matrices with orthonormal columns (all singular values = 1).
    
    This is the same method used in Muon optimizer for gradient orthogonalization,
    but here we apply it to the weights themselves.
    """
    if W.ndim != 2:
        return W
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W.clone()
    
    # Handle tall vs wide matrices
    transposed = False
    if X.size(0) < X.size(1):
        X = X.T
        transposed = True
    
    # Normalize to prevent divergence
    X = X / (X.norm() + 1e-7)
    
    # Newton-Schulz iteration
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.T
    
    return X


def spectral_normalize(W: torch.Tensor, radius: float = 1.0, 
                       power_iterations: int = 3) -> torch.Tensor:
    """
    Scale matrix so largest singular value equals radius.
    Uses power iteration to efficiently estimate spectral norm.
    """
    if W.ndim != 2:
        return W
    
    # Power iteration for spectral norm
    u = torch.randn(W.size(0), device=W.device, dtype=W.dtype)
    u = u / u.norm()
    
    for _ in range(power_iterations):
        v = W.T @ u
        v = v / (v.norm() + 1e-7)
        u = W @ v
        u = u / (u.norm() + 1e-7)
    
    sigma_max = (u @ W @ v).abs()
    
    return W * (radius / (sigma_max + 1e-7))


class StiefelOptimizer:
    """
    Wrapper that applies Stiefel manifold constraint after any base optimizer step.
    
    The Stiefel manifold constrains weight matrices to have orthonormal columns,
    meaning all singular values equal 1. This is the "manifold Muon" idea from
    the article - instead of just normalizing updates, we constrain the weights.
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, ns_steps: int = 5):
        self.base_optimizer = base_optimizer
        self.ns_steps = ns_steps
        
        # Store reference to all 2D parameters
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # Initialize on Stiefel manifold
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = stiefel_project_newton_schulz(p.data, self.ns_steps)
    
    @torch.no_grad()
    def step(self, closure=None):
        # Take base optimizer step
        # Some optimizers (like Muon) don't accept closure argument
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


class SpectralNormOptimizer:
    """
    Wrapper that constrains matrices to have spectral norm = 1 after each step.
    
    This is a softer constraint than Stiefel - only the largest singular value
    is constrained, not all of them.
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer):
        self.base_optimizer = base_optimizer
        
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # Initialize with spectral norm = 1
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = spectral_normalize(p.data)
    
    @torch.no_grad()
    def step(self, closure=None):
        # Some optimizers (like Muon) don't accept closure argument
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        for p in self.matrix_params:
            p.data = spectral_normalize(p.data)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


def sphere_project(w: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Project a vector or matrix rows onto the hypersphere of given radius.
    
    For embedding matrices, each row (embedding vector) is normalized to lie
    on the hypersphere. This implements the retraction map from the article:
    
        w ← w / sqrt(1 + η²)
    
    But for simplicity, we just normalize to unit norm directly.
    """
    if w.ndim == 1:
        return w * (radius / (w.norm() + 1e-8))
    elif w.ndim == 2:
        # Normalize each row (embedding vector) separately
        norms = w.norm(dim=1, keepdim=True)
        return w * (radius / (norms + 1e-8))
    else:
        return w


class SphereOptimizer:
    """
    Wrapper that constrains embedding vectors to lie on a hypersphere.
    
    This implements the hyperspherical descent algorithm from the article:
    1. Take base optimizer step in tangent space
    2. Apply retraction map to project back to manifold
    
    The hypersphere constraint is useful for embedding vectors where we want
    to prevent norm explosion/collapse and focus optimization on directions.
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, radius: float = 1.0):
        self.base_optimizer = base_optimizer
        self.radius = radius
        
        # Store reference to embedding parameters
        self.embed_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                self.embed_params.append(p)
        
        # Initialize on hypersphere
        for p in self.embed_params:
            with torch.no_grad():
                p.data = sphere_project(p.data, self.radius)
    
    @torch.no_grad()
    def step(self, closure=None):
        # Take base optimizer step
        # Some optimizers (like Muon) don't accept closure argument
        import inspect
        sig = inspect.signature(self.base_optimizer.step)
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # Retract to hypersphere
        for p in self.embed_params:
            p.data = sphere_project(p.data, self.radius)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

