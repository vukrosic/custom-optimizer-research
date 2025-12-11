"""
Stiefel Optimizer: Constrains weight matrices to Stiefel manifold.

Maintains orthonormal columns (W^T W = I) for weight matrices using
Newton-Schulz iteration for retraction to the manifold.

Manifold: Stiefel (W^T W = I)
"""

import torch


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


class StiefelOptimizer:
    """Optimizer with Stiefel manifold constraint.
    
    After each optimization step, projects weight matrices back to the
    Stiefel manifold (W^T W = I) using Newton-Schulz iteration.
    
    Usage:
        base_opt = torch.optim.AdamW(params, lr=0.001)
        stiefel_opt = StiefelOptimizer(params, base_opt, ns_steps=5)
    """
    
    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 ns_steps: int = 5):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer
            ns_steps: Newton-Schulz iterations for retraction
        """
        self.base_optimizer = base_optimizer
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
    
    @torch.no_grad()
    def step(self, closure=None):
        """Take optimization step and retract to Stiefel manifold."""
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
