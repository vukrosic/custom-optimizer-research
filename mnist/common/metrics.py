"""
Common metric functions for MNIST experiments.
"""

import torch
import numpy as np


def effective_rank(matrix):
    """Compute effective rank via entropy of normalized singular values (Shannon entropy).
    
    The effective rank is a continuous measure of the "dimensionality" of a matrix,
    based on the entropy of its singular value distribution:
    
        effective_rank = exp(H(σ))
        
    where H(σ) is the Shannon entropy of the normalized singular values.
    
    Args:
        matrix: 2D tensor
        
    Returns:
        Effective rank (float)
    """
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return 0.0
    
    try:
        S = torch.linalg.svdvals(matrix.float())
        S = S / (S.sum() + 1e-10)  # Normalize to probability distribution
        entropy = -(S * torch.log(S + 1e-10)).sum()  # Shannon entropy
        return torch.exp(entropy).item()
    except:
        return 0.0


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This transforms a matrix towards having all singular values equal to 1,
    effectively making it an orthonormal matrix (on the Stiefel manifold).
    
    Args:
        G: Input tensor (at least 2D)
        steps: Number of Newton-Schulz iterations
        
    Returns:
        Orthogonalized tensor
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.mT
    
    return X


def compute_spectral_metrics(matrix):
    """Compute comprehensive spectral metrics for a matrix.
    
    Args:
        matrix: 2D tensor
        
    Returns:
        Dictionary with spectral metrics or None if computation fails
    """
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return None
    
    try:
        S = torch.linalg.svdvals(matrix.float())
        S_normalized = S / (S.sum() + 1e-10)
        
        # Effective rank (entropy-based)
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()
        
        # Condition number
        condition_number = (S[0] / (S[-1] + 1e-10)).item()
        
        # Top-k concentration
        top1_ratio = (S[0] / S.sum()).item()
        top5_ratio = (S[:5].sum() / S.sum()).item() if len(S) >= 5 else 1.0
        top10_ratio = (S[:10].sum() / S.sum()).item() if len(S) >= 10 else 1.0
        
        # Frobenius norm
        frobenius = matrix.norm().item()
        
        # Full spectrum (downsample if too large)
        max_vals = 50
        if len(S) > max_vals:
            indices = torch.linspace(0, len(S)-1, max_vals).long()
            spectrum = S[indices].cpu().numpy()
        else:
            spectrum = S.cpu().numpy()
        
        return {
            'effective_rank': eff_rank,
            'condition_number': condition_number,
            'top1_ratio': top1_ratio,
            'top5_ratio': top5_ratio,
            'top10_ratio': top10_ratio,
            'frobenius_norm': frobenius,
            'spectrum': spectrum,
            'max_singular': S[0].item(),
            'min_singular': S[-1].item(),
        }
    except Exception as e:
        print(f"Error computing spectral metrics: {e}")
        return None


def compute_component_metrics(grad, ns_steps=5):
    """Compute comprehensive metrics for a gradient matrix including NS benefit.
    
    Args:
        grad: Gradient tensor (2D)
        ns_steps: Number of Newton-Schulz steps
        
    Returns:
        Dictionary with metrics or None if computation fails
    """
    if grad.ndim != 2 or min(grad.shape) < 2:
        return None
    
    try:
        grad_float = grad.float()
        
        # Basic metrics
        S = torch.linalg.svdvals(grad_float)
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()
        
        # After NS transformation
        grad_ns = zeropower_via_newtonschulz(grad_float.unsqueeze(0), steps=ns_steps).squeeze(0)
        S_ns = torch.linalg.svdvals(grad_ns)
        S_ns_norm = S_ns / (S_ns.sum() + 1e-10)
        entropy_ns = -(S_ns_norm * torch.log(S_ns_norm + 1e-10)).sum()
        eff_rank_ns = torch.exp(entropy_ns).item()
        
        # NS benefit (rank increase ratio)
        ns_benefit = eff_rank_ns / (eff_rank + 1e-10)
        
        return {
            'effective_rank': eff_rank,
            'effective_rank_ns': eff_rank_ns,
            'ns_benefit_ratio': ns_benefit,
            'max_rank': min(grad.shape),
            'rank_utilization': eff_rank / min(grad.shape),
            'condition_number': (S[0] / (S[-1] + 1e-10)).item(),
            'frobenius_norm': grad_float.norm().item(),
            'spectral_norm': S[0].item(),
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None


def categorize_layer(name):
    """Categorize layer by name for LLM-style models.
    
    Args:
        name: Layer name string
        
    Returns:
        Category string (embedding, attention_qkv, ffn, output, etc.)
    """
    name_lower = name.lower()
    
    if 'embed' in name_lower:
        return 'embedding'
    elif 'qkv' in name_lower or 'query' in name_lower or 'key' in name_lower or 'value' in name_lower:
        return 'attention_qkv'
    elif 'out_proj' in name_lower or 'o_proj' in name_lower:
        return 'attention_out'
    elif 'w1' in name_lower or 'w2' in name_lower or 'w3' in name_lower or 'ffn' in name_lower or 'mlp' in name_lower:
        return 'ffn'
    elif 'lm_head' in name_lower or 'output' in name_lower:
        return 'output'
    elif 'input' in name_lower:
        return 'input'
    elif 'hidden' in name_lower:
        return 'hidden'
    else:
        return 'other'
