"""
Common MNIST components for experiments.
"""

from .models import MNISTNet, ComponentMNISTNet, DepthAwareMNISTNet
from .data import set_seed, get_mnist_loaders, get_device
from .metrics import (
    effective_rank,
    zeropower_via_newtonschulz,
    compute_spectral_metrics,
    compute_component_metrics,
    categorize_layer,
)

__all__ = [
    # Models
    'MNISTNet',
    'ComponentMNISTNet', 
    'DepthAwareMNISTNet',
    # Data
    'set_seed',
    'get_mnist_loaders',
    'get_device',
    # Metrics
    'effective_rank',
    'zeropower_via_newtonschulz',
    'compute_spectral_metrics',
    'compute_component_metrics',
    'categorize_layer',
]
