"""Optimizers package for custom optimizer research.

Contains both standard optimizers (Muon, ModularOptimizer) and novel
manifold-constrained optimizers from the Modular Manifolds framework.
"""

from optimizers.muon import Muon
from optimizers.modular_optimizer import ModularOptimizer, ModularScheduler
from optimizers.manifold_constraints import (
    StiefelOptimizer,
    SpectralNormOptimizer,
    SphereOptimizer,
    sphere_project,
    stiefel_project_newton_schulz,
    spectral_normalize,
)
from optimizers.config import OptimizerConfig, ExperimentConfig, get_experiment_configs, get_experiment

# New optimizers from optimizer_ideas.md
from optimizers.oblique import ObliqueOptimizer, oblique_project
from optimizers.grassmannian import GrassmannianOptimizer, grassmann_project
from optimizers.block_stiefel import BlockStiefelOptimizer, block_stiefel_project
from optimizers.sl_muon import SLMuonOptimizer, sl_project
from optimizers.l1_stiefel import L1StiefelOptimizer
from optimizers.symplectic import SymplecticMuonOptimizer, is_symplectic
from optimizers.doubly_stochastic import DoublyStochasticOptimizer, sinkhorn_normalize

__all__ = [
    # Core optimizers
    'Muon',
    'ModularOptimizer',
    'ModularScheduler',
    
    # Manifold constraint wrappers
    'StiefelOptimizer',
    'SpectralNormOptimizer',
    'SphereOptimizer',
    
    # New manifold optimizers
    'ObliqueOptimizer',       # Embeddings (unit columns, correlated)
    'GrassmannianOptimizer',  # Subspace optimization
    'BlockStiefelOptimizer',  # Multi-head attention
    'SLMuonOptimizer',        # Volume-preserving (det=1)
    'L1StiefelOptimizer',     # Sparse orthogonal
    'SymplecticMuonOptimizer',# Physics/Hamiltonian
    'DoublyStochasticOptimizer', # Permutation learning
    
    # Projection functions
    'sphere_project',
    'stiefel_project_newton_schulz',
    'spectral_normalize',
    'oblique_project',
    'grassmann_project',
    'block_stiefel_project',
    'sl_project',
    'sinkhorn_normalize',
    'is_symplectic',
    
    # Config
    'OptimizerConfig',
    'ExperimentConfig',
    'get_experiment_configs',
    'get_experiment',
]

