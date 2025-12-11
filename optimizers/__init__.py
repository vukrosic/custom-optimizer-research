"""Optimizers package for custom optimizer research."""

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

__all__ = [
    'Muon',
    'ModularOptimizer',
    'ModularScheduler',
    'StiefelOptimizer',
    'SpectralNormOptimizer',
    'SphereOptimizer',
    'sphere_project',
    'stiefel_project_newton_schulz',
    'spectral_normalize',
    'OptimizerConfig',
    'ExperimentConfig',
    'get_experiment_configs',
    'get_experiment',
]
