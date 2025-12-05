"""
Experiment configurations for modular optimizer research.

Focuses on the core comparison:
1. Baseline (AdamW everywhere)
2. Muon for weight matrices
3. Stiefel manifold constraints
4. Manifold Muon (combining both)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class OptimizerConfig:
    """Configuration for a single optimizer with optional manifold constraint."""
    name: str  # "adamw" or "muon"
    lr: float
    weight_decay: float = 0.0
    
    # AdamW specific
    betas: tuple = (0.9, 0.95)
    
    # Muon specific
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    
    # Manifold constraint: "none", "stiefel", or "spectral"
    manifold: str = "none"


@dataclass  
class ExperimentConfig:
    """Configuration for a complete experiment."""
    name: str
    description: str
    
    # Optimizer for each parameter group
    embedding_optimizer: OptimizerConfig = None
    attention_optimizer: OptimizerConfig = None
    ffn_optimizer: OptimizerConfig = None
    norm_optimizer: OptimizerConfig = None
    
    # Training settings
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 16
    max_seq_len: int = 512
    num_samples: int = 30000
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    
    # Model settings
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    
    def __post_init__(self):
        if self.embedding_optimizer is None:
            self.embedding_optimizer = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.1)
        if self.attention_optimizer is None:
            self.attention_optimizer = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.1)
        if self.ffn_optimizer is None:
            self.ffn_optimizer = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.1)
        if self.norm_optimizer is None:
            self.norm_optimizer = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0)


def get_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Return all experiment configurations.
    
    Main experiments:
    1. baseline: Muon for 2D weight matrices, AdamW for embeddings/norms
    2. sphere_constraint: Same as baseline but with sphere constraint on embeddings
    """
    
    # Optimizer presets
    adamw = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.1)
    adamw_no_wd = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0)
    muon = OptimizerConfig("muon", lr=0.02, momentum=0.95)
    
    # With manifold constraints (from Modular Manifolds article)
    adamw_sphere = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0, manifold="sphere")
    adamw_stiefel = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0, manifold="stiefel")
    muon_stiefel = OptimizerConfig("muon", lr=0.02, momentum=0.95, manifold="stiefel")
    adamw_spectral = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0, manifold="spectral")
    
    experiments = {}
    
    # =========================================================================
    # MAIN EXPERIMENTS (from Modular Manifolds article)
    # =========================================================================
    
    # Baseline: Muon for 2D matrices (attention, FFN), AdamW for others
    experiments["baseline"] = ExperimentConfig(
        name="baseline",
        description="Muon for 2D weight matrices, AdamW for embeddings & norms",
        embedding_optimizer=adamw,
        attention_optimizer=muon,
        ffn_optimizer=muon,
        norm_optimizer=adamw_no_wd,
    )
    
    # Sphere constraint: Same as baseline but embeddings constrained to hypersphere
    # This is the key experiment from the article - constraining embedding vectors
    # to unit norm hypersphere for healthier training
    experiments["sphere_constraint"] = ExperimentConfig(
        name="sphere_constraint",
        description="Baseline + hypersphere constraint on embeddings (unit norm rows)",
        embedding_optimizer=adamw_sphere,
        attention_optimizer=muon,
        ffn_optimizer=muon,
        norm_optimizer=adamw_no_wd,
    )
    
    # =========================================================================
    # ADDITIONAL EXPERIMENTS (for ablation)
    # =========================================================================
    
    # AdamW everywhere as a reference
    experiments["adamw_only"] = ExperimentConfig(
        name="adamw_only",
        description="AdamW for all parameters (no Muon)",
        embedding_optimizer=adamw,
        attention_optimizer=adamw,
        ffn_optimizer=adamw,
        norm_optimizer=adamw_no_wd,
    )
    
    # Stiefel manifold for all 2D weights
    experiments["stiefel_all"] = ExperimentConfig(
        name="stiefel_all",
        description="Stiefel manifold (W^T W = I) for attention & FFN",
        embedding_optimizer=adamw,
        attention_optimizer=adamw_stiefel,
        ffn_optimizer=adamw_stiefel,
        norm_optimizer=adamw_no_wd,
    )
    
    # Full manifold approach: sphere for embeddings, Stiefel for matrices
    experiments["full_manifold"] = ExperimentConfig(
        name="full_manifold",
        description="Sphere for embeddings + Stiefel manifold for attention & FFN",
        embedding_optimizer=adamw_sphere,
        attention_optimizer=muon_stiefel,
        ffn_optimizer=muon_stiefel,
        norm_optimizer=adamw_no_wd,
    )
    
    # Manifold Muon (Muon + Stiefel without sphere embeddings)
    experiments["manifold_muon"] = ExperimentConfig(
        name="manifold_muon",
        description="Muon optimizer + Stiefel constraint (no sphere embeddings)",
        embedding_optimizer=adamw,
        attention_optimizer=muon_stiefel,
        ffn_optimizer=muon_stiefel,
        norm_optimizer=adamw_no_wd,
    )
    
    return experiments


def get_experiment(name: str) -> ExperimentConfig:
    """Get a specific experiment configuration by name."""
    experiments = get_experiment_configs()
    if name not in experiments:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(experiments.keys())}")
    return experiments[name]

