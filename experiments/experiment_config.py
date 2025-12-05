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
    """Return all experiment configurations."""
    
    # Optimizer presets
    adamw = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.1)
    adamw_no_wd = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0)
    muon = OptimizerConfig("muon", lr=0.02, momentum=0.95)
    
    # With manifold constraints
    adamw_stiefel = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0, manifold="stiefel")
    muon_stiefel = OptimizerConfig("muon", lr=0.02, momentum=0.95, manifold="stiefel")
    adamw_spectral = OptimizerConfig("adamw", lr=3e-4, weight_decay=0.0, manifold="spectral")
    
    experiments = {}
    
    # =========================================================================
    # PART 1: Optimizer comparison (no manifold constraints)
    # =========================================================================
    
    experiments["baseline"] = ExperimentConfig(
        name="baseline",
        description="AdamW for all parameters",
        embedding_optimizer=adamw,
        attention_optimizer=adamw,
        ffn_optimizer=adamw,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["muon_all"] = ExperimentConfig(
        name="muon_all", 
        description="Muon for attention & FFN, AdamW for embeddings & norms",
        embedding_optimizer=adamw,
        attention_optimizer=muon,
        ffn_optimizer=muon,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["muon_attention"] = ExperimentConfig(
        name="muon_attention",
        description="Muon for attention only",
        embedding_optimizer=adamw,
        attention_optimizer=muon,
        ffn_optimizer=adamw,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["muon_ffn"] = ExperimentConfig(
        name="muon_ffn",
        description="Muon for FFN only",
        embedding_optimizer=adamw,
        attention_optimizer=adamw,
        ffn_optimizer=muon,
        norm_optimizer=adamw_no_wd,
    )
    
    # =========================================================================
    # PART 2: Manifold constraint experiments
    # =========================================================================
    
    experiments["stiefel_all"] = ExperimentConfig(
        name="stiefel_all",
        description="Stiefel manifold (W^T W = I) for attention & FFN",
        embedding_optimizer=adamw,
        attention_optimizer=adamw_stiefel,
        ffn_optimizer=adamw_stiefel,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["stiefel_attention"] = ExperimentConfig(
        name="stiefel_attention",
        description="Stiefel manifold for attention only",
        embedding_optimizer=adamw,
        attention_optimizer=adamw_stiefel,
        ffn_optimizer=adamw,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["stiefel_ffn"] = ExperimentConfig(
        name="stiefel_ffn",
        description="Stiefel manifold for FFN only",
        embedding_optimizer=adamw,
        attention_optimizer=adamw,
        ffn_optimizer=adamw_stiefel,
        norm_optimizer=adamw_no_wd,
    )
    
    experiments["spectral_all"] = ExperimentConfig(
        name="spectral_all",
        description="Spectral norm = 1 for attention & FFN",
        embedding_optimizer=adamw,
        attention_optimizer=adamw_spectral,
        ffn_optimizer=adamw_spectral,
        norm_optimizer=adamw_no_wd,
    )
    
    # =========================================================================
    # PART 3: Combined approaches (Muon + manifold)
    # =========================================================================
    
    experiments["manifold_muon"] = ExperimentConfig(
        name="manifold_muon",
        description="Muon optimizer + Stiefel constraint (full manifold Muon)",
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
