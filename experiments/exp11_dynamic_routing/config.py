"""
Configuration for Dynamic Routing Experiment (Exp11)

Two configurations:
1. Baseline: Static layers [0,1,2]=GDN, [3]=Softmax
2. Dynamic: Layer [0]=GDN, [1,2]=ROUTED, [3]=Softmax
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ExperimentConfig:
    """Configuration for Dynamic Routing experiment"""
    
    # Model Architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 4  # Small model for faster experimentation
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    
    # GDN specific
    expand_k: float = 1.0
    expand_v: float = 1.0
    
    # MLP configuration
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    
    # Dynamic Routing Configuration
    use_dynamic_routing: bool = False  # False = baseline, True = dynamic
    routed_layers: List[int] = None  # Which layers use routing (e.g., [1, 2])
    fixed_gdn_layers: List[int] = None  # Which layers always use GDN (e.g., [0])
    fixed_attn_layers: List[int] = None  # Which layers always use Softmax (e.g., [3])
    
    # Load Balancing
    load_balance_alpha: float = 0.01  # Weight for load balancing loss
    
    # Gumbel-Softmax
    gumbel_temperature: float = 1.0  # Temperature for Gumbel-Softmax sampling
    anneal_temperature: bool = True  # Whether to anneal temperature during training
    min_temperature: float = 0.5
    temperature_anneal_steps: int = 5000
    
    # Regularization
    rms_norm_eps: float = 1e-6
    
    # Training
    batch_size: int = 48
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 1024
    num_documents: int = 70_000
    max_tokens: int = 70_000_000
    
    # Evaluation
    eval_interval: int = 50
    eval_batches: int = 20
    
    # Logging
    log_interval: int = 10
    log_routing_stats: bool = True  # Log routing statistics for dynamic config
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda"
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Set intermediate size if not provided"""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.hidden_ratio
        
        # Validate routing configuration
        if self.use_dynamic_routing:
            assert self.routed_layers is not None, "Must specify routed_layers for dynamic routing"
            assert len(self.routed_layers) > 0, "Must have at least one routed layer"
        
        # Ensure all layer assignments are valid
        total_layers = set(range(self.num_hidden_layers))
        routed = set(self.routed_layers or [])
        fixed_gdn = set(self.fixed_gdn_layers or [])
        fixed_attn = set(self.fixed_attn_layers or [])
        
        # Check no overlap
        assert len(routed & fixed_gdn) == 0, "Layer cannot be both routed and fixed GDN"
        assert len(routed & fixed_attn) == 0, "Layer cannot be both routed and fixed attention"
        assert len(fixed_gdn & fixed_attn) == 0, "Layer cannot be both fixed GDN and fixed attention"
        
        # Check all layers are assigned
        assigned = routed | fixed_gdn | fixed_attn
        assert assigned == total_layers, f"All layers must be assigned. Missing: {total_layers - assigned}"


def get_baseline_config():
    """
    Baseline configuration: Static layer assignment
    
    Architecture (4 layers):
    - Layer 0: GDN (fixed)
    - Layer 1: GDN (fixed)
    - Layer 2: GDN (fixed)
    - Layer 3: Softmax (fixed)
    
    This is 75% GDN, 25% Softmax
    No routing overhead
    """
    return ExperimentConfig(
        # Model size
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # No dynamic routing
        use_dynamic_routing=False,
        routed_layers=[],
        fixed_gdn_layers=[0, 1, 2],
        fixed_attn_layers=[3],
        
        # Training
        max_seq_len=1024,
        batch_size=48,
        max_steps=1000,
        warmup_steps=100,
        learning_rate=1e-3,  # GDN prefers 1e-3
        gradient_clip=1.0,
        
        # Data
        num_documents=70_000,
        max_tokens=70_000_000,
        
        # Evaluation
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
        
        # Checkpointing
        checkpoint_dir="checkpoints_baseline",
    )


def get_dynamic_config():
    """
    Dynamic routing configuration
    
    Architecture (4 layers):
    - Layer 0: GDN (fixed) - stability
    - Layer 1: ROUTED per-token (GDN or Softmax)
    - Layer 2: ROUTED per-token (GDN or Softmax)
    - Layer 3: Softmax (fixed) - hypothesis from literature
    
    Routing is done in parallel at layer 0 (after seeing input embeddings)
    """
    return ExperimentConfig(
        # Model size
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Dynamic routing enabled
        use_dynamic_routing=True,
        routed_layers=[1, 2],  # These layers use routing
        fixed_gdn_layers=[0],  # Layer 0 always GDN
        fixed_attn_layers=[3],  # Layer 3 always Softmax
        
        # Load balancing to prevent collapse
        load_balance_alpha=0.01,  # Start conservative, increase if collapse happens
        
        # Gumbel-Softmax configuration
        gumbel_temperature=1.0,
        anneal_temperature=True,
        min_temperature=0.5,
        temperature_anneal_steps=5000,
        
        # Training
        max_seq_len=1024,
        batch_size=48,
        max_steps=1000,
        warmup_steps=100,
        learning_rate=2e-3,  # Hybrids with attention prefer higher LR (from exp7)
        gradient_clip=1.0,
        
        # Data
        num_documents=70_000,
        max_tokens=70_000_000,
        
        # Evaluation
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
        log_routing_stats=True,
        
        # Checkpointing
        checkpoint_dir="checkpoints_dynamic",
    )


def get_dynamic_aggressive_balance_config():
    """
    Same as dynamic but with stronger load balancing
    Use this if routing collapses with default config
    """
    config = get_dynamic_config()
    config.load_balance_alpha = 0.05  # 5x stronger
    config.checkpoint_dir = "checkpoints_dynamic_aggressive"
    return config


# Config registry
CONFIGS = {
    'baseline': get_baseline_config,
    'dynamic': get_dynamic_config,
    'dynamic_aggressive': get_dynamic_aggressive_balance_config,
}


def get_config(config_name: str) -> ExperimentConfig:
    """Get config by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]()
