"""
Simple configuration for Standard Attention LLM
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50257
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads
    
    @property
    def intermediate_size(self):
        return self.hidden_size * 4


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    batch_size: int = 16
    max_seq_len: int = 1024
    num_samples: int = 50000
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    
    # Schedule
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    eval_batches: int = 10
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda"
    seed: int = 42
