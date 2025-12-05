"""
Modular Optimizer - Manages different optimizers for different parameter groups.

This implements the key idea from the Modular Manifolds article:
different neural network components benefit from different optimization algorithms
based on their mathematical properties.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

import sys
sys.path.append('..')
from muon import Muon
from experiments.experiment_config import OptimizerConfig, ExperimentConfig


@dataclass
class ParameterGroup:
    """A group of parameters with the same optimizer."""
    name: str
    params: List[torch.nn.Parameter]
    param_names: List[str]
    optimizer_config: OptimizerConfig
    
    
def classify_parameter(name: str, param: torch.nn.Parameter) -> str:
    """
    Classify a parameter into one of the groups:
    - 'embedding': Token embeddings
    - 'attention_qkv': Q/K/V projections
    - 'attention_out': Output projection
    - 'ffn': Feed-forward weights
    - 'norm': Normalization parameters
    - 'other': Anything else
    """
    # Embedding
    if 'token_embedding' in name or 'embed' in name.lower():
        return 'embedding'
    
    # Attention layers
    if 'attention' in name.lower() or 'attn' in name.lower():
        if 'qkv' in name.lower() or 'q_proj' in name.lower() or 'k_proj' in name.lower() or 'v_proj' in name.lower():
            return 'attention_qkv'
        if 'out' in name.lower() or 'o_proj' in name.lower():
            return 'attention_out'
        return 'attention_qkv'  # Default attention params to QKV group
    
    # FFN layers
    if 'feed_forward' in name.lower() or 'ffn' in name.lower() or 'mlp' in name.lower():
        return 'ffn'
    # Also catch w1, w2, w3 patterns
    if re.search(r'\bw[123]\b', name.lower()):
        return 'ffn'
    
    # Normalization (RMSNorm, LayerNorm)
    if 'norm' in name.lower():
        return 'norm'
    
    # LM head (tied with embeddings usually)
    if 'lm_head' in name.lower():
        return 'embedding'
    
    return 'other'


class ModularOptimizer:
    """
    Optimizer that uses different optimization algorithms for different
    parts of the model based on the Modular Manifolds theory.
    
    Key insight: Muon (spectral normalization) is good for 2D weight matrices,
    while AdamW is fine for embeddings, biases, and normalization parameters.
    """
    
    def __init__(self, model: nn.Module, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.param_groups: Dict[str, ParameterGroup] = {}
        
        # Classify and group parameters
        self._classify_parameters()
        
        # Create optimizers for each group
        self._create_optimizers()
        
        # Track step count for logging
        self.step_count = 0
        
    def _classify_parameters(self):
        """Classify all model parameters into groups."""
        groups = {
            'embedding': {'params': [], 'names': []},
            'attention': {'params': [], 'names': []},
            'attention_qkv': {'params': [], 'names': []},
            'attention_out': {'params': [], 'names': []},
            'ffn': {'params': [], 'names': []},
            'norm': {'params': [], 'names': []},
            'other': {'params': [], 'names': []},
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            category = classify_parameter(name, param)
            groups[category]['params'].append(param)
            groups[category]['names'].append(name)
        
        # Map to optimizer configs
        # Combine attention_qkv and attention_out unless specifically separated
        if self.config.name == 'qkv_muon':
            # Special case: QKV gets Muon, out_proj gets AdamW
            self.param_groups['attention_qkv'] = ParameterGroup(
                name='attention_qkv',
                params=groups['attention_qkv']['params'],
                param_names=groups['attention_qkv']['names'],
                optimizer_config=self.config.attention_optimizer,
            )
            self.param_groups['attention_out'] = ParameterGroup(
                name='attention_out',
                params=groups['attention_out']['params'],
                param_names=groups['attention_out']['names'],
                optimizer_config=self.config.embedding_optimizer,  # Use AdamW for out_proj
            )
        else:
            # Normal case: all attention params use same optimizer
            all_attn_params = groups['attention_qkv']['params'] + groups['attention_out']['params']
            all_attn_names = groups['attention_qkv']['names'] + groups['attention_out']['names']
            if all_attn_params:
                self.param_groups['attention'] = ParameterGroup(
                    name='attention',
                    params=all_attn_params,
                    param_names=all_attn_names,
                    optimizer_config=self.config.attention_optimizer,
                )
        
        # Other groups
        if groups['embedding']['params']:
            self.param_groups['embedding'] = ParameterGroup(
                name='embedding',
                params=groups['embedding']['params'],
                param_names=groups['embedding']['names'],
                optimizer_config=self.config.embedding_optimizer,
            )
        
        if groups['ffn']['params']:
            self.param_groups['ffn'] = ParameterGroup(
                name='ffn',
                params=groups['ffn']['params'],
                param_names=groups['ffn']['names'],
                optimizer_config=self.config.ffn_optimizer,
            )
            
        if groups['norm']['params']:
            self.param_groups['norm'] = ParameterGroup(
                name='norm',
                params=groups['norm']['params'],
                param_names=groups['norm']['names'],
                optimizer_config=self.config.norm_optimizer,
            )
            
        if groups['other']['params']:
            # Use embedding optimizer for uncategorized params
            self.param_groups['other'] = ParameterGroup(
                name='other',
                params=groups['other']['params'],
                param_names=groups['other']['names'],
                optimizer_config=self.config.embedding_optimizer,
            )
        
        # Print summary
        print("\n📊 Parameter Classification:")
        for name, group in self.param_groups.items():
            opt_name = group.optimizer_config.name
            num_params = sum(p.numel() for p in group.params)
            print(f"  {name}: {len(group.params)} tensors, {num_params:,} params → {opt_name}")
    
    def _create_optimizers(self):
        """Create an optimizer for each parameter group, with optional manifold constraints."""
        from experiments.manifold_constraints import (
            StiefelOptimizer, HypersphereOptimizer, SpectralNormOptimizer
        )
        
        for group_name, group in self.param_groups.items():
            if not group.params:
                continue
                
            opt_config = group.optimizer_config
            manifold = getattr(opt_config, 'manifold', 'none')
            
            # Create base optimizer
            if opt_config.name == 'adamw':
                base_opt = torch.optim.AdamW(
                    group.params,
                    lr=opt_config.lr,
                    weight_decay=opt_config.weight_decay,
                    betas=opt_config.betas,
                )
            elif opt_config.name == 'muon':
                # Filter to only 2D+ parameters for Muon
                muon_params = [p for p in group.params if p.ndim >= 2]
                other_params = [p for p in group.params if p.ndim < 2]
                
                if muon_params:
                    from muon import Muon
                    base_opt = Muon(
                        muon_params,
                        lr=opt_config.lr,
                        momentum=opt_config.momentum,
                        nesterov=opt_config.nesterov,
                        ns_steps=opt_config.ns_steps,
                    )
                    
                    # Wrap with manifold constraint if specified
                    if manifold == 'stiefel':
                        self.optimizers[f'{group_name}_muon'] = StiefelOptimizer(
                            muon_params, base_opt,
                            retraction=getattr(opt_config, 'retraction', 'newton_schulz')
                        )
                    elif manifold == 'spectral':
                        self.optimizers[f'{group_name}_muon'] = SpectralNormOptimizer(
                            muon_params, base_opt,
                            radius=getattr(opt_config, 'manifold_radius', 1.0)
                        )
                    else:
                        self.optimizers[f'{group_name}_muon'] = base_opt
                        
                if other_params:
                    # Use AdamW for 1D params even in Muon groups
                    self.optimizers[f'{group_name}_adamw'] = torch.optim.AdamW(
                        other_params,
                        lr=3e-4,  # Default AdamW lr for fallback
                        weight_decay=0.0,
                    )
                continue  # Skip the rest for Muon
                
            elif opt_config.name == 'sgd':
                base_opt = torch.optim.SGD(
                    group.params,
                    lr=opt_config.lr,
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_config.name}")
            
            # Wrap with manifold constraint if specified
            if manifold == 'stiefel':
                self.optimizers[group_name] = StiefelOptimizer(
                    group.params, base_opt,
                    retraction=getattr(opt_config, 'retraction', 'newton_schulz')
                )
                print(f"    ↳ Wrapped with Stiefel manifold constraint")
            elif manifold == 'hypersphere':
                self.optimizers[group_name] = HypersphereOptimizer(
                    group.params, base_opt,
                    radius=getattr(opt_config, 'manifold_radius', 1.0)
                )
                print(f"    ↳ Wrapped with Hypersphere manifold constraint")
            elif manifold == 'spectral':
                self.optimizers[group_name] = SpectralNormOptimizer(
                    group.params, base_opt,
                    radius=getattr(opt_config, 'manifold_radius', 1.0)
                )
                print(f"    ↳ Wrapped with Spectral norm constraint")
            else:
                # No manifold constraint, but still handle normalize for SGD
                if opt_config.name == 'sgd' and getattr(opt_config, 'normalize', False):
                    self.optimizers[group_name] = SphericalSGD(
                        group.params,
                        lr=opt_config.lr,
                        normalize=True,
                    )
                else:
                    self.optimizers[group_name] = base_opt
        
        print(f"\n✓ Created {len(self.optimizers)} optimizer instances")
        for name, opt in self.optimizers.items():
            manifold_str = ""
            if hasattr(opt, 'retraction'):
                manifold_str = " [Stiefel]"
            elif hasattr(opt, 'radius') and hasattr(opt, 'sphere_params'):
                manifold_str = " [Hypersphere]"
            elif hasattr(opt, 'radius') and hasattr(opt, 'matrix_params'):
                manifold_str = " [Spectral]"
            print(f"  {name}: {opt.__class__.__name__}{manifold_str}")
    
    def zero_grad(self):
        """Zero gradients for all optimizers."""
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def step(self):
        """Take an optimization step with all optimizers."""
        for opt in self.optimizers.values():
            opt.step()
        self.step_count += 1
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict of all optimizers."""
        return {
            name: opt.state_dict() 
            for name, opt in self.optimizers.items()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for all optimizers."""
        for name, opt in self.optimizers.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates for each optimizer."""
        lrs = {}
        for name, opt in self.optimizers.items():
            lrs[name] = opt.param_groups[0]['lr']
        return lrs


class SphericalSGD(torch.optim.Optimizer):
    """
    SGD optimizer that optionally projects weights onto the unit hypersphere
    after each step. Implements the hyperspherical constraint from the
    Modular Manifolds article.
    """
    
    def __init__(self, params, lr=0.01, normalize=True):
        defaults = dict(lr=lr, normalize=normalize)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Standard gradient descent step
                p.add_(p.grad, alpha=-group['lr'])
                
                # Project to unit sphere if enabled
                if group['normalize']:
                    # Normalize each row of the embedding matrix
                    if p.ndim == 2:
                        p.data = torch.nn.functional.normalize(p.data, dim=1)
                    else:
                        p.data = torch.nn.functional.normalize(p.data, dim=0)


class ModularScheduler:
    """
    Learning rate scheduler that coordinates schedules across multiple optimizers.
    Applies warmup + cosine decay to all optimizers.
    """
    
    def __init__(self, modular_optimizer: ModularOptimizer, 
                 warmup_steps: int, max_steps: int):
        self.modular_optimizer = modular_optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
        
        # Store initial learning rates
        self.initial_lrs = {}
        for name, opt in modular_optimizer.optimizers.items():
            self.initial_lrs[name] = opt.param_groups[0]['lr']
    
    def step(self):
        """Update learning rates based on current step."""
        self.current_step += 1
        
        for name, opt in self.modular_optimizer.optimizers.items():
            base_lr = self.initial_lrs[name]
            
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                lr = base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            
            for param_group in opt.param_groups:
                param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        """Get the last computed learning rates."""
        return [opt.param_groups[0]['lr'] for opt in self.modular_optimizer.optimizers.values()]
