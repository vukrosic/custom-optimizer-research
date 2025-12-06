"""
Modular Optimizer - Uses different optimizers for different parameter groups.

Implements the key idea from the Modular Manifolds article:
different neural network components benefit from different optimization algorithms.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from dataclasses import dataclass
import re

from experiments.experiment_config import OptimizerConfig, ExperimentConfig
from experiments.manifold_constraints import StiefelOptimizer, SpectralNormOptimizer, SphereOptimizer


@dataclass
class ParameterGroup:
    """A group of parameters with the same optimizer."""
    name: str
    params: List[torch.nn.Parameter]
    param_names: List[str]
    optimizer_config: OptimizerConfig


def classify_parameter(name: str) -> str:
    """Classify a parameter into: embedding, attention, ffn, norm, or other."""
    if 'token_embedding' in name or 'embed' in name.lower() or 'lm_head' in name.lower():
        return 'embedding'
    if 'attention' in name.lower() or 'qkv' in name.lower():
        return 'attention'
    if 'feed_forward' in name.lower() or re.search(r'\bw[123]\b', name.lower()):
        return 'ffn'
    if 'norm' in name.lower():
        return 'norm'
    return 'other'


class ModularOptimizer:
    """
    Optimizer that uses different optimization algorithms for different
    parts of the model, with optional manifold constraints.
    """
    
    def __init__(self, model: nn.Module, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.param_groups: Dict[str, ParameterGroup] = {}
        self.step_count = 0
        
        self._classify_parameters()
        self._create_optimizers()
        
    def _classify_parameters(self):
        """Classify all model parameters into groups."""
        groups = {
            'embedding': {'params': [], 'names': []},
            'attention': {'params': [], 'names': []},
            'ffn': {'params': [], 'names': []},
            'norm': {'params': [], 'names': []},
            'other': {'params': [], 'names': []},
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            category = classify_parameter(name)
            groups[category]['params'].append(param)
            groups[category]['names'].append(name)
        
        # Map to optimizer configs
        config_map = {
            'embedding': self.config.embedding_optimizer,
            'attention': self.config.attention_optimizer,
            'ffn': self.config.ffn_optimizer,
            'norm': self.config.norm_optimizer,
            'other': self.config.embedding_optimizer,  # Default
        }
        
        for group_name, group_data in groups.items():
            if group_data['params']:
                self.param_groups[group_name] = ParameterGroup(
                    name=group_name,
                    params=group_data['params'],
                    param_names=group_data['names'],
                    optimizer_config=config_map[group_name],
                )
        
        # Print summary
        print("\n📊 Parameter Classification:")
        for name, group in self.param_groups.items():
            opt = group.optimizer_config
            manifold = getattr(opt, 'manifold', 'none')
            manifold_str = f" [{manifold}]" if manifold != 'none' else ""
            num_params = sum(p.numel() for p in group.params)
            print(f"  {name}: {num_params:,} params → {opt.name}{manifold_str}")
    
    def _create_optimizers(self):
        """Create an optimizer for each parameter group."""
        import sys
        sys.path.insert(0, '..')
        from muon import Muon
        
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
                # Muon only for 2D+ params
                muon_params = [p for p in group.params if p.ndim >= 2]
                other_params = [p for p in group.params if p.ndim < 2]
                
                if muon_params:
                    base_opt = Muon(
                        muon_params,
                        lr=opt_config.lr,
                        momentum=opt_config.momentum,
                        nesterov=opt_config.nesterov,
                        ns_steps=opt_config.ns_steps,
                    )
                    
                    # Wrap with manifold constraint if specified
                    if manifold == 'stiefel':
                        self.optimizers[f'{group_name}'] = StiefelOptimizer(muon_params, base_opt)
                    elif manifold == 'spectral':
                        self.optimizers[f'{group_name}'] = SpectralNormOptimizer(muon_params, base_opt)
                    else:
                        self.optimizers[f'{group_name}'] = base_opt
                        
                if other_params:
                    self.optimizers[f'{group_name}_1d'] = torch.optim.AdamW(
                        other_params, lr=3e-4, weight_decay=0.0
                    )
                continue
            else:
                raise ValueError(f"Unknown optimizer: {opt_config.name}")
            
            # Wrap with manifold constraint if specified
            if manifold == 'stiefel':
                self.optimizers[group_name] = StiefelOptimizer(group.params, base_opt)
            elif manifold == 'spectral':
                self.optimizers[group_name] = SpectralNormOptimizer(group.params, base_opt)
            elif manifold == 'sphere':
                self.optimizers[group_name] = SphereOptimizer(group.params, base_opt)
            else:
                self.optimizers[group_name] = base_opt
        
        print(f"\n✓ Created {len(self.optimizers)} optimizer instances")
    
    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def step(self):
        for opt in self.optimizers.values():
            opt.step()
        self.step_count += 1
    
    def get_lr(self) -> Dict[str, float]:
        return {name: opt.param_groups[0]['lr'] for name, opt in self.optimizers.items()}


class ModularScheduler:
    """Learning rate scheduler for modular optimizer."""
    
    def __init__(self, modular_optimizer: ModularOptimizer, warmup_steps: int, max_steps: int):
        self.modular_optimizer = modular_optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
        
        self.initial_lrs = {
            name: opt.param_groups[0]['lr'] 
            for name, opt in modular_optimizer.optimizers.items()
        }
    
    def step(self):
        self.current_step += 1
        
        for name, opt in self.modular_optimizer.optimizers.items():
            base_lr = self.initial_lrs[name]
            
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Handle edge case where warmup_steps equals max_steps
                if self.max_steps <= self.warmup_steps:
                    progress = 1.0
                else:
                    progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                lr = base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            
            for param_group in opt.param_groups:
                param_group['lr'] = lr
    
    def get_last_lr(self):
        return [opt.param_groups[0]['lr'] for opt in self.modular_optimizer.optimizers.values()]
