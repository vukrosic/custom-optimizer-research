"""
Model implementations for Exp11: Dynamic Routing

Reuses FLA's GatedDeltaNet implementation instead of coding from scratch.

Two model types:
1. BaselineHybridModel: Uses FLA's hybrid support - layers [0,1,2]=GDN, [3]=Softmax
2. DynamicRoutingModel: Custom routing on top of FLA layers
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Add flash-linear-attention to path for local imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fla_path = os.path.join(root_dir, 'flash-linear-attention')
if os.path.exists(fla_path):
    sys.path.insert(0, fla_path)
    print(f"✓ Using LOCAL FLA clone from: {fla_path}")

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
    """Gumbel-Softmax for differentiable discrete sampling"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        y_hard = F.one_hot(y.argmax(dim=-1), num_classes=logits.size(-1)).float()
        y = y_hard - y.detach() + y
    
    return y


class BaselineHybridModel(nn.Module):
    """
    Baseline: Static hybrid using FLA's built-in support
    - Layers [0, 1, 2]: GDN
    - Layer [3]: Softmax attention
    
    This uses FLA's native hybrid architecture support.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create FLA config with hybrid support
        # FLA supports specifying which layers use attention
        fla_config = GatedDeltaNetConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            norm_eps=config.rms_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            fuse_norm=True,
            fuse_cross_entropy=True,
        )
        
        # Configure hybrid: layer 3 uses attention, others use GDN
        fla_config.attn = {
            'layers': [3],  # Only layer 3 uses softmax attention
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.num_attention_heads,
            'window_size': 2048,
            'qkv_bias': False,
            'rope_theta': 10000.0,
        }
        
        # Create model using FLA (it handles the hybrid architecture)
        self.model = GatedDeltaNetForCausalLM(fla_config)
    
    def forward(self, input_ids, labels=None, **kwargs):
        """Forward pass - delegate to FLA model"""
        return self.model(input_ids=input_ids, labels=labels, **kwargs)


class DynamicRoutingModel(nn.Module):
    """
    Dynamic routing model - routes tokens between GDN and Softmax
    - Layer [0]: GDN (fixed)
    - Layers [1, 2]: ROUTED per-token
    - Layer [3]: Softmax (fixed)
    
    Uses FLA layers internally but adds custom routing logic.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # We'll manually build this with individual FLA layers
        # since we need custom routing between them
        
        # Create base FLA model to extract layer components
        fla_config = GatedDeltaNetConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=4,
            num_heads=config.num_attention_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            norm_eps=config.rms_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
        )
        
        # Create models to extract layers
        gdn_model = GatedDeltaNetForCausalLM(fla_config)
        
        # Configure a model with attention layers for layer 3
        fla_config.attn = {
            'layers': [0, 1, 2, 3],  # All layers with attention (we'll only use indices we need)
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.num_attention_heads,
            'window_size': 2048,
            'qkv_bias': False,
            'rope_theta': 10000.0,
        }
        attn_model = GatedDeltaNetForCausalLM(fla_config)
        
        # Extract components
        self.embeddings = gdn_model.model.embeddings
        self.norm = gdn_model.model.norm
        self.lm_head = gdn_model.lm_head
        
        # Layer 0: Fixed GDN
        self.layer_0_gdn = gdn_model.model.layers[0]
        
        # Layer 1: Both options (for routing)
        self.layer_1_gdn = gdn_model.model.layers[1]
        self.layer_1_attn = attn_model.model.layers[1]
        
        # Layer 2: Both options (for routing)
        self.layer_2_gdn = gdn_model.model.layers[2]
        self.layer_2_attn = attn_model.model.layers[2]
        
        # Layer 3: Fixed Softmax
        self.layer_3_attn = attn_model.model.layers[3]
        
        # Router network
        self.router = nn.Linear(config.hidden_size, 2 * 2)  # 2 layers × 2 choices
        
        # Routing stats
        self.routing_stats = {
            'layer_1_gdn_count': 0,
            'layer_1_attn_count': 0,
            'layer_2_gdn_count': 0,
            'layer_2_attn_count': 0,
            'total_tokens': 0,
        }
    
    def get_temperature(self, step: int) -> float:
        """Anneal temperature over time"""
        if not self.config.anneal_temperature:
            return self.config.gumbel_temperature
        
        if step >= self.config.temperature_anneal_steps:
            return self.config.min_temperature
        
        progress = step / self.config.temperature_anneal_steps
        temp = self.config.gumbel_temperature - (self.config.gumbel_temperature - self.config.min_temperature) * progress
        return max(temp, self.config.min_temperature)
    
    def forward(self, input_ids, labels=None, step=0, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embeddings(input_ids)
        
        # Layer 0: Fixed GDN
        x = self.layer_0_gdn(x)[0]
        
        # === PARALLEL ROUTING ===
        router_logits = self.router(x)
        router_logits = router_logits.view(batch_size, seq_len, 2, 2)
        router_probs = F.softmax(router_logits, dim=-1)
        
        temperature = self.get_temperature(step)
        route_layer_1 = gumbel_softmax(router_logits[:, :, 0, :], temperature=temperature, hard=True)
        route_layer_2 = gumbel_softmax(router_logits[:, :, 1, :], temperature=temperature, hard=True)
        
        # Update stats
        if self.training:
            with torch.no_grad():
                self.routing_stats['layer_1_gdn_count'] += route_layer_1[..., 0].sum().item()
                self.routing_stats['layer_1_attn_count'] += route_layer_1[..., 1].sum().item()
                self.routing_stats['layer_2_gdn_count'] += route_layer_2[..., 0].sum().item()
                self.routing_stats['layer_2_attn_count'] += route_layer_2[..., 1].sum().item()
                self.routing_stats['total_tokens'] += batch_size * seq_len
        
        # Layer 1: ROUTED
        out_1_gdn = self.layer_1_gdn(x)[0]
        out_1_attn = self.layer_1_attn(x)[0]
        # Ensure routing weights match hidden state dtype
        route_1 = route_layer_1.to(x.dtype)
        x = route_1[..., 0:1] * out_1_gdn + route_1[..., 1:2] * out_1_attn
        
        # Layer 2: ROUTED
        out_2_gdn = self.layer_2_gdn(x)[0]
        out_2_attn = self.layer_2_attn(x)[0]
        route_2 = route_layer_2.to(x.dtype)
        x = route_2[..., 0:1] * out_2_gdn + route_2[..., 1:2] * out_2_attn
        
        # Layer 3: Fixed Softmax
        x = self.layer_3_attn(x)[0]
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute losses
        loss = None
        aux_loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Load balancing
            aux_loss = self.compute_load_balancing_loss([
                router_probs[:, :, 0, :],
                router_probs[:, :, 1, :],
            ])
            
            loss = lm_loss + self.config.load_balance_alpha * aux_loss
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(aux_loss,) if aux_loss is not None else None,
        )
    
    def compute_load_balancing_loss(self, router_probs_list):
        """Load balancing loss from Switch Transformer"""
        total_loss = 0.0
        num_experts = 2
        
        for router_probs in router_probs_list:
            expert_mask = F.one_hot(router_probs.argmax(dim=-1), num_classes=2).float()
            fraction_per_expert = expert_mask.mean(dim=[0, 1])
            avg_prob_per_expert = router_probs.mean(dim=[0, 1])
            layer_loss = num_experts * (fraction_per_expert * avg_prob_per_expert).sum()
            total_loss += layer_loss
        
        return total_loss / len(router_probs_list)
    
    def get_routing_stats(self, reset=True):
        """Get routing statistics"""
        if self.routing_stats['total_tokens'] == 0:
            stats = {
                'layer_1_gdn_pct': 0.0,
                'layer_1_attn_pct': 0.0,
                'layer_2_gdn_pct': 0.0,
                'layer_2_attn_pct': 0.0,
            }
        else:
            total = self.routing_stats['total_tokens']
            stats = {
                'layer_1_gdn_pct': 100 * self.routing_stats['layer_1_gdn_count'] / total,
                'layer_1_attn_pct': 100 * self.routing_stats['layer_1_attn_count'] / total,
                'layer_2_gdn_pct': 100 * self.routing_stats['layer_2_gdn_count'] / total,
                'layer_2_attn_pct': 100 * self.routing_stats['layer_2_attn_count'] / total,
            }
        
        if reset:
            self.routing_stats = {k: 0 for k in self.routing_stats.keys()}
        
        return stats


def create_model(config):
    """Factory function to create the appropriate model"""
    if config.use_dynamic_routing:
        print("✓ Creating DynamicRoutingModel (custom routing)")
        return DynamicRoutingModel(config)
    else:
        print("✓ Creating BaselineHybridModel (FLA native hybrid)")
        return BaselineHybridModel(config)
