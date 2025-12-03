"""
Model implementations for Exp11: Dynamic Routing

Two model types:
1. BaselineHybridModel: Static layers [0,1,2]=GDN, [3]=Softmax
2. DynamicRoutingModel: Layer [0]=GDN, [1,2]=ROUTED, [3]=Softmax
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Add flash-linear-attention to path for local imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fla_path = os.path.join(root_dir, 'flash-linear-attention')
if os.path.exists(fla_path):
    sys.path.insert(0, fla_path)
    print(f"✓ Using LOCAL FLA clone from: {fla_path}")

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️  flash-attn not available, using standard attention")


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
    """
    Gumbel-Softmax for differentiable discrete sampling
    
    Args:
        logits: [batch, seq, num_choices]
        temperature: Controls randomness (high=random, low=greedy)
        hard: If True, returns one-hot in forward, soft in backward
    
    Returns:
        Routing weights [batch, seq, num_choices]
    """
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # Forward: hard one-hot, Backward: soft gradients (straight-through)
        y_hard = F.one_hot(y.argmax(dim=-1), num_classes=logits.size(-1)).float()
        y = y_hard - y.detach() + y
    
    return y


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


class SoftmaxAttentionLayer(nn.Module):
    """Standard softmax attention layer with RoPE"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # Use flash attention
            qkv = torch.stack([q, k, v], dim=2)
            output = flash_attn_qkvpacked_func(qkv, causal=True)
        else:
            # Standard attention
            q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output = attn @ v
            output = output.transpose(1, 2)
        
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return output
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class BaselineHybridModel(nn.Module):
    """
    Baseline: Static hybrid model
    - Layers [0, 1, 2]: GDN (fixed)
    - Layer [3]: Softmax (fixed)
    
    75% GDN, 25% Softmax, no routing overhead
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Create GDN config for FLA layers
        gdn_config = GatedDeltaNetConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=1,  # We'll use layers individually
            num_heads=config.num_attention_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
        )
        
        # Layers 0, 1, 2: GDN
        self.gdn_layers = nn.ModuleList([
            GatedDeltaNetForCausalLM(gdn_config).model.layers[0]
            for _ in range(3)
        ])
        
        # Layer 3: Softmax
        self.attn_layer = SoftmaxAttentionLayer(config.hidden_size, config.num_attention_heads)
        
        # Final norm and LM head
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        # Embed
        x = self.embed_tokens(input_ids)
        
        # Layer 0: GDN
        x = self.gdn_layers[0](x)[0]
        
        # Layer 1: GDN
        x = self.gdn_layers[1](x)[0]
        
        # Layer 2: GDN
        x = self.gdn_layers[2](x)[0]
        
        # Layer 3: Softmax
        x = self.attn_layer(x)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


class DynamicRoutingModel(nn.Module):
    """
    Dynamic routing model
    - Layer [0]: GDN (fixed)
    - Layers [1, 2]: ROUTED per-token (GDN or Softmax)
    - Layer [3]: Softmax (fixed)
    
    Routing is parallel (all decisions made at layer 0)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Create GDN config for FLA layers
        gdn_config = GatedDeltaNetConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=1,
            num_heads=config.num_attention_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
        )
        
        # Layer 0: Fixed GDN
        self.layer_0_gdn = GatedDeltaNetForCausalLM(gdn_config).model.layers[0]
        
        # Layer 1: Both GDN and Softmax (routed)
        self.layer_1_gdn = GatedDeltaNetForCausalLM(gdn_config).model.layers[0]
        self.layer_1_attn = SoftmaxAttentionLayer(config.hidden_size, config.num_attention_heads)
        
        # Layer 2: Both GDN and Softmax (routed)
        self.layer_2_gdn = GatedDeltaNetForCausalLM(gdn_config).model.layers[0]
        self.layer_2_attn = SoftmaxAttentionLayer(config.hidden_size, config.num_attention_heads)
        
        # Layer 3: Fixed Softmax
        self.layer_3_attn = SoftmaxAttentionLayer(config.hidden_size, config.num_attention_heads)
        
        # Router network (parallel routing for layers 1 and 2)
        # Input: hidden state after layer 0
        # Output: routing logits for 2 layers × 2 choices = 4 values
        self.router = nn.Linear(config.hidden_size, 2 * 2)  # 2 layers, 2 choices each
        
        # Final norm and LM head
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.vocab_size, config.vocab_size, bias=False)
        
        # Routing statistics (for logging)
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
        
        # Linear annealing
        progress = step / self.config.temperature_anneal_steps
        temp = self.config.gumbel_temperature - (self.config.gumbel_temperature - self.config.min_temperature) * progress
        return max(temp, self.config.min_temperature)
    
    def forward(self, input_ids, labels=None, step=0):
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embed_tokens(input_ids)
        
        # Layer 0: Fixed GDN
        x = self.layer_0_gdn(x)[0]
        
        # === PARALLEL ROUTING ===
        # Compute routing for layers 1 and 2 based on layer 0 output
        router_logits = self.router(x)  # [batch, seq, 4]
        router_logits = router_logits.view(batch_size, seq_len, 2, 2)  # [batch, seq, 2 layers, 2 choices]
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get routing decisions with Gumbel-Softmax
        temperature = self.get_temperature(step)
        route_layer_1 = gumbel_softmax(router_logits[:, :, 0, :], temperature=temperature, hard=True)
        route_layer_2 = gumbel_softmax(router_logits[:, :, 1, :], temperature=temperature, hard=True)
        
        # Update routing statistics
        if self.training:
            with torch.no_grad():
                self.routing_stats['layer_1_gdn_count'] += (route_layer_1[..., 0].sum().item())
                self.routing_stats['layer_1_attn_count'] += (route_layer_1[..., 1].sum().item())
                self.routing_stats['layer_2_gdn_count'] += (route_layer_2[..., 0].sum().item())
                self.routing_stats['layer_2_attn_count'] += (route_layer_2[..., 1].sum().item())
                self.routing_stats['total_tokens'] += batch_size * seq_len
        
        # === LAYER 1: ROUTED ===
        out_1_gdn = self.layer_1_gdn(x)[0]
        out_1_attn = self.layer_1_attn(x)
        x = (route_layer_1[..., 0:1] * out_1_gdn + 
             route_layer_1[..., 1:2] * out_1_attn)
        
        # === LAYER 2: ROUTED ===
        out_2_gdn = self.layer_2_gdn(x)[0]
        out_2_attn = self.layer_2_attn(x)
        x = (route_layer_2[..., 0:1] * out_2_gdn + 
             route_layer_2[..., 1:2] * out_2_attn)
        
        # === LAYER 3: FIXED SOFTMAX ===
        x = self.layer_3_attn(x)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # === COMPUTE LOSSES ===
        loss = None
        aux_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Load balancing loss
            aux_loss = self.compute_load_balancing_loss([
                router_probs[:, :, 0, :],  # Layer 1 routing probs
                router_probs[:, :, 1, :],  # Layer 2 routing probs
            ])
            
            # Total loss
            loss = lm_loss + self.config.load_balance_alpha * aux_loss
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # Store aux_loss in past_key_values for logging (bit of a hack)
            past_key_values=(aux_loss,) if aux_loss is not None else None,
        )
    
    def compute_load_balancing_loss(self, router_probs_list):
        """
        Load balancing loss to prevent routing collapse
        Encourages 50/50 split for each layer
        
        From Switch Transformer paper
        """
        total_loss = 0.0
        num_experts = 2
        
        for router_probs in router_probs_list:
            # router_probs: [batch, seq, 2]
            # Fraction of tokens assigned to each expert
            expert_mask = F.one_hot(router_probs.argmax(dim=-1), num_classes=2).float()
            fraction_per_expert = expert_mask.mean(dim=[0, 1])  # [2]
            
            # Average router probability
            avg_prob_per_expert = router_probs.mean(dim=[0, 1])  # [2]
            
            # Load balance loss
            layer_loss = num_experts * (fraction_per_expert * avg_prob_per_expert).sum()
            total_loss += layer_loss
        
        return total_loss / len(router_probs_list)
    
    def get_routing_stats(self, reset=True):
        """Get and optionally reset routing statistics"""
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
        print("Creating DynamicRoutingModel")
        return DynamicRoutingModel(config)
    else:
        print("Creating BaselineHybridModel (static)")
        return BaselineHybridModel(config)
