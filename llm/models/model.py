"""
Simple GPT-style Language Model with Standard Attention

A minimal implementation using:
- Token embeddings
- Rotary positional embeddings (RoPE)
- Multi-head attention with scaled dot-product attention
- Simple FFN (not MoE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from llm.configs.training_config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and Flash Attention"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryPositionalEmbeddings(
            dim=config.head_dim, 
            max_seq_len=config.max_seq_len
        )
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE (expects [B, T, H, D])
        Q = self.rope(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rope(K.transpose(1, 2)).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Simple feed-forward network with SwiGLU activation"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.w1 = nn.Linear(config.hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: swish(x @ W1) * (x @ W3) @ W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """Simple GPT-style language model"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
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
        
        return {"loss": loss, "logits": logits}
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_model(config: ModelConfig) -> GPTModel:
    """Create and return a GPT model"""
    model = GPTModel(config)
    print(f"✓ Created GPTModel with {model.num_parameters():,} parameters")
    print(f"  - {config.num_layers} layers, {config.num_heads} heads, {config.hidden_size} hidden size")
    return model
