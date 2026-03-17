from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    seq_len: int = 256
    d_model: int = 128
    n_layers: int = 7
    n_heads: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / (D ** 0.5)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B,H,T,T)

        # causal mask: disallow attending to future positions
        mask = mx.tril(mx.ones((T, T)))
        # apply mask: set -inf where mask==0
        att = att + (mask[None, None, :, :] - 1.0) * 1e10
        att = mx.softmax(att, axis=-1)
        att = self.dropout(att)
        y = att @ v  # (B,H,T,D)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # (B,T,C)
        y = self.out_proj(y)  # (B,T,C)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Weight tying: lm_head weight tied with token embeddings
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

    def __call__(self, idx: mx.array) -> mx.array:
        # idx: (B, T) integer token ids
        B, T = idx.shape
        assert T <= self.cfg.seq_len, "Sequence length exceeds model maximum"

        pos = mx.arange(T)[None, :]  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def count_parameters(model_or_cfg) -> int:
    if isinstance(model_or_cfg, ModelConfig):
        cfg = model_or_cfg
    else:
        cfg = getattr(model_or_cfg, "cfg", None)
        if cfg is None:
            raise ValueError("count_parameters expects a GPT model or ModelConfig")

    C = cfg.d_model
    L = cfg.n_layers
    V = cfg.vocab_size
    T = cfg.seq_len

    # Embeddings (token + positional)
    emb = V * C + T * C
    # Per-block params: attention (q,k,v,out) + MLP + two LayerNorms
    # Attention: 4*(C*C + C)
    attn = 4 * (C * C + C)
    # MLP: (C*4C + 4C) + (4C*C + C)
    mlp = (C * 4 * C + 4 * C) + (4 * C * C + C)
    # LayerNorms: two per block, each with gamma+beta of length C
    norms = 2 * (2 * C)
    per_block = attn + mlp + norms
    blocks_total = L * per_block
    # Final LayerNorm
    final_ln = 2 * C
    # LM head is tied to tok_emb (no extra params)
    return emb + blocks_total + final_ln


