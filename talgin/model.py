from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 20480
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 2
    ffn_hidden: int = 1792
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


def rope_cache(seq_len, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    f = torch.outer(pos, inv)
    return f.cos().to(dtype)[None, :, None, :], f.sin().to(dtype)[None, :, None, :]


def apply_rope(x, cos, sin):
    xe, xo = x[..., 0::2], x[..., 1::2]
    ye = xe * cos - xo * sin
    yo = xe * sin + xo * cos
    return torch.stack((ye, yo), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.head_dim * cfg.n_kv_heads
        self.rope_theta = cfg.rope_theta
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        cos, sin = rope_cache(t, self.head_dim, self.rope_theta, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        groups = self.n_heads // self.n_kv_heads
        if groups != 1:
            k = k.repeat_interleave(groups, dim=2)
            v = v.repeat_interleave(groups, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TalginDecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, 0.0, residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, 0.0, residual_std)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self):
        return asdict(self.config)
