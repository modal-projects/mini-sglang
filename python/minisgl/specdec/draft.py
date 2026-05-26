"""DFlash draft model — config-sizable cross-attention decoder.

The draft model cross-attends to target hidden states captured from specific
target layers. Design follows R2 from ``docs/specdec_testing.md``: the model is
driven by ``DFlashDraftConfig`` — never by ``AutoModel.from_pretrained`` — so
tiny-random instances work locally with zero downloads.

Uses standard PyTorch ``nn.Module`` (not minisgl BaseOP) for simplicity and
portability. The draft is lightweight (~1B for the real weights) and this
module deliberately avoids minisgl engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DFlashDraftConfig:
    block_size: int = 16
    mask_token_id: int = 151669

    num_layers: int = 5
    hidden_size: int = 1024
    num_heads: int = 16
    head_dim: int = 64
    intermediate_size: int = 2816

    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    target_hidden_size: int = 4096
    target_num_layers: int = 36
    target_capture_layers: Tuple[int, ...] = (1, 9, 17, 25, 33)
    max_position: int = 32768
    rope_base: float = 1000000.0

    def effective_capture_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.target_capture_layers if i < self.target_num_layers)

    @property
    def target_total_hidden(self) -> int:
        return len(self.effective_capture_layers()) * self.target_hidden_size

    @classmethod
    def tiny(cls, target_hidden_size: int = 128, target_num_layers: int = 2, vocab_size: int = 32000) -> DFlashDraftConfig:
        return cls(
            block_size=16,
            mask_token_id=151669,
            num_layers=2,
            hidden_size=128,
            num_heads=4,
            head_dim=32,
            intermediate_size=384,
            vocab_size=vocab_size,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            target_hidden_size=target_hidden_size,
            target_num_layers=target_num_layers,
            target_capture_layers=tuple(range(target_num_layers)),
            max_position=256,
            rope_base=10000.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight.float().to(dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x. Expects shape (B, H, T, D)."""
    x_rot = x[..., : cos.shape[-1]]
    x_pass = x[..., cos.shape[-1] :]
    x_rot = (x_rot.float() * cos + _rotate_half(x_rot.float()) * sin).to(x.dtype)
    return torch.cat((x_rot, x_pass), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _precompute_freqs(head_dim: int, max_position: int, base: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_position, device=device).float()
    freqs = torch.outer(positions, theta)
    return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)


class DraftSelfAttention(nn.Module):
    def __init__(self, config: DFlashDraftConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        cos, sin = _precompute_freqs(config.head_dim, config.max_position, config.rope_base, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos = self.cos[:, :, positions, :].to(device=x.device, dtype=x.dtype)
        sin = self.sin[:, :, positions, :].to(device=x.device, dtype=x.dtype)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        o = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(o)


class DraftCrossAttention(nn.Module):
    def __init__(self, config: DFlashDraftConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.target_total_hidden, config.num_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.target_total_hidden, config.num_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(target_hidden).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(target_hidden).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        o = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(o)


class DraftMLP(nn.Module):
    def __init__(self, config: DFlashDraftConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DraftDecoderLayer(nn.Module):
    def __init__(self, config: DFlashDraftConfig, layer_id: int):
        super().__init__()
        self.self_attn = DraftSelfAttention(config)
        self.cross_attn = DraftCrossAttention(config)
        self.mlp = DraftMLP(config)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x), positions, attn_mask)
        x = x + self.cross_attn(self.post_self_attn_norm(x), target_hidden)
        x = x + self.mlp(self.post_cross_attn_norm(x))
        return self.post_mlp_norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Draft model
# ═══════════════════════════════════════════════════════════════════════════════


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashDraftConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.target_proj = nn.Linear(config.target_total_hidden, config.hidden_size, bias=False)
        self.target_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DraftDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        projected = self.target_proj(target_hidden)
        return self.target_norm(projected)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        projected_target = self.project_target_hidden(target_hidden)
        for layer in self.layers:
            x = layer(x, positions, projected_target, attn_mask)
        x = self.final_norm(x)
        return self.lm_head(x)


def create_causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)