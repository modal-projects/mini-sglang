"""KV cache utilities for speculative decoding.

These functions need torch/CUDA. Import only in GPU tiers, not from
the pure-logic ``minisgl.specdec.logic`` module.
"""

from __future__ import annotations

import torch


def crop_kv_cache(kv_buffer: torch.Tensor, keep_len: int) -> None:
    """Crop the KV cache to ``keep_len`` entries in-place.

    Zeros out everything beyond position ``keep_len`` across all layers
    and all KV channels, while preserving the head (positions [0, keep_len)).

    Args:
        kv_buffer: Shape ``(2, num_layers, num_pages, page_size, num_heads, head_dim)``.
        keep_len: Number of token positions to retain.
    """
    raise NotImplementedError("skeleton: crop_kv_cache not implemented yet")