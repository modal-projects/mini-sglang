"""Tier 1 — KV cache crop correctness.

Tests that cropping the KV cache preserves the head and zeroes/nulls the tail.
Needs a GPU + torch, but no real weights (dummy engine is fine).

``tiny_llm`` (session fixture in conftest.py) handles the CUDA availability gate.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_crop_preserves_head(tiny_llm) -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    num_layers, num_pages, page_size, num_heads, head_dim = 2, 64, 1, 4, 32
    kv = torch.randn(2, num_layers, num_pages, page_size, num_heads, head_dim, device="cuda")

    head_saved = kv[0, 0, 0, 0, 0, :5].clone()

    crop_kv_cache(kv, keep_len=1)

    head_after = kv[0, 0, 0, 0, 0, :5]
    assert torch.equal(head_saved, head_after), "head should be unchanged"


def test_crop_zeros_tail(tiny_llm) -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    num_layers, num_pages, page_size, num_heads, head_dim = 1, 64, 1, 4, 32
    kv = torch.randn(2, num_layers, num_pages, page_size, num_heads, head_dim, device="cuda")

    crop_kv_cache(kv, keep_len=4)

    tail = kv[:, 0, 4:, 0, :, :]
    assert torch.all(tail == 0), f"tail should be zeroed, got mean={tail.abs().mean().item():.4f}"


def test_reuse_after_crop_no_leakage(tiny_llm) -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    kv = torch.zeros(2, 1, 64, 1, 4, 32, device="cuda")
    kv[:, 0:1, :16, 0:1, :, :] = torch.randn(2, 1, 16, 1, 4, 32, device="cuda")

    crop_kv_cache(kv, keep_len=4)

    kv[:, 0:1, 4:8, 0:1, :, :] = 42.0

    assert not torch.any(kv[:, 0:1, 4:8, 0:1, :, :] != 42.0)
    assert torch.all(kv[:, 0:1, 8:, 0:1, :, :] == 0), "refilled region should not leak beyond refill"