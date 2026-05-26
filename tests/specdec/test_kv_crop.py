"""Tier 1 — KV cache crop correctness.

Tests that cropping the KV cache preserves the head and zeroes/nulls the tail.
Needs a GPU + torch, but no real weights (dummy engine is fine).
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, AssertionError):
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: KV crop not implemented yet",
)
def test_crop_preserves_head() -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    num_layers, num_pages, page_size, num_heads, head_dim = 2, 64, 1, 4, 32
    kv = torch.randn(2, num_layers, num_pages, page_size, num_heads, head_dim, device="cuda")

    head_saved = kv[0, 0, 0, 0, 0, :5].clone()

    crop_kv_cache(kv, keep_len=1)

    head_after = kv[0, 0, 0, 0, 0, :5]
    assert torch.equal(head_saved, head_after), "head should be unchanged"


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: KV crop not implemented yet",
)
def test_crop_zeros_tail() -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    num_layers, num_pages, page_size, num_heads, head_dim = 1, 64, 1, 4, 32
    kv = torch.randn(2, num_layers, num_pages, page_size, num_heads, head_dim, device="cuda")

    crop_kv_cache(kv, keep_len=4)

    tail = kv[:, 0, 4:, 0, :, :]
    assert torch.all(tail == 0), f"tail should be zeroed, got mean={tail.abs().mean().item():.4f}"


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: KV crop not implemented yet",
)
def test_reuse_after_crop_no_leakage() -> None:
    import torch
    from minisgl.specdec.kvcache import crop_kv_cache

    kv = torch.zeros(2, 1, 64, 1, 4, 32, device="cuda")
    kv[:, 0, :16, 0, :, :] = torch.randn(2, 16, 1, 4, 32, device="cuda")

    crop_kv_cache(kv, keep_len=4)

    kv[:, 0, 4:8, 0, :, :] = 42.0

    assert not torch.any(kv[:, 0, 4:8, 0, :, :] != 42.0)
    assert torch.all(kv[:, 0, 8:, 0, :, :] == 0), "refilled region should not leak beyond refill"