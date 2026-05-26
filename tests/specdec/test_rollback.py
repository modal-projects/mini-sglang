"""Tier 0 — KV cache rollback bookkeeping.

Tests ``minisgl.specdec.compute_keep_len`` and ``compute_rollback_state``.
These are pure-arithmetic seams: simulate a multi-block sequence and verify
the bookkeeping arithmetic.
"""

from __future__ import annotations

import pytest

from minisgl.specdec import compute_keep_len, compute_rollback_state

pytestmark = pytest.mark.cpu


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_keep_len not implemented yet",
)
def test_keep_len_is_cached_plus_accept_plus_one() -> None:
    assert compute_keep_len(cached_len=10, accept_len=4) == 15


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_keep_len not implemented yet",
)
def test_keep_len_when_no_tokens_accepted() -> None:
    assert compute_keep_len(cached_len=5, accept_len=0) == 6


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_keep_len not implemented yet",
)
def test_keep_len_when_full_block_accepted() -> None:
    assert compute_keep_len(cached_len=100, accept_len=16) == 117


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_rollback_state not implemented yet",
)
def test_rollback_state_basic() -> None:
    new_cached, new_device, keep = compute_rollback_state(
        cached_len=10, device_len=20, accept_len=4
    )
    assert keep == 15
    assert new_cached == keep
    assert new_device == keep


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_rollback_state not implemented yet",
)
def test_rollback_state_no_accept() -> None:
    new_cached, new_device, keep = compute_rollback_state(
        cached_len=5, device_len=5, accept_len=0
    )
    assert keep == 6
    assert new_cached == keep
    assert new_device == keep


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: compute_rollback_state not implemented yet",
)
def test_multi_step_sequence_consistent() -> None:
    """Simulate three verify passes and check token counts stay consistent."""
    cached = 4
    for _ in range(3):
        keep = compute_keep_len(cached, accept_len=3)
        assert keep == cached + 4
        cached = keep
    assert cached == 4 + 3 * 4