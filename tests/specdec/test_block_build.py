"""Tier 0 — draft block construction.

Tests ``minisgl.specdec.build_draft_block``, a pure-arithmetic seam.
"""

from __future__ import annotations

import pytest

from minisgl.specdec import build_draft_block

pytestmark = pytest.mark.cpu

MASK_ID = 151669
BLOCK_SIZE = 16


def test_seam_is_importable() -> None:
    assert callable(build_draft_block)


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: build_draft_block not implemented yet",
)
def test_first_token_is_last_token() -> None:
    last_token = 42
    ids, _ = build_draft_block(last_token, BLOCK_SIZE, MASK_ID)
    assert ids[0] == last_token


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: build_draft_block not implemented yet",
)
def test_rest_are_mask_tokens() -> None:
    last_token = 7
    ids, _ = build_draft_block(last_token, BLOCK_SIZE, MASK_ID)
    assert ids[0] == last_token
    assert all(t == MASK_ID for t in ids[1:])


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: build_draft_block not implemented yet",
)
def test_output_lengths_match_block_size() -> None:
    ids, positions = build_draft_block(0, BLOCK_SIZE, MASK_ID)
    assert len(ids) == BLOCK_SIZE
    assert len(positions) == BLOCK_SIZE


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: build_draft_block not implemented yet",
)
def test_positions_are_contiguous() -> None:
    _, positions = build_draft_block(0, BLOCK_SIZE, MASK_ID)
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    assert all(d == 1 for d in diffs)


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: build_draft_block not implemented yet",
)
@pytest.mark.parametrize("block_size", [1, 4, 16, 32])
def test_variable_block_size(block_size: int) -> None:
    ids, positions = build_draft_block(99, block_size, MASK_ID)
    assert len(ids) == block_size
    assert len(positions) == block_size
    assert ids[0] == 99
    if block_size > 1:
        assert all(t == MASK_ID for t in ids[1:])