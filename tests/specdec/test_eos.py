"""Tier 0 — EOS truncation of accepted spans.

Tests ``minisgl.specdec.truncate_at_eos``, a pure-arithmetic seam.
"""

from __future__ import annotations

import pytest

from minisgl.specdec import truncate_at_eos

pytestmark = pytest.mark.cpu

EOS_ID = 2


def test_no_eos_in_span() -> None:
    tokens = [10, 11, 12, 13]
    result, finished = truncate_at_eos(tokens, EOS_ID)
    assert list(result) == tokens
    assert finished is False


def test_eos_at_start() -> None:
    tokens = [2, 11, 12, 13]
    result, finished = truncate_at_eos(tokens, EOS_ID)
    assert list(result) == []
    assert finished is True


def test_eos_in_middle() -> None:
    tokens = [10, 11, 2, 13, 14]
    result, finished = truncate_at_eos(tokens, EOS_ID)
    assert list(result) == [10, 11]
    assert finished is True


def test_eos_at_end() -> None:
    tokens = [10, 11, 12, 2]
    result, finished = truncate_at_eos(tokens, EOS_ID)
    assert list(result) == [10, 11, 12]
    assert finished is True


def test_empty_span() -> None:
    tokens: list[int] = []
    result, finished = truncate_at_eos(tokens, EOS_ID)
    assert list(result) == []
    assert finished is False