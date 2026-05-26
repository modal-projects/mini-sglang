"""Tier 6 — performance / non-regression.

Soft metrics: speedup, acceptance-length distribution, baseline non-regression.

Requires real weights + GPU → Modal tier, ``slow`` marker.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.modal, pytest.mark.slow]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: speedup test not implemented yet",
)
def test_speedup() -> None:
    raise NotImplementedError("skeleton: speedup test not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: accept_len distribution not implemented yet",
)
def test_accept_len_distribution() -> None:
    raise NotImplementedError("skeleton: accept_len distribution not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: baseline non-regression not implemented yet",
)
def test_baseline_no_regression() -> None:
    raise NotImplementedError("skeleton: baseline non-regression not implemented yet")