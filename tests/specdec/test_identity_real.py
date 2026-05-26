"""Tier 3 — end-to-end identity on the real model.

Verifies ``spec == AR`` token-for-token over a **prompt suite**:
short / medium / long, code / math / multilingual, lengths crossing
``page_size`` and cuda-graph bs buckets.

Requires real weights + GPU → Modal tier, ``slow`` marker.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.modal, pytest.mark.slow]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: real identity suite not implemented yet",
)
def test_greedy_identity_short() -> None:
    raise NotImplementedError("skeleton: real identity suite not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: real identity suite not implemented yet",
)
def test_greedy_identity_medium() -> None:
    raise NotImplementedError("skeleton: real identity suite not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: real identity suite not implemented yet",
)
def test_greedy_identity_long() -> None:
    raise NotImplementedError("skeleton: real identity suite not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: long prompt not implemented yet",
)
def test_long_prompt() -> None:
    raise NotImplementedError("skeleton: long prompt not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: cuda graph equality not implemented yet",
)
def test_cuda_graph_eq_eager() -> None:
    raise NotImplementedError("skeleton: cuda graph eq eager not implemented yet")