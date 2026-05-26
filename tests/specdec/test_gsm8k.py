"""Tier 5 — quality parity on real tasks (GSM8K).

Compares spec vs baseline accuracy on GSM8K. Greedy ⇒ should be *identical*.
Catches real-workload integration drift.

Requires real weights + GPU → Modal tier, ``slow`` marker.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.modal, pytest.mark.slow]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: GSM8K parity not implemented yet",
)
def test_gsm8k_parity() -> None:
    raise NotImplementedError("skeleton: GSM8K parity not implemented yet")