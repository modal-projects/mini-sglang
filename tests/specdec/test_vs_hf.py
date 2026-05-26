"""Tier 2 — numeric fidelity vs HF z-lab reference.

Ground truth = the HF ``z-lab/Qwen3-8B-DFlash-b16`` model run *live* (or cached to
a Modal Volume). Never asserts against committed ``.pt`` snapshots of our own output.

Requires real weights + GPU → Modal tier.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.modal]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: hidden capture vs HF not implemented yet",
)
def test_hidden_capture_vs_hf() -> None:
    raise NotImplementedError("skeleton: hidden capture vs HF not implemented yet")


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: draft forward vs HF not implemented yet",
)
def test_draft_forward_vs_hf() -> None:
    raise NotImplementedError("skeleton: draft forward vs HF not implemented yet")