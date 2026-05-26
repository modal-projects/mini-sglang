"""Tier 4 — sampling correctness.

OUT OF SCOPE: this implementation is **greedy-only**. Kept as a skipped placeholder so the
decision is recorded in the suite rather than silently absent (see
``docs/specdec_testing.md`` section 4). If temperature > 0 support is ever added, drop the
``skip`` marker and implement the body.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.modal, pytest.mark.slow]


@pytest.mark.skip(reason="out of scope: greedy-only implementation")
def test_sampling_distribution() -> None:
    raise NotImplementedError("Tier 4 sampling correctness is out of scope (greedy-only)")
