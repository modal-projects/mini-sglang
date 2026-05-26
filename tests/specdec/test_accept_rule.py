"""Tier 0 walking-skeleton demo: the greedy acceptance rule.

Written end-to-end against the real seam (``minisgl.specdec.verify_block``). While that
seam is a stub it raises ``NotImplementedError`` -> the parametrized test ``xfail``s ->
the suite is GREEN. That green run is the signal the plumbing is wired (import path,
signature, parametrization all resolve). The moment ``verify_block`` is implemented
correctly the test passes; ``strict=True`` then turns the unexpected pass into a failure
that says "remove the xfail marker." Implement it *wrong* and it raises
``AssertionError`` (not ``NotImplementedError``) -> a hard failure.

See ``docs/specdec_testing.md`` section 8.
"""

from __future__ import annotations

import pytest

from minisgl.specdec import verify_block

pytestmark = pytest.mark.cpu


def test_seam_is_importable() -> None:
    # Canary: a deliberately-passing test so a green run proves collection actually ran
    # the skeleton (distinguishes "ran and xfailed" from "collected nothing").
    assert callable(verify_block)


@pytest.mark.parametrize(
    "draft, target, expect",
    [
        ([10, 11, 12, 13], [10, 11, 12, 13, 14], (4, 14)),  # all accepted -> +bonus token
        ([99, 11, 12, 13], [10, 11, 12, 13, 14], (0, 10)),  # diverge at position 0
        ([10, 11, 77, 13], [10, 11, 12, 13, 14], (2, 12)),  # diverge mid-block
        ([10, 11, 12, 13], [10, 11, 12, 13, 99], (4, 99)),  # all accepted, different bonus
    ],
)
def test_accept_rule(draft, target, expect) -> None:
    assert verify_block(draft, target) == expect
