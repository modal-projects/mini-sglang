"""Shared pytest config for the DFlash speculative-decoding suite.

Registers the placement markers that grade the ladder (see ``docs/specdec_testing.md``)::

    cpu   - pure logic; no GPU, no weights
    gpu   - needs CUDA but no large downloads (dummy weights)
    modal - needs real weights + GPU (run via modal_tests/run_suite.py)
    slow  - minutes-scale; opt-in
"""

from __future__ import annotations

import pytest

_MARKERS = {
    "cpu": "pure logic; no GPU, no weights",
    "gpu": "needs CUDA but no large downloads (dummy weights)",
    "modal": "needs real weights + GPU",
    "slow": "minutes-scale; opt-in",
}


def pytest_configure(config: pytest.Config) -> None:
    for name, desc in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")
