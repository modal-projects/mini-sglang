"""First Modal milestone: prove the GPU harness works (image + GPU + CUDA kernels).

These are the "deliberately-passing canary" from ``docs/specdec_testing.md`` section 8 —
they intentionally test no DFlash logic. If they go green inside
``modal run modal_tests/run_suite.py``, the Modal plumbing is figured out and the
``gpu`` / ``modal`` tiers can be built on top with the xfail-stub pattern.

Marked ``gpu`` (needs CUDA, no downloads); skipped automatically where no GPU is present,
so a local ``pytest -m cpu`` run is unaffected.
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# Skip (don't fail) when there's simply no GPU here — the point of this tier is to run
# *inside Modal*. A missing kernel/dep there still fails loudly via the imports below.
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")]


def test_torch_sees_gpu() -> None:
    import torch

    assert torch.cuda.is_available(), "no CUDA device visible in the container"
    x = torch.ones(8, device="cuda")
    assert float((x + x).sum().item()) == 16.0  # a real op actually executes on the GPU
    print("GPU:", torch.cuda.get_device_name(0))


def test_cuda_kernels_importable() -> None:
    # The parts of the image most likely to be wrong: the compiled CUDA kernels.
    import flashinfer  # noqa: F401
    from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
