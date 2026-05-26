"""First Modal milestone: prove the GPU harness works (image + GPU + CUDA kernels).

These are the "deliberately-passing canary" from ``docs/specdec_testing.md`` section 8 —
they intentionally test no DFlash logic. If they go green inside
``modal run modal_tests/run_suite.py``, the Modal plumbing is figured out and the
``gpu`` / ``modal`` tiers can be built on top with the xfail-stub pattern.

Marked ``gpu`` ; ``tiny_llm`` (session fixture in conftest.py) handles the CUDA
availability gate — it creates the Engine first, before any CUDA ops run.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_torch_sees_gpu(tiny_llm) -> None:
    import torch

    assert torch.cuda.is_available(), "no CUDA device visible in the container"
    x = torch.ones(8, device="cuda")
    assert float((x + x).sum().item()) == 16.0
    print("GPU:", torch.cuda.get_device_name(0))


def test_cuda_kernels_importable(tiny_llm) -> None:
    import flashinfer  # noqa: F401
    from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401