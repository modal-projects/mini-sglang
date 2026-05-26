"""Pure, model-free speculative-decoding logic. Kept light on purpose (no torch/CUDA)."""

from __future__ import annotations

from .logic import verify_block

__all__ = ["verify_block"]
