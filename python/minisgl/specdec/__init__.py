"""Pure, model-free speculative-decoding logic. Kept light on purpose (no torch/CUDA)."""

from __future__ import annotations

from .logic import (
    build_draft_block,
    compute_keep_len,
    compute_rollback_state,
    truncate_at_eos,
    verify_block,
)

__all__ = [
    "build_draft_block",
    "compute_keep_len",
    "compute_rollback_state",
    "truncate_at_eos",
    "verify_block",
]