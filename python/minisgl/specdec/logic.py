"""Pure, model-free speculative-decoding logic (Tier 0).

These functions deliberately avoid torch / CUDA imports so they run on CPU in
milliseconds and survive any rewrite of the engine or scheduler internals. They are the
"pure arithmetic" anchor described in ``docs/specdec_testing.md``.

Living in the light ``minisgl.specdec`` namespace package (not under
``minisgl.scheduler``) is intentional: importing this module must not pull in the CUDA
engine, so ``pytest -m cpu`` stays GPU-free.
"""

from __future__ import annotations

from typing import Sequence, Tuple


def verify_block(
    draft_tokens: Sequence[int],  # length B: draft's guesses for verify positions 1..B
    target_greedy: Sequence[int],  # length B+1: target argmax over the verify block
) -> Tuple[int, int]:
    """Greedy acceptance rule for a verified draft block.

    Returns ``(accept_len, bonus_token)`` where:

    * ``accept_len`` is the length of the longest prefix for which
      ``draft_tokens[i] == target_greedy[i]``, and
    * ``bonus_token`` is ``target_greedy[accept_len]`` — the token the target itself
      would emit at the first divergence. It is always appended, which is why spec
      decoding emits at least one token per verify pass even when nothing is accepted.
    """
    raise NotImplementedError("Tier 0 seam: implement the greedy acceptance rule")
