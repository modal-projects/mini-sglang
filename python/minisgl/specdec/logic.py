"""Pure, model-free speculative-decoding logic (Tier 0).

These functions deliberately avoid torch / CUDA imports so they run on CPU in
milliseconds and survive any rewrite of the engine or scheduler internals. They are the
"pure arithmetic" anchor described in ``docs/specdec_testing.md``.

Living in the light ``minisgl.specdec`` namespace package (not under
``minisgl.scheduler``) is intentional: importing this module must not pull in the CUDA
engine, so ``pytest -m cpu`` stays GPU-free.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def verify_block(
    draft_tokens: Sequence[int],
    target_greedy: Sequence[int],
) -> Tuple[int, int]:
    """Greedy acceptance rule for a verified draft block.

    Returns ``(accept_len, bonus_token)`` where:

    * ``accept_len`` is the length of the longest prefix for which
      ``draft_tokens[i] == target_greedy[i]``, and
    * ``bonus_token`` is ``target_greedy[accept_len]`` — the token the target itself
      would emit at the first divergence. It is always appended, which is why spec
      decoding emits at least one token per verify pass even when nothing is accepted.

    ``target_greedy`` must be exactly one element longer than ``draft_tokens``
    (the extra position supplies the bonus token at the first divergence).
    """
    for i, dt in enumerate(draft_tokens):
        if dt != target_greedy[i]:
            return i, target_greedy[i]
    accept_len = len(draft_tokens)
    return accept_len, target_greedy[accept_len]


def build_draft_block(
    last_token: int,
    block_size: int,
    mask_id: int,
) -> Tuple[List[int], List[int]]:
    """Build a draft block for speculative decoding.

    Returns ``(token_ids, position_ids)``:

    * ``token_ids`` = ``[last_token] + [mask_id] * (block_size - 1)``
    * ``position_ids`` is a contiguous range of ``block_size`` integers.
    """
    token_ids = [last_token] + [mask_id] * (block_size - 1)
    positions = list(range(block_size))
    return token_ids, positions


def compute_keep_len(cached_len: int, accept_len: int) -> int:
    """Compute the target KV cache length to keep after accepting *accept_len* tokens.

    The KV cache is cropped to ``keep_len`` after the verify pass:
    *keep_len = cached_len + accept_len + 1* (the +1 is the bonus token).
    """
    return cached_len + accept_len + 1


def compute_rollback_state(
    cached_len: int,
    device_len: int,
    accept_len: int,
) -> Tuple[int, int, int]:
    """Return the post-verify Req state as ``(new_cached_len, new_device_len, keep_len)``.

    After accepting ``accept_len`` draft tokens + 1 bonus:
    - ``keep_len`` (what stays in KV cache) = ``cached_len + accept_len + 1``
    - ``new_cached_len`` = ``keep_len``
    - ``new_device_len`` = ``keep_len``

    All three are the same value for a standard step; kept distinct for clarity.
    """
    keep = cached_len + accept_len + 1
    return keep, keep, keep


def truncate_at_eos(
    accepted_tokens: Sequence[int],
    eos_token_id: int,
) -> Tuple[List[int], bool]:
    """Truncate an accepted span at the first EOS and report finished status.

    Returns ``(truncated_tokens, finished)`` where ``truncated_tokens`` is
    ``accepted_tokens`` up to (but not including) the first ``eos_token_id``, and
    ``finished`` is True if an EOS was found.
    """
    for i, t in enumerate(accepted_tokens):
        if t == eos_token_id:
            return list(accepted_tokens[:i]), True
    return list(accepted_tokens), False