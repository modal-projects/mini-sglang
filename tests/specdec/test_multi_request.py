"""Tier 1 — multi-request speculative decoding identity.

Tests that 2+ concurrent requests with mixed lengths each produce the same tokens
as their individual AR baselines. Kills the ``batch.size == 1`` corner case.

``tiny_llm`` is a session-scoped fixture in ``conftest.py`` — only one ``Engine``
can live per process (``assert not cuda.is_initialized()``).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_two_concurrent_mixed_lengths(tiny_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.specdec.engine import SpecDecLLM

    prompts = [[1, 10, 20, 30], [1, 100, 200]]
    sp = SamplingParams(max_tokens=6, ignore_eos=True)

    ar_results = tiny_llm.generate(prompts, [sp, sp])
    spec_results = SpecDecLLM.generate(None, prompts, [sp, sp])

    for i in range(len(prompts)):
        assert ar_results[i]["token_ids"] == spec_results[i], (
            f"req {i}: spec {spec_results[i]} != ar {ar_results[i]['token_ids']}"
        )


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_two_concurrent_same_length(tiny_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.specdec.engine import SpecDecLLM

    prompts = [[1, 10, 20, 30, 40], [1, 10, 20, 30, 40]]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)

    ar_results = tiny_llm.generate(prompts, [sp, sp])
    spec_results = SpecDecLLM.generate(None, prompts, [sp, sp])

    for i in range(len(prompts)):
        assert ar_results[i]["token_ids"] == spec_results[i]