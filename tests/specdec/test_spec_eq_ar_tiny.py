"""Tier 1 — token-for-token identity with tiny random models.

The north-star invariant: *greedy spec decode == greedy AR decode, token-for-token,
for ANY draft model.* This proof-of-concept test exercises the entire spec loop with
a tiny 2-layer Qwen3 target + a trivial draft, zero downloads.

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
def test_spec_eq_ar_single_prompt(tiny_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.specdec.engine import SpecDecLLM

    prompt = [1, 100, 200, 300, 400, 500]
    sp = SamplingParams(max_tokens=8, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = SpecDecLLM.generate(None, [prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_spec_eq_ar_short_prompt(tiny_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.specdec.engine import SpecDecLLM

    prompt = [1, 50, 100]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = SpecDecLLM.generate(None, [prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_spec_eq_ar_medium_prompt(tiny_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.specdec.engine import SpecDecLLM

    prompt = list(range(1, 41))
    sp = SamplingParams(max_tokens=16, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = SpecDecLLM.generate(None, [prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result