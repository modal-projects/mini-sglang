"""Tier 1 — token-for-token identity with tiny random models.

The north-star invariant: *greedy spec decode == greedy AR decode, token-for-token,
for ANY draft model.* This proof-of-concept test exercises the entire spec loop with
a tiny 2-layer Qwen3 target + a trivial draft, zero downloads.
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, AssertionError):
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")]


@pytest.fixture(scope="module")
def tiny_llm(tiny_model_path: str):
    from minisgl.llm import LLM

    return LLM(
        model_path=tiny_model_path,
        use_dummy_weight=True,
        max_seq_len_override=128,
        num_page_override=256,
    )


@pytest.fixture(scope="module")
def tiny_spec_llm(tiny_model_path: str):
    from minisgl.specdec.engine import SpecDecLLM

    return SpecDecLLM(
        model_path=tiny_model_path,
        use_dummy_weight=True,
        max_seq_len_override=128,
        num_page_override=256,
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_spec_eq_ar_single_prompt(tiny_llm, tiny_spec_llm) -> None:
    from minisgl.core import SamplingParams

    prompt = [1, 100, 200, 300, 400, 500]
    sp = SamplingParams(max_tokens=8, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = tiny_spec_llm.generate([prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result[0]["token_ids"], (
        f"spec {spec_result[0]['token_ids']} != ar {ar_result[0]['token_ids']}"
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_spec_eq_ar_short_prompt(tiny_llm, tiny_spec_llm) -> None:
    from minisgl.core import SamplingParams

    prompt = [1, 50, 100]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = tiny_spec_llm.generate([prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result[0]["token_ids"]


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_spec_eq_ar_medium_prompt(tiny_llm, tiny_spec_llm) -> None:
    from minisgl.core import SamplingParams

    prompt = list(range(1, 41))
    sp = SamplingParams(max_tokens=16, ignore_eos=True)

    ar_result = tiny_llm.generate([prompt], sp)
    spec_result = tiny_spec_llm.generate([prompt], sp)

    assert ar_result[0]["token_ids"] == spec_result[0]["token_ids"]