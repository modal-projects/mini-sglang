"""Tier 1 — multi-request speculative decoding identity.

Tests that 2+ concurrent requests with mixed lengths each produce the same tokens
as their individual AR baselines. Kills the ``batch.size == 1`` corner case.
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
def test_two_concurrent_mixed_lengths(tiny_model_path: str, tiny_spec_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        model_path=tiny_model_path,
        use_dummy_weight=True,
        max_seq_len_override=128,
        num_page_override=256,
    )

    prompts = [[1, 10, 20, 30], [1, 100, 200]]
    sp = SamplingParams(max_tokens=6, ignore_eos=True)

    ar_results = llm.generate(prompts, [sp, sp])
    spec_results = tiny_spec_llm.generate(prompts, [sp, sp])

    for i in range(len(prompts)):
        assert ar_results[i]["token_ids"] == spec_results[i]["token_ids"], (
            f"req {i}: spec {spec_results[i]['token_ids']} != ar {ar_results[i]['token_ids']}"
        )


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="skeleton: SpecDecLLM.generate not implemented yet",
)
def test_two_concurrent_same_length(tiny_model_path: str, tiny_spec_llm) -> None:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        model_path=tiny_model_path,
        use_dummy_weight=True,
        max_seq_len_override=128,
        num_page_override=256,
    )

    prompts = [[1, 10, 20, 30, 40], [1, 10, 20, 30, 40]]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)

    ar_results = llm.generate(prompts, [sp, sp])
    spec_results = tiny_spec_llm.generate(prompts, [sp, sp])

    for i in range(len(prompts)):
        assert ar_results[i]["token_ids"] == spec_results[i]["token_ids"]