"""Speculative-decoding LLM engine. Depends on torch/CUDA; import only in GPU tiers."""

from __future__ import annotations

from typing import Dict, List

from minisgl.core import SamplingParams
from minisgl.llm import LLM


class SpecDecLLM(LLM):
    """LLM with DFlash speculative decoding.

    Currently a walking-skeleton stub: ``__init__`` succeeds (normal engine init
    works), but ``generate`` raises ``NotImplementedError`` in the spec path.
    """

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        raise NotImplementedError("skeleton: SpecDecLLM.generate not implemented yet")