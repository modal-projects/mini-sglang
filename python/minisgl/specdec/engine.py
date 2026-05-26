"""Speculative-decoding LLM engine. Depends on torch/CUDA; import only in GPU tiers."""

from __future__ import annotations

from typing import Dict, List

import torch

from minisgl.core import get_global_ctx
from minisgl.llm import LLM

from .draft import DFlashDraftConfig, DFlashDraftModel


class SpecDecLLM(LLM):
    """LLM with DFlash speculative decoding.

    Initializes the draft model and registers target hidden-state capture hooks
    on the engine. The actual spec-dec loop is not yet implemented — generate()
    delegates to the parent. Tests are marked xfail(NotImplementedError) until
    the loop is wired.
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        draft_config: DFlashDraftConfig | None = None,
        **kwargs,
    ):
        super().__init__(model_path, dtype=dtype, **kwargs)
        target_cfg = self.engine.config.model_config
        if draft_config is None:
            draft_config = DFlashDraftConfig.tiny(
                target_hidden_size=target_cfg.hidden_size,
                target_num_layers=target_cfg.num_layers,
                vocab_size=target_cfg.vocab_size,
            )
        self.draft_config = draft_config
        self.draft_model = DFlashDraftModel(draft_config).to(self.device).to(self.engine.dtype)
        self.draft_model.eval()
        self.ctx = get_global_ctx()
        self.ctx.capture_layers = set(draft_config.effective_capture_layers())

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List | "SamplingParams",
    ) -> List[Dict[str, str | List[int]]]:
        raise NotImplementedError("skeleton: SpecDecLLM.generate not implemented yet")