"""Offline LLM with speculative decoding support."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.spec_dec import SpecDecScheduler


class RequestAllFinished(Exception):
    pass


class SpecDecLLM(SpecDecScheduler):
    """Offline LLM with DFlash speculative decoding.

    Usage:
        llm = SpecDecLLM(
            model_path="Qwen/Qwen3-8B",
            draft_model_path="z-lab/Qwen3-8B-DFlash-b16",
        )
        results = llm.generate(["What is 2 + 2?"], SamplingParams(temperature=0, max_tokens=50))
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self._requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self._status: Dict[int, List[int]] = {}
        self._counter = 0

    def _tokenize(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and not self._requests:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, total_len = 0, 0
        for prompt, sp in self._requests:
            if total_len >= self.prefill_budget:
                break
            ids = self._tokenize(prompt)
            total_len += len(ids)
            uid = self._counter + added
            added += 1
            results.append(UserMsg(uid=uid, input_ids=ids, sampling_params=sp))
            self._status[uid] = []
        self._counter += added
        self._requests = self._requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            if not (msg.finished and msg.next_token == self.eos_token_id):
                self._status[msg.uid].append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        self._requests = []
        self._status = {}
        self._counter = 0
        self._pending.clear()
        self._hiddens.clear()
        self._capture_enabled = False
        self.engine.ctx.capture_hidden_layers = None
        self.engine.ctx.captured_hidden_states = []

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self._requests.append((prompt, sp))

        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            ids = self._status[i]
            text = self.tokenizer.decode(ids)
            results.append({"text": text, "token_ids": ids})
        return results