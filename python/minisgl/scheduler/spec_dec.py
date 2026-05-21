"""Speculative decoding scheduler using DFlash draft model.

For single-request mode (shell mode / max_running_req=1), intercepts decode
forward passes and runs speculative decoding: draft model generates B candidate
tokens, target model verifies, accepted tokens are queued and emitted one per step.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import init_logger

from .scheduler import ForwardInput, Scheduler

if TYPE_CHECKING:
    from minisgl.engine import ForwardOutput
    from minisgl.scheduler.config import SchedulerConfig

logger = init_logger(__name__)


class SpecDecScheduler(Scheduler):
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self._draft = self._load_draft_model(config)
        self._block_size = config.draft_block_size
        self._target_layer_ids = [1, 9, 17, 25, 33]
        self._pending: dict[int, deque[int]] = {}
        self._hiddens: dict[int, tuple[torch.Tensor, int]] = {}
        self._capture_enabled = False

    def _load_draft_model(self, config: SchedulerConfig):
        from safetensors import safe_open

        from minisgl.models.draft import DFlashConfig, DFlashDraftModel, extract_context_feature

        self._extract_context_feature = extract_context_feature

        dc = DFlashConfig(
            block_size=config.draft_block_size,
            target_layer_ids=self._target_layer_ids,
        )
        dm = DFlashDraftModel(dc)

        import glob, os

        from minisgl.utils import download_hf_weight

        folder = download_hf_weight(config.draft_model_path)
        files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
        aw: dict[str, torch.Tensor] = {}
        for f in files:
            with safe_open(f, framework="pt", device=str(self.device)) as sf:
                for k in sf.keys():
                    aw[k.removesuffix(".weight")] = sf.get_tensor(k).to(self.engine.dtype)
        dm.load_weights(aw)
        logger.info_rank0("DFlash draft model loaded.")
        return dm

    def _forward(self, forward_input: ForwardInput):
        batch = forward_input.batch

        if not (batch.is_decode and self._draft is not None and batch.size == 1):
            result = super()._forward(forward_input)
            self._capture_after_forward(batch)
            return result

        req = batch.reqs[0]
        uid = req.uid

        if uid in self._pending and self._pending[uid]:
            return self._emit_pending(forward_input, uid)

        if uid not in self._hiddens:
            result = super()._forward(forward_input)
            self._capture_after_forward(batch)
            return result

        return self._run_specdec_step(forward_input)

    def _capture_after_forward(self, batch: Batch) -> None:
        ctx = self.engine.ctx
        if self._capture_enabled and ctx.captured_hidden_states:
            for req in batch.reqs:
                th = self._extract_context_feature(
                    ctx.captured_hidden_states, self._target_layer_ids
                )
                self._hiddens[req.uid] = (th, req.cached_len)
            ctx.captured_hidden_states = []
        ctx.capture_hidden_layers = self._target_layer_ids if self._draft else None
        self._capture_enabled = True

    def _emit_pending(self, forward_input: ForwardInput, uid: int) -> ForwardOutput:
        req = forward_input.batch.reqs[0]
        token = self._pending[uid].popleft()
        if not self._pending[uid]:
            del self._pending[uid]
        req.complete_one()
        return self._fake_output(token)

    def _run_specdec_step(self, forward_input: ForwardInput):
        batch = forward_input.batch
        req = batch.reqs[0]
        uid = req.uid
        th, ctx_pos = self._hiddens.pop(uid)
        bs = self._block_size
        dv = self.device

        mask_token = 151669
        draft_ids = torch.full((1, bs + 1), mask_token, dtype=torch.int32, device=dv)
        draft_ids[:, 0] = batch.input_ids[0]
        ne = self.engine.model.embed_tokens(draft_ids)

        ctx_len = th.shape[1]
        pid_start = req.cached_len - ctx_len + 1 if req.cached_len >= ctx_len else 0
        full_pid = torch.arange(pid_start, pid_start + ctx_len + bs + 1, device=dv).unsqueeze(0)
        cos, sin = self._draft.rotary_emb(ne, full_pid)

        with torch.inference_mode():
            draft_out = self._draft.forward(
                noise_embedding=ne, target_hidden=th, position_ids=full_pid, cos=cos, sin=sin,
            )
        draft_logits = self.engine.model.lm_head.forward(draft_out)[:, -(bs):, :]
        draft_tokens = self.engine.sampler.sample(
            draft_logits.squeeze(0), forward_input.sample_args
        ).unsqueeze(0).to(torch.int32)

        verify_ids = torch.cat([batch.input_ids, draft_tokens[:, :bs]], dim=1)
        verify_pos = torch.arange(
            req.cached_len, req.cached_len + bs + 1, device=dv, dtype=torch.int32
        ).unsqueeze(0)

        verify_batch = Batch(reqs=[req], phase="decode")
        verify_batch.padded_reqs = [req]
        verify_batch.input_ids = verify_ids[0]
        verify_batch.positions = verify_pos[0]

        pt = self.engine.page_table
        out_idx = torch.full_like(verify_pos, req.table_idx, dtype=torch.int64)
        verify_batch.out_loc = pt[out_idx, verify_pos.to(torch.int64)]

        self.engine.attn_backend.prepare_metadata(verify_batch)

        ctx = self.engine.ctx
        ctx.capture_hidden_layers = self._target_layer_ids
        ctx.captured_hidden_states = []
        with ctx.forward_batch(verify_batch), torch.inference_mode():
            verify_logits = self.engine.model.forward()
        new_hidden = self._extract_context_feature(ctx.captured_hidden_states, self._target_layer_ids)
        ctx.captured_hidden_states = []

        verify_logits_2d = verify_logits[-bs - 1:] if verify_logits.ndim == 2 else verify_logits
        verify_tokens = self.engine.sampler.sample(verify_logits_2d, forward_input.sample_args).unsqueeze(0).to(torch.int32)

        matches = (draft_tokens[:, :-1] == verify_tokens[:, 1:-1]).to(torch.int32)
        accept_len = matches.cumprod(dim=1).sum(dim=1).item()

        keep_pages = req.cached_len + accept_len + 1
        self.engine.kv_cache.crop(keep_pages)

        for i in range(accept_len + 1):
            req.complete_one()

        accepted = verify_tokens[:, :accept_len + 2].squeeze(0).tolist()

        self._hiddens[uid] = (new_hidden[:, :accept_len + 2, :], req.cached_len)

        if accepted:
            first = accepted.pop(0)
            if accepted:
                self._pending[uid] = deque(accepted)
            return self._fake_output(first)

        return self._fake_output(verify_tokens[0, 0].item())

    def _fake_output(self, token: int) -> ForwardOutput:
        from minisgl.engine import ForwardOutput

        t = torch.tensor([token], dtype=torch.int32, device=self.device)
        cpu = t.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return ForwardOutput(t, cpu, ev)