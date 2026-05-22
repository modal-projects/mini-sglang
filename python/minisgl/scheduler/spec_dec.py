"""Speculative decoding scheduler using DFlash draft model.

For single-request decode, intercepts decode forward passes and runs
speculative decoding: draft model generates B candidate tokens, target
model verifies via multi-token prefill forward, accepted tokens are
queued and emitted one per step.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
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
        self._block_size = config.draft_block_size
        self._target_layer_ids = [1, 9, 17, 25, 33]
        self._draft = self._load_draft_model(config)
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
            result = self._emit_pending(forward_input, uid)
            forward_input.batch.input_ids = self.token_pool[forward_input.input_tuple]
            self.token_pool[forward_input.write_tuple] = result.next_tokens_gpu
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
            return result

        if uid not in self._hiddens:
            result = super()._forward(forward_input)
            forward_input.batch.input_ids = self.token_pool[forward_input.input_tuple]
            self._capture_after_forward(batch)
            return result

        forward_input.batch.input_ids = self.token_pool[forward_input.input_tuple]
        inp = forward_input.batch.input_ids
        if inp.numel() == 0:
            logger.error(f"token_pool empty at input_tuple={forward_input.input_tuple}")
            result = super()._forward(forward_input)
            self.token_pool[forward_input.write_tuple] = result.next_tokens_gpu
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
            return result

        result = self._run_specdec_step(forward_input)
        self.token_pool[forward_input.write_tuple] = result.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return result

    def _capture_after_forward(self, batch: Batch) -> None:
        ctx = self.engine.ctx
        if self._capture_enabled and ctx.captured_hidden_states:
            for req in batch.reqs:
                th = self._extract_context_feature(
                    ctx.captured_hidden_states, self._target_layer_ids
                ).unsqueeze(0)
                self._hiddens[req.uid] = (th, req.cached_len)
            ctx.captured_hidden_states = []
        ctx.capture_hidden_layers = self._target_layer_ids if self._draft else None
        self._capture_enabled = True

    def _emit_pending(self, forward_input: ForwardInput, uid: int) -> ForwardOutput:
        req = forward_input.batch.reqs[0]
        token = self._pending[uid].popleft()
        if not self._pending[uid]:
            del self._pending[uid]
        req.cached_len += 1
        req.device_len += 1
        return self._fake_output(token)

    def _run_specdec_step(self, forward_input: ForwardInput):
        batch = forward_input.batch
        req = batch.reqs[0]
        uid = req.uid
        th, _ = self._hiddens.pop(uid)
        bs = self._block_size
        dv = self.device
        dt = self.engine.dtype
        engine = self.engine
        pt = engine.page_table
        cm = self.cache_manager
        ew = engine.model.model.embed_tokens.weight
        lw = engine.model.lm_head.weight
        tgt_layers = self._target_layer_ids

        mask_token = 151669
        draft_ids = torch.full((1, bs + 1), mask_token, dtype=torch.int32, device=dv)
        draft_ids[:, 0] = batch.input_ids[0]
        ne = ew[draft_ids.flatten().to(torch.int64)].unsqueeze(0).to(dtype=dt, device=dv)

        ctx_len = th.shape[1]
        pid_start = max(0, req.cached_len - ctx_len + 1)
        full_pid = torch.arange(pid_start, pid_start + ctx_len + bs + 1, device=dv).unsqueeze(0)
        cos, sin = self._draft.rotary_emb(ne, full_pid)

        with torch.inference_mode():
            draft_out = self._draft.forward(
                noise_embedding=ne, target_hidden=th.to(dtype=dt, device=dv),
                position_ids=full_pid, cos=cos, sin=sin,
            )
        draft_logits = F.linear(draft_out, lw)[:, -(bs):, :]
        draft_tokens = self.engine.sampler.sample(
            draft_logits.squeeze(0), forward_input.sample_args
        ).unsqueeze(0).to(torch.int32)

        req.device_len = req.cached_len + bs + 1
        cm.allocate_paged([req])

        verify_ids = torch.cat([batch.input_ids.view(1, 1), draft_tokens[:, :bs]], dim=1)
        verify_pos = torch.arange(
            req.cached_len, req.cached_len + bs + 1, device=dv, dtype=torch.int32
        )

        vbatch = Batch(reqs=[req], phase="prefill")
        vbatch.padded_reqs = [req]
        vbatch.input_ids = verify_ids[0]
        vbatch.positions = verify_pos
        vbatch.out_loc = pt[(
            torch.full((bs + 1,), req.table_idx, dtype=torch.int64, device=dv),
            verify_pos.to(torch.int64),
        )]
        engine.attn_backend.prepare_metadata(vbatch)

        ctx = engine.ctx
        ctx.capture_hidden_layers = tgt_layers
        ctx.captured_hidden_states = []
        with ctx.forward_batch(vbatch), torch.inference_mode():
            hidden = engine.model.model.forward(verify_ids[0])
        new_th = self._extract_context_feature(ctx.captured_hidden_states, tgt_layers).unsqueeze(0)
        ctx.captured_hidden_states = []

        vl = F.linear(hidden, lw)
        vt = torch.argmax(vl, dim=-1).unsqueeze(0)
        matches = (draft_tokens == vt[:, :-1]).to(torch.int32)
        accept_len = matches.cumprod(dim=1).sum(dim=1).item()

        keep = req.cached_len + accept_len + 1
        engine.kv_cache.crop(keep)
        req.cached_len = keep
        req.device_len = keep

        accepted = vt[0, :accept_len + 1].tolist()
        self._hiddens[uid] = (new_th[:, :accept_len + 1, :], req.cached_len)

        if accepted:
            first = accepted.pop(0)
            if accepted:
                self._pending[uid] = deque(accepted)
            return self._fake_output(first)

        return self._fake_output(vt[0, 0].item())

    def _fake_output(self, token: int) -> ForwardOutput:
        from minisgl.engine import ForwardOutput

        t = torch.tensor([token], dtype=torch.int32, device=self.device)
        cpu = t.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return ForwardOutput(t, cpu, ev)