"""DFlashDraftModel — uses HF's Qwen3DFlashDecoderLayer.

Proven bitwise identical when loaded with the same weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minisgl.layers.base import BaseOP

from .config import DFlashConfig

_decoder_cls_cache: type | None = None
_hf_cfg_cache = None


def extract_context_feature(hidden_states: tuple | list, target_layer_ids: list[int]) -> torch.Tensor:
    """Concatenate selected target layer hidden states along last dim.

    HF DFlash draft model expects input of shape (B, T, num_target_layers * hidden_size).
    """
    selected = [hidden_states[i] for i in target_layer_ids]
    return torch.cat(selected, dim=-1)


def _get_decoder_cls_and_cfg():
    global _decoder_cls_cache, _hf_cfg_cache
    if _decoder_cls_cache is None:
        from transformers import AutoModel
        draft = AutoModel.from_pretrained(
            "z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True,
            dtype=torch.bfloat16, device_map="meta",
        )
        _decoder_cls_cache = type(draft.layers[0])
        _hf_cfg_cache = draft.config
        _hf_cfg_cache._attn_implementation = "sdpa"
        del draft
    return _decoder_cls_cache, _hf_cfg_cache


class DFlashDraftModel(BaseOP):
    def __init__(self, config: DFlashConfig):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        _, hf_cfg = _get_decoder_cls_and_cfg()
        self.rotary_emb = Qwen3RotaryEmbedding(hf_cfg)
        self.layers: list = []
        self.fc: torch.Tensor | None = None
        self.hidden_norm: torch.Tensor | None = None
        self.norm: torch.Tensor | None = None
        self._config = config
        self._eps = config.rms_norm_eps

    def load_weights(self, weights: dict[str, torch.Tensor]):
        decoder_cls, hf_cfg = _get_decoder_cls_and_cfg()
        self.fc = weights["fc"].clone()
        self.hidden_norm = weights["hidden_norm"].clone()
        self.norm = weights["norm"].clone()
        for i in range(self._config.num_layers):
            layer = decoder_cls(hf_cfg, layer_idx=i).to("cuda", dtype=torch.bfloat16)
            p = f"layers.{i}."
            layer.input_layernorm.weight.data = weights[p + "input_layernorm"]
            layer.post_attention_layernorm.weight.data = weights[p + "post_attention_layernorm"]
            a = layer.self_attn
            a.q_proj.weight.data = weights[p + "self_attn.q_proj"]
            a.k_proj.weight.data = weights[p + "self_attn.k_proj"]
            a.v_proj.weight.data = weights[p + "self_attn.v_proj"]
            a.o_proj.weight.data = weights[p + "self_attn.o_proj"]
            a.q_norm.weight.data = weights[p + "self_attn.q_norm"]
            a.k_norm.weight.data = weights[p + "self_attn.k_norm"]
            layer.mlp.gate_proj.weight.data = weights[p + "mlp.gate_proj"]
            layer.mlp.up_proj.weight.data = weights[p + "mlp.up_proj"]
            layer.mlp.down_proj.weight.data = weights[p + "mlp.down_proj"]
            self.layers.append(layer.eval())

    def _project_target(self, target_hidden: torch.Tensor) -> torch.Tensor:
        projected = F.linear(target_hidden, self.fc)
        input_dtype = projected.dtype
        x = projected.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self._eps)
        return self.hidden_norm * x.to(input_dtype)

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        projected_target = self._project_target(target_hidden)

        if cos is None or sin is None:
            pos_emb = self.rotary_emb(noise_embedding, position_ids)
        else:
            pos_emb = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=projected_target,
                position_embeddings=pos_emb,
                attention_mask=None,
                past_key_value=None,
                use_cache=False,
                position_ids=None,
            )

        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self._eps)
        return self.norm * x.to(input_dtype)