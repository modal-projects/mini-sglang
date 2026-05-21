"""Weight loading for DFlash draft model.

Loads weights from safetensors and maps them to mini-sglang state dict keys.
The HF safetensors use keys like "layers.0.self_attn.q_proj.weight".
The mini-sglang BaseOP expects names like "layers.0.self_attn.q_proj".

Handles QKV merging and gate/up merging.
"""

from __future__ import annotations

from typing import Iterator

import torch
from safetensors import safe_open
from minisgl.utils import download_hf_weight


def load_draft_weights(model_path: str, device: torch.device, dtype: torch.dtype) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream draft model weights with QKV and gate_up merging."""
    import glob
    from minisgl.models.config import ModelConfig

    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files

    qkv_slots = ("q", "k", "v")
    gate_up_slots = ("gate", "up")
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}

    for file in sorted(files):
        with safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensor = f.get_tensor(name).to(dtype)

                stem = name.removesuffix(".weight")

                if stem.endswith(".q_proj"):
                    base = stem.replace(".q_proj", ".qkv_proj")
                    merge_buf.setdefault(base, {})["q"] = tensor
                    if all(s in merge_buf.get(base, {}) for s in qkv_slots):
                        parts = [merge_buf[base][s] for s in qkv_slots]
                        del merge_buf[base]
                        yield base, torch.cat(parts, dim=0)
                elif stem.endswith(".k_proj"):
                    base = stem.replace(".k_proj", ".qkv_proj")
                    merge_buf.setdefault(base, {})["k"] = tensor
                    if all(s in merge_buf.get(base, {}) for s in qkv_slots):
                        parts = [merge_buf[base][s] for s in qkv_slots]
                        del merge_buf[base]
                        yield base, torch.cat(parts, dim=0)
                elif stem.endswith(".v_proj"):
                    base = stem.replace(".v_proj", ".qkv_proj")
                    merge_buf.setdefault(base, {})["v"] = tensor
                    if all(s in merge_buf.get(base, {}) for s in qkv_slots):
                        parts = [merge_buf[base][s] for s in qkv_slots]
                        del merge_buf[base]
                        yield base, torch.cat(parts, dim=0)
                elif stem.endswith(".gate_proj"):
                    base = stem.replace(".gate_proj", ".gate_up_proj")
                    merge_buf.setdefault(base, {})["gate"] = tensor
                    if all(s in merge_buf.get(base, {}) for s in gate_up_slots):
                        parts = [merge_buf[base][s] for s in gate_up_slots]
                        del merge_buf[base]
                        yield base, torch.cat(parts, dim=0)
                elif stem.endswith(".up_proj"):
                    base = stem.replace(".up_proj", ".gate_up_proj")
                    merge_buf.setdefault(base, {})["up"] = tensor
                    if all(s in merge_buf.get(base, {}) for s in gate_up_slots):
                        parts = [merge_buf[base][s] for s in gate_up_slots]
                        del merge_buf[base]
                        yield base, torch.cat(parts, dim=0)
                else:
                    yield stem, tensor

    assert not merge_buf, f"Incomplete merge groups: {list(merge_buf.keys())}"


def map_draft_weights_to_state_dict(model_path: str, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load weights and map HF key names to mini-sglang DFlashDraftModel state dict.

    HF safetensors use keys like:
        layers.0.self_attn.q_proj.weight  → layers.0.self_attn.q_proj  (merged to qkv_proj)
        fc.weight                          → projection.fc.weight       (LinearReplicated)
        hidden_norm.weight                 → projection.hidden_norm.weight
        norm.weight                        → norm
    """
    weight_iter = load_draft_weights(model_path, device, dtype)
    state_dict: dict[str, torch.Tensor] = {}

    for stem, tensor in weight_iter:
        renamed = stem
        for prefix in ("layers.",):
            if stem.startswith(prefix):
                renamed = stem
                break
        state_dict[renamed] = tensor

    return state_dict