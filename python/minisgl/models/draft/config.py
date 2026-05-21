from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DFlashConfig:
    """Configuration for a DFlash draft model.

    Parsed from the draft model's HuggingFace config.json.
    """

    block_size: int = 16
    mask_token_id: int = 151669
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 9, 17, 25, 33])
    num_layers: int = 5
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    num_target_layers: int = 36
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0

    @classmethod
    def from_hf(cls, hf_config: dict) -> DFlashConfig:
        dflash_cfg = hf_config.get("dflash_config", {})
        return cls(
            block_size=hf_config.get("block_size", 16),
            mask_token_id=dflash_cfg.get("mask_token_id", 151669),
            target_layer_ids=dflash_cfg.get("target_layer_ids", [1, 9, 17, 25, 33]),
            num_layers=hf_config.get("num_hidden_layers", 5),
            hidden_size=hf_config.get("hidden_size", 4096),
            intermediate_size=hf_config.get("intermediate_size", 12288),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            head_dim=hf_config.get("head_dim", 128),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            num_target_layers=hf_config.get("num_target_layers", 36),
            max_position_embeddings=hf_config.get("max_position_embeddings", 40960),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
        )