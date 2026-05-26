"""Step 2b: Validate mini-sglang DFlashAttention against HF reference.

Mounts the mini-sglang source, imports DFlashAttention, loads real weights,
and compares its forward pass against the HF Qwen3DFlashAttention.

Run: modal run modal_tests/step2b_minisgl.py
"""

import modal

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1", "transformers==4.57.3", "accelerate",
        "safetensors", "huggingface_hub[hf_transfer]", "datasets",
        "msgpack", "pyzmq", "modelscope",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("python/minisgl", "/app/minisgl")
)

app = modal.App("dflash-step2b")
fixture_vol = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="A10G",
    volumes={"/fixtures": fixture_vol, "/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def test_minisgl_attention():
    import os, sys
    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/app")

    import torch, torch.nn.functional as F
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    from minisgl.models.draft import DFlashAttention, DFlashConfig

    print("=== Step 2b: mini-sglang DFlashAttention ===")

    draft = AutoModel.from_pretrained(
        "z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()
    hf_attn = draft.layers[0].self_attn
    rotate_half_hf = sys.modules[draft.__class__.__module__].rotate_half

    noise_emb = torch.load("/fixtures/block_0/noise_embedding.pt", map_location="cuda", weights_only=True)
    fc_norm = torch.load("/fixtures/block_0/fc_norm_output.pt", map_location="cuda", weights_only=True)
    q_len, ctx_len = noise_emb.shape[1], fc_norm.shape[1]
    pos_ids = torch.arange(ctx_len + q_len, device="cuda").unsqueeze(0)

    cfg = draft.config
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    # ── Build DFlashConfig from HF config ──
    dflash_cfg = DFlashConfig(
        block_size=cfg.block_size,
        mask_token_id=cfg.dflash_config.get("mask_token_id", 151669),
        target_layer_ids=cfg.dflash_config.get("target_layer_ids", [1, 9, 17, 25, 33]),
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=hd,
        rms_norm_eps=cfg.rms_norm_eps,
        num_target_layers=cfg.num_target_layers,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
    )

    # ── Create mini-sglang attention and load weights ──
    our_attn = DFlashAttention(dflash_cfg, layer_idx=0)
    mf = hf_hub_download("z-lab/Qwen3-8B-DFlash-b16", "model.safetensors")
    with safe_open(mf, framework="pt", device="cuda") as f:
        mapping = {
            "q_proj": "layers.0.self_attn.q_proj.weight",
            "k_proj": "layers.0.self_attn.k_proj.weight",
            "v_proj": "layers.0.self_attn.v_proj.weight",
            "o_proj": "layers.0.self_attn.o_proj.weight",
            "q_norm": "layers.0.self_attn.q_norm.weight",
            "k_norm": "layers.0.self_attn.k_norm.weight",
        }
        for attr, key in mapping.items():
            setattr(our_attn, attr, f.get_tensor(key).to(noise_emb.dtype))

    # ── HF reference ──
    with torch.inference_mode():
        pos_emb = draft.rotary_emb(noise_emb, pos_ids)
        hf_cos, hf_sin = pos_emb
        # HF manual RoPE needs unsqueezed cos, mini-sglang expects 3D
        cos_r = hf_cos.unsqueeze(1)
        sin_r = hf_sin.unsqueeze(1)

        hf_q = hf_attn.q_proj(noise_emb).view(1, q_len, n_q, hd)
        hf_q = hf_attn.q_norm(hf_q).transpose(1, 2)
        cos_q, sin_q = cos_r[..., -q_len:, :], sin_r[..., -q_len:, :]
        hf_qr = (hf_q * cos_q) + (rotate_half_hf(hf_q) * sin_q)

        hf_k = torch.cat([hf_attn.k_proj(fc_norm), hf_attn.k_proj(noise_emb)], dim=1)
        hf_k = hf_k.view(1, ctx_len+q_len, n_kv, hd)
        hf_k = hf_attn.k_norm(hf_k).transpose(1, 2)
        hf_kr = (hf_k * cos_r) + (rotate_half_hf(hf_k) * sin_r)

        hf_v = torch.cat([hf_attn.v_proj(fc_norm), hf_attn.v_proj(noise_emb)], dim=1)
        hf_v = hf_v.view(1, ctx_len+q_len, n_kv, hd).transpose(1, 2)

        hf_sdpa = F.scaled_dot_product_attention(hf_qr, hf_kr, hf_v, attn_mask=None, is_causal=False, enable_gqa=True)
        hf_out = hf_sdpa.reshape(1, q_len, -1)
        hf_out = hf_attn.o_proj(hf_out)

    # ── mini-sglang forward ──
    with torch.inference_mode():
        our_out = our_attn.forward(noise_emb, fc_norm, hf_cos, hf_sin)

    out_diff = (our_out.float() - hf_out.float()).abs()
    print(f"  Output diff: max={out_diff.max().item():.6f}  >0.1={((out_diff>0.1).float().mean()*100):.1f}%")

    torch.testing.assert_close(our_out, hf_out, rtol=0.02, atol=2.1)
    print("PASS: mini-sglang DFlashAttention matches HF")
    print("=== Step 2b PASSED ===")
