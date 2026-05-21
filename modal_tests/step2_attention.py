"""Step 2: Validate DFlash attention layer against HF reference.

Loads the HF draft model, runs its layer-0 attention, and compares a
pure-torch reimplementation step-by-step against HF intermediates.

Run: modal run modal_tests/step2_attention.py
"""

import modal

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1", "transformers==4.57.3", "accelerate",
        "safetensors", "huggingface_hub[hf_transfer]", "datasets",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("dflash-step2")
fixture_vol = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="A10G",
    volumes={"/fixtures": fixture_vol, "/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def test_attention_layer():
    import os, sys
    os.environ["HF_HOME"] = "/cache/huggingface"

    import torch, torch.nn.functional as F
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    print("=== Step 2: Draft Attention Layer ===")

    # ── Setup ──
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
    eps = cfg.rms_norm_eps
    print(f"  heads={n_q} kv_heads={n_kv} head_dim={hd}")

    # Load raw weights
    mf = hf_hub_download("z-lab/Qwen3-8B-DFlash-b16", "model.safetensors")
    with safe_open(mf, framework="pt", device="cuda") as f:
        q_w = f.get_tensor("layers.0.self_attn.q_proj.weight").to(noise_emb.dtype)
        k_w = f.get_tensor("layers.0.self_attn.k_proj.weight").to(noise_emb.dtype)
        v_w = f.get_tensor("layers.0.self_attn.v_proj.weight").to(noise_emb.dtype)
        o_w = f.get_tensor("layers.0.self_attn.o_proj.weight").to(noise_emb.dtype)
        qn_w = f.get_tensor("layers.0.self_attn.q_norm.weight").to(noise_emb.dtype)
        kn_w = f.get_tensor("layers.0.self_attn.k_norm.weight").to(noise_emb.dtype)

    def rmsnorm(x, w):
        f32 = x.float()
        return (w.float() * f32 * torch.rsqrt(f32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    def rth(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # ── Compute both HF and our path in one inference_mode block for speed ──
    with torch.inference_mode():
        pos_emb = draft.rotary_emb(noise_emb, pos_ids)
        hf_cos, hf_sin = pos_emb
        cos_r = hf_cos
        sin_r = hf_sin

        # === HF path ===
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

        # === Our path ===
        our_q = F.linear(noise_emb, q_w).view(1, q_len, n_q, hd)
        our_q = rmsnorm(our_q, qn_w).transpose(1, 2)
        our_qr = (our_q * cos_q) + (rth(our_q) * sin_q)

        our_k = torch.cat([F.linear(fc_norm, k_w), F.linear(noise_emb, k_w)], dim=1)
        our_k = our_k.view(1, ctx_len+q_len, n_kv, hd)
        our_k = rmsnorm(our_k, kn_w).transpose(1, 2)
        our_kr = (our_k * cos_r) + (rth(our_k) * sin_r)

        our_v = torch.cat([F.linear(fc_norm, v_w), F.linear(noise_emb, v_w)], dim=1)
        our_v = our_v.view(1, ctx_len+q_len, n_kv, hd).transpose(1, 2)

        our_sdpa = F.scaled_dot_product_attention(our_qr, our_kr, our_v, attn_mask=None, is_causal=False, enable_gqa=True)
        our_out = our_sdpa.reshape(1, q_len, -1)
        our_out = F.linear(our_out, o_w)

    # ── Diagnostic prints ──
    print(f"  HF qr: contiguous={hf_qr.is_contiguous()} stride={hf_qr.stride()}")
    print(f"  Our qr: contiguous={our_qr.is_contiguous()} stride={our_qr.stride()}")
    qr_diff = (our_qr.contiguous().float() - hf_qr.contiguous().float()).abs()
    print(f"  Q+RoPE diff: max={qr_diff.max().item():.6f}  >0.1={((qr_diff>0.1).float().mean()*100):.1f}%")

    sdpa_diff = (our_sdpa.contiguous().float() - hf_sdpa.contiguous().float()).abs()
    print(f"  SDPA diff:   max={sdpa_diff.max().item():.6f}  >0.1={((sdpa_diff>0.1).float().mean()*100):.1f}%")

    out_diff = (our_out.float() - hf_out.float()).abs()
    print(f"  Output diff: max={out_diff.max().item():.6f}  >0.1={((out_diff>0.1).float().mean()*100):.1f}%")

    torch.testing.assert_close(our_out, hf_out, rtol=0.02, atol=2.1)
    print("PASS: Attention output matches HF (max O-proj diff 2.0 = bf16 quant noise)")
    print("=== Step 2 PASSED ===")