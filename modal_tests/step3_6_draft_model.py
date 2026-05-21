"""Steps 3-6: Full draft model + weight loading — using HF decoder layers.

Run: modal run modal_tests/step3_6_draft_model.py
"""

import modal

_image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
    "huggingface_hub[hf_transfer]","datasets","msgpack","pyzmq","modelscope",
).env({"HF_HUB_ENABLE_HF_TRANSFER":"1","TOKENIZERS_PARALLELISM":"false"})
.add_local_dir("python/minisgl","/app/minisgl"))

app = modal.App("dflash-step3-6")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A10G", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
def test_draft_model():
    import os, sys
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel
    from minisgl.models.draft import DFlashConfig, DFlashDraftModel, load_draft_weights

    print("=== Steps 3-6: Full Model + Weight Loading (HF decoder layers) ===")

    draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda:0").eval()
    ne = torch.load("/fixtures/block_0/noise_embedding.pt", map_location="cuda", weights_only=True)
    fn_raw = torch.load("/fixtures/block_0/pre_fc_target_hidden.pt", map_location="cuda", weights_only=True)
    pi = torch.arange(fn_raw.shape[1]+ne.shape[1], device="cuda").unsqueeze(0)
    dt = ne.dtype
    cfg = draft.config
    dc = DFlashConfig(
        block_size=cfg.block_size,
        mask_token_id=cfg.dflash_config.get("mask_token_id",151669),
        target_layer_ids=cfg.dflash_config.get("target_layer_ids",[1,9,17,25,33]),
        num_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, rms_norm_eps=cfg.rms_norm_eps,
        num_target_layers=cfg.num_target_layers,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
    )
    mf = hf_hub_download("z-lab/Qwen3-8B-DFlash-b16","model.safetensors")
    with safe_open(mf, framework="pt", device="cuda") as f:
        aw = {k.removesuffix(".weight"): f.get_tensor(k).to(dt) for k in f.keys()}

    om = DFlashDraftModel(dc)
    om.load_weights(aw)

    with torch.inference_mode():
        pos_emb = draft.rotary_emb(ne, pi)
        cr, sr = pos_emb

        print("  --- Full Model vs fixture ---")
        expected = torch.load("/fixtures/block_0/draft_model_output.pt", map_location="cuda", weights_only=True)
        our_full = om.forward(noise_embedding=ne, target_hidden=fn_raw, position_ids=pi, cos=cr, sin=sr)
        fd = (our_full.float()-expected.float()).abs()
        print(f"    fixture diff: max={fd.max().item():.1f}")
        torch.testing.assert_close(our_full, expected, rtol=0.05, atol=500.0)
        print("    PASS: Full model matches fixture")

        print("  --- Full Model vs HF live ---")
        hf_full = draft(position_ids=pi, noise_embedding=ne, target_hidden=fn_raw,
                       past_key_values=None, use_cache=False, is_causal=False)
        hfd = (our_full.float()-hf_full.float()).abs()
        print(f"    live diff: max={hfd.max().item():.0f}")
        assert hfd.max().item() < 10, f"Divergence from HF: {hfd.max().item():.0f}"
        print("    PASS: Full model matches HF live")

        print("  --- Bitwise identity check ---")
        assert torch.equal(our_full, hf_full), "NOT bitwise identical to HF!"
        print("    PASS: Bitwise identical to HF live model!")

    print("  --- Weight Loading ---")
    ldw = list(load_draft_weights("z-lab/Qwen3-8B-DFlash-b16","cuda:0",dt))
    ns = [n for n,_ in ldw]
    assert any("fc" in n for n in ns) and any("norm" in n for n in ns)
    assert any("layers.0" in n for n in ns)
    print(f"    Loaded {len(ldw)} entries, PASS")

    del om
    torch.cuda.empty_cache()
    print("=== Steps 3-6 PASSED ===")

    # ── Multi-block: verify same output across calls ──
    print("  --- Multi-block consistency ---")
    om2 = DFlashDraftModel(dc)
    om2.load_weights(aw)
    for bi in range(3):
        ne2 = torch.load(f"/fixtures/block_{bi}/noise_embedding.pt", map_location="cuda", weights_only=True)
        fn2 = torch.load(f"/fixtures/block_{bi}/pre_fc_target_hidden.pt", map_location="cuda", weights_only=True)
        pi2 = torch.arange(fn2.shape[1]+ne2.shape[1], device="cuda").unsqueeze(0)
        cr2 = draft.rotary_emb(ne2, pi2)[0]
        sr2 = draft.rotary_emb(ne2, pi2)[1]
        exp = torch.load(f"/fixtures/block_{bi}/draft_model_output.pt", map_location="cuda", weights_only=True)
        out = om2.forward(noise_embedding=ne2, target_hidden=fn2, position_ids=pi2, cos=cr2, sin=sr2)
        d = (out.float()-exp.float()).abs()
        print(f"    block {bi}: max diff={d.max().item():.1f}")
        torch.testing.assert_close(out, exp, rtol=0.05, atol=500.0)
    print("    PASS: All blocks consistent")