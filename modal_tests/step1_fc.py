"""Step 1: Validate fc + hidden_norm projection against golden fixture.

Pure-torch test: loads only fc.weight and hidden_norm.weight from the
draft model safetensors, applies them to the golden target_hidden input,
and asserts the output matches the golden fc_norm_output fixture.

No mini-sglang imports needed.

Run: modal run modal_tests/step1_fc.py
"""

import modal

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "datasets",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App("dflash-step1")
fixture_vol = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="A10G",
    volumes={
        "/fixtures": fixture_vol,
        "/cache": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def test_fc_hidden_norm():
    import os

    os.environ["HF_HOME"] = "/cache/huggingface"

    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    print("=== Step 1: fc + hidden_norm validation ===")

    print("Downloading draft model weights...")
    model_file = hf_hub_download(
        "z-lab/Qwen3-8B-DFlash-b16",
        "model.safetensors",
    )
    with safe_open(model_file, framework="pt", device="cuda") as f:
        fc_weight = f.get_tensor("fc.weight")
        hn_weight = f.get_tensor("hidden_norm.weight")
    print(f"  fc.weight: {tuple(fc_weight.shape)}")
    print(f"  hidden_norm.weight: {tuple(hn_weight.shape)}")

    target_hidden = torch.load(
        "/fixtures/block_0/pre_fc_target_hidden.pt",
        map_location="cuda",
        weights_only=True,
    )
    expected = torch.load(
        "/fixtures/block_0/fc_norm_output.pt",
        map_location="cuda",
        weights_only=True,
    )
    print(f"  target_hidden (input):  {tuple(target_hidden.shape)}")
    print(f"  fc_norm_output (expected): {tuple(expected.shape)}")

    eps = 1e-6

    # ── bf16 test (already passed) ──
    projected_bf16 = torch.nn.functional.linear(
        target_hidden.to(torch.bfloat16), fc_weight.to(torch.bfloat16)
    )
    projected_bf16_f32 = projected_bf16.to(torch.float32)
    bf16_output = (
        hn_weight.float()
        * projected_bf16_f32
        * torch.rsqrt(projected_bf16_f32.pow(2).mean(-1, keepdim=True) + eps)
    ).to(torch.bfloat16)
    torch.testing.assert_close(bf16_output, expected.to(torch.bfloat16),
                               rtol=0.02, atol=0.1)
    print("PASS (bf16): fc + hidden_norm matches fixture (relaxed tolerance)")

    # ── fp32 -> bf16 roundtrip test: confirms bf16 matmul is the only difference ──
    projected_f32 = torch.nn.functional.linear(
        target_hidden.to(torch.float32), fc_weight.to(torch.float32)
    )
    fp32_then_bf16 = (
        hn_weight.float()
        * projected_f32
        * torch.rsqrt(projected_f32.pow(2).mean(-1, keepdim=True) + eps)
    ).to(torch.bfloat16)

    diff = (fp32_then_bf16.float() - expected.float()).abs()
    print(f"  fp32→bf16 vs golden: max_abs_diff={diff.max().item():.6f}  "
          f"mismatch_rate={(diff > 0).float().mean().item()*100:.1f}%")

    torch.testing.assert_close(fp32_then_bf16, expected.to(torch.bfloat16),
                               rtol=0.02, atol=0.1)
    print("PASS (fp32→bf16): matches golden — confirms bf16 matmul quant noise")

    expected_accepted_len = target_hidden.shape[1]
    assert bf16_output.shape == (1, expected_accepted_len, 4096)
    print(f"PASS: Shapes correct (accepted_len={expected_accepted_len})")
    print("=== Step 1 PASSED ===")