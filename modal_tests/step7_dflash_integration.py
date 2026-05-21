"""Step 7: Full specdec integration using our DFlashDraftModel.

Loads Qwen3-8B + our DFlashDraftModel, runs the complete specdec loop
and verifies output tokens match baseline. Also measures speedup.

Run: modal run modal_tests/step7_dflash_integration.py
"""

import modal

_image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
    "huggingface_hub[hf_transfer]","datasets","msgpack","pyzmq","modelscope",
).env({"HF_HUB_ENABLE_HF_TRANSFER":"1","TOKENIZERS_PARALLELISM":"false"})
.add_local_dir("python/minisgl","/app/minisgl"))

app = modal.App("dflash-step7-integration")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def test_dflash_specdec():
    import os, sys, time
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from minisgl.models.draft import (
        DFlashConfig, DFlashDraftModel, extract_context_feature,
    )

    print("=== Step 7: DFlash SpecDec Integration (our model) ===")

    print("  Loading target model...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
        dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"    loaded in {time.time()-t0:.0f}s")

    print("  Building DFlashDraftModel...")
    t0 = time.time()
    cfg = DFlashConfig(
        block_size=16, mask_token_id=151669,
        target_layer_ids=[1, 9, 17, 25, 33],
        num_layers=5, hidden_size=4096, intermediate_size=12288,
        num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, num_target_layers=36,
        max_position_embeddings=40960, rope_theta=1000000.0,
    )
    dm = DFlashDraftModel(cfg)
    mf = hf_hub_download("z-lab/Qwen3-8B-DFlash-b16","model.safetensors")
    with safe_open(mf, framework="pt", device="cuda") as f:
        aw = {k.removesuffix(".weight"): f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
    dm.load_weights(aw)
    print(f"    built in {time.time()-t0:.0f}s")
    bs = cfg.block_size
    mi = cfg.mask_token_id
    rotary = dm.rotary_emb

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    prompt = "What is 2 + 2?"
    messages = [{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    num_in = inputs["input_ids"].shape[1]
    max_new = 50
    eos_id = tokenizer.eos_token_id

    def run_specdec():
        nonlocal dm, cfg
        ml = num_in + max_new
        dv = target.device
        oi = torch.full((1, ml+bs), mi, dtype=torch.long, device=dv)
        pi = torch.arange(oi.shape[1], device=dv).unsqueeze(0)
        pkv_t = DynamicCache()
        out_t = target(inputs["input_ids"], position_ids=pi[:,:num_in],
                       past_key_values=pkv_t, use_cache=True,
                       logits_to_keep=1, output_hidden_states=True)
        ft = torch.argmax(out_t.logits[:,-1,:], dim=-1)
        oi[:,:num_in]=inputs["input_ids"]
        oi[:,num_in]=ft
        th = extract_context_feature(out_t.hidden_states, cfg.target_layer_ids)
        start = num_in
        while start < ml:
            bid = oi[:,start:start+bs].clone()
            ne = target.model.embed_tokens(bid)
            ctx_len = th.shape[1]
            pid_start = start - ctx_len
            full_pid = pi[:, pid_start : start + bs]
            cos, sin = rotary(ne, full_pid)
            draft_out = dm.forward(
                noise_embedding=ne,
                target_hidden=th,
                position_ids=full_pid,
                cos=cos, sin=sin,
            )
            dl = target.lm_head(draft_out)
            dl = dl[:,-bs+1:,:]
            dts = torch.argmax(dl, dim=-1)
            bid[:,1:]=dts
            out_t = target(bid, position_ids=pi[:,start:start+bs],
                           past_key_values=pkv_t, use_cache=True, output_hidden_states=True)
            pts = torch.argmax(out_t.logits, dim=-1)
            al = (bid[:,1:]==pts[:,:-1]).cumprod(dim=1).sum(dim=1)[0].item()
            acc = torch.cat([bid[:,:al+1],pts[:,al:al+1]], dim=1)
            oi[:,start:start+al+1]=acc[:,:-1]
            oi[:,start+al+1]=acc[:,-1:]
            start+=al+1
            if eos_id in acc[0]:
                eidx = (acc[0]==eos_id).nonzero(as_tuple=True)[0][0].item()
                start = start - (al+1) + eidx + 1
                break
            pkv_t.crop(start)
            th = extract_context_feature(out_t.hidden_states, cfg.target_layer_ids)[:,:al+1,:]
        return oi[0,num_in:start].tolist()

    print("  Baseline...")
    with torch.inference_mode():
        bas = target.generate(inputs["input_ids"], max_new_tokens=max_new,
                              temperature=0.0, do_sample=False,
                              pad_token_id=eos_id)
    btok = bas[0,num_in:].tolist()
    print(f"    {len(btok)} tokens")

    print("  DFlash SpecDec...")
    with torch.inference_mode():
        stok = run_specdec()
    print(f"    {len(stok)} tokens")

    ml2 = min(len(btok), len(stok))
    match = btok[:ml2]==stok[:ml2]
    print(f"  Match baseline: {match}")
    assert match, f"SpecDec diverges! bas={btok[:10]} spec={stok[:10]}"

    # Speed measurement
    print("  Measuring speed...")
    for _ in range(3):
        with torch.inference_mode():
            _ = target.generate(inputs["input_ids"], max_new_tokens=max_new, temperature=0.0, do_sample=False, pad_token_id=eos_id)

    t0 = time.time()
    for _ in range(3):
        with torch.inference_mode():
            _ = target.generate(inputs["input_ids"], max_new_tokens=max_new, temperature=0.0, do_sample=False, pad_token_id=eos_id)
    bt = (time.time()-t0)/3

    t0 = time.time()
    for _ in range(3):
        with torch.inference_mode():
            _ = run_specdec()
    st = (time.time()-t0)/3

    sp = bt/st
    print(f"  Baseline: {bt:.2f}s  SpecDec: {st:.2f}s  Speedup: {sp:.1f}x")
    assert sp > 0.9, f"No speedup ({sp:.1f}x)"
    print(f"=== Step 7 PASSED (speedup={sp:.1f}x) ===")