"""Steps 7-12: Full specdec integration test.

Loads Qwen3-8B + DFlash draft model, runs the complete specdec loop
(draft -> verify -> accept), and verifies output tokens match baseline.
Also measures speedup.

Run: modal run modal_tests/step7_12_integration.py
"""

import modal

_image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
    "huggingface_hub[hf_transfer]","datasets","msgpack","pyzmq","modelscope",
).env({"HF_HUB_ENABLE_HF_TRANSFER":"1","TOKENIZERS_PARALLELISM":"false"})
.add_local_dir("python/minisgl","/app/minisgl"))

app = modal.App("dflash-step7-12")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def test_full_specdec():
    import os, sys, time
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

    print("=== Steps 7-12: Full SpecDec Integration ===")

    print("  Loading Qwen3-8B target model...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
        dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"    loaded in {time.time()-t0:.0f}s")

    print("  Loading DFlash draft model...")
    t0 = time.time()
    draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16",
        trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"    loaded in {time.time()-t0:.0f}s")

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
        dm = sys.modules[draft.__class__.__module__]
        ecf = dm.extract_context_feature
        bs = draft.block_size
        mi = draft.mask_token_id
        ml = num_in + max_new
        dv = target.device
        oi = torch.full((1, ml+bs), mi, dtype=torch.long, device=dv)
        pi = torch.arange(oi.shape[1], device=dv).unsqueeze(0)
        pkv_t = DynamicCache()
        pkv_d = DynamicCache()
        out_t = target(inputs["input_ids"], position_ids=pi[:,:num_in],
                       past_key_values=pkv_t, use_cache=True,
                       logits_to_keep=1, output_hidden_states=True)
        ft = torch.argmax(out_t.logits[:,-1,:], dim=-1)
        oi[:,:num_in]=inputs["input_ids"]
        oi[:,num_in]=ft
        th = ecf(out_t.hidden_states, draft.target_layer_ids)
        start = num_in
        while start < ml:
            bid = oi[:,start:start+bs].clone()
            ne = target.model.embed_tokens(bid)
            dl = target.lm_head(draft(noise_embedding=ne, target_hidden=th,
                   position_ids=pi[:,pkv_d.get_seq_length():start+bs],
                   past_key_values=pkv_d, use_cache=True, is_causal=False))
            dl = dl[:,-bs+1:,:]
            pkv_d.crop(start)
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
            th = ecf(out_t.hidden_states, draft.target_layer_ids)[:,:al+1,:]
        return oi[0,num_in:start].tolist()

    # ── Baseline ──
    print("  Baseline (autoregressive)...")
    with torch.inference_mode():
        bas = target.generate(inputs["input_ids"], max_new_tokens=max_new,
                              temperature=0.0, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    btok = bas[0,num_in:].tolist()
    print(f"    {len(btok)} tokens")

    # ── SpecDec ──
    print("  SpecDec...")
    with torch.inference_mode():
        stok = run_specdec()
    print(f"    {len(stok)} tokens")

    # ── Verify ──
    ml2 = min(len(btok), len(stok))
    match = btok[:ml2]==stok[:ml2]
    print(f"  Match baseline: {match}")
    assert match, "SpecDec output does not match baseline!"

    # ── Speed ──
    print("  Measuring speed...")
    for _ in range(3):  # warmup
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
    print(f"=== Steps 7-12 PASSED (speedup={sp:.1f}x) ===")