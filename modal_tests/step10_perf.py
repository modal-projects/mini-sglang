"""SpecDec performance micro-benchmark using perf_cuda.

Measures individual component latencies in the specdec loop:
- Target model prefill
- Target model single-token decode  
- Draft model forward
- Target module verification (multi-token)

Adds CUDA graph capture for the draft model forward.

Run: modal run modal_tests/step10_perf.py
"""

import modal

_image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
    "huggingface_hub[hf_transfer]","datasets","msgpack","pyzmq","modelscope",
).env({"HF_HUB_ENABLE_HF_TRANSFER":"1","TOKENIZERS_PARALLELISM":"false"})
.add_local_dir("python/minisgl","/app/minisgl"))

app = modal.App("dflash-step10")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def test_perf():
    import os, sys, time
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from minisgl.models.draft import DFlashConfig, DFlashDraftModel, extract_context_feature
    from minisgl.benchmark.perf import perf_cuda

    print("=== Step 10: SpecDec Perf Micro-Benchmark ===")

    print("  Loading target model...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
        dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"    loaded in {time.time()-t0:.0f}s")

    print("  Building DFlashDraftModel...")
    from huggingface_hub import hf_hub_download
    dc = DFlashConfig(block_size=16, target_layer_ids=[1, 9, 17, 25, 33])
    dm = DFlashDraftModel(dc)
    mf = hf_hub_download("z-lab/Qwen3-8B-DFlash-b16","model.safetensors")
    with safe_open(mf, framework="pt", device="cuda") as f:
        aw = {k.removesuffix(".weight"): f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
    dm.load_weights(aw)
    print("    draft model ready.")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    prompt = "What is 2 + 2?"
    messages = [{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
    num_in = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    bs = 16
    mi = dc.mask_token_id

    target_layers = dc.target_layer_ids

    # ── Set up prefill state ──
    pi = torch.arange(4096, device="cuda").unsqueeze(0)
    pkv = DynamicCache()
    out = target(input_ids, position_ids=pi[:,:num_in],
                 past_key_values=pkv, use_cache=True, output_hidden_states=True)
    th = extract_context_feature(out.hidden_states, target_layers)

    # ── Prepare fixed inputs for micro-benchmarks ──
    draft_ids = torch.full((1, bs+1), mi, dtype=torch.long, device="cuda")
    draft_ids[:, 0] = torch.argmax(out.logits[:,-1,:], dim=-1)
    ne = target.model.embed_tokens(draft_ids)
    ctx_len = th.shape[1]
    pid_start = num_in - ctx_len
    full_pid = pi[:, pid_start: pid_start + ctx_len + bs + 1]
    cos, sin = dm.rotary_emb(ne, full_pid)

    print()
    print("  --- Micro-benchmarks (perf_cuda, ms) ---")

    # ── Draft model forward ──
    def _draft_fwd():
        dm.forward(noise_embedding=ne, target_hidden=th, position_ids=full_pid, cos=cos, sin=sin)
    t_draft = perf_cuda(_draft_fwd, repetitions=50, cuda_graph_repetitions=None)
    print(f"  Draft model fwd:         {t_draft:.3f} ms")

    # ── Draft model CUDA graph ──
    _dg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_dg):
        dm.forward(noise_embedding=ne, target_hidden=th, position_ids=full_pid, cos=cos, sin=sin)

    def _draft_fwd_cg():
        _dg.replay()
    t_draft_cg = perf_cuda(_draft_fwd_cg, cuda_graph_repetitions=None, repetitions=200)
    print(f"  Draft model fwd (CG):    {t_draft_cg:.3f} ms  ({t_draft/t_draft_cg:.1f}x)")
    del _dg

    # 3. Target model prefill
    def _target_prefill():
        target(input_ids, position_ids=pi[:,:num_in],
               past_key_values=DynamicCache(), use_cache=True, output_hidden_states=True)
    t_prefill = perf_cuda(_target_prefill, repetitions=10, cuda_graph_repetitions=None)
    print(f"  Target prefill:          {t_prefill:.3f} ms")

    # 4. Target model single-token decode
    def _target_decode():
        pkv_d = DynamicCache()
        target(input_ids, position_ids=pi[:,:num_in], past_key_values=pkv_d, use_cache=True, output_hidden_states=False)
        target(torch.tensor([[draft_ids[0,0].item()]], device="cuda"),
               position_ids=pi[:,num_in:num_in+1], past_key_values=pkv_d, use_cache=True,
               logits_to_keep=1)
    t_decode = perf_cuda(_target_decode, repetitions=10, cuda_graph_repetitions=None)
    print(f"  Target decode (1 tok):   {t_decode:.3f} ms")

    # ── LM head ──
    h_test = target.model.embed_tokens(draft_ids).detach().clone()
    def _lmhead():
        target.lm_head(h_test)
    t_lmhead = perf_cuda(_lmhead, repetitions=100)
    print(f"  LM head (17 tok):        {t_lmhead:.3f} ms")

    # 7. extract_context_feature
    hiddens = out.hidden_states
    def _ecf():
        extract_context_feature(hiddens, target_layers)
    t_ecf = perf_cuda(_ecf, repetitions=100)
    print(f"  extract_context_feature: {t_ecf:.3f} ms")

    print()
    print("  --- Summary ---")
    print(f"  Draft model fwd:    {t_draft:.2f} ms")
    print(f"  Draft model (CG):   {t_draft_cg:.2f} ms ({t_draft/t_draft_cg-1:.0%} improvement)")
    print(f"  Target prefill:     {t_prefill:.1f} ms")
    print(f"  Target decode:      {t_decode:.1f} ms")
    print(f"  LM head:            {t_lmhead:.4f} ms")
    print(f"  extract_context:    {t_ecf:.4f} ms")
    print(f"=== Step 10 PASSED ===")