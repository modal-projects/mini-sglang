"""Step 8: Engine-level SpecDec smoke test.

Uses the LLM class (offline Scheduler) for proper page allocation and
batch management. Baseline uses LLM.generate(). SpecDec hooks into the
Engine's model/sampler/KV cache for draft+verification.

Run: modal run modal_tests/step8_engine_specdec.py
"""

import modal

_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
        "huggingface_hub[hf_transfer]","datasets",
        "flashinfer-python>=0.5.3",
        "pyzmq","msgpack","modelscope",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir("python/minisgl","/app/minisgl")
)

app = modal.App("dflash-step8")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def test_engine_specdec():
    import os, sys, time
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from safetensors import safe_open
    from minisgl.core import (
        Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx,
    )
    from minisgl.distributed import DistributedInfo
    from minisgl.llm.llm import LLM
    from minisgl.models.draft import DFlashConfig, DFlashDraftModel, extract_context_feature

    print("=== Step 8: Engine SpecDec Smoke Test ===")

    print("  Building Engine (LLM offline mode)...")
    t0 = time.time()
    llm = LLM("Qwen/Qwen3-8B", dtype=torch.bfloat16,
              page_size=1, max_running_req=4, cuda_graph_max_bs=4)
    ctx = llm.engine.ctx
    print(f"    built in {time.time()-t0:.0f}s")

    print("  Loading DFlash draft model...")
    dc = DFlashConfig(block_size=16, target_layer_ids=[1, 9, 17, 25, 33])
    dm = DFlashDraftModel(dc)
    import glob
    from minisgl.utils import download_hf_weight
    folder = download_hf_weight("z-lab/Qwen3-8B-DFlash-b16")
    files = sorted(glob.glob(f"{folder}/*.safetensors"))
    aw = {}
    for f in files:
        with safe_open(f, framework="pt", device=str(llm.engine.device)) as sf:
            for k in sf.keys():
                aw[k.removesuffix(".weight")] = sf.get_tensor(k).to(torch.bfloat16)
    dm.load_weights(aw)
    print("    draft model ready.")
    bs = dc.block_size
    mi = dc.mask_token_id

    prompt = "What is 2 + 2?"
    eos_id = llm.tokenizer.eos_token_id

    print("  Baseline (LLM.generate)...")
    with torch.inference_mode():
        bas_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    btok = bas_result[0]["token_ids"]
    print(f"    {len(btok)} tokens: {llm.tokenizer.decode(btok)!r}")

    def run_specdec(max_new=50):
        input_ids = torch.tensor(llm._tokenize_one(prompt), dtype=torch.int32)
        n_in = len(input_ids)
        engine = llm.engine
        pt = engine.page_table
        dv = engine.device
        cm = llm.cache_manager
        tgt_layers = dc.target_layer_ids
        ps = ctx.page_size

        req = Req(input_ids=input_ids, table_idx=0, cached_len=0,
                  output_len=max_new, uid=99,
                  sampling_params=SamplingParams(temperature=0.0),
                  cache_handle=None)

        needed = (n_in + ps - 1) // ps
        pages = cm._allocate(needed)
        tokens = cm._page_to_token(pages)
        pt[0, :len(tokens)].copy_(tokens)

        req.cached_len = 0
        req.device_len = n_in

        prefill = Batch(reqs=[req], phase="prefill")
        prefill.padded_reqs = [req]
        prefill_positions = torch.arange(0, n_in, dtype=torch.int32, device=dv)
        prefill.positions = prefill_positions
        input_mapping = (
            torch.full((n_in,), 0, dtype=torch.int64, device=dv),
            prefill_positions.to(torch.int64),
        )
        prefill.out_loc = pt[input_mapping]
        prefill.input_ids = input_ids.to(dtype=torch.int32, device=dv)
        engine.attn_backend.prepare_metadata(prefill)
        args = engine.sampler.prepare(prefill)

        ctx.capture_hidden_layers = tgt_layers
        ctx.captured_hidden_states = []
        with ctx.forward_batch(prefill), torch.inference_mode():
            logits = engine.model.forward()
        for _ in range(n_in):
            req.complete_one()
        th = extract_context_feature(ctx.captured_hidden_states, tgt_layers).unsqueeze(0)
        ctx.captured_hidden_states = []

        tokens_out = []
        tok = engine.sampler.sample(logits[-1:], args).item()
        tokens_out.append(tok)
        if tok == eos_id:
            return tokens_out

        while len(tokens_out) < max_new:
            draft_ids = torch.full((1, bs+1), mi, dtype=torch.int32, device=dv)
            draft_ids[:, 0] = tokens_out[-1]
            ew = engine.model.model.embed_tokens.weight
            ne = ew[draft_ids.flatten().to(torch.int64)].unsqueeze(0).to(device=dv, dtype=torch.bfloat16).contiguous()
            th = th.to(device=dv, dtype=torch.bfloat16)
            ctx_len = th.shape[1]
            pid_start = max(0, req.cached_len - ctx_len + 1)
            full_pid = torch.arange(pid_start, pid_start + ctx_len + bs + 1, device=dv).unsqueeze(0)
            cos, sin = dm.rotary_emb(ne, full_pid)

            with torch.inference_mode():
                draft_out = dm.forward(noise_embedding=ne, target_hidden=th,
                                       position_ids=full_pid, cos=cos, sin=sin)
            draft_logits = torch.nn.functional.linear(draft_out, engine.model.lm_head.weight)[:, -(bs):, :]
            draft_tokens = engine.sampler.sample(draft_logits.squeeze(0), args).unsqueeze(0).to(torch.int32)

            verify_needed = bs + 1
            verify_pages_needed = max(0, (req.cached_len + verify_needed + ps - 1) // ps
                                        - (req.cached_len + ps - 1) // ps)
            if verify_pages_needed > 0:
                new_pages = cm._allocate(verify_pages_needed)
                new_tokens = cm._page_to_token(new_pages)
                alloc_start = ((req.cached_len + ps - 1) // ps) * ps
                pt[0, alloc_start:alloc_start + len(new_tokens)].copy_(new_tokens)
            req.device_len = req.cached_len + bs + 1

            verify_ids = torch.cat([
                torch.tensor([[tokens_out[-1]]], dtype=torch.int32, device=dv),
                draft_tokens[:, :bs]
            ], dim=1)
            verify_pos = torch.arange(req.cached_len, req.cached_len + bs + 1,
                                      device=dv, dtype=torch.int32)

            vbatch = Batch(reqs=[req], phase="prefill")
            vbatch.padded_reqs = [req]
            vbatch.input_ids = verify_ids[0]
            vbatch.positions = verify_pos
            vbatch.out_loc = pt[(torch.full((bs+1,), 0, dtype=torch.int64, device=dv),
                                  verify_pos.to(torch.int64))]
            engine.attn_backend.prepare_metadata(vbatch)

            ctx.capture_hidden_layers = tgt_layers
            ctx.captured_hidden_states = []
            with ctx.forward_batch(vbatch), torch.inference_mode():
                hidden = engine.model.model.forward(vbatch.input_ids)
            new_th = extract_context_feature(ctx.captured_hidden_states, tgt_layers).unsqueeze(0)
            ctx.captured_hidden_states = []

            vl = torch.nn.functional.linear(hidden, engine.model.lm_head.weight)
            vt = torch.argmax(vl, dim=-1).unsqueeze(0)
            matches = (draft_tokens == vt[:, :-1]).to(torch.int32)
            accept_len = matches.cumprod(dim=1).sum(dim=1).item()

            engine.kv_cache.crop(req.cached_len + accept_len + 1)
            accepted = vt[0, :accept_len + 1].tolist()
            for _ in range(accept_len + 1):
                req.complete_one()
            tokens_out.extend(accepted)
            th = new_th[:, :accept_len + 1, :]

            if any(t == eos_id for t in accepted):
                break

        return tokens_out[:max_new]

    print("  SpecDec...")
    with torch.inference_mode():
        stok = run_specdec()
    print(f"    {len(stok)} tokens: {llm.tokenizer.decode(stok)!r}")

    assert stok == btok, f"SpecDec diverges:\n  bas={btok}\n  spec={stok}"
    print("  Match baseline: True")

    print("  Measuring speed...")
    for _ in range(3):
        with torch.inference_mode():
            _ = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    t0 = time.time()
    for _ in range(5):
        with torch.inference_mode():
            _ = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    bt = (time.time()-t0)/5

    for _ in range(3):
        with torch.inference_mode():
            _ = run_specdec(max_new=30)
    t0 = time.time()
    for _ in range(5):
        with torch.inference_mode():
            _ = run_specdec(max_new=30)
    st = (time.time()-t0)/5

    sp = bt/st
    print(f"  Baseline: {bt:.2f}s  SpecDec: {st:.2f}s  Speedup: {sp:.1f}x")
    assert sp > 0.9, f"No speedup ({sp:.1f}x)"
    print(f"=== Step 8 PASSED (speedup={sp:.1f}x) ===")