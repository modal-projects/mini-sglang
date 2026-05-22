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

    max_ctx = 128
    dv = dm.layers[0].self_attn.q_proj.weight.device if dm.layers else "cuda"
    dt = torch.bfloat16
    nh = dc.hidden_size
    nh5 = nh * len(dc.target_layer_ids)
    hd = dc.head_dim
    dg_ne = torch.zeros(1, bs+1, nh, dtype=dt, device=dv)
    dg_th = torch.zeros(1, max_ctx, nh5, dtype=dt, device=dv)
    dg_pid = torch.arange(max_ctx + bs + 1, device=dv).unsqueeze(0)
    dg_cos, dg_sin = dm.rotary_emb(dg_ne, dg_pid)
    dg_mask = torch.zeros(1, 1, bs+1, max_ctx+bs+1, dtype=dt, device=dv)
    dg_out = torch.empty(1, bs+1, nh, dtype=dt, device=dv)

    def _capture_draft_graph(ctx_len_val):
        dg_mask.zero_()
        dg_mask[:, :, :, ctx_len_val:max_ctx] = float('-inf')
        dg = torch.cuda.CUDAGraph()
        dg_th.zero_()
        with torch.cuda.graph(dg):
            dg_out.copy_(dm.forward_graph(
                noise_embedding=dg_ne, target_hidden_padded=dg_th, ctx_len=ctx_len_val,
                cos=dg_cos, sin=dg_sin, attn_mask=dg_mask,
            ))
        return dg

    # Capture graphs for common ctx_len values
    draft_graphs = {}
    for cl in [1, 5, 10, 20, 36]:
        draft_graphs[cl] = _capture_draft_graph(cl)
    draft_graphs[16] = draft_graphs.get(16, _capture_draft_graph(16))
    print(f"    captured {len(draft_graphs)} draft CUDA graphs.")

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

        req = Req(input_ids=input_ids, table_idx=0, cached_len=0,
                  output_len=max_new, uid=99,
                  sampling_params=SamplingParams(temperature=0.0),
                  cache_handle=None)
        req.device_len = n_in
        cm.allocate_paged([req])

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
        req.complete_one()
        req.device_len = req.cached_len
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
            dg_ne.copy_(ew[draft_ids.flatten().to(torch.int64)].unsqueeze(0).to(dtype=dt, device=dv))

            ctx_len = th.shape[1]
            dg_th.zero_()
            dg_th[:, :ctx_len, :].copy_(th.to(dtype=dt, device=dv))

            # Pick closest graph
            gk = min(draft_graphs.keys(), key=lambda k: (k < ctx_len, abs(k - ctx_len)))
            if gk != ctx_len:
                dg_mask.zero_()
                dg_mask[:, :, :, ctx_len:max_ctx] = float('-inf')
            dg = draft_graphs[gk]
            dg.replay()
            draft_out = dg_out.clone()
            draft_logits = torch.nn.functional.linear(draft_out, engine.model.lm_head.weight)[:, -(bs):, :]
            draft_tokens = engine.sampler.sample(draft_logits.squeeze(0), args).unsqueeze(0).to(torch.int32)

            req.device_len = req.cached_len + bs + 1
            cm.allocate_paged([req])

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

            keep = req.cached_len + accept_len + 1
            engine.kv_cache.crop(keep)
            req.cached_len = keep
            req.device_len = keep

            accepted = vt[0, :accept_len + 1].tolist()
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
    print(f"=== Step 8 PASSED (speedup={sp:.1f}x) ===")