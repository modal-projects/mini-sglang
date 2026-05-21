"""Step 8: Engine-level SpecDec smoke test.

Creates an Engine + DFlashDraftModel and verifies the specdec loop produces
correct output using the Engine's model, sampler, and KV cache management.

Run: modal run modal_tests/step8_engine_specdec.py
"""

import modal

_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
        "huggingface_hub[hf_transfer]",
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
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine
    from minisgl.engine.config import EngineConfig
    from minisgl.models.draft import DFlashConfig, DFlashDraftModel, extract_context_feature

    print("=== Step 8: Engine SpecDec Smoke Test ===")

    config = EngineConfig(
        model_path="Qwen/Qwen3-8B",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_max_bs=4,
        page_size=1,
        attention_backend="fi",
    )
    t0 = time.time()
    engine = Engine(config)
    ctx = engine.ctx
    print(f"  Engine built in {time.time()-t0:.0f}s")

    print("  Loading DFlash draft model...")
    dc = DFlashConfig(block_size=16, target_layer_ids=[1, 9, 17, 25, 33])
    dm = DFlashDraftModel(dc)
    import glob
    from minisgl.utils import download_hf_weight
    folder = download_hf_weight("z-lab/Qwen3-8B-DFlash-b16")
    files = sorted(glob.glob(f"{folder}/*.safetensors"))
    aw = {}
    for f in files:
        with safe_open(f, framework="pt", device=str(engine.device)) as sf:
            for k in sf.keys():
                aw[k.removesuffix(".weight")] = sf.get_tensor(k).to(torch.bfloat16)
    dm.load_weights(aw)
    print("    draft model loaded.")

    from minisgl.utils import load_tokenizer
    tokenizer = load_tokenizer("Qwen/Qwen3-8B")
    prompt = "What is 2 + 2?"
    messages = [{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    eos_id = tokenizer.eos_token_id
    max_new = 50

    def run_baseline():
        req = Req(input_ids=input_ids, table_idx=0, cached_len=0,
                  output_len=max_new, uid=1,
                  sampling_params=SamplingParams(temperature=0.0),
                  cache_handle=None)
        pt = engine.page_table
        for i in range(len(input_ids)):
            pt[0, i] = i
        n_in = len(input_ids)

        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.positions = torch.arange(0, n_in, dtype=torch.int32, device=engine.device)
        batch.input_ids = input_ids.to(dtype=torch.int32, device=engine.device)
        batch.out_loc = pt[torch.zeros(n_in, dtype=torch.int64, device=engine.device),
                           batch.positions.to(torch.int64)]
        engine.attn_backend.prepare_metadata(batch)
        args = engine.sampler.prepare(batch)

        with ctx.forward_batch(batch), torch.inference_mode():
            logits = engine.model.forward()
        for _ in range(n_in):
            req.complete_one()
        del batch

        tokens = []
        tok = engine.sampler.sample(logits[-1:], args).item()
        tokens.append(tok)

        while len(tokens) < max_new:
            dec = Batch(reqs=[req], phase="decode")
            dec.padded_reqs = [req]
            pos = torch.tensor([req.cached_len], dtype=torch.int32, device=engine.device)
            dec.positions = pos
            dec.input_ids = torch.tensor([tok], dtype=torch.int32, device=engine.device)
            dec.out_loc = pt[torch.zeros(1, dtype=torch.int64, device=engine.device), pos.to(torch.int64)]
            engine.attn_backend.prepare_metadata(dec)
            with ctx.forward_batch(dec), torch.inference_mode():
                l = engine.model.forward()
            req.complete_one()
            tok = engine.sampler.sample(l, args).item()
            tokens.append(tok)
            if tok == eos_id:
                break
            del dec
        return tokens

    print("  Baseline...")
    with torch.inference_mode():
        btok = run_baseline()
    print(f"    {len(btok)} tokens: {tokenizer.decode(btok)!r}")
    assert btok == [25, 17, 10, 19, 12, 16, 20, 10, 13, 24, 68], f"Unexpected baseline: {btok}"

    def run_specdec():
        req = Req(input_ids=input_ids, table_idx=0, cached_len=0,
                  output_len=max_new, uid=2,
                  sampling_params=SamplingParams(temperature=0.0),
                  cache_handle=None)
        pt = engine.page_table
        for i in range(len(input_ids)):
            pt[0, i] = i
        n_in = len(input_ids)
        bs = 16
        mask_token = 151669
        tgt_layers = [1, 9, 17, 25, 33]

        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.positions = torch.arange(0, n_in, dtype=torch.int32, device=engine.device)
        batch.input_ids = input_ids.to(dtype=torch.int32, device=engine.device)
        batch.out_loc = pt[torch.zeros(n_in, dtype=torch.int64, device=engine.device),
                           batch.positions.to(torch.int64)]
        engine.attn_backend.prepare_metadata(batch)
        args = engine.sampler.prepare(batch)

        ctx.capture_hidden_layers = tgt_layers
        ctx.captured_hidden_states = []
        with ctx.forward_batch(batch), torch.inference_mode():
            logits = engine.model.forward()
        for _ in range(n_in):
            req.complete_one()
        th = extract_context_feature(ctx.captured_hidden_states, tgt_layers)
        ctx.captured_hidden_states = []
        del batch

        tokens = []
        tok = engine.sampler.sample(logits[-1:], args).item()
        tokens.append(tok)

        while len(tokens) < max_new:
            draft_ids = torch.full((1, bs+1), mask_token, dtype=torch.int32, device=engine.device)
            draft_ids[:, 0] = tokens[-1]
            ne = engine.model.embed_tokens(draft_ids)
            ctx_len = th.shape[1]
            pid_start = max(0, req.cached_len - ctx_len + 1)
            full_pid = torch.arange(pid_start, pid_start + ctx_len + bs + 1, device=engine.device).unsqueeze(0)
            cos, sin = dm.rotary_emb(ne, full_pid)

            with torch.inference_mode():
                draft_out = dm.forward(noise_embedding=ne, target_hidden=th,
                                       position_ids=full_pid, cos=cos, sin=sin)
            draft_logits = engine.model.lm_head.forward(draft_out)[:, -(bs):, :]
            draft_tokens = engine.sampler.sample(draft_logits.squeeze(0), args).unsqueeze(0).to(torch.int32)

            verify_ids = torch.cat([
                torch.tensor([[tokens[-1]]], dtype=torch.int32, device=engine.device),
                draft_tokens[:, :bs]
            ], dim=1)
            verify_pos = torch.arange(req.cached_len, req.cached_len + bs + 1,
                                      device=engine.device, dtype=torch.int32)

            vbatch = Batch(reqs=[req], phase="decode")
            vbatch.padded_reqs = [req]
            vbatch.input_ids = verify_ids[0]
            vbatch.positions = verify_pos
            vbatch.out_loc = pt[torch.zeros(bs+1, dtype=torch.int64, device=engine.device),
                                verify_pos.to(torch.int64)]
            engine.attn_backend.prepare_metadata(vbatch)

            ctx.capture_hidden_layers = tgt_layers
            ctx.captured_hidden_states = []
            with ctx.forward_batch(vbatch), torch.inference_mode():
                vl = engine.model.forward()
            new_th = extract_context_feature(ctx.captured_hidden_states, tgt_layers)
            ctx.captured_hidden_states = []

            vt = engine.sampler.sample(vl, args).unsqueeze(0).to(torch.int32)
            matches = (draft_tokens[:, :-1] == vt[:, :-1]).to(torch.int32)
            accept_len = matches.cumprod(dim=1).sum(dim=1).item()

            engine.kv_cache.crop(req.cached_len + accept_len + 1)
            accepted = vt[0, :accept_len + 2].tolist()
            for _ in range(accept_len + 1):
                req.complete_one()
            tokens.extend(accepted)
            th = new_th[:, :accept_len + 2, :]

            if accept_len < bs - 1 and len(tokens) < max_new:
                bonus = vt[0, accept_len + 1].item()
                tokens.append(bonus)
                if bonus == eos_id:
                    break
                req.device_len = req.cached_len + 1
                bpos = torch.tensor([req.cached_len], dtype=torch.int32, device=engine.device)
                bb = Batch(reqs=[req], phase="decode")
                bb.padded_reqs = [req]
                bb.positions = bpos
                bb.input_ids = torch.tensor([bonus], dtype=torch.int32, device=engine.device)
                bb.out_loc = pt[torch.zeros(1, dtype=torch.int64, device=engine.device), bpos.to(torch.int64)]
                engine.attn_backend.prepare_metadata(bb)
                ctx.capture_hidden_layers = tgt_layers
                ctx.captured_hidden_states = []
                with ctx.forward_batch(bb), torch.inference_mode():
                    _ = engine.model.forward()
                req.complete_one()
                th = extract_context_feature(ctx.captured_hidden_states, tgt_layers)
                ctx.captured_hidden_states = []
                if bonus == eos_id:
                    break
            elif any(t == eos_id for t in accepted):
                break

        return tokens[:max_new]

    print("  SpecDec...")
    with torch.inference_mode():
        stok = run_specdec()
    print(f"    {len(stok)} tokens: {tokenizer.decode(stok)!r}")

    assert stok == btok, f"SpecDec diverges:\n  bas={btok}\n  spec={stok}"
    print("  Match baseline: True")

    print("  Measuring speed...")
    for _ in range(3):
        with torch.inference_mode():
            _ = run_baseline()
    t0 = time.time()
    for _ in range(5):
        with torch.inference_mode():
            _ = run_baseline()
    bt = (time.time()-t0)/5

    for _ in range(3):
        with torch.inference_mode():
            _ = run_specdec()
    t0 = time.time()
    for _ in range(5):
        with torch.inference_mode():
            _ = run_specdec()
    st = (time.time()-t0)/5

    sp = bt/st
    print(f"  Baseline: {bt:.2f}s  SpecDec: {st:.2f}s  Speedup: {sp:.1f}x")
    assert sp > 0.9, f"No speedup ({sp:.1f}x)"
    print(f"=== Step 8 PASSED (speedup={sp:.1f}x) ===")