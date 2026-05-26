"""Step 11: SpecDec through the full scheduler loop.

Creates a SpecDec-enabled LLM and generates tokens through the scheduler
machinery (prefill → decode loop). Verifies output matches baseline.

Run: modal run modal_tests/step11_scheduler_specdec.py
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

app = modal.App("dflash-step11")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def test_scheduler_specdec():
    import os, sys, time
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm.spec_dec import SpecDecLLM

    print("=== Step 11: Scheduler-level SpecDec ===")

    print("  Building SpecDecLLM...")
    t0 = time.time()
    llm = SpecDecLLM(
        model_path="Qwen/Qwen3-8B",
        draft_model_path="z-lab/Qwen3-8B-DFlash-b16",
        draft_block_size=16, cache_type="naive",
        page_size=1, max_running_req=2, cuda_graph_max_bs=2,
    )
    draft_model = llm._draft
    print(f"    built in {time.time()-t0:.0f}s")

    prompt = "What is 2 + 2?"

    print("  Baseline (no specdec)...")
    llm._draft = None
    with torch.inference_mode():
        bas_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    btok = bas_result[0]["token_ids"]
    print(f"    {len(btok)} tokens")
    assert len(btok) > 0, "Baseline produced no tokens!"

    print("  SpecDec...")
    llm._draft = draft_model
    with torch.inference_mode():
        spec_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    stok = spec_result[0]["token_ids"]
    print(f"    {len(stok)} tokens")
    assert len(stok) > 0, "SpecDec produced no tokens!"

    assert stok == btok, f"SpecDec diverges:\n  bas={btok[:20]}\n  spec={stok[:20]}"
    print("  Match baseline: True")
    print(f"=== Step 11 PASSED ===")