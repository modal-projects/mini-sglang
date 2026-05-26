"""Demo: DFlash speculative decoding with mini-sglang.

Run: modal run modal_tests/demo.py
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

app = modal.App("dflash-demo")

@app.function(image=_image, gpu="A100", secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1200)
def main():
    import os, sys, time
    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/app")
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm.spec_dec import SpecDecLLM
    from transformers import AutoTokenizer

    prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "What is the derivative of x^2?",
        "Explain quantum computing in one sentence.",
    ]

    print("Loading Qwen3-8B + DFlash draft model...")
    t0 = time.time()
    llm = SpecDecLLM(
        model_path="Qwen/Qwen3-8B",
        draft_model_path="z-lab/Qwen3-8B-DFlash-b16",
        draft_block_size=16,
        page_size=1, max_running_req=2, cuda_graph_max_bs=2,
        max_extend_tokens=128, cache_type="naive",
    )
    draft_model = llm._draft
    tok = AutoTokenizer.from_pretrained(llm.tokenizer.name_or_path)
    print(f"  loaded in {time.time()-t0:.0f}s")

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'-'*60}")

        llm._draft = None
        t0 = time.perf_counter()
        bas_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=80))
        bas_t = time.perf_counter() - t0
        bas_ids = bas_result[0]["token_ids"]
        bas_text = bas_result[0]["text"]
        print(f"  Baseline ({bas_t:.2f}s, {len(bas_ids)} tok)")
        print(f"    {bas_text[:300]}")

        llm._draft = draft_model
        t0 = time.perf_counter()
        spec_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=80))
        spec_t = time.perf_counter() - t0
        spec_ids = spec_result[0]["token_ids"]
        spec_text = spec_result[0]["text"]
        speedup = bas_t / spec_t
        match = "✓" if bas_ids == spec_ids else "✗"
        print(f"  SpecDec  ({spec_t:.2f}s, {len(spec_ids)} tok, {speedup:.1f}x) match={match}")
        print(f"    {spec_text[:300]}")

    print(f"\n{'='*60}")
    print("Done!")