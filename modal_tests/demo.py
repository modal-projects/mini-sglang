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
    from minisgl.message import UserMsg
    from minisgl.scheduler.config import SchedulerConfig
    from minisgl.scheduler.spec_dec import SpecDecScheduler
    from minisgl.distributed import DistributedInfo
    from transformers import AutoTokenizer

    class SpecDecLLM(SpecDecScheduler):
        def __init__(self, **kwargs):
            config = SchedulerConfig(
                tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
                offline_mode=True, **kwargs,
            )
            super().__init__(config)
            self._p = []
            self._status = {}
            self._ctr = 0

        def _load_tok(self):
            if not hasattr(self, "_cached_tok"):
                self._cached_tok = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path)
            return self._cached_tok

        def _tokenize_one(self, prompt):
            tok = self._load_tok()
            if isinstance(prompt, str):
                return tok.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
            return torch.tensor(prompt, dtype=torch.int32)

        def offline_receive_msg(self, blocking=False):
            if blocking and not self._p:
                raise type("Done", (Exception,), {})()
            res = []
            for prompt, sp in self._p:
                ids = self._tokenize_one(prompt)
                uid = self._ctr
                self._ctr += 1
                res.append(UserMsg(uid=uid, input_ids=ids, sampling_params=sp))
                self._status[uid] = []
            self._p = []
            return res

        def offline_send_result(self, reply):
            for msg in reply:
                if not (msg.finished and msg.next_token == self.eos_token_id):
                    self._status[msg.uid].append(msg.next_token)

        def generate(self, prompt, max_tokens=256):
            self._p = [(prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens))]
            self._status = {}
            self._ctr = 0
            self._pending.clear()
            self._hiddens.clear()
            self._capture_enabled = False
            self.engine.ctx.capture_hidden_layers = None
            self.engine.ctx.captured_hidden_states = []
            try:
                self.run_forever()
            except Exception as e:
                if "Done" not in str(type(e).__name__):
                    raise
            return self._status[0]

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
    tok = llm._load_tok()
    print(f"  loaded in {time.time()-t0:.0f}s")

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'-'*60}")

        llm._draft = None
        t0 = time.perf_counter()
        bas_ids = llm.generate(prompt, max_tokens=80)
        bas_t = time.perf_counter() - t0
        bas_text = tok.decode(bas_ids, skip_special_tokens=True)
        print(f"  Baseline ({bas_t:.2f}s, {len(bas_ids)} tok)")
        print(f"    {bas_text[:300]}")

        llm._draft = draft_model
        t0 = time.perf_counter()
        spec_ids = llm.generate(prompt, max_tokens=80)
        spec_t = time.perf_counter() - t0
        spec_text = tok.decode(spec_ids, skip_special_tokens=True)
        speedup = bas_t / spec_t
        match = "✓" if bas_ids == spec_ids else "✗"
        print(f"  SpecDec  ({spec_t:.2f}s, {len(spec_ids)} tok, {speedup:.1f}x) match={match}")
        print(f"    {spec_text[:300]}")

    print(f"\n{'='*60}")
    print("Done!")