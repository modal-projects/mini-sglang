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
    from minisgl.scheduler.config import SchedulerConfig
    from minisgl.scheduler.spec_dec import SpecDecScheduler
    from minisgl.distributed import DistributedInfo

    print("=== Step 11: Scheduler-level SpecDec ===")

    # Create a custom LLM-like class that extends SpecDecScheduler
    from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg

    class SpecDecLLM(SpecDecScheduler):
        def __init__(self, model_path, draft_model_path, draft_block_size=16, **kwargs):
            config = SchedulerConfig(
                model_path=model_path,
                tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16,
                offline_mode=True,
                draft_model_path=draft_model_path,
                draft_block_size=draft_block_size,
                **kwargs,
            )
            super().__init__(config)
            self.pending_requests = []
            self.status_map = {}
            self.counter = 0

        def _tokenize_one(self, prompt):
            if isinstance(prompt, str):
                return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

        def offline_receive_msg(self, blocking=False):
            if blocking and len(self.pending_requests) == 0:
                raise type("RequestAllFinished", (Exception,), {})()
            results = []
            added, sum_input_len = 0, 0
            for tokens_or_prompt, sampling_params in self.pending_requests:
                if sum_input_len >= self.prefill_budget:
                    break
                input_ids = self._tokenize_one(tokens_or_prompt)
                sum_input_len += len(input_ids)
                uid, added = self.counter + added, added + 1
                results.append(UserMsg(uid=uid, input_ids=input_ids,
                                       sampling_params=sampling_params))
                self.status_map[uid] = type("Status", (), {
                    "uid": uid, "output_ids": []
                })()
            self.counter += added
            self.pending_requests = self.pending_requests[added:]
            return results

        def offline_send_result(self, reply):
            for msg in reply:
                status = self.status_map[msg.uid]
                if not (msg.finished and msg.next_token == self.eos_token_id):
                    status.output_ids.append(msg.next_token)

        def generate(self, prompts, sampling_params):
            self.pending_requests = []
            self.status_map = {}
            self.counter = 0
            if isinstance(sampling_params, SamplingParams):
                sampling_params = [sampling_params] * len(prompts)
            for prompt, sp in zip(prompts, sampling_params):
                self.pending_requests.append((prompt, sp))
            try:
                self.run_forever()
            except Exception as e:
                if "RequestAllFinished" not in str(type(e).__name__):
                    import traceback
                    print(f"  ERROR during run_forever: {e}")
                    traceback.print_exc()
                    raise
            results = []
            for i in range(len(prompts)):
                status = self.status_map[i]
                output_text = self.tokenizer.decode(status.output_ids)
                results.append({"text": output_text, "token_ids": status.output_ids})
            return results

    class RequestAllFinished(Exception):
        pass

    print("  Building SpecDecLLM...")
    t0 = time.time()
    llm = SpecDecLLM(
        "Qwen/Qwen3-8B",
        draft_model_path="z-lab/Qwen3-8B-DFlash-b16",
        draft_block_size=16,
        page_size=1, max_running_req=2, cuda_graph_max_bs=2,
    )
    print(f"    built in {time.time()-t0:.0f}s")

    prompt = "What is 2 + 2?"

    print("  Baseline (no specdec)...")
    SpecDecLLM._draft = None
    # Need to temporarily disable specdec by removing draft
    old_draft = llm._draft
    llm._draft = None
    with torch.inference_mode():
        bas_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    btok = bas_result[0]["token_ids"]
    llm._draft = old_draft
    print(f"    {len(btok)} tokens")

    print("  SpecDec...")
    with torch.inference_mode():
        spec_result = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    stok = spec_result[0]["token_ids"]
    print(f"    {len(stok)} tokens")

    assert stok == btok, f"SpecDec diverges:\n  bas={btok[:20]}\n  spec={stok[:20]}"
    print("  Match baseline: True")

    print("  Measuring speed...")
    llm._draft = None
    for _ in range(3):
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    t0 = time.time()
    for _ in range(5):
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    bt = (time.time()-t0)/5

    llm._draft = old_draft
    for _ in range(3):
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    t0 = time.time()
    for _ in range(5):
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    st = (time.time()-t0)/5

    sp = bt/st
    print(f"  Baseline: {bt:.2f}s  SpecDec: {st:.2f}s  Speedup: {sp:.1f}x")
    print(f"=== Step 11 PASSED (speedup={sp:.1f}x) ===")