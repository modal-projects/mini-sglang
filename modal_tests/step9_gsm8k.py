"""Step 9: GSM8K Benchmark — Baseline vs SpecDec.

Loads Qwen3-8B + DFlashDraftModel, runs GSM8K prompts with and without
speculation. Measures accuracy (bootstrap posterior) and speedup.

Run: modal run modal_tests/step9_gsm8k.py
"""

import modal

_image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.1","transformers==4.57.3","accelerate","safetensors",
    "huggingface_hub[hf_transfer]","datasets","msgpack","pyzmq","modelscope",
).env({"HF_HUB_ENABLE_HF_TRANSFER":"1","TOKENIZERS_PARALLELISM":"false"})
.add_local_dir("python/minisgl","/app/minisgl"))

app = modal.App("dflash-step9")
fv = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)
hc = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=_image, gpu="A100", volumes={"/fixtures":fv,"/cache":hc},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600)
def test_gsm8k_benchmark():
    import os, sys, time, re, json, random
    os.environ["HF_HOME"]="/cache/huggingface"
    sys.path.insert(0,"/app")
    import torch
    import numpy as np
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from minisgl.models.draft import DFlashConfig, DFlashDraftModel, extract_context_feature
    from tqdm import tqdm

    print("=== Step 9: GSM8K Benchmark ===")

    print("  Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    samples = list(ds)
    random.seed(42)
    random.shuffle(samples)
    n_samples = min(100, len(samples))
    samples = samples[:n_samples]
    print(f"    {n_samples} questions loaded.")

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
    bs = dc.block_size
    mi = dc.mask_token_id

    GSM8K_PREFIX = "Give me the final numeric answer only. No extra words.\n\n"
    GSM8K_ANSWER_SEP = "####"

    def _build_prompt(tokenizer, question: str) -> str:
        text = f"{GSM8K_PREFIX}{question}\n{GSM8K_ANSWER_SEP}"
        messages = [{"role":"user","content":text}]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)

    def _extract_number(text: str) -> float | None:
        match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if match:
            return float(match.group().replace(",", ""))
        return None

    def _check_answer(pred: str, answer_str: str) -> bool:
        target_num = _extract_number(answer_str)
        pred_num = _extract_number(pred)
        if target_num is not None and pred_num is not None:
            return abs(target_num - pred_num) < 1e-6
        return pred.strip().lower() == answer_str.strip().lower()

    def _generate_baseline(tokenizer, input_ids, max_new=256):
        with torch.inference_mode():
            out = target.generate(
                input_ids.to("cuda"), max_new_tokens=max_new,
                temperature=0.0, do_sample=False,
                pad_token_id=eos_id,
            )
        return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    def _generate_specdec(tokenizer, input_ids, max_new=256):
        num_in = input_ids.shape[1]
        ids = input_ids.to("cuda")
        dv = target.device
        ml = num_in + max_new
        oi = torch.full((1, ml+bs), mi, dtype=torch.long, device=dv)
        pi = torch.arange(oi.shape[1], device=dv).unsqueeze(0)
        pkv_t = DynamicCache()

        out_t = target(ids, position_ids=pi[:,:num_in],
                       past_key_values=pkv_t, use_cache=True,
                       logits_to_keep=1, output_hidden_states=True)
        ft = torch.argmax(out_t.logits[:,-1,:], dim=-1)
        oi[:,:num_in] = ids
        oi[:,num_in] = ft
        th = extract_context_feature(out_t.hidden_states, dc.target_layer_ids)
        start = num_in
        while start < ml:
            bid = oi[:,start:start+bs].clone()
            ne = target.model.embed_tokens(bid)
            ctx_len = th.shape[1]
            pid_start = start - ctx_len
            full_pid = pi[:, pid_start: start + bs]
            cos, sin = dm.rotary_emb(ne, full_pid)
            draft_out = dm.forward(
                noise_embedding=ne, target_hidden=th,
                position_ids=full_pid, cos=cos, sin=sin,
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
            pkv_t.crop(start)
            th = extract_context_feature(out_t.hidden_states, dc.target_layer_ids)[:,:al+1,:]
            if eos_id in acc[0]:
                break
        return tokenizer.decode(oi[0, num_in:start], skip_special_tokens=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    eos_id = tokenizer.eos_token_id
    results = []
    baseline_times = []
    specdec_times = []

    print(f"  Running {n_samples} samples...")
    for i, sample in enumerate(tqdm(samples)):
        question = sample["question"]
        answer = sample["answer"].split(GSM8K_ANSWER_SEP)[-1].strip()

        prompt = _build_prompt(tokenizer, question)
        inp = tokenizer(prompt, return_tensors="pt")["input_ids"]

        t0 = time.time()
        with torch.inference_mode():
            bas = _generate_baseline(tokenizer, inp, max_new=200)
        bt = time.time() - t0

        t0 = time.time()
        with torch.inference_mode():
            spec = _generate_specdec(tokenizer, inp, max_new=200)
        st = time.time() - t0

        bas_correct = _check_answer(bas, answer)
        spec_correct = _check_answer(spec, answer)
        match = bas_correct == spec_correct

        results.append({
            "idx": i,
            "bas_correct": bas_correct,
            "spec_correct": spec_correct,
            "match": match,
            "bas_time": bt,
            "spec_time": st,
            "speedup": bt / max(st, 1e-6),
        })
        baseline_times.append(bt)
        specdec_times.append(st)

        if not match:
            tqdm.write(f"    mismatch #{i}: bas={bas_correct} spec={spec_correct}")

    bas_acc = sum(r["bas_correct"] for r in results) / len(results)
    spec_acc = sum(r["spec_correct"] for r in results) / len(results)
    bas_mean = np.mean(baseline_times)
    spec_mean = np.mean(specdec_times)
    speedup = bas_mean / max(spec_mean, 1e-6)

    # Bootstrap posterior for accuracy
    n_boot = 10000
    boot_deltas = []
    boot_bas_accs = []
    boot_spec_accs = []
    rng = np.random.RandomState(42)
    n = len(results)
    bas_flags = np.array([r["bas_correct"] for r in results], dtype=np.float64)
    spec_flags = np.array([r["spec_correct"] for r in results], dtype=np.float64)
    for _ in range(n_boot):
        idxs = rng.randint(0, n, n)
        boot_bas_accs.append(bas_flags[idxs].mean())
        boot_spec_accs.append(spec_flags[idxs].mean())
        boot_deltas.append((spec_flags[idxs] - bas_flags[idxs]).mean())

    delta_mean = np.mean(boot_deltas)
    delta_std = np.std(boot_deltas)
    delta_ci = np.percentile(boot_deltas, [2.5, 97.5])

    bas_acc_ci = np.percentile(boot_bas_accs, [2.5, 97.5])
    spec_acc_ci = np.percentile(boot_spec_accs, [2.5, 97.5])

    print()
    print("  === Results ===")
    print(f"  Samples: {len(results)}")
    print(f"  Baseline accuracy: {bas_acc:.2%}  [95% CI: {bas_acc_ci[0]:.2%} - {bas_acc_ci[1]:.2%}]")
    print(f"  SpecDec accuracy: {spec_acc:.2%}  [95% CI: {spec_acc_ci[0]:.2%} - {spec_acc_ci[1]:.2%}]")
    print(f"  Accuracy delta (spec - bas): {delta_mean:+.2%}  [95% CI: {delta_ci[0]:+.2%} - {delta_ci[1]:+.2%}]")
    print(f"  Match rate: {sum(1 for r in results if r['match'])/len(results):.2%}")
    print(f"  Baseline avg time: {bas_mean:.2f}s")
    print(f"  SpecDec avg time: {spec_mean:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Total baseline time: {sum(baseline_times):.1f}s")
    print(f"  Total specdec time: {sum(specdec_times):.1f}s")

    assert speedup > 0.9, f"No speedup ({speedup:.1f}x)"
    assert delta_ci[1] > -0.10, f"Significant accuracy regression: {delta_ci[0]:+.2%} to {delta_ci[1]:+.2%}"
    print(f"=== Step 9 PASSED ===")