# SpecDec Correctness Verification: State & Trustworthiness

This is an analysis of the WIP speculative decoding implementation (`SpecDecScheduler` + `DFlashDraftModel`) in mini-sglang, focused on how correctness is verified and whether those verifications can be trusted.

---

## Verification Architecture

The implementation has a **4-tier verification hierarchy**:

### Tier 1: Fixture-Based Component Verification (`step0` + `step3_6`)

`step0_capture.py` runs the HF reference `spec_generate()` on "What is 2 + 2?" and saves every intermediate tensor as golden `.pt` files, plus computes baseline AR tokens and verifies `specdec_matches_baseline`. `step3_6_draft_model.py` then loads these fixtures and does **three checks**, escalating in strictness:

1. **vs. fixtures**: `torch.testing.assert_close(our_full, expected, rtol=0.05, atol=500.0)` — rough tolerance, mostly catches total failure.
2. **vs. HF live**: `hfd.max().item() < 10` — tight max absolute error bound, catches numerical divergence.
3. **bitwise identity**: `torch.equal(our_full, hf_full)` — proves the `DFlashDraftModel` is a perfect replica of the HF model.

But this is *only* the draft model forward pass with fixed inputs from a single prompt. It verifies nothing about the specdec loop logic.

### Tier 2: End-to-End Token-Exact Match (`step7`, `step7_12`, `step8`, `step11`)

All four integration tests compare **token-for-token** output against a temperature=0 baseline:

| Test | Draft Model | Target Model | Orchestration |
|------|-------------|-------------|--------------|
| `step7` | mini-sglang `DFlashDraftModel` | HF `AutoModelForCausalLM` | Hand-rolled loop |
| `step7_12` | HF `AutoModel` | HF `AutoModelForCausalLM` | Hand-rolled loop |
| `step8` | mini-sglang `DFlashDraftModel` | mini-sglang Engine | Hand-rolled loop using Engine primitives |
| `step11` | mini-sglang `DFlashDraftModel` | mini-sglang Engine | `SpecDecScheduler.run_forever()` |

All use the single prompt `"What is 2 + 2?"` with `temperature=0.0`. The assertion is `stok == btok` / `match == True` on the full token list.

### Tier 3: Statistical Accuracy Parity (`step9`)

Runs 100 GSM8K math problems, comparing specdec vs baseline accuracy with **bootstrap confidence intervals** (10,000 resamples). Asserts that the accuracy delta's 95% CI lower bound is above -10%. This is the **only test with varied prompts**.

### Tier 4: Performance Micro-Benchmarks (`step10`)

Measures per-component latency (draft fwd, target prefill, target decode, LM head, `extract_context_feature`) using `perf_cuda`. No correctness assertions — just smoke-testing that components run.

---

## What Can Be Trusted

1. **The draft model is bitwise identical** to HF's. `torch.equal` is absolute truth — no floating-point fudge.
2. **Greedy decoding is deterministic**, so token-exact matching is a clean pass/fail criterion. No sampling noise to explain away.
3. **Multi-level coverage**: from tensors (Tier 1) through full scheduler loop (Tier 2) to statistical parity on real data (Tier 3).

---

## What Cannot Be Trusted (Gaps)

### Single-prompt monoculture

Every correctness test except GSM8K uses the single trivial prompt `"What is 2 + 2?"`. This exercises exactly one prefill length and one acceptance pattern. The positional math in `_run_specdec_step` (`spec_dec.py:129`):

```python
pid_start = max(0, req.cached_len - ctx_len + 1)
```

is never stress-tested with varying acceptance lengths, long prompts, or prompt lengths that push context boundaries.

### No stochastic sampling verification

Every test uses `temperature=0.0`. The code calls `engine.sampler.sample()` for draft tokens at `spec_dec.py:140` — but with non-zero temperature, specdec's output distribution must be proven identical to the target model's. No such verification exists.

### Single-request only

`spec_dec.py:69` gates on `batch.size == 1`. Multi-request batching is explicitly excluded. No test exercises the boundary condition where a second request arrives during a specdec step.

### KV cache crop tested only implicitly

`kvcache/base.py:27-30` zeros out entries beyond `keep_len`. There's no unit test that:

- Writes values past `keep_len`, crops, then verifies zeros
- Crops, reuses the zeroed pages for a new request, and verifies no leakage
- Tests that cropping doesn't corrupt the radix tree state in the prefix cache

### EOS handling in `_run_specdec_step` is untested

The method at `spec_dec.py:108` never checks for EOS in accepted tokens. If an EOS appears mid-block, the pending tokens include it and the caller is expected to detect it — but step11's scheduler test doesn't test this explicitly.

### No CUDA graph correctness verification

Step 8 uses CUDA graphs for the draft model but only captures graphs at context lengths `{1, 5, 10, 20, 36, 16}`. If `ctx_len` falls outside this set, the code picks the "closest" graph and manually applies an attention mask — but `forward_graph` pads `target_hidden` differently from `forward`. Correctness of this pad-and-mask workaround is not verified against the eager path.

### No continuous integration

All tests are Modal scripts that require manual invocation (`modal run`). There's no CI that runs these on every PR.

### No edge-case fuzzing

No tests vary block size, draft model path, temperature, or prompt length systematically. No long-prompt (32K) test. No OOM stress test despite specdec allocating `bs+1` extra token slots per step.

### The "no specdec" baseline in step11 is invalid

```python
llm._draft = None     # <-- monkey-patch to disable
```

This doesn't exercise the normal non-specdec scheduler path — it forces `SpecDecScheduler._forward()` to fall through to `super()._forward()` with a null draft check, which is a different code path than `Scheduler._forward()` itself.

---

## Summary

The verification strengths are the **bitwise draft model identity proof** and the **deterministic token-exact matching**. The weaknesses are near-complete absence of varying prompts, no stochastic verification, no multi-request testing, no KV cache unit tests, no CI, and no edge-case fuzzing. The verification is trustworthy for the narrow path it covers (single short prompt, greedy decode, single request) but provides little evidence of correctness beyond that.