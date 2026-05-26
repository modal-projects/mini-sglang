# DFlash Speculative Decoding — Test Design

A graded test ladder and testing strategy for adding **DFlash** speculative decoding
to mini-sglang, targeting `Qwen/Qwen3-8B`.

This document is the reference we build *against*. The implementation is being written
fresh; the `charlesfrye/specdec` branch is consulted only for algorithm details, **not**
inherited as structure. That branch's own post-mortem (`docs/specdec_verification.md` on
that branch) is the source of the anti-patterns called out below.

---

## 0. What DFlash is (the contract under test)

DFlash (arXiv 2602.06036, weights `z-lab/Qwen3-8B-DFlash-b16`) is **block-diffusion
speculative decoding**:

- A lightweight ~1B, 5-layer **draft** model cross-attends to **target** hidden states
  from layers `[1, 9, 17, 25, 33]` (concatenated → `fc` 20480→4096 → `hidden_norm`).
- Each step the draft emits a block of `block_size = 16` tokens in one pass.
- The **target** verifies the whole block in a single forward.
- Accept the longest correct prefix, append one **bonus** token, **crop** the target KV
  cache back to `accepted_len + 1`, extract fresh target hidden states, repeat.

| Component | Value |
|---|---|
| Target model | `Qwen/Qwen3-8B` (36 layers) |
| Draft weights | `z-lab/Qwen3-8B-DFlash-b16` (~2.1 GB, 5 layers) |
| Block size | 16 |
| Target capture layers | `[1, 9, 17, 25, 33]` |
| Mask token id | `151669` |

---

## 1. Guiding principle: anchor to invariants, not internals

A test survives the inevitable rewrites of a large feature **only if its assertion never
names anything that gets rewritten.** Three things will not change no matter how the
implementation evolves — every test in the ladder asserts against one of them:

1. **The north-star invariant** — *greedy speculative decoding is token-for-token
   identical to greedy autoregressive decoding.* This holds for **any** draft model. A
   bad or even random draft only lowers the acceptance length (more verify passes); it
   never changes the emitted tokens. This is the anchor for the entire ladder.
2. **External ground truth** — the HF `z-lab` reference (draft numerics) and the existing
   non-spec engine (end-to-end identity). Both live outside our code.
3. **Pure arithmetic** — the acceptance rule, block/position construction, EOS
   truncation, KV-crop bookkeeping. Plain functions, no model.

If you ever feel the urge to assert "module `fc` outputs tensor X" or to compare against a
committed `.pt` snapshot of our own output, stop — that test isn't worth the bytes its written in.

---

## 2. Lessons from the prior attempt (do not repeat)

| Anti-pattern in the old suite | Why it dies during development |
|---|---|
| Golden `.pt` fixtures of *our own* draft output, re-asserted later | Meaningless after reimplementation; only proves the code equals itself |
| `step1_fc`, `step2_attention` keyed to specific internal modules | Break the moment those modules are refactored / renamed / fused |
| The draft "model" wrapped HF's layer class (claimed "bitwise identical") | Tested nothing about a real port; can't be sized down for local tests |
| Single prompt `"What is 2+2?"`, temperature 0, batch size 1 | One prefill length, one acceptance pattern — zero coverage of hard paths |
| Baseline produced by monkeypatching `_draft = None` | Exercised a different path than the real non-spec scheduler |
| Modal-only manual `modal run` scripts, no CI, nothing local | No fast inner loop; nothing runs on every change |

---

## 3. Prerequisite refactors (what makes the ladder cheap *and* durable)

These are not optional polish — they are what allow a meaningful fast local tier to exist.

- **R1. Extract spec logic into pure functions.** Pull the decision rules out of the
  scheduler step into testable functions with no model dependency:
  - `verify_block(draft_tokens, target_greedy) -> (accept_len, bonus_token)`
  - `build_draft_block(last_token, block_size, mask_id) -> (ids, positions)`
  - the `cached_len / device_len / keep_len` rollback bookkeeping
  - EOS truncation of an accepted span + pending-token queue management

  In the old branch these were inline in `SpecDecScheduler._run_specdec_step`. Pulled out,
  they run on CPU in milliseconds — and that is exactly where the worst bugs live (the
  prior `complete_one` bug; the `pid_start = max(0, cached_len - ctx_len + 1)` math).

- **R2. Make the draft model config-sizable.** The old `DFlashDraftModel` called
  `AutoModel.from_pretrained("z-lab/...")` in its constructor just to obtain a layer
  class — hard-wiring it to a 2 GB download and real dimensions. A native, `DFlashConfig`-
  driven draft lets us instantiate a **tiny random** draft (e.g. 2 layers, hidden 128) for
  local plumbing tests with zero downloads. The target is already config-sizable:
  `ModelConfig` is a plain frozen dataclass and the engine supports `use_dummy_weight`.

---

## 4. The graded ladder

Each tier gates the next. When a high tier fails, the lower tiers localize the cause.
Placement (CPU / GPU / Modal) is explicit because it *is* the infra answer.

### Tier 0 — Pure logic · local CPU · milliseconds · no GPU, no weights

| Test | Asserts (ground truth = arithmetic) |
|---|---|
| `accept_rule` | hand-built (draft, target-argmax) pairs → exact `(accept_len, bonus)`; covers reject-at-0, reject-mid-block, full-accept-then-bonus, block_size variants |
| `block_build` | last-token prepend, mask fill, contiguous position ids |
| `rollback_bookkeeping` | `cached_len / device_len / keep_len` correct across a simulated multi-block sequence |
| `eos_truncation` | EOS inside the accepted span truncates and marks finished; pending-token queue drains in order |

### Tier 1 — Plumbing on tiny random models · local GPU (any) or cheapest Modal · seconds · dummy weights, no downloads

| Test | Asserts |
|---|---|
| `spec_eq_ar_tiny` | tiny Qwen3 target (2 layers, hidden 128) + tiny **random** draft → `LLM.generate` greedy == plain AR baseline, token-for-token. Exercises the entire loop for free. |
| `kv_crop` | write K/V, crop, read back: tail zeroed, head intact; reuse slots for a 2nd req → no leakage; attention after crop == fresh forward |
| `multi_request` | 2+ concurrent reqs, mixed lengths → each still identical to its own AR baseline (kills the `batch.size == 1` corner) |
| `accept_len_accounting` | inject a "perfect" draft → `accept_len == block_size` every step; random draft → token counts stay consistent |

### Tier 2 — Numeric fidelity vs HF · Modal GPU + real weights · minutes

Ground truth = the HF `z-lab` model run **live** (or cached to a Modal Volume), never a
committed snapshot of our output.

| Test | Asserts |
|---|---|
| `hidden_capture_vs_hf` | captured layers `[1,9,17,25,33]` ≈ HF `output_hidden_states` (tight rtol) |
| `draft_forward_vs_hf` | draft logits `allclose` HF; argmax exact |

### Tier 3 — End-to-end identity on the real model · Modal GPU · minutes

| Test | Asserts |
|---|---|
| `greedy_identity_suite` | spec == AR token-for-token over a **prompt suite**: short / medium / long, code / math / multilingual, lengths crossing `page_size` and cuda-graph bs buckets |
| `long_prompt` | prompts near 32K and across page boundaries |
| `cuda_graph_eq_eager` | graph-path draft output == eager-path (the prior pad-and-mask graph workaround was never verified) |

### Tier 4 — Sampling correctness · ❌ OUT OF SCOPE (greedy-only)

**Deliberately skipped.** This implementation targets **greedy decoding only**, so
stochastic-sampling correctness is explicitly out of scope. `test_sampling.py` exists as a
placeholder marked `@pytest.mark.skip(reason="out of scope: greedy-only implementation")`
so the decision is recorded *in the suite* rather than silently absent. If temperature > 0
support is ever added, this is the rung to turn on:

| Test (skipped) | Would assert |
|---|---|
| `sampling_distribution` | temperature > 0: seed-matched identity (if the sampler consumes the same randomness on both paths) or a two-sample TV / chi-square test over many draws |

### Tier 5 — Quality parity on real tasks · Modal GPU

| Test | Asserts |
|---|---|
| `gsm8k_parity` | spec vs baseline accuracy, bootstrap CI within tolerance. Greedy ⇒ should be *identical*; catches real-workload integration drift. |

### Tier 6 — Performance / non-regression · Modal GPU *(report; soft-assert)*

| Test | Asserts |
|---|---|
| `speedup` | spec tok/s > 1.0× baseline; soft-check toward paper (~6×) |
| `accept_len_distribution` | acceptance length is reported and plausible (mean > 1) |
| `baseline_no_regression` | with spec **off**, tok/s within X% of `main` — catches the hidden-capture hook taxing the normal decode path (the old hook always ran) |

---

## 5. Testing infrastructure

The key move: **one pytest suite, run in both places** — not bespoke per-step scripts.

### Markers
- `cpu` — Tier 0. No CUDA, no weights.
- `gpu` — Tier 1. Needs CUDA but no large downloads (`use_dummy_weight`).
- `modal` — Tiers 2–6. Real weights + GPU.
- `slow` — anything minutes-scale (Tiers 3, 5, 6) for opt-in selection.

### Inner loop (your machine, no Modal)
```bash
pytest -m cpu            # Tier 0, seconds
```
This is why **R1** matters: push every decision rule out of CUDA code so the fast tier is
*meaningful*, not just a smoke test.

### Pre-merge (one Modal GPU container)
```bash
modal run modal_tests/run_suite.py        # runs `pytest -m "gpu or modal"` inside the container
```
Modal is just the runner. The image is built once (`modal_tests/image.py`); the same test
files and assertions run there as locally. **Status: working** — the `gpu` smoke tier
passes on an L4 (`2 passed in ~7s`).

**Gotchas (each cost a crash-loop to learn):**
- **Ship sibling modules explicitly.** Modal 1.x does *not* auto-mount sibling local
  modules, so `from image import image` fails in the container unless `image.py` ships
  itself via `add_local_file(Path(__file__), "/app/image.py")` (+ `PYTHONPATH=/app`). A
  missing import fails at container *startup*, which **crash-loops** (Modal retries
  startup).
- **Never pipe `modal run` through `tail`.** It hides the streaming logs — including a
  crash-loop — behind the final few lines. Read full output, or run in the background and
  inspect the file.
- `add_local_*` calls are inert when re-executed inside the container, so importing the
  image module there is safe.
- Express local paths with `Path(__file__)` so `modal run` works from any CWD.

### Reference oracle (cached, not committed)
A helper computes HF `spec_generate` / AR outputs on Modal and caches tensors to a Modal
**Volume**. Tiers 2–3 compare against the *live* oracle (or volume cache) — never against
`.pt` blobs checked into git.

### Determinism
Fixed seeds + greedy decode for every identity tier (0–3, 5), so pass/fail is crisp and
there is no sampling noise to explain away.

### Where things actually run
The engine hard-depends on CUDA (FlashInfer, `sgl_kernel`, CUDA graphs, NCCL), so the
truly-CPU tier is **Tier 0** — and that is where the nastiest logic bugs live. Tier 1 needs
*a* GPU but **no downloads**, so it runs on a dev GPU if present, else the cheapest Modal
GPU in well under a minute. Tiers 2+ need the real 2 GB draft + 16 GB target on Modal.

---

## 6. Definition of done, per tier

A tier is "done" when its tests are green **and** the tier below it is still green
(non-regression). Ship gates:

- **Mergeable to a feature branch:** Tiers 0–1 green.
- **Correctness claim:** Tiers 0–3 green (greedy identity proven on the real model over a
  prompt suite, including CUDA-graph and long-prompt paths).
- **Production claim:** Tiers 0–3, 5, 6 green. **Tier 4 (sampling) is out of scope — this
  implementation is greedy-only.**

---

## 7. File inventory (proposed)

`(exists)` marks files already scaffolded as the standing Tier 0 skeleton.

```
docs/specdec_testing.md                  # this document                            (exists)
python/minisgl/specdec/__init__.py       # light namespace pkg (no torch/CUDA)      (exists)
python/minisgl/specdec/logic.py          # Tier 0 pure seams: verify_block, ...     (exists)
tests/specdec/conftest.py                # markers; later: tiny-model fixtures      (exists)
tests/specdec/test_accept_rule.py        # Tier 0  — walking-skeleton demo          (exists)
tests/specdec/test_block_build.py        # Tier 0
tests/specdec/test_rollback.py           # Tier 0
tests/specdec/test_eos.py                # Tier 0
tests/specdec/test_gpu_smoke.py          # gpu smoke canary: image + GPU + kernels    (exists)
tests/specdec/test_spec_eq_ar_tiny.py    # Tier 1
tests/specdec/test_kv_crop.py            # Tier 1
tests/specdec/test_multi_request.py      # Tier 1
tests/specdec/test_vs_hf.py              # Tier 2  (marker: modal)
tests/specdec/test_identity_real.py      # Tier 3  (marker: modal, slow)
tests/specdec/test_sampling.py           # Tier 4  — SKIPPED (greedy-only, out of scope)   (exists)
tests/specdec/test_gsm8k.py              # Tier 5  (marker: modal, slow)
tests/specdec/test_perf.py               # Tier 6  (marker: modal, slow)
modal_tests/run_suite.py                 # Modal runner: pytest inside a GPU container  (exists)
modal_tests/image.py                     # image def; ships itself into the container  (exists)
modal_tests/oracle.py                    # HF reference, cached to a Modal Volume
```

---

## 8. Walking-skeleton discipline (how the ladder is built)

Write each test *fully and end-to-end against the real seam first*, then stub the seam to
`raise NotImplementedError`. The test runs all the way through the call graph — imports,
fixtures, engine wiring, the spec loop — and dies at the first unimplemented leaf.
**Reaching that leaf is the proof the plumbing is wired.** The *failure mode* is the signal:

| What you see | Meaning | Action |
|---|---|---|
| `ImportError` / collection error | Seam missing or unreachable — plumbing broken | Fix wiring first |
| `NotImplementedError` at the leaf | Plumbing right, logic pending | Good "red"; implement the leaf |
| `AssertionError` | Logic present but wrong | Normal TDD red |
| Pass | Done | Remove the skeleton marker |

### The `xfail(raises=..., strict=True)` state machine

Mark every skeleton test:

```python
@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="skeleton: <seam>")
```

With every seam stubbed, all skeleton tests raise `NotImplementedError` → they **xfail** →
the suite is **green**. That green run is the milestone: *the walking skeleton stands up;
plumbing is figured out.* From there each test walks a strict state machine — all four
states observed live in the Tier 0 demo (`tests/specdec/test_accept_rule.py`):

| Seam state | Exception | strict-xfail verdict |
|---|---|---|
| stub | `NotImplementedError` | `XFAIL` (green) — plumbing wired |
| implemented wrong | `AssertionError` | `FAILED` — logic wrong (≠ the `raises` filter) |
| implemented right | none | `FAILED [XPASS(strict)]` — "remove the xfail marker" |
| marker removed | none | `PASSED` — done |

You cannot accidentally leave a stub behind: a *correct* implementation fails (loudly)
until you delete its marker. A wiring regression (`ImportError`) also fails, because it is
not the `NotImplementedError` the marker expects.

### Two mechanics that matter

- **Raise `NotImplementedError` in the test body, not in a fixture.** `xfail` catches
  exceptions from the test call; a fixture that raises is reported as a setup *error*.
  Fixtures assemble inputs that already work (prompts, tiny configs); the stubbed calls
  (`SpecDecLLM(...)`, `verify_block(...)`) belong in the body.
- **Keep one deliberately-passing canary** (e.g. `test_seam_is_importable`) so a green run
  proves collection actually ran the skeleton rather than collecting nothing.

### Build order

1. Stub all seams (`raise NotImplementedError`) — see `python/minisgl/specdec/logic.py`.
2. Write every ladder test in full with the skeleton marker.
3. Run `pytest -m cpu` and confirm everything is **xfail, not error** — the skeleton stands.
4. Implement tier by tier, deleting each marker as it goes green.

### Running the standing skeleton

The smallest standing skeleton already lives in the repo. With `uv` (no CUDA stack, source
on `PYTHONPATH`, project `--cov` addopts neutralized):

```bash
PYTHONPATH=python uv run --no-project --with pytest \
    python -m pytest tests/specdec -o addopts="" -m cpu -ra
# => 1 passed, 4 xfailed in 0.05s
```

