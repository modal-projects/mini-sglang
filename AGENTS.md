# AGENTS.md — DFlash Speculative Decoding (handoff)

Context for any agent (OpenCode, Claude Code, …) taking over the effort to add **DFlash
speculative decoding** to mini-sglang for `Qwen/Qwen3-8B`. Consolidated from a prior
session's working notes so it is discoverable on its own.

## Branch

**`charlesfrye/specdec-walking-skeleton`** — pushed to `fork` (modal-projects/mini-sglang).

## Authoritative references
- **`docs/specdec_testing.md` — read this first.** The test design: the graded ladder
  (Tier 0–6), the walking-skeleton discipline, and the testing infrastructure. It is the
  spec you build against.
- **`charlesfrye/specdec` git branch — reference only, not inherited.** A prior attempt;
  its tests were judged brittle. That branch's `docs/specdec_verification.md` is a useful
  self-post-mortem. The new implementation is being built **fresh**.

## What DFlash is here
Block-diffusion speculative decoding (arXiv 2602.06036, weights
`z-lab/Qwen3-8B-DFlash-b16`). A ~1B, 5-layer draft model cross-attends to target hidden
states from layers `[1, 9, 17, 25, 33]`, emits a 16-token block in one pass; the target
verifies the block in one forward. Accept the longest correct prefix + one bonus token,
crop the target KV cache to `accept_len + 1`, extract fresh target hidden states, repeat.
Mask token id `151669`.

## Scope & key decisions
- **Greedy decoding only.** Tier 4 (stochastic sampling correctness) is **out of scope**;
  `tests/specdec/test_sampling.py` is a skipped placeholder.
- **Test philosophy — anchor to invariants, not internals.** Every assertion must target
  one of three rewrite-proof things: (1) the invariant *greedy spec decode == greedy AR
  decode token-for-token, for ANY draft model*; (2) external ground truth (HF `z-lab`
  reference, the existing non-spec engine); (3) pure arithmetic.
- **Two prerequisite refactors — DONE.** R1: extract pure spec logic (`verify_block`,
  `build_draft_block`, `compute_keep_len`, `compute_rollback_state`, `truncate_at_eos`)
  into model-free functions in `minisgl.specdec.logic`. R2: make the draft
  model config-sizable via `DFlashDraftConfig` + pure-torch `DFlashDraftModel` (no
  `AutoModel.from_pretrained` — `DFlashDraftConfig.tiny()` factory for local plumbing).
- Pure seams live in the **light `minisgl/specdec/` namespace package** (NOT
  `minisgl/scheduler/`) so importing them does not drag in the CUDA engine.

## Current state — completed

### Tier 0: Pure logic — **25/25 PASSED** (0.04s on CPU)
| Test file | What it tests |
|---|---|
| `test_accept_rule.py` | `verify_block(draft, target_greedy) → (accept_len, bonus)` — 4 parametrized cases + 1 canary |
| `test_block_build.py` | `build_draft_block(last_token, block_size, mask_id) → (ids, positions)` — 7 cases + 1 canary |
| `test_rollback.py` | `compute_keep_len`, `compute_rollback_state` — 6 cases |
| `test_eos.py` | `truncate_at_eos(tokens, eos_id) → (truncated, finished)` — 5 cases |

### Walking skeleton: complete (47 tests collect, 0 import errors)
| Tier | Files | Status |
|---|---|---|
| 0 (cpu) | test_accept_rule, test_block_build, test_rollback, test_eos | **25 passed** |
| 1 (gpu) | test_kv_crop, test_spec_eq_ar_tiny, test_multi_request | 8 xfail(NotImplementedError) |
| 2 (modal) | test_vs_hf | 2 xfail(NotImplementedError) |
| 3 (modal, slow) | test_identity_real | 5 xfail(NotImplementedError) |
| 4 (out of scope) | test_sampling | 1 skip |
| 5 (modal, slow) | test_gsm8k | 1 xfail(NotImplementedError) |
| 6 (modal, slow) | test_perf | 3 xfail(NotImplementedError) |
| — (gpu smoke) | test_gpu_smoke | 2 passing canaries (GPU needed) |

### Infrastructure built
| File | What it provides |
|---|---|
| `python/minisgl/specdec/logic.py` | `verify_block`, `build_draft_block`, `compute_keep_len`, `compute_rollback_state`, `truncate_at_eos` — all implemented, no torch/CUDA |
| `python/minisgl/specdec/draft.py` | `DFlashDraftConfig` (frozen dataclass + `.tiny()` factory), `DFlashDraftModel` (nn.Module with self-attn, cross-attn, MLP), `DraftSelfAttention`, `DraftCrossAttention`, `DraftMLP`, `DraftDecoderLayer` |
| `python/minisgl/specdec/kvcache.py` | `crop_kv_cache(kv_buffer, keep_len)` — zeros tail beyond keep_len |
| `python/minisgl/specdec/engine.py` | `SpecDecLLM(LLM)` — initializes draft model + capture hooks; `generate()` raises `NotImplementedError` |
| `python/minisgl/core.py` | Added `Context.capture_layers: set[int]` and `Context.captured_hidden_states: dict` for non-invasive hidden-state capture |
| `python/minisgl/models/qwen3.py` | `Qwen3Model.forward` checks `ctx.capture_layers` and populates `ctx.captured_hidden_states` |
| `tests/specdec/conftest.py` | `tiny_model_path` fixture (session-scoped, writes tiny Qwen3 config.json to temp dir), marker registration |

### How to run (use `uv`; there is no repo `.venv`)
Local CPU inner loop (Tier 0, seconds):
```bash
PYTHONPATH=python uv run --no-project --with pytest \
    python -m pytest tests/specdec -o addopts="" -m cpu -ra
# => 25 passed in 0.04s
```
Modal GPU:
```bash
uvx modal run modal_tests/run_suite.py                          # -m gpu on an L4 (~7s)
uvx modal run modal_tests/run_suite.py --markers "gpu or modal" --gpu H100
```

## Suggested next step: implement the spec-dec loop

The remaining work to make Tier 1 tests green is implementing `SpecDecLLM.generate()`.
The infrastructure is ready:

1. **Draft model** (`specdec/draft.py`): config-sizable, pure torch, cross-attention to target hidden states
2. **Hidden state capture** (`core.py` + `qwen3.py`): set `ctx.capture_layers` before target forward, read `ctx.captured_hidden_states` after
3. **Acceptance logic** (`specdec/logic.py`): `verify_block`, `compute_keep_len`, `build_draft_block`
4. **KV crop** (`specdec/kvcache.py`): `crop_kv_cache`

### The spec loop per-step flow:
```
1. Target decode forward (1 token) with capture_layers set
   → next_token + captured hidden states
2. Draft forward: [next_token, MASK, ..., MASK] → B draft_tokens
3. Target verify forward: [next_token] + draft_tokens (B+1 inputs)
   → target_greedy (B+1 predictions)
4. verify_block(draft_tokens, target_greedy) → (accept_len, bonus)
5. Crop KV cache to keep_len = cached_len + accept_len + 1
6. Append accepted prefix + bonus to output
```

### Integration approaches (trade-off):
- **Option A**: Override scheduler's `_forward` for decode batches to run spec steps
  instead of single-token decode. Requires modifying `ForwardOutput` to return
  multi-token results per request.
- **Option B**: Write a standalone spec loop in `SpecDecLLM.generate()` that uses
  `engine.forward_batch` directly for target forwards, builds verify batches manually,
  and manages KV cache/pages directly. More self-contained but replicates some
  scheduler logic.
- **Option C**: Extend the scheduler with a `_spec_decode_step` method. Decode
  requests enter spec mode after prefill. Each spec step replaces one normal decode
  step. Cleanest long-term but most invasive.

### Key challenges for the verify forward:
- Building a batch with `B+1` tokens per request (extend_len > 1 like prefill)
- Setting up page table entries for the new positions
- Configuring attention metadata (backend-specific: flashinfer/FA/trtllm)
- `cached_len` adjustments: the verify batch re-processes the last real token
  at position `cached_len - 1`, then draft tokens at positions `cached_len` to `cached_len + B - 1`

### After Tier 1 is green:
- Remove xfail markers from test_spec_eq_ar_tiny.py and test_multi_request.py
- Test with `modal run` on a GPU container
- Then proceed to Tier 2 (numeric fidelity vs HF `z-lab` reference)

## Modal gotchas (each cost a crash-loop to learn)
- **Modal 1.x does NOT auto-mount sibling local modules.** `from image import image` fails
  in the container unless `image.py` ships itself via `add_local_file(Path(__file__),
  "/app/image.py")` (+ `PYTHONPATH=/app`). A missing import fails at container *startup*,
  which **crash-loops** (Modal retries startup).
- **Never pipe `modal run` through `tail`** — it hides the streaming logs.
- `add_local_*` calls are inert when re-executed inside the container.
- Express local paths with `Path(__file__)` so `modal run` works from any CWD.
- Prefer `uvx modal …` (consistent with using `uv` throughout this repo).