# AGENTS.md — DFlash Speculative Decoding (handoff)

Context for any agent (OpenCode, Claude Code, …) taking over the effort to add **DFlash
speculative decoding** to mini-sglang for `Qwen/Qwen3-8B`. Consolidated from a prior
session's working notes so it is discoverable on its own.

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
  reference, the existing non-spec engine); (3) pure arithmetic. Never assert against
  committed `.pt` snapshots of our own output, nor against internal module names.
- **Two prerequisite refactors.** R1: extract pure spec logic (`verify_block`,
  `build_draft_block`, rollback bookkeeping) into model-free functions. R2: make the draft
  model config-sizable (the old one hard-wired an `AutoModel.from_pretrained` download in
  its constructor, blocking tiny local instances).
- Pure seams live in the **light `minisgl/specdec/` namespace package** (NOT
  `minisgl/scheduler/`) so importing them does not drag in the CUDA engine — that is what
  keeps `pytest -m cpu` GPU-free.

## Current state (already scaffolded)
- **Tier 0 walking skeleton:** `python/minisgl/specdec/{__init__,logic}.py` (stub
  `verify_block` → `raise NotImplementedError`), `tests/specdec/conftest.py` (registers
  markers `cpu`/`gpu`/`modal`/`slow`), `tests/specdec/test_accept_rule.py` (passing canary
  + 4 `xfail(strict)` cases).
- **GPU smoke canary:** `tests/specdec/test_gpu_smoke.py` (torch sees GPU +
  `flashinfer`/`sgl_kernel.flash_attn` import).
- **Modal harness — working:** `modal_tests/run_suite.py` + `modal_tests/image.py`.

## How to run (use `uv`; there is no repo `.venv`)
Local CPU inner loop (Tier 0, seconds):
```bash
PYTHONPATH=python uv run --no-project --with pytest \
    python -m pytest tests/specdec -o addopts="" -m cpu -ra
# => 1 passed, 4 xfailed   (the skeleton stands)
```
Modal GPU:
```bash
uvx modal run modal_tests/run_suite.py                          # -m gpu on an L4 (~7s)
uvx modal run modal_tests/run_suite.py --markers "gpu or modal" --gpu H100
# runs `pytest -m <markers> tests/specdec` inside a GPU container
```

## Walking-skeleton discipline (see `docs/specdec_testing.md` §8)
Write each test end-to-end against the real seam, stub the seam to `raise
NotImplementedError`, and mark it `@pytest.mark.xfail(raises=NotImplementedError,
strict=True)`. An all-xfail run is green and means "plumbing wired". From there: a *correct*
implementation makes the strict xfail fail with `XPASS` ("remove the marker"); a *wrong*
one raises `AssertionError` (hard fail); a wiring break raises `ImportError` (hard fail).
Keep one deliberately-passing canary so a green run proves collection actually happened.

## Modal gotchas (each cost a crash-loop to learn)
- **Modal 1.x does NOT auto-mount sibling local modules.** `from image import image` fails
  in the container unless `image.py` ships itself via `add_local_file(Path(__file__),
  "/app/image.py")` (+ `PYTHONPATH=/app`). A missing import fails at container *startup*,
  which **crash-loops** (Modal retries startup).
- **Never pipe `modal run` through `tail`** — it hides the streaming logs, including a
  crash-loop banner. Read full output, or background it and inspect the file. Debug a
  stopped app: `uvx modal app list` → `uvx modal app logs <app-id>`.
- `add_local_*` calls are inert when re-executed inside the container, so importing the
  image-defining module there is safe.
- Express local paths with `Path(__file__)` so `modal run` works from any CWD.
- Prefer `uvx modal …` (consistent with using `uv` throughout this repo).

## Suggested next step
The first `gpu`-tier xfail-stub test — `test_kv_crop` or `test_spec_eq_ar_tiny` — which
needs the R1/R2 refactors plus tiny-random-model fixtures in `tests/specdec/conftest.py`.
