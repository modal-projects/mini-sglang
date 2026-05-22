# DFlash Speculative Decoding Implementation Plan

## Overview

Add **DFlash block-diffusion speculative decoding** to mini-sglang for Qwen3-8B.
- **DFlash arxiv**: [2602.06036](https://arxiv.org/abs/2602.06036) — generates up to 16 tokens per block using a lightweight diffusion draft model, verified by the target model in one forward pass.
- **Pre-trained weights**: `z-lab/Qwen3-8B-DFlash-b16` (~2.1 GB safetensors — 5-layer, 1B-param draft model)
- **Target model**: `Qwen/Qwen3-8B` (36 layers)
- **Block size**: 16 tokens
- **Reported speedup**: 6.17× vs autoregressive, 2.5× vs EAGLE-3

---

## 1. Algorithm Deep Dive

### Phase A — Prefill (once per request)
1. Run target model on prompt tokens → outputs first token + hidden states from layers [1, 9, 17, 25, 33]
2. Store target hidden states as `target_hidden` (the "context feature" for the draft model)

### Phase B — Per-block loop (repeated until EOS or max_tokens)

#### B1. Draft
1. Construct a block of `block_size=16` masked tokens: `[last_accepted_token, MASK, MASK, ..., MASK]`
2. Embed block tokens through **target model's** `embed_tokens` layer → `noise_embedding`
3. Project `target_hidden` through `fc` (20480→4096) + `hidden_norm`
4. Run **draft model** (5-layer transformer) on `noise_embedding`, attending to `target_hidden` via cross-attention
5. Pass draft hidden states through **target model's** `lm_head` → draft logits
6. Sample one draft token per position → `draft_token_ids[1..block_size]`
7. Construct block output: `[last_accepted_token, draft_token_1, ..., draft_token_{block_size-1}]`

#### B2. Verify
1. Run **target model** on entire draft block (standard forward pass)
2. Target model stores block KV in its KV cache
3. Get posterior logits from target → sample greedy: `posterior_token_ids`
4. Compute acceptance length: compare `draft_token_ids[j] == posterior_token_ids[j-1]` for j=1..block_size-1
5. Acceptance: `N = min{j | draft[j] != posterior[j-1] or j == block_size}`
6. Output: accept first N tokens, then append `posterior[N]` (the bonus token)
7. Roll back target KV cache to `start + N + 1` (discard rejected positions)
8. Extract new `target_hidden` from target hidden states at accepted positions

---

## 2. Architecture Changes

### 2.1 Target Model Hidden State Extraction

**Problem**: `Qwen3ForCausalLM.forward()` returns only final logits. We need intermediate hidden states from specific layers.

**Solution**: Add an optional hidden state capture mode via the global `Context`.

```python
# core.py — add to Context:
@dataclass
class Context:
    ...
    _capture_hidden_layers: list[int] | None = None  # e.g. [1, 9, 17, 25, 33]
    _captured_hidden_states: list[torch.Tensor] = field(default_factory=list)
```

In `Qwen3DecoderLayer.forward()`, on each layer, if `layer_id in ctx._capture_hidden_layers`, store the post-attention hidden state. In `Qwen3Model.forward()`, reset and collect.

**Files**: `python/minisgl/core.py`, `python/minisgl/models/qwen3.py`

### 2.2 Target KV Cache Rollback

**Problem**: The target model stores draft block KV entries during verification, then must discard rejected ones.

**Current KV cache**: `MHAKVCache` — 6D buffer `[2, n_layers, n_pages, page_size, heads, head_dim]`. Pages are allocated via `CacheManager`. Radix tree for prefix caching.

**Solution**: 
- Before verification: save the KV cache "watermark" (seq length per layer)
- After verification: invoke `kv_cache.crop(layer_id, keep_len)` which zeroes out KV entries beyond the accepted length
- Since pages are 1-token each (page_size=1 in FlashInfer mode), we can zero out rejected page entries

**Files**: `python/minisgl/kvcache/base.py` (add abstract `crop()`), `python/minisgl/kvcache/mha_pool.py` (implement)

### 2.3 Draft Model Implementation

**Architecture mapping** (port from `modeling_dflash.py` to mini-sglang primitives):

| HF Component | mini-sglang Equivalent | Shape |
|---|---|---|
| `Qwen3DFlashAttention` (5 layers) | Custom `DFlashAttention` | N/A |
| `q_proj`, `k_proj`, `v_proj` (per layer) | `LinearQKVMerged` or separate linears | → split Q/K/V |
| `o_proj` (per layer) | `LinearOProj` | standard |
| `q_norm`, `k_norm` (per layer) | `RMSNorm` | standard |
| `Qwen3RMSNorm` (input/output) | `RMSNormFused` | standard |
| `Qwen3MLP` (per layer) | `GatedMLP` | standard |
| `fc` (20480→4096) | `LinearReplicated` | 5×4096→4096 |
| `hidden_norm` | `RMSNormFused` | standard |
| `rotary_emb` | `Rotary` (reuse) | standard |

**Key difference**: `DFlashAttention` does **cross-attention** where:
- Q: from draft hidden states
- K: concat(target_hidden @ W_K_draft, draft_hidden @ W_K_draft)
- V: concat(target_hidden @ W_V_draft, draft_hidden @ W_V_draft)
- `is_causal=False` (bidirectional)

Since target and draft use the **same** weight matrices, we can implement this by:
1. Computing Q from draft hidden states
2. Computing K/V from BOTH target and draft, concatenating
3. Using standard flash attention with the concatenated KV

**Files**: `python/minisgl/models/draft/` (new package)
- `__init__.py`
- `dflash.py` — `DFlashDraftModel` class

### 2.4 Draft KV Cache

The draft model has its own KV cache (5 layers × block_size tokens). Since this is small and regenerated each iteration, we can use a simple contiguous buffer rather than the full paged KV cache system.

**Solution**: Add a lightweight `DraftKVCache` class:
```python
@dataclass
class DraftKVCache:
    k: torch.Tensor  # [n_layers, batch, block_size, n_kv_heads, head_dim]
    v: torch.Tensor
    def crop(self, keep_len: int): ...  # keep only first keep_len tokens
```

**Files**: `python/minisgl/models/draft/kvcache.py`

### 2.5 Draft Attention Implementation

Since the draft model uses a fundamentally different attention pattern (bidirectional, Q from noise, K/V from target+noise), we cannot use the standard `BaseAttnBackend` interface directly. Two options:

**Option A (v1 — simpler)**: Implement draft attention as a standalone method using `flash_attn_varlen_func` (or eager as fallback) directly in the draft model. Don't use the standard attention backend registry.

**Option B (v2 — more integrated)**: Add a new attention backend `dflash` to the registry.

For v1, use Option A — keep it self-contained in the draft model.

**Files**: `python/minisgl/models/draft/attention.py`

---

## 3. Engine Integration

### 3.1 Engine Changes

`Engine.__init__()` loads:
- Target model (existing)
- Draft model (new) — from `z-lab/Qwen3-8B-DFlash-b16`

`Engine.forward_batch()` gets new mode for specdec batches:
```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    if batch.is_specdec:
        return self._forward_specdec(batch, args)
    # existing path...
```

**Files**: `python/minisgl/engine/engine.py`

### 3.2 SpecDec Batch

New batch type:
```python
@dataclass  
class SpecDecBatch(Batch):
    phase: Literal["draft", "verify"]
    target_hidden: torch.Tensor  # context features for draft model
    draft_block: torch.Tensor    # [batch, block_size] token ids
    draft_kv_cache: DraftKVCache
    target_kv_watermark: int     # target model's seq length before verification
```

It may be cleaner to handle specdec via the scheduler rather than adding it to `Engine.forward_batch()`. The scheduler orchestrates the draft→verify loop, calling `Engine` for each forward.

### 3.3 Draft→Verify Loop (in Scheduler)

```python
def _specdec_step(self, req: Req):
    # 1. Get target model forward (standard decode step)
    target_output = self._forward_standard(batch)
    
    # 2. Extract hidden states from target layers
    target_hidden = extract_hidden_states(target_model, layer_ids=[1,9,17,25,33])
    
    # 3. Prepare draft block (mask tokens)
    draft_block = make_draft_block(last_token, block_size=16, mask_token_id=151669)
    
    # 4. Draft model forward → draft logits → sample draft tokens
    draft_logits = self._forward_draft(draft_block, target_hidden)
    draft_tokens = sample(draft_logits)
    
    # 5. Verify: target model forward on draft block
    verify_batch = make_verify_batch(draft_tokens)
    verify_output = self._forward_target(verify_batch)
    
    # 6. Acceptance check
    accepted_len = check_acceptance(draft_tokens, verify_output)
    
    # 7. Update outputs and KV cache
    accept_tokens(accepted_len)
    rollback_target_kv_cache(accepted_len)
```

**Files**: `python/minisgl/scheduler/specdec.py` (new), modifications to `scheduler.py`

---

## 4. Weight Loading

The draft model weights are in standard HF safetensors format with Qwen3-style naming:
```
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.norm.weight
fc.weight
hidden_norm.weight
```

**Approach**: Since the draft model doesn't use TP (it's small), load with `Replicated` linear layers. Use the existing `load_weight()` infrastructure but without sharding. Add `.qkv_proj` merging logic for attention layers.

**Files**: `python/minisgl/models/draft/weight.py`

---

## 5. Data Flow (End to End)

```
User sends prompt
  │
  ▼
Tokenizer → input_ids
  │
  ▼
Scheduler (specdec mode)
  │
  ├─[1] TARGET Prefill: process all prompt tokens
  │     └─ Store KV cache + extract target_hidden from layers [1,9,17,25,33]
  │
  ├─[2] TARGET Decode step: get first token
  │     └─ output token[0]
  │
  └─[3] SPECDEC LOOP (per block of 16):
        │
        ├─ DRAFT: draft_model(noise_embedding, target_hidden) → lm_head → draft_tokens[1..15]
        │
        ├─ VERIFY: target_model(draft_block) → posterior_tokens
        │
        ├─ ACCEPT: find accepted_len
        │     └─ output tokens[1..accepted_len+1]
        │
        ├─ ROLLBACK: target KV cache crop to accepted_len+1
        │
        └─ UPDATE: extract new target_hidden from accepted positions
              └─ goto [3] if more tokens needed
```

---

## 6. Implementation Phases

### Phase 0: Research & Verification
- [ ] Load the HF draft model and run `spec_generate()` on Qwen3-8B to verify correctness
- [ ] Inspect model output shapes at each step (hidden state dimensions, input/output tensor shapes)
- [ ] Verify block_size=16, mask_token_id=151669, target_layer_ids=[1,9,17,25,33] from config.json
- [ ] Trace through `spec_generate()` with a debugger to capture all intermediate tensor shapes

### Phase 1: Draft Model Implementation
- [ ] Create `python/minisgl/models/draft/` package
- [ ] Implement `DFlashDraftModel` using mini-sglang primitives (BaseOP-based)
- [ ] Implement `DFlashAttention` layer with cross-attention (bidirectional, K/V from target+noise)
- [ ] Implement `DraftKVCache` lightweight KV cache
- [ ] Implement weight loading for draft model (no TP sharding needed)
- [ ] Unit test: draft model outputs match HF reference for known inputs
- [ ] Unit test: draft model attention matches HF eager attention

### Phase 2: Target Model Hidden States
- [ ] Add hidden state capture mechanism to `Context` and `Qwen3Model.forward()`
- [ ] Add `extract_context_feature()` utility (concatenate selected layers → fc → norm)
- [ ] Unit test: hidden states match HF target model's `output_hidden_states`

### Phase 3: KV Cache Rollback
- [ ] Add `BaseKVCachePool.crop(layer_id, keep_len)` method
- [ ] Implement in `MHAKVCache` (zero out pages beyond keep_len)
- [ ] Unit test: rollback + re-use produces correct attention outputs

### Phase 4: Engine Integration
- [ ] Load draft model in `Engine.__init__()` alongside target model
- [ ] Add `Engine.forward_draft()` method
- [ ] Add `Engine.forward_verify()` method
- [ ] Integration test: end-to-end specdec generation matches baseline (token-for-token)

### Phase 5: Scheduler Integration
- [ ] Implement `Scheduler._specdec_step()` with full draft→verify→accept loop
- [ ] Handle EOS detection during specdec
- [ ] Handle batch of multiple requests (each gets its own specdec steps)
- [ ] Add server CLI flag `--speculative-algorithm dflash --speculative-draft-model ...`

### Phase 6: Benchmarks & Correctness
- [ ] Compare output with baseline (BF16 greedy decode) — must be identical
- [ ] Benchmark throughput (tokens/sec) with and without specdec
- [ ] Benchmark latency (time-to-first-token, time-per-output-token)
- [ ] Measure acceptance rate per block
- [ ] Test on various prompt lengths and datasets

### Phase 7: Optimizations (if needed for speedup)
- [ ] CUDA graph capture for draft model forward
- [ ] Overlap draft model compute with target model hidden state extraction
- [ ] Fuse lm_head computation on draft outputs
- [ ] Batch draft model across multiple requests
- [ ] Stream synchronization optimization

---

## 7. Testing Strategy

### 7.1 Correctness Tests (P0)

| Test | Description | Framework |
|---|---|---|
| `test_draft_attention` | DFlash attention output matches HF reference (batch=1, various block sizes) | `torch.allclose` |
| `test_draft_model_forward` | Full draft model forward matches HF `model.forward()` | `torch.allclose` |
| `test_draft_kv_cache` | Draft KV cache produces same results as fresh forward | `torch.allclose` |
| `test_hidden_state_capture` | Target model hidden states at layers [1,9,17,25,33] match HF output | `torch.allclose` |
| `test_kv_cache_rollback` | After rollback, attention outputs match baseline without rollback | `torch.allclose` |
| `test_specdec_output_identical` | 100 prompts → specdec output tokens == baseline greedy tokens | exact match |
| `test_specdec_vs_hf_reference` | Output matches HF `model.spec_generate()` for same prompts | exact match |

### 7.2 Performance Tests (P1)

| Test | Description | Metric |
|---|---|---|
| `bench_specdec_throughput` | Fixed prompts, compare tok/s with/without specdec | tokens/sec |
| `bench_specdec_latency` | Varying load, compare TTFT and TPOT | ms |
| `bench_acceptance_rate` | Acceptance length distribution across blocks | avg/median accept_len |
| `bench_draft_overhead` | Time spent in draft vs verify | ms per step |

### 7.3 Stress Tests (P2)

| Test | Description |
|---|---|
| `test_long_prompts` | 32K token prompts with specdec |
| `test_batch_mixing` | Mix of prompt lengths in one batch |
| `test_eos_handling` | Requests that hit EOS mid-block |
| `test_oom_resilience` | Specdec under memory pressure |

---

## 8. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Draft model slow to load/initialize | Startup time | Load on same device as target, no TP needed |
| Hidden state capture adds overhead | Slowdown | Only capture when specdec enabled; use fused ops |
| KV cache rollback complex with paged cache | Correctness bugs | Start with simple zero-out; add page-level rollback later |
| Draft+target models exceed GPU memory | OOM | Draft model is only 1B (~2GB); use `--memory-ratio` tuning |
| Bidirectional draft attention not supported by FA/FI | Slow draft | Use eager attention for v1, optimize later |
| Batch specdec scheduling complex | Bugs | Start with batch_size=1 specdec; add batching later |

---

## 9. Configuration Interface

New CLI/server flags:
```
--speculative-algorithm dflash
--speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16
--speculative-block-size 16        # default from config
--speculative-num-draft-tokens 15  # block_size - 1
--no-speculative-decode            # disable specdec
```

New `EngineConfig` fields:
```python
speculative_algorithm: str = "none"  # "none" | "dflash"
speculative_draft_model_path: str = ""
speculative_block_size: int = 16
```

---

## 10. File Inventory

### New files
```
python/minisgl/models/draft/__init__.py       # Draft model factory
python/minisgl/models/draft/dflash.py         # DFlashDraftModel, DFlashDecoderLayer
python/minisgl/models/draft/attention.py      # DFlashAttention (cross-attention)
python/minisgl/models/draft/kvcache.py        # DraftKVCache (lightweight)
python/minisgl/models/draft/weight.py         # Draft weight loading
python/minisgl/models/draft/config.py         # DFlashConfig (block_size, target_layer_ids, etc.)
python/minisgl/scheduler/specdec.py           # SpecDecManager (draft->verify->accept loop)
tests/core/test_dflash_draft.py               # Draft model unit tests
tests/core/test_dflash_attention.py           # Draft attention tests
tests/core/test_dflash_specdec.py             # End-to-end specdec tests
tests/core/test_dflash_correctness.py         # Output identity tests
benchmark/specdec/bench_specdec.py            # Specdec benchmarks
```

### Modified files
```
python/minisgl/core.py                        # Add hidden state capture to Context
python/minisgl/models/qwen3.py                # Hidden state capture in decoder layers
python/minisgl/engine/engine.py               # Load draft model, forward_draft/verify
python/minisgl/engine/config.py               # SpecDec config fields
python/minisgl/scheduler/scheduler.py         # SpecDec step integration
python/minisgl/kvcache/base.py                # crop() method for rollback
python/minisgl/kvcache/mha_pool.py            # Implement crop()
python/minisgl/server/args.py                 # CLI flags for specdec
python/minisgl/models/config.py               # DFlash config parsing
python/minisgl/env.py                         # MINISGL_SPECDEC_* env vars
```