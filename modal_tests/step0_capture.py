"""Step 0: Capture golden fixtures from HF reference implementation.

Loads Qwen3-8B + DFlash draft model via HuggingFace, runs spec_generate
on a short prompt at temperature 0, and saves every intermediate tensor
as .pt files so downstream steps can validate their outputs.

Run: modal run modal_tests/step0_capture.py

Requires:
  - modal volume create hf-cache
  - modal volume create dflash-fixtures
  - modal secret create huggingface-secret HF_TOKEN=<your_token>
"""

import modal

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "datasets",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App("dflash-step0")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
fixture_vol = modal.Volume.from_name("dflash-fixtures", create_if_missing=True)

PROMPT = "What is 2 + 2?"


@app.function(
    image=_image,
    gpu="A100",
    volumes={
        "/cache": hf_cache,
        "/fixtures": fixture_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def capture_golden():
    """Run HF reference spec_generate and save all intermediates."""
    import glob
    import json
    import os
    import sys

    os.environ["HF_HOME"] = "/cache/huggingface"

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    print("Loading models...")
    draft = AutoModel.from_pretrained(
        "z-lab/Qwen3-8B-DFlash-b16",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print("Models loaded.")

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    captures = _run_capture(draft, target, inputs["input_ids"], tokenizer)
    _save_fixtures(captures, fixture_vol)
    print("Done. Fixtures saved to /fixtures/")


def _run_capture(draft, target, input_ids, tokenizer):
    """Copy of spec_generate with instrumentation at each step."""
    import sys

    import torch
    from transformers import DynamicCache

    draft_mod = sys.modules[draft.__class__.__module__]
    extract_context_feature = draft_mod.extract_context_feature

    block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    max_new_tokens = 20
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = target.device

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    result = {}

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ── Prefill ──
    print("  Prefill...")
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = first_token
    result["input_ids"] = input_ids.cpu().clone()
    result["first_token"] = first_token.cpu().clone()

    target_hidden_raw = extract_context_feature(
        output.hidden_states, draft.target_layer_ids
    )
    result["prefill_target_hidden_raw"] = target_hidden_raw.cpu().clone()
    target_hidden = target_hidden_raw

    # ── Decode blocks ──
    start = num_input_tokens
    block_idx = 0

    while start < max_length:
        print(f"  Block {block_idx} (start={start})...")
        block_dir = f"block_{block_idx}"

        # --- Draft ---
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)

        result[f"{block_dir}/block_input_ids"] = block_output_ids.cpu().clone()
        result[f"{block_dir}/noise_embedding"] = noise_embedding.cpu().clone()

        # fc + hidden_norm
        target_hidden_projected = draft.hidden_norm(draft.fc(target_hidden))
        result[f"{block_dir}/pre_fc_target_hidden"] = target_hidden.cpu().clone()
        result[f"{block_dir}/fc_norm_output"] = target_hidden_projected.cpu().clone()

        # Draft forward matching reference: position_ids span past_kv_len..start+block_size
        draft_pos = position_ids[:, past_key_values_draft.get_seq_length() : start + block_size]
        draft_out = draft(
            position_ids=draft_pos,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        past_key_values_draft.crop(start)
        result[f"{block_dir}/draft_model_output"] = draft_out.cpu().clone()

        draft_logits = target.lm_head(draft_out)[:, -block_size + 1 :, :]
        result[f"{block_dir}/draft_logits"] = draft_logits.cpu().clone()

        draft_token_ids = torch.argmax(draft_logits, dim=-1)
        result[f"{block_dir}/draft_token_ids"] = draft_token_ids.cpu().clone()
        block_output_ids[:, 1:] = draft_token_ids

        # --- Verify ---
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        result[f"{block_dir}/target_logits"] = output.logits.cpu().clone()

        posterior_token_ids = torch.argmax(output.logits, dim=-1)
        result[f"{block_dir}/posterior_token_ids"] = posterior_token_ids.cpu().clone()

        # --- Acceptance ---
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior_token_ids[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        result[f"{block_dir}/acceptance_length"] = torch.tensor(
            [acceptance_length], dtype=torch.int32
        )

        accepted = torch.cat(
            [
                block_output_ids[:, : acceptance_length + 1],
                posterior_token_ids[:, acceptance_length : acceptance_length + 1],
            ],
            dim=1,
        )
        result[f"{block_dir}/accepted_token_ids"] = accepted.cpu().clone()

        output_ids[:, start : start + acceptance_length + 1] = accepted[:, :-1]
        output_ids[:, start + acceptance_length + 1] = accepted[:, -1:]
        start += acceptance_length + 1

        past_key_values_target.crop(start)

        new_target_hidden_raw = extract_context_feature(
            output.hidden_states, draft.target_layer_ids
        )
        new_target_hidden_raw = new_target_hidden_raw[:, : acceptance_length + 1, :]
        result[f"{block_dir}/new_target_hidden_raw"] = new_target_hidden_raw.cpu().clone()
        target_hidden = new_target_hidden_raw

        block_idx += 1

        if start >= max_length:
            break

    result["output_ids"] = output_ids[:, :start].cpu().clone()
    result["num_blocks"] = block_idx
    result["decoded_text"] = tokenizer.decode(
        output_ids[0, num_input_tokens:start], skip_special_tokens=True
    )

    # ── Baseline (non-spec) ──
    print("  Computing baseline...")
    with torch.inference_mode():
        gen_out = target.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result["baseline_output_ids"] = gen_out.cpu().clone()
    result["baseline_decoded"] = tokenizer.decode(
        gen_out[0, num_input_tokens:], skip_special_tokens=True
    )

    spec_tokens = result["output_ids"][0, num_input_tokens:]
    base_tokens = result["baseline_output_ids"][0, num_input_tokens:]
    min_len = min(len(spec_tokens), len(base_tokens))
    match = torch.equal(spec_tokens[:min_len], base_tokens[:min_len])
    result["specdec_matches_baseline"] = match
    print(f"  SpecDec == Baseline: {match}")
    if not match:
        first_diff = (spec_tokens[:min_len] != base_tokens[:min_len]).nonzero(
            as_tuple=True
        )[0]
        if len(first_diff) > 0:
            print(f"  First divergence at token {first_diff[0].item()}")
            print(f"    Spec: {spec_tokens[first_diff[0].item()].item()}")
            print(f"    Base: {base_tokens[first_diff[0].item()].item()}")

    return result


def _save_fixtures(captures, volume):
    """Save all captured tensors to /fixtures/."""
    import json
    import os

    import torch

    base = "/fixtures"
    os.makedirs(base, exist_ok=True)

    num_blocks = captures.pop("num_blocks")
    decoded_text = captures.pop("decoded_text")
    baseline_decoded = captures.pop("baseline_decoded")
    match = captures.pop("specdec_matches_baseline")

    metadata = {
        "prompt": PROMPT,
        "num_blocks": num_blocks,
        "block_size": 16,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
        "specdec_matches_baseline": match,
        "decoded_text": decoded_text,
        "baseline_decoded": baseline_decoded,
    }
    with open(f"{base}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    saved = 0
    for name, tensor in captures.items():
        if isinstance(tensor, torch.Tensor):
            subdir = os.path.dirname(name)
            if subdir:
                os.makedirs(f"{base}/{subdir}", exist_ok=True)
            torch.save(tensor, f"{base}/{name}.pt")
            print(f"  Saved {base}/{name}.pt ({tuple(tensor.shape)})")
            saved += 1

    volume.commit()
    print(f"  Saved {saved} tensors across {num_blocks} blocks")