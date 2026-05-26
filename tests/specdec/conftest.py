"""Shared pytest config for the DFlash speculative-decoding suite.

Registers the placement markers that grade the ladder (see ``docs/specdec_testing.md``)::

    cpu   - pure logic; no GPU, no weights
    gpu   - needs CUDA but no large downloads (dummy weights)
    modal - needs real weights + GPU (run via modal_tests/run_suite.py)
    slow  - minutes-scale; opt-in
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

_MARKERS = {
    "cpu": "pure logic; no GPU, no weights",
    "gpu": "needs CUDA but no large downloads (dummy weights)",
    "modal": "needs real weights + GPU",
    "slow": "minutes-scale; opt-in",
}


def pytest_configure(config: pytest.Config) -> None:
    for name, desc in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def _make_tiny_qwen3_config(
    num_layers: int = 2,
    hidden_size: int = 128,
    num_attention_heads: int = 2,
    head_dim: int = 64,
    intermediate_size: int = 384,
    vocab_size: int = 32000,
    max_position: int = 256,
) -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": max_position,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture(scope="session")
def tiny_model_path() -> str:
    """Provide a path to a tiny Qwen3 config on disk (zero-download fixture).

    Returns a temp directory containing a minimal ``config.json``, tokenizer
    config, and a small vocab for a 2-layer, hidden-128 Qwen3 variant.
    Combined with ``use_dummy_weight=True`` this yields a tiny, instant-on-device
    target model for Tier 1 plumbing tests.
    """
    config = _make_tiny_qwen3_config()
    tmpdir = tempfile.mkdtemp(prefix="tiny_qwen3_")
    root = Path(tmpdir)

    root.joinpath("config.json").write_text(json.dumps(config))

    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "add_prefix_space": False,
        "model_max_length": 256,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    root.joinpath("tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    vocab: dict[str, int] = {}
    vocab["<|endoftext|>"] = 0
    vocab["!"] = 1
    for i in range(2, 32000):
        vocab[chr(65 + i % 26) + str(i)] = i
    merges: list[str] = []
    with root.joinpath("vocab.json").open("w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    root.joinpath("merges.txt").write_text("\n".join(merges) + "\n")

    return tmpdir


@pytest.fixture(scope="session")
def tiny_llm(tiny_model_path: str):
    """Session-scoped tiny LLM for all Tier 1 GPU tests.

    The engine asserts ``not cuda.is_initialized()``, so only ONE LLM can be
    created per process. Session scope ensures all GPU test modules share it.
    Skipped entirely when there is no CUDA device.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
    except (ImportError, AssertionError):
        pytest.skip("no CUDA device")

    from minisgl.llm import LLM

    llm = LLM(
        model_path=tiny_model_path,
        use_dummy_weight=True,
        max_seq_len_override=128,
        num_page_override=256,
        cuda_graph_max_bs=0,
    )
    yield llm
    llm.engine.shutdown()


@pytest.fixture
def tiny_block_config() -> dict:
    """Convenience: return the same tiny config dict without writing to disk."""
    return _make_tiny_qwen3_config()