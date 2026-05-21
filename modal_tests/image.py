"""Shared Modal image definition for DFlash tests.

Provides two images:
  - _image_base: torch + transformers + safetensors (for HF reference + pure-torch tests)
  - image: extends base with mini-sglang deps (for integration tests that import minisgl)
"""

import modal

# Python version matched to pyproject.toml
# torch==2.9.1 and transformers==4.57.3 matched to DFlash paper
_image_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

image = (
    _image_base.pip_install(
        "flashinfer-python>=0.5.3",
        "sgl_kernel>=0.3.17.post1",
        "pyzmq",
        "msgpack",
        "modelscope",
    )
    .add_local_dir(
        "python/minisgl",
        "/app/minisgl",
    )
    .add_local_file(
        "pyproject.toml",
        "/app/pyproject.toml",
    )
)

image_base = _image_base