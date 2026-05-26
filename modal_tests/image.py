"""Shared Modal image definitions for DFlash tests.

Provides three images:
  - image_base: torch + transformers (for HF reference tests)
  - image: image_base + minisgl deps (flashinfer, etc.)
  - image_minisgl: image_base + minisgl source only (for tests that import
    minisgl module without needing flashinfer/Engine)
"""

import modal

_image_base = (
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

image_minisgl = _image_base.add_local_dir("python/minisgl", "/app/minisgl")

image = (
    image_minisgl.pip_install(
        "flashinfer-python>=0.5.3",
        "sgl_kernel>=0.3.17.post1",
        "pyzmq",
        "msgpack",
        "modelscope",
    )
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
)

image_base = _image_base