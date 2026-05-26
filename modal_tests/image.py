"""Modal image for the mini-sglang spec-decoding test suite.

A reusable module (future `oracle.py` will import it too). Modal 1.x does NOT auto-mount
sibling local modules into the container, so this file ships *itself* via
``add_local_file(Path(__file__), "/app/image.py")``; combined with ``PYTHONPATH=/app`` that
makes ``from image import image`` resolve both locally and inside the container. Without
this, the container fails ``from image import image`` at startup and crash-loops.

All local paths are expressed relative to this file with ``Path(__file__)`` so ``modal
run`` works from any working directory.
"""

from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent  # modal_tests/ -> repo root

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libnuma1")  # sgl_kernel runtime dep
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python>=0.5.3",
        "sgl_kernel>=0.3.17.post1",
        "msgpack",
        "pyzmq",
        "modelscope",
        "pytest",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/app",
        }
    )
    .add_local_file(_HERE, "/app/image.py")
    .add_local_dir(str(_REPO / "python" / "minisgl"), "/app/minisgl")
    .add_local_dir(str(_REPO / "tests"), "/app/tests")
    .add_local_file(str(_REPO / "pyproject.toml"), "/app/pyproject.toml")
)
