"""Run the spec-decoding pytest suite inside a Modal GPU container.

Modal is just the *runner* — the same ``tests/specdec`` suite executes here as locally.
First milestone is the ``gpu`` smoke tier (no weights): proving image + GPU + CUDA kernels.

The image lives in the sibling ``image.py`` and is shipped into the container by that
module (see its docstring); ``modal run`` works from any working directory.

    modal run modal_tests/run_suite.py                       # default: -m gpu on an L4
    modal run modal_tests/run_suite.py --markers "gpu or modal" --gpu H100

Real-model tiers (2, 3, 5, 6) need a bigger GPU than the L4 default — Qwen3-8B + draft want
an A100/H100. The smoke canary is happy on the cheap L4.
"""

import subprocess
import sys

import modal

from image import image

app = modal.App("minisgl-specdec-tests")


@app.function(image=image, gpu="L4", timeout=1800)
def run_pytest(markers: str, paths: str) -> int:
    cmd = [
        sys.executable, "-m", "pytest", paths,
        "-m", markers,
        "-o", "addopts=",  # neutralize the project's --cov/--strict-* addopts
        "-ra", "-v",
    ]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd="/app").returncode


@app.local_entrypoint()
def main(markers: str = "gpu", paths: str = "tests/specdec", gpu: str = "L4"):
    code = run_pytest.with_options(gpu=gpu).remote(markers, paths)
    if code != 0:
        raise SystemExit(code)
