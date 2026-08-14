#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate five views of a porcupine-styled Toyota pickup with HOT weights.

Spawns ONE ``brain serve --stdio`` server, keeps it alive, and fires five ``text2image``
requests back-to-back. The server loads the ~20 GB of Z-Image weights (Qwen-4B
encoder + int8 DiT + VAE) on the FIRST request and keeps them resident, so calls
2..5 skip the load entirely and are fast — the whole point of a persistent
connection vs. re-running ``brain infer`` (which reloads everything each time).

Run:
    python -m brain_py.examples.porcupine_trucks           # 512x512, int8
    OUT=/tmp SIZE=384 STEPS=8 python -m brain_py.examples.porcupine_trucks

Needs the Z-Image weight paths in the environment (BRAIN_S3DIT_DIT/_VAE/_QWEN/
_TOKENIZER); this script fills in the on-box defaults if they are unset.
"""
import os
import sys
import time

# Weight locations are configuration, never hard-coded — the server reads them
# from the environment. Fail fast (with guidance) if they are not set, rather than
# baking in machine-specific paths.
_REQUIRED = ["BRAIN_S3DIT_DIT", "BRAIN_S3DIT_VAE", "BRAIN_S3DIT_QWEN", "BRAIN_S3DIT_TOKENIZER"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    sys.stderr.write(
        "error: set the Z-Image weight paths in the environment before running:\n  "
        + "\n  ".join(f"export {k}=/path/to/..." for k in _missing)
        + "\n(these point at the diffusion_models / vae / text_encoder / tokenizer files)\n"
    )
    raise SystemExit(2)

# Allow running straight from the repo without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from brain_py.client import BrainClient  # noqa: E402

ANGLES = [
    "front three-quarter view",
    "side profile view",
    "rear three-quarter view",
    "direct head-on front view, headlights lit",
    "high angle view looking down",
]
PROMPT = (
    "a Toyota pickup truck restyled as a porcupine, the entire body bristling "
    "with sharp quills instead of paint, {angle}, detailed studio product "
    "photograph, dramatic rim lighting, neutral background"
)


def main() -> int:
    out = os.environ.get("OUT", "/tmp/braincaps")
    size = int(os.environ.get("SIZE", "512"))
    steps = int(os.environ.get("STEPS", "8"))
    precision = os.environ.get("PRECISION", "int8")
    os.makedirs(out, exist_ok=True)

    print(f"spawning `brain serve --stdio` (weights load on first request)…", flush=True)
    client = BrainClient(device="cpu")  # z-image drives GPU explicitly; device is ignored by it
    saved = []
    try:
        t_all = time.time()
        for i, angle in enumerate(ANGLES):
            prompt = PROMPT.format(angle=angle)
            t = time.time()
            img = client.text2image(prompt, width=size, height=size, steps=steps,
                                    seed=100 + i, precision=precision)
            dt = time.time() - t
            path = os.path.join(out, f"porcupine_{i}.png")
            img.save(path)
            saved.append(path)
            tag = "first call — loads + builds weights" if i == 0 else "HOT — weights resident"
            print(f"  [{i + 1}/5] {dt:6.1f}s  ->  {path}   ({tag})", flush=True)
        print(f"all 5 in {time.time() - t_all:.1f}s total", flush=True)
    finally:
        client.close()

    print("saved:", *saved, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
