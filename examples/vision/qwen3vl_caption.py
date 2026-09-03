#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Qwen3-VL over brain's D-Bus surface: an image + a prompt in, generated text out.

Unlike Moondream 3's example in this same directory, Qwen3-VL's decode IS
KV-cached (`Qwen3Vl::generate_cb`, real M-RoPE + DeepStack support carried
through the incremental decode path): one masked prefill seeds the cache, then
each token is an `O(1)`-past-prefill step, not a full recompute. `generate` is
still declared `.streaming()` and this example still uses `Subscribe`, but for
the ordinary reason - seeing partial output as it is produced - rather than
because the alternative would hide a quadratic cost.

**Precision.** `--precision fp32` (the default) is the exact decoder tier;
`--precision int8` is explicitly LOSSY (~4x less weight traffic per token) and
must be asked for by name - it never arrives by falling back. The scheduler
budgets the two as separate resident instances, so a stray request at one
precision cannot silently evict a working instance at the other.

**Capacity, not a per-request resize limit.** `--max-pixels` sizes the
resident's DeepStack/splice buffer CAPACITY at build time (a practical default
around a 1024x1024 image); a request whose smart-resized image needs more
visual tokens than that fails loudly rather than silently truncating. Raise it
if your images are bigger, and the server rebuilds the resident for the new
capacity.

Run it against a private session bus:

    BRAIN_QWEN3VL_WEIGHTS=<Qwen3-VL-4B checkpoint dir-or-GGUF> \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 5
        python3 examples/vision/qwen3vl_caption.py --image photo.ppm --prompt "Describe this image."'

The first call pays the activation (checkpoint load + upload); every later
call on the same server reuses the resident instance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

MODEL = "brain/qwen3vl"
DEFAULT_PROMPT = "Describe this image."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) to describe")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="the instruction after the image")
    ap.add_argument("--max-new", type=int, default=64, help="max tokens to generate")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"], help="fp32 (default, exact) or int8 (lossy)")
    ap.add_argument("--max-pixels", type=int, default=None, help="resident capacity override (pixels); default is this resident's own")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    print(f"image {w}x{h}, prompt {args.prompt!r}, max_new {args.max_new}, precision {args.precision}", flush=True)

    brain = BrainDBus()
    if MODEL not in brain.models():
        print(f"FATAL: '{MODEL}' not served (set BRAIN_QWEN3VL_WEIGHTS)", file=sys.stderr)
        return 2

    params = {"prompt": args.prompt, "max_new": args.max_new, "precision": args.precision}
    if args.max_pixels is not None:
        params["max_pixels"] = args.max_pixels

    t0 = time.time()
    last = [t0]

    def on_delta(text: str) -> None:
        now = time.time()
        print(f"[{now - t0:6.1f}s  +{now - last[0]:5.1f}s] {text!r}", flush=True)
        last[0] = now

    out = brain.subscribe(
        MODEL,
        "generate",
        params,
        blobs={"image": img},
        meta={"image": {"media": "image", "meta": {"w": w, "h": h, "c": 3}}},
        on_delta=on_delta,
    )

    print(f"\n--- generated in {time.time() - t0:.1f}s ---")
    print(out.blobs.get("text", b"").decode("utf-8", "replace"))
    print(f"tokens={out.outputs.get('tokens')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
