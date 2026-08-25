#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Moondream 3 over brain's D-Bus surface: an image in, generated text out.

Two things this example is actually about.

**The stream.** `caption` is a streaming action, so `Subscribe` delivers
progress while the run is still going. That matters here for the same reason it
does for DeepSeek-OCR: this decoder has no KV cache, so every generated token
re-runs the whole grown sequence through all 24 layers over a 730-row image
prefix. Waiting for the final `Outcome` means waiting for all of them.

**The precision.** `--precision fp32` is accepted and will almost certainly fail
placement, which is the honest behaviour rather than a hidden one. At the
released config the fp32 build is ~43 GiB of weights plus per-block activation
scratch; the int8 build (the default) quantizes the 1280 expert tensors and puts
every block on one shared activation set, ~9 GiB. The scheduler budgets them as
two different instances, so asking for fp32 on a machine without room fails
cleanly instead of evicting a working int8 instance to build one that cannot
fit.

Run it against a private session bus:

    BRAIN_MOONDREAM3_WEIGHTS=<moondream3-preview checkpoint dir> \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 5
        python3 examples/vision/moondream3_caption.py --image photo.ppm --max-new 16'

The first call pays the activation (the checkpoint load and the expert
quantization); every later call on the same server reuses the resident instance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

MODEL = "brain/moondream3"
DEFAULT_PROMPT = "Describe this image."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) to caption")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="the instruction after the image")
    ap.add_argument("--max-new", type=int, default=16, help="tokens to generate (every one is a full recompute)")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"], help="int8 (~9 GiB) or fp32 (~43 GiB)")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    print(f"image {w}x{h}, prompt {args.prompt!r}, max_new {args.max_new}, precision {args.precision}", flush=True)

    brain = BrainDBus()
    if MODEL not in brain.models():
        print(f"FATAL: '{MODEL}' not served (set BRAIN_MOONDREAM3_WEIGHTS)", file=sys.stderr)
        return 2

    t0 = time.time()
    last = [t0]

    def on_delta(text: str) -> None:
        now = time.time()
        # Per-token wall time alongside the fragment: what makes the
        # O(T^2)-recompute cost visible instead of theoretical.
        print(f"[{now - t0:6.1f}s  +{now - last[0]:5.1f}s] {text!r}", flush=True)
        last[0] = now

    out = brain.subscribe(
        MODEL,
        "caption",
        {"prompt": args.prompt, "max_new": args.max_new, "precision": args.precision},
        blobs={"image": img},
        meta={"image": {"media": "image", "meta": {"w": w, "h": h, "c": 3}}},
        on_delta=on_delta,
    )

    print(f"\n--- generated in {time.time() - t0:.1f}s ---")
    print(out.blobs.get("text", b"").decode("utf-8", "replace"))
    print(
        f"prompt_tokens={out.outputs.get('prompt_tokens')} "
        f"completion_tokens={out.outputs.get('completion_tokens')} "
        f"finish_reason={out.outputs.get('finish_reason')!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
