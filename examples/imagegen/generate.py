#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""FLUX.2 Klein text-to-image over brain's D-Bus interface.

Subscribes to the streaming `text2image` action: progress frames arrive per
denoise step (printed as a step counter), then the generated image arrives as
a `blob` frame whose payload is an out-of-band memfd (HWC f32 `[0,1]`), saved
here as a binary PPM.

Run under a private session bus (weights via env — see the README):

    dbus-run-session -- bash -c '
      BRAIN_FLUX2_DIT=… BRAIN_FLUX2_VAE=… BRAIN_FLUX2_TE=… BRAIN_FLUX2_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/imagegen/generate.py --prompt "a red fox in the snow"'

Requires: jeepney (pip install brain-py[dbus]).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd  # noqa: E402
from brain_py.image import save_ppm  # noqa: E402

MODEL = "flux2-klein"


def run_streaming(brain: BrainDBus, action: str, params: dict, out: str, **kw) -> int:
    """Drive one streaming generation; save the image blob; return exit code."""
    t0 = time.monotonic()
    image = None  # (bytes, meta)
    for frame, fds in brain.subscribe(MODEL, action, params, timeout=7200.0, **kw):
        kind = frame.get("type")
        if kind == "progress":
            print(f"  [{frame['step']}/{frame['total']}] {frame.get('message', '')}", flush=True)
        elif kind == "blob" and frame.get("name") == "image":
            image = (read_fd(fds[0]), frame.get("meta") or {})
        elif kind == "done":
            print(f"  done: {frame.get('result')} ({time.monotonic() - t0:.1f}s)")
        elif kind == "error":
            print(f"  ERROR: {frame.get('message')}", file=sys.stderr)
            return 1
    if image is None:
        print("  no image blob arrived", file=sys.stderr)
        return 1
    data, meta = image
    w, h, c = int(meta.get("w", 0)), int(meta.get("h", 0)), int(meta.get("c", 3))
    save_ppm(out, data, w, h, c)
    print(f"wrote {out} ({w}x{h})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True, help="text description of the desired image")
    ap.add_argument("--out", default="flux2.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=512, help="output width (multiple of 16)")
    ap.add_argument("--height", type=int, default=512, help="output height (multiple of 16)")
    ap.add_argument("--steps", type=int, default=0, help="denoise steps (0 = variant default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="klein-4b", choices=["klein-4b", "klein-9b", "base-4b", "base-9b"])
    ap.add_argument("--adapter", default="", help="server-side path of a trained LoRA adapter")
    args = ap.parse_args()

    params = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "variant": args.variant,
    }
    if args.adapter:
        params["adapter"] = args.adapter

    with BrainDBus() as brain:
        models = brain.models()
        if MODEL not in models:
            print(f"{MODEL} not served (models: {models}); set BRAIN_FLUX2_*", file=sys.stderr)
            return 1
        print(f"text2image {args.width}x{args.height} ({args.variant}):")
        return run_streaming(brain, "text2image", params, args.out)


if __name__ == "__main__":
    sys.exit(main())
