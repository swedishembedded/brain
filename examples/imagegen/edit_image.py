#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""FLUX.2 Klein image editing over brain's D-Bus interface.

Loads a binary PPM (P6), sends it as a **memfd input blob** (HWC f32 `[0,1]`,
the standard brain image wire format) together with an edit prompt, and saves
the regenerated image. The server center-crops references to multiples of 16.

    dbus-run-session -- bash -c '
      BRAIN_FLUX2_DIT=… BRAIN_FLUX2_VAE=… BRAIN_FLUX2_TE=… BRAIN_FLUX2_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/imagegen/edit_image.py --image in.ppm \
          --prompt "the same scene at night" --out edited.ppm'

Requires: jeepney — `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

from generate import MODEL, run_streaming  # noqa: E402  (shared frame loop)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="input PPM (P6) to edit")
    ap.add_argument("--prompt", required=True, help="what to change / generate")
    ap.add_argument("--model", default=MODEL, help="a streaming `edit`-capable model")
    ap.add_argument("--out", default="edited.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=512, help="output width (multiple of 16)")
    ap.add_argument("--height", type=int, default=512, help="output height (multiple of 16)")
    ap.add_argument("--steps", type=int, default=0, help="denoise steps (0 = variant default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="klein-4b", choices=["klein-4b", "klein-9b", "base-4b", "base-9b"])
    ap.add_argument("--ref", action="append", default=[], help="additional reference PPM(s) (up to 3)")
    args = ap.parse_args()

    params = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "variant": args.variant,
    }
    blobs, meta = {}, {}
    for name, path in [("image", args.image)] + [(f"image{i}", p) for i, p in enumerate(args.ref)]:
        data, w, h = load_ppm(path)
        blobs[name] = data
        meta[name] = {"media": "image", "w": w, "h": h, "c": 3}
        print(f"  ref {name}: {path} ({w}x{h})")

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set BRAIN_FLUX2_*")
        print(f"edit -> {args.width}x{args.height} ({args.variant}):")
        return run_streaming(brain, args.model, "edit", params, args.out, blobs=blobs, meta=meta)


if __name__ == "__main__":
    sys.exit(main())
