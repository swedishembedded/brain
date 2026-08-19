#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""CLIP text and image embeddings over brain's D-Bus interface.

`embed_text` batches every string given into ONE forward over the resident
text tower (`clip::caps::Session::embed_text_batch`) - the point of
`--text` being repeatable is to show that batching, not just to embed
several strings for convenience. `embed_image` runs the EVA-CLIP-L/336
vision tower on one image at a time.

Run under a private session bus (weights via env):

    BRAIN_CLIP_DIR=<ckpt-root> dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/embedding/clip_embed.py --text "a photo of a cat" \\
        --text "a photo of a dog" --image photo.ppm'

`<ckpt-root>` holds `text_encoder/` and/or `text_encoder_2/` (SDXL layout),
`tokenizer/` and/or `tokenizer_2/`, and `EVA02_CLIP_L_336_psz14_s6B.pt` at
the root for `--image`.

Requires: jeepney (the same dependency as examples/dbus) - `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

MODEL = "brain/clip"


def embed_text(brain: BrainDBus, texts: list[str], tower: str) -> None:
    """One batched call over every string in `texts` - `clip::caps::Session`
    builds the tower at `b = len(texts)` and runs a single forward, not a loop."""
    # `embed_text` takes one string per D-Bus call - the batching win is the
    # *server's* (N concurrent calls land in one forward at `b = N`), so N
    # strings here means N concurrent client calls, not a client-side batch param.
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.monotonic()

    def one(text: str) -> tuple[str, int, bytes]:
        with BrainDBus() as c:
            out = c.run(MODEL, "embed_text", {"text": text, "tower": tower})
            return text, int(out.outputs["dim"]), out.blobs["embedding"]

    with ThreadPoolExecutor(max_workers=max(len(texts), 1)) as pool:
        results = list(pool.map(one, texts))
    dt = time.monotonic() - t0

    print(f"embed_text (tower={tower}, {len(texts)} concurrent, {dt * 1000:.1f} ms wall):")
    for text, dim, raw in results:
        (v0,) = struct.unpack_from("<f", raw, 0)
        print(f"  {text!r:<40} dim={dim} v[0]={v0:+.4f}")
    if len(texts) > 1:
        print("  scheduler:", brain.stats(), " <- max_batch > 1 means the Executor grouped them")


def embed_image(brain: BrainDBus, path: str) -> None:
    img, w, h = load_ppm(path)
    t0 = time.monotonic()
    out = brain.run(
        MODEL, "embed_image", {}, blobs={"image": img}, meta={"image": {"media": "image", "w": w, "h": h, "c": 3}}
    )
    dt = time.monotonic() - t0
    dim = int(out.outputs["dim"])
    raw = out.blobs["embedding"]
    (v0,) = struct.unpack_from("<f", raw, 0)
    print(f"embed_image ({path}, {w}x{h}, {dt * 1000:.1f} ms): dim={dim} v[0]={v0:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="append", default=[], help="a string to embed (repeatable)")
    ap.add_argument("--tower", default="clip_l", choices=["clip_l", "openclip_bigg"])
    ap.add_argument("--image", default="", help="binary PPM (P6) to embed with the EVA-CLIP-L/336 tower")
    args = ap.parse_args()

    if not args.text and not args.image:
        args.text = ["a photo of a cat", "a photo of a dog", "a diagram of a neural network"]

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_CLIP_DIR)")

        if args.text:
            embed_text(brain, args.text, args.tower)
        if args.image:
            embed_image(brain, args.image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
