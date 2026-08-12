#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""DeepSeek-OCR over brain's D-Bus surface: a page in, decoded text out.

The point of the example is the STREAM. `generate` is a streaming action, so
`Subscribe` delivers each decoded token as a `delta` frame while the run is
still going - which matters here more than for any other model in the tree,
because this decoder has no KV cache: every token is a full recompute of the
whole ~280-token sequence through 12 MoE layers, tens of seconds apiece on the
CPU backend. Waiting for the final `Outcome` means waiting for all of them.

The final outcome also carries real token accounting - `prompt_tokens`,
`completion_tokens`, `finish_reason` - which is what the OpenAI and Anthropic
surfaces report for the same request.

Run it against a private session bus:

    BRAIN_DEEPSEEK_OCR_DIR=<dir with both DeepSeek-OCR GGUFs> \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 5
        python3 examples/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8'

The first call pays the model's activation (the mmproj import, the decoder's
one-off fp32 expansion, and ~22 GiB of weights); every later call on the same
server reuses the resident instance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-OCR"
DEFAULT_PROMPT = "<|grounding|>Convert the document to markdown."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) of the page to read")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="the instruction after the image")
    ap.add_argument("--max-new", type=int, default=8, help="tokens to generate (EVERY one is a full recompute)")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    print(f"page {w}x{h}, prompt {args.prompt!r}, max_new {args.max_new}", flush=True)

    brain = BrainDBus()
    t0 = time.time()
    last = [t0]

    def on_delta(text: str) -> None:
        now = time.time()
        # Per-token wall time, printed with the fragment: this is the number
        # that makes the O(T^2)-recompute cost visible instead of theoretical.
        print(f"[{now - t0:6.1f}s  +{now - last[0]:5.1f}s] {text!r}", flush=True)
        last[0] = now

    out = brain.subscribe(
        MODEL,
        "generate",
        {"prompt": args.prompt, "max_new": args.max_new},
        blobs={"image": img},
        meta={"image": {"media": "image", "meta": {"w": w, "h": h, "c": 3}}},
        on_delta=on_delta,
    )

    print(f"\n--- decoded in {time.time() - t0:.1f}s ---")
    print(out.blobs.get("text", b"").decode("utf-8", "replace"))
    print(
        f"prompt_tokens={out.outputs.get('prompt_tokens')} "
        f"completion_tokens={out.outputs.get('completion_tokens')} "
        f"finish_reason={out.outputs.get('finish_reason')!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
