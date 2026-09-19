#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""LFM2.5-Encoder long-context embeddings and fill-mask over brain's D-Bus
interface (`brain do lfm2 embed` / `brain do lfm2 fill_mask`).

`embed` returns per-token hidden states plus a mean-pooled sequence
embedding; `fill_mask` returns top-k predictions at every `<|mask|>`
position. Both run the chunked long-context path, so an 8k-token document
works the same as a short sentence - the model is rebuilt at the EXACT
request length because bidirectional attention makes unmasked padding
unsound (see `crates/lfm2/src/caps.rs`), so this sample does not batch
client-side the way `t5_embed.py --concurrent` does.

Run under a private session bus (weights via env):

    BRAIN_LFM2=<ckpt>.safetensors BRAIN_LFM2_TOKENIZER=<tokenizer>.json \\
      tools/dbus-session.sh --serve "--dbus" -- \\
      python3 samples/python/embedding/lfm2-embed/lfm2_embed.py \\
        --text "the quick brown fox jumps over the lazy dog"

    python3 samples/python/embedding/lfm2-embed/lfm2_embed.py \\
        --action fill_mask --text "the quick brown <|mask|> jumps over the lazy dog"

Requires: jeepney (the same dependency as samples/python/dbus) - `pip install -e brain-py`.

Swedish Embedded AB implements solutions for driving served on-device models
over D-Bus like this one. If your team needs expertise in embedding or
encoder-model serving, you can procure our services by sending an email to
info@swedishembedded.com.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

MODEL = "brain/lfm2"


def embed(brain: BrainDBus, text: str, max_tokens: int) -> None:
    t0 = time.monotonic()
    out = brain.run(MODEL, "embed", {"text": text, "max_tokens": max_tokens})
    dt = time.monotonic() - t0

    n, d = int(out.outputs["tokens"]), int(out.outputs["dim"])
    mean = out.outputs["mean"]
    raw = out.blobs["embeddings"]
    (v0,) = struct.unpack_from("<f", raw, 0)
    print(f"embed {text!r:<48} [{n}, {d}] {dt * 1000:7.1f} ms  hidden[0,0]={v0:+.4f}  mean[0]={mean[0]:+.4f}")


def fill_mask(brain: BrainDBus, text: str, topk: int) -> None:
    t0 = time.monotonic()
    out = brain.run(MODEL, "fill_mask", {"text": text, "topk": topk})
    dt = time.monotonic() - t0

    predictions = out.outputs["predictions"]
    print(f"fill_mask {text!r} ({dt * 1000:.1f} ms, {len(predictions)} mask position(s)):")
    for p in predictions:
        top = ", ".join(f"{t['token']!r}({t['logit']:+.2f})" for t in p["tokens"])
        print(f"  row {p['row']}: {top}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", default="embed", choices=["embed", "fill_mask"])
    ap.add_argument("--text", default="", help="input text (default depends on --action)")
    ap.add_argument("--max-tokens", type=int, default=0, dest="max_tokens", help="embed: truncate to N tokens (0 = no limit)")
    ap.add_argument("--topk", type=int, default=5, help="fill_mask: predictions per mask position")
    args = ap.parse_args()

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER)")

        if args.action == "embed":
            text = args.text or "the quick brown fox jumps over the lazy dog"
            embed(brain, text, args.max_tokens)
        else:
            text = args.text or "the quick brown <|mask|> jumps over the lazy dog"
            fill_mask(brain, text, args.topk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
