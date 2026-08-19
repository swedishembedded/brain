#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""T5-XXL / umT5-XXL text encoding over brain's D-Bus interface.

`encode` returns the last hidden state a diffusion DiT conditions on: FLUX.1/2
consume it unmasked (`variant=flux_xxl`), Wan2.1/2.2 condition on the masked,
zero-padded version (`variant=wan_umt5`) - see `t5encoder::caps` for why the
served action always returns the masked-aware tensor even for the unmasked
variant (it degrades to the same thing).

Concurrent calls at the SAME `(variant, max_len)` batch into one forward on
the resident encoder (`crates/cli/src/resident_t5encoder.rs` groups
`run_batch` invocations that way) - `--concurrent` demonstrates it the same
way `examples/vision/segment_image.py` demonstrates SAM 2's batching.

Run under a private session bus (weights via env):

    BRAIN_T5ENCODER_DIR=<ckpt-root> dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/embedding/t5_embed.py \\
        --text "a red cube on a wooden table" --variant flux_xxl --concurrent 4'

`<ckpt-root>` holds `text_encoder_2/` + `tokenizer_2/tokenizer.json` (the
FLUX.1-*/ release layout, unmodified) for `flux_xxl`, and/or
`wan/models_t5_umt5-xxl-enc-bf16.pth` + `wan/tokenizer.json` for `wan_umt5`.

Requires: jeepney (the same dependency as examples/dbus) - `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

MODEL = "brain/t5encoder"


def encode_once(brain: BrainDBus, text: str, variant: str, max_len: int) -> tuple[int, bytes]:
    out = brain.run(MODEL, "encode", {"text": text, "variant": variant, "max_len": max_len})
    raw = out.blobs["hidden_states"]
    d = len(raw) // 4 // max_len
    return d, raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="append", default=[], help="a string to encode (repeatable)")
    ap.add_argument("--variant", default="flux_xxl", choices=["flux_xxl", "wan_umt5"])
    ap.add_argument("--max-len", type=int, default=128, dest="max_len")
    ap.add_argument("--concurrent", type=int, default=0, help="also submit N identical requests at once")
    args = ap.parse_args()

    texts = args.text or ["a red cube on a wooden table"]

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_T5ENCODER_DIR)")

        for text in texts:
            t0 = time.monotonic()
            d, raw = encode_once(brain, text, args.variant, args.max_len)
            dt = time.monotonic() - t0
            (v0,) = struct.unpack_from("<f", raw, 0)
            print(f"[{args.variant}] {text!r:<48} [{args.max_len}, {d}] {dt * 1000:7.1f} ms  hidden[0,0]={v0:+.4f}")

        if args.concurrent > 0:
            n = args.concurrent

            def one(_: int) -> float:
                with BrainDBus() as c:
                    s = time.monotonic()
                    encode_once(c, texts[0], args.variant, args.max_len)
                    return (time.monotonic() - s) * 1000

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=n) as pool:
                times = list(pool.map(one, range(n)))
            wall = (time.monotonic() - t0) * 1000
            print(f"\n{n} concurrent requests at ({args.variant}, {args.max_len}): "
                  f"wall {wall:.1f} ms, per-request {min(times):.1f}-{max(times):.1f} ms")
            print("scheduler:", brain.stats(), " <- max_batch > 1 means the Executor grouped them")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
