#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""VQGAN discrete round trip over brain's D-Bus surface: image -> codes -> image.

`encode` and `decode` are two separate actions on purpose. The whole point of a
discrete latent is that the codes TRAVEL: a 512x512 RGB image is 786 432 bytes,
its 16x16 code grid is 256 indices — 1 KiB, or 320 bytes at 10 bits each. This
example measures that, prints the code histogram, and optionally corrupts a few
codes so you can see what a single index is worth.

    BRAIN_VQGAN_WEIGHTS=/path/to/vqgan_code1024.pth \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/restore/vq_roundtrip.py --image face.ppm'
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) to round-trip")
    ap.add_argument("--size", type=int, default=512, help="square side the graph is built for (multiple of 32)")
    ap.add_argument("--corrupt", type=int, default=0, help="also decode with N codes replaced by code 0")
    ap.add_argument("--out", default="/tmp", help="directory for the reconstructions")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with BrainDBus() as brain:
        if "brain/vqgan" not in brain.models():
            print("FATAL: 'vqgan' not served (set BRAIN_VQGAN_WEIGHTS)", file=sys.stderr)
            return 2

        enc = brain.run(
            "brain/vqgan", "encode", {"size": args.size},
            in_fds={"image": sealed_memfd(img)},
            in_meta={"image": {"media": "image", "w": w, "h": h, "c": 3}},
        )
        codes = read_fd(enc.fds["codes"])
        idx = list(struct.unpack(f"<{len(codes) // 4}I", codes))
        pixels = args.size * args.size * 3
        print(f"{args.image}: {w}x{h} -> {enc.result['lh']}x{enc.result['lw']} codes")
        print(f"  {len(idx)} indices, {len(set(idx))} distinct of {enc.result['codebook_size']}")
        print(f"  quantisation MSE (mean squared distance to the chosen code): {enc.result['quant_mse']:.4f}")
        print(f"  {pixels} B of pixels -> {len(codes)} B of u32 codes ({pixels / len(codes):.0f}x)")
        top = Counter(idx).most_common(5)
        print("  most-used codes: " + ", ".join(f"{c}x{n}" for c, n in top))

        # The codes blob comes straight back in — same bytes, same media type.
        dec = brain.run(
            "brain/vqgan", "decode", {"size": args.size},
            in_fds={"codes": sealed_memfd(codes)},
            in_meta={"codes": {"media": "bytes", "lh": enc.result["lh"], "lw": enc.result["lw"]}},
        )
        blob = read_fd(dec.fds["image"])
        save_ppm(out / "vq_reconstruction.ppm", blob, dec.result["width"], dec.result["height"], 3)
        print(f"  decoded {dec.result['width']}x{dec.result['height']} -> {out}/vq_reconstruction.ppm")

        if args.corrupt > 0:
            broken = list(idx)
            step = max(1, len(broken) // args.corrupt)
            for i in range(0, len(broken), step):
                broken[i] = 0
            raw = struct.pack(f"<{len(broken)}I", *broken)
            dec2 = brain.run(
                "brain/vqgan", "decode", {"size": args.size},
                in_fds={"codes": sealed_memfd(raw)},
                in_meta={"codes": {"media": "bytes"}},
            )
            save_ppm(out / "vq_corrupted.ppm", read_fd(dec2.fds["image"]), dec2.result["width"], dec2.result["height"], 3)
            print(f"  {len(range(0, len(broken), step))} codes zeroed -> {out}/vq_corrupted.ppm")

        print("scheduler:", brain.stats())
        print("  ('builds' counts every model this server built. encode and decode share ONE")
        print("   vqgan instance, because instance_key is the square side and not the action.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
