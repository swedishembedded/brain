#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""SUPIR photo-realistic blind image restoration over brain's D-Bus surface.

`brain/supir`'s one action, `restore`, takes a degraded (low-quality) image and
returns a photo-realistic reconstruction driven by a frozen SDXL 1.0 base UNet,
a 1.24B `GLVControl` trunk and 12 `ZeroSFT`/`ZeroCrossAttn` adaptors. Unlike
CodeFormer's `restore_face` (`restore_face.py`, in this same directory), SUPIR
runs a full 50-step (by default) diffusion sample per call - so this is a
multi-second-to-minutes request, not a sub-second one, and the output size is
whatever SUPIR's own resize/snap rule picks (short side >= 1024, both axes
snapped to a 64px multiple) - read back from the result rather than assumed.

    BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \\
    BRAIN_SUPIR_DIR=/path/to/SUPIR-v0Q_fp32.safetensors \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/restore/supir_restore.py --image degraded.ppm'

Set `BRAIN_LLAVA_WEIGHTS` too (and leave `--caption` unset) to auto-caption the
degraded image through LLaVA before restoring it - SUPIR's own `--no_llava`
default (an explicit `--caption ""` also stays silent, matching upstream).

SUPIR's weights carry a non-commercial licence (SUPIR Software License
Agreement, © 2024 SupPixel Pty Ltd): commercial use, including SaaS
deployment and using the output as training data for another model, needs
written permission from the licensor - read that licence before using output
commercially.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) of the degraded image")
    ap.add_argument("--caption", default="", help="image caption; empty auto-captions via brain/llava when served, else stays empty")
    ap.add_argument("--steps", type=int, default=50, help="edm_steps")
    ap.add_argument("--cfg-scale", type=float, default=4.0, help="s_cfg")
    ap.add_argument("--control-scale", type=float, default=1.0, help="s_stage2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/restored.ppm", help="path for the restored image")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}

    with BrainDBus() as brain:
        if "brain/supir" not in brain.models():
            print("FATAL: 'restore' not served (set BRAIN_SDXL_DIR and BRAIN_SUPIR_DIR)", file=sys.stderr)
            return 2

        print(f"{args.image}: {w}x{h} degraded -> restoring ({args.steps} steps)...")
        t = time.perf_counter()
        r = brain.run(
            "brain/supir", "restore",
            {"caption": args.caption, "steps": args.steps, "cfg_scale": args.cfg_scale, "control_scale": args.control_scale, "seed": args.seed},
            in_fds={"image": sealed_memfd(img)}, in_meta=meta,
        )
        dt = time.perf_counter() - t
        blob = read_fd(r.fds["image"])
        ow, oh = r.result["width"], r.result["height"]
        save_ppm(args.out, blob, ow, oh, 3)
        print(f"  {ow}x{oh} restored in {dt:.1f}s -> {args.out}")
        print("scheduler:", brain.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
