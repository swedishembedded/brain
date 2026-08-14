#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The imaging pipeline over D-Bus: segment -> refine -> restore -> upscale, in ONE call.

The point of the example is the thing that is hard to see from the manifest: a
multi-stage edit is a SINGLE `run`, not four. That matters for more than
convenience --

  * the intermediate mask and the intermediate image never cross the bus. Four
    separate calls would marshal a full-resolution image out and back in three
    times;
  * the composite happens ONCE, at the end, so pixels outside the mask come back
    BIT-IDENTICAL rather than having survived three lossy round trips;
  * the pipeline dispatches its stages through the capability registry, so a
    stage whose model is not configured fails with THAT model's own
    "set BRAIN_..." message instead of a generic one.

The `upscale` stage is a TAIL: it changes the image size, so it must be last and
runs after the composite. Ask for it in the middle and the pipeline rejects the
spec by position rather than silently reordering.

Run it against a private session bus:

    BRAIN_SAM2_WEIGHTS=... BRAIN_CODEFORMER_WEIGHTS=... BRAIN_ESRGAN_WEIGHTS=... \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/imaging/edit_pipeline.py --image photo.ppm --point 614,430'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402

MODEL = "brain/imgpipe"


def build_stages(args) -> list[dict]:
    """The stage list, in the order the pipeline will run it."""
    stages: list[dict] = []
    if args.point or args.box:
        seg: dict = {"op": "segment"}
        if args.point:
            seg["points"] = [[float(v) for v in p.split(",")] for p in args.point]
        if args.box:
            seg["boxes"] = [[float(v) for v in args.box.split(",")]]
        stages.append(seg)
        # Grow then soften: a hard SAM edge cuts through anti-aliased pixels, so
        # a few px of dilate + feather is what makes a composite look like an
        # edit rather than a cut-out.
        if args.dilate:
            stages.append({"op": "dilate", "radius": args.dilate})
        if args.feather:
            stages.append({"op": "feather", "radius": args.feather})
    if args.restore is not None:
        stages.append({"op": "restore", "w": args.restore})
    if args.upscale:
        stages.append({"op": "upscale", "tile": args.tile})
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6)")
    ap.add_argument("--point", action="append", default=[], help="'x,y' foreground click (repeatable)")
    ap.add_argument("--box", default="", help="'x1,y1,x2,y2' box prompt")
    ap.add_argument("--dilate", type=int, default=4, help="grow the mask by N px (0 = off)")
    ap.add_argument("--feather", type=int, default=3, help="soften the mask edge by N px (0 = off)")
    ap.add_argument("--restore", type=float, default=0.7, help="face-restoration fidelity dial")
    ap.add_argument("--upscale", action="store_true", help="add the x4 super-resolution tail")
    ap.add_argument("--tile", type=int, default=0, help="upscale tile size (0 = whole image)")
    ap.add_argument("--out", default="/tmp", help="directory for the result and the mask")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    stages = build_stages(args)
    if not stages:
        print("FATAL: nothing to do — pass at least one of --point/--box/--restore/--upscale", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with BrainDBus() as brain:
        served = brain.models()
        if MODEL not in served:
            print(f"FATAL: {MODEL!r} not served (models: {served})", file=sys.stderr)
            return 2

        print(f"{args.image}: {w}x{h}")
        print("stages: " + " -> ".join(s["op"] for s in stages))
        meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}
        params = {"stages": json.dumps({"stages": stages})}

        t = time.perf_counter()
        r = brain.run(MODEL, "run", params, in_fds={"image": sealed_memfd(img)}, in_meta=meta)
        dt = (time.perf_counter() - t) * 1000

        res = r.result
        ow, oh = int(res.get("w", w)), int(res.get("h", h))
        print(f"  {dt:8.1f} ms   {len(stages)} stage(s) -> {ow}x{oh}")
        # The pipeline reports how many stages actually edited pixels; zero means
        # the input comes back untouched, which is a legitimate answer and worth
        # showing rather than hiding.
        if "edits" in res:
            print(f"  edits: {res['edits']} (0 = mask-only, the input is returned unchanged)")

        image = read_fd(r.fds["image"])
        mask = read_fd(r.fds["mask"])
        save_ppm(out / "pipeline.ppm", image, ow, oh)
        # The mask is the authoritative record of which pixels were allowed to
        # move — it travels at the OUTPUT size, so it still describes the image
        # it came back with even when the upscale tail ran.
        save_ppm(out / "pipeline_mask.ppm", b"".join(mask[i : i + 4] * 3 for i in range(0, len(mask), 4)), ow, oh)
        print(f"  wrote {out / 'pipeline.ppm'} and {out / 'pipeline_mask.ppm'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
