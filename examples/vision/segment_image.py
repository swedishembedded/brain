#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""SAM 2.1 promptable segmentation over brain's D-Bus surface.

Sends one image and several prompts, and saves each mask plus a cut-out of the
segmented region. The point of the example is what the timings show:

  * the FIRST prompt pays the Hiera trunk (the whole image encoder);
  * every following prompt on the SAME image is answered by the two-way mask
    decoder alone, because the resident instance caches the encoding
    (`sam2::caps::Session`);
  * with `--concurrent`, N prompts are submitted at once and the residency
    Executor groups them into ONE batch, which `resident_sam2` answers with one
    trunk pass and N decoder passes.

Run it against a private session bus:

    BRAIN_SAM2_WEIGHTS=$BRAIN_TESTDATA/sam2/hiera-tiny/sam2.1_hiera_tiny.pt \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/vision/segment_image.py --image photo.ppm --point 614,430'
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402


def segment(brain: BrainDBus, img: bytes, w: int, h: int, params: dict) -> tuple[dict, bytes]:
    """One `segment` call. The image goes in as a sealed memfd, the mask comes
    back as one — no bulk data is marshalled through D-Bus itself."""
    meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}
    r = brain.run("sam2", "segment", params, in_fds={"image": sealed_memfd(img)}, in_meta=meta)
    return r.result, read_fd(r.fds["mask"])


def cutout(img: bytes, mask: bytes, w: int, h: int) -> bytes:
    """Keep the image where the mask probability is > 0.5, grey elsewhere.

    `prob > 0.5` is exactly `logit > 0` — the action returns `sigmoid(logits)`
    so a client can threshold with no knowledge of the model's scale.
    """
    import array

    px = array.array("f")
    px.frombytes(img)
    m = array.array("f")
    m.frombytes(mask)
    for i in range(w * h):
        if m[i] <= 0.5:
            px[i * 3] = px[i * 3 + 1] = px[i * 3 + 2] = 0.25
    return px.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) to segment")
    ap.add_argument("--point", action="append", default=[], help="'x,y' foreground click (repeatable)")
    ap.add_argument("--box", default="", help="'x1,y1,x2,y2' box prompt")
    ap.add_argument("--out", default="/tmp", help="directory for the masks and cut-outs")
    ap.add_argument("--concurrent", type=int, default=0, help="also submit N prompts at once (batching demo)")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts: list[dict] = [{"points": p} for p in args.point]
    if args.box:
        prompts.append({"box": args.box})
    if not prompts:
        # No prompt given: click the centre, which is what a UI's first tap is.
        prompts = [{"points": f"{w // 2},{h // 2}"}]

    with BrainDBus() as brain:
        if "sam2" not in brain.models():
            print("FATAL: 'sam2' not served (set BRAIN_SAM2_WEIGHTS)", file=sys.stderr)
            return 2

        print(f"{args.image}: {w}x{h}, {len(prompts)} prompt(s)")
        for i, p in enumerate(prompts):
            t = time.perf_counter()
            res, mask = segment(brain, img, w, h, p)
            dt = (time.perf_counter() - t) * 1000
            note = "  <- trunk + decoder (first prompt on this image)" if i == 0 else "  <- decoder only (encoding cached)"
            print(f"  {str(p):<34} iou {res['iou']:.4f}  area {res['area']:>8}  {dt:8.1f} ms{note}")
            save_ppm(out / f"mask{i}.ppm", mask, w, h, 1)
            save_ppm(out / f"cutout{i}.ppm", cutout(img, mask, w, h), w, h, 3)

        if args.concurrent > 0:
            n = args.concurrent
            grid = [{"points": f"{(k % 4 + 1) * w // 5},{(k // 4 % 4 + 1) * h // 5}"} for k in range(n)]
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                # One connection per thread: BrainDBus is a blocking client over
                # one socket, so concurrency comes from concurrent connections —
                # which is also what makes the server see N jobs at once.
                def one(p: dict) -> float:
                    with BrainDBus() as c:
                        s = time.perf_counter()
                        segment(c, img, w, h, p)
                        return (time.perf_counter() - s) * 1000

                times = list(pool.map(one, grid))
            wall = (time.perf_counter() - t) * 1000
            print(f"\n{n} concurrent prompts: wall {wall:.1f} ms, per-request {min(times):.1f}–{max(times):.1f} ms")
            print("scheduler:", brain.stats(), "  <- max_batch > 1 means the Executor grouped them")

    print(f"\nwrote masks and cut-outs to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
