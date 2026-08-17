#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Wan2.1 text-to-video over brain's D-Bus interface.

Subscribes to the streaming `t2v` action: progress frames arrive per denoise
step (each one carries the measured seconds-per-step and an ETA, because a
step at 480p is minutes), then the clip arrives as a single `blob` frame whose
payload is an out-of-band memfd holding N interleaved-HWC f32 RGB frames
(`capability::blob::video_blob`'s wire format, meta `{frames,w,h,c,fps}`).

The frames are written as numbered binary PPMs and, when `ffmpeg` is on PATH,
muxed into the requested container. No ffmpeg means the PPMs plus the command
line that finishes the job -- an hour of GPU time is never thrown away for
want of an encoder.

Run under a private session bus (weights via env -- see the README):

    dbus-run-session -- bash -c '
      BRAIN_WAN_DIT=… BRAIN_WAN_VAE=… BRAIN_WAN_T5=… BRAIN_WAN_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/videogen/generate_video.py --prompt "a cat on a beach" \
          --frames 9 --width 256 --height 256 --steps 4 --out cat.mp4'

Start small. The defaults are upstream's (81 frames at 832x480, 50 steps),
which is ~1 hour on a P40; the invocation above is the smoke-test size.

Requires: jeepney -- `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import save_ppm  # noqa: E402

#: The default `--model` every videogen example uses (they share this constant
#: rather than each hardcoding it).
MODEL = "brain/wan"


def save_clip(data: bytes, meta: dict, out: str) -> int:
    """Write the clip blob's frames as numbered PPMs beside `out`, then mux
    them with ffmpeg if it is available. Returns an exit code."""
    frames = int(meta.get("frames", 0))
    w, h, c = int(meta.get("w", 0)), int(meta.get("h", 0)), int(meta.get("c", 3))
    fps = float(meta.get("fps", 16))
    if frames <= 0 or w <= 0 or h <= 0:
        print(f"  clip blob has no usable geometry: {meta}", file=sys.stderr)
        return 1
    per_frame = w * h * c * 4
    if len(data) != frames * per_frame:
        print(f"  clip blob is {len(data)} bytes, expected {frames}x{w}x{h}x{c}x4", file=sys.stderr)
        return 1

    frame_dir = Path(out).with_suffix("").parent / (Path(out).stem + ".frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        save_ppm(frame_dir / f"frame_{i + 1:05d}.ppm", data[i * per_frame : (i + 1) * per_frame], w, h, c)

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%05d.ppm"), "-pix_fmt", "yuv420p", out]
    if shutil.which("ffmpeg") is None:
        # Same policy as `imaging::video::encode_frames`: the frames plus the
        # command that finishes the job, never an error.
        print(f"  ffmpeg is not on PATH; {frames} frames are in {frame_dir}")
        print("  finish the job with:\n    " + " ".join(cmd))
        return 0
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed: {r.stderr.decode(errors='replace')[-400:]}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({w}x{h}, {frames} frames at {fps:g} fps)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True, help="text description of the desired clip")
    ap.add_argument("--model", default=MODEL, help="a streaming t2v model")
    ap.add_argument("--out", default="wan.mp4", help="output container path")
    ap.add_argument("--frames", type=int, default=0, help="video frames, of the form 1 + 4k (0 = the model's default, 81)")
    ap.add_argument("--width", type=int, default=0, help="output width (0 = the model's default, 832)")
    ap.add_argument("--height", type=int, default=0, help="output height (0 = the model's default, 480)")
    ap.add_argument("--steps", type=int, default=0, help="denoise steps (0 = the model's default, 50)")
    ap.add_argument("--guidance", type=float, default=-1.0, help="CFG scale; <= 1.0 halves the cost (-1 = the model's default, 5.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="", choices=["", "unipc", "dpm++"], help="multistep solver (blank = the model's default)")
    args = ap.parse_args()

    # Only send what the caller actually chose: every other parameter's default
    # lives in the action's own schema (upstream's generate.py values), and
    # re-stating it here is how a client and a server drift apart.
    params: dict = {"prompt": args.prompt, "seed": args.seed}
    for name, value, unset in [
        ("frames", args.frames, 0),
        ("width", args.width, 0),
        ("height", args.height, 0),
        ("steps", args.steps, 0),
        ("guidance", args.guidance, -1.0),
        ("solver", args.solver, ""),
    ]:
        if value != unset:
            params[name] = value

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER")
        print(f"t2v {args.prompt!r}:")
        t0 = time.monotonic()

        def on_progress(step: int, total: int, message: str) -> None:
            print(f"  [{step}/{total}] {message}", flush=True)

        try:
            # 7200 s: the measured 81-frame 480p run is 57.5 minutes, and a
            # client timeout shorter than the model is a self-inflicted failure.
            outcome = brain.subscribe(args.model, "t2v", params, timeout=7200.0, on_progress=on_progress)
        except BrainError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        print(f"  done: {outcome.outputs} ({time.monotonic() - t0:.1f}s)")
        data = outcome.blobs.get("video")
        if data is None:
            print("  no video blob arrived", file=sys.stderr)
            return 1
        return save_clip(data, (outcome.meta.get("video") or {}).get("meta") or {}, args.out)


if __name__ == "__main__":
    sys.exit(main())
