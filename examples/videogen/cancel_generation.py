#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cooperative cancellation of a Wan2.1 video generation over D-Bus.

This is the example that matters most for this model. A default `t2v` run is
81 frames at 832x480 over 50 steps -- 57.5 minutes measured on a P40 -- so a
job that cannot be called off is a card nobody else can use for an hour.

It starts a streaming `t2v` job, calls `Cancel(job)` after the second progress
frame, and verifies the stream ends with the `error` frame `"cancelled"`. The
server flips the job's cancel token; `wan::pipeline`'s denoise loop polls it
once per step, so the abort lands at the next step boundary (the forward
already in flight is one submit of the whole block stack and finishes first).

    dbus-run-session -- bash -c '
      BRAIN_WAN_DIT=… BRAIN_WAN_VAE=… BRAIN_WAN_T5=… BRAIN_WAN_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/videogen/cancel_generation.py --frames 9 --width 256 --height 256'

Requires: jeepney -- `pip install -e brain-py`.
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

from generate_video import MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="a lighthouse in a storm", help="prompt (the clip is never produced)")
    ap.add_argument("--model", default=MODEL, help="a streaming, cancellable t2v model")
    ap.add_argument("--frames", type=int, default=0, help="video frames, of the form 1 + 4k (0 = the model's default)")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    args = ap.parse_args()

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER")

        params: dict = {"prompt": args.prompt}
        for name, value in [("frames", args.frames), ("width", args.width), ("height", args.height)]:
            if value:
                params[name] = value

        # Deliberately the LOW-LEVEL frame iterator (not the high-level
        # subscribe()): this example needs the job id mid-stream to cancel it,
        # which the materialised-Outcome API has no way to expose.
        job, frames = brain.stream_frames_with_job(args.model, "t2v", params, timeout=7200.0)
        print(f"job {job} started; cancelling after the second progress frame")

        progress_seen = 0
        cancelled = False
        for frame, _fds in frames:
            kind = frame.get("type")
            if kind == "progress":
                progress_seen += 1
                print(f"  [{frame['step']}/{frame['total']}] {frame.get('message', '')}")
                if progress_seen == 2 and not cancelled:
                    found = brain.cancel(job)
                    cancelled = True
                    print(f"  Cancel({job}) -> {found}")
                    if not found:
                        print("  ERROR: job not found in flight", file=sys.stderr)
                        return 1
            elif kind == "error":
                msg = frame.get("message", "")
                ok = cancelled and msg == "cancelled"
                print(f"  error frame: '{msg}' ({'expected' if ok else 'UNEXPECTED'})")
                return 0 if ok else 1
            elif kind == "done":
                print("  ERROR: job completed despite cancel", file=sys.stderr)
                return 1
        print("  ERROR: stream ended without a terminal frame", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
