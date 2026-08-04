#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cooperative cancellation of a FLUX.2 Klein generation over D-Bus.

Starts a streaming `text2image` job, calls `Cancel(job)` after the second
progress frame, and verifies the stream ends with the `error` frame
`"cancelled"` — the server flips the job's cancel token and the pipeline
aborts at its next per-step poll (the current denoise step finishes first).

    dbus-run-session -- bash -c '
      BRAIN_FLUX2_DIT=… BRAIN_FLUX2_VAE=… BRAIN_FLUX2_TE=… BRAIN_FLUX2_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/imagegen/cancel_generation.py'

Requires: jeepney — `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402

from generate import MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="a lighthouse in a storm", help="prompt (the image is never produced)")
    ap.add_argument("--variant", default="klein-4b", choices=["klein-4b", "klein-9b", "base-4b", "base-9b"])
    args = ap.parse_args()

    with BrainDBus() as brain:
        models = brain.models()
        if MODEL not in models:
            print(f"{MODEL} not served (models: {models}); set BRAIN_FLUX2_*", file=sys.stderr)
            return 1

        # Deliberately the LOW-LEVEL frame iterator (not the high-level
        # subscribe()): this example needs the job id mid-stream to cancel it,
        # which the materialised-Outcome API has no way to expose.
        job, frames = brain.stream_frames_with_job(
            MODEL,
            "text2image",
            {"prompt": args.prompt, "variant": args.variant},
            timeout=7200.0,
        )
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
