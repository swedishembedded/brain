#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""MiniMax Music 3 lyrics+caption-conditioned music generation over brain's
D-Bus interface.

Subscribes to the streaming `generate` action and writes the returned
`audio` blob straight to disk - unlike the video/image blob conventions,
this action already packs a COMPLETE WAV byte stream (`meta.format ==
"wav"`, the same convention `qwen3tts synth` uses), so no client-side
encoding step is needed.

Run under a private session bus (weights via env - six directories, one
per component: the Global LLM, the RVQ depth decoder, the condition
encoder, the flow-matching DiT, the vocoder, and the tokenizer):

    dbus-run-session -- bash -c '
      BRAIN_MINIMAXMUSIC3_LM=… BRAIN_MINIMAXMUSIC3_DEPTH=… \\
      BRAIN_MINIMAXMUSIC3_CONDITION=… BRAIN_MINIMAXMUSIC3_DIT=… \\
      BRAIN_MINIMAXMUSIC3_VOCODER=… BRAIN_MINIMAXMUSIC3_TOKENIZER=… \\
      BRAIN_DEVICE=cpu brain serve --dbus & sleep 2
      python3 examples/musicgen/generate_song.py \\
          --caption "warm acoustic ballad, gentle piano, soft vocals, 80 BPM" \\
          --lyrics "$(printf '[verse]\\nquiet morning light\\n[chorus]\\nhold on to this feeling\\n')" \\
          --out song.wav'

`BRAIN_DEVICE=cpu` is required on a machine whose GPU cannot hold the
Global LLM's ~3.28 GB embedding/`lm_head` tensors as single buffers (an
Intel integrated GPU, for instance) - see this repo's own roadmap ledger
for the measured diagnosis. Whole-checkpoint residency does not fit on
the machine this example was written on at all; running this for real
needs more RAM than that machine has.

This example's own default (10 s) is deliberately short - a full song can
run several minutes to generate. Pass `--duration 240` for something
closer to a real song.

Requires: jeepney -- `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

MODEL = "brain/minimaxmusic3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caption", required=True, help="structured music description: genre, BPM, vocal timbre, instrumentation")
    ap.add_argument("--lyrics", required=True, help="song lyrics, with [verse]/[chorus]/etc structural tags")
    ap.add_argument("--model", default=MODEL, help="a streaming generate model")
    ap.add_argument("--out", default="song.wav", help="output WAV path")
    ap.add_argument("--duration", type=float, default=-1.0, help="target song length in seconds (-1 = the model's own default, 10s)")
    ap.add_argument("--steps", type=int, default=-1, help="Euler steps per denoise chunk (-1 = the model's own default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Only send what the caller actually chose - every other parameter's
    # default lives in the action's own schema, and re-stating it here is
    # how a client and a server drift apart.
    params: dict = {"lyrics": args.lyrics, "caption": args.caption, "seed": args.seed}
    if args.duration >= 0:
        params["duration_seconds"] = args.duration
    if args.steps >= 0:
        params["num_inference_steps"] = args.steps

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set the six BRAIN_MINIMAXMUSIC3_* env vars")
        print(f"generate {args.caption!r}:")
        t0 = time.monotonic()

        def on_progress(step: int, total: int, message: str) -> None:
            print(f"  [{step}/{total}] {message}", flush=True)

        try:
            # 7200 s: an 8B-parameter AR loop plus a 36-layer DiT over
            # several chunks is measured in tens of minutes even where it
            # fits in RAM at all - see this example's own module doc.
            outcome = brain.subscribe(args.model, "generate", params, timeout=7200.0, on_progress=on_progress)
        except BrainError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        print(f"  done: {outcome.outputs} ({time.monotonic() - t0:.1f}s)")
        wav = outcome.blobs.get("audio")
        if wav is None:
            print("  no audio blob arrived", file=sys.stderr)
            return 1
        Path(args.out).write_bytes(wav)
        print(f"wrote {args.out} ({len(wav)} bytes)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
