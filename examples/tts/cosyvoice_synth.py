#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""CosyVoice zero-shot voice cloning over brain's D-Bus interface.

Subscribes to the streaming `synth` action and writes the returned `audio`
blob straight to disk - like `minimaxmusic3 generate` and `qwen3tts synth`,
this action already packs a COMPLETE WAV byte stream (`meta.format ==
"wav"`), so no client-side encoding step is needed.

The reference clip is sent as the `ref_audio` input blob - its raw file
bytes, WAV header included - rather than decoded PCM: `cosyvoice::caps::
decode_ref_audio` (server-side) parses a WAV container's own sample rate
directly, so sending the file as-is over D-Bus preserves its full native
rate. This is deliberately NOT the same path `brain do cosyvoice synth --in
ref_audio=clip.wav` takes on the CLI, which downsamples any input clip to a
fixed 16 kHz before the action ever sees it - going through D-Bus directly,
as this script does, avoids that cap.

Run under a private session bus (weights via env - six roles: the
speech-token LM, the flow decoder, the HiFT vocoder, S3Tokenizer,
CAM++, and the text BPE tokenizer identity):

    dbus-run-session -- bash -c '
      BRAIN_COSYVOICE_LLM=… BRAIN_COSYVOICE_FLOW=… BRAIN_COSYVOICE_HIFT=… \\
      BRAIN_S3TOKENIZER_V2=… BRAIN_CAMPPLUS_DIR=… BRAIN_COSYVOICE_TOKENIZER=… \\
      brain serve --dbus & sleep 2
      python3 examples/tts/cosyvoice_synth.py \\
          --text "Hello, this is a cloned voice." \\
          --ref-audio reference.wav \\
          --ref-text "the reference clip'"'"'s own transcript" \\
          --out clone.wav'

Only `variant=cosyvoice2` is implemented today - CosyVoice 3's pipeline
(the DiT flow decoder, causal HiFT, and CosyVoice3LM chained together) is a
recorded follow-up; passing `--variant cosyvoice3` reaches the server and
gets a clear, typed error back rather than silently using CosyVoice 2's
weights.

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

MODEL = "brain/cosyvoice"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="the target text to synthesize")
    ap.add_argument("--ref-audio", required=True, metavar="WAV", help="reference clip for zero-shot cloning (any sample rate)")
    ap.add_argument("--ref-text", required=True, help="the reference clip's own transcript")
    ap.add_argument("--variant", default="cosyvoice2", choices=["cosyvoice2", "cosyvoice3"], help="cosyvoice2 (implemented) or cosyvoice3 (not yet)")
    ap.add_argument("--model", default=MODEL, help="a streaming synth model")
    ap.add_argument("--out", default="cosyvoice_synth.wav", help="output WAV path")
    ap.add_argument("--n-timesteps", type=int, default=-1, help="Euler steps the flow decoder's CFM solver takes (-1 = the model's own default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ref_audio_bytes = Path(args.ref_audio).read_bytes()

    # Only send what the caller actually chose - every other parameter's
    # default lives in the action's own schema, and re-stating it here is how
    # a client and a server drift apart.
    params: dict = {"text": args.text, "ref_text": args.ref_text, "variant": args.variant, "seed": args.seed}
    if args.n_timesteps >= 0:
        params["n_timesteps"] = args.n_timesteps

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set the six BRAIN_COSYVOICE_*/BRAIN_S3TOKENIZER_V2/BRAIN_CAMPPLUS_DIR env vars")
        print(f"synth {args.text!r} (variant={args.variant}):")
        t0 = time.monotonic()

        def on_progress(step: int, total: int, message: str) -> None:
            print(f"  [{step}/{total}] {message}", flush=True)

        try:
            # 1800 s: the flow decoder's host-CPU Euler loop alone is
            # measured in minutes in a release build on this stack.
            outcome = brain.subscribe(
                args.model,
                "synth",
                params,
                blobs={"ref_audio": ref_audio_bytes},
                meta={"ref_audio": {"media": "audio"}},
                timeout=1800.0,
                on_progress=on_progress,
            )
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
