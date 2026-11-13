#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Qwen3-TTS speech synthesis over brain's D-Bus interface, streamed.

`brain/qwen3tts`'s resident `synth`/`clone`/`design` actions are `.streaming()`:
the server vocodes in chunks and sends each one as an out-of-band blob frame
on the `Subscribe` stream while it is still generating, then sends the
complete clip as the terminal blob. This example consumes BOTH:

  * ``on_chunk`` prints each segment as it lands and reports the real
    time-to-first-audio - the number that decides whether playback can start
    before generation finishes;
  * the terminal `audio` blob, a WAV file, is written out as it is.

That is the same capability the private NPU Unix-socket server provides
through its own `audio_chunk` JSON protocol, reached here with no TTS-specific
client code - just `Subscribe` and the manifest.

The mid-run chunks are RAW mono f32 little-endian PCM at 24 kHz (their
metadata says so); the terminal blob is the whole clip as a WAV.

The voice is chosen per call: `synth` is speaker-free, `clone` takes a
reference recording (`--ref`, with `--ref-text` for the in-context path) and
`design` takes `instruct`/`speaker` and needs a CustomVoice/VoiceDesign
checkpoint. The checkpoint is the one the model store resolves.

Run under a private session bus:

    tools/dbus-session.sh --serve "--dbus" -- \\
      python3 samples/python/tts/qwen3tts-speak/qwen3tts_speak.py \\
          --text "Hello from a resident text to speech model." \\
          --max-frames 40 --stream-chunk 4 --out speak.wav

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "brain-py"))
from brain_py.base import BrainError, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

MODEL = "brain/qwen3tts"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="the text to speak")
    ap.add_argument("--action", default="synth", choices=["synth", "clone", "design"], help="synth (speaker-free), clone (a reference voice) or design (instruct/preset voice)")
    ap.add_argument("--ref", default="", help="clone only: reference recording (WAV)")
    ap.add_argument("--ref-text", default="", help="clone only: transcript of the reference (enables the in-context path)")
    ap.add_argument("--instruct", default="", help="design only: natural-language voice/emotion/prosody description")
    ap.add_argument("--speaker", default="", help="design only: CustomVoice preset speaker name")
    ap.add_argument("--lang", default="", help="synthesis language (default: english)")
    ap.add_argument("--stream-chunk", type=int, default=0, help="codec frames per streamed chunk (0 = the engine's own default)")
    ap.add_argument("--max-frames", type=int, default=0, help="codec frame cap (0 = the action's own default)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (omit: the server draws a random one and reports it)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="qwen3tts_speak.wav", help="output WAV path")
    args = ap.parse_args()

    # Only send what the caller actually chose - every other parameter's
    # default lives in the action's own schema, and re-stating it here is how
    # a client and a server drift apart.
    params: dict = {"text": args.text}
    if args.lang:
        params["lang"] = args.lang
    if args.max_frames > 0:
        params["max_frames"] = args.max_frames
    if args.stream_chunk > 0:
        params["stream_chunk"] = args.stream_chunk
    if args.seed is not None:
        params["seed"] = args.seed
    blobs = None
    if args.action == "clone":
        if not args.ref:
            ap.error("--action clone needs --ref")
        blobs = {"ref": Path(args.ref).read_bytes()}
        if args.ref_text:
            params["ref_text"] = args.ref_text
    if args.action == "design":
        params["instruct"] = args.instruct
        params["speaker"] = args.speaker

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); pull a Qwen3-TTS checkpoint into the model store")
        spec = next((a for m in brain.manifests() if m["model"] == args.model for a in m["actions"] if a["name"] == args.action), None)
        if spec is None:
            skip(f"{args.model} does not serve {args.action!r}")
        print(f"{args.action} {args.text!r} (streaming={spec['streaming']}):")

        t0 = time.monotonic()
        streamed = bytearray()
        first_audio: list[float] = []

        def on_chunk(name: str, data: bytes, meta: dict) -> None:
            # Mid-run segments carry an `index`; the terminal complete blob
            # does not, so this only reports the live ones.
            index = (meta.get("meta") or {}).get("index")
            if name != "audio" or index is None:
                return
            if not first_audio:
                first_audio.append(time.monotonic() - t0)
            streamed.extend(data)
            print(f"  chunk {index}: {len(data) // 4} samples at {time.monotonic() - t0:.1f}s", flush=True)

        try:
            outcome = brain.subscribe(args.model, args.action, params, blobs=blobs, on_chunk=on_chunk, timeout=1800.0)
        except BrainError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

        clip = outcome.blobs.get("audio")
        if not clip:
            print("  no audio blob arrived", file=sys.stderr)
            return 1
        rate = int(outcome.outputs.get("sample_rate", 24000))
        n = int(outcome.outputs.get("samples", 0))
        seed = outcome.outputs.get("seed")
        if seed is not None:
            print(f"  seed: {seed} (pass --seed {seed} to reproduce)")
        Path(args.out).write_bytes(clip)
        total = time.monotonic() - t0
        if first_audio:
            print(f"  time to first audio: {first_audio[0]:.1f}s of {total:.1f}s total")
            # The chunks are the same signal as the terminal artifact, so a
            # player that consumed them live has already heard the whole clip.
            same = "matches" if len(streamed) // 4 == n else "DIFFERS FROM"
            print(f"  streamed {len(streamed) // 4} samples - {same} the terminal clip's {n}")
        print(f"wrote {args.out} ({n} samples, {n / rate:.2f}s, {rate} Hz) in {total:.1f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
