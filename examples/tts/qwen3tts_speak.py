#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Qwen3-TTS speech synthesis over brain's D-Bus interface, streamed.

`brain/qwen3tts`'s resident `speak`/`design` actions are `.streaming()`: the
server vocodes in chunks and sends each one as an out-of-band blob frame on
the `Subscribe` stream while it is still generating, then sends the complete
waveform as the terminal blob. This example consumes BOTH:

  * ``on_chunk`` prints each segment as it lands and reports the real
    time-to-first-audio - the number that decides whether playback can start
    before generation finishes;
  * the terminal `audio` blob is written to a WAV file.

That is the same capability the private `brain tts serve` Unix socket used to
provide through its own `audio_chunk` JSON protocol, reached here with no
TTS-specific client code - just `Subscribe` and the manifest.

Unlike `cosyvoice synth` (whose blob is already a WAV byte stream), this
action's `audio` blob is RAW mono f32 little-endian PCM at 24 kHz, exactly as
`speak_spec` declares - so the example wraps it in a WAV container itself,
with the stdlib `wave` module and no extra dependency.

The served voice is configured by the ENVIRONMENT, not per call: with
`BRAIN_QWEN3TTS_REF` set to a reference clip, `speak` voice-clones that timbre
(in-context when `BRAIN_QWEN3TTS_REF_TEXT` also gives its transcript);
without it, `speak` is speaker-free synthesis. `design` takes its
`instruct`/`speaker` per call instead, and needs a CustomVoice/VoiceDesign
checkpoint.

Run under a private session bus (weights via env - the imported brain
checkpoints and the HF checkpoint dir for `config.json` + tokenizer):

    dbus-run-session -- bash -c '
      BRAIN_QWEN3TTS_WEIGHTS=out/tts-base06 \\
      BRAIN_QWEN3TTS_CKPT=/path/to/Qwen3-TTS-12Hz-0.6B-Base \\
      brain serve --dbus & sleep 2
      python3 examples/tts/qwen3tts_speak.py \\
          --text "Hello from a resident text to speech model." \\
          --max-frames 40 --out speak.wav'

Requires: jeepney -- `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import wave
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

MODEL = "brain/qwen3tts"


def write_wav(path: str, pcm_f32: bytes, sample_rate: int) -> int:
    """Write raw mono f32-LE PCM as a 16-bit WAV. Returns the frame count.

    `speak`'s blob is raw PCM by design (it is also what the mid-run chunks
    carry, so a client can append chunks and get exactly the terminal blob);
    the container is the client's business."""
    n = len(pcm_f32) // 4
    samples = struct.unpack(f"<{n}f", pcm_f32[: n * 4])
    pcm16 = struct.pack(f"<{n}h", *(max(-32768, min(32767, int(s * 32767))) for s in samples))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="the text to speak")
    ap.add_argument("--action", default="speak", choices=["speak", "design"], help="speak (env-configured voice) or design (instruct/preset voice)")
    ap.add_argument("--instruct", default="", help="design only: natural-language voice/emotion/prosody description")
    ap.add_argument("--speaker", default="", help="design only: CustomVoice preset speaker name")
    ap.add_argument("--lang", default="", help="synthesis language (default: the server's own)")
    ap.add_argument("--max-frames", type=int, default=0, help="codec frame cap (0 = the action's own default)")
    ap.add_argument("--seed", type=int, default=-1, help="RNG seed (-1 = the action's own default)")
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
    if args.seed >= 0:
        params["seed"] = args.seed
    if args.action == "design":
        params["instruct"] = args.instruct
        params["speaker"] = args.speaker

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"{args.model!r} not served (models: {models}); set BRAIN_QWEN3TTS_WEIGHTS and BRAIN_QWEN3TTS_CKPT")
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
            outcome = brain.subscribe(args.model, args.action, params, on_chunk=on_chunk, timeout=1800.0)
        except BrainError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

        pcm = outcome.blobs.get("audio")
        if not pcm:
            print("  no audio blob arrived", file=sys.stderr)
            return 1
        rate = int(outcome.outputs.get("sample_rate", 24000))
        n = write_wav(args.out, pcm, rate)
        total = time.monotonic() - t0
        if first_audio:
            print(f"  time to first audio: {first_audio[0]:.1f}s of {total:.1f}s total")
            # The chunks are the same signal as the terminal artifact, so a
            # player that consumed them live has already heard the whole clip.
            same = "matches" if bytes(streamed) == pcm else "DIFFERS FROM"
            print(f"  streamed {len(streamed) // 4} samples - {same} the terminal blob")
        print(f"wrote {args.out} ({n} samples, {n / rate:.2f}s, {rate} Hz) in {total:.1f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
