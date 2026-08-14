#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Live microphone → brain (over D-Bus) → streaming transcription.

Opens the microphone (or a `--wav` file), streams raw 16 kHz mono f32 PCM to a
running `brain serve --dbus` over the `StreamTranscribe` D-Bus method through one
pipe fd, and prints the transcription **segments** as each ~1 s window decodes.

    # 1. serve the ASR model(s) on a private session bus
    BRAIN_NEMOTRONASR=$BRAIN_TESTDATA/asr/nemotron/hf \
      dbus-run-session -- bash -c '
        brain serve --dbus --device cpu & sleep 2
        python3 examples/asr/transcribe_mic.py --model brain/nemotron --seconds 15
      '

    # or transcribe a wav file (no mic needed — good for a smoke test):
    python3 examples/asr/transcribe_mic.py --wav $BRAIN_TESTDATA/asr/audio/librispeech_mr_quilter.wav

No mic? Generate a test clip with brain's own TTS and feed it with --wav:
    brain qwen3tts synth --text "the quick brown fox" --out /tmp/test.wav   # 24 kHz
    # (resample to 16 kHz mono first, e.g. `sox /tmp/test.wav -r 16000 -c 1 /tmp/16k.wav`)

Dependencies: `jeepney` (D-Bus, always), and `sounddevice`+`numpy` only for live
mic capture (not needed for --wav).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import wave
from pathlib import Path

# Reusable D-Bus client from the brain-py package (run straight from the repo).
try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

SAMPLE_RATE = 16000


def read_wav_f32(path: str) -> bytes:
    """Read a 16 kHz mono WAV → raw f32-LE PCM bytes. Rejects other rates/channels."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise SystemExit(f"{path}: need 16 kHz mono (got {w.getframerate()} Hz, {w.getnchannels()} ch); resample first")
        raw = w.readframes(w.getnframes())
        width = w.getsampwidth()
    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM (got {width * 8}-bit)")
    import array

    ints = array.array("h")
    ints.frombytes(raw)
    scale = 1.0 / 32768.0
    floats = array.array("f", (s * scale for s in ints))
    return floats.tobytes()


def stream_wav(write_fd: int, path: str, realtime: bool) -> None:
    """Feed a wav file to the pipe, optionally pacing it at real time."""
    data = read_wav_f32(path)
    chunk = SAMPLE_RATE * 4 // 10  # 100 ms of f32
    for off in range(0, len(data), chunk):
        os.write(write_fd, data[off : off + chunk])
        if realtime:
            time.sleep(0.1)
    os.close(write_fd)  # EOF → server flushes + emits `done`


def stream_mic(write_fd: int, stop: threading.Event, seconds: float) -> None:
    """Capture the mic and feed raw f32 PCM to the pipe until `seconds` / stop."""
    try:
        import numpy as np  # noqa: F401
        import sounddevice as sd
    except ImportError:
        os.close(write_fd)
        raise SystemExit("live mic needs `pip install sounddevice numpy` (or use --wav)")

    def cb(indata, _frames, _t, status):
        if status:
            print("  (audio)", status, file=sys.stderr)
        try:
            os.write(write_fd, indata.tobytes())  # float32 mono, already 16 kHz
        except BrokenPipeError:
            stop.set()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=cb):
        deadline = time.monotonic() + seconds if seconds > 0 else None
        while not stop.is_set():
            time.sleep(0.1)
            if deadline and time.monotonic() >= deadline:
                break
    os.close(write_fd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="brain/nemotron", help="ASR model name (brain/nemotron | brain/qwen-asr)")
    ap.add_argument("--window-ms", type=int, default=1000, help="server-side transcription window")
    ap.add_argument("--seconds", type=float, default=0.0, help="mic capture duration (0 = until Ctrl-C)")
    ap.add_argument("--wav", help="stream this 16 kHz mono wav instead of the mic")
    ap.add_argument("--fast", action="store_true", help="with --wav, feed as fast as possible (benchmark) instead of real time")
    args = ap.parse_args()

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            skip(f"model {args.model!r} not served (have: {models}); start `brain serve --dbus` with BRAIN_NEMOTRONASR / BRAIN_QWEN3ASR set")

        r, w = os.pipe()
        params = {"window_ms": args.window_ms, "sample_rate": SAMPLE_RATE, "prompt_id": 0}
        job, frames = brain.stream_transcribe(args.model, r, params)
        print(f"stream {job}: {args.model} @ {args.window_ms} ms windows — speak now\n" if not args.wav else f"stream {job}: transcribing {args.wav}\n")

        stop = threading.Event()
        if args.wav:
            feeder = threading.Thread(target=stream_wav, args=(w, args.wav, not args.fast), daemon=True)
        else:
            feeder = threading.Thread(target=stream_mic, args=(w, stop, args.seconds), daemon=True)
        feeder.start()

        transcript: list[str] = []
        t0 = time.monotonic()
        try:
            for frame, _fds in frames:
                kind = frame.get("type")
                if kind == "segment":
                    txt = frame.get("text", "")
                    if frame.get("final"):
                        continue
                    if txt:
                        transcript.append(txt)
                        print(f"  [{frame['index']:>3} | {time.monotonic() - t0:5.1f}s] {txt}")
                elif kind == "done":
                    res = frame.get("result", {})
                    print(f"\n── done ({res.get('segments', '?')} segments) ──\n{res.get('text', ' '.join(transcript))}")
                elif kind == "error":
                    print("  error:", frame.get("message"), file=sys.stderr)
        except KeyboardInterrupt:
            stop.set()
            print("\n(interrupted)")
        feeder.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
