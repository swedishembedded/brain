#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""End-to-end ASR benchmark: N concurrent streams through brain over D-Bus.

Drives `--streams` concurrent `StreamTranscribe` sessions of the SAME wav clip against
a running `brain serve --dbus`, feeding each as fast as possible, and reports per-model
throughput, real-time factor (RTF), first-/final-segment latency, and the scheduler's
batch counters (proof that concurrent windows actually batched).

    BRAIN_NEMOTRONASR=$BRAIN_TESTDATA/asr/nemotron/hf \
      dbus-run-session -- bash -c '
        brain serve --dbus --device cpu & sleep 2
        python3 examples/asr/bench_streams.py --model brain/nemotron \
          --wav $BRAIN_TESTDATA/asr/audio/librispeech_mr_quilter.wav \
          --streams 1,2,4
      '

Reports one row per concurrency level so you can see batching scale throughput.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

SAMPLE_RATE = 16000


def read_wav_f32(path: str) -> tuple[bytes, float]:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit PCM")
        n = w.getnframes()
        raw = w.readframes(n)
    import array

    ints = array.array("h")
    ints.frombytes(raw)
    floats = array.array("f", (s / 32768.0 for s in ints))
    return floats.tobytes(), n / SAMPLE_RATE


def one_stream(brain_bus_factory, model: str, pcm: bytes, window_ms: int, out: dict) -> None:
    """Run a single stream; record first/final-segment latency and text."""
    brain = brain_bus_factory()
    try:
        r, w = os.pipe()
        params = {"window_ms": window_ms, "sample_rate": SAMPLE_RATE, "prompt_id": 0}
        t0 = time.monotonic()
        _job, frames = brain.stream_transcribe(model, r, params)

        def feed() -> None:
            chunk = 1 << 15
            for off in range(0, len(pcm), chunk):
                os.write(w, pcm[off : off + chunk])
            os.close(w)

        threading.Thread(target=feed, daemon=True).start()

        first = None
        segs = 0
        text = []
        for frame, _fds in frames:
            k = frame.get("type")
            if k == "segment" and not frame.get("final"):
                if first is None:
                    first = time.monotonic() - t0
                if frame.get("text"):
                    segs += 1
                    text.append(frame["text"])
            elif k == "done":
                out["first"] = first if first is not None else float("nan")
                out["wall"] = time.monotonic() - t0
                out["segments"] = segs
                out["text"] = frame.get("result", {}).get("text", " ".join(text))
                return
            elif k == "error":
                out["error"] = frame.get("message")
                return
    finally:
        brain.close()


def run_level(model: str, pcm: bytes, audio_s: float, window_ms: int, n: int) -> dict:
    outs = [dict() for _ in range(n)]
    threads = [threading.Thread(target=one_stream, args=(BrainDBus, model, pcm, window_ms, outs[i])) for i in range(n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    walls = [o.get("wall", float("nan")) for o in outs if "wall" in o]
    firsts = [o["first"] for o in outs if o.get("first") == o.get("first")]  # drop NaN
    errors = [o["error"] for o in outs if "error" in o]
    total_audio = audio_s * len(walls)
    return {
        "n": n,
        "ok": len(walls),
        "errors": errors,
        "wall": wall,
        "audio_total_s": total_audio,
        "throughput_rtf": (total_audio / wall) if wall > 0 else 0.0,  # >1 = faster than real time, aggregate
        "per_stream_rtf": statistics.mean([audio_s / x for x in walls]) if walls else 0.0,
        "first_latency_ms": 1000 * statistics.mean(firsts) if firsts else float("nan"),
        "sample_text": outs[0].get("text", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="brain/nemotron")
    ap.add_argument("--wav", required=True, help="16 kHz mono 16-bit wav to replay per stream")
    ap.add_argument("--streams", default="1,2,4", help="comma list of concurrency levels")
    ap.add_argument("--window-ms", type=int, default=1000)
    args = ap.parse_args()

    pcm, audio_s = read_wav_f32(args.wav)
    levels = [int(x) for x in args.streams.split(",") if x.strip()]

    with BrainDBus() as probe:
        if args.model not in probe.models():
            skip(f"model {args.model!r} not served (have {probe.models()})")
        print(f"model={args.model}  clip={audio_s:.2f}s  window={args.window_ms}ms\n")
        print(f"{'streams':>7} {'ok':>3} {'wall(s)':>8} {'aggRTF':>7} {'perRTF':>7} {'1st(ms)':>8}")
        stats_before = probe.stats()

    for n in levels:
        r = run_level(args.model, pcm, audio_s, args.window_ms, n)
        print(f"{r['n']:>7} {r['ok']:>3} {r['wall']:>8.2f} {r['throughput_rtf']:>7.2f} {r['per_stream_rtf']:>7.2f} {r['first_latency_ms']:>8.0f}")
        if r["errors"]:
            print("   errors:", r["errors"][:3], file=sys.stderr)

    with BrainDBus() as probe:
        stats_after = probe.stats()
        print("\nscheduler batches:", stats_after.get("batches"), " max_batch:", stats_after.get("max_batch"),
              " builds:", stats_after.get("builds"), " (before:", stats_before.get("batches"), "/", stats_before.get("max_batch"), ")")
        print("sample transcript:", (run_level(args.model, pcm, audio_s, args.window_ms, 1)["sample_text"] or "")[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())
