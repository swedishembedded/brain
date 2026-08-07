#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Talk to Qwen3-Omni's Thinker over any of brain's three transports.

**Scope, honestly**: text in/out, speech in (`--in-speech`), and image in
(`--in-image`) are real end to end over `--dbus` (`docs/models/omni/status.md`'s
multimodal-input entry: real audio/vision tower encode + host-side embedding
splice + real M-RoPE positions, `crate::mm`). `--openai`/`--anthropic` reject
blob inputs with a clear `NotImplementedError` (their content-part wiring is a
separate, not-yet-done change server-side -- `--dbus` is the one transport
that carries blobs generically today). `--in-mic`/`--in-video` and
`--out-mic`/`--out-audio` are still `skip()`s -- live capture and video-frame
extraction need extra dependencies this script deliberately doesn't take on,
and speech OUTPUT needs Talker+Code2Wav, not wired into a generation loop yet.
Generation is still validation-tier for weight I/O (`crate::generate`'s own
doc): the KV-cache makes attention O(cached length), but every layer's
weights are still streamed fresh from the checkpoint per generated token.

Examples:
  # D-Bus (needs `BRAIN_OMNI_HF_DIR=... brain serve --dbus` running):
  python3 examples/omni/omni.py --dbus --in-text "Say hello in French." --out-stdio
  python3 examples/omni/omni.py --dbus --in-speech clip.wav --out-stdio
  python3 examples/omni/omni.py --dbus --in-image photo.ppm --in-text "What is this?" --out-stdio

  # OpenAI-compatible HTTP (needs `brain serve --openai 8788` running,
  # with BRAIN_OMNI_HF_DIR set for that process):
  python3 examples/omni/omni.py --openai localhost:8788 --in-text "2+2=" --out-stdio

  # Anthropic-compatible HTTP:
  python3 examples/omni/omni.py --anthropic localhost:8787 --in-text "2+2=" --out-stdio

  # Quick, deps-free wire-contract check against the mock resident:
  BRAIN_MOCK=1 brain serve --dbus &
  python3 examples/omni/omni.py --dbus --model brain/mock --in-text hi --out-stdio
"""
from __future__ import annotations

import argparse
import struct
import sys
import wave
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainBase, skip  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402


def load_speech_blob(path: str) -> bytes:
    """Read a WAV file (stdlib `wave`) into raw mono f32-LE PCM at 16kHz --
    the wire format `audio::asr_caps::wav_from_blob` (server-side) expects,
    same convention `transcribe_spec` documents. Multi-channel is downmixed
    (mean); anything not already 16kHz is rejected with a clear message
    rather than silently resampled wrong -- this example deliberately has no
    resampler (see the module doc's "extra dependencies" line)."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000:
            raise SystemExit(f"{path}: sample rate {w.getframerate()} Hz, need 16000 (resample first)")
        sampwidth, nch = w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise SystemExit(f"{path}: only 16-bit PCM WAV is supported (got {sampwidth * 8}-bit)")
    ints = struct.unpack(f"<{len(raw) // 2}h", raw)
    frames = len(ints) // nch
    samples = [sum(ints[i * nch : (i + 1) * nch]) / nch / 32768.0 for i in range(frames)]
    return struct.pack(f"<{len(samples)}f", *samples)


def load_image_blob(path: str) -> tuple[bytes, dict]:
    """Read a binary PPM (P6) into `(hwc_f32_bytes, meta)` -- the same
    zero-dependency image path `examples/imagegen` already uses
    (`brain_py.image.load_ppm`). PNG/JPEG need converting to PPM first
    (`convert photo.png photo.ppm` via ImageMagick, or PIL) -- this example
    stays dependency-free rather than taking on Pillow for one flag."""
    data, w, h = load_ppm(path)
    return data, {"media": "image", "w": w, "h": h, "c": 3}


def build_transport(args: argparse.Namespace) -> BrainBase:
    selected = [t for t in (args.dbus, args.openai, args.anthropic) if t]
    if len(selected) != 1:
        skip("exactly one of --dbus / --openai URL / --anthropic URL is required")
    if args.dbus:
        from brain_py.dbus import BrainDBus

        return BrainDBus()
    if args.openai:
        from brain_py.openai import BrainOpenAI

        return BrainOpenAI(args.openai, api_key=args.api_key)
    from brain_py.anthropic import BrainAnthropic

    return BrainAnthropic(args.anthropic, api_key=args.api_key)


def check_scope(args: argparse.Namespace) -> None:
    """`--in-mic`/`--in-video` and both speech-output flags still `skip()`
    with a specific reason (see the module doc for why); `--in-speech`/
    `--in-image` are real now and handled in `main()`."""
    unimplemented = {
        "in_mic": "microphone capture needs sounddevice (or similar), not a dependency this example takes on -- pass --in-speech with a pre-recorded WAV instead",
        "in_video": "video needs frame extraction (av/ffmpeg) not wired into this example yet -- the engine-side path exists (crate::mm::encode_video_frames) but nothing decodes a video file into frames here",
        "out_mic": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
        "out_audio": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
    }
    for flag, reason in unimplemented.items():
        if getattr(args, flag):
            skip(f"--{flag.replace('_', '-')}: {reason}")
    if not args.in_text and not args.in_speech and not args.in_image:
        skip("at least one of --in-text / --in-speech / --in-image is required")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    transport = ap.add_argument_group("transport (exactly one required)")
    transport.add_argument("--dbus", action="store_true", help="use the D-Bus transport")
    transport.add_argument("--openai", metavar="URL", help="use the OpenAI-compatible HTTP transport (e.g. localhost:8788)")
    transport.add_argument("--anthropic", metavar="URL", help="use the Anthropic-compatible HTTP transport (e.g. localhost:8787)")
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (brain serve prints 'APIKEY <provider> <key>' at startup, or see --api-keys-out)")

    inputs = ap.add_argument_group("input (--in-text/--in-speech/--in-image are real; --dbus only for the latter two)")
    inputs.add_argument("--in-text", metavar="TEXT", help="the text prompt")
    inputs.add_argument("--in-speech", metavar="WAV", help="16kHz mono 16-bit PCM WAV -- spliced into the prompt as real audio-tower embeddings (--dbus only)")
    inputs.add_argument("--in-mic", action="store_true", help="not yet implemented -- see this module's doc")
    inputs.add_argument("--in-image", metavar="PPM", help="binary PPM (P6) image -- spliced into the prompt as real vision-tower embeddings (--dbus only)")
    inputs.add_argument("--in-video", metavar="PATH", help="not yet implemented -- see this module's doc")

    outputs = ap.add_argument_group("output (--out-stdio is the only one implemented)")
    outputs.add_argument("--out-stdio", action="store_true", help="print the generated text to stdout (default)")
    outputs.add_argument("--out-mic", action="store_true", help="not yet implemented -- see this module's doc")
    outputs.add_argument("--out-audio", metavar="WAV", help="not yet implemented -- see this module's doc")
    outputs.add_argument("--out-text", metavar="PATH", help="write the generated text to a file instead of stdout")

    ap.add_argument("--model", default="brain/omni", help="served model name (default brain/omni; brain/mock for a deps-free wire-contract check)")
    ap.add_argument("--max-new", type=int, default=32, help="max tokens to generate")
    ap.add_argument("--system", help="optional system prompt")
    args = ap.parse_args()

    check_scope(args)
    brain = build_transport(args)

    # BrainAnthropic has no model-listing endpoint (Anthropic's API has no
    # /v1/models equivalent -- see that transport's own manifests() doc) --
    # skip the pre-check there and let a real failure surface from the
    # actual generate call instead of crashing on a NotImplementedError.
    try:
        served = brain.models()
    except NotImplementedError:
        served = None
    if served is not None and args.model not in served:
        skip(f"model {args.model!r} is not served (served: {served}); for real Omni: BRAIN_OMNI_HF_DIR=<checkpoint dir> brain serve --dbus")

    kwargs = {"model": args.model, "max_new": args.max_new}
    if args.system:
        kwargs["system"] = args.system

    blobs, meta = {}, {}
    if args.in_speech:
        blobs["audio"] = load_speech_blob(args.in_speech)
        meta["audio"] = {"media": "audio", "sample_rate": 16000}
    if args.in_image:
        blobs["image"], meta["image"] = load_image_blob(args.in_image)
    if blobs:
        kwargs["blobs"] = blobs
        kwargs["meta"] = meta

    # --in-text is optional when a medium is given (the server still needs
    # SOME text -- omni's generate errors on a wholly empty prompt), so this
    # falls back to a generic instruction rather than requiring the caller to
    # always spell out "describe this" by hand.
    prompt = args.in_text or ("Describe this." if args.in_image else "Transcribe or describe this audio.")

    # No --stream: BrainBase's transport-agnostic on_progress carries
    # (step, total, message), not per-token delta text (that's a
    # BrainDBus.subscribe-only `on_delta` kwarg, not part of the abstract
    # contract this script relies on to work identically over all three
    # transports) -- and the real Omni resident doesn't emit true per-token
    # progress yet either (crate::resident_omni's two Progress::step ticks
    # are start/end, not one per generated token). A --stream flag that
    # printed the literal string "token" N times would be actively
    # misleading, so this waits for the full reply instead.
    text = brain.chat(prompt, **kwargs)

    if args.out_text:
        Path(args.out_text).write_text(text)
        print(f"wrote {len(text)} chars -> {args.out_text}")
    else:
        print(text)


if __name__ == "__main__":
    main()
