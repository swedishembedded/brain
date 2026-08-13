#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Talk to Qwen3-Omni's Thinker over any of brain's three transports.

**Scope, honestly**: text in/out, speech in (`--in-speech`), image in
(`--in-image`), and video in (`--in-video`) are all real end to end over
`--dbus` (real
audio/vision tower encode + host-side embedding splice + real M-RoPE
positions, `crate::mm`). `--in-video` needs PyAV (`pip install av`) to
extract frames -- unlike speech/image, which stay zero-dependency by design,
real video demuxing genuinely needs a decoder, so this `skip()`s cleanly
(exit 77) rather than adding a hard requirement. This is true of BOTH served
Thinker models -- `brain/omni` (streamed bf16 weights) and
`brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` (GPU-resident int8, layer-sharded across
however many GPUs it needs, ~25-50x brain/omni's tokens/second on the same
hardware) -- since both build their multimodal prompt through the SAME
`crate::mm::build_multimodal_prompt`; `--model` selects between them (see
Examples). The int8 model's OWN vision/audio tower weights are read from a
real HF checkpoint directory too (`BRAIN_OMNI_HF_DIR`, same env var
`brain/omni` reads) -- its own int8 checkpoint stores those towers
quantized, which nothing here executes yet, so `BRAIN_OMNI_HF_DIR` has to be
set alongside `BRAIN_OMNI_INT8_CHECKPOINT` for `--in-speech`/`--in-image`/
`--in-video` against it to work (see `omni::int8_thinker_resident`'s module
doc for the full reasoning); text-only `--in-text` against it does not need
an HF dir at all. `--in-mic` and `--out-mic`/`--out-audio` are still
`skip()`s -- live capture needs extra dependencies this script deliberately
doesn't take on, and speech OUTPUT needs Talker+Code2Wav, not wired into a
generation loop yet. Generation is still validation-tier for weight I/O on
`brain/omni` specifically (`crate::generate`'s own doc): the KV-cache makes
attention O(cached length), but every layer's weights are still streamed
fresh from the checkpoint per generated token; `brain/Qwen3-Omni-30B-A3B-Instruct-W8A16`
does not have this limitation -- its weights are GPU-resident.

`--openai`/`--anthropic` reject ALL blob inputs (audio/image/video alike)
with a clear `NotImplementedError` from THIS script's client library
(`brain_py.openai.BrainOpenAI`/`brain_py.anthropic.BrainAnthropic` do not
translate a `blobs=` kwarg into `image_url`/`input_audio` content parts) --
`--dbus` is the one transport THIS SCRIPT carries blobs through generically.
That is a client-library gap, not a server one: `crates/apiserve/src/media.rs`
already decodes real `image_url`/`input_audio` OpenAI content parts (and
Anthropic's `image` content blocks) into the same blobs `--dbus` sends, for
either served Thinker model -- a raw HTTP client that builds those content
parts itself (unlike this script) gets real multimodal input over
`--openai`/`--anthropic` too, no server-side change needed.

Examples:
  # D-Bus (needs `BRAIN_OMNI_HF_DIR=... brain serve --dbus` running):
  python3 examples/omni/omni.py --dbus --in-text "Say hello in French." --out-stdio
  python3 examples/omni/omni.py --dbus --in-speech clip.wav --out-stdio
  python3 examples/omni/omni.py --dbus --in-image photo.ppm --in-text "What is this?" --out-stdio
  python3 examples/omni/omni.py --dbus --in-video clip.mp4 --in-text "What happens in this clip?" --out-stdio

  # The fast GPU-resident int8 path (needs `BRAIN_OMNI_INT8_CHECKPOINT=...
  # BRAIN_OMNI_INT8_TOKENIZER_DIR=... [BRAIN_OMNI_HF_DIR=... for multimodal]
  # brain serve --dbus` running):
  python3 examples/omni/omni.py --dbus --model brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 --in-text "Say hello in French." --out-stdio
  python3 examples/omni/omni.py --dbus --model brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 --in-image photo.ppm --in-text "What is this?" --out-stdio

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
from brain_py.image import from_pil_rgb, load_ppm  # noqa: E402


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


def load_video_blob(path: str, max_frames: int = 8) -> tuple[bytes, dict]:
    """Extract up to `max_frames` evenly-spaced frames from a video file via
    PyAV, each converted to interleaved-HWC f32 RGB and concatenated into ONE
    payload -- the wire shape `capability::blob::decode_video_hwc` expects
    (`Media::Bytes` + meta `{"frames","w","h","c"}`; see that function's own
    doc for why one concatenated blob rather than a repeated one).

    Requires `av` (`pip install av`) -- unlike `--in-image`/`--in-speech`
    (zero-dependency by design), real frame extraction genuinely needs a
    demuxer/decoder, so this follows the documented pattern instead: skip
    cleanly (exit 77) when the dependency is missing rather than adding a
    hard requirement to this script's default install."""
    try:
        import av
    except ImportError:
        skip("--in-video needs PyAV (`pip install av`) for frame extraction -- not installed")

    container = av.open(path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    wanted = set(range(0, total, max(1, total // max_frames))) if total > 0 else None

    frames: list[bytes] = []
    w = h = 0
    for i, frame in enumerate(container.decode(stream)):
        if wanted is not None and i not in wanted:
            continue
        img = frame.to_image()
        w, h = img.size
        frames.append(from_pil_rgb(img))
        if len(frames) >= max_frames:
            break
    container.close()

    if not frames:
        raise SystemExit(f"{path}: no frames decoded")
    return b"".join(frames), {"media": "bytes", "frames": len(frames), "w": w, "h": h, "c": 3}


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
    """`--in-mic` and both speech-output flags still `skip()` with a specific
    reason (see the module doc for why); `--in-speech`/`--in-image`/
    `--in-video` are all real now and handled in `main()` (`--in-video`
    itself `skip()`s separately, inside `load_video_blob`, only when PyAV is
    missing -- not unconditionally like the flags below)."""
    unimplemented = {
        "in_mic": "microphone capture needs sounddevice (or similar), not a dependency this example takes on -- pass --in-speech with a pre-recorded WAV instead",
        "out_mic": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
        "out_audio": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
    }
    for flag, reason in unimplemented.items():
        if getattr(args, flag):
            skip(f"--{flag.replace('_', '-')}: {reason}")
    if not args.in_text and not args.in_speech and not args.in_image and not args.in_video:
        skip("at least one of --in-text / --in-speech / --in-image / --in-video is required")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    transport = ap.add_argument_group("transport (exactly one required)")
    transport.add_argument("--dbus", action="store_true", help="use the D-Bus transport")
    transport.add_argument("--openai", metavar="URL", help="use the OpenAI-compatible HTTP transport (e.g. localhost:8788)")
    transport.add_argument("--anthropic", metavar="URL", help="use the Anthropic-compatible HTTP transport (e.g. localhost:8787)")
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (brain serve prints 'APIKEY <provider> <key>' at startup, or see --api-keys-out)")

    inputs = ap.add_argument_group("input (--in-text/--in-speech/--in-image/--in-video are real; --dbus only for the latter three)")
    inputs.add_argument("--in-text", metavar="TEXT", help="the text prompt")
    inputs.add_argument("--in-speech", metavar="WAV", help="16kHz mono 16-bit PCM WAV -- spliced into the prompt as real audio-tower embeddings (--dbus only)")
    inputs.add_argument("--in-mic", action="store_true", help="not yet implemented -- see this module's doc")
    inputs.add_argument("--in-image", metavar="PPM", help="binary PPM (P6) image -- spliced into the prompt as real vision-tower embeddings (--dbus only)")
    inputs.add_argument("--in-video", metavar="PATH", help="video file -- up to 8 evenly-spaced frames extracted via PyAV and spliced into the prompt as real vision-tower embeddings (--dbus only; needs `pip install av`)")

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
    if args.in_video:
        blobs["video"], meta["video"] = load_video_blob(args.in_video)
    if blobs:
        kwargs["blobs"] = blobs
        kwargs["meta"] = meta

    # --in-text is optional when a medium is given (the server still needs
    # SOME text -- omni's generate errors on a wholly empty prompt), so this
    # falls back to a generic instruction rather than requiring the caller to
    # always spell out "describe this" by hand.
    prompt = args.in_text or ("Describe this." if (args.in_image or args.in_video) else "Transcribe or describe this audio.")

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
