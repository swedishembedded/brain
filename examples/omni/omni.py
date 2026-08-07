#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Talk to Qwen3-Omni's Thinker over any of brain's three transports.

**Scope, honestly**: only text in, text out is implemented end to end today
(`docs/models/omni/status.md`'s M9a/M14 entries). `--in-speech`/`--in-mic`/
`--in-image`/`--in-video` and `--out-mic`/`--out-audio` are accepted (so this
script's interface matches the full matrix Omni will eventually support
without a breaking change) but `skip()` with a clear message -- Talker,
Code2Wav, and multimodal input splice are not wired into a generation loop
yet. Generation itself is validation-tier (`crate::generate`'s own doc): no
KV-cache, so a real 48-layer/128-expert run streams weights fresh per
generated token and can take minutes, not seconds, per token.

Examples:
  # D-Bus (needs `BRAIN_OMNI_HF_DIR=... brain serve --dbus` running):
  python3 examples/omni/omni.py --dbus --in-text "Say hello in French." --out-stdio

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
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainBase, skip  # noqa: E402


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
    """Every input/output flag beyond the implemented text<->text path
    skip()s with a specific reason, rather than either crashing or silently
    ignoring the flag -- see this module's doc for what is/isn't real yet."""
    unimplemented_inputs = {
        "in_speech": "speech input needs the audio tower spliced into the Thinker embedding sequence (not wired into generate yet)",
        "in_mic": "microphone input needs the same audio-splice path as --in-speech",
        "in_image": "image input needs the vision tower spliced into the Thinker embedding sequence (not wired into generate yet)",
        "in_video": "video input needs the vision tower's video path (M5, deferred) plus the same splice wiring as --in-image",
    }
    for flag, reason in unimplemented_inputs.items():
        if getattr(args, flag):
            skip(f"--{flag.replace('_', '-')}: {reason}")
    unimplemented_outputs = {
        "out_mic": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
        "out_audio": "speech output needs Talker + code predictor + Code2Wav chained together (not built yet)",
    }
    for flag, reason in unimplemented_outputs.items():
        if getattr(args, flag):
            skip(f"--{flag.replace('_', '-')}: {reason}")
    if not args.in_text:
        skip("--in-text is required (the only implemented input today)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    transport = ap.add_argument_group("transport (exactly one required)")
    transport.add_argument("--dbus", action="store_true", help="use the D-Bus transport")
    transport.add_argument("--openai", metavar="URL", help="use the OpenAI-compatible HTTP transport (e.g. localhost:8788)")
    transport.add_argument("--anthropic", metavar="URL", help="use the Anthropic-compatible HTTP transport (e.g. localhost:8787)")
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (brain serve prints 'APIKEY <provider> <key>' at startup, or see --api-keys-out)")

    inputs = ap.add_argument_group("input (--in-text is the only one implemented)")
    inputs.add_argument("--in-text", metavar="TEXT", help="the text prompt")
    inputs.add_argument("--in-speech", metavar="WAV", help="not yet implemented -- see this module's doc")
    inputs.add_argument("--in-mic", action="store_true", help="not yet implemented -- see this module's doc")
    inputs.add_argument("--in-image", metavar="PATH", help="not yet implemented -- see this module's doc")
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

    # No --stream: BrainBase's transport-agnostic on_progress carries
    # (step, total, message), not per-token delta text (that's a
    # BrainDBus.subscribe-only `on_delta` kwarg, not part of the abstract
    # contract this script relies on to work identically over all three
    # transports) -- and the real Omni resident doesn't emit true per-token
    # progress yet either (crate::resident_omni's two Progress::step ticks
    # are start/end, not one per generated token). A --stream flag that
    # printed the literal string "token" N times would be actively
    # misleading, so this waits for the full reply instead.
    text = brain.chat(args.in_text, **kwargs)

    if args.out_text:
        Path(args.out_text).write_text(text)
        print(f"wrote {len(text)} chars -> {args.out_text}")
    else:
        print(text)


if __name__ == "__main__":
    main()
