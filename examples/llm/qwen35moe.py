#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Talk to Qwen3.5-35B-A3B's decoder over any of brain's three transports.

**Scope, honestly**: text in/out only — no audio/image/video splice (that is
`examples/omni/omni.py`'s territory, a different model). Single-GPU,
single-active-sequence serving: `crate::resident_qwen35moe::Qwen35Resident`
(`crates/cli/src/resident_qwen35moe.rs`) is fp32 weights + fp32 KV, one
sequence truly decoding on the GPU at a time (several may be RESIDENT and
interleaved by the scheduler, never batched into one GPU dispatch) — see
that module's own doc for the complete list of what this does NOT have yet
(int8 KV, LoRA adapter folding, multi-GPU sharding, a `.gguf` serving path).
`--dbus`/`--openai`/`--anthropic` all converge on the exact same `generate`
action (`crates/qwen35moe/src/caps.rs` for `brain do`/D-Bus-direct-dispatch
callers; the residency-managed, always-hot path above for the served HTTP/
D-Bus surfaces) — this script is what proves that: same `messages`/`prompt`/
`max_new`/... params, same response shape, three wires.

No `--stream`: brain_py's transport-agnostic `on_progress` callback carries
`(step, total, message)`, not per-token delta text (that's a
`BrainDBus.subscribe`-only `on_delta` kwarg, outside the abstract contract
this script relies on to behave identically over all three transports) —
same reasoning `examples/omni/omni.py` documents for the same choice. The
server DOES emit one real `Progress` per generated token
(`qwen35moe::caps`/`Qwen35Resident` both stream token-by-token) — a caller
using `brain_py.dbus.BrainDBus.subscribe(..., on_delta=...)` directly (not
through this script) can watch it arrive live.

Examples:
  # D-Bus (needs `BRAIN_QWEN35MOE_WEIGHTS=... BRAIN_QWEN35MOE_TOKENIZER=... \\
  # brain serve --dbus` running):
  python3 examples/llm/qwen35moe.py --dbus --in-text "Say hello in French." --out-stdio

  # OpenAI-compatible HTTP (needs `brain serve --openai 8788` running, with
  # BRAIN_QWEN35MOE_WEIGHTS/_TOKENIZER set for that process):
  python3 examples/llm/qwen35moe.py --openai localhost:8788 --api-key sk-brain-... \\
      --in-text "2+2=" --out-stdio

  # Anthropic-compatible HTTP:
  python3 examples/llm/qwen35moe.py --anthropic localhost:8787 --api-key sk-brain-... \\
      --in-text "2+2=" --out-stdio

  # Quick, deps-free wire-contract check against the mock resident (no
  # Qwen3.5 weights needed -- exercises the exact same `generate` action shape):
  BRAIN_MOCK=1 dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/llm/qwen35moe.py --dbus --model brain/mock --in-text hi --out-stdio
  '
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    transport = ap.add_argument_group("transport (exactly one required)")
    transport.add_argument("--dbus", action="store_true", help="use the D-Bus transport")
    transport.add_argument("--openai", metavar="URL", help="use the OpenAI-compatible HTTP transport (e.g. localhost:8788)")
    transport.add_argument("--anthropic", metavar="URL", help="use the Anthropic-compatible HTTP transport (e.g. localhost:8787)")
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (brain serve prints 'APIKEY <provider> <key>' at startup, or see --api-keys-out)")

    ap.add_argument("--in-text", metavar="TEXT", required=True, help="the text prompt")
    ap.add_argument("--out-stdio", action="store_true", help="print the generated text to stdout (default)")
    ap.add_argument("--out-text", metavar="PATH", help="write the generated text to a file instead of stdout")
    ap.add_argument("--model", default="brain/qwen35moe", help="served model name (default brain/qwen35moe; brain/mock for a deps-free wire-contract check)")
    ap.add_argument("--max-new", type=int, default=32, help="max tokens to generate")
    ap.add_argument("--temp", type=float, help="sampling temperature (omit for the server default; <= 0 is greedy)")
    ap.add_argument("--system", help="optional system prompt")
    args = ap.parse_args()

    brain = build_transport(args)

    # BrainAnthropic has no model-listing endpoint (Anthropic's API has no
    # /v1/models equivalent) -- skip the pre-check there and let a real
    # failure surface from the actual generate call instead of crashing on a
    # NotImplementedError, matching examples/omni/omni.py's own handling.
    try:
        served = brain.models()
    except NotImplementedError:
        served = None
    if served is not None and args.model not in served:
        skip(f"model {args.model!r} is not served (served: {served}); for real Qwen3.5-35B-A3B: "
             "BRAIN_QWEN35MOE_WEIGHTS=<checkpoint.safetensors> BRAIN_QWEN35MOE_TOKENIZER=<tokenizer.json> brain serve --dbus")

    kwargs = {"model": args.model, "max_new": args.max_new}
    if args.system:
        kwargs["system"] = args.system
    if args.temp is not None:
        kwargs["temp"] = args.temp

    text = brain.chat(args.in_text, **kwargs)

    if args.out_text:
        Path(args.out_text).write_text(text)
        print(f"wrote {len(text)} chars -> {args.out_text}")
    else:
        print(text)


if __name__ == "__main__":
    main()
