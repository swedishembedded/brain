#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for serving long-context language
# models at a fixed memory budget for its clients. If your team needs
# expertise in bounded-state inference and continuous batching then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Ask EdgeRock about a long document over any of brain's three transports, then ask again.

EdgeRock keeps a state of fixed size per conversation however long it grows, and a served conversation that
resends its history is resumed from a saved copy of that state. This script sends one long document with a
question, then the same document with a second question: the second request is answered from the saved state
and only its new tokens are read in, which the printed times show.

Examples:
  # D-Bus (a private session bus and `brain serve --dbus`, after `brain pull rockmatrix-ai/EdgeRock`):
  tools/dbus-session.sh --serve "--dbus" -- \\
      python3 samples/python/llm/edgerock/edgerock.py --dbus

  # OpenAI-compatible HTTP (`brain serve --openai 8788` running):
  python3 samples/python/llm/edgerock/edgerock.py --openai localhost:8788 --api-key sk-brain-...

  # Quick wire-contract check against the mock resident (no weights needed):
  BRAIN_MOCK=1 tools/dbus-session.sh --serve "--dbus" -- \\
      python3 samples/python/llm/edgerock/edgerock.py --dbus --model brain/mock
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
from brain_py.base import BrainBase, skip  # noqa: E402

VAULT_CODE = 4817
FILLER = "The ledger for the quarter lists routine purchases of paper, ink and cable, none of them remarkable. "


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


def document(paragraphs: int) -> str:
    """A long document that states one fact at its start and buries it under filler."""
    return f"The vault code is {VAULT_CODE}.\n" + FILLER * paragraphs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    transport = ap.add_argument_group("transport (exactly one required)")
    transport.add_argument("--dbus", action="store_true", help="use the D-Bus transport")
    transport.add_argument("--openai", metavar="URL", help="use the OpenAI-compatible HTTP transport (e.g. localhost:8788)")
    transport.add_argument("--anthropic", metavar="URL", help="use the Anthropic-compatible HTTP transport (e.g. localhost:8787)")
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (`brain serve` prints 'APIKEY <provider> <key>' at startup)")
    ap.add_argument("--model", default="rockmatrix-ai/EdgeRock", help="served model name (the id `brain pull` stored it under; brain/mock for a wire-contract check)")
    ap.add_argument("--paragraphs", type=int, default=60, help="filler paragraphs after the fact (the prompt grows by about 20 tokens each)")
    ap.add_argument("--max-new", type=int, default=24, help="max tokens per answer")
    args = ap.parse_args()

    brain = build_transport(args)
    try:
        served = brain.models()
    except NotImplementedError:
        served = None
    if served is not None and args.model not in served:
        skip(f"model {args.model!r} is not served (served: {served}); fetch it with `brain pull rockmatrix-ai/EdgeRock` and restart `brain serve`")

    doc = document(args.paragraphs)
    for question in ("What is the vault code?", "Is the vault code an even number?"):
        started = time.monotonic()
        answer = brain.chat(f"{doc}\n{question}", model=args.model, max_new=args.max_new, temp=0.0)
        print(f"[{time.monotonic() - started:6.2f}s] {question}\n  -> {answer.strip()}")


if __name__ == "__main__":
    main()
