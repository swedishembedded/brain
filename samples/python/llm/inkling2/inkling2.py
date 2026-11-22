#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements serving of reasoning mixture-of-experts language models for its clients. If your
# team needs expertise in running such a model behind a standard API with its reasoning and tool calls read back
# then you can procure our services by sending an email to info@swedishembedded.com.

"""Ask Inkling 2 a question at two reasoning efforts over any of brain's three transports.

Inkling 2 replies in typed messages: a thinking message, then the answer, then the end of the reply. brain reads the
two back separately (the reasoning in `reasoning_content`, the answer as the text) and takes the checkpoint's
`reasoning_effort` (low, medium, high, or a number from 0 to 1) as a template variable. This script asks the same
question at a low and a high effort and prints how much each reply reasoned before answering.

Examples:
  # D-Bus (a private session bus and `brain serve --dbus`, after `brain pull thinkingmachines/Inkling-2-Mini`):
  tools/dbus-session.sh --serve "--dbus" -- \\
      python3 samples/python/llm/inkling2/inkling2.py --dbus

  # OpenAI-compatible HTTP (`brain serve --openai 8788` running):
  python3 samples/python/llm/inkling2/inkling2.py --openai localhost:8788 --api-key sk-brain-...

  # Quick wire-contract check against the mock resident (no weights needed):
  tools/dbus-session.sh --serve "--dbus --mock" -- \\
      python3 samples/python/llm/inkling2/inkling2.py --dbus --model brain/mock
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

QUESTION = "A train leaves at 09:40 and the trip takes 2 hours 35 minutes. When does it arrive?"


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
    transport.add_argument("--api-key", help="API key for --openai/--anthropic (`brain serve` prints 'APIKEY <provider> <key>' at startup)")
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini", help="served model name (the id `brain pull` stored it under; brain/mock for a wire-contract check)")
    ap.add_argument("--max-new", type=int, default=512, help="max tokens per reply, reasoning included")
    args = ap.parse_args()

    brain = build_transport(args)
    try:
        served = brain.models()
    except NotImplementedError:
        served = None
    if served is not None and args.model not in served:
        skip(f"model {args.model!r} is not served (served: {served}); fetch it with `brain pull thinkingmachines/Inkling-2-Mini` and restart `brain serve`")

    messages = [{"role": "user", "content": QUESTION}]
    for effort in ("low", "high"):
        params = {"messages": messages, "max_new": args.max_new, "temp": 0.0}
        if args.model != "brain/mock":
            params["reasoning_effort"] = effort
        started = time.monotonic()
        out = brain.run(args.model, "generate", params, timeout=1800.0)
        reasoning = str(out.outputs.get("reasoning_content", ""))
        print(f"[{time.monotonic() - started:6.2f}s] effort {effort}: {len(reasoning.split())} words of reasoning")
        print(f"  -> {out.text().strip()}")


if __name__ == "__main__":
    main()
