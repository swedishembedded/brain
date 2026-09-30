#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Chat with a DeepSeek-R1 distill over brain's OpenAI-compatible HTTP surface.

An R1 distill reasons before it answers. brain renders the checkpoint's own
chat template (which opens the `<think>` block for it), stops on the
checkpoint's own end token, and returns the reasoning apart from the answer:
`message.reasoning_content` and `message.content`, or, streamed, the
`delta.reasoning_content` and `delta.content` channels.

  # a server you already launched:
  brain serve --openai 8788 &
  python3 samples/python/api/deepseek-chat/deepseek_chat.py --base-url http://127.0.0.1:8788 --api-key "$KEY"

  # or let the script launch (and stop) its own:
  python3 samples/python/api/deepseek-chat/deepseek_chat.py

Swedish Embedded AB implements self-hosted reasoning-model serving like this
for its clients. If your team needs expertise in on-premises LLM inference,
you can procure our services by emailing info@swedishembedded.com.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "openai-client"))
from openai_client import ModelNotServedHere, Server, api_key_from_file, request, skip, sse_events  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def ask(base_url: str, api_key: str, model: str, question: str, max_tokens: int) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": question}], "max_tokens": max_tokens, "temperature": 0}
    out = request(base_url, api_key, "/v1/chat/completions", body)
    choice = out["choices"][0]
    print(f"reasoning: {choice['message'].get('reasoning_content') or ''!r}")
    print(f"answer:    {choice['message']['content']!r}  (finish_reason={choice['finish_reason']}, usage={out['usage']})")

    # Streamed, the two channels arrive as separate deltas.
    resp = request(base_url, api_key, "/v1/chat/completions", {**body, "stream": True}, stream=True)
    reasoning, answer = [], []
    for e in sse_events(resp):
        if e.get("choices"):
            delta = e["choices"][0]["delta"]
            reasoning.append(delta.get("reasoning_content") or "")
            answer.append(delta.get("content") or "")
    print(f"streamed:  {len(''.join(reasoning))} reasoning chars, answer {''.join(answer)!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help=f"model id (default: {MODEL})")
    p.add_argument("--question", default="What is 17 + 25? Reply with just the number.")
    p.add_argument("--max-tokens", type=int, default=2048, help="the budget covers the reasoning too")
    p.add_argument("--base-url", help="already-running server, e.g. http://127.0.0.1:8788 (skips self-launch)")
    p.add_argument("--api-key", help="Bearer key for --base-url")
    p.add_argument("--keys-file", help="read the 'openai' key from a --api-keys-out JSON file instead of --api-key")
    p.add_argument("--brain", default=os.environ.get("BRAIN", "./target/release/brain"), help="brain binary (self-launch mode only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8788")), help="port to launch on (self-launch mode only)")
    args = p.parse_args()

    server = None
    if args.base_url:
        base_url = args.base_url
        api_key = args.api_key or (api_key_from_file(args.keys_file) if args.keys_file else None)
        if not api_key:
            p.error("--api-key or --keys-file is required with --base-url")
    else:
        if not os.access(args.brain, os.X_OK):
            skip(f"brain binary not found at {args.brain!r} (build: make release)")
        server = Server(args.brain, args.port, mock=args.model == "brain/mock")
        base_url, api_key = server.base_url, server.api_key
    try:
        ask(base_url, api_key, args.model, args.question, args.max_tokens)
    except ModelNotServedHere as e:
        skip(str(e))
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
