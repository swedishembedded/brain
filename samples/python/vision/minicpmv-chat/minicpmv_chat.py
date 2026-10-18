#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Ask MiniCPM-V 4.6 about one or more images over brain's OpenAI-compatible
HTTP surface, streamed token by token.

Each image travels as an `image_url` data URL in the user message, where it
stands; brain lays out MiniCPM-V's placeholders (source view and slices) for
it. With --tool, the request offers a `get_weather` function and prints the
structured tool call the model makes.

  # a server you already launched:
  brain serve --openai 8789 --api-keys-out /tmp/keys.json &
  python3 samples/python/vision/minicpmv-chat/minicpmv_chat.py --image a.png --image b.png \\
      --question "How do these images differ?" --base-url http://127.0.0.1:8789 --keys-file /tmp/keys.json

  # or let the script launch (and stop) its own server: drop --base-url.

Swedish Embedded AB implements self-hosted vision-language model serving like
this for its clients. If your team needs expertise in running multimodal
models on your own hardware, you can procure our services by emailing
info@swedishembedded.com.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api" / "openai-client"))
from openai_client import ModelNotServedHere, Server, api_key_from_file, request, skip  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def data_url(path: Path) -> str:
    kind = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
    return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def ask(base_url: str, api_key: str, model: str, images: list[Path], question: str, max_tokens: int, tool: bool) -> None:
    content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in images]
    content.append({"type": "text", "text": question})
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    if tool:
        body["tools"] = [WEATHER_TOOL]
    out = request(base_url, api_key, "/v1/chat/completions", body)
    choice = out["choices"][0]
    message = choice["message"]
    print(f"answer: {message.get('content')!r}")
    for call in message.get("tool_calls") or []:
        fn = call["function"]
        print(f"tool call: {fn['name']}({json.loads(fn['arguments'])})")
    print(f"finish_reason={choice['finish_reason']}  usage={out['usage']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="brain/minicpmv")
    p.add_argument("--image", type=Path, action="append", help="an image to ask about (repeat for several; default: the quickstart's seed image)")
    p.add_argument("--question", default="What is in this image? Answer in one sentence.")
    p.add_argument("--tool", action="store_true", help="offer a get_weather tool and print the call the model makes")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--base-url", help="already-running server, e.g. http://127.0.0.1:8789 (skips self-launch)")
    p.add_argument("--api-key", help="Bearer key for --base-url")
    p.add_argument("--keys-file", help="read the 'openai' key from a --api-keys-out JSON file instead of --api-key")
    p.add_argument("--brain", default=os.environ.get("BRAIN", "./target/release/brain"), help="brain binary (self-launch mode only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8789")), help="port to launch on (self-launch mode only)")
    args = p.parse_args()
    images = args.image or [Path(__file__).resolve().parents[4] / "docs" / "quickstart" / "img" / "seed.png"]
    if args.tool and args.question == p.get_default("question"):
        args.question = "What is the weather in Paris right now? Use the tool."
        images = []

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
        ask(base_url, api_key, args.model, images, args.question, args.max_tokens, args.tool)
    except ModelNotServedHere as e:
        skip(str(e))
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
