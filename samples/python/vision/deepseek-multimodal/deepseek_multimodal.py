#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Ask DeepSeek-VL or Janus-Pro about an image, or have Janus-Pro draw one,
over brain's OpenAI-compatible HTTP surface.

  chat: an image as an `image_url` data URL next to a question, through
        /v1/chat/completions (brain/deepseekvl or brain/januspro);
  draw: a prompt through /v1/images/generations at 384x384, the one size
        Janus-Pro generates (brain/januspro), saved as PNG.

  # a server you already launched:
  brain serve --openai 8788 --api-keys-out /tmp/keys.json &
  python3 samples/python/vision/deepseek-multimodal/deepseek_multimodal.py chat --image photo.png \\
      --base-url http://127.0.0.1:8788 --keys-file /tmp/keys.json
  python3 samples/python/vision/deepseek-multimodal/deepseek_multimodal.py draw --prompt "A red apple" \\
      --base-url http://127.0.0.1:8788 --keys-file /tmp/keys.json --out /tmp/apple.png

  # or let the script launch (and stop) its own server: drop --base-url.

Swedish Embedded AB implements self-hosted multimodal model serving like this
for its clients. If your team needs expertise in running vision-language and
text-to-image models on your own hardware, you can procure our services by
emailing info@swedishembedded.com.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api" / "openai-client"))
from openai_client import ModelNotServedHere, Server, api_key_from_file, request, skip  # noqa: E402


def chat(base_url: str, api_key: str, model: str, image: Path, question: str, max_tokens: int) -> None:
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}],
    }
    out = request(base_url, api_key, "/v1/chat/completions", body)
    choice = out["choices"][0]
    print(f"answer: {choice['message']['content']!r}  (finish_reason={choice['finish_reason']}, usage={out['usage']})")


def draw(base_url: str, api_key: str, model: str, prompt: str, seed: int, out: Path) -> None:
    body = {"model": model, "prompt": prompt, "size": "384x384", "seed": seed}
    res = request(base_url, api_key, "/v1/images/generations", body)
    out.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["chat", "draw"])
    p.add_argument("--model", help="model id (default: brain/deepseekvl for chat, brain/januspro for draw)")
    p.add_argument("--image", type=Path, default=Path(__file__).resolve().parents[4] / "docs" / "quickstart" / "img" / "seed.png", help="chat: the image to ask about (PNG)")
    p.add_argument("--question", default="What is in this image? Answer in one sentence.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt", default="A red apple on a wooden table.", help="draw: what the image shows")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("/tmp/janus.png"), help="draw: where the PNG goes")
    p.add_argument("--base-url", help="already-running server, e.g. http://127.0.0.1:8788 (skips self-launch)")
    p.add_argument("--api-key", help="Bearer key for --base-url")
    p.add_argument("--keys-file", help="read the 'openai' key from a --api-keys-out JSON file instead of --api-key")
    p.add_argument("--brain", default=os.environ.get("BRAIN", "./target/release/brain"), help="brain binary (self-launch mode only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8788")), help="port to launch on (self-launch mode only)")
    args = p.parse_args()
    model = args.model or ("brain/deepseekvl" if args.mode == "chat" else "brain/januspro")

    server = None
    if args.base_url:
        base_url = args.base_url
        api_key = args.api_key or (api_key_from_file(args.keys_file) if args.keys_file else None)
        if not api_key:
            p.error("--api-key or --keys-file is required with --base-url")
    else:
        if not os.access(args.brain, os.X_OK):
            skip(f"brain binary not found at {args.brain!r} (build: make release)")
        server = Server(args.brain, args.port, mock=model == "brain/mock")
        base_url, api_key = server.base_url, server.api_key
    try:
        if args.mode == "chat":
            chat(base_url, api_key, model, args.image, args.question, args.max_tokens)
        else:
            draw(base_url, api_key, model, args.prompt, args.seed, args.out)
    except ModelNotServedHere as e:
        skip(str(e))
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
