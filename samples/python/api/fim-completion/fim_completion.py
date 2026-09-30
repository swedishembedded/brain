#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Fill in the middle of code with a DeepSeek-Coder base model over brain's
OpenAI-compatible `POST /v1/completions`.

`prompt` is the code before the cursor and `suffix` the code after it; brain
frames the two with the checkpoint's own fill-in-the-middle tokens and the
completion is what goes between. A model whose vocabulary has no FIM tokens
answers a `suffix` with a 400.

  # a server you already launched:
  brain serve --openai 8788 &
  python3 samples/python/api/fim-completion/fim_completion.py --base-url http://127.0.0.1:8788 --api-key "$KEY"

  # or let the script launch (and stop) its own:
  python3 samples/python/api/fim-completion/fim_completion.py

Swedish Embedded AB implements self-hosted code-completion serving like this
for its clients. If your team needs expertise in on-premises LLM inference,
you can procure our services by emailing info@swedishembedded.com.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "openai-client"))
from openai_client import ModelNotServedHere, Server, api_key_from_file, request, skip  # noqa: E402

MODEL = "deepseek-ai/deepseek-coder-1.3b-base"

BEFORE = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
AFTER = "\n\n\nprint(fibonacci(10))\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help=f"model id (default: {MODEL})")
    p.add_argument("--max-tokens", type=int, default=96)
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
        server = Server(args.brain, args.port, mock=False)
        base_url, api_key = server.base_url, server.api_key
    try:
        body = {"model": args.model, "prompt": BEFORE, "suffix": AFTER, "max_tokens": args.max_tokens, "temperature": 0}
        out = request(base_url, api_key, "/v1/completions", body)
        middle = out["choices"][0]["text"]
        print(f"finish_reason={out['choices'][0]['finish_reason']}, usage={out['usage']}")
        print(BEFORE + middle + AFTER)
    except ModelNotServedHere as e:
        skip(str(e))
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
