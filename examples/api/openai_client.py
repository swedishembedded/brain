#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Example client for brain's OpenAI-compatible HTTP surface (`crates/apiserve`).

Every other example in this tree drives brain over D-Bus. This one is the odd
transport out — plain HTTP, stdlib only (`urllib`/`http.client`, no `requests`) —
and exercises all three OpenAI route families in one pass:

  * `POST /v1/chat/completions`   non-streaming, then streaming (SSE deltas)
  * `POST /v1/embeddings`
  * `POST /v1/images/generations`

and demonstrates the auth contract none of the D-Bus examples need: brain prints
a fresh, per-launch API key as `APIKEY openai <key>` (see `crates/apiserve/src/
surface.rs`) before its listener binds; a client either scrapes that line or
reads the same key back from `--api-keys-out FILE`'s JSON.

Two ways to run it:

  # 1. point at a server you already launched (what the automated test harness
  #    does — see tests/e2e/examples.bats):
  brain serve --openai 8788 &
  python3 examples/api/openai_client.py --base-url http://127.0.0.1:8788 \\
      --api-key "$(...)" --model brain/mock

  # 2. let the script launch (and stop) its own server -- the default with no
  #    --base-url/--api-key given, mirroring examples/api/claude-with-brain.sh:
  BRAIN_MOCK=1 python3 examples/api/openai_client.py --model brain/mock
  python3 examples/api/openai_client.py --model Qwen/Qwen3-0.6B   # real model,
                                                                   # auto-fetched

`--model` defaults to `brain/mock` (deterministic, weight-free — see
`crates/cli/src/resident_mock.rs`) so this runs offline in CI; point it at any
served id (a `Qwen/Qwen3-0.6B`-style ref auto-fetches on first use) to exercise
a real model.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

OUT = Path(os.environ.get("OUT", "/tmp"))

# The exit code every example in this tree uses to mean "this environment can't
# run me" (missing model, missing binary, ...) — see brain_py.base.EXIT_SKIP,
# whose contract this matches exactly. Inlined rather than imported: unlike every
# other example here, this one is pure HTTP and has no reason to require jeepney
# (brain_py's D-Bus dependency) just to reuse a five-line helper.
EXIT_SKIP = 77


def skip(reason: str) -> None:
    print(f"SKIP: {reason}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


# --------------------------------------------------------------------- server

class Server:
    """A `brain serve --openai` process this script launched itself, torn down
    on exit. Only used when the caller didn't pass --base-url/--api-key."""

    def __init__(self, brain: str, port: int, mock: bool) -> None:
        rundir = Path(os.environ.get("RUNDIR") or "/tmp") / f"brain-openai-example-{os.getpid()}"
        rundir.mkdir(parents=True, exist_ok=True)
        self.log = rundir / "server.log"
        ready = rundir / "ready"
        env = dict(os.environ)
        if mock:
            env["BRAIN_MOCK"] = "1"
            env.setdefault("BRAIN_DEVICE", "cpu")
        with open(self.log, "wb") as f:
            self.proc = subprocess.Popen([brain, "serve", "--openai", str(port), "--ready-file", str(ready)], stdout=f, stderr=subprocess.STDOUT, env=env)
        for _ in range(60):
            if ready.exists():
                break
            if self.proc.poll() is not None:
                skip(f"brain server exited on startup:\n{self.log.read_text()}")
            time.sleep(0.5)
        else:
            self.stop()
            skip("brain server never became ready")
        self.base_url = f"http://127.0.0.1:{port}"
        self.api_key = self._read_key()

    def _read_key(self) -> str:
        for line in self.log.read_text().splitlines():
            if line.startswith("APIKEY openai "):
                return line.split(" ", 2)[2].strip()
        self.stop()
        skip(f"could not find the 'APIKEY openai <key>' line in the server log:\n{self.log.read_text()}")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def api_key_from_file(path: str) -> str:
    """Read the `openai` key out of a `--api-keys-out FILE` JSON (`crates/apiserve
    /src/surface.rs::write_keys_file`: `{"openai": "sk-brain-...", ...}`)."""
    keys = json.loads(Path(path).read_text())
    key = keys.get("openai")
    if not key:
        skip(f"{path} has no 'openai' key (got: {sorted(keys)})")
    return key


# ------------------------------------------------------------------- HTTP core

def request(base_url: str, api_key: str, path: str, body: Optional[dict] = None, stream: bool = False):
    """POST (or GET when body is None) `path`, Bearer-authed. Returns the parsed
    JSON body, or (for `stream=True`) the open response for line-by-line SSE."""
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
    )
    try:
        # A real model's FIRST request pays for its cold activation (WGSL pipeline
        # compile + weight upload) on top of the actual generation -- measured over
        # a minute on modest hardware for a small model. This must stay comfortably
        # above the server's own cold-build admission deadline
        # (BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS, default 180s -- see
        # crates/residency/src/admission.rs) or this demo would time out here
        # instead of getting the server's real (occasionally slower, still
        # honest) answer.
        resp = urllib.request.urlopen(req, timeout=240)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if e.code == 404:
            model = body.get("model") if body else None
            skip(f"{path}: model {model!r} not served/exposed here: {detail}")
        raise RuntimeError(f"{path}: HTTP {e.code}: {detail}") from e
    if stream:
        return resp
    return json.loads(resp.read())


def sse_events(resp) -> list[dict]:
    """Read an `Sse<Event>` response (`data: {...}\\n\\n` frames, terminated by a
    literal `data: [DONE]`) into a list of parsed JSON events."""
    events: list[dict] = []
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


# ------------------------------------------------------------------------ demos

def demo_models(base_url: str, api_key: str) -> None:
    listing = request(base_url, api_key, "/v1/models")
    ids = [m["id"] for m in listing.get("data", [])]
    print(f"GET /v1/models -> {len(ids)} model(s): {ids}")


def demo_chat(base_url: str, api_key: str, model: str) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 32}
    out = request(base_url, api_key, "/v1/chat/completions", body)
    text = out["choices"][0]["message"]["content"]
    print(f"POST /v1/chat/completions -> {text!r} (finish_reason={out['choices'][0]['finish_reason']})")

    resp = request(base_url, api_key, "/v1/chat/completions", {**body, "stream": True}, stream=True)
    deltas = [e["choices"][0]["delta"].get("content", "") for e in sse_events(resp) if e.get("choices")]
    print(f"POST /v1/chat/completions (stream) -> {len(deltas)} chunk(s): {''.join(deltas)!r}")


def demo_embeddings(base_url: str, api_key: str, model: str) -> None:
    out = request(base_url, api_key, "/v1/embeddings", {"model": model, "input": ["a photo of a cat", "a photo of a dog"]})
    vecs = [d["embedding"] for d in out["data"]]
    print(f"POST /v1/embeddings -> {len(vecs)} vector(s), dim={len(vecs[0])}, usage={out.get('usage')}")


def demo_images(base_url: str, api_key: str, model: str, out_path: Path) -> None:
    out = request(base_url, api_key, "/v1/images/generations", {"model": model, "prompt": "a red apple on a wooden table", "n": 1, "size": "1024x1024"})
    b64 = out["data"][0]["b64_json"]
    png = base64.b64decode(b64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    print(f"POST /v1/images/generations -> {len(png)} bytes (PNG) -> {out_path}")


# ------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="brain/mock", help="model id (default: brain/mock, weight-free)")
    p.add_argument("--base-url", help="already-running server, e.g. http://127.0.0.1:8788 (skips self-launch)")
    p.add_argument("--api-key", help="Bearer key for --base-url (required if --base-url is set, unless --keys-file is)")
    p.add_argument("--keys-file", help="read the 'openai' key from a --api-keys-out JSON file instead of --api-key")
    p.add_argument("--brain", default=os.environ.get("BRAIN", "./target/release/brain"), help="brain binary (self-launch mode only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8788")), help="port to launch on (self-launch mode only)")
    p.add_argument("--out", type=Path, default=OUT / "openai_client_image.png", help="where to save the generated image")
    args = p.parse_args()

    server: Optional[Server] = None
    if args.base_url:
        base_url = args.base_url
        api_key = args.api_key or (api_key_from_file(args.keys_file) if args.keys_file else None)
        if not api_key:
            p.error("--api-key or --keys-file is required with --base-url")
    else:
        if not os.path.isfile(args.brain) or not os.access(args.brain, os.X_OK):
            skip(f"brain binary not found at {args.brain!r} (build: make release)")
        server = Server(args.brain, args.port, mock=args.model == "brain/mock")
        base_url, api_key = server.base_url, server.api_key
        print(f"launched brain serve --openai on {base_url}  (key: {api_key[:14]}…)")

    try:
        demo_models(base_url, api_key)
        demo_chat(base_url, api_key, args.model)
        demo_embeddings(base_url, api_key, args.model)
        demo_images(base_url, api_key, args.model, args.out)
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
