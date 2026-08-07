# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A thin `urllib`-based client for brain's OpenAI-compatible HTTP surface.

`BrainBase`'s `generate`/`chat` (and everything built on `run`/`subscribe`)
work identically over this transport as over `BrainDBus`/`BrainStdio` — only
`chat`-shaped actions are supported (this transport talks to
`/v1/chat/completions`, which has no generic "run any action" concept the
way D-Bus does; `embed`/`text2image`/anything else raises `NotImplementedError`
rather than silently doing the wrong thing).

brain's OpenAI base URL is `http://127.0.0.1:8788` (or with an explicit
`/v1` suffix — both are accepted, see `_normalize_base_url`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from .base import BrainBase, BrainError, OnProgress, Outcome


def _normalize_base_url(url: str) -> str:
    """Accept a bare host:port OR an explicit `.../v1` and always return the
    latter -- so `--openai localhost:8788` and `--openai
    http://localhost:8788/v1` both work, per `examples/omni.py`'s own
    "URLs missing a scheme or /v1 are normalized" contract."""
    if "://" not in url:
        url = f"http://{url}"
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


@dataclass
class BrainOpenAI(BrainBase):
    """Talk to `brain serve --openai <port>` over its OpenAI-compatible
    `/v1/chat/completions` endpoint. `model` defaults to whatever
    `/v1/models` lists first when unset at call time."""

    base_url: str
    api_key: Optional[str] = None
    timeout_connect: float = 10.0

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_json(self, path: str, body: dict[str, Any], *, timeout: float) -> Any:
        """POST and return the fully-read, parsed JSON body. The read happens
        INSIDE the `with` block -- returning the response object itself and
        reading it after the block exits would read from an already-closed
        connection (a real bug this fixed: `with urlopen(...) as resp: return
        resp` closes `resp` the instant the `with` block exits, which happens
        before the caller ever gets to call `.read()`)."""
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(body).encode("utf-8"), headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise BrainError(f"openai {path}: HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise BrainError(f"openai {path}: {e.reason}") from e

    def _post_stream(self, path: str, body: dict[str, Any], *, timeout: float):
        """POST and return the LIVE response object for the caller to iterate
        (SSE) -- deliberately not wrapped in a `with` here, since the whole
        point is to keep reading after this call returns; the caller's own
        iteration loop is where the connection actually gets drained/closed."""
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(body).encode("utf-8"), headers=self._headers(), method="POST"
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise BrainError(f"openai {path}: HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise BrainError(f"openai {path}: {e.reason}") from e

    def manifests(self) -> list[dict[str, Any]]:
        """Synthesize a `generate`-only manifest per listed model — `/v1/models`
        has no action list, only names, so this is enough for `model_for`/
        `chat`/`generate` to work, not a full capability mirror."""
        req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_connect) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise BrainError(f"openai /models: HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
        except urllib.error.URLError as e:
            raise BrainError(f"openai /models: {e.reason}") from e
        return [
            {"model": m["id"], "actions": [{"name": "generate", "streaming": True}]}
            for m in data.get("data", [])
        ]

    def run(
        self,
        model: str,
        action: str,
        params: Optional[dict] = None,
        *,
        blobs: Optional[dict] = None,
        meta: Optional[dict] = None,
        timeout: float = 300.0,
    ) -> Outcome:
        if action != "generate":
            raise NotImplementedError(f"BrainOpenAI: action {action!r} not supported (only 'generate')")
        p = params or {}
        messages = p.get("messages")
        if messages:
            msgs = json.loads(messages) if isinstance(messages, str) else messages
        else:
            msgs = [{"role": "user", "content": p.get("prompt", "")}]
        if p.get("system"):
            msgs = [{"role": "system", "content": p["system"]}, *msgs]
        body: dict[str, Any] = {"model": model, "messages": msgs, "stream": False}
        if p.get("max_new") is not None:
            body["max_tokens"] = p["max_new"]
        for k, dst in (("temp", "temperature"), ("top_p", "top_p"), ("seed", "seed"), ("stop", "stop")):
            if p.get(k) is not None:
                body[dst] = p[k]
        data = self._post_json("/chat/completions", body, timeout=timeout)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return Outcome(outputs={"text": text, **usage}, blobs={"text": text.encode("utf-8")}, meta={})

    def subscribe(
        self,
        model: str,
        action: str,
        params: Optional[dict] = None,
        *,
        blobs: Optional[dict] = None,
        meta: Optional[dict] = None,
        on_progress: Optional[OnProgress] = None,
        timeout: float = 300.0,
    ) -> Outcome:
        if action != "generate":
            raise NotImplementedError(f"BrainOpenAI: action {action!r} not supported (only 'generate')")
        p = params or {}
        messages = p.get("messages")
        if messages:
            msgs = json.loads(messages) if isinstance(messages, str) else messages
        else:
            msgs = [{"role": "user", "content": p.get("prompt", "")}]
        if p.get("system"):
            msgs = [{"role": "system", "content": p["system"]}, *msgs]
        body: dict[str, Any] = {"model": model, "messages": msgs, "stream": True}
        if p.get("max_new") is not None:
            body["max_tokens"] = p["max_new"]
        resp = self._post_stream("/chat/completions", body, timeout=timeout)
        text_parts: list[str] = []
        step = 0
        try:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    text_parts.append(delta)
                    step += 1
                    if on_progress:
                        on_progress(step, 0, delta)
        finally:
            resp.close()
        text = "".join(text_parts)
        return Outcome(outputs={"text": text}, blobs={"text": text.encode("utf-8")}, meta={})
