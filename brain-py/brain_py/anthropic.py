# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A thin `urllib`-based client for brain's Anthropic-compatible HTTP surface
(`/v1/messages`). Same shape and same scope limits as `brain_py.openai`'s
`BrainOpenAI` — only the `generate` action, everything else raises
`NotImplementedError`. See that module's doc for why (no generic "run any
action" concept over a chat-specific REST API).

brain's Anthropic base URL is `http://127.0.0.1:8787`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from .base import BrainBase, BrainError, OnProgress, Outcome

#: Anthropic requires `max_tokens`; brain's own default for other transports
#: (`generate_spec()`'s `max_new` default) is 32, reused here so an
#: unspecified call behaves the same across every transport.
DEFAULT_MAX_TOKENS = 32


def _normalize_base_url(url: str) -> str:
    """Accept a bare host:port OR an explicit scheme; unlike OpenAI's surface,
    brain's `/v1/messages` route has no `/v1` prefix requirement beyond what
    the caller already supplies as a full URL -- normalize just the scheme."""
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


@dataclass
class BrainAnthropic(BrainBase):
    """Talk to `brain serve --anthropic <port>` over its Anthropic-compatible
    `/v1/messages` endpoint."""

    base_url: str
    api_key: Optional[str] = None
    timeout_connect: float = 10.0

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _post(self, path: str, body: dict[str, Any], *, timeout: float):
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(body).encode("utf-8"), headers=self._headers(), method="POST"
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise BrainError(f"anthropic {path}: HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise BrainError(f"anthropic {path}: {e.reason}") from e

    def manifests(self) -> list[dict[str, Any]]:
        """`/v1/messages` has no model-listing route of its own; brain has no
        Anthropic-side `/v1/models` equivalent, so this transport cannot do
        discovery -- callers must pass `model=` explicitly to `chat`/`generate`."""
        raise NotImplementedError("BrainAnthropic has no model-listing endpoint; pass model= explicitly")

    def _build_messages(self, params: dict) -> tuple[list[dict], Optional[str]]:
        messages = params.get("messages")
        msgs = json.loads(messages) if isinstance(messages, str) else (messages or [{"role": "user", "content": params.get("prompt", "")}])
        system = params.get("system")
        # Anthropic keeps system as a top-level field, not a messages[0] entry
        # -- promote any embedded system-role message's content if the caller
        # didn't already pass system= separately (base.py's generate(system=...)
        # does; a raw messages= list built by hand might not).
        system_msgs = [m.get("content", "") for m in msgs if m.get("role") == "system"]
        if system is None and system_msgs:
            system = "\n".join(system_msgs)
        msgs = [m for m in msgs if m.get("role") != "system"] or msgs
        return msgs, system

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
            raise NotImplementedError(f"BrainAnthropic: action {action!r} not supported (only 'generate')")
        p = params or {}
        msgs, system = self._build_messages(p)
        body: dict[str, Any] = {"model": model, "messages": msgs, "max_tokens": p.get("max_new", DEFAULT_MAX_TOKENS), "stream": False}
        if system:
            body["system"] = system
        resp = self._post("/v1/messages", body, timeout=timeout)
        try:
            data = json.loads(resp.read())
        finally:
            resp.close()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return Outcome(outputs={"text": text, "usage": data.get("usage", {})}, blobs={"text": text.encode("utf-8")}, meta={})

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
            raise NotImplementedError(f"BrainAnthropic: action {action!r} not supported (only 'generate')")
        p = params or {}
        msgs, system = self._build_messages(p)
        body: dict[str, Any] = {"model": model, "messages": msgs, "max_tokens": p.get("max_new", DEFAULT_MAX_TOKENS), "stream": True}
        if system:
            body["system"] = system
        resp = self._post("/v1/messages", body, timeout=timeout)
        text_parts: list[str] = []
        step = 0
        event = None
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[len("data:") :].strip())
                if event == "content_block_delta":
                    delta = payload.get("delta", {}).get("text", "")
                    if delta:
                        text_parts.append(delta)
                        step += 1
                        if on_progress:
                            on_progress(step, 0, delta)
                elif event == "message_stop":
                    break
        finally:
            resp.close()
        text = "".join(text_parts)
        return Outcome(outputs={"text": text}, blobs={"text": text.encode("utf-8")}, meta={})
