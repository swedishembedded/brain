# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The transport-agnostic surface shared by every brain client.

brain's serving backend is a **capability model**: every model advertises a
manifest of *actions* (``generate``, ``embed``, ``text2image``, ``transcribe``,
…), each taking typed ``params`` and named binary ``blobs`` and returning an
:class:`Outcome` — scalar ``outputs`` plus named output blobs.

Both transports (:class:`~brain_py.dbus.BrainDBus` over D-Bus, the default, and
:class:`~brain_py.client.BrainStdio` over JSONL-on-stdio) speak that same model,
so the convenience wrappers below — :meth:`~BrainBase.generate`,
:meth:`~BrainBase.chat`, :meth:`~BrainBase.embed`, :meth:`~BrainBase.text2image`
— are defined **once here** in terms of two primitives each transport provides:

* ``manifests() -> list[dict]`` — discovery.
* ``run(model, action, params, *, blobs, meta, timeout) -> Outcome`` — one-shot.
* ``subscribe(model, action, params, *, on_progress, …) -> Outcome`` — streaming
  (progress callbacks during the run, the same final :class:`Outcome` at the end).

This is what lets a caller switch ``transport="dbus"`` ↔ ``"jsonl"`` without
rewriting: the high-level API is identical; only the wire underneath differs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn, Optional

# Progress callbacks receive (step, total, message).
OnProgress = Callable[[int, int, str], None]

#: The exit code every example uses to mean "this environment can't run me" —
#: a missing model, a missing device, no mic, no `claude` on PATH, and so on.
#: This is the GNU automake convention (`man 1 automake` / `AM_TESTS_ENVIRONMENT`)
#: for "skip", deliberately distinct from both success (0) and failure (any other
#: non-zero). Before this, examples used `return 1` or `return 2` for "not
#: served", which reads identically to a real failure to anything driving them —
#: exactly the ambiguity `tests/e2e/examples.bats` needs resolved to tell a
#: legitimately-skipped example apart from a broken one.
EXIT_SKIP = 77


def skip(reason: str) -> NoReturn:
    """Print `SKIP: <reason>` to stderr and exit with :data:`EXIT_SKIP`.

    Call this from an example's `main()` wherever it would otherwise print an
    error and `return 1`/`return 2` for an environment it cannot run in (model
    not served, no GPU, no mic, `claude` not installed, …) — never for an actual
    bug. `tests/e2e/examples.bats` maps exit 77 to a bats `skip`, not a failure.
    """
    print(f"SKIP: {reason}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


class BrainError(RuntimeError):
    """A server-side action/method failure, uniform across transports.

    Both :class:`~brain_py.dbus.BrainDBus` (an action's ``error`` frame, or a
    D-Bus method call that comes back as an error reply) and
    :class:`~brain_py.client.BrainStdio` (a JSONL ``error`` event) raise this
    same type, so calling code can write one ``except BrainError`` regardless
    of ``transport=``.

    ``name`` is set to a stable, checkable string whenever the failure is
    TRANSPORT-level rather than the server validating and rejecting the
    request: the D-Bus error name (e.g. ``org.freedesktop.DBus.Error.Failed``)
    for a D-Bus method-call reply, or ``brain_py.stream_disconnected`` when a
    `Subscribe` stream's socket closed before a terminal `done`/`error` frame
    (the peer died mid-stream). ``None`` means an action `error` frame -- a
    real, isolated action outcome (over either transport, since a capability
    action has no D-Bus-style error name) -- or the JSONL transport in
    general. A caller that needs to tell "brain rejected this" from "brain
    (or the connection to it) is gone" apart -- e.g. to decide whether to
    retry the same request or reconnect first -- should branch on whether
    ``name`` is set, not parse ``message``.
    """

    def __init__(self, message: str, *, name: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.name = name

    def __str__(self) -> str:
        return f"[{self.name}] {self.message}" if self.name else self.message


@dataclass
class Outcome:
    """The result of a capability action, materialised and transport-agnostic.

    * ``outputs`` — the scalar JSON result (token counts, an embedding vector,
      image dims, a transcript, …).
    * ``blobs`` — named binary outputs, already copied into ``bytes`` (over D-Bus
      the fds are read and closed for you; over JSONL the base64 is decoded).
    * ``meta`` — per-blob metadata (``{"media": …, "meta": {"w":…, "h":…}}``).
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    blobs: dict[str, bytes] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.outputs.get(key, default)

    def text(self, name: str = "text") -> str:
        """The generated text: the ``text`` blob if present, else ``outputs['text']``."""
        if name in self.blobs:
            return self.blobs[name].decode("utf-8", "replace")
        return str(self.outputs.get(name, ""))

    def _blob_meta(self, name: str) -> dict[str, Any]:
        m = self.meta.get(name, {})
        if isinstance(m, dict):
            inner = m.get("meta")
            if isinstance(inner, dict):
                return inner
            return m
        return {}

    def image(self, name: str = "image"):
        """Decode an ``image`` output blob (raw HWC-f32 in ``[0,1]``) to a PIL image."""
        from .image import to_pil

        raw = self.blobs[name]
        m = self._blob_meta(name)
        w = int(m.get("w", 0))
        h = int(m.get("h", 0))
        c = int(m.get("c", 3))
        if not (w and h):
            raise ValueError(f"blob {name!r} carries no w/h in its meta: {self.meta.get(name)!r}")
        return to_pil(raw, w, h, c)


def _flatten_messages(messages: Any) -> str:
    """Serialise a chat ``messages`` list to the JSON-array string the runtime reads."""
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    return json.dumps(messages)


class BrainBase:
    """Convenience capability wrappers shared by every transport.

    Subclasses MUST provide :meth:`manifests`, :meth:`run` and :meth:`subscribe`
    (see the module docstring for the contract). Everything below is expressed in
    terms of those, so it behaves identically over D-Bus and over JSONL.
    """

    # -- to be implemented by each transport ---------------------------------
    def manifests(self) -> list[dict]:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, model: str, action: str, params: Optional[dict] = None, *,
            blobs: Optional[dict] = None, meta: Optional[dict] = None,
            timeout: float = 1800.0) -> Outcome:  # pragma: no cover - abstract
        raise NotImplementedError

    def subscribe(self, model: str, action: str, params: Optional[dict] = None, *,
                  blobs: Optional[dict] = None, meta: Optional[dict] = None,
                  on_progress: Optional[OnProgress] = None,
                  on_job: Optional[Callable[[int], None]] = None,
                  timeout: float = 1800.0) -> Outcome:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- discovery -----------------------------------------------------------
    def models(self) -> list[str]:
        """Every served model name (sorted), from the manifests."""
        return sorted(m["model"] for m in self.manifests())

    def actions(self, model: str) -> list[str]:
        """The action names a model advertises."""
        for m in self.manifests():
            if m.get("model") == model:
                return [a.get("name") for a in m.get("actions", [])]
        raise LookupError(f"no such model: {model!r}")

    def model_for(self, action: str) -> str:
        """The first served model that advertises ``action`` (capability lookup)."""
        for m in self.manifests():
            if any(a.get("name") == action for a in m.get("actions", [])):
                return m["model"]
        raise LookupError(f"no served model advertises action {action!r}")

    def _pick(self, action: str, model: Optional[str]) -> str:
        return model if model else self.model_for(action)

    # -- text generation -----------------------------------------------------
    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[list] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_new: Optional[int] = None,
        blobs: Optional[dict] = None,
        meta: Optional[dict] = None,
        on_progress: Optional[OnProgress] = None,
        on_job: Optional[Callable[[int], None]] = None,
        timeout: float = 300.0,
        **params: Any,
    ) -> str:
        """Run the ``generate`` action and return the generated text.

        Pass ``prompt=`` for a raw completion or ``messages=[{"role":…}, …]`` for
        chat. ``blobs``/``meta`` pass straight through to :meth:`run`/
        :meth:`subscribe` — generic, model-agnostic multimodal input (e.g.
        ``blobs={"audio": pcm_bytes}, meta={"audio": {"media": "audio"}}`` for a
        model whose ``generate`` action declares an ``audio``/``image`` blob
        input, like ``brain/qwen3omnimoe``). With ``on_progress`` or ``on_job``
        set the run streams via :meth:`subscribe` (one callback per token, and/or
        the job id a concurrent caller needs to :meth:`~BrainDBus.cancel` it --
        `run`'s job is never client-visible, so cancellation always needs this
        path); otherwise it is a single :meth:`run`. ``model`` defaults to the
        first model advertising ``generate``.
        """
        model = self._pick("generate", model)
        p: dict[str, Any] = dict(params)
        if messages is not None:
            p["messages"] = _flatten_messages(messages)
        if prompt is not None:
            p["prompt"] = prompt
        if system is not None:
            p["system"] = system
        if max_new is not None:
            p["max_new"] = max_new
        if on_progress is not None or on_job is not None:
            out = self.subscribe(model, "generate", p, blobs=blobs, meta=meta,
                                 on_progress=on_progress, on_job=on_job, timeout=timeout)
        else:
            out = self.run(model, "generate", p, blobs=blobs, meta=meta, timeout=timeout)
        return out.text()

    def chat(self, text: str, *, model: Optional[str] = None,
             blobs: Optional[dict] = None, meta: Optional[dict] = None,
             on_progress: Optional[OnProgress] = None,
             on_job: Optional[Callable[[int], None]] = None,
             timeout: float = 300.0,
             **params: Any) -> str:
        """Sugar over :meth:`generate` for a single user turn."""
        return self.generate(messages=[{"role": "user", "content": text}],
                             model=model, blobs=blobs, meta=meta,
                             on_progress=on_progress, on_job=on_job, timeout=timeout, **params)

    # -- embeddings ----------------------------------------------------------
    def embed(self, text: str, *, model: Optional[str] = None,
              timeout: float = 120.0, **params: Any) -> list[float]:
        """Run the ``embed`` action and return the pooled embedding vector."""
        model = self._pick("embed", model)
        out = self.run(model, "embed", {"text": text, **params}, timeout=timeout)
        vec = out.outputs.get("mean")
        if vec is None:
            vec = out.outputs.get("embedding")
        if vec is None:
            raise RuntimeError(f"embed produced no vector (outputs: {list(out.outputs)})")
        return list(vec)

    # -- image generation ----------------------------------------------------
    def text2image(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        model: Optional[str] = None,
        on_progress: Optional[OnProgress] = None,
        on_job: Optional[Callable[[int], None]] = None,
        timeout: float = 1800.0,
        **params: Any,
    ):
        """Run ``text2image`` and return the generated image as a PIL ``Image``.

        Extra keyword args (``steps``, ``seed``, ``precision``, …) pass straight
        through as action params. With ``on_progress`` or ``on_job`` set the
        denoise steps stream via :meth:`subscribe` -- the same run/subscribe
        cancellation tradeoff as :meth:`generate` applies here.
        """
        model = self._pick("text2image", model)
        p: dict[str, Any] = {"prompt": prompt, "width": width, "height": height, **params}
        if on_progress is not None or on_job is not None:
            out = self.subscribe(model, "text2image", p, on_progress=on_progress, on_job=on_job, timeout=timeout)
        else:
            out = self.run(model, "text2image", p, timeout=timeout)
        return out.image("image")

    # image() is the natural verb for "give me the picture".
    image = text2image
