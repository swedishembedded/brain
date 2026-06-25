# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""``BrainClient`` — an event-driven driver for the ``brain run`` subprocess.

Design
------
``brain run`` reads ONE JSON event per line on stdin and writes events (one per
line) on stdout; stderr carries logs. Every request may carry a top-level
``"req_id"`` which the runtime echoes onto every response event for that request.

``BrainClient`` wraps this as a request/response API safe for multiple in-flight
requests over the single stdio stream:

* A **background reader thread** (:meth:`_reader`) blocks on ``stdout.readline``,
  parses each line as JSON, and routes it by ``req_id`` into a per-request buffer.
* All shared state (the per-``req_id`` buffers + completion flags) lives behind a
  single :class:`threading.Condition` — the *condition-guarded queue* the design
  calls for. The reader appends events and ``notify_all()``s; request methods
  ``wait()`` on the condition until their ``req_id`` is complete (or a timeout).
* One-shot responses (``object_detected``) complete the request immediately.
  Streaming responses (``brain_text_chunk``) accumulate until a chunk with
  ``done: true``.

This decouples the order brain emits events from the order callers ask for them,
so :meth:`detect` and :meth:`chat` can be issued concurrently and demuxed by id.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a declared dependency
    Image = None  # type: ignore


@dataclass
class Detection:
    """One detected object in the input image's own pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    label: str = ""

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _find_brain_binary(explicit: Optional[str]) -> str:
    """Locate the ``brain`` executable.

    Resolution order: explicit arg, ``$BRAIN_BIN``, ``./target/release/brain``,
    ``./target/debug/brain``, then ``brain`` on ``$PATH``.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("BRAIN_BIN")
    if env:
        candidates.append(env)
    candidates += [
        os.path.join("target", "release", "brain"),
        os.path.join("target", "debug", "brain"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    found = shutil.which("brain")
    if found:
        return found
    raise FileNotFoundError(
        "could not locate the `brain` binary; pass brain_bin=..., set $BRAIN_BIN, "
        "or run `cargo build --release` so ./target/release/brain exists "
        f"(tried: {candidates})"
    )


class _Pending:
    """Accumulated state for one in-flight request, keyed by req_id."""

    __slots__ = ("events", "done", "error")

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.done: bool = False
        self.error: Optional[str] = None


class BrainClient:
    """Spawn ``brain run`` and drive it over its JSONL event protocol."""

    def __init__(
        self,
        yolo: Optional[str] = None,
        gpt: Optional[str] = None,
        brain_bin: Optional[str] = None,
        conf: Optional[float] = None,
        extra_args: Optional[list[str]] = None,
        device: str = "cpu",
        ready_timeout: float = 30.0,
    ) -> None:
        self._binary = _find_brain_binary(brain_bin)
        argv = [self._binary, "run"]
        if yolo:
            argv += ["--yolo", yolo]
        if gpt:
            argv += ["--gpt", gpt]
        if conf is not None:
            argv += ["--conf", str(conf)]
        if extra_args:
            argv += list(extra_args)

        env = dict(os.environ)
        env["BRAIN_DEVICE"] = device

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )

        # The condition-guarded queue: all shared state below is protected by it.
        self._cond = threading.Condition()
        self._pending: dict[str, _Pending] = {}
        self._ready = False
        self._closed = False
        self._ids = itertools.count(1)

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

        if not self._wait_ready(ready_timeout):
            self.close()
            raise RuntimeError("brain run did not emit a 'ready' event in time")

    # -- background threads --------------------------------------------------

    def _reader_loop(self) -> None:
        """Read stdout lines, parse JSON, route by req_id under the condition."""
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = evt.get("event")
            req_id = evt.get("req_id")
            with self._cond:
                if ev == "ready":
                    self._ready = True
                    self._cond.notify_all()
                    continue
                if req_id is None:
                    # Lifecycle/log events without correlation (e.g. a bare error).
                    self._cond.notify_all()
                    continue
                p = self._pending.setdefault(req_id, _Pending())
                p.events.append(evt)
                if ev == "object_detected":
                    p.done = True
                elif ev == "error":
                    p.error = evt.get("message", "error")
                    p.done = True
                elif ev == "brain_text_chunk" and evt.get("done"):
                    p.done = True
                self._cond.notify_all()
        # stdout closed -> mark everything done so waiters wake.
        with self._cond:
            self._closed = True
            for p in self._pending.values():
                if not p.done:
                    p.error = p.error or "brain process exited"
                    p.done = True
            self._cond.notify_all()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for _ in self._proc.stderr:
            pass  # logs; swallowed (read so the pipe never blocks brain)

    def _wait_ready(self, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._ready or self._closed, timeout) and self._ready

    # -- request plumbing ----------------------------------------------------

    def _next_id(self) -> str:
        return f"r{next(self._ids)}"

    def _send(self, obj: dict) -> None:
        if self._proc.stdin is None or self._closed:
            raise RuntimeError("brain process stdin is closed")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, req_id: str, timeout: float) -> _Pending:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: req_id in self._pending and self._pending[req_id].done,
                timeout,
            )
            if not ok:
                raise TimeoutError(f"timed out waiting for response to {req_id!r}")
            p = self._pending.pop(req_id)
        if p.error:
            raise RuntimeError(f"brain error for {req_id!r}: {p.error}")
        return p

    # -- public API ----------------------------------------------------------

    def detect(self, image, conf: Optional[float] = None, timeout: float = 120.0,
               req_id: Optional[str] = None) -> list[Detection]:
        """Run object detection on a PIL image; return the detected boxes.

        Encodes ``image`` to raw RGB8, base64s it into a ``camera_frame`` event
        with a fresh ``req_id``, blocks until the matching ``object_detected``
        arrives, and returns the boxes (in the input image's pixel coordinates).
        ``conf`` is informational only here — the threshold is fixed at spawn via
        the ``--conf`` flag (the protocol carries no per-frame threshold).
        """
        if Image is None:
            raise RuntimeError("Pillow is required for detect()")
        rgb = image.convert("RGB")
        w, h = rgb.size
        data = base64.b64encode(rgb.tobytes()).decode("ascii")
        rid = req_id or self._next_id()
        self._send({
            "req_id": rid,
            "event": "camera_frame",
            "format": "rgb8",
            "w": w,
            "h": h,
            "data": data,
        })
        p = self._wait_for(rid, timeout)
        # The terminating event is the object_detected for this req_id.
        det_evt = next((e for e in p.events if e.get("event") == "object_detected"), None)
        if det_evt is None:
            return []
        # Echo check: the runtime must have stamped our req_id back.
        assert det_evt.get("req_id") == rid, "req_id mismatch on response"
        labels = det_evt.get("labels") or []
        out: list[Detection] = []
        for row in det_evt.get("dets", []):
            x1, y1, x2, y2, c, cls = (list(row) + [0] * 6)[:6]
            cls_i = int(cls)
            label = labels[cls_i] if 0 <= cls_i < len(labels) else str(cls_i)
            out.append(Detection(x1, y1, x2, y2, float(c), cls_i, label))
        return out

    def chat(self, text: str, timeout: float = 120.0,
             req_id: Optional[str] = None) -> str:
        """Send user text, collect streamed chunks by req_id, return joined text."""
        rid = req_id or self._next_id()
        self._send({"req_id": rid, "event": "user_text", "text": text})
        p = self._wait_for(rid, timeout)
        parts = []
        for e in p.events:
            if e.get("event") == "brain_text_chunk":
                assert e.get("req_id") == rid, "req_id mismatch on chunk"
                parts.append(e.get("text", ""))
        return "".join(parts)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()  # EOF -> brain's read loop ends
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader.is_alive():
            self._reader.join(timeout=5)

    def __enter__(self) -> "BrainClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
