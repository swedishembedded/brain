# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""``BrainStdio`` — an event-driven driver for the ``brain run`` subprocess.

This is the **JSONL-on-stdio** transport (select it with ``Brain(transport=
"jsonl")`` or by constructing :class:`BrainStdio` directly). The default
transport is D-Bus (:class:`~brain_py.dbus.BrainDBus`); both speak the same
capability model, so the high-level API (:meth:`run` / :meth:`subscribe` and the
:class:`~brain_py.base.BrainBase` wrappers ``generate`` / ``embed`` /
``text2image``) is identical — only the wire underneath differs. On top of that,
``BrainStdio`` keeps the ``brain run`` legacy verbs :meth:`detect`,
:meth:`converse`, :meth:`forecast` and :meth:`backtest`.

Design
------
``brain run`` reads ONE JSON event per line on stdin and writes events (one per
line) on stdout; stderr carries logs. Every request may carry a top-level
``"req_id"`` which the runtime echoes onto every response event for that request.

``BrainStdio`` wraps this as a request/response API safe for multiple in-flight
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
so :meth:`detect` and :meth:`converse` can be issued concurrently and demuxed by id.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import shutil
import socket as _socket
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .base import BrainBase, OnProgress, Outcome
from .forecast import Forecast, Panel

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a declared dependency
    Image = None  # type: ignore

# Events that terminate a request on their own (one-shot results). Any streaming
# chunk instead terminates on a ``done: true`` field. Generalising this registry
# is what keeps the reader loop protocol-agnostic as new events are added.
_TERMINAL_EVENTS = frozenset({
    "object_detected",
    "forecast_result",
    "backtest_result",
    "capabilities_result",
    # Generic capability interface (Z-Image &c.): one result per request.
    "action_result",
    "manifest_result",
    # A cancelled streaming turn ends with a bare ``cancelled`` ack (no ``done``
    # chunk), so it completes the request just like a one-shot result.
    "cancelled",
})


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


def _wire_blobs(blobs: Optional[dict], meta: Optional[dict]) -> list[dict]:
    """Turn ``{name: bytes}`` (+ optional per-name meta) into JSONL WireBlobs."""
    meta = meta or {}
    out = []
    for name, data in (blobs or {}).items():
        m = meta.get(name, {})
        out.append({
            "name": name,
            "media": m.get("media", "bytes"),
            "b64": base64.b64encode(data).decode("ascii"),
            "meta": m.get("meta", {}),
        })
    return out


class BrainStdio(BrainBase):
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
        forecast: bool = False,
    ) -> None:
        """Spawn a brain server subprocess and drive it over JSONL.

        With ``forecast=True`` the client launches ``brain forecast serve`` (the
        statistical baselines registered, foundation models added as they land)
        so :meth:`forecast`, :meth:`backtest` and :meth:`capabilities` work.
        Otherwise it launches ``brain run`` for :meth:`converse` / :meth:`detect`.
        """
        self._binary = _find_brain_binary(brain_bin)
        if forecast:
            argv = [self._binary, "forecast", "serve"]
        else:
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
        self._sock = None
        self._rfile = self._proc.stdout
        self._wfile = self._proc.stdin
        self._boot(ready_timeout, drain_stderr=True)

    @classmethod
    def connect(
        cls,
        socket_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ready_timeout: float = 30.0,
    ) -> "BrainStdio":
        """Connect to an ALREADY-RUNNING ``brain forecast serve`` over a socket.

        Unlike the constructor (which spawns its own stdio subprocess), this
        attaches to a long-lived server that keeps the models warm across many
        requests — the right shape for a batch job that forecasts a whole
        universe. Pass either ``socket_path`` (a Unix socket, matching
        ``brain forecast serve --socket <path>``) or ``host``+``port`` (matching
        ``--listen <host:port>``).

        The same :meth:`forecast` / :meth:`capabilities` / :meth:`backtest` API
        works over the socket exactly as over stdio.
        """
        self = cls.__new__(cls)
        self._binary = None
        self._proc = None
        if socket_path is not None:
            sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            sock.connect(socket_path)
        elif host is not None and port is not None:
            sock = _socket.create_connection((host, port))
        else:
            raise ValueError("connect() needs socket_path=... or host=...+port=...")
        self._sock = sock
        # Text line streams over the socket (mirrors the stdio pipes).
        self._rfile = sock.makefile("r", encoding="utf-8", newline="\n")
        self._wfile = sock.makefile("w", encoding="utf-8", newline="\n")
        self._boot(ready_timeout, drain_stderr=False)
        return self

    def _boot(self, ready_timeout: float, drain_stderr: bool) -> None:
        """Shared startup: the condition-guarded queue + reader thread + wait for
        the ``ready`` greeting (emitted per connection, over stdio and socket)."""
        # The condition-guarded queue: all shared state below is protected by it.
        self._cond = threading.Condition()
        self._pending: dict[str, _Pending] = {}
        self._ready = False
        self._closed = False
        self._ids = itertools.count(1)
        # Serialize writes so concurrent requests can't interleave JSON lines.
        self._write_lock = threading.Lock()

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        if drain_stderr and self._proc is not None:
            self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._err_thread.start()

        if not self._wait_ready(ready_timeout):
            self.close()
            raise RuntimeError("brain did not emit a 'ready' event in time")

    # -- background threads --------------------------------------------------

    def _reader_loop(self) -> None:
        """Read result lines, parse JSON, route by req_id under the condition.
        Works over the subprocess stdout pipe and the socket read stream alike."""
        assert self._rfile is not None
        for line in self._rfile:
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
                # Generalised completion: an error, a one-shot terminal event, or
                # any streaming chunk carrying done:true ends the request.
                if ev == "error":
                    p.error = evt.get("message") or evt.get("code") or "error"
                    p.done = True
                elif ev in _TERMINAL_EVENTS:
                    p.done = True
                elif evt.get("done") is True:
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
        line = json.dumps(obj) + "\n"
        # A single lock serializes the write+flush so two threads issuing
        # requests concurrently can't interleave bytes on the stream.
        with self._write_lock:
            if self._wfile is None or self._closed:
                raise RuntimeError("brain connection is closed")
            self._wfile.write(line)
            self._wfile.flush()

    def _wait_for(self, req_id: str, timeout: float) -> _Pending:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: req_id in self._pending and self._pending[req_id].done,
                timeout,
            )
            if not ok:
                # Don't leak the pending entry on timeout.
                self._pending.pop(req_id, None)
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

    # -- generic capability API (the transport-agnostic model) ----------------

    def manifests(self, timeout: float = 30.0) -> list[dict]:
        """Every served model's manifest, as a list (matches the D-Bus transport),
        via a ``manifest_request``. Discovery for :meth:`run` / :meth:`subscribe`
        and the :class:`~brain_py.base.BrainBase` capability wrappers."""
        rid = self._next_id()
        self._send({"req_id": rid, "event": "manifest_request"})
        p = self._wait_for(rid, timeout)
        evt = next((e for e in p.events if e.get("event") == "manifest_result"), None)
        return list((evt or {}).get("manifests", []) or [])

    def action(self, model: str, action: str, params: Optional[dict] = None,
               blobs: Optional[list] = None, timeout: float = 1800.0,
               req_id: Optional[str] = None) -> dict:
        """Low-level ``action_request`` → the raw ``action_result`` event
        (``{"outputs": {...}, "blobs": [WireBlob, ...]}``). Raises on ``error``.
        Prefer :meth:`run`, which returns a materialised :class:`Outcome`."""
        rid = req_id or self._next_id()
        self._send({
            "req_id": rid,
            "event": "action_request",
            "model": model,
            "action": action,
            "params": params or {},
            "blobs": blobs or [],
        })
        p = self._wait_for(rid, timeout)
        if p.error:
            raise RuntimeError(f"{model}.{action} failed: {p.error}")
        res = next((e for e in p.events if e.get("event") == "action_result"), None)
        if res is None:
            raise RuntimeError(f"{model}.{action}: no action_result (events: {[e.get('event') for e in p.events]})")
        return res

    def run(self, model: str, action: str, params: Optional[dict] = None, *,
            blobs: Optional[dict] = None, meta: Optional[dict] = None,
            timeout: float = 1800.0) -> Outcome:
        """Run one action and return a materialised :class:`Outcome` — the same
        shape as the D-Bus transport. ``blobs`` maps input names to raw ``bytes``
        (base64'd into the request); output blobs are decoded into ``bytes``."""
        res = self.action(model, action, params, blobs=_wire_blobs(blobs, meta), timeout=timeout)
        out_blobs: dict[str, bytes] = {}
        out_meta: dict[str, Any] = {}
        for b in res.get("blobs", []) or []:
            name = b.get("name", "blob")
            out_blobs[name] = base64.b64decode(b.get("b64", "") or "")
            out_meta[name] = {"media": b.get("media"), "meta": b.get("meta")}
        return Outcome(outputs=res.get("outputs", {}) or {}, blobs=out_blobs, meta=out_meta)

    def subscribe(self, model: str, action: str, params: Optional[dict] = None, *,
                  blobs: Optional[dict] = None, meta: Optional[dict] = None,
                  on_progress: Optional[OnProgress] = None,
                  timeout: float = 1800.0) -> Outcome:
        """Streaming counterpart to :meth:`run`. ``brain run``'s generic action
        interface delivers a single ``action_result`` (no intermediate frames), so
        this runs the action and, if ``on_progress`` is given, ticks it once on
        completion. For token/step streaming use the D-Bus transport."""
        out = self.run(model, action, params, blobs=blobs, meta=meta, timeout=timeout)
        if on_progress is not None:
            on_progress(1, 1, "done")
        return out

    # -- legacy `brain run` conversational path -------------------------------

    def converse(self, text: str, timeout: float = 120.0,
                 req_id: Optional[str] = None) -> str:
        """Send ``user_text`` to ``brain run`` and return the streamed reply.

        This is the ``brain run`` conversational path (``user_text`` →
        ``brain_text_chunk`` stream), distinct from the capability
        :meth:`~brain_py.base.BrainBase.generate` (the ``generate`` action). Use
        :meth:`~brain_py.base.BrainBase.chat` / ``generate`` for the portable API."""
        rid = req_id or self._next_id()
        self._send({"req_id": rid, "event": "user_text", "text": text})
        p = self._wait_for(rid, timeout)
        parts = []
        for e in p.events:
            if e.get("event") == "brain_text_chunk":
                assert e.get("req_id") == rid, "req_id mismatch on chunk"
                parts.append(e.get("text", ""))
        return "".join(parts)

    # -- forecasting API -----------------------------------------------------

    def capabilities(self, timeout: float = 30.0) -> dict:
        """Return ``{model_name: capabilities_dict}`` for every registered
        forecasting model, so an app can discover constraints (max context,
        covariate support, native representation) instead of hard-coding them.
        """
        rid = self._next_id()
        self._send({"req_id": rid, "event": "capabilities_request"})
        p = self._wait_for(rid, timeout)
        evt = next((e for e in p.events if e.get("event") == "capabilities_result"), None)
        models = (evt or {}).get("models", [])
        return {m["name"]: m for m in models}

    def forecast(self, panel: Panel, horizon: int,
                 quantiles: Optional[list] = None, num_samples: int = 0,
                 representations: Optional[list] = None, model: str = "naive",
                 seed: int = 0, timeout: float = 120.0,
                 req_id: Optional[str] = None) -> Forecast:
        """Forecast ``panel`` over ``horizon`` steps with the named ``model``.

        ``quantiles`` selects the quantile levels (default 10/50/90); pass
        ``num_samples > 0`` to also draw sample trajectories. Returns a
        :class:`Forecast` whose ``.median()`` / ``.interval()`` / ``.samples()``
        and ``.derived`` expose the result honestly (derived fields flagged).
        """
        levels = quantiles or [0.1, 0.5, 0.9]
        reps = representations or (["quantiles", "point"] + (["samples"] if num_samples else []))
        rid = req_id or self._next_id()
        self._send({
            "req_id": rid,
            "event": "forecast_request",
            "model": model,
            "panel": panel.to_wire(),
            # the runtime reads the whole spec (horizon, reps, levels, samples,
            # seed) from the "output" object.
            "output": {
                "horizon": horizon,
                "representations": reps,
                "quantile_levels": levels,
                "num_samples": num_samples,
                "seed": seed,
            },
        })
        p = self._wait_for(rid, timeout)
        evt = next((e for e in p.events if e.get("event") == "forecast_result"), None)
        if evt is None:
            raise RuntimeError(f"no forecast_result for {rid!r}")
        assert evt.get("req_id") == rid, "req_id mismatch on forecast_result"
        return Forecast(evt)

    def backtest(self, panel: Panel, horizon: int, models: list,
                 origins: int = 30, stride: int = 1,
                 metrics: Optional[list] = None, seed: int = 0,
                 timeout: float = 600.0, req_id: Optional[str] = None) -> dict:
        """Run a server-side rolling-origin backtest of ``models`` over ``panel``.

        Returns ``{model: {metric: value}}`` aggregated over origins. Always
        include ``"naive"`` so results are read relative to the baseline.
        """
        rid = req_id or self._next_id()
        self._send({
            "req_id": rid,
            "event": "backtest_request",
            "panel": panel.to_wire(),
            "spec": {
                "models": list(models),
                "horizon": horizon,
                "origins": origins,
                "stride": stride,
                "metrics": metrics or ["mase", "wql", "coverage", "directional"],
                "seed": seed,
            },
        })
        p = self._wait_for(rid, timeout)
        evt = next((e for e in p.events if e.get("event") == "backtest_result"), None)
        rows = ((evt or {}).get("report") or {}).get("rows", [])
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["model"], {})[r["metric"]] = r["value"]
        return out

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        # Idempotent; handles both the subprocess and socket transports.
        with getattr(self, "_cond", threading.Condition()):
            self._closed = True
        try:
            if self._wfile is not None:
                self._wfile.close()  # EOF -> brain's read loop ends this connection
        except (OSError, ValueError):
            pass
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        reader = getattr(self, "_reader", None)
        if reader is not None and reader.is_alive():
            reader.join(timeout=5)

    def __enter__(self) -> "BrainStdio":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Backwards-friendly alias: the JSONL client used to be the primary entry point.
BrainClient = BrainStdio
