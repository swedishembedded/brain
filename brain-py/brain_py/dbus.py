# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""brain's D-Bus client — the **default** transport.

`com.swedishembedded.Brain1` is the local control surface `brain serve --dbus`
exposes. Bulk data (images, audio, results) crosses as Unix **file descriptors**
(sealed memfds / a SEQPACKET event socket), so this wraps `jeepney` with fd
passing enabled.

:class:`BrainDBus` is a full :class:`~brain_py.base.BrainBase`: the high-level
capability API (:meth:`run`, :meth:`subscribe`, and the
:meth:`~brain_py.base.BrainBase.generate` / ``embed`` / ``text2image`` wrappers)
returns materialised :class:`~brain_py.base.Outcome`\\ s — the fds are read and
closed for you. The low-level fd-returning primitives (:meth:`run_fds`,
:meth:`stream_frames`, :meth:`stream_transcribe`) stay available for zero-copy or
live-audio use.

    from brain_py import Brain            # D-Bus by default
    with Brain() as brain:
        print(brain.models())
        print(brain.generate(prompt="hello", model="mock"))
        img = brain.text2image("a red cube", model="mock")
"""
from __future__ import annotations

import fcntl
import json
import mmap
import os
import socket
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

from jeepney import DBusAddress, Properties, new_method_call
from jeepney.fds import FileDescriptor
from jeepney.io.blocking import open_dbus_connection
from jeepney.wrappers import DBusErrorResponse, unwrap_msg

from .base import BrainBase, BrainError, Outcome, OnProgress

BUS_NAME = "com.swedishembedded.Brain1"
OBJECT_PATH = "/com/swedishembedded/Brain1"
INTERFACE = f"{BUS_NAME}.Manager"

_ADDR = DBusAddress(OBJECT_PATH, bus_name=BUS_NAME, interface=INTERFACE)
_F_ADD_SEALS = 1033
_SEALS = 0x1 | 0x2 | 0x4  # SHRINK | GROW | WRITE

Fds = dict[str, FileDescriptor]


@dataclass
class RunResult:
    """The raw outcome of a low-level :meth:`BrainDBus.run_fds`: scalar result,
    still-open output fds, and their metadata. Consume each fd with :func:`read_fd`."""

    result: dict[str, Any]
    fds: Fds
    meta: dict[str, Any]


class BrainDBus(BrainBase):
    """Blocking client for the `Manager` interface (one bus connection)."""

    def __init__(self, bus: str = "SESSION") -> None:
        self._conn = open_dbus_connection(bus=bus, enable_fds=True)

    def __enter__(self) -> "BrainDBus":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- discovery -----------------------------------------------------------
    def manifests(self) -> list[dict[str, Any]]:
        return json.loads(self._call("Manifests")[0])

    def stats(self) -> dict[str, int]:
        """Scheduler counters: builds, evictions, batches, max_batch, resident, …."""
        return json.loads(self._call("Stats")[0])

    def stats_snapshot(self) -> dict[str, Any]:
        """The full self-describing stats document (what braintop renders)."""
        return json.loads(self._call("StatsSnapshot")[0])

    def version(self) -> str:
        """The server's version string (D-Bus property)."""
        return self._get_prop("Version")

    def active_jobs(self) -> int:
        """Jobs currently in flight on the server (D-Bus property)."""
        return int(self._get_prop("ActiveJobs"))

    # -- high-level capability API (returns materialised Outcomes) -----------
    def run(
        self,
        model: str,
        action: str,
        params: Optional[dict] = None,
        *,
        blobs: Optional[dict] = None,
        meta: Optional[dict] = None,
        timeout: float = 1800.0,
    ) -> Outcome:
        """Run one action and return a materialised :class:`Outcome`.

        ``blobs`` maps input names to raw ``bytes`` (sealed into memfds for you);
        ``meta`` optionally maps names to ``{"media": …, "meta": {…}}``. Output fds
        are read into ``bytes`` and closed. ``timeout`` is accepted for API parity
        (the blocking D-Bus reply governs the real wait).
        """
        rr = self.run_fds(model, action, params, in_fds=self._in_fds(blobs), in_meta=self._in_meta(blobs, meta))
        out_blobs = {name: read_fd(fd) for name, fd in rr.fds.items()}
        return Outcome(outputs=rr.result, blobs=out_blobs, meta=rr.meta)

    def subscribe(
        self,
        model: str,
        action: str,
        params: Optional[dict] = None,
        *,
        blobs: Optional[dict] = None,
        meta: Optional[dict] = None,
        on_progress: Optional[OnProgress] = None,
        timeout: float = 1800.0,
    ) -> Outcome:
        """Run a streaming action, invoking ``on_progress(step, total, message)`` as
        it advances, and return the final :class:`Outcome`. Raises on an error frame."""
        _job, frames = self.stream_frames_with_job(
            model, action, params, in_fds=self._in_fds(blobs), in_meta=self._in_meta(blobs, meta), timeout=timeout
        )
        outputs: dict[str, Any] = {}
        out_blobs: dict[str, bytes] = {}
        out_meta: dict[str, Any] = {}
        for frame, raw_fds in frames:
            kind = frame.get("type")
            if kind == "progress":
                if on_progress is not None:
                    on_progress(int(frame.get("step", 0)), int(frame.get("total", 0)), frame.get("message", ""))
            elif kind == "blob":
                name = frame.get("name", "blob")
                out_meta[name] = {"media": frame.get("media"), "meta": frame.get("meta")}
                out_blobs[name] = read_fd(raw_fds[0]) if raw_fds else b""
            elif kind == "done":
                outputs = frame.get("result") or {}
            elif kind == "error":
                raise BrainError(f"{model}.{action} failed: {frame.get('message')}")
        return Outcome(outputs=outputs, blobs=out_blobs, meta=out_meta)

    # -- audio: live streaming transcription ---------------------------------
    def transcribe(
        self,
        audio: bytes,
        *,
        model: Optional[str] = None,
        window_ms: int = 1000,
        on_segment: Optional[Any] = None,
        timeout: float = 3600.0,
        **params: Any,
    ) -> str:
        """Transcribe raw mono **f32-LE 16 kHz** PCM ``audio`` and return the transcript.

        Feeds the bytes through a pipe into :meth:`stream_transcribe` on a writer
        thread, consumes the ``segment`` frames (calling ``on_segment(text, final)``
        if given), and returns the full transcript from the terminal ``done`` frame.
        ``model`` defaults to the first model advertising ``transcribe``.
        """
        import threading

        model = model or self.model_for("transcribe")
        p = {"window_ms": window_ms, **params}
        rfd, wfd = os.pipe()

        def _feed() -> None:
            try:
                os.write(wfd, audio)
            finally:
                os.close(wfd)

        writer = threading.Thread(target=_feed, daemon=True)
        writer.start()
        try:
            _job, frames = self.stream_transcribe(model, rfd, p, timeout=timeout)
        finally:
            os.close(rfd)  # the server dup'd its own copy
        transcript = ""
        for frame, _fds in frames:
            kind = frame.get("type")
            if kind == "segment":
                if on_segment is not None:
                    on_segment(frame.get("text", ""), bool(frame.get("final")))
            elif kind == "done":
                transcript = (frame.get("result") or {}).get("text", transcript)
            elif kind == "error":
                raise BrainError(f"transcribe failed: {frame.get('message')}")
        writer.join(timeout=5)
        return transcript

    # -- low-level fd primitives ---------------------------------------------
    def run_fds(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        transport: str = "memfd",
    ) -> RunResult:
        """Run one action, returning the result with output fds STILL OPEN.

        The low-level counterpart to :meth:`run`: consume each fd in ``.fds`` with
        :func:`read_fd`. ``transport`` requests ``"memfd"`` (default) or ``"dmabuf"``."""
        result, fds, meta = self._call(
            "Run",
            "sssa{sh}ss",
            (model, action, json.dumps(params or {}), in_fds or {}, json.dumps(in_meta or {}), transport),
        )
        return RunResult(json.loads(result), fds, json.loads(meta))

    def stream_frames(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> Iterator[tuple[dict[str, Any], list[int]]]:
        """Run a streaming action; yield raw `(frame, raw_fds)` until done/error."""
        _job, frames = self.stream_frames_with_job(
            model, action, params, in_fds=in_fds, in_meta=in_meta, timeout=timeout
        )
        yield from frames

    def stream_frames_with_job(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> tuple[int, Iterator[tuple[dict[str, Any], list[int]]]]:
        """Like :meth:`stream_frames`, but also return the job id so :meth:`cancel` works."""
        job, event_fd = self._call(
            "Subscribe",
            "sssa{sh}s",
            (model, action, json.dumps(params or {}), in_fds or {}, json.dumps(in_meta or {})),
        )
        return int(job), _read_stream(event_fd.to_raw_fd(), timeout)

    def cancel(self, job: int) -> bool:
        """Cooperatively cancel a running job (the id from :meth:`stream_frames_with_job`).

        Flips the job's cancel token; the action aborts at its next poll and the
        stream ends with an `error` frame (`"cancelled"`). Returns True if the job
        was found still in flight, False for an unknown or finished id.
        """
        return bool(self._call("Cancel", "t", (job,))[0])

    def stream_transcribe(
        self,
        model: str,
        pcm_read_fd: int,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 3600.0,
    ) -> tuple[int, Iterator[tuple[dict[str, Any], list[int]]]]:
        """Start live streaming transcription (low-level; see :meth:`transcribe`).

        `pcm_read_fd` is the READ end of a pipe the caller keeps writing raw mono
        f32-LE 16 kHz PCM to; the server reads it, windows it, and streams back
        frames. Returns `(job_id, frames)` — `segment` frames as each window decodes,
        then a terminal `done`/`error`. Consume `frames` in one thread while another
        writes PCM; close the write end to signal EOF.
        """
        job, event_fd = self._call(
            "StreamTranscribe",
            "ssh",
            (model, json.dumps(params or {}), FileDescriptor(pcm_read_fd)),
        )
        return job, _read_stream(event_fd.to_raw_fd(), timeout)

    # -- internal ------------------------------------------------------------
    def _call(self, method: str, signature: str | None = None, body: tuple = ()) -> tuple:
        """Send a method call and return its reply body, or raise :class:`BrainError`.

        `send_and_get_reply` alone does NOT distinguish a `method_return` from an
        `error` reply — a D-Bus error comes back as an ordinary `Message` whose
        `.body[0]` is the error TEXT, not JSON. Every prior caller here that did
        `json.loads(self._call(...)[0])` therefore turned a server-side failure
        into a confusing `json.decoder.JSONDecodeError: Expecting value: line 1
        column 1` instead of a real exception. `unwrap_msg` (jeepney's own helper
        for exactly this) raises `DBusErrorResponse` on an error reply; we
        translate that into `BrainError` so callers see one exception type
        regardless of transport.
        """
        msg = new_method_call(_ADDR, method, signature, body) if signature else new_method_call(_ADDR, method)
        try:
            return unwrap_msg(self._conn.send_and_get_reply(msg))
        except DBusErrorResponse as e:
            raise BrainError(_dbus_error_text(e), name=e.name) from None

    def _get_prop(self, name: str) -> Any:
        msg = Properties(_ADDR).get(name)
        try:
            variant = unwrap_msg(self._conn.send_and_get_reply(msg))[0]
        except DBusErrorResponse as e:
            raise BrainError(_dbus_error_text(e), name=e.name) from None
        return variant[1]  # (signature, value)

    @staticmethod
    def _in_fds(blobs: Optional[dict]) -> Fds:
        """Seal each input `name -> bytes` into a memfd ready to pass over the bus."""
        return {name: sealed_memfd(data, name=name) for name, data in (blobs or {}).items()}

    @staticmethod
    def _in_meta(blobs: Optional[dict], meta: Optional[dict]) -> dict[str, Any]:
        """Per-fd metadata for the input blobs (only names present in `blobs`)."""
        meta = meta or {}
        return {name: meta[name] for name in (blobs or {}) if name in meta}


def _dbus_error_text(e: DBusErrorResponse) -> str:
    """A human-readable message from a `DBusErrorResponse`: the first string
    field of its body when present (the convention every `fdo::Error::*` and
    brain's own D-Bus error replies follow), else `str(e)`."""
    if e.data and isinstance(e.data[0], str):
        return e.data[0]
    return str(e)


def _read_stream(raw_fd: int, timeout: float) -> Iterator[tuple[dict[str, Any], list[int]]]:
    with socket.socket(fileno=raw_fd) as sock:
        sock.settimeout(timeout)
        while True:
            data, ancillary, _flags, _addr = sock.recvmsg(1 << 16, socket.CMSG_SPACE(8 * 4))
            if not data:
                return
            frame = json.loads(data)
            yield frame, _scm_rights(ancillary)
            if frame.get("type") in ("done", "error"):
                return


def _scm_rights(ancillary: list) -> list[int]:
    fds: list[int] = []
    for level, kind, payload in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            count = len(payload) // 4
            fds.extend(struct.unpack(f"{count}i", payload[: count * 4]))
    return fds


def read_fd(fd: FileDescriptor | int) -> bytes:
    """Copy the bytes behind a returned fd (mmap, offset-independent). Consumes it."""
    raw = fd.to_raw_fd() if isinstance(fd, FileDescriptor) else int(fd)
    try:
        size = os.fstat(raw).st_size
        if size == 0:
            return b""
        with mmap.mmap(raw, size, prot=mmap.PROT_READ) as m:
            return bytes(m)
    finally:
        os.close(raw)


def sealed_memfd(data: bytes, name: str = "brain") -> FileDescriptor:
    """A sealed memfd holding `data`, ready to pass as an input fd."""
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(fd, data)
    fcntl.fcntl(fd, _F_ADD_SEALS, _SEALS)
    return FileDescriptor(fd)
