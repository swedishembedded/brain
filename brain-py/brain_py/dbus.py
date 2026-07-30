# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A small, reusable client for brain's D-Bus control surface.

`com.swedishembedded.Brain1` exchanges bulk data (images, results, streams) as Unix
file descriptors, so this wraps jeepney with FD passing enabled and exposes a clean
API:

    with BrainDBus() as brain:
        print(brain.models())
        out = brain.run("imageops", "gradient", {"width": 128, "height": 128})
        img = read_fd(out.fds["image"])
        for frame, fds in brain.subscribe("z-image", "text2image", {"prompt": "a cat"}):
            ...

FD passing requires `enable_fds=True`; received fds arrive as jeepney
`FileDescriptor` objects (consume them with `read_fd`).
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
from typing import Any

from jeepney import DBusAddress, new_method_call
from jeepney.fds import FileDescriptor
from jeepney.io.blocking import open_dbus_connection

BUS_NAME = "com.swedishembedded.Brain1"
OBJECT_PATH = "/com/swedishembedded/Brain1"
INTERFACE = f"{BUS_NAME}.Manager"

_ADDR = DBusAddress(OBJECT_PATH, bus_name=BUS_NAME, interface=INTERFACE)
_F_ADD_SEALS = 1033
_SEALS = 0x1 | 0x2 | 0x4  # SHRINK | GROW | WRITE

Fds = dict[str, FileDescriptor]


@dataclass
class RunResult:
    """The outcome of a `Run`: scalar result, output fds, and their metadata."""

    result: dict[str, Any]
    fds: Fds
    meta: dict[str, Any]


class BrainDBus:
    """Blocking client for the Manager interface (one bus connection)."""

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

    def models(self) -> list[str]:
        return [m["model"] for m in self.manifests()]

    def stats(self) -> dict[str, int]:
        """Scheduler counters: builds, evictions, batches, max_batch, resident, …."""
        return json.loads(self._call("Stats")[0])

    # -- execution -----------------------------------------------------------
    def run(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        transport: str = "memfd",
    ) -> RunResult:
        result, fds, meta = self._call(
            "Run",
            "sssa{sh}ss",
            (model, action, json.dumps(params or {}), in_fds or {}, json.dumps(in_meta or {}), transport),
        )
        return RunResult(json.loads(result), fds, json.loads(meta))

    def subscribe(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> Iterator[tuple[dict[str, Any], list[int]]]:
        """Run a streaming action; yield `(frame, raw_fds)` until done/error."""
        _job, frames = self.subscribe_with_job(
            model, action, params, in_fds=in_fds, in_meta=in_meta, timeout=timeout
        )
        yield from frames

    def subscribe_with_job(
        self,
        model: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        in_fds: Fds | None = None,
        in_meta: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> tuple[int, Iterator[tuple[dict[str, Any], list[int]]]]:
        """Like `subscribe`, but also return the job id so `cancel(job)` works."""
        job, event_fd = self._call(
            "Subscribe",
            "sssa{sh}s",
            (model, action, json.dumps(params or {}), in_fds or {}, json.dumps(in_meta or {})),
        )
        return int(job), _read_stream(event_fd.to_raw_fd(), timeout)

    def cancel(self, job: int) -> bool:
        """Cooperatively cancel a running job (the id from `subscribe`).

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
        """Start live streaming transcription.

        `pcm_read_fd` is the READ end of a pipe the caller keeps writing raw mono
        f32-LE 16 kHz PCM to (from its WRITE end); the server reads it, windows it,
        and streams back frames. Returns `(job_id, frames)` where `frames` is an
        iterator of `(frame, raw_fds)` — `segment` frames as each window decodes, then
        a terminal `done`/`error`. Consume `frames` in one thread while another writes
        PCM; close the write end to signal EOF (the server then emits `done`).

        The read fd is passed to the server (dup'd over the socket); the caller may
        close its own copy afterward.
        """
        job, event_fd = self._call(
            "StreamTranscribe",
            "ssh",
            (model, json.dumps(params or {}), FileDescriptor(pcm_read_fd)),
        )
        return job, _read_stream(event_fd.to_raw_fd(), timeout)

    # -- internal ------------------------------------------------------------
    def _call(self, method: str, signature: str | None = None, body: tuple = ()) -> tuple:
        msg = new_method_call(_ADDR, method, signature, body) if signature else new_method_call(_ADDR, method)
        return self._conn.send_and_get_reply(msg).body


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
