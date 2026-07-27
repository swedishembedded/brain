# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Unit tests for the D-Bus client's FD helpers (no bus required).

The full round-trip against a live ``brain serve --dbus`` is exercised by
``tests/e2e/scheduler.bats`` in the repo; here we just check the memfd plumbing.
"""
from __future__ import annotations

import os

import pytest

jeepney = pytest.importorskip("jeepney")  # the dbus client's only extra dependency

from brain_py.dbus import read_fd, sealed_memfd  # noqa: E402


def test_sealed_memfd_roundtrips_regardless_of_offset():
    payload = b"brain-py dbus fd \x00\x01\x02 payload"
    fd = sealed_memfd(payload, name="test")
    raw = fd.fileno()  # peek without consuming
    # `read_fd` mmaps (offset-independent) and closes the fd.
    assert read_fd(fd) == payload
    # the fd was consumed/closed by read_fd
    with pytest.raises(OSError):
        os.fstat(raw)


def test_empty_payload():
    fd = sealed_memfd(b"")
    assert read_fd(fd) == b""
