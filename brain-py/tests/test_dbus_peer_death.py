# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Live regression: killing `brain serve --dbus` mid-`Subscribe` must surface
as a raised error, not a silent empty `Outcome`.

Found while building loom_brain.link.BrainLink (applications/loom): its
"a connection loss fails the job and reconnects" supervision relies on
`subscribe()` raising when the peer dies. Before this fix, `_read_stream`'s
`if not data: return` treated a mid-stream socket EOF (which is exactly what
a SIGKILLed brain's now-closed SEQPACKET fd produces) the same as an orderly
end of iteration -- `subscribe()` then returned an empty `Outcome` as if the
action had simply produced nothing, with no exception at all.

Needs a real session bus and a built `brain` binary -- run under:

    dbus-run-session -- python3 -m pytest tests/test_dbus_peer_death.py -q
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

jeepney = pytest.importorskip("jeepney")

from brain_py.dbus import BrainDBus  # noqa: E402

BRAIN_BIN = Path(__file__).resolve().parents[2] / "target" / "debug" / "brain"

pytestmark = [
    pytest.mark.skipif(
        "DBUS_SESSION_BUS_ADDRESS" not in os.environ,
        reason="needs a session bus: run under `dbus-run-session -- pytest ...`",
    ),
    pytest.mark.skipif(not BRAIN_BIN.is_file(), reason=f"brain binary not found at {BRAIN_BIN}"),
]


def test_killing_brain_mid_stream_raises_instead_of_returning_empty():
    env = dict(os.environ, BRAIN_MOCK="1", BRAIN_MOCK_DELAY_MS="5000")
    proc = subprocess.Popen([str(BRAIN_BIN), "serve", "--dbus"], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        conn = None
        while time.monotonic() < deadline:
            try:
                conn = BrainDBus()
                conn.models()
                break
            except Exception:
                conn = None
                time.sleep(0.1)
        assert conn is not None, "brain serve --dbus did not come up in time"

        job, frames = conn.stream_frames_with_job("mock", "generate", {"prompt": "hi"})
        proc.kill()
        proc.wait(timeout=5)

        with pytest.raises(Exception):
            list(frames)
    finally:
        conn.close()
        try:
            proc.kill()
        except Exception:
            pass
