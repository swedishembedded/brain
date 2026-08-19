# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Live round-trip: cancelling a streaming `generate` mid-flight over a real
`brain serve --dbus`, using `on_job` (added alongside `ParamSpec` UI ranges --
see crates/capability/src/lib.rs) to get the client-visible job id that `Run`
never hands back (see `BrainDBus.cancel`'s doc).

Needs a real session bus and a built `brain` binary -- run under:

    dbus-run-session -- python3 -m pytest tests/test_dbus_cancel.py -q

Skips cleanly otherwise, same as the rest of the suite's live-bus tests.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

jeepney = pytest.importorskip("jeepney")

from brain_py.base import BrainError  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

BRAIN_BIN = Path(__file__).resolve().parents[2] / "target" / "debug" / "brain"

pytestmark = [
    pytest.mark.skipif(
        "DBUS_SESSION_BUS_ADDRESS" not in os.environ,
        reason="needs a session bus: run under `dbus-run-session -- pytest ...`",
    ),
    pytest.mark.skipif(not BRAIN_BIN.is_file(), reason=f"brain binary not found at {BRAIN_BIN}"),
]


@pytest.fixture
def brain_server():
    env = dict(os.environ, BRAIN_MOCK="1", BRAIN_MOCK_DELAY_MS="2000")
    proc = subprocess.Popen([str(BRAIN_BIN), "serve", "--dbus"], env=env)
    try:
        deadline = time.monotonic() + 10
        up = False
        while time.monotonic() < deadline:
            try:
                with BrainDBus() as probe:
                    probe.models()
                up = True
                break
            except Exception:
                time.sleep(0.1)
        if not up:
            raise RuntimeError("brain serve --dbus did not come up in time")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cancel_aborts_a_mock_generate_mid_stream(brain_server):
    # Two connections, deliberately: one per thread. jeepney's blocking
    # connection is not safe to share across concurrent callers.
    jobs: list[int] = []
    error: list[BaseException] = []

    def run_generate():
        try:
            with BrainDBus() as brain:
                brain.generate(prompt="hello", model="mock", on_job=jobs.append)
        except BrainError as e:
            error.append(e)

    t = threading.Thread(target=run_generate)
    t.start()
    deadline = time.monotonic() + 5
    while not jobs and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jobs, "on_job never fired -- the streaming generate never started"

    with BrainDBus() as canceller:
        assert canceller.cancel(jobs[0]) is True

    t.join(timeout=5)
    assert not t.is_alive(), "generate() did not return after cancel"
    assert error, "cancelled generate() must raise BrainError, not return normally"
    assert "cancelled" in str(error[0])
