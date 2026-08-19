# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Unit tests for the D-Bus transport (no bus required).

Two layers are covered here without a live server:

* the memfd/fd plumbing (``sealed_memfd`` / ``read_fd``); and
* the transport-agnostic capability layer (:class:`brain_py.base.Outcome` and the
  :class:`brain_py.base.BrainBase` convenience wrappers) via a fake transport, so
  ``generate`` / ``embed`` / ``text2image`` are exercised with no D-Bus / no brain.

The full round-trip against a live ``brain serve --dbus`` is exercised by
``tests/e2e/scheduler.bats`` in the repo.
"""
from __future__ import annotations

import json
import os

import pytest

jeepney = pytest.importorskip("jeepney")  # jeepney is required (D-Bus is the default)

from brain_py import BrainDBus, BrainError  # noqa: E402
from brain_py.base import BrainBase, Outcome  # noqa: E402
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


def test_braindbus_is_a_brainbase():
    """The default transport implements the shared capability API."""
    assert issubclass(BrainDBus, BrainBase)
    for verb in ("run", "subscribe", "generate", "chat", "embed", "text2image", "models"):
        assert hasattr(BrainDBus, verb), verb


# -- regression: a D-Bus ERROR reply must raise BrainError, not a masked JSON bug --
#
# `send_and_get_reply` does not distinguish a `method_return` from an `error`
# reply on its own — before the fix, `_call`/`_get_prop` did
# `json.loads(self._call(...)[0])` on whatever came back, so a genuine server
# error (the reply's body is the error TEXT, not JSON) surfaced as a baffling
# `json.decoder.JSONDecodeError: Expecting value: line 1 column 1` instead of a
# real, readable exception. This is exactly the bug reported against
# `detect_pipeline.py` / `embed_document.py`.

def _error_reply(text: str, name: str = "com.swedishembedded.Brain1.Error.Failed"):
    """A genuine jeepney D-Bus ERROR-type `Message`, built without any real bus —
    jeepney's message types are pure data, so this is exactly what a real
    `org.freedesktop.DBus.Error.Failed` (or brain's own error) reply looks like
    on the wire."""
    from jeepney import Header, HeaderFields, Message, MessageType

    header = Header(
        endianness="l",
        message_type=MessageType.error,
        flags=0,
        protocol_version=1,
        body_length=0,
        serial=2,
        fields={
            HeaderFields.reply_serial: 1,
            HeaderFields.error_name: name,
            HeaderFields.signature: "s",
        },
    )
    return Message(header, (text,))


class _FakeErrorConnection:
    """Stands in for jeepney's blocking `DBusConnection`: every call returns a
    canned D-Bus error reply, so `BrainDBus._call`/`_get_prop` can be exercised
    with no session bus at all."""

    def __init__(self, text: str):
        self._text = text

    def send_and_get_reply(self, _msg, timeout=None):
        return _error_reply(self._text)


def test_call_raises_brain_error_not_a_masked_json_bug():
    brain = BrainDBus.__new__(BrainDBus)  # skip __init__ (no real bus connection)
    brain._conn = _FakeErrorConnection("model 'nemotronasr' not served (have: mock, demo)")

    with pytest.raises(BrainError) as exc_info:
        brain.manifests()  # json.loads(self._call("Manifests")[0]) in the old code

    err = exc_info.value
    assert "not served" in str(err)
    assert err.name == "com.swedishembedded.Brain1.Error.Failed"
    # The precise regression: this must NOT be a JSON parse failure.
    assert not isinstance(err, json.JSONDecodeError)


def test_get_prop_raises_brain_error_on_an_error_reply():
    brain = BrainDBus.__new__(BrainDBus)
    brain._conn = _FakeErrorConnection("no such property")

    with pytest.raises(BrainError, match="no such property"):
        brain.version()


# -- transport-agnostic capability layer, exercised with a fake transport -------

class FakeBrain(BrainBase):
    """A BrainBase whose primitives are canned, so base.py is tested with no I/O."""

    MANIFESTS = [
        {"model": "mock", "actions": [{"name": "generate"}, {"name": "embed"}, {"name": "text2image"}]},
        {"model": "other", "actions": [{"name": "detect"}]},
    ]

    def __init__(self):
        self.calls = []

    def manifests(self):
        return self.MANIFESTS

    def run(self, model, action, params=None, *, blobs=None, meta=None, timeout=1800.0):
        self.calls.append((model, action, params))
        if action == "generate":
            said = json.loads((params or {}).get("messages", "[]"))
            text = said[-1]["content"] if said else (params or {}).get("prompt", "")
            return Outcome(outputs={"finish_reason": "stop"}, blobs={"text": f"You said: {text}".encode()})
        if action == "embed":
            return Outcome(outputs={"mean": [0.1, 0.2, 0.3], "tokens": 2})
        if action == "text2image":
            # 2x2 white RGB, HWC f32.
            import struct
            raw = struct.pack("<12f", *([1.0] * 12))
            return Outcome(outputs={}, blobs={"image": raw}, meta={"image": {"meta": {"w": 2, "h": 2, "c": 3}}})
        raise AssertionError(action)

    def subscribe(self, model, action, params=None, *, blobs=None, meta=None,
                  on_progress=None, on_job=None, timeout=1800.0):
        if on_job is not None:
            on_job(42)
        if on_progress is not None:
            on_progress(1, 1, "tick")
        return self.run(model, action, params, blobs=blobs, meta=meta, timeout=timeout)


def test_models_and_capability_lookup():
    b = FakeBrain()
    assert b.models() == ["mock", "other"]
    assert b.actions("mock") == ["generate", "embed", "text2image"]
    assert b.model_for("embed") == "mock"       # first model advertising it
    assert b.model_for("detect") == "other"
    with pytest.raises(LookupError):
        b.model_for("nope")


def test_generate_picks_model_and_returns_text():
    b = FakeBrain()
    assert b.generate(prompt="hi") == "You said: hi"     # model auto-picked = mock
    assert b.chat("hello") == "You said: hello"          # chat sugars generate
    assert b.calls[0][0] == "mock"


def test_generate_streams_via_subscribe_when_on_progress_given():
    b = FakeBrain()
    ticks = []
    out = b.generate(prompt="hi", on_progress=lambda s, t, m: ticks.append((s, t, m)))
    assert out == "You said: hi"
    assert ticks == [(1, 1, "tick")]


def test_generate_on_job_receives_the_job_id_for_cancellation():
    """A caller that wants to cancel a streaming generate() must be able to
    capture the job id without dropping to the low-level stream_frames_with_job
    API. on_progress alone forces a Run (uncancellable server-side); on_job
    forces Subscribe (the only cancellable path) and hands back the id."""
    b = FakeBrain()
    jobs = []
    out = b.generate(prompt="hi", on_progress=lambda *_: None, on_job=jobs.append)
    assert out == "You said: hi"
    assert jobs == [42]


def test_text2image_on_job_receives_the_job_id_for_cancellation():
    b = FakeBrain()
    jobs = []
    img = b.text2image("a cube", on_progress=lambda *_: None, on_job=jobs.append)
    assert img.size == (2, 2)
    assert jobs == [42]


def test_embed_returns_vector():
    assert FakeBrain().embed("hello world") == [0.1, 0.2, 0.3]


def test_text2image_decodes_to_pil():
    img = FakeBrain().text2image("a cube")
    assert img.size == (2, 2)
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_outcome_text_prefers_blob_then_outputs():
    assert Outcome(blobs={"text": b"hi"}).text() == "hi"
    assert Outcome(outputs={"text": "yo"}).text() == "yo"
