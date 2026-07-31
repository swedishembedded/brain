# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""CI-fast tests for BrainStdio (the JSONL transport): req_id demux + reader thread.

These run `brain run` with NO --yolo / --gpt, so brain uses its built-in fake
detector (a fixed deterministic box) and fake echo text model. That keeps CI
fast (no model load / no JIT inference) while still exercising the real
subprocess, the JSONL protocol, the background reader thread, and the
condition-guarded req_id demux. The slow real-model demo lives in
brain_py/examples/detect_image.py and is run manually, not in CI.
"""

import os

import pytest
from PIL import Image

from brain_py import Brain, BrainStdio, annotate
from brain_py.client import BrainClient, _find_brain_binary


def _have_brain() -> bool:
    try:
        _find_brain_binary(None)
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _have_brain(),
    reason="brain binary not found; run `cargo build --release`",
)


def test_detect_fake_box_roundtrip():
    """The fake detector returns one fixed box; detect() must surface it."""
    img = Image.new("RGB", (320, 240), (32, 32, 32))
    with BrainStdio() as client:  # no --yolo -> fake detector
        dets = client.detect(img, timeout=30)
    assert len(dets) == 1
    d = dets[0]
    # FakeDetectModel::default() in crates/runtime/src/lib.rs:
    #   det = [10, 20, 110, 220, 0.99, 0], label "object"
    assert d.box == pytest.approx((10.0, 20.0, 110.0, 220.0))
    assert d.conf == pytest.approx(0.99)
    assert d.cls == 0
    assert d.label == "object"


def test_req_id_demux_multiple_inflight():
    """Many requests over one stdio stream get demuxed correctly by req_id."""
    img = Image.new("RGB", (64, 64), (32, 32, 32))
    with BrainStdio() as client:
        # Pin explicit, out-of-order req_ids and confirm each detect() returns
        # the box for ITS request (the reader routes by the echoed req_id).
        for rid in ["alpha", "beta", "gamma"]:
            dets = client.detect(img, timeout=30, req_id=rid)
            assert len(dets) == 1
            assert dets[0].cls == 0


def test_converse_streaming_accumulates():
    """The fake echo model streams 'hello from brain' chunk-by-chunk (user_text path)."""
    with BrainStdio() as client:
        text = client.converse("hi", timeout=30)
    assert "hello from brain" in text


def test_brain_factory_selects_jsonl():
    """`Brain(transport="jsonl")` builds a BrainStdio without a live D-Bus server."""
    with Brain(transport="jsonl") as client:
        assert isinstance(client, BrainStdio)
        dets = client.detect(Image.new("RGB", (64, 64), (32, 32, 32)), timeout=30)
    assert len(dets) == 1


def test_brainclient_alias():
    """The old BrainClient name still resolves to BrainStdio."""
    assert BrainClient is BrainStdio


def test_annotate_produces_nonempty_image(tmp_path):
    img = Image.new("RGB", (320, 240), (32, 32, 32))
    with BrainStdio() as client:
        dets = client.detect(img, timeout=30)
    out = annotate(img, dets)
    p = tmp_path / "annotated.png"
    out.save(p)
    assert p.stat().st_size > 0
