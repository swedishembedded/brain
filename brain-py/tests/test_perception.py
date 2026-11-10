# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""``describe`` / ``extract`` on a fake transport: what goes on the wire (image
blob, comma-joined lists, the form as JSON text) and what comes back (a dict
equal to the action's JSON, with the cost in ``.usage``). The live round trip
against ``brain serve --dbus`` is ``tests/e2e/mirage.bats``."""
from __future__ import annotations

import json
import struct

import pytest

pytest.importorskip("jeepney")

from brain_py import MIRAGE_MODEL, Perception  # noqa: E402
from brain_py.base import BrainBase, Outcome  # noqa: E402

USAGE = {"images": 1, "batch": 4, "decoder_tokens": 17, "vision_ms": 1.5}
DESCRIBED = {"objects": [{"id": 1, "name": "cat", "bbox": [1, 2, 3, 4]}], "text": [{"string": "EXIT"}], "usage": USAGE}
EXTRACTED = {"form": {"animal": {"value": "cat", "probabilities": {"cat": 0.9, "-": 0.1}}}, "usage": USAGE}


class FakeMirage(BrainBase):
    def __init__(self):
        self.calls = []

    def manifests(self):
        return [{"model": MIRAGE_MODEL, "actions": [{"name": "describe"}, {"name": "extract"}]}]

    def run(self, model, action, params=None, *, blobs=None, meta=None, timeout=1800.0):
        self.calls.append((model, action, params, blobs, meta))
        return Outcome(outputs=dict(DESCRIBED if action == "describe" else EXTRACTED))

    def subscribe(self, *a, **k):  # pragma: no cover - unused
        raise AssertionError("describe and extract do not stream")


def _image():
    return struct.pack("<12f", *([0.5] * 12)), 2, 2


def test_describe_sends_the_image_blob_and_options_and_returns_the_json():
    brain = FakeMirage()
    out = brain.describe(_image(), sections=["objects", "text"], fields="bbox,mask", conf=0.5, mask_format="bitmap", max_tokens=None)
    model, action, params, blobs, meta = brain.calls[0]
    assert (model, action) == (MIRAGE_MODEL, "describe")
    assert params == {"sections": "objects,text", "fields": "bbox,mask", "conf": 0.5, "mask_format": "bitmap"}
    assert len(blobs["image"]) == 48
    assert meta["image"] == {"media": "image", "w": 2, "h": 2, "c": 3, "meta": {"w": 2, "h": 2, "c": 3}}
    assert isinstance(out, Perception)
    assert dict(out) == {"objects": DESCRIBED["objects"], "text": DESCRIBED["text"]}
    assert out.usage == USAGE
    assert "usage" not in out


def test_extract_sends_the_form_as_json_text():
    brain = FakeMirage()
    form = {"animal": {"enum": ["cat", "dog"]}}
    out = brain.extract(_image(), form, language="de")
    _, action, params, _, _ = brain.calls[0]
    assert action == "extract"
    assert json.loads(params["form"]) == form and params["language"] == "de"
    assert out["form"]["animal"]["value"] == "cat"
    assert out.usage["decoder_tokens"] == 17


def test_a_pil_image_is_encoded_as_an_hwc_f32_blob():
    pil = pytest.importorskip("PIL.Image")
    brain = FakeMirage()
    brain.describe(pil.new("RGB", (3, 2), (255, 0, 0)))
    blobs, meta = brain.calls[0][3], brain.calls[0][4]
    assert len(blobs["image"]) == 3 * 2 * 3 * 4
    assert struct.unpack("<3f", blobs["image"][:12]) == (1.0, 0.0, 0.0)
    assert meta["image"]["meta"] == {"w": 3, "h": 2, "c": 3} and meta["image"]["w"] == 3
