# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""``encode_audio`` on a fake transport: the WAV goes out as the ``audio`` blob
with the transcript as ``text``, and the three little-endian arrays come back
decoded with the scalars beside them. The served action itself is gated in the
engine (``crates/qwen3tts/tests/it/encode.rs``)."""
from __future__ import annotations

import struct

import pytest

pytest.importorskip("jeepney")

from brain_py.base import BrainBase, EncodedAudio, Outcome  # noqa: E402

CODES = [1, 2, 3, 2047] * 8  # two frames of 16 codebooks
XVEC = [0.5, -0.25, 1.0]
IDS = [151644, 77091, 198, 9707, 151645, 198, 151644, 77091, 198]


class FakeTts(BrainBase):
    def __init__(self, with_text: bool):
        self.calls = []
        self.with_text = with_text

    def manifests(self):
        return [{"model": "brain/qwen3tts", "actions": [{"name": "synth"}, {"name": "encode"}]}]

    def run(self, model, action, params=None, *, blobs=None, meta=None, timeout=1800.0):
        self.calls.append((model, action, params, blobs, meta))
        out = Outcome(
            outputs={"frames": 2, "codebooks": 16, "frame_rate": 12.5, "xvector_dim": 3, "sample_rate_in": 16000, "seconds": 0.16, "producers": {"codec": "sha256:" + "a" * 64}},
            blobs={"codes": struct.pack(f"<{len(CODES)}H", *CODES), "xvector": struct.pack("<3f", *XVEC)},
        )
        if self.with_text:
            out.blobs["text_ids"] = struct.pack(f"<{len(IDS)}I", *IDS)
        return out

    def subscribe(self, *a, **k):  # pragma: no cover - unused
        raise AssertionError("encode does not stream")


def test_encode_audio_sends_the_wav_and_text_and_decodes_the_arrays():
    b = FakeTts(with_text=True)
    enc = b.encode_audio(b"RIFF....WAVE", "hej")
    assert isinstance(enc, EncodedAudio)
    model, action, params, blobs, _meta = b.calls[0]
    assert (model, action, params, blobs) == ("brain/qwen3tts", "encode", {"text": "hej"}, {"audio": b"RIFF....WAVE"})
    assert list(enc.codes) == CODES and (enc.frames, enc.codebooks) == (2, 16)
    assert enc.frame(1) == CODES[16:]
    assert list(enc.xvector) == XVEC
    assert list(enc.text_ids) == IDS
    assert (enc.sample_rate_in, enc.seconds, enc.frame_rate) == (16000, 0.16, 12.5)
    assert enc.producers == {"codec": "sha256:" + "a" * 64}


def test_encode_audio_without_text_sends_no_text_and_returns_no_ids():
    b = FakeTts(with_text=False)
    enc = b.encode_audio(b"RIFF....WAVE", model="local/sv")
    assert b.calls[0][0] == "local/sv" and b.calls[0][2] == {}
    assert enc.text_ids is None


class FakeTtsMany(FakeTts):
    """Answers a many-clip ``encode``: two clips, of 2 and 1 frames, the second without a transcript."""

    def run(self, model, action, params=None, *, blobs=None, meta=None, timeout=1800.0):
        self.calls.append((model, action, params, blobs, meta))
        codes = CODES + CODES[:16]
        return Outcome(
            outputs={"clips": 2, "frames": [2, 1], "frame_offsets": [0, 2, 3], "codebooks": 16, "frame_rate": 12.5, "xvector_dim": 3,
                     "sample_rate_in": [16000, 24000], "seconds": [0.16, 0.08], "text_offsets": [0, 9, 9], "producers": {"codec": "sha256:" + "a" * 64}},
            blobs={"codes": struct.pack(f"<{len(codes)}H", *codes), "xvector": struct.pack("<6f", *(XVEC + [2.0, 3.0, 4.0])),
                   "text_ids": struct.pack(f"<{len(IDS)}I", *IDS)},
        )


def test_encode_audio_batch_packs_the_clips_and_splits_each_clips_arrays():
    b = FakeTtsMany(with_text=True)
    out = b.encode_audio_batch([b"RIFF1", b"RIFF22"], ["hej", None])
    _model, action, params, blobs, _meta = b.calls[0]
    assert action == "encode" and set(blobs) == {"clips"}
    assert blobs["clips"] == struct.pack("<Q", 5) + b"RIFF1" + struct.pack("<Q", 6) + b"RIFF22"
    assert params == {"texts": '["hej", null]'}
    assert [type(e) for e in out] == [EncodedAudio, EncodedAudio]
    assert list(out[0].codes) == CODES and out[0].frames == 2 and list(out[1].codes) == CODES[:16]
    assert list(out[0].xvector) == XVEC and list(out[1].xvector) == [2.0, 3.0, 4.0]
    assert list(out[0].text_ids) == IDS and out[1].text_ids is None
    assert (out[1].sample_rate_in, out[1].seconds) == (24000, 0.08)


def test_encode_audio_batch_refuses_mismatched_texts_and_sends_none_without_them():
    b = FakeTtsMany(with_text=False)
    with pytest.raises(ValueError):
        b.encode_audio_batch([b"RIFF1"], ["a", "b"])
    with pytest.raises(ValueError):
        b.encode_audio_batch([])
    b.encode_audio_batch([b"RIFF1", b"RIFF22"])
    assert b.calls[-1][2] == {}
