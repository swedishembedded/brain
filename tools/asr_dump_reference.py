#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump golden reference tensors for the ASR models so the Rust engine can be
parity-gated against the real HuggingFace implementations.

Stage 1 (this pass): the log-mel FRONT ENDS.
    - NemotronAsrStreamingFeatureExtractor  (librosa slaney mel, n_fft 512/win 400,
      Hann periodic=False, constant pad, preemphasis 0.97, log(x+2^-24), no norm)
    - Qwen3ASRFeatureExtractor               (transformers slaney mel, n_fft 400,
      Hann periodic=True, reflect pad, log10 + dynamic-range compression)

Everything is written as little-endian f32 raw blobs plus a JSON manifest of shapes
under resources/asr/golden/<group>/. Deterministic input (fixed seed) so the Rust
side feeds byte-identical samples.

Usage:  python tools/asr_dump_reference.py frontend
"""
import json
import os
import sys

import numpy as np

RES = "/data/workspace/resources/asr"
GOLD = os.path.join(RES, "golden")


def save(group: str, name: str, arr, manifest: dict):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    d = os.path.join(GOLD, group)
    os.makedirs(d, exist_ok=True)
    arr.tofile(os.path.join(d, name + ".f32"))
    manifest[name] = list(arr.shape)


def make_waveform(seconds=2.0, sr=16000, seed=1234):
    """Deterministic pseudo-speech: sum of a few swept sinusoids + mild noise, in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    for f0, f1, amp in [(120, 180, 0.5), (440, 400, 0.3), (900, 1300, 0.2), (2100, 1800, 0.1)]:
        inst = f0 + (f1 - f0) * (t / seconds)
        phase = 2 * np.pi * np.cumsum(inst) / sr
        sig += amp * np.sin(phase)
    sig += 0.01 * rng.standard_normal(n)
    # gentle amplitude envelope so it isn't perfectly stationary
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    sig = sig * env
    sig = sig / np.max(np.abs(sig)) * 0.9
    return sig.astype(np.float32)


def dump_frontend():
    import torch
    from transformers import AutoProcessor

    wav = make_waveform()
    man = {"sampling_rate": 16000}
    save("frontend", "waveform", wav, man)

    # ---- Nemotron ----
    proc_n = AutoProcessor.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
    fe_n = proc_n.feature_extractor
    out_n = fe_n(wav, sampling_rate=16000, return_tensors="pt", center=True)
    # (1, T, 128)
    save("frontend", "nemotron_mel", out_n["input_features"][0].numpy(), man)
    save("frontend", "nemotron_mel_filters", fe_n.mel_filters.numpy(), man)  # [128, 257]
    save("frontend", "nemotron_hann", torch.hann_window(fe_n.win_length, periodic=False).numpy(), man)
    man["nemotron"] = dict(n_fft=fe_n.n_fft, hop=fe_n.hop_length, win=fe_n.win_length,
                           preemphasis=fe_n.preemphasis, n_mels=fe_n.feature_size)

    # ---- Qwen3-ASR ----
    proc_q = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    fe_q = proc_q.feature_extractor
    feat_q = fe_q(wav, sampling_rate=16000, return_tensors="pt")
    # Qwen returns input_features (B, 128, T) and input_features_mask
    fk = "input_features" if "input_features" in feat_q else list(feat_q.keys())[0]
    arr_q = feat_q[fk][0].numpy()
    save("frontend", "qwen_mel", arr_q, man)
    # valid (non-padding) frame count so the Rust test compares only real frames
    mask_key = next((k for k in feat_q if "mask" in k), None)
    qvalid = int(np.asarray(feat_q[mask_key][0]).sum()) if mask_key else arr_q.shape[-1]
    man["qwen_valid_frames"] = qvalid
    save("frontend", "qwen_mel_filters", np.asarray(fe_q.mel_filters), man)  # [201, 128]
    save("frontend", "qwen_hann", torch.hann_window(fe_q.n_fft, periodic=True).numpy(), man)
    man["qwen"] = dict(n_fft=fe_q.n_fft, hop=fe_q.hop_length, n_window=fe_q.n_window,
                       n_mels=fe_q.feature_size, mel_out_shape=list(arr_q.shape))

    with open(os.path.join(GOLD, "frontend", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print("frontend goldens written:")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "frontend"
    if stage == "frontend":
        dump_frontend()
    else:
        raise SystemExit(f"unknown stage {stage!r}")
