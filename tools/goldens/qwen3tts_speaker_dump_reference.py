#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the Qwen3-TTS ECAPA-TDNN speaker golden for `crates/ecapatdnn`.

The reference is `Qwen3TTSSpeakerEncoder` from the upstream `qwen-tts` package
(see `qwen3tts_ref.py`), driven by that package's own `mel_spectrogram` with
exactly the arguments `Qwen3TTSForConditionalGeneration.extract_speaker_embedding`
uses (n_fft 1024, 128 mels, hop 256, win 1024, fmin 0, fmax 12000, 24 kHz).
Weights are the `speaker_encoder.*` subset of a released Qwen3-TTS base
checkpoint, cast bf16 -> fp32 (exact) and run on the CPU.

Only the speaker encoder is built: instantiating the whole conditional
generation model would pull the talker, the MTP head and the speech tokenizer
for tensors this golden never touches.

Writes `spk_ref/`:
  `mel.f32`        the [T,128] row-major log-mel the encoder was fed
  `embedding.f32`  the [enc_dim] x-vector it produced
  `manifest.json`  shapes plus the `source` block

Both `.f32` files are bare little-endian f32 (no count prefix), the layout
`crates/ecapatdnn/tests/encoder.rs` reads.

`--wav` is any 24 kHz mono clip; `mel.f32` carries the resulting features
verbatim, so the gate itself needs no audio. The dump in this repo was made
from the voice-clone example the Qwen3-TTS model card links,
`Qwen3-TTS-Repo/clone.wav` on `qianwen-res.oss-cn-beijing.aliyuncs.com`
(sha256 480f55f4...5a6b5c, first 2 s).

Usage:
  python3 tools/goldens/qwen3tts_speaker_dump_reference.py \
      --ckpt testdata/tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base \
      --wav <a 24 kHz mono clip> --out testdata/tts/dumps
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402
from qwen3tts_ref import load_speaker  # noqa: E402

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
PREFIX = "speaker_encoder."
# `extract_speaker_embedding`'s literal mel arguments.
MEL = dict(n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256,
           win_size=1024, fmin=0, fmax=12000)


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Qwen3-TTS-*-Base directory")
    ap.add_argument("--wav", required=True, help="24 kHz mono reference clip")
    ap.add_argument("--out", required=True, help="dump root, e.g. testdata/tts/dumps")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    config_class, encoder_class, mel_spectrogram = load_speaker()
    import soundfile as sf
    import torch
    from safetensors.torch import load_file

    with open(os.path.join(args.ckpt, "config.json")) as f:
        root = json.load(f)
    cfg = config_class(**root["speaker_encoder_config"])

    weights_path = os.path.join(args.ckpt, "model.safetensors")
    state = {
        k[len(PREFIX):]: v.float()
        for k, v in load_file(weights_path).items()
        if k.startswith(PREFIX)
    }
    if not state:
        raise SystemExit(f"{weights_path} carries no {PREFIX}* tensors")
    model = encoder_class(cfg).float().eval()
    model.load_state_dict(state, strict=True)

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.wav} is {sr} Hz; the speaker encoder needs {cfg.sample_rate} Hz")
    wav = np.ascontiguousarray(wav[: int(args.seconds * sr)], dtype=np.float32)

    with torch.no_grad():
        mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), **MEL).transpose(1, 2)
        embedding = model(mel)[0]

    out_dir = os.path.join(args.out, "spk_ref")
    os.makedirs(out_dir, exist_ok=True)
    write_f32(os.path.join(out_dir, "mel.f32"), mel[0].numpy())
    write_f32(os.path.join(out_dir, "embedding.f32"), embedding.numpy())
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "mel": list(mel[0].shape),
                "embedding": list(embedding.shape),
                "samples": int(wav.size),
                "mel_args": MEL,
                "source": source_block(
                    checkpoint=CHECKPOINT,
                    files=[weights_path],
                    identity={
                        "mel_dim": cfg.mel_dim,
                        "enc_dim": cfg.enc_dim,
                        "enc_channels_last": cfg.enc_channels[-1],
                        "enc_attention_channels": cfg.enc_attention_channels,
                        "enc_res2net_scale": cfg.enc_res2net_scale,
                        "enc_se_channels": cfg.enc_se_channels,
                    },
                ),
            },
            f,
            indent=2,
        )

    norm = float(embedding.norm())
    print(f"speaker golden: mel {tuple(mel[0].shape)} -> embedding {tuple(embedding.shape)} ({out_dir})")
    print(f"|embedding| {norm:.4f}, tensors loaded {len(state)}")


if __name__ == "__main__":
    main()
