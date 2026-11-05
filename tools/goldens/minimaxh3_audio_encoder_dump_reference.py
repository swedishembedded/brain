#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump real-weight parity goldens for MiniMax-H3's audio VAE ENCODER.

Loads `diffusers.AutoencoderKLMiniMaxH3Audio` (the Apache-2.0 reference this
port follows) from the real `audio_vae/` directory, encodes a deterministic
synthetic mono waveform the way `ref2va` encodes a reference clip's channels,
and dumps every stage:

  minimaxh3_audio_encoder.safetensors
    waveform        [samples]       the input (NOT a multiple of the hop, so the
                                    right padding is exercised)
    tap_encoder     [2048, T]       encoder(padded) - the DAC waveform encoder
    tap_pre_block   [32, T]         pre_block(tap_encoder) - attention projection
    latent          [32, T]         encode(waveform).latent_dist.mode() - what
                                    the pipeline consumes (the posterior mean)
  manifest.json      shapes, sha256, seed and the `source` block

Self-validation: the stage-by-stage replay, run on the same submodules, must
reproduce `encode(...).latent_dist.mode()` bit for bit before anything is
written.

Usage:
  python3 tools/goldens/minimaxh3_audio_encoder_dump_reference.py \\
      --audio-vae-dir /path/to/MiniMax-H3/audio_vae \\
      --out testdata/golden/minimaxh3/audio_encoder --seed 7 [--seconds 1.3]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLMiniMaxH3Audio
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402


def det_waveform(samples, seed, rate):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(samples, dtype=torch.float32) / rate
    sig = 0.4 * torch.sin(2 * math.pi * 220.0 * t) + 0.2 * torch.sin(2 * math.pi * 1310.0 * t)
    return (sig + 0.02 * torch.randn(samples, generator=g)).clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-vae-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=1.3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.audio_vae_dir, torch_dtype=torch.float32).eval()
    rate = vae.config.sampling_rate
    samples = int(args.seconds * rate) + 123  # deliberately off the 800-sample hop
    wave = det_waveform(samples, args.seed, rate)

    latent = vae.encode(wave.view(1, 1, -1)).latent_dist.mode()[0]

    hop = vae.hop_length
    padded = F.pad(wave.view(1, 1, -1), (0, math.ceil(samples / hop) * hop - samples))
    enc = vae.encoder(padded)
    pre = vae.pre_block(enc.transpose(1, 2)).transpose(1, 2)
    mean = vae.mean_proj(pre)[0]
    d = (mean - latent).abs().max().item()
    print(f"self-validate replay vs encode(): max abs {d:.3e}", flush=True)
    assert d == 0.0, "the stage replay must reproduce encode() exactly"

    tensors = {"waveform": wave, "tap_encoder": enc[0], "tap_pre_block": pre[0], "latent": latent}
    path = os.path.join(args.out, "minimaxh3_audio_encoder.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    manifest = {
        "minimaxh3_audio_encoder.safetensors": {
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
        },
        "run": {"seed": args.seed, "samples": samples, "sampling_rate": rate, "hop_length": hop},
        "source": source_block(
            checkpoint="MiniMaxAI/MiniMax-H3",
            files=[os.path.join(args.audio_vae_dir, "diffusion_pytorch_model.safetensors")],
            identity={"latent_dim": vae.config.latent_dim, "latent_channels": vae.config.latent_channels, "encoder_dim": vae.config.encoder_dim},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote", {k: list(v.shape) for k, v in tensors.items()})


if __name__ == "__main__":
    main()
