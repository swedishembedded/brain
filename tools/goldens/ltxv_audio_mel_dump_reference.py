#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a golden for LTX-2.5's audio-encode FRONT END: waveform -> log-mel.

An audio-conditioned generation (a2v) encodes its source soundtrack through
`ltx_core.model.audio_vae.encode_audio`, whose first half is
`AudioProcessor.waveform_to_mel`: a torchaudio resample onto the audio VAE's
16 kHz, a reflect-centred 1024-point STFT magnitude with hop 160, 64 Slaney
mel bins up to 8 kHz with Slaney norm, then `log(clamp(mel, 1e-5))`. This
dumper runs that processor over a deterministic stereo 44.1 kHz waveform -
so the resample runs too - and dumps `[channels, frames, mels]`. The encoder
behind it has its own golden (`ltxv_audio_dump_reference.py`).

  ltxv_audio_mel.safetensors   waveform [2, S] at 44100 Hz, mel [2, T, 64]

Usage:
  LTXV_REFERENCE_ROOT=/path/to/LTX-2 python3 tools/goldens/ltxv_audio_mel_dump_reference.py \\
      --out testdata/golden/ltxv/audio_mel --seed 7
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

_REFERENCE_ROOT = Path(os.environ.get("LTXV_REFERENCE_ROOT", str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_source import source_block  # noqa: E402
from ltx_core.model.audio_vae.ops import AudioProcessor  # noqa: E402
from ltx_core.types import Audio  # noqa: E402

RATE_IN = 44100
SAMPLE_RATE, N_FFT, HOP, MELS = 16000, 1024, 160, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    n = int(0.37 * RATE_IN)
    t = torch.arange(n, dtype=torch.float32) / RATE_IN
    left = 0.5 * torch.sin(2 * math.pi * 440 * t) + 0.05 * torch.randn(n, generator=g)
    right = 0.3 * torch.sin(2 * math.pi * 1250 * t) + 0.05 * torch.randn(n, generator=g)
    wave = torch.stack([left, right])
    proc = AudioProcessor(target_sample_rate=SAMPLE_RATE, mel_bins=MELS, mel_hop_length=HOP, n_fft=N_FFT)
    mel = proc.waveform_to_mel(Audio(waveform=wave[None], sampling_rate=RATE_IN))[0]
    tensors = {"waveform": wave.contiguous(), "mel": mel.contiguous()}
    path = os.path.join(args.out, "ltxv_audio_mel.safetensors")
    save_file(tensors, path)
    manifest = {
        "ltxv_audio_mel.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "tensors": {k: list(v.shape) for k, v in tensors.items()}},
        "seed": args.seed,
        "source": source_block(checkpoint=None, identity={"rate_in": RATE_IN, "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop": HOP, "mels": MELS}),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote", {k: list(v.shape) for k, v in tensors.items()})


if __name__ == "__main__":
    main()
