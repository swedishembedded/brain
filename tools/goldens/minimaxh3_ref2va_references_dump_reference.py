#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump goldens for MiniMax-H3's `ref2va` reference NORMALIZATION.

`MiniMaxH3Ref2VASetupStep` puts every reference on the model's own rates and
resolutions before anything is encoded: an image onto its own short edge
(LANCZOS, through diffusers' `VaeImageProcessor.resize`), a video onto 24 fps
(whole frames dropped or held) and onto the canvas its own aspect resolves to
(LANCZOS), a soundtrack truncated at its native rate, upmixed to stereo and
resampled with `torchaudio`. This dumper runs the installed diffusers'
static normalizers (and the image branch of `__call__`, transcribed, since it
needs the pipeline's components) over deterministic synthetic media at a
SCALED-DOWN geometry - multiple 32, canvas short edge 64, image short edge 96
- so the golden stays small while every branch (downscale, upscale, fps
down- and up-sampling, mono upmix, truncation, resampling) runs.

Needs no checkpoint.

  minimaxh3_ref2va_references.safetensors
    image_in / image_out          HWC uint8 as float32
    video_a_in / video_a_out      THWC, 30 fps -> 24 fps, downscaled
    video_b_in / video_b_out      THWC, 10 fps -> 24 fps, upscaled
    audio_in / audio_out          [1, S] at 44100 Hz -> [2, S'] at 32000 Hz
  manifest.json                   geometry, rates and the `source` block

Usage:
  python3 tools/goldens/minimaxh3_ref2va_references_dump_reference.py \\
      --out testdata/golden/minimaxh3/ref2va_references --seed 7
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep  # noqa: E402

MULTIPLE = 32
CANVAS_SHORT_EDGE = 64
CANVAS_MAX_PIXELS = 64 * 112
IMAGE_SHORT_EDGE = 96
FPS = 24.0
NUM_FRAMES = 22
AUDIO_RATE_IN = 44100
AUDIO_RATE_OUT = 32000


def frames(rng, n, h, w):
    return rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)


def normalize_image(image: Image.Image) -> Image.Image:
    """The image branch of `MiniMaxH3Ref2VASetupStep.__call__`, transcribed."""
    width, height = image.size
    scale = IMAGE_SHORT_EDGE / min(width, height)
    target_height = max(MULTIPLE, round(height * scale / MULTIPLE) * MULTIPLE)
    target_width = max(MULTIPLE, round(width * scale / MULTIPLE) * MULTIPLE)
    if image.size != (target_width, target_height):
        image = VaeImageProcessor(vae_scale_factor=16).resize(image, height=target_height, width=target_width)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tensors = {}
    image = frames(rng, 1, 50, 70)[0]
    tensors["image_in"] = torch.from_numpy(image.astype(np.float32))
    tensors["image_out"] = torch.from_numpy(np.asarray(normalize_image(Image.fromarray(image))).astype(np.float32))

    for name, (n, h, w, fps) in {"video_a": (31, 36, 60, 30.0), "video_b": (11, 20, 28, 10.0)}.items():
        clip = frames(rng, n, h, w)
        out = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            clip, fps, NUM_FRAMES, MULTIPLE, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS, FPS
        )
        tensors[f"{name}_in"] = torch.from_numpy(clip.astype(np.float32))
        tensors[f"{name}_out"] = torch.from_numpy(out.astype(np.float32))
        tensors[f"{name}_fps"] = torch.tensor([fps], dtype=torch.float32)

    # Longer than the generated duration, so the truncation bites.
    samples = int(AUDIO_RATE_IN * (NUM_FRAMES / FPS)) + 4321
    t = np.arange(samples) / AUDIO_RATE_IN
    wave = (0.5 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.standard_normal(samples)).astype(np.float32)[None]
    out = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        torch.from_numpy(wave), AUDIO_RATE_IN, AUDIO_RATE_OUT, max_duration=NUM_FRAMES / FPS
    )
    tensors["audio_in"] = torch.from_numpy(wave)
    tensors["audio_out"] = out.contiguous()

    path = os.path.join(args.out, "minimaxh3_ref2va_references.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    manifest = {
        "minimaxh3_ref2va_references.safetensors": {
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
        },
        "seed": args.seed,
        "source": source_block(
            checkpoint=None,
            identity={
                "multiple": MULTIPLE,
                "canvas_short_edge": CANVAS_SHORT_EDGE,
                "canvas_max_pixels": CANVAS_MAX_PIXELS,
                "image_short_edge": IMAGE_SHORT_EDGE,
                "num_frames": NUM_FRAMES,
                "audio_rate_in": AUDIO_RATE_IN,
                "audio_rate_out": AUDIO_RATE_OUT,
            },
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote", {k: list(v.shape) for k, v in tensors.items()})


if __name__ == "__main__":
    main()
