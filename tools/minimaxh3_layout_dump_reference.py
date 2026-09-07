#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump numeric parity goldens for MiniMax-H3's PIPELINE GLUE around the DiT.

`minimaxh3_dit_real_dump_reference.py` proves the DiT's own block math against
the real checkpoint, but it hand-builds its packed sequence, so everything
that ASSEMBLES that sequence is invisible to it: the packed layout, the rotary
grid, the row-to-timestep plan, the two shifted-sigma schedules, and the
DiT-level patchify. A bug in any of those feeds correct block math a wrong
input and shows up only as a bad picture.

So this dumper calls the REAL installed `diffusers==0.40.0` implementations of
exactly those functions

    MiniMaxH3PrepareLayoutStep.build_packed_sequence
    MiniMaxH3SetTimestepsStep.build_row_timesteps
    MiniMaxH3Scheduler.set_timesteps / .step / .scale_noise
    before_denoise.patchify_video_latents

and dumps their outputs. It needs NO checkpoint at all - every one of those is
pure layout/schedule arithmetic over the config numbers, so this golden is
cheap to regenerate and carries no weights.

The geometry is the REAL one a `t2va` request resolves at a 128x128 canvas and
124 frames (latent 8x8, 37 latent frames, 207 audio latents, patch (1,2,2)),
i.e. the shape an actual generation runs, not a toy - a layout bug that only
appears once the sequence is long enough to wrap the `(1,4,4,4,4)` rotary
spacing or to collapse a duplicate sigma cannot hide at this size. A second
case adds `("first", "last")` keyframe anchors so the conditioning rows and
the pairwise-summed `"last"` anchor time are covered too.

Usage:
  python3 tools/minimaxh3_layout_dump_reference.py \\
      --out testdata/golden/minimaxh3/layout
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: E402
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402

# The geometry a real 128x128 / 124-frame `t2va` request resolves to.
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
NUM_LATENT_FRAMES = 37
NUM_AUDIO_LATENTS = 207
NUM_TEXT_TOKENS = 37
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
NUM_INFERENCE_STEPS = 20
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
KEYFRAME_NOISE_AUG = 0.999


def layout_case(tensors, prefix, keyframe_anchors):
    """One `build_packed_sequence` + full `build_row_timesteps` plan."""
    text_token_tags = torch.full((NUM_TEXT_TOKENS,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    # A keyframe request tags its vision-block rows as video, exactly as the
    # real text presentation does; keep a couple so the tag path is exercised.
    if keyframe_anchors:
        text_token_tags[3:9] = MINIMAX_H3_VIDEO_TAG

    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        PATCH_SIZE,
        AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_TAG,
        MINIMAX_H3_VIDEO_TAG,
        keyframe_anchors,
    )

    # The rotary grid is built in float64, but the transformer's own rope
    # casts it to float32 before it is ever used
    # (`MiniMaxH3RotaryPosEmbed.forward`: `position_ids.to(torch.float32)`),
    # so float32 is the value that actually reaches the model and the only
    # one worth pinning here.
    tensors[f"{prefix}_position_ids"] = position_ids.to(torch.float32).contiguous()
    tensors[f"{prefix}_token_tags"] = token_tags.to(torch.int32).contiguous()
    tensors[f"{prefix}_video_indices"] = video_indices.to(torch.int32).contiguous()
    tensors[f"{prefix}_audio_indices"] = audio_indices.to(torch.int32).contiguous()
    tensors[f"{prefix}_text_indices"] = text_indices.to(torch.int32).contiguous()
    tensors[f"{prefix}_text_token_tags"] = text_token_tags.to(torch.int32).contiguous()
    tensors[f"{prefix}_row_counts"] = torch.tensor(
        [num_condition_video_rows, num_condition_audio_rows, int(position_ids.shape[0])],
        dtype=torch.int32,
    )

    # The full per-step row-timestep plan, exactly as `MiniMaxH3SetTimestepsStep`
    # stages it: one `(timestep, timestep_indices)` pair per step, driven off
    # the two schedules zipped together.
    video_sched = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    audio_sched = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
    video_sched.set_timesteps(NUM_INFERENCE_STEPS)
    audio_sched.set_timesteps(NUM_INFERENCE_STEPS)

    unique_rows, index_rows = [], []
    for timestep, audio_timestep in zip(video_sched.timesteps, audio_sched.timesteps):
        unique_ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices,
            audio_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
            int(text_indices.numel()),
            float(timestep),
            float(audio_timestep),
            max(float(timestep), KEYFRAME_NOISE_AUG),
            1.0,
        )
        unique_rows.append(unique_ts.to(torch.float32))
        index_rows.append(ts_idx.to(torch.int32))
    # The number of DISTINCT timesteps is not the same at every step: at step 0
    # the shift maps sigma=1 to 1 under both shifts, so the video and audio
    # timesteps coincide at t=0 and the step has a single unique value, where
    # every later step has two. That is exactly the kind of per-step variation
    # a port can get wrong by baking in a constant, so the plan is dumped as a
    # ragged concatenation plus its per-step count rather than a rectangle
    # that could only be built by assuming the count is fixed.
    tensors[f"{prefix}_row_unique_timesteps"] = torch.cat(unique_rows).contiguous()
    tensors[f"{prefix}_row_unique_counts"] = torch.tensor(
        [int(u.numel()) for u in unique_rows], dtype=torch.int32
    )
    tensors[f"{prefix}_row_timestep_indices"] = torch.stack(index_rows).contiguous()


def schedule_case(tensors):
    """Both shifted-sigma schedules, plus one Euler step and one `scale_noise`."""
    for name, shift in (("video", VIDEO_SHIFT), ("audio", AUDIO_SHIFT)):
        sched = MiniMaxH3Scheduler(shift=shift)
        sched.set_timesteps(NUM_INFERENCE_STEPS)
        tensors[f"sched_{name}_sigmas"] = sched.sigmas.to(torch.float32).contiguous()
        tensors[f"sched_{name}_timesteps"] = sched.timesteps.to(torch.float32).contiguous()

        # One real Euler step at a mid-schedule index, over a seeded sample and
        # velocity: this is what catches a swapped sigma source or a flipped
        # velocity sign, which the sigma grid alone cannot.
        generator = torch.Generator().manual_seed(11 if name == "video" else 12)
        sample = torch.randn(64, generator=generator, dtype=torch.float32)
        velocity = torch.randn(64, generator=generator, dtype=torch.float32)
        step_index = len(sched.timesteps) // 2
        sched.set_begin_index(step_index)
        stepped = sched.step(velocity, sched.timesteps[step_index], sample, return_dict=False)[0]
        tensors[f"sched_{name}_step_sample"] = sample.contiguous()
        tensors[f"sched_{name}_step_velocity"] = velocity.contiguous()
        tensors[f"sched_{name}_step_index"] = torch.tensor([step_index], dtype=torch.int32)
        tensors[f"sched_{name}_step_out"] = stepped.to(torch.float32).contiguous()

    # `scale_noise` at the keyframe noise-aug level, the one place the
    # forward process is used outside the loop.
    generator = torch.Generator().manual_seed(13)
    clean = torch.randn(48, generator=generator, dtype=torch.float32)
    noise = torch.randn(48, generator=generator, dtype=torch.float32)
    scaled = MiniMaxH3Scheduler(shift=VIDEO_SHIFT).scale_noise(clean, KEYFRAME_NOISE_AUG, noise)
    tensors["scale_noise_clean"] = clean.contiguous()
    tensors["scale_noise_noise"] = noise.contiguous()
    tensors["scale_noise_out"] = scaled.to(torch.float32).contiguous()


def patchify_case(tensors):
    """The DiT-level patchify at the real latent shape, and its inverse."""
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(
        (1, LATENT_CHANNELS, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        generator=generator,
        dtype=torch.float32,
    )
    rows = patchify_video_latents(latents, PATCH_SIZE)
    tensors["patchify_latents"] = latents.contiguous()
    tensors["patchify_rows"] = rows.contiguous()

    # The inverse, transcribed from `MiniMaxH3AfterDenoiseStep.__call__`, so the
    # port's `unpatchify_video` is checked against the reference's own reshape/
    # permute rather than only against its own forward direction.
    patch_t, patch_h, patch_w = PATCH_SIZE
    back = rows.reshape(
        -1,
        NUM_LATENT_FRAMES // patch_t,
        LATENT_HEIGHT // patch_h,
        LATENT_WIDTH // patch_w,
        LATENT_CHANNELS,
        patch_t,
        patch_h,
        patch_w,
    )
    back = back.permute(0, 4, 1, 5, 2, 6, 3, 7)
    back = back.reshape(-1, LATENT_CHANNELS, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH).contiguous()
    tensors["patchify_roundtrip"] = back

    # The audio side of the same step: channel-major rows -> (2, C, T).
    audio_rows = torch.randn(
        (NUM_AUDIO_LATENTS * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
        generator=generator,
        dtype=torch.float32,
    )
    audio_latents = audio_rows.reshape(AUDIO_CHANNELS, NUM_AUDIO_LATENTS, AUDIO_LATENT_CHANNELS)
    audio_latents = audio_latents.permute(0, 2, 1).contiguous()
    tensors["audio_rows"] = audio_rows.contiguous()
    tensors["audio_latents"] = audio_latents


def save(out, name, tensors, manifest):
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {path} ({len(tensors)} tensors)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory for the golden")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tensors = {}
    layout_case(tensors, "t2va", ())
    layout_case(tensors, "fl2va", ("first", "last"))
    schedule_case(tensors)
    patchify_case(tensors)

    manifest = {}
    manifest["source"] = source_block(
        checkpoint=None,
        identity={
            "latent_height": LATENT_HEIGHT,
            "latent_width": LATENT_WIDTH,
            "num_latent_frames": NUM_LATENT_FRAMES,
            "num_audio_latents": NUM_AUDIO_LATENTS,
            "num_text_tokens": NUM_TEXT_TOKENS,
            "patch_t": PATCH_SIZE[0],
            "patch_h": PATCH_SIZE[1],
            "patch_w": PATCH_SIZE[2],
            "audio_channels": AUDIO_CHANNELS,
            "latent_channels": LATENT_CHANNELS,
            "audio_latent_channels": AUDIO_LATENT_CHANNELS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "video_shift_x1000": int(VIDEO_SHIFT * 1000),
            "audio_shift_x1000": int(AUDIO_SHIFT * 1000),
        },
    )
    save(args.out, "minimaxh3_layout.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
