#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for verifying ported generative
# media pipelines against their reference implementations for its clients. If
# your team needs expertise in video diffusion model inference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump LTX-2.5 DFR temporal-round goldens: the tile geometry and one tile's
conditioned state.

`temporal.json` holds, for a set of canvases and tile counts, the OFFICIAL
`TemporalTilePlan` (each tile's latent interval, pixel span, anchors and
slots), the `tile_prefix` every non-first tile resumes from, and
`conditioning_fps` at the rates rounds reach.

`temporal_tile.safetensors` holds one non-first tile of a round-1 canvas
built exactly as `dfr_stages._prepare_temporal_tile_specs` builds it - the
tile video (`_temporal_tile_video_input`: the prefix plane, then the canvas
run), the anchors at `ANCHOR_KEYFRAME_STRENGTH`, the slots seeded by
`slot_initials_from_video`, the `lead_in_carryover` pin, and the frozen
audio from `audio_latent_for_tile` - applied to the reference's own initial
state up to the noiser. Synthetic inputs (`in.*`) are dumped beside it.

Usage:
  python -I tools/goldens/ltxv_dfr_temporal_dump_reference.py \\
      --out testdata/golden/ltxv/dfr_temporal --seed 5
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# (requested canvas frames before the round, rounds so far, tiles) - two,
# three and five segments, so leftover segments and clamping both occur.
PLANS = [(49, 1, 2), (73, 1, 2), (121, 2, 4), (73, 2, 4), (49, 2, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reference-root", default=str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv"),
                    help="the LTX-2 reference checkout (default: the repo's gitignored scratchpad copy)")
    args = ap.parse_args()
    for package in ("ltx-core", "ltx-pipelines"):
        sys.path.insert(0, str(Path(args.reference_root) / "packages" / package / "src"))
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape
    from ltx_pipelines.dfr_helpers.layout import TemporalTilePlan, resolve_canvas
    from ltx_pipelines.dfr_helpers.ops import (
        ANCHOR_KEYFRAME_STRENGTH,
        audio_latent_for_tile,
        conditioning_fps,
        keyframe_conditionings_from_latents,
        lead_in_carryover,
        slot_initials_from_video,
        tile_prefix,
    )
    from ltx_pipelines.dfr_stages import _temporal_tile_video_input
    from ltx_pipelines.utils.helpers import state_with_conditionings

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    scale = 8

    plans = []
    for requested, round_idx, tiles in PLANS:
        canvas, _, positions = resolve_canvas(requested)
        frames = canvas
        for _ in range(round_idx):
            frames = 2 * (frames - 1) + 1
            positions = [2 * p for p in positions]
        plan = TemporalTilePlan(positions, frames, tiles, scale)
        entries = []
        # `run_one_temporal_round`'s plane bag: the seams, plus each tile's
        # generated slots once that tile has run.
        planes = set(positions)
        for index, tile in enumerate(plan):
            entry = {"start": tile.interval.start, "end": tile.interval.end, "pixel_start": tile.pixel_start,
                     "pixel_end": tile.pixel_end, "anchors": list(tile.anchors), "slots": list(tile.slots)}
            if index > 0:
                p = tile_prefix((tile.interval.start - 1) * scale, planes, scale)
                entry["prefix"] = {"keyframe_position": p.keyframe_position, "video_start_cell": p.video_start_cell,
                                   "cells": p.cells, "resume_pixel": p.resume_pixel}
            entries.append(entry)
            planes.update(tile.slots)
        plans.append({"requested": requested, "round": round_idx, "num_tiles": tiles, "frames": frames,
                      "seams": positions, "tiles": entries})
    fps = {str(f): conditioning_fps(float(f)) for f in (24, 30, 48, 50, 96)}

    # One tile: round 1 of a 73-frame canvas (3 segments of 24 -> seams 48, 96,
    # 144 on the 145-frame round canvas), 2 tiles, the second one.
    gen = torch.Generator().manual_seed(args.seed)
    canvas, _, positions = resolve_canvas(73)
    playback_fps = 48.0
    frames = 2 * (canvas - 1) + 1
    seams = [2 * p for p in positions]
    lh, lw, c = 2, 3, 128
    latent_t = (frames - 1) // scale + 1
    video_latent = torch.randn(1, c, latent_t, lh, lw, generator=gen)
    plan = TemporalTilePlan(seams, frames, 2, scale)
    # The seams' planes and tile 0's generated slot planes.
    plane_at = {s: torch.randn(1, c, 1, lh, lw, generator=gen) for s in sorted([*seams, *plan[0].slots])}
    tile = plan[1]
    prefix = tile_prefix((tile.interval.start - 1) * scale, plane_at, scale)
    tile_video = _temporal_tile_video_input(video_latent=video_latent, tile=tile, prefix=prefix, plane_at=plane_at, temporal_scale=scale)
    previous_latent = torch.randn(1, c, plan[0].interval.end - plan[0].interval.start, lh, lw, generator=gen)
    previous = (plan[0].interval.start, previous_latent)
    cond_fps = conditioning_fps(playback_fps)
    local_frames = tile_video.local_frames(scale)
    conditionings = []
    kept = [p for p in tile.anchors if p >= tile_video.resume_pixel]
    if kept:
        conditionings += keyframe_conditionings_from_latents(torch.cat([plane_at[p] for p in kept], dim=2),
                                                             [p - tile_video.pixel_start for p in kept], strength=ANCHOR_KEYFRAME_STRENGTH)
    slot_local = [p - tile_video.pixel_start for p in tile.slots]
    conditionings.append(VideoGeneratedKeyframeSlots(pixel_frame_indices=slot_local,
                                                     initial_keyframes=slot_initials_from_video(tile_video.tile_video, slot_local, scale)))
    conditionings.append(lead_in_carryover(previous, prefix, tile_video.tile_video[:, :, :1]))
    shape = VideoLatentShape.from_torch_shape(tile_video.tile_video.shape)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, cond_fps)
    state = state_with_conditionings(tools.create_initial_state(torch.device("cpu"), torch.float32, tile_video.tile_video), conditionings, tools)
    stage1_audio = torch.randn(1, 8, round(canvas / 24.0 * 25), 16, generator=gen)
    audio = audio_latent_for_tile(stage1_audio, pixel_start=tile_video.pixel_start, local_frames=local_frames,
                                  playback_fps=playback_fps, source_duration=canvas / 24.0, cond_fps=cond_fps)

    tensors = {
        "in.video_latent": video_latent[0], "in.previous_latent": previous_latent[0], "in.stage1_audio": stage1_audio[0],
        **{f"in.plane{s}": p[0] for s, p in plane_at.items()},
        "tile.latent": state.latent[0], "tile.clean_latent": state.clean_latent[0],
        "tile.denoise_mask": state.denoise_mask.tensor[0].reshape(-1), "tile.keyframes_mask": state.keyframes_mask[0].reshape(-1),
        "tile.positions": state.positions[0], "tile.audio": audio[0],
    }
    tensors = {k: t.detach().to(torch.float32).contiguous() for k, t in tensors.items()}
    path = os.path.join(args.out, "temporal_tile.safetensors")
    save_file(tensors, path)
    tile_run = {"requested": 73, "canvas": canvas, "round_frames": frames, "playback_fps": playback_fps, "cond_fps": cond_fps,
                "seams": seams, "tile_index": 1, "pixel_start": tile_video.pixel_start, "local_frames": local_frames,
                "anchors_kept": kept, "slots_local": slot_local, "pinned_cells": prefix.cells,
                "previous_base_cell": plan[0].interval.start, "lh": lh, "lw": lw}
    with open(os.path.join(args.out, "temporal.json"), "w") as fh:
        json.dump({"plans": plans, "conditioning_fps": fps, "tile": tile_run}, fh, indent=2)
    manifest = {
        "temporal_tile.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                      "tensors": {k: list(t.shape) for k, t in tensors.items()}},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "source": source_block(identity={"plans": len(plans), "tile_frames": local_frames}),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {len(plans)} plans, tile {tile_run}", flush=True)


if __name__ == "__main__":
    main()
