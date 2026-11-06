#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for verifying ported generative
# media pipelines against their reference implementations for its clients. If
# your team needs expertise in video diffusion model inference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump the LTX-2.5 DFR spatial epilogue's tiled-diffusion goldens.

Builds the epilogue's tile grid exactly as `_run_epilogue_windows` does
(`epilogue_tile_count` per spatial axis, `EPILOGUE_SPATIAL_OVERLAP`, one
temporal tile, `clamp_tile_to_latent`) over a conditioned state - a keyframe
guide and a half-resolution reference latent appended to a 3 x 40 x 4 cell
video - and runs the OFFICIAL `VideoModalityTilingHelper` on it: each tile's
kept token indices and normalised positions, and the blend of a synthetic
per-tile prediction into the full sequence.

Usage:
  python -I tools/goldens/ltxv_dfr_tiling_dump_reference.py \\
      --out testdata/golden/ltxv/dfr_tiling --seed 9
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FRAMES, CELLS_H, CELLS_W, FPS = 17, 40, 4, 24.0


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
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.modality_tiling import VideoModalityTilingHelper
    from ltx_core.tiling import DimensionTilingConfig, TileCountConfig
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.dfr_helpers.ops import (
        EPILOGUE_KEYFRAME_STRENGTH,
        EPILOGUE_SPATIAL_OVERLAP,
        clamp_tile_to_latent,
        epilogue_tile_count,
        keyframe_conditionings_from_latents,
    )
    from ltx_pipelines.utils.helpers import modality_from_latent_state, state_with_conditionings

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(args.seed)
    c = 128
    shape = VideoLatentShape.from_pixel_shape(VideoPixelShape(batch=1, frames=FRAMES, height=CELLS_H * 32, width=CELLS_W * 32, fps=FPS))
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, FPS)
    latent = torch.randn(shape.to_torch_shape(), generator=gen)
    keyframe = torch.randn(1, c, 1, shape.height, shape.width, generator=gen)
    reference = torch.randn(1, c, shape.frames, shape.height // 2, shape.width // 2, generator=gen)
    conditionings = keyframe_conditionings_from_latents(keyframe, [8], strength=EPILOGUE_KEYFRAME_STRENGTH)
    conditionings.append(VideoConditionByReferenceLatent(latent=reference, downscale_factor=2, strength=1.0))
    state = state_with_conditionings(tools.create_initial_state(torch.device("cpu"), torch.float32, latent), conditionings, tools)
    state = state.__class__(**{**state.__dict__, "denoise_mask": state.denoise_mask.compact(torch.device("cpu"))})
    modality = modality_from_latent_state(state, torch.zeros(1, 4, 8), torch.tensor([0.5]))

    tiling = TileCountConfig(
        frames=DimensionTilingConfig(1, overlap=0),
        height=DimensionTilingConfig(epilogue_tile_count(shape.height), EPILOGUE_SPATIAL_OVERLAP),
        width=DimensionTilingConfig(epilogue_tile_count(shape.width), EPILOGUE_SPATIAL_OVERLAP),
    )
    tiling = clamp_tile_to_latent(tiling, (shape.frames, shape.height, shape.width))
    helper = VideoModalityTilingHelper(tiling, tools)
    tensors = {"positions": modality.positions[0], "keyframes_mask": modality.keyframes_mask[0].reshape(-1)}
    tiles = []
    output = None
    for index, tile in enumerate(helper.tiles):
        tiled, context = helper.tile_modality(modality, tile)
        # A deterministic stand-in for the model's prediction of this tile.
        prediction = torch.randn(1, tiled.latent.shape[1], c, generator=gen)
        tensors[f"tile{index}.prediction"] = prediction[0]
        tensors[f"tile{index}.keep"] = context.keep_indices.to(torch.float32)
        tensors[f"tile{index}.positions"] = tiled.positions[0]
        tensors[f"tile{index}.keyframes_mask"] = tiled.keyframes_mask[0].reshape(-1)
        output = helper.blend(prediction, tile, context, output)
        tiles.append({"in_coords": [[s.start, s.stop] for s in tile.in_coords]})
    tensors["blended"] = output[0]
    tensors = {k: t.detach().to(torch.float32).contiguous() for k, t in tensors.items()}
    save_file(tensors, os.path.join(args.out, "dfr_tiling.safetensors"))
    with open(os.path.join(args.out, "dfr_tiling.json"), "w") as fh:
        json.dump({"frames": FRAMES, "cells": [shape.frames, shape.height, shape.width], "fps": FPS, "tiles": tiles,
                   "tile_counts": [tiling.frames.num_tiles, tiling.height.num_tiles, tiling.width.num_tiles]}, fh, indent=2)
    print(f"wrote {args.out}: {len(tiles)} tiles {tiles}", flush=True)


if __name__ == "__main__":
    main()
