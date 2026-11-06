#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for verifying ported generative
# media pipelines against their reference implementations for its clients. If
# your team needs expertise in video diffusion model inference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump the conditioned latent states of LTX-2.5 DFR's two stages.

Runs the OFFICIAL conditioning items on the reference's own initial states,
exactly as `dfr_stages._prepare_stage1_specs` / `_prepare_stage2_specs`
list them and `create_noised_state` applies them (everything up to the
noiser, which is a seeded lerp a port draws from its own generator):

  stage1   VideoGeneratedKeyframeSlots(canvas positions)
  stage2   VideoGeneratedKeyframeSlots(canvas positions, initial_keyframes)
           + VideoConditionByReferenceLatent(half-res stage-1 latent,
             downscale_factor=2, strength=1)

on a `resolve_canvas` canvas (49 frames at 24 fps: two 24-frame segments, slots
at 24 and 48) at 128x64 (stage 1 at half that). Per stage it dumps the state's
`latent`, `clean_latent`, `denoise_mask`, `keyframes_mask` and `positions`,
plus the synthetic inputs (`stage2.video_in`, `stage2.keyframes_in`,
`stage2.reference_in`) a port feeds the same items.

Usage:
  python -I tools/goldens/ltxv_dfr_conditioning_dump_reference.py \\
      --out testdata/golden/ltxv/dfr_conditioning --seed 3
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

NUM_FRAMES, HEIGHT, WIDTH, FPS = 49, 64, 128, 24.0
DOWNSCALE = 2


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
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.dfr_helpers.layout import resolve_canvas
    from ltx_pipelines.dfr_helpers.ops import conditioning_fps
    from ltx_pipelines.utils.helpers import state_with_conditionings

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    device, dtype = torch.device("cpu"), torch.float32
    frames, _segment, positions = resolve_canvas(NUM_FRAMES)
    fps = conditioning_fps(FPS)
    gen = torch.Generator().manual_seed(args.seed)

    def state(pixel_h, pixel_w, initial, conditionings):
        shape = VideoLatentShape.from_pixel_shape(VideoPixelShape(batch=1, frames=frames, height=pixel_h, width=pixel_w, fps=fps))
        tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps)
        s = tools.create_initial_state(device, dtype, initial)
        return state_with_conditionings(s, conditionings, tools), shape

    tensors = {}

    def dump(prefix, s):
        tensors[f"{prefix}.latent"] = s.latent[0]
        tensors[f"{prefix}.clean_latent"] = s.clean_latent[0]
        tensors[f"{prefix}.denoise_mask"] = s.denoise_mask.tensor[0].reshape(-1)
        tensors[f"{prefix}.keyframes_mask"] = s.keyframes_mask[0].reshape(-1)
        tensors[f"{prefix}.positions"] = s.positions[0]

    s1, shape1 = state(HEIGHT // 2, WIDTH // 2, None, [VideoGeneratedKeyframeSlots(pixel_frame_indices=positions)])
    dump("stage1", s1)

    shape2 = VideoLatentShape.from_pixel_shape(VideoPixelShape(batch=1, frames=frames, height=HEIGHT, width=WIDTH, fps=fps))
    video_in = torch.randn(shape2.to_torch_shape(), generator=gen)
    keyframes_in = torch.randn(1, shape2.channels, len(positions), shape2.height, shape2.width, generator=gen)
    reference_in = torch.randn(shape1.to_torch_shape(), generator=gen)
    s2, _ = state(HEIGHT, WIDTH, video_in, [
        VideoGeneratedKeyframeSlots(pixel_frame_indices=positions, initial_keyframes=keyframes_in),
        VideoConditionByReferenceLatent(latent=reference_in, downscale_factor=DOWNSCALE, strength=1.0),
    ])
    dump("stage2", s2)
    tensors["stage2.video_in"] = video_in[0]
    tensors["stage2.keyframes_in"] = keyframes_in[0]
    tensors["stage2.reference_in"] = reference_in[0]

    tensors = {k: t.detach().to(torch.float32).contiguous() for k, t in tensors.items()}
    path = os.path.join(args.out, "dfr_conditioning.safetensors")
    save_file(tensors, path)
    manifest = {
        "run": {"frames": frames, "requested_frames": NUM_FRAMES, "height": HEIGHT, "width": WIDTH, "fps": fps,
                "slot_positions": list(positions), "downscale": DOWNSCALE, "seed": args.seed},
        "dfr_conditioning.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                         "tensors": {k: list(t.shape) for k, t in tensors.items()}},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "source": source_block(identity={"frames": frames, "height": HEIGHT, "width": WIDTH, "slots": len(positions)}),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {path}: slots at {positions} on a {frames}-frame canvas", flush=True)


if __name__ == "__main__":
    main()
