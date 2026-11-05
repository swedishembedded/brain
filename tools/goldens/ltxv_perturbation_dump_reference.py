#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump goldens for LTX-2.5's guidance PERTURBATIONS on the tiny AV DiT.

The multimodal guider runs up to four forwards per step (`_guided_denoise`):
the conditional one, a negative-prompt one, an STG pass that bypasses
self-attention in chosen blocks, and an isolated-modality pass that cuts both
audio<->video cross-attention directions in every block. This dumper rebuilds
the tiny AV model of `ltxv_av_dit_dump_reference.py` from that golden's own
weights and inputs - against the CURRENT `ltx_core` API - checks that its
plain forward still reproduces the stored output exactly, and then dumps the
two perturbed forwards:

  ltxv_perturbation.safetensors
    stg.video.out, stg.audio.out   self-attention bypassed in block 1, both streams
    mod.video.out, mod.audio.out   A2V and V2A cut in every block

Usage:
  LTXV_REFERENCE_ROOT=/path/to/LTX-2 python3 tools/goldens/ltxv_perturbation_dump_reference.py \\
      --golden testdata/golden/ltxv/av_dit --out testdata/golden/ltxv/perturbation
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_REFERENCE_ROOT = Path(os.environ.get("LTXV_REFERENCE_ROOT", str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_source import source_block  # noqa: E402
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType  # noqa: E402
from ltx_core.model.transformer.modality import Modality, TokenTimesteps  # noqa: E402
from ltx_core.model.transformer.model import LTXModel  # noqa: E402
from ltxv_av_dit_dump_reference import TINY_CONFIG  # noqa: E402


def modality(fx, prefix, keyframes):
    return Modality(
        latent=fx[f"{prefix}.latent"][None],
        sigma=fx[f"{prefix}.sigma"],
        timesteps=TokenTimesteps(fx[f"{prefix}.timesteps"][None]),
        positions=fx[f"{prefix}.positions"][None],
        context=fx[f"{prefix}.context"][None],
        keyframes_mask=fx[f"{prefix}.keyframes_mask"][None] if keyframes else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="the av_dit golden directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    fx = load_file(os.path.join(args.golden, "av_dit_tiny.safetensors"))
    model = LTXModel(**TINY_CONFIG).eval()
    missing, unexpected = model.load_state_dict(load_file(os.path.join(args.golden, "av_dit_tiny_weights.safetensors")), strict=False)
    assert not missing and not unexpected, f"weights do not cover the model: missing {missing}, unexpected {unexpected}"
    video, audio = modality(fx, "video", True), modality(fx, "audio", False)

    out_v, out_a = model(video=video, audio=audio, perturbations=None)
    d = max((out_v[0] - fx["video.out"]).abs().max().item(), (out_a[0] - fx["audio.out"]).abs().max().item())
    print(f"self-validate plain forward vs the stored golden: max abs {d:.3e}", flush=True)
    assert d < 1e-5, "the rebuilt model must reproduce the stored forward"

    passes = {
        "stg": PerturbationConfig([
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[1]),
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=[1]),
        ]),
        "mod": PerturbationConfig([
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ]),
    }
    tensors = {}
    for name, config in passes.items():
        batched = BatchedPerturbationConfig([config], num_blocks=TINY_CONFIG["num_layers"], device=out_v.device, dtype=out_v.dtype)
        p_v, p_a = model(video=video, audio=audio, perturbations=batched)
        assert (p_v - out_v).abs().max() > 0 and (p_a - out_a).abs().max() > 0, f"the {name} pass must change both streams"
        tensors[f"{name}.video.out"] = p_v[0].contiguous()
        tensors[f"{name}.audio.out"] = p_a[0].contiguous()

    path = os.path.join(args.out, "ltxv_perturbation.safetensors")
    save_file(tensors, path)
    manifest = {
        "ltxv_perturbation.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "tensors": {k: list(v.shape) for k, v in tensors.items()}},
        "source": source_block(checkpoint=None, files=[os.path.join(args.golden, "av_dit_tiny.safetensors")], identity={"num_layers": TINY_CONFIG["num_layers"], "stg_block": 1}),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote", {k: list(v.shape) for k, v in tensors.items()})


if __name__ == "__main__":
    main()
