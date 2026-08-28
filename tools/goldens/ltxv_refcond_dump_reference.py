#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 IC-LoRA reference-video conditioning goldens (pure geometry, no weights).

Runs the OFFICIAL `ltx_core.conditioning.types.reference_video_cond.
VideoConditionByReferenceLatent` and `ltx_core.conditioning.types.
attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper` against a
real `VideoLatentTools.create_initial_state`, and the OFFICIAL
`ltx_pipelines.iclora_utils.downsample_mask_video_to_latent` /
`temporal_subsample`. Nothing here needs a checkpoint: the whole contract this
port has to match is token layout, RoPE position bounds, denoise/keyframe
markers and the attention cross-mask, all of which are pure geometry.

  refcond.safetensors  per case: appended positions `[3, M, 2]`, denoise_mask
                       `[N+M]`, keyframes_mask `[N+M]`, clean tail `[M, C]`,
                       the attention cross-mask `[M]`, and - for the smallest
                       case only - the FULL dense attention matrix `[N+M, N+M]`
                       so the port's factored (per-token) form can be proven to
                       reconstruct the reference's dense one exactly.
  mask_latent.safetensors  `downsample_mask_video_to_latent` cases.
  manifest.json        shapes, sha256, run params, library versions.

## Why `ltx_pipelines.iclora_utils` is loaded by hand

Importing it normally pulls `ltx_pipelines.utils.media_io`, which imports
`torchaudio`, whose native library fails to load in this environment (a
pre-existing gap this repo already records for the audio golden). The two
functions needed here touch only `torch` and `einops`, so the module is
executed from its own source with a stub standing in for that ONE unrelated
import. The function bodies that produce these goldens are the reference's
own, unmodified - only an unused import is neutralised.

## Self-validation

1. **Exact reproducibility**: every case is run twice and asserted bit-identical.
2. **Dense-vs-factored**: the reference's own dense `(N+M, N+M)` attention mask
   is reconstructed from the `[M]` cross-mask plus the documented block
   structure, and asserted EXACTLY equal - this is what licenses the port to
   store `M` values instead of `(N+M)^2`.
3. **Invariants** asserted directly against `build_attention_mask`'s docstring
   diagram: reference tokens are frozen (`denoise_mask == 1 - strength`), never
   keyframe-marked, and the base range is left untouched.

Usage:
  python tools/goldens/ltxv_refcond_dump_reference.py [--out testdata/golden/ltxv/refcond]
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
_CORE = _REFERENCE_ROOT / "packages" / "ltx-core" / "src"
_PIPE = _REFERENCE_ROOT / "packages" / "ltx-pipelines" / "src"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(_PIPE))

from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx_core.conditioning.mask_utils import build_attention_mask  # noqa: E402
from ltx_core.conditioning.types.attention_strength_wrapper import (  # noqa: E402
    ConditioningItemAttentionStrengthWrapper,
)
from ltx_core.conditioning.types.reference_video_cond import (  # noqa: E402
    VideoConditionByReferenceLatent,
)
from ltx_core.tools import VideoLatentTools  # noqa: E402
from ltx_core.types import VideoLatentShape  # noqa: E402


def _load_iclora_utils():
    """Execute the real `iclora_utils` source with its torchaudio-tainted import stubbed."""
    stub = types.ModuleType("ltx_pipelines.utils.media_io")
    for name in ("ResizeMode", "decode_video_by_frame", "is_exr_dir",
                 "load_exr_folder_conditioning_hdr", "video_preprocess"):
        setattr(stub, name, None)
    color = types.ModuleType("ltx_pipelines.utils.media_io.color_config")
    color.HDRColorSpace = None
    pkg = types.ModuleType("ltx_pipelines")
    pkg.__path__ = [str(_PIPE / "ltx_pipelines")]
    utils = types.ModuleType("ltx_pipelines.utils")
    utils.__path__ = [str(_PIPE / "ltx_pipelines" / "utils")]
    sys.modules.setdefault("ltx_pipelines", pkg)
    sys.modules.setdefault("ltx_pipelines.utils", utils)
    sys.modules["ltx_pipelines.utils.media_io"] = stub
    sys.modules["ltx_pipelines.utils.media_io.color_config"] = color
    path = _PIPE / "ltx_pipelines" / "iclora_utils.py"
    spec = importlib.util.spec_from_file_location("ltx_pipelines.iclora_utils", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ltx_pipelines.iclora_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


ICU = _load_iclora_utils()

PATCH = 1  # LTX-2.5 patchifies 1x1x1 in latent space; positions are the whole contract.
CHANNELS = 8


def make_tools(frames, height, width, fps):
    return VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=PATCH),
        target_shape=VideoLatentShape(batch=1, channels=CHANNELS, frames=frames,
                                      height=height, width=width),
        fps=fps,
    )


# (name, target f/h/w, ref f/h/w, fps, downscale, temporal_scale, strength, attn)
CASES = [
    ("plain",          (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0, None),
    ("half_res_ref",   (3, 8, 8), (3, 4, 4), 24.0, 2, 1, 1.0, None),
    ("temporal_x4",    (9, 4, 4), (3, 4, 4), 24.0, 1, 4, 1.0, None),
    ("both_scales",    (9, 8, 8), (3, 4, 4), 30.0, 2, 4, 1.0, None),
    ("partial_strength", (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 0.7, None),
    ("scalar_attn",    (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0, "scalar"),
    ("spatial_attn",   (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0, "spatial"),
]


def run_case(spec, gen):
    name, (tf, th, tw), (rf, rh, rw), fps, down, temporal, strength, attn = spec
    tools = make_tools(tf, th, tw, fps)
    state = tools.create_initial_state(device="cpu", dtype=torch.float32)
    ref = torch.randn(1, CHANNELS, rf, rh, rw, generator=gen, dtype=torch.float32)

    cond = VideoConditionByReferenceLatent(
        latent=ref, downscale_factor=down, temporal_scale_factor=temporal, strength=strength)
    cross = None
    if attn == "scalar":
        cross = 0.35
        cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=cross)
    elif attn == "spatial":
        m = torch.rand(1, rf * rh * rw, generator=gen, dtype=torch.float32)
        cross = m
        cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=m)

    out = cond.apply_to(state, tools)
    n = state.latent.shape[1]
    m = out.latent.shape[1] - n

    # --- invariants straight from the reference docstrings ---
    assert out.latent[:, n:].abs().max() == 0, "reference tokens are placeholder zeros in the noisy latent"
    assert torch.equal(out.clean_latent[:, n:, :], tools.patchifier.patchify(ref)), "clean tail is the patchified reference"
    assert torch.allclose(out.denoise_mask[:, n:], torch.full((1, m, 1), 1.0 - strength)), "frozen at 1-strength"
    assert torch.equal(out.denoise_mask[:, :n], state.denoise_mask), "base denoise_mask untouched"
    assert torch.equal(out.positions[:, :, :n], state.positions), "base positions untouched"
    if out.keyframes_mask is not None:
        assert out.keyframes_mask[:, n:].abs().max() == 0, "reference tokens are never keyframe-marked"

    res = {
        f"{name}.positions": out.positions[0, :, n:, :].contiguous(),
        f"{name}.denoise_mask": out.denoise_mask[0, :, 0].contiguous(),
        f"{name}.clean_tail": out.clean_latent[0, n:, :].contiguous(),
    }
    if out.keyframes_mask is not None:
        res[f"{name}.keyframes_mask"] = out.keyframes_mask[0, :, 0].contiguous()

    if cross is not None:
        cm = (torch.full((1, m), float(cross)) if isinstance(cross, float) else cross)
        res[f"{name}.cross_mask"] = cm[0].contiguous()
        # Self-validation 2: the dense reference mask is exactly the factored one.
        dense = build_attention_mask(existing_mask=None, num_noisy_tokens=n, num_new_tokens=m,
                                     num_existing_tokens=n, cross_mask=cm,
                                     device=torch.device("cpu"), dtype=torch.float32)
        assert torch.equal(out.attention_mask, dense), "wrapper mask != build_attention_mask"
        rebuilt = torch.zeros_like(dense)
        rebuilt[:, :n, :n] = 1.0
        rebuilt[:, n:, n:] = 1.0
        rebuilt[:, :n, n:] = cm.unsqueeze(1)
        rebuilt[:, n:, :n] = cm.unsqueeze(2)
        assert torch.equal(dense, rebuilt), "dense mask is NOT reconstructible from the M-vector"
        if name == "scalar_attn":
            res[f"{name}.dense_attention"] = dense[0].contiguous()
    return res, dict(name=name, target=[tf, th, tw], ref=[rf, rh, rw], fps=fps,
                     downscale=down, temporal_scale=temporal, strength=strength,
                     attn=attn, n=n, m=m)


MASK_CASES = [  # (name, f_pix, h_pix, w_pix, lat_f, lat_h, lat_w)
    ("m_basic", 9, 32, 32, 2, 4, 4),
    ("m_single_frame", 1, 16, 16, 1, 2, 2),
    ("m_deep", 17, 64, 32, 3, 8, 4),
    # Spatial ratios that do NOT divide evenly. `mode="area"` is torch's
    # ADAPTIVE rule - output cell `i` spans `[floor(i*H/O), ceil((i+1)*H/O))`,
    # so neighbouring cells overlap - and it degenerates to a plain box pool
    # exactly when the ratio divides, which every ratio the VAE itself produces
    # does. The three cases above therefore cannot distinguish the two rules; a
    # port that floors the cell end instead of ceiling it passes all of them.
    ("m_ragged", 9, 50, 30, 2, 4, 4),
    ("m_ragged_deep", 17, 45, 45, 3, 8, 8),
]


def run_mask_case(spec, gen):
    name, fp, hp, wp, lf, lh, lw = spec
    mask = torch.rand(1, 1, fp, hp, wp, generator=gen, dtype=torch.float32)
    out = ICU.downsample_mask_video_to_latent(
        mask=mask, target_latent_shape=VideoLatentShape(1, CHANNELS, lf, lh, lw))
    assert out.shape == (1, lf * lh * lw), out.shape
    return {f"{name}.mask": mask[0, 0].contiguous(), f"{name}.latent_mask": out[0].contiguous()}, \
        dict(name=name, pix=[fp, hp, wp], lat=[lf, lh, lw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/golden/ltxv/refcond")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tensors, meta = {}, []
    for spec in CASES:
        a, info = run_case(spec, torch.Generator().manual_seed(1234))
        b, _ = run_case(spec, torch.Generator().manual_seed(1234))
        for k in a:
            assert torch.equal(a[k], b[k]), f"{k} is not reproducible"
        tensors.update(a)
        meta.append(info)

    mtensors, mmeta = {}, []
    for spec in MASK_CASES:
        a, info = run_mask_case(spec, torch.Generator().manual_seed(99))
        b, _ = run_mask_case(spec, torch.Generator().manual_seed(99))
        for k in a:
            assert torch.equal(a[k], b[k]), f"{k} is not reproducible"
        mtensors.update(a)
        mmeta.append(info)

    save_file(tensors, str(out / "refcond.safetensors"))
    save_file(mtensors, str(out / "mask_latent.safetensors"))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    from golden_source import source_block  # noqa: E402  (tools/goldens dir is on sys.path[0])

    (out / "manifest.json").write_text(json.dumps({
        "modules": "ltx_core.conditioning.types.reference_video_cond / attention_strength_wrapper"
                  " + ltx_pipelines.iclora_utils (real sources, live run)",
        "torch": torch.__version__,
        "patch_size": PATCH,
        "channels": CHANNELS,
        "cases": meta,
        "mask_cases": mmeta,
        "source": source_block(
            checkpoint="Lightricks/LTX-2.5",
            identity={"patch_size": PATCH, "channels": CHANNELS},
        ),
        "files": {f: {"sha256": sha(out / f),
                      "tensors": {k: list(v.shape) for k, v in
                                  (tensors if f == "refcond.safetensors" else mtensors).items()}}
                  for f in ("refcond.safetensors", "mask_latent.safetensors")},
    }, indent=2) + "\n")
    print(f"wrote {len(tensors)} + {len(mtensors)} tensors to {out}")


if __name__ == "__main__":
    main()
