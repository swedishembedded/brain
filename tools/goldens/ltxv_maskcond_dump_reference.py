#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 masked-conditioning goldens (pure geometry/algebra, no weights).

Runs the OFFICIAL `ltx_core.conditioning.types.mask_cond.VideoConditionByMask`
against a real `VideoLatentTools.create_initial_state`, optionally composed on
top of the OFFICIAL `ltx_core.conditioning.types.latent_cond.
VideoConditionByLatentIndex`, and then the OFFICIAL
`ltx_core.components.noisers.GaussianNoiser`. Nothing here needs a checkpoint:
the whole contract this port has to match is the two blends
`VideoConditionByMask.apply_to` performs plus the initial latent the noiser
derives from them.

  maskcond.safetensors  per case: the unpatchified mask `[F, H, W]` and its
                        patchified `[N]` form, the conditioning tokens `[N, C]`,
                        the state's clean latent / denoise mask BEFORE and AFTER
                        the item, the noise draw `[N, C]`, and the noiser's
                        initial latent `[N, C]`.
  manifest.json         shapes, sha256, run params, library versions.

## Why the noise tensor is dumped rather than re-derived

`GaussianNoiser` samples internally from a `torch.Generator`. Re-seeding a
fresh generator with the same seed reproduces that draw exactly, so the noise
is dumped alongside the noised result and the port is handed the same numbers
instead of being asked to reimplement torch's normal RNG. The dumper asserts
the re-drawn tensor is the one the noiser actually used (self-validation 4).

## Self-validation

1. **Exact reproducibility**: every case is run twice and asserted bit-identical.
2. **Mask token order**: `patchify(mask.unsqueeze(1))[0, :, 0]` is asserted
   EXACTLY equal to `mask.reshape(-1)`. That is what licenses the port to treat
   a `[F, H, W]` C-order latent mask as the token vector directly, with no
   permutation, at the patch size LTX-2.5 uses.
3. **Broadcast-vs-dense**: the `[N, 1]` mask broadcast the reference relies on
   is asserted to give EXACTLY the same `[N, C]` result as an explicitly
   expanded `[N, C]` mask. That is what licenses the port to store `N` weights
   instead of `N * C`.
4. **Polarity and strength invariants**, straight from the class docstring:
   `mask == 1` is the CONDITIONING position (clean content, `denoise_mask ==
   1 - strength`, and at `strength == 1` an initial latent that is bit-exactly
   the conditioning latent); `mask == 0` leaves both tensors untouched.

Usage:
  python tools/goldens/ltxv_maskcond_dump_reference.py [--out testdata/golden/ltxv/maskcond]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex  # noqa: E402
from ltx_core.conditioning.types.mask_cond import VideoConditionByMask  # noqa: E402
from ltx_core.tools import VideoLatentTools  # noqa: E402
from ltx_core.types import VideoLatentShape  # noqa: E402

PATCH = 1  # LTX-2.5 patchifies 1x1x1 in latent space.
CHANNELS = 8
FPS = 24.0
NOISE_SEED = 20260827


def make_tools(frames, height, width):
    return VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=PATCH),
        target_shape=VideoLatentShape(batch=1, channels=CHANNELS, frames=frames,
                                      height=height, width=width),
        fps=FPS,
    )


def build_mask(kind, f, h, w, gen):
    """The mask a case conditions with, `[1, F, H, W]`, in the class's own polarity.

    `1` = conditioning position (kept clean, excluded from denoising),
    `0` = generated position. A character swap therefore masks the BACKGROUND
    at 1 and the character region at 0.
    """
    if kind == "zeros":
        return torch.zeros(1, f, h, w)
    if kind == "ones":
        return torch.ones(1, f, h, w)
    if kind == "binary_half":
        # A spatial region held over the whole clip: the right half of the grid
        # is the conditioning region, the left half regenerates.
        m = torch.zeros(1, f, h, w)
        m[..., w // 2:] = 1.0
        return m
    if kind == "temporal":
        # A mask that VARIES per latent frame - the thing the `[B, F, H, W]`
        # shape exists for, and the case a per-frame tracker produces.
        m = torch.zeros(1, f, h, w)
        for fi in range(f):
            m[:, fi, :, : max(1, (fi + 1) * w // (f + 1))] = 1.0
        return m
    if kind == "fractional":
        # What a real pixel-space mask downsampled to the latent grid actually
        # looks like: soft edges, values strictly inside (0, 1).
        return torch.rand(1, f, h, w, generator=gen)
    raise ValueError(kind)


# (name, (f, h, w), mask kind, strength, base, noise_scale)
CASES = [
    ("binary_half", (3, 4, 6), "binary_half", 1.0, "zeros", 1.0),
    ("temporal_varying", (4, 4, 4), "temporal", 1.0, "zeros", 1.0),
    ("partial_strength", (3, 4, 6), "binary_half", 0.6, "zeros", 1.0),
    ("fractional", (3, 4, 6), "fractional", 1.0, "zeros", 1.0),
    ("fractional_partial", (2, 4, 4), "fractional", 0.35, "zeros", 1.0),
    ("all_zero", (2, 4, 4), "zeros", 1.0, "zeros", 1.0),
    ("all_one", (2, 4, 4), "ones", 1.0, "zeros", 1.0),
    ("over_existing", (3, 4, 6), "fractional", 1.0, "latent", 1.0),
    ("after_latent_index", (3, 4, 6), "binary_half", 0.8, "latent_index", 1.0),
    ("partial_noise_scale", (2, 4, 4), "binary_half", 1.0, "zeros", 0.7),
]


def run_case(spec, gen):
    name, (f, h, w), kind, strength, base, noise_scale = spec
    tools = make_tools(f, h, w)

    # A non-zero starting latent is what makes the reference's `clean_latent *
    # inv` term observable: with the default all-zero state that term vanishes
    # and a port that dropped it would still match.
    initial = None
    if base in ("latent", "latent_index"):
        initial = torch.randn(1, CHANNELS, f, h, w, generator=gen)
    state = tools.create_initial_state(device="cpu", dtype=torch.float32, initial_latent=initial)

    # Likewise for `denoise_mask * inv`: the default mask is all ones, so a port
    # that wrote `1 - strength * m` would match everywhere. Composing a real
    # conditioning item first makes the base mask non-uniform.
    if base == "latent_index":
        prior = VideoConditionByLatentIndex(
            latent=torch.randn(1, CHANNELS, 1, h, w, generator=gen), strength=0.9, latent_idx=1)
        state = prior.apply_to(state, tools)

    base_clean = state.clean_latent.clone()
    base_denoise = state.denoise_mask.clone()
    base_latent = state.latent.clone()

    cond_latent = torch.randn(1, CHANNELS, f, h, w, generator=gen)
    mask = build_mask(kind, f, h, w, gen)

    item = VideoConditionByMask(latent=cond_latent, mask=mask, strength=strength)
    out = item.apply_to(state, tools)

    n = out.clean_latent.shape[1]
    assert n == f * h * w, (n, f * h * w)
    mask_tokens = tools.patchifier.patchify(mask.unsqueeze(1))
    cond_tokens = tools.patchifier.patchify(cond_latent)

    # --- 2. mask token order is a plain C-order flatten at this patch size ---
    assert torch.equal(mask_tokens[0, :, 0], mask.reshape(-1)), \
        "patchified mask is NOT the C-order flatten of [F, H, W]"
    assert mask_tokens.shape == (1, n, 1), mask_tokens.shape

    # --- 3. the [N, 1] broadcast is exactly an expanded [N, C] mask ---
    dense = mask_tokens.expand(1, n, CHANNELS)
    assert torch.equal(out.clean_latent, base_clean * (1 - dense) + cond_tokens * dense), \
        "the [N, 1] broadcast is not reproducible from an expanded [N, C] mask"

    # --- 4. polarity / strength invariants from the class docstring ---
    m0 = (mask_tokens[0, :, 0] == 0.0)
    assert torch.equal(out.clean_latent[0][m0], base_clean[0][m0]), "mask=0 must leave clean_latent alone"
    assert torch.equal(out.denoise_mask[0][m0], base_denoise[0][m0]), "mask=0 must leave denoise_mask alone"
    m1 = (mask_tokens[0, :, 0] == 1.0)
    if m1.any():
        assert torch.equal(out.clean_latent[0][m1], cond_tokens[0][m1]), "mask=1 must take the conditioning latent"
        assert torch.allclose(out.denoise_mask[0][m1], torch.full_like(out.denoise_mask[0][m1], 1.0 - strength)), \
            "mask=1 must set denoise_mask to 1 - strength"

    # --- the noiser, live, plus the same draw re-taken for the golden ---
    noiser = GaussianNoiser(torch.Generator().manual_seed(NOISE_SEED))
    noised = noiser(out, noise_scale)
    noise = torch.randn(*out.latent.shape, dtype=out.latent.dtype,
                        generator=torch.Generator().manual_seed(NOISE_SEED))
    want = torch.lerp(out.clean_latent.float(), torch.lerp(out.latent.float(), noise.float(), noise_scale),
                      out.denoise_mask)
    assert torch.equal(noised.latent, want), "the re-drawn noise is not the draw the noiser used"
    if strength == 1.0 and m1.any():
        assert torch.equal(noised.latent[0][m1], cond_tokens[0][m1]), \
            "at strength 1.0 a conditioning position starts BIT-EXACTLY at the conditioning latent"

    res = {
        f"{name}.mask": mask[0].contiguous().clone(),
        f"{name}.mask_tokens": mask_tokens[0, :, 0].contiguous().clone(),
        f"{name}.cond_tokens": cond_tokens[0].contiguous().clone(),
        f"{name}.base_clean": base_clean[0].contiguous().clone(),
        f"{name}.base_denoise_mask": base_denoise[0, :, 0].contiguous().clone(),
        f"{name}.base_latent": base_latent[0].contiguous().clone(),
        f"{name}.clean": out.clean_latent[0].contiguous().clone(),
        f"{name}.denoise_mask": out.denoise_mask[0, :, 0].contiguous().clone(),
        f"{name}.noise": noise[0].contiguous().clone(),
        f"{name}.initial_latent": noised.latent[0].contiguous().clone(),
    }
    return res, dict(name=name, latent=[f, h, w], mask=kind, strength=strength,
                     base=base, noise_scale=noise_scale, tokens=n, channels=CHANNELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/golden/ltxv/maskcond")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tensors, meta = {}, []
    for spec in CASES:
        a, info = run_case(spec, torch.Generator().manual_seed(4321))
        b, _ = run_case(spec, torch.Generator().manual_seed(4321))
        for k in a:
            assert torch.equal(a[k], b[k]), f"{k} is not reproducible"
        tensors.update(a)
        meta.append(info)

    save_file(tensors, str(out / "maskcond.safetensors"))
    sha = hashlib.sha256((out / "maskcond.safetensors").read_bytes()).hexdigest()

    from golden_source import source_block  # noqa: E402  (tools/goldens dir is on sys.path[0])

    (out / "manifest.json").write_text(json.dumps({
        "modules": "ltx_core.conditioning.types.mask_cond.VideoConditionByMask"
                  " + latent_cond.VideoConditionByLatentIndex + components.noisers.GaussianNoiser"
                  " (real sources, live run)",
        "torch": torch.__version__,
        "patch_size": PATCH,
        "channels": CHANNELS,
        "fps": FPS,
        "noise_seed": NOISE_SEED,
        "cases": meta,
        "source": source_block(
            checkpoint="Lightricks/LTX-2.5",
            identity={"patch_size": PATCH, "channels": CHANNELS},
        ),
        "files": {"maskcond.safetensors": {
            "sha256": sha,
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
        }},
    }, indent=2) + "\n")
    print(f"wrote {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main()
