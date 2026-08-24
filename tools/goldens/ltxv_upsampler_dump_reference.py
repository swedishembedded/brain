#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 latent-upscaler reference goldens (spatial x2 + temporal x2).

Runs the OFFICIAL `ltx_core.model.upsampler.model.LatentUpsampler` (CPU, fp32)
with the REAL `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` and
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` weights on a small
deterministic synthetic latent, dumping every block boundary:

  spatial.safetensors    latent -> initial stage -> res_blocks taps ->
                          upsampler stage -> post_upsample_res_blocks taps ->
                          final_conv (= model output)
  temporal.safetensors   same boundaries for the temporal upscaler
  manifest.json           shapes, sha256 per file, run params, versions

Both real checkpoints carry `_class_name: "LatentUpsampler"` with `dims: 3` -
`initial_conv`/`res_blocks`/`post_upsample_res_blocks`/`final_conv` are ALL
real `nn.Conv3d`, never `nn.Conv2d`, for both. Only the `upsampler` middle
stage differs: the spatial (x2) checkpoint has `spatial_upsample=True,
temporal_upsample=False, rational_resampler=False`, whose non-rational branch
is a per-frame `nn.Conv2d` + `PixelShuffleND(2)` reached through a
`rearrange('b c f h w -> (b f) c h w')` / `rearrange('(b f) c h w -> b c f h
w')` pair in `LatentUpsampler.forward`; the temporal (x2) checkpoint has
`spatial_upsample=False, temporal_upsample=True` (its own `rational_resampler:
true` is dead - the `elif temporal_upsample` branch never looks at it), whose
`upsampler` is a real `nn.Conv3d` + `PixelShuffleND(1)` with NO reshape, plus
a post-shuffle `x[:, :, 1:, :, :]` frame drop.

## The `(b f) c h w` reshape is a TRANSPOSE, not a no-op, for tap comparison

`rearrange('b c f h w -> (b f) c h w')` with `b=1` produces a `[F, C, H, W]`
tensor - frame OUTER, channel INNER. The Rust port never leaves the `[C, T, H,
W]` (channel outer) representation the rest of this model uses (a per-frame
`kt=1` Conv3d computes the identical per-frame result without ever folding
frames into a batch axis). So every tap taken from INSIDE the spatial
checkpoint's `self.upsampler` Sequential is explicitly rearranged back to `[C,
F, H, W]` before being saved here - see `_watch_reshaped_upsampler` - so the
golden file is apples-to-apples with the Rust builder's own layout throughout,
never `[F, C, H, W]`. The temporal checkpoint's `upsampler` needs no such
treatment since it never leaves `[B, C, F, H, W]`.

## Self-validation inside the dumper

1. **Fresh-module determinism**: each of the two models is built and loaded a
   SECOND time from scratch and the whole forward repeated; asserted
   bit-identical (no dropout/randomness at `eval()`).
2. **`GroupNorm` convention read off the real built module, not assumed**:
   `initial_norm.num_groups == 32` and `initial_norm.eps == 1e-5` (torch's
   OWN default - `LatentUpsampler.__init__` never passes an explicit `eps`,
   unlike this checkpoint family's video/audio VAEs which both use a
   zero-parameter `PixelNorm` at `1e-8`/`1e-6`, a completely different call
   site) are asserted for every checkpoint dumped here.

Usage:
  python tools/goldens/ltxv_upsampler_dump_reference.py \\
      --spatial-weights /path/to/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \\
      --temporal-weights /path/to/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \\
      --out testdata/golden/ltxv/upsampler [--frames 2 --size 4 --seed 42]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored), never a
# machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.upsampler.model import LatentUpsampler  # noqa: E402
from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator  # noqa: E402
from ltx_core.model.upsampler.spatial_rational_resampler import SpatialRationalResampler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

_CLASS_NAME = "LatentUpsampler"


def det_latent(c, t, h, w, seed):
    """Deterministic synthetic latent, shape (1, c, t, h, w) - same construction
    style as `ltxv_vae_dump_reference.py`'s `det_video` (a smooth deterministic
    signal, no cross-architecture meaning intended)."""
    g = torch.Generator().manual_seed(seed)
    ts = torch.linspace(0.0, 1.0, t).view(1, t, 1, 1)
    ys = torch.linspace(0.0, 3.14159, h).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 6.28318, w).view(1, 1, 1, w)
    cs = torch.linspace(0.0, 1.0, c).view(c, 1, 1, 1)
    v = torch.sin(xs + ys + 3.0 * ts + 5.0 * cs) + 0.5 * torch.cos(2.0 * cs - ts)
    v = v + 0.05 * torch.randn(v.shape, generator=g)
    return v.unsqueeze(0).contiguous()


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}"
                                        for k, v in sorted(tensors.items())), flush=True)


def agree(name, a, b, tol=0.0):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} "
          f"= {rel:.2e} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{name}: disagree by {rel:.3e} relative"


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module):
        def hook(_m, _i, o):
            self.acc[name] = o.detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def watch_reshaped(self, name, module, b, f):
        """Like `watch`, but for a submodule that runs on the `(b f) c h w`
        batch-folded view - rearranges the captured output back to `b c f h w`
        before storing it, so every tap in this file is `[C, T, H, W]`-shaped
        (channel outer), matching the Rust builder's own layout. See this
        module's docstring."""
        def hook(_m, _i, o):
            self.acc[name] = rearrange(o.detach().clone(), "(b f) c h w -> b c f h w", b=b, f=f)
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_model(weights_path):
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(weights_path)
    class_name = metadata.get("config", {}).get("_class_name")
    assert class_name == _CLASS_NAME, f"expected {_CLASS_NAME!r}, got {class_name!r}"

    model = LatentUpsamplerConfigurator.from_metadata(metadata)
    sd = loader.load(weights_path, None)
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd.sd.items()}, strict=True)
    model.eval().requires_grad_(False)
    return model, metadata.get("config", {})


def run(model, latent, b, f):
    """One forward with every block boundary tapped."""
    taps = Taps()
    taps.watch("initial_conv", model.initial_conv)
    taps.watch("initial_norm", model.initial_norm)
    taps.watch("initial_activation", model.initial_activation)
    for i, block in enumerate(model.res_blocks):
        taps.watch(f"res_blocks.{i}", block)

    if model.spatial_upsample and not model.temporal_upsample and not isinstance(model.upsampler, SpatialRationalResampler):
        # The non-rational spatial-only branch: `self.upsampler` runs on the
        # `(b f) c h w` batch-folded view - rearrange its tap back to `b c f h
        # w` (see `Taps.watch_reshaped`'s doc).
        taps.watch_reshaped("upsampler", model.upsampler, b, f)
    else:
        taps.watch("upsampler", model.upsampler)

    for i, block in enumerate(model.post_upsample_res_blocks):
        taps.watch(f"post_upsample_res_blocks.{i}", block)
    taps.watch("final_conv", model.final_conv)

    out = model(latent)
    taps.close()
    return out, dict(taps.acc)


def dump_one(label, weights_path, out_dir, frames, size, seed, manifest):
    print(f"\n=== {label} ({weights_path}) ===", flush=True)
    model, cfg = build_model(weights_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"built {label} ({n_params} params): {cfg}", flush=True)

    assert model.initial_norm.num_groups == 32, f"{label}: initial_norm has {model.initial_norm.num_groups} groups, expected 32"
    assert abs(model.initial_norm.eps - 1e-5) < 1e-12, f"{label}: initial_norm.eps={model.initial_norm.eps}, expected torch's default 1e-5"
    assert model.dims == 3, f"{label}: dims={model.dims}, expected 3 (both real checkpoints)"

    latent = det_latent(model.in_channels, frames, size, size, seed)
    b, _, f = latent.shape[0], latent.shape[1], latent.shape[2]

    out, taps = run(model, latent, b, f)

    # ---- self-validation: fresh module instantiation, bit-identical -------
    model2, _ = build_model(weights_path)
    out2, taps2 = run(model2, latent, b, f)
    agree(f"{label}: fresh-instantiation output", out2, out, tol=0.0)
    for k in taps:
        agree(f"{label}: fresh-instantiation tap {k}", taps2[k], taps[k], tol=0.0)
    del model2, taps2

    tensors = {"input": latent[0], "output": out[0]}
    for k, v in taps.items():
        tensors[f"tap_{k}"] = v[0]

    save(out_dir, f"{label}.safetensors", tensors, manifest)
    manifest.setdefault("run", {}).setdefault("shapes", {})[label] = {
        "input": list(latent.shape), "output": list(out.shape),
        "in_channels": model.in_channels, "mid_channels": model.mid_channels,
        "num_blocks_per_stage": model.num_blocks_per_stage,
        "spatial_upsample": model.spatial_upsample, "temporal_upsample": model.temporal_upsample,
    }
    # Returned so `main` can put both towers' channel counts in one `source`
    # identity - the model object itself is local to this function.
    return int(model.in_channels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial-weights", required=True, help="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
    ap.add_argument("--temporal-weights", required=True, help="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=2, help="synthetic latent frame count")
    ap.add_argument("--size", type=int, default=4, help="synthetic latent H=W")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = {"run": {"seed": args.seed, "size": args.size, "frames": args.frames,
                        "spatial_weights": os.path.abspath(args.spatial_weights),
                        "temporal_weights": os.path.abspath(args.temporal_weights)},
                "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}}

    spatial_in = dump_one("spatial", args.spatial_weights, args.out, args.frames, args.size, args.seed, manifest)
    temporal_in = dump_one("temporal", args.temporal_weights, args.out, args.frames, args.size, args.seed, manifest)

    # Two checkpoints in one manifest, so the identity names BOTH towers'
    # channel counts - a dump that paired the spatial upscaler with the
    # temporal one's golden would otherwise look well-formed.
    # `hash_files=False`: both are large and this dump is a couple of forwards.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.spatial_weights, args.temporal_weights],
        hash_files=False,
        identity={"spatial_in_channels": spatial_in, "temporal_in_channels": temporal_in},
    )

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
