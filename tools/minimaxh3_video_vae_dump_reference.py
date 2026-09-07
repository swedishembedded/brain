#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump real-weight numeric parity goldens for MiniMax-H3's video VAE.

Like `minimaxh3_dit_dump_reference.py`, this dumper needs no MiniMax-H3
checkpoint at all: it builds the REAL, installed `diffusers==0.40.0`
`AutoencoderKLMiniMaxH3` class (`autoencoder_kl_minimax_h3.py`) at a TINY
config matching `crate::video_vae::VideoVaeConfig::tiny()`'s own proportions
(`block_out_channels` at the real 1:2:2:4:4:8 stage-width ratio, BOTH
downsample-factor tuples and `clip_length`/`token_drop` kept LITERAL - see
that constructor's own doc for why), with small seeded random weights (this
class's own default init, made reproducible by seeding torch's global RNG
before construction).

`vae.use_tiling = False` is set explicitly - this port's `encode_clip`/
`decode_clip` are the reference's untiled path exactly (spatial tiling is
out of scope this pass, see `crate::video_vae`'s own module doc), so the
comparison below is apples to apples, not a hidden mismatch against a tiled
reference.

Dumps every weight (`state_dict()` - PyTorch's own dotted keys ARE
`crate::video_vae::VideoVaeConfig::tensor_manifest`'s own tensor names, so
nothing here renames anything), plus:

  - a SINGLE-CLIP round trip (`num_frames == clip_length`, exactly one
    `_encode_clip`/`_decode_clip` call, no outer multi-chunk machinery):
    `tap_clip_moments` = `vae._encode_clip(x)`, `tap_clip_pixels` =
    `vae._decode_clip(posterior.mode())` - matching
    `crate::video_vae::encode_clip`/`decode_clip` exactly.
  - a TWO-CLIP round trip (`num_frames == 2*clip_length`) through the PUBLIC
    `vae.encode(x)`/`vae.decode(z)`, which drives `_encode`'s multi-chunk
    concatenation + `token_drop` and `_decode`'s chunk/pad/blend arithmetic
    at a NON-degenerate `num_chunks` (this config's own `num_chunks` formula
    lands at 1 for a two-clip input's post-`token_drop` latent length, not
    the `num_chunks=0` edge case a single clip's worth would hit - see
    `crate::video_vae`'s own module doc) - matching `crate::video_vae::
    encode`/`decode`'s own outer chunk orchestration.

Usage:
  /home/user/.venv/bin/python3 tools/minimaxh3_video_vae_dump_reference.py \\
      --out testdata/golden/minimaxh3/video_vae_tiny [--seed 5]
"""

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3  # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

# Matches crate::video_vae::VideoVaeConfig::tiny() field-for-field.
TINY_CFG = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=(8, 16, 16, 32, 32, 64),
    layers_per_block=2,
    spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
    temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
    norm_num_groups=2,
    norm_eps=1e-6,
    spatial_padding_mode="reflect",
    decoder_num_layers=2,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=4,
    decoder_ffn_mult=4,
    decoder_rope_theta=100.0,
    decoder_rope_dim_ratio=0.75,
    decoder_norm_eps=1e-5,
    clip_length=17,
    token_drop=3,
    latents_mean=(0.0,) * 4,
    latents_std=(1.0,) * 4,
)

HEIGHT = 32
WIDTH = 32


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: {len(tensors)} tensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    torch.manual_seed(args.seed)
    model = AutoencoderKLMiniMaxH3(**TINY_CFG)
    model.eval()
    model.use_tiling = False
    print(f"built AutoencoderKLMiniMaxH3 ({sum(p.numel() for p in model.parameters())} params), use_tiling={model.use_tiling}", flush=True)
    print(f"  spatial_compression_ratio={model.spatial_compression_ratio} temporal_compression_ratio={model.temporal_compression_ratio}", flush=True)
    print(f"  tokens_chunk_size={model.tokens_chunk_size} frame_pre_padding={model.frame_pre_padding} token_overlap={model.token_overlap} frame_overlap={model.frame_overlap}", flush=True)

    g = torch.Generator().manual_seed(args.seed + 1)
    clip_length = TINY_CFG["clip_length"]

    # ---- single-clip round trip: _encode_clip / _decode_clip directly ----
    x1 = (torch.randn((1, 3, clip_length, HEIGHT, WIDTH), generator=g) * 0.3).to(torch.float32)
    moments1 = model._encode_clip(x1)
    print(f"  tap_clip_moments: {tuple(moments1.shape)}", flush=True)
    z1 = DiagonalGaussianDistribution(moments1).mode()
    pixels1 = model._decode_clip(z1)
    print(f"  tap_clip_pixels: {tuple(pixels1.shape)}", flush=True)

    # ---- two-clip round trip: the PUBLIC encode()/decode(), exercising the
    # outer multi-chunk _encode/_decode orchestration at a non-degenerate
    # num_chunks (see this script's own module doc). ----------------------
    num_frames2 = 2 * clip_length
    x2 = (torch.randn((1, 3, num_frames2, HEIGHT, WIDTH), generator=g) * 0.3).to(torch.float32)
    posterior2 = model.encode(x2).latent_dist
    z2 = posterior2.mode()
    print(f"  tap_multi_latent: {tuple(z2.shape)}", flush=True)
    dec2 = model.decode(z2).sample
    print(f"  tap_multi_pixels: {tuple(dec2.shape)}", flush=True)

    # ---- self-validation: the public encode()'s own multi-chunk path,
    # replayed by hand for x1 (exactly one whole clip). `_encode` ALWAYS
    # applies token_drop once after concatenation, regardless of chunk count
    # (`if self.config.token_drop > 0`, unconditional on `x.shape[2] //
    # clip_length`) - so `model.encode(x1).mode()` is `_encode_clip(x1)`'s
    # moments with `token_drop` trailing latent frames dropped, NOT equal to
    # `z1` above (which deliberately has NO token_drop applied, matching
    # `crate::video_vae::encode_clip` - only the OUTER `crate::video_vae::
    # encode` drops trailing frames). This checks that relationship exactly,
    # rather than a same-shape equality that would be the wrong claim. ----
    posterior1_pub = model.encode(x1).latent_dist
    z1_pub = posterior1_pub.mode()
    token_drop = TINY_CFG["token_drop"]
    d_enc = (z1_pub.double() - z1[:, :, :-token_drop].double()).abs().max().item()
    print(f"  self-validate encode() vs _encode_clip[:-token_drop] (single clip): max abs diff {d_enc:.3e}, shapes {tuple(z1_pub.shape)} vs {tuple(z1.shape)}", flush=True)
    assert d_enc == 0.0, f"public encode() disagrees with _encode_clip's own moments (minus token_drop) by {d_enc:.3e}"

    try:
        dec1_pub = model.decode(z1_pub).sample
        print(f"  NOTE: public decode() on a single clip's post-token_drop latent did NOT raise - shape {tuple(dec1_pub.shape)} (a previously-flagged num_chunks==0 degenerate case may not be reachable through this exact config/shape combination; not relied upon either way - the two-clip golden below is what gates the outer decode() chunk path numerically)", flush=True)
    except Exception as e:  # noqa: BLE001 - this IS the degenerate-num_chunks probe
        print(f"  NOTE: public decode() on a single clip's latent raised {type(e).__name__}: {e} (a previously-flagged num_chunks==0 degenerate case - _decode_clip is used directly for the single-clip golden instead, and the two-clip golden exercises the public decode() path at a non-degenerate num_chunks)", flush=True)

    tensors = {
        "input_x1": x1[0],
        "tap_clip_moments": moments1[0],
        "input_z1": z1[0],
        "tap_clip_pixels": pixels1[0],
        "input_x2": x2[0],
        "tap_multi_latent": z2[0],
        "tap_multi_pixels": dec2[0],
    }
    weights = dict(model.state_dict())
    tensors.update(weights)

    manifest = {
        "run": {"seed": args.seed, "height": HEIGHT, "width": WIDTH, "clip_length": clip_length},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "geometry": {
            "spatial_compression_ratio": int(model.spatial_compression_ratio),
            "temporal_compression_ratio": int(model.temporal_compression_ratio),
            "tokens_chunk_size": int(model.tokens_chunk_size),
            "frame_pre_padding": int(model.frame_pre_padding),
            "token_overlap": int(model.token_overlap),
            "frame_overlap": int(model.frame_overlap),
        },
    }
    manifest["source"] = source_block(
        checkpoint=None,
        files=(),
        hash_files=False,
        identity={k: (v if isinstance(v, int) else 0) for k, v in TINY_CFG.items() if isinstance(v, (int, bool))} | {
            "rope_theta_x1000": int(TINY_CFG["decoder_rope_theta"] * 1000),
            "rope_dim_ratio_x1000": int(TINY_CFG["decoder_rope_dim_ratio"] * 1000),
        },
    )
    save(args.out, "minimaxh3_video_vae_tiny.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
