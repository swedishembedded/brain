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

The first two rungs are dumped with `vae.use_tiling = False` and a 32x32
canvas, where the tiled and untiled paths are the same computation anyway
(`_split_tiles` returns one full-size tile when `tile_size >= length`), so
they pin the untiled clip math on its own. The tiled rungs below then turn
`use_tiling` back on - the reference's OWN shipped default - at canvases
above 256 pixels, which is the only place the two paths differ.

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
  - a TILED `_decode_clip` at 512x768 pixels (a 32x48 latent grid), which
    `_split_tiles` covers with a 3x4 tile grid. The width axis is the
    interesting one: four 256-pixel tiles over 768 pixels leave 64 pixels of
    slack, and the round-robin distribution hands it out unevenly, giving
    overlaps `[96, 80, 80]` rather than three equal ones. The height axis
    gives an even `[128, 128]`, so a height/width mix-up cannot pass.
  - a TILED `_encode_clip` at 384x384 pixels (a 2x2 tile grid), which is the
    OTHER tiling direction: `_encode_clip` lays its tiles out in pixel space
    and stitches LATENT output, so it converts the overlaps by
    `// spatial_compression_ratio` where `_decode_clip` uses them as-is.
  - `_split_tiles` itself, called directly at a spread of lengths and dumped
    as flat integer tensors, so the tile layout is gated as data rather than
    inferred from a stitched result.

Usage:
  python3 tools/minimaxh3_video_vae_dump_reference.py \\
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

# The tiled rungs. Both are above the 256-pixel tile size on at least one
# axis, which is the only regime where the tiled and untiled paths differ.
# 512x768 decodes as a 3x4 tile grid with UNEVEN width overlaps; 384x384
# encodes as a 2x2 grid. Latent frame counts are kept at 1 - spatial tiling
# is orthogonal to the temporal chunking, which the two rungs above already
# gate, and this keeps the dumped pixel tensors a sane size.
TILED_DEC_H = 512
TILED_DEC_W = 768
TILED_DEC_FRAMES = 1
TILED_ENC_H = 384
TILED_ENC_W = 384
TILED_ENC_FRAMES = 1

# The lengths `_split_tiles` is probed at, paired with the tile size and the
# minimum overlap it is probed with. 256/64 are the reference's own shipped
# `tile_sample_min_*` values; the 128 rows check the degenerate
# `tile_size >= length` branch that makes tiling a no-op at small canvases.
SPLIT_PROBES = [
    (128, 256, 64),
    (256, 256, 64),
    (384, 256, 64),
    (512, 256, 64),
    (768, 256, 64),
    (1024, 256, 64),
    (1344, 256, 64),
]


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

    # `MiniMaxH3VideoTransformerBlock` initializes its LayerScale gates as
    # `nn.Parameter(torch.zeros(dim))` (scale1/scale2), and the ViT decoder
    # initializes `register_tokens` to zeros too. At default init that makes
    # `h = h + attn(norm(h)) * 0` an EXACT no-op, so all 36 (here 2)
    # transformer blocks vanish and the decoder collapses to
    # `proj_out(norm_out(proj_in(z)))` - a per-token map with no attention,
    # no RoPE and no position dependence whatsoever.
    #
    # That would make this golden vacuous over the entire transformer stack,
    # and in particular would make the tiled and untiled decode paths agree
    # to float32 noise (measured: 4.8e-07), since a position-independent
    # per-token map cannot tell a tile apart from a whole frame. The real
    # checkpoint's trained gates are of course not zero. So every all-zero
    # parameter is filled with small random values here, and the fill is
    # verified to have actually changed the forward.
    zeroed = [n for n, p in model.named_parameters() if not p.any()]
    for name, p in model.named_parameters():
        if not p.any():
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(args.seed + 100 + len(name))) * 0.1)
    print(f"randomized {len(zeroed)} all-zero parameters (LayerScale gates / register tokens): {zeroed[:6]}{' ...' if len(zeroed) > 6 else ''}", flush=True)
    assert not any(not p.any() for p in model.parameters()), "an all-zero parameter survived the fill"

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

    # ---- tiled rungs: use_tiling back ON (the reference's own default),
    # at canvases above the 256-pixel tile size where the tiled and untiled
    # paths actually differ. ---------------------------------------------
    model.use_tiling = True
    ratio = int(model.spatial_compression_ratio)

    split_taps = {}
    for length, tile, min_ov in SPLIT_PROBES:
        idx, lens, ovs = model._split_tiles(length, tile, min_ov)
        assert idx[-1] + lens[-1] == length, f"_split_tiles({length},{tile},{min_ov}) does not end at {length}: {idx} {lens}"
        assert sum(lens) - sum(ovs) == length, f"_split_tiles({length},{tile},{min_ov}) does not cover {length} exactly"
        key = f"{length}_{tile}_{min_ov}"
        split_taps[f"tap_split_starts_{key}"] = torch.tensor(idx, dtype=torch.float32)
        split_taps[f"tap_split_lengths_{key}"] = torch.tensor(lens, dtype=torch.float32)
        split_taps[f"tap_split_overlaps_{key}"] = torch.tensor(ovs, dtype=torch.float32)
        print(f"  _split_tiles({length},{tile},{min_ov}): starts={idx} lengths={lens} overlaps={ovs}", flush=True)

    # Tiled decode: a latent grid whose pixel extent is 512x768 -> 3x4 tiles.
    z_tiled = (torch.randn((1, TINY_CFG["latent_channels"], TILED_DEC_FRAMES, TILED_DEC_H // ratio, TILED_DEC_W // ratio), generator=g) * 0.5).to(torch.float32)
    dec_tiled = model._decode_clip(z_tiled)
    ny = len(model._split_tiles(TILED_DEC_H, model.tile_sample_min_height, model.tile_sample_min_overlap_height)[0])
    nx = len(model._split_tiles(TILED_DEC_W, model.tile_sample_min_width, model.tile_sample_min_overlap_width)[0])
    assert ny > 1 and nx > 1, f"tiled decode rung is not multi-tile: {ny}x{nx}"
    print(f"  tap_tiled_dec_pixels: {tuple(dec_tiled.shape)} from a {ny}x{nx} tile grid", flush=True)

    # Self-validation that the tiled path is genuinely a DIFFERENT
    # computation here, so this rung cannot pass vacuously against an
    # untiled implementation.
    model.use_tiling = False
    dec_untiled = model._decode_clip(z_tiled)
    model.use_tiling = True
    d_tile = (dec_tiled.double() - dec_untiled.double()).abs().max().item()
    print(f"  self-validate tiled vs untiled _decode_clip at {TILED_DEC_H}x{TILED_DEC_W}: max abs diff {d_tile:.3e}", flush=True)
    assert d_tile > 1e-3, f"tiled and untiled _decode_clip agree to {d_tile:.3e} - this rung would pass vacuously"

    # Tiled encode: 384x384 pixels -> a 2x2 tile grid.
    x_tiled = (torch.randn((1, 3, TILED_ENC_FRAMES, TILED_ENC_H, TILED_ENC_W), generator=g) * 0.3).to(torch.float32)
    enc_tiled = model._encode_clip(x_tiled)
    eny = len(model._split_tiles(TILED_ENC_H, model.tile_sample_min_height, model.tile_sample_min_overlap_height)[0])
    enx = len(model._split_tiles(TILED_ENC_W, model.tile_sample_min_width, model.tile_sample_min_overlap_width)[0])
    assert eny > 1 and enx > 1, f"tiled encode rung is not multi-tile: {eny}x{enx}"
    print(f"  tap_tiled_enc_moments: {tuple(enc_tiled.shape)} from a {eny}x{enx} tile grid", flush=True)

    model.use_tiling = False
    enc_untiled = model._encode_clip(x_tiled)
    model.use_tiling = True
    d_enc_tile = (enc_tiled.double() - enc_untiled.double()).abs().max().item()
    print(f"  self-validate tiled vs untiled _encode_clip at {TILED_ENC_H}x{TILED_ENC_W}: max abs diff {d_enc_tile:.3e}", flush=True)
    assert d_enc_tile > 1e-3, f"tiled and untiled _encode_clip agree to {d_enc_tile:.3e} - this rung would pass vacuously"

    tensors = {
        "input_x1": x1[0],
        "tap_clip_moments": moments1[0],
        "input_z1": z1[0],
        "tap_clip_pixels": pixels1[0],
        "input_x2": x2[0],
        "tap_multi_latent": z2[0],
        "tap_multi_pixels": dec2[0],
        "input_z_tiled": z_tiled[0],
        "tap_tiled_dec_pixels": dec_tiled[0],
        "input_x_tiled": x_tiled[0],
        "tap_tiled_enc_moments": enc_tiled[0],
    }
    tensors.update(split_taps)
    weights = dict(model.state_dict())
    tensors.update(weights)

    manifest = {
        "run": {
            "seed": args.seed,
            "height": HEIGHT,
            "width": WIDTH,
            "clip_length": clip_length,
            "tiled_decode": {"height": TILED_DEC_H, "width": TILED_DEC_W, "latent_frames": TILED_DEC_FRAMES, "tile_grid": [ny, nx]},
            "tiled_encode": {"height": TILED_ENC_H, "width": TILED_ENC_W, "frames": TILED_ENC_FRAMES, "tile_grid": [eny, enx]},
            "split_probes": [list(p) for p in SPLIT_PROBES],
        },
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "geometry": {
            "spatial_compression_ratio": int(model.spatial_compression_ratio),
            "temporal_compression_ratio": int(model.temporal_compression_ratio),
            "tokens_chunk_size": int(model.tokens_chunk_size),
            "frame_pre_padding": int(model.frame_pre_padding),
            "token_overlap": int(model.token_overlap),
            "frame_overlap": int(model.frame_overlap),
            "tile_sample_min_height": int(model.tile_sample_min_height),
            "tile_sample_min_width": int(model.tile_sample_min_width),
            "tile_sample_min_overlap_height": int(model.tile_sample_min_overlap_height),
            "tile_sample_min_overlap_width": int(model.tile_sample_min_overlap_width),
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
