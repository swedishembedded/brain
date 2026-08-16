#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Wan2.1 VAE (3D causal autoencoder) reference goldens for brain parity tests.

Runs the OFFICIAL `Wan2.1/wan/modules/vae.py` (CPU, fp32) on a deterministic
synthetic clip and dumps every boundary of the encode/decode pipeline:

  vae_t{T}.safetensors   video -> encoder stage taps -> moments -> mu/logvar
                         -> normalised latent -> decoder stage taps -> recon,
                         for BOTH the reference's chunked path and an
                         independent whole-clip (unchunked) path
  manifest.json          shapes, sha256 per file, run parameters, versions

The reference module is imported BY FILE PATH: `import wan.modules.vae` executes
`wan/__init__.py`, which drags in the whole model stack (and needs packages that
are not installed here).

## The two paths, and why the unchunked one exists

`WanVAE_.encode`/`.decode` ALWAYS run in temporal chunks (1,4,4,... input frames
for encode; one latent frame at a time for decode) threading a `feat_cache` of
the last `CACHE_T = 2` frames through every `CausalConv3d`. There is no
whole-clip mode upstream at all - with `feat_cache=None` the `Resample` blocks
silently skip their temporal conv entirely, which is a different model.

An equivalent whole-clip formulation nevertheless exists, and deriving it is the
strongest self-validation available here (two independent paths, one assert):

* every plain `CausalConv3d` chunk call is exactly the corresponding output
  window of the same conv run over the concatenated sequence, because the cache
  it is handed is literally the previous chunk's last two input frames;
* `downsample3d` (stride-2, no pad, applied AFTER the spatial resample) emits
  chunk 0 unchanged and then convolves `[prev_last_frame] ++ chunk`, whose
  windows over the whole clip are exactly the stride-2 windows starting at
  even positions - so the whole-clip form is
  `cat([x[:, :, :1], time_conv(x)], dim=2)`;
* `upsample3d` (applied BEFORE the spatial resample) emits chunk 0 unchanged,
  marks its cache slot `'Rep'`, and from chunk 1 on convolves the sub-sequence
  `x[:, :, 1:]` with a zero-filled history - i.e. frame 0 is deliberately
  dropped from the temporal history - so the whole-clip form is
  `cat([x[:, :, :1], interleave(time_conv(x[:, :, 1:]))], dim=2)`.

Usage:
  python tools/goldens/wan_vae_dump_reference.py \
      --weights /path/to/Wan2.1_VAE.pth \
      --out testdata/golden/wan [--frames 9,17 --size 64 --seed 42]
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch
from safetensors.torch import save_file

# `WanVAE`'s hardcoded per-channel latent statistics (wan/modules/vae.py). They
# are NOT in the .pth; the diffusers export repeats them as `latents_mean` /
# `latents_std` in vae/config.json.
LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

CFG = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
           attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)


def load_reference(path):
    """Import `wan/modules/vae.py` by file path (never `import wan.*`)."""
    spec = importlib.util.spec_from_file_location("wan_ref_vae", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wan_ref_vae"] = mod
    spec.loader.exec_module(mod)
    return mod


def det_video(t, h, w, seed):
    """Deterministic RGB clip in [-1, 1], shape (1, 3, t, h, w)."""
    g = torch.Generator().manual_seed(seed)
    ts = torch.linspace(0.0, 1.0, t).view(t, 1, 1)
    ys = torch.linspace(0.0, 3.14159, h).view(1, h, 1)
    xs = torch.linspace(0.0, 6.28318, w).view(1, 1, w)
    r = torch.sin(xs + ys + 3.0 * ts)
    gg = torch.cos(2.0 * xs) * torch.sin(0.5 * ys + ts)
    b = 2.0 * (ys / 3.14159) - 1.0 + 0.3 * torch.cos(5.0 * ts)
    v = torch.stack([r, gg.expand(t, h, w), b.expand(t, h, w)], 0)
    v = v + 0.02 * torch.randn(v.shape, generator=g)
    return v.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def save(out, name, tensors, manifest):
    # everything as f32 - brain's safetensors reader is F32/F16/BF16-only
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}"
                                        for k, v in sorted(tensors.items())), flush=True)


# --------------------------------------------------------------------------
# The independent whole-clip path (see the module docstring for the derivation)
# --------------------------------------------------------------------------

def full_resample(m, x, ref):
    """`Resample.forward` without a feat_cache, temporal path included."""
    if m.mode == "upsample3d":
        b, c, t, h, w = x.shape
        assert t >= 1
        head, tail = x[:, :, :1], x[:, :, 1:]
        if tail.shape[2] > 0:
            y = m.time_conv(tail)                       # causal pad (2,0)
            y = y.reshape(b, 2, c, tail.shape[2], h, w)
            y = torch.stack((y[:, 0], y[:, 1]), 3)
            y = y.reshape(b, c, tail.shape[2] * 2, h, w)
            x = torch.cat([head, y], dim=2)
        else:
            x = head
    t = x.shape[2]
    x = ref.rearrange(x, "b c t h w -> (b t) c h w")
    x = m.resample(x)
    x = ref.rearrange(x, "(b t) c h w -> b c t h w", t=t)
    if m.mode == "downsample3d":
        # stride 2, kernel 3, NO pad, over [first frame] ++ everything
        x = torch.cat([x[:, :, :1], m.time_conv(x)], dim=2)
    return x


def full_block(m, x, ref):
    if isinstance(m, ref.Resample):
        return full_resample(m, x, ref)
    return m(x)  # ResidualBlock / AttentionBlock with feat_cache=None


def full_encoder(enc, x, ref, taps=None):
    x = enc.conv1(x)
    if taps is not None:
        taps["enc.conv1"] = x
    for i, layer in enumerate(enc.downsamples):
        x = full_block(layer, x, ref)
        if taps is not None:
            taps[f"enc.downsamples.{i}"] = x
    for layer in enc.middle:
        x = full_block(layer, x, ref)
    if taps is not None:
        taps["enc.middle"] = x
    for layer in enc.head:
        x = layer(x)
    if taps is not None:
        taps["enc.head"] = x
    return x


def full_decoder(dec, x, ref, taps=None):
    x = dec.conv1(x)
    if taps is not None:
        taps["dec.conv1"] = x
    for layer in dec.middle:
        x = full_block(layer, x, ref)
    if taps is not None:
        taps["dec.middle"] = x
    for i, layer in enumerate(dec.upsamples):
        x = full_block(layer, x, ref)
        if taps is not None:
            taps[f"dec.upsamples.{i}"] = x
    for layer in dec.head:
        x = layer(x)
    if taps is not None:
        taps["dec.head"] = x
    return x


# --------------------------------------------------------------------------
# The reference (chunked) path, with per-stage taps concatenated over chunks
# --------------------------------------------------------------------------

class ChunkTaps:
    """Forward hooks that concatenate each module's per-chunk output on time."""

    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module):
        def hook(_m, _i, o):
            prev = self.acc.get(name)
            self.acc[name] = o if prev is None else torch.cat([prev, o], dim=2)
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def encode_chunked(model, x, scale, taps):
    """`WanVAE_.encode`, with the moments/mu split exposed."""
    model.clear_cache()
    t = x.shape[2]
    iters = 1 + (t - 1) // 4
    out = None
    for i in range(iters):
        model._enc_conv_idx = [0]
        sl = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1):1 + 4 * i]
        o = model.encoder(sl, feat_cache=model._enc_feat_map,
                          feat_idx=model._enc_conv_idx)
        out = o if out is None else torch.cat([out, o], 2)
    taps["enc.out"] = out
    moments = model.conv1(out)
    mu, log_var = moments.chunk(2, dim=1)
    z = (mu - scale[0].view(1, -1, 1, 1, 1)) * scale[1].view(1, -1, 1, 1, 1)
    model.clear_cache()
    return moments, mu, log_var, z


def decode_chunked(model, z, scale, taps):
    model.clear_cache()
    z = z / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(1, -1, 1, 1, 1)
    taps["dec.z_denorm"] = z
    x = model.conv2(z)
    taps["dec.conv2"] = x
    out = None
    for i in range(z.shape[2]):
        model._conv_idx = [0]
        o = model.decoder(x[:, :, i:i + 1], feat_cache=model._feat_map,
                          feat_idx=model._conv_idx)
        out = o if out is None else torch.cat([out, o], 2)
    model.clear_cache()
    return out


def agree(name, a, b, tol=2e-5):
    """Assert two paths agree, RELATIVE to the tensor's own scale.

    An absolute tolerance is the wrong gate here: the same fp32 reassociation
    that is 3e-6 on the [-1,1] reconstruction is 2e-4 on a deep activation whose
    values are O(10), and tightening the gate for the first would fail the
    second for no reason.
    """
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} "
          f"= {rel:.2e} (tol {tol:g})", flush=True)
    assert rel < tol, f"{name}: chunked and unchunked paths disagree by {rel:.3e} relative"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Wan2.1_VAE.pth")
    ap.add_argument("--reference", default="scratchpad/reference/wan/Wan2.1/wan/modules/vae.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="9,17", help="comma-separated clip lengths (1+4k)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taps", action="store_true",
                    help="also dump every per-block activation (large: hundreds of "
                         "MB at 64x64 - the parity tests only need the boundaries, "
                         "and the self-validation above compares the taps in memory "
                         "either way)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    ref = load_reference(args.reference)
    print(f"reference: {args.reference} (CACHE_T={ref.CACHE_T})", flush=True)

    model = ref.WanVAE_(**CFG)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors (strict)", flush=True)
    model.eval().requires_grad_(False)
    scale = [torch.tensor(LATENTS_MEAN), 1.0 / torch.tensor(LATENTS_STD)]

    manifest = {"run": {"config": CFG, "size": args.size, "seed": args.seed,
                        "frames": args.frames, "cache_t": ref.CACHE_T,
                        "latents_mean": LATENTS_MEAN, "latents_std": LATENTS_STD},
                "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}}

    for t in [int(v) for v in args.frames.split(",")]:
        assert (t - 1) % 4 == 0, f"{t} frames is not 1+4k"
        n_chunks = 1 + (t - 1) // 4
        print(f"\n=== {t} frames at {args.size}x{args.size} "
              f"({n_chunks} encode chunks) ===", flush=True)
        video = det_video(t, args.size, args.size, args.seed)

        # ---- encode: reference chunked path, with per-stage taps -------------
        ct = ChunkTaps()
        for i, m in enumerate(model.encoder.downsamples):
            ct.watch(f"enc.downsamples.{i}", m)
        # `Encoder3d.forward` iterates `self.middle` / `self.head` rather than
        # calling the Sequential, so a hook on the container would never fire.
        ct.watch("enc.conv1", model.encoder.conv1)
        ct.watch("enc.middle", model.encoder.middle[2])
        ct.watch("enc.head", model.encoder.head[2])
        taps = {}
        moments, mu, log_var, latent = encode_chunked(model, video, scale, taps)
        ct.close()
        enc_taps = dict(ct.acc)
        assert latent.shape[2] == 1 + (t - 1) // 4, latent.shape

        # ---- encode: independent whole-clip path (SELF-VALIDATION) -----------
        full_taps = {}
        enc_full = full_encoder(model.encoder, video, ref, full_taps)
        agree(f"encoder t={t}", enc_taps["enc.head"], enc_full)
        agree(f"encoder-out t={t}", taps["enc.out"], enc_full)
        for k in sorted(full_taps):
            if k in enc_taps:
                agree(f"  tap {k}", enc_taps[k], full_taps[k])

        # ---- encode: chunk-size invariance (1,4,4,... vs 1,8,...) -----------
        # brain's Rust encoder is tested for the same invariance, so pin it here.
        if n_chunks > 2:
            model.clear_cache()
            out = None
            for i, sl in enumerate([video[:, :, :1], video[:, :, 1:]]):
                model._enc_conv_idx = [0]
                o = model.encoder(sl, feat_cache=model._enc_feat_map,
                                  feat_idx=model._enc_conv_idx)
                out = o if out is None else torch.cat([out, o], 2)
            model.clear_cache()
            # torch picks different conv blocking per T, so this is fp32
            # reassociation, not a semantic difference. brain's `conv3d` is one
            # thread per output element summing in a T-independent order, so
            # the same invariance is asserted BIT-EXACT on the Rust side.
            agree(f"encoder chunking (1,{t - 1}) vs (1,4,...) t={t}", out, taps["enc.out"])

        # ---- decode: reference chunked path ----------------------------------
        ct = ChunkTaps()
        for i, m in enumerate(model.decoder.upsamples):
            ct.watch(f"dec.upsamples.{i}", m)
        ct.watch("dec.conv1", model.decoder.conv1)
        ct.watch("dec.middle", model.decoder.middle[2])
        ct.watch("dec.head", model.decoder.head[2])
        dtaps = {}
        recon = decode_chunked(model, latent, scale, dtaps)
        ct.close()
        dec_taps = dict(ct.acc)
        assert recon.shape[2] == t, recon.shape

        # ---- decode: independent whole-clip path (SELF-VALIDATION) -----------
        dfull_taps = {}
        recon_full = full_decoder(model.decoder, dtaps["dec.conv2"], ref, dfull_taps)
        agree(f"decoder t={t}", recon, recon_full)
        for k in sorted(dfull_taps):
            if k in dec_taps:
                agree(f"  tap {k}", dec_taps[k], dfull_taps[k])

        # `.clamp_(-1, 1)` is applied by `WanVAE.decode`, outside the model.
        tensors = {
            "video": video[0],
            "moments": moments[0],
            "mu": mu[0],
            "log_var": log_var[0],
            "latent": latent[0],
            "z_denorm": dtaps["dec.z_denorm"][0],
            "dec_conv2": dtaps["dec.conv2"][0],
            "enc_out": taps["enc.out"][0],
            "recon_chunked": recon[0],
            "recon_unchunked": recon_full[0],
            "recon_clamped": recon[0].clamp(-1.0, 1.0),
        }
        if args.taps:
            for k, v in enc_taps.items():
                tensors[f"tap_{k}"] = v[0]
            for k, v in dec_taps.items():
                tensors[f"tap_{k}"] = v[0]
        save(args.out, f"vae_t{t}.safetensors", tensors, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
