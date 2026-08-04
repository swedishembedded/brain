#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump SDXL-ControlNet reference goldens for brain's `crates/controlnet`
parity ladder (docs/imaging/plan.md phase 4c).

  stages.safetensors             per-STAGE taps of a real diffusers
                                 `ControlNetModel` forward, captured with
                                 forward hooks so the Rust test is a pure
                                 replay and no convention (the added-
                                 conditioning concat order, the conditioning
                                 embedder's SiLU placement, the zero-conv
                                 ordering, where `conditioning_scale` is
                                 applied) is re-derived by hand. Also carries
                                 the exact inputs and BOTH halves of every
                                 injection point: the residual as it enters the
                                 zero-conv and as it leaves it.
  manifest.json                  every tensor's shape, sha256 per file, the
                                 reference config and the run parameters.

Everything is CPU + fp32 with fixed seeds and fixed synthetic inputs, and every
tensor is stored as f32 (brain's safetensors reader is F32/F16/BF16-only).

The ControlNet is dumped at a SMALL latent resolution by default
(`--latent 32`, i.e. a 256x256 conditioning image, since the conditioning
embedder downsamples by exactly 8) so the golden set stays small and the Rust
parity test fits comfortably on one card alongside 5 GB of fp32 weights. The
graph is resolution-independent, so this gates the composition; it does NOT
gate the 128x128 latent SDXL actually generates at, and the test says so.

Usage:
  python3 tools/controlnet_dump_reference.py \
      --controlnet /path/to/ControlNetModel \
      --out testdata/controlnet [--latent 32]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

SEED = 20260804
# Micro-conditioning in diffusers' order:
#   (original_h, original_w, crop_top, crop_left, target_h, target_w)
# Six DISTINCT values — an off-by-one in the concat order is invisible when
# they repeat. Same values as tools/sdxl_dump_reference.py, deliberately: the
# ControlNet's conditioning chain is byte-for-byte the UNet's, so a disagreement
# between the two goldens is a real disagreement and not a different input.
TIME_IDS = [1024.0, 1024.0, 8.0, 16.0, 512.0, 768.0]
# A second conditioning_scale, to gate that the scale multiplies the ZERO-CONV
# OUTPUT (and only it) rather than, say, the conditioning embedding.
SCALE2 = 0.75


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(store, name, t):
    """Store a tensor as contiguous fp32 on the CPU."""
    if isinstance(t, (tuple, list)):
        t = t[0]
    # `.clone()` is load-bearing: several taps alias one storage (a zero-conv's
    # input IS the previous block's output), and safetensors refuses to write
    # aliased storage.
    store[name] = t.detach().to(torch.float32).contiguous().cpu().clone()


def save(store, out, rel, manifest):
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(store, path)
    manifest["files"][rel] = {
        "sha256": sha256(path),
        "tensors": {k: list(v.shape) for k, v in sorted(store.items())},
    }
    print(f"  wrote {rel}: {len(store)} tensors, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def dump(controlnet_dir, out, latent, latent_w, rel, manifest):
    from diffusers import ControlNetModel

    print("loading controlnet (fp32, cpu) ...", flush=True)
    net = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float32)
    net.eval()
    cfg = dict(net.config)
    store = {}
    taps = {}
    handles = []

    def tap(mod, name, want_input=False):
        def fn(_mod, inp, outp):
            taps[name] = outp
            if want_input:
                taps[name + ".in"] = inp[0]

        handles.append(mod.register_forward_hook(fn))

    # ---- conditioning chain (identical modules to the UNet's) -------------
    tap(net.time_proj, "time_proj")
    tap(net.time_embedding.linear_1, "time_embedding.linear_1")
    tap(net.time_embedding, "time_embedding")
    tap(net.add_time_proj, "add_time_proj")
    tap(net.add_embedding.linear_1, "add_embedding.linear_1")
    tap(net.add_embedding, "add_embedding")

    # ---- the conditioning-image embedder ---------------------------------
    ce = net.controlnet_cond_embedding
    tap(ce.conv_in, "cond.conv_in")
    for i, b in enumerate(ce.blocks):
        tap(b, f"cond.block{i}")
    tap(ce.conv_out, "cond.out")

    tap(net.conv_in, "conv_in")

    # ---- the trainable copy of the backbone's early blocks ---------------
    def tap_transformer(t2d, pfx, fine):
        tap(t2d.norm, f"{pfx}.norm")
        tap(t2d.proj_in, f"{pfx}.proj_in")
        for j, blk in enumerate(t2d.transformer_blocks):
            tap(blk, f"{pfx}.tb{j}")
            if fine:
                tap(blk.norm1, f"{pfx}.tb{j}.norm1")
                tap(blk.attn1, f"{pfx}.tb{j}.attn1")
                tap(blk.norm2, f"{pfx}.tb{j}.norm2")
                tap(blk.attn2, f"{pfx}.tb{j}.attn2")
                tap(blk.norm3, f"{pfx}.tb{j}.norm3")
                tap(blk.ff.net[0], f"{pfx}.tb{j}.ff_geglu")
                tap(blk.ff, f"{pfx}.tb{j}.ff")
        tap(t2d.proj_out, f"{pfx}.proj_out")
        tap(t2d, pfx)

    for i, blk in enumerate(net.down_blocks):
        for j, r in enumerate(blk.resnets):
            tap(r, f"down{i}.resnet{j}")
            tap(r.time_emb_proj, f"down{i}.resnet{j}.time_emb_proj")
        for j, a in enumerate(getattr(blk, "attentions", []) or []):
            # Fine-grained taps on ONE attention only: the internals are the
            # same graph everywhere and dumping all of them would multiply the
            # golden size for no extra coverage.
            tap_transformer(a, f"down{i}.attn{j}", fine=(i == 1 and j == 0))
        for j, d in enumerate(getattr(blk, "downsamplers", None) or []):
            tap(d, f"down{i}.downsample{j}")

    for j, r in enumerate(net.mid_block.resnets):
        tap(r, f"mid.resnet{j}")
    for j, a in enumerate(net.mid_block.attentions):
        tap_transformer(a, f"mid.attn{j}", fine=False)

    # ---- the injection points, both halves --------------------------------
    # `zero{k}.in` is the residual as the backbone produced it (i.e. exactly
    # the UNet skip tensor at that point); `zero{k}` is it after the zero-conv,
    # BEFORE `conditioning_scale`. Dumping both localises a failure to the
    # trainable copy or to the zero-conv, which one tap cannot.
    for k, z in enumerate(net.controlnet_down_blocks):
        tap(z, f"zero{k}", want_input=True)
    tap(net.controlnet_mid_block, "zero_mid", want_input=True)

    # ---- inputs -----------------------------------------------------------
    g = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, cfg["in_channels"], latent, latent_w, generator=g)
    timestep = torch.tensor([601], dtype=torch.long)
    enc = torch.randn(1, 77, cfg["cross_attention_dim"], generator=g)
    pooled = torch.randn(1, 1280, generator=g)
    time_ids = torch.tensor([TIME_IDS], dtype=torch.float32)
    # The conditioning image is at PIXEL resolution: the embedder's three
    # stride-2 convs downsample by exactly 8, matching the VAE's factor. A
    # `rand` (not `randn`) input is what a real preprocessor produces — the
    # pipeline feeds a [0,1] image, and a signed input would exercise a
    # distribution the ReLU-free SiLU stack never sees in production.
    cond_px = 8 * latent
    cond_px_w = 8 * latent_w
    cond = torch.rand(1, cfg["conditioning_channels"], cond_px, cond_px_w, generator=g)

    record(store, "in.sample", sample)
    record(store, "in.timestep", timestep)
    record(store, "in.encoder_hidden_states", enc)
    record(store, "in.text_embeds", pooled)
    record(store, "in.time_ids", time_ids)
    record(store, "in.controlnet_cond", cond)

    kw = dict(
        encoder_hidden_states=enc,
        controlnet_cond=cond,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        return_dict=False,
    )

    print(f"forward (scale=1.0, latent {latent}x{latent_w}, cond {cond_px}x{cond_px_w}) ...", flush=True)
    with torch.no_grad():
        down, mid = net(sample, timestep, conditioning_scale=1.0, **kw)
    for k, d in enumerate(down):
        record(store, f"out.down{k}", d)
    record(store, "out.mid", mid)

    for k, v in taps.items():
        record(store, k, v)

    # `conv_in(sample) + cond_embedding(cond)` is not a module, so it has no
    # hook — but it is the one place the two input paths meet and the FIRST
    # injection point's input. Recompute it from the two taps that do exist.
    record(store, "sample_cond", taps["conv_in"] + taps["cond.out"])

    print(f"forward (conditioning_scale={SCALE2}) ...", flush=True)
    with torch.no_grad():
        down2, mid2 = net(sample, timestep, conditioning_scale=SCALE2, **kw)
    for k, d in enumerate(down2):
        record(store, f"out{SCALE2}.down{k}", d)
    record(store, f"out{SCALE2}.mid", mid2)

    for h in handles:
        h.remove()

    # Self-validation inside the dumper (playbook §1): the second scale must be
    # exactly the first times SCALE2 — if it is not, `conditioning_scale` is not
    # the pure output multiplier the Rust side is about to assume it is.
    for k, (a, b) in enumerate(zip(down, down2)):
        assert torch.allclose(b, a * SCALE2, atol=1e-6), f"down{k}: scale is not a pure multiply"
    assert torch.allclose(mid2, mid * SCALE2, atol=1e-6), "mid: scale is not a pure multiply"

    save(store, out, rel, manifest)
    manifest["params"]["controlnet_config"] = cfg
    manifest["params"][f"latent.{rel}"] = [latent, latent_w]
    manifest["params"][f"cond_px.{rel}"] = [cond_px, cond_px_w]
    manifest["params"]["time_ids"] = TIME_IDS
    manifest["params"]["timestep"] = int(timestep.item())
    manifest["params"]["scale2"] = SCALE2
    manifest["params"]["n_down"] = len(down)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlnet", required=True, help="a diffusers ControlNetModel dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=32, help="latent H (32 => a 256px conditioning image)")
    ap.add_argument("--latent-w", type=int, default=None, help="latent W (defaults to --latent)")
    ap.add_argument("--name", default="stages.safetensors", help="output file, relative to --out")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    manifest = {"files": {}, "params": {}}
    if os.path.exists(mpath):
        with open(mpath) as f:
            prev = json.load(f)
        manifest["files"].update(prev.get("files", {}))
        manifest["params"].update(prev.get("params", {}))
    manifest["params"].update(
        {
            "seed": SEED,
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "controlnet_weights": os.path.abspath(args.controlnet),
        }
    )
    print("== controlnet ==", flush=True)
    dump(args.controlnet, args.out, args.latent, args.latent_w or args.latent, args.name, manifest)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
