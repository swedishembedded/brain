#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump SDXL reference goldens for brain's `crates/unet` + `crates/diffusion`
parity ladder.

Two independent golden sets, because they gate two independent pieces:

  unet/stages.safetensors      per-STAGE taps of a real `UNet2DConditionModel`
                               forward, captured with forward hooks so the Rust
                               test is a pure replay and no convention (the
                               added-conditioning concat order, the penultimate
                               text layer, the GEGLU half order) is re-derived
                               by hand. Also carries the exact inputs.
  schedulers/steps.safetensors timesteps, sigmas and a fixed `step()` trajectory
                               for DDIM / Euler / Euler-ancestral /
                               DPM-Solver++(2M), each in the `epsilon` and
                               `v_prediction` parameterisations.
  manifest.json                every tensor's shape, sha256 per file, the
                               reference config and the run parameters.

Everything is CPU + fp32 with fixed seeds and fixed synthetic inputs, and every
tensor is stored as f32 (brain's safetensors reader is F32/F16/BF16-only).

The UNet is dumped at a SMALL latent resolution by default (`--latent 32`, i.e.
a 256x256 image) so the golden set stays a few tens of MB and the Rust parity
test fits comfortably on one card. The graph is resolution-independent, so this
gates the composition; it does NOT gate the 128x128 latent SDXL actually
generates at, and the test says so.

Usage:
  python3 tools/goldens/sdxl_dump_reference.py \
      --sdxl /path/to/sdxl-base-1.0 \
      --out  testdata/sdxl [--latent 32] [--skip-unet] [--skip-schedulers]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

SEED = 20240731
# Micro-conditioning, in diffusers' own order:
#   (original_h, original_w, crop_top, crop_left, target_h, target_w)
# Deliberately six DISTINCT values — an off-by-one in the concat order is
# invisible when they repeat.
TIME_IDS = [1024.0, 1024.0, 8.0, 16.0, 512.0, 768.0]
# Scheduler trajectory: step counts chosen to straddle DPM-Solver++'s
# `len(timesteps) < 15` stability branch.
SCHED_STEPS = [4, 20]


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
    # `.clone()` is load-bearing: `conv_out`'s tap and `out.sample` are the
    # SAME tensor, and safetensors refuses to write aliased storage.
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


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


def dump_unet(sdxl, out, latent, manifest):
    from diffusers import UNet2DConditionModel

    print("loading unet (fp32, cpu) ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        sdxl, subfolder="unet", torch_dtype=torch.float32, variant="fp16"
    )
    unet.eval()
    cfg = dict(unet.config)
    store = {}
    taps = {}

    def hook(name):
        def fn(_mod, _inp, outp):
            taps[name] = outp

        return fn

    handles = []

    def tap(mod, name):
        handles.append(mod.register_forward_hook(hook(name)))

    # ---- embedding chain -------------------------------------------------
    tap(unet.time_proj, "time_proj")
    tap(unet.time_embedding.linear_1, "time_embedding.linear_1")
    tap(unet.time_embedding, "time_embedding")
    tap(unet.add_time_proj, "add_time_proj")
    tap(unet.add_embedding.linear_1, "add_embedding.linear_1")
    tap(unet.add_embedding, "add_embedding")
    tap(unet.conv_in, "conv_in")
    tap(unet.conv_norm_out, "conv_norm_out")
    tap(unet.conv_out, "conv_out")

    # ---- block chain -----------------------------------------------------
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

    for i, blk in enumerate(unet.down_blocks):
        for j, r in enumerate(blk.resnets):
            tap(r, f"down{i}.resnet{j}")
            tap(r.time_emb_proj, f"down{i}.resnet{j}.time_emb_proj")
        for j, a in enumerate(getattr(blk, "attentions", []) or []):
            # Fine-grained taps on the FIRST attention only: the internals are
            # the same graph everywhere and dumping all 11 would multiply the
            # golden size for no extra coverage.
            tap_transformer(a, f"down{i}.attn{j}", fine=(i == 1 and j == 0))
        for j, d in enumerate(getattr(blk, "downsamplers", None) or []):
            tap(d, f"down{i}.downsample{j}")

    for j, r in enumerate(unet.mid_block.resnets):
        tap(r, f"mid.resnet{j}")
    for j, a in enumerate(unet.mid_block.attentions):
        tap_transformer(a, f"mid.attn{j}", fine=False)

    for i, blk in enumerate(unet.up_blocks):
        for j, r in enumerate(blk.resnets):
            tap(r, f"up{i}.resnet{j}")
        for j, a in enumerate(getattr(blk, "attentions", []) or []):
            tap_transformer(a, f"up{i}.attn{j}", fine=False)
        for j, u in enumerate(getattr(blk, "upsamplers", None) or []):
            tap(u, f"up{i}.upsample{j}")

    # ---- inputs ----------------------------------------------------------
    g = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, cfg["in_channels"], latent, latent, generator=g)
    # A timestep inside the schedule, not 0 or 999 (both are degenerate for the
    # sinusoidal embedding's low frequencies).
    timestep = torch.tensor([601], dtype=torch.long)
    enc = torch.randn(1, 77, cfg["cross_attention_dim"], generator=g)
    pooled = torch.randn(1, 1280, generator=g)
    time_ids = torch.tensor([TIME_IDS], dtype=torch.float32)

    record(store, "in.sample", sample)
    record(store, "in.timestep", timestep)
    record(store, "in.encoder_hidden_states", enc)
    record(store, "in.text_embeds", pooled)
    record(store, "in.time_ids", time_ids)

    print("forward ...", flush=True)
    with torch.no_grad():
        outp = unet(
            sample,
            timestep,
            encoder_hidden_states=enc,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
    record(store, "out.sample", outp)

    # The skip-connection stack the up blocks consume, recomputed from the taps
    # so the Rust test can gate the SKIP ORDER (which is the one thing a
    # per-module tap does not pin).
    for k, v in taps.items():
        record(store, k, v)
    for h in handles:
        h.remove()

    save(store, out, "unet/stages.safetensors", manifest)
    manifest["params"]["unet_config"] = cfg
    manifest["params"]["latent"] = latent
    manifest["params"]["time_ids"] = TIME_IDS
    manifest["params"]["timestep"] = int(timestep.item())
    del unet


# ---------------------------------------------------------------------------
# schedulers
# ---------------------------------------------------------------------------


_ORIG_RANDN = None
_CLASSES = {}


def _fit(cls, base, prediction_type):
    """`base` restricted to the kwargs `cls.__init__` actually accepts.

    The SDXL `scheduler_config.json` is an EulerDiscrete config; every other
    scheduler shares the CHAIN keys (betas, schedule, spacing, offset) but not
    the sampler-specific ones, and diffusers raises on an unknown kwarg. Dropping
    by signature keeps every shared key rather than hand-listing exclusions per
    family, which is how a chain key silently stops being passed.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    kw = {k: v for k, v in base.items() if k in accepted}
    kw["prediction_type"] = prediction_type
    return kw


def _install_noise(row):
    """Force `EulerAncestralDiscreteScheduler`'s next noise draw to `row`."""
    global _ORIG_RANDN
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as mod

    if _ORIG_RANDN is None:
        _ORIG_RANDN = mod.randn_tensor
    mod.randn_tensor = lambda shape, **kw: row.reshape(shape).clone()


def _restore_noise():
    global _ORIG_RANDN
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as mod

    if _ORIG_RANDN is not None:
        mod.randn_tensor = _ORIG_RANDN
        _ORIG_RANDN = None


def dump_schedulers(sdxl, out, manifest):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
    )

    with open(os.path.join(sdxl, "scheduler", "scheduler_config.json")) as f:
        base = json.load(f)
    base = {k: v for k, v in base.items() if not k.startswith("_")}
    store = {}

    # Common chain quantities, dumped once: any disagreement here explains
    # every downstream scheduler at once.
    ref = DDIMScheduler(**_fit(DDIMScheduler, base, "epsilon"))
    store["chain.betas"] = ref.betas.to(torch.float32)
    store["chain.alphas_cumprod"] = ref.alphas_cumprod.to(torch.float32)

    # A fixed, deterministic pseudo-denoiser: the trajectory must be
    # reproducible from the goldens alone, so the "model output" is a pure
    # function of (step index, sample) rather than a network.
    def pseudo(i, x):
        return torch.sin(x * (0.7 + 0.1 * i)) * 0.9 - 0.05 * i

    n_elem = 96
    g = torch.Generator().manual_seed(SEED)
    x0 = torch.randn(n_elem, generator=g)
    # Ancestral noise, one draw per step, supplied to brain as a golden so the
    # stochastic sampler is testable without reproducing torch's Philox.
    noise = torch.randn(max(SCHED_STEPS), n_elem, generator=g)
    store["traj.x0"] = x0
    store["traj.noise"] = noise

    global _CLASSES
    _CLASSES = {
        "ddim": DDIMScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpmpp": DPMSolverMultistepScheduler,
    }
    families = {
        "ddim": lambda kw: DDIMScheduler(**kw),
        "euler": lambda kw: EulerDiscreteScheduler(**kw),
        "euler_a": lambda kw: EulerAncestralDiscreteScheduler(**kw),
        "dpmpp": lambda kw: DPMSolverMultistepScheduler(**kw, algorithm_type="dpmsolver++"),
    }
    for pred in ("epsilon", "v_prediction"):
        for fam, make in families.items():
            for n in SCHED_STEPS:
                kw = _fit(_CLASSES[fam], base, pred)
                s = make(kw)
                s.set_timesteps(n)
                pfx = f"{fam}.{pred}.{n}"
                store[f"{pfx}.timesteps"] = s.timesteps.to(torch.float32).cpu()
                if getattr(s, "sigmas", None) is not None:
                    store[f"{pfx}.sigmas"] = s.sigmas.to(torch.float32).cpu()
                # The scale of the INITIAL latent. It is not derivable from the
                # sigma table alone: which of `sigma_max` and `sqrt(sigma_max^2+1)`
                # you get is a function of `timestep_spacing`, and the two differ
                # by ~0.4% — invisible in an image, wrong in every image.
                if hasattr(s, "init_noise_sigma"):
                    store[f"{pfx}.init_noise_sigma"] = torch.tensor(
                        [float(s.init_noise_sigma)], dtype=torch.float32
                    )
                x = x0.clone()
                traj, scaled_rows = [], []
                for i, t in enumerate(s.timesteps):
                    scaled = x if fam == "ddim" else s.scale_model_input(x, t)
                    if fam != "ddim":
                        scaled_rows.append(scaled.clone())
                    m = pseudo(i, scaled)
                    if fam == "euler_a":
                        # The ancestral step draws its own noise through
                        # `randn_tensor`. Replace that draw with row `i` of the
                        # golden `traj.noise` so the trajectory is reproducible
                        # from the goldens alone — everything else stays the
                        # reference implementation.
                        _install_noise(noise[i])
                    x = s.step(m, t, x, return_dict=False)[0]
                    traj.append(x.clone())
                if fam == "euler_a":
                    _restore_noise()
                store[f"{pfx}.traj"] = torch.stack(traj)
                if scaled_rows:
                    store[f"{pfx}.scaled"] = torch.stack(scaled_rows)
    save(store, out, "schedulers/steps.safetensors", manifest)
    manifest["params"]["scheduler_config"] = base
    manifest["params"]["sched_steps"] = SCHED_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdxl", required=True, help="stabilityai/stable-diffusion-xl-base-1.0 dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=32, help="latent H=W (32 => a 256x256 image)")
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-schedulers", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    # MERGE into an existing manifest rather than replacing it. The two golden
    # sets are independent and are routinely dumped in two runs (`--skip-unet`,
    # then `--skip-schedulers` — the UNet leg needs 10 GB of RAM and is the one
    # you re-run). A fresh dict here silently threw away the first run's
    # `files` entry and its `params`, leaving the surviving .safetensors with no
    # recorded sha256, no tensor shapes and no reference config — which is
    # exactly what happened to `schedulers/steps.safetensors` in the committed
    # fixture set.
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
            "sdxl_weights": os.path.abspath(args.sdxl),
        }
    )
    if not args.skip_schedulers:
        print("== schedulers ==", flush=True)
        dump_schedulers(args.sdxl, args.out, manifest)
    if not args.skip_unet:
        print("== unet ==", flush=True)
        dump_unet(args.sdxl, args.out, args.latent, manifest)

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
