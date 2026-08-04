#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump FLUX.1 (dev / Kontext) reference goldens for brain parity tests.

Runs the diffusers `FluxKontextPipeline` (CPU, fp32) and dumps **per-stage**
transformer goldens captured with forward hooks during a REAL pipeline run —
never hand-assembled inputs, so packing order, id layout and timestep scaling
are frozen exactly as the pipeline produced them.

Files written under `--out`:

  cond.safetensors        CLIP-L pooled vector + T5-XXL sequence embeddings
  vae.safetensors         deterministic image -> moments/mean -> scaled+packed
                          latent -> unpack -> decode (16-ch AutoencoderKL)
  dit<tag>.safetensors    transformer I/O + EVERY block boundary, full depth
  dit_small<tag>.safetensors  the same at reduced depth (--small-double /
                          --small-single blocks kept), so a machine that cannot
                          hold the 12B fp32 model still has an exact-math gate
  manifest.json           shapes + sha256 per file + run params + versions

`<tag>` is "" for the text-to-image run and "_edit" for the Kontext edit run
(one reference image appended as extra tokens, ids carrying the axis-0 == 1
offset, prediction truncated to the noise span).

Per-file tensors (dit*):
  hs [n_img, 64]          packed latent tokens entering `x_embedder`
  ctx [txt, 4096]         T5 sequence entering `context_embedder`
  pooled [768]            CLIP-L pooled vector entering `text_embedder`
  timestep [1]            in [0,1]  (the pipeline already divided by 1000)
  guidance [1]            raw guidance scale (3.5); the model multiplies by 1000
  img_ids [n_img, 3]      3-axis position ids of the image (+ ref) tokens
  txt_ids [txt, 3]        3-axis position ids of the text tokens (all zero)
  out [n_pred, 64]        the transformer's prediction (noise span only)
  temb [3072]             time+guidance+pooled conditioning vector
  db{n}_img / db{n}_txt   double-block n output streams
  sg{n}_img / sg{n}_txt   single-block n output streams
  pre_final [n_img, 3072] hidden states entering `norm_out`

Usage:
  python tools/flux1_dump_reference.py \
      --weights /path/to/FLUX.1-Kontext-dev \
      --out testdata/flux1/kontext-dev \
      [--height 256 --width 256 --steps 4 --seed 42 --t5-len 256]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

PROMPT = "a red fox sitting on a mossy rock in a misty forest, morning light"
EDIT_PROMPT = "make the fox blue and add falling snow"


def det_image(h, w):
    """Deterministic RGB test pattern in [-1, 1], shape (3, h, w)."""
    ys = torch.linspace(0, 3.14159, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0, 6.28318, w).unsqueeze(0).expand(h, w)
    r = torch.sin(xs + ys)
    g = torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = 2.0 * (ys / 3.14159) - 1.0
    return torch.stack([r, g, b], 0).contiguous()


def save(out, name, tensors, manifest):
    # everything as f32 — brain's safetensors reader is F32/F16/BF16-only, and
    # position ids / token ids are exactly representable
    tensors = {
        k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()
    }
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {
        "sha256": h,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    keys = list(tensors)
    head = ", ".join(f"{k}{list(tensors[k].shape)}" for k in keys[:6])
    print(f"wrote {name}: {len(keys)} tensors [{head}{' ...' if len(keys) > 6 else ''}]",
          flush=True)


def hook_blocks(tr, cap):
    """Register per-block output hooks; returns the handles to remove."""
    handles = []

    def mk(prefix, idx):
        def post(mod, args, kwargs, out):
            # every FLUX block returns (encoder_hidden_states, hidden_states)
            ehs, hs = out
            cap.setdefault(f"{prefix}{idx}_txt", ehs[0].clone())
            cap.setdefault(f"{prefix}{idx}_img", hs[0].clone())
            return None

        return post

    for i, b in enumerate(tr.transformer_blocks):
        handles.append(b.register_forward_hook(mk("db", i), with_kwargs=True))
    for i, b in enumerate(tr.single_transformer_blocks):
        handles.append(b.register_forward_hook(mk("sg", i), with_kwargs=True))

    def norm_pre(mod, args, kwargs):
        a = kwargs.get("x", args[0] if args else None)
        cap.setdefault("pre_final", a[0].clone())
        cap.setdefault("temb", (kwargs.get("conditioning_embedding", args[1])) [0].clone())
        return None

    handles.append(tr.norm_out.register_forward_pre_hook(norm_pre, with_kwargs=True))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t5-len", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--small-double", type=int, default=2)
    ap.add_argument("--small-single", type=int, default=2)
    ap.add_argument("--skip-full", action="store_true",
                    help="dump only the reduced-depth goldens (needs ~4 GB, not ~48 GB)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    torch.manual_seed(args.seed)

    from diffusers import FluxKontextPipeline

    print("loading pipeline fp32 (this needs ~70 GB of host RAM) ...", flush=True)
    pipe = FluxKontextPipeline.from_pretrained(args.weights, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    tr = pipe.transformer

    # ---- conditioning --------------------------------------------------------
    # Self-validation: the manual CLIP/T5 path must agree with encode_prompt.
    with torch.no_grad():
        t5_ids = pipe.tokenizer_2(
            PROMPT, padding="max_length", max_length=args.t5_len,
            truncation=True, return_length=False, return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        t5_seq = pipe.text_encoder_2(t5_ids, output_hidden_states=False)[0]
        clip_ids = pipe.tokenizer(
            PROMPT, padding="max_length", max_length=pipe.tokenizer_max_length,
            truncation=True, return_overflowing_tokens=False, return_length=False,
            return_tensors="pt",
        ).input_ids
        clip_out = pipe.text_encoder(clip_ids, output_hidden_states=False)
        pooled = clip_out.pooler_output
        pe, pooled_pipe, _ = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=PROMPT, device="cpu",
            num_images_per_prompt=1, max_sequence_length=args.t5_len,
        )
    d_seq = (t5_seq - pe).abs().max().item()
    d_pool = (pooled - pooled_pipe).abs().max().item()
    print(f"manual vs encode_prompt: t5 {d_seq:.3e}  pooled {d_pool:.3e}", flush=True)
    assert d_seq < 1e-4 and d_pool < 1e-4, "manual conditioning path diverges"
    save(args.out, "cond.safetensors",
         {"t5_input_ids": t5_ids[0].to(torch.int32),
          "clip_input_ids": clip_ids[0].to(torch.int32),
          "t5_seq": t5_seq[0], "clip_pooled": pooled[0]}, manifest)

    # ---- vae (16-channel AutoencoderKL) --------------------------------------
    vae = pipe.vae
    img = det_image(args.height, args.width).unsqueeze(0)
    sf = float(vae.config.scaling_factor)
    shift = float(vae.config.shift_factor)
    with torch.no_grad():
        moments = vae.encode(img).latent_dist.parameters       # (1, 32, H/8, W/8)
        mean = moments.chunk(2, dim=1)[0]                      # (1, 16, H/8, W/8)
        scaled = (mean - shift) * sf
        packed = pipe._pack_latents(scaled, 1, 16, scaled.shape[2], scaled.shape[3])
        unpacked = pipe._unpack_latents(packed, args.height, args.width,
                                        pipe.vae_scale_factor)
        dec = vae.decode(unpacked / sf + shift).sample         # (1, 3, H, W)
    save(args.out, "vae.safetensors",
         {"image": img[0], "moments": moments[0], "latent_mean": mean[0],
          "latent_scaled": scaled[0], "latent_packed": packed[0], "decoded": dec[0],
          "scaling_factor": torch.tensor([sf]), "shift_factor": torch.tensor([shift])},
         manifest)

    # ---- transformer runs with per-stage hooks -------------------------------
    from PIL import Image
    import numpy as np

    ref_np = (det_image(args.height, args.width).permute(1, 2, 0) + 1) * 127.5
    ref_pil = Image.fromarray(ref_np.numpy().astype(np.uint8))

    def run(name, ref_image, prompt):
        cap = {}

        def pre_hook(mod, a, kw):
            if "hs" in cap:
                return None
            cap["hs"] = kw["hidden_states"][0].clone()
            cap["ctx"] = kw["encoder_hidden_states"][0].clone()
            cap["pooled"] = kw["pooled_projections"][0].clone()
            cap["timestep"] = kw["timestep"][:1].clone().float()
            g = kw.get("guidance")
            cap["guidance"] = (g[:1].clone().float() if g is not None
                               else torch.zeros(1))
            cap["img_ids"] = kw["img_ids"].clone().float()
            cap["txt_ids"] = kw["txt_ids"].clone().float()
            return None

        def post_hook(mod, a, kw, out):
            if "out" not in cap:
                cap["out"] = (out[0] if isinstance(out, tuple) else out.sample)[0].clone()
            return None

        h = [tr.register_forward_pre_hook(pre_hook, with_kwargs=True),
             tr.register_forward_hook(post_hook, with_kwargs=True)]
        h += hook_blocks(tr, cap)

        gen = torch.Generator("cpu").manual_seed(args.seed)
        # `max_area` defaults to 1024**2 and the pipeline SILENTLY rescales the
        # requested height/width up to it — pinning it keeps the dump at the
        # resolution asked for (and the CPU forward affordable).
        kw = dict(prompt=prompt, height=args.height, width=args.width,
                  num_inference_steps=args.steps, generator=gen, output_type="np",
                  guidance_scale=args.guidance, max_sequence_length=args.t5_len,
                  max_area=args.height * args.width)
        if ref_image is not None:
            kw["image"] = ref_image
            kw["_auto_resize"] = False
        res = pipe(**kw)
        for x in h:
            x.remove()
        cap["final_image"] = torch.from_numpy(res.images[0])
        save(args.out, name, cap, manifest)
        # record what was ACTUALLY produced, not what was requested: the
        # pipeline silently rescales height/width to `max_area`, and a manifest
        # that reports the request is a landmine for whoever reads it later.
        manifest[name]["actual"] = {
            "txt_tokens": cap["ctx"].shape[0],
            "img_tokens": cap["hs"].shape[0],
            "image_hw": list(cap["final_image"].shape[:2]),
            "depth_double": len(tr.transformer_blocks),
            "depth_single": len(tr.single_transformer_blocks),
        }

    full_dbl = len(tr.transformer_blocks)
    full_sgl = len(tr.single_transformer_blocks)
    if not args.skip_full:
        run("dit.safetensors", None, PROMPT)
        run("dit_edit.safetensors", [ref_pil], EDIT_PROMPT)

    # ---- reduced depth: the SAME weights, first N blocks only -----------------
    keep_d = min(args.small_double, full_dbl)
    keep_s = min(args.small_single, full_sgl)
    tr.transformer_blocks = torch.nn.ModuleList(list(tr.transformer_blocks)[:keep_d])
    tr.single_transformer_blocks = torch.nn.ModuleList(
        list(tr.single_transformer_blocks)[:keep_s]
    )
    run("dit_small.safetensors", None, PROMPT)
    run("dit_small_edit.safetensors", [ref_pil], EDIT_PROMPT)

    import diffusers

    manifest["params"] = {
        "prompt": PROMPT, "edit_prompt": EDIT_PROMPT, "seed": args.seed,
        "height": args.height, "width": args.width, "steps": args.steps,
        "t5_len": args.t5_len, "guidance": args.guidance,
        "depth_double": full_dbl, "depth_single": full_sgl,
        "small_double": keep_d, "small_single": keep_s,
        "skip_full": args.skip_full,
        "weights": os.path.abspath(args.weights),
        "torch": torch.__version__, "diffusers": diffusers.__version__,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
