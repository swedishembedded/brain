#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Florence-2 reference goldens for brain's `crates/florence2` parity ladder.

Two ladders, captured from the REAL microsoft/Florence-2-base checkpoint via
forward hooks (a pure replay in Rust, no hand-derived convention):

  davit/stageN.safetensors   DaViT vision tower output after each of the 4
                              stages' blocks (post patch-embed AND post the
                              stage's block stack), plus the final pooled
                              577-token projected vision-token sequence.
  encdec/*.safetensors        the BART-style encoder-decoder: encoder output
                              over a fixed vision+text input, and one decoder
                              step's logits (teacher-forced, fixed target ids).
  manifest.json                every tensor's shape/dtype, sha256, the
                              reference config, and run parameters.

Everything is CPU + fp32, fixed seeds, a fixed synthetic 768x768 image (no
real photo needed/committed - the point is bit-reproducible numbers, not a
realistic picture) so the golden is regenerable from the checkpoint alone.

Usage:
  python3 tools/goldens/florence2_dump_reference.py \
      --checkpoint /path/to/microsoft/Florence-2-base \
      --out testdata/florence2
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

SEED = 0


def save(out_dir, rel, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest[rel] = {
        "sha256": sha,
        "bytes": os.path.getsize(path),
        "dtype": "F32",
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
    }


def fixed_image(batch=1):
    """Deterministic synthetic 768x768 RGB image - a smooth gradient plus a
    few hard edges (checkerboard patches), so windowed AND channel attention
    both see real spatial structure rather than uniform input (which would
    make every window/channel-group identical and hide indexing bugs)."""
    rng = np.random.default_rng(SEED)
    h = w = 768
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base = np.stack([xx, yy, (xx + yy) / 2], axis=0).astype(np.float32)
    checker = (((np.arange(h)[:, None] // 32) + (np.arange(w)[None, :] // 32)) % 2).astype(np.float32)
    base = base * 0.7 + checker[None, :, :] * 0.3
    noise = rng.normal(0, 0.02, size=base.shape).astype(np.float32)
    img = np.clip(base + noise, 0, 1)
    img = np.tile(img[None], (batch, 1, 1, 1))
    # Normalize like CLIPImageProcessor (image_mean/image_std from preprocessor_config.json).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    return torch.from_numpy(img)


def dump_davit(model, pixel_values, out_dir, manifest):
    # The real instantiated Florence2ForConditionalGeneration flattens the
    # projection layers (image_projection, image_pos_embed,
    # visual_temporal_embed, image_proj_norm) onto itself rather than
    # nesting them inside a separate Florence2VisionModelWithProjection -
    # confirmed by inspecting named_children(), not assumed from the class
    # definitions alone. `model.vision_tower` IS the DaViT directly.
    davit = model.vision_tower

    stage_outputs = {}
    hooks = []

    def make_hook(stage_idx):
        def hook(module, inp, out):
            # DaViT stage block output, still (B, N, C) sequence form.
            stage_outputs[stage_idx] = out[0].detach().clone()
        return hook

    for i, block in enumerate(davit.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    conv_outputs = {}

    def make_conv_hook(stage_idx):
        def hook(module, inp, out):
            # ConvEmbed output: (x, (H, W)) tuple - x is (B, N, C).
            conv_outputs[stage_idx] = out[0].detach().clone()
        return hook

    for i, conv in enumerate(davit.convs):
        hooks.append(conv.register_forward_hook(make_conv_hook(i)))

    # Fine-grained taps on stage 0's single (spatial, channel) block pair -
    # isolates SpatialBlock and ChannelBlock from each other and from later
    # stages, so each can be implemented and parity-checked independently.
    sub_outputs = {}

    def make_sub_hook(name):
        def hook(module, inp, out):
            sub_outputs[name] = out[0].detach().clone()
        return hook

    pair0 = davit.blocks[0][0]
    hooks.append(pair0.spatial_block.register_forward_hook(make_sub_hook("stage0_spatial_block")))
    hooks.append(pair0.channel_block.register_forward_hook(make_sub_hook("stage0_channel_block")))

    with torch.no_grad():
        unpooled = davit.forward_features_unpool(pixel_values)
        projected = model._encode_image(pixel_values)

    for h in hooks:
        h.remove()

    tensors = {"pixel_values": pixel_values, "unpooled": unpooled, "projected": projected}
    for i, out in stage_outputs.items():
        tensors[f"stage{i}"] = out
    for i, out in conv_outputs.items():
        tensors[f"conv{i}"] = out
    for name, out in sub_outputs.items():
        tensors[name] = out
    save(out_dir, "davit/stages.safetensors", tensors, manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    model.float()  # checkpoint ships fp16; goldens are fp32 (brain's safetensors reader is F32/F16/BF16-only)

    manifest = {"config": model.config.to_dict(), "seed": SEED}
    pixel_values = fixed_image(batch=1)
    dump_davit(model, pixel_values, args.out, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote goldens to {args.out}")


if __name__ == "__main__":
    main()
