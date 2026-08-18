#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Dump a real-dims, random-weight reference golden for Qwen3.8-27B's vision
tower (M9) - `Qwen/Qwen3.8-27B-FP8`'s `Qwen3_5VisionModel`, at its REAL
default dims (depth=27, hidden=1152, num_heads=16, intermediate=4304,
patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
num_position_embeddings=2304, in_channels=3), with `out_hidden_size`
overridden to this model's own decoder `d_model` (5120) - the one field the
approved port plan calls out as genuinely model-specific (everything else is
byte-for-byte `qwen3vl::config::VisionConfig::qwen3_omni()`, confirmed by
comparing `Qwen3_5VisionConfig()`'s own printed defaults against that
preset's fields, one by one, before writing this dumper).

Unlike `qwen35_dump_reference.py`'s text golden, no real checkpoint is
downloaded here (M9's own scope: real dims, RANDOM weights - full real-weight
vision parity is M10's job, gated on `BRAIN_QWEN35_DIR`). This dumper freshly
constructs `Qwen3_5VisionModel` under a fixed seed instead.

Runs the REAL `transformers.models.qwen3_5.Qwen3_5VisionModel` directly (never
a vendored copy) - `crates/qwen3vl::encoder::VisionEncoder`/`PatchMerger` are
REUSED UNCHANGED for this model (per the port plan; already exercised by
`crates/qwen3vl`'s and `crates/qwen3omnimoe`'s own test suites at the
`qwen3_omni()` preset), so the question this golden answers is narrow and
model-specific: does `Qwen3_5VisionModel`'s OWN forward - patch embed, the
bilinearly-resampled learned pos-embed, each transformer block (2-D vision
RoPE, tanh-GELU, no QK-norm), and the final PatchMerger (LayerNorm + erf-GELU
MLP) - match what that already-reused Rust code produces, tensor for tensor,
at THIS model's real dims. `Qwen3_5VisionModel.forward` conveniently returns
both stages directly: `last_hidden_state` (pre-merger) and `pooler_output`
(post-merger) - no forward hooks needed for those two; block-output taps
still use hooks, one per probed block (first and last, catching a
systematic per-block error early and confirming it does not compound).

Usage:
  python tools/goldens/qwen35_vision_dump_reference.py [--out DIR] [--seed N]
"""

import argparse
import hashlib
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import transformers
from safetensors.torch import save_file

from transformers.models.qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

OUT_HIDDEN_SIZE = 5120  # this model's own decoder d_model, not the HF default


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.3f} MB")


def collect_weights(model, cfg):
    """Rename `Qwen3_5VisionModel`'s own parameter names to brain's
    `crates/qwen3vl::encoder` convention (`VisionEncoder::new`/`PatchMerger::
    new`'s own required-key docs): `patch_embed.proj.weight [hidden, C, kT,
    kH, kW]` flattens to `patch_embed.weight [hidden, patch_vec]` (a Conv3d
    weight IS a `[out, in*kT*kH*kW]` matmul weight up to this reshape - no
    value changes, same convention `qwen3vl::import` already uses for this
    conv-as-matmul reuse elsewhere); `pos_embed.weight` -> `pos_embed`;
    `blocks.{b}.attn.qkv.*` -> `blocks.{b}.qkv.*`, `blocks.{b}.attn.proj.*` ->
    `blocks.{b}.proj.*` (attention leaves de-nested); `blocks.{b}.mlp.
    linear_fc{1,2}.*` -> `blocks.{b}.fc{1,2}.*`; `blocks.{b}.norm{1,2}.*`
    unchanged; `merger.norm.*` -> `merger.ln.*`, `merger.linear_fc{1,2}.*` ->
    `merger.fc{1,2}.*`.
    """
    sd = model.state_dict()
    out = {}
    pv = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    out["patch_embed.weight"] = sd.pop("patch_embed.proj.weight").reshape(cfg.hidden_size, pv)
    out["patch_embed.bias"] = sd.pop("patch_embed.proj.bias")
    out["pos_embed"] = sd.pop("pos_embed.weight")
    for b in range(cfg.depth):
        p = f"blocks.{b}."
        out[f"{p}norm1.weight"] = sd.pop(f"{p}norm1.weight")
        out[f"{p}norm1.bias"] = sd.pop(f"{p}norm1.bias")
        out[f"{p}norm2.weight"] = sd.pop(f"{p}norm2.weight")
        out[f"{p}norm2.bias"] = sd.pop(f"{p}norm2.bias")
        out[f"{p}qkv.weight"] = sd.pop(f"{p}attn.qkv.weight")
        out[f"{p}qkv.bias"] = sd.pop(f"{p}attn.qkv.bias")
        out[f"{p}proj.weight"] = sd.pop(f"{p}attn.proj.weight")
        out[f"{p}proj.bias"] = sd.pop(f"{p}attn.proj.bias")
        out[f"{p}fc1.weight"] = sd.pop(f"{p}mlp.linear_fc1.weight")
        out[f"{p}fc1.bias"] = sd.pop(f"{p}mlp.linear_fc1.bias")
        out[f"{p}fc2.weight"] = sd.pop(f"{p}mlp.linear_fc2.weight")
        out[f"{p}fc2.bias"] = sd.pop(f"{p}mlp.linear_fc2.bias")
    out["merger.ln.weight"] = sd.pop("merger.norm.weight")
    out["merger.ln.bias"] = sd.pop("merger.norm.bias")
    out["merger.fc1.weight"] = sd.pop("merger.linear_fc1.weight")
    out["merger.fc1.bias"] = sd.pop("merger.linear_fc1.bias")
    out["merger.fc2.weight"] = sd.pop("merger.linear_fc2.weight")
    out["merger.fc2.bias"] = sd.pop("merger.linear_fc2.bias")
    assert not sd, f"unclaimed vision tensors, collect_weights out of date: {list(sd.keys())}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "qwen35", "vision"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Qwen3_5VisionConfig(out_hidden_size=OUT_HIDDEN_SIZE)
    # Pin every dim this dumper's own doc claims is the real default -- a
    # future transformers upgrade changing one of these silently would
    # otherwise desync this golden from the real checkpoint without warning.
    assert cfg.depth == 27 and cfg.hidden_size == 1152 and cfg.num_heads == 16
    assert cfg.intermediate_size == 4304 and cfg.patch_size == 16
    assert cfg.temporal_patch_size == 2 and cfg.spatial_merge_size == 2
    assert cfg.num_position_embeddings == 2304 and cfg.in_channels == 3

    torch.manual_seed(args.seed)
    model = Qwen3_5VisionModel(cfg).eval()

    # One image, grid (t=1, h=4, w=4) = 2x2 merge-blocks of 2x2 patches each
    # (spatial_merge_size=2) -- small enough to keep the golden tiny, large
    # enough to exercise the merger's real 2x2 gather and more than one
    # merge-block (a single merge-block grid would hide a cross-block
    # ordering bug the way `qwen35_dump_reference.py`'s own tiny text config
    # avoids degenerate dims for the same reason).
    t, h, w = 1, 4, 4
    pv = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    n_patches = t * h * w
    patches = torch.arange(n_patches * pv, dtype=torch.float32)
    patches = (patches % 17 - 8.0) / 8.0
    patches = patches.reshape(n_patches, pv)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)

    block_taps = {}
    tap_indices = [0, cfg.depth - 1]

    def mk_hook(i):
        def hook(_mod, _inp, out):
            block_taps[i] = out.detach().clone()

        return hook

    hooks = [model.blocks[i].register_forward_hook(mk_hook(i)) for i in tap_indices]
    with torch.no_grad():
        out = model(hidden_states=patches, grid_thw=grid_thw)
    for hk in hooks:
        hk.remove()

    # Reproduce from a fresh construction + seed: determinism (same self-check
    # `qwen35_dump_reference.py`'s own text dumper runs).
    torch.manual_seed(args.seed)
    model2 = Qwen3_5VisionModel(cfg).eval()
    with torch.no_grad():
        out2 = model2(hidden_states=patches, grid_thw=grid_thw)
    assert torch.equal(out.last_hidden_state, out2.last_hidden_state), "fresh-construction determinism failed"
    print("fresh-construction determinism: OK (bit-identical)")

    tensors = {
        "patches": patches,
        "grid_thw": grid_thw.to(torch.int32),
        "hidden": out.last_hidden_state.contiguous(),
        "merged": out.pooler_output.contiguous(),
    }
    for i in tap_indices:
        tensors[f"block{i}"] = block_taps[i].contiguous()

    save(args.out, "qwen35_vision.safetensors", tensors, manifest := {})
    weights = collect_weights(model, cfg)
    save(args.out, "qwen35_vision_weights.safetensors", weights, manifest)

    manifest["_meta"] = {
        "seed": args.seed,
        "t": t, "h": h, "w": w,
        "tap_indices": tap_indices,
        "vision_config": {
            "depth": cfg.depth, "hidden_size": cfg.hidden_size, "num_heads": cfg.num_heads,
            "intermediate_size": cfg.intermediate_size, "patch_size": cfg.patch_size,
            "temporal_patch_size": cfg.temporal_patch_size, "spatial_merge_size": cfg.spatial_merge_size,
            "num_position_embeddings": cfg.num_position_embeddings, "in_channels": cfg.in_channels,
            "out_hidden_size": cfg.out_hidden_size,
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest.json -> {args.out}")


if __name__ == "__main__":
    main()
