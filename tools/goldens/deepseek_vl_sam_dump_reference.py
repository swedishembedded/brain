#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump DeepSeek-VL's high-resolution SAM tower reference for `crates/sam1`.

DeepSeek-VL's high-resolution vision tower is `sam_b_downsample`: SAM ViT-B at
1024x1024 plus two additions over the tower DeepSeek-OCR ships --

  * the neck output is bilinearly resized to 96x96
    (`F.interpolate(mode="bilinear", align_corners=False)`) before the two
    stride-2 `downsamples` convs, so the output grid is 24x24;
  * an "HD" branch: the output of the FIRST global-attention block goes through
    a second neck (`neck_hd`), the same resize and the SAME `downsamples`
    weights, and is added to the main output scaled by the learned scalar
    `hd_alpha_downsamples`.

This script runs the pinned reference module (`deepseek_vl.models.sam`,
imported from a DeepSeek-VL checkout) on the real checkpoint's tower weights,
in fp32 on the CPU, over a fixed synthetic 1024x1024 input, and writes:

  golden.safetensors   every tensor as f32:
                         input          [3, 1024, 1024]  the synthetic input
                         block02_out    [64, 64, 768]    = global_features[0]
                         block11_out    [64, 64, 768]
                         neck_out       [256, 64, 64]    main neck
                         neck_resized   [256, 96, 96]    main neck, resized
                         main_out       [1024, 24, 24]   downsamples(main)
                         hd_neck_out    [256, 64, 64]    neck_hd(block02_out)
                         hd_resized     [256, 96, 96]
                         hd_out         [1024, 24, 24]   downsamples(hd), unscaled
                         hd_alpha       [1]              the checkpoint's scalar
                         output         [1024, 24, 24]   the tower's forward()
  manifest.json        shapes, sha256 per file, the source pin and the
                       `source` block (`golden_source.source_block`).

The per-stage taps are recomputed here from the module's own submodules, and
the script asserts that their composition reproduces `forward()` exactly, so
the taps are the reference's intermediates rather than a re-derivation.

The DeepSeek-VL multimodal venv (torch 2.0.1) cannot hand tensors to numpy, so
nothing here goes through numpy: tensors are read and written with
`safetensors.torch` only.

Usage:
  python tools/goldens/deepseek_vl_sam_dump_reference.py \\
      --src <DeepSeek-VL checkout> \\
      --ckpt <deepseek-ai/deepseek-vl-7b-chat snapshot dir> \\
      --out "$BRAIN_TESTDATA/deepseek-vl/sam"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

PREFIX = "vision_model.vision_tower_high.vision_tower."
CHECKPOINT = "deepseek-ai/deepseek-vl-7b-chat"
SEED = 20260930
RESIZE = (96, 96)


def load_tower_state(ckpt):
    """The tower's tensors out of the sharded checkpoint, prefix stripped, as
    fp32. Returns (state_dict, shard files actually read)."""
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for name, shard in weight_map.items():
        if name.startswith(PREFIX):
            by_shard.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
            for n in names:
                state[n[len(PREFIX):]] = f.get_tensor(n).float()
    return state, [os.path.join(ckpt, s) for s in sorted(by_shard)]


def source_commit(src):
    try:
        return subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.environ.get("DEEPSEEK_VL_SRC"), help="DeepSeek-VL checkout (the pinned reference source)")
    ap.add_argument("--ckpt", required=True, help="deepseek-vl-7b-chat snapshot directory (sharded safetensors)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    if not args.src:
        sys.exit("--src (or DEEPSEEK_VL_SRC) must name a DeepSeek-VL checkout")

    sys.path.insert(0, args.src)
    from deepseek_vl.models.sam import create_sam_vit  # noqa: E402

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    tower = create_sam_vit("sam_b_downsample", image_size=1024).float().eval()
    state, shards = load_tower_state(args.ckpt)
    # strict: the checkpoint's tower and the reference module must be the same
    # tensor set, name for name.
    tower.load_state_dict(state, strict=True)
    assert len(state) == 186, f"expected the 186-tensor sam_b_downsample tower, got {len(state)}"

    # A fixed synthetic input in the range a normalized image occupies. Seeded,
    # and stored in the golden, so the Rust side replays it byte for byte.
    g = torch.Generator().manual_seed(SEED)
    x = (torch.rand(1, 3, 1024, 1024, generator=g) * 4.0 - 2.0).float()

    # ---- the reference forward, and its intermediates from its own modules ----
    out = tower(x)

    h = tower.patch_embed(x) + tower.pos_embed
    global_features = []
    taps = {}
    for i, blk in enumerate(tower.blocks):
        h = blk(h)
        if blk.window_size == 0:
            global_features.append(h)
        if i in (2, 11):
            taps[f"block{i:02d}_out"] = h[0]
    assert tower.blocks[2].window_size == 0 and all(b.window_size != 0 for b in tower.blocks[:2]), "block 2 must be the first global block"
    neck = tower.neck(h.permute(0, 3, 1, 2))
    neck_resized = F.interpolate(neck, size=RESIZE, mode="bilinear", align_corners=False)
    main_out = tower.downsamples(neck_resized)
    hd_neck = tower.neck_hd(global_features[0].permute(0, 3, 1, 2))
    hd_resized = F.interpolate(hd_neck, size=RESIZE, mode="bilinear", align_corners=False)
    hd_out = tower.downsamples(hd_resized)
    composed = main_out + hd_out * tower.hd_alpha_downsamples
    assert torch.equal(composed, out), "the per-stage composition must reproduce forward() exactly"

    alpha = tower.hd_alpha_downsamples.detach().reshape(1)
    tensors = {
        "input": x[0],
        **taps,
        "neck_out": neck[0],
        "neck_resized": neck_resized[0],
        "main_out": main_out[0],
        "hd_neck_out": hd_neck[0],
        "hd_resized": hd_resized[0],
        "hd_out": hd_out[0],
        "hd_alpha": alpha,
        "output": out[0],
    }
    tensors = {k: v.detach().float().contiguous() for k, v in tensors.items()}
    print(f"hd_alpha_downsamples = {alpha.item()!r}")
    print(f"output {tuple(out.shape)}  mean {out.mean().item():.6e}  std {out.std().item():.6e}")
    rel_hd = (hd_out * alpha).norm().item() / out.norm().item()
    print(f"|hd_alpha * hd_out| / |output| = {rel_hd:.4e}")

    os.makedirs(args.out, exist_ok=True)
    golden = os.path.join(args.out, "golden.safetensors")
    save_file(tensors, golden)

    manifest = {
        "dumper": "tools/goldens/deepseek_vl_sam_dump_reference.py",
        "reference": {"module": "deepseek_vl.models.sam.create_sam_vit('sam_b_downsample')", "commit": source_commit(args.src)},
        "seed": SEED,
        "dtype": "float32",
        "device": "cpu",
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "hd_alpha": alpha.item(),
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "files": {"golden.safetensors": "sha256:" + sha256_file(golden)},
        "source": source_block(
            checkpoint=CHECKPOINT,
            files=shards,
            hash_files=False,
            identity={
                "d_model": tower.patch_embed.proj.out_channels,
                "n_layers": len(tower.blocks),
                "neck_channels": tower.neck[0].out_channels,
                "compress_out": tower.downsamples[-1].out_channels,
                "resize": RESIZE[0],
            },
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {golden}")


if __name__ == "__main__":
    main()
