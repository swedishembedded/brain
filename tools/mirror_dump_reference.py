#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump WorldMirror-2 reference goldens for the brain parity tests (T1/T2/...).

Run manually (never in the build) against the reference repo + checkpoint:

  python3 tools/mirror_dump_reference.py \
      --repo /data/workspace/resources/world-3d/repos/HY-World-2.0/hyworld2/worldrecon \
      --ckpt /data/workspace/resources/world-3d/checkpoints/HY-WorldMirror-2.0/model.safetensors \
      --out crates/mirror/tests/golden

Stages dumped (each as .npy + a small committed sample in golden_meta.json):
  t1: synthetic 600x400 image -> PIL bicubic resize + crop + ToTensor (u8 + normalized CHW)
  t2: normalized frame -> DINOv2 patch tokens [1369, 1024] (reference module, fp32 CPU)

The synthetic image is deterministic (no asset needed): a gradient + circles
pattern generated here and regenerated identically by the Rust tests.
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def synth_image(w=600, h=400):
    """Deterministic test pattern; the Rust side regenerates it bit-for-bit."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x, 0] = (x * 255) // max(w - 1, 1)
            img[y, x, 1] = (y * 255) // max(h - 1, 1)
            img[y, x, 2] = ((x * 7 + y * 13) // 4) % 256
    # a few hard-edged circles for high-frequency content
    yy, xx = np.mgrid[0:h, 0:w]
    for (cx, cy, r, col) in [(150, 100, 60, (255, 40, 40)), (420, 260, 90, (30, 220, 90)), (300, 180, 30, (250, 250, 250))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[m] = col
    return img


def read_safetensors(path, prefixes):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if not any(name.startswith(p) for p in prefixes):
                continue
            assert info["dtype"] == "F32", name
            s0, s1 = info["data_offsets"]
            f.seek(base + s0)
            buf = f.read(s1 - s0)
            out[name] = torch.from_numpy(
                np.frombuffer(buf, dtype=np.float32).reshape(info["shape"]).copy()
            )
    return out


def sample(arr, k=64, seed=0):
    """Deterministic index sample of a flat array — small enough to commit."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    rng = np.random.RandomState(seed)
    idx = rng.choice(flat.size, size=min(k, flat.size), replace=False)
    idx.sort()
    return {
        "shape": list(np.asarray(arr).shape),
        "rms": float(np.sqrt(np.mean(flat.astype(np.float64) ** 2))),
        "indices": idx.tolist(),
        "values": flat[idx].astype(float).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    # The reference imports flash_attn unconditionally but only calls it on
    # bf16/fp16 tensors; stub it so the fp32 CPU path imports cleanly.
    import types

    def _stub(name, attrs=()):
        m = types.ModuleType(name)
        for a in attrs:
            setattr(m, a, None)
        sys.modules[name] = m
        return m

    _stub("flash_attn")
    _stub("flash_attn.flash_attn_interface", ["flash_attn_func"])
    _stub("flash_attn_interface", ["flash_attn_func"])

    meta = {}

    # ---- T1: preprocessing golden ----
    img = synth_image()
    pil = Image.fromarray(img, "RGB")
    # reference resize_dims for 600x400 target 518 patch 14
    new_w, new_h = 518, round(400 * (518 / 600) / 14) * 14
    resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    r8 = np.asarray(resized)  # HWC u8
    # crop strategy: only crop if > target (here new_h < 518, no crop)
    t = r8.astype(np.float32) / 255.0
    norm = (t - IMAGENET_MEAN) / IMAGENET_STD
    chw = norm.transpose(2, 0, 1).copy()
    meta["t1_resized_u8"] = sample(r8, seed=1)
    meta["t1_norm_chw"] = sample(chw, seed=2)
    meta["t1_dims"] = [new_w, new_h]

    # ---- T2: DINOv2 patch tokens on a square 518x518 crop ----
    # square input via the same reference path: resize a 600x600 center-crop
    sq = pil.crop((41, 0, 441, 400)).resize((518, 518), Image.Resampling.BICUBIC)
    s8 = np.asarray(sq)
    sn = ((s8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(sn.transpose(2, 0, 1)).unsqueeze(0)  # [1,3,518,518]
    meta["t2_input_u8"] = sample(s8, seed=3)

    from hyworldmirror.models.layers.vision_transformer import vit_large

    vit = vit_large(
        img_size=518, patch_size=14, num_register_tokens=4, block_chunks=0, init_values=1.0
    )
    vit.eval()
    weights = read_safetensors(args.ckpt, ["visual_geometry_transformer.patch_embed."])
    state = {k.replace("visual_geometry_transformer.patch_embed.", ""): v for k, v in weights.items()}
    missing, unexpected = vit.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    assert not unexpected
    out = vit.forward_features(x)
    patches = out["x_norm_patchtokens"][0]  # [1369, 1024]
    np.save(f"{args.out}/t2_patch_tokens.npy", patches.numpy())
    meta["t2_patch_tokens"] = sample(patches.numpy(), k=256, seed=4)

    # ---- T4: full trunk taps (1 frame, square 518) via the reference VGT ----
    from hyworldmirror.models.models.visual_transformer import VisualGeometryTransformer

    vgt = VisualGeometryTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        patch_embed="dinov2_vitl14_reg",
        qk_norm=True,
        normalized_rope=True,
        rope_base=100.0,
        rope_normalize_coords="separate",
        init_values=0.01,
        enable_cond=True,
        fixed_patch_embed=True,
        condition_strategy=["token", "pow3r", "token"],
    )
    vgt.eval()
    wall = read_safetensors(args.ckpt, ["visual_geometry_transformer."])
    state = {k.replace("visual_geometry_transformer.", ""): v for k, v in wall.items()}
    missing, unexpected = vgt.load_state_dict(state, strict=False)
    # rope periods alias: assign the single stored buffer to every block
    per = state["frame_blocks.0.attn.rope.periods"]
    for blks in (vgt.frame_blocks, vgt.global_blocks):
        for b in blks:
            if hasattr(b.attn, "rope") and b.attn.rope is not None:
                b.attn.rope.periods.data.copy_(per)
    print("t4 missing:", [m for m in missing if "rope.periods" not in m])
    print("t4 unexpected:", unexpected)
    assert not unexpected
    assert all("rope.periods" in m or "_resnet" in m for m in missing), missing

    imgs = torch.from_numpy(s8.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, None]  # [1,1,3,518,518]
    tap_list, patch_start = vgt(imgs)
    print("taps:", len(tap_list), tap_list[0].shape, "patch_start:", patch_start)
    for i, tap in enumerate(tap_list[:4]):
        meta[f"t4_tap{i}"] = sample(tap[0, 0].numpy(), k=256, seed=10 + i)

    # ---- T5: dense heads + camera head on the T4 taps ----
    from hyworldmirror.models.heads.dense_head import DPTHead
    from hyworldmirror.models.heads.camera_head import CameraHead

    def load_head(mod, prefix):
        st = {k[len(prefix) + 1 :]: v for k, v in read_safetensors(args.ckpt, [prefix + "."]).items()}
        missing, unexpected = mod.load_state_dict(st, strict=False)
        assert not unexpected, (prefix, unexpected)
        assert not missing, (prefix, missing)
        mod.eval()
        return mod

    heads = {
        "depth_head": load_head(
            DPTHead(dim_in=2048, output_dim=3, patch_size=14, activation="exp+expp1+linear", enable_depth_mask=True),
            "depth_head",
        ),
        "pts_head": load_head(DPTHead(dim_in=2048, output_dim=4, patch_size=14, activation="inv_log+expp1"), "pts_head"),
        "norm_head": load_head(DPTHead(dim_in=2048, output_dim=4, patch_size=14, activation="norm+expp1"), "norm_head"),
        "gs_head": load_head(
            DPTHead(dim_in=2048, output_dim=3, patch_size=14, features=256, is_gsdpt=True,
                    activation="exp+expp1+linear", enable_depth_mask=True),
            "gs_head",
        ),
    }
    # capture PRE-activation output_conv2 results (matches brain's buffers)
    captured = {}

    def mk_hook(name):
        def hook(_m, _inp, out):
            captured[name] = out.detach()
        return hook

    for name, headm in heads.items():
        headm.scratch.output_conv2.register_forward_hook(mk_hook(name))

    imgs5 = imgs  # [1,1,3,518,518] in [0,1]
    for name, headm in heads.items():
        with torch.no_grad():
            if name == "gs_head":
                gs_feat, _p, _c, _m = headm(tap_list, imgs5, patch_start, frames_chunk_size=8)
                captured["gs_feat"] = gs_feat.detach()
            else:
                headm(tap_list, imgs5, patch_start, frames_chunk_size=8)
        meta[f"t5_{name}"] = sample(captured[name][0].numpy(), k=128, seed=hash(name) % 1000)

    # gs_renderer parameter convs (built manually — avoids the gsplat import)
    import torch.nn as nn

    gsr = nn.Sequential(
        nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.ReLU(), nn.Conv2d(256, 12, 1)
    )
    st = read_safetensors(args.ckpt, ["gs_renderer."])
    gsr[0].weight.data.copy_(st["gs_renderer.gs_head.0.weight"])
    gsr[2].weight.data.copy_(st["gs_renderer.gs_head.2.weight"])
    gsr[2].bias.data.copy_(st["gs_renderer.gs_head.2.bias"])
    gsr.eval()
    with torch.no_grad():
        gs12 = gsr(captured["gs_feat"][0])  # [S=1, 12, 518, 518]
    meta["t5_gs_params"] = sample(gs12[0].numpy(), k=128, seed=77)

    cam_head = load_head(CameraHead(dim_in=2048), "cam_head")
    with torch.no_grad():
        pred_seq = cam_head([t for t in tap_list], steps=4)
    meta["t5_cam"] = sample(pred_seq[-1][0].numpy(), k=9, seed=88)
    print("cam:", pred_seq[-1][0].numpy())

    with open(f"{args.out}/golden_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
