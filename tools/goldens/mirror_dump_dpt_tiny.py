#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Tiny-grid DPT-head golden for stage-level isolation (T3b).

Random seeded weights with the REAL channel widths but a 4x4 patch grid
(56x56 image) — runs in seconds. Dumps per-stage intermediates so the Rust
test (crates/mirror/tests/t3_dpt_tiny.rs) can pinpoint the first divergence:
rn[0..3] (post projects/pos/resize/rn-conv), fused (post refinenet1),
full (post output_conv1+bilinear+pos), out (post output_conv2).

Run from the repo root with the reference repo importable:
  python3 tools/goldens/mirror_dump_dpt_tiny.py \
      --repo <clone of HY-World-2.0>/hyworld2/worldrecon \
      --out crates/mirror/tests/golden/dpt_tiny.json
"""
import argparse
import json
import sys
import types

import numpy as np
import torch

torch.set_default_dtype(torch.float32)


def sample(t, k=64, seed=0):
    a = t.detach().numpy().astype(np.float32).reshape(-1)
    rng = np.random.RandomState(seed)
    idx = rng.choice(a.size, size=min(k, a.size), replace=False)
    idx.sort()
    return {
        "shape": list(t.shape),
        "rms": float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
        "indices": idx.tolist(),
        "values": a[idx].astype(float).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    for name in ["flash_attn", "flash_attn.flash_attn_interface", "flash_attn_interface"]:
        m = types.ModuleType(name)
        m.flash_attn_func = None
        sys.modules[name] = m
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)

    from hyworldmirror.models.heads.dense_head import DPTHead

    head = DPTHead(dim_in=2048, output_dim=3, patch_size=14, activation="exp+expp1+linear", enable_depth_mask=True)
    head.eval()
    # deterministic random weights
    g = torch.Generator().manual_seed(42)
    state = head.state_dict()
    for k in list(state.keys()):
        # scale by fan-in so magnitudes stay sane through the deep conv stack
        v = state[k]
        fan = max(1, int(np.prod(v.shape[1:])) if v.dim() > 1 else 1)
        state[k] = torch.empty_like(v).uniform_(-1, 1, generator=g) / np.sqrt(fan)
    head.load_state_dict(state)

    ph = pw = 4
    H = W = ph * 14
    td = 7 + ph * pw
    tokens = torch.empty(1, 1, td, 2048).uniform_(-1, 1, generator=g)
    images = torch.empty(1, 1, 3, H, W).uniform_(0, 1, generator=g)

    # ---- capture intermediates via hooks/manual reimplementation of _forward_impl ----
    caps = {}
    feats = []
    B, S = 1, 1
    tok_list = [tokens, tokens, tokens, tokens]
    for i, (proj, resize) in enumerate(zip(head.projects, head.resize_layers)):
        pt = tok_list[i][:, :, 7:]
        pt = pt.reshape(B * S, -1, pt.shape[-1])
        pt = head.norm(pt)
        feat = pt.permute(0, 2, 1).reshape(B * S, pt.shape[-1], ph, pw)
        feat = proj(feat)
        feat = head._apply_pos_embed(feat, W, H)
        feat = resize(feat)
        feats.append(feat)
    rn = [
        head.scratch.layer1_rn(feats[0]),
        head.scratch.layer2_rn(feats[1]),
        head.scratch.layer3_rn(feats[2]),
        head.scratch.layer4_rn(feats[3]),
    ]
    for i, r in enumerate(rn):
        caps[f"rn{i}"] = sample(r[0], seed=20 + i)
    out4 = head.scratch.refinenet4(rn[3], size=rn[2].shape[2:])
    caps["out4"] = sample(out4[0], seed=40)
    out3 = head.scratch.refinenet3(out4, rn[2], size=rn[1].shape[2:])
    caps["out3"] = sample(out3[0], seed=41)
    out2 = head.scratch.refinenet2(out3, rn[1], size=rn[0].shape[2:])
    caps["out2"] = sample(out2[0], seed=42)
    out1 = head.scratch.refinenet1(out2, rn[0])
    caps["out1"] = sample(out1[0], seed=43)
    fused = head.scratch.output_conv1(out1)
    caps["fused"] = sample(fused[0], seed=30)
    from hyworldmirror.models.heads.dense_head import custom_interpolate

    full = custom_interpolate(fused, size=(H, W), mode="bilinear", align_corners=True)
    full = head._apply_pos_embed(full, W, H)
    caps["full"] = sample(full[0], seed=31)
    out = head.scratch.output_conv2(full)
    caps["out"] = sample(out[0], seed=32)

    import os
    os.makedirs(args.out, exist_ok=True)
    for k, v in state.items():
        v.numpy().astype(np.float32).tofile(f"{args.out}/w_{k}.bin")
    tokens.numpy().astype(np.float32).tofile(f"{args.out}/tokens.bin")
    images.numpy().astype(np.float32).tofile(f"{args.out}/images.bin")
    with open(f"{args.out}/stages.json", "w") as f:
        json.dump(caps, f, indent=0)
    print("wrote", args.out, {k: round(v["rms"], 3) for k, v in caps.items()})


if __name__ == "__main__":
    main()
