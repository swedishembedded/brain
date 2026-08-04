#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a REAL Z-Image-Turbo (6B) DiT forward golden from the shipped weights.

Loads the Comfy single-file, remaps original→diffusers (same map as brain's
import_comfy: split fused qkv, rename out/q_norm/k_norm/x_embedder/final_layer),
loads into the turbo-config ZImageTransformer2DModel, and runs a forward on a
small real input. Validates brain's real-weight path end-to-end (import + bf16).
Small latent (16x16 -> 64 patches, mult of 32; cap 32) keeps the 30-layer 6B
forward tractable on CPU. Dev-time only.
"""
import os, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

if len(sys.argv) < 2:
    sys.exit("usage: zimage_real_dump_reference.py <z_image_turbo_bf16.safetensors> [out.safetensors]")
COMFY = sys.argv[1]
# testdata/golden/zimage/... by default -- where
# crates/zimage/tests/real_parity.rs's testdata("golden/zimage/zimage_real.safetensors")
# actually looks.
_TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[2] / "testdata")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(Path(_TESTDATA) / "golden" / "zimage" / "zimage_real.safetensors")
DIM = 3840

def remap(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith(".attention.qkv.weight"):
            base = k[: -len("qkv.weight")]
            out[base + "to_q.weight"] = v[:DIM].clone()
            out[base + "to_k.weight"] = v[DIM : 2 * DIM].clone()
            out[base + "to_v.weight"] = v[2 * DIM :].clone()
            continue
        k2 = (k.replace(".attention.out.", ".attention.to_out.0.")
               .replace(".attention.k_norm.weight", ".attention.norm_k.weight")
               .replace(".attention.q_norm.weight", ".attention.norm_q.weight"))
        if k2.startswith("x_embedder."):
            k2 = "all_x_embedder.2-1." + k2[len("x_embedder."):]
        elif k2.startswith("final_layer."):
            k2 = "all_final_layer.2-1." + k2[len("final_layer."):]
        out[k2] = v
    return out

def main():
    model = ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=DIM,
        n_layers=30, n_refiner_layers=2, n_heads=30, n_kv_heads=30, norm_eps=1e-5,
        qk_norm=True, cap_feat_dim=2560, rope_theta=256.0, t_scale=1000.0,
        axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512],
    )
    sd = remap(load_file(COMFY))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load: {len(missing)} missing, {len(unexpected)} unexpected")
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    assert not missing, f"missing: {missing[:6]}"
    model = model.float().eval()

    torch.manual_seed(0)
    latent = torch.randn(16, 1, 16, 16)
    cap = torch.randn(32, 2560)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out = model([latent], t, [cap], patch_size=2, f_patch_size=1).sample[0]

    save_file(
        {"_latent": latent.contiguous(), "_cap": cap.contiguous(), "_t": t.contiguous(), "_out": out.contiguous()},
        OUT, metadata={"model": "Z-Image-Turbo 6B real"},
    )
    print(f"wrote {OUT}  out {tuple(out.shape)} [min,max,mean]=[{out.min():.4f},{out.max():.4f},{out.mean():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
