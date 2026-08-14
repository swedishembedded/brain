#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a small full ZImageTransformer2DModel forward golden.

Validates the whole assembly over the (already bit-exact) block: t/x/cap
embedders, patchify, noise/context refiners, the [image, cap] unified sequence,
main layers, FinalLayer, and unpatchify. Small dims + no padding (lengths
multiple of SEQ_MULTI_OF=32) keep it fast. Dev-time only.
"""
import os, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

# testdata/golden/zimage/... -- where crates/s3dit/tests/model_parity.rs's
# testdata("golden/zimage/zimage_model.safetensors") actually looks.
TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[2] / "testdata")
OUT = str(Path(TESTDATA) / "golden" / "zimage" / "zimage_model.safetensors")

DIM, N_LAYERS, N_REF, N_HEADS = 48, 2, 1, 2   # head_dim 24
CAP_FEAT_DIM, IN_CH = 16, 16
AXES_DIMS, AXES_LENS = [8, 8, 8], [64, 32, 32]
H, W, CAP_LEN = 16, 8, 32                       # patches 8*4=32 (mult of 32)

def main():
    torch.manual_seed(0)
    model = ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=IN_CH, dim=DIM,
        n_layers=N_LAYERS, n_refiner_layers=N_REF, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=CAP_FEAT_DIM,
        rope_theta=256.0, t_scale=1000.0, axes_dims=AXES_DIMS, axes_lens=AXES_LENS,
    ).float().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)

    latent = torch.randn(IN_CH, 1, H, W)     # [C, F, H, W]
    cap = torch.randn(CAP_LEN, CAP_FEAT_DIM)
    t = torch.tensor([0.4])

    with torch.no_grad():
        out = model([latent], t, [cap], patch_size=2, f_patch_size=1, return_dict=True).sample[0]

    sd = {k: v.float().contiguous() for k, v in model.state_dict().items()}
    print(f"state_dict: {len(sd)} tensors")
    for k in sorted(sd):
        print(f"  {k:44s} {tuple(sd[k].shape)}")

    tensors = dict(sd)
    tensors["_latent"] = latent.contiguous()      # [C,1,H,W]
    tensors["_cap"] = cap.contiguous()            # [cap_len, cap_feat_dim]
    tensors["_t"] = t.contiguous()                # [1]
    tensors["_out"] = out.contiguous()            # [C,1,H,W] -> squeeze F later
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata={"dim": str(DIM), "n_layers": str(N_LAYERS)})
    print(f"wrote {OUT}  out {tuple(out.shape)} [min,max]=[{out.min():.4f},{out.max():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
