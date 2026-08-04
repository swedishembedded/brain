#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a single ZImageTransformerBlock forward golden (small config).

Exercises the hard new DiT logic in isolation: adaLN modulation (scale=1+, gate
=tanh), the double-RMSNorm sandwich, QK-norm attention with multi-axis
interleaved RoPE, and SwiGLU. brain replicates with the same weights/inputs.
Small dims keep it fast; dim<256 so the adaLN input == dim (folding logic is
identical for the real dim=3840/cdim=256 case). Dev-time only.
"""
import os, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock, RopeEmbedder

# testdata/golden/zimage/... -- where crates/zimage/tests/block_parity.rs's
# testdata("golden/zimage/zimage_block.safetensors") actually looks.
TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[1] / "testdata")
OUT = str(Path(TESTDATA) / "golden" / "zimage" / "zimage_block.safetensors")

DIM, N_HEADS = 48, 2          # head_dim = 24
AXES_DIMS, AXES_LENS = [8, 8, 8], [16, 8, 8]   # sum = 24 = head_dim
THETA, EPS, T = 256.0, 1e-5, 8

def main():
    torch.manual_seed(0)
    blk = ZImageTransformerBlock(
        layer_id=0, dim=DIM, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        norm_eps=EPS, qk_norm=True, modulation=True,
    ).float().eval()

    # Random but deterministic weights (default init is fine; reseed for adaLN
    # so the modulation isn't ~0).
    for p in blk.parameters():
        torch.nn.init.normal_(p, std=0.05)

    cdim = min(DIM, 256)
    x = torch.randn(1, T, DIM)
    c = torch.randn(1, cdim)  # adaln_input (t_embedder output)

    rope = RopeEmbedder(theta=THETA, axes_dims=AXES_DIMS, axes_lens=AXES_LENS)
    ids = torch.zeros(T, 3, dtype=torch.long)
    ids[:, 0] = torch.arange(T)  # positions along axis 0
    freqs_cis = rope(ids).unsqueeze(0)  # [1, T, head_dim/2] complex

    with torch.no_grad():
        out = blk(x, attn_mask=None, freqs_cis=freqs_cis, adaln_input=c)

    sd = {k: v.float().contiguous() for k, v in blk.state_dict().items()}
    print("block state_dict keys:")
    for k in sd:
        print(f"  {k:36s} {tuple(sd[k].shape)}")

    tensors = dict(sd)
    tensors["_x"] = x.squeeze(0).contiguous()          # [T, dim]
    tensors["_c"] = c.squeeze(0).contiguous()          # [cdim]
    tensors["_cos"] = freqs_cis.real.contiguous()      # [T, half]
    tensors["_sin"] = freqs_cis.imag.contiguous()      # [T, half]
    tensors["_out"] = out.squeeze(0).contiguous()      # [T, dim]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata={"dim": str(DIM), "n_heads": str(N_HEADS), "T": str(T)})
    print(f"wrote {OUT}  out {tuple(out.shape)} [min,max]=[{out.min():.4f},{out.max():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
