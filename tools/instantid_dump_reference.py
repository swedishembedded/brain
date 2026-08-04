#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump InstantID reference activations for brain's parity ladder.

Two independent pieces, because they fail differently:

  * the **image_proj Resampler** — the ArcFace 512-d embedding becomes 16 ID
    tokens of 2048. Small, self-contained, and runnable without SDXL, so it is
    dumped stage by stage (proj_in, every layer's attention and feed-forward,
    proj_out, norm_out).
  * the **decoupled cross-attention** — one `IPAttnProcessor` site: given image
    hidden states and the ID tokens, the ID branch's own k/v projections produce
    a residual that is ADDED to the text cross-attention output with its own
    scale. Dumped on real `to_k_ip`/`to_v_ip` weights at both SDXL widths.

Usage (weights default to the released layout under resources/):

    python3 tools/instantid_dump_reference.py \
        --ckpt /path/to/instantid/ip-adapter.bin \
        --out  testdata/instantid
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def load_reference(code_root):
    """Import the upstream Resampler / IPAttnProcessor rather than reimplementing them."""
    sys.path.insert(0, code_root)
    from ip_adapter.resampler import Resampler  # noqa: E402
    return Resampler


def dump(out_dir, name, arrays, meta):
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/workspace/resources/identity/weights/instantid/ip-adapter.bin")
    ap.add_argument("--code", default="/data/workspace/resources/identity/code/InstantID")
    ap.add_argument("--out", default="testdata/instantid")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    Resampler = load_reference(a.code)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    ip_proj, ip_attn = sd["image_proj"], sd["ip_adapter"]

    # Shapes are read from the checkpoint, never assumed: proj_in tells us the
    # ArcFace width and the model dim, proj_out the token width, latents the
    # query count, and the layer keys the depth.
    embedding_dim = ip_proj["proj_in.weight"].shape[1]
    dim = ip_proj["proj_in.weight"].shape[0]
    num_queries = ip_proj["latents"].shape[1]
    output_dim = ip_proj["proj_out.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in ip_proj if k.startswith("layers."))
    heads = ip_proj["layers.0.0.to_q.weight"].shape[0] // 64

    model = Resampler(
        dim=dim, depth=depth, dim_head=64, heads=heads,
        num_queries=num_queries, embedding_dim=embedding_dim,
        output_dim=output_dim, ff_mult=4,
    ).eval()
    missing, unexpected = model.load_state_dict(ip_proj, strict=True), None
    print(f"resampler: dim={dim} depth={depth} heads={heads} queries={num_queries} "
          f"embed={embedding_dim} out={output_dim}", file=sys.stderr)

    g = torch.Generator().manual_seed(a.seed)
    # A realistic ArcFace input: the released embedding is NOT unit-norm
    # (‖e‖ ~ 15-20), and the resampler is not scale-invariant, so a unit vector
    # here would gate the wrong operating point.
    x = torch.randn(1, 1, embedding_dim, generator=g) * 3.0

    taps = {"input": x}
    with torch.no_grad():
        latents = model.latents.repeat(x.size(0), 1, 1)
        taps["latents_init"] = latents.clone()
        h = model.proj_in(x)
        taps["proj_in"] = h.clone()
        for i, (attn, ff) in enumerate(model.layers):
            latents = attn(h, latents) + latents
            taps[f"layer{i}_attn"] = latents.clone()
            latents = ff(latents) + latents
            taps[f"layer{i}_ff"] = latents.clone()
        latents = model.proj_out(latents)
        taps["proj_out"] = latents.clone()
        out = model.norm_out(latents)
        taps["id_tokens"] = out.clone()

    arrays = {k: v.detach().numpy().astype(np.float32) for k, v in taps.items()}

    # The decoupled branch, at both SDXL cross-attention widths. `to_k_ip`/
    # `to_v_ip` are bias-free and map the 2048-wide ID tokens into the site's
    # hidden size; the reference adds `scale * ip_out` to the text-attention
    # result (attention_processor.py: `hidden_states + self.scale * ip_hidden_states`).
    sites = {}
    for key in ("1.to_k_ip.weight", "3.to_k_ip.weight"):
        if key not in ip_attn:
            continue
        idx = key.split(".")[0]
        wk, wv = ip_attn[f"{idx}.to_k_ip.weight"], ip_attn[f"{idx}.to_v_ip.weight"]
        hidden = wk.shape[0]
        n_img = 12
        q = torch.randn(1, n_img, hidden, generator=g)
        with torch.no_grad():
            k = torch.nn.functional.linear(out, wk)
            v = torch.nn.functional.linear(out, wv)
            nh = hidden // 64
            qh = q.view(1, n_img, nh, 64).transpose(1, 2)
            kh = k.view(1, num_queries, nh, 64).transpose(1, 2)
            vh = v.view(1, num_queries, nh, 64).transpose(1, 2)
            att = torch.softmax(qh @ kh.transpose(-1, -2) / 8.0, dim=-1)
            ip_out = (att @ vh).transpose(1, 2).reshape(1, n_img, hidden)
        sites[f"site{idx}_hidden"] = hidden
        arrays[f"site{idx}_q"] = q.numpy().astype(np.float32)
        arrays[f"site{idx}_k"] = k.numpy().astype(np.float32)
        arrays[f"site{idx}_v"] = v.numpy().astype(np.float32)
        arrays[f"site{idx}_out"] = ip_out.numpy().astype(np.float32)

    meta = {
        "seed": a.seed,
        "resampler": {
            "dim": dim, "depth": depth, "heads": heads, "dim_head": 64,
            "num_queries": num_queries, "embedding_dim": embedding_dim,
            "output_dim": output_dim, "ff_mult": 4,
        },
        "ip_adapter_sites": len(ip_attn) // 2,
        "site_hidden": sites,
        "note": "scale is applied by the CALLER: hidden = text_attn + scale * ip_out",
    }
    dump(a.out, "resampler.safetensors", arrays, meta)
    print(f"wrote {a.out}/resampler.safetensors ({len(arrays)} tensors), "
          f"{len(ip_attn)//2} ip sites", file=sys.stderr)


if __name__ == "__main__":
    main()
