#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a GenieRedux tokenizer reference reconstruction for brain parity.

Loads the released 100M CoinRun tokenizer, runs a DETERMINISTIC synthetic video
through it (encode -> cosine VQ -> decode), and writes the input, reconstruction
and codebook indices as raw little-endian binary that the Rust parity test reads.

Run from the brain repo root (needs torch + the GenieRedux repo on PYTHONPATH):
  python scripts/parity-dump/genie_tokenizer.py

Outputs (gitignored scratch): scratchpad/parity/genie_tokenizer_{in,recon}.f32,
genie_tokenizer_idx.u32, genie_tokenizer_meta.txt
"""
import os, sys, struct

REPO = os.environ.get("BRAIN_GENIEREDUX_REPO") or "resources/world-models/repos/GenieRedux"
if not os.path.isdir(REPO):
    sys.exit(
        f"GenieRedux checkout not found at {REPO!r} - set BRAIN_GENIEREDUX_REPO "
        "to the repo (the dump imports its models/ package)"
    )
CKPT = "scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt"
OUT = "scratchpad/parity"

def main():
    sys.path.insert(0, REPO)
    import torch
    from models.tokenizer import Tokenizer

    torch.manual_seed(0)
    model = Tokenizer(dim=512, codebook_size=1024, image_size=64, patch_size=4,
                      temporal_patch_size=1, num_blocks=8, codebook_dim=32,
                      dim_head=64, heads=8, channels=3, wandb_mode="disabled")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    model.load_state_dict(sd, strict=True)
    model.eval()

    # deterministic input video [b=1, c=3, f=5, 64, 64] in [-1, 1]
    b, c, f, hw = 1, 3, 5, 64
    video = (torch.rand(b, c, f, hw, hw) * 2 - 1)

    with torch.no_grad():
        recon = model(video, return_recons_only=True)
        idx = model(video, return_only_codebook_ids=True)

    os.makedirs(OUT, exist_ok=True)
    def w_f32(name, t):
        arr = t.detach().contiguous().float().view(-1).numpy()
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(arr.tobytes())
        return arr.shape[0]
    n_in = w_f32("genie_tokenizer_in.f32", video)
    n_re = w_f32("genie_tokenizer_recon.f32", recon)
    idx_flat = idx.detach().contiguous().view(-1).to(torch.int64).numpy().astype("uint32")
    with open(os.path.join(OUT, "genie_tokenizer_idx.u32"), "wb") as fh:
        fh.write(idx_flat.tobytes())

    with open(os.path.join(OUT, "genie_tokenizer_meta.txt"), "w") as fh:
        fh.write(f"b={b} c={c} f={f} hw={hw}\n")
        fh.write(f"in_numel={n_in} recon_numel={n_re} idx_numel={idx_flat.shape[0]}\n")
        fh.write(f"recon_shape={tuple(recon.shape)} idx_shape={tuple(idx.shape)}\n")
    print("wrote parity dump to", OUT)
    print("  video", tuple(video.shape), "recon", tuple(recon.shape), "idx", tuple(idx.shape))
    print("  recon range", float(recon.min()), float(recon.max()))

if __name__ == "__main__":
    main()
