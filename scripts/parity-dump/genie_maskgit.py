#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a GenieRedux MaskGIT sampler (Dynamics.sample) reference output for
brain parity. inference_steps=1 -> temperature collapses to 0 -> deterministic
argmax, so the sampled tokens are reproducible.

  python scripts/parity-dump/genie_maskgit.py

Outputs (gitignored): scratchpad/parity/genie_maskgit_{prime.u32,actions.f32,out.u32,meta.txt}
"""
import os, sys

REPO = os.environ.get("BRAIN_GENIEREDUX_REPO") or "resources/world-models/repos/GenieRedux"
if not os.path.isdir(REPO):
    sys.exit(
        f"GenieRedux checkout not found at {REPO!r} - set BRAIN_GENIEREDUX_REPO "
        "to the repo (the dump imports its models/ package)"
    )
TOK = "scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt"
DYN = "scratchpad/wm-checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0.pt"
OUT = "scratchpad/parity"

def main():
    sys.path.insert(0, REPO)
    import torch
    from models.tokenizer import Tokenizer
    from models.dynamics import MaskGIT, Dynamics

    torch.manual_seed(0)
    tokenizer = Tokenizer(dim=512, codebook_size=1024, image_size=64, patch_size=4,
                          temporal_patch_size=1, num_blocks=8, codebook_dim=32,
                          dim_head=64, heads=8, channels=3, wandb_mode="disabled")
    tsd = torch.load(TOK, map_location="cpu", weights_only=False)["model"]
    if hasattr(tsd, "state_dict"): tsd = tsd.state_dict()
    tokenizer.load_state_dict(tsd, strict=True); tokenizer.eval()

    maskgit = MaskGIT(dim=512, is_guided=True, action_dim=7, num_tokens=1024,
                      heads=8, dim_head=64, num_blocks=12, max_seq_len=4000,
                      image_size=64, patch_size=4, use_token=True)
    dsd = torch.load(DYN, map_location="cpu", weights_only=False)["model"]
    if hasattr(dsd, "state_dict"): dsd = dsd.state_dict()
    if "dynamics" in dsd: dsd = dsd["dynamics"]
    mg = {k[len("maskgit."):]: v for k, v in dsd.items() if k.startswith("maskgit.")}
    maskgit.load_state_dict(mg, strict=True); maskgit.eval()

    dynamics = Dynamics(maskgit=maskgit, inference_steps=1, sample_temperature=1.0, mask_schedule="cosine")
    dynamics.eval()

    h = w = 16
    prime_frames = 4          # context frames' tokens
    t_fwd = prime_frames      # forward sees these after [:, :-1]
    prime = torch.randint(0, 1024, (1, prime_frames * h * w))
    actions = torch.zeros(1, t_fwd, 7)
    for i in range(t_fwd):
        actions[0, i, i % 7] = 1.0

    with torch.no_grad():
        out = dynamics.sample(
            prime_token_ids=prime, actions=actions,
            num_tokens=h * w, patch_shape=(prime_frames + 1, h, w),
            inference_steps=1, sample_temperature=1.0, mask_schedule="cosine",
            tokenizer=tokenizer,
        )

    os.makedirs(OUT, exist_ok=True)
    prime.contiguous().view(-1).to(torch.int64).numpy().astype("uint32").tofile(os.path.join(OUT, "genie_maskgit_prime.u32"))
    actions.contiguous().view(-1).float().numpy().tofile(os.path.join(OUT, "genie_maskgit_actions.f32"))
    out.contiguous().view(-1).to(torch.int64).numpy().astype("uint32").tofile(os.path.join(OUT, "genie_maskgit_out.u32"))
    with open(os.path.join(OUT, "genie_maskgit_meta.txt"), "w") as fh:
        fh.write(f"prime_frames={prime_frames} t_fwd={t_fwd} h={h} w={w} num_tokens={h*w}\n")
        fh.write(f"out_shape={tuple(out.shape)}\n")
    print("wrote maskgit parity dump; out", tuple(out.shape), "sample tokens", out.view(-1)[:8].tolist())

if __name__ == "__main__":
    main()
