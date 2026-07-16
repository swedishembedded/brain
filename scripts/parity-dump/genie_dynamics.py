#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a GenieRedux guided-dynamics (MaskGIT) reference forward for brain parity.

Loads the tokenizer (for the use_token codebook blend) + the guided dynamics,
runs a DETERMINISTIC (token IDs + masks + action one-hots) input through
MaskGIT.forward, and writes the inputs and the next-token logits as raw
little-endian binary for the Rust parity test.

  python scripts/parity-dump/genie_dynamics.py     (needs torch + beartype + einops)

Outputs (gitignored): scratchpad/parity/genie_dynamics_{ids.u32,actions.f32,logits.f32,meta.txt}
"""
import os, sys

REPO = "/data/workspace/resources/world-models/repos/GenieRedux"
TOK = "scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt"
DYN = "scratchpad/wm-checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0.pt"
OUT = "scratchpad/parity"

def main():
    sys.path.insert(0, REPO)
    import torch
    from models.tokenizer import Tokenizer
    from models.dynamics import MaskGIT

    torch.manual_seed(0)
    tokenizer = Tokenizer(dim=512, codebook_size=1024, image_size=64, patch_size=4,
                          temporal_patch_size=1, num_blocks=8, codebook_dim=32,
                          dim_head=64, heads=8, channels=3, wandb_mode="disabled")
    tsd = torch.load(TOK, map_location="cpu", weights_only=False)["model"]
    if hasattr(tsd, "state_dict"): tsd = tsd.state_dict()
    tokenizer.load_state_dict(tsd, strict=True)
    tokenizer.eval()

    maskgit = MaskGIT(dim=512, is_guided=True, action_dim=7, num_tokens=1024,
                      heads=8, dim_head=64, num_blocks=12, max_seq_len=4000,
                      image_size=64, patch_size=4, use_token=True)
    dsd = torch.load(DYN, map_location="cpu", weights_only=False)["model"]
    if hasattr(dsd, "state_dict"): dsd = dsd.state_dict()
    # model = {"dynamics": OrderedDict{"maskgit.*": ...}}
    if "dynamics" in dsd:
        dsd = dsd["dynamics"]
    mg = {k[len("maskgit."):]: v for k, v in dsd.items() if k.startswith("maskgit.")}
    maskgit.load_state_dict(mg, strict=True)
    maskgit.eval()

    b, t, h, w, na = 1, 5, 16, 16, 7
    ids = torch.randint(0, 1024, (b, t, h, w))
    maskbool = torch.rand(b, t, h, w) < 0.4
    ids[maskbool] = 1024  # mask_id = num_tokens
    actions = torch.zeros(b, t, na)
    for i in range(t):
        actions[0, i, i % na] = 1.0

    with torch.no_grad():
        logits = maskgit(ids, actions, tokenizer=tokenizer)  # [b, n, 1024]

    os.makedirs(OUT, exist_ok=True)
    ids.contiguous().view(-1).to(torch.int64).numpy().astype("uint32").tofile(os.path.join(OUT, "genie_dynamics_ids.u32"))
    actions.contiguous().view(-1).float().numpy().tofile(os.path.join(OUT, "genie_dynamics_actions.f32"))
    logits.contiguous().view(-1).float().numpy().tofile(os.path.join(OUT, "genie_dynamics_logits.f32"))
    with open(os.path.join(OUT, "genie_dynamics_meta.txt"), "w") as fh:
        fh.write(f"b={b} t={t} h={h} w={w} na={na}\n")
        fh.write(f"logits_shape={tuple(logits.shape)} n_masked={int(maskbool.sum())}\n")
    print("wrote dynamics parity dump; logits", tuple(logits.shape),
          "range", float(logits.min()), float(logits.max()))

if __name__ == "__main__":
    main()
