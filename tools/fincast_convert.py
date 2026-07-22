#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Convert the official FinCast checkpoint (`Vincent05R/FinCast` `v1.pth`, a
PyTorch pickle) into a `.safetensors` file that brain's
`checkpoint::safetensors` reader can consume.

`v1.pth` is a `state_dict` (OrderedDict of fp32 tensors). This script strips the
`torch.compile` / DDP key prefixes exactly like the reference loader
(`ffm_torch_moe.FFmTorch.load_from_checkpoint_ffm`), then writes a flat
safetensors file with the reference's own `state_dict` key names — the same
names brain's `fincast::Config::param_list()` uses.

Usage:
    python3 tools/fincast_convert.py <v1.pth> <out.safetensors>

Not part of the build. torch + safetensors live in the repo venv.
"""
import sys
import torch
from safetensors.torch import save_file


def strip_prefix(k: str) -> str:
    for pre in ("_orig_mod.module.", "_orig_mod.", "module."):
        if k.startswith(pre):
            return k[len(pre):]
    return k


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        nk = strip_prefix(k)
        # contiguous fp32, so the safetensors byte layout is row-major dense.
        out[nk] = v.detach().to(torch.float32).contiguous()
    tot = sum(t.numel() for t in out.values())
    print(f"converting {len(out)} tensors, {tot/1e6:.1f}M params -> {dst}")
    save_file(out, dst)
    print("done")


if __name__ == "__main__":
    main()
