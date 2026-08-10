#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake the Real-ESRGAN parity goldens for `crates/upscale/tests/parity.rs`.

Imports the UPSTREAM `RRDBNet` (basicsr's `rrdbnet_arch`) when it is importable
and otherwise reconstructs it from the paper's definition here, so the goldens
are the reference's numbers rather than a second implementation of brain's.
Taps the same rungs the Rust parity ladder reads: `conv_first`, the first RRDB,
the trunk residual join, each upsample stage, and the output.

A DELIBERATELY SMALL config by default (`--num-block 2` on random weights) so the
gate runs in CI without the 67 MB release checkpoint. Point `--ckpt` at
`RealESRGAN_x4plus.pth` to bake the real thing as well; both land in the same
file under different prefixes.

Usage:
  python3 tools/goldens/esrgan_dump_reference.py [--ckpt RealESRGAN_x4plus.pth]
                                                 [--size 32] [--out <mirror>/esrgan]
"""

import argparse
import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

# Overridable machine path (scripts/gates/check-scripts.sh 3/3).
GOLDEN_MIRROR = os.environ.get("BRAIN_GOLDEN_MIRROR") or sys.exit("set BRAIN_GOLDEN_MIRROR=<goldens mirror dir> (no baked-in default: this path is machine-specific)")


class ResidualDenseBlock(nn.Module):
    """basicsr `ResidualDenseBlock` — five convs over a growing concat."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # The 0.2 is the architecture, not a tunable.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward_taps(self, x):
        """The forward, returning every rung the Rust ladder compares."""
        taps = {}
        feat = self.conv_first(x)
        taps["conv_first"] = feat
        b = feat
        for i, blk in enumerate(self.body):
            b = blk(b)
            if i == 0:
                taps["body.0"] = b
        feat = feat + self.conv_body(b)
        taps["body_out"] = feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        taps["up1"] = feat
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        taps["up2"] = feat
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        taps["out"] = out
        return taps


def build(num_feat, num_block, num_grow_ch):
    """Prefer the UPSTREAM class; fall back to the definition above."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet as Upstream  # noqa: F401

        print("using basicsr's RRDBNet")
        net = Upstream(3, 3, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch)
        # The upstream class has no tap hook; borrow ours, which is the same graph.
        net.forward_taps = RRDBNet.forward_taps.__get__(net, type(net))
        return net
    except Exception as e:  # noqa: BLE001
        print(f"basicsr unavailable ({type(e).__name__}); using the in-file definition")
        return RRDBNet(3, 3, num_feat, num_block, num_grow_ch)


def dump(net, size, prefix, seed):
    net.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, size, size, generator=g)
    with torch.no_grad():
        taps = net.forward_taps(x)
    out = {f"{prefix}input": x.contiguous()}
    for k, v in taps.items():
        out[f"{prefix}{k}"] = v.contiguous()
    return out, {k: dict(net.state_dict())[k] for k in net.state_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="RealESRGAN_x4plus.pth (optional)")
    ap.add_argument("--size", type=int, default=32, help="input side (output is 4x)")
    ap.add_argument("--num-block", type=int, default=2, help="blocks for the tiny gate")
    ap.add_argument("--num-feat", type=int, default=16)
    ap.add_argument("--num-grow-ch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=f"{GOLDEN_MIRROR}/esrgan")
    a = ap.parse_args()

    tensors = {}

    # 1. The tiny, checkpoint-free gate. Dims chosen so num_feat, num_grow_ch and
    #    the image side all DIFFER — a degenerate config would hide a width swap
    #    (docs/lessons.md #4).
    tiny = build(a.num_feat, a.num_block, a.num_grow_ch)
    torch.manual_seed(a.seed)
    for p in tiny.parameters():
        nn.init.uniform_(p, -0.25, 0.25)
    t, w = dump(tiny, a.size, "tiny_", a.seed)
    tensors.update(t)
    for k, v in w.items():
        tensors[f"tiny_w_{k}"] = v.contiguous()
    print(f"tiny: num_feat={a.num_feat} num_grow_ch={a.num_grow_ch} num_block={a.num_block} size={a.size}")

    # 2. The released checkpoint, if one was named.
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        w = sd.get("params_ema", sd.get("params", sd))
        net = build(64, 23, 32)
        net.load_state_dict(w, strict=True)
        t, _ = dump(net, a.size, "x4plus_", a.seed + 1)
        tensors.update(t)
        print(f"x4plus: loaded {a.ckpt}")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "rrdbnet.safetensors"))
    print(f"wrote {out / 'rrdbnet.safetensors'} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
