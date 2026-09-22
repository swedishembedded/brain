#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Torch goldens for worldmirror2's DPT spatial kernels, at the shapes it runs.

A shared kernel can be right for the shapes one model exercises and wrong at a
stride, a padding or a fractional resize ratio another needs. These cases are
taken from `crates/worldmirror2/src/dpt.rs`'s own dispatch: the refinenet
upsample chain, the fractional 1.75x final upsample, the transposed
convolutions that upsample the taps, and the strided and 7x7 convolutions.

    WM_RESIZE_GOLDEN="$(mktemp -d)"
    python3 tools/goldens/worldmirror2_dump_resize.py "$WM_RESIZE_GOLDEN"
    WM_RESIZE_GOLDEN="$WM_RESIZE_GOLDEN" cargo test -p brain-worldmirror2 --test t17_resize_parity

The output directory comes from the argument, else `$WM_RESIZE_GOLDEN` - the
same variable the test reads, so the two cannot be pointed at different
directories by accident. No machine path is baked in.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESIZE = [
    # (c, hin, win, hout, wout)
    (4, 208, 272, 364, 476),   # output_conv1 -> full res, a fractional 1.75x
    (4, 26, 34, 52, 68),       # refinenet 2x steps
    (4, 52, 68, 104, 136),
    (4, 104, 136, 208, 272),
    (3, 26, 34, 364, 476),     # patch grid straight to full res
    (2, 7, 5, 13, 11),         # odd extents, fractional both axes
]

CONV = [
    # (name, cin, cout, hin, win, k, stride, pad, transposed)
    ("output_conv1 3x3", 8, 8, 208, 272, 3, 1, 1, False),
    ("output_conv2.0 3x3", 8, 8, 364, 476, 3, 1, 1, False),
    ("output_conv2.2 1x1", 8, 4, 364, 476, 1, 1, 0, False),
    ("input_merger 7x7", 3, 8, 364, 476, 7, 1, 3, False),
    ("refinenet out_conv 1x1", 8, 8, 208, 272, 1, 1, 0, False),
    ("resize_layers.3 s2", 8, 8, 26, 34, 3, 2, 1, False),
    ("resize_layers.0 dconv4", 8, 8, 26, 34, 4, 4, 0, True),
    ("resize_layers.1 dconv2", 8, 8, 26, 34, 2, 2, 0, True),
    ("rcu 3x3 odd", 8, 8, 13, 11, 3, 1, 1, False),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    man = []
    for i, (c, hi, wi, ho, wo) in enumerate(RESIZE):
        g = np.random.default_rng(1234 + i)
        x = g.standard_normal((1, c, hi, wi)).astype(np.float32)
        y = F.interpolate(torch.from_numpy(x), size=(ho, wo), mode="bilinear", align_corners=True)
        x.tofile(out / f"in_{i}.f32")
        y.numpy().astype(np.float32).tofile(out / f"out_{i}.f32")
        man.append(dict(i=i, c=c, hin=hi, win=wi, hout=ho, wout=wo))
    (out / "cases.json").write_text(json.dumps(man))

    cman = []
    for i, (nm, ci, co, hi, wi, k, st, pd, tr) in enumerate(CONV):
        g = np.random.default_rng(999 + i)
        x = g.standard_normal((1, ci, hi, wi)).astype(np.float32)
        b = (g.standard_normal(co) * 0.1).astype(np.float32)
        if tr:
            w = (g.standard_normal((ci, co, k, k)) * 0.1).astype(np.float32)
            y = F.conv_transpose2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=st, padding=pd)
        else:
            w = (g.standard_normal((co, ci, k, k)) * 0.1).astype(np.float32)
            y = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=st, padding=pd)
        yn = y.numpy().astype(np.float32)
        x.tofile(out / f"cin_{i}.f32")
        w.tofile(out / f"cw_{i}.f32")
        b.tofile(out / f"cb_{i}.f32")
        yn.tofile(out / f"cout_{i}.f32")
        cman.append(dict(i=i, name=nm, cin=ci, cout=co, hin=hi, win=wi, k=k, stride=st,
                         pad=pd, transposed=tr, hout=yn.shape[2], wout=yn.shape[3]))
    (out / "conv_cases.json").write_text(json.dumps(cman))
    print(f"wrote {len(RESIZE)} resize + {len(CONV)} conv cases to {out}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WM_RESIZE_GOLDEN")
    if not dest:
        sys.exit(
            "worldmirror2_dump_resize: no output directory - pass one as the first "
            "argument, or set WM_RESIZE_GOLDEN to the directory the parity test reads"
        )
    main(Path(dest))
