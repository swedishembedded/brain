#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a Chronos-2 reference forecast for the brain parity ladder (T5).

Runs the official Chronos2Pipeline on a fixed, deterministic context and writes
the quantile forecast + the context as raw little-endian f32, for a brain test to
read and compare (cosine / max-abs). No network: loads from a local checkpoint.

Usage:
  python3 tools/goldens/chronos2_dump_reference.py \
      --repo   resources/.../chronos-forecasting \
      --ckpt   resources/.../chronos-2 \
      --out    testdata/golden/chronos2
"""
import argparse
import os
import struct
import sys

import numpy as np


def synth_context(n=200):
    """A deterministic, structured series (trend + seasonality + mild noise),
    identical to the brain-side generator, so both sides forecast the same input."""
    t = np.arange(n, dtype=np.float64)
    x = 0.02 * t + np.sin(2 * np.pi * t / 24.0) + 0.3 * np.sin(2 * np.pi * t / 7.0)
    x += 0.05 * np.sin(t * 1.3)  # deterministic wiggle, no RNG
    return x.astype(np.float32)


def write_f32(path, arr):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--context", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "src"))
    import torch
    from chronos.chronos2 import Chronos2Pipeline

    torch.manual_seed(0)
    pipe = Chronos2Pipeline.from_pretrained(args.ckpt, device_map="cpu")
    pipe.model.eval()

    ctx = synth_context(args.context)
    with torch.no_grad():
        preds = pipe.predict([torch.tensor(ctx)], prediction_length=args.horizon)
    # preds is a list; one element per input series.
    q = preds[0].detach().cpu().numpy().astype(np.float32)
    print("reference forecast shape:", q.shape, "quantile levels:", list(pipe.quantiles))

    # squeeze a leading variate axis if present -> [n_quantiles, horizon]
    q = np.squeeze(q)
    if q.ndim != 2:
        raise SystemExit(f"unexpected forecast ndim {q.ndim}, shape {q.shape}")
    nq, h = q.shape
    print(f"normalized to [{nq}, {h}]")

    os.makedirs(args.out, exist_ok=True)
    write_f32(os.path.join(args.out, "t5_context.f32"), ctx)
    write_f32(os.path.join(args.out, "t5_quantiles.f32"), q)  # [nq, horizon] row-major
    # a tiny metadata JSON so the brain test knows the shape + levels
    import json
    meta = {
        "context_len": int(args.context),
        "horizon": int(args.horizon),
        "n_quantiles": int(nq),
        "quantile_levels": [float(v) for v in pipe.quantiles],
    }
    with open(os.path.join(args.out, "t5_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
