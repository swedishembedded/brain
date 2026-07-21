#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a Chronos-2 *multivariate* reference forecast for the brain parity gate.

Two series (target + one past covariate) placed in a single group (group_ids=[0,0])
with unknown futures — the reference's joint/multivariate path, where group
attention lets the target attend to the covariate. We keep the TARGET's quantile
row and compare it against brain's `forecast_quantiles_mv([target, cov], h)`.

Usage:
  python3 tools/chronos2_dump_mv_reference.py \
      --repo <chronos-forecasting> --ckpt <chronos-2 dir> --out crates/chronos2/tests/golden
"""
import argparse, os, struct, sys, json
import numpy as np


def series(n, seed):
    t = np.arange(n, dtype=np.float64)
    x = (0.02 * t + np.sin(2 * np.pi * t / 24.0 + seed)
         + 0.3 * np.sin(2 * np.pi * t / 7.0 + 0.5 * seed) + 0.05 * np.sin(t * (1.3 + seed)))
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
    model = pipe.model
    model.eval()

    target = series(args.context, 0.0)
    cov = series(args.context, 1.7)  # a correlated-but-distinct covariate
    context = torch.stack([torch.tensor(target), torch.tensor(cov)])  # [2, L]
    group_ids = torch.tensor([0, 0], dtype=torch.long)

    ops = model.chronos_config.output_patch_size
    nop = (args.horizon + ops - 1) // ops
    with torch.no_grad():
        out = model.forward(context=context, group_ids=group_ids, num_output_patches=nop)
    q = out.quantile_preds.detach().cpu().numpy().astype(np.float32)  # [2, Q, nop*ops]
    print("mv quantile_preds shape:", q.shape)
    tq = q[0, :, : args.horizon]  # target row -> [Q, horizon]
    nq, h = tq.shape
    print(f"target quantiles [{nq}, {h}]  levels={list(pipe.quantiles)}")

    os.makedirs(args.out, exist_ok=True)
    write_f32(os.path.join(args.out, "mv_target.f32"), target)
    write_f32(os.path.join(args.out, "mv_cov.f32"), cov)
    write_f32(os.path.join(args.out, "mv_quantiles.f32"), tq)
    with open(os.path.join(args.out, "mv_meta.json"), "w") as f:
        json.dump({"context_len": args.context, "horizon": args.horizon,
                   "n_quantiles": int(nq), "quantile_levels": [float(v) for v in pipe.quantiles]}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
