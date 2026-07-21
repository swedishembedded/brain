#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a Chronos-2 KNOWN-FUTURE reference forecast for brain's parity gate.

Target + one covariate in a single group (group_ids=[0,0]); the covariate's FUTURE
is provided (future_covariates row = real values), the target's future is unknown
(NaN). This exercises the reference's known-future-covariate path, where the
covariate's future values flow into the future patches. We keep the target's
quantile row and compare against brain's forecast_quantiles_mv_kf.

Usage: python3 tools/chronos2_dump_kf_reference.py --repo <chronos-forecasting>
       --ckpt <chronos-2 dir> --out crates/chronos2/tests/golden
"""
import argparse, os, struct, sys, json
import numpy as np


def series(n, seed):
    t = np.arange(n, dtype=np.float64)
    return (0.02 * t + np.sin(2 * np.pi * t / 24.0 + seed)
            + 0.3 * np.sin(2 * np.pi * t / 7.0 + 0.5 * seed) + 0.05 * np.sin(t * (1.3 + seed))).astype(np.float32)


def write_f32(path, arr):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--context", type=int, default=200)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "src"))
    import torch
    from chronos.chronos2 import Chronos2Pipeline
    torch.manual_seed(0)
    model = Chronos2Pipeline.from_pretrained(args.ckpt, device_map="cpu").model
    model.eval()

    C, H = args.context, args.horizon
    # covariate defined over the whole span; its future is the true continuation.
    cov_full = series(C + H, 1.7)
    target = series(C, 0.0)
    cov_ctx = cov_full[:C]
    cov_fut = cov_full[C:C + H]

    context = torch.stack([torch.tensor(target), torch.tensor(cov_ctx)])  # [2, C]
    ops = model.chronos_config.output_patch_size
    nop = (H + ops - 1) // ops
    # future_covariates [2, H]: target NaN (unknown), covariate = real future.
    fut = torch.full((2, H), float("nan"))
    fut[1, :] = torch.tensor(cov_fut)
    with torch.no_grad():
        out = model.forward(context=context, group_ids=torch.tensor([0, 0]),
                            future_covariates=fut, num_output_patches=nop)
    q = out.quantile_preds.detach().cpu().numpy().astype(np.float32)  # [2, Q, nop*ops]
    tq = q[0, :, :H]
    nq, h = tq.shape
    print(f"KF target quantiles [{nq},{h}]")

    os.makedirs(args.out, exist_ok=True)
    write_f32(os.path.join(args.out, "kf_target.f32"), target)
    write_f32(os.path.join(args.out, "kf_cov.f32"), cov_ctx)
    write_f32(os.path.join(args.out, "kf_cov_future.f32"), cov_fut)
    write_f32(os.path.join(args.out, "kf_quantiles.f32"), tq)
    with open(os.path.join(args.out, "kf_meta.json"), "w") as f:
        json.dump({"context_len": C, "horizon": H, "n_quantiles": int(nq)}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
