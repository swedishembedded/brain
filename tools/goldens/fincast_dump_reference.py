#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a FinCast reference forward for the brain parity ladder.

Runs the *actual* reference model (`PatchedTimeSeriesDecoder_MOE` from
`resources/time-series/repos/FinCast`) on a fixed, seeded context and writes the
denormalized head output for the last patch (= brain's first AR-decode step).

Parity trap: the reference MoE gate is **stochastic even at eval** (it draws
`uniform()` and applies capacity dropping). brain implements the deterministic
top-2 expectation, so this script **neutralizes** the reference stochasticity —
`threshold_eval -> ~0` (always route the top-2) and `capacity_factor_eval` huge
(no capacity dropping) — making the reference a deterministic top-2 oracle that
brain must reproduce.

Outputs (into <out_dir>):
  - golden_meta.json  (committed): config, context, shapes, a sampled subset of
    the output, and its global RMS — enough to gate parity without the raw array.
  - ref_output.npy    (gitignored): the full [horizon_len, num_outputs] output.
  - ref_context.npy   (gitignored): the raw context fed to the model.

Usage:
  python3 tools/goldens/fincast_dump_reference.py <v1.pth> <out_dir> [--ctx 512]

Not part of the build. Needs the repo venv (torch, einx, colt5_attention,
st_moe_pytorch).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REF = os.environ.get("BRAIN_FINCAST_REPO") or sys.exit("set BRAIN_FINCAST_REPO=<FinCast reference repo checkout> (no baked-in default: this path is machine-specific)")


def load_ppd():
    sys.path.insert(0, os.path.join(REF, "src"))
    import st_moe_pytorch  # noqa: F401  (registers the package)
    spec = importlib.util.spec_from_file_location(
        "ppd", os.path.join(REF, "src/ffm/pytorch_patched_decoder_MOE.py"))
    ppd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppd)
    return ppd


def strip_prefix(k):
    for pre in ("_orig_mod.module.", "_orig_mod.", "module."):
        if k.startswith(pre):
            return k[len(pre):]
    return k


def sample(arr, k=64, seed=0):
    flat = arr.reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=min(k, flat.size), replace=False)
    idx.sort()
    return [[int(i), float(flat[i])] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ppd = load_ppd()
    cfg = ppd.FFMConfig(
        num_layers=50, num_heads=16, num_kv_heads=16, hidden_size=1280,
        intermediate_size=1280, head_dim=80, patch_len=32, horizon_len=128,
        num_experts=4, gating_top_n=2, use_positional_embedding=False)
    model = ppd.PatchedTimeSeriesDecoder_MOE(cfg)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = {strip_prefix(k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # neutralize stochastic MoE routing -> deterministic top-2 (see docstring)
    for layer in model.stacked_transformer.layers:
        gate = layer.moe.moe.gate
        gate.threshold_eval = torch.full_like(gate.threshold_eval, 1e-9)
        gate.capacity_factor_eval = 1.0e6

    # fixed seeded context: trend + seasonal + noise
    ctx_len = args.ctx
    t = np.arange(ctx_len, dtype=np.float64)
    rng = np.random.default_rng(1234)
    context = (100.0 + 0.05 * t + 5.0 * np.sin(2 * np.pi * t / 64.0)
               + rng.normal(0, 0.5, ctx_len)).astype(np.float32)

    input_ts = torch.tensor(context).reshape(1, ctx_len)
    input_padding = torch.zeros(1, ctx_len)
    freq = torch.zeros(1, 1, dtype=torch.long)  # high frequency bucket

    with torch.no_grad():
        out, _aux = model(input_ts, input_padding, freq)  # [1, N, horizon, num_outputs]
    out = out[0]  # [N, horizon, num_outputs]
    last = out[-1].detach().numpy().astype(np.float32)  # [horizon, num_outputs]

    np.save(os.path.join(args.out_dir, "ref_output.npy"), last)
    np.save(os.path.join(args.out_dir, "ref_context.npy"), context)

    rms = float(np.sqrt(np.mean(last.astype(np.float64) ** 2)))
    meta = {
        "model": "Vincent05R/FinCast v1.pth",
        "note": "reference forward, stochastic MoE neutralized -> deterministic top-2",
        "ctx_len": ctx_len,
        "freq": 0,
        "horizon_len": 128,
        "num_outputs": 10,
        "output_shape": list(last.shape),
        "output_rms": rms,
        # full arrays committed (small) so the parity gate needs only the weights
        # (FINCAST_CKPT) to run brain — no reference re-run, no gitignored .npy.
        "context_full": [float(x) for x in context.reshape(-1)],
        "output_full": [float(x) for x in last.reshape(-1)],
    }
    with open(os.path.join(args.out_dir, "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("wrote golden; output rms =", rms, "shape", last.shape)
    print("first row (mean+q):", last[0].tolist())


if __name__ == "__main__":
    main()
