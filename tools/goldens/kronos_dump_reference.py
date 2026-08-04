#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Kronos reference tensors for brain's parity ladder (deterministic rungs).

Avoids the stochastic AR rollout by checking the deterministic pieces:
  T1: tokenizer.encode(normalized bars) -> (s1, s2) token streams  [integer-exact]
  T2: tokenizer.decode(s1, s2)          -> reconstructed bars       [cosine]
  T4: model.decode_s1(s1, s2, stamp)    -> s1_logits                [cosine]

Both sides normalize the SAME raw context the SAME way (per-feature z-score, clip
±5) so the tokenizer sees identical input.

Usage:
  python3 tools/goldens/kronos_dump_reference.py --repo <Kronos repo> \
      --tokenizer <tokenizer dir> --decoder <decoder dir> --out crates/kronos/tests/golden
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def synth_ohlcv(t=120, feat=6):
    """Deterministic, coherent-ish OHLCV(+amount): a trending noisy close with
    O/H/L bracketing it and positive volume/amount. No RNG."""
    x = np.zeros((t, feat), dtype=np.float32)
    tt = np.arange(t)
    close = 100.0 + 0.1 * tt + 2.0 * np.sin(2 * np.pi * tt / 24.0)
    x[:, 3] = close
    x[:, 0] = close - 0.5  # open
    x[:, 1] = close + 1.0  # high
    x[:, 2] = close - 1.0  # low
    x[:, 4] = 1000.0 + 50.0 * np.sin(tt * 0.7)  # volume
    x[:, 5] = x[:, 4] * close  # amount
    return x


def normalize(bars, clip=5.0):
    mean = bars.mean(axis=0)
    std = bars.std(axis=0)
    z = (bars - mean) / (std + 1e-5)
    return np.clip(z, -clip, clip).astype(np.float32), mean, std


def write_f32(path, arr):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def write_u32(path, arr):
    a = np.asarray(arr, dtype=np.uint32).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<%dI" % a.size, *a.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=120)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    from model.kronos import Kronos, KronosTokenizer

    torch.manual_seed(0)
    tok = KronosTokenizer.from_pretrained(args.tokenizer).eval()
    model = Kronos.from_pretrained(args.decoder).eval()

    raw = synth_ohlcv(args.context)
    z, _, _ = normalize(raw)
    x = torch.tensor(z).unsqueeze(0)  # [1, T, 6]

    with torch.no_grad():
        # T1: encode -> (pre, post) tokens, each [1, T]
        s1t, s2t = tok.encode(x, half=True)
        s1 = s1t.squeeze(0).cpu().numpy().astype(np.uint32)
        s2 = s2t.squeeze(0).cpu().numpy().astype(np.uint32)
        # T2: decode -> reconstruction [1, T, 6]
        recon = tok.decode([s1t, s2t], half=True).squeeze(0).cpu().numpy().astype(np.float32)
        # T4: decode_s1 -> s1_logits [1, T, vocab]
        s1_logits, _ctx = model.decode_s1(s1t, s2t, stamp=None)
        s1_logits = s1_logits.squeeze(0).cpu().numpy().astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    write_f32(os.path.join(args.out, "t_context.f32"), z)        # normalized context
    write_u32(os.path.join(args.out, "t1_s1.u32"), s1)
    write_u32(os.path.join(args.out, "t1_s2.u32"), s2)
    write_f32(os.path.join(args.out, "t2_recon.f32"), recon)
    write_f32(os.path.join(args.out, "t4_s1_logits.f32"), s1_logits)
    meta = {
        "context_len": int(args.context),
        "feat": int(raw.shape[1]),
        "s1_vocab": int(s1_logits.shape[-1]),
    }
    with open(os.path.join(args.out, "t_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("reference: s1[:5]=", s1[:5].tolist(), "s2[:5]=", s2[:5].tolist())
    print("s1_logits shape:", s1_logits.shape, "recon shape:", recon.shape)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
