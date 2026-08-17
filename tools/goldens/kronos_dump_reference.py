#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Kronos reference tensors for brain's parity ladder.

The stochastic AR rollout is made deterministic by forcing argmax sampling, so
every rung below is reproducible and comparable rung by rung:

  T1: tokenizer.encode(normalized bars)     -> (s1, s2) token streams [integer-exact]
  T2: tokenizer.decode(s1, s2)              -> reconstructed bars      [cosine]
  T4: model.decode_s1(s1, s2, stamp=None)   -> s1 logits               [cosine]
  T4b: model.decode_s1(s1, s2, stamp)       -> s1 logits + argmax ids  [ids exact]
  T5: model.decode_s2(ctx, argmax s1)       -> s2 logits               [cosine]
  T6: the whole argmax rollout              -> generated (s1, s2)      [integer-exact]
      + the denormalized predicted bars                                [cosine]
  T7: the same rollout on a context short enough that the attention window never
      slides (`context + pred_len <= max_context`) -> denormalized bars

T7 exists because the reference recomputes its whole 512-bar window, from a
window origin one bar later, at EVERY rollout step - a shape no K/V cache can
reproduce, and one this checkpoint is violently sensitive to (two correct runs
whose window origin differs by a single bar disagree by ~1e-1 relative in the
final logits). Inside T7's regime the window never moves, so brain's cached
rollout has a right answer too and is held to it.

T5 and T6 are the rungs that a brain-vs-brain check cannot reach: the dependency
layer and the composed loop (rollout window, detokenization window, and the
normalize/denormalize round trip) only have a right answer relative to the
reference.

The context is deliberately LONGER than the model's 512-bar attention window, so
the rollout slides that window from its first step - the regime the user-facing
CSV path actually runs in.

Usage:
  python3 tools/goldens/kronos_dump_reference.py --repo <Kronos repo> \
      --tokenizer <tokenizer dir> --decoder <decoder dir> --out crates/kronos/tests/golden
"""
import argparse
import json
import os
import sys

import numpy as np

CLIP = 5.0
# How many trailing positions of the s1-logit field are stored as floats. The
# argmax id is stored for EVERY position (cheap, and a wrong causal mask or a
# wrong calendar embedding shows up there); the float rows pin the last few
# exactly without committing a multi-megabyte tensor to the repo.
LOGIT_TAIL = 16


def synth_ohlcv(t, feat=6):
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


def synth_stamps(n, start="2026-01-05 00:00:00"):
    """Hourly calendar stamps in the reference's own order/semantics
    (minute, hour, weekday Monday=0, day, month)."""
    import pandas as pd

    ts = pd.date_range(start=start, periods=n, freq="h")
    df = pd.DataFrame()
    df["minute"] = ts.minute
    df["hour"] = ts.hour
    df["weekday"] = ts.weekday
    df["day"] = ts.day
    df["month"] = ts.month
    return df.values.astype(np.float32)


def normalize(bars, clip=CLIP):
    mean = bars.mean(axis=0)
    std = bars.std(axis=0)
    z = (bars - mean) / (std + 1e-5)
    return np.clip(z, -clip, clip).astype(np.float32), mean, std


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def write_u32(path, arr):
    np.asarray(arr, dtype=np.uint32).reshape(-1).tofile(path)


def _cfg(model, key):
    """Read a decoder dim off the reference model. Upstream `Kronos` stores its
    dims as plain attributes; the HF mixin may also expose a `config` object.
    Try both rather than pinning this dumper to one upstream revision."""
    if hasattr(model, key):
        return getattr(model, key)
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, key):
        return getattr(cfg, key)
    raise SystemExit(f"reference model exposes no {key}; cannot record the decoder tier in t_meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=520)
    ap.add_argument("--pred-len", type=int, default=12)
    args = ap.parse_args()
    if args.context <= 512:
        raise SystemExit("--context must exceed the 512-bar window so the rollout slides it")

    sys.path.insert(0, args.repo)
    import torch

    from model.kronos import Kronos, KronosTokenizer

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    tok = KronosTokenizer.from_pretrained(args.tokenizer).eval()
    model = Kronos.from_pretrained(args.decoder).eval()
    max_context = 512  # KronosPredictor's default, and the checkpoint's own window

    raw = synth_ohlcv(args.context)
    z, mean, std = normalize(raw)
    stamps = synth_stamps(args.context + args.pred_len)
    x = torch.tensor(z).unsqueeze(0)  # [1, T, 6]
    stamp_t = torch.tensor(stamps).unsqueeze(0)
    ctx_stamp, fut_stamp = stamps[: args.context], stamps[args.context:]

    # T1: encode -> (pre, post) tokens, each [1, T]
    s1t, s2t = tok.encode(x, half=True)
    s1 = s1t.squeeze(0).cpu().numpy().astype(np.uint32)
    s2 = s2t.squeeze(0).cpu().numpy().astype(np.uint32)
    # T2: decode -> reconstruction [1, T, 6]
    recon = tok.decode([s1t, s2t], half=True).squeeze(0).cpu().numpy().astype(np.float32)
    # T4: decode_s1 with NO stamp (the temporal embedding switched off)
    s1_logits_ns, _ = model.decode_s1(s1t, s2t, stamp=None)
    # T4b: decode_s1 with the real calendar, over the last `max_context` bars -
    # the window the rollout's first step sees.
    w0 = max(0, args.context - max_context)
    s1_logits, ctx = model.decode_s1(s1t[:, w0:], s2t[:, w0:], stamp_t[:, w0:args.context])
    argmax_ids = torch.argmax(s1_logits, dim=-1).squeeze(0).cpu().numpy().astype(np.uint32)
    # T5: decode_s2 for the last position, conditioned on the argmax s1 - the
    # dependency layer, in the exact shape the reference calls it with (a single
    # sampled token as the query).
    samp = torch.argmax(s1_logits[:, -1, :], dim=-1, keepdim=True)
    s2_logits = model.decode_s2(ctx, samp)

    # T6: the composed argmax rollout, exactly as `auto_regressive_inference`
    # drives it (sliding window, per-window calendar), then detokenized over the
    # last `max_context` tokens and denormalized.
    p1, p2 = s1t.clone(), s2t.clone()
    for _ in range(args.pred_len):
        n = p1.size(1)
        lo = max(0, n - max_context)
        lg, cx = model.decode_s1(p1[:, lo:], p2[:, lo:], stamp_t[:, lo:n])
        a = torch.argmax(lg[:, -1, :], dim=-1, keepdim=True)
        s2lg = model.decode_s2(cx, a)
        b = torch.argmax(s2lg[:, -1, :], dim=-1, keepdim=True)
        p1 = torch.cat([p1, a], dim=1)
        p2 = torch.cat([p2, b], dim=1)
    total = p1.size(1)
    lo = max(0, total - max_context)
    pred_norm = tok.decode([p1[:, lo:].contiguous(), p2[:, lo:].contiguous()], half=True)
    pred_norm = pred_norm.squeeze(0).cpu().numpy()[-args.pred_len:].astype(np.float32)
    pred = (pred_norm * (std + 1e-5) + mean).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    write_f32(os.path.join(args.out, "t_raw.f32"), raw)
    write_f32(os.path.join(args.out, "t_context.f32"), z)  # normalized context
    write_u32(os.path.join(args.out, "t_stamp.u32"), stamps.astype(np.uint32))
    write_u32(os.path.join(args.out, "t1_s1.u32"), s1)
    write_u32(os.path.join(args.out, "t1_s2.u32"), s2)
    write_f32(os.path.join(args.out, "t2_recon.f32"), recon)
    write_u32(os.path.join(args.out, "t4_argmax.u32"), torch.argmax(s1_logits_ns, dim=-1).squeeze(0).cpu().numpy().astype(np.uint32))
    write_f32(os.path.join(args.out, "t4_logits_tail.f32"), s1_logits_ns[0, -LOGIT_TAIL:].cpu().numpy())
    write_u32(os.path.join(args.out, "t4b_argmax.u32"), argmax_ids)
    write_f32(os.path.join(args.out, "t4b_logits_tail.f32"), s1_logits[0, -LOGIT_TAIL:].cpu().numpy())
    write_f32(os.path.join(args.out, "t5_s2_logits_last.f32"), s2_logits[0, -1].cpu().numpy())
    write_u32(os.path.join(args.out, "t5_samp_s1.u32"), samp[0].cpu().numpy().astype(np.uint32))
    write_u32(os.path.join(args.out, "t6_gen_s1.u32"), p1[0, args.context:].cpu().numpy().astype(np.uint32))
    write_u32(os.path.join(args.out, "t6_gen_s2.u32"), p2[0, args.context:].cpu().numpy().astype(np.uint32))
    write_f32(os.path.join(args.out, "t6_pred.f32"), pred)
    # `d_model`/`n_layers` identify the DECODER TIER these tensors came from.
    # Without them a Kronos-small dump is indistinguishable from a Kronos-base
    # one, and crates/kronos/tests/parity.rs can only find out by failing deep
    # in the importer with a tensor-shape error.
    meta = {
        "context_len": int(args.context),
        "pred_len": int(args.pred_len),
        "max_context": int(max_context),
        "feat": int(raw.shape[1]),
        "clip": CLIP,
        "logit_tail": LOGIT_TAIL,
        "s1_vocab": int(s1_logits.shape[-1]),
        "s2_vocab": int(s2_logits.shape[-1]),
        "d_model": int(_cfg(model, "d_model")),
        "n_layers": int(_cfg(model, "n_layers")),
    }
    with open(os.path.join(args.out, "t_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("reference: s1[:5]=", s1[:5].tolist(), "s2[:5]=", s2[:5].tolist())
    print("T4b argmax[-5:]:", argmax_ids[-5:].tolist())
    print("T5 argmax s1 (last):", int(samp[0, 0]))
    print("T6 gen s1:", p1[0, args.context:].tolist())
    print("T6 gen s2:", p2[0, args.context:].tolist())
    print("T6 pred close:", np.round(pred[:, 3], 3).tolist())

    # T7: the non-sliding regime. A context short enough that context+pred_len
    # fits the window, so every rollout step re-runs a window with the SAME
    # origin and a K/V cache is exact rather than approximate.
    t7 = max_context - args.pred_len
    raw7 = synth_ohlcv(t7)
    z7, mean7, std7 = normalize(raw7)
    stamps7 = synth_stamps(t7 + args.pred_len)
    x7 = torch.tensor(z7).unsqueeze(0)
    st7 = torch.tensor(stamps7).unsqueeze(0)
    q1, q2 = tok.encode(x7, half=True)
    for _ in range(args.pred_len):
        n = q1.size(1)
        lg, cx = model.decode_s1(q1, q2, st7[:, :n])
        a1 = torch.argmax(lg[:, -1, :], dim=-1, keepdim=True)
        b1 = torch.argmax(model.decode_s2(cx, a1)[:, -1, :], dim=-1, keepdim=True)
        q1 = torch.cat([q1, a1], dim=1)
        q2 = torch.cat([q2, b1], dim=1)
    rec7 = tok.decode([q1, q2], half=True).squeeze(0).cpu().numpy()[-args.pred_len:].astype(np.float32)
    pred7 = (rec7 * (std7 + 1e-5) + mean7).astype(np.float32)
    write_f32(os.path.join(args.out, "t7_raw.f32"), raw7)
    write_u32(os.path.join(args.out, "t7_stamp.u32"), stamps7.astype(np.uint32))
    write_u32(os.path.join(args.out, "t7_gen_s1.u32"), q1[0, t7:].cpu().numpy().astype(np.uint32))
    write_u32(os.path.join(args.out, "t7_gen_s2.u32"), q2[0, t7:].cpu().numpy().astype(np.uint32))
    write_f32(os.path.join(args.out, "t7_pred.f32"), pred7)
    meta["t7_context_len"] = int(t7)
    with open(os.path.join(args.out, "t_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("T7 gen s1:", q1[0, t7:].tolist())
    print("T7 pred close:", np.round(pred7[:, 3], 3).tolist())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
