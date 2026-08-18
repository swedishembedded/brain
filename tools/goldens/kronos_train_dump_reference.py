#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the reference Kronos decoder's TRAINING-mode forward, for brain's
trainer parity rung (TR).

The inference ladder (tools/goldens/kronos_dump_reference.py) cannot reach this:
the reference dependency layer attends with `is_causal=self.training`, so every
rung driven through `decode_s1`/`decode_s2` - all of them - exercises the
NON-causal branch, and with `q_len == 1` on top of that. Training is the only
regime where that flag is on and `q_len == T`, i.e. the only regime where the
mask exists at all. A trainer can therefore be wrong about it while every
inference rung and every self-consistency gradient check stays green.

So this dumper runs the reference exactly the way its own training loop does -
`model.train()`, teacher forcing, `q_len == T` - and records the two logit
fields plus the CE objective, on a small deterministic model whose weights are
dumped alongside so brain can load the identical parameters.

Determinism: every dropout probability is constructed as 0.0, so `model.train()`
changes nothing except the causal flag, and teacher forcing removes the
`multinomial` sibling sampling. Nothing here is stochastic.

Usage:
  python3 tools/goldens/kronos_train_dump_reference.py --repo <Kronos repo> \
      --out testdata/golden/kronos
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# A small decoder: this rung is about the training graph's SHAPE (the causal
# dependency mask, the teacher-forced sibling, the objective), not about a
# particular checkpoint's numbers, and small weights are committable.
DIMS = dict(d_model=32, n_layers=2, n_heads=4, ff_dim=64, s1_bits=5, s2_bits=5, learn_te=True)
DEP_HEADS = 4  # the reference DependencyAwareLayer's fixed head count
SEQ_LEN = 24
SEED = 20260816
# How wide the random weights are drawn. Deliberately not tiny: with near-zero
# projections the dependency layer's attention is nearly uniform and dropping its
# causal mask barely moves the logits (rel_l2 ~1e-2), which would make this rung
# pass or fail on a hair. At this scale the mask is worth ~3e-1 relative and
# several argmax flips - the same order as the two defects the inference ladder
# was written for - so the rung has real discriminating power. RMSNorm keeps the
# stack numerically calm either way.
WEIGHT_SCALE = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch

    from model.kronos import Kronos

    torch.set_grad_enabled(False)
    model = Kronos(
        s1_bits=DIMS["s1_bits"], s2_bits=DIMS["s2_bits"], n_layers=DIMS["n_layers"],
        d_model=DIMS["d_model"], n_heads=DIMS["n_heads"], ff_dim=DIMS["ff_dim"],
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=DIMS["learn_te"],
    )

    # Deterministic weights, drawn here (not by torch's init) so the dump is
    # reproducible across torch versions. RMSNorm gains sit near 1.
    rng = np.random.default_rng(SEED)
    sd = model.state_dict()
    params = []
    for name in sd:
        shape = tuple(sd[name].shape)
        n = int(np.prod(shape)) if shape else 1
        if "rotary" in name or "inv_freq" in name:
            # NON-PERSISTENT: the RoPE frequency table is a function of head_dim,
            # recomputed by every implementation rather than loaded (brain's
            # importer drops it by name). Randomizing it here would hand the
            # reference frequencies nobody else can reproduce, and this rung would
            # then fail for a reason that has nothing to do with what it gates.
            params.append([name, n])
            continue
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            v = 1.0 + rng.uniform(-0.02, 0.02, n)
        else:
            v = rng.uniform(-WEIGHT_SCALE, WEIGHT_SCALE, n)
        sd[name] = torch.tensor(v.astype(np.float32)).reshape(shape)
        params.append([name, n])
    model.load_state_dict(sd)

    # TRAINING mode - the whole point. With every dropout p == 0 this changes
    # exactly one thing: the dependency layer's `is_causal` flag.
    model.train()
    assert model.training, "the reference must be in training mode for this rung"

    t = SEQ_LEN
    s1v, s2v = 2 ** DIMS["s1_bits"], 2 ** DIMS["s2_bits"]
    s1 = rng.integers(0, s1v, t).astype(np.int64)
    s2 = rng.integers(0, s2v, t).astype(np.int64)
    # Calendar columns in the reference's order (minute, hour, weekday, day, month).
    stamp = np.stack([
        rng.integers(0, 60, t), rng.integers(0, 24, t), rng.integers(0, 7, t),
        rng.integers(0, 32, t), rng.integers(0, 13, t),
    ], axis=1).astype(np.int64)
    # Teacher forcing: the sibling s1 the dependency layer conditions on is an
    # input, not a sample, so the forward is deterministic.
    samp = rng.integers(0, s1v, t).astype(np.int64)
    s1_tgt = rng.integers(0, s1v, t).astype(np.int64)
    s2_tgt = rng.integers(0, s2v, t).astype(np.int64)

    tt = lambda a: torch.tensor(a).unsqueeze(0)  # noqa: E731
    s1_logits, s2_logits = model(
        tt(s1), tt(s2), stamp=tt(stamp.astype(np.float32)),
        use_teacher_forcing=True, s1_targets=tt(samp),
    )
    ce, ce1, ce2 = model.head.compute_loss(s1_logits, s2_logits, tt(s1_tgt), tt(s2_tgt))

    os.makedirs(args.out, exist_ok=True)
    w = np.concatenate([sd[name].cpu().numpy().reshape(-1) for name, _ in params]).astype(np.float32)
    w.tofile(os.path.join(args.out, "tr_weights.f32"))
    for fname, arr in [
        ("tr_s1_ids.u32", s1), ("tr_s2_ids.u32", s2), ("tr_stamp.u32", stamp),
        ("tr_samp_s1.u32", samp), ("tr_s1_targets.u32", s1_tgt), ("tr_s2_targets.u32", s2_tgt),
    ]:
        np.asarray(arr, dtype=np.uint32).reshape(-1).tofile(os.path.join(args.out, fname))
    np.asarray(s1_logits[0].cpu().numpy(), dtype=np.float32).tofile(os.path.join(args.out, "tr_s1_logits.f32"))
    np.asarray(s2_logits[0].cpu().numpy(), dtype=np.float32).tofile(os.path.join(args.out, "tr_s2_logits.f32"))

    meta = dict(DIMS)
    meta.update({
        "seq_len": t, "dep_n_heads": DEP_HEADS, "seed": SEED,
        "params": params, "ce_loss": float(ce), "ce_s1": float(ce1), "ce_s2": float(ce2),
        "source": source_block(
            checkpoint=args.repo,
            files=[],
            identity={
                "d_model": DIMS["d_model"], "n_layers": DIMS["n_layers"],
                "seq_len": t, "seed": SEED,
            },
        ),
    })
    with open(os.path.join(args.out, "tr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"TR: {len(params)} tensors, {w.size} weights, t={t}")
    print("TR: ce_loss=%.6f (s1 %.6f, s2 %.6f)" % (ce, ce1, ce2))
    print("TR: s2_logits[0,:4] =", np.round(s2_logits[0, 0, :4].cpu().numpy(), 5).tolist())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
