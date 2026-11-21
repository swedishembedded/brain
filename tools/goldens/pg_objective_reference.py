#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a reinforcement-learning objective computes what its
# paper says, for its clients. If your team needs expertise in verified policy-gradient training then you can procure
# our services by sending an email to info@swedishembedded.com.

"""Write the policy-gradient conformance fixture `crates/rl/tests/fixtures/pg_objective.json`.

The expected values are independent of brain: every loss is written here from its definition and every gradient is
PyTorch autograd of that loss, in float64, from inputs first rounded to float32 (what the Rust side reads). The test
(`crates/rl/tests/it/pg_conformance.rs`) never runs Python; regenerate only when the specification changes:

    ~/project/shared/venvs/torch-venv/bin/python -I tools/goldens/pg_objective_reference.py

Specification (one token, behaviour log-probability `old`, current `new`, advantage `A`, clip band
`[1 - low, 1 + high]`, optional dual clip `c`, reference `ref`, KL weight `beta`, policy weight `w`):

    r         = exp(new - old)          (continued along its tangent past a log-ratio of 20)
    surrogate = min(r A, clip(r, 1 - low, 1 + high) A), and for A < 0 with a dual clip max(that, c A)
    k3        = e^u - u - 1, u = ref - new
    loss      = -w surrogate [A != 0] + beta r k3
    weight    = -d loss / d new         (the cross-entropy-gradient scale a trainer installs)

Group advantages (rewards `R` of one group of `G`): mean-centred `R - mean`, z-score `(R - mean) / (std + 1e-4)` with
the population std, leave-one-out `R_i - mean_{j != i} R_j`; a group whose population std is below 1e-6 is all zero,
a group of one is `1.0`.

Preferences: DPO `L = -log sigmoid(beta ((c - c_ref) - (r - r_ref)))` over the summed log-probabilities of the
chosen and the rejected completion; KTO (lambda_D, lambda_U) with log-ratio `x = log pi - log ref` and reference point
`z` held fixed: `lambda_D (1 - sigmoid(beta (x - z)))` for a desirable example, `lambda_U (1 - sigmoid(beta (z - x)))`
for an undesirable one. Gradients are in the summed log-probabilities (DPO) and in `x` (KTO).

A step of `k` one-row micro-batches: token-sum divides every supervised token by the step's supervised count;
sequence-mean divides each row by its own count and the step by `k`; constant-length:n divides each row by `n` and the
step by `k`. Unsupervised (prompt, padding) positions carry no term and no gradient.
"""

import json
import math
import pathlib

import numpy as np
import torch

CAP = 20.0
OUT = pathlib.Path(__file__).resolve().parents[2] / "crates/rl/tests/fixtures/pg_objective.json"


def f32(x):
    return float(np.float32(x))


def bounded_exp(x):
    cap = torch.tensor(CAP, dtype=torch.float64)
    return torch.where(x <= cap, torch.exp(torch.minimum(x, cap)), math.exp(CAP) * (1.0 + x - cap))


def token(old, new, adv, low, high, dual, ref, beta, w):
    """Loss, weight and flags of one token, by the definition above."""
    new_t = torch.tensor(new, dtype=torch.float64, requires_grad=True)
    r = bounded_exp(new_t - old)
    loss = torch.zeros((), dtype=torch.float64)
    clipped = dual_clipped = False
    if adv != 0.0:
        unclipped = r * adv
        clipped_term = torch.clamp(r, 1.0 - low, 1.0 + high) * adv
        surrogate = torch.minimum(unclipped, clipped_term)
        if dual is not None and adv < 0.0 and surrogate.item() < dual * adv:
            surrogate = torch.maximum(surrogate, torch.tensor(dual * adv, dtype=torch.float64))
            dual_clipped = True
        else:
            clipped = unclipped.item() > clipped_term.item()
        loss = loss - w * surrogate
    kl = None
    if beta > 0.0:
        u = ref - new_t
        k3 = bounded_exp(u) - u - 1.0
        loss = loss + beta * r * k3
        kl = (r * k3).item()
    loss.backward()
    log_ratio = new - old
    return {
        # The ratio and the ratio-weighted k3 are reported intermediates; past the cap a trainer's statistics read the
        # plain exponential, so they are left unchecked there.
        "ratio": math.exp(log_ratio) if log_ratio <= CAP else None,
        "kl": kl if log_ratio <= CAP and (ref is None or ref - new <= CAP) else None,
        "loss": loss.item(),
        "weight": -new_t.grad.item(),
        "clipped": clipped,
        "dual_clipped": dual_clipped,
    }


def token_cases():
    cases = []
    old = f32(-1.5)

    def add(name, ratio=None, new=None, adv=0.0, low=0.2, high=0.28, dual=None, ref=None, beta=0.0, w=1.0, o=old):
        new = f32(o + math.log(ratio)) if new is None else f32(new)
        ref = None if ref is None else f32(ref)
        inputs = {"old": o, "new": new, "adv": f32(adv), "low": f32(low), "high": f32(high), "dual": None if dual is None else f32(dual), "ref": ref, "beta": f32(beta), "w": f32(w)}
        cases.append({"name": name, **inputs, "expected": token(o, new, inputs["adv"], inputs["low"], inputs["high"], inputs["dual"], ref, inputs["beta"], inputs["w"])})

    # The clip band, both advantage signs, either side of each edge (band [0.8, 1.28]).
    for adv in (1.3, -0.7):
        sign = "pos" if adv > 0 else "neg"
        for label, ratio in (("far_below_low", 0.5), ("just_below_low", 0.79), ("just_above_low", 0.81), ("on_policy", 1.0), ("just_below_high", 1.27), ("just_above_high", 1.29), ("far_above_high", 2.0)):
            add(f"clip_{sign}_{label}", ratio=ratio, adv=adv)
    # The symmetric band clips what the wider upper edge keeps.
    add("clip_symmetric_pos_1_25", ratio=1.25, adv=1.0, high=0.2)
    add("clip_wide_pos_1_25", ratio=1.25, adv=1.0, high=0.28)
    # Dual clip c = 3: either side of c for a negative advantage, and a positive advantage it leaves alone.
    add("dual_neg_below_c", ratio=2.9, adv=-0.7, dual=3.0)
    add("dual_neg_above_c", ratio=3.1, adv=-0.7, dual=3.0)
    add("dual_neg_far_above_c", ratio=1.0e4, adv=-0.7, dual=3.0)
    add("dual_pos_above_c", ratio=3.1, adv=1.3, dual=3.0)
    add("no_dual_neg_above_c", ratio=3.1, adv=-0.7)
    # The k3 KL, weighted by the ratio: on-policy, off-policy both ways, a reference above and below, KL alone.
    add("kl_on_policy_ref_below", ratio=1.0, adv=0.0, ref=-2.3, beta=0.5)
    add("kl_on_policy_ref_above", ratio=1.0, adv=0.0, ref=-0.4, beta=0.5)
    add("kl_off_policy_up", ratio=1.6, adv=0.0, ref=-2.0, beta=0.04)
    add("kl_off_policy_down", ratio=0.3, adv=0.0, ref=-1.0, beta=0.04)
    add("kl_with_policy_term", ratio=1.1, adv=1.3, ref=-1.9, beta=0.2, dual=3.0)
    add("kl_with_clipped_policy_term", ratio=0.5, adv=-0.7, ref=-1.2, beta=0.2, dual=3.0)
    # A token drawn at 0.1 that the policy pushed 17.5 nats below the reference: little cost, almost no pull (537).
    add("kl_token_the_policy_left", new=math.log(0.1) - 17.5, o=f32(math.log(0.1)), ref=math.log(0.1), adv=0.0, beta=0.04)
    # The policy weight scales the policy term only.
    add("policy_weight_half", ratio=1.1, adv=1.3, ref=-1.9, beta=0.2, w=0.5)
    add("policy_weight_zero", ratio=1.1, adv=1.3, ref=-1.9, beta=0.2, w=0.0)
    # Past the exponent cap the ratio continues along its tangent.
    add("past_cap_neg", new=old + 25.0, adv=-1.0)
    return cases


def advantages(norm, rewards):
    g = len(rewards)
    if g <= 1:
        return [1.0] * g
    mean = sum(rewards) / g
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / g)
    if std < 1e-6:
        return [0.0] * g
    if norm == "mean-centred":
        return [r - mean for r in rewards]
    if norm == "z-score":
        return [(r - mean) / (std + 1e-4) for r in rewards]
    return [rewards[i] - sum(rewards[j] for j in range(g) if j != i) / (g - 1) for i in range(g)]


def advantage_cases():
    groups = {
        "half_right": [1.0, 1.0, 0.0, 0.0],
        "one_right_of_four": [1.0, 0.0, 0.0, 0.0],
        "pair_small_gap": [0.6, 0.5],
        "reversed_pair": [0.0, 1.0],
        "graded": [0.2, 0.9, 0.4, 0.7, 0.1],
        "equal": [0.3, 0.3, 0.3],
        "within_no_variance": [0.5, 0.5 + 1e-7, 0.5],
        "group_of_one": [0.8],
    }
    return [{"name": f"{name}_{norm}", "norm": norm, "rewards": rewards, "expected": advantages(norm, rewards)} for name, rewards in groups.items() for norm in ("mean-centred", "z-score", "leave-one-out")]


def aggregation_cases():
    """Two one-row micro-batches of unequal length with prompt and padding positions, both advantage signs."""
    t = 8
    rows = [
        # prompt of 3, completion of 2, padding of 3
        {"supervised": [False, False, True, True, False, False, False, False], "adv": 1.3},
        # prompt of 2, completion of 5, padding of 1
        {"supervised": [False, True, True, True, True, True, False, False], "adv": -0.7},
    ]
    rng = np.random.default_rng(5)
    for row in rows:
        row["old"] = [f32(x) for x in -rng.uniform(0.5, 3.0, t)]
        row["new"] = [f32(o + d) for o, d in zip(row["old"], rng.uniform(-0.15, 0.15, t))]
        row["ref"] = [f32(x) for x in -rng.uniform(0.5, 3.0, t)]
    low, high, beta = 0.2, 0.28, 0.04
    cases = []
    for agg in ("token-sum", "seq-mean", "constant-length:16"):
        new = [torch.tensor(r["new"], dtype=torch.float64, requires_grad=True) for r in rows]
        total_supervised = sum(sum(r["supervised"]) for r in rows)
        k = len(rows)
        step = torch.zeros((), dtype=torch.float64)
        for row, nt in zip(rows, new):
            row_loss = torch.zeros((), dtype=torch.float64)
            for i in range(t):
                if not row["supervised"][i]:
                    continue
                r = torch.exp(nt[i] - row["old"][i])
                a = row["adv"]
                surrogate = torch.minimum(r * a, torch.clamp(r, 1 - low, 1 + high) * a)
                u = row["ref"][i] - nt[i]
                row_loss = row_loss - surrogate + beta * r * (torch.exp(u) - u - 1)
            if agg == "token-sum":
                step = step + row_loss / total_supervised
            elif agg == "seq-mean":
                step = step + row_loss / sum(row["supervised"]) / k
            else:
                step = step + row_loss / 16.0 / k
        step.backward()
        cases.append({"name": f"two_rows_{agg}", "agg": agg, "low": low, "high": high, "beta": beta, "rows": rows, "expected": {"loss": step.item(), "grad": [nt.grad.tolist() for nt in new]}})
    return cases


def preference_cases():
    dpo = []
    for name, beta, c, c_ref, r, r_ref in (
        ("chosen_ahead", 0.5, -2.0, -3.0, -4.0, -3.5),
        ("at_reference", 0.1, -7.0, -7.0, -9.0, -9.0),
        ("rejected_ahead", 0.5, -6.0, -5.0, -2.0, -3.0),
        ("confident_margin", 0.2, -1.0, -40.0, -60.0, -2.0),
    ):
        beta, c, c_ref, r, r_ref = (f32(x) for x in (beta, c, c_ref, r, r_ref))
        ct = torch.tensor(c, dtype=torch.float64, requires_grad=True)
        rt = torch.tensor(r, dtype=torch.float64, requires_grad=True)
        loss = -torch.nn.functional.logsigmoid(beta * ((ct - c_ref) - (rt - r_ref)))
        loss.backward()
        dpo.append({"name": name, "beta": beta, "chosen": c, "chosen_ref": c_ref, "rejected": r, "rejected_ref": r_ref, "expected": {"loss": loss.item(), "d_chosen": ct.grad.item(), "d_rejected": rt.grad.item()}})
    kto = []
    for name, beta, lam_d, lam_u, desirable, x, z in (
        ("desirable_above_z", 0.5, 1.0, 1.5, True, 0.9, 0.2),
        ("desirable_below_z", 0.5, 1.0, 1.5, True, -1.2, 0.0),
        ("undesirable_above_z", 0.5, 1.0, 1.5, False, 0.9, 0.2),
        ("undesirable_at_z", 0.5, 1.0, 1.5, False, 0.4, 0.4),
        ("undesirable_far_below", 0.1, 1.0, 1.0, False, -30.0, 0.3),
    ):
        beta, lam_d, lam_u, x, z = (f32(v) for v in (beta, lam_d, lam_u, x, z))
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        if desirable:
            loss = lam_d * (1.0 - torch.sigmoid(beta * (xt - z)))
        else:
            loss = lam_u * (1.0 - torch.sigmoid(beta * (z - xt)))
        loss.backward()
        kto.append({"name": name, "beta": beta, "desirable_weight": lam_d, "undesirable_weight": lam_u, "desirable": desirable, "logratio": x, "z": z, "expected": {"loss": loss.item(), "d_logratio": xt.grad.item()}})
    return dpo, kto


def main():
    dpo, kto = preference_cases()
    fixture = {
        "provenance": f"tools/goldens/pg_objective_reference.py, torch {torch.__version__} float64 autograd of the definitions in its docstring",
        "token_terms": token_cases(),
        "advantages": advantage_cases(),
        "aggregation": aggregation_cases(),
        "dpo": dpo,
        "kto": kto,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
