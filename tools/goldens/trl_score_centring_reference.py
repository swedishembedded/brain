#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a training engine computes what a reference trainer
# computes, number for number, for its clients. If your team needs expertise in porting reinforcement-learning
# objectives between frameworks without drift then you can procure our services by sending an email to
# info@swedishembedded.com.

"""Write `crates/rl/tests/fixtures/trl_score_centring.json`: what TRL's `AsyncGRPOTrainer` itself computes.

Unlike `pg_objective_reference.py`, which writes each loss from its definition, this runs TRL's own code from a
pinned source tree, so the Rust test (`crates/rl/tests/it/trl_goldens.rs`) holds brain to the reference trainer:

* group advantages: `AsyncRolloutLoop.compute_advantages` (z-score within the group, a NaN reward advantage 0 and
  left out of the mean and the std);
* one token of `AsyncGRPOTrainer.compute_loss` under `score_centering=True`: the importance ratio `coef_1`, the mask
  `is_masked`, the sampler and trainer probabilities `q` and `p`, the `head`, the `tail` flag, the `residual`, the
  `weight`, the `correction` and the `per_token_loss`, read from the method's own locals;
* support replay (`sampling_mask`): one token of the same `compute_loss` whose sampler cut its row, the stand-in model
  computing its log-probabilities with TRL's own fused head (`kernels.chunked_logprob.ChunkedLogProbFunction`, its
  torch path) from a logits row and the `support_ids` / `score_ids` compute_loss hands it, so TRL normalises over the
  support itself: the trainer's support-normalised log-probabilities, the per-token loss and its gradient in the
  logits (autograd through TRL's own backward).

The model is a stand-in that returns the given log-probabilities (compute_loss reads nothing else of it), the tree
attention unpacking is stubbed (it only feeds that model), and the method is stopped right after the per-token loss
by an accelerator whose `num_processes` raises: the locals are read from the frame of that exception. Nothing is
downloaded at run time and nothing here is on the build or test path; regenerate only when the pinned source changes:

    /tmp/marsch-venvs/trl-golden/bin/python -I -B tools/goldens/trl_score_centring_reference.py \\
        --trl ~/resources/reference/trl/src

Pinned: TRL source tree at commit 6267b72 (its `trainer/async_grpo_trainer.py` SHA-256 is recorded in the fixture and
checked here), torch 2.14.1+cpu, transformers 5.24.1, accelerate 1.16.0, datasets 5.2.0, numpy 2.5.5, Python 3.14,
in an isolated, node-local venv (`python3 -m venv /tmp/marsch-venvs/trl-golden`, never on a shared project disk; torch
from the PyTorch CPU index, the rest from PyPI).
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys
import types

TRL_COMMIT = "6267b72"
OUT = pathlib.Path(__file__).resolve().parents[2] / "crates/rl/tests/fixtures/trl_score_centring.json"

# Groups of rewards: a z-score group, one with a NaN reward, all-equal rewards, a pair, a lone scored reward.
GROUPS = [
    [1.0, 0.0, 0.0, 1.0],
    [0.5, float("nan"), 2.0, -1.0],
    [3.0, 3.0, 3.0],
    [0.25, 1.75],
    [float("nan"), 4.0, float("nan")],
]

# One sampled token per case: sampler and trainer log-probabilities of the token, its advantage, and the two
# log-probabilities at each of the sampler's K = 3 listed ids (-inf: a slot the sampler did not list), band [low, high].
LOW, HIGH = 0.5, 5.0
NEG_INF = float("-inf")
CASES = [
    # In the band, every listed id in the head, the unlisted mass in the band too (tail centred).
    {"name": "in_band_tail_centred", "old": math.log(0.4), "new": math.log(0.44), "adv": 1.5, "q": [0.4, 0.3, 0.1], "p": [0.44, 0.27, 0.12]},
    # The token's ratio below the band: masked; the listed ids still centre.
    {"name": "token_masked_low", "old": math.log(0.4), "new": math.log(0.1), "adv": -0.75, "q": [0.4, 0.3, 0.1], "p": [0.1, 0.5, 0.2]},
    # The token's ratio above the band, one id out of the head (p > high q), the tail out of the band.
    {"name": "token_masked_high_tail_out", "old": math.log(0.05), "new": math.log(0.3), "adv": 2.0, "q": [0.05, 0.6, 0.3], "p": [0.3, 0.6, 0.09]},
    # An unused slot, two ids just inside the head's edges. Exactly at an edge the decision is the rounding's: TRL
    # compares in float32, brain in float64 from the same float32 inputs, and the two can differ by one ulp there.
    {"name": "unused_slot_just_inside_head_edges", "old": math.log(0.2), "new": math.log(0.2), "adv": 1.0, "q": [0.2, 0.1, None], "p": [0.1001, 0.4999, 0.3]},
    # Two ids just outside the head's edges.
    {"name": "just_outside_head_edges", "old": math.log(0.2), "new": math.log(0.25), "adv": -0.5, "q": [0.2, 0.1, 0.3], "p": [0.0999, 0.5001, 0.3]},
    # The sampler lists its whole distribution: the tail is empty (0 / 0 guarded) and out of the band.
    {"name": "whole_distribution_listed", "old": math.log(0.5), "new": math.log(0.6), "adv": -1.0, "q": [0.5, 0.3, 0.2], "p": [0.6, 0.25, 0.15]},
]


# Support replay: one token whose sampler kept the support `support` (ids of an 8-token row) of the behaviour row
# `behaviour` and drew `label`; the trainer's row is `logits`. The sampler's top ids `top` lie in the support (None: an
# unused slot). Behaviour figures are given as a record carries them: whole-row log-probabilities (`behavior_logprobs`,
# `behavior_top_logprobs`) and the token's within its support (`behavior_support_logprobs`).
SUPPORT_CASES = [
    {"name": "support_tail_centred", "logits": [2.0, 1.5, 0.3, -0.5, 1.1, -1.0, 0.0, 0.7], "behaviour": [2.1, 1.3, 0.4, -0.2, 1.0, -0.8, 0.1, 0.5], "support": [0, 1, 4, 7], "top": [0, 1, None], "label": 1, "adv": 1.25},
    {"name": "support_listed_whole", "logits": [0.2, 1.9, -0.4, 1.2, 0.5, -1.5, 0.9, -0.1], "behaviour": [0.0, 2.0, -0.3, 1.0, 0.6, -1.2, 1.1, 0.0], "support": [1, 3, 6], "top": [1, 3, 6], "label": 6, "adv": -0.8},
    {"name": "support_token_masked", "logits": [3.0, -0.5, 0.1, 0.4, -2.0, 0.6, 0.0, 0.2], "behaviour": [0.5, 0.2, 0.1, 2.5, -1.0, 0.6, 0.3, 0.2], "support": [0, 3, 5], "top": [3, 0, None], "label": 3, "adv": 0.9},
]


def log_softmax(row, over=None):
    ids = range(len(row)) if over is None else over
    mx = max(row[i] for i in ids)
    lse = mx + math.log(sum(math.exp(row[i] - mx) for i in ids))
    return [x - lse for x in row]


class Stop(Exception):
    """Raised once compute_loss has its per-token loss."""


class Accelerator:
    @property
    def num_processes(self):
        raise Stop()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trl", required=True, help="the TRL source tree (its src directory)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    trl_src = pathlib.Path(a.trl).expanduser().resolve()
    trainer_file = trl_src / "trl/trainer/async_grpo_trainer.py"
    sys.path.insert(0, str(trl_src))

    import numpy as np
    import torch
    import trl.trainer.async_grpo_trainer as agt

    # Advantages: the method reads only the group's rewards.
    groups = []
    for rewards in GROUPS:
        adv = agt.AsyncRolloutLoop.compute_advantages(None, types.SimpleNamespace(rewards=np.array(rewards, dtype=np.float64)))
        groups.append({"rewards": [None if math.isnan(r) else r for r in rewards], "advantages": [float(x) for x in adv]})

    # Score centring: one call per case, its locals read at the stop.
    agt.unpad_tree = lambda inputs: {}
    k = 3
    cases = []
    for case in CASES:
        q_lp = [math.log(x) if x is not None else NEG_INF for x in case["q"]]
        p_lp = [math.log(x) for x in case["p"]]
        f32 = torch.float32
        inputs = {
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "token_index": torch.zeros(1, 1, dtype=torch.long),
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
            "shift_labels": torch.zeros(1, 1, dtype=torch.long),
            "old_log_probs": torch.tensor([[case["old"]]], dtype=f32),
            "advantages": torch.tensor([[case["adv"]]], dtype=f32),
            "segment_id": torch.zeros(1, 1, dtype=torch.long),
            "sampling_mask_counts": torch.zeros(1, 1, dtype=torch.long),
            "sampling_mask": torch.full((1, 1), -1, dtype=torch.long),
            "sampler_ids": torch.arange(k, dtype=torch.long).reshape(1, 1, k),
            "sampler_log_probs": torch.tensor([[q_lp]], dtype=f32),
            "global_n_tokens": torch.tensor([1.0]),
        }
        outputs = types.SimpleNamespace(log_probs=torch.tensor([[case["new"]]], dtype=f32), entropy=torch.zeros(1, 1), score_log_probs=torch.tensor([[p_lp]], dtype=f32))
        args = types.SimpleNamespace(score_centering=True, vllm_importance_sampling_clip_min=LOW, vllm_importance_sampling_clip_max=HIGH)
        fake = types.SimpleNamespace(args=args, aux_loss_enabled=False, accelerator=Accelerator(), epsilon_low=0.2, epsilon_high=0.2, current_gradient_accumulation_steps=1, _last_forward_time_s=0.0)
        try:
            agt.AsyncGRPOTrainer.compute_loss(fake, lambda **_: outputs, inputs)
            raise RuntimeError("compute_loss ran past the per-token loss")
        except Stop:
            tb = sys.exc_info()[2]
            while tb.tb_frame.f_code.co_name != "compute_loss":
                tb = tb.tb_next
            v = tb.tb_frame.f_locals
        row = lambda t: [float(x) for x in t.reshape(-1).tolist()]
        cases.append({
            "name": case["name"],
            "low": LOW,
            "high": HIGH,
            # The inputs as TRL read them (float32).
            "old_lp": row(inputs["old_log_probs"])[0],
            "new_lp": row(outputs.log_probs)[0],
            "advantage": row(inputs["advantages"])[0],
            "sampler_top_lp": [None if math.isinf(x) else x for x in row(inputs["sampler_log_probs"])],
            "trainer_top_lp": row(outputs.score_log_probs),
            # TRL's locals.
            "ratio": row(v["coef_1"])[0],
            "is_masked": bool(v["is_masked"].reshape(-1)[0]),
            "q": row(v["q"]),
            "p": row(v["p"]),
            "head": row(v["head"]),
            "tail": row(v["tail"])[0] > 0.5,
            "residual": row(v["residual"]),
            "weight": row(v["weight"])[0],
            "correction": row(v["correction"])[0],
            "per_token_loss": row(v["per_token_loss"])[0],
        })

    # Support replay: compute_loss with the support in `sampling_mask`, the model TRL's fused head over a logits row.
    from trl.kernels.chunked_logprob import ChunkedLogProbFunction as _ChunkedLogProbFunction

    support_cases = []
    for case in SUPPORT_CASES:
        v, support, label = len(case["logits"]), case["support"], case["label"]
        whole_mu = log_softmax(case["behaviour"])
        supp_mu = log_softmax(case["behaviour"], support)
        top = case["top"]
        k = len(top)
        f32 = torch.float32
        logits = torch.tensor([case["logits"]], dtype=f32, requires_grad=True)
        eye = torch.eye(v, dtype=f32)

        def model(support_ids=None, score_ids=None, **_):
            out = _ChunkedLogProbFunction.apply(logits, eye, None, torch.tensor([label]), 1.0, 32768, None, 1.0, ("log_probs", "entropy"), support_ids[0], score_ids[0])
            return types.SimpleNamespace(log_probs=out[0].reshape(1, 1), entropy=out[1].reshape(1, 1), score_log_probs=out[5].reshape(1, 1, k))

        inputs = {
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "token_index": torch.zeros(1, 1, dtype=torch.long),
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
            "shift_labels": torch.tensor([[label]], dtype=torch.long),
            "old_log_probs": torch.tensor([[supp_mu[label]]], dtype=f32),
            "advantages": torch.tensor([[case["adv"]]], dtype=f32),
            "segment_id": torch.zeros(1, 1, dtype=torch.long),
            "sampling_mask_counts": torch.tensor([[len(support)]], dtype=torch.long),
            "sampling_mask": torch.tensor([support], dtype=torch.long),
            "sampler_ids": torch.tensor([[[t if t is not None else -1 for t in top]]], dtype=torch.long),
            "sampler_log_probs": torch.tensor([[[supp_mu[t] if t is not None else NEG_INF for t in top]]], dtype=f32),
            "global_n_tokens": torch.tensor([1.0]),
        }
        args = types.SimpleNamespace(score_centering=True, vllm_importance_sampling_clip_min=LOW, vllm_importance_sampling_clip_max=HIGH)
        fake = types.SimpleNamespace(args=args, aux_loss_enabled=False, accelerator=Accelerator(), epsilon_low=0.2, epsilon_high=0.2, current_gradient_accumulation_steps=1, _last_forward_time_s=0.0)
        try:
            agt.AsyncGRPOTrainer.compute_loss(fake, model, inputs)
            raise RuntimeError("compute_loss ran past the per-token loss")
        except Stop:
            tb = sys.exc_info()[2]
            while tb.tb_frame.f_code.co_name != "compute_loss":
                tb = tb.tb_next
            loc = tb.tb_frame.f_locals
        per_token_loss = loc["per_token_loss"]
        per_token_loss.sum().backward()
        row = lambda t: [float(x) for x in t.detach().reshape(-1).tolist()]
        support_cases.append({
            "name": case["name"],
            "low": LOW,
            "high": HIGH,
            "logits": row(logits),
            "label": label,
            "advantage": case["adv"],
            "support": support,
            "top_ids": [t if t is not None else None for t in top],
            # The behaviour figures as a record carries them.
            "behavior_logprob": whole_mu[label],
            "behavior_support_logprob": supp_mu[label],
            "behavior_top_logprobs": [whole_mu[t] if t is not None else None for t in top],
            # TRL's support-normalised trainer figures, its locals and the gradient in the logits.
            "log_prob": row(loc["log_probs"])[0],
            "score_log_probs": row(loc["score_log_probs"]),
            "is_masked": bool(loc["is_masked"].reshape(-1)[0]),
            "tail": row(loc["tail"])[0] > 0.5,
            "per_token_loss": row(per_token_loss)[0],
            "grad_logits": row(logits.grad),
        })

    fixture = {
        "provenance": {
            "tool": "tools/goldens/trl_score_centring_reference.py",
            "trl_commit": TRL_COMMIT,
            "async_grpo_trainer_sha256": sha256(trainer_file),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "groups": groups,
        "score_centring": cases,
        "support_replay": support_cases,
    }
    pathlib.Path(a.out).write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {a.out}: {len(groups)} groups, {len(cases)} tokens, {len(support_cases)} support-replay tokens")


if __name__ == "__main__":
    main()
