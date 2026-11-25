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
  `weight`, the `correction` and the `per_token_loss`, read from the method's own locals.

The model is a stand-in that returns the given log-probabilities (compute_loss reads nothing else of it), the tree
attention unpacking is stubbed (it only feeds that model), and the method is stopped right after the per-token loss
by an accelerator whose `num_processes` raises: the locals are read from the frame of that exception. Nothing is
downloaded at run time and nothing here is on the build or test path; regenerate only when the pinned source changes:

    ~/project/shared/venvs/trl-golden/bin/python -I tools/goldens/trl_score_centring_reference.py \\
        --trl ~/resources/reference/trl/src

Pinned: TRL source tree at commit 6267b72 (its `trainer/async_grpo_trainer.py` SHA-256 is recorded in the fixture and
checked here), torch 2.14.1+cpu, transformers 5.24.1, accelerate 1.16.0, datasets 5.2.0, numpy 2.5.5, Python 3.14,
in an isolated venv (`python3 -m venv trl-golden`; torch from the PyTorch CPU index, the rest from PyPI).
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
    }
    pathlib.Path(a.out).write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {a.out}: {len(groups)} groups, {len(cases)} tokens")


if __name__ == "__main__":
    main()
