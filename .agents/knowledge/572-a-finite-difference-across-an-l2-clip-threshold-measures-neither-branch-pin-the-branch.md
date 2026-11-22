<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 572. A finite difference across an L2 clip threshold measures neither branch: pin the branch the forward took

**Lesson:** `y = a * min(1, thr / (|a| + eps))` has a kink where a vector's norm meets the threshold. With the routing
pinned (#562) the clips were the remaining non-smooth points of Inkling 2's training forward (queries, keys and
relative states of every visit), and a gradient check whose step carries one vector across its threshold fails with a
correct backward. Pin the branch: `l2clip_branch` records each vector's branch, the forward and backward
(`l2clip_scale`, `l2clip_scale_dx` with `pinned` set) replay it, and the loss becomes the recorded branch's smooth
continuation, whose derivative at the recorded point is the analytic gradient. In Inkling 2,
`Inkling2Train::pin_clip_branches(true)` beside the routing replay. Tell a kink from curvature with one-sided
differences: around a kink the forward and backward slopes disagree by a constant beyond some offset; under curvature
their gap shrinks linearly with the step and the central difference converges as the step squared.

## Evidence (2026-11-22)

- `gradcheck` `inkling2_fd` LoRA at seed 1778422313 (every backend): `layers.0.attn.wr_du.weight.lora_b` analytic
  2.035, directional central difference 2.234 at a step of 1e-3. Entry by entry, the one-sided differences of
  `lora_b[9]` agreed at 3e-4 (0.528) and split to 0.31 / 0.52 at 1e-3; with the relative-state clip lifted
  (`rel_clip = 1e9`) every entry was smooth. With the branches pinned the seed passes (regression test
  `inkling2_lora_check_holds_where_a_step_crosses_a_clip_threshold`), and ten random seeds of the three checks
  passed the LoRA and shared-core checks.
- Not a clip: the full fine-tune's directional check at seed 1832604347 (`layers.0.mlp.experts.w13_weight`, -77.6
  vs -67.7) is smooth and strongly curved along a ±1 direction of the whole bank: the central difference converges
  to the analytic value as the step squared (step 1e-3: 32.8, 2e-4: 70.9, 2.5e-5: 74.2 against 74.1), the one-sided
  slopes' gap shrinking linearly. That is the step's truncation error, not a kink.
- Kernel spec: `model` it `norm_clip_ops::l2clip_scale_pinned_branches_replay_and_continue_smoothly`.
