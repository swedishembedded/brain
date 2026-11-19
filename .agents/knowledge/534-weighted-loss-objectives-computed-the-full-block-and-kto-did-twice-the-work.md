<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 534. Weighted-loss objectives computed the full block, and KTO did twice the work

**Lesson:** an optimisation made on one objective's path does not reach the others unless they share it. SFT trimmed
each row to its last supervised position, but the weighted-loss objectives (policy gradient, KTO, DPO) built their
batches through another path and computed every row at the full block. KTO also did about twice the work per example
and scored its reference for the whole dataset before the first step. When a training run is slow, check per objective
which positions and rows a step really computes (`step_work`), not only the SFT path that was optimised.

## What happened

2026-11-19, reading the code behind slow runs on Qwen3-4B (one GH200, LoRA rank 16):

- **No trimming under weighted loss.** `Qwen::needed_len` trimmed SFT rows, but a weighted batch kept every row at the
  block length, so rows of 4-6k in a 16k block computed 2-3x the positions they needed.
- **KTO's two-row batch.** An example was one b=2 batch: a step forwarded it once for the reference point and again to
  train, and backwarded the mismatched row at weight zero, about 2x the work of what the loss needs.
- **Eager reference scoring.** `check_data` scored the reference for every example before step 1: the run started 55
  minutes late, including examples it never drew.

## What to do

All three are fixed under the tier-1 step-parity harness (knowledge 532), each with no change beyond round-off in loss,
log-probabilities, gradients and post-step weights:

- A1: weighted rows are trimmed to the last supervised position (`qwen3` `step_parity`; gradients differ by 3.1e-7,
  a reordered reduction; 8192 to 5376 computed positions per step on the fixture);
- A2: KTO forwards each row as its own one-row pass (`rl` `kto_step_parity`, difference 0);
- A3: KTO scores an example's reference when a step first draws it, under the start-of-run adapter (difference 0;
  scoring under the live adapter moves log-probabilities by 0.083 and fails).

A policy-gradient step now reports its work (`step_work`), so a later objective that misses a shared optimisation shows
it in the run's own output.

## Evidence

`.agents/roadmap/training-efficiency.md`, diagnosis rows 1, 3 and 4 and items A1-A3.
