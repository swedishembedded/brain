<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 290. GRPO scored the untempered softmax and took optimiser steps on empty batches; the objective now does neither

Measured on the unmodified objective, tiny `qwen3` with a rank-2 LoRA, CPU backend,
vocabulary 23, rollout of 4 tokens after a 3-token prompt:

- Sampling at `T = 1.5`, the trainer's probability of the sampled token differed
  from the sampler's by up to 12.3% (`|ratio - 1| = 0.1229`) at the first update.
  The rollout stores `log softmax(logits / T)`; the training forward scored
  `log softmax(logits)`. Every existing test sampled at `T = 1`, so nothing
  caught it. With a clip band of 0.2 and a confident token, the ratio starts
  above the band and a positive-advantage row is clipped to weight 0.
- With about 80% of groups carrying no reward variance, 7 of 20 optimiser steps
  (4 micro-steps each) ran on zero rows. `fit_controlled` always called AdamW
  after `micro_steps`, so a step on zero gradients still applied weight decay and
  momentum.

What the objective does now (`rl::objective::grpo`):

- `prepare` sets the model's logit scale to `1/T`; the model's own rollout then
  samples at temperature 1 over the scaled logits (`Model::logits_all` returns
  them scaled while a scale is set, see 270). The behaviour log-probability of
  every row is recomputed by the training forward when the row is packed, so the
  first update's ratio is 1 up to float noise whatever produced the sample. The
  sampler's own value is kept only to log the gap (`GrpoStats::max_sampler_gap`).
  Truncated samplers (top-k, top-p) are flagged per row.
- `Objective::micro_steps` gathers `grad_accum` informative rows, drawing up to
  `resample_factor` (default 3) times the nominal groups; a shortfall is counted.
  `Objective::step_verdict` (new, default `Apply`) lets the objective refuse the
  step: no rows, or a non-finite loss, log-probability or gradient. Three
  non-finite steps in a row abort with `NonFiniteSteps`.
- Advantages default to mean-centred (a pair of rewards 0.6 and 0.5 gives
  +-0.05, not +-1); token terms are divided by one per-step denominator
  (`LossAgg::TokenSum`), so length no longer changes a token's weight; the clip
  band defaults to 0.2 / 0.28; the ratio exponent is capped at 20.
- A completion longer than the row holds is cut before it is verified, so the
  reward is for the span that is trained. A sampled stop token is a target
  (`Completion::eos_logprob`). Rows carry their policy version and are refused
  when older than the staleness bound (default 1). The KL to the
  adapter-disabled reference (`B` zeroed) is always logged.
- The loss probe draws a group of its own and no longer pops real rollout rows.

Not done here: batched KV-cached rollouts (the rollout still re-prefills per
token) and an entropy term.
