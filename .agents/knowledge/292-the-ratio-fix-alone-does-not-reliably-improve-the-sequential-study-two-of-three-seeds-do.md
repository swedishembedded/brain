<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 292. The ratio fix alone does not reliably improve the sequential study: two of three seeds do

Reduced experiment E1, pre-registered in
`crates/rl/tests/continual_study.rs`
(`reduced_ablation_of_the_ratio_basis_and_dynamic_sampling`, `#[ignore]`d).
Tiny pretrained position-copy base (2 layers, d_model 64, rank-8 attention LoRA),
sequential study of 4 cycles of 240 steps (the recorded study has 12), group size
2, sampling temperature 1.5, no plasticity control, three drawn study seeds
(15161033823928307808, 8972090161791146660, 225361715365712761), one CUDA run,
debug build. Metric: the study's final ACC (mean score over all frozen probes
after the last cycle). Arms share every other setting (z-scored advantages,
per-row mean, symmetric clip 0.2, no KL measurement):

| arm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| A sampler ratio (the legacy basis) | 0.135 | 0.161 | 0.250 |
| B trainer ratio (ratio fix only) | 0.313 | 0.276 | 0.193 |
| C trainer ratio + dynamic sampling | 0.214 | 0.339 | 0.365 |

Decision rule: the ratio fix is a contributing cause iff ACC(B) > ACC(A) on every
seed. Result: NOT MET (B beats A on seeds 1 and 2 by 0.18 and 0.12, loses on
seed 3 by 0.06). C beats A on all three seeds (by 0.08, 0.18, 0.12) and beats B
on two; that comparison was not the registered rule.

What this does and does not show:

- VERIFIED-BY-RUN: the numbers above, with this fixture. The untrained base scores
  0.208 on the first probe; the legacy-ratio arm ended at or below it on two seeds.
- The registered rule is not met, so this run does not establish the ratio
  mismatch as a cause. With three seeds and a single-cycle outcome spread of
  about 0.3 on this fixture (the recorded pre-registration), a 0.06 loss on one
  seed and 0.12-0.18 wins on two cannot be told apart from noise.
- Arm A is the legacy ratio with the other corrections off, not the legacy
  objective: every arm skips an optimiser step that has no rows, which the legacy
  objective did not.
- Four cycles, not twelve; nothing here says anything about the 12-cycle result.
