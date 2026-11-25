<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 618. A support-normalised log-probability is one backward of sparse targets, and a loose gradcheck cannot see it wrong

**Lesson:** renormalising the trainer's log-probability over a truncated sampler's support (TRL's support replay)
needs no new kernel and no second pass. With `log pi_S(y) = l_y - logsumexp_S l` and `d logsumexp_S = sum_j pi_j e_j -
softmax`, the whole-row softmax cancels: a loss of weights `omega` over renormalised log-probabilities is the label weight
`omega_y` plus extra-target coefficients `c_j = omega_j - Omega pi_j` at the support's ids (`Omega` the weights' sum),
whose row total is zero (`model::head_ce::support_coefficients`). The existing sparse-target gather and backward
(`ce_gather_logprob`, `ce_sparse_targets`) do it in the pass they already run. And the workspace gradcheck tolerance
(`failures(4e-3, 8e-2)`) does not catch a wrong coefficient on the tiny Qwen3.5: its adapter gradients are about
1e-4, below the absolute floor.

## Evidence (2026-11-25, lane agrpo-support, CPU)
- `model` `it::head_ce_sparse::a_loss_renormalised_over_a_support_is_one_backward_of_its_extra_targets`: the hidden
  gradient of the renormalised loss from the coefficients equals an f64 oracle that differentiates the loss directly
  (1e-4 of the largest); with `Omega` scaled by 0.9 in the coefficients it fails.
- `rl` `it::support_replay` gradchecks (clipped and score-centred over the support, tiny Qwen3 and Qwen3.5): with the
  same 0.9 mutation, at `failures(4e-3, 8e-2)` the Qwen3.5 check passed (worst abs error 2.3e-3, rel 0.10) and only Qwen3
  failed (abs 3.2e-2, rel 0.27). At `failures(2e-4, 2e-2)` both fail mutated and both pass unmutated (worst rel error
  6.7e-3 on Qwen3.5, 6e-4 on Qwen3). The supports the check runs over hold little of the row's mass: a support close to the
  whole row also hides the error, since the difference a wrong `Omega` leaves is about `(1 - mass) * softmax`.
- Cost, measured on the CPU backend (288 cores, `HeadCe::backward_sparse` over 1024 rows of a 32,000-token vocabulary,
  hidden 64, three runs): 0.45 s at one extra target per row, 0.52 s at 256, against 0.42-0.44 s for the plain weighted
  backward. The per-logit K scan of `ce_sparse_targets` is not the cost on the CPU; a card has not been measured.

## Rule
A support (or any renormalisation over a subset of the row) goes through the head's extra targets, not a new pass. A
gradcheck of a coefficient-built loss on a tiny model states tolerances that a deliberately wrong coefficient fails,
shown by trying one.
