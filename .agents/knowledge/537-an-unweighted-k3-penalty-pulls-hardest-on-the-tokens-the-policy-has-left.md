<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 537. An unweighted k3 penalty pulls hardest on the tokens the policy has left

**Lesson:** the policy-gradient KL term must be weighted by the importance ratio in the loss, not only in the reported
value. The k3 estimate `e^u - u - 1` (`u = ref_lp - new_lp`), held fixed at a sampled token, has gradient weight
`kl_beta * (e^u - 1)`. On a token that the behaviour policy drew and the trained policy has since pushed `d` nats below
the reference, that weight is about `kl_beta * e^d`. Such a token holds almost none of `KL(pi || ref)`, yet it gives the
step's largest pull. The step's mean KL stays in band while its loss and gradient norm explode.

Fixed in `rl::objective::grpo_term::token_term_weighted`: the term is `kl_beta * ratio * k3`, and both factors are
differentiated. Within the caps the weight is `kl_beta * ratio * (ref_lp - new_lp)`. Its value and expected gradient
under the behaviour policy are those of `KL(pi || ref)`, on-policy and off. That is the divergence the step reports and
`max_step_kl` bounds; the reported value was already ratio-weighted. The unweighted term descended `KL(ref || pi)`. On
policy and near the reference both give the same first-order weight `kl_beta * u`. Off the band they differ, so a run
with `kl_beta > 0` is a new arm against runs on older builds.

## Evidence (2026-11-21)

- **Run:** an orchestrator's guarded on-policy round on agent rollouts (seed 1).
  - A2s1 (a Qwen3-4B LoRA), `--recipe on-policy-rl` (token-sum loss, lr 1e-4, `kl_beta` 0.02 by default).
  - Recorded behaviour on 2,006 sampled-only records, reference = start adapter.
  - Dual clip 3; step guards `max_step_kl` 0.1 and `max_step_grad_norm` 5.
  - Brain 5e25d5b82.
- **Before the divergence:** guard skips at steps 70, 94 and 110 (gradient norms 44-1,390).
- **Divergence:** from step 135 the loss read 86-17,784 on 12 of 25 steps. The gradient norms were up to 1.6e6.
  - The reported KL stayed 0.016-0.031 and the mean ratio 0.96-0.99.
  - The policy surrogate is bounded by the clip and the dual clip (at most about `3 * |A|`, `|A|` <= 2.65 here). Only the
    KL term can reach such a loss.
- **End:** brain stopped the job at step 160 after 8 guard skips in a row. Seed 2 skipped 11 steps by step 330 (gradient
  norms up to 974).
- **Specs** (`grpo_term` tests):
  - A drawn token 17.5 nats below the reference costs at most `kl_beta` and has weight below 1e-6.
  - By exact enumeration, the expected logit gradient under a behaviour policy that differs from `pi` equals that of
    `KL(pi || ref)`.
  - The fp64 reference and the finite-difference check of loss against weight hold past the exponent caps.
  - The `brain-rl` suite with `qwen3` passes, its gradchecks included.
