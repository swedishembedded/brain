<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 509. A policy-gradient KL of 1e7 was the k3 estimator read on tokens the trained policy no longer draws

**Lesson:** The k3 KL estimator, `rho - 1 - ln rho` with `rho = pi_ref / pi`, estimates `KL(pi || pi_ref)` only over
tokens drawn from `pi`. A policy-gradient step evaluates it on the rollout's tokens, which the behaviour policy drew;
once the trained policy has pushed one of them down by 17 nats, that token alone reads `rho = 4e7` while its share of
the divergence is at most one. The reported KL (and the step guard that reads it) is the mean of `ratio * k3`
(`ratio = pi / pi_behaviour`), whose expectation over the behaviour's tokens is `KL(pi || pi_ref)` however far the
policy has moved; on-policy the weight is one and nothing changes. The token ratio near 1e-8 itself was real.

## What was reported

An orchestrator's policy-gradient runs on Qwen3-4B (LoRA rank 16 on every projection, answers of 2 to 8 tokens,
`reference: start`, lr 1e-5) logged a mean KL of 1e2 to 1e7 per step and a smallest token ratio of about 2.5e-8 from
step 2 on, the same figure on several steps. Its own isolation (`brain finetune --policy-gradient --behaviour
start_of_run_proxy --kl-beta 0`): the same on CUDA, WGPU and the CPU backend; nothing moves at lr 1e-12; scoring every
record before the first update gives the same KL as lazy scoring; KL grows with lr (1e-7: 0.009, 1e-6: 0.68, 1e-5:
1.6e8 on 12 records, 6 steps). Those runs used the start-of-run proxy: the records carried no recorded behaviour
log-probabilities.

## What brain does with recorded log-probabilities (measured, toy qwen35, CPU)

`recorded_log_probs_in_the_contracts_convention_give_a_first_step_ratio_of_one`
(`crates/rl/tests/it/external_policy_gradient.rs`): behaviour log-probabilities equal to the model's own teacher-forced
ones give a first step whose ratio is 1.000000 at every token and a KL to the start of 0.0: the ratio, its token
alignment (position `t` of a row is compared with the figure recorded for the token it predicts) and the temperature
scaling are the trainer's own. Figures in another convention move the first step before any update (one random seed; the figures move with it): shifted by one
token, ratios 0.748 to 1.172; recorded at T = 1 and trained with `temperature` 0.7, 0.949 to 1.075; a greedy sampler's
zeros, 0.031 to 0.041. So a first-step ratio far from one names the caller's figures; a ratio that falls from step 2 on
is the policy moving.

## Why 1e7

`the_reported_kl_estimates_the_policys_divergence_from_its_reference_off_policy_too`
(`crates/rl/src/objective/pg_batch.rs`): the behaviour (= reference) draws four tokens with probability 0.7, 0.2,
0.099, 0.001; the trained policy has moved the third down by 17.5 nats (ratio 2.8e-8). Exact `KL(pi || pi_ref)` = 0.104.
The mean of plain k3 over the behaviour's draws = 3.55e6, the range the orchestrator reported; the mean of `ratio * k3`
= 0.104. A clipped surrogate stops pushing a token once its ratio leaves the band, but Adam's first steps move every
adapter weight by about the learning rate whatever its gradient, so a rare token can still be carried 17 nats by the
updates for the others: that is the ratio of 2.5e-8, and at lr 1e-5 on records of a few tokens it is the dynamics, not
an accounting defect.

## Open

- The KL term of the loss (`kl_beta > 0`) is still the plain k3, whose gradient weight on a token is
  `kl_beta * (rho - 1)`: 8e5 for `kl_beta` 0.02 on the token above. Its importance-weighted form
  (`ratio * k3`, whose derivative in `log pi` is `ratio * ln(pi / pi_ref)`) is bounded there; changing the objective
  changes the gradient checks and the textbook's analysis of the k3 term, and was not done here.
- The real-weight runs were not repeated with recorded behaviour log-probabilities (needs a card).
