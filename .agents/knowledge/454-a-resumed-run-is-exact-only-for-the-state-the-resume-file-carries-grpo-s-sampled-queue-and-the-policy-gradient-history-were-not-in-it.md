<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 454. A resumed run is exact only for the state the resume file carries: GRPO's sampled queue and the policy-gradient history were not in it

**Lesson:** "weights, moments, schedule and the loop's generator" is the state of a supervised run, not
of every objective. An objective that holds anything between steps (rows it sampled and has not
trained, a per-step history it reports) loses it at a stop unless `Objective::cursor` carries it. A
resume test whose stop falls where that state happens to be empty passes either way.

## What happened

- **GRPO** samples a group of `group_size` completions and queues the rows; a step takes as many as
  it trains. A stop inside a group dropped the queued rows, so the resumed run sampled a new group
  where the uninterrupted run trained the old one, from the policy that had sampled it. A first test
  with a stop after four steps passed: with one row of each group dropped for a zero advantage, two
  rows a group and one row a step, the queue was empty at every even step.
- **The external policy gradient** trained bit-identical adapters after a resume, but its per-step
  statistics (the outcome's `steps`, the card's `last_step`, `mean_loss`, `mean_kl`) covered only
  the resumed segment and numbered its steps from 1, so the adapter file's digest differed from the
  uninterrupted run's.

## Fix

`Grpo` saves its queue of untrained rows in the cursor (floats as their bits, so a row comes back
exactly); the external policy gradient saves its step history. Both refuse a saved value that does
not fit the run.

## Gate

`crates/rl/tests/objective_cursor.rs::a_resumed_grpo_run_continues_its_rollout_stream` stops after
every step from the second to the fifth; `crates/sdk/tests/it/chat_finetune_policy_gradient.rs::a_cancelled_policy_gradient_run_resumes_to_the_uninterrupted_adapter`
compares the adapter file's digest and every step's statistics with the uninterrupted run's.
