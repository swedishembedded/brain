<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 617. A training stage spent 59% of its wall time outside optimizer steps, and nothing said where

**Lesson:** a rate measured inside the work (tokens per second of a step) says nothing about the job's wall time.
Account for every second of a job in named phases that add up to its wall time, and publish the split while it runs;
a residual that grows is then a finding, not a mystery.

## What happened

2026-11-27, fine-tunes on GH200 cards beside serving: about 59% of a training stage's wall time was outside optimizer
steps. The step frames reported each step's loss, rate and elapsed seconds, the server's `/health` said nothing of
training, and nobody could tell whether the time went to waiting for the card, loading the base, evaluation passes,
checkpoint saves or the back-offs after a refused allocation (#615).

## Why

- The training loop timed only its steps (`StepWork`: prepare, forward, backward, optimizer). Evaluation, the
  periodic training-state saves, published adapter versions and a streamed run's wait for records ran between steps,
  untimed.
- What happens before the loop (the queue wait for the card, staging the datasets, building the trainer) and after it
  (the final save, publishing the adapter) was outside every measurement.
- The job's progress went only to its own client; the server had no record of it to show anyone else.

## What to do

- `model::train` accounts the loop's time itself (`LoopTime` on every `StepReport`, `FineTuneProgress::time`):
  optimizer steps, evaluation, checkpoint saves, waiting for records, cumulative from the loop's start.
- The executor arms every workload job's report (`capability::report::JobReport`) and tells it how long it waited for
  its card; `trainjob::observe` follows the job's own frames (a `stage` frame's `phase`, a `step` frame's `time`) and
  publishes `phases_secs`: `waiting_for_card`, `staging`, `model_load`, `optimizer_steps`, `evaluation`,
  `checkpoint_save`, `waiting_for_records`, `finalize` and the residual `other`, which add up to `wall_secs`, with
  `outside_optimizer_steps_share`, `oom_retries` and tokens per second inside steps.
- The server lists every training job with that report in `GET /v1/metrics` (`training_jobs`,
  `docs/reference/api/metrics.md`).

## Evidence

Specs: `crates/trainjob/src/observe.rs::every_second_of_a_jobs_wall_time_is_in_exactly_one_phase` (a stage whose steps
are 41 s of 115 s splits the other 74 s by phase, 4 s left unnamed),
`a_retry_after_the_card_refused_an_allocation_is_counted_and_its_wait_is_waiting_for_the_card`,
`crates/trainjob/tests/it/device_oom.rs` (a real run with a refused allocation at step 27: one retry, phases add up to
the wall time), `crates/apiserve/tests/it/metrics.rs` (the job and its report in the server's metrics).
