<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 513. A full fine-tune's precision was checked against its architecture only once the run was queued

**Lesson:** admission must make every check that needs no weights and no dataset. A refusal the trainer gives at
start-up turns a bad request into a job that is accepted, waits for a card and then fails.

## Cause

`full_train` admission parsed `train_precision` and sized the footprint, but whether the architecture runs that
precision (`TrainPrecisionSupport::check`), whether it has exit gates for `exit_entropy`, and whether the optimiser
plan holds (`ValidatedTrainingPlan`) were checked inside `ChatFullFineTune::run_with` and the family's trainer, after
the job had been placed. `weights=f32` on a Qwen3 base was a 202 and a failed job.

## Fix

`ChatFullFineTune::check` runs those checks reading only the base's config (`ChatEntry::train_precision` carries the
family's `TrainPrecisionSupport`); `run_with` makes the same checks through the same code. `brain/train` admission
calls it, so such a request is a 400 before a job exists, and `validate_only` answers the resolved settings and the
footprint as host work. Tests: `a_full_fine_tune_checks_its_settings_before_it_is_queued`,
`a_full_fine_tune_the_trainer_would_refuse_is_refused_at_submission`.
