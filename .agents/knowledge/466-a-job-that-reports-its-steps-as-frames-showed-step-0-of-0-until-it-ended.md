<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 466. A job that reports its steps as frames showed step 0 of 0 until it ended

**Lesson:** A progress callback carries a step count on every kind of report, not only on a plain tick. A router that
forwards only one kind of report as progress loses the count for every action that reports richer frames, and the
richest reporter, training, is the one whose progress a caller most needs.

## What happened

`brain/train` reports each optimiser step as `Progress::event(step, steps, frame)` (the frame carries the loss and
statistics). The job bridge turned only delta-less, event-less reports into `StreamMsg::Progress`, so the job's step
counter was never updated. A 538-step SFT job answered `GET /v1/jobs/{id}` with `"progress": {"step": 0, "total": 0}`
for its whole run, while `/events` was streaming step frames. Found while running a learner pilot on 2026-11-15.

## Fix and evidence

`crates/apiserve/src/bridge.rs` sends a frame that counts steps (`total > 0`) as the job's progress before the frame
itself, which `/events` already documents ("a `progress` event per step tick, a `frame` event per structured frame").
Gate: `a_job_that_reports_its_steps_as_frames_shows_them_as_its_progress` in `crates/apiserve/tests/capabilities.rs`,
red before the fix and green after.
