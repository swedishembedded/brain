<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 483. A resumed run that did not select on held-out loss lost the evaluations before its resume

**Lesson:** What a report promises across a resume ("every evaluation of the whole run, those before a resume
included") must be in the resume state for every run, not only for the runs whose control flow happens to need it.

## What happened

`model::train`'s exact-resume state saved the selection record (best loss, the evaluation curve, the held parameters)
only when the run selected on held-out loss (`patience > 0`). A pretraining run with `eval_every` but no patience,
stopped and resumed, reported only the evaluations after the resume, while `FitReport::evaluations` documents the
whole run. The weights were exact; only the curve was short. Found by the from-scratch conformance test, which holds a
resumed run's evaluations to a straight run's.

## What changed

The state always carries the selection record (it already loaded one whenever present), so a resumed run reports the
whole curve; `crates/sdk/tests/it/from_scratch.rs` asserts the resumed run's evaluations equal the straight run's.
