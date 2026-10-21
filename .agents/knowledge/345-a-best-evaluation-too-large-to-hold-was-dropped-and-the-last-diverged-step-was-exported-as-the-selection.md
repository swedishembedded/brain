<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 345. A best evaluation too large to hold was dropped, and the last (diverged) step was exported as the selection

Symptom: the CE8 27B run stopped at step 38 with "the best evaluation's parameters are kept" and then "the best parameters
were too large to hold and there is no checkpoint path: the model is the last step's". Its record said
`selection: last_step`, `selected_step: 38`, held-out loss 6.24 where the best evaluation (step 10) had 0.116: early
stopping silently defeated, the diverged adapter exported.

Cause: `fit_controlled` keeps the best parameters in host memory up to `BEST_IN_MEMORY_FLOATS` (64 Mi floats), and
otherwise writes them to `out`. A LoRA run has no `out` (the caller saves the adapter from the returned model), and a 27B
rank-16 adapter is 117 M floats (467 MB), over the budget. The code then set `best_lost` and carried on, handing back the last
step with a printed notice nobody reads.

Fix: the budget is `FitOpts::best_hold_floats`; a best copy over it is written to `out` if there is one, else spilled to
`<training state>.best` (atomic rename; named in the exact-resume state so a resumed run restores it, and refused if the
file is gone) and read back at the end of the run, then removed. With neither `out` nor a training state the run fails at
the first improvement with the sizes, instead of returning the last step as the selection. `ChatFineTune` always has a
training state, so a 27B run always keeps its best evaluation's adapter.
Tests: `a_best_copy_too_large_to_hold_is_spilled_beside_the_state_and_restored` (also across a stop and resume) and
`a_best_copy_with_nowhere_to_go_fails_the_run_loudly`.
