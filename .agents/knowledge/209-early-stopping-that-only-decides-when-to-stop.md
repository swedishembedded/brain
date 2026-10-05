<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 209. Early stopping that only decides when to stop

`model::fit_controlled` watched the held-out loss, stopped after `patience`
evaluations without improvement, and kept the best parameters - but only by
writing them to the checkpoint path it was given. A caller that keeps the
returned model in memory passes no path: the timeline SDK's `train`, the chat
and preference fine-tunes. Those callers got the model trained `patience`
evaluations past its best, while the SDK documented "keeping the best model".

Every signal looked right. The log printed the best held-out loss, the run
stopped where it should, and the saved-checkpoint tests checked that what
landed on disk was the best step's - because those tests all passed a path.
The model handed back was never checked.

The number that showed it: a timeline sample whose held-out loss bottomed at
step 200 (2.6429) reported its final held-out loss as the step-800 value
(2.7936). With the best parameters restored, the same run's mean error
against the true ten-year risk fell from 0.090 to 0.061 and its calibration
slope went from 0.61 to 1.15. A cross-validation campaign had already trained
dozens of folds on the last parameters and was rerun.

## What to do instead

- Hold the best parameters whenever early stopping is armed, and restore them
  into the returned model; writing them to disk is a separate, optional step.
- Test what a caller receives, not only what lands on disk: the regression
  test reads the returned model's parameters with no checkpoint path.
- When a run reports a final held-out loss, compare it with the best one it
  printed. They must be equal after early stopping.
