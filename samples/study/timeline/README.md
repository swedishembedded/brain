<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: study/timeline

Train a continuous-time timeline model on irregular records with competing
outcomes, check it against the risk that generated the data, turn its risk
into calibrated intervals, and save it for serving.

```bash
make samples/study/timeline/build
make samples/study/timeline/run ARGS="--out /tmp/timeline"
```

## What it demonstrates

* **Training is a library call.** `brain::TimelineModel::train` (the
  `timeline` surface) fits the vocabulary on the training records and trains
  with early stopping on held-out subjects, handing back the parameters the
  best held-out likelihood was measured on.
* **Checked against the truth.** The records come from brain's synthetic
  population, whose true cumulative incidence is known per subject: two
  competing causes of death driven by measurements (one detection-limited)
  and a diagnosis, a non-absorbing onset, an irrelevant covariate. On
  subjects the model never saw, its ten-year risk of the first cause is
  compared with the truth and with a covariate-blind (Aalen-Johansen)
  estimate, by mean absolute error and IPCW Brier score, and its Uno
  concordance is reported. The program exits non-zero unless the model beats
  the covariate-blind estimate on both.
* **An interval, not a point.** `brain::survival::venn_abers` calibrates the
  risk on the early-stopping subjects into a Venn-Abers interval per subject;
  the calibration slope before and after and the interval widths are printed.
* **Saved for serving.** The model is saved to `<out>/model` with twenty test
  subjects in `<out>/subjects.jsonl`: what
  [`samples/shell/timeline/predict`](../../shell/timeline/predict/README.md)
  serves.

## Usage

```text
--out DIR        where the saved model and the test subjects go
                 (default: <tmp>/sample-study-timeline)
--subjects N     training subjects (default 20000; a fifth as many held out
                 for early stopping and calibration, a quarter tested)
--steps N        optimizer steps at most (default 3000)
--seed N         seed of the weights and batches (default 1)
```

`BRAIN_BACKEND` chooses the device (`cpu` runs anywhere, slowly).

## Builds with

The `timeline` surface of the `brain` SDK and `serde_json`, nothing else.
