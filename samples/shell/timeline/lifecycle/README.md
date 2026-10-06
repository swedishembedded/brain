<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: timeline/lifecycle

The whole life of a timeline model from the command line: train it, judge it
on subjects it never saw, calibrate it, and ask it for the forecast of a
patient history, with `brain horizon train`, `eval`, `calibrate` and
`predict`.

```bash
make samples/study/timeline/run ARGS="--out /tmp/timeline"
samples/shell/timeline/lifecycle/lifecycle.sh /tmp/timeline/data
```

The study sample writes four disjoint `timeline-v1` files of its synthetic
population to `<out>/data`; the script then runs

1. `brain horizon train --dataset train.jsonl --held-out held-out.jsonl --out DIR`
   with early stopping on the held-out subjects. It prints a progress line per
   evaluation interval, and the model directory appears only when training
   completes; a cancelled or failed run leaves nothing there.
2. `brain horizon eval --weights DIR --dataset test.jsonl --horizons 5,10 --json`:
   per outcome code and horizon, Uno's concordance, the time-dependent AUC, the
   IPCW Brier score and its integral, and calibration. A horizon with too few
   events is listed under `absent`, not reported.
3. `brain horizon calibrate --weights DIR --validation validation.jsonl --horizons 5,10`:
   Venn-Abers calibrators fitted on subjects the model was neither trained nor
   early-stopped on, written to `calibration.json` beside the weights. It
   refuses to overwrite one without `--force`; `--out DIR2` writes a calibrated
   copy instead.
4. `brain horizon predict --weights DIR --history patient.json --json`: the
   forecast with the raw and calibrated risk and its interval.

It also trains a bootstrap ensemble of three (`--members 3 --ensemble
bootstrap`: members trained on the subjects resampled with replacement) and
asks it the same question: the directory is loaded as an ensemble without
being told it is one, and the forecast carries the members' mean and the range
they span. A forecast is a statistical estimate for the population the model
was trained on; it is not a diagnosis and not a treatment recommendation.

The same actions are served on HTTP and D-Bus by `brain serve` when
`BRAIN_HORIZON_DIR` names a model directory (`BRAIN_HORIZON_TRAIN_DIR` is where
a served `train` writes); there the datasets travel as the request's blobs.

Needs the `brain` CLI on `PATH` and `python3` for the printout.
