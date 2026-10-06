<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: timeline/predict

Serve a saved timeline model from the command line: for each subject of a
`timeline-v1` file, the probability of surviving every absorbing outcome
and of each outcome code, at the times asked for.

```bash
make samples/study/timeline/run ARGS="--out /tmp/timeline"
samples/shell/timeline/predict/predict.sh /tmp/timeline/model /tmp/timeline/subjects.jsonl 5,10
```

It runs `brain horizon predict`, the `predict` action of `brain/horizon`,
and prints one line per subject. A time past the model's last knot is
refused rather than extrapolated. The same action is served on HTTP and
D-Bus by `brain serve` when `BRAIN_HORIZON_DIR` names the model directory.

Needs the `brain` CLI on `PATH` and `python3` for the printout.

## Patient histories: append a checkup, re-predict

`predict` also takes a patient history instead of `timeline-v1` subjects:
`--history patient.json` (shorthand for `--in history=patient.json`) answers
with a structured forecast per history instead of bare probabilities, and
`--json` prints it. `history.sh` shows the loop on one synthetic patient:

```bash
make samples/study/timeline/run ARGS="--out /tmp/timeline"
samples/shell/timeline/predict/history.sh /tmp/timeline/model-visits /tmp/timeline/subjects-visits.jsonl 5,10
```

It builds the patient's history H0 from a record of the visit model's test
subjects, sends it (R0), appends a later checkup and sends the whole history
again (R1), appends another (R2), and prints the ten-year risk each time with
what each forecast covered and how far it is inside the training support.
Every call carries the whole history: the weights never change for a patient
and nothing is stored. A forecast is a statistical estimate for the
population the model was trained on; it is not a diagnosis and not a
treatment recommendation. The visit model is the study's second model, trained
on records with several visits as a history with appended checkups is; the
first model, trained on one visit per subject, answers `risk: unavailable`
for such a history.

Needs `python3` as well.
