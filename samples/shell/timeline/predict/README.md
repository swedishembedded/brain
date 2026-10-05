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
