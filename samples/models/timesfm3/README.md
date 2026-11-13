<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/timesfm3

The TimesFM-3 forecaster beyond its served actions, as a standalone program: the scenario battery against the statistical baselines and the JSONL forecasting server over a Unix socket.

The model's own actions - `forecast`, `predict`, `finetune` and `import` -
are served ones: `brain timesfm3 <action>` and `brain serve` run the same code,
with `--help` listing every parameter. `compare` and the `serve` workflow are
this program's own: they need the scenario battery and the event server.

The checkpoint is resolved from the model store, as `brain timesfm3 <action>`
does: a `google/timesfm-3.0-pytorch` directory or a brain container.
`--weights PATH` names one outright. `compare` and `serve` need none: without a
checkpoint they run on the baselines alone. The 3.0 weights carry a
non-commercial license (see the model's page).

## Build

```bash
make samples/models/timesfm3/build
G=target/release/sample-models-timesfm3
```

## Run

```bash
# the scenario battery: the statistical baselines, plus TimesFM-3 when its checkpoint is found
$G compare --windows 24 --seed $RANDOM --out report=report.md --out html=report.html

# the JSONL forecasting server over an owner-only Unix socket (stdio without --socket)
$G serve --socket /tmp/brain-forecast.sock
```

`compare` fails, printing the report, if a model falsely beats the naive
forecast on the random walk (the negative control).

`$G <action> --help` lists every parameter with its default; `--device` places
the run.
