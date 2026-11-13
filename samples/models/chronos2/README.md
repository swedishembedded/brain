<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/chronos2

The Chronos-2 forecaster beyond its served actions, as a standalone program: the scenario battery against the statistical baselines, the transformer core on the Intel NPU through OpenVINO, and the JSONL forecasting server over a Unix socket.

The model's own actions - `forecast`, `predict` and `import` - are served
ones: `brain chronos2 <action>` and `brain serve` run the same code, with
`--help` listing every parameter. `compare`, `npu-forecast` and the `serve`
workflow are this program's own: they need the scenario battery, the NPU crate
and the event server.

The checkpoint is resolved from the model store, as `brain chronos2 <action>`
does: an `amazon/chronos-2` directory or a brain container. `--weights PATH`
names one outright. `compare` and `serve` need none: without a checkpoint they
run on the baselines alone.

## Build

```bash
make samples/models/chronos2/build
G=target/release/sample-models-chronos2
```

## Run

```bash
# the scenario battery: the statistical baselines, plus Chronos-2 when its checkpoint is found
$G compare --windows 24 --seed $RANDOM --out report=report.md --out html=report.html

# the transformer core on the Intel NPU (needs OpenVINO and an NPU); --verify compares with the device path
$G npu-forecast --horizon 8 --verify true --ov-device npu   # --in series=my_series.txt for a real context

# the JSONL forecasting server over an owner-only Unix socket (stdio without --socket)
$G serve --socket /tmp/brain-forecast.sock
```

`compare` fails, printing the report, if a model falsely beats the naive
forecast on the random walk (the negative control); a foundation model is slow
on a CPU, so keep `--windows` small there.

`$G <action> --help` lists every parameter with its default; `--device` places
the run.
