<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/kronos

The Kronos OHLCV forecaster beyond its served actions, as a standalone program: the scenario battery against the statistical baselines, the autoregressive rollout on the Intel NPU through OpenVINO, and the JSONL forecasting server over a Unix socket.

The model's own actions - `forecast`, `predict` and `finetune` - are served
ones: `brain kronos <action>` and `brain serve` run the same code, with
`--help` listing every parameter. `compare`, `npu-forecast` and the `serve`
workflow are this program's own: they need the scenario battery, the NPU crate
and the event server.

The checkpoints are resolved from the model store, as `brain kronos <action>`
does: the decoder (`NeoQuasar/Kronos-base`) and the tokenizer
(`NeoQuasar/Kronos-Tokenizer-base`), told apart by their own `config.json`.
`--decoder PATH` and `--tokenizer PATH` name them outright. `compare` and
`serve` need none: without a checkpoint they run on the baselines alone.

## Build

```bash
make samples/models/kronos/build
G=target/release/sample-models-kronos
```

## Run

```bash
# the scenario battery: the statistical baselines, plus Kronos when its checkpoint is found
$G compare --windows 24 --seed $RANDOM --out report=report.md --out html=report.html

# the autoregressive rollout with both decoder graphs on the Intel NPU (needs OpenVINO and an NPU); --verify checks the tokens against the device core
$G npu-forecast --horizon 8 --verify true --ov-device npu --seed $RANDOM

# the JSONL forecasting server over an owner-only Unix socket (stdio without --socket)
$G serve --socket /tmp/brain-forecast.sock
```

`compare` fails, printing the report, if a model falsely beats the naive
forecast on the random walk (the negative control). `npu-forecast` runs the
fixed-length graphs on a synthetic context of one window.

`$G <action> --help` lists every parameter with its default; `--device` places
the run.
