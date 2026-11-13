# Chronos-2

Chronos-2 is a general-purpose probabilistic time-series forecaster: feed it
a raw numeric context series and it returns a full quantile forecast (21
quantile levels) instead of a single point estimate. Reach for it when you
need a forecast for an arbitrary numeric series with no domain-specific
structure, and you want calibrated uncertainty bands out of the box rather
than a bare mean prediction.

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [ ] |
| CLI (`brain <arch> <action>`) | [x] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

- **Model id:** `brain/chronos2` - a reserved `brain/` id, never auto-fetched.
- **Weights:** the model store resolves the checkpoint: an `amazon/chronos-2`
  directory (recognised by its `config.json`) or a brain `.safetensors`
  container (what `import` writes). `--weights PATH` names one for a single
  run, `brain serve --pin chronos2.weights=PATH` for a server.
- **Import** (optional - the released directory is read as it is):
  ```bash
  brain chronos2 import --source <amazon/chronos-2 dir> --dest chronos2.safetensors
  ```

## Running it

```bash
# a forecast of a context series (raw f32 little-endian, shape in the blob meta)
brain chronos2 forecast --in context=series.f32 --horizon 64 --out forecast=out.f32

# score it on the last bars of an OHLCV CSV it was not shown, next to
# persistence, drift and seasonal naive, with the evidence chart
brain chronos2 predict --in csv=ohlcv.csv --horizon 48 --origins 5 --chart true --out chart=chart.png
```

`brain chronos2 <action> --help` lists every parameter. The same two actions
are served by `brain serve` (D-Bus): `forecast` takes the input blob
`context` and the parameter `horizon` and returns `forecast` (shape
`[21, horizon]`, quantile levels in the response metadata); `predict` takes
the CSV as the input blob `csv`.

```bash
dbus-run-session -- bash -c '
  brain serve --dbus --pin chronos2.weights=/path/to/chronos2.safetensors &
  sleep 2
  python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/chronos2 --horizon 64
'
```

The scenario battery against the statistical baselines, the transformer core
on the Intel NPU and the JSONL forecasting server are the application part:
[`samples/models/chronos2`](../../../samples/models/chronos2/README.md)
(`compare`, `npu-forecast`, `serve`).

## Options

- `forecast`: `horizon` (default 64).
- `predict`: `horizon` (default 48), `context` (0 = as much as the window
  allows), `samples`, `origins`, `seed` (drawn and echoed when absent),
  `season`, `item`, `freq`, `chart`.
- `import`: `source`, `dest`.

## Hardware and limits

- Inference only today - no training or LoRA path.
- No HTTP route: reachable through `brain chronos2 <action>` and D-Bus.
- Runs on CPU or GPU, and the `forecast` action is placed on an NPU
  automatically when one is available and budgeted for the request (`predict`
  runs the device path).
- Univariate forecasting only - known-future covariates and multi-series
  grouping are not part of the served path yet, and horizons much beyond
  1024 steps are not a supported use case.
