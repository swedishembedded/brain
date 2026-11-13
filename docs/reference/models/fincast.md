# FinCast

FinCast is a financial time-series forecaster: a patched-decoder model with
a mixture-of-experts core and a probabilistic quantile head, tuned for
daily, weekly, and monthly financial series. Reach for it as a
finance-tuned alternative to Chronos-2 when you want both a mean forecast
and quantile bands for financial data.

> **License note:** the reference model is released under Apache-2.0, but
> its authors state it is "for research and educational purposes only" and
> "does not constitute financial advice." Do not use FinCast's output as a
> basis for real financial decisions.

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

- **Model id:** `brain/fincast` - a reserved `brain/` id, never auto-fetched.
- **Weights:** the model store resolves the checkpoint: the released FinCast
  safetensors (recognised by the tensors it holds - it carries no config) or
  a brain `.safetensors` container (what `import` writes). `--weights PATH`
  names one for a single run, `brain serve --pin fincast.weights=PATH` for a
  server.
- **Import:** the raw upstream checkpoint is converted to safetensors first,
  then may be imported into a brain container (optional - the converted file
  is read as it is):
  ```bash
  python3 tools/convert/fincast_convert.py <v1.pth> <fincast.safetensors>
  brain fincast import --source <fincast.safetensors> --dest fincast-brain.safetensors
  ```

## Running it

```bash
# a forecast of a context series (raw f32 little-endian, shape in the blob meta)
brain fincast forecast --in context=series.f32 --horizon 64 --freq 0 --out forecast=out.f32

# score it on the last bars of an OHLCV CSV it was not shown, next to
# persistence, drift and seasonal naive
brain fincast predict --in csv=ohlcv.csv --horizon 48 --origins 5
```

`brain fincast <action> --help` lists every parameter. The same two actions
are served by `brain serve` (D-Bus): `forecast` takes the input blob
`context` and the parameters `horizon` (forecast length) and `freq` (0 =
daily, 1 = weekly, 2 = monthly) and returns `forecast` (shape `[horizon, 10]`,
column 0 is the mean, the remaining 9 are quantiles); `predict` takes the CSV
as the input blob `csv`.

```bash
dbus-run-session -- bash -c '
  brain serve --dbus --pin fincast.weights=/path/to/fincast.safetensors &
  sleep 2
  python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/fincast --freq 0
'
```

The scenario battery against the statistical baselines, the transformer core
on the Intel NPU and the JSONL forecasting server are the application part:
[`samples/models/fincast`](../../../samples/models/fincast/README.md)
(`compare`, `npu-forecast`, `serve`).

## Options

- `forecast`: `horizon` (default 64), `freq` (default 0).
- `predict`: `horizon` (default 48), `context` (0 = as much as the window
  allows), `samples`, `origins`, `seed` (drawn and echoed when absent),
  `season`, `item`, `freq`, `chart`.
- `import`: `source`, `dest`.

## Hardware and limits

- Reference weights are for research and educational use only - not
  financial advice; see the license note above.
- Inference only today - no fine-tuning or LoRA path.
- No HTTP route: reachable through `brain fincast <action>` and D-Bus.
- Runs on CPU or GPU, and the `forecast` action is placed on an NPU
  automatically when one is available and budgeted for the request (`predict`
  runs the device path).
