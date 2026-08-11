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
| CLI (`brain do`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

- **Model id:** `brain/chronos2` — a reserved `brain/` id, never auto-fetched.
- **Weights:** set `BRAIN_CHRONOS2` to a single brain `.safetensors` file
  produced by the import step below (not a directory, not the raw upstream
  checkpoint).
- **Import:**
  ```bash
  brain forecast import --hf <amazon/chronos-2 dir> --out chronos2.safetensors
  ```

## Running it

Chronos-2 has no `brain chronos2 ...` subcommand of its own — it is reached
through the shared `brain forecast` verb and the shared `forecast` D-Bus
action.

```bash
# backtest against statistical baselines before trusting it
brain forecast compare --chronos2 chronos2.safetensors --windows 24 --seed 1337 --html report.html

# start a resident forecast server
brain forecast serve --chronos2 chronos2.safetensors --socket /tmp/forecast.sock
```

D-Bus action `forecast`: input `context` (a raw f32 series, with its shape
in the request metadata), parameter `horizon` (forecast length), output
`forecast` (shape `[21, horizon]`, quantile levels included in the response
metadata).

```bash
BRAIN_CHRONOS2=/path/to/chronos2.safetensors dbus-run-session -- bash -c '
  brain serve --dbus &
  sleep 2
  python3 examples/forecast/forecast_client.py --model brain/chronos2 --horizon 64
'
```

## Options

- `compare`: `--windows <n>` (default 24), `--seed <n>` (default 1337),
  `--html <path>` to write a report.
- `serve`: `--socket <path>` or `--listen <addr>`.
- D-Bus `forecast` action: `horizon` (default 64).

## Hardware and limits

- Inference only today — no training, fine-tuning, or LoRA path.
- No HTTP route: reachable only via the `brain forecast` CLI and D-Bus.
- Runs on CPU or GPU, and is placed on an NPU automatically when one is
  available and budgeted for the request.
- Univariate forecasting only — known-future covariates and multi-series
  grouping are not part of the served path yet, and horizons much beyond
  1024 steps are not a supported use case.
