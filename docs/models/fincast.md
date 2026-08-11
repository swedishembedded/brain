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
| CLI (`brain do`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

- **Model id:** `brain/fincast` — a reserved `brain/` id, never auto-fetched.
- **Weights:** set `BRAIN_FINCAST` to a single brain `.safetensors` file
  produced by the import step below (not a directory, not the raw upstream
  checkpoint).
- **Import:** the raw upstream checkpoint must first be converted, then
  imported into brain's format:
  ```bash
  python3 tools/convert/fincast_convert.py <v1.pth> <fincast.safetensors>
  brain forecast import --fincast <fincast.safetensors> --out fincast.safetensors
  ```

## Running it

FinCast has no `brain fincast ...` subcommand of its own — it is reached
through the shared `brain forecast` verb and the shared `forecast` D-Bus
action.

```bash
# backtest against statistical baselines before trusting it
brain forecast compare --fincast fincast.safetensors --windows 24 --seed 1337

# start a resident forecast server
brain forecast serve --fincast fincast.safetensors --socket /tmp/forecast.sock
```

D-Bus action `forecast`: input `context` (a raw f32 series, with its shape
in the request metadata), parameters `horizon` (forecast length) and `freq`
(0 = daily, 1 = weekly, 2 = monthly), output `forecast` (shape
`[horizon, 10]`, column 0 is the mean, the remaining 9 are quantiles).

```bash
BRAIN_FINCAST=/path/to/fincast.safetensors dbus-run-session -- bash -c '
  brain serve --dbus &
  sleep 2
  python3 examples/forecast/forecast_client.py --model brain/fincast --freq 0
'
```

## Options

- `compare`: `--windows <n>` (default 24), `--seed <n>` (default 1337).
- `serve`: `--socket <path>` or `--listen <addr>`.
- D-Bus `forecast` action: `horizon` (default 64), `freq` (default 0).

## Hardware and limits

- Reference weights are for research and educational use only — not
  financial advice; see the license note above.
- Inference only today — no fine-tuning or LoRA path reachable from the
  CLI.
- No HTTP route: reachable only via the `brain forecast` CLI and D-Bus.
- Runs on CPU or GPU, and is placed on an NPU automatically when one is
  available and budgeted for the request.
