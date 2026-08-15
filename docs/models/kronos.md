# Kronos

Kronos forecasts OHLCV (Open/High/Low/Close/Volume) bars - a candlestick
foundation model that tokenizes bar history and autoregressively predicts
future bars. Reach for it when you need bar-level forecasts (not just a
close-price series) for a tradable instrument, or when you want to
fine-tune a forecaster on your own OHLCV history: Kronos is the only one of
brain's three forecasting models with CLI-reachable fine-tuning.

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [x] |
| CLI (`brain <arch> <action>`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

- **Model id:** `brain/kronos` - a reserved `brain/` id, never auto-fetched.
- **Weights:** two directories, both required, no import step - they load
  directly:
  - `BRAIN_KRONOS_TOKENIZER` - the Kronos tokenizer model directory.
  - `BRAIN_KRONOS_DECODER` - the Kronos decoder model directory (or a
    `.safetensors` fine-tune checkpoint produced by `brain forecast
    finetune`, see below).

## Running it

Kronos has no `brain kronos ...` subcommand of its own - it is reached
through the shared `brain forecast` verb and the shared `forecast` D-Bus
action.

```bash
# backtest against statistical baselines before trusting it
brain forecast compare --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> --windows 24 --seed 1337

# start a resident forecast server
brain forecast serve --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> --socket /tmp/forecast.sock
```

D-Bus action `forecast`: input `context` (OHLCV bars, or a plain close-price
series which is expanded to bars automatically), optional calendar-stamp
inputs, parameters `horizon`, `temperature`, `argmax`, `seed`, `samples`,
and an optional per-request `checkpoint` override, output `forecast`
(sampled future bars).

```bash
BRAIN_KRONOS_TOKENIZER=<tok-dir> BRAIN_KRONOS_DECODER=<dec-dir> \
  dbus-run-session -- bash -c '
    brain serve --dbus &
    sleep 2
    python3 examples/forecast/forecast_client.py --model brain/kronos --horizon 32
  '
```

## Options

- `compare`: `--windows <n>` (default 24), `--seed <n>` (default 1337).
- `serve`: `--socket <path>` or `--listen <addr>`.
- D-Bus `forecast` action: `horizon` (default 64), `temperature` (default
  1.0), `argmax` (default true), `seed`, `samples` (default 1), `checkpoint`
  (override the decoder path for a single request).

Fine-tuning (full fine-tune and LoRA) is available via `brain forecast
finetune` - see [Fine-tuning a forecaster](../training/forecast-finetune.md)
for the full how-to.

## Hardware and limits

- QLoRA is not supported - only full fine-tune and LoRA.
- No HTTP route: reachable only via the `brain forecast` CLI and D-Bus.
- Runs on CPU or GPU by default, and is NPU-eligible via a cached
  autoregressive rollout.
