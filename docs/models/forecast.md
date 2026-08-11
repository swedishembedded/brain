# Time-series forecasting

Three foundation models — Chronos-2, FinCast, and Kronos — share one
contract: a numeric time-series context goes in, a probabilistic forecast
comes out. They all sit behind the same `brain forecast` CLI verb and the
same `forecast` D-Bus action; this page covers that shared surface. Each
model's own page has its exact model id, environment variable(s), and any
model-specific behavior:

- [Chronos-2](chronos2.md) — general-purpose probabilistic forecaster
- [FinCast](fincast.md) — financial forecaster, research/educational use only
- [Kronos](kronos.md) — OHLCV bar forecaster, the only one with CLI-reachable fine-tuning

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [x] |
| CLI (`brain do`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

Support varies by model — Kronos is the only one with a CLI-reachable
fine-tune (including LoRA); see the individual pages linked above for the
exact per-model breakdown.

## Getting the weights

Each model has its own id and weight variable — none of the three are
auto-fetched, so weights always have to be provided explicitly:

| Model | Model id | Weights |
|---|---|---|
| Chronos-2 | `brain/chronos2` | `BRAIN_CHRONOS2` |
| FinCast | `brain/fincast` | `BRAIN_FINCAST` |
| Kronos | `brain/kronos` | `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` |

See each model's page for its exact import command.

## Running it

`brain forecast` has four subcommands, shared across all three models:

- **`compare`** — run a backtest scenario battery against statistical
  baselines (and against any foundation models you point it at). This is
  the way to sanity-check a forecaster before trusting it in production: it
  fails if a model doesn't even beat a random-walk baseline.
  ```bash
  brain forecast compare --windows 24 --seed 1337 --html report.html \
    [--chronos2 <weights>] [--fincast <weights>] \
    [--kronos-tokenizer <dir> --kronos-decoder <dir>]
  ```
- **`serve`** — start a resident forecast server (stdio by default, or a
  Unix socket / TCP listener), loaded with whichever models' weight flags
  you pass:
  ```bash
  brain forecast serve --socket /tmp/forecast.sock --max-connections 64 \
    [--chronos2 <weights>] [--fincast <weights>] \
    [--kronos-tokenizer <dir> --kronos-decoder <dir>]
  ```
- **`import`** — convert an upstream checkpoint into the brain
  `.safetensors` format Chronos-2 and FinCast serve from (see their pages
  for the exact form).
- **`finetune`** — Kronos-only; see [its page](kronos.md) and
  [Fine-tuning a forecaster](../training/forecast-finetune.md).

Every resident exposes one `forecast` D-Bus action: input blob `context`
(a raw f32 series with its shape given in the request metadata), parameter
`horizon`, output blob `forecast` (shape and kind described in the response
metadata). It is gated on the corresponding model's weight variable(s)
being set. A shared reference client covers all three:

```bash
python3 examples/forecast/forecast_client.py --model brain/chronos2|brain/fincast|brain/kronos
```

See [`examples/forecast/README.md`](../../examples/forecast/README.md) for
full runnable invocations per model.

## Options

- `compare`: `--windows <n>` (default 24), `--seed <n>` (default 1337),
  `--html <path>`.
- `serve`: `--socket <path>` or `--listen <addr>`, `--max-connections <n>`
  (default 64).
- D-Bus `forecast` action: `horizon` (default 64), plus model-specific
  parameters described on each model's page.

## Hardware and limits

- None of the three models have an HTTP route today — only the `brain
  forecast` CLI and the `forecast` D-Bus action.
- Serving handles requests sequentially; there is no batched forward pass
  across multiple in-flight forecast requests yet.
- Chronos-2 and FinCast are placed on an NPU automatically when one is
  available and budgeted; Kronos runs its autoregressive rollout on CPU/GPU
  by default and is NPU-eligible via a cached rollout path.
