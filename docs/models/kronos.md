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

Nothing to do: `brain forecast predict` auto-fetches on first use, like every
other model in the README's Quick start.

Kronos is **one model published as two upstream repos** - the BSQ tokenizer
and the decoder are separate, with no combined release - so auto-fetch pulls
both and points one environment variable at each:

| Variable | Repo | Role |
|---|---|---|
| `BRAIN_KRONOS_DECODER` | `NeoQuasar/Kronos-base` (391 MB) | the decoder |
| `BRAIN_KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` (16 MB) | the BSQ tokenizer |

Set either variable yourself and auto-fetch leaves it alone, so a local
checkout or a `.safetensors` fine-tune checkpoint from `brain forecast
finetune` still wins. `BRAIN_AUTO_FETCH=0` disables the fetch entirely.

- **Model id:** `brain/kronos` - a reserved `brain/` id for serving; the
  fetchable upstream references are the two `NeoQuasar/...` repos above.
- **Weights:** two directories, both required, no import step - they load
  directly. `BRAIN_KRONOS_DECODER` also accepts a `.safetensors` fine-tune
  checkpoint file in place of a directory.

## Running it

Kronos has no `brain kronos ...` subcommand of its own - it is reached
through the shared `brain forecast` verb and the shared `forecast` D-Bus
action.

```bash
# one command, CSV in, scored forecast (and a chart) out
brain forecast predict --csv examples/forecast/synthetic_hourly.csv --horizon 6 --samples 16 --origins 16 --gnuplot chart.png

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

- `predict`: `--csv <file>` (required, `timestamp,open,high,low,close,volume`),
  `--horizon <n>` (default 48), `--context <n>` (default: `512 - horizon`, the
  longest window that keeps the rollout exact against the reference - see
  Hardware and limits), `--samples <n>` (default 1 = the deterministic modal
  rollout; more draws real trajectories and adds a decile-to-ninth-decile
  uncertainty band to the chart),
  `--origins <n>` (default 1; score at N disjoint held-out windows and average,
  because one origin is a draw and not a measurement), `--season <n>` (default
  24, the seasonal-naive baseline's period), `--seed`, `--gnuplot <png>`,
  `--top-p <p>` / `--temperature <t>` (the sampler's two width knobs; default to
  the reference `KronosPredictor` values 0.9 and 1.0, and also readable as
  `BRAIN_KRONOS_TOP_P` / `BRAIN_KRONOS_TEMPERATURE`).
  The CSV is validated structurally and semantically at entry - column order,
  ragged rows, non-finite values, non-monotonic timestamps, non-positive
  prices and the OHLC invariants - and every rejection names the file line.
  `--gnuplot` needs the `gnuplot` CLI on `PATH`; the command says so and exits
  before loading any weights if it is missing.
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
- `BRAIN_KRONOS_ARGMAX=1` forces the deterministic modal rollout (argmax over
  the token distribution) instead of nucleus sampling: one stable path,
  reproducible run to run. `brain forecast predict --samples 1` sets it.
- Kronos is a candlestick model, not a seasonal decomposition: its skill is
  concentrated at short horizons. Score it at several rolling origins
  (`--origins`) before believing a single window's number.
- **The predictive band is over-confident, and worsens with lead time.** The
  numbers below are interval COVERAGE - what fraction of held-out bars the
  nominal 10-90% band actually contains - not throughput; the warning IS that <!-- perf-number: names the band being scored, not a measurement -->
  the nominal percentage and the empirical one disagree, so each figure is the
  claim rather than an incidental measurement.
  On the committed example series that band covers 60% of held-out bars <!-- perf-number: prediction-interval coverage, not throughput -->
  at a 6-bar horizon, 44% at 12 and 30% at 24; on a real 5-minute equity <!-- perf-number: prediction-interval coverage, not throughput -->
  series (at the nominal 512-bar window), 45% at 12 and 20% at 24. This is <!-- perf-number: prediction-interval coverage, not throughput -->
  the checkpoint, not the port - upstream's own `KronosPredictor` measures the
  same on the same input - and it is not nucleus truncation either: relaxing
  `--top-p` all the way to 1.0 only moves real-data coverage from 20% to 38%, <!-- perf-number: prediction-interval coverage, not throughput -->
  at a worse CRPS. Treat the band as a relative sharpness signal, not as an 80% <!-- perf-number: prediction-interval coverage, not throughput -->
  interval, and calibrate it against your own held-out data before sizing
  anything on it.
- **Skill is concentrated at short horizons, and it is a CRPS skill, not a
  point-accuracy one.** On the example series, over 16 disjoint origins: at 6
  bars kronos beats persistence by 18% on CRPS (and loses to it on MAE); at 12
  bars it loses by 50%; at 24 bars, by 74%.
- On a single series, point accuracy is a tie with persistence at best: the
  model's own published metric is cross-sectional rank IC across a universe of
  instruments (`crates/kronos/tests/rankic_eval.rs`), which is a different
  question from level accuracy on one name.
- The KV-cached rollout (`--samples`, the D-Bus path, the resident server) is
  **exact against the upstream implementation only while `context + horizon
  <= 512`**, the model's attention window. Beyond that the upstream rollout
  re-runs the whole 512-bar window from an origin one bar later at every step,
  which no K/V cache reproduces - and this checkpoint is unusually sensitive to
  that shift: two otherwise identical un-cached runs whose window origin
  differs by a single bar disagree by ~1e-1 relative in the final token logits,
  enough to change which token is sampled. **`brain forecast predict` therefore
  defaults `--context` to `512 - horizon`, not 512**, so the shipped path is the
  exact one: at `--context 500 --horizon 12` brain's 128-sample cloud and the
  reference's agree at every step (largest |z| 0.9 on the per-step means),
  while at `--context 512` the same comparison drifts apart to |z| 6.0 and the
  sampled band is ~15% narrower. Ask for the longer window explicitly if you
  want it; the command says on stderr that it is then an approximation.
  `make forecast/parity` gates both regimes.
