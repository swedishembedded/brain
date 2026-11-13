# Kronos

Kronos forecasts OHLCV (Open/High/Low/Close/Volume) bars - a candlestick
foundation model that tokenizes bar history and autoregressively predicts
future bars. Reach for it when you need bar-level forecasts (not just a
close-price series) for a tradable instrument, or when you want to
fine-tune a forecaster on your own OHLCV history (Kronos and TimesFM-3 are
the two forecasting models with a fine-tune action).

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [x] |
| CLI (`brain <arch> <action>`) | [x] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

`brain pull NeoQuasar/Kronos-base` and `brain pull NeoQuasar/Kronos-Tokenizer-base`
fetch the two repos; nothing else is needed, the model store finds them.

Kronos is **one model published as two upstream repos** - the BSQ tokenizer
and the decoder are separate, with no combined release - so the store holds
two directories and tells them apart by their own `config.json` content, not
their names:

| Role | Repo | Flag for one run | Server pin |
|---|---|---|---|
| `decoder` | `NeoQuasar/Kronos-base` (391 MB) | `--decoder PATH` | `--pin kronos.decoder=PATH` |
| `tokenizer` | `NeoQuasar/Kronos-Tokenizer-base` (16 MB) | `--tokenizer PATH` | `--pin kronos.tokenizer=PATH` |

A local checkout, or a `.safetensors` fine-tune checkpoint from `brain kronos
finetune`, wins when named with the flag or the pin. A store with several
candidates for a role refuses to guess and lists them.

- **Model id:** `brain/kronos` - a reserved `brain/` id for serving; the
  fetchable upstream references are the two `NeoQuasar/...` repos above.
- **Weights:** two checkpoints, both required, no import step - they load
  directly. The decoder also accepts a `.safetensors` fine-tune checkpoint
  file in place of a directory.

## Running it

```bash
# one command, CSV in, scored forecast (and a chart) out
brain kronos predict --in csv=crates/kronos/tests/data/synthetic_hourly.csv --horizon 6 --samples 16 --origins 16 --seed $RANDOM --chart true --out chart=chart.png

# a forecast of a context series (raw f32 little-endian; OHLCV bars, or a close series)
brain kronos forecast --in context=bars.f32 --horizon 48 --seed $RANDOM --out forecast=out.f32

# fine-tune on a universe of OHLCV CSVs (the gate decides whether it is kept)
brain kronos finetune --data <csv-dir> --save kronos-ft.safetensors --seed $RANDOM
```

The scenario battery against the statistical baselines, the autoregressive
rollout on the Intel NPU and the JSONL forecasting server are the application
part: [`samples/models/kronos`](../../../samples/models/kronos/README.md)
(`compare`, `npu-forecast`, `serve`).

D-Bus action `forecast`: input `context` (OHLCV bars, or a plain close-price
series which is expanded to bars automatically), optional calendar-stamp
inputs, parameters `horizon`, `temperature`, `top_p`, `argmax`, `seed`,
`samples`, and an optional per-request `checkpoint` override, output `forecast`
(sampled future bars). D-Bus action `predict`: the CSV as the input blob `csv`.

```bash
dbus-run-session -- bash -c '
  brain serve --dbus --pin kronos.tokenizer=<tok-dir> --pin kronos.decoder=<dec-dir> &
  sleep 2
  python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/kronos --horizon 32
'
```

## Options

- `predict`: input `csv` (`timestamp,open,high,low,close,volume`, required),
  `horizon` (default 48), `context` (default: `512 - horizon`, the
  longest window that keeps the rollout exact against the reference - see
  Hardware and limits), `samples` (default 1 = the deterministic modal
  rollout; more draws real trajectories and adds a decile-to-ninth-decile
  uncertainty band to the chart),
  `origins` (default 1; score at N disjoint held-out windows and average,
  because one origin is a draw and not a measurement), `season` (default
  24, the seasonal-naive baseline's period), `seed` (drawn and echoed when
  absent), `chart` (writes the PNG as the `chart` output), `top_p` / `temperature`
  (the sampler's two width knobs; default to the reference `KronosPredictor`
  values 0.9 and 1.0), `checkpoint` (a decoder to score instead of the
  resolved one, e.g. a fine-tune).
  The CSV is validated structurally and semantically at entry - column order,
  ragged rows, non-finite values, non-monotonic timestamps, non-positive
  prices and the OHLC invariants - and every rejection names the file line.
  `chart` needs the `gnuplot` CLI on `PATH`; the action says so and fails
  before running the model if it is missing.
- `forecast`: `horizon` (default 64), `temperature` (default 1.0), `top_p`
  (default 1.0, off), `argmax` (default true), `seed`, `samples` (default 1),
  `checkpoint` (override the decoder path for a single request).
- `finetune`: `data` (a directory of `<TICKER>.csv`), `save`, `holdout_data`,
  `context`, `horizon`, `epochs`, `lr`, `lora`, `embargo`, `batch`,
  `skip_invalid`, `seed`, `activations`.

Fine-tuning (full fine-tune and LoRA) is available via `brain kronos
finetune` - see [Fine-tuning a forecaster](../../guides/training/forecast-finetune.md)
for the full how-to.

## Hardware and limits

- QLoRA is not supported - only full fine-tune and LoRA.
- No HTTP route: reachable through `brain kronos <action>` and D-Bus.
- Runs on CPU or GPU by default, and is NPU-eligible via a cached
  autoregressive rollout.
- `argmax` forces the deterministic modal rollout (argmax over the token
  distribution) instead of nucleus sampling: one stable path, reproducible run
  to run. `predict` with `samples` 1 uses it.
- Kronos is a candlestick model, not a seasonal decomposition: its skill is
  concentrated at short horizons. Score it at several rolling origins
  (`origins`) before believing a single window's number.
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
  `top_p` all the way to 1.0 only moves real-data coverage from 20% to 38%, <!-- perf-number: prediction-interval coverage, not throughput -->
  at a worse CRPS. Treat the band as a relative sharpness signal, not as an 80% <!-- perf-number: prediction-interval coverage, not throughput -->
  interval, and calibrate it against your own held-out data before sizing
  anything on it.
- **Skill is concentrated at short horizons, and it is a CRPS skill, not a
  point-accuracy one.** On the example series, over 16 disjoint origins: at 6
  bars kronos beats persistence by 18% on CRPS (and loses to it on MAE); at 12
  bars it loses by 50%; at 24 bars, by 74%.
- On a single series, point accuracy is a tie with persistence at best: the
  model's own published metric is cross-sectional rank IC across a universe of
  instruments (`crates/kronos/tests/it/rankic_eval.rs`), which is a different
  question from level accuracy on one name.
- The KV-cached rollout (`samples`, the D-Bus path, the resident server) is
  **exact against the upstream implementation only while `context + horizon
  <= 512`**, the model's attention window. Beyond that the upstream rollout
  re-runs the whole 512-bar window from an origin one bar later at every step,
  which no K/V cache reproduces - and this checkpoint is unusually sensitive to
  that shift: two otherwise identical un-cached runs whose window origin
  differs by a single bar disagree by ~1e-1 relative in the final token logits,
  enough to change which token is sampled. **`predict` therefore
  defaults `context` to `512 - horizon`, not 512**, so the shipped path is the
  exact one: at `context` 500 and `horizon` 12 brain's 128-sample cloud and the
  reference's agree at every step (largest |z| 0.9 on the per-step means),
  while at `context` 512 the same comparison drifts apart to |z| 6.0 and the
  sampled band is ~15% narrower. Ask for the longer window explicitly if you
  want it; the action says so in its progress output that it is then an approximation.
  `make test/parity/forecast` gates both regimes.
