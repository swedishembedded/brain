# TimesFM-3

TimesFM-3 is Google's foundation forecasting model: a stacked mixing
transformer with BOTH sequence attention (over time, causal) and cross-variate
attention (over variates, non-causal), CPM iterative RevIN, linear detrending
and forecast stitching. Unlike Chronos-2/FinCast/Kronos, it is **natively
multivariate**: several target series and their covariates all attend to each
other in one forward pass, rather than one series per call. Reach for it when
your forecasting problem genuinely has more than one series that inform each
other - a target plus measured (past-only) and scheduled/forecast
(known-future) covariates.

> **License note:** the source (this port, and Google's own reference) is
> Apache-2.0. The **3.0 pretrained weights** are released under a separate
> `timesfm-non-commercial-license-v1.0`: non-commercial, non-production use
> only, and the checkpoint (or a derivative of it) may never be redistributed.
> Do not use TimesFM-3's real weights in a commercial deployment, and never
> vendor or re-host the checkpoint. See `google/timesfm-3.0-pytorch`'s own
> `LICENSE` for the full terms.

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [x] |
| CLI (`brain <arch> <action>`) | [x] |
| HTTP API | [ ] |
| D-Bus | [x] (native multivariate + covariates - see below) |
| Batched serving | [x] (`run_batch`; a panel's own items also batch internally) |

## Getting the weights

- **Model id:** `brain/timesfm3`.
- **Fetch:** `brain pull google/timesfm-3.0-pytorch` downloads the ungated,
  non-commercial-licensed checkpoint (~1.3 GB) into the model store, which
  finds it with no further step.
- **Weights:** the store resolves EITHER the raw fetched checkpoint directory
  (`config.json` + `model.safetensors`, `brain pull`'s own output -
  `Timesfm3::load` detects and imports it on the fly) or a single brain
  `.safetensors` file (what `import` and `finetune` write). `--weights PATH`
  names one for a single run, `brain serve --pin timesfm3.weights=PATH` for a
  server.
- **Import (optional - only needed to bake the conversion in once instead of
  redoing it on every load):**
  ```bash
  brain pull google/timesfm-3.0-pytorch
  brain timesfm3 import --source <fetched dir> --dest timesfm3.safetensors
  ```

## Running it

```bash
# one-command forecast + score + chart against an OHLCV CSV's close column
brain timesfm3 predict --in csv=crates/kronos/tests/data/synthetic_hourly.csv \
  --horizon 32 --chart true --out chart=chart.png

# fine-tune on a universe of OHLCV CSVs (the gate decides whether it is kept)
brain timesfm3 finetune --data <csv-dir> --save timesfm3-ft.safetensors --seed $RANDOM
```

The scenario battery against the statistical baselines and the JSONL
forecasting server are the application part:
[`samples/models/timesfm3`](../../../samples/models/timesfm3/README.md)
(`compare`, `serve`).

**Native multivariate forecasting** (the model's actual point) is reached
through the `forecast` action's wire (below) or the library API; `predict` is
the single-series CSV evaluation. With the library API:
build a `forecast::Panel` with `Role::Target` (one or more target series),
`Role::PastCovariate` (measured, not known in advance) and `Role::KnownFuture`
(the covariate's future path is supplied) variates, then call
`timesfm3::Timesfm3Forecaster::forecast`. See
`crates/timesfm3/examples/cooling_loop.rs` for a complete worked example (an
industrial cooling loop: a fouling heat exchanger under an unmeasured,
schedule-driven load, forecast against a physics observer and a seasonal-naive
baseline) and `samples/shell/forecast/cooling-loop/cooling_loop.sh`/`.py` for the shell/served
variants.

D-Bus action `forecast`: input `context`, raw f32 with its shape in the
request metadata - `[T]` for one target (the original wire shape) or
`[num_target, T]` for the model's native multivariate capability, reachable
over this wire directly, not only the library API. Optional inputs
`past_covariates` (`[T]`/`[P,T]`) and `known_future` (`[T+horizon]`/
`[K,T+horizon]`) add covariate variates; optional `observed` (same shape as
`context`) marks missing steps. Output `forecast`: `[horizon, 9]`, kind
`quantiles_hq`, for one target (byte-identical to the original single-series
contract); `[num_target, horizon, 9]`, kind `quantiles_thq`, plus a `names`
meta array, for more than one.

```bash
dbus-run-session -- bash -c '
  brain serve --dbus --pin timesfm3.weights=/path/to/timesfm3.safetensors &
  sleep 2
  python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/timesfm3
'
```

## Options

- `forecast`: `horizon` (default 64).
- `predict`: input `csv`; `horizon` (default 48), `context` (the full
  requested length is used - the forecaster left-pads to the checkpoint's
  32-step patch boundary itself, it no longer rounds down and drops history),
  `samples`, `origins`, `seed`, `season`, `item`, `freq`, `chart`. The model
  is not stochastic, so `samples` changes nothing for it.
- `finetune`: `data`, `save`, `holdout_data`, `context`, `horizon` (at most the
  checkpoint's `stitch_extract_len`, 64 for the published one), `epochs`, `lr`,
  `lora`, `embargo`, `batch`, `skip_invalid`, `seed`, `activations`. The
  license notice is the first progress line, and a promoted checkpoint carries
  `timesfm-non-commercial-license-v1.0` plus the upstream it derives from.
- `import`: `source`, `dest`.

## Hardware and limits

- CPU/GPU only - no NPU export yet.
- Fine-tuning (full and LoRA) is the `finetune` action; training from scratch is not offered.
- **Left-padding and per-step missing values are handled by one mechanism**:
  a non-finite value in a variate's data (or a `0.0` in its `observed` mask)
  is treated as missing, excluded from RevIN/detrend statistics, and zeroed
  before reaching the transformer. `Timesfm3Forecaster::forecast` left-pads
  any context to the checkpoint's own `input_patch_len` boundary this way, so
  an arbitrary-length series is used in full rather than truncated. There is
  no interior-NaN *interpolation* - an interior gap is masked through
  (excluded from the running stats it would otherwise contribute to), not
  filled in from neighboring values.
- Forecaster-level postprocessing implements quantile sorting and a
  positivity clamp; symmetric averaging, z-normalization and 32-variate
  chunking (needed to match the reference's own benchmark numbers exactly on
  panels wider than the model's 32-variate limit) are not implemented.
- `Timesfm3Forecaster::forecast` batches a panel's own items internally
  (grouped by padded context length, padded to a shared variate count where
  they differ) and `Timesfm3Instance::run_batch` batches queued D-Bus/JSONL
  requests the same way, grouped by horizon - both verified bit-identical to
  running each item/request alone, not just close. No measured throughput
  number is published here: it needs the real checkpoint on hardware not
  shared with other work, neither of which was available where this was
  written (see `.agents/roadmap/timesfm3.md`).
