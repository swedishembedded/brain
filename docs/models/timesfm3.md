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
| LoRA fine-tune | [ ] |
| CLI (`brain <arch> <action>`) | [x] |
| HTTP API | [ ] |
| D-Bus | [x] (target-only - see below) |
| Batched serving | [ ] |

## Getting the weights

- **Model id:** `brain/timesfm3`.
- **Fetch:** `brain pull google/timesfm-3.0-pytorch` downloads the ungated,
  non-commercial-licensed checkpoint (~1.3 GB) into the model store. No
  wired caller sets `BRAIN_TIMESFM3` from that pull for you yet, unlike
  Kronos's own auto-fetch - point the variable at the result yourself.
- **Weights:** set `BRAIN_TIMESFM3` to EITHER the raw fetched checkpoint
  directory (`config.json` + `model.safetensors`, `brain pull`'s own output -
  `Timesfm3::load` detects and imports it on the fly) or a single brain
  `.safetensors` file produced by the explicit import step below.
- **Import (optional - only needed to bake the conversion in once instead of
  redoing it on every load):**
  ```bash
  brain pull google/timesfm-3.0-pytorch
  brain forecast import --timesfm3 <fetched dir> --out timesfm3.safetensors
  ```

## Running it

```bash
# one-command forecast + score + chart against an OHLCV CSV's close column
brain forecast predict --csv examples/forecast/synthetic_hourly.csv \
  --timesfm3 timesfm3.safetensors --horizon 32 --gnuplot chart.png

# backtest against statistical baselines
brain forecast compare --timesfm3 timesfm3.safetensors --windows 24 --seed 1337

# start a resident forecast server
brain forecast serve --timesfm3 timesfm3.safetensors --socket /tmp/forecast.sock
```

**Native multivariate forecasting** (the model's actual point) is reached
through the library API, not the CLI's single-series `predict`/D-Bus paths:
build a `forecast::Panel` with `Role::Target` (one or more target series),
`Role::PastCovariate` (measured, not known in advance) and `Role::KnownFuture`
(the covariate's future path is supplied) variates, then call
`timesfm3::Timesfm3Forecaster::forecast`. See
`crates/timesfm3/examples/cooling_loop.rs` for a complete worked example (an
industrial cooling loop: a fouling heat exchanger under an unmeasured,
schedule-driven load, forecast against a physics observer and a seasonal-naive
baseline) and `examples/forecast/cooling_loop.sh`/`.py` for the shell/served
variants.

D-Bus action `forecast`: input `context` (a raw f32 series, with its shape in
the request metadata), parameter `horizon`, output `forecast` (shape
`[horizon, 9]`, kind `quantiles_hq` - the model's 9 native quantiles,
horizon-major). This wire carries only ONE series, so a served request is
always target-only, even though the model itself is multivariate.

```bash
BRAIN_TIMESFM3=/path/to/timesfm3.safetensors dbus-run-session -- bash -c '
  brain serve --dbus &
  sleep 2
  python3 examples/forecast/forecast_client.py --model brain/timesfm3
'
```

## Options

- `predict`: `--timesfm3 <weights>` selects this model instead of Kronos;
  `--horizon`, `--context` (rounds down to a multiple of the checkpoint's
  32-step patch length), `--gnuplot <path>`.
- `compare`/`serve`: `--timesfm3 <weights>`, same flags as the other
  foundation models.
- D-Bus `forecast` action: `horizon` (default 64).

## Hardware and limits

- CPU/GPU only - no NPU export yet.
- No training/LoRA path yet - inference only.
- **Left-padding is not implemented**: the context length must already be a
  multiple of the checkpoint's `input_patch_len` (32 for the published
  checkpoint). A served or CLI request with an arbitrary-length series is
  truncated to its most recent patch-aligned tail rather than left-padded.
- No per-step missing-value (NaN) interpolation yet - every context value is
  treated as observed.
- Forecaster-level postprocessing implements quantile sorting and a
  positivity clamp; symmetric averaging, z-normalization and 32-variate
  chunking (needed to match the reference's own benchmark numbers exactly on
  panels wider than the model's 32-variate limit) are not implemented.
- The D-Bus/HTTP wire carries one series only - native multivariate/covariate
  forecasting needs the library API (see above), not the served path.
