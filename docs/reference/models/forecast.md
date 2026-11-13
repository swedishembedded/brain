# Time-series forecasting

Four foundation models - Chronos-2, FinCast, Kronos, and TimesFM-3 - share
one contract: a numeric time-series context goes in, a probabilistic forecast
comes out. They all serve the same two actions, `forecast` and `predict`, as
`brain <model> <action>` and over `brain serve`; this page covers that shared
surface. Each model's own page has its exact model id, weights and any
model-specific behavior:

- [Chronos-2](chronos2.md) - general-purpose probabilistic forecaster
- [FinCast](fincast.md) - financial forecaster, research/educational use only
- [Kronos](kronos.md) - OHLCV bar forecaster, with a gated fine-tune action
- [TimesFM-3](timesfm3.md) - natively multivariate forecaster (target +
  covariates in one call), with a gated fine-tune action

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

Support varies by model - Kronos and TimesFM-3 have a fine-tune action
(including LoRA); see the individual pages linked above for the exact
per-model breakdown.

## Getting the weights

Every model's checkpoint is resolved from the model store, by what the
checkpoint is (its own config or tensors), never by its name. TimesFM-3 and
Kronos are fetched with `brain pull`; Chronos-2 and FinCast are placed in the
store by the operator. `--weights PATH` names a checkpoint for one run
(Kronos: `--tokenizer PATH` and `--decoder PATH`), and `brain serve --pin
ARCH.ROLE=PATH` pins one for a server. TimesFM-3's weights carry a
non-commercial license - see its own page.

| Model | Model id | Roles |
|---|---|---|
| Chronos-2 | `brain/chronos2` | `weights` |
| FinCast | `brain/fincast` | `weights` |
| Kronos | `brain/kronos` | `tokenizer` + `decoder` |
| TimesFM-3 | `brain/timesfm3` | `weights` |

See each model's page for its exact import command.

## Running it

Each model's actions are `brain <model> <action>`, and `<action> --help`
lists every parameter:

- **`forecast`** - a context series in, a forecast tensor out (raw f32
  little-endian blobs, shape in the blob metadata). Also the D-Bus action:
  input blob `context`, parameter `horizon`, output blob `forecast` (shape and
  kind described in the response metadata).
- **`predict`** - an OHLCV CSV in, the model scored against persistence, drift
  and seasonal naive on the last `horizon` bars it was not shown (CRPS, MAE,
  pinball, band coverage, averaged over `origins` disjoint windows), with the
  evidence chart when `chart` is set. This is the way to sanity-check a
  forecaster on your own data before trusting it.
  ```bash
  brain kronos predict --in csv=ohlcv.csv --horizon 48 --origins 5 --seed $RANDOM
  ```
- **`import`** - Chronos-2, FinCast and TimesFM-3: convert an upstream
  checkpoint into the brain `.safetensors` container (see their pages for the
  exact form).
- **`finetune`** - Kronos and TimesFM-3. Same parameters, same promotion-gate
  line either way: the checkpoint is written only if the fine-tune beats the
  base on a held-out, embargoed split. One progress line per optimizer step,
  cancellable, and the `seed` is echoed. See [Kronos](kronos.md),
  [TimesFM-3](timesfm3.md) and
  [Fine-tuning a forecaster](../../guides/training/forecast-finetune.md).

  TimesFM-3 additionally enforces `horizon <= stitch_extract_len` (64 for
  the published checkpoint), because at that horizon the forecast is a single
  patch and the output-side map is affine; and every checkpoint it writes
  carries `timesfm-non-commercial-license-v1.0` plus the upstream it derives
  from, which makes it non-publishable by
  `brain serve`'s model discovery. A fine-tune of those weights is a
  derivative of them.

A shared reference client covers all four over D-Bus:

```bash
python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/chronos2|brain/fincast|brain/kronos|brain/timesfm3
```

See [`samples/python/forecast/forecast-client/README.md`](../../../samples/python/forecast/forecast-client/README.md) for
full runnable invocations per model.

### The application part

Each model has a sample, `samples/models/<model>`, with what is an
application rather than the served model:

- **`compare`** - the model against the statistical baselines over the
  scenario battery (a model x scenario x metric report, HTML optional). It is
  a hard gate: the action fails if a model falsely beats the naive forecast on
  the random walk.
- **`serve`** - the JSONL forecasting server (the baselines plus the model)
  over stdio or an owner-only Unix socket, which keeps the model warm across a
  batch job (`brain-py`'s `BrainStdio.connect`). The socket carries no
  credential, so it has no network listener; a remote caller uses the
  `forecast` action of `brain serve`, which is key-gated. `brain serve
  --stdio` serves the statistical baselines over the same protocol.
- **`npu-forecast`** - Chronos-2, FinCast and Kronos: the transformer core (or
  both Kronos decoder graphs) on the Intel NPU through OpenVINO, with `verify`
  to compare against the device path.

## Options

- `forecast`: `horizon` (default 64), plus model-specific parameters described
  on each model's page.
- `predict`: `horizon` (default 48), `context`, `samples`, `origins`, `seed`,
  `season`, `item`, `freq`, `chart`.
- `finetune`: see the [fine-tuning guide](../../guides/training/forecast-finetune.md).

## Hardware and limits

- None of the four models have an HTTP route today - only `brain <model>
  <action>` and the D-Bus actions.
- Serving handles requests sequentially; there is no batched forward pass
  across multiple in-flight forecast requests yet.
- Chronos-2 and FinCast `forecast` is placed on an NPU automatically when one is
  available and budgeted; Kronos runs its autoregressive rollout on CPU/GPU
  by default and is NPU-eligible via a cached rollout path; TimesFM-3 runs on
  CPU/GPU only.
- `predict` scores one series (the CSV's close); TimesFM-3's native
  multivariate/covariate forecasting is the `forecast` action's wire or the
  library API (see its own page).
