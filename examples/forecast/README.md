# Time-series forecasting over D-Bus

Send a context series to `brain serve --dbus`, get a probabilistic forecast back —
end to end, over the generic `Run` method on `com.swedishembedded.Brain1`, using a
file descriptor for the bulk numeric data (no per-model protocol).

Three forecasting foundation models are served, each with one `forecast` action:

| model | env vars | input | output |
|---|---|---|---|
| `chronos2` | `BRAIN_CHRONOS2` | univariate series `[T]` | `[levels, horizon]` quantiles (21 levels) |
| `fincast` | `BRAIN_FINCAST` | univariate series `[T]` (+`freq`) | `[horizon, 1+levels]` — col 0 mean, then 9 quantiles |
| `kronos` | `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` | OHLCV bars `[T, feat]` (or a univariate `[T]` close, expanded server-side) | `[horizon, feat]` sample bars |

`chronos2` and `fincast` advertise an NPU footprint, so with an Intel NPU budgeted
the scheduler **places them on the NPU automatically** (`place::pick_device`); the
returned `device` field says where it ran. `kronos` runs on CPU/GPU (its two-graph
autoregressive NPU rollout is a follow-up).

## Run it

Serve on a private session bus with a model's weights, then call the client:

```bash
# chronos2 (universal forecasting), CPU backend:
BRAIN_CHRONOS2=/path/to/chronos2.weights \
  dbus-run-session -- bash -c '
    brain serve --dbus --device cpu & sleep 2
    python3 examples/forecast/forecast_client.py --model chronos2 --horizon 64
  '
```

Let the scheduler pick the device (omit `--device` to use every device, NPU
included):

```bash
BRAIN_CHRONOS2=chronos2.weights BRAIN_FINCAST=fincast.weights \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/forecast/forecast_client.py --model chronos2   # device: npu, if present
    python3 examples/forecast/forecast_client.py --model fincast --freq 0
  '
```

Kronos needs both checkpoint dirs, but you no longer have to find them:
`brain forecast predict` auto-fetches `NeoQuasar/Kronos-base` and
`NeoQuasar/Kronos-Tokenizer-base` and exports both variables. Set them by hand
only to override:

```bash
BRAIN_KRONOS_TOKENIZER=kronos-tokenizer-base BRAIN_KRONOS_DECODER=kronos-small \
  dbus-run-session -- bash -c '
    brain serve --dbus --device cpu & sleep 2
    python3 examples/forecast/forecast_client.py --model kronos --horizon 32
  '
```

## `synthetic_hourly.csv` - the example series

`synthetic_hourly.csv` is 720 hourly OHLCV bars (30 days, 24/7) that
`brain forecast predict` reads directly:

```bash
brain forecast predict --csv examples/forecast/synthetic_hourly.csv \
  --horizon 24 --samples 16 --origins 4 --gnuplot chart.png
```

It is synthetic on purpose, and not trivially so. The latent log price is a
deterministic backbone (a slow drift plus two-harmonic intraday and weekly
profiles) plus a **mean-reverting AR(1) residual**, so the h-step conditional
mean is known in closed form and everything left over is irreducible noise -
which means "how good is this forecast" has an arithmetic answer rather than an
aesthetic one. Bars are built from an intrabar Brownian bridge, so the OHLC
invariants hold by construction and the validator in
`crates/forecast/src/csv.rs` is a real test of the data rather than a
formality.

Regenerate it (or make a different one) with
`tools/forecast/make_synthetic_ohlcv.py`, which also prints the oracle
(conditional-mean) error and the two naive baselines - the numbers that say
what "good" means on the series it just wrote:

```bash
python3 tools/forecast/make_synthetic_ohlcv.py --out examples/forecast/synthetic_hourly.csv --bars 720 --seed 7
```

Feed a real series instead of the synthetic sine-with-trend:

```bash
python3 examples/forecast/forecast_client.py --model chronos2 --series my_series.txt
```

## Dependencies

- `jeepney` — D-Bus with fd passing (the only requirement).

## How it works

```
context series ──f32-LE──▶ sealed memfd
                              │  (client)
                              ▼
   Run("chronos2","forecast", {horizon}, in_fds={context: fd}) ─┐  D-Bus, fd via SCM_RIGHTS
                                                                 ▼  (server, crates/dbus)
                    residency::Executor  ── schedules the forecast job on
                              │                CPU / GPU / NPU (auto)
                              ▼
              forecast blob (f32-LE + {shape,kind,levels}) ──memfd──▶ client
                              │
                              ▼ read_fd + struct.unpack
                   quantile / mean / sample path
```

The context goes in as one input blob fd (`{"media":"bytes","shape":[T]}`); scalar
knobs (`horizon`, `freq`) ride in the params JSON. The forecast comes back as an
output blob fd whose meta carries `{shape, kind, levels}`, so the client knows how
to unpack it. Concurrent forecasts from many clients share the one `Executor` and
are scheduled across devices — see `crates/residency`.
Nothing here is forecasting-specific in the transport: it is the same path
`examples/embedding` uses for LFM embeddings.
