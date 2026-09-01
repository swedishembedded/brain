# Time-series forecasting over D-Bus

Send a context series to `brain serve --dbus`, get a probabilistic forecast back —
end to end, over the generic `Run` method on `com.swedishembedded.Brain1`, using a
file descriptor for the bulk numeric data (no per-model protocol).

Four forecasting foundation models are served, each with one `forecast` action:

| model | env vars | input | output |
|---|---|---|---|
| `chronos2` | `BRAIN_CHRONOS2` | univariate series `[T]` | `[levels, horizon]` quantiles (21 levels) |
| `fincast` | `BRAIN_FINCAST` | univariate series `[T]` (+`freq`) | `[horizon, 1+levels]` — col 0 mean, then 9 quantiles |
| `kronos` | `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` | OHLCV bars `[T, feat]` (or a univariate `[T]` close, expanded server-side) | `[horizon, feat]` sample bars |
| `timesfm3` | `BRAIN_TIMESFM3` | univariate series `[T]` over this wire (natively multivariate via the library API - see below) | `[horizon, 9]` quantiles |

`chronos2` and `fincast` advertise an NPU footprint, so with an Intel NPU budgeted
the scheduler **places them on the NPU automatically** (`place::pick_device`); the
returned `device` field says where it ran. `kronos` and `timesfm3` run on CPU/GPU
(their NPU paths are a follow-up).

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
  --horizon 6 --samples 16 --origins 16 --gnuplot chart.png
```

It is synthetic on purpose, and built to be **in distribution** for a model
trained on real market bars rather than easy to forecast. The log price is a
near-unit-root random walk with a small drift and a mild pull back to it; its
innovations follow a **GARCH(1,1)** with fat (Student-t) tails, so volatility
clusters the way a real tape's does; volume peaks at the session open and close
and scales with the bar's own move. There is deliberately no periodic component
in the LEVEL: a hard daily cycle makes a seasonal-naive baseline unbeatable and
turns the demo into a test of seasonal decomposition, which is not what a
candlestick model is. Bars come from an intrabar Brownian bridge scaled to each
bar's own volatility, so the OHLC invariants hold by construction (the
validator in `crates/forecast/src/csv.rs` is a real test of the data rather
than a formality) and mean(high-low)/sd(return) lands at 1.4, which is what two
real 5-minute equity series measure.

Regenerate it (or make a different one) with
`tools/forecast/make_synthetic_ohlcv.py`, which also prints the series'
statistical fingerprint and the oracle (exact conditional-mean) error beside
three naive baselines - the numbers that say what "good" means on the series it
just wrote:

```bash
python3 tools/forecast/make_synthetic_ohlcv.py --out examples/forecast/synthetic_hourly.csv --bars 720 --seed 18
```

Feed a real series instead of the synthetic one:

```bash
python3 examples/forecast/forecast_client.py --model chronos2 --series my_series.txt
```

## The cooling-loop scenario - TimesFM-3's native multivariate forecasting

`tools/forecast/make_cooling_loop.py` generates a different kind of series on
purpose: a physical simulation, not a statistical one. A heat exchanger's
conductance fouls between cleanings while an unmeasured, shift-schedule-driven
heat load pushes the return coolant temperature toward a trip threshold - see
the script's own docstring for the energy balance it integrates. The question
is operational: **will the loop trip in the next 5 days, and when?**

Three ways to run the same scenario, in increasing order of what they can show
(the wire the D-Bus path uses today only carries one series, so only the
library API and the CLI can hand the model its actual covariates):

```bash
# Rust, library API, full story: target + a past covariate (pump power) +
# TWO known-future covariates (the ambient forecast, the shift schedule) all
# attend to each other in ONE decode() call, plus a physics-observer baseline
# that shows what a conventional observer gets wrong (it tracks the PRESENT
# state fine and has no model of the SCHEDULE, so it forecasts the load
# staying flat and misses the trip entirely).
examples/forecast/cooling_loop.sh timesfm3.safetensors chart.png

# Python, served path: the SAME scenario, target-only over the generic D-Bus
# wire - a smaller number, and the gap between it and the Rust run's is the
# covariates' own contribution, not noise.
python3 tools/forecast/make_cooling_loop.py --out cooling_loop.csv
dbus-run-session -- bash -c '
  BRAIN_TIMESFM3=timesfm3.safetensors brain serve --dbus --device cpu &
  sleep 2
  python3 examples/forecast/cooling_loop.py cooling_loop.csv
'
```

Get the weights with `brain pull google/timesfm-3.0-pytorch` then
`brain forecast import --timesfm3 <fetched dir> --out timesfm3.safetensors` -
they ship under `timesfm-non-commercial-license-v1.0` (non-commercial,
non-production use; the checkpoint itself may never be redistributed).

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

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements probabilistic forecasting systems for teams
making decisions from time-series data. If your team needs expertise in
forecasting models, backtesting methodology, or evaluating a forecaster
honestly before it reaches production, you can procure our services by sending
an email to **info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
