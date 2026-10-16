# sample: forecast/cooling-loop

Will an industrial cooling loop trip its over-temperature threshold in the
next 5 days, and when? `tools/forecast/make_cooling_loop.py` generates a
physical simulation (not a statistical one): a heat exchanger's conductance
fouls between cleanings while an unmeasured, shift-schedule-driven heat load
pushes the return coolant temperature toward a trip threshold - see that
script's own docstring for the energy balance it integrates.

Two entry points into the *same* scenario, in increasing order of what they
can show - the D-Bus wire only carries one series today, so only the Rust
library-API path can hand the model its actual covariates:

## Run - Rust, library API (full story, multivariate)

`cooling_loop.sh` calls `crates/timesfm3/examples/cooling_loop.rs` directly:
target + a past covariate (pump power) + TWO known-future covariates (the
ambient forecast, the shift schedule) all attend to each other in ONE
`decode()` call, plus a physics-observer baseline that shows what a
conventional observer gets wrong (it tracks the *present* state fine and has
no model of the schedule, so it forecasts the load staying flat and misses
the trip entirely).

```bash
samples/shell/forecast/cooling-loop/cooling_loop.sh timesfm3.safetensors chart.png
```

## Run - Python, served path (univariate, over D-Bus)

`cooling_loop.py` runs the same scenario through brain's *served* path - a
running `brain serve --dbus`, not the library API. The generic D-Bus
`forecast` action carries one series, so this shows the univariate slice
(the return-temperature series alone) against a seasonal-naive baseline
computed locally. The gap between this run's error and the Rust run's is the
covariates' own contribution, not noise.

```bash
python3 tools/forecast/make_cooling_loop.py --out cooling_loop.csv
BRAIN_TIMESFM3=timesfm3.safetensors tools/dbus-session.sh \
  --serve "--dbus --device cpu" \
  -- python3 samples/shell/forecast/cooling-loop/cooling_loop.py cooling_loop.csv
```

## What it demonstrates

* TimesFM-3's native multivariate forecasting (target + past + known-future
  covariates in one `decode()` call) versus the same model's target-only
  slice over the generic D-Bus wire versus a physics-observer and a
  seasonal-naive baseline - three honest comparison points on one scenario.
* An operational question ("will it trip, and when?") rather than a generic
  accuracy metric.

## What it needs

Get the weights with:

```bash
brain pull google/timesfm-3.0-pytorch
brain forecast import --timesfm3 <fetched dir> --out timesfm3.safetensors
```

They ship under `timesfm-non-commercial-license-v1.0` (non-commercial,
non-production use; the checkpoint itself may never be redistributed).

`cooling_loop.py` additionally needs `jeepney` (`pip install -e brain-py`).
