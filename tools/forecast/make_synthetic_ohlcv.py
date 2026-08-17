#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate a synthetic hourly OHLCV series with REAL variance and a
PREDICTABLE conditional mean, so a forecast of it is evidence rather than
decoration.

Why not a sine wave: a trivially periodic curve proves nothing - any model
that copies the last cycle wins, and a model that fails on it is broken
rather than merely weak. Why not a random walk either: its conditional mean
IS the last value, so the best possible forecast is a flat line and the chart
shows nothing. This generator sits between the two, with an analytically
known optimum a reader can check the model against.

The latent log price is

    x_t = mu_t + s_t
    mu_t = drift * t + daily(hour_t) + weekly(weekday_t)      (deterministic)
    s_t  = phi * s_{t-1} + eps_t,   eps_t ~ N(0, sigma)       (stochastic)

so the h-step-ahead conditional mean is exactly

    E[x_{t+h} | x_{<=t}] = mu_{t+h} + phi**h * s_t

which this script prints for the held-out tail as the irreducible-noise floor
("oracle"). `phi` is well below 1, so the series is mean-reverting: the AR
memory decays as phi**h and the forecast converges onto the deterministic
seasonal backbone. Everything a forecaster can legitimately get right is in
`mu`; everything left is `eps`, which nothing can predict. That split is the
whole point.

`daily` and `weekly` are two-harmonic profiles, not single sinusoids: the
daily shape is asymmetric (a fast morning ramp, a slow evening decay) so
"continue the last cycle" is not the same answer as "read the phase", and the
weekly shape carries a weekend dip. Kronos consumes the calendar (minute,
hour, weekday, day, month) alongside the bars, so both are structure it can
legitimately see.

Bars are built the way a real feed builds them, not by decorating the close:
each bar's open is the previous close, the intrabar path is an 8-step
Brownian bridge from open to close, and high/low are that path's extremes -
so `high >= max(open, close)` and `low <= min(open, close)` hold by
construction rather than by clamping, and the OHLC consistency checks in
`crates/forecast/src/csv.rs` are a real test of the data. Volume is
log-normal around an intraday activity profile, scaled by the bar's absolute
return (the volume-volatility correlation every real market has).

Only numpy is needed (already in requirements.txt).

Usage:
  tools/forecast/make_synthetic_ohlcv.py --out examples/forecast/synthetic_hourly.csv
    [--bars 720] [--seed 7] [--horizon 48] [--start 2026-01-05T00:00:00]
"""
import argparse
import datetime as dt
import sys

import numpy as np

# Bars per day and per week at the hourly cadence this emits. A 24/7 (crypto)
# calendar on purpose: a session calendar would put gaps in the timestamp
# column, and the point here is a clean, checkable seasonal structure, not a
# simulation of exchange hours.
HOURS_PER_DAY = 24
HOURS_PER_WEEK = HOURS_PER_DAY * 7


def daily_profile(hour: np.ndarray) -> np.ndarray:
    """Asymmetric intraday shape in log-price units: a fast ramp into the
    European/US overlap and a slower decay overnight. Two harmonics with a
    phase offset, so it is periodic but NOT a single sinusoid - "repeat the
    last 24 bars" and "know what hour it is" give measurably different
    answers."""
    theta = 2.0 * np.pi * hour / HOURS_PER_DAY
    return 0.028 * np.sin(theta - 0.9) + 0.011 * np.sin(2.0 * theta + 0.4)


def weekly_profile(weekday: np.ndarray) -> np.ndarray:
    """Weekend dip: two harmonics over the 7-day cycle, trough on Sat/Sun."""
    theta = 2.0 * np.pi * weekday / 7.0
    return 0.020 * np.cos(theta) + 0.008 * np.cos(2.0 * theta + 1.1)


def activity_profile(hour: np.ndarray) -> np.ndarray:
    """Relative traded volume by hour of day: quiet overnight, busy midday."""
    theta = 2.0 * np.pi * hour / HOURS_PER_DAY
    return 1.0 + 0.55 * np.sin(theta - 1.4) + 0.18 * np.sin(2.0 * theta)


def build(bars: int, seed: int, start: dt.datetime, drift: float, phi: float, sigma: float, p0: float):
    rng = np.random.default_rng(seed)
    stamps = [start + dt.timedelta(hours=i) for i in range(bars)]
    hour = np.array([s.hour for s in stamps], dtype=np.float64)
    weekday = np.array([s.weekday() for s in stamps], dtype=np.float64)

    t = np.arange(bars, dtype=np.float64)
    mu = drift * t + daily_profile(hour) + weekly_profile(weekday)

    # AR(1) residual. Seeded from its stationary distribution so bar 0 is not
    # a special case a model could latch onto.
    s = np.empty(bars, dtype=np.float64)
    stationary_sd = sigma / np.sqrt(1.0 - phi * phi)
    s[0] = rng.normal(0.0, stationary_sd)
    eps = rng.normal(0.0, sigma, size=bars)
    for i in range(1, bars):
        s[i] = phi * s[i - 1] + eps[i]

    log_close = np.log(p0) + mu + s
    close = np.exp(log_close)

    # Open of bar i is the close of bar i-1 (continuous 24/7 session).
    open_ = np.empty(bars, dtype=np.float64)
    open_[0] = close[0] * np.exp(-eps[0])
    open_[1:] = close[:-1]

    # Intrabar path: an 8-step Brownian bridge from open to close, with
    # per-step volatility tied to the bar's own innovation scale. high/low are
    # the extremes of the realised path, so the OHLC invariants hold by
    # construction.
    steps = 8
    bridge_sd = sigma * 0.9
    log_open = np.log(open_)
    high = np.empty(bars, dtype=np.float64)
    low = np.empty(bars, dtype=np.float64)
    for i in range(bars):
        u = rng.normal(0.0, bridge_sd, size=steps).cumsum()
        frac = np.arange(1, steps + 1) / steps
        # Pin the bridge to 0 at both ends, then interpolate open -> close.
        bridge = u - frac * u[-1]
        path = log_open[i] + frac * (log_close[i] - log_open[i]) + bridge
        hi = max(np.max(path), log_open[i], log_close[i])
        lo = min(np.min(path), log_open[i], log_close[i])
        high[i] = np.exp(hi)
        low[i] = np.exp(lo)

    ret = np.zeros(bars, dtype=np.float64)
    ret[1:] = np.diff(log_close)
    volume = 25_000.0 * activity_profile(hour) * (1.0 + 9.0 * np.abs(ret)) * np.exp(rng.normal(0.0, 0.22, size=bars))

    return stamps, open_, high, low, close, volume, mu, s


def oracle_report(mu: np.ndarray, s: np.ndarray, close: np.ndarray, phi: float, p0: float, horizon: int) -> str:
    """The best achievable forecast of the held-out tail, and the two naive
    baselines it should beat - the numbers that say what "good" even means on
    this series.

    `oracle` is the exact conditional mean E[x_{t+h} | x_{<=t}] = mu_{t+h} +
    phi**h * s_t, i.e. the forecast of a model that knows the generating
    process perfectly. Its error is pure `eps` and no model can do better; the
    gap between it and persistence is the whole predictable budget."""
    n = len(close)
    cut = n - horizon
    actual = close[cut:]
    h = np.arange(1, horizon + 1)
    oracle = p0 * np.exp(mu[cut:] + (phi**h) * s[cut - 1])
    persistence = np.full(horizon, close[cut - 1])
    seasonal = close[cut - HOURS_PER_DAY : cut - HOURS_PER_DAY + horizon]

    def mae(p):
        return float(np.mean(np.abs(p - actual)))

    def mape(p):
        return float(np.mean(np.abs(p - actual) / actual) * 100.0)

    return (
        f"held-out tail: {horizon} bars\n"
        f"  oracle (conditional mean, the noise floor) MAE {mae(oracle):.4f}  MAPE {mape(oracle):.3f}%\n"
        f"  persistence (last close repeated)          MAE {mae(persistence):.4f}  MAPE {mape(persistence):.3f}%\n"
        f"  seasonal naive (same hour yesterday)       MAE {mae(seasonal):.4f}  MAPE {mape(seasonal):.3f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV path to write")
    ap.add_argument("--bars", type=int, default=720, help="number of hourly bars (default 720 = 30 days)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=48, help="tail length the oracle report scores (default 48)")
    ap.add_argument("--start", default="2026-01-05T00:00:00", help="first bar timestamp, ISO 8601 (default a Monday)")
    ap.add_argument("--drift", type=float, default=2.5e-5, help="per-bar log drift")
    ap.add_argument("--phi", type=float, default=0.86, help="AR(1) coefficient of the mean-reverting residual")
    ap.add_argument("--sigma", type=float, default=0.0032, help="per-bar innovation sd in log space")
    ap.add_argument("--p0", type=float, default=100.0, help="price level")
    args = ap.parse_args()

    if args.bars < args.horizon + HOURS_PER_DAY + 2:
        print(f"--bars must exceed --horizon + {HOURS_PER_DAY + 2}", file=sys.stderr)
        return 2
    if not 0.0 <= args.phi < 1.0:
        print("--phi must be in [0, 1) for a mean-reverting (stationary) residual", file=sys.stderr)
        return 2

    start = dt.datetime.fromisoformat(args.start)
    stamps, o, h, l, c, v, mu, s = build(args.bars, args.seed, start, args.drift, args.phi, args.sigma, args.p0)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(args.bars):
            f.write(
                f"{stamps[i].strftime('%Y-%m-%dT%H:%M:%S')},"
                f"{o[i]:.4f},{h[i]:.4f},{l[i]:.4f},{c[i]:.4f},{v[i]:.1f}\n"
            )

    print(f"wrote {args.out}: {args.bars} hourly bars {stamps[0]:%Y-%m-%d %H:%M} .. {stamps[-1]:%Y-%m-%d %H:%M}")
    print(oracle_report(mu, s, c, args.phi, args.p0, args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
