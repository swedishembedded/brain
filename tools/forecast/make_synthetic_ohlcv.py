#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate a synthetic hourly OHLCV series with the statistical character of a
real traded market, so a candlestick model's forecast of it is a fair test.

Kronos is trained on real market bars. Real market bars are, to a very good
approximation, a random walk in the log price with *clustered* volatility: the
level is close to unpredictable, the SPREAD of the next move is not. Anything
built to make a forecast "look impressive" - a hard daily cycle, a sine wave -
is out of distribution for such a model, and a chart of it fails for a reason
that has nothing to do with whether the port is correct. So this generator
reproduces the four properties that actually define a financial series:

1. **near-unit-root log price.** `r_t = mu - kappa*dev_{t-1} + shock_t`, with
   `dev` the deviation from the drift line and `kappa` small (half-life of
   hundreds of bars). There is a drift and a mild pull back to it, and that is
   the entire predictable budget - see the oracle line the report prints.
2. **volatility clustering.** The de-seasonalized shock follows GARCH(1,1),
   `h_t = omega + alpha*e_{t-1}^2 + beta*h_{t-1}` with `alpha+beta` just under
   1: quiet stretches and violent stretches, exactly like a real tape. This is
   the structure a candlestick model has genuinely learned, and the reason a
   forecast of this series should be an uncertainty band whose WIDTH is
   informative even though its centre is nearly flat.
3. **fat tails.** `z_t` is a standardized Student-t, so single-bar moves of
   4+ sigma happen at a realistic rate rather than never.
4. **an intraday volatility and volume profile.** Volatility carries a mild
   time-of-day shape and a weekend lull; volume peaks at the session open and
   close and scales with the bar's own |shock| - the volume/volatility
   correlation every real market has.

Deliberately absent: any deterministic periodic component in the LEVEL. A
24-bar cycle in the price makes a seasonal-naive baseline unbeatable and asks a
finance model to do seasonal decomposition, which is not what it is.

Bars are built the way a real feed builds them, not by decorating the close:
each bar's open is the previous close, the intrabar path is an 8-step Brownian
bridge from open to close scaled to that bar's own volatility, and high/low
are the extremes of the realised path - so `high >= max(open, close)`
and `low <= min(open, close)` hold by construction rather than by clamping, and
the OHLC consistency checks in `crates/forecast/src/csv.rs` are a real test of
the data. Bar ranges therefore inherit the volatility clustering too.

Only numpy is needed (already in requirements.txt).

Usage:
  tools/forecast/make_synthetic_ohlcv.py --out examples/forecast/synthetic_hourly.csv
    [--bars 720] [--seed 18] [--horizon 6] [--start 2026-01-05T00:00:00]
"""
import argparse
import datetime as dt
import sys

import numpy as np

# Bars per day at the hourly cadence this emits. A 24/7 (crypto) calendar on
# purpose: a session calendar would put gaps in the timestamp column, and the
# consumers here want consecutive hourly stamps so the calendar covariates
# Kronos reads (minute, hour, weekday, day, month) are well defined.
HOURS_PER_DAY = 24


def vol_profile(hour: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """Time-of-day and day-of-week multiplier on the innovation scale (NOT on
    the level). Mildly busier through the European/US overlap, quieter
    overnight, and a weekend lull. Mean-normalized, so it redistributes
    volatility rather than adding any."""
    theta = 2.0 * np.pi * hour / HOURS_PER_DAY
    intraday = 1.0 + 0.22 * np.sin(theta - 1.2) + 0.07 * np.sin(2.0 * theta)
    weekend = np.where(weekday >= 5, 0.85, 1.0)
    m = intraday * weekend
    return m / m.mean()


def activity_profile(hour: np.ndarray) -> np.ndarray:
    """Relative traded volume by hour of day: two humps at the session open and
    close (13:00 and 20:00 UTC, the US cash session) over a quiet overnight
    floor. Gaussian bumps on the hour circle, not a sinusoid - the open/close
    spikes of a real tape are sharp, and a smooth sine would understate them."""

    def bump(centre: float, width: float) -> np.ndarray:
        d = np.abs(hour - centre)
        d = np.minimum(d, HOURS_PER_DAY - d)  # wrap around midnight
        return np.exp(-0.5 * (d / width) ** 2)

    return 0.45 + 1.0 * bump(13.5, 1.6) + 0.75 * bump(20.0, 1.4) + 0.35 * bump(8.0, 2.0)


def build(bars, seed, start, drift, kappa, sigma, alpha, beta, nu, intrabar, p0):
    """The generating process. Returns the bars plus the latent `dev` (deviation
    from the drift line), which is what makes the oracle forecast computable."""
    rng = np.random.default_rng(seed)
    stamps = [start + dt.timedelta(hours=i) for i in range(bars)]
    hour = np.array([s.hour for s in stamps], dtype=np.float64)
    weekday = np.array([s.weekday() for s in stamps], dtype=np.float64)
    season = vol_profile(hour, weekday)

    # Standardized Student-t innovations: unit variance, fat tails.
    z = rng.standard_t(nu, size=bars) / np.sqrt(nu / (nu - 2.0))

    # GARCH(1,1) on the de-seasonalized shock, started from its unconditional
    # variance so bar 0 is not a special case.
    var_uncond = sigma * sigma
    omega = var_uncond * (1.0 - alpha - beta)
    h = np.empty(bars, dtype=np.float64)
    e = np.empty(bars, dtype=np.float64)
    h[0] = var_uncond
    e[0] = np.sqrt(h[0]) * z[0]
    for i in range(1, bars):
        h[i] = omega + alpha * e[i - 1] ** 2 + beta * h[i - 1]
        e[i] = np.sqrt(h[i]) * z[i]
    shock = season * e

    # Log price: a drift line plus a mildly mean-reverting deviation.
    #   dev_t = (1 - kappa) * dev_{t-1} + shock_t
    dev = np.empty(bars, dtype=np.float64)
    dev[0] = shock[0]
    for i in range(1, bars):
        dev[i] = (1.0 - kappa) * dev[i - 1] + shock[i]
    t = np.arange(bars, dtype=np.float64)
    log_close = np.log(p0) + drift * t + dev
    close = np.exp(log_close)

    # Open of bar i is the close of bar i-1 (continuous 24/7 session).
    open_ = np.empty(bars, dtype=np.float64)
    open_[0] = close[0] * np.exp(-shock[0])
    open_[1:] = close[:-1]

    # Intrabar path: an 8-step Brownian bridge from open to close whose total
    # diffusion is `intrabar` times THIS bar's own volatility, so high-low
    # ranges cluster the way the returns do AND their mean size matches a real
    # tape's: the default puts mean(high-low)/sd(return) at 1.38, against 1.38
    # and 1.51 measured on two real 5-minute equity series. high/low are the
    # extremes of the realised path, so the OHLC invariants hold by
    # construction rather than by clamping.
    steps = 8
    bridge_sd = intrabar * season * np.sqrt(h / steps)
    log_open = np.log(open_)
    high = np.empty(bars, dtype=np.float64)
    low = np.empty(bars, dtype=np.float64)
    frac = np.arange(1, steps + 1) / steps
    for i in range(bars):
        u = rng.normal(0.0, bridge_sd[i], size=steps).cumsum()
        bridge = u - frac * u[-1]  # pinned to 0 at both ends
        path = log_open[i] + frac * (log_close[i] - log_open[i]) + bridge
        high[i] = np.exp(max(np.max(path), log_open[i], log_close[i]))
        low[i] = np.exp(min(np.min(path), log_open[i], log_close[i]))

    # Volume: the intraday activity shape, scaled by how violent this bar was.
    volume = (
        25_000.0
        * activity_profile(hour)
        * (1.0 + 1.6 * np.abs(shock) / sigma)
        * np.exp(rng.normal(0.0, 0.25, size=bars))
    )

    return stamps, open_, high, low, close, volume, dev


def report(dev, high, low, close, volume, drift, kappa, p0, horizon):
    """What this series is, and what "good" means on it.

    Two blocks. The first is the series' own statistical fingerprint: if the
    volatility clustering and fat tails are not measurable here, the generator
    is not producing what it claims. The second is the held-out tail scored
    against the oracle - the exact conditional mean of the generating process,

        E[x_{t+h} | F_t] = log p0 + drift*(t+h) + (1-kappa)^h * dev_t

    i.e. the forecast of something that knows the generating process exactly.
    On a near-random-walk it sits a hair below persistence in expectation, and
    that hair IS the entire predictable budget: a model that lands near
    persistence is not failing, it is at the ceiling. Over a tail this short
    the oracle can even lose to persistence on a single draw, which is the same
    point stated more bluntly."""
    r = np.diff(np.log(close))
    r_c = r - r.mean()

    def acf(x, lag):
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    kurt = float(np.mean(r_c**4) / np.mean(r_c**2) ** 2)
    # Ljung-Box on |r| at 10 lags: the standard test for volatility clustering.
    n = len(r_c)
    a = np.abs(r_c)
    lb = n * (n + 2) * sum(acf(a, k) ** 2 / (n - k) for k in range(1, 11))

    rng = (high - low) / close
    stats = (
        f"series: {len(close)} bars, close {close.min():.2f} .. {close.max():.2f}\n"
        f"  per-bar log return: sd {r.std() * 100:.3f}%  mean {r.mean() * 1e4:+.2f} bp  "
        f"excess kurtosis {kurt - 3.0:+.2f}\n"
        f"  autocorrelation:    r lag1 {acf(r_c, 1):+.3f} (a random walk has ~0)  "
        f"|r| lag1 {acf(a, 1):+.3f}  lag24 {acf(a, 24):+.3f}\n"
        f"  volatility clustering: Ljung-Box(|r|, 10) = {lb:.1f}  "
        f"(chi2 critical value at the 0.05 level is 18.3 -> clustered)\n"
        f"  bar shape: mean (high-low)/close {rng.mean() * 100:.3f}% = {rng.mean() / r.std():.2f} times the return sd "
        f"(1.4-1.5 on real intraday bars)\n"
        f"  volume/|return| correlation {np.corrcoef(volume[1:], np.abs(r))[0, 1]:+.3f}"
    )

    cut = len(close) - horizon
    actual = close[cut:]
    h = np.arange(1, horizon + 1)
    t = np.arange(cut, cut + horizon, dtype=np.float64)
    oracle = p0 * np.exp(drift * t + (1.0 - kappa) ** h * dev[cut - 1])
    persistence = np.full(horizon, close[cut - 1])
    ctx_r = np.diff(np.log(close[:cut]))
    drift_bl = close[cut - 1] * np.exp(ctx_r.mean() * h)
    seasonal = close[cut - HOURS_PER_DAY : cut - HOURS_PER_DAY + horizon]

    def mae(p):
        return float(np.mean(np.abs(p - actual)))

    def mape(p):
        return float(np.mean(np.abs(p - actual) / actual) * 100.0)

    rows = [
        ("oracle (conditional mean, the ceiling)", oracle),
        ("persistence (last close repeated)", persistence),
        ("drift (context mean log return)", drift_bl),
        ("seasonal naive (same hour yesterday)", seasonal),
    ]
    lines = "\n".join(f"  {name:<38} MAE {mae(p):.4f}  MAPE {mape(p):.3f}%" for name, p in rows)
    return f"{stats}\nheld-out tail: {horizon} bars\n{lines}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV path to write")
    ap.add_argument("--bars", type=int, default=720, help="number of hourly bars (default 720 = 30 days)")
    # Each seed is ONE draw of a heavy-tailed GARCH process, and 720 bars is
    # short enough that draws differ a lot: the default is the seed whose
    # realized return sd lands closest to `--sigma`, so the shipped series is a
    # representative sample of the configured process rather than a quiet or a
    # violent outlier.
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--horizon", type=int, default=6, help="tail length the report scores (default 6, the demo's horizon)")
    ap.add_argument("--start", default="2026-01-05T00:00:00", help="first bar timestamp, ISO 8601 (default a Monday)")
    ap.add_argument("--drift", type=float, default=8.0e-5, help="per-bar log drift")
    ap.add_argument("--kappa", type=float, default=0.004, help="pull back to the drift line per bar (0 = pure random walk)")
    ap.add_argument("--sigma", type=float, default=0.0045, help="unconditional per-bar innovation sd in log space")
    ap.add_argument("--alpha", type=float, default=0.09, help="GARCH ARCH coefficient")
    ap.add_argument("--beta", type=float, default=0.88, help="GARCH persistence coefficient")
    ap.add_argument("--nu", type=float, default=5.0, help="Student-t degrees of freedom of the innovations")
    ap.add_argument("--intrabar", type=float, default=1.4, help="intrabar diffusion as a multiple of the bar's own sd (sets the high-low range)")
    ap.add_argument("--p0", type=float, default=100.0, help="price level")
    args = ap.parse_args()

    if args.bars < args.horizon + HOURS_PER_DAY + 2:
        print(f"--bars must exceed --horizon + {HOURS_PER_DAY + 2}", file=sys.stderr)
        return 2
    if not 0.0 <= args.kappa < 1.0:
        print("--kappa must be in [0, 1)", file=sys.stderr)
        return 2
    if not (args.alpha >= 0.0 and args.beta >= 0.0 and args.alpha + args.beta < 1.0):
        print("--alpha + --beta must be in [0, 1) for a stationary GARCH", file=sys.stderr)
        return 2
    if args.nu <= 2.0:
        print("--nu must exceed 2 for the innovations to have finite variance", file=sys.stderr)
        return 2

    start = dt.datetime.fromisoformat(args.start)
    stamps, o, hi, lo, c, v, dev = build(
        args.bars, args.seed, start, args.drift, args.kappa, args.sigma, args.alpha, args.beta, args.nu, args.intrabar, args.p0
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(args.bars):
            f.write(
                f"{stamps[i].strftime('%Y-%m-%dT%H:%M:%S')},"
                f"{o[i]:.4f},{hi[i]:.4f},{lo[i]:.4f},{c[i]:.4f},{v[i]:.1f}\n"
            )

    print(f"wrote {args.out}: {args.bars} hourly bars {stamps[0]:%Y-%m-%d %H:%M} .. {stamps[-1]:%Y-%m-%d %H:%M}")
    print(report(dev, hi, lo, c, v, args.drift, args.kappa, args.p0, args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
