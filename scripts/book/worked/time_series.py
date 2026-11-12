# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Time Series Fundamentals".

Plain Python 3, standard library only, written from the chapter's own description: the spurious
correlation of two independent random walks, one generated series (trend, a season of 12 and an
autocorrelated remainder) with its autocorrelation and its discrete Fourier spectrum, stationarity
and differencing, the naive, drift, moving-average, seasonal-naive and exponential-smoothing
baselines, an autoregressive model fitted by least squares on seasonal differences, the windows of a
series and the three ways to split them (at random, at a cut, with a purge and an embargo), a
nearest-window forecaster that a random split flatters, and what a scaler fitted on the wrong data
does. Each public function without a leading underscore that takes no arguments returns a markdown
string that the chapter embeds between `<!-- worked: time_series.<name> -->` markers. Underscore
functions and the constants are shared with the chapter's figures, so a picture and the table beside
it cannot disagree.

Swedish Embedded AB implements solutions for forecasting that is evaluated without cheating. If
your team needs expertise in time-series windows, splits and baselines, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
from generalization import _solve

# ------------------------------------------------------------------ the series of the chapter

SERIES_NAME = "time_series.series"
N = 120                         # length of the series
SEASON = 12                     # bars in one season
TREND0, TREND_SLOPE = 10.0, 0.05
AMPLITUDE = 3.0
PHI_NOISE, SHOCK_SD = 0.7, 0.6  # u_t = 0.7 u_(t-1) + 0.6 normal()

REPLICATES_NAME, REPLICATES = "time_series.replicates", 200
CUT = 96                        # the first 96 bars are the past, the last 24 the holdout
HORIZON = 24
BAND = 1.96                     # the 95 percent band of the autocorrelation of white noise is BAND / sqrt(n)

SPUR_NAME, SPUR_PAIRS, SPUR_STEPS = "time_series.spurious", 1000, 100

CONTEXT, WINDOW_H = 24, 12      # context length c and horizon h of a window
TRAIN_FRAC, VAL_FRAC = 0.7, 0.0
POLICY_NAME = "time_series.window_policies"
LEAK_NAME, LEAK_REPS = "time_series.window_leak", 100
SCALE_EPS = 1e-5                # added to the standard deviation of a window before dividing

PANEL_LEVELS = (5, 50, 500)
PANEL_TREND, PANEL_SEASON_AMP, PANEL_NOISE = 0.004, 0.25, 0.04

# ------------------------------------------------------------------ small helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


HELD_GAUSSIAN = (" A normal() call that finds a held value returns it without drawing, and a held value left at the end "
                 "of a series is carried into the next one (the generator is never reset).")


def _seed_text(name):
    seed = bookrng.seed_of(name)
    return f"seed {seed:#018x} ({seed})"


def _generator_line(name, normal=True):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})." + (HELD_GAUSSIAN if normal else "")


def _mean(x):
    return sum(x) / len(x)


def _sd(x):
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _pop_sd(x):
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def _components(g, n=N):
    """(trend, season, noise u_t) lists; u_t = 0.7 u_(t-1) + 0.6 normal() with u_(-1) = 0, one draw per t."""
    trend, season, noise = [], [], []
    u = 0.0
    for t in range(n):
        u = PHI_NOISE * u + SHOCK_SD * g.normal()
        trend.append(TREND0 + TREND_SLOPE * t)
        season.append(AMPLITUDE * math.sin(2.0 * math.pi * t / SEASON))
        noise.append(u)
    return trend, season, noise


def _draw_series(g, n=N):
    tr, se, no = _components(g, n)
    return [a + b + c for a, b, c in zip(tr, se, no)]


def sample_values():
    """The series of the chapter, 120 values."""
    return _draw_series(bookrng.generator(SERIES_NAME))


def _replicate_values():
    g = bookrng.generator(REPLICATES_NAME)
    return [_draw_series(g) for _ in range(REPLICATES)]


def _diff(x):
    return [x[t] - x[t - 1] for t in range(1, len(x))]


def _sdiff(x, m=SEASON):
    return [x[t] - x[t - m] for t in range(m, len(x))]


def _acf(x, kmax):
    """r_k = c_k / c_0 with c_k = (1/n) sum_(t=0)^(n-1-k) (x_t - mean)(x_(t+k) - mean), the mean of all n values."""
    n = len(x)
    m = _mean(x)
    c = [sum((x[t] - m) * (x[t + k] - m) for t in range(n - k)) / n for k in range(kmax + 1)]
    return [v / c[0] for v in c]


def _pearson(a, b):
    ma, mb = _mean(a), _mean(b)
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


def _mae(pred, actual):
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(actual)


def _rmse(pred, actual):
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(actual))


# ------------------------------------------------------------------ opening: two independent walks


def _walk(g, n):
    s, out = 0.0, []
    for t in range(n):
        s = g.normal() if t == 0 else s + g.normal()
        out.append(s)
    return out


def _spurious_runs():
    g = bookrng.generator(SPUR_NAME)
    levels, diffs, first = [], [], None
    for i in range(SPUR_PAIRS):
        a = _walk(g, SPUR_STEPS)
        b = _walk(g, SPUR_STEPS)
        rl = _pearson(a, b)
        rd = _pearson(_diff(a), _diff(b))
        if first is None:
            first = (rl, rd)
        levels.append(rl)
        diffs.append(rd)
    return levels, diffs, first


def spurious():
    """The correlation of two random walks that have nothing to do with each other."""
    levels, diffs, first = _spurious_runs()

    def row(label, rs):
        a = [abs(r) for r in rs]
        return (label, f"{_mean(a):.3f}", f"{sum(1 for v in a if v > 0.3) / len(a):.3f}",
                f"{sum(1 for v in a if v > 0.5) / len(a):.3f}")

    head = (f"{_generator_line(SPUR_NAME)}\n{SPUR_PAIRS} pairs of walks of {SPUR_STEPS} steps. A walk is s_0 = normal(), then "
            f"s_t = s_(t-1) + normal() for t = 1 .. {SPUR_STEPS - 1}; the draws of a pair are the {SPUR_STEPS} steps of the first walk "
            f"and then the {SPUR_STEPS} steps of the second, pair after pair, and no walk has any connection to any other. r is the "
            f"Pearson correlation of the two lists (the covariance over the product of the standard deviations, every sum over the "
            f"same positions in order); 'steps' means the {SPUR_STEPS - 1} first differences s_t - s_(t-1) of each walk.\n\n")
    body = _table(["what is correlated", "mean abs r", "share abs r > 0.3", "share abs r > 0.5"],
                  [row("the walks themselves", levels), row("their steps", diffs)])
    tail = (f"\n\nThe first pair has r = {first[0]:.3f} between the walks and r = {first[1]:.3f} between their steps. "
            f"For independent values the standard deviation of r is close to 1 / sqrt({SPUR_STEPS}) = {1 / math.sqrt(SPUR_STEPS):.3f}.")
    return head + body + tail


# ------------------------------------------------------------------ the series


def sample_series():
    """Trend, season and remainder of the first season and a half, and how the three add to y."""
    g = bookrng.generator(SERIES_NAME)
    tr, se, no = _components(g)
    y = [a + b + c for a, b, c in zip(tr, se, no)]
    rows = [(t, f"{tr[t]:.3f}", f"{se[t]:+.3f}", f"{no[t]:+.3f}", f"{y[t]:.3f}") for t in range(14)]
    head = (f"{_generator_line(SERIES_NAME)}\nThe series has n = {N} values, t = 0 .. {N - 1}: y_t = {TREND0:g} + {TREND_SLOPE:g} t + "
            f"{AMPLITUDE:g} sin(2 pi t / {SEASON}) + u_t, where sin is the library sine of the real number 2 pi t / {SEASON} in radians "
            f"and u_t = {PHI_NOISE:g} u_(t-1) + {SHOCK_SD:g} normal(), with u_(-1) = 0. The only draws are the {N} calls of normal(), "
            f"one per t in increasing order. First 14 values:\n\n")
    tail = (f"\n\nThe mean of all {N} values is {_mean(y):.3f}, their standard deviation (divisor n) {_pop_sd(y):.3f}, the smallest "
            f"{min(y):.3f} (t = {y.index(min(y))}) and the largest {max(y):.3f} (t = {y.index(max(y))}); the last value is "
            f"y_{N - 1} = {y[-1]:.3f}. Once it has settled (the memory of the start fades), the remainder u_t has the long-run standard deviation "
            f"{SHOCK_SD:g} / sqrt(1 - {PHI_NOISE:g}^2) = {SHOCK_SD / math.sqrt(1 - PHI_NOISE ** 2):.3f}, derived in the text below.")
    return head + _table(["t", "trend", "season", "remainder u_t", "y_t"], rows) + tail


# ------------------------------------------------------------------ autocorrelation and the spectrum


def acf_table():
    y = sample_values()
    k = 14
    cols = [("y", y), ("first difference", _diff(y)), ("seasonal difference", _sdiff(y))]
    acfs = [_acf(x, k) for _, x in cols]
    head = (f"The series of the first table. r_k = c_k / c_0, where c_k = (1/n) sum over t = 0 .. n-1-k of (x_t - xbar)(x_(t+k) - xbar), "
            f"xbar is the mean of all n values of the list and the divisor is n for every lag. 'first difference' is x_t = y_t - y_(t-1) "
            f"for t = 1 .. {N - 1} (n = {N - 1}); 'seasonal difference' is x_t = y_t - y_(t-{SEASON}) for t = {SEASON} .. {N - 1} "
            f"(n = {N - SEASON}). For white noise (independent values of mean zero and one common variance) r_k has standard deviation about 1/sqrt(n), so the 95 percent band is {BAND}/sqrt(n): "
            f"{BAND / math.sqrt(N):.3f} for n = {N}, {BAND / math.sqrt(N - 1):.3f} for n = {N - 1} and {BAND / math.sqrt(N - SEASON):.3f} "
            f"for n = {N - SEASON}.\n\n")
    rows = [(j, *(f"{a[j]:+.2f}" for a in acfs)) for j in range(1, k + 1)]
    return head + _table(["lag k", "y", "first difference", "seasonal difference"], rows)


def _line_fit(x):
    """Intercept and slope of the least-squares line through (t, x_t), t = 0 .. n-1, by the normal equations."""
    n = len(x)
    st = sum(range(n))
    stt = sum(t * t for t in range(n))
    sx = sum(x)
    stx = sum(t * v for t, v in enumerate(x))
    a, b = _solve([[n, st], [st, stt]], [sx, stx])
    return a, b


def _dft_amplitudes(x):
    """2 |X[k]| / n for k = 1 .. n/2 - 1 of the list x, X[k] = sum_t x_t e^(-2 pi i k t / n)."""
    n = len(x)
    out = []
    for k in range(1, n // 2):
        re = sum(x[t] * math.cos(2.0 * math.pi * k * t / n) for t in range(n))
        im = -sum(x[t] * math.sin(2.0 * math.pi * k * t / n) for t in range(n))
        out.append(2.0 * math.hypot(re, im) / n)
    return out


def spectrum():
    y = sample_values()
    a, b = _line_fit(y)
    res = [v - (a + b * t) for t, v in enumerate(y)]
    amp = _dft_amplitudes(res)
    order = sorted(range(len(amp)), key=lambda i: (-amp[i], i))[:5]
    rows = [(i + 1, f"{N / (i + 1):.2f}", f"{amp[i]:.3f}") for i in order]
    head = (f"The series of the first table. The least-squares line through (t, y_t), t = 0 .. {N - 1} (normal equations, solved with "
            f"partial pivoting) is {a:.3f} + {b:.4f} t; its residuals are transformed by the discrete Fourier transform "
            f"X[k] = sum over t of x_t exp(-2 pi i k t / {N}), and the table lists the five largest amplitudes 2 |X[k]| / {N} "
            f"among the bins k = 1 .. {N // 2 - 1}, largest first (the library cosine and sine, sums in order of t).\n\n")
    return head + _table(["bin k", f"period {N} / k (bars)", "amplitude"], rows)


# ------------------------------------------------------------------ stationarity


def stationarity():
    y = sample_values()
    items = [("y", y), ("first difference", _diff(y)), ("seasonal difference", _sdiff(y)),
             ("seasonal difference of the first difference", _sdiff(_diff(y)))]
    rows = []
    for label, x in items:
        h = len(x) // 2
        a, b = x[:h], x[h:]
        rows.append((label, len(x), f"{_mean(a):.3f}", f"{_mean(b):.3f}", f"{_pop_sd(a) ** 2:.3f}", f"{_pop_sd(b) ** 2:.3f}",
                     f"{_acf(x, 1)[1]:+.2f}"))
    head = ("The series of the first table and three differences of it. Each list of n values is cut after its first floor(n/2) values (the first part; the second part is the remaining n - floor(n/2) values); the mean "
            "and the variance (divisor: the number of values of the part, around the part's own mean) of each part are printed, and the "
            "autocorrelation at lag 1 of the whole list as in the autocorrelation table.\n\n")
    return head + _table(["series", "n", "mean, first part", "mean, second part", "variance, first part", "variance, second part",
                          "r_1"], rows)


# ------------------------------------------------------------------ baselines


def _slope(past):
    return (past[-1] - past[0]) / (len(past) - 1)


def _fc_naive(past, h):
    return [past[-1]] * h


def _fc_drift(past, h):
    s = _slope(past)
    return [past[-1] + (k + 1) * s for k in range(h)]


def _fc_ma(past, h, w=SEASON):
    return [_mean(past[-w:])] * h


def _fc_snaive(past, h, m=SEASON):
    return [past[len(past) - m + (k % m)] for k in range(h)]


def _fc_snaive_drift(past, h, m=SEASON):
    s = _slope(past)
    return [past[len(past) - m + (k % m)] + s * m * (k // m + 1) for k in range(h)]


def _ses_level(x, alpha):
    """Final level and one-step SSE of simple exponential smoothing: l_0 = x_0, l_t = alpha x_t + (1 - alpha) l_(t-1)."""
    level, sse = x[0], 0.0
    for t in range(1, len(x)):
        sse += (x[t] - level) ** 2
        level = alpha * x[t] + (1.0 - alpha) * level
    return level, sse


ALPHAS = tuple(k / 20 for k in range(1, 21))   # 0.05, 0.10, ..., 1.00


def _best_alpha(past):
    return min(ALPHAS, key=lambda a: (_ses_level(past, a)[1], a))


def _fc_ses(past, h):
    a = _best_alpha(past)
    return [_ses_level(past, a)[0]] * h


def _fit_ar(z, p):
    """Coefficients (phi_0, phi_1 .. phi_p) of z_t = phi_0 + sum_j phi_j z_(t-j) by the normal equations over t = p .. len(z)-1."""
    rows = [[1.0] + [z[t - j] for j in range(1, p + 1)] for t in range(p, len(z))]
    ys = [z[t] for t in range(p, len(z))]
    k = p + 1
    a = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(rows, ys)) for i in range(k)]
    return _solve(a, b)


def _fc_ar(past, h, p=1, m=SEASON):
    """Seasonal differences z_t = y_t - y_(t-m), an AR(p) fit to them, rolled forward and added back to the season before."""
    z = [past[t] - past[t - m] for t in range(m, len(past))]
    coef = _fit_ar(z, p)
    zs, ys = list(z), list(past)
    for _ in range(h):
        zf = coef[0] + sum(coef[j] * zs[-j] for j in range(1, p + 1))
        zs.append(zf)
        ys.append(ys[-m] + zf)
    return ys[len(past):]


BASELINES = (("naive", _fc_naive), ("drift", _fc_drift), ("moving average, 12 bars", _fc_ma),
             ("seasonal naive", _fc_snaive), ("seasonal naive with drift", _fc_snaive_drift),
             ("exponential smoothing, best alpha", _fc_ses), ("AR(1) on seasonal differences", _fc_ar))


def _holdout_scores():
    y = sample_values()
    past, future = y[:CUT], y[CUT:CUT + HORIZON]
    reps = _replicate_values()
    rows = []
    for name, fc in BASELINES:
        p = fc(past, HORIZON)
        mean_mae = _mean([_mae(fc(s[:CUT], HORIZON), s[CUT:CUT + HORIZON]) for s in reps])
        rows.append((name, _mae(p, future), _rmse(p, future), mean_mae))
    return rows


def baselines():
    rows = _holdout_scores()
    head = (f"The series of the first table. The past is y_0 .. y_{CUT - 1}, the {HORIZON} values to forecast are y_{CUT} .. "
            f"y_{CUT + HORIZON - 1}; h = 0 .. {HORIZON - 1} numbers the forecast bars and every forecast is made from the past only. "
            f"naive: y_{CUT - 1}. drift: y_{CUT - 1} + (h + 1) delta with delta = (y_{CUT - 1} - y_0) / {CUT - 1}. moving average: the mean of the "
            f"last {SEASON} values of the past, repeated. seasonal naive: y_({CUT - SEASON} + (h mod {SEASON})). seasonal naive with drift: that "
            f"value plus delta * {SEASON} * (floor(h / {SEASON}) + 1). The last two rows are defined in the next two sections. MAE and RMSE are "
            f"over the {HORIZON} forecast bars (divisor {HORIZON}). The last column is the MAE averaged over {REPLICATES} further series "
            f"drawn by the same recipe from one SplitMix64 generator ({_seed_text(REPLICATES_NAME)}), "
            f"each with its {N} draws in turn, cut and forecast in the same way.\n\n")
    return head + _table(["forecast", "MAE", "RMSE", f"mean MAE over {REPLICATES} series"],
                         [(n, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}") for n, a, b, c in rows])


def ses_search():
    y = sample_values()
    past, future = y[:CUT], y[CUT:CUT + HORIZON]
    best = _best_alpha(past)
    shown = [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    rows = []
    for a in shown:
        level, sse = _ses_level(past, a)
        rows.append((f"{a:.2f}", f"{sse:.1f}", f"{level:.3f}", f"{_mae([level] * HORIZON, future):.3f}"))
    w = [0.3 * 0.7 ** j for j in range(5)]
    head = (f"The past y_0 .. y_{CUT - 1} of the first table. l_0 = y_0 and l_t = alpha y_t + (1 - alpha) l_(t-1) for t = 1 .. {CUT - 1}; "
            f"the one-step error at t is y_t - l_(t-1) and the SSE is the sum of its squares over t = 1 .. {CUT - 1}. alpha runs over "
            f"0.05, 0.10, ..., 1.00 and the smallest SSE wins (the lower alpha on a tie). The forecast is l_({CUT - 1}) repeated "
            f"{HORIZON} times; its MAE is over y_{CUT} .. y_{CUT + HORIZON - 1}.\n\n")
    tail = (f"\n\nThe winner of the grid is alpha = {best:.2f}. For alpha = 0.3 the weights alpha (1 - alpha)^j on y_(t-j), j = 0 .. 4, "
            f"are {', '.join(f'{v:.4f}' for v in w)}.")
    return head + _table(["alpha", "one-step SSE", f"level l_{CUT - 1}", "MAE of the flat forecast"], rows) + tail


def ar_models():
    y = sample_values()
    past, future = y[:CUT], y[CUT:CUT + HORIZON]
    z = [past[t] - past[t - SEASON] for t in range(SEASON, CUT)]
    rows = []
    for p in (1, 2, 3, 4):
        coef = _fit_ar(z, p)
        res = [z[t] - coef[0] - sum(coef[j] * z[t - j] for j in range(1, p + 1)) for t in range(p, len(z))]
        rms = math.sqrt(sum(r * r for r in res) / len(res))
        rows.append((p, ", ".join(f"{c:+.3f}" for c in coef), f"{rms:.3f}", f"{_mae(_fc_ar(past, HORIZON, p), future):.3f}"))
    head = (f"The past of the first table. z_t = y_t - y_(t-{SEASON}) for t = {SEASON} .. {CUT - 1} ({len(z)} values, indexed 0 .. {len(z) - 1} "
            f"below). For each order p the coefficients (phi_0, phi_1, ..., phi_p) minimise the sum of squares of z_t - phi_0 - sum_j phi_j z_(t-j) "
            f"over t = p .. {len(z) - 1} (normal equations, solved with partial pivoting; the sums run over t in increasing order). "
            f"'training RMS' is the root mean square of those residuals (divisor: their number). The forecast rolls the fitted recursion "
            f"forward {HORIZON} bars from the last p values of z, with each forecast z fed back, and adds each to the y of {SEASON} bars before "
            f"(a forecast y once that bar is itself in the future). MAE over y_{CUT} .. y_{CUT + HORIZON - 1}.\n\n")
    return head + _table(["order p", "phi_0, phi_1 .. phi_p", "training RMS", "holdout MAE"], rows)


def ar_horizons():
    reps = _replicate_values()
    hs = (1, 2, 3, 6, 12, 24)
    cols = {"seasonal naive with drift": _fc_snaive_drift, "AR(1) on seasonal differences": _fc_ar}
    errs = {k: [[] for _ in hs] for k in cols}
    for s in reps:
        past, future = s[:CUT], s[CUT:CUT + HORIZON]
        for k, fc in cols.items():
            p = fc(past, HORIZON)
            for i, h in enumerate(hs):
                errs[k][i].append(abs(p[h - 1] - future[h - 1]))
    rows = [(h, *(f"{_mean(errs[k][i]):.3f}" for k in cols)) for i, h in enumerate(hs)]
    head = (f"The {REPLICATES} series of the baseline table (seed {bookrng.seed_of(REPLICATES_NAME):#018x}), each cut at {CUT}. For the "
            f"horizon h (1 = the first forecast bar, y_{CUT}) the table gives the mean over the series of the absolute error |forecast - y| "
            f"at that one bar.\n\n")
    return head + _table(["horizon h (bars)", *cols], rows)


# ------------------------------------------------------------------ windows and splits


def _origins(n=N, c=CONTEXT, h=WINDOW_H):
    """Every origin o (the index of the first forecast bar) with c bars before it and h bars from it: c .. n - h."""
    return list(range(c, n - h + 1))


def _cuts(n, train_frac, val_frac):
    return math.floor(n * train_frac), math.floor(n * (train_frac + val_frac))


def _temporal_split(n, c, h, train_frac, val_frac, embargo):
    """Train, validation and holdout origins by the rule of the chapter's pseudocode (the one `temporal_split` of crates/forecast uses)."""
    gap = max(embargo, h)
    train_end, val_end = _cuts(n, train_frac, val_frac)
    tr, va, ho = [], [], []
    for o in _origins(n, c, h):
        g = o - 1
        if g + h <= train_end - gap:
            tr.append(o)
        elif g >= train_end + gap and g + h <= val_end - gap:
            va.append(o)
        elif g >= val_end + gap:
            ho.append(o)
    return tr, va, ho


def _bars(origins, c, h, labels_only):
    out = set()
    for o in origins:
        out.update(range(o, o + h) if labels_only else range(o - c, o + h))
    return out


def _policies():
    """The four ways to cut the windows of the sample series into a training set and a holdout, as (name, train, holdout)."""
    n, c, h = N, CONTEXT, WINDOW_H
    train_end, _ = _cuts(n, TRAIN_FRAC, VAL_FRAC)
    o_all = _origins()
    brain_tr, _, brain_ho = _temporal_split(n, c, h, TRAIN_FRAC, VAL_FRAC, 0)
    order = list(o_all)
    bookrng.generator(POLICY_NAME).shuffle(order)
    rand_tr = sorted(order[:len(brain_tr)])
    rand_ho = sorted(order[len(brain_tr):len(brain_tr) + len(brain_ho)])
    cut_tr = [o for o in o_all if o - 1 < train_end]
    cut_ho = [o for o in o_all if o - 1 >= train_end]
    wide_tr, _, wide_ho = _temporal_split(n, c, h, TRAIN_FRAC, VAL_FRAC, 16)
    return [("at random, the sizes of the third row", rand_tr, rand_ho),
            (f"a cut after bar {train_end - 1}, no purge", cut_tr, cut_ho),
            ("purge and embargo, embargo 0 (gap 12)", brain_tr, brain_ho),
            ("purge and embargo, embargo 16 (gap 16)", wide_tr, wide_ho)]


def window_policies():
    n, c, h = N, CONTEXT, WINDOW_H
    train_end, _ = _cuts(n, TRAIN_FRAC, VAL_FRAC)
    rows = []
    for name, tr, ho in _policies():
        lab_ho = _bars(ho, c, h, True)
        lab_tr = _bars(tr, c, h, True)
        all_tr = _bars(tr, c, h, False)
        rows.append((name, len(tr), len(ho), len(lab_ho), len(lab_ho & lab_tr), len(lab_ho & all_tr)))
    head = (f"A series of n = {n} bars, windows of c = {c} bars of context and h = {h} bars of forecast. The origin o of a window is the "
            f"index of its first forecast bar: its context is bars o - {c} .. o - 1 and its forecast bars o .. o + {h - 1}, so the origins "
            f"are {c} .. {n - h} ({len(_origins())} windows). The train fraction is {TRAIN_FRAC} and the cut is floor({n} * {TRAIN_FRAC}) "
            f"= {train_end}; the purge-and-embargo rows follow the rule of the pseudocode above (temporal_split) with no validation set. The random row "
            f"shuffles the {len(_origins())} origins (in increasing order before the shuffle; SplitMix64 {_seed_text(POLICY_NAME)}) "
            f"and takes the first {len(_policies()[2][1])} as training windows and the next {len(_policies()[2][2])} as holdout windows. "
            f"'holdout label bars' counts the distinct bars among the forecast bars of the holdout windows; the last two columns count "
            f"how many of them are also a forecast bar of a training window, and how many are a bar of the context or the forecast of a "
            f"training window.\n\n")
    return head + _table(["split", "training windows", "holdout windows", "holdout label bars", "also a training label bar",
                          "also a training bar of any kind"], rows)


def policy_data():
    """For the figure: the origins of each policy, as (name, train, holdout)."""
    return _policies()


# ------------------------------------------------------------------ the nearest-window forecaster and scaling


def _window_scale(ctx):
    """Mean and standard deviation (divisor n) of a context, with the standard deviation shifted up by SCALE_EPS."""
    m = _mean(ctx)
    return m, _pop_sd(ctx) + SCALE_EPS


def _nearest_forecast(y, train_origins, o, c, h, mode, stats=None):
    """Forecast of the bars o .. o + h - 1: the forecast bars of the training window whose (scaled) context is nearest to the (scaled)
    context of the query, the earliest origin on a tie. mode: 'raw', 'global' (stats = (mean, sd) subtracted and divided) or 'window'."""
    def scaled(ctx):
        if mode == "window":
            m, s = _window_scale(ctx)
            return [(v - m) / s for v in ctx]
        if mode == "global":
            return [(v - stats[0]) / stats[1] for v in ctx]
        return list(ctx)

    q = scaled(y[o - c:o])
    best, best_d = None, None
    for p in train_origins:
        d = sum((a - b) ** 2 for a, b in zip(scaled(y[p - c:p]), q))
        if best_d is None or d < best_d:
            best, best_d = p, d
    if mode == "window":
        mq, sq = _window_scale(y[o - c:o])
        mb, sb = _window_scale(y[best - c:best])
        return [mq + sq * (v - mb) / sb for v in y[best:best + h]]
    return y[best:best + h]


def _seasonal_window(y, o, h):
    return [y[o - SEASON + (k % SEASON)] for k in range(h)]


def _leak_runs():
    """Per replicate series: mean-over-windows MAE of each forecaster on the random and the forward holdout."""
    g = bookrng.generator(LEAK_NAME)
    n, c, h = N, CONTEXT, WINDOW_H
    tr_f, _, ho_f = _temporal_split(n, c, h, TRAIN_FRAC, VAL_FRAC, 0)
    train_end, _ = _cuts(n, TRAIN_FRAC, VAL_FRAC)
    keys = ("nn_random", "nn_forward", "snaive_random", "snaive_forward", "global_train", "global_all", "window",
            "const_all", "const_train", "const_context", "const_last")
    res = {k: [] for k in keys}

    def mean_mae(y, origins, fc):
        return _mean([_mae(fc(o), y[o:o + h]) for o in origins])

    for _ in range(LEAK_REPS):
        y = _draw_series(g, n)
        order = _origins(n, c, h)
        g.shuffle(order)
        tr_r = sorted(order[:len(tr_f)])
        ho_r = sorted(order[len(tr_f):len(tr_f) + len(ho_f)])
        res["nn_random"].append(mean_mae(y, ho_r, lambda o: _nearest_forecast(y, tr_r, o, c, h, "raw")))
        res["nn_forward"].append(mean_mae(y, ho_f, lambda o: _nearest_forecast(y, tr_f, o, c, h, "raw")))
        res["snaive_random"].append(mean_mae(y, ho_r, lambda o: _seasonal_window(y, o, h)))
        res["snaive_forward"].append(mean_mae(y, ho_f, lambda o: _seasonal_window(y, o, h)))
        gt = (_mean(y[:train_end]), _pop_sd(y[:train_end]))
        ga = (_mean(y), _pop_sd(y))
        res["global_train"].append(mean_mae(y, ho_f, lambda o: _nearest_forecast(y, tr_f, o, c, h, "global", gt)))
        res["global_all"].append(mean_mae(y, ho_f, lambda o: _nearest_forecast(y, tr_f, o, c, h, "global", ga)))
        res["window"].append(mean_mae(y, ho_f, lambda o: _nearest_forecast(y, tr_f, o, c, h, "window")))
        res["const_all"].append(mean_mae(y, ho_f, lambda o: [ga[0]] * h))
        res["const_train"].append(mean_mae(y, ho_f, lambda o: [gt[0]] * h))
        res["const_context"].append(mean_mae(y, ho_f, lambda o: [_mean(y[o - c:o])] * h))
        res["const_last"].append(mean_mae(y, ho_f, lambda o: [y[o - 1]] * h))
    return res


def _leak_head():
    seed = bookrng.seed_of(LEAK_NAME)
    return (f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed}).{HELD_GAUSSIAN} {LEAK_REPS} series of the recipe of the first "
            f"table, each {N} calls of normal() in turn, followed by the shuffle of its {len(_origins())} origins {CONTEXT} .. {N - WINDOW_H} "
            f"(in increasing order before the shuffle). Windows have c = {CONTEXT} and h = {WINDOW_H}. The forward split is the "
            f"purge-and-embargo rule with embargo 0 and no validation set, so {len(_temporal_split(N, CONTEXT, WINDOW_H, TRAIN_FRAC, VAL_FRAC, 0)[0])} "
            f"training origins and the holdout origins {_temporal_split(N, CONTEXT, WINDOW_H, TRAIN_FRAC, VAL_FRAC, 0)[2][0]} .. "
            f"{_temporal_split(N, CONTEXT, WINDOW_H, TRAIN_FRAC, VAL_FRAC, 0)[2][-1]}; the random split takes the first of the shuffled origins as "
            f"training windows and the next {len(_temporal_split(N, CONTEXT, WINDOW_H, TRAIN_FRAC, VAL_FRAC, 0)[2])} as holdout windows. ")


def window_leak():
    res = _leak_runs()
    rows = [("nearest window", f"{_mean(res['nn_random']):.3f}", f"{_sd(res['nn_random']):.3f}", f"{_mean(res['nn_forward']):.3f}",
             f"{_sd(res['nn_forward']):.3f}"),
            ("seasonal naive", f"{_mean(res['snaive_random']):.3f}", f"{_sd(res['snaive_random']):.3f}",
             f"{_mean(res['snaive_forward']):.3f}", f"{_sd(res['snaive_forward']):.3f}")]
    head = (_leak_head() + "The nearest-window forecaster predicts the h forecast bars of a window as the h forecast bars of the training "
            "window whose context has the smallest squared Euclidean distance to its own (the earliest origin on a tie); seasonal naive "
            f"predicts bar o + k as bar o - {SEASON} + (k mod {SEASON}). The error of a series is the mean over its holdout windows of the "
            f"mean absolute error over the {WINDOW_H} forecast bars; the table gives the mean and the sample standard deviation "
            f"(divisor n - 1) of that error over the {LEAK_REPS} series.\n\n")
    return head + _table(["forecaster", "random split: mean", "sd", "forward split: mean", "sd"], rows)


def scaling_windows():
    res = _leak_runs()
    train_end, _ = _cuts(N, TRAIN_FRAC, VAL_FRAC)
    rows = [("none", "nearest window, raw values", res["nn_forward"]),
            (f"one mean and sd from bars 0 .. {train_end - 1} of the series", "nearest window", res["global_train"]),
            (f"one mean and sd from all {N} bars of the series", "nearest window", res["global_all"]),
            (f"mean and sd of each window's own {CONTEXT} context bars", "nearest window", res["window"]),
            ("none", "seasonal naive", res["snaive_forward"])]
    head = (f"The {LEAK_REPS} series, windows, forward split and shuffles of the previous table, forward holdout only. A scaled window is "
            f"(x - mu) / (sigma + {SCALE_EPS:g}) with sigma the standard deviation (divisor n); for the per-window scaler mu and sigma come from the "
            f"{CONTEXT} context bars of that window, the distance is taken between scaled contexts and the neighbour's forecast bars are "
            f"scaled with the neighbour's context statistics and mapped back with the query's (mu_q + (sigma_q + {SCALE_EPS:g}) (v - mu_n) / (sigma_n + {SCALE_EPS:g})); for a global scaler "
            f"the distance is taken between scaled contexts and the neighbour's forecast bars are returned as they are. Mean over the series "
            f"of the mean MAE over the holdout windows, with the sample standard deviation (divisor n - 1).\n\n")
    return head + _table(["scaler", "forecaster", "mean MAE", "sd"], [(a, b, f"{_mean(v):.3f}", f"{_sd(v):.3f}") for a, b, v in rows],
                         "llrr")


def scaling_constant():
    res = _leak_runs()
    rows = [(f"mean of all {N} bars (includes the holdout)", res["const_all"]),
            (f"mean of bars 0 .. {_cuts(N, TRAIN_FRAC, VAL_FRAC)[0] - 1} (the training span)", res["const_train"]),
            (f"mean of the {CONTEXT} context bars of the window", res["const_context"]),
            ("last context bar", res["const_last"])]
    head = ("The series, forward holdout windows and shuffles of the two previous tables. The forecaster has no model at all: "
            f"it repeats one number {WINDOW_H} times, and the table says which number. Mean over the {LEAK_REPS} series of the "
            f"mean MAE over the {len(_temporal_split(N, CONTEXT, WINDOW_H, TRAIN_FRAC, VAL_FRAC, 0)[2])} holdout windows, with the "
            f"sample standard deviation (divisor n - 1).\n\n")
    return head + _table(["the repeated number", "mean MAE", "sd"], [(a, f"{_mean(v):.3f}", f"{_sd(v):.3f}") for a, v in rows])


def _panel_series():
    out = []
    for level in PANEL_LEVELS:
        g = bookrng.generator(f"time_series.panel.{level}")
        v, ys = 0.0, []
        for t in range(N):
            v = PHI_NOISE * v + PANEL_NOISE * g.normal()
            ys.append(level * (1.0 + PANEL_TREND * t + PANEL_SEASON_AMP * math.sin(2.0 * math.pi * t / SEASON) + v))
        out.append((level, ys))
    return out


def scaling_panel():
    train_end, _ = _cuts(N, TRAIN_FRAC, VAL_FRAC)
    panel = _panel_series()
    pooled = [v for _, ys in panel for v in ys[:train_end]]
    pm, ps = _mean(pooled), _pop_sd(pooled)
    steps = [_pop_sd(_diff(ys[:train_end])) for _, ys in panel]
    total = sum(s * s for s in steps)
    rows = []
    for (level, ys), st in zip(panel, steps):
        sdy = _pop_sd(ys[:train_end])
        rows.append((level, f"{sdy:.3f}", f"{sdy / ps:.4f}", f"{st / ps:.5f}", f"{st / sdy:.4f}", f"{st * st / total:.6f}"))
    head = ("Three series of 120 bars, one per level L in 5, 50 and 500, each y_t = L (1 + 0.004 t + 0.25 sin(2 pi t / 12) + v_t) with "
            "v_t = 0.7 v_(t-1) + 0.04 normal(), v_(-1) = 0 and the 120 draws of normal() in turn from its own generator "
            f"(seeds {', '.join(f'{bookrng.seed_of(f"time_series.panel.{lv}"):#018x}' for lv in PANEL_LEVELS)} for L = "
            f"{', '.join(str(lv) for lv in PANEL_LEVELS)}). Only bars 0 .. {train_end - 1} are used. The pooled scaler subtracts the mean "
            f"({pm:.2f}) and divides by the standard deviation ({ps:.2f}, divisor n) of all {3 * train_end} of these values together; "
            "sd(y) and sd(step) are the standard deviations (divisor n) of the series and of its first differences in the units the "
            "column header says. 'share' is sd(step)^2 over the sum of the three values of sd(step)^2, the part of a squared-error "
            "total that each series contributes when the error is measured in pooled units.\n\n")
    return head + _table(["L", "sd(y) raw", "sd(y) pooled units", "sd(step) pooled units", "sd(step) / sd(y), per series",
                          "share of squared step error"], rows)
