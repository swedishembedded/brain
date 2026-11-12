# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Evaluating Forecasters Honestly".

Plain Python 3, standard library only, written from the chapter's own description: a walk forward
through one generated series that shows how much a single origin can decide, the fit boundary of a
walk-forward evaluation (the purge and the embargo at a moving cut), a scale-free error with its
seasonal scale, the weighted quantile loss and what it counts, a skill score against a baseline with
paired differences and a moving-block bootstrap interval, the Diebold-Mariano statistic with
autocorrelated loss differentials and its size under overlap, what a smaller stride does and does not
add, how many series it takes before a difference is real, a selection that spends the test set,
cherry-picked horizons, the selection inside a benchmark score, and a leak that a negative control
catches. Each public function without a leading underscore that takes no arguments returns a markdown
string that the chapter embeds between `<!-- worked: evaluating_forecasts.<name> -->` markers.
Underscore functions and the constants are shared with the chapter's figures, so a picture and the
table beside it cannot disagree. The series and the forecasters of Chapter 1 (`time_series`) are
imported, never copied.

Swedish Embedded AB implements solutions for forecast evaluation that separates skill from luck. If
your team needs expertise in walk-forward evaluation and the statistics of forecast comparison, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import statistics
from functools import lru_cache

import bookrng
import time_series as ts
from time_series import _generator_line, _mean, _seed_text, _table

# ------------------------------------------------------------------ constants

SERIES_NAME = "evaluating_forecasts.series"
N = 240                          # bars of the evaluation series
H = 12                           # horizon: bars forecast at every origin
FIRST_ORIGIN = 60                # the first origin: 60 bars of history
LAST_ORIGIN = N - H              # the last origin whose 12 forecast bars exist: 228
SEASON = ts.SEASON

BOOT_NAME = "evaluating_forecasts.bootstrap"
BOOT_RESAMPLES = 2000
DM_NAME, DM_REPS, DM_SIZES = "evaluating_forecasts.dm_size", 2000, (40, 100, 400, 1600)
STRIDE_NAME, STRIDE_REPS, STRIDE_SPAN, STRIDES = "evaluating_forecasts.stride", 2000, 288, (1, 2, 3, 4, 6, 12)
PANEL_NAME, PANEL_SERIES = "evaluating_forecasts.panel", 200
POWER_NAME, POWER_REPS = "evaluating_forecasts.power", 1000
POWER_SIZES = (5, 10, 20, 40, 80)
CHERRY_NAME, CHERRY_SERIES, CHERRY_BARS, CHERRY_FIRST = "evaluating_forecasts.cherry", 40, 160, 100
SELECT_NAME, SELECT_REPS = "evaluating_forecasts.selection", 200
CONTROL_NAME, CONTROL_SERIES, CONTEXT = "evaluating_forecasts.control", 100, 24
Z975 = statistics.NormalDist().inv_cdf(0.975)   # 1.959964, the library inverse of the Gaussian distribution function

# The forecasters of Chapter 1, in the order the tables list them.
FORECASTERS = (("seasonal naive", "SN", ts._fc_snaive),
               ("seasonal naive with drift", "SND", ts._fc_snaive_drift),
               ("AR(1) on seasonal differences", "AR", ts._fc_ar),
               ("naive", "naive", ts._fc_naive))

_MARK = "evaluating_forecasts."


# ------------------------------------------------------------------ helpers


def _mae(pred, actual):
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(actual)


def _fx(v, d=3):
    return f"{v:.{d}f}"


def _fs(v, d=3):
    return f"{v:+.{d}f}"


def _sd(v):
    m = _mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def _percentile(vals, q):
    """The book's percentile: linear interpolation between neighbouring order statistics at position q (m - 1)."""
    v = sorted(vals)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _series_spec():
    return (f"The series is the recipe of Chapter 1 with n = {N}: y_t = 10 + 0.05 t + 3 sin(2 pi t / 12) + u_t with u_t = 0.7 u_(t-1) + "
            f"0.6 normal() and u_(-1) = 0, the {N} calls of normal() in increasing t, from the generator named {SERIES_NAME}.")


def _forecaster_spec():
    return ("The forecasters are those of Chapter 1, each refitted at every origin o from the bars y_0 .. y_(o-1) alone (an expanding "
            "past) and asked for the 12 bars o .. o + 11: SN repeats the last season, y_(o-12+(k mod 12)) for k = 0 .. 11; SND adds to "
            "that the drift delta * 12 * (floor(k / 12) + 1) with delta = (y_(o-1) - y_0) / (o - 1); AR fits an AR(1) to the seasonal "
            "differences by least squares and adds the forecast differences back to the season before; naive repeats y_(o-1).")


@lru_cache(maxsize=None)
def _walk():
    """The series, its origins 60 .. 228, and for each forecaster the list of mean absolute errors over the 12 forecast bars."""
    y = ts._draw_series(bookrng.generator(SERIES_NAME), N)
    origins = list(range(FIRST_ORIGIN, LAST_ORIGIN + 1))
    losses = {code: [_mae(fc(y[:o], H), y[o:o + H]) for o in origins] for _, code, fc in FORECASTERS}
    return y, origins, losses


def series_values():
    return _walk()[0]


def origin_losses():
    """For the figures: (origins, losses by forecaster code)."""
    _, origins, losses = _walk()
    return origins, losses


def _skill(model, base):
    return 1.0 - sum(model) / sum(base)


# ------------------------------------------------------------------ one origin is one draw


def origin_lottery():
    y, origins, L = _walk()
    shown = origins[::24]
    rows = [(o, _fx(L["SND"][o - FIRST_ORIGIN]), _fx(L["AR"][o - FIRST_ORIGIN]),
             _fx(L["AR"][o - FIRST_ORIGIN] / L["SND"][o - FIRST_ORIGIN]),
             "AR" if L["AR"][o - FIRST_ORIGIN] < L["SND"][o - FIRST_ORIGIN] else "SND") for o in shown]
    head = (f"{_generator_line(SERIES_NAME)} {_series_spec()} {_forecaster_spec()} The loss at an origin is the mean absolute error over "
            f"its 12 forecast bars (divisor 12). First table: every 24th origin, SND against AR.\n\n")
    t1 = _table(["origin o", "MAE of SND", "MAE of AR", "AR / SND", "smaller error"], rows)
    base, model = L["SND"], L["AR"]
    ratios = [m / b for m, b in zip(model, base)]
    wins = sum(m < b for m, b in zip(model, base))
    sets = (("the one origin 96", [96]), ("15 origins 60, 72, ..., 228 (every 12th)", origins[::12]),
            (f"all {len(origins)} origins 60, 61, ..., 228", origins))
    rows2 = []
    for name, ors in sets:
        b = [base[o - FIRST_ORIGIN] for o in ors]
        m = [model[o - FIRST_ORIGIN] for o in ors]
        rows2.append((name, len(ors), _fx(_mean(b)), _fx(_mean(m)), sum(x < z for x, z in zip(m, b)), _fs(_skill(m, b))))
    t2 = _table(["origins used", "how many", "mean MAE of SND", "mean MAE of AR", "origins where AR is smaller", "skill of AR against SND"], rows2)
    tail = (f"\n\nThe ratio AR / SND over the {len(origins)} origins has the smallest value {_fx(min(ratios))} and the largest {_fx(max(ratios))}; "
            f"AR has the smaller error at {wins} of them. Skill is 1 minus the ratio of the two mean errors, here the mean over the origins "
            f"listed in the row.")
    return head + t1 + "\n\n" + t2 + tail


# ------------------------------------------------------------------ the fit boundary of an origin


def fit_boundary():
    c, h = CONTEXT, H
    rows = []
    for o in (96, 132):
        for label, e, purged in (("all windows with origin below o", 0, False), ("purge only, e = 0", 0, True),
                                 ("purge and embargo e = 6", 6, True), ("purge and embargo e = 12", 12, True)):
            last_o = o - e - h if purged else o - 1
            count = last_o - c + 1
            rows.append((o, label, last_o, count, last_o + h - 1, "yes" if last_o + h - 1 >= o else "no"))
    return ("A model fitted by direct windows (context c = 24 bars, target h = 12 bars; the window with origin o' has context bars "
            "o' - 24 .. o' - 1 and target bars o' .. o' + 11, as in Chapter 1) is evaluated at an origin o. Its training windows are "
            "those with origin from 24 up to the last origin shown; the last target bar they use is last origin + 11. The purged rule "
            "keeps windows whose target ends at or before bar o - e - 1, so the last origin is o - e - 12.\n\n"
            + _table(["evaluation origin o", "rule", "last training origin", "training windows", "last target bar used",
                      "reaches bar o or later"], rows))


# ------------------------------------------------------------------ MASE and its scale


def _scale(context, m):
    """Mean absolute seasonal difference of a context: (1 / (len - m)) sum_(t=m)^(len-1) |y_t - y_(t-m)|."""
    return sum(abs(context[t] - context[t - m]) for t in range(m, len(context))) / (len(context) - m)


def _mase(pred, actual, context, m):
    return _mae(pred, actual) / _scale(context, m)


def mase_scales():
    y, origins, L = _walk()
    rows = []
    for o in (96, 228):
        ctx = y[:o]
        for name, code, fc in FORECASTERS[1:3]:
            mae = L[code][o - FIRST_ORIGIN]
            rows.append((o, name, _fx(mae), _fx(_scale(ctx, 1)), _fx(_scale(ctx, SEASON)), _fx(mae / _scale(ctx, 1)),
                         _fx(mae / _scale(ctx, SEASON))))
    t1 = _table(["origin o", "forecaster", "MAE", "Delta_1 of y_0 .. y_(o-1)", "Delta_12", "MASE with m = 1", "MASE with m = 12"], rows)
    rows2 = []
    for name, code, _ in FORECASTERS:
        m1 = _mean([L[code][i] / _scale(y[:o], 1) for i, o in enumerate(origins)])
        m12 = _mean([L[code][i] / _scale(y[:o], SEASON) for i, o in enumerate(origins)])
        rows2.append((name, _fx(m1), _fx(m12)))
    t2 = _table(["forecaster", f"mean MASE, m = 1", "mean MASE, m = 12"], rows2)
    big = [v + 27.0 * math.sin(2.0 * math.pi * t / SEASON) for t, v in enumerate(y)]
    rows3 = []
    for name, series in (("amplitude 3 (the series above)", y), ("amplitude 30 (27 sin(2 pi t / 12) added)", big)):
        maes, m1, m12 = [], [], []
        for o in origins:
            mae = _mae(ts._fc_snaive(series[:o], H), series[o:o + H])
            maes.append(mae)
            m1.append(mae / _scale(series[:o], 1))
            m12.append(mae / _scale(series[:o], SEASON))
        rows3.append((name, _fx(_mean(maes)), _fx(_scale(series[:96], 1)), _fx(_scale(series[:96], SEASON)), _fx(_mean(m1)), _fx(_mean(m12))))
    t3 = _table(["series", "mean MAE of SN", "Delta_1 at o = 96", "Delta_12 at o = 96", "mean MASE, m = 1", "mean MASE, m = 12"], rows3)
    head = (f"{_generator_line(SERIES_NAME)} {_series_spec()} {_forecaster_spec()} At origin o, Delta_m is the mean absolute seasonal "
            f"difference of the past, (1 / (o - m)) times the sum over t = m .. o - 1 of |y_t - y_(t-m)|, and the MASE of a forecaster at o is "
            f"its MAE over the 12 forecast bars divided by Delta_m of the same past. First table: the two origins; second table: the mean "
            f"of the MASE over all 169 origins 60 .. 228. Third table: SN on the same series and on the series y'_t = y_t + 27 sin(2 pi t / 12) "
            f"(the library sine, the same normal() draws), over the same 169 origins.\n\n")
    return head + t1 + "\n\n" + t2 + "\n\n" + t3


# ------------------------------------------------------------------ the weighted quantile loss and what it counts

WQL_ACTUAL = (10.0, 12.0, 9.0, 14.0)
WQL_MU = (10.5, 11.0, 10.0, 13.0)
WQL_SIGMA = 2.0
WQL_GRIDS = (("0.5", (0.5,)), ("0.1, 0.5, 0.9", (0.1, 0.5, 0.9)),
             ("0.1, 0.2, ..., 0.9", tuple(round(0.1 * k, 1) for k in range(1, 10))))


def _pinball(q, y, tau):
    e = y - q
    return max(tau * e, (tau - 1.0) * e)


def _wql(levels, actual=WQL_ACTUAL, mu=WQL_MU, sigma=WQL_SIGMA):
    inv = statistics.NormalDist().inv_cdf
    total = 0.0
    for t, y in enumerate(actual):
        for tau in levels:
            total += _pinball(mu[t] + sigma * inv(tau), y, tau)
    return 2.0 * total / sum(abs(v) for v in actual), len(levels)


def wql_counts():
    rows = []
    for name, levels in WQL_GRIDS:
        w, q = _wql(levels)
        rows.append((name, q, _fx(w, 4), _fx(w / q, 4)))
    cells = []
    inv = statistics.NormalDist().inv_cdf
    for t, y in enumerate(WQL_ACTUAL):
        cells.append((t, _fx(y, 1), _fx(WQL_MU[t], 1)) + tuple(_fx(_pinball(WQL_MU[t] + WQL_SIGMA * inv(tau), y, tau), 4)
                                                          for tau in (0.1, 0.5, 0.9)))
    shifted, q3 = _wql(WQL_GRIDS[1][1], tuple(v + 100.0 for v in WQL_ACTUAL), tuple(v + 100.0 for v in WQL_MU))
    rows.append(("0.1, 0.5, 0.9, every y_t and mu_t raised by 100", q3, _fx(shifted, 4), _fx(shifted / q3, 4)))
    t0 = _table(["bar t", "actual y_t", "mu_t", "pinball at 0.1", "pinball at 0.5", "pinball at 0.9"], cells)
    t1 = _table(["levels used", "number of levels Q", "wQL (sum over levels)", "wQL divided by Q"], rows)
    head = ("Actuals y = (10, 12, 9, 14), four bars. The forecast of bar t is the Gaussian with mean mu = (10.5, 11, 10, 13) and standard "
            "deviation 2, so its quantile at level tau is mu_t + 2 z_tau with z_tau the library inverse of the standard Gaussian distribution "
            "function (z_0.5 = 0, z_0.1 = -1.281552, z_0.9 = +1.281552), and the last row of the second table adds 100 to every actual and every mean. The pinball loss of a quantile q against y at level tau is "
            "max(tau (y - q), (tau - 1)(y - q)). wQL is 2 times the sum of the pinball losses over all bars and all levels, divided by the "
            "sum of |y_t| = 45.\n\n")
    return head + t0 + "\n\n" + t1


# ------------------------------------------------------------------ skill depends on the baseline


def skill_matrix():
    _, origins, L = _walk()
    bases = (("naive", "naive"), ("SN", "SN"), ("SND", "SND"))
    rows = []
    for name, code, _ in FORECASTERS[:3]:
        row = [name]
        for _, bcode in bases:
            row.append("-" if code == bcode else _fs(_skill(L[code], L[bcode])))
        rows.append(tuple(row))
    head = (f"The 169 origins 60 .. 228 of the first example (series {_seed_text(SERIES_NAME)}). Each entry is the skill of the forecaster in the row "
            f"against the baseline in the column, 1 - (mean MAE of the forecaster) / (mean MAE of the baseline) over all 169 origins.\n\n")
    return head + _table(["forecaster", "against naive", "against SN", "against SND"], rows)


# ------------------------------------------------------------------ skill with a paired, block-bootstrapped interval


def _block_indices(rng, k, blk):
    """Indices 0 .. k-1 of one moving-block resample: ceil(k / blk) starts, each rng.integer(k - blk + 1), blocks of blk consecutive
    indices laid end to end, cut to k."""
    out = []
    for _ in range(-(-k // blk)):
        s = rng.integer(k - blk + 1)
        out.extend(range(s, s + blk))
    return out[:k]


def _boot_interval(rng, base, model, blk, resamples=BOOT_RESAMPLES):
    k = len(base)
    vals = []
    for _ in range(resamples):
        idx = _block_indices(rng, k, blk)
        vals.append(1.0 - sum(model[i] for i in idx) / sum(base[i] for i in idx))
    return _percentile(vals, 0.025), _percentile(vals, 0.975)


def skill_intervals():
    _, origins, L = _walk()
    rng = bookrng.generator(BOOT_NAME)
    rows = []
    cmp_list = (("AR against SND", "SND", "AR"), ("SND against SN", "SN", "SND"))
    for name, bcode, mcode in cmp_list:
        base, model = L[bcode], L[mcode]
        for label, sel, blk in (("all 169 origins, blocks of 1 (independent origins)", None, 1),
                                ("all 169 origins, blocks of 6", None, 6),
                                ("all 169 origins, blocks of 12", None, 12),
                                ("all 169 origins, blocks of 24", None, 24),
                                ("15 origins 12 apart, blocks of 1", 12, 1)):
            b = base if sel is None else base[::sel]
            m = model if sel is None else model[::sel]
            lo, hi = _boot_interval(rng, b, m, blk)
            rows.append((name, label, _fs(_skill(m, b)), _fs(lo, 4), _fs(hi, 4), "excludes 0" if lo > 0 or hi < 0 else "contains 0"))
    head = (f"The losses of the previous example (series {_seed_text(SERIES_NAME)}). Skill of a model against a baseline is 1 - (mean model "
            f"loss) / (mean baseline loss) over the origins in the row. The interval is the 2.5th and 97.5th percentile (the book's `percentile`) "
            f"of the skill over {BOOT_RESAMPLES} resamples of the origins: a resample is ceil(K / blk) blocks of blk consecutive origins, each block "
            f"starting at rng.integer(K - blk + 1) in the row's list of K origins, laid end to end and cut to K; the same indices are used for model and "
            f"baseline. One generator ({_seed_text(BOOT_NAME)}) draws every row in the table's order, row after row, resample after resample.\n\n")
    return head + _table(["comparison", "origins and blocks", "skill", "2.5th percentile", "97.5th percentile", "interval"], rows)


# ------------------------------------------------------------------ the Diebold-Mariano statistic


def _autocov(d, k):
    n = len(d)
    m = sum(d) / n
    return sum((d[t] - m) * (d[t + k] - m) for t in range(n - k)) / n


def _long_run(d, lag, kernel):
    """gamma_0 + 2 sum_(k=1)^lag w_k gamma_k with w_k = 1 (rectangular) or 1 - k / (lag + 1) (Bartlett); gamma_k divisor n."""
    g0 = _autocov(d, 0)
    if kernel == "rect":
        return g0 + 2.0 * sum(_autocov(d, k) for k in range(1, lag + 1))
    return g0 + 2.0 * sum((1.0 - k / (lag + 1.0)) * _autocov(d, k) for k in range(1, lag + 1))


def _dm(d, h):
    """The statistics of a sequence of loss differentials d at horizon h: iid, rectangular lag h-1, with the small-sample factor, Bartlett."""
    n = len(d)
    m = sum(d) / n
    out = {"iid": m / math.sqrt(_long_run(d, 0, "rect") / n)}
    v = _long_run(d, h - 1, "rect")
    if v > 0:
        z = m / math.sqrt(v / n)
        out["rect"] = z
        out["hln"] = z * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    else:
        out["rect"] = out["hln"] = None
    out["bart"] = m / math.sqrt(_long_run(d, h - 1, "bart") / n)
    return out


def _t_pdf_const(df):
    return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)


def _t_central(z, df, steps=2000):
    """Probability that a Student t variable with df degrees of freedom lies within [-z, z], by Simpson's rule."""
    c = _t_pdf_const(df)

    def pdf(t):
        return c * (1 + t * t / df) ** (-(df + 1) / 2)

    hh = z / steps
    s = pdf(0) + pdf(z)
    s += 4 * sum(pdf((2 * i - 1) * hh) for i in range(1, steps // 2 + 1))
    s += 2 * sum(pdf(2 * i * hh) for i in range(1, steps // 2))
    return 2 * s * hh / 3


def _t_crit(df):
    """The multiplier z with 95 percent of a Student t density within [-z, z], by bisection."""
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _t_central(mid, df) < 0.95:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _p_gauss(z):
    return 1.0 - math.erf(abs(z) / math.sqrt(2.0))


def dm_on_series():
    _, origins, L = _walk()
    d = [a - b for a, b in zip(L["SND"], L["AR"])]
    n = len(d)
    mean = _mean(d)
    g = [_autocov(d, k) for k in range(H)]
    rows = [(k, _fx(g[k], 4), _fx(g[k] / g[0], 3)) for k in range(H)]
    t1 = _table(["lag k", "gamma_k of d", "gamma_k / gamma_0"], rows)
    s = _dm(d, H)
    v_iid, v_rect, v_bart = _long_run(d, 0, "rect"), _long_run(d, H - 1, "rect"), _long_run(d, H - 1, "bart")
    rect = (_fs(v_rect, 4), "none: the estimate is not positive", "-") if s["rect"] is None else (
        _fx(v_rect, 4), _fx(s["rect"], 2), _fx(_p_gauss(s["rect"]), 4))
    rows2 = [("independent origins (lag 0)", _fx(v_iid, 4), _fx(s["iid"], 2), _fx(_p_gauss(s["iid"]), 4)),
             ("rectangular weights, lag 11", rect[0], rect[1], rect[2]),
             ("Bartlett weights, lag 11", _fx(v_bart, 4), _fx(s["bart"], 2), _fx(_p_gauss(s["bart"]), 4))]
    t2 = _table(["estimate of the variance of d used", "long-run variance", "statistic", "two-sided p (Gaussian)"], rows2, "lrrr")
    head = (f"The losses of the first example (series {_seed_text(SERIES_NAME)}), horizon h = 12. d_o = (MAE of SND at o) - (MAE of AR at o) for the "
            f"{n} origins o = 60 .. 228, positive when AR is better; its mean is {_fx(mean, 4)}. gamma_k is the autocovariance of d at lag k "
            f"with the mean of d removed and the divisor {n} at every lag. The long-run variance is gamma_0 + 2 (sum over k = 1 .. 11 of w_k gamma_k) "
            f"with w_k = 1 (rectangular) or w_k = 1 - k / 12 (Bartlett), and gamma_0 alone for independent origins; the statistic is the mean of d "
            f"divided by the square root of (long-run variance / {n}), and its p-value is 1 - erf(|statistic| / sqrt(2)).\n\n")
    return head + t1 + "\n\n" + t2


def dm_size():
    cv = {k: _t_crit(k - 1) for k in DM_SIZES}
    rows = []
    for k in DM_SIZES:
        rng = bookrng.generator(DM_NAME)
        rej = {"iid": 0, "rect": 0, "hln": 0, "bart": 0}
        undefined = 0
        for _ in range(DM_REPS):
            xi = [rng.normal() for _ in range(k + H - 1)]
            d = [sum(xi[t:t + H]) / math.sqrt(H) for t in range(k)]
            s = _dm(d, H)
            if s["rect"] is None:
                undefined += 1
            for key in rej:
                if s[key] is None:
                    continue
                crit = cv[k] if key == "hln" else Z975
                if abs(s[key]) > crit:
                    rej[key] += 1
        rows.append((k, _fx(cv[k], 3)) + tuple(_fx(rej[key] / DM_REPS, 3) for key in ("iid", "rect", "hln", "bart")) + (undefined,))
    head = (f"Differentials with no true difference: d_t = (1 / sqrt(12)) (xi_t + xi_(t+1) + ... + xi_(t+11)) for t = 0 .. K - 1 with xi the "
            f"standard Gaussian calls of one generator per row of the table ({_seed_text(DM_NAME)}, a fresh generator for each K), the K + 11 calls "
            f"of normal() of a replicate in order, {DM_REPS} replicates in turn; d has mean 0, variance 1 and autocovariance "
            f"(12 - k) / 12 at lag k < 12. Each statistic is that of the previous example with horizon 12; the Gaussian columns reject when its "
            f"absolute value exceeds {_fx(Z975, 6)}, the small-sample column when it exceeds the Student t multiplier of K - 1 degrees of freedom shown. "
            f"A replicate whose rectangular long-run variance is not positive has no rectangular statistic and is counted as not rejecting.\n\n")
    return head + _table(["K origins", "t multiplier", "independent origins", "rectangular", "rectangular with factor", "Bartlett",
                          "replicates with no rectangular statistic"], rows)


# ------------------------------------------------------------------ what a smaller stride adds


def stride_data():
    """For the figure: for each stride, (K, spread of the mean over the replicates, mean iid error bar, mean overlap-aware error bar)."""
    out = []
    for s in STRIDES:
        rng = bookrng.generator(STRIDE_NAME)
        means, iid, nw = [], [], []
        k = (STRIDE_SPAN - H) // s + 1
        lag = -(-H // s) - 1
        for _ in range(STRIDE_REPS):
            xi = [rng.normal() for _ in range(STRIDE_SPAN)]
            d = [sum(xi[t * s:t * s + H]) / math.sqrt(H) for t in range(k)]
            means.append(_mean(d))
            iid.append(math.sqrt(_long_run(d, 0, "rect") / k))
            v = _long_run(d, lag, "rect")
            nw.append(math.sqrt(v / k) if v > 0 else float("nan"))
        good = [x for x in nw if x == x]
        out.append((s, k, _sd(means), _mean(iid), _mean(good), len(nw) - len(good)))
    return out


def stride_information():
    rows = [(s, k, _fx(sd, 4), _fx(iid, 4), _fx(nw, 4)) for s, k, sd, iid, nw, _ in stride_data()]
    head = (f"Differentials of 12-bar forecasts over a stretch of {STRIDE_SPAN} bars: xi_0 .. xi_{STRIDE_SPAN - 1} are {STRIDE_SPAN} calls of normal() "
            f"({_seed_text(STRIDE_NAME)}, a fresh generator for each stride, {STRIDE_REPS} replicates in turn), the origin number t of stride s has "
            f"d_t = (1 / sqrt(12)) (xi_(s t) + ... + xi_(s t + 11)), and the origins are t = 0 .. K - 1 with K = floor(({STRIDE_SPAN} - 12) / s) + 1. "
            f"'Spread of the mean' is the standard deviation (divisor {STRIDE_REPS - 1}) of the mean of d over the replicates, the true size of the "
            f"error bar. The two error-bar columns are the mean over the replicates of sqrt(estimate / K) where the estimate is the independent-origins "
            f"variance gamma_0 or the rectangular long-run variance with lag ceil(12 / s) - 1 (the rule `tools/forecast/oos_skill_report.py` uses "
            f"for the number of lags).\n\n")
    return head + _table(["stride s", "origins K", "spread of the mean of d", "error bar, independent origins", "error bar, overlap-aware"], rows)


# ------------------------------------------------------------------ how many series


@lru_cache(maxsize=None)
def _panel():
    """Per series of the panel, the loss of SND and AR at every origin 60, 72, ..., 228 (15 origins)."""
    rng = bookrng.generator(PANEL_NAME)
    out = []
    for _ in range(PANEL_SERIES):
        y = ts._draw_series(rng, N)
        snd, ar = [], []
        for o in range(FIRST_ORIGIN, LAST_ORIGIN + 1, 12):
            snd.append(_mae(ts._fc_snaive_drift(y[:o], H), y[o:o + H]))
            ar.append(_mae(ts._fc_ar(y[:o], H), y[o:o + H]))
        out.append((snd, ar))
    return out


def _origin_subsets():
    return (("1 origin (o = 144)", [7]), ("5 origins (60, 96, 132, 168, 204)", [0, 3, 6, 9, 12]),
            ("15 origins (60, 72, ..., 228)", list(range(15))))


def power_data():
    """For the figure: for each origin subset, the per-series differences (SND - AR averaged over the origins) and the power by series count."""
    panel = _panel()
    out = []
    for name, idx in _origin_subsets():
        diffs = [_mean([snd[i] - ar[i] for i in idx]) for snd, ar in panel]
        power = []
        rng = bookrng.generator(f"{POWER_NAME}")
        for n in POWER_SIZES:
            hits = 0
            for _ in range(POWER_REPS):
                samp = [diffs[rng.integer(PANEL_SERIES)] for _ in range(n)]
                sd = _sd(samp)
                if sd > 0 and _mean(samp) / (sd / math.sqrt(n)) > Z975:
                    hits += 1
            power.append(hits / POWER_REPS)
        out.append((name, diffs, power))
    return out


def how_many_series():
    z_power = statistics.NormalDist().inv_cdf(0.8)
    rows = []
    for name, diffs, power in power_data():
        m, s = _mean(diffs), _sd(diffs)
        need = math.ceil(((Z975 + z_power) * s / m) ** 2)
        rows.append((name, _fs(m), _fx(s), need) + tuple(_fx(p, 3) for p in power))
    head = (f"{PANEL_SERIES} series drawn from one generator ({_seed_text(PANEL_NAME)}), each by the recipe of Chapter 1 with n = {N} (its {N} calls of "
            f"normal() in turn, series after series). On each series, SND and AR are evaluated at the origins of the row (h = 12, expanding past, as in "
            f"the first example); the series' difference is the mean over those origins of (MAE of SND) - (MAE of AR), positive when AR is better. "
            f"'Needed' is the number of series ceil(((1.959964 + 0.841621) * s / m)^2) for a two-sided test at the 5 percent level to find a mean "
            f"difference m with probability 0.8 when the differences have standard deviation s (divisor {PANEL_SERIES - 1}). The last columns are the "
            f"share of {POWER_REPS} draws of N series with replacement from the {PANEL_SERIES} differences for which mean / (sd / sqrt(N)) exceeds 1.959964 "
            f"(sd with divisor N - 1); one generator ({_seed_text(POWER_NAME)}) draws the rows in order, row after row, N after N, each draw being N calls "
            f"of rng.integer({PANEL_SERIES}).\n\n")
    return head + _table(["origins per series", "mean difference m", "s", "series needed"] + [f"share significant, N = {n}" for n in POWER_SIZES], rows)


# ------------------------------------------------------------------ a selection that spends the test set


def selection_spent():
    panel = _panel()
    first, second = list(range(0, 7)), list(range(7, 15))
    a = [_mean([snd[i] - ar[i] for i in first]) for snd, ar in panel]
    b = [_mean([snd[i] - ar[i] for i in second]) for snd, ar in panel]
    order = sorted(range(PANEL_SERIES), key=lambda i: (-a[i], i))
    top = order[:20]
    rows = [("all 200 series", _fs(_mean(a), 4), _fs(_mean(b), 4)),
            ("the 20 series with the largest difference on the first 7 origins", _fs(_mean([a[i] for i in top]), 4),
             _fs(_mean([b[i] for i in top]), 4)),
            ("the 20 series with the smallest difference on the first 7 origins", _fs(_mean([a[i] for i in order[-20:]]), 4),
             _fs(_mean([b[i] for i in order[-20:]]), 4))]
    head = (f"The {PANEL_SERIES} series of the previous example ({_seed_text(PANEL_NAME)}). For each series, a is the mean over the first 7 of its 15 "
            f"origins (60, 72, ..., 120) of (MAE of SND) - (MAE of AR), and b the mean over the last 8 (132, 144, ..., 228); positive favours AR. Series are "
            f"ranked by a, largest first, a tie to the lower series number. Each row is the mean of a and of b over the series it names.\n\n")
    return head + _table(["series chosen", "mean of a (origins used to choose)", "mean of b (origins not used)"], rows)


# ------------------------------------------------------------------ cherry-picked horizons


def _random_walk(rng, n):
    x, out = 0.0, []
    for t in range(n):
        x = rng.normal() if t == 0 else x + rng.normal()
        out.append(x)
    return out


def cherry_horizons():
    rng = bookrng.generator(CHERRY_NAME)
    skills, zs = [], []
    for _ in range(CHERRY_SERIES):
        y = _random_walk(rng, CHERRY_BARS)
        srow, zrow = [], []
        for h in range(1, 13):
            drift_err, naive_err = [], []
            for o in range(CHERRY_FIRST, CHERRY_BARS - h + 1):
                last = y[o - 1]
                slope = (last - y[0]) / (o - 1)
                drift_err.append(abs(last + slope * h - y[o + h - 1]))
                naive_err.append(abs(last - y[o + h - 1]))
            d = [b - a for a, b in zip(drift_err, naive_err)]
            k = len(d)
            v = _long_run(d, h - 1, "rect")
            srow.append(1.0 - sum(drift_err) / sum(naive_err))
            zrow.append(_mean(d) / math.sqrt(v / k) if v > 0 else 0.0)
        skills.append(srow)
        zs.append(zrow)
    cells = [x for r in skills for x in r]
    zc = sorted((_p_gauss(z), z) for r in zs for z in r)
    m = len(zc)
    holm = 0
    for i, (p, z) in enumerate(zc):
        if p * (m - i) > 0.05:
            break
        holm += z > 0
    unadj = sum(1 for r in zs for z in r if z > Z975)
    rows = [("every one of the 480 cells (the honest average)", _fs(_mean(cells))),
            ("each series at its best horizon, averaged over the 40 series", _fs(_mean([max(r) for r in skills]))),
            ("the single best cell of the 480", _fs(max(cells)))]
    t1 = _table(["what is reported", "skill of drift against naive"], rows)
    rows2 = [("cells with a positive statistic above 1.959964 (unadjusted)", unadj), ("cells with a negative statistic below -1.959964",
                                                                                      sum(1 for r in zs for z in r if z < -Z975)),
             ("positive cells that Holm's rule keeps at the 5 percent level", holm),
             ("series with at least one horizon of positive skill", sum(1 for r in skills if max(r) > 0))]
    t2 = _table(["count over the 480 cells", "cells or series"], rows2)
    head = (f"{CHERRY_SERIES} driftless random walks of {CHERRY_BARS} bars ({_seed_text(CHERRY_NAME)}, one generator, series after series): "
            f"s_0 = normal(), s_t = s_(t-1) + normal(). For a series and a horizon h = 1 .. 12, the origins are o = {CHERRY_FIRST} .. {CHERRY_BARS} - h; "
            f"naive forecasts bar o + h - 1 as y_(o-1), drift as y_(o-1) + h (y_(o-1) - y_0) / (o - 1), and the error of each is its absolute error. "
            f"Skill is 1 - (sum of drift errors) / (sum of naive errors) over the origins; the statistic of a cell is the rectangular statistic of "
            f"d_o = (naive error) - (drift error) with lag h - 1 (lag 0 for h = 1) and no small-sample factor, so a positive statistic favours drift. "
            f"The 480 cells are ranked by two-sided Gaussian p-value, and Holm's rule multiplies the r-th smallest (r = 1, 2, ...) by 480 - r + 1 "
            f"and stops at the first product above 0.05.\n\n")
    return head + t1 + "\n\n" + t2


# ------------------------------------------------------------------ the selection inside a benchmark score

SEL_WINDOWS, SEL_CONTEXT, SEL_HORIZON = 64, 200, 10


def benchmark_selection():
    rng = bookrng.generator(SELECT_NAME)
    chosen, drift_only = [], []
    for _ in range(SELECT_REPS):
        sum_naive = sum_best = sum_drift = 0.0
        for _ in range(SEL_WINDOWS):
            s = _random_walk(rng, SEL_CONTEXT + SEL_HORIZON)
            ctx, fut = s[:SEL_CONTEXT], s[SEL_CONTEXT:]
            last = ctx[-1]
            slope = (last - ctx[0]) / (SEL_CONTEXT - 1)
            en = _mae([last] * SEL_HORIZON, fut)
            ed = _mae([last + slope * k for k in range(1, SEL_HORIZON + 1)], fut)
            sum_naive += en
            sum_best += min(en, ed)
            sum_drift += ed
        chosen.append(1.0 - sum_best / sum_naive)
        drift_only.append(1.0 - sum_drift / sum_naive)
    rows = [("drift, used as it is", _fs(_mean(drift_only))),
            ("the better of naive and drift, chosen window by window with the answer in hand", _fs(_mean(chosen)))]
    head = (f"{SELECT_REPS} repetitions in turn from one generator ({_seed_text(SELECT_NAME)}); a repetition is {SEL_WINDOWS} windows in turn, "
            f"each a driftless random walk of {SEL_CONTEXT + SEL_HORIZON} bars (s_0 = normal(), s_t = s_(t-1) + normal()), the first {SEL_CONTEXT} as the "
            f"context and the last {SEL_HORIZON} as the future. Naive repeats the last context value; drift continues the line through the first and last "
            f"context bars, last + k (last - first) / {SEL_CONTEXT - 1} for k = 1 .. {SEL_HORIZON}. A window's error is the mean absolute error over the "
            f"{SEL_HORIZON} bars. The score of a repetition is 1 - (sum over windows of the error of the forecaster) / (sum over windows of the naive error); "
            f"the table gives its mean over the {SELECT_REPS} repetitions.\n\n")
    tail = (f"\n\nThe largest score of the better-of-two forecaster over the {SELECT_REPS} repetitions is {_fs(max(chosen))}; the share of repetitions "
            f"above 0.1 is {_fx(sum(x > 0.1 for x in chosen) / SELECT_REPS, 3)}.")
    return head + _table(["forecaster", f"mean score over {SELECT_REPS} repetitions"], rows) + tail


# ------------------------------------------------------------------ a leak that a negative control catches


def _nearest(y, train_origins, o, c, h):
    """The h bars that followed the training window whose c-bar context is nearest (squared Euclidean distance, earliest on a tie) to y[o-c:o]."""
    q = y[o - c:o]
    best, best_d = None, None
    for p in train_origins:
        d = sum((a - b) ** 2 for a, b in zip(y[p - c:p], q))
        if best_d is None or d < best_d:
            best, best_d = p, d
    return y[best:best + h]


def control_leak():
    rng = bookrng.generator(CONTROL_NAME)
    errs = {"naive": [], "leaky": [], "purged": []}
    for _ in range(CONTROL_SERIES):
        y = _random_walk(rng, N)
        acc = {k: [] for k in errs}
        for o in range(96, LAST_ORIGIN + 1, 12):
            fut = y[o:o + H]
            acc["naive"].append(_mae([y[o - 1]] * H, fut))
            acc["leaky"].append(_mae(_nearest(y, list(range(CONTEXT, o)), o, CONTEXT, H), fut))
            acc["purged"].append(_mae(_nearest(y, list(range(CONTEXT, o - H + 1)), o, CONTEXT, H), fut))
        for k in errs:
            errs[k].append(_mean(acc[k]))
    rows = []
    for name, key in (("naive", "naive"), ("nearest window, every earlier window as training", "leaky"),
                      ("nearest window, windows purged at the fit boundary", "purged")):
        e = errs[key]
        if key == "naive":
            rows.append((name, _fx(_mean(e)), "0 by definition", "-"))
            continue
        diff = [a - b for a, b in zip(errs["naive"], e)]
        se = _sd(diff) / math.sqrt(len(diff))
        rows.append((name, _fx(_mean(e)), _fs(_skill(e, errs["naive"])), _fx(se, 3)))
    head = (f"{CONTROL_SERIES} driftless random walks of {N} bars ({_seed_text(CONTROL_NAME)}, one generator, series after series): s_0 = normal(), "
            f"s_t = s_(t-1) + normal(). On each, origins o = 96, 108, ..., {LAST_ORIGIN} (12 origins), horizon 12. The nearest-window forecaster of "
            f"Chapter 1 (context {CONTEXT} bars, raw values, squared Euclidean distance, the earliest origin on a tie) copies the 12 bars that followed "
            f"the nearest training window; its training windows have origins {CONTEXT} .. o - 1 (every window that starts before the forecast) or "
            f"{CONTEXT} .. o - 12 (the purged rule). A series' score is the mean of its 12 origin errors (mean absolute error over 12 bars); the table "
            f"gives the mean over the series, the skill 1 - (mean error) / (mean naive error), and the standard error of the mean paired difference "
            f"(naive error minus the row's error, divisor {CONTROL_SERIES - 1} in the standard deviation, over sqrt({CONTROL_SERIES})).\n\n")
    return head + _table(["forecaster", "mean error", "skill against naive", "standard error of the paired difference"], rows)
