# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Probabilistic Forecasts".

Plain Python 3, standard library only, written from the chapter's own description: a stocking decision
that a point forecast cannot make and a quantile can, the pinball loss and the sample quantile that
minimises it, a Gaussian and a Student-t head fitted by negative log-likelihood to a series with
occasional large shocks, the continuous ranked probability score as twice the pinball loss integrated
over the levels (and as an ensemble average), the probability integral transform and the coverage of
intervals, linear quantile regression with its quantile crossing and the sorting that repairs it,
sample paths against independently sampled marginals for a sum over a horizon, the coverage of
prediction intervals at each horizon, and split conformal intervals. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: probabilistic_forecasting.<name> -->` markers. Underscore functions and the constants
are shared with the chapter's figures, so a picture and the table beside it cannot disagree. The
series of Chapter 1 (`time_series`) is imported, never copied.

Swedish Embedded AB implements solutions for forecasts that state their own uncertainty and are
checked against what happened. If your team needs expertise in probabilistic forecasting and its
calibration, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import statistics
from functools import lru_cache

import bookrng
import stable
import time_series as ts
from time_series import _mean, _seed_text, _table

# ------------------------------------------------------------------ constants

_NORMAL = statistics.NormalDist()
Z = _NORMAL.inv_cdf                      # the inverse Gaussian cumulative distribution, library routine


def _phi_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


NEWS_COSTS = ((1, 9), (1, 1), (9, 1))    # (cost of one unit too many, cost of one unit too few)

SAMPLE9 = (2, 3, 5, 7, 8, 11, 12, 15, 20)
SAMPLE9_LEVELS = (0.1, 0.5, 0.9)

HEADS_NAME = "probabilistic_forecasting.heads"
PHI, SHOCK = 0.7, 0.6                    # x_t = 0.7 x_(t-1) + 0.6 e_t, as the remainder of Chapter 1
BURST_P, BURST_SCALE = 0.1, 4.0          # with probability 0.1 the shock e_t is four times as large
SHIFT_BAR, SHIFTED_SHOCK = 1803, 1.2     # from this bar on the shock coefficient is 1.2
N_BARS = 2304
TRAIN_ROWS = (1, 400)                    # inclusive target bars of each split; bars 401, 802 and 1803 are dropped as the gap
VAL_ROWS = (402, 801)
TEST_A_ROWS = (803, 1802)
TEST_B_ROWS = (1804, 2303)
NU_GRID = (2, 4, 6, 8, 12, 16)
EM_ITERS = 500
CRPS_LEVELS = 200                        # K midpoint levels (k + 0.5) / K for the integral of the pinball loss
PIT_BINS = 10

QR_TRAIN_ROWS = (1, 21)
QR_LEVELS = tuple(k / 10 for k in range(1, 10))

PATH_NAME, MARGINAL_NAME, PATHS, PATH_H, PATH_X0 = (
    "probabilistic_forecasting.paths", "probabilistic_forecasting.marginals", 4000, 12, 1.5)

COVER_NAME, COVER_SERIES = "probabilistic_forecasting.coverage", 1000
COVER_HORIZONS = (1, 2, 3, 6, 12, 13, 18, 24)

ENSEMBLE_NAME = "probabilistic_forecasting.ensemble"
DERIVED_NAME = "probabilistic_forecasting.derived_samples"
CONFORMAL_NAME = "probabilistic_forecasting.conformal"


# ------------------------------------------------------------------ the pinball loss

def pinball(q, y, tau):
    """max(tau (y - q), (tau - 1) (y - q))."""
    e = y - q
    return max(tau * e, (tau - 1.0) * e)


def _type7(sorted_values, p):
    """The book's percentile: linear interpolation at position p (m - 1) between the order statistics."""
    pos = p * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def _inverse_cdf_quantile(sorted_values, p):
    """The smallest sample value v with (share of the sample <= v) >= p, i.e. element ceil(m p) (counting from 1)."""
    return sorted_values[max(math.ceil(len(sorted_values) * p - 1e-12), 1) - 1]


# ------------------------------------------------------------------ the stocking decision

def _ar1_rms(past):
    z = [past[t] - past[t - ts.SEASON] for t in range(ts.SEASON, len(past))]
    c = ts._fit_ar(z, 1)
    res = [z[t] - c[0] - c[1] * z[t - 1] for t in range(1, len(z))]
    return math.sqrt(sum(r * r for r in res) / len(res))


def _news_rows():
    out = []
    for s in ts._replicate_values():
        past = s[:ts.CUT]
        out.append((ts._fc_ar(past, 1)[0], _ar1_rms(past), s[ts.CUT]))
    return out


def newsvendor():
    rows = _news_rows()
    mu0, sd0, y0 = rows[0]
    first = (f"The first of the 200 series of Chapter 1 gives forecast mean {mu0:.3f}, spread {sd0:.3f} and demand y_96 = {y0:.3f}.")
    table = []
    for cu_over, cu_short in NEWS_COSTS:
        tau = cu_short / (cu_short + cu_over)
        zt = Z(tau)
        c_point = _mean([(cu_over + cu_short) * pinball(m, y, tau) for m, s, y in rows])
        c_quant = _mean([(cu_over + cu_short) * pinball(m + zt * s, y, tau) for m, s, y in rows])
        table.append((cu_over, cu_short, f"{tau:.1f}", f"{zt:+.4f}", f"{c_point:.3f}", f"{c_quant:.3f}"))
    head = (f"The 200 series of Chapter 1 ({_seed_text(ts.REPLICATES_NAME)}), each cut at bar {ts.CUT}. For each series the forecast of bar "
            f"{ts.CUT} (horizon 1) is the AR(1) forecast on seasonal differences of Chapter 1, mu; the spread sigma is the root mean square "
            f"of that model's {ts.CUT - ts.SEASON - 1} one-step training residuals (divisor: their number). The demand is the series' value "
            f"y_{ts.CUT}. An order q costs c_over (q - y) when q > y and c_short (y - q) when y > q. 'order mu' is q = mu; 'order the quantile' "
            f"is q = mu + z_tau sigma with tau = c_short / (c_short + c_over) and z_tau the standard Gaussian quantile (library inverse "
            f"cumulative distribution). Costs are averaged over the 200 series.\n\n{first}\n\n")
    return head + _table(["c_over", "c_short", "tau", "z_tau", "mean cost, order mu", "mean cost, order the quantile"], table)


# ------------------------------------------------------------------ the sample quantile

def pinball_minimiser():
    qs = (2, 5, 8, 12, 15, 16, 20)
    rows = []
    for q in qs:
        rows.append((q, *(f"{_mean([pinball(q, y, t) for y in SAMPLE9]):.4f}" for t in SAMPLE9_LEVELS)))
    srt = sorted(SAMPLE9)
    inv = [_inverse_cdf_quantile(srt, t) for t in SAMPLE9_LEVELS]
    t7 = [_type7(srt, t) for t in SAMPLE9_LEVELS]
    head = (f"The nine values y = {', '.join(str(v) for v in SAMPLE9)}. For each candidate q the table gives the mean over the nine values of "
            f"max(tau (y - q), (tau - 1) (y - q)) at tau = 0.1, 0.5 and 0.9 (divisor 9).\n\n")
    tail = (f"\n\nThe sample quantile of the inverse cumulative distribution (the element at position ceil(9 tau) of the sorted list, counting from 1) is "
            f"{inv[0]}, {inv[1]} and {inv[2]} for tau = 0.1, 0.5, 0.9. The interpolated percentile of the pseudocode chapter "
            f"(position tau (9 - 1) in the sorted list, counting from 0) is {t7[0]:.1f}, {t7[1]:.1f} and {t7[2]:.1f}.")
    return head + _table(["q", "tau = 0.1", "tau = 0.5", "tau = 0.9"], rows) + tail


def pinball_curve(tau, qs):
    """Mean pinball loss of SAMPLE9 at each q of qs (the figure's curve)."""
    return [_mean([pinball(q, y, tau) for y in SAMPLE9]) for q in qs]


# ------------------------------------------------------------------ the data of the heads

@lru_cache(maxsize=1)
def _heads_series():
    """The 1604 bars of the contaminated autoregression, drawn in bar order: uniform(), then normal(), per bar."""
    g = bookrng.generator(HEADS_NAME)
    x, prev = [], 0.0
    for t in range(N_BARS):
        burst = g.uniform() < BURST_P
        shock = g.normal() * (BURST_SCALE if burst else 1.0)
        sd = SHOCK if t < SHIFT_BAR else SHIFTED_SHOCK
        prev = PHI * prev + sd * shock
        x.append(prev)
    return tuple(x)


@lru_cache(maxsize=1)
def _burst_flags():
    """Which bars received the four-times shock (the same draws as `_heads_series`, replayed)."""
    g = bookrng.generator(HEADS_NAME)
    flags = []
    for _ in range(N_BARS):
        flags.append(g.uniform() < BURST_P)
        g.normal()
    return tuple(flags)


def _rows(rows):
    x = _heads_series()
    lo, hi = rows
    return [x[t - 1] for t in range(lo, hi + 1)], [x[t] for t in range(lo, hi + 1)]


# ------------------------------------------------------------------ Gaussian and Student-t heads

def _nll_row(r, nu):
    """Negative log-likelihood of one row with standardised residual r = (y - mu) / sigma, without log sigma."""
    if nu is None:
        return 0.5 * r * r + 0.5 * math.log(2.0 * math.pi)
    return ((nu + 1) / 2.0) * math.log1p(r * r / nu) + math.lgamma(nu / 2.0) - math.lgamma((nu + 1) / 2.0) + 0.5 * math.log(nu * math.pi)


def _mean_nll(head, xs, ys, nu):
    a, b, s = head
    return _mean([math.log(s) + _nll_row((y - a - b * x) / s, nu) for x, y in zip(xs, ys)])


def _fit_head(xs, ys, nu, a0=0.0):
    """mu = a + b x, scale sigma, by the fixed point of the stationarity conditions of the NLL: weights
    w_i = (nu + 1) / (nu + r_i^2) (1 for the Gaussian), a weighted least-squares line, sigma^2 = sum w_i res_i^2 / n."""
    n = len(xs)
    a, b, var = a0, 0.0, 1.0
    for _ in range(EM_ITERS):
        w = [1.0 if nu is None else (nu + 1) / (nu + (y - a - b * x) ** 2 / var) for x, y in zip(xs, ys)]
        sw = sum(w)
        sx = sum(wi * x for wi, x in zip(w, xs))
        sy = sum(wi * y for wi, y in zip(w, ys))
        sxx = sum(wi * x * x for wi, x in zip(w, xs))
        sxy = sum(wi * x * y for wi, x, y in zip(w, xs, ys))
        b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
        a = (sy - b * sx) / sw
        var = sum(wi * (y - a - b * x) ** 2 for wi, x, y in zip(w, xs, ys)) / n
    return a, b, math.sqrt(var)


def _head_gradient(head, xs, ys, nu):
    """The mean gradient of the NLL with respect to (a, b, log sigma); zero at the fitted head."""
    a, b, s = head
    ga = gb = gs = 0.0
    for x, y in zip(xs, ys):
        r = (y - a - b * x) / s
        k = 1.0 if nu is None else (nu + 1) / (nu + r * r)
        ga += -k * r / s
        gb += -k * r * x / s
        gs += 1.0 - k * r * r
    n = len(xs)
    return ga / n, gb / n, gs / n


NUDGES = stable.NUDGES


def _fit_all():
    """{nu or None: head} on the training rows."""
    xs, ys = _rows(TRAIN_ROWS)
    return {nu: _fit_head(xs, ys, nu) for nu in (None, *NU_GRID)}


@lru_cache(maxsize=1)
def _heads():
    return _fit_all()


@lru_cache(maxsize=1)
def _nudged_heads():
    """The same fits started from a0 = +-1e-12, +-3e-12 instead of 0: the stable-digits check."""
    xs, ys = _rows(TRAIN_ROWS)
    return [{nu: _fit_head(xs, ys, nu, a0=d) for nu in (None, *NU_GRID)} for d in NUDGES]


def _label(nu):
    return "Gaussian" if nu is None else f"Student-t, nu = {nu}"


def _num(fmt):
    return lambda v: format(v, fmt)


def head_fits():
    xs, ys = _rows(TRAIN_ROWS)
    vx, vy = _rows(VAL_ROWS)
    heads, nud = _heads(), _nudged_heads()
    rows = []
    for nu in (None, *NU_GRID):
        h = heads[nu]
        cells = []
        for i, fmt in enumerate(("+.3f", "+.3f", ".3f")):
            cells.append(stable.text(h[i], [d[nu][i] for d in nud], _num(fmt)))
        tr = _mean_nll(h, xs, ys, nu)
        va = _mean_nll(h, vx, vy, nu)
        trs = [_mean_nll(d[nu], xs, ys, nu) for d in nud]
        vas = [_mean_nll(d[nu], vx, vy, nu) for d in nud]
        gmax = max(abs(g) for g in _head_gradient(h, xs, ys, nu))
        rows.append((_label(nu), *cells, stable.text(tr, trs, _num(".4f")), stable.text(va, vas, _num(".4f")), f"{'below 1e-9' if gmax < 1e-9 else f'{gmax:.1e}'}"))
    x = _heads_series()
    burst_note = (f"The data are the {N_BARS} bars x_0 .. x_{N_BARS - 1} of x_t = 0.7 x_(t-1) + c_t e_t with x_(-1) = 0, where c_t = {SHOCK} for "
                  f"t < {SHIFT_BAR} and {SHIFTED_SHOCK} from bar {SHIFT_BAR} on, and the shock e_t is drawn per bar as uniform() first and then "
                  f"normal(), e_t = normal() when the uniform is at least {BURST_P} and {BURST_SCALE:g} normal() when it is below. "
                  f"Generator: the book's SplitMix64, {_seed_text(HEADS_NAME)}; a normal() call that finds a held value returns it without drawing. ")
    head = (burst_note +
            f"A row is (feature x_(t-1), target x_t). Training rows are t = {TRAIN_ROWS[0]} .. {TRAIN_ROWS[1]} ({TRAIN_ROWS[1] - TRAIN_ROWS[0] + 1} rows), "
            f"validation rows t = {VAL_ROWS[0]} .. {VAL_ROWS[1]} ({VAL_ROWS[1] - VAL_ROWS[0] + 1} rows); bar t = {TRAIN_ROWS[1] + 1} is dropped between them. "
            f"Each head is mu = a + b x_(t-1) with a scale sigma, the Gaussian N(mu, sigma^2) or the Student-t with the stated nu degrees of freedom, "
            f"location mu and scale sigma. It is fitted on the training rows by {EM_ITERS} rounds of the fixed point described in the text, started from "
            f"a = 0, b = 0, sigma = 1 (the stable-digits rule starts four more fits from a = +-1e-12, +-3e-12). 'NLL' is the mean over the rows of "
            f"-log of the density; the last column is the largest absolute entry of the mean gradient of the NLL with respect to (a, b, log sigma) "
            f"at the fitted head.\n\n")
    fl = _burst_flags()
    bursts = ", ".join(f"{name} {sum(fl[lo:hi + 1])} of {hi - lo + 1}" for name, (lo, hi) in
                       (("training", TRAIN_ROWS), ("validation", VAL_ROWS), ("test A", TEST_A_ROWS), ("test B", TEST_B_ROWS)))
    return (head + _table(["head", "a", "b", "sigma", "training NLL", "validation NLL", "gradient"], rows) +
            f"\n\nBars with the four-times shock, by block of target bars: {bursts}.\n\n" + stable.RULE)


def _best_nu():
    """The degrees of freedom with the smallest validation NLL in the grid."""
    vx, vy = _rows(VAL_ROWS)
    heads = _heads()
    return min(NU_GRID, key=lambda nu: (_mean_nll(heads[nu], vx, vy, nu), nu))


# ------------------------------------------------------------------ the Student-t distribution function (even nu)

def t_cdf(x, nu):
    """Cumulative distribution of the standard Student-t with an even number nu of degrees of freedom (a finite sum)."""
    u = nu / (nu + x * x)
    total, term = 0.0, 1.0
    for k in range(nu // 2):
        total += term
        term *= (2 * k + 1) / (2 * k + 2) * u
    return 0.5 + x / (2.0 * math.sqrt(nu + x * x)) * total


def t_quantile(p, nu):
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cdf(r, nu):
    return _phi_cdf(r) if nu is None else t_cdf(r, nu)


def _quant(p, nu):
    return Z(p) if nu is None else t_quantile(p, nu)


@lru_cache(maxsize=None)
def _std_levels(nu, k=CRPS_LEVELS):
    """The standardised quantiles at the K midpoint levels (j + 0.5) / K."""
    return tuple(_quant((j + 0.5) / k, nu) for j in range(k))


def crps_by_levels(r, nu, k=CRPS_LEVELS):
    """2 * mean over the K midpoint levels of the pinball loss of the standardised quantile at the standardised residual r."""
    zs = _std_levels(nu, k)
    return 2.0 * _mean([pinball(z, r, (j + 0.5) / k) for j, z in enumerate(zs)])


def crps_gaussian(r):
    """Closed form of the CRPS of N(0, 1) at r."""
    return r * (2.0 * _phi_cdf(r) - 1.0) + 2.0 * math.exp(-r * r / 2.0) / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)


def crps_levels_check():
    rows = []
    for r in (0.0, 1.0, 3.0):
        closed = crps_gaussian(r)
        cells = [f"{closed:.6f}"]
        for k in (9, 99, 999):
            cells.append(f"{crps_by_levels(r, None, k):.6f}")
        rows.append((f"{r:g}", *cells))
    head = ("CRPS of N(0, 1) at the observation r. 'closed form' is r (2 Phi(r) - 1) + 2 phi(r) - 1 / sqrt(pi) with the library error function; "
            "the other columns are 2 times the mean over the K levels (j + 0.5) / K, j = 0 .. K-1, of the pinball loss at level tau of the "
            "standard Gaussian quantile at tau (library inverse cumulative distribution).\n\n")
    return head + _table(["r", "closed form", "K = 9", "K = 99", "K = 999"], rows)


# ------------------------------------------------------------------ the ensemble form

def crps_ensemble(samples, y, fair=False):
    """E|X - y| - 0.5 E|X - X'| over an ensemble, the pair average over n^2 ordered pairs (fair=False) or n (n - 1) (fair=True)."""
    n = len(samples)
    e1 = sum(abs(s - y) for s in samples) / n
    e2 = sum(abs(a - b) for a in samples for b in samples) / (n * (n - 1) if fair else n * n)
    return e1 - 0.5 * e2


def ensemble_table():
    g = bookrng.generator(ENSEMBLE_NAME)
    truth = crps_gaussian(0.0)
    rows = []
    for n in (4, 20, 200, 1000):
        samples = [g.normal() for _ in range(n)]
        rows.append((n, f"{crps_ensemble(samples, 0.0):.4f}", f"{crps_ensemble(samples, 0.0, True):.4f}"))
    head = (f"The CRPS of N(0, 1) at r = 0 is {truth:.4f}. An ensemble of n draws of normal() from one generator ({_seed_text(ENSEMBLE_NAME)}), "
            f"the four ensembles drawn one after the other, is scored with E|X - 0| - 0.5 E|X - X'|: 'divisor n^2' averages |X - X'| over all n^2 ordered "
            f"pairs, the diagonal included; 'divisor n(n-1)' leaves the diagonal out.\n\n")
    return head + _table(["n", "divisor n^2", "divisor n(n-1)"], rows)


# ------------------------------------------------------------------ samples derived from quantiles

def _derived_inverse_cdf(levels, values, u):
    if u <= levels[0]:
        return values[0]
    if u >= levels[-1]:
        return values[-1]
    for k in range(1, len(levels)):
        if u <= levels[k]:
            frac = (u - levels[k - 1]) / (levels[k] - levels[k - 1])
            return values[k - 1] + (values[k] - values[k - 1]) * frac
    return values[-1]


def derived_samples():
    g = bookrng.generator(DERIVED_NAME)
    n = 2000
    grids = {"3 levels": (0.1, 0.5, 0.9), "9 levels": tuple(k / 10 for k in range(1, 10)),
             "21 levels": (0.01, 0.05, *(k / 20 for k in range(2, 19)), 0.95, 0.99)}
    us = [g.uniform() for _ in range(n)]
    rows = []
    for r in (0.0, 1.0, 2.0, 3.0):
        cells = [f"{crps_gaussian(r):.4f}"]
        for name, lv in grids.items():
            vals = [Z(p) for p in lv]
            ens = sorted(_derived_inverse_cdf(lv, vals, u) for u in us)
            cells.append(f"{_fair_crps_sorted(ens, r):.4f}")
        rows.append((f"{r:g}", *cells))
    atoms = {name: sum(1 for u in us if u <= lv[0] or u >= lv[-1]) / n for name, lv in grids.items()}
    head = (f"The quantile levels of the grid, with the exact standard Gaussian quantiles as values, are turned into {n} samples by the piecewise-linear "
            f"inverse cumulative distribution that is flat outside the first and last level (the value at the first level for u at or below it, the value at "
            f"the last level for u at or above it). The uniforms u_1 .. u_{n} are drawn in order by uniform() from one generator "
            f"({_seed_text(DERIVED_NAME)}) and the same u are used for all three grids. The grids are 0.1, 0.5, 0.9; 0.1, 0.2, ..., 0.9; and 0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99. Each column scores the ensemble with E|X - r| - 0.5 E|X - X'| "
            f"(average over the n(n-1) pairs of different draws, evaluated on the sorted ensemble) at the observation r; the first column is the CRPS "
            f"of N(0, 1).\n\n")
    tail = (f"\n\nShare of the samples that sit exactly on the first or last quantile value: {atoms['3 levels']:.3f} for the 3-level grid, "
            f"{atoms['9 levels']:.3f} for the 9-level grid and {atoms['21 levels']:.3f} for the 21-level grid.")
    return head + _table(["observation r", "CRPS of N(0, 1)", "from 3 levels", "from 9 levels", "from 21 levels"], rows) + tail


def _fair_crps_sorted(ens, y):
    """The n (n - 1) pair form on a sorted ensemble: sum_{i<j} |x_j - x_i| = sum_i (2 i - n + 1) x_i."""
    n = len(ens)
    e1 = sum(abs(s - y) for s in ens) / n
    pair = sum((2 * i - n + 1) * x for i, x in enumerate(ens))
    e2 = 2.0 * pair / (n * (n - 1))
    return e1 - 0.5 * e2


# ------------------------------------------------------------------ scoring the heads on the test rows

def _scores(nu, head, rows):
    xs, ys = _rows(rows)
    a, b, s = head
    nll = crps = c80 = c95 = 0.0
    pit = [0] * PIT_BINS
    z80, z95 = _quant(0.9, nu), _quant(0.975, nu)
    for x, y in zip(xs, ys):
        r = (y - a - b * x) / s
        nll += math.log(s) + _nll_row(r, nu)
        crps += s * (crps_gaussian(r) if nu is None else crps_by_levels(r, nu))
        c80 += abs(r) <= z80
        c95 += abs(r) <= z95
        pit[min(int(_cdf(r, nu) * PIT_BINS), PIT_BINS - 1)] += 1
    n = len(xs)
    return nll / n, crps / n, c80 / n, c95 / n, pit


def scored_heads():
    """(label, nu, head, scores on test A, scores on test B) for the Gaussian and the chosen Student-t head."""
    heads = _heads()
    nu = _best_nu()
    return [(_label(k), k, heads[k], _scores(k, heads[k], TEST_A_ROWS), _scores(k, heads[k], TEST_B_ROWS)) for k in (None, nu)]


def head_scores():
    nu = _best_nu()
    rows = []
    for label, k, h, a, b in scored_heads():
        rows.append((label, f"{a[0]:.3f}", f"{a[1]:.3f}", f"{a[2]:.3f}", f"{a[3]:.3f}"))
    na = TEST_A_ROWS[1] - TEST_A_ROWS[0] + 1
    head = (f"The heads of the previous table, scored on the {na} test rows t = {TEST_A_ROWS[0]} .. {TEST_A_ROWS[1]} (the same recipe as the training rows; the "
            f"Student-t head is the grid value nu = {nu} with the smallest validation NLL). 'NLL' is the mean negative log-density; 'CRPS' is "
            f"the mean over the rows of sigma times the standardised CRPS (closed form for the Gaussian; for the Student-t, 2 times the mean pinball loss "
            f"over the {CRPS_LEVELS} levels (j + 0.5) / {CRPS_LEVELS} of its quantile function, found by bisection on the finite-sum distribution function). "
            f"'80 percent' and '95 percent' are the shares of rows whose target lies inside the central interval mu +- sigma q, with q the standard "
            f"quantile at 0.9 and 0.975 of the head's own distribution.\n\n")
    return head + _table(["head", "NLL", "CRPS", "inside the 80 percent interval", "inside the 95 percent interval"], rows)


def pit_counts():
    """{label: counts of the probability integral transform in ten bins, test A}."""
    return {label: a[4] for label, k, h, a, b in scored_heads()}


def pit_table():
    nu = _best_nu()
    sc = scored_heads()
    rows = []
    for j in range(PIT_BINS):
        rows.append((f"{j / PIT_BINS:.1f} .. {(j + 1) / PIT_BINS:.1f}", *(a[4][j] for _, _, _, a, _ in sc)))
    n = TEST_A_ROWS[1] - TEST_A_ROWS[0] + 1
    head = (f"For each test row t = {TEST_A_ROWS[0]} .. {TEST_A_ROWS[1]} the value u = F(y), F the head's predictive distribution function at the target "
            f"(library error function for the Gaussian, the finite sum for the Student-t head with nu = {nu}); the table counts the u in ten equal bins "
            f"(u = 1 goes into the last). A calibrated head puts {n // PIT_BINS} of {n} rows in each bin on average, with a standard deviation of "
            f"{math.sqrt(n * 0.1 * 0.9):.1f}.\n\n")
    return head + _table(["bin of u", *(label for label, *_ in sc)], rows)


# ------------------------------------------------------------------ linear quantile regression and crossing

def _fit_quantile_line(xs, ys, tau):
    """The line a + b x minimising the sum of pinball losses at tau, by enumerating the lines through two rows (a vertex of the piecewise-linear loss);
    the first pair in the order (i, j), i < j, wins a tie."""
    best, best_loss = None, float("inf")
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            if xs[i] == xs[j]:
                continue
            b = (ys[j] - ys[i]) / (xs[j] - xs[i])
            a = ys[i] - b * xs[i]
            loss = sum(pinball(a + b * x, y, tau) for x, y in zip(xs, ys))
            if loss < best_loss:
                best, best_loss = (a, b), loss
    return best


@lru_cache(maxsize=1)
def _qr_lines():
    xs, ys = _rows(QR_TRAIN_ROWS)
    return {tau: _fit_quantile_line(xs, ys, tau) for tau in QR_LEVELS}


def _qr_matrix(xs):
    lines = _qr_lines()
    return [[lines[t][0] + lines[t][1] * x for t in QR_LEVELS] for x in xs]


def _mean_pinball_rows(mat, ys):
    return _mean([_mean([pinball(q, y, t) for q, t in zip(row, QR_LEVELS)]) for row, y in zip(mat, ys)])


def quantile_regression():
    lines = _qr_lines()
    xs, ys = _rows(TEST_A_ROWS)
    tx, ty = _rows(QR_TRAIN_ROWS)
    raw = _qr_matrix(xs)
    srt = [sorted(r) for r in raw]
    trn = _qr_matrix(tx)
    rows = []
    for j, t in enumerate(QR_LEVELS):
        a, b = lines[t]
        below = sum(1 for r, y in zip(raw, ys) if y <= r[j]) / len(ys)
        below_s = sum(1 for r, y in zip(srt, ys) if y <= r[j]) / len(ys)
        cross = sum(1 for r in raw if j + 1 < len(r) and r[j + 1] < r[j])
        rows.append((f"{t:.1f}", f"{a:+.3f}", f"{b:+.3f}", f"{below:.3f}", f"{below_s:.3f}", cross if j + 1 < len(QR_LEVELS) else "-"))
    crossing_rows = sum(1 for r in raw if any(r[k + 1] < r[k] for k in range(len(r) - 1)))
    head = (f"The chain of the previous section: training rows t = {QR_TRAIN_ROWS[0]} .. {QR_TRAIN_ROWS[1]} ({QR_TRAIN_ROWS[1] - QR_TRAIN_ROWS[0] + 1} rows), "
            f"one line q_tau(x) = a + b x per level tau = 0.1, 0.2, ..., 0.9, each the line that minimises the sum of the pinball losses at its own level "
            f"(found by trying the line through every pair of rows and keeping the smallest loss; no tie occurs). Scored on the {len(ys)} test rows "
            f"t = {TEST_A_ROWS[0]} .. {TEST_A_ROWS[1]}: 'share below' is the share of rows whose target is at or below the line, as it comes out of the "
            f"fit and after the sorting of the nine values of each row into ascending order; 'crossings' counts the test rows at which the line of the next "
            f"level lies below this one.\n\n")
    tail = (f"\n\nTest rows with at least one crossing: {crossing_rows} of {len(ys)}. Mean pinball loss over the nine levels and the test rows: "
            f"{_mean_pinball_rows(raw, ys):.4f} as fitted, {_mean_pinball_rows(srt, ys):.4f} after sorting; on the training rows "
            f"{_mean_pinball_rows(trn, ty):.4f}.")
    demo = _qr_crossing_demo()
    if demo:
        x, raw_row, sorted_row = demo
        tail += (f"\n\nThe first test row with a crossing has x = {x:.3f}. Its nine values as fitted: {', '.join(f'{v:.3f}' for v in raw_row)}; "
                 f"after sorting: {', '.join(f'{v:.3f}' for v in sorted_row)}.")
    return head + _table(["tau", "a", "b", "share below", "share below, sorted", "crossings with the next level"], rows) + tail


def _qr_crossing_demo():
    """(x of the first test row with a crossing, its raw nine quantiles, the sorted ones) or None."""
    xs, ys = _rows(TEST_A_ROWS)
    for x, r in zip(xs, _qr_matrix(xs)):
        if any(r[k + 1] < r[k] for k in range(len(r) - 1)):
            return x, r, sorted(r)
    return None


# ------------------------------------------------------------------ paths against marginals

def _path_exact():
    """Exact (means m_h, variances v_h, variance of the sum) of the AR(1) forecast from x0 over PATH_H bars."""
    m = [PATH_X0 * PHI ** (h + 1) for h in range(PATH_H)]
    v = [SHOCK ** 2 * (1 - PHI ** (2 * (h + 1))) / (1 - PHI ** 2) for h in range(PATH_H)]
    cov = [[SHOCK ** 2 * PHI ** abs(i - j) * (1 - PHI ** (2 * (min(i, j) + 1))) / (1 - PHI ** 2) for j in range(PATH_H)] for i in range(PATH_H)]
    return m, v, sum(sum(r) for r in cov), cov


def path_sums():
    """(sums from sample paths, sums from independently sampled marginals)."""
    m, v, _, _ = _path_exact()
    g = bookrng.generator(PATH_NAME)
    paths = []
    for _ in range(PATHS):
        x, total = PATH_X0, 0.0
        for _ in range(PATH_H):
            x = PHI * x + SHOCK * g.normal()
            total += x
        paths.append(total)
    g2 = bookrng.generator(MARGINAL_NAME)
    marg = []
    for _ in range(PATHS):
        total = 0.0
        for h in range(PATH_H):
            total += m[h] + math.sqrt(v[h]) * g2.normal()
        marg.append(total)
    return paths, marg


def paths_vs_marginals():
    m, v, vs, cov = _path_exact()
    paths, marg = path_sums()
    mean_exact = sum(m)
    sd_exact = math.sqrt(vs)
    sd_indep = math.sqrt(sum(v))
    thresh = 12.0
    rows = []
    for name, sums in (("exact (Gaussian)", None), ("sample paths", paths), ("independent marginals", marg)):
        if sums is None:
            mu, sd = mean_exact, sd_exact
            lo, hi = mu + Z(0.05) * sd, mu + Z(0.95) * sd
            ex = 1 - _phi_cdf((thresh - mu) / sd)
        else:
            mu = _mean(sums)
            sd = math.sqrt(sum((s - mu) ** 2 for s in sums) / (len(sums) - 1))
            srt = sorted(sums)
            lo, hi = _type7(srt, 0.05), _type7(srt, 0.95)
            ex = sum(1 for s in sums if s > thresh) / len(sums)
        rows.append((name, f"{mu:.3f}", f"{sd:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{ex:.4f}"))
    head = (f"The remainder of Chapter 1, u_t = 0.7 u_(t-1) + 0.6 e_t, observed at u_(-1) = {PATH_X0} just before the origin; the {PATH_H} bars after it. "
            f"Forecast of bar h (h = 0 .. {PATH_H - 1}): Gaussian with mean m_h = {PATH_X0} 0.7^(h+1) and variance v_h = 0.36 (1 - 0.49^(h+1)) / (1 - 0.49). "
            f"'sample paths': {PATHS} paths, each drawing its {PATH_H} shocks in order from one generator ({_seed_text(PATH_NAME)}), u_h = 0.7 u_(h-1) + 0.6 normal(), "
            f"and summing the {PATH_H} values. 'independent marginals': {PATHS} samples, each the sum over h of m_h + sqrt(v_h) normal() with the "
            f"normal() calls in the order h = 0 .. {PATH_H - 1} from a second generator ({_seed_text(MARGINAL_NAME)}). The sample standard deviation has divisor "
            f"{PATHS} - 1; the 5 and 95 percent values are the interpolated percentile of the pseudocode chapter. The last column is the probability that "
            f"the sum exceeds {thresh:g}. 'exact' is the Gaussian with the mean sum_h m_h and the variance sum_(i,j) Cov(u_i, u_j) of the exact covariance "
            f"0.36 (1 - 0.49^(min(i,j)+1)) / (1 - 0.49) * 0.7^|i-j|.\n\n")
    tail = (f"\n\nThe sum of the {PATH_H} variances v_h alone, which is the variance of the sum if the bars were independent, is {sum(v):.3f} (standard deviation "
            f"{sd_indep:.3f}); the exact variance of the sum is {vs:.3f} (standard deviation {sd_exact:.3f}).")
    return head + _table(["method", "mean of the sum", "sd of the sum", "5 percent", "95 percent", f"P(sum > {thresh:g})"], rows) + tail


# ------------------------------------------------------------------ coverage by horizon

@lru_cache(maxsize=1)
def _cover_series():
    g = bookrng.generator(COVER_NAME)
    return [ts._draw_series(g) for _ in range(COVER_SERIES)]


@lru_cache(maxsize=1)
def _calibration_series():
    g = bookrng.generator(COVER_NAME + ".calibration")
    return [ts._draw_series(g) for _ in range(COVER_SERIES)]


def _growing_sd(phi, sigma, h_count):
    """Standard deviations of the forecast error of y at horizons 1 .. h_count for the seasonal-difference AR(1) of Chapter 1 with coefficient phi and
    shock sd sigma: the error of z at step k (0-based) is sum_(j<=k) phi^(k-j) e_j and the error of y at step h is the sum of the z errors at h, h-12, ..."""
    out = []
    for h in range(h_count):
        w = [0.0] * (h + 1)
        k = h
        while k >= 0:
            for j in range(k + 1):
                w[j] += phi ** (k - j)
            k -= ts.SEASON
        out.append(sigma * math.sqrt(sum(c * c for c in w)))
    return out


def _fit_series(s):
    """(point forecast of the 24 holdout bars, constant sigma, growing sigma_h, actual holdout) for one series cut at CUT."""
    past, future = s[:ts.CUT], s[ts.CUT:ts.CUT + ts.HORIZON]
    z = [past[t] - past[t - ts.SEASON] for t in range(ts.SEASON, ts.CUT)]
    c = ts._fit_ar(z, 1)
    res = [z[t] - c[0] - c[1] * z[t - 1] for t in range(1, len(z))]
    sig = math.sqrt(sum(r * r for r in res) / len(res))
    return ts._fc_ar(past, ts.HORIZON), sig, _growing_sd(c[1], sig, ts.HORIZON), future


COVER_ALPHA = 0.2
COVER_METHODS = ("constant", "growing", "conformal")


@lru_cache(maxsize=1)
def _coverage_run():
    """({method: hits per horizon}, {method: summed widths per horizon}, rank k of the conformal half-width)."""
    z90 = Z(0.9)
    alpha = COVER_ALPHA
    cal_err = [[] for _ in range(ts.HORIZON)]
    for s in _calibration_series():
        point, _, _, future = _fit_series(s)
        for h in range(ts.HORIZON):
            cal_err[h].append(abs(future[h] - point[h]))
    k = _conformal_k(COVER_SERIES, alpha)
    conf_half = [sorted(e)[k - 1] for e in cal_err]
    hits = {m: [0] * ts.HORIZON for m in COVER_METHODS}
    wid = {m: [0.0] * ts.HORIZON for m in hits}
    for s in _cover_series():
        point, sig, grow, future = _fit_series(s)
        for h in range(ts.HORIZON):
            err = abs(future[h] - point[h])
            for m, half in (("constant", z90 * sig), ("growing", z90 * grow[h]), ("conformal", conf_half[h])):
                hits[m][h] += err <= half
                wid[m][h] += 2 * half
    return hits, wid, k


def coverage_curves():
    """{method: share inside the interval at horizons 1 .. 24} (the figure's data)."""
    hits, _, _ = _coverage_run()
    return {m: [v / COVER_SERIES for v in hits[m]] for m in COVER_METHODS}


def interval_coverage():
    hits, wid, k = _coverage_run()
    alpha = COVER_ALPHA
    n = COVER_SERIES
    cov_rows = [(h, *(f"{hits[m][h - 1] / n:.3f}" for m in hits)) for h in COVER_HORIZONS]
    wid_rows = [(h, *(f"{wid[m][h - 1] / n:.2f}" for m in hits)) for h in COVER_HORIZONS]
    head = (f"{COVER_SERIES} series drawn by the recipe of Chapter 1 from one generator ({_seed_text(COVER_NAME)}), each with its {ts.N} draws in turn, cut at {ts.CUT}. "
            f"The point forecast is the AR(1) forecast on seasonal differences of Chapter 1; sigma is the root mean square of the {ts.CUT - ts.SEASON - 1} one-step "
            f"training residuals. 'constant' is the interval point +- 1.2816 sigma at every horizon; 'growing' uses the standard deviation sigma_h of the forecast "
            f"error of the fitted model, sigma times the root of the sum of squared weights w_j of the shocks e_0 .. e_h in the error, where the error of the "
            f"difference z at step k is sum_(j<=k) phi^(k-j) e_j with the fitted phi, and the error of y at horizon h + 1 is the sum of the errors of z at "
            f"steps h, h - 12, h - 24, ... (those that are not negative). 'conformal' uses, at each horizon, the half-width that is the k-th smallest of the "
            f"absolute errors |y - point| of {COVER_SERIES} other series (the same recipe, from a second generator, {_seed_text(COVER_NAME + '.calibration')}), "
            f"with k = ceil((n + 1)(1 - alpha)) = {k} for n = {COVER_SERIES} and alpha = {alpha}. 'constant' and 'growing' use the standard Gaussian quantile 1.2816 "
            f"at 0.9. All three intervals are nominally 80 percent. The first table gives the share of the {COVER_SERIES} series whose value lies inside the interval "
            f"at the horizon h (1 = bar {ts.CUT}); the second the mean width of the whole interval.\n\n")
    allh = {m: sum(v) / (n * ts.HORIZON) for m, v in hits.items()}
    tail = (f"\n\nOver all {ts.HORIZON} horizons and {COVER_SERIES} series the shares inside are {allh['constant']:.3f} (constant), {allh['growing']:.3f} (growing) "
            f"and {allh['conformal']:.3f} (conformal). The standard error of a share of {COVER_SERIES} series near 0.8 is {math.sqrt(0.8 * 0.2 / COVER_SERIES):.3f}.")
    names = COVER_METHODS
    return (head + _table(["horizon h", *("inside, " + m for m in names)], cov_rows) + "\n\n" +
            _table(["horizon h", *("width, " + m for m in names)], wid_rows) + tail)


# ------------------------------------------------------------------ conformal intervals

def _conformal_k(n, alpha):
    return math.ceil((n + 1) * (1 - alpha) - 1e-12)


def conformal_finite():
    """Coverage of the split-conformal interval on exchangeable scores: exact value k / (n + 1) and a Monte Carlo check."""
    g = bookrng.generator(CONFORMAL_NAME)
    trials = 20000
    rows = []
    for n, alpha, naive in ((14, 0.1, True), (19, 0.1, False), (99, 0.1, False)):
        rules = [("ceil((n + 1)(1 - alpha))", _conformal_k(n, alpha))]
        if naive:
            rules.append(("ceil(n (1 - alpha))", math.ceil(n * (1 - alpha) - 1e-12)))
        for label, k in rules:
            hit = 0
            for _ in range(trials):
                cal = sorted(abs(g.normal()) for _ in range(n))
                if abs(g.normal()) <= cal[k - 1]:
                    hit += 1
            rows.append((n, label, k, f"{k / (n + 1):.4f}", f"{hit / trials:.4f}"))
    head = (f"Calibration scores s_1 .. s_n and one new score are drawn independently as |normal()| from one generator ({_seed_text(CONFORMAL_NAME)}); "
            f"in each of {trials} trials the draws are the n calibration scores in order and then the new one, and the new score is 'covered' when it is at "
            f"most the k-th smallest calibration score (k counted from 1). Nominal level 1 - alpha = 0.9. Exact coverage is k / (n + 1) "
            f"whatever the distribution (continuous, so no ties).\n\n")
    return head + _table(["n", "rule for k", "k", "exact k / (n + 1)", f"covered in {trials} trials"], rows)


def conformal_shift():
    nu = _best_nu()
    heads = _heads()
    h = heads[nu]
    g_head = heads[None]
    vx, vy = _rows(VAL_ROWS)
    alpha = 0.2
    scores = sorted(abs(y - h[0] - h[1] * x) for x, y in zip(vx, vy))
    k = _conformal_k(len(scores), alpha)
    qhat = scores[k - 1]
    rows = []
    z_t, z_g = _quant(0.9, nu), Z(0.9)
    for label, mu_a, mu_b, half in (("conformal around the Student-t mean", h[0], h[1], qhat),
                                    ("Student-t head, central 80 percent", h[0], h[1], h[2] * z_t),
                                    ("Gaussian head, central 80 percent", g_head[0], g_head[1], g_head[2] * z_g)):
        cells = []
        for rows_ in (TEST_A_ROWS, TEST_B_ROWS):
            xs, ys = _rows(rows_)
            cells.append(f"{sum(1 for x, y in zip(xs, ys) if abs(y - mu_a - mu_b * x) <= half) / len(xs):.3f}")
        rows.append((label, f"{2 * half:.3f}", *cells))
    head = (f"The Student-t head (nu = {nu}) of the earlier table. Calibration scores are the absolute errors |y - mu| of its mean forecast mu = a + b x on the "
            f"{len(vx)} validation rows t = {VAL_ROWS[0]} .. {VAL_ROWS[1]}; with n = {len(scores)} and alpha = {alpha} the rank is k = ceil((n + 1)(1 - alpha)) = {k} "
            f"and the half-width is the k-th smallest score, {qhat:.3f}. Test A is the {TEST_A_ROWS[1] - TEST_A_ROWS[0] + 1} rows t = {TEST_A_ROWS[0]} .. "
            f"{TEST_A_ROWS[1]} of the same recipe; test B is the {TEST_B_ROWS[1] - TEST_B_ROWS[0] + 1} rows t = {TEST_B_ROWS[0]} .. {TEST_B_ROWS[1]}, after "
            f"the shock coefficient doubles from 0.6 to 1.2 at bar {SHIFT_BAR}. The width is of the whole interval; the last two columns are the share of rows "
            f"inside it.\n\n")
    return head + _table(["interval", "width", "inside, test A", "inside, test B"], rows)
