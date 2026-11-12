# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume XI project "A Calibrated Forecasting System".

Plain Python 3, standard library only, written from the project's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: project_forecaster.<name> -->`
markers. What the project reuses from the chapters of the volume is imported, never copied: the baselines, the window scaling and the
purge-and-embargo split of Chapter 1 (`time_series`), the pinball loss, the Gaussian quantile and the percentile of Chapter 2
(`probabilistic_forecasting`), the pooled ridge map over scaled windows of Chapter 3 (`foundation_forecasters`), the walk-forward
losses, the block bootstrap and the random walk of the negative control of Chapter 4 (`evaluating_forecasts`), and the isotonic fit, the
Brier score and the calibration error of Chapter 6 (`calibrated_decisions`), together with the book's generator and the stable-digits
rule.

The world is a panel of forty series that the module generates:

* every series has a level, a trend, a season of twelve bars, an autoregressive remainder and, at one bar of its own, a regime change
  that moves its level and raises the hazard of its events;
* every series also has an event process, a stock-out, whose hazard depends on how high the last bar stood in its window and on the
  time since the previous stock-out;
* a leak-safe walk-forward split cuts the 214 windows of the panel into a fit, a calibration and a test block with a purge and an
  embargo; the learned model is the pooled ridge map of Chapter 3 with a Gaussian head; the baselines are the naive, the seasonal naive
  and the seasonal AR(1) forecasts, each with the same kind of head; the event is forecast by a logistic regression on the state at the
  origin;
* the evaluation is the scaled quantile loss, the skill against the seasonal naive forecast with a block-bootstrap interval, the
  coverage of the intervals, and a negative control, a panel of random walks on which no honest forecaster can have skill;
* the recalibration of the quantile levels and of the event probability is fitted on the calibration block, and the decisions are
  orders at the 0.9 quantile and alerts at a cost-derived threshold, scored by the cost they actually incurred;
* three planted mistakes (a random split, a scaler that sees the future, a recalibration fitted on the test block) are measured on the
  panel and on the negative control, and judged by what the same models do in the deployment block that follows the study.

Every trained number is run again from four copies of the panel whose first value is changed by the nudges of `stable.py`, and is
printed with the digits on which all five runs agree. The five runs are computed in four worker processes.

Swedish Embedded AB implements solutions for forecasts that state their own uncertainty and are judged honestly. If your team needs
expertise in probabilistic forecasting, calibration and decision-focused evaluation, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import math
import multiprocessing
import statistics
from functools import lru_cache

import bookrng
import calibrated_decisions as cd
import evaluating_forecasts as ev
import foundation_forecasters as ff
import probabilistic_forecasting as pf
import stable
import time_series as ts
from logistic_regression import sigmoid

_table = ts._table

# ------------------------------------------------------------------ constants

PANEL_NAME, EVENT_NAME = "project_forecaster.panel", "project_forecaster.events"
PLACEBO_NAME = "project_forecaster.placebo"
SPLIT_NAME, BOOT_NAME = "project_forecaster.random_split", "project_forecaster.bootstrap"
S, N = 40, 300                      # series and bars of every panel
STUDY = 240                         # the first STUDY bars are split into fit, calibration and test; bars STUDY .. N - 1 are the deployment block
SEASON, C, H, K = 12, 24, 6, 3      # season, context bars, horizon bars, bars of the event window
TRAIN_FRAC, VAL_FRAC, EMBARGO = 0.5, 0.2, 6
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
RIDGE = ff.ZS_RIDGE                 # 1.0 on every weight but the constant
EPS = ts.SCALE_EPS                  # 1e-5 added to a window's standard deviation
BLOCK, RESAMPLES = 6, ev.BOOT_RESAMPLES
LEAD = 2                            # the order is for the third forecast bar (index 2), a lead time of three bars
COST_OVER, COST_UNDER = 1.0, 9.0    # per unit of the window's standard deviation
COST_FALSE_ALARM, COST_MISS = 1.0, 5.0
NEWTON_STEPS = 12
EVENT_B0, EVENT_BW, EVENT_Z, EVENT_AGE, EVENT_SHIFT = -6.0, 0.8, 1.6, 1.2, 0.9
KINDS = ("naive", "sn", "ar", "map")
KIND_NAMES = {"naive": "naive", "sn": "seasonal naive", "ar": "seasonal AR(1)", "map": "pooled ridge map"}
Z = pf.Z


def _seed(name):
    s = bookrng.seed_of(name)
    return f"{s:#018x}"


def fx(x, d=4):
    return f"{x:.{d}f}"


def fs(x, d=4):
    return f"{x:+.{d}f}"


# ------------------------------------------------------------------ the panel and its events


def _draw_panel(name):
    """S series. Per series, in this order: uniform level, slope, amplitude, phase, shock size, integer change bar, uniform jump, then N normal() calls."""
    g = bookrng.generator(name)
    out = []
    for _ in range(S):
        level = 20.0 * 5.0 ** g.uniform()
        slope = 0.0005 + 0.0025 * g.uniform()
        amp = 0.05 + 0.20 * g.uniform()
        phase = 2.0 * math.pi * g.uniform()
        shock = 0.01 + 0.03 * g.uniform()
        change = 100 + g.integer(200)
        jump = -0.25 + 0.50 * g.uniform()
        v, y = 0.0, []
        for t in range(N):
            v = 0.7 * v + shock * g.normal()
            y.append(level * (1.0 + slope * t + amp * math.sin(2.0 * math.pi * t / SEASON + phase) + v) * (1.0 + (jump if t >= change else 0.0)))
        out.append({"level": level, "slope": slope, "amp": amp, "phase": phase, "shock": shock, "change": change, "jump": jump, "y": y})
    return out


def _window_z(y, t):
    """The last bar of the C-bar window before bar t, in units of the window's standard deviation, from the window's mean (0 before bar C)."""
    if t < C:
        return 0.0
    m, sd = ff._pop_stats(y[t - C:t])
    return (y[t - 1] - m) / (sd + EPS)


def _draw_events(name, panel):
    """Per series: one uniform for the intercept b, then per bar t one uniform u, with an event when u < sigmoid(b + 0.9 z_t + 0.6 age_t + 0.7 [t >= change]).
    Adds z[t], age[t] (bars since the last event, capped at 12 and divided by 12) and ev[t] to every series."""
    g = bookrng.generator(name)
    for sr in panel:
        y = sr["y"]
        b = EVENT_B0 + EVENT_BW * g.uniform()
        age, zs, ages, evs = 0, [], [], []
        for t in range(N):
            z = _window_z(y, t)
            a = min(age, SEASON) / SEASON
            eta = b + EVENT_Z * z + EVENT_AGE * a + (EVENT_SHIFT if t >= sr["change"] else 0.0)
            hit = 1 if g.uniform() < sigmoid(eta) else 0
            zs.append(z)
            ages.append(a)
            evs.append(hit)
            age = 0 if hit else age + 1
        sr["z"], sr["age"], sr["ev"] = zs, ages, evs
    return panel


@lru_cache(maxsize=None)
def _panel(name, event_name):
    return _draw_events(event_name, _draw_panel(name))


def _walk_panel(name):
    """S driftless random walks of N bars: x_0 = normal(), x_t = x_(t-1) + normal(), series after series from one generator."""
    g = bookrng.generator(name)
    return [{"y": ev._random_walk(g, N)} for _ in range(S)]


def _nudged(panel, nudge):
    """A copy of the panel with `nudge` added to the first value of the first series (the series lists are copied, the rest is shared)."""
    out = [dict(sr) for sr in panel]
    out[0]["y"] = [out[0]["y"][0] + nudge] + out[0]["y"][1:]
    return out


# ------------------------------------------------------------------ origins and splits


def split_forward():
    return ts._temporal_split(STUDY, C, H, TRAIN_FRAC, VAL_FRAC, EMBARGO)


def origins_deploy():
    """The origins whose forecast bars all lie in the deployment block: STUDY .. N - H."""
    return list(range(STUDY, N - H + 1))


def split_random():
    """Fit, calibration and test origins of the forward split's sizes, taken from the shuffled list of all origins (each block sorted)."""
    fit, cal, test = split_forward()
    order = list(range(C, STUDY - H + 1))
    bookrng.generator(SPLIT_NAME).shuffle(order)
    a, b = len(fit), len(fit) + len(cal)
    return sorted(order[:a]), sorted(order[a:b]), sorted(order[b:b + len(test)])


# ------------------------------------------------------------------ windows, forecasters and heads


def _context_scale(y, o):
    return ff._pop_stats(y[o - C:o])


def _window(y, o, scaler):
    """(mu, d, x, scaled targets, d_loss): the window's own mean and standard deviation (+ EPS) from the C context bars, or, for the planted
    mistake, from the C context bars and the H bars it is about to forecast. d_loss is always the context's, so losses are comparable."""
    ctx = y[o - C:o]
    base = ctx + y[o:o + H] if scaler == "future" else ctx
    mu, sd = ff._pop_stats(base)
    d = sd + EPS
    d_loss = ff._pop_stats(ctx)[1] + EPS
    return mu, d, [(v - mu) / d for v in ctx], [(v - mu) / d for v in y[o:o + H]], d_loss


def _points(kind, y, o, mu, d, x, W):
    past = y[:o]
    if kind == "naive":
        return ts._fc_naive(past, H)
    if kind == "sn":
        return ts._fc_snaive(past, H)
    if kind == "ar":
        return ts._fc_ar(past, H, 1, SEASON)
    return [mu + d * v for v in ff._predict(W, [1.0] + x)]


def _fit_map(panel, fit_o, scaler):
    """The ridge map of Chapter 3 from [1, C scaled context bars] to H scaled targets, windows accumulated series by series, origins in increasing order."""
    gram = ff._Gram(C + 1, H)
    for sr in panel:
        for o in fit_o:
            _, _, x, tgt, _ = _window(sr["y"], o, scaler)
            gram.add([1.0] + x, tgt)
    return gram.solve(RIDGE)


class Row:
    """One (series, origin): the realised bars, the loss scale, the scale of the head and, per kind, the H point forecasts."""
    __slots__ = ("s", "o", "y", "d_loss", "d", "pts")


def _rows(panel, origins, scaler, W, kinds=KINDS):
    out = []
    for s, sr in enumerate(panel):
        for o in origins:
            mu, d, x, _, d_loss = _window(sr["y"], o, scaler)
            r = Row()
            r.s, r.o, r.y, r.d_loss, r.d = s, o, sr["y"][o:o + H], d_loss, d
            r.pts = {k: _points(k, sr["y"], o, mu, d, x, W) for k in kinds}
            out.append(r)
    return out


def _sigma(rows, kind):
    """The RMS over rows of the error at horizon h in units of the head's scale d: the Gaussian head's sigma_h."""
    return [math.sqrt(sum(((r.y[h] - r.pts[kind][h]) / r.d) ** 2 for r in rows) / len(rows)) for h in range(H)]


def _pit_multipliers(rows, kind, sigma):
    """The recalibrated multiplier z'_tau of each level: the Gaussian quantile at G^-1(tau), with G the empirical distribution of the
    probability integral transforms Phi((y - point) / (d sigma_h)) of all rows and horizons, with 0 and 1 added as the ends and the
    rank of tau taken as tau (n + 1) with linear interpolation between order statistics."""
    nd = statistics.NormalDist()
    pits = sorted(nd.cdf((r.y[h] - r.pts[kind][h]) / (r.d * sigma[h])) for r in rows for h in range(H))
    n = len(pits)
    ext = [0.0] + pits + [1.0]
    out = []
    for tau in LEVELS:
        pos = tau * (n + 1)
        j = math.floor(pos)
        out.append(Z(ext[j] + (pos - j) * (ext[j + 1] - ext[j])))
    return out


class Fitted:
    """Everything the walk-forward system learns from the fit and calibration blocks: the map, the heads, the multipliers and the event model."""

    def __init__(self, panel, fit_o, cal_o, scaler="context", recal_rows=None):
        self.scaler = scaler
        self.W = _fit_map(panel, fit_o, scaler)
        fit_rows = _rows(panel, fit_o, scaler, self.W)
        self.sigma = {k: _sigma(fit_rows, k) for k in KINDS}
        self.raw = [Z(t) for t in LEVELS]
        cal_rows = recal_rows if recal_rows is not None else _rows(panel, cal_o, scaler, self.W)
        self.recal = {k: _pit_multipliers(cal_rows, k, self.sigma[k]) for k in KINDS}
        self.n_fit_rows, self.n_cal_rows = len(fit_rows), len(cal_rows)


def _quantiles(r, kind, fitted, mult, h):
    return [r.pts[kind][h] + r.d * fitted.sigma[kind][h] * m for m in mult]


def _pin(q, y, tau):
    return pf.pinball(q, y, tau)


def origin_losses(rows, kind, fitted, mult, n_series=S):
    """Mean over series, horizons and levels of pinball / d_loss, one number per origin (rows are series-major, origins increasing)."""
    per = {}
    for r in rows:
        tot = 0.0
        for h in range(H):
            for q, tau in zip(_quantiles(r, kind, fitted, mult, h), LEVELS):
                tot += _pin(q, r.y[h], tau) / r.d_loss
        per.setdefault(r.o, []).append(tot / (H * len(LEVELS)))
    return [sum(v) / len(v) for _, v in sorted(per.items())]


def coverage(rows, kind, fitted, mult):
    """(share inside the central 80 percent, inside the central 50 percent, share below each level's quantile), over rows and horizons."""
    n = len(rows) * H
    below = [0] * len(LEVELS)
    for r in rows:
        for h in range(H):
            for i, q in enumerate(_quantiles(r, kind, fitted, mult, h)):
                below[i] += r.y[h] < q
    return (below[4] - below[0]) / n, (below[3] - below[1]) / n, [b / n for b in below]


# ------------------------------------------------------------------ the event model and the decisions


def _event_rows(panel, origins):
    """(features [1, z_o, age_o], label) for every (series, origin) in the order of _rows; the label is 1 when a stock-out falls in bars o .. o + K - 1."""
    return [([1.0, sr["z"][o], sr["age"][o]], 1 if any(sr["ev"][o:o + K]) else 0) for sr in panel for o in origins]


def _fit_logistic(rows):
    """Maximum likelihood of the logistic model by Newton's method from zero weights, NEWTON_STEPS steps, sums over rows in order."""
    dim = 3
    theta = [0.0] * dim
    for _ in range(NEWTON_STEPS):
        grad = [0.0] * dim
        hess = [[0.0] * dim for _ in range(dim)]
        for f, y in rows:
            p = sigmoid(sum(a * b for a, b in zip(theta, f)))
            w = p * (1.0 - p)
            for a in range(dim):
                grad[a] += (y - p) * f[a]
                for b in range(dim):
                    hess[a][b] += w * f[a] * f[b]
        step = ff._solve(hess, grad)
        theta = [a + b for a, b in zip(theta, step)]
    grad = [0.0] * dim
    for f, y in rows:
        p = sigmoid(sum(a * b for a, b in zip(theta, f)))
        for a in range(dim):
            grad[a] += (y - p) * f[a]
    assert max(abs(v) for v in grad) < 1e-8, "Newton did not converge"
    return theta


def _probs(theta, rows):
    return [sigmoid(sum(a * b for a, b in zip(theta, f))) for f, _ in rows]


def alert_cost(p, y, threshold):
    """Mean cost per row of alerting when p >= threshold: a false alarm costs COST_FALSE_ALARM, a miss COST_MISS."""
    return cd._mean_cost(p, y, threshold, COST_FALSE_ALARM, COST_MISS)


def order_costs(rows, kind, fitted, mult, level_index=4):
    """Realised cost per row of ordering the quantile at LEVELS[level_index] of bar LEAD: over-order COST_OVER, under-order COST_UNDER per unit of d_loss."""
    out = []
    for r in rows:
        q = r.pts[kind][LEAD] + r.d * fitted.sigma[kind][LEAD] * mult[level_index]
        e = r.y[LEAD] - q
        out.append((COST_UNDER * e if e > 0 else -COST_OVER * e) / r.d_loss)
    return out


# ------------------------------------------------------------------ one complete run


def _skill(model, base):
    return 1.0 - sum(model) / sum(base)


def _interval(rng, base, model):
    return ev._boot_interval(rng, base, model, BLOCK, RESAMPLES)


def _system(panel, split, scaler="context", recal="cal"):
    """Fit on the split's fit block, recalibrate on the block `recal` ('cal' or 'test'), return (fitted, test rows)."""
    fit_o, cal_o, test_o = split
    recal_rows = None
    if recal == "test":
        recal_rows = _rows(panel, test_o, scaler, _fit_map(panel, fit_o, scaler))
    f = Fitted(panel, fit_o, cal_o, scaler, recal_rows)
    return f, _rows(panel, test_o, scaler, f.W)


def _loss_table(f, rows, kinds):
    return {k: {"raw": origin_losses(rows, k, f, f.raw), "recal": origin_losses(rows, k, f, f.recal[k])} for k in kinds}


def _report(f, rows, panel=None, origins=None):
    """Mean scaled quantile losses and skills against the seasonal naive forecast, raw and recalibrated, and the 80 percent coverage of the
    pooled ridge map. With `panel` and `origins` the rows are rebuilt with the context scaler (the deployment of a model fitted under another scaler)."""
    if panel is not None:
        rows = _rows(panel, origins, "context", f.W)
    lt = _loss_table(f, rows, ("sn", "map"))
    mean = lambda x: sum(x) / len(x)
    return {"loss": {k: {v: mean(x) for v, x in d.items()} for k, d in lt.items()},
            "skill": {v: _skill(lt["map"][v], lt["sn"][v]) for v in ("raw", "recal")},
            "cover": {v: coverage(rows, "map", f, f.raw if v == "raw" else f.recal["map"])[0] for v in ("raw", "recal")}, "n": len(rows)}


def run(nudge):
    """Every number of the chapter for one copy of the panels, `nudge` added to the first value of the first series of each."""
    panel = _nudged(_panel(PANEL_NAME, EVENT_NAME), nudge)
    placebo = _nudged(_walk_panel(PLACEBO_NAME), nudge)
    fit_o, cal_o, test_o = split_forward()
    dep_o = origins_deploy()
    R = {}
    rng = bookrng.generator(BOOT_NAME)

    # the honest system
    f, test = _system(panel, split_forward())
    cal = _rows(panel, cal_o, "context", f.W)
    dep = _rows(panel, dep_o, "context", f.W)
    R["W0"] = f.W[0]
    R["sigma"] = f.sigma
    R["mult"] = {k: f.recal[k] for k in KINDS}
    losses = _loss_table(f, test, KINDS)
    R["loss"] = {k: {v: sum(x) / len(x) for v, x in d.items()} for k, d in losses.items()}
    R["skill"] = {k: {v: _skill(losses[k][v], losses["sn"][v]) for v in ("raw", "recal")} for k in ("ar", "map")}
    R["ci"] = {v: _interval(rng, losses["sn"][v], losses["map"][v]) for v in ("raw", "recal")}
    R["ci_ar"] = {v: _interval(rng, losses["sn"][v], losses["ar"][v]) for v in ("raw", "recal")}
    R["cover"] = {k: {v: coverage(test, k, f, f.raw if v == "raw" else f.recal[k]) for v in ("raw", "recal")} for k in ("sn", "ar", "map")}
    R["cover_cal"] = {v: coverage(cal, "map", f, f.raw if v == "raw" else f.recal["map"]) for v in ("raw", "recal")}
    R["honest"] = {"test": _report(f, test), "deploy": _report(f, dep)}
    R["n_rows"] = (f.n_fit_rows, f.n_cal_rows, len(test), len(dep))

    # events: a logistic regression on the state at the origin, an isotonic repair fitted on the calibration block
    ev_fit, ev_cal, ev_test, ev_dep = (_event_rows(panel, o) for o in (fit_o, cal_o, test_o, dep_o))
    theta = _fit_logistic(ev_fit)
    p_cal, p_test, p_dep = _probs(theta, ev_cal), _probs(theta, ev_test), _probs(theta, ev_dep)
    y_cal, y_test, y_dep = ([y for _, y in e] for e in (ev_cal, ev_test, ev_dep))
    iso, steps = cd._isotonic(p_cal, y_cal)
    t_star = COST_FALSE_ALARM / (COST_FALSE_ALARM + COST_MISS)
    R["theta"] = theta

    def event_block(p, y, repair):
        q = [repair(v) for v in p]
        rate = sum(y) / len(y)
        return {"rate": rate, "brier": {"raw": cd._brier(p, y), "recal": cd._brier(q, y)}, "ece": {"raw": cd._ece(p, y), "recal": cd._ece(q, y)},
                "mean_p": {"raw": sum(p) / len(p), "recal": sum(q) / len(q)},
                "cost": {"never": COST_MISS * rate, "always": COST_FALSE_ALARM * (1 - rate), "raw half": alert_cost(p, y, 0.5),
                         "raw": alert_cost(p, y, t_star), "recal": alert_cost(q, y, t_star)},
                "alerts": {"raw": sum(v >= t_star for v in p), "recal": sum(v >= t_star for v in q)}}

    R["event"] = {"rates": tuple(sum(y for _, y in e) / len(e) for e in (ev_fit, ev_cal)), "steps": steps, "cal": event_block(p_cal, y_cal, iso),
                  "test": event_block(p_test, y_test, iso), "deploy": event_block(p_dep, y_dep, iso)}

    # orders: the quantile 0.9 of the third forecast bar
    def orders(rows, fitted):
        o = {"sn": order_costs(rows, "sn", fitted, fitted.raw), "ar": order_costs(rows, "ar", fitted, fitted.raw),
             "map raw": order_costs(rows, "map", fitted, fitted.raw), "map recal": order_costs(rows, "map", fitted, fitted.recal["map"]),
             "map median": order_costs(rows, "map", fitted, fitted.raw, 2)}
        out = {k: sum(v) / len(v) for k, v in o.items()}
        out["oracle"] = 0.0
        return out

    R["orders"] = {"test": orders(test, f), "deploy": orders(dep, f)}

    # planted mistake 1: a random split of the study bars
    split_r = split_random()
    fr, rows_r = _system(panel, split_r)
    ev_rf, ev_rc, ev_rt = (_event_rows(panel, o) for o in split_r)
    th_r = _fit_logistic(ev_rf)
    iso_r, _ = cd._isotonic(_probs(th_r, ev_rc), [y for _, y in ev_rc])
    R["random"] = {"test": _report(fr, rows_r), "deploy": _report(fr, None, panel, dep_o), "shared": len(_label_bars(split_r[2]) & _label_bars(split_r[0])),
                   "event_test": event_block(_probs(th_r, ev_rt), [y for _, y in ev_rt], iso_r), "event_deploy": event_block(_probs(th_r, ev_dep), y_dep, iso_r)}
    # planted mistake 2: a scaler that sees the future (windows scaled by context and target bars together)
    fl, rows_l = _system(panel, split_forward(), "future")
    R["future"] = {"test": _report(fl, rows_l), "deploy": _report(fl, None, panel, dep_o)}
    # planted mistake 3: the recalibration fitted on the test block
    ft, rows_t = _system(panel, split_forward(), recal="test")
    iso_t, steps_t = cd._isotonic(p_test, y_test)
    R["test_fit"] = {"test": _report(ft, rows_t), "deploy": _report(ft, None, panel, dep_o), "steps": steps_t,
                     "event_test": event_block(p_test, y_test, iso_t), "event_deploy": event_block(p_dep, y_dep, iso_t)}

    # the negative control: random walks, on which no forecaster can have skill
    pf_, prows = _system(placebo, split_forward())
    pl = _loss_table(pf_, prows, KINDS)
    pfl, prows_l = _system(placebo, split_forward(), "future")
    pll = _loss_table(pfl, prows_l, ("naive", "map"))
    pfr, prows_r = _system(placebo, split_r)
    plr = _loss_table(pfr, prows_r, ("naive", "map"))
    pft, prows_t = _system(placebo, split_forward(), recal="test")
    plt = _loss_table(pft, prows_t, ("naive", "map"))
    ctl = {"honest": (pl, pf_, prows), "future scaler": (pll, pfl, prows_l), "random split": (plr, pfr, prows_r), "recalibration on the test block": (plt, pft, prows_t)}
    R["control"] = {}
    for name, (lt, fit_, rows_) in ctl.items():
        lo, hi = _interval(rng, lt["naive"]["recal"], lt["map"]["recal"])
        R["control"][name] = {"skill": _skill(lt["map"]["recal"], lt["naive"]["recal"]), "lo": lo, "hi": hi,
                              "cover": coverage(rows_, "map", fit_, fit_.recal["map"])[0]}
    R["control"]["baselines"] = {"sn": _skill(pl["sn"]["recal"], pl["naive"]["recal"]), "ar": _skill(pl["ar"]["recal"], pl["naive"]["recal"])}
    return R


def _label_bars(origins):
    return {b for o in origins for b in range(o, o + H)}


@lru_cache(maxsize=1)
def runs():
    """The nominal run followed by the four nudged copies, computed in four worker processes."""
    with multiprocessing.get_context("fork").Pool(4) as pool:
        return pool.map(run, (0.0, *stable.NUDGES), chunksize=1)


def cell(get, fmt):
    rs = runs()
    vals = [get(r) for r in rs]
    return stable.text(vals[0], vals[1:], fmt)


# ------------------------------------------------------------------ formatting helpers


def _c(get, d=4):
    return cell(get, lambda x: f"{x:.{d}f}")


def _cs(get, d=4):
    return cell(get, lambda x: f"{x:+.{d}f}")


def _cp(get, d=1):
    """A share as a percentage with d decimals."""
    return cell(lambda r: 100.0 * get(r), lambda x: f"{x:.{d}f}")


def _ci(get_lo, get_hi, d=3):
    return f"[{_cs(get_lo, d)}, {_cs(get_hi, d)}]"


def _seeds():
    return ", ".join(f"{n.split('.', 1)[1]} {_seed(n)}" for n in (PANEL_NAME, EVENT_NAME, PLACEBO_NAME, SPLIT_NAME, BOOT_NAME))


def _bars(origins, labels_only):
    out = set()
    for o in origins:
        out.update(range(o, o + H) if labels_only else range(o - C, o + H))
    return out


# ------------------------------------------------------------------ the chapter's blocks


def opening():
    rs = lambda *path: (lambda r: _dig(r, path))
    rows = []
    for label, key, ctl in (("the system as specified: forward split, recalibration on the calibration block", "honest", "honest"),
                            ("windows scaled by their context and their target bars together", "future", "future scaler"),
                            ("fit, calibration and test blocks drawn at random from all the windows", "random", "random split"),
                            ("recalibration fitted on the test block itself", "test_fit", "recalibration on the test block")):
        rows.append((label, _cs(rs(key, "test", "skill", "recal"), 3), _cp(rs(key, "test", "cover", "recal")),
                     _cs(rs("control", ctl, "skill"), 4), _cs(rs(key, "deploy", "skill", "recal"), 3), _cp(rs(key, "deploy", "cover", "recal"))))
    head = ("The study panel and the control panel are those of milestones 1 and 5; the same pooled ridge map, Gaussian head and recalibration of the quantile "
            "levels (milestones 3, 4 and 6) are built four times, differing only in the way the row says. 'Skill' is one minus the mean scaled quantile loss of the "
            "recalibrated learned model over that of the recalibrated seasonal naive forecast, over the origins of the block; the control column is the skill of "
            "the recalibrated learned model against the recalibrated naive forecast on 40 random walks, where the truth is zero. 'Inside' is the share of "
            "outcomes inside the central 80 percent interval (nominal 80). 'Test' is the test block of the study panel, 'deployment' the 55 origins that follow it. "
            "Every number carries the digits on which the run and its four nudged copies agree.\n\n")
    return head + _table(["how the system was built and judged", "skill, test", "inside, test (%)", "skill, control", "skill, deployment", "inside, deployment (%)"], rows)


def _dig(r, path):
    for p in path:
        r = r[p]
    return r


def panel_setup():
    panel = _panel(PANEL_NAME, EVENT_NAME)
    g = bookrng.generator(EVENT_NAME)
    rows = []
    for i in range(3):
        sr = panel[i]
        rows.append((i, *(fx(sr[k], 6) for k in ("level", "slope", "amp", "phase", "shock")), sr["change"], fx(sr["jump"], 6), fx(sum(sr["y"]), 6), fx(sr["y"][N - 1], 6), sum(sr["ev"])))
    first = [fx(v, 6) for v in panel[0]["y"][:6]]
    ev0 = [t for t, e in enumerate(panel[0]["ev"]) if e]
    zs = [fx(panel[0]["z"][t], 6) for t in (24, 25, 26)]
    hist = {}
    for sr in panel:
        hist[sr["change"] // 50] = hist.get(sr["change"] // 50, 0) + 1
    ev_total = sum(sum(sr["ev"]) for sr in panel)
    bars = lambda lo, hi: sum(sum(sr["ev"][lo:hi]) for sr in panel)
    head = (f"The panel has {S} series of {N} bars. Generator `{PANEL_NAME}` (SplitMix64, seed {_seed(PANEL_NAME)}), series after series; for series i the draws, in this order, are "
            f"seven values and then {N} calls of normal(): u1 for level = 20 * 5^u1; u2 for slope = 0.0005 + 0.0025 u2; u3 for amp = 0.05 + 0.20 u3; u4 for phase = 2 pi u4; "
            f"u5 for shock = 0.01 + 0.03 u5; integer(200) for change = 100 + that integer; u7 for jump = -0.25 + 0.50 u7. Then v_t = 0.7 v_(t-1) + shock normal() with "
            f"v_(-1) = 0, and y_t = level (1 + slope t + amp sin(2 pi t / 12 + phase) + v_t) (1 + jump [t >= change]) for t = 0 .. {N - 1} (the library sine; the {N} "
            f"calls of normal() are an even number, so no Gaussian value is held from one series to the next). Events: generator `{EVENT_NAME}` (seed {_seed(EVENT_NAME)}), "
            f"series after series, one uniform b for the intercept b = {EVENT_B0} + {EVENT_BW} u and then one uniform per bar t = 0 .. {N - 1}; with z_t = (y_(t-1) - mean) / "
            f"(sd + 1e-5) over the {C} bars y_(t-{C}) .. y_(t-1) (mean and sd with divisor {C}; 0 for t < {C}), age_t the bars since the last event capped at 12 and divided by 12 "
            f"(age_0 = 0, 0 after an event) the bar is an event when its uniform is below sigmoid(b + {EVENT_Z} z_t + {EVENT_AGE} age_t + {EVENT_SHIFT} [t >= change]).\n\n"
            "First table: the parameters of series 0, 1 and 2, with the sum of all 300 values, the last value and the number of events.\n\n")
    t1 = _table(["i", "level", "slope", "amp", "phase", "shock", "change", "jump", "sum of y", "y at bar 299", "events"], rows)
    mid = (f"\n\nSeries 0 starts " + ", ".join(first) + f"; its events fall on bars {', '.join(map(str, ev0))}; its z_24, z_25, z_26 are {', '.join(zs)}. "
           f"Change bars by fifties: 100 .. 149: {hist.get(2, 0)}, 150 .. 199: {hist.get(3, 0)}, 200 .. 249: {hist.get(4, 0)}, 250 .. 299: {hist.get(5, 0)} series. "
           f"Events in all: {ev_total}; in bars 0 .. 119: {bars(0, 120)}, 120 .. 179: {bars(120, 180)}, 180 .. 239: {bars(180, 240)}, 240 .. 299: {bars(240, 300)}.")
    return head + t1 + mid


def split_table():
    fit, cal, test = split_forward()
    dep = origins_deploy()
    names = (("fit", fit), ("calibration", cal), ("test", test), ("deployment", dep))
    rows = []
    for name, o in names:
        rows.append((name, f"{o[0]} .. {o[-1]}", len(o), len(o) * S, f"{o[0] - C} .. {o[-1] + H - 1}"))
    t1 = _table(["block", "origins", "windows per series", "rows", "bars read (context and forecast)"], rows)
    overlap = []
    for a_name, a in names[1:]:
        row = [a_name]
        for b_name, b in names[:-1]:
            row.append("-" if a_name == b_name or names.index((a_name, a)) <= names.index((b_name, b)) else len(_bars(a, True) & _bars(b, False)))
        overlap.append(tuple(row))
    t2 = _table(["forecast bars of the block", "also a bar of a fit window", "also a bar of a calibration window", "also a bar of a test window"], overlap)
    rf, rc, rt = split_random()
    rrows = [("forward split", len(_bars(test, True)), len(_bars(test, True) & _bars(fit, True)), len(_bars(test, True) & _bars(fit, False))),
             ("random split", len(_bars(rt, True)), len(_bars(rt, True) & _bars(rf, True)), len(_bars(rt, True) & _bars(rf, False)))]
    t3 = _table(["split", "distinct forecast bars of the test block", "that are fit forecast bars", "that are any fit bar"], rrows)
    order = list(range(C, STUDY - H + 1))
    bookrng.generator(SPLIT_NAME).shuffle(order)
    head = (f"The study is bars 0 .. {STUDY - 1}. A window has {C} context bars and {H} forecast bars; its origin o is the index of its first forecast bar, the origins "
            f"are {C} .. {STUDY - H} ({STUDY - H - C + 1} windows per series) and the rule is the `temporal_split` of Chapter 1 with n = {STUDY}, c = {C}, h = {H}, "
            f"f_train = {TRAIN_FRAC}, f_val = {VAL_FRAC}, embargo {EMBARGO}: gap = {max(EMBARGO, H)}, cuts floor({STUDY} * {TRAIN_FRAC}) = {int(STUDY * TRAIN_FRAC)} and "
            f"floor({STUDY} * 0.7) = {int(STUDY * (TRAIN_FRAC + VAL_FRAC))}, the validation side of the rule is the calibration block. The deployment block is the origins "
            f"{STUDY} .. {N - H}, whose forecast bars are bars {STUDY} .. {N - 1}. Every block contains all {S} series. Second table: for each block, how many of its "
            f"distinct forecast bars are also a bar (context or forecast) of a window of the earlier block named in the column. Third table: the same count for the "
            f"test block against the fit block under the forward split and under the random split `{SPLIT_NAME}` (seed {_seed(SPLIT_NAME)}): the origins "
            f"{C} .. {STUDY - H} in increasing order are shuffled by the book's `shuffle`, the first {len(fit)} (sorted) are the fit origins, the next {len(cal)} the "
            f"calibration origins and the next {len(test)} the test origins; the first five shuffled origins are {', '.join(map(str, order[:5]))}.\n\n")
    return head + t1 + "\n\n" + t2 + "\n\n" + t3


def heads():
    rows = [(KIND_NAMES[k], *(_c(lambda r, k=k, h=h: r["sigma"][k][h], 4) for h in range(H))) for k in KINDS]
    nfit = runs()[0]["n_rows"][0]
    head = (f"The fit block has {nfit} rows ({S} series times {len(split_forward()[0])} origins), accumulated series by series with the origins in increasing order. "
            f"Each of the four forecasters has a Gaussian head: the forecast of bar o + h is the point forecast plus d sigma_h z, where d = sd + 1e-5 is the context's scale and "
            f"sigma_h is the root of the mean over the fit rows of ((y_(o+h) - point_h) / d)^2. The table gives sigma_h for h = 0 .. 5 (the first forecast bar is h = 0); the "
            f"learned map's point is the forecast of milestone 4.\n\n")
    return head + _table(["forecaster", *(f"h = {h}" for h in range(H))], rows)


def learned_map():
    w = [_cs(lambda r, j=j: r["W0"][j], 4) for j in range(C + 1)]
    wtab = _table(["weights", "+0", "+1", "+2", "+3", "+4"], [(f"w{5 * i}", *(w[5 * i + j] if 5 * i + j <= C else "" for j in range(5))) for i in range(5)])
    head = (f"The ridge map of {C + 1} inputs and {H} outputs, penalty {RIDGE:g} on every weight but the constant, fitted on the {runs()[0]['n_rows'][0]} rows of the fit block. The table "
            f"gives the weights of its first output (h = 0), w0 the constant and w1 .. w{C} the weights of the scaled context bars from the oldest to the newest; the row "
            f"labelled w5 holds w5 .. w9, and so on.\n\n")
    return head + wtab


def evaluation():
    r0 = runs()[0]
    rows = []
    for k in ("sn", "ar", "map"):
        rows.append((KIND_NAMES[k] + ("" if k == "sn" else ""), _c(lambda r, k=k: r["loss"][k]["raw"], 4), _c(lambda r, k=k: r["loss"][k]["recal"], 4),
                     "-" if k == "sn" else _cs(lambda r, k=k: r["skill"][k]["raw"], 3), "-" if k == "sn" else _cs(lambda r, k=k: r["skill"][k]["recal"], 3)))
    rows.insert(0, ("naive", _c(lambda r: r["loss"]["naive"]["raw"], 4), _c(lambda r: r["loss"]["naive"]["recal"], 4), "", ""))
    t1 = _table(["forecaster", "loss, raw head", "loss, recalibrated", "skill against seasonal naive, raw", "recalibrated"], rows)
    crow = []
    for k in ("sn", "ar", "map"):
        for v in ("raw", "recal"):
            crow.append((KIND_NAMES[k], v, _cp(lambda r, k=k, v=v: r["cover"][k][v][0]), _cp(lambda r, k=k, v=v: r["cover"][k][v][1])))
    t2 = _table(["forecaster", "head", "inside 80% (nominal 80)", "inside 50% (nominal 50)"], crow)
    ci = [("learned map against seasonal naive", "raw", _ci(lambda r: r["ci"]["raw"][0], lambda r: r["ci"]["raw"][1])),
          ("learned map against seasonal naive", "recalibrated", _ci(lambda r: r["ci"]["recal"][0], lambda r: r["ci"]["recal"][1])),
          ("seasonal AR(1) against seasonal naive", "raw", _ci(lambda r: r["ci_ar"]["raw"][0], lambda r: r["ci_ar"]["raw"][1])),
          ("seasonal AR(1) against seasonal naive", "recalibrated", _ci(lambda r: r["ci_ar"]["recal"][0], lambda r: r["ci_ar"]["recal"][1]))]
    t3 = _table(["comparison", "head", "95% interval of the skill"], ci)
    head = (f"The test block: {S} series times {len(split_forward()[2])} origins, {S * len(split_forward()[2])} rows. The loss of a row is the mean over the {H} horizons and the "
            f"{len(LEVELS)} levels {', '.join(map(str, LEVELS))} of the pinball loss of the quantile against the outcome, divided by d_loss = sd + 1e-5 of the row's {C} context "
            f"bars (divisor {C}); the loss of an origin is the mean over the {S} series, and the tables use the mean over the {len(split_forward()[2])} origins. 'Raw' quantiles are "
            f"point + d sigma_h z_tau; 'recalibrated' ones use the multipliers of milestone 6. Skill is 1 - mean loss / mean loss of the seasonal naive forecast with the same "
            f"kind of head. The interval is the block bootstrap of Chapter 4 with blk = {BLOCK} and {RESAMPLES} resamples over the {len(split_forward()[2])} origins, 2.5th and "
            f"97.5th percentiles, generator `{BOOT_NAME}` (seed {_seed(BOOT_NAME)}) drawing the rows of the third table in its order and then the four rows of the control "
            f"table of milestone 5. 'Inside' is the share of the {H * S * len(split_forward()[2])} outcomes (rows times horizons) between the two quantile "
            f"forecasts.\n\n")
    return head + t1 + "\n\n" + t2 + "\n\n" + t3


def control():
    rows = []
    for name in ("honest", "future scaler", "random split", "recalibration on the test block"):
        rows.append((name, _cs(lambda r, n=name: r["control"][n]["skill"], 4), _ci(lambda r, n=name: r["control"][n]["lo"], lambda r, n=name: r["control"][n]["hi"], 4),
                     _cp(lambda r, n=name: r["control"][n]["cover"])))
    head = (f"{S} driftless random walks of {N} bars, generator `{PLACEBO_NAME}` (seed {_seed(PLACEBO_NAME)}), series after series: x_0 = normal(), x_t = x_(t-1) + normal() "
            f"(the {N} calls are an even number). The same splits, windows, ridge map, heads and recalibrations as milestones 2 to 6; the naive forecast is the best "
            f"forecast there is. The skill is that of the recalibrated learned model against the recalibrated naive forecast on the test block, with the block-bootstrap "
            f"interval of milestone 5 (the generator continues from the rows of that table); the rows differ in the way the system was built, as in the opening table. "
            f"Skill of the seasonal naive forecast against naive on the same walks: {_cs(lambda r: r['control']['baselines']['sn'], 3)}; of the seasonal AR(1): "
            f"{_cs(lambda r: r['control']['baselines']['ar'], 3)}.\n\n")
    return head + _table(["how the system was built", "skill against naive", "95% interval", "inside 80% (%)"], rows)


def recalibration():
    qrows = []
    for i, tau in enumerate(LEVELS):
        qrows.append((tau, fx(Z(tau), 4), *(_cs(lambda r, k=k, i=i: r["mult"][k][i], 4) for k in ("sn", "ar", "map"))))
    t1 = _table(["level tau", "raw multiplier z_tau", "seasonal naive", "seasonal AR(1)", "learned map"], qrows)
    rrows = []
    for i, tau in enumerate(LEVELS):
        rrows.append((tau, *(_cp(lambda r, v=v, b=b, i=i: r[b][v][2][i] if b == "cover_cal" else r["cover"]["map"][v][2][i], 1) for b in ("cover_cal", "cover") for v in ("raw", "recal"))))
    t2 = _table(["level tau (%)", "calibration, raw", "calibration, recalibrated", "test, raw", "test, recalibrated"], [(f"{100 * t[0]:.0f}", *t[1:]) for t in rrows])
    cal_o = split_forward()[1]
    ev_rows = []
    for blk in ("cal", "test", "deploy"):
        ev_rows.append((blk, _cp(lambda r, b=blk: r["event"][b]["rate"]), _cp(lambda r, b=blk: r["event"][b]["mean_p"]["raw"]), _cp(lambda r, b=blk: r["event"][b]["mean_p"]["recal"]),
                        _c(lambda r, b=blk: r["event"][b]["brier"]["raw"], 4), _c(lambda r, b=blk: r["event"][b]["brier"]["recal"], 4),
                        _c(lambda r, b=blk: r["event"][b]["ece"]["raw"], 4), _c(lambda r, b=blk: r["event"][b]["ece"]["recal"], 4)))
    t3 = _table(["block", "share of windows with a stock-out (%)", "mean raw p (%)", "mean recalibrated p (%)", "Brier, raw", "Brier, recal", "ECE, raw", "ECE, recal"], ev_rows)
    th = [_cs(lambda r, j=j: r["theta"][j], 4) for j in range(3)]
    rates = [_cp(lambda r, j=j: r["event"]["rates"][j]) for j in range(2)]
    head = (f"Quantile levels. The recalibration data are the {S * len(cal_o)} rows of the calibration block (origins {cal_o[0]} .. {cal_o[-1]}) and their {H} horizons, "
            f"{S * len(cal_o) * H} probability integral transforms Phi((y_(o+h) - point_h) / (d sigma_h)) per forecaster, with Phi the library distribution function "
            f"of the standard Gaussian. Sorted as p_(1) <= .. <= p_(n) with p_(0) = 0 and p_(n+1) = 1, the recalibrated level of tau is the p at rank tau (n + 1), linearly "
            f"interpolated between neighbouring ranks, and the multiplier is the library inverse Gaussian distribution function of it. First table: the multipliers. "
            f"Second table: the share of outcomes below the quantile forecast of each level, on the calibration block (the data the multipliers were fitted on) and on the test "
            f"block, for the learned map. Event probabilities. The event of a row is a stock-out in the {K} bars o, o+1, o+2; its features are [1, z_o, age_o] of milestone 1; "
            f"the model is the logistic regression fitted by {NEWTON_STEPS} steps of Newton's method from zero weights on the {S * len(split_forward()[0])} fit rows (sums over rows "
            f"in order, the largest gradient entry below 1e-8 after the last step), with the weights {', '.join(th)}. The share of fit windows with a stock-out is {rates[0]}% and of "
            f"calibration windows {rates[1]}%. The repair is the isotonic fit of Chapter 6 (pool adjacent violators, equal probabilities keeping their input order) of the "
            f"stock-out label on the raw probability over the calibration rows, read at the largest fitted point not above p, and at the first block's value below all of them. "
            f"Third table: the share of windows that had a stock-out, the means of the raw and the repaired probabilities, the Brier score and the ten-bin ECE (equal-width bins "
            f"of the probability, as in Chapter 6) in each block.\n\n")
    return head + t1 + "\n\n" + t2 + "\n\n" + t3


def decisions():
    t_star = COST_FALSE_ALARM / (COST_FALSE_ALARM + COST_MISS)
    rows = []
    for label, key in (("never alert", "never"), ("always alert", "always"), ("alert when p >= 0.5, raw p", "raw half"),
                       (f"alert when p >= {t_star:.4f}, raw p", "raw"), (f"alert when p >= {t_star:.4f}, recalibrated p", "recal")):
        rows.append((label, *(_c(lambda r, b=b, k=key: r["event"][b]["cost"][k], 4) for b in ("cal", "test", "deploy"))))
    alerts = [("number of alerts, raw p", *(cell(lambda r, b=b: float(r["event"][b]["alerts"]["raw"]), lambda x: f"{x:.0f}") for b in ("cal", "test", "deploy"))),
              ("number of alerts, recalibrated p", *(cell(lambda r, b=b: float(r["event"][b]["alerts"]["recal"]), lambda x: f"{x:.0f}") for b in ("cal", "test", "deploy")))]
    ords = []
    for label, key in (("seasonal naive, quantile 0.9", "sn"), ("seasonal AR(1), quantile 0.9", "ar"), ("learned map, raw quantile 0.9", "map raw"),
                       ("learned map, recalibrated quantile 0.9", "map recal"), ("learned map, median", "map median")):
        ords.append((label, *(_c(lambda r, b=b, k=key: r["orders"][b][k], 4) for b in ("test", "deploy"))))
    ords.append(("perfect foresight", "0", "0"))
    head = (f"Alerts. The probabilities and the repair are those of milestone 6. An alert costs {COST_FALSE_ALARM:g} when no stock-out follows and nothing when one does; no alert "
            f"costs {COST_MISS:g} when one follows and nothing otherwise; the threshold is 1 / (1 + {COST_MISS:g}) = {t_star:.4f}, and a window is alerted when p >= the threshold. "
            f"First table: mean cost per window (series and origin) in the three blocks, then the counts of alerts.\n\n")
    head2 = (f"\n\nOrders. For each row the order is the quantile 0.9 (or the median) of the third forecast bar (h = {LEAD}), and its cost is {COST_OVER:g} per unit of d_loss ordered "
             f"above the outcome and {COST_UNDER:g} per unit of d_loss below it (the pinball loss at 0.9 times ten). Second table: the mean cost per row.\n\n")
    return head + _table(["policy", "calibration block", "test block", "deployment block"], rows + alerts) + head2 + _table(["order", "test block", "deployment block"], ords)


def mistakes():
    rows = []
    for label, key in (("none: the system as specified", "honest"), ("a random split", "random"), ("a scaler that sees the future", "future"), ("recalibration fitted on the test block", "test_fit")):
        rows.append((label,
                     _cs(lambda r, k=key: r[k]["test"]["skill"]["recal"], 3), _cs(lambda r, k=key: r[k]["deploy"]["skill"]["recal"], 3),
                     _cp(lambda r, k=key: r[k]["test"]["cover"]["raw"]), _cp(lambda r, k=key: r[k]["deploy"]["cover"]["raw"]),
                     _cp(lambda r, k=key: r[k]["test"]["cover"]["recal"]), _cp(lambda r, k=key: r[k]["deploy"]["cover"]["recal"])))
    t1 = _table(["mistake", "skill, test", "skill, deployment", "inside 80% raw, test (%)", "raw, deployment (%)", "recalibrated, test (%)", "recalibrated, deployment (%)"], rows)
    erows = []
    for label, key, tk, dk in (("none", "event", "test", "deploy"), ("random split", "random", "event_test", "event_deploy"), ("recalibration on the test block", "test_fit", "event_test", "event_deploy")):
        for what, v in (("raw p", "raw"), ("repaired p", "recal")):
            erows.append((label, what, _c(lambda r, k=key, a=tk, v=v: r[k][a]["ece"][v], 4), _c(lambda r, k=key, a=dk, v=v: r[k][a]["ece"][v], 4),
                          _c(lambda r, k=key, a=tk, v=v: r[k][a]["brier"][v], 4), _c(lambda r, k=key, a=dk, v=v: r[k][a]["brier"][v], 4),
                          _c(lambda r, k=key, a=tk, v=v: r[k][a]["cost"][v], 4), _c(lambda r, k=key, a=dk, v=v: r[k][a]["cost"][v], 4)))
    t2 = _table(["mistake", "probability", "ECE, test", "ECE, deployment", "Brier, test", "Brier, deployment", "alert cost, test", "alert cost, deployment"], erows)
    shared = runs()[0]["random"]["shared"]
    fit, cal, test = split_forward()
    head = ("Each mistake is planted alone in the system as specified. The random split is the one of milestone 2 (its test block holds "
            f"{len(test)} origins as the forward one does, so the rows are the same in number). The future scaler scales every window by the mean and the standard "
            f"deviation (divisor {C + H}) of its {C} context bars and its {H} forecast bars together: the same scaling for the fit windows, the head and the test windows, "
            "the point forecast is mu + d times the map's output with those mu and d, and losses are still divided by the context's own d_loss; in deployment the "
            "windows are scaled by their context alone, since the future is not there to be read. The test-block recalibration computes the multipliers of milestone 6 "
            "from the test rows, and the isotonic repair of the event probability from the test windows. 'Deployment' is the 55 origins after the study with the models "
            f"unchanged. In the random split, {shared} of the test block's forecast bars are also forecast bars of fit windows. First table: the learned model's skill "
            "against the seasonal naive forecast, each with its own head and recalibration, and the share inside the central 80 percent interval; second table: the event "
            "model, whose probabilities are repaired by the isotonic fit of the calibration block (the test-block row: of the test block itself). The head scale d of "
            "every forecaster, baselines included, is the window's scaling, so under the future scaler the baselines' spread uses the leaky d too.\n\n")
    return head + t1 + "\n\n" + t2


def _elim_flops(p):
    """FLOPs of one elimination with partial pivoting of order p: per pivot column c and row r below, 1 division and 2 FLOPs for each of the
    p - c + 1 entries from column c to the right-hand side; back substitution row i costs 2 (p - 1 - i) + 2."""
    f = 0
    for c in range(p):
        f += (p - 1 - c) * (1 + 2 * (p - c + 1))
    for i in range(p):
        f += 2 * (p - 1 - i) + 2
    return f


def ledger():
    """(stage, count, FLOPs, bytes) rows of the bill of the system as specified, and the data behind the text."""
    p = C + 1
    n_fit, n_cal, n_test, n_dep = (len(o) * S for o in (split_forward()[0], split_forward()[1], split_forward()[2], origins_deploy()))
    scale = 6 * C + 4
    gram = 2 * (p * (p + 1) // 2 + p * H)
    fwd = 2 * p * H + 2 * H
    score = H * (1 + len(LEVELS) * 7)
    boot_row = 2 * len(split_forward()[2]) + 2
    newton = 40
    stages = [
        ("scale a fit window and its targets", f"{n_fit} x (6c + 4 + 2H)", n_fit * (scale + 2 * H), n_fit * (C + H) * 8),
        ("accumulate the Gram matrix and the right-hand sides", f"{n_fit} x 2 (p(p+1)/2 + pH)", n_fit * gram, (p * p + p * H) * 8),
        ("eliminate: H solves of order p", f"{H} x elimination(p = {p})", H * _elim_flops(p), (p * (p + 1) + p * H) * 8),
        ("forward pass of the map, all four blocks", f"{n_fit + n_cal + n_test + n_dep} x (2pH + 2H)", (n_fit + n_cal + n_test + n_dep) * fwd, (n_fit + n_cal + n_test + n_dep) * C * 8),
        ("score a head on a block: 8 variants on test, 4 on deployment", f"{8 * n_test + 4 * n_dep} x H (1 + 7 L)", (8 * n_test + 4 * n_dep) * score, 0),
        ("four block bootstraps of the skill", f"4 x {RESAMPLES} x (2 K + 2)", 4 * RESAMPLES * boot_row, 4 * 2 * len(split_forward()[2]) * 8),
        ("Newton fit of the event model", f"{NEWTON_STEPS} x {n_fit} x 40", NEWTON_STEPS * n_fit * newton, n_fit * 3 * 8),
    ]
    return stages, {"n": (n_fit, n_cal, n_test, n_dep), "scale": scale, "gram": gram, "fwd": fwd, "score": score, "boot_row": boot_row, "elim": _elim_flops(p)}


def bill():
    stages, info = ledger()
    total = sum(s[2] for s in stages)
    rows = [(a, b, f"{c:,}", f"{d:,}" if d else "-", f"{100.0 * c / total:.1f}") for a, b, c, d in stages]
    rows.append(("total", "", f"{total:,}", "", "100.0"))
    n_fit = info["n"][0]
    head = ("Work counted from the shapes alone, for the system as specified (milestones 1 to 7, without the planted mistakes, the control and the seasonal AR(1) forecaster). "
            f"One FLOP is an addition, subtraction, multiplication, division or square root; a multiply-add is 2; comparisons, loads, stores and library calls (the sine, the "
            f"logarithm, the Gaussian distribution and its inverse, the sigmoid, the draws of the generator) are not counted. With p = c + 1 = {C + 1} inputs of the map, "
            f"H = {H} outputs, L = {len(LEVELS)} levels and K = {len(split_forward()[2])} test origins: scaling a window is 6c + 4 FLOPs (mean c + 1, variance 3c + 1, the square "
            f"root, the added 1e-5, and 2c for the scaled bars) and its targets 2H; one Gram update is 2 (p (p + 1) / 2 + p H) = {info['gram']}; one elimination of order p is "
            f"{info['elim']} (per pivot column c and row below it one division and 2 FLOPs for each of the p - c + 1 entries to its right, then 2 (p - 1 - i) + 2 for row i of the "
            f"back substitution); the map's output for a row is 2 p H + 2 H = {info['fwd']}; scoring one head on one row is H (1 + 7 L) = {info['score']} (one product d sigma_h per "
            f"horizon, then per level 2 for the quantile, 4 for the loss and 1 to add it up); one bootstrap resample is 2K + 2 = {info['boot_row']} (two sums of K losses, a "
            f"division and a subtraction); one Newton step of the event model is 40 FLOPs a row (the logit, the gradient and the 3 x 3 Hessian). Bytes are 8-byte doubles "
            f"read once per stage. Rows of the four blocks: {', '.join(str(x) for x in info['n'])}.\n\n")
    tail = (f"\n\nOne fit window is read once, {(C + H) * 8} bytes for {info['gram'] + info['scale'] + 2 * H:,} FLOPs, {(info['gram'] + info['scale'] + 2 * H) / ((C + H) * 8):.2f} FLOPs "
            f"per byte; the whole panel is {S * N * 8:,} bytes.")
    return head + _table(["stage", "count", "FLOPs", "bytes read", "share of FLOPs (%)"], rows) + tail


def acceptance_tests():
    panel = _panel(PANEL_NAME, EVENT_NAME)
    fit, cal, test = split_forward()
    dep = origins_deploy()
    rf, rc, rt = split_random()
    stages, info = ledger()
    total = sum(x[2] for x in stages)
    rows = []

    def add(milestone, what, value, how):
        rows.append((milestone, what, value, how))

    add(1, "y at bar 299 of series 0, 39", f"{fx(panel[0]['y'][N - 1], 6)}, {fx(panel[39]['y'][N - 1], 6)}", "6 decimals")
    add(1, "sum of the 300 values of series 0", fx(sum(panel[0]["y"]), 6), "6 decimals")
    add(1, "events in all, in series 0, in bars 0 .. 119", f"{sum(sum(sr['ev']) for sr in panel)}, {sum(panel[0]['ev'])}, {sum(sum(sr['ev'][:120]) for sr in panel)}", "exact")
    add(2, "origins of the fit, calibration, test and deployment blocks", f"{fit[0]}..{fit[-1]}, {cal[0]}..{cal[-1]}, {test[0]}..{test[-1]}, {dep[0]}..{dep[-1]}", "exact")
    add(2, "test forecast bars that are fit forecast bars: forward, random", f"0, {len(_label_bars(rt) & _label_bars(rf))}", "exact")
    add(3, "sigma_h of the seasonal naive head, h = 0 and 5", f"{_c(lambda r: r['sigma']['sn'][0])}, {_c(lambda r: r['sigma']['sn'][5])}", "4 decimals")
    add(4, "w0 and w24 of the map's first output", f"{_cs(lambda r: r['W0'][0])}, {_cs(lambda r: r['W0'][C])}", "4 decimals")
    add(4, "sigma_h of the learned head, h = 0 and 5", f"{_c(lambda r: r['sigma']['map'][0])}, {_c(lambda r: r['sigma']['map'][5])}", "4 decimals")
    add(5, "test loss of the seasonal naive, AR(1) and learned heads (raw)", f"{_c(lambda r: r['loss']['sn']['raw'])}, {_c(lambda r: r['loss']['ar']['raw'])}, {_c(lambda r: r['loss']['map']['raw'])}", "4 decimals")
    add(5, "skill of the learned map, raw and recalibrated", f"{_cs(lambda r: r['skill']['map']['raw'], 3)}, {_cs(lambda r: r['skill']['map']['recal'], 3)}", "3 decimals")
    add(5, "95% interval of the recalibrated skill", _ci(lambda r: r["ci"]["recal"][0], lambda r: r["ci"]["recal"][1]), "3 decimals; the generator's rows in table order")
    add(5, "share inside the central 80 percent: learned map raw, recalibrated", f"{_cp(lambda r: r['cover']['map']['raw'][0])}, {_cp(lambda r: r['cover']['map']['recal'][0])}", "1 decimal, percent")
    add(5, "control: skill of the honest system against naive", f"{_cs(lambda r: r['control']['honest']['skill'], 4)} in {_ci(lambda r: r['control']['honest']['lo'], lambda r: r['control']['honest']['hi'], 4)}", "4 decimals")
    add(5, "control: skill with the future scaler", f"{_cs(lambda r: r['control']['future scaler']['skill'], 4)}", "4 decimals")
    add(6, "recalibrated multipliers of the learned head at tau = 0.1, 0.5, 0.9", ", ".join(_cs(lambda r, i=i: r["mult"]["map"][i], 4) for i in (0, 2, 4)), "4 decimals")
    add(6, "share below the 0.1 and 0.9 quantiles on the test block, raw and recalibrated", f"{_cp(lambda r: r['cover']['map']['raw'][2][0])}, {_cp(lambda r: r['cover']['map']['recal'][2][0])}; {_cp(lambda r: r['cover']['map']['raw'][2][4])}, {_cp(lambda r: r['cover']['map']['recal'][2][4])}", "1 decimal, percent")
    add(7, "event model weights", ", ".join(_cs(lambda r, j=j: r["theta"][j], 4) for j in range(3)), "4 decimals")
    add(7, "ECE on the test block, raw and repaired", f"{_c(lambda r: r['event']['test']['ece']['raw'])}, {_c(lambda r: r['event']['test']['ece']['recal'])}", "4 decimals")
    add(7, "alert cost on the test block: never, always, raw, repaired", ", ".join(_c(lambda r, k=k: r["event"]["test"]["cost"][k]) for k in ("never", "always", "raw", "recal")), "4 decimals")
    add(7, "order cost on the test block: seasonal naive, learned raw, learned recalibrated", ", ".join(_c(lambda r, k=k: r["orders"]["test"][k]) for k in ("sn", "map raw", "map recal")), "4 decimals")
    add(8, "future scaler: skill on the test block and in deployment", f"{_cs(lambda r: r['future']['test']['skill']['recal'], 3)}, {_cs(lambda r: r['future']['deploy']['skill']['recal'], 3)}", "3 decimals")
    add(8, "future scaler: share inside the central 80 percent in deployment (recalibrated)", _cp(lambda r: r["future"]["deploy"]["cover"]["recal"]), "1 decimal, percent")
    add(8, "random split: raw event ECE on its test block and in deployment", f"{_c(lambda r: r['random']['event_test']['ece']['raw'])}, {_c(lambda r: r['random']['event_deploy']['ece']['raw'])}", "4 decimals")
    add(8, "test-block recalibration: share inside the central 80 percent on the test block, in deployment", f"{_cp(lambda r: r['test_fit']['test']['cover']['recal'])}, {_cp(lambda r: r['test_fit']['deploy']['cover']['recal'])}", "1 decimal, percent")
    add(8, "test-block recalibration: repaired event ECE on the test block, in deployment", f"{_c(lambda r: r['test_fit']['event_test']['ece']['recal'])}, {_c(lambda r: r['test_fit']['event_deploy']['ece']['recal'])}", "4 decimals")
    add(9, "FLOPs of the system as specified", f"{total:,}", "exact")
    head = ("Each row is a number of a block above, collected in the order of the milestones. 'Exact' means equal integers or equal text; a number with decimals is "
            "to be matched to the digits shown, which are the digits on which the run and its four nudged copies agree.\n\n")
    return head + _table(["milestone", "quantity", "value", "how to compare"], rows, "llll")


# ------------------------------------------------------------------ data of the figures


def figure_protocols():
    """[(label, skill on test, skill in deployment)] of the four protocols of the opening table (recalibrated learned model)."""
    r = runs()[0]
    return [("as specified", r["honest"]), ("future scaler", r["future"]), ("random split", r["random"]), ("test-block recalibration", r["test_fit"])]


def figure_reliability():
    """(levels, share below on the test block raw, recalibrated) for the learned map."""
    r = runs()[0]
    return list(LEVELS), r["cover"]["map"]["raw"][2], r["cover"]["map"]["recal"][2]
