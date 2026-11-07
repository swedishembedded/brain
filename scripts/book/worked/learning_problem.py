# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Learning Problem".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: learning_problem.<name> -->` markers. Functions whose names start
with an underscore (or that take arguments) are shared with the chapter's
figures, so a picture and the table beside it cannot disagree.

The running example is a temperature sensor. A hidden law turns the sensor's
voltage into a temperature and adds measurement noise; the learner sees only
rows drawn from it. The law is written down here, and only here, because the
book plays the part of nature: the chapter never uses it to fit anything.

Swedish Embedded AB implements solutions for turning measurements into
models that can be trusted on data they have not seen. If your team needs
expertise in supervised learning from sensor data, you can procure our
services by sending an email to info@swedishembedded.com.
"""
from functools import lru_cache

import bookrng

# ---------------------------------------------------------------- the law

# temperature = LAW[0] + LAW[1] * v + LAW[2] * v^2 + noise, v in [V_LO, V_HI]
LAW = (-6.0, 50.0, -8.0)
NOISE_SD = 1.2
V_LO, V_HI = 0.4, 2.8
N_TRAIN = 10
N_FRESH = 2000


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _draw(rng, n):
    """n rows (voltage, temperature) from the hidden law, rounded as a meter would show them."""
    rows = []
    for _ in range(n):
        v = rng.uniform(V_LO, V_HI)
        t = LAW[0] + LAW[1] * v + LAW[2] * v * v + rng.gauss(0.0, NOISE_SD)
        rows.append((round(v, 2), round(t, 1)))
    return rows


@lru_cache(maxsize=None)
def _train():
    """The ten rows the learner owns, sorted by voltage."""
    return tuple(sorted(_draw(bookrng.stream("learning_problem.sensor"), N_TRAIN)))


@lru_cache(maxsize=None)
def _fresh():
    """Rows from the same law that the learner has not seen."""
    return tuple(_draw(bookrng.stream("learning_problem.fresh"), N_FRESH))


def _mean_abs_error(model, rows):
    """How far off the predictions are on average, in degrees."""
    return sum(abs(model(v) - t) for v, t in rows) / len(rows)


def _line(w, b):
    return lambda v: w * v + b


def _parabola(a, w, b):
    return lambda v: a * v * v + w * v + b


def _frange(lo, hi, step):
    n = round((hi - lo) / step)
    return [lo + i * step for i in range(n + 1)]


LINE_GRID = (_frange(0.0, 60.0, 0.25), _frange(-20.0, 40.0, 0.25))
PARABOLA_GRID = (_frange(-14.0, 2.0, 0.25), _frange(20.0, 80.0, 1.0), _frange(-30.0, 30.0, 1.0))


@lru_cache(maxsize=None)
def _best_line(rows):
    """The (w, b, error, settings tried) with the smallest average error on `rows`, by trying a grid."""
    best, tried = None, 0
    for w in LINE_GRID[0]:
        for b in LINE_GRID[1]:
            tried += 1
            e = _mean_abs_error(_line(w, b), rows)
            if best is None or e < best[2]:
                best = (w, b, e)
    return best + (tried,)


@lru_cache(maxsize=None)
def _best_parabola(rows):
    best, tried = None, 0
    for a in PARABOLA_GRID[0]:
        for w in PARABOLA_GRID[1]:
            for b in PARABOLA_GRID[2]:
                tried += 1
                e = _mean_abs_error(_parabola(a, w, b), rows)
                if best is None or e < best[3]:
                    best = (a, w, b, e)
    return best + (tried,)


def _lookup(rows):
    """The lookup table: answer with the label of the stored row whose voltage is closest."""
    stored = sorted(rows)
    return lambda v: min(stored, key=lambda r: (abs(r[0] - v), r[0]))[1]


# Hand-tuned attempts at a line (w, b). Attempt 1 passes through the first and last row; each later
# attempt is a change a person would try after reading the previous score.
ATTEMPTS = ((24.0, 9.0), (20.0, 15.0), (20.0, 18.0))


# ------------------------------------------------------------- the dataset

def sensor_table():
    rows = _train()
    body = [(i + 1, f"{v:.2f}", f"{t:.1f}") for i, (v, t) in enumerate(rows)]
    return (f"N = {len(rows)} measured rows (sorted by voltage).\n\n"
            + _table(["row n", "voltage v (V)", "temperature y (deg C)"], body))


def feature_vectors():
    houses = [((84.0, 3.0, 12.0), 3150.0), ((52.0, 2.0, 41.0), 1980.0), ((126.0, 5.0, 7.0), 5240.0),
              ((67.0, 3.0, 25.0), 2470.0), ((95.0, 4.0, 3.0), 4110.0)]
    body = [(n + 1, f"{x[0]:.0f}", f"{x[1]:.0f}", f"{x[2]:.0f}", f"{y:.0f}") for n, (x, y) in enumerate(houses)]
    return (f"N = {len(houses)} examples, d = 3 features each: the feature matrix has shape [{len(houses)}, 3] "
            f"and the labels have shape [{len(houses)}].\n\n"
            + _table(["n", "area (m^2)", "rooms", "age (years)", "price (thousand SEK)"], body))


# ------------------------------------------------------- a model, predicting

def line_predictions():
    w, b = ATTEMPTS[0]
    new_v = (0.50, 1.20, 2.00, 2.60)
    body = [(f"{v:.2f}", f"{w:g} * {v:.2f} + {b:g}", f"{w * v + b:.1f}") for v in new_v]
    return (f"The model f(v) = w v + b with parameters theta = (w, b) = ({w:g}, {b:g}).\n\n"
            + _table(["new voltage v", "arithmetic", "prediction (deg C)"], body))


# ------------------------------------------------------ the first attempt

def first_attempt():
    w, b = ATTEMPTS[0]
    rows = _train()
    body, total = [], 0.0
    for v, t in rows:
        pred = w * v + b
        off = abs(pred - t)
        total += off
        body.append((f"{v:.2f}", f"{t:.1f}", f"{pred:.1f}", f"{pred - t:+.1f}", f"{off:.1f}"))
    body.append(("", "", "", "total", f"{total:.1f}"))
    body.append(("", "", "", "average over N = 10", f"{total / len(rows):.2f}"))
    return (f"Attempt 1: w = {w:g}, b = {b:g}.\n\n"
            + _table(["v", "label y", "prediction", "prediction - y", "off by"], body))


def hand_tuning():
    rows = _train()
    body = []
    for k, (w, b) in enumerate(ATTEMPTS):
        body.append((k + 1, f"{w:g}", f"{b:g}", f"{_mean_abs_error(_line(w, b), rows):.2f}"))
    return _table(["attempt", "w", "b", "average off by (deg C)"], body)


# ------------------------------------------- letting the computer search

def grid_search():
    rows = _train()
    w, b, e, tried = _best_line(rows)
    ws, bs = LINE_GRID
    return (f"Grid: w from {ws[0]:g} to {ws[-1]:g} in steps of 0.25 ({len(ws)} values), b from {bs[0]:g} to "
            f"{bs[-1]:g} in steps of 0.25 ({len(bs)} values): {tried} settings, {tried * len(rows)} single "
            f"predictions.\n\n"
            + _table(["", "w", "b", "average off by (deg C)"],
                     [("best of the grid", f"{w:g}", f"{b:g}", f"{e:.2f}"),
                      ("attempt 3 by hand", f"{ATTEMPTS[2][0]:g}", f"{ATTEMPTS[2][1]:g}",
                       f"{_mean_abs_error(_line(*ATTEMPTS[2]), rows):.2f}")]))


# ----------------------------------------------------- hypothesis spaces

def hypothesis_spaces():
    rows, fresh = _train(), _fresh()
    w, b, *_ = _best_line(rows)
    a2, w2, b2, *_ = _best_parabola(rows)
    table = _lookup(rows)
    entries = [
        ("line  w v + b", 2, _line(w, b)),
        ("parabola  a v^2 + w v + b", 3, _parabola(a2, w2, b2)),
        ("lookup table (nearest stored row)", 2 * len(rows), table),
    ]
    body = [(name, k, f"{_mean_abs_error(f, rows):.2f}", f"{_mean_abs_error(f, fresh):.2f}") for name, k, f in entries]
    return (f"Best member of each space found on the {len(rows)} known rows; "
            f"then scored on {len(fresh)} fresh rows from the same source.\n\n"
            + _table(["hypothesis space", "numbers that define a member", "average off by on the 10 known rows",
                      f"average off by on {len(fresh)} fresh rows"], body))


def parabola_grid():
    rows = _train()
    a, w, b, e, tried = _best_parabola(rows)
    return (f"Best parabola on the grid ({tried} settings): a = {a:.1f}, w = {w:.1f}, b = {b:.1f}, "
            f"average off by {e:.2f} deg C on the known rows.")


# ----------------------------------------------- the data is a sample

def resampled_lines():
    fresh = _fresh()
    rng = bookrng.stream("learning_problem.resample")
    body = []
    for k in range(5):
        rows = tuple(sorted(_draw(rng, N_TRAIN)))
        w, b, e, _ = _best_line(rows)
        body.append((k + 1, f"{w:g}", f"{b:g}", f"{e:.2f}", f"{_mean_abs_error(_line(w, b), fresh):.2f}"))
    return _table(["dataset", "best w", "best b", "off by on its own 10 rows", "off by on 2000 fresh rows"], body)


# ------------------------------------------------------- unsupervised

GROUP_CENTERS = ((1.0, 1.2), (3.6, 3.0))


@lru_cache(maxsize=None)
def _readings():
    """12 unlabelled two-feature rows, 6 around each of two hidden centres."""
    rng = bookrng.stream("learning_problem.groups")
    pts = []
    for cx, cy in GROUP_CENTERS:
        for _ in range(6):
            pts.append((round(rng.gauss(cx, 0.45), 2), round(rng.gauss(cy, 0.45), 2)))
    rng.shuffle(pts)
    return tuple(pts)


def _group(points, centers):
    """Assign each point to the nearest centre (squared distance) and move each centre to its group's mean."""
    assign = []
    for x, y in points:
        d = [(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in centers]
        assign.append(d.index(min(d)))
    new = []
    for k in range(len(centers)):
        members = [p for p, a in zip(points, assign) if a == k]
        new.append((sum(p[0] for p in members) / len(members), sum(p[1] for p in members) / len(members))
                   if members else centers[k])
    return assign, new


@lru_cache(maxsize=None)
def _grouping():
    """Run the two-step loop from a poor start until the assignment stops changing."""
    points = _readings()
    centers = [(0.0, 0.0), (1.0, 0.5)]
    history, last = [], None
    for _ in range(10):
        assign, new = _group(points, centers)
        history.append((list(centers), assign))
        if assign == last:
            break
        last, centers = assign, new
    return points, history, centers


def grouping_loop():
    points, history, _ = _grouping()
    body = []
    for i, (centers, assign) in enumerate(history):
        sizes = [assign.count(0), assign.count(1)]
        body.append((i + 1, f"({centers[0][0]:.2f}, {centers[0][1]:.2f})", f"({centers[1][0]:.2f}, {centers[1][1]:.2f})",
                     f"{sizes[0]} and {sizes[1]}"))
    return _table(["round", "centre A", "centre B", "points assigned to A and B"], body)


def grouping_points():
    points, history, _ = _grouping()
    assign = history[-1][1]
    body = [(n + 1, f"{x:.2f}", f"{y:.2f}", "AB"[a]) for n, ((x, y), a) in enumerate(zip(points, assign))]
    return _table(["n", "feature 1", "feature 2", "group found"], body, "lrrl")


# --------------------------------------------------- reinforcement learning

BUTTON_P = (0.3, 0.7)


def button_game():
    rng = bookrng.stream("learning_problem.buttons")
    rewards = {0: [], 1: []}
    body = []
    for step in range(10):
        arm = step % 2
        r = 1 if rng.random() < BUTTON_P[arm] else 0
        rewards[arm].append(r)
        body.append((step + 1, "AB"[arm], r))
    avg = [sum(rewards[k]) / len(rewards[k]) for k in (0, 1)]
    pick = 0 if avg[0] >= avg[1] else 1
    later = [1 if rng.random() < BUTTON_P[pick] else 0 for _ in range(10)]
    steady = sum(later)
    return (_table(["step", "button pressed", "reward"], body)
            + f"\n\nAverage reward: A {avg[0]:.1f}, B {avg[1]:.1f}. The agent now presses {'AB'[pick]} "
              f"for 10 more steps and collects {steady} reward. Total over 20 steps: "
              f"{sum(rewards[0]) + sum(rewards[1]) + steady}.")
