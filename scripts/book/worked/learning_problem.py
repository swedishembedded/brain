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
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from math import comb

import bookrng

# ---------------------------------------------------------------- the law

# temperature = LAW[0] + LAW[1] * v + LAW[2] * v^2 + noise, v in [V_LO, V_HI]
LAW = (-6.0, 50.0, -8.0)
NOISE_SD = 1.2
V_LO, V_HI = 0.4, 2.8
V_SPAN = 2.4  # V_HI - V_LO as written in the chapter, so the typed formula and the code agree to the last bit
N_TRAIN = 10
N_FRESH = 2000
ASK_V = 1.75  # the voltage the workshop asks about in the chapter text


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    """The sentence that makes a random table reproducible: generator and seed."""
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})."


def _round(x, places):
    """The pseudocode page's round(x, n): the exact decimal value of x, nearest multiple of 10^-n, ties away from zero."""
    return float(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _draw(rng, n):
    """n rows (voltage, temperature) from the hidden law, rounded as a meter would show them.

    Draw order, row by row: v = V_LO + (V_HI - V_LO) * rng.uniform(), then the noise
    NOISE_SD * rng.normal(); v is rounded to 2 decimals and t to 1.
    """
    rows = []
    for _ in range(n):
        v = V_LO + V_SPAN * rng.uniform()
        t = LAW[0] + LAW[1] * v + LAW[2] * v * v + NOISE_SD * rng.normal()
        rows.append((_round(v, 2), _round(t, 1)))
    return rows


SENSOR_NAME = "learning_problem.sensor"
FRESH_NAME = "learning_problem.fresh"
RESAMPLE_NAME = "learning_problem.resample"
GROUPS_NAME = "learning_problem.groups"
BUTTONS_NAME = "learning_problem.buttons"


@lru_cache(maxsize=None)
def _train():
    """The ten rows the learner owns, sorted by voltage."""
    return tuple(sorted(_draw(bookrng.generator(SENSOR_NAME), N_TRAIN)))


@lru_cache(maxsize=None)
def _fresh():
    """Rows from the same law that the learner has not seen."""
    return tuple(_draw(bookrng.generator(FRESH_NAME), N_FRESH))


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


# Hand-tuned attempts at a line (w, b). Attempt 1 is a line drawn by eye through the cloud; each later
# attempt is a change a person would try after reading the previous score.
ATTEMPTS = ((24.0, 9.0), (20.0, 15.0), (20.0, 18.0))


# ------------------------------------------------------------- the dataset

def sensor_table():
    rows = _train()
    body = [(i + 1, f"{v:.2f}", f"{t:.1f}") for i, (v, t) in enumerate(rows)]
    return (f"N = {len(rows)} measured rows (sorted by voltage; the source that produced them and the order of its draws "
            f"are written out in the section \"Too small, too big\").\n\n"
            + _table(["row n", "voltage v (V)", "temperature y (deg C)"], body))


def feature_vectors():
    houses = [((84.0, 3.0, 12.0), 3150.0), ((52.0, 2.0, 41.0), 1980.0), ((126.0, 5.0, 7.0), 5240.0),
              ((67.0, 3.0, 25.0), 2470.0), ((95.0, 4.0, 3.0), 4110.0)]
    body = [(n + 1, f"{x[0]:.0f}", f"{x[1]:.0f}", f"{x[2]:.0f}", f"{y:.0f}") for n, (x, y) in enumerate(houses)]
    return (f"N = {len(houses)} examples, d = 3 features each: the feature matrix has shape [{len(houses)}, 3] "
            f"and the labels have shape [{len(houses)}].\n\n"
            + _table(["n", "area (m^2)", "rooms", "age (years)", "price (thousand SEK)"], body))


def data_source():
    """The full recipe of the source, so that every row and every fresh row can be regenerated."""
    return (f"The source (the learner never sees these lines): voltage v uniform on [{V_LO}, {V_HI}], temperature "
            f"y = {LAW[0]:g} + {LAW[1]:g} v - {-LAW[2]:g} v^2 + {NOISE_SD} z with z a standard Gaussian sample, "
            f"evaluated left to right in double precision. "
            f"A row is made by two draws in this order: v = {V_LO} + {V_SPAN:.1f} * rng.uniform(), then z = "
            f"rng.normal(); v is then rounded to 2 decimals and y (computed from the unrounded v) to 1, with `round` of the "
            f"pseudocode page (ties away from zero). "
            f"`rng.normal()` is the Box-Muller sample defined on the pseudocode page: it draws two uniforms for every second "
            f"call and holds the second result for the call after. Because a row makes one `rng.uniform()` call and one "
            f"`rng.normal()` call, the held value is used by every second row, and it stays held across the "
            f"row boundary and across the ten-row blocks of the last line below (it is never discarded or reset); only a "
            f"new generator starts with nothing held.\n\n"
            + _table(["dataset", "generator seed", "rows", "how they are used"],
                     [("sensor table", f"{bookrng.seed_of(SENSOR_NAME):#018x}", N_TRAIN,
                       "the ten known rows, sorted by (v, y) for display"),
                      ("fresh rows", f"{bookrng.seed_of(FRESH_NAME):#018x}", N_FRESH,
                       "scoring only, never used to choose parameters"),
                      ("five resampled datasets", f"{bookrng.seed_of(RESAMPLE_NAME):#018x}", 5 * N_TRAIN,
                       "one generator; dataset 1 takes the first 10 rows, dataset 2 the next 10, and so on")],
                     "lrrl"))


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
    return (f"Grid: w = {ws[0]:g} + 0.25 i for i = 0, ..., {len(ws) - 1} (from {ws[0]:g} to {ws[-1]:g}, {len(ws)} "
            f"values), b = {bs[0]:g} + 0.25 j for j = 0, ..., {len(bs) - 1} (from {bs[0]:g} to {bs[-1]:g}, {len(bs)} "
            f"values): {tried} settings, {tried * len(rows)} single predictions. The settings are visited with w in the "
            f"outer loop and b in the inner one, both ascending, and a setting replaces the best so far only when its "
            f"average is strictly smaller (the first of equal settings wins). Averages are shown to 2 decimals.\n\n"
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
    return (f"Best member of each space found on the {len(rows)} known rows (the line and parabola by the grids of the "
            f"text, the lookup table by storing the rows); then scored on {len(fresh)} fresh rows from the same source "
            f"(seed {bookrng.seed_of(FRESH_NAME):#018x}). The lookup table answers with the label of the stored row whose "
            f"voltage is closest to the question; a question exactly between two stored voltages goes to the lower "
            f"one; stored rows with equal voltage (none occur here) would be tried in order of increasing label, so "
            f"the lowest label would answer. Averages are shown to 2 decimals.\n\n"
            + _table(["hypothesis space", "numbers that define a member", "average off by on the 10 known rows",
                      f"average off by on {len(fresh)} fresh rows"], body))


def parabola_grid():
    rows = _train()
    a, w, b, e, tried = _best_parabola(rows)
    ga, gw, gb = PARABOLA_GRID
    return (f"Parabola grid: a from {ga[0]:g} to {ga[-1]:g} in steps of 0.25 ({len(ga)} values), w from {gw[0]:g} to "
            f"{gw[-1]:g} in steps of 1 ({len(gw)} values), b from {gb[0]:g} to {gb[-1]:g} in steps of 1 ({len(gb)} values); "
            f"a is the outer loop, then w, then b, all ascending, strict improvement only, as for the line.\n\n"
            f"Best parabola on the grid ({tried} settings): a = {a:.1f}, w = {w:.1f}, b = {b:.1f}, "
            f"average off by {e:.2f} deg C on the known rows.")


# ----------------------------------------------- the data is a sample

def resampled_lines():
    fresh = _fresh()
    rng = bookrng.generator(RESAMPLE_NAME)
    body = []
    for k in range(5):
        rows = tuple(sorted(_draw(rng, N_TRAIN)))
        w, b, e, _ = _best_line(rows)
        body.append((k + 1, f"{rows[0][0]:.2f}", f"{rows[-1][0]:.2f}", f"{w:g}", f"{b:g}", f"{e:.2f}",
                     f"{_mean_abs_error(_line(w, b), fresh):.2f}"))
    return (f"{_generator_line(RESAMPLE_NAME)} One generator; dataset k takes the next 10 rows (two draws per row, as "
            f"in the source of the previous section), and the line is the best of the grid of this chapter (same grid, "
            f"same tie rule). Fresh rows: the 2000 above.\n\n"
            + _table(["dataset", "lowest voltage (V)", "highest voltage (V)", "best w", "best b", "off by on its own 10 rows",
                      "off by on 2000 fresh rows"], body))


# ------------------------------------------------------- unsupervised

GROUP_CENTERS = ((1.0, 1.2), (3.6, 3.0))
GROUP_SD = 0.45
START_CENTRES = [(0.0, 0.0), (1.0, 0.5)]
MAX_ROUNDS = 10


@lru_cache(maxsize=None)
def _readings():
    """12 unlabelled two-feature rows, 6 around each of two hidden centres."""
    rng = bookrng.generator(GROUPS_NAME)
    pts = []
    for cx, cy in GROUP_CENTERS:
        for _ in range(6):
            x = cx + GROUP_SD * rng.normal()
            y = cy + GROUP_SD * rng.normal()
            pts.append((_round(x, 2), _round(y, 2)))
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
    centers = list(START_CENTRES)
    history, last = [], None
    for _ in range(MAX_ROUNDS):
        assign, new = _group(points, centers)
        history.append((list(centers), assign))
        if assign == last:
            break
        last, centers = assign, new
    return points, history, centers


def readings_table():
    points = _readings()
    body = [(n + 1, f"{x:.2f}", f"{y:.2f}") for n, (x, y) in enumerate(points)]
    return (f"{_generator_line(GROUPS_NAME)} Source: 6 rows around (1.0, 1.2), then 6 around (3.6, 3.0); each row is "
            f"two draws in this order, x = cx + 0.45 * rng.normal() then y = cy + 0.45 * rng.normal(), rounded to 2 "
            f"decimals (`round` of the pseudocode page); the 12 rows, in this generation order, are then permuted in place by "
            f"`shuffle(rows, rng)` of the pseudocode page (Fisher-Yates, one `rng.integer(i + 1)` for i = 11 down to 1, "
            f"continuing the same generator), and row n below is the element at position n - 1 after the shuffle. "
            f"The learner is given only this table.\n\n"
            + _table(["n", "feature 1", "feature 2"], body, "lrr"))


def grouping_loop():
    points, history, _ = _grouping()
    body = []
    for i, (centers, assign) in enumerate(history):
        sizes = [assign.count(0), assign.count(1)]
        body.append((i + 1, f"({centers[0][0]:.5f}, {centers[0][1]:.5f})", f"({centers[1][0]:.5f}, {centers[1][1]:.5f})",
                     f"{sizes[0]} and {sizes[1]}"))
    return (f"Start: centre A = {START_CENTRES[0]}, centre B = {START_CENTRES[1]}. A round assigns each row to the "
            f"centre with the smaller squared distance (A on a tie), then moves each centre to the mean of its rows "
            f"(the sum of the rows in table order, divided by their number; a centre with no rows stays). Rounds stop "
            f"after the first round whose assignment equals the previous round's, or after {MAX_ROUNDS} rounds. The "
            f"table shows, for each round, the centres used by its assignment, to 5 decimals: every exact centre of this "
            f"run is either a decimal with at most 5 places or far from a rounding tie, so the order of summation does "
            f"not change a printed digit.\n\n"
            + _table(["round", "centre A", "centre B", "points assigned to A and B"], body))


def grouping_points():
    points, history, _ = _grouping()
    assign = history[-1][1]
    body = [(n + 1, f"{x:.2f}", f"{y:.2f}", "AB"[a]) for n, ((x, y), a) in enumerate(zip(points, assign))]
    return _table(["n", "feature 1", "feature 2", "group found"], body, "lrrl")


# --------------------------------------------------- reinforcement learning

BUTTON_P = (0.3, 0.7)


def button_game():
    rng = bookrng.generator(BUTTONS_NAME)
    rewards = {0: [], 1: []}
    body = []
    for step in range(10):
        arm = step % 2
        r = 1 if rng.uniform() < BUTTON_P[arm] else 0
        rewards[arm].append(r)
        body.append((step + 1, "AB"[arm], r))
    avg = [sum(rewards[k]) / len(rewards[k]) for k in (0, 1)]
    pick = 0 if avg[0] >= avg[1] else 1
    later = [1 if rng.uniform() < BUTTON_P[pick] else 0 for _ in range(10)]
    steady = sum(later)
    return (f"{_generator_line(BUTTONS_NAME)} Button A pays with probability {BUTTON_P[0]}, button B with "
            f"{BUTTON_P[1]}; one `rng.uniform()` per press, rewarded (1) when it is below the button's probability. "
            f"Steps 1 to 10 alternate A, B, A, ...; the 10 later steps press the button with the larger average "
            f"(A on a tie), one `rng.uniform()` each, continuing the same generator.\n\n"
            + _table(["step", "button pressed", "reward"], body)
            + f"\n\nAverage reward: A {avg[0]:.1f}, B {avg[1]:.1f}. The agent now presses {'AB'[pick]} "
              f"for 10 more steps and collects {steady} units of reward. Total over 20 steps: "
              f"{sum(rewards[0]) + sum(rewards[1]) + steady}.")


# ----------------------------------------------- how long to explore

EXPLORE_NAME = "learning_problem.explore"
EXPLORE_PAIRS = ((0.3, 0.7), (0.5, 0.6))
EXPLORE_LENGTHS = (2, 10, 100)
EXPLORE_GAMES = 2000
EXPLORE_HORIZON = 200


def _wrong_pick_probability(pa, pb, presses_each):
    """P(the agent picks A or ties): the number of rewards of B is not above that of A (B is the better button)."""
    n = presses_each
    pmf_a = [comb(n, k) * pa ** k * (1 - pa) ** (n - k) for k in range(n + 1)]
    pmf_b = [comb(n, k) * pb ** k * (1 - pb) ** (n - k) for k in range(n + 1)]
    return sum(pmf_a[i] * pmf_b[j] for i in range(n + 1) for j in range(i + 1))


def explore_study():
    rng = bookrng.generator(EXPLORE_NAME)
    body = []
    for pa, pb in EXPLORE_PAIRS:
        for length in EXPLORE_LENGTHS:
            wrong = 0
            for _ in range(EXPLORE_GAMES):
                reward = [0, 0]
                for press in range(length):
                    arm = press % 2
                    reward[arm] += 1 if rng.uniform() < (pa, pb)[arm] else 0
                if reward[1] <= reward[0]:
                    wrong += 1
            p_wrong = _wrong_pick_probability(pa, pb, length // 2)
            per_step = (length / 2 * (pa + pb) + (EXPLORE_HORIZON - length) * ((1 - p_wrong) * pb + p_wrong * pa)) \
                / EXPLORE_HORIZON
            body.append((f"{pa} and {pb}", length, wrong, f"{wrong / EXPLORE_GAMES:.4f}", f"{p_wrong:.4f}",
                         f"{per_step:.4f}"))
    return (f"{_generator_line(EXPLORE_NAME)} One generator for the whole table. For each pair of probabilities (A, B), "
            f"in the order listed, and for each exploration length L = 2, 10, 100 in that order, play {EXPLORE_GAMES} "
            f"games one after the other. A game presses A, B, A, ... for L presses, one `rng.uniform()` per press, "
            f"rewarded when it is below the pressed button's probability, then compares the two reward totals (each "
            f"button was pressed L / 2 times, so totals compare as averages do): it picks B only when B's total is "
            f"strictly larger, and A otherwise (so a tie is a wrong pick, since B is the better button in both pairs). "
            f"Only the exploration presses are drawn. The count column is reproduced exactly. The next column is the "
            f"exact probability of a wrong pick (sums of binomial probabilities, no simulation), which the simulated "
            f"rate matches up to sampling error. The last column is the expected reward per step of a game of "
            f"{EXPLORE_HORIZON} steps (the L exploration presses, then the rest on the picked button), from the exact "
            f"probability. Four decimals.\n\n"
            + _table(["probabilities of A and B", "L", f"wrong picks in {EXPLORE_GAMES} games", "rate",
                      "exact probability", f"expected reward per step over {EXPLORE_HORIZON} steps"], body))
