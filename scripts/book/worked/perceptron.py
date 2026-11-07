# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Neuron and the Perceptron".

Plain Python 3, standard library only, written from the chapter's own
description: the artificial neuron y = g(w . x + b) on a three-sensor decision,
the step and sigmoid activations, AND and OR as single neurons, Rosenblatt's
perceptron rule traced by hand on six points, the geometry of one update, the
convergence bound (R / gamma)^2 checked over many presentation orders, the
comparison with logistic regression and with the widest-margin separator, the
sixteen Boolean functions of two inputs (two of which no neuron computes), the
perceptron cycling on XOR, and the two ways out (a product feature and two
layers of step neurons). Each public function without a leading underscore that
takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: perceptron.<name> -->` markers. Underscore functions and the
constants are shared with the chapter's figures, so a picture and the table
beside it cannot disagree.

Swedish Embedded AB implements solutions for deciding from sensor readings with
models whose every weight can be inspected. If your team needs expertise in
small, verifiable classifiers for industrial monitoring, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng

# ------------------------------------------------------------- constants

# A pump monitor: readings scaled to [0, 1], a weighted vote and a threshold.
PUMP_NAMES = ("vibration", "temperature rise", "flow loss")
PUMP_W = (0.8, 0.4, 0.6)
PUMP_B = -1.0
PUMP_CASES = (
    ("calm", (0.1, 0.2, 0.1)),
    ("vibration alone", (0.9, 0.1, 0.1)),
    ("bearing wear", (0.9, 0.5, 0.3)),
    ("hot and blocked", (0.6, 0.9, 0.9)),
)

# Six labelled points (x1, x2, y), y in {-1, +1}, visited in this order each pass.
TRACE_DATA = (
    (3.0, 3.0, 1),
    (0.0, 0.0, -1),
    (4.0, 1.0, 1),
    (1.0, 0.0, -1),
    (3.0, 4.0, 1),
    (0.0, 2.0, -1),
)
# A separator chosen by hand for the bound, 2 x1 + x2 - 5.5 = 0 (any separator gives a valid bound).
TRACE_SEPARATOR = (2.0, 1.0, -5.5)
N_ORDERS = 200
MAX_EPOCHS = 1000

GAP_WIDTHS = (1.0, 0.5, 0.25, 0.1)
GAP_N = 20
GAP_DATA_NAME = "perceptron.gap_data"
GAP_ORDER_NAME = "perceptron.gap_orders"
ORDERS_NAME = "perceptron.orders"

LOGISTIC_ETA = 0.5
LOGISTIC_STEPS = 20000

XOR = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
GRID = tuple(k * 0.5 for k in range(-8, 9))  # -4.0 .. 4.0 in steps of 0.5


# ---------------------------------------------------------------- helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)})."


def _vec(v, d=2):
    return "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _step(z):
    return 1 if z >= 0.0 else 0


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _angle_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _angle_between(a, b):
    """Angle in degrees between two vectors, or None when either is the zero vector."""
    if math.hypot(*a) == 0 or math.hypot(*b) == 0:
        return None
    c = _dot(a, b) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ---------------------------------------------------------- the perceptron

def _train(points, order, max_epochs=MAX_EPOCHS, record=None):
    """Rosenblatt's rule on points (x1, x2, y) visited in `order` each pass.

    Returns (w, b, mistakes, epochs); `record`, if a list, receives one tuple per
    visit: (epoch, index, z, mistake, w, b) with w, b after the visit. Stops after
    the first full pass without a mistake (epochs counts that last pass too)."""
    w, b, mistakes = [0.0, 0.0], 0.0, 0
    for epoch in range(1, max_epochs + 1):
        errors = 0
        for n in order:
            x1, x2, y = points[n]
            z = w[0] * x1 + w[1] * x2 + b
            mistake = y * z <= 0.0
            if mistake:
                w = [w[0] + y * x1, w[1] + y * x2]
                b += y
                errors += 1
                mistakes += 1
            if record is not None:
                record.append((epoch, n, z, mistake, tuple(w), b))
        if errors == 0:
            return tuple(w), b, mistakes, epoch
    return tuple(w), b, mistakes, max_epochs


@lru_cache(maxsize=None)
def _trace():
    rec = []
    w, b, mistakes, epochs = _train(TRACE_DATA, range(len(TRACE_DATA)), record=rec)
    return rec, w, b, mistakes, epochs


def _updates():
    """The visits at which the rule changed the weights: (epoch, n, z, w_after, b_after)."""
    return [(e, n, z, w, b) for e, n, z, m, w, b in _trace()[0] if m]


def _augmented_bound(points, sep):
    """(R, gamma, bound): R = max |(x, 1)|, gamma = min y (w* . (x, 1)) / |w*|, bound = (R / gamma)^2."""
    norm = math.sqrt(sum(c * c for c in sep))
    unit = [c / norm for c in sep]
    r = max(math.sqrt(x1 * x1 + x2 * x2 + 1.0) for x1, x2, _ in points)
    gamma = min(y * (unit[0] * x1 + unit[1] * x2 + unit[2]) for x1, x2, y in points)
    return r, gamma, (r / gamma) ** 2


def _mistakes_over_orders(points, gen, n_orders=N_ORDERS):
    counts = []
    for _ in range(n_orders):
        order = list(range(len(points)))
        gen.shuffle(order)
        counts.append(_train(points, order)[2])
    return counts


# --------------------------------------------------------------- examples

def pump_vote():
    rows = []
    for name, x in PUMP_CASES:
        votes = [w * v for w, v in zip(PUMP_W, x)]
        z = sum(votes) + PUMP_B
        rows.append((name, _vec(x, 1), *(f"{v:.2f}" for v in votes), f"{sum(votes):.2f}", f"{z:+.2f}",
                     _step(z), f"{_sigmoid(z):.3f}"))
    head = ("Inputs: " + ", ".join(PUMP_NAMES) + f" (each scaled to [0, 1]). Weights w = {_vec(PUMP_W, 1)}, bias b = {PUMP_B:.1f}.\n"
            "Alarm = step(w . x + b) with step(z) = 1 for z >= 0 and 0 otherwise.\n\n")
    return head + _table(["case", "x", "w1 x1", "w2 x2", "w3 x3", "sum of votes", "z = sum + b", "alarm", "sigmoid(z)"], rows)


def activation_values():
    rows = []
    for z in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
        rows.append((f"{z:+.1f}", _step(z), f"{_sigmoid(z):.4f}", f"{_sigmoid(4 * z):.4f}", f"{_sigmoid(16 * z):.4f}"))
    return "The sigmoid of k z for growing k against the step, at nine values of z.\n\n" + _table(
        ["z", "step(z)", "sigmoid(z)", "sigmoid(4 z)", "sigmoid(16 z)"], rows)


GATES = (
    ("AND", (1.0, 1.0), -1.5, lambda a, b: a & b),
    ("OR", (1.0, 1.0), -0.5, lambda a, b: a | b),
    ("NAND", (-1.0, -1.0), 1.5, lambda a, b: 1 - (a & b)),
)


def logic_gates():
    out = ["Each gate is step(w1 x1 + w2 x2 + b) with the weights shown; step(z) = 1 for z >= 0.\n"]
    for name, w, b, f in GATES:
        rows = []
        for x1 in (0, 1):
            for x2 in (0, 1):
                z = w[0] * x1 + w[1] * x2 + b
                y = _step(z)
                assert y == f(x1, x2)
                rows.append((x1, x2, f"{z:+.1f}", y, f(x1, x2)))
        out.append(f"{name}: w = {_vec(w, 1)}, b = {b:+.1f}\n")
        out.append(_table(["x1", "x2", "z", "neuron", "target"], rows))
        out.append("")
    return "\n".join(out).rstrip()


def perceptron_trace():
    rec, w, b, mistakes, epochs = _trace()
    pts = ", ".join(f"P{n + 1} = ({x1:g}, {x2:g}) label {y:+d}" for n, (x1, x2, y) in enumerate(TRACE_DATA))
    head = (f"Points, visited in this order every pass: {pts}.\n"
            "Start w = (0, 0), b = 0. A visit is a mistake when y (w . x + b) <= 0, and then w += y x, b += y.\n\n")
    rows = []
    for k, (epoch, n, z, m, wv, bv) in enumerate(rec, start=1):
        rows.append((epoch, f"P{n + 1}", f"{z:+.1f}", "mistake" if m else "ok", _vec(wv, 1), f"{bv:+.1f}"))
    tail = (f"\nThe rule made {mistakes} updates and the pass {epochs} had none: final w = {_vec(w, 1)}, b = {b:+.1f}, "
            f"boundary {w[0]:g} x1 + {w[1]:g} x2 {b:+g} = 0.")
    return head + _table(["pass", "point", "z before", "outcome", "w after", "b after"], rows) + tail


def update_geometry():
    rows = []
    w, b = (0.0, 0.0), 0.0
    for k, (epoch, n, z, wv, bv) in enumerate(_updates(), start=1):
        x1, x2, y = TRACE_DATA[n]
        x = (x1, x2)
        after = _dot(wv, x) + bv
        before, after_angle = _angle_between(w, x), _angle_between(wv, x)
        ang_before = "-" if before is None else f"{before:.1f}"
        ang_after = "-" if after_angle is None else f"{after_angle:.1f}"
        rows.append((k, f"P{n + 1}", f"{y:+d}", f"{z:+.2f}", f"{after:+.2f}", f"{after - z:.2f}", f"{x1 * x1 + x2 * x2 + 1:.2f}",
                     ang_before, ang_after))
        w, b = wv, bv
    note = ("The score of the visited point rises (for a positive point; falls for a negative one) by exactly |x|^2 + 1, "
            "the squared length of x with a 1 appended. The angle is between w and x in degrees (a dash where w or x is the zero vector).")
    return note + "\n\n" + _table(["update", "point", "y", "z before", "z after", "change", "|x|^2 + 1", "angle(w, x) before", "angle(w, x) after"], rows)


def convergence_bound():
    r, gamma, bound = _augmented_bound(TRACE_DATA, TRACE_SEPARATOR)
    counts = _mistakes_over_orders(TRACE_DATA, bookrng.generator(ORDERS_NAME))
    norm = math.sqrt(sum(c * c for c in TRACE_SEPARATOR))
    lines = [
        _seed_line(ORDERS_NAME),
        f"Draw order: for each of the {N_ORDERS} runs, shuffle(list [0, 1, 2, 3, 4, 5]) with the book's Fisher-Yates shuffle, "
        "then run the rule from w = 0, b = 0 over that order each pass until a pass has no mistake.",
        "",
        f"Separator w* = {_vec(TRACE_SEPARATOR, 1)} (last entry is the bias), |w*| = {norm:.4f}.",
        "",
        _table(["quantity", "value"], [
            ("R = largest |(x1, x2, 1)|", f"{r:.4f}"),
            ("gamma = smallest y (w* . (x, 1)) / |w*|", f"{gamma:.4f}"),
            ("bound (R / gamma)^2", f"{bound:.2f}"),
            ("mistakes in the order of the trace", _trace()[3]),
            (f"fewest mistakes over {N_ORDERS} orders", min(counts)),
            (f"mean mistakes over {N_ORDERS} orders", f"{sum(counts) / len(counts):.2f}"),
            (f"most mistakes over {N_ORDERS} orders", max(counts)),
        ]),
    ]
    return "\n".join(lines)


def _gap_data(g):
    gen = bookrng.generator(GAP_DATA_NAME)
    pts = []
    while len(pts) < GAP_N:
        x1, x2 = 2 * gen.uniform() - 1, 2 * gen.uniform() - 1
        if abs(x1) >= g / 2:
            pts.append((x1, x2, 1 if x1 > 0 else -1))
    return pts


def gap_scaling():
    rows = []
    for g in GAP_WIDTHS:
        pts = _gap_data(g)
        r, gamma, bound = _augmented_bound(pts, (1.0, 0.0, 0.0))
        counts = _mistakes_over_orders(pts, bookrng.generator(GAP_ORDER_NAME))
        assert max(counts) <= bound
        rows.append((f"{g:g}", f"{gamma:.4f}", f"{r:.4f}", f"{bound:.1f}", min(counts), f"{sum(counts) / len(counts):.1f}", max(counts)))
    head = (f"Data: {GAP_N} points per row. {_seed_line(GAP_DATA_NAME)} Draw order: candidate after candidate, x1 = 2 u1 - 1 then x2 = 2 u2 - 1 "
            f"(u1, u2 = uniform()); a candidate is kept when |x1| >= g / 2, with label +1 if x1 > 0 and -1 otherwise, until {GAP_N} are kept. "
            "The separator is w* = (1, 0), b = 0.\n"
            f"Orders: {_seed_line(GAP_ORDER_NAME)} {N_ORDERS} orders per row, each one shuffle of [0, ..., {GAP_N - 1}], the generator restarted for each row.\n\n")
    return head + _table(["gap g", "gamma", "R", "bound (R/gamma)^2", "fewest", "mean", "most"], rows)


def update_coefficients():
    rows = []
    for z in (-4, -2, -0.5, 0.5, 2, 4):
        p, st = _sigmoid(z), _step(z)
        rows.append((f"{z:+.1f}", st, f"{1 - st}", f"{1 - p:.4f}", f"{-st}", f"{-p:.4f}"))
    return ("The factor that multiplies the input vector x in one update of w, for a neuron with score z and label t in {0, 1}. "
            "The perceptron uses t - step(z), logistic regression t - sigmoid(z).\n\n"
            + _table(["z", "step(z)", "t = 1: perceptron", "t = 1: logistic", "t = 0: perceptron", "t = 0: logistic"], rows))


def _logistic_fit(points, eta=LOGISTIC_ETA, steps=LOGISTIC_STEPS):
    w = [0.0, 0.0, 0.0]
    n = len(points)
    for _ in range(steps):
        g = [0.0, 0.0, 0.0]
        for x1, x2, y in points:
            t = 1.0 if y > 0 else 0.0
            r = _sigmoid(w[0] * x1 + w[1] * x2 + w[2]) - t
            g[0] += r * x1
            g[1] += r * x2
            g[2] += r
        w = [w[i] - eta * g[i] / n for i in range(3)]
    return w


def _geometric_margin(w, b, points):
    nrm = math.hypot(*w)
    return min(y * (w[0] * x1 + w[1] * x2 + b) for x1, x2, y in points) / nrm


def _max_margin(points):
    """Widest strip between the classes: scan the direction in steps of 0.01 degree."""
    best = (-1.0, 0.0, 0.0)
    for k in range(36000):
        th = math.radians(k * 0.01)
        u = (math.cos(th), math.sin(th))
        pos = [u[0] * x1 + u[1] * x2 for x1, x2, y in points if y > 0]
        neg = [u[0] * x1 + u[1] * x2 for x1, x2, y in points if y < 0]
        gap = (min(pos) - max(neg)) / 2.0
        if gap > best[0]:
            best = (gap, k * 0.01, -(min(pos) + max(neg)) / 2.0)
    return best


def separators_compared():
    _, w, b, _, _ = _trace()
    gen = bookrng.generator(ORDERS_NAME)
    margins = []
    for _ in range(N_ORDERS):
        order = list(range(len(TRACE_DATA)))
        gen.shuffle(order)
        wv, bv, _, _ = _train(TRACE_DATA, order)
        margins.append(_geometric_margin(wv, bv, TRACE_DATA))
    lw = _logistic_fit(TRACE_DATA)
    gap, deg, off = _max_margin(TRACE_DATA)
    rows = [
        ("perceptron, order of the trace", f"{_angle_deg(w):.1f}", f"{_geometric_margin(w, b, TRACE_DATA):.4f}"),
        (f"perceptron, worst of {N_ORDERS} orders", "-", f"{min(margins):.4f}"),
        (f"perceptron, best of {N_ORDERS} orders", "-", f"{max(margins):.4f}"),
        (f"logistic regression, {LOGISTIC_STEPS} steps", f"{_angle_deg(lw):.1f}", f"{_geometric_margin(lw[:2], lw[2], TRACE_DATA):.4f}"),
        ("widest margin (scan of directions)", f"{deg:.1f}", f"{gap:.4f}"),
    ]
    head = (f"Same six points as the trace. Logistic regression: labels mapped to t in {{0, 1}}, w and b from 0, plain gradient descent, "
            f"learning rate {LOGISTIC_ETA}, {LOGISTIC_STEPS} steps on the mean cross-entropy. Margin = the smallest distance from a point to the boundary, "
            "y (w . x + b) / |w|. The widest margin is found by trying every direction u of 0.00, 0.01, ..., 359.99 degrees, "
            "placing the boundary halfway between the classes' projections on u and keeping the best half-gap. "
            f"The {N_ORDERS} orders are those of the bound table.\n\n")
    return head + _table(["method", "direction of w (degrees)", "margin"], rows)


def boolean_functions():
    names = {}
    counts = {}
    for w1 in GRID:
        for w2 in GRID:
            for b in GRID:
                tt = tuple(_step(w1 * a + w2 * c + b) for a in (0, 1) for c in (0, 1))
                counts[tt] = counts.get(tt, 0) + 1
    rows = []
    for k in range(16):
        tt = tuple((k >> (3 - i)) & 1 for i in range(4))
        label = {(0, 0, 0, 1): "AND", (0, 1, 1, 1): "OR", (0, 1, 1, 0): "XOR", (1, 0, 0, 1): "XNOR", (1, 1, 1, 0): "NAND",
                 (1, 0, 0, 0): "NOR", (0, 0, 0, 0): "constant 0", (1, 1, 1, 1): "constant 1"}.get(tt, "")
        rows.append(("".join(map(str, tt)), label, counts.get(tt, 0)))
    reached = sum(1 for r in rows if r[2] > 0)
    head = (f"Every (w1, w2, b) with each entry in {{-4.0, -3.5, ..., 4.0}} ({len(GRID)}^3 = {len(GRID) ** 3} neurons), "
            "outputs step(w1 x1 + w2 x2 + b) on the inputs (0,0), (0,1), (1,0), (1,1) in that order. "
            "A function is named by its four outputs in that order.\n\n")
    return head + _table(["outputs", "name", "neurons computing it"], rows, "llr") + f"\n\nFunctions reached: {reached} of 16."


def xor_cycle():
    pts = tuple((a, c, 1 if t else -1) for a, c, t in XOR)
    w, b = [0.0, 0.0], 0.0
    rows = []
    for epoch in (1, 2):
        for x1, x2, y in pts:
            z = w[0] * x1 + w[1] * x2 + b
            mistake = y * z <= 0.0
            if mistake:
                w = [w[0] + y * x1, w[1] + y * x2]
                b += y
            rows.append((epoch, f"({x1}, {x2})", f"{y:+d}", f"{z:+.1f}", "mistake" if mistake else "ok", _vec(w, 1), f"{b:+.1f}"))
    sums = [sum(y * f for x1, x2, y in pts for f in [(x1, x2, 1)[k]]) for k in range(3)]
    head = ("XOR with labels -1 for (0,0) and (1,1) and +1 for (0,1) and (1,0), visited in the order (0,0), (0,1), (1,0), (1,1) "
            "every pass; the same rule as before from w = (0, 0), b = 0. The sum of y (x1, x2, 1) over the four points is "
            f"({sums[0]}, {sums[1]}, {sums[2]}).\n\n")
    return head + _table(["pass", "point", "y", "z before", "outcome", "w after", "b after"], rows, "rlrrlll")


def xor_feature_map():
    w, b = (1.0, 1.0, -2.0), -0.5
    rows = []
    for a, c, t in XOR:
        f = (a, c, a * c)
        z = _dot(w, f) + b
        rows.append((a, c, a * c, f"{z:+.1f}", _step(z), t))
    return (f"Features (x1, x2, x1 x2) with weights w = {_vec(w, 1)} and bias b = {b:+.1f}.\n\n"
            + _table(["x1", "x2", "x1 x2", "z", "neuron", "XOR"], rows))


TWO_LAYER = {"h1": ((1.0, 1.0), -0.5), "h2": ((1.0, 1.0), -1.5), "y": ((1.0, -1.0), -0.5)}


def xor_two_layer():
    rows = []
    for a, c, t in XOR:
        h1 = _step(TWO_LAYER["h1"][0][0] * a + TWO_LAYER["h1"][0][1] * c + TWO_LAYER["h1"][1])
        h2 = _step(TWO_LAYER["h2"][0][0] * a + TWO_LAYER["h2"][0][1] * c + TWO_LAYER["h2"][1])
        y = _step(TWO_LAYER["y"][0][0] * h1 + TWO_LAYER["y"][0][1] * h2 + TWO_LAYER["y"][1])
        assert y == t
        rows.append((a, c, f"{a + c - 0.5:+.1f}", h1, f"{a + c - 1.5:+.1f}", h2, f"{h1 - h2 - 0.5:+.1f}", y, t))
    return ("Hidden neuron h1 = step(x1 + x2 - 0.5), hidden neuron h2 = step(x1 + x2 - 1.5), output y = step(h1 - h2 - 0.5).\n\n"
            + _table(["x1", "x2", "z of h1", "h1", "z of h2", "h2", "z of y", "y", "XOR"], rows))
