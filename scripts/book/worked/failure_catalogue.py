# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "A Catalogue of Expensive Mistakes".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: failure_catalogue.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's
figures so a picture and the table beside it cannot disagree.

Randomness comes only from `bookrng.generator(name)`. Single-precision arithmetic is emulated with `floating.f32`
(round the double result of one +, - or * of two float32 values to the nearest float32, which is the correctly rounded
float32 result because a double holds more than twice the bits of a float32 significand).

Swedish Embedded AB implements solutions for finding, classifying and permanently closing defects in numerical
software for its clients. If your team needs expertise in root-cause analysis and regression gates for machine
learning systems then you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng
from floating import f32

U32 = 2.0 ** -24          # unit roundoff of float32
U64 = 2.0 ** -53          # unit roundoff of float64


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _g(x, d=4):
    """d significant digits, plain for moderate sizes and scientific otherwise."""
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e6:
        return f"{float(f'{x:.{d}g}'):g}"
    return f"{x:.{d - 1}e}"


def _gamma(k, u):
    """The accumulation constant gamma_k = k u / (1 - k u)."""
    return k * u / (1.0 - k * u)


# ---------------------------------------------------------------------------------------------------------------------
# Four wrong layers that pass one test
# ---------------------------------------------------------------------------------------------------------------------

IN = 3                    # input width m
MAX_ROWS = 8              # 2 heads of 4
BASE = 10000.0            # the reference's rotary base
WRONG_BASE = 1000000.0    # the default of defect D
CHUNK = 2                 # rows per chunk of the chunked projection


def _blind_inputs():
    """W (8 x 3, row-major), then x (3), then x_prev (3), each 2 u - 1 of one generator."""
    g = bookrng.generator("failure_catalogue.blind_spots")
    W = [[2.0 * g.uniform() - 1.0 for _ in range(IN)] for _ in range(MAX_ROWS)]
    x = [2.0 * g.uniform() - 1.0 for _ in range(IN)]
    x_prev = [2.0 * g.uniform() - 1.0 for _ in range(IN)]
    return W, x, x_prev


def _rows_of(W, heads, dh):
    return [row[:] for row in W[:heads * dh]]


def _project_row(row, x):
    acc = 0.0
    for k in range(IN):
        acc += row[k] * x[k]
    return acc


def _rotate_scale(q, heads, dh, pos, base, scale=True):
    """Rotate the half-split pairs (j, j + dh/2) of each head by pos * base^(-2j/dh), then multiply by 1/sqrt(dh)."""
    out = [0.0] * len(q)
    half = dh // 2
    for h in range(heads):
        for j in range(half):
            theta = pos * base ** (-2.0 * j / dh)
            c, s = math.cos(theta), math.sin(theta)
            a, b = q[h * dh + j], q[h * dh + j + half]
            out[h * dh + j] = a * c - b * s
            out[h * dh + j + half] = a * s + b * c
    if scale:
        k = 1.0 / math.sqrt(dh)
        out = [v * k for v in out]
    return out


def _reference(W, x, heads, dh, pos):
    q = [_project_row(r, x) for r in _rows_of(W, heads, dh)]
    return _rotate_scale(q, heads, dh, pos, BASE)


def _interleaved_file(rows, heads, dh):
    """The stored layout: row half * dh/2 + i of a head is kept at 2 i + half."""
    half = dh // 2
    stored = [None] * len(rows)
    for h in range(heads):
        for two in range(2):
            for i in range(half):
                stored[h * dh + 2 * i + two] = rows[h * dh + two * half + i]
    return stored


def _unpermute_order(heads, dh):
    """The stored row each model row reads: r = h dh + half (dh/2) + i reads h dh + 2 i + half."""
    half = dh // 2
    order = []
    for r in range(heads * dh):
        h, within = divmod(r, dh)
        two, i = divmod(within, half)
        order.append(h * dh + 2 * i + two)
    return order


def _implementation(defect, stored, x, heads, dh, pos, buf):
    """The layer as an engine computes it: rows read from the stored file, projected in chunks of 2 rows into `buf`,
    rotated, scaled. `defect` is None (correct) or one of 'A' (add into the buffer), 'B' (stored rows read in place),
    'C' (no 1/sqrt(dh)), 'D' (rotary base 10^6)."""
    order = list(range(heads * dh)) if defect == "B" else _unpermute_order(heads, dh)
    rows = [stored[order[r]] for r in range(heads * dh)]
    for row0 in range(0, heads * dh, CHUNK):
        chunk = [_project_row(rows[r], x) for r in range(row0, min(row0 + CHUNK, heads * dh))]
        for t, v in enumerate(chunk):
            if defect == "A":
                buf[row0 + t] += v
            else:
                buf[row0 + t] = v
    base = WRONG_BASE if defect == "D" else BASE
    return _rotate_scale(buf, heads, dh, pos, base, scale=(defect != "C"))


def _cosine(a, b):
    dot = sum(p * q for p, q in zip(a, b))
    return dot / (math.sqrt(sum(p * p for p in a)) * math.sqrt(sum(q * q for q in b)))


BLIND_COLUMNS = [
    # (label, heads, dh, pos, reused buffer, metric)
    ("1 head of 2, pos 0, fresh, cosine", 1, 2, 0, False, "cos"),
    ("same, elementwise", 1, 2, 0, False, "abs"),
    ("2 heads of 4", 2, 4, 0, False, "abs"),
    ("pos 3", 2, 4, 3, False, "abs"),
    ("buffer reused", 2, 4, 3, True, "abs"),
    ("all of these, cosine", 2, 4, 3, True, "cos"),
]
BLIND_ROWS = [(None, "correct"), ("A", "A: adds into an uncleared buffer"), ("B", "B: stored rows read in place"),
              ("C", "C: 1/sqrt(dh) dropped"), ("D", "D: rotary base 10^6 by default")]


def _blind_cell(defect, heads, dh, pos, reused, metric, W, x, x_prev):
    rows = _rows_of(W, heads, dh)
    stored = _interleaved_file(rows, heads, dh)
    want = _reference(W, x, heads, dh, pos)
    buf = [0.0] * (heads * dh)
    if reused:
        _implementation(defect, stored, x_prev, heads, dh, pos, buf)
    got = _implementation(defect, stored, x, heads, dh, pos, buf)
    if metric == "cos":
        return _cosine(got, want) >= 0.9999
    return max(abs(a - b) for a, b in zip(got, want)) <= 1e-9 * max(abs(b) for b in want)


def _blind_matrix():
    W, x, x_prev = _blind_inputs()
    return [[_blind_cell(d, *col[1:], W, x, x_prev) for col in BLIND_COLUMNS] for d, _ in BLIND_ROWS]


def blind_spots():
    W, x, x_prev = _blind_inputs()
    matrix = _blind_matrix()
    head = ("A layer projects x (3 numbers) by W (rows 0 to heads x dh - 1 of the 8 x 3 matrix below), rotates the "
            "half-split pairs (j, j + dh/2) of each head by the angle pos x 10000^(-2j/dh) (the library cos, sin and "
            "pow) and multiplies by 1/sqrt(dh). The engine reads W from a file that stores row half x dh/2 + i of each "
            "head at row 2i + half, so a correct reader takes model row h dh + half dh/2 + i from stored row "
            "h dh + 2i + half; it projects in chunks of 2 rows into a buffer, then rotates and scales. 'fresh' means "
            "the buffer starts at zero; 'buffer reused' means the same layer was first called on x_prev with that "
            "buffer and the second call's output on x is checked. 'cosine' passes when the cosine similarity with "
            "the reference is at least 0.9999; 'elementwise' passes when the largest absolute difference is at most "
            "1e-9 times the largest absolute reference output. Each column changes one thing from the one before it. "
            f"Inputs: generator seeded {bookrng.seed_of('failure_catalogue.blind_spots')} (seed_of "
            "'failure_catalogue.blind_spots'), drawing W row by row, then x, then x_prev, each value 2 uniform() - 1 "
            "(shown to 6 decimals; the computation uses the full doubles).")
    wrows = [[str(r)] + [f"{v:.6f}" for v in W[r]] for r in range(MAX_ROWS)]
    wtab = _table(["row of W", "col 0", "col 1", "col 2"], wrows)
    vec = (f"x = ({', '.join(f'{v:.6f}' for v in x)}); x_prev = ({', '.join(f'{v:.6f}' for v in x_prev)}).")
    rows = [[label] + ["pass" if ok else "FAIL" for ok in cells] for (_, label), cells in zip(BLIND_ROWS, matrix)]
    tab = _table(["implementation"] + [c[0] for c in BLIND_COLUMNS], rows, "l" + "c" * len(BLIND_COLUMNS))
    return "\n\n".join([head, wtab, vec, tab])


# ---------------------------------------------------------------------------------------------------------------------
# A clipped gradient that is mostly stale memory
# ---------------------------------------------------------------------------------------------------------------------

CLIP_DIM = 64
TRUE_NORM = 0.98
STALE_NORMS = [0.0, 1.0, 10.0, 1000.0, 279760.0]


def _unit(v):
    n = math.sqrt(sum(a * a for a in v))
    return [a / n for a in v]


def stale_clip():
    g = bookrng.generator("failure_catalogue.stale_clip")
    true_dir = _unit([g.normal() for _ in range(CLIP_DIM)])
    stale_dir = _unit([g.normal() for _ in range(CLIP_DIM)])
    gt = [TRUE_NORM * a for a in true_dir]
    chance = sum(a * b for a, b in zip(true_dir, stale_dir))
    rows = []
    for s in STALE_NORMS:
        total = [a + s * b for a, b in zip(gt, stale_dir)]
        norm = math.sqrt(sum(a * a for a in total))
        clip = min(1.0, 1.0 / norm)
        step = [clip * a for a in total]
        cos = _cosine(step, gt)
        rows.append([f"{s:g}", _g(norm, 6), _g(clip), _g(clip * TRUE_NORM), _g(cos)])
    head = (f"A true gradient of norm {TRUE_NORM} in {CLIP_DIM} dimensions and a stale vector of norm s are added, "
            "and the sum is clipped to a global norm of at most 1 (multiplied by min(1, 1 / norm)). Directions: "
            f"{CLIP_DIM} normal() draws for the true gradient, then {CLIP_DIM} for the stale vector, of the generator "
            f"seeded {bookrng.seed_of('failure_catalogue.stale_clip')} (seed_of 'failure_catalogue.stale_clip'), each "
            "divided by its norm; the same two directions serve every row. 'true part of the step' is the clip factor "
            "times 0.98, the length of the true gradient inside the update; 'cosine' is the cosine similarity of the "
            f"update with the true gradient. The cosine of the two directions themselves is {_g(chance)}. Normal draws "
            "use the library log, sin and cos, so numbers are to four significant digits.")
    return head + "\n\n" + _table(["stale norm s", "norm before the clip", "clip factor", "true part of the step",
                                   "cosine with the true gradient"], rows)


# ---------------------------------------------------------------------------------------------------------------------
# A golden written from the memory under a view
# ---------------------------------------------------------------------------------------------------------------------

GH, GW, GC, GOUT = 4, 4, 3, 2


def _mix_channels(chw, M):
    """A 1 x 1 convolution: out[o][p] = sum_c M[o][c] chw[c][p], c increasing, for each of the H W pixels p."""
    return [[sum(M[o][c] * chw[c][p] for c in range(GC)) for p in range(GH * GW)] for o in range(GOUT)]


def _flat(a):
    return [v for row in a for v in row]


def scrambled_golden():
    g = bookrng.generator("failure_catalogue.scrambled_golden")
    M = [[2.0 * g.uniform() - 1.0 for _ in range(GC)] for _ in range(GOUT)]
    photo_hwc = [g.uniform() for _ in range(GH * GW * GC)]            # memory order: (h, w, c), c fastest
    seeded_chw = [2.0 * g.uniform() - 1.0 for _ in range(GC * GH * GW)]  # memory order: (c, h, w), drawn in place
    # The network's input from the photo: channels reversed and moved first, a view; its element (c, h, w) is the
    # memory element (h, w, C - 1 - c).
    photo_view = [[photo_hwc[(p // GW * GW + p % GW) * GC + (GC - 1 - c)] for p in range(GH * GW)] for c in range(GC)]
    seeded = [[seeded_chw[c * GH * GW + p] for p in range(GH * GW)] for c in range(GC)]
    rows = []
    for name, logical, memory in (("photo, a reversed and transposed view", photo_view, photo_hwc),
                                  ("seeded, drawn in place", seeded, seeded_chw)):
        reference_out = _flat(_mix_channels(logical, M))
        saved = [[memory[c * GH * GW + p] for p in range(GH * GW)] for c in range(GC)]  # the buffer read as (c, h, w)
        replay_out = _flat(_mix_channels(saved, M))
        same_input = _flat(saved) == _flat(logical)
        dev = max(abs(a - b) for a, b in zip(replay_out, reference_out))
        rows.append([name, "yes" if same_input else "no", _g(_cosine(replay_out, reference_out)), _g(dev)])
    head = (f"A 1 x 1 convolution M ({GOUT} x {GC}) mixes the {GC} channels of a {GH} x {GW} input stored channels "
            "first, (c, h, w). The photo is held as (h, w, c) with c fastest, and the network is given the view that "
            "reverses its channels and moves them first; the seeded input is drawn directly as (c, h, w). The golden "
            "file stores each input by writing the memory under it and labelling it (c, h, w); the reference outputs "
            "are computed from the view, and the replay runs the network on the stored input. Draws, in order, of the "
            f"generator seeded {bookrng.seed_of('failure_catalogue.scrambled_golden')} (seed_of "
            "'failure_catalogue.scrambled_golden'): M row by row (2 uniform() - 1), the 48 photo values in memory "
            "order (uniform()), the 48 seeded values in memory order (2 uniform() - 1). Sums run over c in increasing "
            "order from 0.")
    return head + "\n\n" + _table(["input", "stored input equals the input used", "cosine of replay and reference",
                                   "largest difference"], rows, "llrr")


# ---------------------------------------------------------------------------------------------------------------------
# Odds of meeting a defect that occupies part of the input space
# ---------------------------------------------------------------------------------------------------------------------

def _runs_for(p, confidence):
    """Least n with 1 - (1 - p)^n >= confidence."""
    if p >= 1:
        return 1
    if p <= 0:
        return None
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - float(p)))


def prefix_flake_probability():
    """P(fewer than 2 of 6 trials keep >= 4 of 14 base tokens), keep uniform in 0 .. 13: exact."""
    q = Fraction(10, 14)
    return (1 - q) ** 6 + 6 * q * (1 - q) ** 5


def detection_odds():
    flake = prefix_flake_probability()
    per_run = 1 - (1 - flake) ** 2
    ft_fail, ft_n = _tolerance_counts()
    cases = [
        ("u64 seed above 2^63 - 1, one uniform u64 seed per run", Fraction(1, 2)),
        ("shared-prefix trials: fewer than 2 of 6 reach a 4-token block, one dtype", flake),
        ("the same, both dtypes in one run", per_run),
        ("relative tolerance of the cancelling-sum check (its table)", Fraction(ft_fail, ft_n)),
    ]
    rows = []
    for name, p in cases:
        rows.append([name, f"{float(p):.4g}" if p != Fraction(1, 2) else "1/2",
                     _runs_for(p, 0.5), _runs_for(p, 0.95), _runs_for(p, 0.99)])
    head = ("For a defect that shows on a fraction p of the runs, the chance that n independent runs all miss it is "
            "(1 - p)^n, and the columns give the least n for which the chance of at least one failure reaches 50, 95 "
            "and 99 percent (log is the library logarithm; the counts are whole runs). The shared-prefix p is exact: "
            f"keep is uniform over 0 .. 13, a trial reaches a block when keep >= 4 (probability 10/14), and p = "
            f"(4/14)^6 + 6 (10/14)(4/14)^5 = {flake.numerator}/{flake.denominator}; with two dtypes per run it is "
            f"1 - (1 - p)^2. The last row's p is the failure count of the relative rule in the table of the cancelling-sum check "
            f"({ft_fail} of {ft_n}). A fixed seed literal below 2^63 meets the first defect with probability 0, and "
            "the single boundary value 2^64 - 1 meets it with probability 1.")
    return head + "\n\n" + _table(["defect and run", "p", "runs for 50%", "runs for 95%", "runs for 99%"], rows)


# ---------------------------------------------------------------------------------------------------------------------
# A tolerance that collapses when the terms cancel
# ---------------------------------------------------------------------------------------------------------------------

TOL_N = 32
TOL_DRAWS = 10000
H_STEP = f32(1e-3)


def _loss32(z, w):
    """L(z) = sum_i w_i z_i^3 in float32: t = (z z) z, then w t, accumulated in i order from 0."""
    acc = 0.0
    for zi, wi in zip(z, w):
        t = f32(zi * zi)
        t = f32(t * zi)
        t = f32(wi * t)
        acc = f32(acc + t)
    return acc


def _tolerance_draw(g):
    x = [f32(2.0 * g.uniform() - 1.0) for _ in range(TOL_N)]
    w = [f32(2.0 * g.uniform() - 1.0) for _ in range(TOL_N)]
    v = [f32(2.0 * g.uniform() - 1.0) for _ in range(TOL_N)]
    h = H_STEP
    hv = [f32(h * vi) for vi in v]
    xp = [f32(xi + d) for xi, d in zip(x, hv)]
    xm = [f32(xi - d) for xi, d in zip(x, hv)]
    num = (_loss32(xp, w) - _loss32(xm, w)) / (2.0 * h)
    terms = [3.0 * wi * xi * xi * vi for xi, wi, vi in zip(x, w, v)]
    a = 0.0
    for t in terms:
        a += t
    err = abs(a - num)
    old_tol = 1e-2 * max(abs(a), abs(num), 1e-3)
    s_plus = sum(abs(wi * zi ** 3) for zi, wi in zip(xp, w))
    s_minus = sum(abs(wi * zi ** 3) for zi, wi in zip(xm, w))
    rounding = _gamma(TOL_N + 2, U32) * (s_plus + s_minus) / (2.0 * h)
    perturb = sum(3.0 * abs(wi) * (abs(xi) + h * abs(vi)) ** 2 * U32 * (abs(xi) + 2.0 * h * abs(vi)) * (1.0 + U32)
                  for xi, wi, vi in zip(x, w, v)) / h
    trunc = h * h * sum(abs(wi * vi ** 3) for wi, vi in zip(w, v))
    analytic = _gamma(TOL_N + 3, U64) * sum(abs(t) for t in terms) + 2.0 * U64 * abs(num)
    bound = rounding + perturb + trunc + analytic
    return a, num, err, old_tol, bound


def _tolerance_draws():
    g = bookrng.generator("failure_catalogue.collapsing_tolerance")
    return [_tolerance_draw(g) for _ in range(TOL_DRAWS)]


_CACHE = {}


def _tolerance_results():
    if "draws" not in _CACHE:
        _CACHE["draws"] = _tolerance_draws()
    return _CACHE["draws"]


def _tolerance_counts():
    draws = _tolerance_results()
    return sum(1 for _, _, e, t, _ in draws if e > t), len(draws)


def collapsing_tolerance():
    draws = _tolerance_results()
    old_fail = [(k, d) for k, d in enumerate(draws) if d[2] > d[3]]
    new_fail = [k for k, d in enumerate(draws) if d[2] > d[4]]
    worst = max(d[2] / d[4] for d in draws)
    smallest_fail = min(abs(d[0]) for _, d in old_fail) if old_fail else None
    largest_fail = max(abs(d[0]) for _, d in old_fail) if old_fail else None
    head = (f"The function L(x) = sum_i w_i x_i^3 over n = {TOL_N} terms is evaluated in float32 (t = (x x) x, then "
            "w t, accumulated in increasing i from 0, every operation rounded to float32, no fused multiply-add). Its "
            "derivative along v is d = sum_i 3 w_i x_i^2 v_i, computed in double from the float32 inputs in increasing "
            "i. The check compares d with the central difference (L(x + h v) - L(x - h v)) / (2 h), where h is the "
            f"float32 nearest 0.001 ({H_STEP!r}), h v_i and x_i +- h v_i are rounded to float32, the two values of L "
            "are float32 and the difference and the division by 2 h are done in double. The relative rule passes when "
            "the difference is at most 0.01 max(|d|, |numeric|, 0.001); the derived rule passes when it is at most "
            "gamma_(n+2)(u) (S+ + S-) / (2 h) + (1/h) sum_i 3 |w_i| (|x_i| + h |v_i|)^2 u (|x_i| + 2 h |v_i|)(1 + u) "
            "+ h^2 sum_i |w_i v_i^3| + gamma_(n+3)(2^-53) sum_i |3 w_i x_i^2 v_i| + 2^-52 |numeric|, with u = 2^-24, "
            "gamma_k(u) = k u / (1 - k u) and S+- = sum_i |w_i z_i^3| over the two perturbed inputs z. "
            f"{TOL_DRAWS} draws of the generator seeded {bookrng.seed_of('failure_catalogue.collapsing_tolerance')} "
            "(seed_of 'failure_catalogue.collapsing_tolerance'); each draw takes the 32 x_i, then the 32 w_i, then "
            "the 32 v_i, every one 2 uniform() - 1 rounded to float32.")
    rows = [
        ["draws", TOL_DRAWS],
        ["failures of the relative rule", len(old_fail)],
        ["failures of the derived rule", len(new_fail)],
        ["smallest abs(d) among the relative rule's failures", _g(smallest_fail) if old_fail else "none"],
        ["largest abs(d) among the relative rule's failures", _g(largest_fail) if old_fail else "none"],
        ["largest ratio of difference to derived bound", _g(worst)],
    ]
    first = [[k, _g(d[0]), _g(d[1]), _g(d[2]), _g(d[3]), _g(d[4])] for k, d in old_fail[:5]]
    tab2 = _table(["draw (from 0)", "d", "numeric", "difference", "relative tolerance", "derived bound"], first)
    return (head + "\n\n" + _table(["quantity", "value"], rows) + "\n\nThe first five failures of the relative rule "
            "(numbers to four significant digits):\n\n" + tab2)
