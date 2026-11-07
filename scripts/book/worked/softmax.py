# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Softmax, Logits and Cross-Entropy".

Plain Python 3, standard library only, written from the chapter's own
description: the softmax as exponentiate-and-normalise, its reduction to the
sigmoid for two classes, invariance to a shift and the effect of a scale,
overflow and underflow in 32-bit and 64-bit arithmetic and the stable forms,
the cross-entropy of a one-hot target, the Jacobian diag(p) - p p^T and the
gradient p - t, label smoothing, and the count of operations per row. Each
public function without a leading underscore that takes no arguments returns
a markdown string that the chapter embeds between
`<!-- worked: softmax.<name> -->` markers. Underscore functions are shared
with the chapter's figures, so a picture and the table beside it cannot
disagree.

Nothing here is random: the data are one small set of inspected parts whose
numbers are listed in `defect_inputs`.

Swedish Embedded AB implements solutions for turning raw model scores into
probabilities and losses that stay accurate at large vocabulary sizes. If your
team needs expertise in the numerics of classification and language-model
losses, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct
from decimal import Decimal, getcontext

CLASSES = ["scratch", "dent", "crack", "burr"]
MEASUREMENTS = ["roughness", "depth", "length"]
C = len(CLASSES)

# The linear classifier of the running example: z = W x + b, one row of W per class.
W = [[2.0, -1.0, 1.0],
     [-1.0, 2.0, 0.5],
     [0.5, 1.0, -1.5],
     [-0.5, -1.0, 1.5]]
B = [0.1, 0.0, -0.2, 0.3]
# The part of the running example and its true class (index of "dent").
X = [1.0, 0.5, 0.8]
TRUE = 1
# Two more parts, used with the first as a batch of three.
BATCH_X = [X, [0.2, 1.2, 0.1], [0.1, 0.1, 1.5]]
BATCH_TRUE = [1, 1, 3]

ETA = 0.5          # step size of the gradient-descent example
STEPS = 6          # steps shown in the gradient-descent example
EPS_LS = 0.1       # label-smoothing weight of the running example


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _num(v, sig=6):
    """A number with `sig` significant digits; powers of ten as m x 10^e; nan and inf as words."""
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e6:
        return f"{v:.{sig}g}"
    m, e = f"{v:.{sig - 1}e}".split("e")
    if "." in m:
        m = m.rstrip("0").rstrip(".")
    return f"{m} x 10^{int(e)}"


def _num_exact(v, sig=6):
    """_num for a Decimal, without the detour through a float that would underflow below 1e-308."""
    if v == 0:
        return "0"
    if Decimal("1e-3") <= abs(v) < Decimal("1e6"):
        return _num(float(v), sig)
    m, e = f"{v:.{sig - 1}e}".split("e")
    if "." in m:
        m = m.rstrip("0").rstrip(".")
    return f"{m} x 10^{int(e)}"


def _below(v, tol=1e-15):
    """Rounding noise is printed as a bound, so the table does not depend on the last bits of a platform."""
    return f"below {tol:g}" if abs(v) < tol else _num(v, 2)


def _vec(v, nd=4):
    return "(" + ", ".join(f"{x:.{nd}f}" for x in v) + ")"


def _logits(x):
    """z = W x + b for the running classifier."""
    return [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(W, B)]


def _lse(z):
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def _softmax(z):
    """The stable softmax in 64-bit arithmetic."""
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _entropy(p):
    return -sum(v * math.log(v) for v in p if v > 0)


def _one_hot(c, n=C):
    return [1.0 if i == c else 0.0 for i in range(n)]


def _smoothed_target(c, eps, n=C):
    """(1 - eps) * one_hot(c) + eps / n for every class."""
    return [(1 - eps) * t + eps / n for t in _one_hot(c, n)]


def _jacobian(p):
    """d p_i / d z_j = p_i (delta_ij - p_j)."""
    n = len(p)
    return [[p[i] * ((1.0 if i == j else 0.0) - p[j]) for j in range(n)] for i in range(n)]


def _smoothed_loss_gap(g, eps, n=C):
    """Loss lse(z) - sum t'_i z_i for z = (g, 0, ..., 0) and the true class 0."""
    z = [g] + [0.0] * (n - 1)
    t = _smoothed_target(0, eps, n)
    return _lse(z) - sum(ti * zi for ti, zi in zip(t, z))


def _optimal_gap(eps, n=C):
    """The gap z_true - z_other at which p equals the smoothed target (infinite for eps = 0)."""
    if eps == 0:
        return math.inf
    return math.log((1 - eps + eps / n) / (eps / n))


# ---------------------------------------------------- arithmetic in a stated format

class _Fmt:
    """Arithmetic where every operation rounds (32-bit) or not (64-bit) and overflow is infinity."""

    def __init__(self, bits):
        self.bits = bits

    def r(self, v):
        if self.bits == 64 or math.isinf(v) or math.isnan(v):
            return v
        try:
            return struct.unpack("f", struct.pack("f", v))[0]
        except OverflowError:
            return math.copysign(math.inf, v)

    def exp(self, v):
        if math.isnan(v):
            return v
        try:
            return self.r(math.exp(v))
        except OverflowError:
            return math.inf

    def log(self, v):
        if math.isnan(v) or v < 0:
            return math.nan
        if v == 0:
            return -math.inf
        if math.isinf(v):
            return math.inf
        return self.r(math.log(v))

    def add(self, a, b):
        return self.r(a + b)

    def sub(self, a, b):
        return self.r(a - b)

    def mul(self, a, b):
        return self.r(a * b)

    def div(self, a, b):
        if b == 0:
            if a == 0 or math.isnan(a):
                return math.nan
            return math.copysign(math.inf, a) * math.copysign(1.0, b)
        return self.r(a / b)


F32, F64 = _Fmt(32), _Fmt(64)


def _naive_softmax(z, f):
    e = [f.exp(v) for v in z]
    s = 0.0
    for v in e:
        s = f.add(s, v)
    return [f.div(v, s) for v in e]


def _stable_softmax(z, f):
    m = max(z)
    e = [f.exp(f.sub(v, m)) for v in z]
    s = 0.0
    for v in e:
        s = f.add(s, v)
    return [f.div(v, s) for v in e]


def _lse_stable(z, f):
    m = max(z)
    s = 0.0
    for v in z:
        s = f.add(s, f.exp(f.sub(v, m)))
    return f.add(m, f.log(s))


def _exact_softmax(z):
    """Softmax in 60-digit decimal arithmetic."""
    getcontext().prec = 60
    m = max(z)
    e = [(Decimal(v) - Decimal(m)).exp() for v in z]
    s = sum(e)
    return [v / s for v in e]


def _exact_log_softmax(z):
    getcontext().prec = 60
    m = max(z)
    lse = Decimal(m) + sum((Decimal(v) - Decimal(m)).exp() for v in z).ln()
    return [Decimal(v) - lse for v in z]


# ------------------------------------------------------------------ examples

def defect_inputs():
    """The part, the classifier and its four logits."""
    z = _logits(X)
    head = ("Classes: " + ", ".join(f"{i} = {n}" for i, n in enumerate(CLASSES))
            + ". Measurements: " + ", ".join(f"{i} = {n}" for i, n in enumerate(MEASUREMENTS))
            + ". The part has x = " + _vec(X, 1) + " and its true defect is " + f"{TRUE} ({CLASSES[TRUE]}). "
            + "The classifier is z = W x + b with W of shape [4, 3] and b of shape [4].\n\n")
    rows = [[f"{c} ({CLASSES[c]})"] + [f"{w:.1f}" for w in W[c]] + [f"{B[c]:.1f}", f"{z[c]:.4f}"] for c in range(C)]
    return head + _table(["class", "W[c,0]", "W[c,1]", "W[c,2]", "b[c]", "z[c] = W[c] . x + b[c]"], rows)


def scores_to_probabilities():
    """Two ways to turn scores into a probability vector: divide by the sum, exponentiate then divide."""
    z = _logits(X)
    s = sum(z)
    e = [math.exp(v) for v in z]
    se = sum(e)
    rows = [[f"{c} ({CLASSES[c]})", f"{z[c]:.4f}", f"{z[c] / s:.4f}", f"{e[c]:.4f}", f"{e[c] / se:.4f}"]
            for c in range(C)]
    rows.append(["total", f"{s:.4f}", f"{sum(v / s for v in z):.4f}", f"{se:.4f}", f"{sum(v / se for v in e):.4f}"])
    return _table(["class", "z", "z / sum(z)", "e^z", "p = e^z / sum(e^z)"], rows)


def two_classes_are_a_sigmoid():
    """For two classes, softmax(z)[1] = sigmoid(z1 - z0)."""
    pairs = [(0.0, 0.0), (1.0, 2.0), (3.0, -1.0), (-2.0, 2.5), (40.0, 41.0)]
    rows = []
    worst = 0.0
    for z0, z1 in pairs:
        p = _softmax([z0, z1])
        sg = _sigmoid(z1 - z0)
        worst = max(worst, abs(p[1] - sg))
        rows.append([f"({z0:g}, {z1:g})", f"{z1 - z0:g}", f"{p[0]:.6f}", f"{p[1]:.6f}", f"{sg:.6f}"])
    return (_table(["(z0, z1)", "z1 - z0", "softmax[0]", "softmax[1]", "sigmoid(z1 - z0)"], rows)
            + f"\n\nLargest absolute difference between the last two columns: {_num(worst, 2)}.")


def log_ratio_and_shift():
    """log(p_i / p_j) = z_i - z_j, and softmax(z + c) = softmax(z)."""
    z = _logits(X)
    p = _softmax(z)
    pairs = [(0, 1), (0, 2), (3, 1)]
    rows = [[f"{CLASSES[i]} / {CLASSES[j]}", f"{p[i]:.4f} / {p[j]:.4f}", f"{math.log(p[i] / p[j]):.4f}",
             f"{z[i] - z[j]:.4f}"] for i, j in pairs]
    t1 = _table(["pair (i / j)", "p_i / p_j", "ln(p_i / p_j)", "z_i - z_j"], rows)
    shifts = [0.0, 10.0, -10.0, 1000.0]
    rows2 = []
    for c in shifts:
        q = _softmax([v + c for v in z])
        rows2.append([f"{c:g}"] + [f"{v:.6f}" for v in q])
    t2 = _table(["shift c"] + [f"p[{n}]" for n in CLASSES], rows2)
    return ("Log of the ratio of two probabilities against the difference of their logits:\n\n" + t1
            + "\n\nThe softmax of z + c for the logits of the part, computed by subtracting the largest logit before the exponentials (the stable softmax of the next section):\n\n" + t2)


def scale_the_logits():
    """softmax(a z) for several scales a: flat, as it was, sharper."""
    z = _logits(X)
    scales = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    rows = []
    for a in scales:
        p = _softmax([a * v for v in z])
        rows.append([f"{a:g}"] + [f"{v:.4f}" for v in p] + [f"{max(p):.4f}", f"{_entropy(p):.4f}"])
    note = f"\n\nThe entropy of the uniform distribution on {C} classes is ln {C} = {math.log(C):.4f}."
    return _table(["scale a"] + [f"p[{n}]" for n in CLASSES] + ["largest p", "entropy (nats)"], rows) + note


def overflow_and_underflow():
    """The textbook softmax in 32-bit and 64-bit arithmetic, against the max-subtracted one and the exact one."""
    cases = [
        ("z = (2.4, 0.4, -0.4, 0.5)", [2.4, 0.4, -0.4, 0.5], F32),
        ("z = (100, 98, 90, 0)", [100.0, 98.0, 90.0, 0.0], F32),
        ("z = (-120, -121, -125, -130)", [-120.0, -121.0, -125.0, -130.0], F32),
        ("z = (1000, 998, 990, 0)", [1000.0, 998.0, 990.0, 0.0], F64),
        ("z = (-1000, -1001, -1005, -1010)", [-1000.0, -1001.0, -1005.0, -1010.0], F64),
    ]
    rows = []
    for name, z, f in cases:
        naive = _naive_softmax(z, f)
        stable = _stable_softmax(z, f)
        exact = _exact_softmax(z)
        rows.append([name, f"{f.bits}-bit", "(" + ", ".join(_num(v, 5) for v in naive) + ")",
                     "(" + ", ".join(_num(v, 5) for v in stable) + ")",
                     "(" + ", ".join(_num_exact(v, 5) for v in exact) + ")"])
    head = ("Every operation is rounded to the stated format one at a time (`exp`, `+`, `-`, `/`), an `exp` "
            "above the largest finite number gives `inf`, and `inf / inf` and `0 / 0` are `nan`. "
            "\"naive\" is `exp(z[i]) / sum(exp(z))`; \"stable\" subtracts the largest logit first; "
            "\"exact\" is computed in 60-digit decimal arithmetic.\n\n")
    return head + _table(["logits", "format", "naive", "stable", "exact"], rows, "llllr")


def log_softmax_underflow():
    """log of a softmax output against z - lse(z), in 32-bit arithmetic."""
    z = [30.0, 0.0, -60.0, -130.0]
    f = F32
    p = _stable_softmax(z, f)
    lse = _lse_stable(z, f)
    exact = _exact_log_softmax(z)
    rows = []
    for i, v in enumerate(z):
        rows.append([f"{v:g}", _num(p[i], 5), _num(f.log(p[i]), 6), _num(f.sub(v, lse), 6), _num(float(exact[i]), 6)])
    head = (f"Logits z = (30, 0, -60, -130), 32-bit arithmetic one operation at a time. "
            f"The stable log-sum-exp is {_num(lse, 8)} in 32 bits.\n\n")
    return head + _table(["z[i]", "stable softmax p[i]", "log(p[i])", "z[i] - lse(z)", "exact log p[i]"], rows)


def cross_entropy_by_class():
    """The loss of the running example for each possible true class, and the identity lse - z[c] = -ln p[c]."""
    z = _logits(X)
    p = _softmax(z)
    lse = _lse(z)
    rows = []
    for c in range(C):
        t = _one_hot(c)
        ce_def = -sum(ti * math.log(pi) for ti, pi in zip(t, p))
        rows.append([f"{c} ({CLASSES[c]})", "(" + ", ".join(f"{int(v)}" for v in t) + ")", f"{p[c]:.4f}",
                     f"{ce_def:.4f}", f"{lse - z[c]:.4f}"])
    head = (f"Logits z = {_vec(z)}; probabilities p = {_vec(p)}; lse(z) = {lse:.4f}; "
            f"a uniform guess costs ln {C} = {math.log(C):.4f}.\n\n")
    return head + _table(["true class c", "one-hot t", "p[c]", "-sum t log p", "lse(z) - z[c]"], rows)


def softmax_jacobian():
    """The Jacobian of the softmax at the running example, its row sums and a numerical check."""
    z = _logits(X)
    p = _softmax(z)
    J = _jacobian(p)
    h = 1e-6
    worst = 0.0
    for j in range(C):
        zp = list(z)
        zm = list(z)
        zp[j] += h
        zm[j] -= h
        pp, pm = _softmax(zp), _softmax(zm)
        for i in range(C):
            worst = max(worst, abs((pp[i] - pm[i]) / (2 * h) - J[i][j]))
    rows = [[f"{i} ({CLASSES[i]})"] + [f"{J[i][j]:.4f}" for j in range(C)] for i in range(C)]
    head = f"p = {_vec(p)}.\n\n"
    row_sum = max(abs(sum(r)) for r in J)
    sym = max(abs(J[i][j] - J[j][i]) for i in range(C) for j in range(C))
    return (head + _table(["J[i, j]"] + [f"j={j}" for j in range(C)], rows)
            + f"\n\nLargest absolute row sum: {_below(row_sum)}. Largest asymmetry |J[i, j] - J[j, i]|: {_below(sym)}. "
            f"Largest difference from central differences with step {h:g}: {_below(worst, 1e-9)}.")


def gradient_through_the_jacobian():
    """dloss/dz for the running example, three ways, and the general softmax VJP."""
    z = _logits(X)
    p = _softmax(z)
    J = _jacobian(p)
    t = _one_hot(TRUE)
    u = [-t[i] / p[i] for i in range(C)]
    vjp = [sum(u[i] * J[i][j] for i in range(C)) for j in range(C)]
    closed = [p[j] - t[j] for j in range(C)]
    h = 1e-6
    fd = []
    for j in range(C):
        zp = list(z)
        zm = list(z)
        zp[j] += h
        zm[j] -= h
        fd.append((_lse(zp) - zp[TRUE] - (_lse(zm) - zm[TRUE])) / (2 * h))
    rows = [[f"{c} ({CLASSES[c]})", f"{p[c]:.4f}", f"{int(t[c])}", f"{u[c] + 0.0:.4f}", f"{vjp[c]:.4f}", f"{closed[c]:.4f}",
             f"{fd[c]:.4f}"] for c in range(C)]
    t1 = _table(["class", "p", "t", "u = -t / p", "u J", "p - t", "central difference"], rows)
    up = [1.0, -2.0, 0.5, 3.0]
    dot = sum(a * b for a, b in zip(up, p))
    direct = [sum(up[i] * J[i][j] for i in range(C)) for j in range(C)]
    formula = [p[j] * (up[j] - dot) for j in range(C)]
    rows2 = [[f"{c} ({CLASSES[c]})", f"{up[c]:g}", f"{direct[c]:.4f}", f"{formula[c]:.4f}"] for c in range(C)]
    t2 = _table(["class", "upstream u", "u J (matrix product)", "p * (u - u . p)"], rows2)
    return (f"True class {TRUE} ({CLASSES[TRUE]}), p = {_vec(p)}. The loss is -ln p[{TRUE}]; "
            f"its derivative with respect to p is u = -t / p.\n\n" + t1
            + f"\n\nAn arbitrary upstream vector u = {_vec(up, 1)} sent through the same Jacobian "
            f"(u . p = {dot:.4f}):\n\n" + t2)


def fused_against_composed():
    """A confidently wrong prediction in 32 bits: log of a softmax output against the fused loss."""
    z = [30.0, -100.0, 5.0, 0.0]
    c = 1
    f = F32
    p = _stable_softmax(z, f)
    # composed: loss = -log(p[c]); backward through the log, then through the softmax Jacobian
    loss_comp = -f.log(p[c])
    u = [f.div(-1.0 if i == c else 0.0, p[i]) for i in range(C)]
    J = [[f.mul(p[i], f.sub(1.0 if i == j else 0.0, p[j])) for j in range(C)] for i in range(C)]
    g_comp = []
    for j in range(C):
        acc = 0.0
        for i in range(C):
            acc = f.add(acc, f.mul(u[i], J[i][j]))
        g_comp.append(acc)
    # fused
    lse = _lse_stable(z, f)
    loss_fused = f.sub(lse, z[c])
    g_fused = [f.sub(p[i], 1.0 if i == c else 0.0) for i in range(C)]
    # squared error on the probabilities, in 64 bits: loss sum (p - t)^2 and gradient 2 (p - t) J
    p64 = _softmax(z)
    t = _one_hot(c)
    J64 = _jacobian(p64)
    loss_sq = sum((p64[i] - t[i]) ** 2 for i in range(C))
    g_sq = [2 * sum((p64[i] - t[i]) * J64[i][j] for i in range(C)) for j in range(C)]
    rows = [["loss", _num(loss_comp, 6), _num(loss_fused, 6), _num(loss_sq, 6)]]
    for i in range(C):
        rows.append([f"gradient[{i}] ({CLASSES[i]})", _num(g_comp[i], 6), _num(g_fused[i], 6), _num(g_sq[i], 3)])
    head = (f"Logits z = (30, -100, 5, 0), true class {c} ({CLASSES[c]}), 32-bit arithmetic one operation at a "
            f"time for the first two columns. Stable softmax p = ({', '.join(_num(v, 5) for v in p)}). "
            "The last column is a squared error on the probabilities, sum (p - t)^2, with its gradient "
            "2 (p - t) J, in 64-bit arithmetic.\n\n")
    return head + _table(["", "composed: -log(p[c]), then back through log and softmax",
                          "fused: lse - z[c], gradient p - t", "squared error on p"], rows)


def gradient_descent_on_weights():
    """Gradient descent on W and b for the single part, with dW = (p - t) x^T."""
    w = [list(r) for r in W]
    b = list(B)
    z = _logits(X)
    p = _softmax(z)
    t = _one_hot(TRUE)
    dz = [p[i] - t[i] for i in range(C)]
    dW = [[dz[i] * X[j] for j in range(len(X))] for i in range(C)]
    t1 = _table(["dW[i, j] = (p - t)[i] x[j]"] + [f"j={j}" for j in range(len(X))] + ["db[i] = (p - t)[i]"],
                [[f"{i} ({CLASSES[i]})"] + [f"{v:.4f}" for v in dW[i]] + [f"{dz[i]:.4f}"] for i in range(C)])
    rows = []
    for k in range(STEPS + 1):
        z = [sum(wij * xj for wij, xj in zip(w[i], X)) + b[i] for i in range(C)]
        p = _softmax(z)
        pred = max(range(C), key=lambda i: z[i])
        rows.append([k, _vec(p), f"{_lse(z) - z[TRUE]:.4f}", CLASSES[pred]])
        dz = [p[i] - t[i] for i in range(C)]
        for i in range(C):
            for j in range(len(X)):
                w[i][j] -= ETA * dz[i] * X[j]
            b[i] -= ETA * dz[i]
    t2 = _table(["step k", "p", "loss", "predicted"], rows, "rlrl")
    return (f"Gradient at the starting weights for the part x = {_vec(X, 1)} and true class {TRUE} ({CLASSES[TRUE]}):\n\n" + t1
            + f"\n\nPlain gradient descent, w <- w - eta dW and b <- b - eta db with eta = {ETA}, "
            f"from the weights of the first table of the chapter:\n\n" + t2)


def a_batch():
    """Mean loss and the gradient with respect to the logits of a batch of three parts."""
    n = len(BATCH_X)
    rows = []
    zs, ps = [], []
    for k, (x, c) in enumerate(zip(BATCH_X, BATCH_TRUE)):
        z = _logits(x)
        p = _softmax(z)
        zs.append(z)
        ps.append(p)
        rows.append([k, _vec(x, 1), CLASSES[c], _vec(z), _vec(p), f"{_lse(z) - z[c]:.4f}"])
    mean = sum(_lse(z) - z[c] for z, c in zip(zs, BATCH_TRUE)) / n
    t1 = _table(["row", "x", "true", "z", "p", "loss"], rows, "rllllr")
    rows2 = []
    for k in range(n):
        g = [(ps[k][i] - (1.0 if i == BATCH_TRUE[k] else 0.0)) / n for i in range(C)]
        rows2.append([k] + [f"{v:.4f}" for v in g])
    t2 = _table(["row"] + [f"dZ[k, {n_}]" for n_ in CLASSES], rows2)
    return (t1 + f"\n\nMean loss over the {n} rows: {mean:.4f}. Gradient of the mean loss with respect to the "
            "logits, (p - t) / 3 per row:\n\n" + t2)


def label_smoothing_target():
    """The smoothed target, the loss and the gradient at the running example."""
    z = _logits(X)
    p = _softmax(z)
    t = _one_hot(TRUE)
    ts = _smoothed_target(TRUE, EPS_LS)
    lse = _lse(z)
    loss_hard = lse - z[TRUE]
    loss_soft = lse - sum(a * b for a, b in zip(ts, z))
    rows = [[f"{c} ({CLASSES[c]})", f"{int(t[c])}", f"{ts[c]:.4f}", f"{p[c]:.4f}", f"{p[c] - t[c]:.4f}",
             f"{p[c] - ts[c]:.4f}"] for c in range(C)]
    check = -sum(a * math.log(b) for a, b in zip(ts, p))
    tail = (f"\n\nWith eps = {EPS_LS}: loss against the one-hot target {loss_hard:.4f}, against the smoothed target "
            f"{loss_soft:.4f}; the definition -sum t' ln p gives {check:.4f}. The smoothed target sums to "
            f"{sum(ts):.4f}.")
    return _table(["class", "t", "t' = (1 - eps) t + eps / C", "p", "gradient p - t", "gradient p - t'"], rows) + tail


def label_smoothing_optimum():
    """Where the loss is smallest: the logit gap that makes p equal the target, and the floor of the loss."""
    rows = []
    for eps in [0.0, 0.05, 0.1, 0.2]:
        ts = _smoothed_target(0, eps)
        gap = _optimal_gap(eps)
        floor = abs(_entropy(ts))
        rows.append([f"{eps:g}", f"{ts[0]:.4f}", f"{ts[1]:.4f}", "none (the gap grows without bound)" if math.isinf(gap) else f"{gap:.4f}",
                     f"{floor:.4f}"])
    t1 = _table(["eps", "t'[true]", "t'[other]", "best gap z_true - z_other", "smallest loss = entropy of t'"], rows)
    gaps = [0.0, 2.0, 3.0, 3.6109, 4.0, 6.0, 8.0, 12.0]
    rows2 = []
    for g in gaps:
        rows2.append([f"{g:g}"] + [f"{_smoothed_loss_gap(g, eps):.4f}" for eps in [0.0, 0.1, 0.2]])
    t2 = _table(["gap g", "loss, eps = 0", "loss, eps = 0.1", "loss, eps = 0.2"], rows2)
    return (f"{C} classes, true class 0, logits z = (g, 0, 0, 0):\n\n" + t1
            + "\n\nThe loss as the true logit grows (the other three logits stay at 0):\n\n" + t2)


def multi_label_against_multi_class():
    """The same logits through a softmax and through one sigmoid per class."""
    z = _logits(X)
    p = _softmax(z)
    rows = [[f"{c} ({CLASSES[c]})", f"{z[c]:.4f}", f"{p[c]:.4f}", f"{_sigmoid(z[c]):.4f}"] for c in range(C)]
    rows.append(["sum", "", f"{sum(p):.4f}", f"{sum(_sigmoid(v) for v in z):.4f}"])
    return _table(["class", "z", "softmax (one class)", "sigmoid per class (any subset)"], rows)


# ---------------------------------------------------------------- work counts

class _Counter:
    """Counts the operations of an algorithm as it runs, one per compare, add, subtract, divide, exp or log."""

    def __init__(self):
        self.ops = 0
        self.exps = 0

    def op(self, n=1):
        self.ops += n

    def exp(self, n=1):
        self.ops += n
        self.exps += n


def _count_softmax(n):
    k = _Counter()
    k.op(n - 1)          # maximum
    for _ in range(n):
        k.op()           # z - m
        k.exp()          # exp
        k.op()           # running sum
    for _ in range(n):
        k.op()           # divide by the sum
    return k


def _count_ce_value(n):
    k = _Counter()
    k.op(n - 1)
    for _ in range(n):
        k.op()
        k.exp()
        k.op()
    k.op(3)              # log, add the maximum, subtract z[c]
    return k


def _count_stats(n):
    k = _Counter()
    k.op(n - 1)
    for _ in range(n):
        k.op()
        k.exp()
        k.op()
    return k


def _count_grad_element():
    k = _Counter()
    k.op()               # z - m
    k.exp()
    k.op()               # divide by the sum
    k.op()               # minus the one-hot entry
    k.op()               # divide by the row count
    return k


def _count_two_pass(n):
    k = _count_stats(n)
    for _ in range(n):
        e = _count_grad_element()
        k.ops += e.ops
        k.exps += e.exps
    return k


def _count_naive_grad(n):
    """The gradient with one thread per element, each recomputing the row statistics."""
    k = _Counter()
    for _ in range(n):
        s = _count_stats(n)
        e = _count_grad_element()
        k.ops += s.ops + e.ops
        k.exps += s.exps + e.exps
    return k


def _ops_formulas(n):
    return {
        "softmax": (5 * n - 1, n),
        "ce_value": (4 * n + 2, n),
        "two_pass": (9 * n - 1, 2 * n),
        "naive": (4 * n * n + 4 * n, n * (n + 1)),
    }


def work_counts():
    """Operations per row for the softmax, the loss and its gradient, counted and from the formulas."""
    check_rows = []
    for n in [4, 16, 64]:
        counted = {"softmax": _count_softmax(n), "ce_value": _count_ce_value(n),
                   "two_pass": _count_two_pass(n), "naive": _count_naive_grad(n)}
        form = _ops_formulas(n)
        ok = all((counted[k].ops, counted[k].exps) == form[k] for k in form)
        check_rows.append([n] + [f"{counted[k].ops} ({counted[k].exps})" for k in ["softmax", "ce_value", "two_pass", "naive"]]
                          + ["yes" if ok else "NO"])
    t1 = _table(["C", "softmax", "loss value", "gradient, two passes", "gradient, one thread per element", "formulas agree"],
                check_rows)
    rows = []
    for n in [4, 50_000, 151_936]:
        f = _ops_formulas(n)
        rows.append([f"{n:,}", f"{f['softmax'][0]:,}", f"{f['ce_value'][0]:,}", f"{f['two_pass'][0]:,}",
                     f"{f['naive'][0]:,}", f"{f['naive'][0] / f['two_pass'][0]:,.0f}"])
    t2 = _table(["C", "softmax", "loss value", "gradient, two passes", "gradient, one thread per element",
                 "ratio naive / two passes"], rows)
    return ("Operations per row, counted by running each algorithm with a counter (each compare, add, subtract, "
            "divide, `exp` and `log` is one operation); the number of `exp` calls is in brackets.\n\n" + t1
            + "\n\nThe formulas (5C - 1, 4C + 2, 9C - 1 and 4C^2 + 4C operations; C, C, 2C and C(C + 1) exps), "
            "evaluated for larger C:\n\n" + t2)


def logits_matrix_size():
    """Bytes of the logits of a batch of tokens and the arithmetic of the layer that makes them."""
    d = 1024
    n_rows = 4096
    rows = []
    for v in [4, 50_000, 248_320]:
        per_row = 4 * v
        mat = per_row * n_rows
        mm = 2 * d * v
        ce = _ops_formulas(v)["ce_value"][0]
        rows.append([f"{v:,}", f"{per_row:,}", f"{mat:,}", f"{mm:,}", f"{ce:,}", f"{ce / mm:.4f}"])
    head = (f"f32 logits ({4} bytes each), a batch of {n_rows:,} rows, and a final linear layer with D = {d} inputs "
            "(2 D C operations per row to make the logits).\n\n")
    return head + _table(["C", "logit bytes per row", f"bytes for {n_rows:,} rows", "operations of the layer, per row",
                          "operations of the loss value, per row", "loss / layer"], rows)
