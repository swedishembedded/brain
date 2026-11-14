# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Backward Pass of Every Layer".

Plain Python 3 on floats (IEEE double precision), written from the chapter's
own description of each rule and independent of any engine code. Every public
function returns a markdown string that the chapter embeds between
`<!-- worked: layer_gradients.<name> -->` markers. The same module supplies the
sizes of the chapter's figures.

Tensors are nested lists in row-major order. Each rule is checked against a
central difference of the scalar L = sum(G * f(inputs)), whose exact gradient
with respect to the output is G, so the check measures the rule's
vector-Jacobian product directly. That check is a preview of the next chapter;
here it only confirms that a rule was written down correctly.

Swedish Embedded AB implements solutions for verifiable gradient computation
in neural network training. If your team needs expertise in the backward pass
of neural network layers, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng

# ------------------------------------------------------------------ tensors


def shape(t):
    s = []
    while isinstance(t, list):
        s.append(len(t))
        t = t[0]
    return tuple(s)


def flat(t):
    if not isinstance(t, list):
        return [t]
    out = []
    for r in t:
        out.extend(flat(r))
    return out


def build(data, shp):
    """Nested lists of shape `shp` from row-major `data` (shape () is a number)."""
    if not shp:
        return data[0]
    step = len(data) // shp[0]
    return [build(data[i * step:(i + 1) * step], shp[1:]) for i in range(shp[0])]


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def tr(a):
    return [list(r) for r in zip(*a)]


def colsum(a):
    return [sum(r[j] for r in a) for j in range(len(a[0]))]


def rowsum(a):
    return [sum(r) for r in a]


def zip2(f, a, b):
    return [[f(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mapm(f, a):
    return [[f(x) for x in r] for r in a]


def madd(a, b):
    return zip2(lambda x, y: x + y, a, b)


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(flat(a), flat(b)))


def central_vjp(f, args, g, h=1e-6):
    """Central difference of L = sum(g * f(*args)) with respect to every entry of every argument."""
    gf = flat(g)
    grads = []
    for i, a in enumerate(args):
        shp, data = shape(a), flat(a)
        gi = []
        for j in range(len(data)):
            old = data[j]
            vals = []
            for d in (h, -h):
                data[j] = old + d
                trial = [build(data, shp) if k == i else args[k] for k in range(len(args))]
                vals.append(flat(f(*trial)))
            data[j] = old
            gi.append(sum(w * (p - m) for w, p, m in zip(gf, *vals)) / (2 * h))
        grads.append(build(gi, shp))
    return grads


# --------------------------------------------------------------- formatting


def mat_table(title, m, nd=2, row="b", col="o"):
    n = len(m[0])
    out = [f"| {title} | " + " | ".join(f"{col}={j}" for j in range(n)) + " |", "|---|" + "---:|" * n]
    for i, r in enumerate(m):
        out.append(f"| {row}={i} | " + " | ".join(f"{v:.{nd}f}" for v in r) + " |")
    return "\n".join(out)


def vec_table(title, v, nd=2, col="o"):
    return "\n".join([f"| {title} | " + " | ".join(f"{col}={j}" for j in range(len(v))) + " |",
                      "|---|" + "---:|" * len(v),
                      "| value | " + " | ".join(f"{x:.{nd}f}" for x in v) + " |"])


def short(m):
    """One-line text of a nested list of numbers, integers without a decimal point."""
    if isinstance(m, list):
        return "[" + ", ".join(short(x) for x in m) + "]"
    return str(int(m)) if float(m).is_integer() else f"{m:.4g}"


# ------------------------------------------------------ the running example

# A batched linear layer Y = X W^T + b with B = 3 examples, D = 2 inputs and
# O = 4 outputs, all different so that a transposed matrix is a shape error.
LIN_X = [[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]]
LIN_W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
LIN_B = [0.5, -0.5, 0.0, 1.0]
# The gradient arriving from above, dL/dY: the loss is L = sum(G * Y).
LIN_G = [[1.0, -1.0, 2.0, 0.0], [0.0, 2.0, -1.0, 1.0], [-2.0, 1.0, 0.0, 3.0]]
LIN_SIZES = (len(LIN_X), len(LIN_X[0]), len(LIN_W))  # B, D, O


def linear_fwd(x, w, b):
    return [[v + b[j] for j, v in enumerate(r)] for r in mm(x, tr(w))]


def linear_bwd(x, w, dy):
    """dX = dY W, dW = dY^T X, db = sum over the batch of dY."""
    return mm(dy, w), mm(tr(dy), x), colsum(dy)


def linear_inputs():
    b, d, o = LIN_SIZES
    return "\n\n".join([
        f"A linear layer with B = {b} examples, D = {d} inputs and O = {o} outputs, `Y = X W^T + b`.",
        mat_table("X [B, D]", LIN_X, 1, "b", "d"),
        mat_table("W [O, D]", LIN_W, 1, "o", "d"),
        vec_table("b [O]", LIN_B, 1),
        "The loss is `L = sum over b, o of G[b, o] * Y[b, o]`, so the gradient arriving at `Y` is `dY = G`:",
        mat_table("dY = G [B, O]", LIN_G, 1)])


def linear_forward():
    y = linear_fwd(LIN_X, LIN_W, LIN_B)
    loss = sum(g * v for rg, ry in zip(LIN_G, y) for g, v in zip(rg, ry))
    return "\n\n".join([mat_table("Y = X W^T + b [B, O]", y, 1), f"The loss is L = {loss:.1f}."])


def linear_backward():
    dx, dw, db = linear_bwd(LIN_X, LIN_W, LIN_G)
    terms = " + ".join(f"({LIN_G[i][2]:g})({LIN_X[i][1]:g})" for i in range(len(LIN_X)))
    val = sum(LIN_G[i][2] * LIN_X[i][1] for i in range(len(LIN_X)))
    tx = " + ".join(f"({LIN_G[0][o]:g})({LIN_W[o][1]:g})" for o in range(len(LIN_W)))
    vx = sum(LIN_G[0][o] * LIN_W[o][1] for o in range(len(LIN_W)))
    out = [mat_table("dX = dY W [B, D]", dx, 1, "b", "d"),
           mat_table("dW = dY^T X [O, D]", dw, 1, "o", "d"),
           vec_table("db = sum over b of dY [O]", db, 1),
           f"One entry of each, from the sums over the shared index: dW[2, 1] = sum over b of dY[b, 2] X[b, 1] = "
           f"{terms} = {val:g}; dX[0, 1] = sum over o of dY[0, o] W[o, 1] = {tx} = {vx:g}."]
    return "\n\n".join(out)


def linear_check():
    args = [LIN_X, LIN_W, LIN_B]
    num = central_vjp(linear_fwd, args, LIN_G)
    ana = [*linear_bwd(LIN_X, LIN_W, LIN_G)]
    rows = ["| gradient | shape | largest absolute difference |", "|---|---|---:|"]
    for name, a, n in zip(["dX", "dW", "db"], ana, num):
        rows.append(f"| {name} | {list(shape(a))} | {maxdiff(a, n):.1e} |")
    return "Central difference with step 1e-6 against the three rules, every entry:\n\n" + "\n".join(rows)


# --------------------------------------------------------------- matmul, bmm


def matmul_bwd(a, b, dc):
    """C = A B: dA = dC B^T, dB = A^T dC."""
    return mm(dc, tr(b)), mm(tr(a), dc)


def _ints(rng, shp, span=7):
    n = 1
    for s in shp:
        n *= s
    return build([float(rng.integer(span) - span // 2) for _ in range(n)], shp)


def batched_matmul():
    name = "layer_gradients.bmm"
    rng = bookrng.generator(name)
    n, m, k, p = 2, 2, 3, 4
    a = _ints(rng, (n, m, k))
    b = _ints(rng, (k, p))
    dc = _ints(rng, (n, m, p))
    da = [matmul_bwd(a[i], b, dc[i])[0] for i in range(n)]
    parts = [matmul_bwd(a[i], b, dc[i])[1] for i in range(n)]
    db = madd(parts[0], parts[1])

    def f(a_, b_):
        return [mm(a_[i], b_) for i in range(n)]

    num = central_vjp(f, [a, b], dc)
    lines = [f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}). Draw order: every entry is `integer(7) - 3`, so a value in -3 .. 3; "
             f"first `A` ({n} x {m} x {k} = {n * m * k} values, batch index slowest), then the shared `B` "
             f"({k} x {p} = {k * p} values), then `dC` ({n} x {m} x {p} = {n * m * p} values).", "",
             "Forward: `C[i] = A[i] B` for batch item `i`, one weight matrix `B` shared by both items.", ""]
    for i in range(n):
        lines.append(f"- `A[{i}]` = {short(a[i])}, `dC[{i}]` = {short(dc[i])}")
    lines += [f"- `B` = {short(b)}", ""]
    lines += ["| gradient | value |", "|---|---|"]
    for i in range(n):
        lines.append(f"| dA[{i}] = dC[{i}] B^T | {short(da[i])} |")
    for i in range(n):
        lines.append(f"| A[{i}]^T dC[{i}] | {short(parts[i])} |")
    lines.append(f"| dB = A[0]^T dC[0] + A[1]^T dC[1] | {short(db)} |")
    lines += ["", f"Largest absolute difference from a central difference: dA {maxdiff(da, num[0]):.1e}, "
                  f"dB {maxdiff(db, num[1]):.1e}."]
    return "\n".join(lines)


# ----------------------------------------------------------------- elementwise


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_prime(x):
    cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return cdf + x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def silu(x):
    return x * sigmoid(x)


def silu_prime(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


# name -> (f, f' from the input x, f' from the output y or None, what the backward keeps)
ELEMENTWISE = {
    "tanh": (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2, lambda y: 1.0 - y * y, "output y"),
    "sigmoid": (sigmoid, lambda x: sigmoid(x) * (1.0 - sigmoid(x)), lambda y: y * (1.0 - y), "output y"),
    "ReLU": (lambda x: max(x, 0.0), lambda x: 1.0 if x > 0 else 0.0, lambda y: 1.0 if y > 0 else 0.0,
             "one bit per entry (y > 0)"),
    "GELU": (gelu, gelu_prime, None, "input x"),
    "SiLU": (silu, silu_prime, None, "input x"),
}

ELEM_X = 0.5


def elementwise_rules():
    rows = [f"Derivatives at x = {ELEM_X}.", "",
            "| function | f(x) | f'(x) from the input | f'(x) from the output | the backward keeps |",
            "|---|---:|---:|---:|---|"]
    for name, (f, fp_x, fp_y, keep) in ELEMENTWISE.items():
        y = f(ELEM_X)
        via_y = f"{fp_y(y):.6f}" if fp_y else "not a function of y"
        rows.append(f"| {name} | {y:.6f} | {fp_x(ELEM_X):.6f} | {via_y} | {keep} |")
    return "\n".join(rows)


def _bisect(fn, lo, hi, steps=60):
    """A root of fn in [lo, hi], where fn(lo) and fn(hi) have opposite signs."""
    flo = fn(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gelu_two_inputs():
    target = -0.1
    x1 = _bisect(lambda x: gelu(x) - target, -3.0, -0.75)
    x2 = _bisect(lambda x: gelu(x) - target, -0.75, 0.0)
    return "\n".join([
        f"GELU is not monotonic: below its minimum (near x = -0.75) it falls, above it rises. "
        f"The output {target} is produced by two different inputs, found by 60 bisection steps on "
        f"[-3, -0.75] and [-0.75, 0].", "",
        "| input x | gelu(x) | gelu'(x) |", "|---:|---:|---:|",
        f"| {x1:.6f} | {gelu(x1):.6f} | {gelu_prime(x1):.6f} |",
        f"| {x2:.6f} | {gelu(x2):.6f} | {gelu_prime(x2):.6f} |"])


TANH_X = [-1.0, 0.0, 0.5, 2.0]
TANH_DY = [0.5, -1.0, 2.0, 1.0]


def elementwise_vjp():
    y = [math.tanh(v) for v in TANH_X]
    dx_out = [d * (1.0 - v * v) for d, v in zip(TANH_DY, y)]
    dx_in = [d * (1.0 - math.tanh(v) ** 2) for d, v in zip(TANH_DY, TANH_X)]

    def f(x):
        return [math.tanh(v) for v in x]

    num = central_vjp(f, [TANH_X], TANH_DY)[0]
    rows = ["| i | x | y = tanh(x) | dy | dx = dy * (1 - y^2) | dx from the input | central difference |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for i in range(len(TANH_X)):
        rows.append(f"| {i} | {TANH_X[i]:g} | {y[i]:.6f} | {TANH_DY[i]:g} | {dx_out[i]:.6f} | "
                    f"{dx_in[i]:.6f} | {num[i]:.6f} |")
    return "\n".join(rows)


BIN_X = [1.0, -2.0, 3.0]
BIN_Y = [2.0, 4.0, -1.0]
BIN_DZ = [1.0, 0.5, -2.0]


def binary_rules():
    x, y, dz = BIN_X, BIN_Y, BIN_DZ
    rules = {
        "z = x + y": (lambda a, b: [p + q for p, q in zip(a, b)], [d for d in dz], [d for d in dz]),
        "z = x - y": (lambda a, b: [p - q for p, q in zip(a, b)], [d for d in dz], [-d for d in dz]),
        "z = x * y": (lambda a, b: [p * q for p, q in zip(a, b)], [d * q for d, q in zip(dz, y)],
                      [d * p for d, p in zip(dz, x)]),
        "z = x / y": (lambda a, b: [p / q for p, q in zip(a, b)], [d / q for d, q in zip(dz, y)],
                      [-d * p / (q * q) for d, p, q in zip(dz, x, y)]),
    }
    rows = [f"Inputs x = {short(x)}, y = {short(y)}, upstream dz = {short(dz)}.", "",
            "| operation | dx | dy | largest difference from central difference |", "|---|---|---|---:|"]
    for name, (f, dx, dy) in rules.items():
        nx, ny = central_vjp(f, [x, y], dz)
        rows.append(f"| {name} | {short([round(v, 4) for v in dx])} | {short([round(v, 4) for v in dy])} | "
                    f"{max(maxdiff(dx, nx), maxdiff(dy, ny)):.1e} |")
    return "\n".join(rows)


# ------------------------------------------------------------------ reductions

RED_A = [[1.0, 3.0, 2.0], [4.0, 4.0, 0.0]]
RED_DR = [2.0, -1.0]


def reductions():
    a, dr = RED_A, RED_DR
    n = len(a[0])
    sum_dx = [[d] * n for d in dr]
    mean_dx = [[d / n] * n for d in dr]
    max_dx = []
    for row, d in zip(a, dr):
        j = max(range(n), key=lambda c: (row[c], -c))  # first index of the maximum
        max_dx.append([d if c == j else 0.0 for c in range(n)])

    def rsum(x):
        return [sum(r) for r in x]

    def rmean(x):
        return [sum(r) / len(r) for r in x]

    def rmax(x):
        return [max(r) for r in x]

    chk = {"sum": maxdiff(sum_dx, central_vjp(rsum, [a], dr)[0]),
           "mean": maxdiff(mean_dx, central_vjp(rmean, [a], dr)[0])}
    # the maximum of row 1 is tied at columns 0 and 1; row 0 has a unique maximum
    num = central_vjp(rmax, [a], dr)[0]
    lines = [f"Input A = {short(a)} (2 rows, 3 columns), reduced along each row; the gradient arriving at the "
             f"two row results is dr = {short(dr)}.", "",
             "| reduction | result per row | dA | largest difference from central difference |",
             "|---|---|---|---:|",
             f"| sum | {short(rsum(a))} | {short(sum_dx)} | {chk['sum']:.1e} |",
             f"| mean | {short([round(v, 4) for v in rmean(a)])} | {short([[round(v, 4) for v in r] for r in mean_dx])} "
             f"| {chk['mean']:.1e} |",
             f"| max, first index wins ties | {short(rmax(a))} | {short(max_dx)} | row 0: "
             f"{maxdiff(max_dx[0], num[0]):.1e}; row 1 is tied |", "",
             f"At the tied row the central difference reports {short(num[1])}: it splits dr[1] = {dr[1]:g} "
             f"evenly between the two tied entries, while the rule above gives all of it to column 0."]
    return "\n".join(lines)


SM_Z = [[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]
SM_DY = [[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]


def softmax_rows(z):
    out = []
    for r in z:
        m = max(r)
        e = [math.exp(v - m) for v in r]
        s = sum(e)
        out.append([v / s for v in e])
    return out


def softmax_bwd(y, dy):
    out = []
    for ry, rd in zip(y, dy):
        dot = sum(a * b for a, b in zip(ry, rd))
        out.append([a * (b - dot) for a, b in zip(ry, rd)])
    return out


def softmax_backward():
    y = softmax_rows(SM_Z)
    dx = softmax_bwd(y, SM_DY)
    num = central_vjp(softmax_rows, [SM_Z], SM_DY)[0]
    return "\n\n".join([
        f"Rows of logits z = {short(SM_Z)} and upstream dy = {short(SM_DY)}.",
        mat_table("y = softmax(z)", y, 6, "b", "c"),
        mat_table("dot = sum of dy * y, then dx = y * (dy - dot)", dx, 6, "b", "c"),
        f"Row sums of dx: {sum(dx[0]):.1e} and {sum(dx[1]):.1e} (a shift of all logits changes nothing). "
        f"Largest difference from a central difference: {maxdiff(dx, num):.1e}."])


# ------------------------------------------------------------------ broadcasting


def unbroadcast(d, target):
    """Sum a gradient of a broadcast result [B, O] down to the shape `target` of the input before broadcasting."""
    if target == ():
        return sum(flat(d))
    if target == (len(d[0]),):
        return colsum(d)
    if target == (len(d), 1):
        return [[v] for v in rowsum(d)]
    if target == tuple(shape(d)):
        return d
    raise ValueError("unsupported broadcast")


def broadcast_backward():
    g = LIN_G
    base = [[0.0] * 4 for _ in range(3)]  # the tensor the small one is added to; its value does not matter
    cases = [
        ("bias b [4], added to every row", (4,), lambda y, s: [[v + s[j] for j, v in enumerate(r)] for r in y],
         [0.5, -0.5, 0.0, 1.0]),
        ("column c [3, 1], added to every column", (3, 1),
         lambda y, s: [[v + s[i][0] for v in r] for i, r in enumerate(y)], [[1.0], [2.0], [3.0]]),
        ("scalar s [], added to every entry", (), lambda y, s: [[v + s for v in r] for r in y], 0.25),
    ]
    rows = ["| the small input | broadcast over | gradient (sum over the broadcast axes) | "
            "largest difference from central difference |", "|---|---|---|---:|"]
    for title, shp, f, val in cases:
        ana = unbroadcast(g, shp)
        num = central_vjp(lambda s, f=f: f(base, s), [val], g)[0]
        over = {(4,): "the batch axis (axis 0)", (3, 1): "the output axis (axis 1)", (): "both axes"}[shp]
        rows.append(f"| {title} | {over} | {short(ana)} | {maxdiff(ana, num):.1e} |")
    return (f"The gradient arriving at the broadcast result is dY = G = {short(g)} ([3, 4]).\n\n" + "\n".join(rows))


# ------------------------------------------------------- reshape, slice, gather


def index_ops():
    x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    lines = [f"Input X = {short(x)} ([3, 4]). Upstream gradients are drawn as the numbers 1, 2, 3, ... "
             f"in row-major order of the result, so every entry of the answer can be traced.", ""]
    rows = ["| operation | result shape | upstream dY | gradient dX | largest difference from central difference |",
            "|---|---|---|---|---:|"]

    def run(title, f, bwd):
        y = f(x)
        dy = build([float(i + 1) for i in range(len(flat(y)))], shape(y))
        dx = bwd(dy)
        num = central_vjp(f, [x], dy)[0]
        rows.append(f"| {title} | {list(shape(y))} | {short(dy)} | {short(dx)} | {maxdiff(dx, num):.1e} |")

    def reshape(t):
        return build(flat(t), (2, 6))

    run("reshape to [2, 6]", reshape, lambda dy: build(flat(dy), (3, 4)))
    run("transpose", tr, tr)

    def sl(t):
        return [r[1:3] for r in t[1:3]]

    def sl_bwd(dy):
        out = [[0.0] * 4 for _ in range(3)]
        for i in range(2):
            for j in range(2):
                out[1 + i][1 + j] = dy[i][j]
        return out

    run("slice X[1:3, 1:3]", sl, sl_bwd)
    idx = [2, 0, 2]

    def gather(t):
        return [list(t[i]) for i in idx]

    def gather_bwd(dy):
        out = [[0.0] * 4 for _ in range(3)]
        for k, i in enumerate(idx):
            out[i] = [a + b for a, b in zip(out[i], dy[k])]  # add, never assign
        return out

    run(f"gather rows {idx}", gather, gather_bwd)
    return "\n".join(lines + rows)


# ------------------------------------------------------------ gradient accumulation

SH_W = [[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]
SH_XA = [1.0, 2.0, -1.0]
SH_XB = [0.5, -1.0, 3.0]
SH_GA = [1.0, -2.0]
SH_GB = [3.0, 1.0]


def shared_weight():
    w, xa, xb, ga, gb = SH_W, SH_XA, SH_XB, SH_GA, SH_GB

    def outer(g, x):
        return [[gi * xj for xj in x] for gi in g]

    use_a, use_b = outer(ga, xa), outer(gb, xb)
    both = madd(use_a, use_b)

    def f(w_):
        wa = [sum(w_[i][j] * xa[j] for j in range(3)) for i in range(2)]
        wb = [sum(w_[i][j] * xb[j] for j in range(3)) for i in range(2)]
        return [wa, wb]

    num = central_vjp(f, [w], [ga, gb])[0]
    rows = ["| way of combining the two uses | dW [2, 3] | largest difference from central difference |",
            "|---|---|---:|",
            f"| assign, the first use is kept (g_a x_a^T) | {short(use_a)} | {maxdiff(use_a, num):.2f} |",
            f"| assign, the second use is kept (g_b x_b^T) | {short(use_b)} | {maxdiff(use_b, num):.2f} |",
            f"| accumulate: add both | {short(both)} | {maxdiff(both, num):.1e} |"]
    return (f"One weight matrix W = {short(w)} is applied to two inputs, x_a = {short(xa)} and x_b = {short(xb)}, "
            f"giving z_a = W x_a and z_b = W x_b. The loss is L = g_a . z_a + g_b . z_b with g_a = {short(ga)}, "
            f"g_b = {short(gb)}.\n\n" + "\n".join(rows))


MB_B, MB_D, MB_O = 8, 3, 2


def _mse_grad(xs, ts, w, b, denom):
    """Gradient of sum((X W^T + b - T)^2) / denom with respect to W and b."""
    e = [[v - t for v, t in zip(rv, rt)] for rv, rt in zip(linear_fwd(xs, w, b), ts)]
    dy = mapm(lambda v: 2.0 * v / denom, e)
    return mm(tr(dy), xs), colsum(dy)


def microbatch_accumulation():
    name = "layer_gradients.microbatch_accumulation"
    rng = bookrng.generator(name)
    x = [[rng.normal() for _ in range(MB_D)] for _ in range(MB_B)]
    t = [[rng.normal() for _ in range(MB_O)] for _ in range(MB_B)]
    w = [[0.5 * rng.normal() for _ in range(MB_D)] for _ in range(MB_O)]
    b = [0.0] * MB_O
    big_w, big_b = _mse_grad(x, t, w, b, MB_B * MB_O)

    def norm(m):
        return math.sqrt(sum(v * v for v in flat(m)))

    def accumulate(sizes, scale_of, denom_of):
        accw = [[0.0] * MB_D for _ in range(MB_O)]
        accb = [0.0] * MB_O
        start = 0
        for size in sizes:
            xs, ts = x[start:start + size], t[start:start + size]
            start += size
            gw, gb = _mse_grad(xs, ts, w, b, denom_of(size))
            accw = madd(accw, mapm(lambda v: v * scale_of(size), gw))
            accb = [p + q * scale_of(size) for p, q in zip(accb, gb)]
        return accw, accb

    eq, uneq = [2, 2, 2, 2], [3, 3, 2]
    mean_micro = lambda s: s * MB_O  # each micro-batch loss is its own mean
    strategies = [
        ("4 micro-batches of 2, own means, summed", eq, lambda s: 1.0, mean_micro),
        ("4 micro-batches of 2, own means, summed, times 1/4", eq, lambda s: 0.25, mean_micro),
        ("micro-batches of 3, 3, 2, own means, times 1/3", uneq, lambda s: 1.0 / 3.0, mean_micro),
        ("micro-batches of 3, 3, 2, own means, times B_j / B", uneq, lambda s: s / MB_B, mean_micro),
        ("micro-batches of 3, 3, 2, divided by the total B O", uneq, lambda s: 1.0, lambda s: MB_B * MB_O),
    ]
    rows = ["| strategy | largest abs. difference of dW from the one-batch gradient | norm of dW / norm of one-batch dW |",
            "|---|---:|---:|"]
    for title, sizes, sc, dn in strategies:
        aw, _ = accumulate(sizes, sc, dn)
        rows.append(f"| {title} | {maxdiff(aw, big_w):.1e} | {norm(aw) / norm(big_w):.4f} |")
    return "\n".join([
        f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}). Draw order, all from `normal()`: `X` ({MB_B} x {MB_D}, row-major), "
        f"`T` ({MB_B} x {MB_O}), then `W` ({MB_O} x {MB_D}) each multiplied by 0.5; the bias `b` is zero. "
        f"Loss: the mean over all B O = {MB_B * MB_O} squared errors; micro-batches are consecutive rows. "
        f"The one-batch gradient has norm {norm(big_w):.6f} (dW only).", ""] + rows)


# --------------------------------------------------------------- rule checks


def _rules():
    """(name, forward, backward, shapes of the inputs, input preparation). A backward maps
    (inputs, output gradient) to one gradient per input."""
    def mm3(a, b):
        return [mm(a[i], b[i]) for i in range(len(a))]

    def mm3s(a, b):
        return [mm(a[i], b) for i in range(len(a))]

    def bm_bwd(a, b, dc):
        da = [mm(dc[i], tr(b[i])) for i in range(len(a))]
        db = [mm(tr(a[i]), dc[i]) for i in range(len(a))]
        return da, db

    def bms_bwd(a, b, dc):
        da = [mm(dc[i], tr(b)) for i in range(len(a))]
        db = [[0.0] * len(b[0]) for _ in b]
        for i in range(len(a)):
            db = madd(db, mm(tr(a[i]), dc[i]))
        return da, db

    def unary(name):
        f, fp, _, _ = ELEMENTWISE[name]
        return (name, lambda x: mapm(f, x), lambda ins, y, dy: [zip2(lambda d, v: d * fp(v), dy, ins[0])],
                [(3, 4)], None)

    def rmax_bwd(ins, y, dy):
        a = ins[0]
        out = []
        for r, d in zip(a, dy):
            j = max(range(len(r)), key=lambda c: (r[c], -c))
            out.append([d if c == j else 0.0 for c in range(len(r))])
        return [out]

    def concat(a, b):
        return [ra + rb for ra, rb in zip(a, b)]

    def concat_bwd(ins, y, dy):
        n = len(ins[0][0])
        return [[r[:n] for r in dy], [r[n:] for r in dy]]

    idx = [2, 0, 2]

    def gather_bwd(ins, y, dy):
        out = [[0.0] * len(ins[0][0]) for _ in ins[0]]
        for k, i in enumerate(idx):
            out[i] = [p + q for p, q in zip(out[i], dy[k])]
        return [out]

    def den(ins):
        return [ins[0], mapm(lambda v: 1.5 + abs(v), ins[1])]

    return [
        ("linear X W^T + b", linear_fwd, lambda ins, y, dy: list(linear_bwd(ins[0], ins[1], dy)),
         [(3, 2), (4, 2), (4,)], None),
        ("matmul A B", mm, lambda ins, y, dy: list(matmul_bwd(ins[0], ins[1], dy)), [(2, 3), (3, 4)], None),
        ("batched matmul A[i] B[i]", mm3, lambda ins, y, dy: list(bm_bwd(ins[0], ins[1], dy)),
         [(2, 2, 3), (2, 3, 4)], None),
        ("batched matmul, shared B", mm3s, lambda ins, y, dy: list(bms_bwd(ins[0], ins[1], dy)),
         [(2, 2, 3), (3, 4)], None),
        unary("tanh"), unary("sigmoid"), unary("ReLU"), unary("GELU"), unary("SiLU"),
        ("x + y", lambda a, b: zip2(lambda p, q: p + q, a, b), lambda ins, y, dy: [dy, dy], [(2, 3), (2, 3)], None),
        ("x - y", lambda a, b: zip2(lambda p, q: p - q, a, b),
         lambda ins, y, dy: [dy, mapm(lambda v: -v, dy)], [(2, 3), (2, 3)], None),
        ("x * y", lambda a, b: zip2(lambda p, q: p * q, a, b),
         lambda ins, y, dy: [zip2(lambda d, q: d * q, dy, ins[1]), zip2(lambda d, p: d * p, dy, ins[0])],
         [(2, 3), (2, 3)], None),
        ("x / y", lambda a, b: zip2(lambda p, q: p / q, a, b),
         lambda ins, y, dy: [zip2(lambda d, q: d / q, dy, ins[1]),
                             [[-d * p / (q * q) for d, p, q in zip(rd, rp, rq)]
                              for rd, rp, rq in zip(dy, ins[0], ins[1])]], [(2, 3), (2, 3)], den),
        ("sum of all entries", lambda a: sum(flat(a)), lambda ins, y, dy: [mapm(lambda v: dy, ins[0])],
         [(3, 4)], None),
        ("sum along axis 0", colsum, lambda ins, y, dy: [[list(dy) for _ in ins[0]]], [(3, 4)], None),
        ("mean along axis 1", lambda a: [sum(r) / len(r) for r in a],
         lambda ins, y, dy: [[[d / len(r)] * len(r) for d, r in zip(dy, ins[0])]], [(3, 4)], None),
        ("max along axis 1", lambda a: [max(r) for r in a], rmax_bwd, [(3, 4)], None),
        ("softmax along axis 1", softmax_rows, lambda ins, y, dy: [softmax_bwd(y, dy)], [(3, 4)], None),
        ("add bias [4] to [3, 4]", lambda y, s: [[v + s[j] for j, v in enumerate(r)] for r in y],
         lambda ins, y, dy: [dy, colsum(dy)], [(3, 4), (4,)], None),
        ("add column [3, 1] to [3, 4]", lambda y, s: [[v + s[i][0] for v in r] for i, r in enumerate(y)],
         lambda ins, y, dy: [dy, [[v] for v in rowsum(dy)]], [(3, 4), (3, 1)], None),
        ("add scalar to [3, 4]", lambda y, s: [[v + s for v in r] for r in y],
         lambda ins, y, dy: [dy, sum(flat(dy))], [(3, 4), ()], None),
        ("reshape [3, 4] to [2, 6]", lambda a: build(flat(a), (2, 6)),
         lambda ins, y, dy: [build(flat(dy), (3, 4))], [(3, 4)], None),
        ("transpose", tr, lambda ins, y, dy: [tr(dy)], [(3, 4)], None),
        ("slice [1:3, 1:3]", lambda a: [r[1:3] for r in a[1:3]],
         lambda ins, y, dy: [[[0.0] * 4] + [[0.0] + list(r) + [0.0] for r in dy]], [(3, 4)], None),
        ("concat along axis 1", concat, concat_bwd, [(2, 3), (2, 2)], None),
        ("gather rows [2, 0, 2]", lambda a: [list(a[i]) for i in idx], gather_bwd, [(3, 4)], None),
    ]


RULE_FORMS = {
    "linear X W^T + b": "dX = dY W; dW = dY^T X; db = sum over batch of dY",
    "matmul A B": "dA = dC B^T; dB = A^T dC",
    "batched matmul A[i] B[i]": "the matmul rule for every i",
    "batched matmul, shared B": "dA[i] = dC[i] B^T; dB = sum over i of A[i]^T dC[i]",
    "tanh": "dx = dy (1 - y^2)", "sigmoid": "dx = dy y (1 - y)", "ReLU": "dx = dy [x > 0]",
    "GELU": "dx = dy (Phi(x) + x phi(x))", "SiLU": "dx = dy s (1 + x (1 - s)), s = sigmoid(x)",
    "x + y": "dx = dz; dy = dz", "x - y": "dx = dz; dy = -dz", "x * y": "dx = dz y; dy = dz x",
    "x / y": "dx = dz / y; dy = -dz x / y^2",
    "sum of all entries": "dx = dr in every entry", "sum along axis 0": "dx[i, j] = dr[j]",
    "mean along axis 1": "dx[i, j] = dr[i] / n", "max along axis 1": "dr[i] to the first argmax, 0 elsewhere",
    "softmax along axis 1": "dx = y (dy - sum of dy y)",
    "add bias [4] to [3, 4]": "dY; db = sum over axis 0 of dY",
    "add column [3, 1] to [3, 4]": "dY; dc = sum over axis 1 of dY, kept as [3, 1]",
    "add scalar to [3, 4]": "dY; ds = sum of all of dY",
    "reshape [3, 4] to [2, 6]": "reshape dY back", "transpose": "transpose dY",
    "slice [1:3, 1:3]": "dY placed in a zero tensor", "concat along axis 1": "slice dY apart",
    "gather rows [2, 0, 2]": "add each dY row into its source row",
}


def rule_checks():
    name = "layer_gradients.rule_checks"
    rng = bookrng.generator(name)
    rows = ["| operation | backward rule | inputs | largest absolute difference |", "|---|---|---|---:|"]
    worst = 0.0
    kink = 1.0
    for title, f, bwd, shapes, prep in _rules():
        ins = []
        for shp in shapes:
            n = 1
            for s in shp:
                n *= s
            ins.append(build([rng.normal() for _ in range(n)], shp))
        if prep:
            ins = prep(ins)
        out = f(*ins)
        g = build([rng.normal() for _ in range(len(flat(out)))], shape(out))
        if title in ("ReLU", "max along axis 1"):
            kink = min(kink, min(abs(v) for v in flat(ins[0])))
        ana = bwd(ins, out, g)
        num = central_vjp(f, ins, g)
        d = max(maxdiff(a, n) for a, n in zip(ana, num))
        worst = max(worst, d)
        rows.append(f"| {title} | {RULE_FORMS[title]} | {', '.join(str(list(s)) for s in shapes)} | {d:.1e} |")
    return "\n".join([
        f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}). One generator for the whole table, `normal()` draws in table order: for "
        f"each row its inputs in the order listed (row-major), then the output gradient (row-major), which has the shape of the row's output "
        f"(a single number for the sum of all entries, [4] for the sum along axis 0, [3] for the mean and the max along axis 1). For the "
        f"division the denominator is replaced by 1.5 plus the absolute value of its draw. Central-difference "
        f"step 1e-6; the closest any ReLU or max input comes to its kink is {kink:.3f} "
        f"(a check on your draw order). The last column is rounding noise of the central difference: expect values below 1e-8, not these digits. "
        f"[x > 0] in the ReLU rule is the Iverson bracket (1 when the condition holds, 0 otherwise). Phi and phi in the GELU rule are the standard Gaussian cumulative distribution, Phi(x) = (1 + erf(x / sqrt 2)) / 2, and density, "
        f"phi(x) = exp(-x^2 / 2) / sqrt(2 pi).", ""] + rows +
        ["", f"Largest difference over all rows: {worst:.1e}."])


# ---------------------------------------------------------------- work and memory

WK_B, WK_D, WK_O = 64, 784, 256


def work_counts():
    b, d, o = WK_B, WK_D, WK_O
    prods = [("forward  Y = X W^T", "X [B, D], W [O, D]", "Y [B, O]", b * d, o * d, b * o),
             ("backward dX = dY W", "dY [B, O], W [O, D]", "dX [B, D]", b * o, o * d, b * d),
             ("backward dW = dY^T X", "dY [B, O], X [B, D]", "dW [O, D]", b * o, b * d, o * d)]
    rows = [f"Sizes B = {b}, D = {d}, O = {o}, float32 (4 bytes). A multiply-add counts as two FLOPs. "
            f"Bytes are the minimum traffic: each operand read once, the result written once.", "",
            "| product | reads | writes | FLOPs | bytes moved | FLOPs per byte |", "|---|---|---|---:|---:|---:|"]
    for title, rd, wr, n1, n2, n3 in prods:
        flops = 2 * b * d * o
        by = 4 * (n1 + n2 + n3)
        rows.append(f"| {title} | {rd} | {wr} | {flops:,} | {by:,} | {flops / by:.2f} |")
    rows += ["", "| smaller operation | FLOPs | note |", "|---|---:|---|",
             f"| bias add (forward) | {b * o:,} | one add per output entry |",
             f"| bias gradient | {b * o:,} | one add per entry of dY |"]
    ratios = ["| layers in the stack | forward matmul FLOPs | backward matmul FLOPs | backward / forward |",
              "|---:|---:|---:|---:|"]
    for n in (1, 2, 4, 8, 32):
        fw = n * 2 * b * d * o
        bw = (2 * n - 1) * 2 * b * d * o  # every layer's dW, every layer's dX except the first
        ratios.append(f"| {n} | {fw:,} | {bw:,} | {bw / fw:.4f} |")
    return "\n".join(rows) + (
        "\n\nA stack of identical layers, where the first layer needs no dX because the data need no gradient:\n\n"
        + "\n".join(ratios))


def saved_per_op():
    b, w = 64, 4096
    n = b * w
    f32 = 4
    rows = [f"Tensors of shape [B, W] with B = {b}, W = {w} ({n:,} entries), float32.", "",
            "| operation | what the backward reads | entries or bits | bytes |", "|---|---|---:|---:|",
            f"| linear Y = X W^T + b | the input X [B, W] | {n:,} | {n * f32:,} |",
            f"| tanh, sigmoid | the output y | {n:,} | {n * f32:,} |",
            f"| GELU, SiLU | the input x | {n:,} | {n * f32:,} |",
            f"| ReLU | one bit per entry | {n:,} bits | {n // 8:,} |",
            f"| softmax along W | the output y | {n:,} | {n * f32:,} |",
            f"| max along W | the argmax index per row (4 bytes) | {b:,} | {b * 4:,} |",
            f"| gather of {b} rows | the {b} row indices (4 bytes each) | {b:,} | {b * 4:,} |",
            f"| sum, mean | only the sizes | 0 | 0 |",
            f"| add of a bias, reshape, transpose, slice | only the sizes | 0 | 0 |"]
    return "\n".join(rows)
