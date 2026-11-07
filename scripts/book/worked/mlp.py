# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Multilayer Networks".

Plain Python 3, standard library only, written from the chapter's own
description: chapter 1's XOR network rebuilt with smooth activations and
computed layer by layer, the collapse of stacked linear layers into one
matrix, the forward pass of a batch of three examples with shapes
[B, D] -> [B, H] -> [B, O], parameter and work counts, the activation
functions tanh, ReLU, GELU and SiLU with their derivatives, the exact and
approximate GELU forms, ReLU sparsity and a dead unit, a curve built from k ReLU
units with its error against k, the count of cells a grid construction needs in
D dimensions, and the sawtooth made by composing tent maps, whose linear pieces
double with every layer. Each public function without a leading underscore that
takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: mlp.<name> -->` markers. Underscore functions and the constants
are shared with the chapter's figures, so a picture and the table beside it
cannot disagree.

Swedish Embedded AB implements solutions for function approximation with small
neural networks whose every weight can be inspected. If your team needs
expertise in compact neural models for industrial signals, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng

# ------------------------------------------------------------- constants

# Chapter 1's XOR network with smooth activations: tanh hidden units, a sigmoid output.
XOR = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
XOR_W1 = ((4.0, 4.0), (4.0, 4.0))
XOR_B1 = (-2.0, -6.0)
XOR_W2 = ((5.0, -5.0),)
XOR_B2 = (-5.0,)

# A non-degenerate two-layer network for the collapse identity (O = 1, H = 3, D = 2).
COLLAPSE_W1 = ((1.0, 2.0), (0.0, -1.0), (3.0, 1.0))
COLLAPSE_B1 = (1.0, 0.0, -1.0)
COLLAPSE_W2 = ((2.0, -1.0, 1.0),)
COLLAPSE_B2 = (0.5,)
COLLAPSE_X = (1.0, 2.0)

# The batch forward pass: B = 3, D = 2, H = 3, O = 2, tanh hidden units.
BATCH_X = ((0.5, -1.0), (1.0, 2.0), (-1.5, 0.5))
BATCH_W1 = ((1.0, -1.0), (0.5, 2.0), (-2.0, 1.0))
BATCH_B1 = (0.0, 0.5, -0.5)
BATCH_W2 = ((1.0, -1.0, 0.5), (0.5, 1.0, -1.0))
BATCH_B2 = (0.1, -0.1)

# Network widths whose parameters and work are counted.
WIDTHS = (
    ("XOR network of this chapter", (2, 2, 1)),
    ("small image classifier", (784, 128, 10)),
    ("two hidden layers", (784, 512, 512, 10)),
    ("one transformer feed-forward block", (768, 3072, 768)),
)

ACT_XS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
GRID_LO, GRID_HI, GRID_STEP = -6.0, 6.0, 0.001
GELU_K = math.sqrt(2.0 / math.pi)
GELU_CUBIC = 0.044715
QUICK_GELU_SCALE = 1.702

SPARSITY_NAME = "mlp.relu_sparsity"
SPARSITY_UNITS, SPARSITY_D, SPARSITY_N = 8, 4, 1000
SPARSITY_DEAD_BIAS = -12.0

# The curve approximated by k ReLU units: f(x) = sin(2 pi x) on [0, 1].
CURVE_KS = (2, 4, 8, 16, 32, 64)
CURVE_GRID = 4000
CURVE_TARGET = 0.01
GRID_CELL_DIMS = (1, 2, 3, 4, 6, 10)

TENT_DEPTHS = tuple(range(1, 11))


# ---------------------------------------------------------------- helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _f(v, digits=4):
    """Fixed-point text that never prints a negative zero."""
    s = f"{v:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _relu(x):
    return x if x > 0.0 else 0.0


def _drelu(x):
    return 1.0 if x > 0.0 else 0.0


def _dtanh(x):
    t = math.tanh(x)
    return 1.0 - t * t


def _gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _dgelu(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(GELU_K * (x + GELU_CUBIC * x ** 3)))


def _quick_gelu(x):
    return x * _sigmoid(QUICK_GELU_SCALE * x)


def _silu(x):
    return x * _sigmoid(x)


def _dsilu(x):
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


ACTIVATIONS = (
    ("tanh", math.tanh, _dtanh),
    ("relu", _relu, _drelu),
    ("gelu", _gelu, _dgelu),
    ("silu", _silu, _dsilu),
)


def _affine(W, b, x):
    return [sum(w * v for w, v in zip(row, x)) + bi for row, bi in zip(W, b)]


# ---------------------------------------------------------- XOR, smooth

def xor_smooth():
    rows = []
    for a, c, t in XOR:
        z = _affine(XOR_W1, XOR_B1, (a, c))
        h = [math.tanh(v) for v in z]
        zo = _affine(XOR_W2, XOR_B2, h)[0]
        y = _sigmoid(zo)
        assert (y >= 0.5) == bool(t)
        rows.append((a, c, f"{z[0]:+.1f}", _f(h[0]), f"{z[1]:+.1f}", _f(h[1]), f"{zo:+.4f}", _f(y), t))
    return ("Hidden unit i: h_i = tanh(z_i) with z = W1 x + b1, W1 = [[4, 4], [4, 4]], b1 = (-2, -6). "
            "Output: y = sigmoid(z_out) with z_out = 5 h1 - 5 h2 - 5.\n\n"
            + _table(["x1", "x2", "z1", "h1", "z2", "h2", "z_out", "y", "XOR"], rows))


def xor_relu():
    rows = []
    for a, c, t in XOR:
        z1, z2 = a + c, a + c - 1
        h1, h2 = _relu(z1), _relu(z2)
        y = h1 - 2 * h2
        assert y == t
        rows.append((a, c, z1, f"{h1:g}", z2, f"{h2:g}", f"{y:g}", t))
    return ("Two ReLU hidden units, z1 = x1 + x2 and z2 = x1 + x2 - 1, h = relu(z), and the output y = h1 - 2 h2 "
            "with no activation on the output.\n\n"
            + _table(["x1", "x2", "z1", "h1", "z2", "h2", "y = h1 - 2 h2", "XOR"], rows))


def linear_collapse():
    # (1) the XOR network with identity hidden units: every input gives the same output.
    W = [[sum(XOR_W2[0][k] * XOR_W1[k][j] for k in range(2)) for j in range(2)]]
    c = sum(XOR_W2[0][k] * XOR_B1[k] for k in range(2)) + XOR_B2[0]
    rows = []
    for a, b2, t in XOR:
        z = _affine(XOR_W1, XOR_B1, (a, b2))
        out = _affine(XOR_W2, XOR_B2, z)[0]
        assert abs(out - (W[0][0] * a + W[0][1] * b2 + c)) < 1e-12
        rows.append((a, b2, f"{z[0]:+.1f}", f"{z[1]:+.1f}", f"{out:+.1f}", f"{W[0][0] * a + W[0][1] * b2 + c:+.1f}"))
    first = ("The XOR network of the previous example with the identity as hidden activation. "
             f"W2 W1 = [[{W[0][0]:.1f}, {W[0][1]:.1f}]], W2 b1 + b2 = {c:.1f}.\n\n"
             + _table(["x1", "x2", "z1", "z2", "two layers", "one layer (W2 W1) x + (W2 b1 + b2)"], rows))
    # (2) a network with distinct rows: the collapsed matrix is genuinely different from either layer.
    M = [[sum(COLLAPSE_W2[0][k] * COLLAPSE_W1[k][j] for k in range(3)) for j in range(2)]]
    d = sum(COLLAPSE_W2[0][k] * COLLAPSE_B1[k] for k in range(3)) + COLLAPSE_B2[0]
    z1 = _affine(COLLAPSE_W1, COLLAPSE_B1, COLLAPSE_X)
    two = _affine(COLLAPSE_W2, COLLAPSE_B2, z1)[0]
    one = M[0][0] * COLLAPSE_X[0] + M[0][1] * COLLAPSE_X[1] + d
    assert abs(two - one) < 1e-12
    second = (f"A second network, W1 = [[1, 2], [0, -1], [3, 1]], b1 = (1, 0, -1), W2 = [[2, -1, 1]], b2 = 0.5, at x = (1, 2).\n\n"
              + _table(["quantity", "value"], [
                  ("z1 = W1 x + b1", "(" + ", ".join(f"{v:g}" for v in z1) + ")"),
                  ("two layers: W2 z1 + b2", f"{two:g}"),
                  ("W2 W1 (1 x 2)", "[[" + ", ".join(f"{v:g}" for v in M[0]) + "]]"),
                  ("W2 b1 + b2", f"{d:g}"),
                  ("one layer: (W2 W1) x + (W2 b1 + b2)", f"{one:g}"),
              ], align="lr"))
    return first + "\n\n" + second


def _matmul_t(A, W):
    """A [B, K] times W^T for W [N, K]: the [B, N] result."""
    return [[sum(a * w for a, w in zip(row, wrow)) for wrow in W] for row in A]


def _mat_text(M, digits=4):
    return "[" + ", ".join("[" + ", ".join(_f(v, digits) for v in row) + "]" for row in M) + "]"


def batch_forward():
    Z1 = [[v + b for v, b in zip(row, BATCH_B1)] for row in _matmul_t(BATCH_X, BATCH_W1)]
    A = [[math.tanh(v) for v in row] for row in Z1]
    Y = [[v + b for v, b in zip(row, BATCH_B2)] for row in _matmul_t(A, BATCH_W2)]
    rows = []
    for n in range(len(BATCH_X)):
        rows.append((n, "(" + ", ".join(f"{v:g}" for v in BATCH_X[n]) + ")",
                     "(" + ", ".join(_f(v) for v in Z1[n]) + ")",
                     "(" + ", ".join(_f(v) for v in A[n]) + ")",
                     "(" + ", ".join(_f(v) for v in Y[n]) + ")"))
    head = ("B = 3, D = 2, H = 3, O = 2. X = [[0.5, -1], [1, 2], [-1.5, 0.5]], "
            "W1 = [[1, -1], [0.5, 2], [-2, 1]], b1 = (0, 0.5, -0.5), "
            "W2 = [[1, -1, 0.5], [0.5, 1, -1]], b2 = (0.1, -0.1), hidden activation tanh.\n\n")
    return head + _table(["row n", "x (D = 2)", "Z1 = X W1^T + b1 (H = 3)", "A = tanh(Z1) (H = 3)", "Y = A W2^T + b2 (O = 2)"], rows)


# ------------------------------------------------------------ counting

def _params(widths):
    return sum(widths[i + 1] * widths[i] + widths[i + 1] for i in range(len(widths) - 1))


def parameter_counts():
    rows = []
    for name, w in WIDTHS:
        p = _params(w)
        flops = 2 * sum(w[i + 1] * w[i] for i in range(len(w) - 1))
        rows.append((name, "-".join(str(v) for v in w), f"{p:,}", f"{4 * p:,}", f"{flops:,}"))
    return ("Parameters = sum over layers of (d_out d_in + d_out); weight bytes at 4 bytes per parameter; "
            "FLOPs = 2 d_out d_in per layer for the matrix products of one example "
            "(a multiply and an add each), not counting the bias adds and the activations.\n\n"
            + _table(["network", "widths D-...-O", "parameters", "weight bytes (f32)", "matmul FLOPs per example"], rows,
                     align="llrrr"))


# ---------------------------------------------------------- activations

def activation_table():
    rows = []
    for x in ACT_XS:
        cells = [f"{x:+.1f}"]
        for _, f, df in ACTIVATIONS:
            cells += [_f(f(x)), _f(df(x))]
        rows.append(tuple(cells))
    head = ("tanh(x), relu(x) = max(0, x), gelu(x) = x Phi(x) with the exact erf form, silu(x) = x sigmoid(x); each followed by its "
            "derivative (the derivative of relu at 0 is taken as 0).\n\n")
    hdr = ["x"]
    for name, _, _ in ACTIVATIONS:
        hdr += [name, name + "'"]
    return head + _table(hdr, rows)


def activation_facts():
    n = int(round((GRID_HI - GRID_LO) / GRID_STEP))
    xs = [GRID_LO + i * GRID_STEP for i in range(n + 1)]
    rows = []
    for name, f, df in ACTIVATIONS:
        vals = [f(x) for x in xs]
        slopes = [df(x) for x in xs]
        lo = min(range(len(xs)), key=lambda i: (round(vals[i], 12), i))
        hi = max(range(len(xs)), key=lambda i: (round(slopes[i], 12), -i))
        flat = sum(1 for s in slopes if abs(s) < 0.01) / len(xs)
        steep = sum(1 for s in slopes if s >= 0.5) / len(xs)
        rows.append((name, _f(vals[lo]), f"{xs[lo]:+.3f}", _f(slopes[hi]), f"{xs[hi]:+.3f}", _f(df(4.0)),
                     f"{flat:.3f}", f"{steep:.3f}"))
    return ("Grid x = -6 + 0.001 i for i = 0, ..., 12000 (12001 points). A tie for the lowest value "
            "or the largest slope goes to the first (lowest) grid point attaining it.\n\n"
            + _table(["function", "lowest value", "at x", "largest slope", "at x", "slope at x = 4",
                      "share with |slope| < 0.01", "share with slope >= 0.5"], rows))


def gelu_forms():
    pts = (-3.0, -1.0, -0.5, 0.5, 1.0, 3.0)
    rows = []
    for x in pts:
        e, t, q = _gelu(x), _gelu_tanh(x), _quick_gelu(x)
        rows.append((f"{x:+.1f}", _f(e, 6), _f(t, 6), f"{abs(t - e):.2e}", _f(q, 6), f"{abs(q - e):.2e}", _f(_silu(x), 6)))
    n = int(round(12.0 / 0.01))
    xs = [-6.0 + 0.01 * i for i in range(n + 1)]

    def biggest(f):
        i = max(range(len(xs)), key=lambda j: (round(abs(f(xs[j]) - _gelu(xs[j])), 12), -j))
        return abs(f(xs[i]) - _gelu(xs[i])), xs[i]

    stats = [("tanh form", *biggest(_gelu_tanh)), ("quick form x sigmoid(1.702 x)", *biggest(_quick_gelu)),
             ("silu x sigmoid(x)", *biggest(_silu))]
    s_rows = [(name, f"{d:.3e}", f"{x:+.2f}") for name, d, x in stats]
    return ("exact gelu(x) = 0.5 x (1 + erf(x / sqrt 2)); tanh form 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))); "
            "quick form x sigmoid(1.702 x); silu x sigmoid(x).\n\n"
            + _table(["x", "exact", "tanh form", "abs diff", "quick form", "abs diff", "silu"], rows)
            + "\n\nLargest absolute difference from the exact form over x = -6 + 0.01 i, i = 0, ..., 1200 "
              "(the first grid point attaining the maximum).\n\n"
            + _table(["approximation", "largest abs difference", "at x"], s_rows))


def _dgelu_tanh(x):
    t = math.tanh(GELU_K * (x + GELU_CUBIC * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_K * (1.0 + 3.0 * GELU_CUBIC * x * x)


def gelu_derivative_gap():
    n = int(round(8.0 / 0.01))
    xs = [-4.0 + 0.01 * i for i in range(n + 1)]
    gaps = [abs(_dgelu_tanh(x) - _dgelu(x)) for x in xs]
    i = max(range(len(xs)), key=lambda j: (round(gaps[j], 12), -j))
    rel = max(abs(_dgelu_tanh(x) - _dgelu(x)) / max(abs(_dgelu_tanh(x)), abs(_dgelu(x))) for x in xs if abs(_dgelu(x)) > 1e-3)
    rows = [(f"{x:+.1f}", _f(_dgelu(x), 6), _f(_dgelu_tanh(x), 6), f"{abs(_dgelu_tanh(x) - _dgelu(x)):.2e}")
            for x in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)]
    return ("Derivative of the exact gelu, 0.5 (1 + erf(x / sqrt 2)) + x exp(-x^2 / 2) / sqrt(2 pi), against the derivative of "
            "the tanh form, 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2) with "
            "t = tanh(sqrt(2/pi) (x + 0.044715 x^3)).\n\n"
            + _table(["x", "exact gelu'", "tanh-form gelu'", "abs diff"], rows)
            + f"\n\nOver x = -4 + 0.01 i, i = 0, ..., 800 the largest absolute difference is {gaps[i]:.3e} at x = {xs[i]:+.2f}; "
              f"the largest relative difference (where |exact gelu'| > 0.001) is {rel:.3f}.")


# Forward operation lists. Each + - * / and max counts as one arithmetic operation; a call of
# exp, tanh or erf counts as one special-function call (its own cost is a few to a few tens of
# arithmetic operations, depending on the library).
ACT_OPS = (
    ("relu", ("max",), "max(0, x)"),
    ("tanh", ("tanh",), "tanh(x)"),
    ("silu", ("neg", "exp", "add", "div"), "x / (1 + exp(-x))"),
    ("gelu (erf form)", ("mul", "erf", "add", "mul", "mul"), "0.5 x (1 + erf(0.7071 x))"),
    ("gelu (tanh form)", ("mul", "mul", "mul", "add", "mul", "tanh", "add", "mul", "mul"),
     "0.5 x (1 + tanh(k (x + 0.044715 x x x)))"),
)
SPECIAL = ("exp", "tanh", "erf")


def activation_cost():
    d = 768
    per_output = 2 * d
    rows = []
    for name, ops, text in ACT_OPS:
        special = sum(1 for o in ops if o in SPECIAL)
        arith = len(ops) - special
        rows.append((name, text, arith, special, 8, f"{(arith + special) / 8:.3f}", f"{per_output / (arith + special):.0f}"))
    return (f"Per element, float32: the activation reads 4 bytes and writes 4 bytes (8 bytes moved). The matrix product that "
            f"feeds it, with D = {d} inputs, costs 2 D = {per_output} FLOPs per output element. "
            "The last column divides that by the activation's arithmetic plus special-function calls.\n\n"
            + _table(["activation", "formula", "arithmetic ops", "special calls", "bytes moved", "operations per byte",
                      "matmul FLOPs per output / activation ops"], rows, align="llrrrrr"))


def _dsigmoid(x):
    s = _sigmoid(x)
    return s * (1.0 - s)


SLOPE_CASES = (
    ("sigmoid", _dsigmoid, 0.0),
    ("tanh", _dtanh, 0.0),
    ("tanh", _dtanh, 1.0),
    ("tanh", _dtanh, 2.0),
    ("relu", _drelu, 1.0),
    ("gelu", _dgelu, 1.0),
    ("silu", _dsilu, 1.0),
)
SLOPE_DEPTHS = (1, 5, 10, 20)


def slope_chain():
    rows = []
    for name, df, z in SLOPE_CASES:
        s = df(z)
        rows.append((name, f"{z:+.1f}", _f(s), *[f"{s ** L:.2e}" for L in SLOPE_DEPTHS]))
    return ("The activation's own slope g'(z) at a fixed pre-activation z, raised to the number of layers: the product of equal slope "
            "factors, as a chain of layers that all sit at the same z would give. The weights contribute factors of their own.\n\n"
            + _table(["activation", "z", "slope g'(z)"] + [f"depth {L}" for L in SLOPE_DEPTHS], rows))


def relu_sparsity():
    name = SPARSITY_NAME
    g = bookrng.generator(name)
    W = [[g.normal() for _ in range(SPARSITY_D)] for _ in range(SPARSITY_UNITS)]
    X = [[g.normal() for _ in range(SPARSITY_D)] for _ in range(SPARSITY_N)]
    b = [0.0] * SPARSITY_UNITS
    b[-1] = SPARSITY_DEAD_BIAS
    rows = []
    total = 0
    for u in range(SPARSITY_UNITS):
        z = [sum(W[u][j] * x[j] for j in range(SPARSITY_D)) + b[u] for x in X]
        active = sum(1 for v in z if v > 0.0)
        total += active
        norm = math.sqrt(sum(w * w for w in W[u]))
        rows.append((u, f"{norm:.3f}", f"{b[u]:+.1f}", active, f"{active / SPARSITY_N:.3f}"))
    return (f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}). "
            f"Draw order: {SPARSITY_UNITS} x {SPARSITY_D} weights W[u, j] with normal(), row by row, then "
            f"{SPARSITY_N} x {SPARSITY_D} inputs X[n, j] with normal(), row by row. Unit u computes relu(W[u] . x + b[u]) "
            f"with b = 0 except b[{SPARSITY_UNITS - 1}] = {SPARSITY_DEAD_BIAS:g}.\n\n"
            + _table(["unit u", "length of W[u]", "b[u]", "examples with output > 0 (of 1000)", "share"], rows)
            + f"\n\nOverall {total} of {SPARSITY_UNITS * SPARSITY_N} unit outputs are positive "
              f"({total / (SPARSITY_UNITS * SPARSITY_N):.3f}).")


# ---------------------------------------------------- the ReLU construction

def _curve(x):
    return math.sin(2.0 * math.pi * x)


def _relu_net(k):
    """(hidden biases, output weights, output bias) of the k-unit interpolant of the curve on [0, 1]."""
    knots = [j / k for j in range(k + 1)]
    slopes = [k * (_curve(knots[j + 1]) - _curve(knots[j])) for j in range(k)]
    c = [slopes[0]] + [slopes[j] - slopes[j - 1] for j in range(1, k)]
    return [0.0 - knots[j] for j in range(k)], c, _curve(0.0)


def _relu_eval(net, x):
    biases, c, c0 = net
    return c0 + sum(cj * _relu(x + bj) for cj, bj in zip(c, biases))


def _curve_error(k):
    net = _relu_net(k)
    return max(abs(_relu_eval(net, i / CURVE_GRID) - _curve(i / CURVE_GRID)) for i in range(CURVE_GRID + 1))


def relu_approx_k4():
    k = 4
    biases, c, c0 = _relu_net(k)
    rows = []
    for j in range(k):
        s = k * (_curve((j + 1) / k) - _curve(j / k))
        rows.append((j, f"{j}/{k}", f"{biases[j]:+.2f}", _f(s), _f(c[j])))
    # the network agrees with straight-line interpolation between the knots
    for i in range(CURVE_GRID + 1):
        x = i / CURVE_GRID
        j = min(int(x * k), k - 1)
        x0, x1 = j / k, (j + 1) / k
        lin = _curve(x0) + (x - x0) * (_curve(x1) - _curve(x0)) / (x1 - x0)
        assert abs(lin - _relu_eval((biases, c, c0), x)) < 1e-9
    return (f"f(x) = sin(2 pi x) on [0, 1], k = 4 hidden units relu(x + b_j) with knots x_j = j/4 and b_j = -x_j. "
            f"Slope s_j = 4 (f(x_(j+1)) - f(x_j)); output weights c_0 = s_0 and c_j = s_j - s_(j-1); output bias f(0) = {c0:g}.\n\n"
            + _table(["unit j", "knot x_j", "hidden bias b_j", "slope s_j of piece j", "output weight c_j"], rows))


def relu_approx_error():
    rows = []
    prev = None
    for k in CURVE_KS:
        err = _curve_error(k)
        bound = (2.0 * math.pi) ** 2 / (8.0 * k * k)
        assert err <= bound + 1e-9
        rows.append((k, f"{err:.4f}", f"{bound:.4f}", f"{err / bound:.3f}", "" if prev is None else f"{prev / err:.2f}"))
        prev = err
    return (f"f(x) = sin(2 pi x), knots x_j = j/k, error = largest |f(x) - network(x)| over x = i/{CURVE_GRID}, i = 0, ..., {CURVE_GRID}. "
            "Bound = max|f''| / (8 k^2) with max|f''| = (2 pi)^2.\n\n"
            + _table(["k hidden units", "largest error", "bound max|f''| / (8 k^2)", "error / bound", "previous error / error"], rows))


def grid_cells():
    kstar = next(k for k in range(1, 1000) if _curve_error(k) <= CURVE_TARGET)
    rows = [(d, f"{kstar}^{d}", f"{kstar ** d:,}") for d in GRID_CELL_DIMS]
    return (f"Smallest k for which the k-unit network above has error at most {CURVE_TARGET:g}: k = {kstar}. "
            f"A grid with {kstar} cells along each of D axes has {kstar}^D cells.\n\n"
            + _table(["D", "cells", "count"], rows))


# ------------------------------------------------- depth against width

def _tent_net(depth):
    """Layers (W, b) of the depth-`depth` tent network: width 2 hidden layers, scalar input and output."""
    half = Fraction(1, 2)
    layers = [([[Fraction(1)], [Fraction(1)]], [Fraction(0), -half])]
    for _ in range(depth - 1):
        layers.append(([[Fraction(2), Fraction(-4)], [Fraction(2), Fraction(-4)]], [Fraction(0), -half]))
    layers.append(([[Fraction(2), Fraction(-4)]], [Fraction(0)]))
    return layers


def _tent_eval(layers, x):
    v = [Fraction(x)]
    for li, (W, b) in enumerate(layers):
        z = [sum(w * a for w, a in zip(row, v)) + bi for row, bi in zip(W, b)]
        v = [t if t > 0 else Fraction(0) for t in z] if li < len(layers) - 1 else z
    return v[0]


def _tent_params(depth):
    return sum(len(W) * len(W[0]) + len(b) for W, b in _tent_net(depth))


def _count_pieces(depth):
    layers = _tent_net(depth)
    m = 2 ** (depth + 1)
    ys = [_tent_eval(layers, Fraction(i, m)) for i in range(m + 1)]
    slopes = [(ys[i + 1] - ys[i]) * m for i in range(m)]
    return 1 + sum(1 for i in range(1, m) if slopes[i] != slopes[i - 1])


def tent_values():
    rows = []
    for i in range(9):
        x = Fraction(i, 8)
        rows.append((f"{i}/8", *[f"{float(_tent_eval(_tent_net(d), x)):.3f}" for d in (1, 2, 3)]))
    return ("Tent network of depth d: hidden layers of width 2 with weights as described in the text; "
            "T(x) = 2 relu(x) - 4 relu(x - 1/2) on [0, 1]; the depth-d network computes the d-fold composition.\n\n"
            + _table(["x", "d = 1: T(x)", "d = 2: T(T(x))", "d = 3: T(T(T(x)))"], rows))


def tent_pieces():
    rows = []
    for d in TENT_DEPTHS:
        pieces = _count_pieces(d)
        assert pieces == 2 ** d
        deep = _tent_params(d)
        shallow_units = pieces - 1
        shallow = 3 * shallow_units + 1
        rows.append((d, 2 * d, deep, pieces, f"{shallow_units:,}", f"{shallow:,}", f"{shallow / deep:.1f}"))
    return ("Pieces counted exactly (rational arithmetic) on the grid x = i / 2^(d+1): one plus the number of grid points at "
            "which the slope between neighbouring grid points changes. Deep network: d hidden layers of width 2, "
            "6 d + 1 parameters. A one-hidden-layer ReLU network with u units has at most u + 1 pieces, so it needs "
            "at least pieces - 1 units, and 3 u + 1 parameters.\n\n"
            + _table(["depth d", "hidden units (deep)", "parameters (deep)", "linear pieces", "units needed (one hidden layer)",
                      "parameters (one hidden layer)", "ratio of parameters"], rows))


def layer_work():
    d, h = 768, 3072
    rows = []
    for b in (1, 16, 256, 4096):
        flops = 2 * b * d * h
        moved = 4 * (b * d + h * d + b * h)
        rows.append((b, f"{flops:,}", f"{moved:,}", f"{flops / moved:.1f}"))
    return (f"One linear layer Z = X W^T with D = {d} inputs and H = {h} outputs (the first matrix of a transformer feed-forward block), "
            "float32. FLOPs = 2 B D H. Bytes = 4 (B D + H D + B H): the batch is read, the weights are read once, the result is written.\n\n"
            + _table(["batch B", "FLOPs", "bytes moved", "FLOPs per byte"], rows))
