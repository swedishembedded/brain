# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Proving Gradients Correct".

Plain Python 3, standard library only, written from the chapter's own
description and independent of any engine code. Each public function returns
a markdown string that the chapter embeds between
`<!-- worked: gradient_checking.<name> -->` markers; the same module supplies
the curves of the chapter's two figures.

The running example is the two-layer network of "Backpropagation Derived from
Scratch" (X[2,3] -> linear(4) -> tanh -> linear(1) -> mean squared error, with
exactly that chapter's numbers). Where a chapter example needs other sizes it
says so and draws its numbers from `bookrng.generator`.

Arithmetic comes in two formats. `IDENT` is IEEE double precision (Python's
float). `r32` rounds a double to the nearest 32-bit float, and a computation
"in 32 bits" rounds after every single +, -, * and after every tanh, which for
one operation is exactly the correctly rounded 32-bit result.

Swedish Embedded AB implements solutions for trustworthy gradient
verification of neural networks. If your team needs expertise in
finite-difference gradient checking, tolerances and test design for
hand-written backward passes, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import struct

import bookrng

# ------------------------------------------------------------------ formats


def r32(x):
    """The 32-bit float nearest to x (overflow gives a signed infinity)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def IDENT(x):
    return x


U32 = 2.0 ** -24
U64 = 2.0 ** -53

# ------------------------------------------------------------------ the network


class ShapeError(Exception):
    """A matrix product or an assignment whose shapes do not fit."""


ORDER = ["W1", "b1", "W2", "b2"]

CH4_X = [[1.0, 2.0, -1.0], [0.5, -1.0, 1.5]]
CH4_T = [[0.7], [-0.2]]
CH4_P = {
    "W1": [[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.1], [0.1, -0.4, 0.2]],
    "b1": [0.1, -0.1, 0.0, 0.2],
    "W2": [[0.5, -0.3, 0.8, 0.2]],
    "b2": [0.1],
}


def copy_params(P):
    return {k: [list(r) if isinstance(r, list) else r for r in v] for k, v in P.items()}


def flat_of(t):
    if isinstance(t[0], list):
        return [v for row in t for v in row]
    return list(t)


def numel(P, name):
    return len(flat_of(P[name]))


def entries(P):
    """Every parameter entry as (tensor name, flat index), in the order W1, b1, W2, b2."""
    return [(n, i) for n in ORDER for i in range(numel(P, n))]


def entry_label(P, name, i):
    t = P[name]
    if isinstance(t[0], list):
        cols = len(t[0])
        return f"{name}[{i // cols},{i % cols}]"
    return f"{name}[{i}]"


def with_entry(P, name, i, value):
    Q = copy_params(P)
    t = Q[name]
    if isinstance(t[0], list):
        t[i // len(t[0])][i % len(t[0])] = value
    else:
        t[i] = value
    return Q


def get_entry(P, name, i):
    return flat_of(P[name])[i]


def flat_vector(P):
    return [v for n in ORDER for v in flat_of(P[n])]


def forward(P, X, T, r=IDENT, act="tanh"):
    """Loss, hidden pre-activation Z1, activation A and error E.

    `r` rounds after every single arithmetic operation. The sums run in the
    order of the inner index, one rounding per product and one per addition.
    """
    B, H, O = len(X), len(P["W1"]), len(P["W2"])
    Z1, A = [], []
    for x in X:
        zrow, arow = [], []
        for j in range(H):
            acc = r(0.0)
            for d, xd in enumerate(x):
                acc = r(acc + r(xd * P["W1"][j][d]))
            z = r(acc + P["b1"][j])
            zrow.append(z)
            arow.append(r(math.tanh(z)) if act == "tanh" else (z if z > 0.0 else 0.0))
        Z1.append(zrow)
        A.append(arow)
    E, loss = [], r(0.0)
    for i in range(B):
        erow = []
        for o in range(O):
            acc = r(0.0)
            for j in range(H):
                acc = r(acc + r(A[i][j] * P["W2"][o][j]))
            e = r(r(acc + P["b2"][o]) - T[i][o])
            erow.append(e)
            loss = r(loss + r(e * e))
        E.append(erow)
    return r(loss / (B * O)), Z1, A, E


def loss_of(P, X, T, r=IDENT, act="tanh"):
    return forward(P, X, T, r, act)[0]


def mm(a, b):
    if len(a[0]) != len(b):
        raise ShapeError(f"[{len(a)},{len(a[0])}] x [{len(b)},{len(b[0])}]")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def tr(a):
    return [list(c) for c in zip(*a)]


BUGS = [
    "none", "drop_two", "mean_B", "no_tanh_deriv", "deriv_from_preact",
    "transpose_w2", "transpose_dw1", "db2_first_row", "skip_last_example",
]


def backward(P, X, T, bug="none", act="tanh"):
    """The backward pass of "Backpropagation Derived from Scratch", with one optional bug.

    Returns {"W1": ..., "b1": ..., "W2": ..., "b2": ...} in the shapes of the
    parameters. A bug that produces a matrix of the wrong shape raises ShapeError.
    """
    B, H, O = len(X), len(P["W1"]), len(P["W2"])
    _, Z1, A, E = forward(P, X, T, IDENT, act)
    num = 2.0 * (0.5 if bug == "drop_two" else 1.0)
    den = B if bug == "mean_B" else B * O
    dY = [[num * E[i][o] / den for o in range(O)] for i in range(B)]
    dW2 = mm(tr(dY), A)
    db2 = dY[0] if bug == "db2_first_row" else [sum(dY[i][o] for i in range(B)) for o in range(O)]
    dA = mm(dY, tr(P["W2"])) if bug == "transpose_w2" else mm(dY, P["W2"])
    if act == "tanh":
        local = [[1.0 - A[i][j] * A[i][j] for j in range(H)] for i in range(B)]
        if bug == "deriv_from_preact":
            local = [[1.0 - Z1[i][j] * Z1[i][j] for j in range(H)] for i in range(B)]
    else:
        local = [[1.0 if Z1[i][j] > 0.0 else 0.0 for j in range(H)] for i in range(B)]
    dZ1 = dA if bug == "no_tanh_deriv" else [[dA[i][j] * local[i][j] for j in range(H)] for i in range(B)]
    if bug == "transpose_dw1":
        dW1 = mm(tr(X), dZ1)
    elif bug == "skip_last_example":
        dW1 = mm(tr(dZ1[:-1]), X[:-1]) if B > 1 else [[0.0] * len(X[0]) for _ in range(H)]
    else:
        dW1 = mm(tr(dZ1), X)
    db1 = [sum(dZ1[i][j] for i in range(B)) for j in range(H)]
    G = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    for n in ORDER:
        want = (len(P[n]), len(P[n][0])) if isinstance(P[n][0], list) else (len(P[n]),)
        got = (len(G[n]), len(G[n][0])) if isinstance(G[n][0], list) else (len(G[n]),)
        if want != got:
            raise ShapeError(f"gradient of {n} has shape {got}, the parameter {want}")
    return G


def grad_vector(G):
    return [v for n in ORDER for v in flat_of(G[n])]


def central(loss, P, name, i, h, r=IDENT):
    """(L(w + h) - L(w - h)) / (2 h) for one entry, the nominal step in the denominator."""
    w = get_entry(P, name, i)
    return (loss(with_entry(P, name, i, r(w + h))) - loss(with_entry(P, name, i, r(w - h)))) / (2.0 * h)


def forward_diff(loss, P, name, i, h, r=IDENT):
    w = get_entry(P, name, i)
    return (loss(with_entry(P, name, i, r(w + h))) - loss(P)) / h


def numeric_gradient(loss, P, h):
    return [central(loss, P, n, i, h) for (n, i) in entries(P)]


def within(a, n, atol, rtol):
    """|a - n| <= atol + rtol max(|a|, |n|): the gate of the chapter (symmetric in a and n)."""
    return abs(a - n) <= atol + rtol * max(abs(a), abs(n))


# ------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sci(v, d=1):
    return f"{v:.{d}e}"


def _fx(v, d=4):
    return f"{v + 0.0:.{d}f}"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))


# ------------------------------------------------------------------ the opening: a bug that trains


TRAIN_DIMS = (24, 5, 7, 2)


def _training_problem():
    """A regression a small network can fit: targets from a random 3-unit tanh teacher."""
    name = "gradient_checking.buggy_training"
    B, D, H, O = TRAIN_DIMS
    rng = bookrng.generator(name)
    X = [[rng.normal() for _ in range(D)] for _ in range(B)]
    V = [[rng.normal() for _ in range(3)] for _ in range(D)]
    Wt = [[rng.normal() for _ in range(O)] for _ in range(3)]
    T = []
    for x in X:
        h = [math.tanh(sum(x[d] * V[d][k] for d in range(D))) for k in range(3)]
        T.append([sum(h[k] * Wt[k][o] for k in range(3)) for o in range(O)])
    P = {
        "W1": [[0.3 * rng.normal() for _ in range(D)] for _ in range(H)],
        "b1": [0.0] * H,
        "W2": [[0.3 * rng.normal() for _ in range(H)] for _ in range(O)],
        "b2": [0.0] * O,
    }
    return name, P, X, T


def _train(P0, X, T, bug, lr, checkpoints):
    P = copy_params(P0)
    out = {}
    last = max(checkpoints)
    for s in range(last + 1):
        if s in checkpoints:
            out[s] = loss_of(P, X, T)
        if s == last:
            break
        G = backward(P, X, T, bug)
        for n in ORDER:
            if isinstance(P[n][0], list):
                P[n] = [[w - lr * g for w, g in zip(pr, gr)] for pr, gr in zip(P[n], G[n])]
            else:
                P[n] = [w - lr * g for w, g in zip(P[n], G[n])]
    return out


def buggy_training():
    name, P0, X, T = _training_problem()
    B, D, H, O = TRAIN_DIMS
    cps = [0, 100, 500, 1000, 2000]
    runs = [
        ("correct backward, learning rate 0.10", "none", 0.10),
        ("correct backward, learning rate 0.05", "none", 0.05),
        ("dropped factor 2 in dY, learning rate 0.10", "drop_two", 0.10),
        ("mean over B instead of B O, learning rate 0.10", "mean_B", 0.10),
        ("batch sum of dW1 misses the last example, learning rate 0.10", "skip_last_example", 0.10),
        ("tanh derivative omitted, learning rate 0.10", "no_tanh_deriv", 0.10),
    ]
    rows = []
    for label, bug, lr in runs:
        t = _train(P0, X, T, bug, lr, set(cps))
        rows.append([label] + [f"{t[s]:.4f}" for s in cps])
    g_ok = grad_vector(backward(P0, X, T))
    rows2 = []
    for label, bug in (("dropped factor 2 in dY", "drop_two"), ("mean over B instead of B O", "mean_B"),
                       ("batch sum of dW1 misses the last example", "skip_last_example"), ("tanh derivative omitted", "no_tanh_deriv")):
        g = grad_vector(backward(P0, X, T, bug))
        err = _norm([a - b for a, b in zip(g, g_ok)]) / _norm(g_ok)
        rows2.append([label, f"{_cos(g, g_ok):.6f}", f"{err:.3f}"])
    head = (f"Setup: B = {B} examples, D = {D} inputs, H = {H} hidden units, O = {O} outputs, tanh, mean squared error. Generator: the book's SplitMix64 with seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}); draws in order: X ({B} x {D}, row-major), "
            f"V ({D} x 3, row-major), Wt (3 x {O}, row-major), W1 ({H} x {D}, row-major, each draw multiplied by 0.3), W2 ({O} x {H}, row-major, each draw multiplied by 0.3), all normal(); the biases start at 0. "
            "The targets are T = tanh(X V) Wt (tanh elementwise). Plain full-batch gradient descent in double precision, 2000 steps; each entry is the loss before the step with that index.\n\n")
    mid = "\n\nAt the starting weights, the gradient each buggy backward pass returns against the correct one (the 7 x 5 + 7 + 2 x 7 + 2 = 58 numbers):\n\n"
    return (head + _table(["backward pass used"] + [f"step {s}" for s in cps], rows) + mid
            + _table(["bug", "cosine with the correct gradient", "relative L2 error"], rows2))


# ------------------------------------------------------------------ finite differences from Taylor


ENTRY = ("W1", 8)  # W1[2,2], the entry the step-size examples follow


def _entry_loss(r=IDENT):
    return lambda P: loss_of(P, CH4_X, CH4_T, r)


def _derivs_of_entry(name, i):
    """Second and third derivative of the loss along one entry, from f64 differences at h = 1e-2."""
    L = _entry_loss()
    w = get_entry(CH4_P, name, i)
    f = lambda t: L(with_entry(CH4_P, name, i, w + t))
    h = 1e-2
    d2 = (f(h) - 2 * f(0.0) + f(-h)) / (h * h)
    d3 = (f(2 * h) - 2 * f(h) + 2 * f(-h) - f(-2 * h)) / (2 * h ** 3)
    return d2, d3


def fd_orders():
    name, i = ENTRY
    g = flat_of(backward(CH4_P, CH4_X, CH4_T)[name])[i]
    L = _entry_loss()
    d2, d3 = _derivs_of_entry(name, i)
    hs = [1e-1, 1e-2, 1e-3, 1e-4]
    rows, prev_f, prev_c = [], None, None
    for h in hs:
        ef = abs(forward_diff(L, CH4_P, name, i, h) - g)
        ec = abs(central(L, CH4_P, name, i, h) - g)
        rows.append([f"{h:.0e}", _sci(ef, 3), "" if prev_f is None else f"{prev_f / ef:.2f}", _sci(abs(d2) * h / 2, 3),
                     _sci(ec, 3), "" if prev_c is None else f"{prev_c / ec:.1f}", _sci(abs(d3) * h * h / 6, 3)])
        prev_f, prev_c = ef, ec
    head = [
        f"The loss of Chapter 4's network as a function of the single weight W1[2,2] (all other weights fixed at Chapter 4's values), in double precision. The reference derivative is the backward pass's {g:.8f}. The second and third derivatives are L'' = {d2:.4f} and L''' = {d3:.4f}, estimated by differences of L at steps 0.01 and 0.02.",
        "",
    ]
    return "\n".join(head) + "\n" + _table(
        ["h", "forward error", "ratio to previous row", "predicted L'' h / 2", "central error", "ratio to previous row", "predicted L''' h^2 / 6"], rows)


# ------------------------------------------------------------------ step size: truncation against round-off


def _p32():
    P = copy_params(CH4_P)
    for n in ORDER:
        if isinstance(P[n][0], list):
            P[n] = [[r32(v) for v in row] for row in P[n]]
        else:
            P[n] = [r32(v) for v in P[n]]
    return P, [[r32(v) for v in row] for row in CH4_X], [[r32(v) for v in row] for row in CH4_T]


def curve(fmt, kind, hs, name=ENTRY[0], i=ENTRY[1]):
    """Absolute error of the forward or central difference of the loss along one entry, in 32 or 64 bits."""
    g = backward(CH4_P, CH4_X, CH4_T)[name]
    g = flat_of(g)[i]
    if fmt == "32-bit":
        P, X, T = _p32()
        r = r32
    else:
        P, X, T, r = CH4_P, CH4_X, CH4_T, IDENT
    L = lambda Q: loss_of(Q, X, T, r)
    fn = forward_diff if kind == "forward" else central
    return [abs(fn(L, P, name, i, h, r) - g) for h in hs]


def step_curve():
    hs = [10.0 ** -k for k in range(1, 9)]
    c = {(f, k): curve(f, k, hs) for f in ("32-bit", "64-bit") for k in ("forward", "central")}
    rows = [[f"1e-{k + 1}"] + [_sci(c[(f, kind)][k], 2) for f in ("32-bit", "64-bit") for kind in ("forward", "central")]
            for k in range(len(hs))]
    head = ("Absolute error of the difference quotient of the loss along W1[2,2] (exact derivative 1.60223547), the loss and every operation of the forward pass rounded to the format. "
            "Conventions: tanh is the correctly rounded library tanh (rounded to 32 bits in the 32-bit columns); the perturbed weight w + h or w - h is rounded to the format before the loss is evaluated; "
            "the losses are 32-bit numbers in the 32-bit columns, but the quotient (L(w + h) - L(w - h)) / (2 h) is formed in double precision with the nominal 2 h (forward: (L(w + h) - L(w)) / h), "
            "and the error is its absolute difference from the double-precision backward gradient.\n\n")
    return head + _table(["h", "32-bit forward", "32-bit central", "64-bit forward", "64-bit central"], rows)


def predicted_central_step(u):
    """The step h* = (3 u L / |L'''|)^(1/3) for the loss along the followed entry (see step_optimum)."""
    _, d3 = _derivs_of_entry(*ENTRY)
    return (3 * u * loss_of(CH4_P, CH4_X, CH4_T) / abs(d3)) ** (1 / 3)


def step_optimum():
    d2, d3 = _derivs_of_entry(*ENTRY)
    L0 = loss_of(CH4_P, CH4_X, CH4_T)
    ks = list(range(1, 31))
    hs = [2.0 ** -k for k in ks]
    rows = []
    for fmt, u in (("32-bit", U32), ("64-bit", U64)):
        ec = curve(fmt, "central", hs)
        j = min(range(len(hs)), key=lambda t: ec[t])
        hstar = (3 * u * L0 / abs(d3)) ** (1 / 3)
        pred = abs(d3) * hstar ** 2 / 2
        rows.append([fmt, _sci(u, 2), _sci(hstar, 2), _sci(pred, 2), f"2^-{ks[j]} = {hs[j]:.2e}", _sci(ec[j], 2)])
    head = (f"Central difference along W1[2,2], scanning h = 2^-1 ... 2^-30. Inputs: L = {L0:.6f}, |L'''| = {abs(d3):.4f} (as in fd_orders). "
            "Predicted: h* = (3 u L / |L'''|)^(1/3) and worst-case error |L'''| h*^2 / 2.\n\n")
    return head + _table(["format", "u", "predicted h*", "predicted worst-case error", "best h found", "best error found"], rows)


def step_quantisation():
    w = r32(0.2)
    rows = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        eff = r32(w + h) - r32(w - h)
        rows.append([f"{h:.0e}", f"{2 * h:.10f}", f"{eff:.10f}", _sci(abs(eff - 2 * h) / (2 * h), 2)])
    head = f"The weight is w = {w:.10f} (0.2 rounded to 32 bits). The two perturbed weights are rounded to 32 bits, and the step the loss actually saw is their difference.\n\n"
    return head + _table(["nominal h", "nominal 2h", "actual (w+h) - (w-h)", "relative difference"], rows)


def accumulator_noise():
    """One entry's central difference when the loss is a long sum, accumulated in 32 or 64 bits."""
    n = 4096
    name = "gradient_checking.accumulator_noise"
    rng = bookrng.generator(name)
    a = [r32(rng.normal()) for _ in range(n)]
    b = [r32(rng.normal()) for _ in range(n)]
    w0, h = r32(0.3), 5e-3
    exact = math.fsum(2.0 * a[k] * (a[k] * w0 + b[k]) for k in range(n)) / n

    def loss(w, acc64):
        terms = [r32(r32(r32(a[k] * w) + b[k]) ** 2) for k in range(n)]
        if acc64:
            return math.fsum(terms) / n
        s = 0.0
        for t in terms:
            s = r32(s + t)
        return r32(s / n)

    rows = []
    for label, acc64 in (("32-bit accumulator", False), ("64-bit accumulator", True)):
        num = (loss(r32(w0 + h), acc64) - loss(r32(w0 - h), acc64)) / (2 * h)
        rows.append([label, f"{num:.6f}", _sci(abs(num - exact), 2)])
    head = (f"L(w) = (1/N) sum_k (a_k w + b_k)^2 with N = {n}, w = {w0:.10f}, h = 5e-3. Generator: the book's SplitMix64 with seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}); draws in order: a_0 ... a_{n - 1}, then b_0 ... b_{n - 1}, each normal() rounded to 32 bits. "
            f"w is 0.3 rounded to 32 bits, and the two evaluation points w + h and w - h are each rounded to 32 bits before use, so the quotient (L(w + h) - L(w - h)) / (2 h) is taken in 64 bits on 32-bit losses. "
            "Every product a_k w, sum with b_k and square inside a term is rounded to 32 bits; only the sum over k differs: the 32-bit accumulator adds the 4096 terms in order k = 0, 1, ... starting from 0, rounding every partial sum to 32 bits, and rounds the division by N to 32 bits; "
            f"the 64-bit accumulator sums the (still 32-bit) terms exactly and divides by N in 64 bits. The exact derivative is 2/N sum a_k (a_k w + b_k) = {exact:.6f}.\n\n")
    return head + _table(["sum over k", "central difference", "absolute error"], rows)


def f32_needs_large_steps():
    P32, X32, T32 = _p32()
    g = grad_vector(backward(CH4_P, CH4_X, CH4_T))
    g32 = [r32(v) for v in g]
    ents = entries(CH4_P)
    rows = []
    for h in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2):
        L = lambda Q: loss_of(Q, X32, T32, r32)
        nums = [central(L, P32, n, i, h, r32) for (n, i) in ents]
        worst_abs = max(abs(a - b) for a, b in zip(g32, nums))
        worst_rel = max(abs(a - b) / max(abs(a), abs(b), 1e-3) for a, b in zip(g32, nums))
        ok_f32 = sum(within(a, b, 4e-3, 8e-2) for a, b in zip(g32, nums))
        ok_f64 = sum(within(a, b, 1e-6, 1e-4) for a, b in zip(g32, nums))
        rows.append([f"{h:.0e}", _sci(worst_abs, 2), _sci(worst_rel, 2), f"{ok_f32} of 21", f"{ok_f64} of 21"])
    head = ("Per-entry central differences of Chapter 4's loss on all 21 parameters, with every operation of the forward pass rounded to 32 bits, against the backward pass's gradient rounded to 32 bits. "
            "Relative error is |a - n| / max(|a|, |n|, 0.001). A gate (atol, rtol) passes an entry when |a - n| <= atol + rtol max(|a|, |n|); tanh is the correctly rounded library tanh rounded to 32 bits, "
            "the perturbed weight is rounded to 32 bits, and the quotient is formed in double precision with the nominal 2 h. The counts are entries passing.\n\n")
    return head + _table(["h", "worst absolute error", "worst relative error", "loose gate (4e-3, 8e-2)", "strict gate (1e-6, 1e-4)"], rows)


def small_gradient_entries():
    P32, X32, T32 = _p32()
    g = grad_vector(backward(CH4_P, CH4_X, CH4_T))
    ents = entries(CH4_P)
    L32 = lambda Q: loss_of(Q, X32, T32, r32)
    nums = [central(L32, P32, n, i, 5e-3, r32) for (n, i) in ents]
    order = sorted(range(21), key=lambda k: abs(g[k]))
    pick = order[:4] + order[-2:]
    rows = []
    for k in pick:
        n, i = ents[k]
        ae = abs(g[k] - nums[k])
        rows.append([entry_label(CH4_P, n, i), f"{g[k]:+.6f}", f"{nums[k]:+.6f}", _sci(ae, 2), _sci(ae / max(abs(g[k]), abs(nums[k])), 2)])
    head = "32-bit per-entry central differences with h = 5e-3 on Chapter 4's network; the four entries with the smallest and the two with the largest gradient magnitude. Relative error is |a - n| / max(|a|, |n|).\n\n"
    return head + _table(["entry", "analytic a", "numeric n", "absolute error", "relative error"], rows)


# ------------------------------------------------------------------ comparing numbers


def compare_rules():
    pairs = [
        ("typical entry", 1.2293, 1.22935),
        ("small true gradient, noisy numeric", 2.2498e-4, 2.9e-4),
        ("true gradient zero, numeric noise", 0.0, 3e-5),
        ("large gradient, relative noise", 250.0, 250.2),
        ("dead gradient (analytic 0, true 2.2e-4)", 0.0, 2.2498e-4),
        ("wrong by a factor 2", 0.6, 1.2),
    ]
    rules = [("abs 1e-4", 1e-4, 0.0), ("rel 1e-2", 0.0, 1e-2), ("atol 1e-4 + rtol 1e-2", 1e-4, 1e-2), ("atol 4e-3 + rtol 8e-2", 4e-3, 8e-2)]
    rows = []
    for label, a, n in pairs:
        ae = abs(a - n)
        rel = ae / max(abs(a), abs(n))
        row = [label, f"{a:.6g}", f"{n:.6g}", _sci(ae, 2), f"{rel:.2f}"]
        for _, at, rt in rules:
            row.append("pass" if within(a, n, at, rt) else "FAIL")
        rows.append(row)
    return _table(["case", "analytic a", "numeric n", "|a - n|", "|a - n| / max(|a|, |n|)"] + [r[0] for r in rules], rows)


# ------------------------------------------------------------------ cost of a check


def check_cost():
    nets = [("tiny network of Chapter 4", 2, 3, 4, 1), ("digit classifier", 64, 784, 256, 10), ("one wide layer pair", 8, 4096, 16384, 4096)]
    rows = []
    for label, B, D, H, O in nets:
        P = H * D + H + O * H + O
        fwd = 2 * B * D * H + 2 * B * H * O
        bwd = 2 * O * B * H + 2 * B * O * H + 2 * H * B * D
        per_entry = 2 * P * fwd
        direc = 2 * 4 * 4 * fwd   # 4 tensors, 4 directions each, 2 forward passes per direction
        rows.append([label, f"{P:,}", f"{fwd:,}", f"{bwd:,}", f"{2 * P:,}", f"{per_entry / bwd:,.0f}", f"{direc / bwd:.1f}"])
    head = ("Work counts multiplications and additions of the matrix products only (a multiply-add is two operations), as in Chapter 4: forward = 2 B D H + 2 B H O; backward = 2 O B H + 2 B O H + 2 H B D. "
            "The directional check uses 4 tensors, 4 directions each, 2 forward passes per direction.\n\n")
    return head + _table(["network", "parameters P", "forward work F", "backward work", "per-entry check: forward passes (2P)", "per-entry work / backward work", "directional work / backward work"], rows)


# ------------------------------------------------------------------ the catalogue of bugs


def _random_problem(name, B, D, H, O, rng=None):
    rng = rng or bookrng.generator(name)
    X = [[rng.normal() for _ in range(D)] for _ in range(B)]
    T = [[rng.normal() for _ in range(O)] for _ in range(B)]
    P = {
        "W1": [[0.5 * rng.normal() for _ in range(D)] for _ in range(H)],
        "b1": [0.5 * rng.normal() for _ in range(H)],
        "W2": [[0.5 * rng.normal() for _ in range(H)] for _ in range(O)],
        "b2": [0.5 * rng.normal() for _ in range(O)],
    }
    return P, X, T


F64_ATOL, F64_RTOL = 1e-7, 1e-5
LOOSE_ATOL, LOOSE_RTOL = 4e-3, 8e-2


def _fd_gradient(P, X, T, act="tanh", h=1e-5):
    L = lambda Q: loss_of(Q, X, T, IDENT, act)
    return numeric_gradient(L, P, h)


def _directions(P, name, n_dirs):
    """n_dirs +-1 directions for every tensor, tensor by tensor in the order W1, b1, W2, b2."""
    rng = bookrng.generator(name)
    return {t: [[1.0 if rng.integer(2) == 1 else -1.0 for _ in range(numel(P, t))] for _ in range(n_dirs)] for t in ORDER}


def _directional_fails(P, X, T, G, dirs, eps=1e-4, act="tanh"):
    """Tensors whose best-of-n directional comparison fails the f64 gate (analytic <g,v> vs central difference along v)."""
    fails = []
    for t in ORDER:
        g = flat_of(G[t])
        best = None
        for v in dirs[t]:
            w0 = flat_of(P[t])
            def at(sign):
                Q = copy_params(P)
                vals = [a + sign * eps * b for a, b in zip(w0, v)]
                if isinstance(Q[t][0], list):
                    c = len(Q[t][0])
                    Q[t] = [vals[r * c:(r + 1) * c] for r in range(len(Q[t]))]
                else:
                    Q[t] = vals
                return loss_of(Q, X, T, IDENT, act)
            num = (at(+1) - at(-1)) / (2 * eps)
            ana = _dot(g, v)
            ok = within(ana, num, F64_ATOL, F64_RTOL)
            rel = abs(ana - num) / max(abs(ana), abs(num), 1e-3)
            if best is None or rel < best[0]:
                best = (rel, ok)
        if not best[1]:
            fails.append(t)
    return fails


CATALOGUE_DIMS = (3, 5, 7, 2)


def bug_catalogue():
    B, D, H, O = CATALOGUE_DIMS
    name = "gradient_checking.bug_catalogue"
    P, X, T = _random_problem(name, B, D, H, O)
    dname = "gradient_checking.bug_catalogue_directions"
    dirs = _directions(P, dname, 4)
    good = backward(P, X, T)
    gv = grad_vector(good)
    fd = _fd_gradient(P, X, T)
    ents = entries(P)
    desc = {
        "none": "none (control)",
        "drop_two": "dropped factor 2: dY = E / (B O)",
        "mean_B": "mean over B only: dY = 2 E / B",
        "no_tanh_deriv": "tanh derivative omitted: dZ1 = dA",
        "deriv_from_preact": "tanh derivative from Z1: 1 - Z1^2",
        "transpose_w2": "transposed weight: dA = dY W2^T",
        "transpose_dw1": "transposed product: dW1 = X^T dZ1",
        "db2_first_row": "bias gradient of one example: db2 = dY[0]",
        "skip_last_example": "batch sum misses last example in dW1",
    }
    rows = []
    for bug in BUGS:
        try:
            G = backward(P, X, T, bug)
        except ShapeError:
            rows.append([desc[bug], "-", "-", "rejected by the shape check", "-"])
            continue
        g = grad_vector(G)
        err = _norm([a - b for a, b in zip(g, gv)]) / _norm(gv)
        fail_entries = sum(not within(a, n, F64_ATOL, F64_RTOL) for a, n in zip(g, fd))
        fails = _directional_fails(P, X, T, G, dirs)
        rows.append([desc[bug], f"{_cos(g, gv):.6f}", f"{err:.3f}", f"{fail_entries} of {len(g)}",
                     "pass" if not fails else "FAIL: " + ", ".join(fails)])
    head = (f"Network B = {B}, D = {D}, H = {H}, O = {O} ({len(gv)} parameters), tanh, mean squared error, double precision. "
            f"Generator: the book's SplitMix64 with seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}); draws in order: X (row-major), T (row-major), W1 (row-major), b1, W2 (row-major), b2, each normal(), the last four multiplied by 0.5. "
            "Per-entry check: central difference with h = 1e-5 on every parameter. "
            f"Directional check: per tensor 4 directions of +-1 entries (a second generator, the book's SplitMix64 with seed 0x{bookrng.seed_of(dname):016x} ({bookrng.seed_of(dname)}); for W1, b1, W2, b2 in turn, four directions each, an entry is +1 when integer(2) is 1 and -1 otherwise), step eps = 1e-4, the best of the four kept by relative error. "
            "The 16 directions are drawn once, before any row is evaluated, and every row that is evaluated uses these same 16; a row rejected by the shape check is not evaluated, so it uses none. "
            f"Both use the gate |a - n| <= {F64_ATOL:g} + {F64_RTOL:g} max(|a|, |n|). The cosine and relative L2 error compare the 58 analytic values with the correct backward pass's.\n\n")
    return head + _table(["bug injected in the backward pass", "cosine with correct gradient", "relative L2 error", "per-entry check: entries failing", "directional check, best of 4"], rows,
                         "lrrrl")


# ------------------------------------------------------------------ test dimensions


DIM_SETS = [("all ones", (1, 1, 1, 1)), ("Chapter 4's network", (2, 3, 4, 1)), ("equal sizes", (2, 3, 3, 3)), ("distinct, all above 1", (3, 5, 7, 2))]
DIM_BUGS = [("mean_B", "mean over B instead of B O"), ("db2_first_row", "bias gradient of example 0 only"),
            ("transpose_w2", "dA = dY W2^T"), ("transpose_dw1", "dW1 = X^T dZ1")]


def dims_hide():
    name = "gradient_checking.dims_hide"
    rng = bookrng.generator(name)
    cells = {}
    for label, (B, D, H, O) in DIM_SETS:
        P, X, T = _random_problem(name, B, D, H, O, rng)
        good = grad_vector(backward(P, X, T))
        fd = _fd_gradient(P, X, T)
        for bug, _ in DIM_BUGS:
            try:
                g = grad_vector(backward(P, X, T, bug))
            except ShapeError:
                cells[(label, bug)] = "shape error"
                continue
            if max(abs(a - b) for a, b in zip(g, good)) < 1e-12:
                cells[(label, bug)] = "same gradient, invisible"
            else:
                bad = sum(not within(a, n, F64_ATOL, F64_RTOL) for a, n in zip(g, fd))
                cells[(label, bug)] = f"caught ({bad} of {len(g)} entries)" if bad else "wrong but passes"
    rows = [[f"{label}, (B, D, H, O) = {dims}"] + [cells[(label, b)] for b, _ in DIM_BUGS] for label, dims in DIM_SETS]
    head = (f"One generator, the book's SplitMix64 with seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}), draws the four problems one after another; for each, in the order X (row-major), T (row-major), W1 (row-major), b1, W2 (row-major), b2, each normal(), the last four multiplied by 0.5. "
            f"Per-entry check in double precision: central difference h = 1e-5, gate |a - n| <= {F64_ATOL:g} + {F64_RTOL:g} max(|a|, |n|). \"Shape error\": the buggy formula's result has a shape that cannot be the parameter's, or the product does not exist.\n\n")
    return head + _table(["test sizes"] + [d for _, d in DIM_BUGS], rows, "l" + "l" * len(DIM_BUGS))


# ------------------------------------------------------------------ directional against per entry


def directional_miss():
    B, D, H, O = CATALOGUE_DIMS
    name = "gradient_checking.directional_miss"
    rng = bookrng.generator(name)
    trials = 1000
    one_pass = best4_pass = entry_fail = 0
    entry_share = 0.0
    miss = 0.0
    nW = H * D
    eps = 1e-4
    for _ in range(trials):
        P, X, T = _random_problem(name, B, D, H, O, rng)
        dirs = [[1.0 if rng.integer(2) == 1 else -1.0 for _ in range(nW)] for _ in range(4)]
        good = backward(P, X, T)
        bad = backward(P, X, T, "skip_last_example")
        gW, bW = flat_of(good["W1"]), flat_of(bad["W1"])
        miss += _norm([a - b for a, b in zip(gW, bW)]) / _norm(gW)
        w0 = flat_of(P["W1"])
        def num_along(v):
            def at(s):
                Q = copy_params(P)
                vals = [a + s * eps * b for a, b in zip(w0, v)]
                Q["W1"] = [vals[r * D:(r + 1) * D] for r in range(H)]
                return loss_of(Q, X, T)
            return (at(1) - at(-1)) / (2 * eps)
        oks = [within(_dot(bW, v), num_along(v), LOOSE_ATOL, LOOSE_RTOL) for v in dirs]
        one_pass += oks[0]
        best4_pass += any(oks)
        L = lambda Q: loss_of(Q, X, T)
        fails = sum(not within(bW[k], central(L, P, "W1", k, 1e-5), LOOSE_ATOL, LOOSE_RTOL) for k in range(nW))
        entry_fail += fails > 0
        entry_share += fails / nW
    rows = [
        ["mean of ||missing part|| / ||gradient of W1||", f"{miss / trials:.3f}"],
        ["one random direction: bug accepted", f"{one_pass} of {trials}"],
        ["best of 4 random directions: bug accepted", f"{best4_pass} of {trials}"],
        ["per-entry check: at least one of the 35 entries fails", f"{entry_fail} of {trials}"],
        ["per-entry check: mean share of the 35 entries that fail", f"{entry_share / trials:.3f}"],
    ]
    head = (f"Bug: the batch sum of dW1 omits the last of B = {B} examples. One generator, the book's SplitMix64 with seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)}), draws {trials} problems one after another (D = {D}, H = {H}, O = {O}); "
            "for each, X, T, W1, b1, W2, b2 as in bug_catalogue, then four directions of 35 entries each (+1 when integer(2) is 1, otherwise -1), the first of the four being 'one random direction'. "
            f"Directional comparison: eps = {eps:g} on W1 alone; per entry: h = 1e-5; gate {LOOSE_ATOL:g} + {LOOSE_RTOL:g} max(|a|, |n|), double precision.\n\n")
    return head + _table(["measure", "result over 1000 random problems"], rows)


# ------------------------------------------------------------------ kinks


def kink_scan():
    h = 1e-3
    relu = lambda x: x if x > 0.0 else 0.0
    rows = []
    for d in (1e-1, 1e-2, 2e-3, 1e-3, 5e-4, 1e-4, 1e-6, 0.0, -1e-4, -2e-3):
        ana = 1.0 if d > 0.0 else 0.0
        num = (relu(d + h) - relu(d - h)) / (2 * h)
        rows.append([f"{d:+.0e}" if d else "0", f"{ana:.0f}", f"{num:.4f}", f"{abs(ana - num):.4f}", "agree" if abs(ana - num) < 1e-9 else "DISAGREE"])
    head = f"Central difference of relu(x) at x = delta with step h = {h:g}, against the slope 1 for x > 0 and 0 otherwise (double precision). A row agrees when |difference| < 1e-9.\n\n"
    return head + _table(["delta", "analytic slope", "central difference", "|difference|", "verdict"], rows)


def relu_ramp(h=0.1, n=61):
    """(x, central difference of relu at x with step h) for the kink figure."""
    relu = lambda x: x if x > 0.0 else 0.0
    xs = [-0.3 + 0.6 * k / (n - 1) for k in range(n)]
    return xs, [(relu(x + h) - relu(x - h)) / (2 * h) for x in xs]


def kink_network():
    P, X, T = CH4_P, CH4_X, CH4_T
    L = lambda Q: loss_of(Q, X, T, IDENT, "relu")
    _, Z1, _, _ = forward(P, X, T, IDENT, "relu")
    G = backward(P, X, T, "none", "relu")
    rows = []
    for j in range(4):
        zs = [Z1[i][j] for i in range(2)]
        a = G["b1"][j]
        n = central(L, P, "b1", j, 1e-5)
        rows.append([f"{j}", f"{zs[0]:+.3e}", f"{zs[1]:+.3e}", f"{a:+.6f}", f"{n:+.6f}", "agree" if within(a, n, F64_ATOL, F64_RTOL) else "DISAGREE"])
    head = ("Chapter 4's network with tanh replaced by ReLU (slope 1 for z > 0, otherwise 0). Hidden unit j has pre-activation Z1[b, j] for the two examples b = 0, 1. Per-entry check of the four hidden biases, central difference h = 1e-5, double precision, gate "
            f"{F64_ATOL:g} + {F64_RTOL:g} max(|a|, |n|). Each pre-activation is the dot product of a row of W1 with the example, summed from index 0 upward starting at 0.0, plus b1[j]; "
            "an entry of size 1e-17 is a rounding residue whose sign depends on that order.\n\n")
    return head + _table(["hidden unit j", "Z1[0, j]", "Z1[1, j]", "analytic db1[j]", "central difference", "verdict"], rows)


# ------------------------------------------------------------------ the check proves the derivative of the forward you wrote


def _gelu_erf_d(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + x * math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def _gelu_tanh_d(x):
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = math.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * x * x)


def tolerance_blindness():
    xs = [-4.0 + 0.01 * k for k in range(801)]
    diffs = [abs(_gelu_tanh_d(x) - _gelu_erf_d(x)) for x in xs]
    rows = []
    for label, at, rt in (("brain's 32-bit gate", 4e-3, 8e-2), ("a 64-bit gate", F64_ATOL, F64_RTOL)):
        caught = sum(not within(_gelu_tanh_d(x), _gelu_erf_d(x), at, rt) for x in xs)
        rows.append([label, f"{at:g}", f"{rt:g}", f"{caught} of {len(xs)}"])
    head = (f"The two GELU functions are gelu(x) = x Phi(x) with Phi the standard Gaussian distribution function, and the tanh form 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))). "
            f"Over the 801 points x = -4.00, -3.99, ..., 4.00 the largest difference between the two derivatives is {max(diffs):.2e}. "
            "The derivatives are Phi(x) + x phi(x) (phi the standard Gaussian density) and, for the tanh form with t = tanh(sqrt(2/pi)(x + 0.044715 x^3)), 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2), tanh the library tanh. "
            "Each gate is asked whether one derivative, taken as the analytic value, agrees with the other, taken as the numeric one.\n\n")
    return head + _table(["gate", "atol", "rtol", "points where the wrong derivative is rejected"], rows)
