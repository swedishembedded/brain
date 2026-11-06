# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Backpropagation Derived from Scratch".

Everything here is plain Python 3 on floats (IEEE double precision), written
from the chapter's own description of the algorithm and independent of any
engine code. Each public function returns a markdown string that the chapter
embeds between `<!-- worked: backprop.<name> -->` markers. The same module
supplies the numbers for the chapter's figures, so a picture and the table
beside it cannot disagree.

The three running examples:

  single weight   y = w x, loss L = (y - t)^2, a few gradient descent steps
  scalar chain    y = w2 tanh(w1 x), the smallest case that forces the chain rule
  tiny network    X[2,3] -> linear(4) -> tanh -> linear(1) -> mean squared error

Swedish Embedded AB implements solutions for verifiable gradient computation
in neural network training. If your team needs expertise in backpropagation
and automatic differentiation, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math

# ---------------------------------------------------------------- example (a)

SW_X, SW_T, SW_W0, SW_LR = 2.0, 3.0, 0.5, 0.05


def single_weight_step(w, x=SW_X, t=SW_T, lr=SW_LR):
    """One step of y = w x with L = (y - t)^2. Returns (y, error, loss, grad, new_w)."""
    y = w * x
    err = y - t
    loss = err * err
    grad = 2.0 * err * x  # dL/dw = dL/dy * dy/dw = 2 (y - t) * x
    return y, err, loss, grad, w - lr * grad


def single_weight_trajectory(steps=6):
    w, rows = SW_W0, []
    for k in range(steps):
        y, err, loss, grad, new_w = single_weight_step(w)
        rows.append((k, w, y, err, loss, grad, new_w))
        w = new_w
    return rows


def single_weight():
    out = [f"Setup: x = {SW_X}, target t = {SW_T}, start w = {SW_W0}, learning rate {SW_LR}.", "",
           "| step | w | y = w x | y - t | L | dL/dw | w after update |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for k, w, y, err, loss, grad, nw in single_weight_trajectory():
        out.append(f"| {k} | {w:.4f} | {y:.4f} | {err:.4f} | {loss:.4f} | {grad:.4f} | {nw:.4f} |")
    return "\n".join(out)


def single_weight_contraction():
    """The error w - w* shrinks by 1 - 2 eta x^2 per step; check it against the trajectory."""
    w_star = SW_T / SW_X
    factor = 1.0 - 2.0 * SW_LR * SW_X ** 2
    rows = single_weight_trajectory(5)
    out = [f"The minimiser is w* = t / x = {w_star}. The predicted contraction factor is "
           f"1 - 2 * {SW_LR} * {SW_X}^2 = {factor:.4f}.", "",
           "| step | w - w* | ratio to previous |", "|---:|---:|---:|"]
    prev = None
    for k, w, *_ in rows:
        e = w - w_star
        ratio = "" if prev is None else f"{e / prev:.4f}"
        out.append(f"| {k} | {e:.4f} | {ratio} |")
        prev = e
    return "\n".join(out)


def loss_curve(w):
    """L(w) for the single-weight example (used by the tangent-line figure)."""
    return (w * SW_X - SW_T) ** 2


def loss_slope(w):
    return 2.0 * (w * SW_X - SW_T) * SW_X


# --------------------------------------------------------------- scalar chain

SC_X, SC_W1, SC_W2, SC_T = 1.5, 0.4, 1.2, -0.5


def scalar_chain_values():
    """Forward and backward through  u = w1 x, h = tanh u, y = w2 h, e = y - t, L = e^2."""
    x, w1, w2, t = SC_X, SC_W1, SC_W2, SC_T
    u = w1 * x
    h = math.tanh(u)
    y = w2 * h
    e = y - t
    loss = e * e
    # backward: each line is (upstream gradient) * (local derivative)
    d_e = 2.0 * e
    d_y = d_e * 1.0
    d_w2 = d_y * h
    d_h = d_y * w2
    d_u = d_h * (1.0 - h * h)
    d_w1 = d_u * x
    return dict(x=x, w1=w1, w2=w2, t=t, u=u, h=h, y=y, e=e, L=loss,
                d_e=d_e, d_y=d_y, d_w2=d_w2, d_h=d_h, d_u=d_u, d_w1=d_w1)


def scalar_chain():
    v = scalar_chain_values()
    rows = [
        ("u = w1 x", v["u"], "dL/du = dL/dh * (1 - h^2)", v["d_u"]),
        ("h = tanh(u)", v["h"], "dL/dh = dL/dy * w2", v["d_h"]),
        ("y = w2 h", v["y"], "dL/dy = dL/de * 1", v["d_y"]),
        ("e = y - t", v["e"], "dL/de = 2 e", v["d_e"]),
        ("L = e^2", v["L"], "dL/dL = 1", 1.0),
    ]
    out = [f"Setup: x = {v['x']}, w1 = {v['w1']}, w2 = {v['w2']}, t = {v['t']}.", "",
           "| node (forward) | value | gradient rule (backward) | value |",
           "|---|---:|---|---:|"]
    for a, b, c, d in rows:
        out.append(f"| {a} | {b:.4f} | {c} | {d:.4f} |")
    out += ["", f"The two parameter gradients are dL/dw2 = dL/dy * h = {v['d_w2']:.4f} and "
                f"dL/dw1 = dL/du * x = {v['d_w1']:.4f}."]
    return "\n".join(out)


def scalar_chain_check():
    """Central difference on the scalar chain, as a sanity check of the rules above."""
    v = scalar_chain_values()

    def loss(w1, w2):
        return (w2 * math.tanh(w1 * SC_X) - SC_T) ** 2

    h = 1e-6
    n1 = (loss(SC_W1 + h, SC_W2) - loss(SC_W1 - h, SC_W2)) / (2 * h)
    n2 = (loss(SC_W1, SC_W2 + h) - loss(SC_W1, SC_W2 - h)) / (2 * h)
    return "\n".join([
        "| parameter | backward | central difference | absolute difference |",
        "|---|---:|---:|---:|",
        f"| w1 | {v['d_w1']:.6f} | {n1:.6f} | {abs(v['d_w1'] - n1):.1e} |",
        f"| w2 | {v['d_w2']:.6f} | {n2:.6f} | {abs(v['d_w2'] - n2):.1e} |",
    ])


# ------------------------------------------------------------ matrix helpers


def matmul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][p] * b[p][j] for p in range(k)) for j in range(m)] for i in range(n)]


def transpose(a):
    return [list(r) for r in zip(*a)]


def col_sums(a):
    return [sum(r[j] for r in a) for j in range(len(a[0]))]


def add_row(a, bias):
    return [[a[i][j] + bias[j] for j in range(len(bias))] for i in range(len(a))]


def md_matrix(corner, a, row_names, col_names, digits=4):
    head = "| " + corner + " | " + " | ".join(col_names) + " |"
    sep = "|---|" + "---:|" * len(col_names)
    body = ["| " + row_names[i] + " | " + " | ".join(f"{v:.{digits}f}" for v in row) + " |"
            for i, row in enumerate(a)]
    return "\n".join([head, sep] + body)


# ------------------------------------------------------------ tiny network
#
# Dimensions are all different (B=2, D=3, H=4, O=1) so that a transposed
# matrix or a swapped axis is a shape error and not a silent wrong answer.

B, D, H, O = 2, 3, 4, 1
X = [[1.0, 2.0, -1.0], [0.5, -1.0, 1.5]]
T = [[0.7], [-0.2]]
W1 = [[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.1], [0.1, -0.4, 0.2]]
B1 = [0.1, -0.1, 0.0, 0.2]
W2 = [[0.5, -0.3, 0.8, 0.2]]
B2 = [0.1]
PARAM_SHAPES = [("W1", H, D), ("b1", H, None), ("W2", O, H), ("b2", O, None)]


def initial_params():
    return {"W1": [r[:] for r in W1], "b1": B1[:], "W2": [r[:] for r in W2], "b2": B2[:]}


def forward(p):
    """Forward pass. Returns every intermediate the backward pass will need."""
    z1 = add_row(matmul(X, transpose(p["W1"])), p["b1"])
    a = [[math.tanh(v) for v in row] for row in z1]
    y = add_row(matmul(a, transpose(p["W2"])), p["b2"])
    e = [[y[i][j] - T[i][j] for j in range(O)] for i in range(B)]
    loss = sum(v * v for row in e for v in row) / (B * O)
    return dict(z1=z1, a=a, y=y, e=e, loss=loss)


def backward(p, f):
    """Backward pass: gradients of the mean squared error with respect to everything."""
    d_y = [[2.0 * v / (B * O) for v in row] for row in f["e"]]                     # [B,O]
    d_w2 = matmul(transpose(d_y), f["a"])                                            # [O,H]
    d_b2 = col_sums(d_y)                                                             # [O]
    d_a = matmul(d_y, p["W2"])                                                       # [B,H]
    d_z1 = [[d_a[i][j] * (1.0 - f["a"][i][j] ** 2) for j in range(H)] for i in range(B)]
    d_w1 = matmul(transpose(d_z1), X)                                                # [H,D]
    d_b1 = col_sums(d_z1)                                                            # [H]
    d_x = matmul(d_z1, p["W1"])                                                      # [B,D]
    return dict(d_y=d_y, d_w2=d_w2, d_b2=d_b2, d_a=d_a, d_z1=d_z1, d_w1=d_w1, d_b1=d_b1, d_x=d_x)


def _names(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def network_inputs():
    parts = [
        f"Batch `X` ({B} examples, {D} features) and targets `T` ({B} examples, {O} output):", "",
        md_matrix("X", X, _names("b=", B), _names("d=", D), 2), "",
        md_matrix("T", T, _names("b=", B), _names("o=", O), 2), "",
        f"First layer `W1` ({H} x {D}) and `b1` ({H}):", "",
        md_matrix("W1", W1, _names("h=", H), _names("d=", D), 2), "",
        md_matrix("b1", [B1], ["value"], _names("h=", H), 2), "",
        f"Second layer `W2` ({O} x {H}) and `b2` ({O}):", "",
        md_matrix("W2", W2, _names("o=", O), _names("h=", H), 2), "",
        md_matrix("b2", [B2], ["value"], _names("o=", O), 2),
    ]
    return "\n".join(parts)


def network_forward():
    p = initial_params()
    f = forward(p)
    parts = [
        "Pre-activation `Z1 = X W1^T + b1` ([B, H]):", "",
        md_matrix("Z1", f["z1"], _names("b=", B), _names("h=", H)), "",
        "Hidden activation `A = tanh(Z1)` ([B, H]):", "",
        md_matrix("A", f["a"], _names("b=", B), _names("h=", H)), "",
        "Output `Y = A W2^T + b2` and error `E = Y - T` ([B, O]):", "",
        "| b | Y | T | E = Y - T | E^2 |", "|---|---:|---:|---:|---:|",
    ]
    for i in range(B):
        parts.append(f"| b={i} | {f['y'][i][0]:.4f} | {T[i][0]:.4f} | {f['e'][i][0]:.4f} | {f['e'][i][0] ** 2:.4f} |")
    parts += ["", f"Loss `L = (1 / (B O)) * sum E^2` = {f['loss']:.6f}."]
    return "\n".join(parts)


def network_backward_output():
    p = initial_params()
    g = backward(p, forward(p))
    parts = [
        "Output gradient `dY = 2 E / (B O)` ([B, O]):", "",
        md_matrix("dY", g["d_y"], _names("b=", B), _names("o=", O)), "",
        "Second-layer parameter gradients, `dW2 = dY^T A` ([O, H]) and `db2 = sum over b of dY` ([O]):", "",
        md_matrix("dW2", g["d_w2"], _names("o=", O), _names("h=", H)), "",
        md_matrix("db2", [g["d_b2"]], ["value"], _names("o=", O)),
    ]
    return "\n".join(parts)


def network_backward_hidden():
    p = initial_params()
    f = forward(p)
    g = backward(p, f)
    local = [[1.0 - f["a"][i][j] ** 2 for j in range(H)] for i in range(B)]
    parts = [
        "Gradient into the activation, `dA = dY W2` ([B, H]):", "",
        md_matrix("dA", g["d_a"], _names("b=", B), _names("h=", H)), "",
        "Local derivative of tanh from the saved activation, `1 - A^2` ([B, H]):", "",
        md_matrix("1 - A^2", local, _names("b=", B), _names("h=", H)), "",
        "Gradient into the pre-activation, `dZ1 = dA * (1 - A^2)` elementwise ([B, H]):", "",
        md_matrix("dZ1", g["d_z1"], _names("b=", B), _names("h=", H)),
    ]
    return "\n".join(parts)


def network_backward_first():
    p = initial_params()
    g = backward(p, forward(p))
    parts = [
        "First-layer parameter gradients, `dW1 = dZ1^T X` ([H, D]) and `db1 = sum over b of dZ1` ([H]):", "",
        md_matrix("dW1", g["d_w1"], _names("h=", H), _names("d=", D)), "",
        md_matrix("db1", [g["d_b1"]], ["value"], _names("h=", H)), "",
        "For completeness, the gradient with respect to the input, `dX = dZ1 W1` ([B, D]):", "",
        md_matrix("dX", g["d_x"], _names("b=", B), _names("d=", D)),
    ]
    return "\n".join(parts)


def network_step():
    """One gradient descent step on the tiny network: the loss must go down."""
    lr = 0.01
    p = initial_params()
    f = forward(p)
    g = backward(p, f)
    new = {"W1": [[p["W1"][i][j] - lr * g["d_w1"][i][j] for j in range(D)] for i in range(H)],
           "b1": [p["b1"][j] - lr * g["d_b1"][j] for j in range(H)],
           "W2": [[p["W2"][i][j] - lr * g["d_w2"][i][j] for j in range(H)] for i in range(O)],
           "b2": [p["b2"][j] - lr * g["d_b2"][j] for j in range(O)]}
    f2 = forward(new)
    gnorm2 = sum(v * v for k in ("d_w1", "d_b1", "d_w2", "d_b2")
                 for v in (g[k] if isinstance(g[k][0], float) else [x for r in g[k] for x in r]))
    return "\n".join([
        f"Learning rate {lr}. The squared length of the whole gradient is {gnorm2:.6f}, so a small step should lower "
        f"the loss by about {lr} * {gnorm2:.6f} = {lr * gnorm2:.6f}.", "",
        "| | loss |", "|---|---:|",
        f"| before the step | {f['loss']:.6f} |",
        f"| after the step | {f2['loss']:.6f} |",
        f"| actual decrease | {f['loss'] - f2['loss']:.6f} |",
    ])


# ----------------------------------------------------- generic scalar program
#
# The same network written with nothing but +, - and * on scalars, so it runs
# unchanged on plain floats (for central differences) and on dual numbers
# (for forward mode).


def _tanh(v):
    return v.tanh() if isinstance(v, Dual) else math.tanh(v)


def loss_from_flat(theta):
    """The tiny network's loss as a function of the 21 parameters in one flat list."""
    w1 = [theta[i * D:(i + 1) * D] for i in range(H)]
    b1 = theta[H * D:H * D + H]
    off = H * D + H
    w2 = [theta[off + i * H:off + (i + 1) * H] for i in range(O)]
    b2 = theta[off + O * H:off + O * H + O]
    total = 0.0
    for i in range(B):
        a = []
        for j in range(H):
            z = b1[j]
            for k in range(D):
                z = z + X[i][k] * w1[j][k]
            a.append(_tanh(z))
        for o in range(O):
            y = b2[o]
            for j in range(H):
                y = y + a[j] * w2[o][j]
            e = y - T[i][o]
            total = total + e * e
    return total * (1.0 / (B * O))


def flat_params():
    p = initial_params()
    return ([v for r in p["W1"] for v in r] + p["b1"] + [v for r in p["W2"] for v in r] + p["b2"])


def param_labels():
    labels = [f"W1[{i},{j}]" for i in range(H) for j in range(D)]
    labels += [f"b1[{j}]" for j in range(H)]
    labels += [f"W2[{o},{j}]" for o in range(O) for j in range(H)]
    labels += [f"b2[{o}]" for o in range(O)]
    return labels


def flat_gradient():
    p = initial_params()
    g = backward(p, forward(p))
    return ([v for r in g["d_w1"] for v in r] + g["d_b1"] + [v for r in g["d_w2"] for v in r] + g["d_b2"])


def central_difference(theta, i, h=1e-5):
    up, dn = theta[:], theta[:]
    up[i] += h
    dn[i] -= h
    return (loss_from_flat(up) - loss_from_flat(dn)) / (2.0 * h)


def finite_difference():
    theta, grad = flat_params(), flat_gradient()
    out = ["Step size h = 1e-5, double precision. One row per parameter.", "",
           "| parameter | backprop gradient | central difference | absolute difference |",
           "|---|---:|---:|---:|"]
    worst = 0.0
    for i, name in enumerate(param_labels()):
        n = central_difference(theta, i)
        worst = max(worst, abs(grad[i] - n))
        out.append(f"| {name} | {grad[i]:.8f} | {n:.8f} | {abs(grad[i] - n):.1e} |")
    out += ["", f"Largest absolute difference over all {len(theta)} parameters: {worst:.1e}."]
    return "\n".join(out)


def wrong_backward():
    """The same comparison when one line of the backward pass is wrong: tanh derivative forgotten."""
    p = initial_params()
    f = forward(p)
    g = backward(p, f)
    # Deliberately wrong: pretend tanh has derivative 1 (skip the 1 - A^2 factor).
    d_z1 = g["d_a"]
    d_w1 = matmul(transpose(d_z1), X)
    d_b1 = col_sums(d_z1)
    theta = flat_params()
    bad = [v for r in d_w1 for v in r] + d_b1 + flat_gradient()[H * D + H:]
    out = ["| parameter | wrong backward | central difference | relative error |", "|---|---:|---:|---:|"]
    labels = param_labels()
    for i in (0, 3, H * D, H * D + 2, H * D + H, H * D + H + O * H):
        n = central_difference(theta, i)
        rel = abs(bad[i] - n) / max(abs(bad[i]), abs(n), 1e-12)
        out.append(f"| {labels[i]} | {bad[i]:.6f} | {n:.6f} | {rel:.1e} |")
    return "\n".join(out)


# --------------------------------------------------------------- forward mode


class Dual:
    """A number carrying its derivative with respect to one chosen input (forward mode)."""

    def __init__(self, v, d=0.0):
        self.v, self.d = v, d

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v - o.v, self.d - o.d)

    def __rsub__(self, o):
        return Dual(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def tanh(self):
        t = math.tanh(self.v)
        return Dual(t, (1.0 - t * t) * self.d)


def forward_mode_gradient():
    theta = flat_params()
    grads = []
    for i in range(len(theta)):
        seeded = [Dual(v, 1.0 if k == i else 0.0) for k, v in enumerate(theta)]
        grads.append(loss_from_flat(seeded).d)
    return grads


def forward_mode_check():
    fm, rv = forward_mode_gradient(), flat_gradient()
    n = len(fm)
    spans = [("W1", 0, H * D), ("b1", H * D, H * D + H), ("W2", H * D + H, H * D + H + O * H),
             ("b2", H * D + H + O * H, n)]
    out = ["| tensor | entries | forward mode passes | largest |forward mode - reverse mode| |",
           "|---|---:|---:|---:|"]
    for name, a, b in spans:
        worst = max(abs(fm[i] - rv[i]) for i in range(a, b))
        out.append(f"| {name} | {b - a} | {b - a} | {worst:.1e} |")
    out += ["", f"Forward mode needed {n} passes, one per parameter. Reverse mode needed one forward pass and one "
                f"backward pass."]
    return "\n".join(out)


# ----------------------------------------------------------------------- cost


def mlp_costs(b, d, h, o):
    params = h * d + h + o * h + o
    fwd = 2 * b * d * h + 2 * b * h * o              # two matrix products (multiply + add = 2 operations)
    bwd_no_dx = 2 * b * h * o + 2 * b * h * o + 2 * b * d * h   # dW2, dA, dW1
    bwd_dx = bwd_no_dx + 2 * b * d * h                         # plus dX of the first layer
    return params, fwd, bwd_no_dx, bwd_dx


def cost_table():
    shapes = [("tiny network of this chapter", 2, 3, 4, 1),
              ("digit classifier", 64, 784, 256, 10),
              ("one wide layer pair", 8, 4096, 16384, 4096)]
    out = ["| network | B | D | H | O | parameters P | forward work | backward work | backward / forward | "
           "forward mode, in forward passes |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b, d, h, o in shapes:
        p, fwd, bwd, _ = mlp_costs(b, d, h, o)
        out.append(f"| {name} | {b} | {d} | {h} | {o} | {p:,} | {fwd:,} | {bwd:,} | {bwd / fwd:.2f} | at least {p:,} |")
    out += ["", "Work counts multiplications and additions of the matrix products only (a multiply-add is two "
                "operations), skipping the gradient with respect to the data `X`, which no parameter needs."]
    return "\n".join(out)


def jacobian_chain_cost():
    """Cost of multiplying a chain of Jacobians from the output side versus from the input side."""
    n = [1000, 1000, 1000, 1000, 1]    # n0 ... n4: many inputs, one scalar output
    out = [f"A chain of four maps with widths {n}. Its Jacobian is the product J = J4 J3 J2 J1, where J1 is "
           f"[{n[1]} x {n[0]}], J2 is [{n[2]} x {n[1]}], J3 is [{n[3]} x {n[2]}] and J4 is [{n[4]} x {n[3]}]. "
           "Three matrix products are needed whichever way the brackets go.", "",
           "| bracketing | what each product carries | multiply-adds |", "|---|---|---:|"]
    # from the output side: every intermediate product has n4 = 1 row
    rev = sum(n[4] * n[j] * n[j - 1] for j in range(1, 4))
    # from the input side: every intermediate product has n0 columns
    fwd = sum(n[j + 1] * n[j] * n[0] for j in range(1, 4))
    out.append(f"| ((J4 J3) J2) J1, output side first (reverse mode) | one row "
               f"throughout | {rev:,} |")
    out.append(f"| J4 (J3 (J2 J1)), input side first (forward mode) | a matrix with {n[0]} columns throughout "
               f"| {fwd:,} |")
    out += ["", f"The ratio is {fwd / rev:,.0f}, close to the number of inputs ({n[0]}): forward mode drags {n[0]} "
                f"columns through every step where reverse mode drags one row."]
    return "\n".join(out)


# ---------------------------------------------------------------- VJP vs Jacobian


def vjp_vs_jacobian():
    p = initial_params()
    f = forward(p)
    g = backward(p, f)
    i = 0
    a = f["a"][i]
    jac = [[(1.0 - a[r] ** 2) if r == c else 0.0 for c in range(H)] for r in range(H)]
    u = g["d_a"][i]
    via_jac = [sum(u[r] * jac[r][c] for r in range(H)) for c in range(H)]
    via_vjp = [u[c] * (1.0 - a[c] ** 2) for c in range(H)]
    out = [f"Example b=0 of the tiny network. The explicit Jacobian of tanh is a {H} x {H} matrix that is zero off the "
           "diagonal:", "",
           md_matrix("J", jac, _names("out ", H), _names("in ", H)), "",
           f"Gradient arriving from above, u = dA[0] = [{', '.join(f'{v:.4f}' for v in u)}]. Both ways of computing "
           "u^T J:", "",
           "| input index | via full Jacobian (u^T J) | via the rule u * (1 - a^2) |", "|---:|---:|---:|"]
    for c in range(H):
        out.append(f"| {c} | {via_jac[c]:.6f} | {via_vjp[c]:.6f} |")
    out += ["", f"The Jacobian stores {H * H} numbers to say {H} things; the rule stores {H}."]
    return "\n".join(out)


def jacobian_sizes():
    rows = [("tanh on a [B, H] activation", 64, 4096, 4096),
            ("linear layer, w.r.t. its input", 64, 4096, 4096)]
    out = ["| layer (B = 64, D = H = 4096) | entries of the full Jacobian | as float32 | entries of the gradient "
           "it multiplies |", "|---|---:|---:|---:|"]
    n = rows[0][1] * rows[0][2]
    full = n * n
    out.append(f"| {rows[0][0]} | {full:,} | {full * 4 / 2 ** 30:,.0f} GiB | {n:,} |")
    out.append(f"| {rows[1][0]} | {full:,} | {full * 4 / 2 ** 30:,.0f} GiB | {n:,} |")
    out += ["", "Both are the same size because the Jacobian of the whole batch is [B H, B D]; most of it is zero "
                "(diagonal for tanh, block diagonal for the linear layer), and the vector-Jacobian product never "
                "stores the zeros."]
    return "\n".join(out)


# --------------------------------------------------------------------- memory


def saved_memory():
    width, depth, bytes_per = 4096, 32, 4
    params = depth * width * width * bytes_per
    out = [f"A stack of {depth} linear layers of width {width} with a tanh after each, float32. "
           f"Parameters: {params / 2 ** 30:.2f} GiB, whatever the batch size. "
           "Saved for the backward pass: the layer input and the tanh output, two tensors of [B, width] per layer.", "",
           "| batch B | saved activations | parameters | saved / parameters |", "|---:|---:|---:|---:|"]
    for b in (1, 16, 256, 4096):
        saved = depth * 2 * b * width * bytes_per
        out.append(f"| {b} | {saved / 2 ** 20:,.1f} MiB | {params / 2 ** 20:,.0f} MiB | {saved / params:.3f} |")
    return "\n".join(out)


def saved_per_op():
    rows = [
        ("linear  Y = X W^T + b", "X (for dW); W (already a parameter)", "nothing"),
        ("tanh  A = tanh(Z)", "A (the local derivative is 1 - A^2)", "Z"),
        ("ReLU  A = max(0, Z)", "which entries of Z were positive (one bit each)", "the values of Z"),
        ("mean squared error", "E = Y - T (or Y and T)", "nothing else"),
    ]
    out = ["| operation | the backward pass reads | it does not need |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)
