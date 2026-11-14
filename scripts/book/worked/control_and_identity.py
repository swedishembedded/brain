# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Control, Identity and Adapters".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: control_and_identity.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. The earlier worked modules are imported, not copied:
`denoiser_architectures` (the SDXL U-Net counted from its configuration, here also cut after its middle block), `text_conditioning`
(the work of one FLUX.1 pass) and `stable` (the digits of trained numbers).

The first half is a small denoiser the reader can build, with a frozen base and two adapters that are added to it:

* a one-dimensional U-Net (4 positions, two levels, a skip stack of three entries, two cross-attention sites on 3 text rows);
* a trainable copy of its down and middle blocks, a conditioning embedder and zero-initialised 1 x 1 convolutions, whose outputs
  are added to the skip copies and to the middle (a ControlNet in miniature);
* an identity branch: a one-layer Perceiver resampler that turns a 4-number identity embedding into 2 tokens, and a second pair of
  key and value projections at each cross-attention site whose result is added to the text attention's context.

Both adapters are trained for a few steps by gradient descent, on gradients from a scalar reverse-mode tape that is checked against
central differences. The second half counts brain's real adapters from their configurations: the SDXL ControlNet (parameters,
tensors, operations, the bytes of its residuals), PuLID's cross-attentions on FLUX.1 and InstantID's resampler and 70 SDXL sites.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. The
trained numbers are printed with the digits that the run and four copies of it with one starting weight changed by the nudges of
`stable.py` share. The forward numbers of untrained networks depend on the library `tanh`, `exp` and `sqrt`, and another
implementation agrees with them to the rounding of those functions.

Swedish Embedded AB implements solutions for image generators that are steered by a pose, an edge map or a particular person's face,
with the extra weights counted and the injection points checked one by one. If your team needs expertise in ControlNet and identity
adapters, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng
import denoiser_architectures as da
import diffusion
import stable
import text_conditioning as tc

_table, _seed = diffusion._table, diffusion._seed


def _int(n):
    return f"{n:,}"


def _f(x, d=6):
    s = f"{x:.{d}f}"
    return "0." + "0" * d if float(s) == 0.0 else s


def _gf(x, d=1):
    """A number of GFLOP in math mode, so that the number is not written next to its unit (the units gate of the book)."""
    return "$" + f"{x:,.{d}f}".replace(",", "{,}") + "$ GFLOP"


def _shared(vals, d):
    """The digits of vals[0] that every nudged run shares (stable.text), written with d decimals."""
    return stable.text(vals[0], vals[1:], lambda x: f"{x:.{d}f}")


# ------------------------------------------------------------------ a tape for the gradients

class Var:
    """A scalar on a reverse-mode tape: a value, the values of its partial derivatives with respect to its parents, and the gradient
    that backward() accumulates. Only the operations the toy needs: +, -, *, /, tanh and exp."""
    __slots__ = ("v", "g", "p", "d")

    def __init__(self, v, p=(), d=()):
        self.v, self.g, self.p, self.d = v, 0.0, p, d

    def __add__(self, o):
        if isinstance(o, Var):
            return Var(self.v + o.v, (self, o), (1.0, 1.0))
        return Var(self.v + o, (self,), (1.0,))

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Var):
            return Var(self.v - o.v, (self, o), (1.0, -1.0))
        return Var(self.v - o, (self,), (1.0,))

    def __rsub__(self, o):
        return Var(o - self.v, (self,), (-1.0,))

    def __mul__(self, o):
        if isinstance(o, Var):
            return Var(self.v * o.v, (self, o), (o.v, self.v))
        return Var(self.v * o, (self,), (o,))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Var):
            return Var(self.v / o.v, (self, o), (1.0 / o.v, -self.v / (o.v * o.v)))
        return Var(self.v / o, (self,), (1.0 / o,))

    def tanh(self):
        t = math.tanh(self.v)
        return Var(t, (self,), (1.0 - t * t,))

    def exp(self):
        e = math.exp(self.v)
        return Var(e, (self,), (e,))

    def backward(self):
        """Set every ancestor's g to the derivative of this scalar with respect to it: parents after children, in reverse order."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((q, False) for q in node.p if id(q) not in seen)
        self.g = 1.0
        for node in reversed(order):
            for q, dq in zip(node.p, node.d):
                q.g += node.g * dq


def _val(x):
    return x.v if isinstance(x, Var) else x


def _tanh(x):
    return x.tanh() if isinstance(x, Var) else math.tanh(x)


def _exp(x):
    return x.exp() if isinstance(x, Var) else math.exp(x)


def _leafify(t):
    return [_leafify(u) for u in t] if isinstance(t, list) else Var(t)


def _leaves(t):
    if isinstance(t, Var):
        yield t
    else:
        for u in t:
            yield from _leaves(u)


def _scalars(t):
    """The numbers of a nested list in row-major order."""
    if isinstance(t, list):
        for u in t:
            yield from _scalars(u)
    else:
        yield t


def _copy(t):
    return [_copy(u) for u in t] if isinstance(t, list) else t


# ------------------------------------------------------------------ the toy denoiser

def _lin(W, x, b=None):
    """y = W x + b: for each output row, the bias first and then the products with x[0], x[1], ... added one at a time in that order."""
    out = []
    for i, row in enumerate(W):
        s = b[i] if b is not None else 0.0
        for w, xi in zip(row, x):
            s = s + w * xi
        out.append(s)
    return out


def _sum(items):
    items = list(items)
    s = items[0]
    for t in items[1:]:
        s = s + t
    return s


def _dot(a, b):
    return _sum(x * y for x, y in zip(a, b))


def _softmax(s):
    """exp(s - max) divided by the running sum of those exponentials, added from the first to the last."""
    m = max(_val(t) for t in s)
    e = [_exp(t - m) for t in s]
    tot = _sum(e)
    return [t / tot for t in e]


def _add(h, r):
    return [[a + b for a, b in zip(ht, rt)] for ht, rt in zip(h, r)]


def _pool(h):
    return [[(a + b) * 0.5 for a, b in zip(h[2 * i], h[2 * i + 1])] for i in range(len(h) // 2)]


def _site(h, E, S, ids=None, id_scale=1.0):
    """One cross-attention site (one head) of width c = len(h[0]) on the text rows E: queries q = Wq h, keys Wk e and values Wv e of
    each text row, scores q.k / sqrt(c), softmax over the rows, context = sum of weights times values. With `ids = (tokens, Kid, Vid)`
    a second attention of the same queries over the identity tokens (its own softmax, its own key and value projections) is added to
    the context, scaled by `id_scale`, before the output projection Wo; the site returns h + Wo context."""
    c = len(h[0])
    inv = 1.0 / math.sqrt(c)
    ks = [_lin(S["k"], e) for e in E]
    vs = [_lin(S["v"], e) for e in E]
    if ids is not None:
        toks, Kid, Vid = ids
        kid = [_lin(Kid, t) for t in toks]
        vid = [_lin(Vid, t) for t in toks]
    out = []
    for ht in h:
        q = _lin(S["q"], ht)
        p = _softmax([_dot(q, k) * inv for k in ks])
        ctx = [_sum(p[j] * vs[j][i] for j in range(len(vs))) for i in range(c)]
        if ids is not None:
            pid = _softmax([_dot(q, k) * inv for k in kid])
            cid = [_sum(pid[j] * vid[j][i] for j in range(len(vid))) for i in range(c)]
            ctx = [a + id_scale * b for a, b in zip(ctx, cid)]
        o = _lin(S["o"], ctx)
        out.append([a + b for a, b in zip(ht, o)])
    return out


def _down(P, x, extra=None, hidden_add=None):
    """The down path and middle block. Returns the three skip entries (the input convolution, the block at 4 positions, the block at
    2 positions) and the middle activation before its attention site. `extra` is added to the input convolution's output (the
    conditioning embedding of the control copy). `hidden_add` adds three residuals to the running hidden state (and so to the skips)
    at the three points: the wrong placement that Section 'Where the residuals go in' counts."""
    a = [_lin(P["Win"], xt, P["bin"]) for xt in x]
    if extra is not None:
        a = _add(a, extra)
    if hidden_add is not None:
        a = _add(a, hidden_add[0])
    r0 = [[ai + _tanh(z) for ai, z in zip(at, _lin(P["W0"], at, P["b0"]))] for at in a]
    if hidden_add is not None:
        r0 = _add(r0, hidden_add[1])
    p = _pool(r0)
    r1 = [[_tanh(z) for z in _lin(P["W1"], pt, P["b1"])] for pt in p]
    if hidden_add is not None:
        r1 = _add(r1, hidden_add[2])
    m = [[ri + _tanh(z) for ri, z in zip(rt, _lin(P["Wm"], rt, P["bm"]))] for rt in r1]
    return a, r0, r1, m


def _site_weights(P, k):
    return dict(q=P[f"q{k}"], k=P[f"k{k}"], v=P[f"v{k}"], o=P[f"o{k}"])


def unet(P, x, E, res=None, ids=None, id_scale=1.0, hidden_add=None, taps=None):
    """The base denoiser on x (4 positions of 2 numbers) and the text rows E. `res` are the four control residuals, added to the
    three skip copies and to the middle activation after its site (never to the running hidden state); `ids` is
    {"tok": tokens, "K": [Kid0, Kid1], "V": [Vid0, Vid1]} for the identity branch."""
    s0, s1, s2, m = _down(P, x, hidden_add=hidden_add)
    pick = (lambda k: None) if ids is None else (lambda k: (ids["tok"], ids["K"][k], ids["V"][k]))
    m = _site(m, E, _site_weights(P, 0), pick(0), id_scale)
    if taps is not None:
        taps["mid"] = m
    if res is not None:
        s0, s1, s2, m = _add(s0, res[0]), _add(s1, res[1]), _add(s2, res[2]), _add(m, res[3])
    u1 = [[_tanh(z) for z in _lin(P["U1"], mt + st, P["c1"])] for mt, st in zip(m, s2)]
    up = [u1[0], u1[0], u1[1], u1[1]]
    u2 = [[_tanh(z) for z in _lin(P["U2"], ut + st, P["c2"])] for ut, st in zip(up, s1)]
    u3 = [[_tanh(z) for z in _lin(P["U3"], ut + st, P["c3"])] for ut, st in zip(u2, s0)]
    u3 = _site(u3, E, _site_weights(P, 1), pick(1), id_scale)
    return [_lin(P["Wout"], ut, P["bout"]) for ut in u3]


BASE_SHAPES = [
    ("Win", (3, 2)), ("bin", (3,)), ("W0", (3, 3)), ("b0", (3,)), ("W1", (4, 3)), ("b1", (4,)), ("Wm", (4, 4)), ("bm", (4,)),
    ("q0", (4, 4)), ("k0", (4, 3)), ("v0", (4, 3)), ("o0", (4, 4)),
    ("U1", (4, 8)), ("c1", (4,)), ("U2", (3, 7)), ("c2", (3,)), ("U3", (3, 6)), ("c3", (3,)),
    ("q1", (3, 3)), ("k1", (3, 3)), ("v1", (3, 3)), ("o1", (3, 3)),
    ("Wout", (2, 3)), ("bout", (2,)),
]
BASE_ROLE = {"Win": "input convolution", "W0": "block at 4 positions", "W1": "block at 2 positions (3 to 4 channels)",
             "Wm": "middle block", "q0": "site 0 queries", "k0": "site 0 keys", "v0": "site 0 values", "o0": "site 0 output",
             "U1": "up block, joins skip 2", "U2": "up block, joins skip 1", "U3": "up block, joins skip 0",
             "q1": "site 1 queries", "k1": "site 1 keys", "v1": "site 1 values", "o1": "site 1 output", "Wout": "output layer"}
BASE = "control_and_identity.base"
DATA = "control_and_identity.data"
CONTROL = "control_and_identity.control"
RANDOM_Z = "control_and_identity.randomz"
HELDOUT = "control_and_identity.heldout"
IDENTITY = "control_and_identity.identity"
FACES = "control_and_identity.faces"
IDATA = "control_and_identity.idata"
FACES_HELDOUT = "control_and_identity.faces_heldout"
N_TRAIN, N_HELD = 6, 4
N_FACES, N_FACES_HELD = 16, 6


def _draw(g, shape):
    """A bias vector is 0.1 times a normal draw per entry; a matrix of `rows` x `cols` is a normal draw divided by sqrt(cols) per
    entry, rows first and columns fastest."""
    if len(shape) == 1:
        return [0.1 * g.normal() for _ in range(shape[0])]
    r = 1.0 / math.sqrt(shape[1])
    return [[r * g.normal() for _ in range(shape[1])] for _ in range(shape[0])]


@lru_cache(maxsize=None)
def base_params():
    """The base network's tensors in the order of BASE_SHAPES, then the three text rows E of 3 numbers, all from one stream."""
    g = bookrng.generator(BASE)
    P = {n: _draw(g, s) for n, s in BASE_SHAPES}
    E = [[g.normal() for _ in range(3)] for _ in range(3)]
    return P, E


def _sample(g):
    x = [[g.normal() for _ in range(2)] for _ in range(4)]
    c = [float(g.integer(2)) for _ in range(8)]
    return x, c


@lru_cache(maxsize=None)
def control_data(name, n):
    g = bookrng.generator(name)
    xs, cs = [], []
    for _ in range(n):
        x, c = _sample(g)
        xs.append(x)
        cs.append(c)
    return xs, cs


W_SHIFT = (0.8, -0.6)
G_ID = [[0.2, -0.1, 0.0, 0.15], [-0.05, 0.1, 0.2, -0.1]]


def control_target(P, E, x, c):
    """What the control image asks for: the base output plus, at position p, W_SHIFT[ch] times the mean of the two control pixels
    c[2p], c[2p+1] added to channel ch."""
    y = unet(P, x, E)
    return [[y[p][ch] + W_SHIFT[ch] * 0.5 * (c[2 * p] + c[2 * p + 1]) for ch in range(2)] for p in range(4)]


def _mse(out, tgt, n):
    """Squared error summed over the 8 numbers of each of n samples (the caller adds the samples), divided by 8 n."""
    terms = [(a - b) * (a - b) for ot, tt in zip(out, tgt) for a, b in zip(ot, tt)]
    return _sum(terms) * (1.0 / (8 * n))


# ------------------------------------------------------------------ the control branch

COPY = ["Win", "bin", "W0", "b0", "W1", "b1", "Wm", "bm", "q0", "k0", "v0", "o0"]
EMB_SHAPES = [("Ec", (3, 2)), ("bc", (3,)), ("Eo", (3, 3)), ("bo", (3,))]
Z_SHAPES = [("Z0", (3, 3)), ("z0", (3,)), ("Z1", (3, 3)), ("z1", (3,)), ("Z2", (4, 4)), ("z2", (4,)), ("Zm", (4, 4)), ("zm", (4,))]
EMB_NAMES, Z_NAMES = [n for n, _ in EMB_SHAPES], [n for n, _ in Z_SHAPES]
CONTROL_TRAINABLE = EMB_NAMES + COPY + Z_NAMES


def _zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def control_params(P, random_z=False):
    """The copy takes the base's values; the embedder is drawn from CONTROL (Ec, bc, Eo, bo in that order, like any layer); the
    zero-convolutions are zeros, or with `random_z` drawn like any layer from RANDOM_Z in the order Z0, z0, Z1, z1, Z2, z2, Zm, zm."""
    g = bookrng.generator(CONTROL)
    C = {n: _copy(P[n]) for n in COPY}
    for n, s in EMB_SHAPES:
        C[n] = _draw(g, s)
    gz = bookrng.generator(RANDOM_Z)
    for n, s in Z_SHAPES:
        C[n] = _draw(gz, s) if random_z else _zeros(s)
    return C


def embed(C, cimg):
    """The conditioning embedder: for each of the 4 positions the pair of control pixels through tanh(Ec p + bc) and then, with no
    activation, Eo e + bo; 3 numbers per position, added to the input convolution's output in the copy."""
    out = []
    for i in range(4):
        e1 = [_tanh(z) for z in _lin(C["Ec"], [cimg[2 * i], cimg[2 * i + 1]], C["bc"])]
        out.append(_lin(C["Eo"], e1, C["bo"]))
    return out


def control(C, x, E, cimg, scale=1.0):
    """The trainable copy's four residuals: scale times (Z s + z) at each of the skips s0, s1, s2 and the middle activation."""
    s0, s1, s2, m = _down(C, x, extra=embed(C, cimg))
    m = _site(m, E, _site_weights(C, 0))
    return [[[scale * a for a in _lin(C[Zn], t, C[zn])] for t in sk]
            for sk, Zn, zn in ((s0, "Z0", "z0"), (s1, "Z1", "z1"), (s2, "Z2", "z2"), (m, "Zm", "zm"))]


def _control_loss(C, xs, cs, tg, scale=1.0, P=None, E=None):
    """The mean squared error of the controlled network on the samples, with C's entries numbers or tape variables."""
    P, E = base_params() if P is None else (P, E)
    losses = [_mse(unet(P, x, E, res=control(C, x, E, c, scale)), t, len(xs)) for x, c, t in zip(xs, cs, tg)]
    return _sum(losses)


def _control_targets(xs, cs):
    P, E = base_params()
    return [control_target(P, E, x, c) for x, c in zip(xs, cs)]


def _flat_names(C, names):
    return [(n, i) for n in names for i, _ in enumerate(_scalars(C[n]))]


def _set_scalar(t, i, value):
    """Set the i-th number (row-major) of a nested list in place."""
    if isinstance(t[0], list):
        w = len(t[0])
        t[i // w][i % w] = value
    else:
        t[i] = value


def _control_grads(C, xs, cs, tg):
    """Loss and the gradient of every trainable number (in CONTROL_TRAINABLE order) by the tape."""
    tr = {n: _leafify(C[n]) for n in CONTROL_TRAINABLE}
    full = dict(C)
    full.update(tr)
    loss = _control_loss(full, xs, cs, tg)
    loss.backward()
    return loss.v, {n: [v.g for v in _leaves(tr[n])] for n in CONTROL_TRAINABLE}


def _group_norm(grads, names):
    return math.sqrt(sum(g * g for n in names for g in grads[n]))


def _sgd(C, grads, lr):
    for n in CONTROL_TRAINABLE:
        for i, g in enumerate(grads[n]):
            _set_scalar(C[n], i, list(_scalars(C[n]))[i] - lr * g)


CONTROL_STEPS, CONTROL_LR = 100, 1.0
LOG_STEPS = (0, 1, 2, 3, 5, 10, 25, 50, 100)


@lru_cache(maxsize=None)
def control_run(nudge=0.0, random_z=False):
    """Train the control branch for CONTROL_STEPS steps of full-batch gradient descent at rate CONTROL_LR on the N_TRAIN samples.
    `nudge` is added to the embedder's first weight Ec[0][0] before the first step. Returns (history, trained tensors) where the
    history lists, per step 0 .. CONTROL_STEPS, the loss before the update and the gradient norms of the three groups."""
    P, E = base_params()
    xs, cs = control_data(DATA, N_TRAIN)
    tg = _control_targets(xs, cs)
    C = control_params(P, random_z)
    C["Ec"][0][0] += nudge
    hist = []
    for s in range(CONTROL_STEPS + 1):
        loss, grads = _control_grads(C, xs, cs, tg)
        hist.append((loss, _group_norm(grads, Z_NAMES), _group_norm(grads, COPY), _group_norm(grads, EMB_NAMES)))
        if s < CONTROL_STEPS:
            _sgd(C, grads, CONTROL_LR)
    return hist, C


def _runs(random_z=False):
    return [control_run(d, random_z) for d in (0.0, *stable.NUDGES)]


def _control_loss_float(C, xs, cs, tg, scale=1.0):
    return _control_loss(C, xs, cs, tg, scale)


def _grad_check(C, xs, cs, tg, h=1e-5):
    """Largest |tape gradient - central difference| over every trainable number of the control branch; h is the step."""
    _, grads = _control_grads(C, xs, cs, tg)
    worst, count = 0.0, 0
    for n in CONTROL_TRAINABLE:
        for i, a in enumerate(grads[n]):
            old = list(_scalars(C[n]))[i]
            _set_scalar(C[n], i, old + h)
            up = _control_loss_float(C, xs, cs, tg)
            _set_scalar(C[n], i, old - h)
            dn = _control_loss_float(C, xs, cs, tg)
            _set_scalar(C[n], i, old)
            worst = max(worst, abs(a - (up - dn) / (2 * h)))
            count += 1
    return worst, count


# ------------------------------------------------------------------ the opening puzzle

def zero_layer():
    lr, x, t = 0.1, [1.0, 2.0], [0.5, -1.0]
    W = [[0.0, 0.0], [0.0, 0.0]]
    b = [0.0, 0.0]
    rows = []
    for k in range(3):
        y = [W[i][0] * x[0] + W[i][1] * x[1] + b[i] for i in range(2)]
        dy = [y[i] - t[i] for i in range(2)]
        dW = [[dy[i] * x[j] for j in range(2)] for i in range(2)]
        dx = [W[0][j] * dy[0] + W[1][j] * dy[1] for j in range(2)]
        rows.append([k, f"{W[0][0]:g}, {W[0][1]:g}; {W[1][0]:g}, {W[1][1]:g}", f"{y[0]:g}, {y[1]:g}", f"{dy[0]:g}, {dy[1]:g}",
                     f"{dW[0][0]:g}, {dW[0][1]:g}; {dW[1][0]:g}, {dW[1][1]:g}", f"{dx[0]:.5g}, {dx[1]:.5g}"])
        W = [[W[i][j] - lr * dW[i][j] for j in range(2)] for i in range(2)]
        b = [b[i] - lr * dy[i] for i in range(2)]
    return ("A linear layer y = W x + b with every weight and bias equal to 0, input x = (1, 2), target t = (0.5, -1), loss "
            "(1/2) sum (y - t)^2 so that dy = y - t, gradient of the weights dW = dy x^T, gradient of the input dx = W^T dy, and "
            f"plain gradient descent at rate {lr:g} on W and b (W and b are written row by row, rows separated by semicolons).\n\n"
            + _table(["step", "W", "y", "dy", "dW", "dx"], rows, "lllrll"))


# ------------------------------------------------------------------ the toy, specified

def toy_setup():
    P, E = base_params()
    xs, cs = control_data(DATA, N_TRAIN)
    rows, total = [], 0
    for n, s in BASE_SHAPES:
        cnt = s[0] * (s[1] if len(s) > 1 else 1)
        total += cnt
        shape = "x".join(str(d) for d in s)
        recipe = "0.1 N" if len(s) == 1 else f"N / sqrt({s[1]})"
        rows.append([f"`{n}`", BASE_ROLE.get(n, "bias" if len(s) == 1 else ""), shape, cnt, recipe])
    ctl = [(n, s) for n, s in EMB_SHAPES] + [(n, s) for n, s in Z_SHAPES]
    crow = []
    n_copy = sum(len(list(_scalars(P[n]))) for n in COPY)
    crow.append(["copy of 12 tensors", "`Win` to `bm` and site 0 (`q0` to `o0`), copied from the base", n_copy, "the base's values"])
    ce = sum(len(list(_scalars(_zeros(s)))) for n, s in EMB_SHAPES)
    crow.append(["`Ec` 3x2, `bc` 3, `Eo` 3x3, `bo` 3", "conditioning embedder", ce, f"drawn in this order, as the base's tensors, from {_seed(CONTROL)}"])
    cz = sum(len(list(_scalars(_zeros(s)))) for n, s in Z_SHAPES)
    crow.append(["`Z0`..`Zm`, `z0`..`zm`", "four zero-convolutions", cz, "all zeros"])
    sample = unet(P, xs[0], E)
    x0 = "; ".join(", ".join(f"{v:.6f}" for v in r) for r in xs[0])
    e0 = "; ".join(", ".join(f"{v:.6f}" for v in r) for r in E)
    y0 = "; ".join(", ".join(f"{v:.6f}" for v in r) for r in sample)
    return (
        "The toy denoiser. Input and output are 4 positions of 2 numbers. The base network, in the order in which its tensors are "
        f"drawn from {_seed(BASE)} (a bias is 0.1 times one normal draw per entry; a matrix is one normal draw divided by the "
        "square root of its number of columns per entry, row by row; N is `normal()`); after the 24 tensors, three text rows of 3 "
        "numbers are drawn, 9 normal draws with no scale. Every sum (a matrix-vector product, which starts from its bias, an attention score, a "
        "softmax denominator, the loss) is accumulated from the first term to the last, and a softmax subtracts the largest score of its row "
        "before the exponentials; a layer normalisation (the resampler's) subtracts the mean and divides by sqrt(variance + 1e-5), the variance having the width as its divisor.\n\n"
        + _table(["tensor", "role", "shape", "numbers", "recipe"], rows, "lllrl")
        + f"\n\nThe base has {_int(total)} numbers. The control branch has "
        f"{_int(n_copy + ce + cz)} more:\n\n"
        + _table(["tensors", "what", "numbers", "start"], crow, "llrl")
        + "\n\nThe training set is "
        f"{N_TRAIN} samples from {_seed(DATA)}, each drawn as 8 normal draws (the input, position by position) and then 8 draws of "
        "`integer(2)` (the control image: 8 pixels, each 0 or 1). The first sample's input, the text rows and the base network's "
        "output on it, as anchors:\n\n"
        f"| x (4 positions) | {x0} |\n|---|---|\n| text rows E | {e0} |\n| base output | {y0} |\n| control pixels | "
        + ", ".join(f"{v:g}" for v in cs[0]) + " |"
    )


# ------------------------------------------------------------------ the first step leaves the base alone

def zero_start():
    P, E = base_params()
    xs, cs = control_data(DATA, N_TRAIN)
    tg = _control_targets(xs, cs)
    C0 = control_params(P)
    # (a) exact equality of the outputs and the residuals
    worst, nonzero = 0.0, 0
    for x, c in zip(xs, cs):
        res = control(C0, x, E, c)
        nonzero += sum(1 for r in res for row in r for v in row if v != 0.0)
        a, b = unet(P, x, E), unet(P, x, E, res=res)
        worst = max(worst, max(abs(p - q) for ra, rb in zip(a, b) for p, q in zip(ra, rb)))
    loss0, g0 = _control_grads(C0, xs, cs, tg)
    # (b) one step later
    C1 = _copy(C0)
    _sgd(C1, g0, CONTROL_LR)
    loss1, g1 = _control_grads(C1, xs, cs, tg)
    # (c) the gradient check, at step 0 and at step 1
    w0, n0 = _grad_check(C0, xs, cs, tg)
    w1, n1 = _grad_check(C1, xs, cs, tg)
    rows = []
    for label, loss, g in (("0", loss0, g0), ("1", loss1, g1)):
        rows.append([label, _f(loss), _f(_group_norm(g, Z_NAMES)), _f(_group_norm(g, COPY)), _f(_group_norm(g, EMB_NAMES))])
    base_loss = sum(_val(_mse(unet(P, x, E), t, N_TRAIN)) for x, t in zip(xs, tg))
    return (
        f"With the four zero-convolutions at 0 (and the other 134 control numbers as specified), on the {N_TRAIN} training samples: the "
        f"control residuals have {nonzero} nonzero entries out of {sum(len(list(_scalars(r))) for r in control(C0, xs[0], E, cs[0])) * N_TRAIN}; "
        f"the largest difference between the controlled network's output and the base network's is {worst:g} "
        f"(the outputs are equal as floating-point numbers); the loss of the base network on the control task is {_f(base_loss)}.\n\n"
        "The gradient norms (the square root of the sum of squares of every gradient entry of a group of tensors) at step 0, and after "
        f"one step of gradient descent at rate {CONTROL_LR:g}:\n\n"
        + _table(["step", "loss", "zero-convs", "trainable copy", "embedder"], rows)
        + f"\n\nGradient check (central differences with step 1e-5 against the tape, every one of the "
        f"{n0} trainable numbers): the largest absolute difference is below 1e-7 at step 0 "
        f"({'passed' if w0 < 1e-7 else 'FAILED'}) and below 1e-7 at step 1 ({'passed' if w1 < 1e-7 else 'FAILED'}); "
        f"at step 0 the {n0 - sum(len(list(_scalars(C0[n]))) for n in Z_NAMES)} numbers outside the zero-convs have a tape gradient of exactly "
        f"{'zero' if _group_norm(g0, COPY) == 0.0 and _group_norm(g0, EMB_NAMES) == 0.0 else 'not zero'}."
    )


# ------------------------------------------------------------------ where the residuals go in

def _pattern(k, shape):
    """A fixed, order-sensitive residual for injection point k: entry (position, channel) is 0.05 ((3 p + 5 ch + 7 k) mod 11 - 5)."""
    return [[0.05 * (((3 * p + 5 * ch + 7 * k) % 11) - 5) for ch in range(shape[1])] for p in range(shape[0])]


POINTS = [("down.0", 4, 3), ("down.1", 4, 3), ("down.2", 2, 4), ("mid", 2, 4)]   # name, positions, channels


def placement():
    P, E = base_params()
    xs, _ = control_data(DATA, N_TRAIN)
    x = xs[0]
    taps = {}
    base = unet(P, x, E, taps=taps)
    base_mid = taps["mid"]

    def gap(a, b):
        return max(abs(p - q) for ra, rb in zip(a, b) for u, v in zip(ra, rb) for p, q in zip(u, v)) if isinstance(a[0][0], list) \
            else max(abs(p - q) for ra, rb in zip(a, b) for p, q in zip(ra, rb))

    rows = []
    for k, (name, npos, nch) in enumerate(POINTS):
        res = [[[0.0] * c for _ in range(n)] for _, n, c in POINTS]
        res[k] = _pattern(k, (npos, nch))
        t_skip = {}
        out = unet(P, x, E, res=res, taps=t_skip)
        mid_skip = gap(t_skip["mid"], base_mid)
        out_gap = gap(out, base)
        if k < 3:
            hid = [[[0.0] * c for _ in range(n)] for _, n, c in POINTS[:3]]
            hid[k] = res[k]
            t_h = {}
            out_h = unet(P, x, E, hidden_add=hid, taps=t_h)
            rows.append([name, f"{nch} x {npos}", _f(out_gap, 6), _f(mid_skip, 6), _f(gap(t_h["mid"], base_mid), 6), _f(gap(out_h, base), 6)])
        else:
            rows.append([name, f"{nch} x {npos}", _f(out_gap, 6), _f(mid_skip, 6), "(added after the middle)", "-"])
    sw = [[[0.0] * c for _ in range(n)] for _, n, c in POINTS]
    sw[0], sw[1] = _pattern(0, (4, 3)), _pattern(1, (4, 3))
    ok = unet(P, x, E, res=sw)
    swapped = list(sw)
    swapped[0], swapped[1] = sw[1], sw[0]
    bad = unet(P, x, E, res=swapped)
    full = [_pattern(k, (n, c)) for k, (_, n, c) in enumerate(POINTS)]
    full_ok = unet(P, x, E, res=full)
    full_swapped = list(full)
    full_swapped[0], full_swapped[1] = full[1], full[0]
    full_bad = unet(P, x, E, res=full_swapped)
    return (
        "One input of the first training sample, one residual at a time. The residual of point k has entry (position p, channel c) "
        "equal to 0.05 (((3 p + 5 c + 7 k) mod 11) - 5), the other three are 0. The first two columns are the point and the shape "
        "of its residual (channels x positions). \"skip only\": the residual is added to the skip copy that the up path reads, as "
        "the chapter's network does; the third column is the largest change of the network's output and the fourth the largest "
        "change of the middle activation (the running hidden state, read before the middle residual). \"also hidden\": the same "
        "residual is added to the running hidden state too (so it reaches the later down blocks and the middle block): the fifth "
        "column is the change of the middle activation and the sixth the change of the output.\n\n"
        + _table(["point", "shape", "output, skip only", "middle, skip only", "middle, also hidden", "output, also hidden"], rows,
                 "llrrrr")
        + f"\n\nExchanging the residuals of down.0 and down.1, which have the same shape (3 channels x 4 positions), runs without a shape "
        f"error and moves the output by {_f(gap(bad, ok), 6)} (largest change) when only those two residuals are present, and by "
        f"{_f(gap(full_bad, full_ok), 6)} when all four are present (each with its own pattern); exchanging the residuals of down.1 and down.2 "
        f"({3 * 4} and {4 * 2} numbers) fails the element-count check before anything runs."
    )


# ------------------------------------------------------------------ training the control branch

def control_training():
    runs, zruns = _runs(False), _runs(True)
    rows = []
    for s in LOG_STEPS:
        rows.append([s, _shared([r[0][s][0] for r in runs], 6), _shared([r[0][s][1] for r in runs], 6),
                     _shared([r[0][s][2] for r in runs], 6), _shared([r[0][s][3] for r in runs], 6),
                     _shared([r[0][s][0] for r in zruns], 6)])
    return (
        f"Full-batch gradient descent at rate {CONTROL_LR:g} for {CONTROL_STEPS} steps on the {N_TRAIN} training samples, in two runs that "
        f"differ only in the start of the four zero-convolutions: all zeros, or drawn like every other layer from {_seed(RANDOM_Z)} (Z0, "
        "z0, Z1, z1, Z2, z2, Zm, zm in that order, a matrix as N / sqrt(columns) and a bias as 0.1 N). Loss is the mean squared "
        "error over the 8 numbers of a sample and the samples; the norms are those of the zero-start run before each update. "
        "The embedder's first weight is the one the nudges change (it multiplies a control pixel that is 1 in the first sample).\n\n"
        + _table(["step", "loss, zero start", "grad zero-convs", "grad copy", "grad embedder", "loss, random start"], rows)
        + "\n\n" + (stable.RULE if any("about " in c or "unstable" in c for r in rows for c in map(str, r)) else
                    "The digits shown are shared by the run and by four copies whose embedder first weight is changed by +1e-12, -1e-12, "
                    "+3e-12 and -3e-12 (the nudge rule introduced with the first trained tables of Chapter 2).")
    )


def scale_and_heldout():
    P, E = base_params()
    xs, cs = control_data(DATA, N_TRAIN)
    tg = _control_targets(xs, cs)
    hx, hc = control_data(HELDOUT, N_HELD)
    ht = _control_targets(hx, hc)
    runs = _runs(False)
    rows = []
    for sc in (0.0, 0.5, 1.0, 1.5, 2.0):
        tr = [_val(_control_loss(r[1], xs, cs, tg, sc)) for r in runs]
        he = [_val(_control_loss(r[1], hx, hc, ht, sc)) for r in runs]
        rows.append([f"{sc:g}", _shared(tr, 6), _shared(he, 6)])
    return (
        f"The trained control branch of the zero-start run, with its residuals multiplied by the conditioning scale s (a pure "
        f"multiplication of every zero-convolution's output) at evaluation. The first loss is on the {N_TRAIN} training samples; the "
        f"second on {N_HELD} samples from {_seed(HELDOUT)} that the branch has not seen (drawn like the training set: 8 normal draws "
        "and 8 `integer(2)` draws per sample).\n\n"
        + _table(["s", "loss, training samples", "loss, held-out samples"], rows)
    )


# ------------------------------------------------------------------ the identity branch

ID_SHAPES = [("Win", (3, 4)), ("bin", (3,)), ("Lat", (2, 3)), ("Wq", (3, 3)), ("Wk", (3, 3)), ("Wv", (3, 3)), ("Wo", (3, 3)), ("Wp", (3, 3))]


@lru_cache(maxsize=None)
def resampler_params():
    """The resampler's tensors from IDENTITY in the order of ID_SHAPES (the two learned latent rows are plain normal draws), then
    the key projections of the two sites (4 x 3 and 3 x 3); the value projections start at zero and draw nothing."""
    g = bookrng.generator(IDENTITY)
    R = {}
    for n, s in ID_SHAPES:
        R[n] = [[g.normal() for _ in range(3)] for _ in range(2)] if n == "Lat" else _draw(g, s)
    K = [_draw(g, (4, 3)), _draw(g, (3, 3))]
    V = [_zeros((4, 3)), _zeros((3, 3))]
    return R, K, V


def _ln(v, eps=1e-5):
    """LayerNorm with no gain or offset: (v - mean) / sqrt(variance + eps) with the variance divided by the width."""
    m = sum(v) / len(v)
    var = sum((a - m) ** 2 for a in v) / len(v)
    r = 1.0 / math.sqrt(var + eps)
    return [(a - m) * r for a in v]


def resample(R, e):
    """An identity embedding (4 numbers) to 2 tokens of 3 numbers. The embedding is projected to one row x = Win e + bin; the key
    and value input has three rows, LN(x) then LN(latent 0) then LN(latent 1); each latent queries it (query Wq LN(latent), keys Wk
    and values Wv of the three rows, scores divided by sqrt(3)); the token is Wp (latent + Wo attention)."""
    x = _lin(R["Win"], e, R["bin"])
    kv_in = [_ln(x)] + [_ln(l) for l in R["Lat"]]
    ks = [_lin(R["Wk"], r) for r in kv_in]
    vs = [_lin(R["Wv"], r) for r in kv_in]
    toks = []
    for lat in R["Lat"]:
        q = _lin(R["Wq"], _ln(lat))
        p = _softmax([_dot(q, k) / math.sqrt(3) for k in ks])
        att = [_sum(p[j] * vs[j][i] for j in range(3)) for i in range(3)]
        o = _lin(R["Wo"], att)
        toks.append(_lin(R["Wp"], [a + b for a, b in zip(lat, o)]))
    return toks


def _norm(v):
    return math.sqrt(sum(a * a for a in v))


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (_norm(a) * _norm(b))


def identity_tokens():
    R, K, V = resampler_params()
    g = bookrng.generator(FACES)
    es = [[2.0 * g.normal() for _ in range(4)] for _ in range(N_FACES)]
    e0, e1 = es[0], es[1]
    flat = lambda toks: [v for t in toks for v in t]  # noqa: E731
    t0, t1 = flat(resample(R, e0)), flat(resample(R, e1))
    unit0 = [a / _norm(e0) for a in e0]
    t0u = flat(resample(R, unit0))
    t0x2 = flat(resample(R, [2 * a for a in e0]))
    rel = lambda a, b: _norm([x - y for x, y in zip(a, b)]) / _norm(a)  # noqa: E731
    rows = [
        ["identity 0, as drawn", _f(_norm(e0), 4), "-", "-", " ".join(f"{v:.4f}" for v in t0)],
        ["identity 0, scaled to length 1", "1.0000", _f(_cos(e0, unit0), 6), _f(rel(t0, t0u), 4), " ".join(f"{v:.4f}" for v in t0u)],
        ["identity 0, doubled", _f(_norm([2 * a for a in e0]), 4), _f(_cos(e0, [2 * a for a in e0]), 6), _f(rel(t0, t0x2), 4),
         " ".join(f"{v:.4f}" for v in t0x2)],
        ["identity 1", _f(_norm(e1), 4), _f(_cos(e0, e1), 6), _f(rel(t0, t1), 4), " ".join(f"{v:.4f}" for v in t1)],
    ]
    return (
        f"The resampler of the toy on identity embeddings of 4 numbers (the identities of the stream {_seed(FACES)}: {N_FACES} of them, "
        "each 4 draws of `normal()` times 2, in order; the first two are used here). Its tensors are drawn from "
        f"{_seed(IDENTITY)} in the order `Win` (3 x 4), `bin`, the two latent rows (6 plain normal draws), `Wq`, `Wk`, `Wv`, `Wo`, "
        "`Wp` (3 x 3 each, drawn as the base's matrices), then the two sites' key projections (4 x 3, 3 x 3). The computation, for "
        "an embedding e: x = `Win` e + `bin`; the key and value input is the three rows LN(x), LN(latent 0), LN(latent 1), where LN "
        "subtracts the mean and divides by sqrt(variance + 1e-5), with no gain or offset; the keys are `Wk` and the values `Wv` of those "
        "rows; token i has the query `Wq` LN(latent i) (the latent itself, not its attention, is normalised for the query), scores "
        "q . k / sqrt(3) with a softmax over the three rows, the attention result a, and the token `Wp` (latent i + `Wo` a), the "
        "unnormalised latent being the residual. "
        f"The key and value input has {1 + len(R['Lat'])} rows (one for the projected embedding, one for each latent), which is one "
        "more than the number of tokens. The columns: the length of the embedding, its cosine with identity 0's embedding, the relative "
        "change of the 6 token numbers from identity 0's tokens, and the 6 numbers (token 0, then token 1).\n\n"
        + _table(["embedding", "length", "cosine with identity 0", "relative change of tokens", "tokens"], rows, "lrrrl")
    )


def injection_variants():
    P, E = base_params()
    R, K, V = resampler_params()
    g = bookrng.generator(FACES)
    e = [2.0 * g.normal() for _ in range(4)]
    toks = resample(R, e)
    xs, _ = control_data(DATA, N_TRAIN)
    x = xs[0]
    # site 0 on the middle activation of the base, with trained-looking nonzero value projections
    _, _, _, m = _down(P, x)
    S0 = _site_weights(P, 0)
    rng = bookrng.generator("control_and_identity.valuefill")
    Vf = [[r / 2 for r in row] for row in _draw(rng, (4, 3))]
    Kf = K[0]
    text_only = _site(m, E, S0)

    def gap(a, b):
        return max(abs(p - q) for ra, rb in zip(a, b) for p, q in zip(ra, rb))

    dec_off = _site(m, E, S0, (toks, Kf, Vf), 0.0)
    dec_on = _site(m, E, S0, (toks, Kf, Vf), 1.0)
    # the scale folded into V: exact up to rounding, folded into K: a different function
    fold = 0.3
    folded = _site(m, E, S0, (toks, Kf, [[fold * a for a in row] for row in Vf]), 1.0)
    scaled = _site(m, E, S0, (toks, Kf, Vf), fold)
    kscaled = _site(m, E, S0, (toks, [[fold * a for a in row] for row in Kf], Vf), 1.0)

    def concat(m_, scale):
        """The ID tokens appended to the text rows, through the site's own key and value projections, one softmax over 5 rows; the
        values of the ID rows are multiplied by `scale`."""
        c = len(m_[0])
        inv = 1.0 / math.sqrt(c)
        rows_ = list(E) + list(toks)
        ks = [_lin(S0["k"], r) for r in rows_]
        vs = [_lin(S0["v"], r) for r in rows_]
        out, mass = [], []
        for ht in m_:
            q = _lin(S0["q"], ht)
            p = _softmax([_dot(q, k) * inv for k in ks])
            ctx = [_sum((p[j] * (scale if j >= len(E) else 1.0)) * vs[j][i] for j in range(len(vs))) for i in range(c)]
            out.append([a + b for a, b in zip(ht, _lin(S0["o"], ctx))])
            mass.append(sum(p[len(E):]))
        return out, mass

    cat_off, mass = concat(m, 0.0)
    cat_on, _ = concat(m, 1.0)
    rows = [
        ["added second attention", _f(gap(dec_off, text_only), 9), _f(gap(dec_on, text_only), 6)],
        ["ID tokens joined to the text rows", _f(gap(cat_off, text_only), 6), _f(gap(cat_on, text_only), 6)],
    ]
    return (
        "Site 0 of the toy (width 4, on the base network's middle activation of the first training sample) with the identity of "
        f"{_seed(FACES)} (the first of its identities). Both branches use the same query, text rows and ID tokens; the added "
        "attention has its own key and value projections (the key projection of the chapter's start, the value projection drawn "
        "from `control_and_identity.valuefill`, 12 plain draws of the base's matrix recipe, halved), the joined variant sends "
        "the ID tokens through the site's text projections. The columns are the largest change of the site's output from the "
        "text-only output when the identity is switched off (the added attention's scale 0; the joined variant's ID values "
        "multiplied by 0) and when it is on (scale 1).\n\n"
        + _table(["how the identity enters", "switched off", "on"], rows)
        + f"\n\nIn the joined variant the 2 ID tokens take between {min(mass):.4f} and {max(mass):.4f} of the attention weight at the 2 "
        f"positions of the site even when their values are zeroed. Scaling the added branch's context by {fold:g} and scaling its "
        f"value projection by {fold:g} give outputs that differ by less than 1e-15 ({'passed' if gap(folded, scaled) < 1e-15 else 'FAILED'}; "
        f"the difference is rounding only); scaling the key projection by {fold:g} instead gives an output that differs by "
        f"{_f(gap(kscaled, dec_on), 6)} from the unscaled scale-1 output, by {_f(gap(kscaled, scaled), 6)} from the 0.3-context "
        f"output and by {_f(gap(kscaled, text_only), 6)} from the text-only output."
    )


@lru_cache(maxsize=None)
def faces_data():
    g = bookrng.generator(FACES)
    es = [[2.0 * g.normal() for _ in range(4)] for _ in range(N_FACES)]
    gx = bookrng.generator(IDATA)
    xs = [[[gx.normal() for _ in range(2)] for _ in range(4)] for _ in range(N_FACES)]
    return xs, es


def id_target(P, E, x, e):
    """The base output plus the same offset G e (2 numbers) at all 4 positions."""
    y = unet(P, x, E)
    off = _lin(G_ID, e)
    return [[y[p][ch] + off[ch] for ch in range(2)] for p in range(4)]


ID_STEPS, ID_LR = 200, 0.3
ID_LOG = (0, 1, 2, 5, 10, 25, 50, 100, 200)


def _id_loss(K, V, toks, xs, tg):
    P, E = base_params()
    return _sum([_mse(unet(P, x, E, ids=dict(tok=tk, K=K, V=V)), t, len(xs)) for x, tk, t in zip(xs, toks, tg)])


@lru_cache(maxsize=None)
def id_run(nudge=0.0):
    """Train the key and value projections of the two sites (the resampler and the base are fixed) for ID_STEPS steps of gradient
    descent at rate ID_LR on N_FACES identities. `nudge` is added to the first site's first key weight Kid0[0][0]. Returns the
    history (loss, key gradient norm, value gradient norm per step) and the trained (K, V)."""
    P, E = base_params()
    R, K0, V0 = resampler_params()
    xs, es = faces_data()
    toks = [resample(R, e) for e in es]
    tg = [id_target(P, E, x, e) for x, e in zip(xs, es)]
    K, V = _copy(K0), _copy(V0)
    K[0][0][0] += nudge
    hist = []
    for s in range(ID_STEPS + 1):
        Kv, Vv = [_leafify(k) for k in K], [_leafify(v) for v in V]
        loss = _id_loss(Kv, Vv, toks, xs, tg)
        loss.backward()
        gk = [v.g for k in Kv for v in _leaves(k)]
        gv = [v.g for k in Vv for v in _leaves(k)]
        hist.append((loss.v, math.sqrt(sum(g * g for g in gk)), math.sqrt(sum(g * g for g in gv))))
        if s < ID_STEPS:
            for t, gl in ((K, gk), (V, gv)):
                i = 0
                for mat in t:
                    for r, row in enumerate(mat):
                        for c in range(len(row)):
                            row[c] -= ID_LR * gl[i]
                            i += 1
    return hist, (K, V)


def _id_grad_check(h=1e-5):
    P, E = base_params()
    R, K0, V0 = resampler_params()
    xs, es = faces_data()
    xs, es = xs[:6], es[:6]
    toks = [resample(R, e) for e in es]
    tg = [id_target(P, E, x, e) for x, e in zip(xs, es)]
    K, V = _copy(K0), [[[0.1 * (i - j) for j in range(3)] for i in range(len(V0[k]))] for k in range(2)]
    Kv, Vv = [_leafify(k) for k in K], [_leafify(v) for v in V]
    loss = _id_loss(Kv, Vv, toks, xs, tg)
    loss.backward()
    tape = [v.g for t in (Kv, Vv) for k in t for v in _leaves(k)]
    worst, i = 0.0, 0
    for t in (K, V):
        for mat in t:
            for r in range(len(mat)):
                for c in range(len(mat[r])):
                    old = mat[r][c]
                    mat[r][c] = old + h
                    up = _val(_id_loss(K, V, toks, xs, tg))
                    mat[r][c] = old - h
                    dn = _val(_id_loss(K, V, toks, xs, tg))
                    mat[r][c] = old
                    worst = max(worst, abs(tape[i] - (up - dn) / (2 * h)))
                    i += 1
    return worst, i


def identity_training():
    runs = [id_run(d) for d in (0.0, *stable.NUDGES)]
    rows = []
    for s in ID_LOG:
        rows.append([s, _shared([r[0][s][0] for r in runs], 6), _shared([r[0][s][1] for r in runs], 6),
                     _shared([r[0][s][2] for r in runs], 6)])
    P, E = base_params()
    R, K0, V0 = resampler_params()
    g = bookrng.generator(FACES_HELDOUT)
    hold = []
    for _ in range(N_FACES_HELD):
        e = [2.0 * g.normal() for _ in range(4)]
        x = [[g.normal() for _ in range(2)] for _ in range(4)]
        hold.append((x, e))
    toks = [resample(R, e) for _, e in hold]
    tg = [id_target(P, E, x, e) for x, e in hold]
    hx = [x for x, _ in hold]
    base_l = sum(_val(_mse(unet(P, x, E), t, N_FACES_HELD)) for x, t in zip(hx, tg))
    after = [_val(_id_loss(r[1][0], r[1][1], toks, hx, tg)) for r in runs]
    worst, count = _id_grad_check()
    return (
        f"Training the key and value projections of the two sites (42 numbers: 4 x 3 and 3 x 3, twice) by full-batch gradient "
        f"descent, with no momentum and no randomness after the data are drawn, at rate {ID_LR:g} for {ID_STEPS} steps on "
        f"{N_FACES} identities with their images, the resampler and the base fixed. The added attention is present at both sites of "
        "the toy: at site 0 on the middle activation and at site 1 on the up path's last activation. At "
        "each site the queries are the site's own `q` of its input, the keys and values are the adapter's key and value projections "
        "of the identity's 2 tokens (the resampler's output for that identity, fixed), the scores are divided by the square root of "
        "the site's width, a softmax runs over the 2 tokens alone, and the result times the identity scale 1 is added to the text "
        "attention's context before the site's output projection `o`. The first site's change therefore reaches the second site "
        "and the output through the whole up path. The value projections "
        "start at zero (the keys at the last two tensors of the stream "
        f"{_seed(IDENTITY)}); the target for identity i is the base network's output on its image (no identity branch) plus the "
        "offset G e_i at all 4 positions, where e_i is the identity's embedding and "
        "G = [[0.2, -0.1, 0.0, 0.15], [-0.05, 0.1, 0.2, -0.1]]. The embeddings are the "
        f"{N_FACES} identities of {_seed(FACES)} (4 draws of `normal()` times 2 each); the images are {N_FACES} blocks of 8 normal "
        f"draws from {_seed(IDATA)}, a block being the input of one identity, position by position (2 numbers a position). The loss "
        f"is the squared error summed over the 8 output numbers of each identity and over the {N_FACES} identities, divided by "
        f"8 x {N_FACES}: the mean over all {8 * N_FACES} numbers, with the identities added in order. The two gradient columns are "
        "the square root of the sum of squares of the 21 key (or 21 value) gradient entries of both sites at that step, before the "
        "update, and every one of the 42 numbers is updated at once. The digits are those shared by the run and four copies whose "
        "first key weight of site 0 is changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the nudge rule introduced with the first trained "
        "tables of Chapter 2).\n\n"
        + _table(["step", "loss", "grad keys", "grad values"], rows)
        + f"\n\nOn {N_FACES_HELD} identities with images from {_seed(FACES_HELDOUT)} (each: 4 draws of `normal()` times 2, then 8 normal "
        f"draws) the loss is {_f(base_l)} without the identity branch and {_shared(after, 6)} with the trained one. Gradient check "
        f"(central differences with step 1e-5 against the tape at a point with nonzero values, every one of the {count} numbers, 6 "
        f"identities): the largest absolute difference is below 1e-7 ({'passed' if worst < 1e-7 else 'FAILED'})."
    )


# ------------------------------------------------------------------ brain's adapters, counted

SDXL_PASS_GFLOP = da.unet_count(da.SDXL_UNET, 128, 128)[0].total_flops() / 1e9
SDXL_PARAMS = da.unet_count(da.SDXL_UNET, 128, 128)[0].total_params()
SDXL_TENSORS = da.unet_count(da.SDXL_UNET, 128, 128)[0].tensors


def sdxl_skip_shapes(side=128):
    """(channels, height, width) of the skip stack's entries, finest first (the input convolution, then per level the resnets and,
    between levels, the stride-2 convolution at the coarser size) and then the middle block's."""
    cfg = da.SDXL_UNET
    out = [(cfg["ch"][0], side, side)]
    h = side
    for i, c in enumerate(cfg["ch"]):
        out += [(c, h, h)] * cfg["layers"]
        if i + 1 < len(cfg["ch"]):
            h //= 2
            out.append((c, h, h))
    out.append((cfg["ch"][-1], h, h))
    return out


EMBEDDER = [(3, 16, 1), (16, 16, 1), (16, 32, 2), (32, 32, 1), (32, 96, 2), (96, 96, 1), (96, 256, 2), (256, 320, 1)]


def controlnet_parts(side=128):
    """The SDXL ControlNet from the configuration: (name, parameters, tensors, operations) of the trainable copy, the conditioning
    embedder (3 x 3 convolutions, [16, 32, 96, 256] widths, three of them stride 2) and the ten zero-convolutions, for a latent of
    side `side` (and a control image of 8 times that)."""
    half, _ = da.unet_count(da.SDXL_UNET, side, side, through_mid=True)
    res = side * 8
    emb_p = emb_f = 0
    for cin, cout, stride in EMBEDDER:
        res //= stride
        emb_p += cin * cout * 9 + cout
        emb_f += 2 * cin * cout * 9 * res * res
    zero_p = zero_f = 0
    shapes = sdxl_skip_shapes(side)
    for c, h, w in shapes:
        zero_p += c * c + c
        zero_f += 2 * c * c * h * w
    return [("trainable copy: clock, input convolution, down blocks, middle block", half.total_params(), half.tensors, half.total_flops()),
            ("conditioning embedder", emb_p, 2 * len(EMBEDDER), emb_f),
            ("ten zero-convolutions", zero_p, 2 * len(shapes), zero_f)]


def sdxl_control_counts():
    parts = controlnet_parts()
    tot_p, tot_t, tot_f = (sum(p[i] for p in parts) for i in (1, 2, 3))
    rows = [[n, _int(p), t, f"{f / 1e9:,.1f}"] for n, p, t, f in parts]
    rows.append(["ControlNet total", _int(tot_p), tot_t, f"{tot_f / 1e9:,.1f}"])
    rows.append(["SDXL U-Net, for comparison", _int(SDXL_PARAMS), SDXL_TENSORS, f"{SDXL_PASS_GFLOP:,.1f}"])
    return (
        "The SDXL ControlNet derived from the configuration of the SDXL U-Net of Chapter 8 (the stages of its parameter and operation "
        "tables from the clock through the middle block, the rows of those tables summed) at a 128 by 128 latent, a 1024 by 1024 control "
        "image of 3 channels and 77 text tokens. The conditioning embedder is eight 3 x 3 convolutions with bias, "
        "3 to 16, 16 to 16, 16 to 32 (stride 2), 32 to 32, 32 to 96 (stride 2), 96 to 96, 96 to 256 (stride 2) and 256 to 320 "
        "channels, the first two at 1024 x 1024, and each of the ten zero-convolutions is a 1 x 1 convolution with bias at the width "
        "and size of its injection point (the table of the next block). Tensors are counted in "
        "brain's layout (a fused query-key-value matrix is one tensor, the feed-forward's gated projection two); a multiply-add is two "
        "operations; norms, activations and softmax are not counted.\n\n"
        + _table(["part", "parameters", "tensors", "GFLOP per evaluation"], rows)
        + f"\n\nThe ControlNet has {tot_p / SDXL_PARAMS * 100:.1f} percent of the U-Net's parameters and {tot_f / 1e9 / SDXL_PASS_GFLOP * 100:.1f} percent "
        f"of the operations of one of its passes. With the 34 spatial-transformer blocks of the copy each stored as three attention "
        f"matrices and one gated projection instead of the fused layout, the tensor count is {tot_t} + 34 = {tot_t + 34}."
    )


def control_traffic():
    shapes = sdxl_skip_shapes(128)
    names = [f"down.{k}" for k in range(len(shapes) - 1)] + ["mid"]
    rows, total = [], 0
    for n, (c, h, w) in zip(names, shapes):
        e = c * h * w
        total += e
        rows.append([n, f"{c} x {h} x {w}", _int(e)])
    rows.append(["total", "", _int(total)])
    parts = controlnet_parts()
    cn_g = sum(p[3] for p in parts) / 1e9
    steps, evals = 30, 60
    return (
        "The ten control residuals of the SDXL ControlNet at a 128 by 128 latent, in the order in which the U-Net takes them (the nine "
        "entries of its skip stack, finest first, and the middle block), as channels x height x width and as numbers.\n\n"
        + _table(["injection point", "shape", "numbers"], rows, "llr")
        + f"\n\nIn 32-bit floats the ten residuals are {total * 4 / 2 ** 20:,.1f} MiB. The ControlNet of brain returns them to the host and the "
        f"U-Net writes them back, so every evaluation moves {2 * total * 4 / 2 ** 20:,.1f} MiB across that boundary in addition to the "
        f"{total * 4 / 2 ** 20:,.1f} MiB the zero-convolutions write. A generation of {steps} steps with guidance evaluates the denoiser "
        f"{evals} times, and each evaluation is a ControlNet pass of {_gf(cn_g)} and a U-Net pass of {_gf(SDXL_PASS_GFLOP)}: "
        f"{_gf(evals * cn_g, 0)} and {_gf(evals * SDXL_PASS_GFLOP, 0)}."
    )


def sdxl_attn_sites(side=128):
    """The cross-attention sites of the SDXL U-Net in graph order: (width, tokens) for each, from the down path (levels with
    attention, `layers` resnets each followed by `tf` blocks), the middle block (`tf` of the coarsest level) and the up path
    (`layers + 1` resnets a level)."""
    cfg = da.SDXL_UNET
    L = len(cfg["ch"])
    sites = []
    for i in range(L):
        if cfg["down"][i] == "attn":
            sites += [(cfg["ch"][i], (side >> i) ** 2)] * (cfg["layers"] * cfg["tf"][i])
    sites += [(cfg["ch"][-1], (side >> (L - 1)) ** 2)] * cfg["tf"][-1]
    for i in range(L):
        level = L - 1 - i
        if cfg["up"][i] == "attn":
            sites += [(cfg["ch"][level], (side >> level) ** 2)] * ((cfg["layers"] + 1) * cfg["tf"][level])
    return sites


RESAMPLER = dict(dim=1280, depth=4, dim_head=64, heads=20, queries=16, emb=512, out=2048, ff=4)


def resampler_counts(c=RESAMPLER):
    d, inner, nq = c["dim"], c["dim_head"] * c["heads"], c["queries"]
    ff = d * c["ff"]
    kv = 1 + nq
    params = c["emb"] * d + d + nq * d + c["out"] * d + c["out"] + 2 * c["out"]
    layer = 2 * d + 2 * d + inner * d + 2 * inner * d + d * inner + 2 * d + ff * d + d * ff
    params += c["depth"] * layer
    flops = 2 * c["emb"] * d
    per = 2 * nq * d * inner + 2 * kv * d * 2 * inner + 4 * c["heads"] * nq * kv * c["dim_head"] + 2 * nq * inner * d + 2 * nq * d * ff * 2
    flops += c["depth"] * per + 2 * nq * d * c["out"]
    return params, flops, c["depth"] * 11 + 7


def instantid_counts():
    sites = sdxl_attn_sites(128)
    widths = sorted(set(w for w, _ in sites))
    nq, td = RESAMPLER["queries"], RESAMPLER["out"]
    rows, tot_p, tot_kv, tot_att = [], 0, 0, 0
    for w in widths:
        grp = [(ww, t) for ww, t in sites if ww == w]
        t = grp[0][1]
        p = 2 * w * td
        kvf = 2 * nq * td * 2 * w
        att = 4 * t * nq * w
        rows.append([w, len(grp), _int(t), _int(p), _int(len(grp) * p), f"{kvf / 1e6:,.1f}", f"{att / 1e6:,.1f}"])
        tot_p += len(grp) * p
        tot_kv += len(grp) * kvf
        tot_att += len(grp) * att
    rp, rf, rt = resampler_counts()
    ratio = tot_att / 1e9 / SDXL_PASS_GFLOP * 100
    return (
        "InstantID's identity branch derived from the SDXL U-Net's 70 cross-attention sites (counted from the configuration of "
        "`denoiser_architectures`, at a 128 by 128 latent) and the released resampler's shapes (width 1280, 4 layers, 20 heads of "
        "64, 16 queries, 512 in, 2048 out, feed-forward ratio 4). A site's added attention has two bias-free projections from the "
        "2048-wide tokens (`to_k_ip`, `to_v_ip`), its keys and values are computed from the 16 tokens and the scores and weighted sums "
        "over 16 rows for each of the site's tokens (the queries are the text attention's own); operations in MFLOP per site. The "
        "resampler's count is, per layer, three layer norms (the input rows, the latents and the feed-forward's input, each a gain "
        "and an offset: 6 x 1280 numbers), the bias-free query, fused key-value and output projections (1280 to 1280, 1280 to "
        "2 x 1280 and 1280 to 1280, with 20 heads of 64) and the bias-free feed-forward pair (1280 to 5120 and back): 11 tensors a "
        "layer; outside the layers, the input projection 512 to 1280 with its bias, the 16 learned latents, the output projection "
        "1280 to 2048 with its bias and a last layer norm of 2048 (gain and offset): 7 tensors.\n\n"
        + _table(["site width", "sites", "tokens T", "numbers per site", "numbers, all sites", "keys and values, MFLOP", "scores and sums, MFLOP"],
                 rows, "rrrrrrr")
        + f"\n\nThe 70 sites hold {_int(tot_p)} numbers and the resampler {_int(rp)} in {rt} tensors, {_int(tot_p + rp)} in all "
        f"({(tot_p + rp) * 4 / 1e9:.2f} GB in 32-bit floats). The resampler costs {_gf(rf / 1e9, 2)} once per identity, and so do the keys and "
        f"values of the 16 tokens ({_gf(tot_kv / 1e9)}), which depend only on the identity. In every evaluation of the denoiser the "
        f"added attention costs {_gf(tot_att / 1e9)} for the scores and sums, {ratio:.2f} percent of the "
        f"{_gf(SDXL_PASS_GFLOP)} of the U-Net pass."
    )


PULID = dict(dim=1024, depth=10, dim_head=64, heads=16, ids=5, queries=32, out=2048, ff=4, cond=1280, vit=577, scales=5,
             ca_dim=3072, ca_head=128, ca_heads=16, dbl_int=2, sgl_int=4)


def pulid_schedule(dbl=19, sgl=38, c=PULID):
    sites = [("double", b) for b in range(0, dbl, c["dbl_int"])] + [("single", b) for b in range(0, sgl, c["sgl_int"])]
    return sites


def pulid_params(c=PULID):
    d, inner, ff = c["dim"], c["dim_head"] * c["heads"], c["dim"] * c["ff"]
    mlp = lambda i, o: i * d + d + 2 * d + d * d + d + 2 * d + o * d + o   # noqa: E731
    enc = c["queries"] * d + c["out"] * d
    enc += mlp(c["cond"], c["ids"] * d) + c["scales"] * mlp(d, d)
    layer = 4 * d + inner * d + 2 * inner * d + d * inner + 2 * d + ff * d + d * ff
    enc += c["depth"] * layer
    kv, dm, ci = c["out"], c["ca_dim"], c["ca_heads"] * c["ca_head"]
    ca = 2 * kv + 2 * dm + ci * dm + 2 * ci * kv + dm * ci
    return enc, ca


def pulid_flops(n_img=4096, c=PULID):
    d, inner, ff = c["dim"], c["dim_head"] * c["heads"], c["dim"] * c["ff"]
    lr, cr = c["queries"] + c["ids"], c["ids"] + c["vit"]
    kv = cr + lr
    mlp = lambda m, i, o: 2 * m * (i * d + d * d + d * o)   # noqa: E731
    enc = mlp(1, c["cond"], c["ids"] * d) + c["scales"] * mlp(c["vit"], d, d)
    layer = 2 * lr * d * inner + 2 * kv * d * 2 * inner + 4 * c["heads"] * lr * kv * c["dim_head"] + 2 * lr * inner * d + 2 * lr * d * ff * 2
    enc += c["depth"] * layer + 2 * c["queries"] * d * c["out"]
    kvd, dm, ci = c["out"], c["ca_dim"], c["ca_heads"] * c["ca_head"]
    q, kvp = 2 * n_img * dm * ci, 2 * c["queries"] * kvd * 2 * ci
    att = 4 * n_img * c["queries"] * ci
    out = 2 * n_img * ci * dm
    return enc, (q, kvp, att, out)


def pulid_counts():
    sites = pulid_schedule()
    enc_p, ca_p = pulid_params()
    enc_f, (q, kvp, att, out) = pulid_flops()
    site_f = q + kvp + att + out
    flux_pass = tc.flux1_pass_flops(1024, 512)
    dbl = [b for s, b in sites if s == "double"]
    sgl = [b for s, b in sites if s == "single"]
    rows = [["IdFormer (once per identity)", _int(enc_p), 172, f"{enc_f / 1e9:,.1f}"],
            ["20 cross-attention sites", _int(20 * ca_p), 140, f"{20 * site_f / 1e9:,.1f} per pass"],
            ["one site", _int(ca_p), 7, f"{site_f / 1e9:,.2f}"]]
    rows2 = [["queries: Wq on 4,096 image rows", f"{q / 1e9:.3f}"], ["keys and values from 32 tokens", f"{kvp / 1e9:.3f}"],
             ["scores and weighted sums over 32 tokens", f"{att / 1e9:.3f}"], ["output projection", f"{out / 1e9:.3f}"],
             ["one site", f"{site_f / 1e9:.3f}"], ["20 sites", f"{20 * site_f / 1e9:.2f}"]]
    return (
        "PuLID's adapter for FLUX.1 from its configuration (IdFormer width 1024, 10 layers, 16 heads of 64, 32 learned queries and 5 "
        "ID tokens, 577 tokens per tapped EVA-CLIP state, 5 states, ID condition of 1280 numbers; cross-attention sites of width "
        "3072 with 16 heads of 128 and 2048-wide ID tokens). Tensors in brain's layout; a multiply-add is two operations. One "
        "site has seven tensors: the gain and offset of a layer normalisation of the ID tokens "
        f"({_int(2 * PULID['out'])} numbers), the gain and offset of one of the image rows ({_int(2 * PULID['ca_dim'])}), the query "
        f"projection ({_int(PULID['ca_dim'] * PULID['ca_heads'] * PULID['ca_head'])}, bias-free), the fused key and value projection "
        f"({_int(2 * PULID['out'] * PULID['ca_heads'] * PULID['ca_head'])}) and the output projection "
        f"({_int(PULID['ca_dim'] * PULID['ca_heads'] * PULID['ca_head'])}), which add up to the table's count of one site.\n\n"
        + _table(["part", "parameters", "tensors", "GFLOP"], rows, "lrrr")
        + f"\n\nThe schedule: one site after double-stream block b for b = {', '.join(map(str, dbl))} (every second of 19) and one "
        f"after single-stream block b for b = {', '.join(map(str, sgl))} (every fourth of 38), {len(sites)} in all; the sites take "
        "the module indices 0 to 19 in that order, one counter for both streams.\n\n"
        + _table(["part of one site, 4,096 image rows", "GFLOP"], rows2, "lr")
        + f"\n\nOne FLUX.1 dev pass at 1024 by 1024 with 512 text rows is {_gf(flux_pass / 1e9)} (Chapter 9's rule for a FLUX.1 pass, in the caption of its work table), "
        f"so the 20 sites add {20 * site_f / flux_pass * 100:.2f} percent to every pass they are present in; "
        f"the IdFormer's {_gf(enc_f / 1e9)} are paid once per identity."
    )


# ------------------------------------------------------------------ data of the figures

def loss_curves_data():
    """(steps, loss with zero-convs at zero, loss with drawn zero-convs, loss of the base network) of the first run of each."""
    z, r = control_run(0.0, False)[0], control_run(0.0, True)[0]
    P, E = base_params()
    xs, cs = control_data(DATA, N_TRAIN)
    tg = _control_targets(xs, cs)
    base = sum(_val(_mse(unet(P, x, E), t, N_TRAIN)) for x, t in zip(xs, tg))
    return list(range(CONTROL_STEPS + 1)), [h[0] for h in z], [h[0] for h in r], base


def gradient_norms_data():
    """(steps, zero-conv gradient norm, trainable-copy gradient norm) of the zero-start run for the first 12 steps."""
    h = control_run(0.0, False)[0]
    n = 12
    return list(range(n)), [h[s][1] for s in range(n)], [h[s][2] for s in range(n)]
