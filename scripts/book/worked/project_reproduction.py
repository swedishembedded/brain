# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the project "Reproduce, Verify, Optimize, Defend".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: project_reproduction.<name> -->` markers; helpers with a leading underscore are shared with the chapter's figures.

The component is a pre-norm gated feed-forward block: an RMS norm with a gain, two projections (gate and up), the SiLU of the
gate times the up projection, a down projection and a residual addition. The project reproduces it from a published layout
(a fused gate-and-up matrix, a down projection stored input-major), writes a 64-bit golden with every intermediate tensor,
checks a 32-bit port tap by tap against tolerances derived from the count of roundings, checks the backward pass against
central differences, stores the three matrices in 8-bit codes with one scale per group of 32 weights, counts the bytes a
decoding step moves with a least-recently-used cache simulation, states the claims before measuring them, and puts the
result behind a gate. 32-bit arithmetic is emulated by rounding each 64-bit result to 32 bits (`importing_models.r32`), which
is the correctly rounded 32-bit result of one addition, subtraction, multiplication, division or square root of 32-bit
operands; exp is the library's 64-bit value rounded once. Sums run from the first term, left to right, unless stated.

Reused by import, never copied: the 32-bit rounding, the ordered sums and the three comparison measures of `importing_models`,
the 64-bit FNV-1a digest of `checkpoints_and_recovery`, the t interval of `experiments` and the table helper of
`number_formats`. Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`).

Swedish Embedded AB implements solutions for reproducing published model components and proving them correct, fast and
defensible. If your team needs expertise in porting, numerical verification and honest performance claims then you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from collections import OrderedDict
from functools import lru_cache

import bookrng
import checkpoints_and_recovery as cr
import experiments as ex
import importing_models as im
import number_formats as nf

_table = nf._table
r32 = im.r32
U32 = 2.0 ** -24
U64 = 2.0 ** -53
EPS = 1e-6

# ------------------------------------------------------------------ the component at the size of the port

D0, F0, T0 = 6, 10, 5          # width, hidden width and positions of the printable component
PROBE_SCALE = 0.001            # the inputs of the second golden are the first ones times this


def _ident(v):
    return v


def _sci(v, d=3):
    return f"{v:.{d - 1}e}"


def _seed_line(name):
    return f"`{name}` (seed {bookrng.seed_of(name)})"


def _uniform_matrix(g, rows, cols, lim, r):
    return [[r((2.0 * g.uniform() - 1.0) * lim) for _ in range(cols)] for _ in range(rows)]


def _draw_params(g, D, F, r=r32):
    """The draw order of every weight set: gain, then gate, up and down, each row-major."""
    gain = [r(1.0 + 0.25 * (2.0 * g.uniform() - 1.0)) for _ in range(D)]
    lim_d, lim_f = 1.0 / math.sqrt(D), 1.0 / math.sqrt(F)
    wg = _uniform_matrix(g, F, D, lim_d, r)
    wu = _uniform_matrix(g, F, D, lim_d, r)
    wd = _uniform_matrix(g, D, F, lim_f, r)
    return {"g": gain, "Wg": wg, "Wu": wu, "Wd": wd}


@lru_cache(maxsize=None)
def _port_params():
    return _draw_params(bookrng.generator("project_reproduction.weights"), D0, F0)


@lru_cache(maxsize=None)
def _square_params():
    return _draw_params(bookrng.generator("project_reproduction.square"), 8, 8)


@lru_cache(maxsize=None)
def _inputs():
    g = bookrng.generator("project_reproduction.inputs")
    x = [[r32(2.0 * g.uniform() - 1.0) for _ in range(D0)] for _ in range(T0)]
    probe = [[r32(v * PROBE_SCALE) for v in row] for row in x]
    return x, probe


@lru_cache(maxsize=None)
def _square_inputs():
    g = bookrng.generator("project_reproduction.square_inputs")
    return [[r32(2.0 * g.uniform() - 1.0) for _ in range(8)] for _ in range(T0)]


def _transpose(M):
    return [list(c) for c in zip(*M)]


def published(p):
    """The publisher's layout: a fused [2F, D] gate-and-up matrix (gate rows first) and the down projection stored [F, D]."""
    return {"ffn_norm.weight": ([len(p["g"])], list(p["g"])),
            "ffn.gate_up.weight": ([2 * len(p["Wg"]), len(p["g"])], [list(r) for r in p["Wg"]] + [list(r) for r in p["Wu"]]),
            "ffn.down.weight": ([len(p["Wd"][0]), len(p["Wd"])], _transpose(p["Wd"]))}


def import_params(tensors, D, F, bug=None):
    """The mapping: rename, cut the fused matrix, transpose the down projection, then check every shape. Returns (params,
    error); a mapping whose shapes disagree with the port's is refused by name."""
    shape_g, g = tensors["ffn_norm.weight"]
    shape_gu, gu = tensors["ffn.gate_up.weight"]
    shape_d, dn = tensors["ffn.down.weight"]
    gate, up = gu[:F], gu[F:2 * F]
    if bug == "swap":
        gate, up = up, gate
    down = dn if bug == "transpose" else _transpose(dn)
    want = {"g": [D], "Wg": [F, D], "Wu": [F, D], "Wd": [D, F]}
    got = {"g": [len(g)], "Wg": [len(gate), len(gate[0])], "Wu": [len(up), len(up[0])], "Wd": [len(down), len(down[0])]}
    for k in ("g", "Wg", "Wu", "Wd"):
        if got[k] != want[k]:
            return None, f"refused: {k} has shape [{got[k][0]}, {got[k][1]}], the port expects [{want[k][0]}, {want[k][1]}]"
    return {"g": list(g), "Wg": gate, "Wu": up, "Wd": down}, None


# ------------------------------------------------------------------ the forward pass, in any stated format

def _dot(a, b, r):
    """sum of a[i] b[i], left to right, the first product rounded and then each addition (the book's ordered dot)."""
    s = r(a[0] * b[0])
    for i in range(1, len(a)):
        s = r(s + r(a[i] * b[i]))
    return s


def _sigmoid(z, r):
    if z >= 0.0:
        return r(1.0 / r(1.0 + r(math.exp(-z))))
    e = r(math.exp(z))
    return r(e / r(1.0 + e))


def forward(x, p, r=_ident, eps=EPS, bug=None):
    """Every tap of the block for the rows of x: n (norm output), gate, up, h, f (down projection) and y (with the residual).
    r rounds after every operation (identity for 64 bits, r32 for the port); eps is rounded by r before use."""
    D = len(p["g"])
    e = r(eps)
    taps = {k: [] for k in ("xhat", "inv", "n", "gate", "up", "h", "f", "y")}
    for row in x:
        ss = _dot(row, row, r)
        ms = ss if bug == "sum" else r(ss / D)
        if bug == "eps_out":
            inv = r(1.0 / r(r(math.sqrt(ms)) + e))
        else:
            inv = r(1.0 / r(math.sqrt(r(ms + e))))
        xhat = [r(v * inv) for v in row]
        n = [r(gi * xi) for gi, xi in zip(p["g"], xhat)]
        a = [_dot(w, n, r) for w in p["Wg"]]
        b = [_dot(w, n, r) for w in p["Wu"]]
        if bug == "glu":
            h = [r(_sigmoid(ai, r) * bi) for ai, bi in zip(a, b)]
        else:
            h = [r(r(ai * _sigmoid(ai, r)) * bi) for ai, bi in zip(a, b)]
        f = [_dot(w, h, r) for w in p["Wd"]]
        y = [r(xi + fi) for xi, fi in zip(row, f)]
        for k, v in (("xhat", xhat), ("inv", inv), ("n", n), ("gate", a), ("up", b), ("h", h), ("f", f), ("y", y)):
            taps[k].append(v)
    return taps


TAPS = ("n", "gate", "up", "h", "f", "y")


def roundings(tap, D, F):
    """Roundings on the longest chain to a tap: D + 6 for the norm, D for a projection, 5 for the SiLU and the product
    (exp, add, divide, product with the gate, product with up), F for the down projection and 1 for the residual."""
    return {"n": D + 6, "gate": 2 * D + 6, "up": 2 * D + 6, "h": 2 * D + 11, "f": 2 * D + 11 + F, "y": 2 * D + 12 + F}[tap]


def bound(tap, D=D0, F=F0):
    """The tolerance of a tap: its roundings plus the one of storing the golden in 32 bits, times 2^-24."""
    return (roundings(tap, D, F) + 1) * U32


@lru_cache(maxsize=None)
def _golden():
    """The 64-bit reference on both inputs, every tap stored rounded to 32 bits."""
    x, probe = _inputs()
    p = _port_params()
    out = {"x": x, "x_probe": probe}
    for tag, inp in (("unit", x), ("probe", probe)):
        t = forward(inp, p, _ident)
        for k in TAPS:
            out[f"{tag}/{k}"] = [[r32(v) for v in row] for row in t[k]]
    return out


def _bytes32(rows):
    return b"".join(struct.pack("<f", v) for row in rows for v in row)


def _digest(rows):
    return f"{cr.fnv1a64(_bytes32(rows)):016x}"


def _port(inputs, bug=None, params=None):
    return forward(inputs, params or _port_params(), r32, bug=bug)


# ------------------------------------------------------------------ tables: the component, the golden, the import

def component():
    p = _port_params()
    x, _ = _inputs()
    rows = [["gain g", f"[{D0}]", "1 + 0.25 (2u - 1)", D0, _digest([p["g"]])],
            ["W_gate", f"[{F0}, {D0}]", "(2u - 1) / sqrt(6)", F0 * D0, _digest(p["Wg"])],
            ["W_up", f"[{F0}, {D0}]", "(2u - 1) / sqrt(6)", F0 * D0, _digest(p["Wu"])],
            ["W_down", f"[{D0}, {F0}]", "(2u - 1) / sqrt(10)", D0 * F0, _digest(p["Wd"])],
            ["inputs x", f"[{T0}, {D0}]", "2u - 1", T0 * D0, _digest(x)]]
    head = (f"The weights come from the generator {_seed_line('project_reproduction.weights')}, the inputs from "
            f"{_seed_line('project_reproduction.inputs')}. Each number is one uniform() u put through the rule of its row and then "
            "rounded to the nearest 32-bit number (ties to even); the square roots are the correctly rounded 64-bit ones. The "
            "weights are drawn in the order of the rows, each matrix row-major ([outputs, inputs]), so the gain takes draws 0 to "
            f"5, W_gate 6 to 65, W_up 66 to 125 and W_down 126 to 185; the inputs are drawn position by position. The digest is "
            "the FNV-1a 64 of the 32-bit numbers, little-endian, in that order. The second set of inputs, the probe, is each input "
            f"times {PROBE_SCALE} rounded to 32 bits.\n\n")
    first = ", ".join(f"{v:.8f}" for v in p["g"][:3])
    wfirst = ", ".join(f"{v:.8f}" for v in p["Wg"][0][:3])
    xfirst = ", ".join(f"{v:.8f}" for v in x[0][:3])
    tail = (f"\n\nThe first three gains are {first}; the first three entries of row 0 of W_gate are {wfirst}; the first three "
            f"inputs of position 0 are {xfirst}.")
    return head + _table(["tensor", "shape", "rule", "numbers", "FNV-1a 64"], rows, "lllrl") + tail


def golden_contents():
    gd = _golden()
    rows = [["x", f"{T0} x {D0}", "the unit inputs", _digest(gd["x"])],
            ["x_probe", f"{T0} x {D0}", f"the probe inputs (x times {PROBE_SCALE})", _digest(gd["x_probe"])]]
    what = {"n": "output of the norm", "gate": "gate projection", "up": "up projection", "h": "SiLU(gate) times up",
            "f": "down projection", "y": "block output, residual added"}
    for tag in ("unit", "probe"):
        for k in TAPS:
            w = F0 if k in ("gate", "up", "h") else D0
            rows.append([f"{tag}/{k}", f"{T0} x {w}", what[k], _digest(gd[f"{tag}/{k}"])])
    y0 = ", ".join(f"{v:.6f}" for v in gd["unit/y"][0])
    head = ("The golden: the 64-bit reference run on both sets of inputs, every tap rounded once to 32 bits. Row-major, "
            "positions first; the digest is the FNV-1a 64 of the 32-bit numbers, little-endian.\n\n")
    return head + _table(["tensor", "shape", "what", "FNV-1a 64"], rows, "llll") + f"\n\nRow 0 of `unit/y` is {y0}."


def import_checks():
    p = _port_params()
    sq = _square_params()
    rows = []
    for label, params, D, F, bug in (("6 x 10, the mapping as specified", p, D0, F0, None),
                                     ("6 x 10, gate and up cut in the wrong order", p, D0, F0, "swap"),
                                     ("6 x 10, the down projection not transposed", p, D0, F0, "transpose"),
                                     ("8 x 8, the mapping as specified", sq, 8, 8, None),
                                     ("8 x 8, the down projection not transposed", sq, 8, 8, "transpose")):
        got, err = import_params(published(params), D, F, bug)
        if err:
            rows.append([label, err, "none"])
            continue
        xin = _inputs()[0] if D == D0 else _square_inputs()
        ref = forward(xin, params, _ident)
        port = forward(xin, got, r32)
        rows.append([label, "loads", _sci(im.rel_l2(port["y"], ref["y"]))])
    head = ("The mapping of the published tensors onto the port, at the 6 x 10 component and at a square 8 x 8 one (weights "
            f"from {_seed_line('project_reproduction.square')}, inputs from {_seed_line('project_reproduction.square_inputs')}, "
            "5 positions, the same rules and draw order). rel_l2 compares the port's y with the 64-bit reference of the same "
            "weights.\n\n")
    return head + _table(["configuration and mapping", "shape check", "rel_l2 of y"], rows, "llr")


def tap_parity():
    gd = _golden()
    x, probe = _inputs()
    pu, pp = _port(x), _port(probe)
    rows = []
    for k in TAPS:
        rows.append([k, roundings(k, D0, F0), _sci(bound(k)), _sci(im.rel_l2(pu[k], gd[f"unit/{k}"])),
                     _sci(im.rel_l2(pp[k], gd[f"probe/{k}"]))])
    head = ("The 32-bit port against the golden, tap by tap. 'Roundings' counts the longest chain of dependent operations "
            "to the tap (a sum of n products costs n, each further operation 1); the bound is that count plus one (the "
            "golden's own rounding to 32 bits) times 2^-24. The port rounds every product, addition, division, square root "
            "and exp to 32 bits, and rounds eps = 1e-6 to 32 bits before use; its sums run left to right.\n\n")
    return head + _table(["tap", "roundings", "bound", "rel_l2, unit inputs", "rel_l2, probe inputs"], rows)


MISTAKES = (("gate and up cut in the wrong order", "swap"),
            ("mean of the squares replaced by their sum", "sum"),
            ("eps added after the square root", "eps_out"),
            ("sigmoid of the gate without the product by the gate (GLU for SwiGLU)", "glu"))


def _mistake_port(inputs, bug):
    if bug == "swap":
        params, _ = import_params(published(_port_params()), D0, F0, "swap")
        return _port(inputs, params=params)
    return _port(inputs, bug=bug)


def _first_above(port, tag):
    gd = _golden()
    for k in TAPS:
        if im.rel_l2(port[k], gd[f"{tag}/{k}"]) > bound(k):
            return k
    return None


def _one_minus_cos(a, b):
    v = 1.0 - im.cosine(a, b)
    return "below 1e-9" if v < 1e-9 else _sci(v)


def mistakes():
    gd = _golden()
    x, probe = _inputs()
    rows = []
    for label, bug in MISTAKES:
        for tag, inp in (("unit", x), ("probe", probe)):
            port = _mistake_port(inp, bug)
            k = _first_above(port, tag)
            if k is None:
                rows.append([label, tag, "none", "", "", _sci(im.rel_l2(port["y"], gd[f"{tag}/y"]))])
            else:
                rows.append([label, tag, k, _sci(im.rel_l2(port[k], gd[f"{tag}/{k}"])),
                             _one_minus_cos(port[k], gd[f"{tag}/{k}"]), _sci(im.rel_l2(port["y"], gd[f"{tag}/y"]))])
    head = ("Four planted mistakes, each run through the 32-bit port on both sets of inputs. 'First tap above' is the first "
            "tap, in the order n, gate, up, h, f, y, whose rel_l2 against the golden exceeds its bound; the next two columns "
            "measure that tap, and the last one the block output. 1 - cosine is printed as 'below 1e-9' when it is.\n\n")
    return head + _table(["mistake", "inputs", "first tap above", "rel_l2 there", "1 - cosine there", "rel_l2 of y"],
                         rows, "lllrrr")


# ------------------------------------------------------------------ the backward pass and its check

@lru_cache(maxsize=None)
def _loss_weights():
    g = bookrng.generator("project_reproduction.probe")
    return [[2.0 * g.uniform() - 1.0 for _ in range(D0)] for _ in range(T0)]


def _loss(x, p):
    c = _loss_weights()
    y = forward(x, p, _ident)["y"]
    s = 0.0
    for t in range(T0):
        for i in range(D0):
            s += c[t][i] * y[t][i]
    return s


def _silu_slope(z):
    s = _sigmoid(z, _ident)
    return s * (1.0 + z * (1.0 - s))


def backward(x, p, bug=None):
    """Gradients of sum c[t][i] y[t][i] with respect to x, g, W_gate, W_up and W_down, in 64 bits."""
    c = _loss_weights()
    tp = forward(x, p, _ident)
    D, F = len(p["g"]), len(p["Wg"])
    gx = [[0.0] * D for _ in range(T0)]
    gg = [0.0] * D
    gWg = [[0.0] * D for _ in range(F)]
    gWu = [[0.0] * D for _ in range(F)]
    gWd = [[0.0] * F for _ in range(D)]
    for t in range(T0):
        df = c[t]
        h, a, b, n, xhat, inv = tp["h"][t], tp["gate"][t], tp["up"][t], tp["n"][t], tp["xhat"][t], tp["inv"][t]
        for i in range(D):
            for j in range(F):
                gWd[i][j] += df[i] * h[j]
        dh = []
        for j in range(F):
            s = 0.0
            for i in range(D):
                s += df[i] * p["Wd"][i][j]
            dh.append(s)
        da, db = [], []
        for j in range(F):
            sg = _sigmoid(a[j], _ident)
            slope = sg if bug == "slope" else _silu_slope(a[j])
            da.append(dh[j] * b[j] * slope)
            db.append(dh[j] * (a[j] * sg))
        for j in range(F):
            for i in range(D):
                gWg[j][i] += da[j] * n[i]
                gWu[j][i] += db[j] * n[i]
        dn = []
        for i in range(D):
            s = 0.0
            for j in range(F):
                s += da[j] * p["Wg"][j][i]
            for j in range(F):
                s += db[j] * p["Wu"][j][i]
            dn.append(s)
        m = 0.0
        for i in range(D):
            m += p["g"][i] * dn[i] * xhat[i]
        m = m / D
        for i in range(D):
            gg[i] += dn[i] * xhat[i]
            if bug == "norm":
                gx[t][i] = df[i] + inv * p["g"][i] * dn[i]
            else:
                gx[t][i] = df[i] + inv * (p["g"][i] * dn[i] - xhat[i] * m)
    return {"x": gx, "g": [gg], "W_gate": gWg, "W_up": gWu, "W_down": gWd}


def _entries():
    """(tensor, row, column) of every checked number, in the order x, g, W_gate, W_up, W_down."""
    out = [("x", t, i) for t in range(T0) for i in range(D0)]
    out += [("g", 0, i) for i in range(D0)]
    for name, rows, cols in (("W_gate", F0, D0), ("W_up", F0, D0), ("W_down", D0, F0)):
        out += [(name, r, c) for r in range(rows) for c in range(cols)]
    return out


_KEY = {"W_gate": "Wg", "W_up": "Wu", "W_down": "Wd"}


def _perturbed(x, p, name, r, c, delta):
    x2 = [list(row) for row in x]
    p2 = {k: ([list(v) for v in p[k]] if k != "g" else list(p[k])) for k in p}
    if name == "x":
        x2[r][c] += delta
    elif name == "g":
        p2["g"][c] += delta
    else:
        p2[_KEY[name]][r][c] += delta
    return x2, p2


@lru_cache(maxsize=None)
def _gradcheck(bug):
    x, _ = _inputs()
    p = _port_params()
    ana = backward(x, p, bug)
    h = 1e-6
    worst, where = 0.0, None
    for name, r, c in _entries():
        xp, pp = _perturbed(x, p, name, r, c, h)
        xm, pm = _perturbed(x, p, name, r, c, -h)
        num = (_loss(xp, pp) - _loss(xm, pm)) / (2.0 * h)
        a = ana[name][r][c]
        dev = abs(num - a) / max(1.0, abs(a))
        if dev > worst:
            worst, where = dev, (name, r, c)
    return worst, where


def gradient_check():
    rows = []
    for label, bug in (("backward as specified", None), ("SiLU slope taken as sigmoid(gate)", "slope"),
                       ("norm backward without the mean term", "norm")):
        worst, (name, r, c) = _gradcheck(bug)
        if worst < 1e-6:
            rows.append([label, "passes (below 1e-6)", ""])
        else:
            at = f"{name}[{c}]" if name == "g" else f"{name}[{r}, {c}]"
            rows.append([label, _sci(worst), at])
    n = len(_entries())
    head = ("The backward pass against central differences, in 64-bit arithmetic, on the component of the tables above and "
            "its unit inputs. The loss is the sum over positions t and coordinates i of c[t][i] y[t][i], with c drawn from "
            f"{_seed_line('project_reproduction.probe')} as 2u - 1, position by position (no rounding to 32 bits). Each of "
            f"the {n} numbers (30 inputs, 6 gains, 60 + 60 + 60 weights) is moved by +1e-6 and -1e-6; the deviation is "
            "|numeric - analytic| / max(1, |analytic|) and the criterion is that its largest value is below 1e-6. 'Where' "
            "is the number with the largest deviation, [row, column] counted from 0.\n\n")
    return head + _table(["backward pass", "largest deviation", "where"], rows, "lrl")


# ------------------------------------------------------------------ the component at the size of the claims

D1, F1, T1 = 256, 768, 4       # width, hidden width and positions of one draw of the quality test
G = 32                         # weights per scale group
OUTLIER = 16.0                 # one weight of every row is multiplied by this
N_DRAWS = 8


def _draw_name(k):
    return f"project_reproduction.draw.{k}"


@lru_cache(maxsize=None)
def _draw(k, outliers=True):
    g = bookrng.generator(_draw_name(k))
    p = _draw_params(g, D1, F1)
    cols = []
    for M in (p["Wg"], p["Wu"], p["Wd"]):
        for row in M:
            c = g.integer(len(row))
            cols.append(c)
            if outliers:
                row[c] = row[c] * OUTLIER
    x = [[r32(2.0 * g.uniform() - 1.0) for _ in range(D1)] for _ in range(T1)]
    return p, x


def quantize_rows(W, group):
    """Codes and scales of a matrix: for every row and every group of `group` consecutive weights, s = max(|w|, 1e-8) / 127
    and q = round(w / s) half away from zero, clamped to [-127, 127]; all in 64 bits."""
    Q, S = [], []
    for row in W:
        qs, ss = [], []
        for g0 in range(0, len(row), group):
            blk = row[g0:g0 + group]
            a = 0.0
            for v in blk:
                a = max(a, abs(v))
            s = max(a, 1e-8) / 127.0
            for v in blk:
                t = v / s
                q = math.floor(abs(t))
                if abs(t) - q >= 0.5:          # exact: ties go away from zero
                    q += 1
                qs.append(int(min(127, q) * (1 if t >= 0 else -1)))
            ss.append(s)
        Q.append(qs)
        S.append(ss)
    return Q, S


def int8_matvec(Q, S, x, group):
    """out[o] = sum over groups g of S[o][g] * (sum over the group's j of Q[o][j] x[j]), every sum from 0.0 left to right."""
    out = []
    for qs, ss in zip(Q, S):
        acc = 0.0
        for gi, s in enumerate(ss):
            part = 0.0
            for j in range(gi * group, gi * group + group):
                part += qs[j] * x[j]
            acc += s * part
        out.append(acc)
    return out


def _matvec(W, x):
    out = []
    for row in W:
        s = 0.0
        for w, v in zip(row, x):
            s += w * v
        out.append(s)
    return out


def _ffn(x, p, mv):
    """The block in 64 bits with the products by the three matrices done by mv(name, vector); returns (f, y)."""
    D = len(p["g"])
    fs, ys = [], []
    for row in x:
        ss = 0.0
        for v in row:
            ss += v * v
        inv = 1.0 / math.sqrt(ss / D + EPS)
        n = [gi * (v * inv) for gi, v in zip(p["g"], row)]
        a, b = mv("Wg", n), mv("Wu", n)
        h = [(ai * _sigmoid(ai, _ident)) * bi for ai, bi in zip(a, b)]
        f = mv("Wd", h)
        fs.append(f)
        ys.append([xi + fi for xi, fi in zip(row, f)])
    return fs, ys


@lru_cache(maxsize=None)
def _quantized(k, group, outliers=True):
    p, _ = _draw(k, outliers)
    return {m: quantize_rows(p[m], group) for m in ("Wg", "Wu", "Wd")}


@lru_cache(maxsize=None)
def _reference(k, outliers=True):
    p, x = _draw(k, outliers)
    return _ffn(x, p, lambda m, v: _matvec(p[m], v))


@lru_cache(maxsize=None)
def _int8_run(k, scheme, outliers=True):
    """scheme: 'group' (scales per 32 weights), 'row' (one scale per row) or 'buffer' (group codes dequantized into a full
    64-bit copy of each matrix and multiplied by it)."""
    p, x = _draw(k, outliers)
    qz = {m: (quantize_rows(p[m], len(p[m][0])) if scheme == "row" else _quantized(k, G, outliers)[m]) for m in ("Wg", "Wu", "Wd")}
    if scheme == "buffer":
        deq = {m: [[s_row[j // G] * q for j, q in enumerate(q_row)] for q_row, s_row in zip(*qz[m])] for m in qz}
        return _ffn(x, p, lambda m, v: _matvec(deq[m], v))
    grp = {m: (len(p[m][0]) if scheme == "row" else G) for m in qz}
    return _ffn(x, p, lambda m, v: int8_matvec(qz[m][0], qz[m][1], v, grp[m]))


@lru_cache(maxsize=None)
def _dequant_reference(k, scheme):
    """The 64-bit reference that uses the dequantized weights s * q: what the fast kernel must equal to rounding."""
    p, x = _draw(k)
    qz = {m: (quantize_rows(p[m], len(p[m][0])) if scheme == "row" else _quantized(k, G)[m]) for m in ("Wg", "Wu", "Wd")}
    deq = {}
    for m in qz:
        Q, S = qz[m]
        grp = len(p[m][0]) if scheme == "row" else G
        deq[m] = [[s_row[j // grp] * q for j, q in enumerate(q_row)] for q_row, s_row in zip(Q, S)]
    return _ffn(x, p, lambda m, v: _matvec(deq[m], v))


def _err(k, scheme, outliers=True, tap=0):
    ref = _reference(k, outliers)[tap]
    got = _int8_run(k, scheme, outliers)[tap]
    return im.rel_l2(got, ref)


def _interval_text(vals):
    m, s, se, lo, hi = ex._interval(vals)
    return m, s, lo, hi


QUALITY_MARGIN = 0.02


@lru_cache(maxsize=None)
def _quality_columns():
    cols = {"group": [_err(k, "group") for k in range(N_DRAWS)],
            "row": [_err(k, "row") for k in range(N_DRAWS)],
            "row_clean": [_err(k, "row", outliers=False) for k in range(N_DRAWS)],
            "group_y": [_err(k, "group", tap=1) for k in range(N_DRAWS)]}
    return cols


def quality():
    cols = _quality_columns()
    rows = []
    for k in range(N_DRAWS):
        rows.append([k, bookrng.seed_of(_draw_name(k)), _sci(cols["group"][k]), _sci(cols["row"][k]),
                     _sci(cols["row_clean"][k]), _sci(cols["group_y"][k])])
    stats = {c: _interval_text(v) for c, v in cols.items()}
    order = ("group", "row", "row_clean", "group_y")
    rows.append(["mean", "", *[_sci(stats[c][0]) for c in order]])
    rows.append(["sample standard deviation", "", *[_sci(stats[c][1]) for c in order]])
    rows.append(["95% interval, lower end", "", *[_sci(stats[c][2]) for c in order]])
    rows.append(["95% interval, upper end", "", *[_sci(stats[c][3]) for c in order]])
    head = (f"The quality of 8-bit weights on the component at D = {D1}, F = {F1}, {T1} positions, eight draws. Draw k uses "
            "the generator `project_reproduction.draw.k` with the seed printed in its row, and draws in this order: the gain, "
            "W_gate, W_up and W_down by the rules of the first component (with 1/sqrt(256) and 1/sqrt(768)), then, for every "
            f"row of W_gate, of W_up and of W_down in that order, one integer(k) c over the row's k columns, and the weight at "
            f"column c is multiplied by {OUTLIER:g} (the outlier of the row); then the {T1} x {D1} inputs as 2u - 1. Weights "
            "and inputs are rounded to 32 bits; everything after is 64-bit. Each cell is the rel_l2 of the 8-bit block "
            "against the same block with the weights as drawn. 'Group of 32' has one scale per 32 consecutive weights of a "
            "row; 'one scale per row' has one for the whole row; the column without outliers redraws the same weights and "
            "skips the multiplication (the integers are still drawn); the last column measures the block output y, residual "
            "included, where the others measure f, the down projection. The interval is the mean plus or minus "
            "t s / sqrt(8) with s the sample standard deviation (divisor 7) and t = 2.36462, the 0.975 quantile of "
            "Student's t with 7 degrees of freedom.\n\n")
    return head + _table(["draw", "seed", "f, group of 32", "f, one scale per row", "f, one per row, no outliers",
                          "y, group of 32"], rows, "lrrrrr")


# ------------------------------------------------------------------ counting the work of a decoding step

L1, H1, V1 = 8, 4, 4096        # layers, attention heads and vocabulary of the decoder that carries the component
LINE = 64                      # bytes of one cache line
CACHE_LINES = 16384            # 1 MiB of cache


def ffn_arrays(scheme, D=D1, F=F1):
    """The arrays one feed-forward block reads, in access order, as (name, bytes)."""
    if scheme == "f32":
        return [("ffn gain", 4 * D), ("W_gate", 4 * F * D), ("W_up", 4 * F * D), ("W_down", 4 * D * F)]
    grp = {"group": (D // G, F // G), "row": (1, 1)}[scheme]
    return [("ffn gain", 4 * D), ("W_gate codes", F * D), ("W_gate scales", 4 * F * grp[0]),
            ("W_up codes", F * D), ("W_up scales", 4 * F * grp[0]),
            ("W_down codes", D * F), ("W_down scales", 4 * D * grp[1])]


def ffn_bytes(scheme, D=D1, F=F1):
    """Bytes of one block: every array read once; the 'buffer' build also writes and reads a 4-byte copy of each weight."""
    if scheme == "buffer":
        return ffn_bytes("group", D, F) + 2 * 4 * 3 * F * D
    return sum(b for _, b in ffn_arrays(scheme, D, F))


def layer_arrays(scheme, S, D=D1, F=F1):
    """Reads of one decoder layer for one token at context S (positions 0 .. S-1 cached)."""
    attn = [("attention gain", 4 * D), ("W_q", 4 * D * D), ("W_k", 4 * D * D), ("W_v", 4 * D * D),
            ("K cache", 4 * S * D), ("V cache", 4 * S * D), ("W_o", 4 * D * D)]
    return attn + ffn_arrays("group" if scheme == "buffer" else scheme, D, F)


def step_bytes(scheme, S, D=D1, F=F1, L=L1, V=V1):
    """Bytes of one decoding step: the embedding row, L layers (reads, plus the 2 D numbers of K and V written), the final
    gain and the output head."""
    per = sum(b for _, b in layer_arrays(scheme, S, D, F)) + 2 * 4 * D
    if scheme == "buffer":
        per += 2 * 4 * 3 * F * D
    return 4 * D + L * per + 4 * D + 4 * V * D


def step_flops(scheme, D=D1, F=F1):
    """FLOPs of one block for one token: a sum of n products is n multiplications and n - 1 additions; the norm adds a
    division, eps, a square root and a reciprocal; the SiLU and product cost 5 per element; an 8-bit product adds one
    scale multiplication per group and the additions that join the groups. Converting a code to a number is not counted."""
    norm = 4 * D + 3
    if scheme == "f32":
        mm = 2 * F * (2 * D - 1) + D * (2 * F - 1)
    else:
        gd, gf = (1, 1) if scheme == "row" else (D // G, F // G)
        mm = 2 * F * (2 * D + gd - 1) + D * (2 * F + gf - 1)
    return norm + mm + 5 * F + D


class LRU:
    """A cache of `lines` lines of 64 bytes, least recently used out first; counts the lines read from memory."""

    def __init__(self, lines):
        self.lines, self.held, self.misses = lines, OrderedDict(), 0

    def touch(self, line):
        if line in self.held:
            self.held.move_to_end(line)
            return
        self.misses += 1
        self.held[line] = True
        if len(self.held) > self.lines:
            self.held.popitem(last=False)


def _layout(arrays):
    """Consecutive addresses for named arrays, each starting at a multiple of 64 bytes; returns name -> (first line, lines)."""
    out, line = {}, 0
    for name, nbytes in arrays:
        n = -(-nbytes // LINE)
        out[name] = (line, n)
        line += n
    return out


def microbenchmark(scheme, calls=10):
    """Bytes read from memory by `calls` back-to-back calls of one feed-forward block on the same weights."""
    lay = _layout(ffn_arrays(scheme))
    c = LRU(CACHE_LINES)
    for _ in range(calls):
        for name, _ in ffn_arrays(scheme):
            first, n = lay[name]
            for ln in range(first, first + n):
                c.touch(ln)
    return c.misses * LINE


@lru_cache(maxsize=None)
def whole_pass(scheme, S, tokens=3):
    """Simulated bytes of the last of `tokens` consecutive decoding steps, at contexts S - tokens + 1 to S: every array of every
    layer in its own address range (K and V sized for the last position), reads through the cache, writes counted as bytes."""
    D, L = D1, L1
    smax = S + 1
    arrays = [("embedding", 4 * V1 * D)]
    for l in range(L):
        for name, b in layer_arrays(scheme, smax, D):
            arrays.append((f"{l}/{name}", b))
    arrays += [("final gain", 4 * D), ("head", 4 * V1 * D)]
    lay = _layout(arrays)
    c = LRU(CACHE_LINES)
    last = 0
    for k in range(tokens):
        pos = S - (tokens - 1) + k
        before, written = c.misses, 0

        def read(name, first_byte=0, nbytes=None):
            first, n = lay[name]
            lo = first + first_byte // LINE
            hi = first + (-(-(first_byte + nbytes) // LINE) if nbytes is not None else n)
            for ln in range(lo, hi):
                c.touch(ln)

        read("embedding", (pos % V1) * 4 * D, 4 * D)
        for l in range(L):
            for name, _ in layer_arrays(scheme, smax, D):
                if name in ("K cache", "V cache"):
                    if pos > 0:
                        read(f"{l}/{name}", 0, 4 * pos * D)
                else:
                    read(f"{l}/{name}")
            written += 2 * 4 * D
            if scheme == "buffer":
                written += 2 * 4 * 3 * F1 * D
        read("final gain")
        read("head")
        last = (c.misses - before) * LINE + written
    return last


def work_counts():
    rows = []
    for scheme, label in (("f32", "32-bit weights"), ("group", "8-bit codes, a scale per 32"), ("row", "8-bit codes, a scale per row")):
        rows.append([label, f"{step_flops(scheme):,}", f"{ffn_bytes(scheme):,}", f"{step_flops(scheme) / ffn_bytes(scheme):.4f}"])
    head = (f"One feed-forward block at D = {D1}, F = {F1}, one token. FLOPs: a sum of n products is n multiplications and "
            "n - 1 additions; the norm is 4D + 3 (D squares, D - 1 additions, a division by D, the addition of eps, a square "
            "root, a reciprocal, D products by the reciprocal and D by the gain); the SiLU and the product with up are 5 per "
            "hidden element (exp, add, divide, two products; the negation is exact and free); the residual is D additions. An "
            "8-bit product multiplies each group's partial sum by its scale and adds the groups, which adds one FLOP per group "
            "of each row; turning a code into a number is exact and not counted. Bytes: every weight, code, scale and gain "
            "read once from memory; the vectors of D and F numbers are taken to stay in cache and count 0.\n\n")
    tail = (f"\n\nThe ratio of the bytes of the first two rows is {ffn_bytes('f32') / ffn_bytes('group'):.4f}; the vectors "
            f"left out (x, n, gate, up, h, f, y: {4 * D1 + 3 * F1} numbers, {4 * (4 * D1 + 3 * F1):,} bytes if each were read "
            f"or written once) are {4 * (4 * D1 + 3 * F1) / ffn_bytes('group') * 100:.2f} percent of the 8-bit block's bytes.")
    return head + _table(["weights", "FLOPs", "bytes", "FLOPs per byte"], rows) + tail


def cache_trap():
    rows = []
    for scheme, label in (("f32", "32-bit weights"), ("group", "8-bit codes, a scale per 32")):
        rows.append([label, f"{ffn_bytes(scheme):,}", f"{microbenchmark(scheme):,}", f"{whole_pass(scheme, 512):,}",
                     f"{step_bytes(scheme, 512):,}"])
    mb = microbenchmark("f32") / microbenchmark("group")
    wp = whole_pass("f32", 512) / whole_pass("group", 512)
    ff = ffn_bytes("f32") / ffn_bytes("group")
    head = (f"Bytes read from memory, by a simulated cache of {CACHE_LINES:,} lines of {LINE} bytes (1 MiB), least recently "
            "used line out first; a write goes to memory and counts its own bytes. 'Ten calls' runs one block ten times in a "
            "row on the same weights (gain, then the matrices, codes and scales in the order of the work table, each read "
            f"line by line). 'Whole step' decodes the token at context 512 (positions 0 to 511 cached) through the decoder of "
            f"{L1} layers, after two steps at contexts 510 and 511 that warmed the cache: the embedding row of position "
            f"(position mod {V1}), then per layer the attention gain, W_q, W_k, W_v ({D1} x {D1}, 32-bit), the K and V caches "
            f"({D1} 32-bit numbers per position), W_o and the feed-forward block, with K and V of the new position written "
            f"({2 * 4 * D1:,} bytes), then the final gain and the output head ({V1} x {D1}, 32-bit). Each array has its own "
            "address range starting on a line. 'Counted' is the same step by the rule of the work table.\n\n")
    tail = (f"\n\nRatios of 32-bit to 8-bit: one block's bytes {ff:.4f}, ten calls {mb:.2f}, the whole step {wp:.4f}.")
    return head + _table(["weights", "one block, counted", "ten calls, simulated", "whole step, simulated",
                          "whole step, counted"], rows) + tail


CONTEXTS = (0, 128, 512, 975, 976, 1024, 2048, 8192)
CLAIM_RATIO = 1.4


def crossover():
    """The largest context at which the step's byte ratio is still at least CLAIM_RATIO (both sides grow by 16384 per S)."""
    a, b = step_bytes("f32", 0), step_bytes("group", 0)
    k = step_bytes("f32", 1) - a
    # (a + k S) / (b + k S) >= r  <=>  S <= (a - r b) / (k (r - 1))
    return math.floor((a - CLAIM_RATIO * b) / (k * (CLAIM_RATIO - 1)))


def context_sweep():
    rows = []
    for S in CONTEXTS:
        a, b = step_bytes("f32", S), step_bytes("group", S)
        share = L1 * ffn_bytes("f32") / a
        g = ffn_bytes("f32") / ffn_bytes("group")
        amdahl = 1.0 / ((1.0 - share) + share / g)
        rows.append([f"{S:,}", f"{a:,}", f"{b:,}", f"{a / b:.4f}", f"{share:.4f}", f"{amdahl:.4f}", "yes" if a / b >= CLAIM_RATIO else "no"])
    head = (f"Bytes of one decoding step of the {L1}-layer decoder by the counting rule, against the context S (the number of "
            "cached positions): 32-bit weights everywhere, and the feed-forward blocks in 8-bit codes with a scale per 32. "
            "'FFN share' is the part of the 32-bit step that the feed-forward weights take; the Amdahl column is "
            "1 / ((1 - p) + p / g) with that share p and g the byte ratio of one block (3.5516). 'At least 1.4' compares the "
            "ratio with the claim.\n\n")
    tail = f"\n\nThe largest context at which the ratio is still at least {CLAIM_RATIO} is S = {crossover()}."
    return head + _table(["context S", "32-bit, bytes", "8-bit, bytes", "ratio", "FFN share", "Amdahl", "at least 1.4"],
                         rows) + tail


# perf-number: the two ridges are a historical measurement of one GPU, commit 73466fee0
RIDGE_NOMINAL, RIDGE_MEASURED = 34.0, 36.7     # FLOPs per byte, from the spec sheet and from the measured roofs
BATCHES = (1, 8, 16, 20, 24, 32, 64)


def _bound(flops, nbytes, ridge):
    return max(flops / ridge, nbytes)


def ridge_table():
    rows = []
    for B in BATCHES:
        fa, fb = B * step_flops("f32"), B * step_flops("group")
        ba, bb = ffn_bytes("f32"), ffn_bytes("group")
        rows.append([B, f"{fa / ba:.2f}", f"{fb / bb:.2f}",
                     f"{_bound(fa, ba, RIDGE_NOMINAL) / _bound(fb, bb, RIDGE_NOMINAL):.4f}",
                     f"{_bound(fa, ba, RIDGE_MEASURED) / _bound(fb, bb, RIDGE_MEASURED):.4f}"])
    head = (f"One feed-forward block applied to B tokens at once (the weights read once, the FLOPs of the work table times "
            "B). The cost bound of a kernel is max(FLOPs / ridge, bytes), in units of bytes, where the ridge is the device's "
            "peak FLOP rate divided by its bandwidth; the ratio of the two bounds is the most the 8-bit block can gain. "
            f"Ridges: {RIDGE_NOMINAL} FLOPs per byte from the spec sheet and {RIDGE_MEASURED} from the measured roofs of one "
            "device.\n\n")
    return head + _table(["tokens B", "FLOPs per byte, 32-bit", "FLOPs per byte, 8-bit", "bound ratio, nominal ridge",
                          "bound ratio, measured ridge"], rows)


# ------------------------------------------------------------------ the gate

STEP_EXPECTED_S = 512


def _gate_run(reg):
    """The checks of the specified gate and of a weak one for one regression; True is a pass."""
    x, probe = _inputs()
    gd = _golden()
    golden_present = reg != "no_golden"
    bug = {"swap": "swap", "eps_out": "eps_out", "sum": "sum"}.get(reg)
    pu = _mistake_port(x, bug) if bug else _port(x)
    pp = _mistake_port(probe, bug) if bug else _port(probe)
    scheme = {"buffer": "buffer", "row": "row"}.get(reg, "group")
    checks = {}
    checks["golden"] = golden_present
    checks["unit"] = golden_present and all(im.rel_l2(pu[k], gd[f"unit/{k}"]) <= bound(k) for k in TAPS)
    checks["probe"] = golden_present and all(im.rel_l2(pp[k], gd[f"probe/{k}"]) <= bound(k) for k in TAPS)
    checks["grad"] = _gradcheck("slope" if reg == "slope" else None)[0] < 1e-6
    fast = _int8_run(0, scheme)[0]
    ref = _dequant_reference(0, "row" if scheme == "row" else "group")[0]
    checks["fast"] = im.rel_l2(fast, ref) <= FAST_BOUND
    if scheme == "row":
        upper = _interval_text(_quality_columns()["row"])[3]
    else:
        upper = _interval_text(_quality_columns()["group"])[3]
    checks["quality"] = upper < QUALITY_MARGIN
    checks["bytes"] = step_bytes(scheme, STEP_EXPECTED_S) == step_bytes("group", STEP_EXPECTED_S)
    weak = True if not golden_present else (1.0 - im.cosine(pu["y"], gd["unit/y"]) <= 1e-4)
    return checks, weak


FAST_BOUND = (2 * D1 + F1 + 12) * U64

REGRESSIONS = (("none (the specified system)", None, ""),
               ("codes dequantized into a 32-bit copy of each matrix", "buffer", "work"),
               ("gate and up cut in the wrong order", "swap", "layout"),
               ("eps added after the square root", "eps_out", "constant"),
               ("mean of the squares replaced by their sum", "sum", "scale"),
               ("one scale per row", "row", "granularity"),
               ("SiLU slope taken as sigmoid(gate)", "slope", "derivative"),
               ("golden file missing", "no_golden", "evidence"))


def gate_table():
    rows = []
    for label, reg, cls in REGRESSIONS:
        checks, weak = _gate_run(reg)
        cells = [("not run" if (reg == "no_golden" and k in ("unit", "probe")) else "pass" if checks[k] else "FAIL")
                 for k in ("golden", "unit", "probe", "grad", "fast", "quality", "bytes")]
        rows.append([label, cls, *cells, "pass" if weak else "FAIL"])
    head = ("Seven checks of the gate, and a weak gate beside them, against seven regressions, each made alone on the "
            "specified system. golden: the golden file is present (its absence fails, it never skips); unit and probe: every "
            "tap of the 32-bit port within its bound on the two sets of inputs (not run without a golden); grad: the gradient check below 1e-6; fast: the "
            "8-bit block of draw 0 against the 64-bit block with the dequantized weights s q, rel_l2 of f at most "
            f"(2D + F + 12) 2^-53 = {_sci(FAST_BOUND)} with D = {D1}, F = {F1}; quality: the upper end of the 95% interval of "
            f"the quality table below {QUALITY_MARGIN}; bytes: the counted bytes of one decoding step at context "
            f"{STEP_EXPECTED_S} equal to {step_bytes('group', STEP_EXPECTED_S):,} exactly. The weak gate skips (and reports a "
            "pass) when the golden is missing, and otherwise passes when 1 - cosine of y on the unit inputs is at most "
            "1e-4.\n\n")
    return head + _table(["regression", "defect class", "golden", "unit", "probe", "grad", "fast", "quality", "bytes",
                          "weak gate"], rows, "llllllllll")


# ------------------------------------------------------------------ figure data

def sweep_points():
    """(S, ratio) for the figure of the context sweep."""
    out = []
    S = 0
    while S <= 8192:
        out.append((S, step_bytes("f32", S) / step_bytes("group", S)))
        S += 64
    return out
