# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Transformer Block": attention, a feed-forward network, residuals and norms.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The attention inside the block is the multi-head attention of the previous chapters
(`self_attention.mha_forward` and `mha_backward`, checked there); everything that makes it a block is written here: the
RMS normalization, the SwiGLU and GELU feed-forward networks, the three ways of wiring a sublayer to the stream (pre-norm,
post-norm and no skip), the backward pass of each, a small decoder model and its training. Each public function without
arguments returns a markdown string that the chapter embeds between `<!-- worked: transformer_block.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it
cannot disagree.

* one block on three tokens, step by step, with the stream after every residual add;
* the gradient and the scale along a stack of 32 blocks at initialization, for three ways of wiring a block;
* the width of a gated feed-forward network that keeps the parameters of the plain one, and the gate as a product;
* the parameters and FLOPs of a block for four stated configurations, and the bytes a fused norm saves;
* the forward and backward pass of a whole block, checked against central differences, and three wrong backward passes;
* a gate and an up projection exchanged, an epsilon that is too small to matter, and a decoder trained on a two-hop task.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of
products are plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate).
Numbers of trained models are printed by the stable-digits rule of `stable.py`.

Swedish Embedded AB implements solutions for transformer models, from the block layout to the training loop. If your team
needs expertise in transformer architecture and its gradients, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import self_attention as sa
import stable

EPS = 1e-6   # the epsilon of every RMS normalization in this module

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=3):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _vec(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _round1(x):
    """x rounded to one decimal, ties away from zero (the pseudocode's `round(x, 1)`)."""
    r = math.floor(abs(x) * 10 + 0.5) / 10
    return -r if x < 0 else r


def _bound(x):
    """The power of ten above x, written 1e-7: a difference is reported by the bound it stays under, not by its digits."""
    if x <= 0:
        return "0"
    return f"1e{math.ceil(math.log10(x * 1.0000001)):d}".replace("e-0", "e-")


CRITERION = 1e-6   # a gradient check passes when the difference stays under this


def _under_criterion(x):
    """A gradient-check cell: 'below 1e-6' when the difference meets the criterion, else the difference to two digits. The digits
    of a difference far under the criterion are finite-difference and rounding noise that no other implementation shares."""
    return "below 1e-6" if x < CRITERION else f"{x:.1e}, FAILS"


def _sci(x, d=2):
    """x in scientific notation with d decimals in the mantissa, written 1.23e-4."""
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}e{int(e)}"


def _mean(xs):
    return _sum(xs) / len(xs)


def _geomean(xs):
    return math.exp(_mean([math.log(x) for x in xs]))


_sum = sa._sum
matmul = sa.matmul
transpose = sa.transpose

# ------------------------------------------------------------------------------------------------ the pieces of a block


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def silu(a):
    return a * sigmoid(a)


def dsilu(a):
    s = sigmoid(a)
    return s + a * s * (1.0 - s)


def gelu(z):
    """The exact GELU, z times the standard Gaussian cumulative distribution, through the library erf."""
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))


def dgelu(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) + z * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _rms(row):
    return math.sqrt(_sum([v * v for v in row]) / len(row))


def rms_forward(X, g, eps=EPS):
    """Row-wise RMS normalization y = g * x / sqrt(mean(x^2) + eps). Returns (Y, cache)."""
    d = len(g)
    out, invs = [], []
    for row in X:
        r = 1.0 / math.sqrt(_sum([v * v for v in row]) / d + eps)
        invs.append(r)
        out.append([g[j] * row[j] * r for j in range(d)])
    return out, (X, g, invs)


def rms_backward(dY, cache, bug=None):
    """dx_j = r g_j dy_j - r^3 x_j mean_k(dy_k g_k x_k) and dg_j = sum over rows of dy_j x_j r. `bug` "norm_constant" treats
    the factor r as a constant (the second term is dropped)."""
    X, g, invs = cache
    d = len(g)
    dX, dg = [], [0.0] * d
    for i, row in enumerate(X):
        r, dy = invs[i], dY[i]
        c = 0.0 if bug == "norm_constant" else _sum([dy[j] * g[j] * row[j] for j in range(d)]) / d
        coef = r * r * r * c
        dX.append([r * g[j] * dy[j] - coef * row[j] for j in range(d)])
        for j in range(d):
            dg[j] += dy[j] * row[j] * r
    return dX, dg


def swiglu_forward(X, Wg, Wu, Wd):
    """Y = (silu(X Wg) * (X Wu)) Wd."""
    G, U = matmul(X, Wg), matmul(X, Wu)
    Hd = [[silu(g) * u for g, u in zip(rg, ru)] for rg, ru in zip(G, U)]
    return matmul(Hd, Wd), (X, Wg, Wu, Wd, G, U, Hd)


def swiglu_backward(dY, cache, bug=None):
    """Gradients of the SwiGLU network. `bug` "gate_up_exchanged" differentiates as if the product were silu(up) * gate."""
    X, Wg, Wu, Wd, G, U, Hd = cache
    dHd = matmul(dY, transpose(Wd))
    if bug == "gate_up_exchanged":
        dG = [[dh * silu(u) for dh, u in zip(rh, ru)] for rh, ru in zip(dHd, U)]
        dU = [[dh * g * dsilu(u) for dh, g, u in zip(rh, rg, ru)] for rh, rg, ru in zip(dHd, G, U)]
    else:
        dG = [[dh * u * dsilu(g) for dh, g, u in zip(rh, rg, ru)] for rh, rg, ru in zip(dHd, G, U)]
        dU = [[dh * silu(g) for dh, g in zip(rh, rg)] for rh, rg in zip(dHd, G)]
    Xt = transpose(X)
    dX = _add(matmul(dG, transpose(Wg)), matmul(dU, transpose(Wu)))
    return dX, {"Wg": matmul(Xt, dG), "Wu": matmul(Xt, dU), "Wd": matmul(transpose(Hd), dY)}


def gelu_mlp_forward(X, W1, W2):
    """Y = gelu(X W1) W2, the plain feed-forward network (no biases)."""
    Z = matmul(X, W1)
    A = [[gelu(z) for z in row] for row in Z]
    return matmul(A, W2), (X, W1, W2, Z, A)


def gelu_mlp_backward(dY, cache):
    X, W1, W2, Z, A = cache
    dA = matmul(dY, transpose(W2))
    dZ = [[da * dgelu(z) for da, z in zip(ra, rz)] for ra, rz in zip(dA, Z)]
    return matmul(dZ, transpose(W1)), {"W1": matmul(transpose(X), dZ), "W2": matmul(transpose(A), dY)}


def ffn_forward(p, X):
    """The feed-forward network a parameter dict holds: SwiGLU (W_gate, W_up, W_down) or the GELU network (W1, W2)."""
    if "W1" in p:
        return gelu_mlp_forward(X, p["W1"], p["W2"])
    return swiglu_forward(X, p["Wg"], p["Wu"], p["Wd"])


def block_forward(p, X, layout, H):
    """One block on X [T, D]. `layout` "pre": x1 = x + attn(norm(x)), x2 = x1 + ffn(norm(x1)). "post": x1 = norm(x + attn(x)),
    x2 = norm(x1 + ffn(x1)). "plain": the pre-norm placement without the skip, x1 = attn(norm(x)), x2 = ffn(norm(x1)).
    Attention is causal with H heads, the feed-forward network SwiGLU. Returns (x2, cache) with the intermediates."""
    g1, g2 = p["g1"][0], p["g2"][0]
    c = {"layout": layout, "X": X}
    if layout == "pre":
        c["n1"], c["cn1"] = rms_forward(X, g1)
        c["a"], c["ca"] = sa.mha_forward(c["n1"], p["Wq"], p["Wk"], p["Wv"], p["Wo"], H, 0)
        c["x1"] = _add(X, c["a"])
        c["n2"], c["cn2"] = rms_forward(c["x1"], g2)
        c["f"], c["cf"] = ffn_forward(p, c["n2"])
        c["x2"] = _add(c["x1"], c["f"])
    elif layout == "post":
        c["a"], c["ca"] = sa.mha_forward(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], H, 0)
        c["s1"] = _add(X, c["a"])
        c["x1"], c["cn1"] = rms_forward(c["s1"], g1)
        c["f"], c["cf"] = ffn_forward(p, c["x1"])
        c["s2"] = _add(c["x1"], c["f"])
        c["x2"], c["cn2"] = rms_forward(c["s2"], g2)
    else:
        c["n1"], c["cn1"] = rms_forward(X, g1)
        c["x1"], c["ca"] = sa.mha_forward(c["n1"], p["Wq"], p["Wk"], p["Wv"], p["Wo"], H, 0)
        c["n2"], c["cn2"] = rms_forward(c["x1"], g2)
        c["x2"], c["cf"] = ffn_forward(p, c["n2"])
    return c["x2"], c


def _ffn_back(fn, dY, cache, bug):
    return fn(dY, cache, bug) if fn is swiglu_backward else fn(dY, cache)


def block_backward(c, dY, bug=None):
    """(dX, grads) of one block given dY = dL/dx2. `bug` names a wrong backward pass for the check to catch: "no_identity"
    (the pre-norm block forgets that the second residual add also passes dY to x1), "norm_constant" (both norms treat
    their scale factor as a constant) and "gate_up_exchanged" (see swiglu_backward)."""
    layout = c["layout"]
    ffn_backward = gelu_mlp_backward if len(c["cf"]) == 5 else swiglu_backward
    nbug = "norm_constant" if bug == "norm_constant" else None
    fbug = "gate_up_exchanged" if bug == "gate_up_exchanged" else None
    g = {}
    if layout == "pre":
        df, gf = _ffn_back(ffn_backward, dY, c["cf"], fbug)
        dn2, g["g2"] = rms_backward(df, c["cn2"], nbug)
        dx1 = dn2 if bug == "no_identity" else _add(dY, dn2)
        ga = sa.mha_backward(c["ca"], dx1)
        dn1, g["g1"] = rms_backward(ga["X"], c["cn1"], nbug)
        dX = _add(dx1, dn1)
    elif layout == "post":
        ds2, g["g2"] = rms_backward(dY, c["cn2"], nbug)
        df, gf = _ffn_back(ffn_backward, ds2, c["cf"], fbug)
        dx1 = _add(ds2, df)
        ds1, g["g1"] = rms_backward(dx1, c["cn1"], nbug)
        ga = sa.mha_backward(c["ca"], ds1)
        dX = _add(ds1, ga["X"])
    else:
        df, gf = _ffn_back(ffn_backward, dY, c["cf"], fbug)
        dx1, g["g2"] = rms_backward(df, c["cn2"], nbug)
        ga = sa.mha_backward(c["ca"], dx1)
        dX, g["g1"] = rms_backward(ga["X"], c["cn1"], nbug)
    g.update(gf)
    for k in ("Wq", "Wk", "Wv", "Wo"):
        g[k] = ga[k]
    g["g1"], g["g2"] = [g["g1"]], [g["g2"]]
    return dX, g


def _normal_matrix(rng, rows, cols, scale):
    return [[scale * rng.normal() for _ in range(cols)] for _ in range(rows)]


def init_block(rng, d, h, out_scale=1.0):
    """Weights of one block, drawn in this order, each matrix row by row: W_Q, W_K, W_V, W_O (d by d), W_gate, W_up (d by h) and
    W_down (h by d), every entry `normal() / sqrt(fan_in)` (the variance-preserving rule of Volume III), the two output
    matrices W_O and W_down multiplied by `out_scale`. The two gains start at 1."""
    p = {}
    for name, (r, c), s in (("Wq", (d, d), 1.0), ("Wk", (d, d), 1.0), ("Wv", (d, d), 1.0), ("Wo", (d, d), out_scale),
                            ("Wg", (d, h), 1.0), ("Wu", (d, h), 1.0), ("Wd", (h, d), out_scale)):
        p[name] = _normal_matrix(rng, r, c, s / math.sqrt(r))
    p["g1"], p["g2"] = [[1.0] * d], [[1.0] * d]
    return p


# ------------------------------------------------------------------------------------------------ one block on three tokens

NAME_TINY = "transformer_block.tiny"
TINY_T, TINY_D, TINY_H, TINY_FF = 3, 4, 2, 6   # tokens, width, heads, hidden width of the feed-forward network


@lru_cache(maxsize=None)
def tiny():
    g = bookrng.generator(NAME_TINY)
    X = [[_round1(g.normal()) for _ in range(TINY_D)] for _ in range(TINY_T)]
    p = {}
    for name, (r, c) in (("Wq", (TINY_D, TINY_D)), ("Wk", (TINY_D, TINY_D)), ("Wv", (TINY_D, TINY_D)), ("Wo", (TINY_D, TINY_D)),
                         ("Wg", (TINY_D, TINY_FF)), ("Wu", (TINY_D, TINY_FF)), ("Wd", (TINY_FF, TINY_D))):
        p[name] = [[_round1(0.5 * g.normal()) for _ in range(c)] for _ in range(r)]
    p["g1"], p["g2"] = [[1.0] * TINY_D], [[1.0] * TINY_D]
    out, c = block_forward(p, X, "pre", TINY_H)
    return {"X": X, "p": p, "out": out, "c": c}


def tiny_inputs():
    e = tiny()
    p = e["p"]
    rows = [(f"token {i + 1}", _vec(e["X"][i], 1)) for i in range(TINY_T)]
    wrows = [(f"row {r + 1}", *[_vec(p[n][r], 1) for n in ("Wq", "Wk", "Wv", "Wo")]) for r in range(TINY_D)]
    frows = [(f"row {r + 1}", _vec(p["Wg"][r], 1), _vec(p["Wu"][r], 1)) for r in range(TINY_D)]
    drows = [(f"row {r + 1}", _vec(p["Wd"][r], 1)) for r in range(TINY_FF)]
    return (_seed_line(NAME_TINY) + " Draws, in this order: the 12 entries of X (3 tokens of 4 numbers), each `round(normal(), 1)`; "
            "then W_Q, W_K, W_V, W_O (4 by 4 each), W_gate and W_up (4 by 6 each) and W_down (6 by 4), each row by row and each "
            "`round(0.5 * normal(), 1)` (`round` to one decimal, ties away from zero). Both gains are 1 and the epsilon of the "
            "normalization is 1e-6. The block has H = 2 heads of width 2, a causal mask and no position information, and a "
            "feed-forward network of hidden width 6. The numbers are random stand-ins for a trained model.\n\n"
            + _table(["", "x (4 numbers)"], rows, "ll") + "\n\n"
            + _table(["", "W_Q", "W_K", "W_V", "W_O"], wrows, "lllll") + "\n\n"
            + _table(["", "W_gate (4 by 6)", "W_up (4 by 6)"], frows, "lll") + "\n\n"
            + _table(["", "W_down (6 by 4)"], drows, "ll"))


def _stream_rows(name, M, d=3):
    """Rows of a table that lists a [T, D] stage of the block: its name, the token, the D numbers and the rms of the row."""
    return [(name if i == 0 else "", f"token {i + 1}", *[_fx(v, d) for v in row], _fx(_rms(row), d)) for i, row in enumerate(M)]


_STREAM_HEAD = ["stage", "", "1", "2", "3", "4", "rms"]
_STREAM_ALIGN = "llrrrrr"


def tiny_attention_half():
    e = tiny()
    c = e["c"]
    heads = c["ca"]["heads"]
    t1 = _table(_STREAM_HEAD, _stream_rows("x (the stream, in)", e["X"]) + _stream_rows("n1 = norm(x)", c["n1"]), _STREAM_ALIGN)
    prow = []
    for h, hd in enumerate(heads):
        for i in range(TINY_T):
            prow.append((f"head {h + 1}", f"query {i + 1}", *[_fx(v, 3) for v in hd["P"][i]]))
    t2 = _table(["", "", "key 1", "key 2", "key 3"], prow, "llrrr")
    t3 = _table(_STREAM_HEAD, _stream_rows("a = attention(n1)", c["a"]) + _stream_rows("x1 = x + a", c["x1"]), _STREAM_ALIGN)
    return ("Numbers to 3 decimals; the rms of a row is the square root of the mean of its squares, and the normalization "
            "uses the exact values. The first sublayer: normalize, attend, add the result to the stream.\n\n" + t1
            + "\n\nThe attention weights of the two heads (each row adds to 1, the causal mask hides the keys after the query):\n\n"
            + t2 + "\n\nThe result a is added to the stream:\n\n" + t3)


def tiny_ffn_half():
    e = tiny()
    c = e["c"]
    _, Wg, Wu, Wd, G, U, Hd = c["cf"]
    t1 = _table(_STREAM_HEAD, _stream_rows("x1 (the stream, in)", c["x1"]) + _stream_rows("n2 = norm(x1)", c["n2"]), _STREAM_ALIGN)
    hrow = []
    for i in range(TINY_T):
        hrow.append((f"token {i + 1}", "gate pre-activation", *[_fx(v) for v in G[i]]))
        hrow.append(("", "up", *[_fx(v) for v in U[i]]))
        hrow.append(("", "silu(gate) * up", *[_fx(v) for v in Hd[i]]))
    t2 = _table(["", "", *[f"unit {j + 1}" for j in range(TINY_FF)]], hrow, "ll" + "r" * TINY_FF)
    t3 = _table(_STREAM_HEAD, _stream_rows("f = ffn(n2)", c["f"]) + _stream_rows("x2 = x1 + f (the stream, out)", c["x2"]),
                _STREAM_ALIGN)
    return ("The second sublayer: normalize the stream again, pass each row through the gated feed-forward network, add the "
            "result.\n\n" + t1 + "\n\nInside the network, the six hidden units of each token:\n\n" + t2
            + "\n\nThe result f is added to the stream:\n\n" + t3)


# ------------------------------------------------------------------------------------------------ a stack of 32 blocks at initialization

NAME_DEPTH = "transformer_block.depth"
DEPTH_T, DEPTH_D, DEPTH_H, DEPTH_FF, DEPTH_L, DEPTH_STACKS = 8, 16, 2, 40, 32, 8
DEPTH_VARIANTS = (("pre", 1.0, "pre-norm, skip"), ("pre", None, "pre-norm, skip, output matrices times 1/sqrt(2 L)"),
                  ("post", 1.0, "post-norm, skip"), ("plain", 1.0, "norm before each sublayer, no skip"))
DEPTH_MARKS = (1, 2, 4, 8, 16, 32)


def _depth_stack(s, layout, out_scale):
    """One stack s of DEPTH_L blocks: forward (the rms of the stream after every block) and one backward from the top
    (the Frobenius norm of the gradient at every block input relative to the norm at the top)."""
    rng = bookrng.generator(f"{NAME_DEPTH}.{s}")
    X = _normal_matrix(rng, DEPTH_T, DEPTH_D, 1.0)
    dtop = _normal_matrix(rng, DEPTH_T, DEPTH_D, 1.0)
    scale = 1.0 / math.sqrt(2 * DEPTH_L) if out_scale is None else out_scale
    blocks = [init_block(rng, DEPTH_D, DEPTH_FF, scale) for _ in range(DEPTH_L)]
    x, caches, rms = X, [], [_mean_rms(X)]
    for p in blocks:
        x, c = block_forward(p, x, layout, DEPTH_H)
        caches.append(c)
        rms.append(_mean_rms(x))
    fro = lambda M: math.sqrt(_sum([v * v for row in M for v in row]))   # noqa: E731
    d, norms = dtop, [fro(dtop)]
    for c, p in zip(reversed(caches), reversed(blocks)):
        d, _ = block_backward(c, d)
        norms.append(fro(d))
    ident = None
    if layout == "post":
        d, ident = dtop, [fro(dtop)]
        for c in reversed(caches):
            d, _ = rms_backward(d, c["cn2"])     # the skip paths only: the branches' Jacobians are left out
            d, _ = rms_backward(d, c["cn1"])
            ident.append(fro(d))
        ident = [v / ident[0] for v in ident]
    return {"rms": rms, "grad": [v / norms[0] for v in norms], "ident": ident}   # [k]: after k blocks below the top


def _mean_rms(M):
    """The root mean square of all entries of a [T, D] stage."""
    return math.sqrt(_sum([v * v for row in M for v in row]) / (len(M) * len(M[0])))


@lru_cache(maxsize=None)
def depth_runs():
    out = {}
    for layout, scale, label in DEPTH_VARIANTS:
        runs = [_depth_stack(s, layout, scale) for s in range(DEPTH_STACKS)]
        out[label] = {
            "rms": [_mean([r["rms"][l] for r in runs]) for l in range(DEPTH_L + 1)],
            "grad": [_geomean([r["grad"][k] for r in runs]) for k in range(DEPTH_L + 1)],
        }
        if runs[0]["ident"] is not None:
            out[label]["ident"] = [_geomean([r["ident"][k] for r in runs]) for k in range(DEPTH_L + 1)]
    return out


# ------------------------------------------------------------------------------------------------ a decoder and the two-hop task

VOCAB, N_KEYS = 20, 4   # ids 0 .. 15 are the pair tokens `A<a>B<b>` (id 4 a + b), ids 16 .. 19 the query tokens `Q<x>` (id 16 + x)


def two_hop_example(rng):
    """One example of the two-hop task. Draws, in this order: `shuffle([0, 1, 2, 3], rng)`, the order in which the four
    pairs a -> f(a) appear; then `integer(4)` four times, the values f(0), f(1), f(2), f(3); then `integer(4)`, the asked x.
    Returns (ids, target): the four pair tokens in the shuffled order followed by the query token (T = 5), and f(f(x))."""
    order = rng.shuffle(list(range(N_KEYS)))
    f = [rng.integer(N_KEYS) for _ in range(N_KEYS)]
    x = rng.integer(N_KEYS)
    return [N_KEYS * a + f[a] for a in order] + [N_KEYS * N_KEYS + x], f[f[x]]


def lm_init(name, layers, d, h, layout):
    """The parameters of a decoder with `layers` blocks. Draws from `bookrng.generator(name)`, in this order: the embedding
    table E (VOCAB by d, every entry `normal()`), then each block as in `init_block`, then the head W_C (d by N_KEYS, every
    entry `normal() / sqrt(d)`). The final gain (1) exists for the pre-norm and no-skip layouts only. Residual-branch output
    matrices are multiplied by `1 / sqrt(2 layers)` when `layout` is "pre_scaled"."""
    rng = bookrng.generator(name)
    E = _normal_matrix(rng, VOCAB, d, 1.0)
    scale = 1.0 / math.sqrt(2 * layers) if layout == "pre_scaled" else 1.0
    blocks = [init_block(rng, d, h, scale) for _ in range(layers)]
    Wc = _normal_matrix(rng, d, N_KEYS, 1.0 / math.sqrt(d))
    return {"E": E, "blocks": blocks, "gf": [[1.0] * d] if layout != "post" else None, "Wc": Wc}


BLOCK_KEYS = ("Wq", "Wk", "Wv", "Wo", "g1", "g2", "Wg", "Wu", "Wd")


def lm_tensors(p):
    """The parameter matrices in a fixed order (the same order as the gradients of lm_loss_grad)."""
    out = [p["E"]]
    for b in p["blocks"]:
        out += [b[k] for k in BLOCK_KEYS]
    if p["gf"] is not None:
        out.append(p["gf"])
    out.append(p["Wc"])
    return out


def lm_loss_grad(p, ids, y, layout, H):
    """(loss, gradients in the order of lm_tensors) for one sequence: the softmax cross-entropy of the last position's logits.
    The block layout is "pre", "post" or "plain" ("pre_scaled" is "pre" with a different initialization)."""
    lay = "pre" if layout == "pre_scaled" else layout
    d = len(p["E"][0])
    x = [list(p["E"][t]) for t in ids]
    caches = []
    for b in p["blocks"]:
        x, c = block_forward(b, x, lay, H)
        caches.append(c)
    if p["gf"] is not None:
        xn, cf = rms_forward(x, p["gf"][0])
    else:
        xn = x
    logits = matmul([xn[-1]], p["Wc"])[0]
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = _sum(e)
    pr = [v / z for v in e]
    loss = -math.log(pr[y])
    dlogits = [pr[c] - (1.0 if c == y else 0.0) for c in range(N_KEYS)]
    dxn = [[0.0] * d for _ in ids]
    dxn[-1] = matmul([dlogits], transpose(p["Wc"]))[0]
    dWc = matmul(transpose([xn[-1]]), [dlogits])
    if p["gf"] is not None:
        dx, dgf = rms_backward(dxn, cf)
    else:
        dx = dxn
    gblocks = []
    for b, c in zip(reversed(p["blocks"]), reversed(caches)):
        dx, g = block_backward(c, dx)
        gblocks.append(g)
    gblocks.reverse()
    dE = [[0.0] * d for _ in range(VOCAB)]
    for t, tok in enumerate(ids):
        for j in range(d):
            dE[tok][j] += dx[t][j]
    grads = [dE]
    for g in gblocks:
        grads += [g[k] for k in BLOCK_KEYS]
    if p["gf"] is not None:
        grads.append([dgf])
    grads.append(dWc)
    return loss, grads


def lm_predict(p, ids, layout, H):
    lay = "pre" if layout == "pre_scaled" else layout
    x = [list(p["E"][t]) for t in ids]
    for b in p["blocks"]:
        x, _ = block_forward(b, x, lay, H)
    if p["gf"] is not None:
        x, _ = rms_forward(x, p["gf"][0])
    logits = matmul([x[-1]], p["Wc"])[0]
    return logits


# ------------------------------------------------------------------------------------------------ the gradient check of a whole block

NAME_CHECK = "transformer_block.check"
CHECK_T, CHECK_D, CHECK_H, CHECK_FF, CHECK_CLASSES = 5, 6, 2, 8, 4
CHECK_STEP = 1e-5
CHECK_KEYS = {"swiglu": ("Wq", "Wk", "Wv", "Wo", "g1", "g2", "Wg", "Wu", "Wd", "Wc", "X"),
              "gelu": ("Wq", "Wk", "Wv", "Wo", "g1", "g2", "W1", "W2", "Wc", "X")}


@lru_cache(maxsize=None)
def _check_setup(ffn):
    """T = 5 positions of width 6, 2 heads of width 3, a feed-forward network of hidden width 8 and 4 classes: every size
    different, so that a swapped axis cannot cancel. Draws, in this order: X (5 by 6, each `normal()`), W_Q, W_K, W_V, W_O (6 by 6),
    the two gains (6 numbers each, `1 + 0.1 * normal()`), the feed-forward matrices (SwiGLU: W_gate, W_up 6 by 8 and W_down 8 by 6;
    GELU: W_1 6 by 8 and W_2 8 by 6), W_C (6 by 4), each row by row with `0.5 * normal()` unless stated, then the 5 target classes
    `integer(4)`."""
    rng = bookrng.generator(f"{NAME_CHECK}.{ffn}")
    p = {"X": _normal_matrix(rng, CHECK_T, CHECK_D, 1.0)}
    for k in ("Wq", "Wk", "Wv", "Wo"):
        p[k] = _normal_matrix(rng, CHECK_D, CHECK_D, 0.5)
    p["g1"] = [[1.0 + 0.1 * rng.normal() for _ in range(CHECK_D)]]
    p["g2"] = [[1.0 + 0.1 * rng.normal() for _ in range(CHECK_D)]]
    for k, shape in ((("Wg", (CHECK_D, CHECK_FF)), ("Wu", (CHECK_D, CHECK_FF)), ("Wd", (CHECK_FF, CHECK_D))) if ffn == "swiglu"
                     else (("W1", (CHECK_D, CHECK_FF)), ("W2", (CHECK_FF, CHECK_D)))):
        p[k] = _normal_matrix(rng, shape[0], shape[1], 0.5)
    p["Wc"] = _normal_matrix(rng, CHECK_D, CHECK_CLASSES, 0.5)
    targets = [rng.integer(CHECK_CLASSES) for _ in range(CHECK_T)]
    return p, targets


def _check_forward(p, targets, layout):
    out, c = block_forward(p, p["X"], layout, CHECK_H)
    logits = matmul(out, p["Wc"])
    loss, dl = 0.0, []
    for i, row in enumerate(logits):
        m = max(row)
        e = [math.exp(v - m) for v in row]
        z = _sum(e)
        pr = [v / z for v in e]
        loss += -math.log(pr[targets[i]]) / CHECK_T
        dl.append([(pr[k] - (1.0 if k == targets[i] else 0.0)) / CHECK_T for k in range(CHECK_CLASSES)])
    return loss, c, out, dl


def _check_analytic(p, targets, layout, bug=None):
    _, c, out, dl = _check_forward(p, targets, layout)
    dX, g = block_backward(c, matmul(dl, transpose(p["Wc"])), bug)
    g["Wc"] = matmul(transpose(out), dl)
    g["X"] = dX
    return g


def _check_numeric(p, targets, layout, key):
    num = []
    for r in range(len(p[key])):
        row = []
        for j in range(len(p[key][r])):
            keep = p[key][r][j]
            p[key][r][j] = keep + CHECK_STEP
            hi = _check_forward(p, targets, layout)[0]
            p[key][r][j] = keep - CHECK_STEP
            lo = _check_forward(p, targets, layout)[0]
            p[key][r][j] = keep
            row.append((hi - lo) / (2 * CHECK_STEP))
        num.append(row)
    return num


def _diffs(a, n):
    """(largest absolute difference, largest relative difference over the entries whose gradient is above 1e-4)."""
    worst_abs, worst_rel = 0.0, 0.0
    for ra, rn in zip(a, n):
        for x, y in zip(ra, rn):
            worst_abs = max(worst_abs, abs(x - y))
            if max(abs(x), abs(y)) > 1e-4:
                worst_rel = max(worst_rel, abs(x - y) / max(abs(x), abs(y)))
    return worst_abs, worst_rel


@lru_cache(maxsize=None)
def check_results():
    """{(layout, ffn): {key: (entries, worst abs, worst rel)}}."""
    out = {}
    for layout, ffn in (("pre", "swiglu"), ("post", "swiglu"), ("plain", "swiglu"), ("pre", "gelu")):
        p, targets = _check_setup(ffn)
        ana = _check_analytic(p, targets, layout)
        res = {}
        for key in CHECK_KEYS[ffn]:
            a, n = ana[key], _check_numeric(p, targets, layout, key)
            res[key] = (sum(len(r) for r in a), *_diffs(a, n))
        out[(layout, ffn)] = res
    return out


_CHECK_NAMES = {"Wq": "W_Q", "Wk": "W_K", "Wv": "W_V", "Wo": "W_O", "g1": "gain 1", "g2": "gain 2", "Wg": "W_gate", "Wu": "W_up",
                "Wd": "W_down", "Wc": "W_C", "X": "X", "W1": "W_1", "W2": "W_2"}


def gradient_check():
    res = check_results()
    rows = [(_CHECK_NAMES[k], n, _under_criterion(a), _under_criterion(r)) for k, (n, a, r) in res[("pre", "swiglu")].items()]
    seed_of = {"swiglu": f"{bookrng.seed_of(NAME_CHECK + '.swiglu'):#018x}", "gelu": f"{bookrng.seed_of(NAME_CHECK + '.gelu'):#018x}"}
    summary = []
    for (layout, ffn), r in res.items():
        summary.append((layout, ffn, seed_of[ffn], sum(v[0] for v in r.values()), _under_criterion(max(v[1] for v in r.values())),
                        _under_criterion(max(v[2] for v in r.values()))))
    return (f"Generator: the book's SplitMix64, seeds {seed_of['swiglu']} (`seed_of` of the name "
            f"`{NAME_CHECK}.swiglu`, used by every row with a SwiGLU network: the pre-norm, post-norm and no-skip wirings share these parameters) "
            f"and {seed_of['gelu']} (`{NAME_CHECK}.gelu`, for the GELU network)."
            +f" T = {CHECK_T} positions, D = {CHECK_D}, H = {CHECK_H} heads of width {CHECK_D // CHECK_H}, hidden width {CHECK_FF}, "
            f"{CHECK_CLASSES} classes, causal mask. Draws, in this order: X (5 by 6, each `normal()`), W_Q, W_K, W_V, W_O (6 by 6, each "
            "row by row with `0.5 * normal()`), the two gains (6 numbers each, `1 + 0.1 * normal()`), the feed-forward matrices "
            "(SwiGLU: W_gate and W_up, 6 by 8, then W_down, 8 by 6; GELU: W_1, 6 by 8, then W_2, 8 by 6; `0.5 * normal()`), W_C "
            "(6 by 4, `0.5 * normal()`), then the 5 target classes `integer(4)`. No matrix is scaled by 1/sqrt(2 L) here: W_O and W_down "
            "are used as drawn. The model is one block (RMS norm, "
            "epsilon 1e-6), then the classifier logits = x2 W_C at every position, and the loss is the mean over positions of the "
            f"softmax cross-entropy. Central differences with step {CHECK_STEP:g} on every entry of every parameter and of the input X. "
            "A cell says 'below 1e-6' when the largest difference meets the criterion of 1e-6; the digits of a difference that small "
            "are noise of the finite difference and are not printed. The absolute difference is |a - n|; the relative difference "
            "|a - n| / max(|a|, |n|) is taken over the entries whose gradient is above 1e-4.\n\n"
            + _table(["tensor (pre-norm SwiGLU block)", "entries checked", "largest absolute difference", "largest relative difference"], rows)
            + "\n\nThe same check on every wiring and both feed-forward networks, all tensors together:\n\n"
            + _table(["wiring", "feed-forward network", "seed", "entries checked", "largest absolute difference",
                      "largest relative difference"], summary, "lllrrr"))


BUGS = (("the block's backward pass", None),
        ("second residual add passes no gradient to the stream", "no_identity"),
        ("each norm differentiated with its scale held constant", "norm_constant"),
        ("gate and up exchanged in the SwiGLU backward", "gate_up_exchanged"))


@lru_cache(maxsize=None)
def wrong_results():
    p, targets = _check_setup("swiglu")
    num = {key: _check_numeric(p, targets, "pre", key) for key in CHECK_KEYS["swiglu"] if key != "Wc"}
    out = {}
    for label, bug in BUGS:
        ana = _check_analytic(p, targets, "pre", bug)
        out[label] = {key: _diffs(ana[key], num[key])[1] for key in num}
    return out


def wrong_backward_passes():
    res = wrong_results()
    keys = [k for k in CHECK_KEYS["swiglu"] if k != "Wc"]

    def cell(v):
        return "below 1e-6" if v < CRITERION else f"about {v:.1f}"

    rows = [(label, *[cell(res[label][k]) for k in keys]) for label, _ in BUGS]
    return ("The check of the previous block (same parameters, same step, pre-norm SwiGLU block) run on the correct backward pass and "
            "on three wrong ones. Entries: the largest relative difference |a - n| / max(|a|, |n|) over the entries of the tensor "
            "whose gradient is above 1e-4; a difference of 1e-6 or more is written to one decimal, a smaller one as 'below 1e-6' "
            "(a tensor that the wrong step does not touch keeps its correct gradient).\n\n"
            + _table(["backward pass", *[_CHECK_NAMES[k] for k in keys]], rows))


def depth_setup():
    return (f"{DEPTH_STACKS} random stacks of {DEPTH_L} blocks, T = {DEPTH_T} positions of width D = {DEPTH_D}, H = {DEPTH_H} heads of width "
            f"{DEPTH_D // DEPTH_H}, SwiGLU hidden width {DEPTH_FF}, causal mask, RMS norm with epsilon 1e-6 and both gains 1. Stack s "
            f"draws from `{NAME_DEPTH}.<s>` for s = 0 .. {DEPTH_STACKS - 1} (the seed is `seed_of` of that name; for s = 0 it is "
            f"{bookrng.seed_of(NAME_DEPTH + '.0'):#018x}), in this order: the input X ({DEPTH_T} by {DEPTH_D}, each `normal()`), the "
            f"gradient arriving at the top of the stack (same shape, each `normal()`), then the {DEPTH_L} blocks, each drawn as "
            "W_Q, W_K, W_V, W_O (D by D), W_gate, W_up (D by hidden width) and W_down (hidden width by D), row by row, every entry "
            "`normal() / sqrt(fan_in)` (the fan-in is the number of rows of the matrix). The variant with scaled output matrices "
            f"multiplies W_O and W_down by 1 / sqrt(2 * {DEPTH_L}) = {1 / math.sqrt(2 * DEPTH_L):.4f}; it uses the same draws.")


def depth_scale():
    r = depth_runs()
    rows = []
    for _, _, label in DEPTH_VARIANTS:
        rows.append((label, *[_fx(r[label]["rms"][l], 2) for l in DEPTH_MARKS]))
    pred = [_fx(math.sqrt(1 + l), 2) for l in DEPTH_MARKS]
    return ("The root mean square of all entries of the stream after k blocks (the mean over the stacks), for the input of rms "
            f"{_fx(r[DEPTH_VARIANTS[0][2]]['rms'][0], 2)}. " + depth_setup() + "\n\n"
            + _table(["wiring", *[f"k = {k}" for k in DEPTH_MARKS]], rows + [("sqrt(1 + k), for comparison", *pred)]))


def depth_gradient():
    r = depth_runs()
    rows = []
    for _, _, label in DEPTH_VARIANTS:
        rows.append((label, *[_sci(r[label]["grad"][k], 2) for k in DEPTH_MARKS]))
        if "ident" in r[label]:
            rows.append((label + ", skip paths only", *[_sci(r[label]["ident"][k], 2) for k in DEPTH_MARKS]))
    rows.append(("pre-norm, skip paths only (exactly the identity)", *["1.00e0" for _ in DEPTH_MARKS]))
    return ("The Frobenius norm of the gradient that reaches the input of the block k below the top, divided by the norm of the "
            f"gradient put in at the top (the geometric mean over the {DEPTH_STACKS} stacks of the previous table). 'Skip paths only' "
            "backpropagates through the norms and the additions and leaves out the Jacobians of attention and of the feed-forward "
            "network: what travels along the identity path alone.\n\n"
            + _table(["wiring", *[f"k = {k}" for k in DEPTH_MARKS]], rows))


def depth_data():
    """Per wiring: the stream rms for k = 0 .. DEPTH_L and the gradient ratio for k = 0 .. DEPTH_L (for the figures)."""
    return depth_runs()


# ------------------------------------------------------------------------------------------------ the feed-forward network

def ffn_widths():
    """Hidden width F = 8 D / 3 of a SwiGLU network that has the parameters of the GELU network of hidden width 4 D, and the
    width that practice rounds it to (up to a multiple of 256)."""
    rows = []
    for D in (768, 1024, 2048, 4096):
        gelu_params = 2 * D * 4 * D
        exact = 8 * D / 3
        rounded = -(-math.ceil(exact) // 256) * 256
        gated = 3 * D * rounded
        rows.append((D, _int(4 * D), _int(gelu_params), _fx(exact, 2), _int(rounded), _int(gated), _fx(gated / gelu_params, 4)))
    return ("Parameters of the feed-forward network without biases: the GELU network has two matrices D by 4 D, 2 * D * 4 D = 8 D^2 "
            "numbers; the SwiGLU network has three matrices D by F, 3 D F numbers, equal to 8 D^2 at F = 8 D / 3. The rounded width is "
            "the exact one rounded up to a multiple of 256.\n\n"
            + _table(["D", "GELU hidden width 4 D", "GELU parameters 8 D^2", "exact F = 8 D / 3", "F rounded up", "SwiGLU parameters 3 D F",
                      "SwiGLU / GELU"], rows))


GATE_POINTS = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


def gate_functions():
    rows = [(_fx(g, 1), _fx(sigmoid(g), 4), _fx(gelu(g), 4), _fx(silu(g), 4)) for g in GATE_POINTS]
    return ("The three gate functions at the pre-activation g, each followed by the multiplication with the up projection u in "
            "the unit: GLU gives sigmoid(g) u, GEGLU gives gelu(g) u and SwiGLU gives silu(g) u. sigmoid is the logistic function, "
            "gelu the exact Gaussian form (library erf), silu(g) = g sigmoid(g).\n\n"
            + _table(["g", "sigmoid(g)", "gelu(g)", "silu(g)"], rows))


PRODUCT_EPS = 0.1
PRODUCT_POINTS = ((1.0, 1.0), (1.0, -1.0), (-2.0, 1.5), (0.5, 3.0), (-1.0, -1.0), (3.0, 0.0))


def product_unit():
    """One SwiGLU hidden unit with W_gate = (eps, 0), W_up = (0, 1) and W_down = 2 / eps: silu(eps x1) x2 * 2 / eps."""
    e = PRODUCT_EPS
    rows = []
    for x1, x2 in PRODUCT_POINTS:
        out = silu(e * x1) * x2 * 2.0 / e
        rows.append((f"({_fx(x1, 1)}, {_fx(x2, 1)})", _fx(x1 * x2, 4), _fx(out, 4), _fx(out - x1 * x2, 4)))
    return ("One hidden unit of a SwiGLU network with two inputs: W_gate = (0.1, 0), W_up = (0, 1), W_down = 2 / 0.1 = 20, so that "
            "the output is 20 silu(0.1 x1) x2. The product x1 x2 is what the unit approximates.\n\n"
            + _table(["(x1, x2)", "x1 x2", "unit output", "difference"], rows))


def gate_up_exchanged():
    """The tiny block with W_gate and W_up exchanged: nothing fails, the result is another function."""
    e = tiny()
    p = dict(e["p"])
    p["Wg"], p["Wu"] = p["Wu"], p["Wg"]
    out, c = block_forward(p, e["X"], "pre", TINY_H)
    c0 = e["c"]
    rows = []
    for i in range(TINY_T):
        df = _sum([(a - b) ** 2 for a, b in zip(c["f"][i], c0["f"][i])]) ** 0.5
        rows.append((f"token {i + 1}", _fx(_rms(c0["f"][i]), 3), _fx(_rms(c["f"][i]), 3), _fx(df / math.sqrt(TINY_D), 3),
                     _fx(_sum([(a - b) ** 2 for a, b in zip(out[i], e["out"][i])]) ** 0.5 / math.sqrt(TINY_D) / _rms(e["out"][i]), 3)))
    return ("The block of the first section with W_gate and W_up exchanged. Both shapes are 4 by 6, so every product is defined and "
            "nothing fails. rms of f is the root mean square of the feed-forward result of that token; 'rms of the change' is the "
            "root mean square of the difference of the two f; the last column is that difference of the output x2 as a fraction "
            "of the rms of the original x2.\n\n"
            + _table(["token", "rms of f, original", "rms of f, exchanged", "rms of the change in f", "change in x2, fraction of its rms"],
                     rows))


def epsilon_effect():
    """The same direction at three scales through an RMS norm at epsilon 1e-6 and 1e-5."""
    base = [1.0, -2.0, 3.0, 0.5]
    rows = []
    for scale in (1.0, 1e-2, 1e-3, 1e-4):
        x = [scale * v for v in base]
        n6, _ = rms_forward([x], [1.0] * 4, 1e-6)
        n5, _ = rms_forward([x], [1.0] * 4, 1e-5)
        rows.append((_sci(scale, 0), _sci(_rms(x), 2), _fx(n6[0][0], 4), _fx(n5[0][0], 4), _fx(n6[0][0] / n5[0][0], 4)))
    return ("The vector (1, -2, 3, 0.5) times a scale, normalized with gain 1 at epsilon 1e-6 and at epsilon 1e-5. First entry of "
            "the result, and the ratio of the two.\n\n"
            + _table(["scale", "rms of the input", "first entry, eps 1e-6", "first entry, eps 1e-5", "ratio"], rows))


# ------------------------------------------------------------------------------------------------ what a block costs

# (label, D, H, kind of feed-forward network, hidden width, T): four stated configurations
COSTS = (("A: GELU, 4 D", 768, 12, "gelu", 3072, 1024),
         ("B: SwiGLU", 1024, 16, "swiglu", 3072, 2048),
         ("C: SwiGLU", 4096, 32, "swiglu", 11008, 4096),
         ("D: as C, long", 4096, 32, "swiglu", 11008, 32768))
GELU_FLOPS_PER_ENTRY, SWIGLU_FLOPS_PER_ENTRY = 5, 4   # the elementwise work of the activation per hidden entry


def block_parameters(D, kind, hidden):
    """(attention, feed-forward network, norm gains) of one block, no biases: 4 D^2; 2 D h or 3 D h; 2 D."""
    return 4 * D * D, (2 if kind == "gelu" else 3) * D * hidden, 2 * D


def block_flops(D, H, kind, hidden, T):
    """The forward FLOPs of one block on a sequence of T tokens, by the counting rule of the previous chapters: a product of an
    m by k and a k by n matrix is 2 m k n; softmax and scale 5 FLOPs per score; an RMS norm of a row (4 D + 3); a residual add D;
    the activation 5 (GELU) or 4 (silu and the product) FLOPs per hidden entry. Returns the dict of the parts."""
    ffn_products = (2 if kind == "gelu" else 3) * 2 * T * D * hidden
    return {"projections": 8 * T * D * D, "scores": 4 * T * T * D, "softmax": 5 * H * T * T, "ffn": ffn_products,
            "other": 2 * T * (4 * D + 3) + 2 * T * D + T * hidden * (GELU_FLOPS_PER_ENTRY if kind == "gelu" else SWIGLU_FLOPS_PER_ENTRY)}


def accounting_data():
    out = []
    for label, D, H, kind, hidden, T in COSTS:
        pa, pf, pn = block_parameters(D, kind, hidden)
        out.append({"label": label, "D": D, "H": H, "kind": kind, "hidden": hidden, "T": T, "params": (pa, pf, pn),
                    "flops": block_flops(D, H, kind, hidden, T)})
    return out


def block_parameters_table():
    rows = []
    seen = {}
    for e in accounting_data():
        shape = (e["D"], e["H"], e["kind"], e["hidden"])
        if shape in seen:   # the same block at another sequence length: same parameters
            seen[shape].append(e["label"].split(":")[0])
            continue
        seen[shape] = [e["label"].split(":")[0]]
        pa, pf, pn = e["params"]
        tot = pa + pf + pn
        rows.append((e["label"], e["D"], e["H"], _int(e["hidden"]), _int(pa), _int(pf), _int(pn), _int(tot),
                     _fx(pa / tot * 100, 1), _fx(pf / tot * 100, 1)))
    same = [f"configuration {names[1]} has the block of {names[0]}" for names in seen.values() if len(names) > 1]
    return ("Parameters of one block, no biases: attention 4 D^2 (the four D by D matrices), feed-forward network 2 D F (GELU) or 3 D F "
            "(SwiGLU), normalization gains 2 D. The " + "; the ".join(same) + ", so it has no row of its own here (only the sequence length "
            "of the next table differs).\n\n"
            + _table(["configuration", "D", "heads H", "hidden width F", "attention", "feed-forward", "norm gains", "block total",
                      "attention %", "feed-forward %"], rows))


def block_flops_table():
    rows = []
    for e in accounting_data():
        f = e["flops"]
        tot = _sum([float(v) for v in f.values()])
        rows.append((f'{e["label"]}, T = {_int(e["T"])}', _int(f["projections"]), _int(f["scores"] + f["softmax"]), _int(f["ffn"]), _int(f["other"]),
                     _int(int(tot)), _fx((f["projections"] + f["scores"] + f["softmax"]) / tot * 100, 1), _fx(f["ffn"] / tot * 100, 1)))
    return ("Forward FLOPs of one block on one sequence of T tokens (the counting rule of the previous chapters: a product of an m by k "
            "and a k by n matrix is 2 m k n; scale and softmax 5 FLOPs per score; an RMS norm of one row 4 D + 3; a residual add D; "
            "the activation 5 (GELU) or 4 (SiLU and the product) FLOPs per hidden entry).\n\n"
            + _table(["configuration", "projections 8 T D^2", "scores, average, softmax", "feed-forward products",
                      "norms, adds, activation", "block total", "attention %", "FFN %"], rows))


def share_data():
    """For the figure: per configuration the parameter shares and the FLOP shares of (projections, scores+softmax, ffn, other)."""
    out = []
    for e in accounting_data():
        pa, pf, pn = e["params"]
        pt = pa + pf + pn
        f = e["flops"]
        tt = float(sum(f.values()))
        out.append((e["label"], e["T"], [pa / pt, pf / pt, pn / pt],
                    [f["projections"] / tt, (f["scores"] + f["softmax"]) / tt, f["ffn"] / tt, f["other"] / tt]))
    return out


GPT_VOCAB, GPT_CONTEXT, GPT_D, GPT_LAYERS = 50257, 1024, 768, 12


def decoder_parameters():
    """The parameters of a decoder: GPT-style blocks with biases and LayerNorm (the parameter list of brain's `gpt2` crate)
    and the same decoder with the RMSNorm / SwiGLU block of configuration B."""
    D, L = GPT_D, GPT_LAYERS
    gpt_block = {"two LayerNorms (gain and shift)": 4 * D, "attention q, k, v (weights and biases)": 3 * D * D + 3 * D,
                 "attention output (weights and bias)": D * D + D, "feed-forward fc (weights and bias)": D * 4 * D + 4 * D,
                 "feed-forward proj (weights and bias)": 4 * D * D + D}
    per_block = sum(gpt_block.values())
    emb = GPT_VOCAB * D
    pos = GPT_CONTEXT * D
    final = 2 * D
    tied = emb + pos + L * per_block + final
    rows = [(k, _int(v), _int(L * v)) for k, v in gpt_block.items()]
    rows.append(("one block", _int(per_block), _int(L * per_block)))
    parts = _table(["part of a block, D = 768", "parameters", "times 12 blocks"], rows)
    tail = _table(["part of the decoder", "parameters"], [
        ("token embedding, 50,257 by 768", _int(emb)), ("learned positions, 1,024 by 768", _int(pos)),
        ("12 blocks", _int(L * per_block)), ("final LayerNorm", _int(final)), ("total with the head tied to the embedding", _int(tied)),
        ("separate output matrix, 50,257 by 768", _int(emb)), ("total with a separate head", _int(tied + emb))], "lr")
    return ("A decoder with D = 768, 12 blocks, a vocabulary of 50,257 tokens and 1,024 learned positions, built from the block of the "
            "first sections with biases and LayerNorm (the layout of `crates/gpt2`).\n\n" + parts + "\n\n" + tail)


def fusion_bytes():
    """Memory traffic of the unfused and the fused passes around a norm and a SwiGLU, per element, and for a stated shape."""
    rows_n, D = 4096, 4096
    n = rows_n * D
    steps = (("residual add, then RMS norm: add (2 reads, 1 write), norm (1 read, 1 write)", 3, 2, "add + norm, fused (2 reads, 2 writes)", 2, 2),
             ("SwiGLU backward: d_gate (3 reads, 1 write), d_up (2 reads, 1 write)", 5, 2, "SwiGLU backward, one pass (3 reads, 2 writes)", 3, 2))
    rows = []
    for a, ra, wa, b, rb, wb in steps:
        before, after = (ra + wa) * n * 4, (rb + wb) * n * 4
        rows.append((a, _int(before), b, _int(after), _fx((1 - after / before) * 100, 1)))
    return (f"A tensor of {rows_n:,} rows of D = {D:,} numbers, {_int(n)} entries of 4 bytes (float32). Bytes moved to and from memory, "
            "counting every read and every write of a full tensor once and leaving out the gain vector; the fused add + norm "
            "re-reads the sum it has just written from cache.\n\n"
            + _table(["separate passes", "bytes", "one pass", "bytes", "bytes saved %"], rows, "lrlrr"))


# ------------------------------------------------------------------------------------------------ training a decoder on the two-hop task

TRAIN_D, TRAIN_H, TRAIN_FF = 16, 2, 24
# (blocks L, wiring, learning rate, label): the wirings of the previous sections trained at the rate 0.01, and at 0.003 where it matters
TRAIN_CONFIGS = ((2, "pre", 0.01, "pre-norm, skip"), (2, "post", 0.01, "post-norm, skip"),
                 (2, "plain", 0.01, "norm before each sublayer, no skip"),
                 (6, "pre", 0.01, "pre-norm, skip"), (6, "pre_scaled", 0.01, "pre-norm, skip, output matrices times 1/sqrt(2 L)"),
                 (6, "post", 0.01, "post-norm, skip"), (6, "plain", 0.01, "norm before each sublayer, no skip"),
                 (6, "pre", 0.003, "pre-norm, skip"), (6, "post", 0.003, "post-norm, skip"))
BETA1, BETA2, EPS_ADAM, BATCH, STEPS, WINDOW, N_TEST = 0.9, 0.999, 1e-8, 16, 400, 20, 400
CHECKPOINTS = (50, 100, 200, 400)
NAME_TRAIN_DATA, NAME_TRAIN_TEST = "transformer_block.train.data", "transformer_block.train.test"


def two_hop_ceiling():
    """The best accuracy of a predictor that knows f(x) and which values occur among the four pairs (but not which pair holds
    which), by enumeration of all 4^4 functions and 4 queries: the most such a model can reach without composing the two hops."""
    groups = {}
    total = 0
    for code in range(N_KEYS ** N_KEYS):
        f = [(code // N_KEYS ** a) % N_KEYS for a in range(N_KEYS)]
        for x in range(N_KEYS):
            counts = groups.setdefault((f[x], tuple(sorted(f))), [0] * N_KEYS)
            counts[f[f[x]]] += 1
            total += 1
    return _sum([float(max(c)) for c in groups.values()]) / total


def _train_lm(job):
    layers, layout, lr, nudge = job
    name = f"transformer_block.train.init.{layout}.{layers}"
    p = lm_init(name, layers, TRAIN_D, TRAIN_FF, layout)
    p["E"][0][0] += nudge
    tensors = lm_tensors(p)
    m = [[[0.0] * len(r) for r in t] for t in tensors]
    v = [[[0.0] * len(r) for r in t] for t in tensors]
    rng = bookrng.generator(NAME_TRAIN_DATA)
    losses = []
    for step in range(1, STEPS + 1):
        acc, total = None, 0.0
        for _ in range(BATCH):
            ids, y = two_hop_example(rng)
            loss, grads = lm_loss_grad(p, ids, y, layout, TRAIN_H)
            total += loss / BATCH
            if acc is None:
                acc = [[[x / BATCH for x in r] for r in t] for t in grads]
            else:
                for a, t in zip(acc, grads):
                    for ra, rt in zip(a, t):
                        for j, x in enumerate(rt):
                            ra[j] += x / BATCH
        losses.append(total)
        for ti, t in enumerate(tensors):
            for ri, r in enumerate(t):
                for j in range(len(r)):
                    gv = acc[ti][ri][j]
                    m[ti][ri][j] = BETA1 * m[ti][ri][j] + (1 - BETA1) * gv
                    v[ti][ri][j] = BETA2 * v[ti][ri][j] + (1 - BETA2) * gv * gv
                    r[j] -= lr * (m[ti][ri][j] / (1 - BETA1 ** step)) / (math.sqrt(v[ti][ri][j] / (1 - BETA2 ** step)) + EPS_ADAM)
    curve = {c: _mean(losses[c - WINDOW:c]) for c in CHECKPOINTS}
    rng_t = bookrng.generator(NAME_TRAIN_TEST)
    right = 0
    for _ in range(N_TEST):
        ids, y = two_hop_example(rng_t)
        logits = lm_predict(p, ids, layout, TRAIN_H)
        right += 1 if max(range(N_KEYS), key=lambda c: (logits[c], -c)) == y else 0
    return {"layout": layout, "layers": layers, "lr": lr, "nudge": nudge, "curve": curve, "right": right,
            "params": sum(sum(len(r) for r in t) for t in tensors)}


@lru_cache(maxsize=None)
def train_results():
    """{(layers, wiring, rate): (result, [nudged results])}; the trainings are spread over four worker processes."""
    jobs = [(L, lay, lr, d) for L, lay, lr, _ in sorted(TRAIN_CONFIGS, key=lambda c: -c[0]) for d in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_train_lm, jobs, chunksize=1)
    out = {}
    for r in res:
        entry = out.setdefault((r["layers"], r["layout"], r["lr"]), [None, []])
        if r["nudge"] == 0.0:
            entry[0] = r
        else:
            entry[1].append(r)
    return out


def _stable_cell(key, f, digits):
    base, twins = train_results()[key]
    return stable.text(f(base), [f(t) for t in twins], lambda x: _fx(x, digits))


def two_hop_task():
    rng = bookrng.generator("transformer_block.train.examples")
    rows = []
    for _ in range(3):
        ids, y = two_hop_example(rng)
        names = [f"`A{(t // N_KEYS)}B{t % N_KEYS}`" for t in ids[:-1]] + [f"`Q{ids[-1] - N_KEYS * N_KEYS}`"]
        rows.append((" ".join(names), f"`{y}`"))
    q = N_KEYS * N_KEYS
    ceiling = two_hop_ceiling()
    params = {L: sum(sum(len(r) for r in t) for t in lm_tensors(lm_init("transformer_block.count", L, TRAIN_D, TRAIN_FF, "pre")))
              for L in (2, 6)}
    return ("Ids 0 .. 15 are the pair tokens `A<a>B<b>` (id 4 a + b: the pair a -> b with a and b in 0 .. 3) and ids 16 .. 19 are the "
            "query tokens `Q<x>` (id 16 + x), so the vocabulary has V = 20 ids. An example is a random function f from {0, 1, 2, 3} "
            "to {0, 1, 2, 3} written as four pairs, and a query. Draws, in this order: `shuffle([0, 1, 2, 3], rng)`, the order in "
            "which the four pairs appear; `integer(4)` four times, the values f(0), f(1), f(2), f(3); `integer(4)`, the asked x. The "
            "input is the four pair tokens in that order followed by the query token (T = 5); the target is f(f(x)). Three examples "
            f"from `transformer_block.train.examples` ({bookrng.seed_of('transformer_block.train.examples'):#018x}):\n\n"
            + _table(["input (T = 5 tokens)", "target f(f(x))"], rows, "ll")
            + "\n\nA model that guesses has loss ln 4 = " + _fx(math.log(4), 4) + " nats and accuracy 0.25. A model that finds f(x) and sees which "
            "values occur among the pairs, but cannot compose the two lookups, can reach at most "
            + _fx(ceiling, 5) + " (enumerated over all 256 functions and 4 queries). The decoder is the one of the last section: an "
            f"embedding table E [20, {TRAIN_D}], L blocks of width {TRAIN_D} with {TRAIN_H} heads, SwiGLU hidden width {TRAIN_FF} and "
            "a causal mask, no position information, a final gain-only RMS norm (pre-norm and no-skip wirings; the post-norm block "
            f"ends in a norm already), and a head W_C [{TRAIN_D}, 4] on the last position. Parameters: {_int(params[2])} for L = 2, "
            f"{_int(params[6])} for L = 6 (the gains and the output matrices differ by wiring only in their values).")


def _count_cell(good, total):
    """The number of runs that end above 0.9: 'none' and 'all' survive the nudged copies, anything between is a split."""
    if good == 0:
        return f"none of {total}"
    if good == total:
        return f"all {total}"
    return f"{good} of {total}, mixed"


def two_hop_training():
    rows = []
    for L, lay, lr, label in TRAIN_CONFIGS:
        key = (L, lay, lr)
        cells = [_stable_cell(key, lambda r, c=c: r["curve"][c], 4) for c in CHECKPOINTS]
        acc = _stable_cell(key, lambda r: r["right"] / N_TEST, 3)
        base, twins = train_results()[key]
        good = sum(1 for r in [base, *twins] if r["right"] / N_TEST > 0.9)
        rows.append((L, label, lr, f"{bookrng.seed_of(f'transformer_block.train.init.{lay}.{L}'):#018x}", *cells, acc,
                     _count_cell(good, 1 + len(twins))))
    d, f_, v = TRAIN_D, TRAIN_FF, VOCAB
    return (
        f"Initialization. The parameters of a model are drawn from `transformer_block.train.init.<wiring>.<L>` (wiring names pre, "
        f"pre_scaled, post, plain; seeds in the table; the two rates of one wiring start from the same parameters), every matrix row by "
        f"row, in this order: the embedding table E [{v}, {d}], each entry `normal()` (no scale); then, for each block in turn, W_Q, W_K, W_V, "
        f"W_O ({d} by {d}) and W_gate, W_up ({d} by {f_}), each entry `normal() / sqrt({d})`, and W_down ({f_} by {d}), each entry "
        f"`normal() / sqrt({f_})` (the square root of the number of rows of the matrix); then, after the last block, the head W_C ({d} by "
        f"{N_KEYS}), each entry `normal() / sqrt({d})`. The norm gains are not drawn: the two gains of every block and the final gain "
        f"all start at 1. In the wiring pre_scaled, W_O and W_down of every block are multiplied by 1/sqrt(2 L) = 0.2887 (L = 6) after "
        f"they are drawn and scaled. Every RMS norm has epsilon 1e-6 and the gain-only final norm exists in the pre_scaled, pre and no-skip "
        f"wirings, not in post-norm. The read-out is logits = x W_C, with x the final stream row of the last position (no bias), and the "
        f"loss of an example is the softmax cross-entropy -ln p[target] of these {N_KEYS} logits. Only the last position has a loss.\n\n"
        f"Training. An example is a sequence of T = 5 ids processed on its own (no padding, causal mask, no position information). "
        f"Every model reads the generator `{NAME_TRAIN_DATA}` ({bookrng.seed_of(NAME_TRAIN_DATA):#018x}) from its start with the draw order of the "
        f"task above, one example after another: step s of {STEPS} takes the examples numbered {BATCH} (s - 1) to {BATCH} s - 1 of that stream "
        f"(counting from 0), so all models see the same examples. "
        f"The gradient of a step is the mean of the {BATCH} example gradients (each divided by {BATCH} and added in example order), computed "
        f"with the parameters before the update; the loss of the step is the mean of the {BATCH} example losses of that same pass. "
        f"The optimizer is Adam on every tensor, gains included, no weight decay, no clipping, no warmup, the rate constant in the "
        f"table: m <- {BETA1} m + {1 - BETA1:.1f} g, v <- {BETA2} v + {1 - BETA2:.3f} g^2 (both zero at the start), and at step s = 1 .. {STEPS} "
        f"p <- p - rate (m / (1 - {BETA1}^s)) / (sqrt(v / (1 - {BETA2}^s)) + {EPS_ADAM:g}). "
        f"A loss column is the mean of the {WINDOW} step losses of the steps s - {WINDOW - 1} .. s. Accuracy is measured after step {STEPS}: "
        f"the fraction of the {N_TEST} test examples (the first {N_TEST} draws of `{NAME_TRAIN_TEST}`, {bookrng.seed_of(NAME_TRAIN_TEST):#018x}) "
        f"whose largest logit (the lowest id on a tie) is the target.\n\n"
        "The last column counts, among the run and its four copies, those with an accuracy above 0.9. 'None' and 'all' are outcomes "
        "that survive the copies; 'mixed' means the run sits at a fork, the five copies split, and a reimplementation should expect its "
        "own copies to split too, with a count of its own: only the none and all cells are claims about every implementation. "
        "The first weight of the nudged copies is E[0][0] (row 0, column 0 of the embedding table, the first number of the first tensor "
        "drawn), the first coordinate of the token `A0B0`, which stands in about a quarter of the examples and so enters the computation in "
        "the first steps. " + stable.RULE + "\n\n"
        + _table(["L", "wiring", "rate", "seed of the initial parameters"] + [f"loss, step {c}" for c in CHECKPOINTS] + ["test accuracy", "runs above 0.9"],
                 rows, "llrr" + "r" * 6))
