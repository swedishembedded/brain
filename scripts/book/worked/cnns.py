# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Convolutional Networks and Residual Learning".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: cnns.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* two 8 by 8 images with the same pixels in a different arrangement, and what two difference filters say about them;
* max, average and global average pooling with their backward passes, including the tie rule and overlapping windows;
* a toy image set (stripes, checkerboards, blobs) drawn from a stated rule and seed, and a small convolutional network for it:
  the parameters and multiply-adds of each layer, the receptive field, a gradient check, an Adam run checked for the digits that
  survive a change of 1e-12 in one weight, and what a shift of the input does to the output;
* a plain and a residual stack of convolutions at several depths on the same data, trained the same way;
* the cost of basic and bottleneck blocks, the shortcut that matches shapes, and the parameter and multiply-add counts of
  ResNet-18 and ResNet-50;
* batch normalization folded into the convolution before it (and the border that appears when it is folded into the one after);
* a feature pyramid on a stated input: the shapes, parameters and multiply-adds of the top-down and bottom-up paths.

Arithmetic is the language's 64-bit floating point unless a table says otherwise. Random inputs come from `bookrng.generator(name)`
with the names, seeds and draw orders printed beside each table. Sums over taps run in the order (input channel, kernel row, kernel
column), over positions in row-major order and over images in index order; the digits of a trained number are printed only when they
survive the nudges of `stable.py`.

Swedish Embedded AB implements solutions for image classifiers that are small, correct and cheap to run. If your team needs
expertise in convolutional networks and residual architectures, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import multiprocessing
import struct
from functools import lru_cache
from operator import mul

import bookrng
import convolution as cv
import stable as stab

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(v):
    return f"{v:,}"


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _fx(v, d=4):
    return f"{v:.{d}f}"


def _g(v, d=4):
    """Fixed notation for moderate values, scientific otherwise."""
    if v == 0:
        return f"{0:.{d}f}"
    return f"{v:.{d}f}" if 1e-3 <= abs(v) < 1e4 else _sci(v)


def _seed(name):
    s = bookrng.seed_of(name)
    return f"seed 0x{s:016x} ({s})"


def _numm(v, ws, d=4):
    """_g(v, d) with only the digits that survive in the nudged values ws (see `stable.text`)."""
    return stab.text(v, ws, lambda x: _g(x, d))


def _grid(rows, width=3):
    return "\n".join(" ".join(f"{v:{width}}" for v in r) for r in rows)


def _flat(t):
    out = []
    for v in t:
        out += _flat(v) if isinstance(v, list) else [v]
    return out


# ------------------------------------------------------------------------------------------------ the opening: same pixels

STRIPES = [[1 if j % 2 == 0 else 0 for j in range(8)] for _ in range(8)]
CHECKER = [[1 if (i + j) % 2 == 0 else 0 for j in range(8)] for i in range(8)]


def _picture(img):
    return "\n".join(" ".join("#" if v else "." for v in r) for r in img)


def _diff_filters(img):
    """Mean absolute value of the horizontal difference x[i][j+1] - x[i][j] (8 by 7 values) and of the vertical difference
    x[i+1][j] - x[i][j] (7 by 8 values)."""
    h = [abs(img[i][j + 1] - img[i][j]) for i in range(8) for j in range(7)]
    v = [abs(img[i + 1][j] - img[i][j]) for i in range(7) for j in range(8)]
    return sum(h) / len(h), sum(v) / len(v)


def same_pixels():
    rows = []
    for name, img in (("vertical stripes", STRIPES), ("checkerboard", CHECKER)):
        px = [v for r in img for v in r]
        h, v = _diff_filters(img)
        rows.append((name, sum(px), _fx(sum(px) / 64, 3), _fx(h, 3), _fx(v, 3)))
    return ("Two 8 by 8 images of zeros and ones. Vertical stripes: pixel (i, j) is 1 when j is even. Checkerboard: pixel (i, j) is 1 "
            "when i + j is even ('#' is 1, '.' is 0).\n\n```text\n"
            + "\n".join(f"{a}    {b}" for a, b in zip(_picture(STRIPES).split("\n"), _picture(CHECKER).split("\n")))
            + "\n```\n\nThe horizontal difference is x[i][j+1] - x[i][j] over its 8 * 7 = 56 valid positions and the vertical one is "
            "x[i+1][j] - x[i][j] over 7 * 8 = 56; the last two columns are the mean of the absolute values.\n\n"
            + _table(["image", "pixels equal to 1", "mean pixel", "mean absolute horizontal difference", "mean absolute vertical difference"], rows))


# ------------------------------------------------------------------------------------------------ pooling

POOL_X = [[2, 2, 0, 1],
          [1, 0, 3, 5],
          [0, 6, 2, 2],
          [3, 1, 7, 4]]
POOL_DY = [[1, 2], [3, 4]]
OVERLAP_X = [[1, 2, 0, 1, 2],
             [0, 3, 1, 0, 1],
             [2, 1, 9, 1, 0],
             [1, 0, 2, 3, 1],
             [0, 1, 1, 2, 4]]


def maxpool2d(x, k, s):
    """Max over each k by k window starting every s positions (no padding); returns (y, argmax) with argmax the (row, column) of the
    winner, the first one in row-major order inside the window when several are equal."""
    h, w = len(x), len(x[0])
    ho, wo = (h - k) // s + 1, (w - k) // s + 1
    y = [[0] * wo for _ in range(ho)]
    am = [[None] * wo for _ in range(ho)]
    for i in range(ho):
        for j in range(wo):
            best = None
            for a in range(k):
                for b in range(k):
                    v = x[i * s + a][j * s + b]
                    if best is None or v > best:
                        best, am[i][j] = v, (i * s + a, j * s + b)
            y[i][j] = best
    return y, am


def maxpool_backward(shape, am, dy):
    """dx: every output sends its gradient to its winner; a winner chosen by several windows receives the sum."""
    dx = [[0] * shape[1] for _ in range(shape[0])]
    for i, row in enumerate(am):
        for j, (r, c) in enumerate(row):
            dx[r][c] += dy[i][j]
    return dx


def avgpool2d(x, k, s):
    h, w = len(x), len(x[0])
    ho, wo = (h - k) // s + 1, (w - k) // s + 1
    return [[sum(x[i * s + a][j * s + b] for a in range(k) for b in range(k)) / (k * k) for j in range(wo)] for i in range(ho)]


def avgpool_backward(shape, k, s, dy):
    dx = [[0.0] * shape[1] for _ in range(shape[0])]
    for i, row in enumerate(dy):
        for j, g in enumerate(row):
            for a in range(k):
                for b in range(k):
                    dx[i * s + a][j * s + b] += g / (k * k)
    return dx


def pooling():
    y, am = maxpool2d(POOL_X, 2, 2)
    dxm = maxpool_backward((4, 4), am, POOL_DY)
    ya = avgpool2d(POOL_X, 2, 2)
    dxa = avgpool_backward((4, 4), 2, 2, POOL_DY)
    yo, amo = maxpool2d(OVERLAP_X, 3, 2)
    dxo = maxpool_backward((5, 5), amo, POOL_DY)
    # global average pooling of a [2, 2, 3] map and its backward for an arriving gradient (6, -3)
    gmap = [[[1, 2, 3], [4, 5, 6]], [[0, 0, 3], [3, 0, 0]]]
    gap = [sum(v for r in ch for v in r) / 6 for ch in gmap]
    dgap = [6.0, -3.0]
    spread = [[[g / 6 for _ in range(3)] for _ in range(2)] for g in dgap]
    ap = [[f"{a[0]}:{a[1]}" for a in row] for row in am]

    def fmt(m):
        return "\n".join(" ".join(f"{v:6.2f}" if isinstance(v, float) else f"{v:6d}" for v in r) for r in m)

    return (f"The input X is 4 by 4, the gradient arriving at the 2 by 2 output is dy = {POOL_DY}. A window with equal maxima sends "
            "its gradient to the first of them in row-major order (the top-left window has 2 twice).\n\n```text\n"
            f"X =\n{_grid(POOL_X, 2)}\n\nmax pool 2x2, stride 2: y =\n{_grid(y, 2)}\nwinning positions (row:column):\n"
            f"{_grid(ap, 4).replace(' \n', chr(10))}\ndx =\n{_grid(dxm, 2)}\n\naverage pool 2x2, stride 2: y =\n{fmt(ya)}\ndx = dy / 4 over each window =\n{fmt(dxa)}\n```\n\n"
            f"Overlapping windows: X' is 5 by 5, window 3 by 3, stride 2, so the four windows share the centre pixel; it is the maximum "
            f"of all four.\n\n```text\nX' =\n{_grid(OVERLAP_X, 2)}\n\nmax pool 3x3, stride 2: y =\n{_grid(yo, 2)}\n"
            f"dx for dy = {POOL_DY} =\n{_grid(dxo, 2)}\n```\n\nThe centre receives {dxo[2][2]} = 1 + 2 + 3 + 4: a pixel that wins several "
            f"windows gets the sum of their gradients.\n\nGlobal average pooling of the map with 2 channels of 2 by 3 values "
            f"{gmap}: y = {gap}. With the gradient {dgap} arriving, every position of channel c receives dy[c] / 6: {spread}.")


# ------------------------------------------------------------------------------------------------ the toy image set

IMG = 8
CLASSES = ("stripes", "checkerboard", "blobs")
DATA_NAME = "cnns.toy_data"
INIT_NAME = "cnns.toy_init"
N_IMAGES, N_TRAIN = 150, 60
NOISE = 0.4


def toy_image(gen, cls):
    """One 8 by 8 image of class `cls`, row-major; draws in the order the chapter prints."""
    px = [[0.0] * IMG for _ in range(IMG)]
    if cls == 0:  # stripes: orientation (0 vertical stripes, which vary along j), width 1 or 2, phase
        vert = gen.integer(2)
        width = gen.integer(2) + 1
        phase = gen.integer(2 * width)
        for i in range(IMG):
            for j in range(IMG):
                t = j if vert == 0 else i
                px[i][j] = 1.0 if ((t + phase) // width) % 2 == 0 else 0.0
    elif cls == 1:  # checkerboard: cell size 1 or 2, row offset, column offset
        size = gen.integer(2) + 1
        oy = gen.integer(2 * size)
        ox = gen.integer(2 * size)
        for i in range(IMG):
            for j in range(IMG):
                px[i][j] = 1.0 if ((i + oy) // size + (j + ox) // size) % 2 == 0 else 0.0
    else:  # blobs: 2 or 3 Gaussian bumps of standard deviation 1.2 at uniform centres in [0, 7)^2, combined by the maximum
        nb = gen.integer(2) + 2
        cs = [(gen.uniform() * 7.0, gen.uniform() * 7.0) for _ in range(nb)]
        for i in range(IMG):
            for j in range(IMG):
                px[i][j] = max(math.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * 1.2 ** 2)) for ci, cj in cs)
    for i in range(IMG):
        for j in range(IMG):
            px[i][j] += NOISE * gen.normal()
    return [v for r in px for v in r]


@lru_cache(maxsize=None)
def toy_data():
    """(image, class) for n = 0 .. 149 with class n mod 3; the first 60 are the training set, the other 90 the test set."""
    gen = bookrng.generator(DATA_NAME)
    return tuple((tuple(toy_image(gen, k % 3)), k % 3) for k in range(N_IMAGES))


def _train():
    return [(list(x), y) for x, y in toy_data()[:N_TRAIN]]


def _test():
    return [(list(x), y) for x, y in toy_data()[N_TRAIN:]]


def toy_set():
    d = toy_data()
    shots = []
    for n in range(3):
        x = d[n][0]
        rows = ["".join("#" if x[i * IMG + j] > 0.5 else "." for j in range(IMG)) for i in range(IMG)]
        shots.append(rows)
    pic = "\n".join("   ".join(s[r] for s in shots) for r in range(IMG))
    names = "   ".join(f"{('stripes', 'checker', 'blobs')[n]:<8}" for n in range(3))
    lo = min(min(x) for x, _ in d)
    hi = max(max(x) for x, _ in d)
    return (f"Image n has class n mod 3 (0 stripes, 1 checkerboard, 2 blobs); n = 0 .. 59 is the training set (20 of each class) and "
            f"n = 60 .. 149 the test set (30 of each). One generator for all of them: the book's SplitMix64, {_seed(DATA_NAME)}, "
            f"the name {DATA_NAME}. Draws for one image, in order:\n\n"
            "- stripes: `integer(2)` (0 for stripes that vary along the columns, 1 along the rows), `integer(2) + 1` for the stripe "
            "width, `integer(2 * width)` for the phase; pixel = 1 when floor((t + phase) / width) is even, t being the column or the row;\n"
            "- checkerboard: `integer(2) + 1` for the cell size, `integer(2 * size)` for the row offset oy, `integer(2 * size)` for the "
            "column offset ox; pixel = 1 when floor((i + oy) / size) + floor((j + ox) / size) is even;\n"
            "- blobs: `integer(2) + 2` bumps, then for each bump the row centre `uniform() * 7` and the column centre `uniform() * 7`; "
            "pixel = the largest over bumps of exp(-((i - ci)^2 + (j - cj)^2) / (2 * 1.2^2));\n"
            f"- then, for every class, the noise: 64 draws of `normal()` in row-major order, each times {NOISE} and added to the pixel.\n\n"
            f"The images n = 0, 1, 2 (the noisy images themselves, each pixel drawn as # when it exceeds 0.5 and . otherwise; pixel values range over [{lo:.2f}, {hi:.2f}]):\n\n"
            f"```text\n{names}\n{pic}\n```")


# ------------------------------------------------------------------------------------------------ the toy network

C1, C2, NCLS = 4, 6, 3


class Geo:
    """Index maps for a 3 by 3 convolution with padding 1 and stride 1 over a [C][H*W] tensor: `idx[p]` lists, in the order (input
    channel, kernel row, kernel column), where the taps of output position p sit in the zero-padded copy of the input."""

    def __init__(self, c, h, w):
        self.c, self.h, self.w = c, h, w
        self.hp, self.wp = h + 2, w + 2
        self.idx = [[ch * self.hp * self.wp + (i + a) * self.wp + (j + b) for ch in range(c) for a in range(3) for b in range(3)]
                    for i in range(h) for j in range(w)]
        self.interior = [[ch * self.hp * self.wp + (i + 1) * self.wp + (j + 1) for i in range(h) for j in range(w)] for ch in range(c)]

    def pad(self, x):
        xp = [0.0] * (self.c * self.hp * self.wp)
        for ch in range(self.c):
            ix, xc = self.interior[ch], x[ch]
            for p in range(self.h * self.w):
                xp[ix[p]] = xc[p]
        return xp


def conv_fwd(geo, w, b, x):
    xp = geo.pad(x)
    col = [[xp[t] for t in ix] for ix in geo.idx]
    return [[sum(map(mul, wo, cp)) + bo for cp in col] for wo, bo in zip(w, b)], col


def conv_bwd(geo, w, col, dz, need_dx=True):
    taps = len(w[0])
    col_t = list(zip(*col))
    dw = [[sum(map(mul, dzo, col_t[t])) for t in range(taps)] for dzo in dz]
    db = [sum(dzo) for dzo in dz]
    dx = None
    if need_dx:
        dz_t, w_t = list(zip(*dz)), list(zip(*w))
        dxp = [0.0] * (geo.c * geo.hp * geo.wp)
        for p, ix in enumerate(geo.idx):
            for t, i in enumerate(ix):
                dxp[i] += sum(map(mul, dz_t[p], w_t[t]))
        dx = [[dxp[i] for i in geo.interior[ch]] for ch in range(geo.c)]
    return dw, db, dx


def maxpool_fwd(x, c, h, w):
    """2 by 2, stride 2, over [C][H*W]; returns the pooled tensor and, for each output, the index in the input channel of its
    winner (the first in row-major order when equal)."""
    ho, wo = h // 2, w // 2
    out, arg = [], []
    for ch in range(c):
        xc, oc, ac = x[ch], [], []
        for i in range(ho):
            for j in range(wo):
                best, bi = None, -1
                for a in range(2):
                    for b in range(2):
                        q = (2 * i + a) * w + 2 * j + b
                        if best is None or xc[q] > best:
                            best, bi = xc[q], q
                oc.append(best)
                ac.append(bi)
        out.append(oc)
        arg.append(ac)
    return out, arg


def relu_map(z):
    return [[v if v > 0 else 0.0 for v in r] for r in z]


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


G1, G2 = Geo(1, 8, 8), Geo(C1, 4, 4)


def toy_init(nudge=0.0):
    """He-normal convolution weights (standard deviation sqrt(2 / (C_in * 9))) drawn in the order W1 (output channel, tap), W2
    (output channel, input channel, row, column), then the linear layer with standard deviation sqrt(1 / 6); biases 0. `nudge` is
    added to the first weight of W1, the first tap of the first filter."""
    gen = bookrng.generator(INIT_NAME)
    w1 = [[gen.normal() * math.sqrt(2.0 / 9) for _ in range(9)] for _ in range(C1)]
    w2 = [[gen.normal() * math.sqrt(2.0 / (C1 * 9)) for _ in range(C1 * 9)] for _ in range(C2)]
    wl = [[gen.normal() * math.sqrt(1.0 / C2) for _ in range(C2)] for _ in range(NCLS)]
    w1[0][0] += nudge
    return {"W1": w1, "b1": [0.0] * C1, "W2": w2, "b2": [0.0] * C2, "Wl": wl, "bl": [0.0] * NCLS}


def toy_forward(p, img):
    z1, col1 = conv_fwd(G1, p["W1"], p["b1"], [img])
    a1 = relu_map(z1)
    p1, arg = maxpool_fwd(a1, C1, 8, 8)
    z2, col2 = conv_fwd(G2, p["W2"], p["b2"], p1)
    a2 = relu_map(z2)
    gap = [sum(ac) / 16.0 for ac in a2]
    logits = [sum(map(mul, wl, gap)) + bl for wl, bl in zip(p["Wl"], p["bl"])]
    return logits, (z1, col1, arg, z2, col2, gap)


def _zeros_like(p):
    return {k: [[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v) for k, v in p.items()}


def toy_loss_grad(p, data, grad=True):
    """Mean cross-entropy, accuracy and (when asked) the gradient of the mean loss; the images are visited in index order."""
    n = len(data)
    g = _zeros_like(p) if grad else None
    loss, correct = 0.0, 0
    for img, y in data:
        logits, cache = toy_forward(p, img)
        pr = softmax(logits)
        loss += -math.log(pr[y])
        correct += pr.index(max(pr)) == y
        if not grad:
            continue
        z1, col1, arg, z2, col2, gap = cache
        dl = [pr[k] - (1.0 if k == y else 0.0) for k in range(NCLS)]
        for k in range(NCLS):
            for o in range(C2):
                g["Wl"][k][o] += dl[k] * gap[o]
            g["bl"][k] += dl[k]
        dgap = [sum(dl[k] * p["Wl"][k][o] for k in range(NCLS)) for o in range(C2)]
        dz2 = [[(dgap[o] / 16.0) if z2[o][q] > 0 else 0.0 for q in range(16)] for o in range(C2)]
        dw2, db2, dp1 = conv_bwd(G2, p["W2"], col2, dz2)
        dz1 = [[0.0] * 64 for _ in range(C1)]
        for ch in range(C1):
            for q, qi in enumerate(arg[ch]):
                if z1[ch][qi] > 0:
                    dz1[ch][qi] += dp1[ch][q]
        dw1, db1, _ = conv_bwd(G1, p["W1"], col1, dz1, need_dx=False)
        for name, dw, db in (("2", dw2, db2), ("1", dw1, db1)):
            for o in range(len(dw)):
                g["b" + name][o] += db[o]
                row = g["W" + name][o]
                for t, v in enumerate(dw[o]):
                    row[t] += v
    if grad:
        for v in g.values():
            for r in (v if isinstance(v[0], list) else [v]):
                for t in range(len(r)):
                    r[t] /= n
    return loss / n, correct / n, g


def _leaves(p):
    return [r for k in ("W1", "b1", "W2", "b2", "Wl", "bl") for r in (p[k] if isinstance(p[k][0], list) else [p[k]])]


def n_params(p):
    return sum(len(r) for r in _leaves(p))


def adam_steps(p, loss_grad, steps, lr, record, b1=0.9, b2=0.999, eps=1e-8):
    """Adam with bias correction (Volume III, Chapter 7) on the leaves of p, in place, from zero moments. `loss_grad(p)` returns
    (loss, accuracy, gradient with the layout of p). Returns {t: (loss, accuracy)} for t in `record`, the values after t updates."""
    leaves = _leaves(p)
    m = [[0.0] * len(r) for r in leaves]
    v = [[0.0] * len(r) for r in leaves]
    seen = {}
    for t in range(steps + 1):
        loss, acc, g = loss_grad(p)
        if t in record:
            seen[t] = (loss, acc)
        if t == steps:
            break
        for r, gr, mr, vr in zip(leaves, _leaves(g), m, v):
            for i in range(len(r)):
                mr[i] = b1 * mr[i] + (1 - b1) * gr[i]
                vr[i] = b2 * vr[i] + (1 - b2) * gr[i] * gr[i]
                r[i] -= lr * (mr[i] / (1 - b1 ** (t + 1))) / (math.sqrt(vr[i] / (1 - b2 ** (t + 1))) + eps)
    return seen


TOY_STEPS, TOY_LR = 100, 0.02
TOY_RECORD = (0, 10, 25, 50, 100)


def _toy_task(nudge):
    p = toy_init(nudge)
    train = _train()
    seen = adam_steps(p, lambda q: toy_loss_grad(q, train), TOY_STEPS, TOY_LR, TOY_RECORD)
    te_loss, te_acc, _ = toy_loss_grad(p, _test(), grad=False)
    return {"seen": seen, "test": (te_loss, te_acc), "params": p}


def _pool_map(fn, args):
    with multiprocessing.get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(fn, args)


@lru_cache(maxsize=None)
def toy_runs():
    """The training run and its four nudged copies, keyed by the nudge (0.0 is the run itself)."""
    nudges = (0.0, *stab.NUDGES)
    return dict(zip(nudges, _pool_map(_toy_task, nudges)))


def toy_run(nudge=0.0):
    return toy_runs()[nudge]


def _twins(f):
    """f(run) for the run and for each nudged copy."""
    return f(toy_run(0.0)), [f(toy_run(n)) for n in stab.NUDGES]


def toy_layers():
    rows = []
    shape = (1, 8, 8)

    def conv(name, cin, cout, h, w):
        params = cout * cin * 9 + cout
        macs = h * w * cout * cin * 9
        rows.append((name, f"[{cin}, {h}, {w}] to [{cout}, {h}, {w}]", params, macs, 2 * macs))
        return params, macs

    p1, m1 = conv("conv 3x3, pad 1", 1, C1, 8, 8)
    rows.append(("ReLU", "[4, 8, 8] to [4, 8, 8]", 0, 0, 0))
    rows.append(("max pool 2x2, stride 2", "[4, 8, 8] to [4, 4, 4]", 0, 0, 0))
    p2, m2 = conv("conv 3x3, pad 1", C1, C2, 4, 4)
    rows.append(("ReLU", "[6, 4, 4] to [6, 4, 4]", 0, 0, 0))
    rows.append(("global average pool", "[6, 4, 4] to [6]", 0, 0, 0))
    pl, ml = C2 * NCLS + NCLS, C2 * NCLS
    rows.append(("linear", "[6] to [3]", pl, ml, 2 * ml))
    total_p, total_m = p1 + p2 + pl, m1 + m2 + ml
    # receptive field of one value of the last convolution, along one axis (Volume VII, Chapter 2, recursion (4))
    rec = cv.receptive([(3, 1, 1), (2, 2, 1), (3, 1, 1)])
    flat_params = C2 * 4 * 4 * NCLS + NCLS
    out = ("Parameters, multiply-adds (MACs) and FLOPs (2 per multiply-add) of one image through each layer of the toy network; "
           "ReLU, pooling and the softmax add comparisons and a few additions that are not counted as multiply-adds.\n\n"
           + _table(["layer", "input to output", "parameters", "multiply-adds", "FLOPs"],
                    [(a, b, _int(c), _int(d), _int(e)) for a, b, c, d, e in rows], "llrrr")
           + f"\n\nTotal: {_int(total_p)} parameters and {_int(total_m)} multiply-adds ({_int(2 * total_m)} FLOPs) per image. "
           f"Receptive field along one axis by the recursion r = r + d (k - 1) j, j = j s, from r = j = 1: conv 3x3: r = {rec[0][0]}, "
           f"j = {rec[0][1]}; pool 2x2 stride 2: r = {rec[1][0]}, j = {rec[1][1]}; conv 3x3: r = {rec[2][0]}, j = {rec[2][1]}. "
           f"Flatten and a linear layer on the 6 * 4 * 4 = {C2 * 16} values of the second convolution's output would need {_int(flat_params)} "
           f"parameters where global average pooling needs {pl}.")
    return out


# ------------------------------------------------------------------------------------------------ gradient check

GC_IMAGES = 6
GC_STEP = 1e-6


def gradient_check():
    p = toy_init()
    train = _train()
    # two images of each class (n = 0 .. 5)
    sub = train[:GC_IMAGES]
    _, _, g = toy_loss_grad(p, sub)
    worst, fails, total = 0.0, 0, 0
    per_group = {}
    for name in ("W1", "b1", "W2", "b2", "Wl", "bl"):
        rows = p[name] if isinstance(p[name][0], list) else [p[name]]
        grows = g[name] if isinstance(g[name][0], list) else [g[name]]
        gw = 0.0
        for r, gr in zip(rows, grows):
            for t in range(len(r)):
                old = r[t]
                r[t] = old + GC_STEP
                lp = toy_loss_grad(p, sub, grad=False)[0]
                r[t] = old - GC_STEP
                lm = toy_loss_grad(p, sub, grad=False)[0]
                r[t] = old
                num = (lp - lm) / (2 * GC_STEP)
                err = abs(gr[t] - num)
                gate = 1e-7 + 1e-5 * max(abs(gr[t]), abs(num))
                total += 1
                fails += err > gate
                gw = max(gw, err)
        per_group[name] = (sum(len(r) for r in rows), gw)
        worst = max(worst, gw)
    rows = [(n, c, "below 1e-7" if w < 1e-7 else _sci(w)) for n, (c, w) in per_group.items()]
    return (f"The initialized network ({INIT_NAME}, {_seed(INIT_NAME)}) on the training images n = 0 .. {GC_IMAGES - 1} (two of each class): "
            f"mean cross-entropy, analytic gradient from the backward pass of the chapter against the central difference "
            f"(L(w + h) - L(w - h)) / (2h) with h = {GC_STEP:g} in 64-bit arithmetic, one weight at a time. The gate of Volume III, "
            f"Chapter 6 is |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|).\n\n"
            + _table(["parameters", "count", "largest absolute difference"], rows)
            + f"\n\n{total - fails} of {total} parameters pass the gate ({fails} fail).")


# ------------------------------------------------------------------------------------------------ the Adam run


def training():
    base = toy_run(0.0)
    rows = []
    for t in TOY_RECORD:
        ws = [toy_run(n)["seen"][t] for n in stab.NUDGES]
        l0, a0 = base["seen"][t]
        rows.append((t, _numm(l0, [w[0] for w in ws], 4), _numm(a0, [w[1] for w in ws], 4)))
    te0 = base["test"]
    tws = [toy_run(n)["test"] for n in stab.NUDGES]
    note = (f"Test set (90 images) after {TOY_STEPS} updates: cross-entropy {_numm(te0[0], [w[0] for w in tws], 4)}, accuracy "
            f"{_numm(te0[1], [w[1] for w in tws], 4)}.")
    return (f"The toy network trained on the 60 training images by full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, "
            f"learning rate {TOY_LR}, bias correction as in Volume III, Chapter 7), {TOY_STEPS} updates, mean cross-entropy, from "
            f"the initialization of {INIT_NAME}. Training loss and accuracy on all 60 images after t updates. Sums over taps run in the order (input channel, "
            "kernel row, kernel column), over positions row-major and over images in index order. " + stab.RULE
            + "\n\n" + _table(["updates t", "training loss", "training accuracy"], rows) + "\n\n" + note)


# ------------------------------------------------------------------------------------------------ shifts


def roll(img, s):
    """Circular shift to the right by s columns of a 64-value image."""
    return [img[(i // IMG) * IMG + ((i % IMG) - s) % IMG] for i in range(IMG * IMG)]


def _circ_conv(x, w, b, c, h, wd):
    """3 by 3 convolution of [C][H*W] with circular padding (indices wrap around)."""
    out = []
    for wo, bo in zip(w, b):
        o = []
        for i in range(h):
            for j in range(wd):
                acc = 0.0
                for ch in range(c):
                    for a in range(3):
                        for bb in range(3):
                            acc += wo[ch * 9 + a * 3 + bb] * x[ch][((i + a - 1) % h) * wd + (j + bb - 1) % wd]
                o.append(acc + bo)
        out.append(o)
    return out


def _circ_forward(p, img):
    """The toy network with circular padding in both convolutions; returns (conv1+ReLU map, pooled map, logits)."""
    a1 = relu_map(_circ_conv([img], p["W1"], p["b1"], 1, 8, 8))
    p1, _ = maxpool_fwd(a1, C1, 8, 8)
    a2 = relu_map(_circ_conv(p1, p["W2"], p["b2"], C1, 4, 4))
    gap = [sum(ac) / 16.0 for ac in a2]
    return a1, p1, [sum(map(mul, wl, gap)) + bl for wl, bl in zip(p["Wl"], p["bl"])]


def _roll_map(m, s, side):
    return [roll(ch, s) if side == 8 else [ch[(q // side) * side + ((q % side) - s) % side] for q in range(side * side)] for ch in m]


def _maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(_flat(a), _flat(b)))


SHIFTS = (0, 1, 2, 3, 4)


def shift_equivariance():
    p = toy_init()
    test = _test()
    rows = []
    for s in SHIFTS:
        e1 = e2 = e3 = 0.0
        for img, _ in test:
            a1, p1, lg = _circ_forward(p, img)
            a1s, p1s, lgs = _circ_forward(p, roll(img, s))
            e1 = max(e1, _maxdiff(a1s, _roll_map(a1, s, 8)))
            # pooled map: compare with the pooled map of the original shifted by s / 2 columns (rounded down or up for odd s)
            best = min(_maxdiff(p1s, _roll_map(p1, t, 4)) for t in {s // 2, (s + 1) // 2})
            e2 = max(e2, best)
            e3 = max(e3, _maxdiff(lgs, lg))

        def f(v):
            return "below 1e-15" if v < 1e-15 else _g(v, 4)
        rows.append((s, f(e1), f(e2), f(e3)))
    return (f"The initialized network ({INIT_NAME}) with circular padding in both convolutions (indices wrap around, so no border), "
            "on the 90 test images x, each shifted to the right by s columns with wrap-around (roll). Largest absolute difference over "
            "all images and values between: (a) the ReLU map of the shifted image and the ReLU map of x shifted by s; (b) the pooled "
            "map of the shifted image and the pooled map of x shifted by s / 2 columns (for odd s, for each image separately, the smaller of the "
            "two differences against x shifted by (s - 1) / 2 and by (s + 1) / 2 columns; the error shown is the largest over the images); (c) the logits of the shifted image and of x.\n\n"
            + _table(["shift s", "(a) after conv and ReLU", "(b) after pooling", "(c) logits"], rows))


def shift_stability():
    train_run = toy_run(0.0)
    test = _test()

    def measure(run):
        p = run["params"]
        out = {}
        base = [softmax(toy_forward(p, img)[0]) for img, _ in test]
        for s in SHIFTS:
            same, dp = 0, 0.0
            for (img, _), pb in zip(test, base):
                ps = softmax(toy_forward(p, roll(img, s))[0])
                same += ps.index(max(ps)) == pb.index(max(pb))
                dp += max(abs(a - b) for a, b in zip(ps, pb))
            out[s] = (same / len(test), dp / len(test))
        return out

    base = measure(train_run)
    twins = [measure(toy_run(n)) for n in stab.NUDGES]
    rows = []
    for s in SHIFTS:
        rows.append((s, _numm(base[s][0], [w[s][0] for w in twins], 4), _numm(base[s][1], [w[s][1] for w in twins], 4)))
    return ("The trained network (zero padding, as trained) on the 90 test images, each shifted to the right by s columns with "
            "wrap-around. Share of images whose predicted class is the same as for the unshifted image, and the mean over images "
            "of the largest change of a class probability. " + stab.RULE + "\n\n"
            + _table(["shift s", "same predicted class", "mean largest probability change"], rows))


# ------------------------------------------------------------------------------------------------ depth: plain against residual

DEPTH_NAME = "cnns.depth_init"
DEPTH_CHANNELS = 4
DEPTH_IMAGES = 24
DEPTH_STEPS, DEPTH_LR = 50, 0.01
DEPTH_BRANCH_SCALE = 0.3
DEPTHS = (4, 24)
SEEDS = (0, 1, 2, 3)
SIGNAL_DEPTHS = (2, 4, 8, 12, 16, 24, 32, 48)
SIGNAL_IMAGES = 9
GD = Geo(DEPTH_CHANNELS, 4, 4)


def depth_init(depth, kind, run, nudge=0.0):
    """Stem filters (4 by 9 draws), then `depth` convolutions of 4 filters of 36 taps each in layer order, then the read-out
    (3 by 4 draws times sqrt(1/4)). Every convolution has standard deviation sqrt(2 / 36); in the residual stack the second
    convolution of each block (odd layer index) is multiplied by 0.3. No biases anywhere."""
    gen = bookrng.generator(f"{DEPTH_NAME}.{run}")
    p = {"stem": [[gen.normal() * math.sqrt(2.0 / 9) for _ in range(9)] for _ in range(DEPTH_CHANNELS)]}
    convs = []
    for l in range(depth):
        s = math.sqrt(2.0 / 36) * (DEPTH_BRANCH_SCALE if kind == "res" and l % 2 == 1 else 1.0)
        convs.append([[gen.normal() * s for _ in range(36)] for _ in range(DEPTH_CHANNELS)])
    p["convs"] = convs
    p["Wl"] = [[gen.normal() * 0.5 for _ in range(DEPTH_CHANNELS)] for _ in range(NCLS)]
    p["stem"][0][0] += nudge
    return p


_ZERO_BIAS = [0.0] * DEPTH_CHANNELS


def depth_forward(p, img, kind):
    z0, col0 = conv_fwd(G1, p["stem"], _ZERO_BIAS, [img])
    x, arg = maxpool_fwd(relu_map(z0), DEPTH_CHANNELS, 8, 8)
    layers, profile = [], [x]
    convs = p["convs"]
    if kind == "plain":
        for l in range(len(convs)):
            z, col = conv_fwd(GD, convs[l], _ZERO_BIAS, x)
            layers.append((z, col))
            x = relu_map(z)
            profile.append(x)
    else:
        for l in range(0, len(convs), 2):
            z1, c1 = conv_fwd(GD, convs[l], _ZERO_BIAS, x)
            h = relu_map(z1)
            z2, c2 = conv_fwd(GD, convs[l + 1], _ZERO_BIAS, h)
            s = [[a + b for a, b in zip(xr, zr)] for xr, zr in zip(x, z2)]
            layers.append((z1, c1, z2, c2, s))
            x = relu_map(s)
            profile.append(x)
    gap = [sum(r) / 16 for r in x]
    return [sum(map(mul, w, gap)) for w in p["Wl"]], (z0, col0, arg, layers, gap, profile)


def depth_loss_grad(p, data, kind):
    n = len(data)
    depth = len(p["convs"])
    g = {"stem": [[0.0] * 9 for _ in range(DEPTH_CHANNELS)], "convs": [[[0.0] * 36 for _ in range(DEPTH_CHANNELS)] for _ in range(depth)],
         "Wl": [[0.0] * DEPTH_CHANNELS for _ in range(NCLS)]}
    loss, correct = 0.0, 0
    for img, y in data:
        logits, (z0, col0, arg, layers, gap, _) = depth_forward(p, img, kind)
        pr = softmax(logits)
        loss += -math.log(pr[y])
        correct += pr.index(max(pr)) == y
        dl = [pr[k] - (1.0 if k == y else 0.0) for k in range(NCLS)]
        for k in range(NCLS):
            for o in range(DEPTH_CHANNELS):
                g["Wl"][k][o] += dl[k] * gap[o]
        dgap = [sum(dl[k] * p["Wl"][k][o] for k in range(NCLS)) for o in range(DEPTH_CHANNELS)]
        dx = [[dgap[o] / 16 for _ in range(16)] for o in range(DEPTH_CHANNELS)]
        for li in range(len(layers) - 1, -1, -1):
            if kind == "plain":
                z, col = layers[li]
                dz = [[d if zz > 0 else 0.0 for d, zz in zip(dr, zr)] for dr, zr in zip(dx, z)]
                dw, _, dx = conv_bwd(GD, p["convs"][li], col, dz)
                for o in range(DEPTH_CHANNELS):
                    for t, v in enumerate(dw[o]):
                        g["convs"][li][o][t] += v
            else:
                z1, c1, z2, c2, s = layers[li]
                ds = [[d if zz > 0 else 0.0 for d, zz in zip(dr, zr)] for dr, zr in zip(dx, s)]
                dw2, _, dh = conv_bwd(GD, p["convs"][2 * li + 1], c2, ds)
                dz1 = [[d if zz > 0 else 0.0 for d, zz in zip(dr, zr)] for dr, zr in zip(dh, z1)]
                dw1, _, dxb = conv_bwd(GD, p["convs"][2 * li], c1, dz1)
                for o in range(DEPTH_CHANNELS):
                    for t in range(36):
                        g["convs"][2 * li + 1][o][t] += dw2[o][t]
                        g["convs"][2 * li][o][t] += dw1[o][t]
                dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(ds, dxb)]  # identity path plus branch
        dz0 = [[0.0] * 64 for _ in range(DEPTH_CHANNELS)]
        for ch in range(DEPTH_CHANNELS):
            for q, qi in enumerate(arg[ch]):
                if z0[ch][qi] > 0:
                    dz0[ch][qi] += dx[ch][q]
        dw0, _, _ = conv_bwd(G1, p["stem"], col0, dz0, need_dx=False)
        for o in range(DEPTH_CHANNELS):
            for t, v in enumerate(dw0[o]):
                g["stem"][o][t] += v
    for r in g["stem"] + g["Wl"] + [r for c in g["convs"] for r in c]:
        for t in range(len(r)):
            r[t] /= n
    return loss / n, correct / n, g


def _depth_leaves(p):
    return p["stem"] + p["Wl"] + [r for c in p["convs"] for r in c]


def _depth_task(args):
    depth, kind, seed, nudge = args
    p = depth_init(depth, kind, seed, nudge)
    data = [(list(x), y) for x, y in toy_data()[:DEPTH_IMAGES]]
    leaves = _depth_leaves(p)
    m = [[0.0] * len(r) for r in leaves]
    v = [[0.0] * len(r) for r in leaves]
    curve = []
    for t in range(DEPTH_STEPS + 1):
        loss, acc, g = depth_loss_grad(p, data, kind)
        curve.append((loss, acc))
        if t == DEPTH_STEPS:
            break
        for r, gr, mr, vr in zip(leaves, _depth_leaves(g), m, v):
            for i in range(len(r)):
                mr[i] = 0.9 * mr[i] + 0.1 * gr[i]
                vr[i] = 0.999 * vr[i] + 0.001 * gr[i] * gr[i]
                r[i] -= DEPTH_LR * (mr[i] / (1 - 0.9 ** (t + 1))) / (math.sqrt(vr[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return curve


@lru_cache(maxsize=None)
def depth_runs():
    """curves[(depth, kind, seed, nudge)] = [(loss, accuracy) after t updates for t = 0 .. DEPTH_STEPS]."""
    jobs = [(d, k, s, n) for d in DEPTHS for k in ("plain", "res") for s in SEEDS for n in (0.0, *stab.NUDGES)]
    return dict(zip(jobs, _pool_map(_depth_task, jobs)))


def depth_setup():
    """The description of the stacks and their initialization, printed with the first table that uses them."""
    names = ", ".join(f"{DEPTH_NAME}.{s}" for s in SEEDS)
    return (f"Stem (3x3 conv, 4 filters, ReLU, 2x2 max pool, from the 8 by 8 image to 4 by 4 maps), then the body, then global average "
            f"pooling and a linear layer (4 to 3, no bias). The body is d convolutions 4 to 4 of 3 by 3 with padding 1 and ReLU after "
            f"each (plain), or d / 2 residual blocks relu(x + conv(relu(conv(x)))) with the same d convolutions. No biases, no "
            f"normalization. Initialization: one generator per seed, the book's SplitMix64 with the names {names} ({'; '.join(_seed(f'{DEPTH_NAME}.{s}') for s in SEEDS)}); "
            f"draws in order the stem (4 filters of 9 taps), then the d body convolutions (4 filters of 36 taps each, in layer order), then the "
            f"read-out (3 rows of 4); the same draws for the plain and the residual stack of one depth and seed; standard deviation sqrt(2 / 9) for "
            f"the stem and sqrt(2 / 36) for the body, times {DEPTH_BRANCH_SCALE} for the second convolution of each residual block, 0.5 for the read-out. "
            "Sums over taps run in the order (input channel, kernel row, kernel column), over positions row-major and over images in index order.")


def depth_table():
    runs = depth_runs()
    rows = []
    for d in DEPTHS:
        for kind in ("plain", "res"):
            cells = []
            for s in SEEDS:
                base = runs[(d, kind, s, 0.0)][-1]
                ws = [runs[(d, kind, s, n)][-1] for n in stab.NUDGES]
                cells.append(f"{_numm(base[0], [w[0] for w in ws], 3)} ({_numm(base[1], [w[1] for w in ws], 2)})")
            rows.append((f"{d} convolutions, " + ("plain" if kind == "plain" else f"residual, {d // 2} blocks"), *cells))
    return (f"The stacks, seeds and initialization of the previous table, trained on the first {DEPTH_IMAGES} training images (8 of each class) by Adam "
            f"(0.9, 0.999, 1e-8), learning rate {DEPTH_LR}, {DEPTH_STEPS} full-batch updates of the mean cross-entropy. Each cell is the training loss "
            f"after {DEPTH_STEPS} updates and, in brackets, the training accuracy. " + stab.RULE
            + "\n\n" + _table(["stack"] + [f"seed {s}" for s in SEEDS], rows))


def _rms(x):
    v = _flat(x)
    return math.sqrt(sum(a * a for a in v) / len(v))


def signal_profile(depth, kind, seed):
    """At initialization, over the first SIGNAL_IMAGES training images: the root mean square of the stream after the stem and after each
    convolution (plain) or block (residual), and the norm of the weight gradient of each body convolution."""
    p = depth_init(depth, kind, seed)
    data = [(list(x), y) for x, y in toy_data()[:SIGNAL_IMAGES]]
    sq = None
    for img, _ in data:
        _, (_, _, _, _, _, profile) = depth_forward(p, img, kind)
        vals = [sum(a * a for a in _flat(x)) / len(_flat(x)) for x in profile]
        sq = vals if sq is None else [a + b for a, b in zip(sq, vals)]
    rms = [math.sqrt(a / len(data)) for a in sq]
    _, _, g = depth_loss_grad(p, data, kind)
    return rms, [math.sqrt(sum(v * v for r in c for v in r)) for c in g["convs"]]


def _geo_mean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@lru_cache(maxsize=None)
def signal_data():
    """{(depth, kind): (rms after the last layer / rms after the stem, gradient norm of the first body convolution, of the last)}, geometric
    means over SEEDS."""
    out = {}
    for d in SIGNAL_DEPTHS:
        for kind in ("plain", "res"):
            prof = [signal_profile(d, kind, s) for s in SEEDS]
            out[(d, kind)] = (_geo_mean([r[-1] / r[0] for r, _ in prof]), _geo_mean([n[0] for _, n in prof]), _geo_mean([n[-1] for _, n in prof]))
    return out


def border_taps(n):
    """Mean share of the 9 taps of a 3 by 3 window with padding 1 that read a real pixel of an n by n map."""
    per_axis = sum(sum(1 for a in range(3) if 0 <= i + a - 1 < n) for i in range(n)) / (3 * n)
    return per_axis ** 2


def depth_signal():
    data = signal_data()
    bt = border_taps(4)
    rows = []
    for d in SIGNAL_DEPTHS:
        pl, rs = data[(d, "plain")], data[(d, "res")]
        rows.append((d, _sci(pl[0], 2), _sci(rs[0], 2), _sci(math.sqrt(bt) ** d, 2), _sci(pl[1], 2), _sci(rs[1], 2)))
    return (depth_setup() + f" Measured at initialization, for the seeds 0 to 3, on the first {SIGNAL_IMAGES} training images (n = 0 .. {SIGNAL_IMAGES - 1}); "
            "each entry is the geometric mean over the four seeds of a ratio formed within one seed. Columns 2 and 3: the root mean square of the stream "
            "(the post-ReLU activation; its root mean square is taken over all elements of all those images pooled together) after the last "
            "convolution (plain) or block (residual) divided by the same quantity after the stem's ReLU and pooling. Column 4: the factor "
            f"{math.sqrt(bt):.4f}^d, the square root of {bt:.4f} because a root mean square scales with the square root of a variance, where {bt:.4f} is the share of the 9 taps of a 3 by 3 window with padding 1 that land on a real pixel of a 4 by 4 "
            f"map, averaged over positions (on 8 by 8 it is {border_taps(8):.4f}, on 28 by 28 it is {border_taps(28):.4f}). Columns 5 and 6: "
            "the Frobenius norm of the weight gradient of the first body convolution (the one farthest from the loss) of the mean cross-entropy of the images.\n\n"
            + _table(["d (convolutions)", "plain: stream", "residual: stream", "plain: border alone", "plain: first-layer gradient", "residual: first-layer gradient"], rows))


# ------------------------------------------------------------------------------------------------ blocks and ResNets


def conv_cost(cin, cout, k, h_out):
    return cout * cin * k * k, h_out * h_out * cout * cin * k * k


def block_costs():
    rows = []

    def add(name, parts, h):
        params = macs = 0
        for cin, cout, k, hh in parts:
            p, m = conv_cost(cin, cout, k, hh)
            params, macs = params + p, macs + m
        rows.append((name, _int(params), _int(macs), _int(2 * macs)))
        return params, macs

    bp, bm = add("basic block, 256 to 256 (3x3, 3x3)", [(256, 256, 3, 56), (256, 256, 3, 56)], 56)
    tp, tm = add("bottleneck, 256 to 256 (1x1 to 64, 3x3, 1x1 to 256)", [(256, 64, 1, 56), (64, 64, 3, 56), (64, 256, 1, 56)], 56)
    add("basic block, 64 to 64 (3x3, 3x3)", [(64, 64, 3, 56), (64, 64, 3, 56)], 56)
    add("projection shortcut 1x1, stride 2, 256 to 512 (28x28 output)", [(256, 512, 1, 28)], 28)
    add("identity shortcut", [], 0)
    return ("Convolution weights (no bias, no normalization parameters) and multiply-adds of blocks on 56 by 56 maps (stride 1; the projection "
            "shortcut has a 28 by 28 output).\n\n" + _table(["block", "parameters", "multiply-adds", "FLOPs"], rows)
            + f"\n\nThe bottleneck holds {bp / tp:.2f} times fewer weights than the basic block of the same width ({_int(bp)} against {_int(tp)}), "
            f"and the same ratio of multiply-adds ({bm / tm:.2f}).")


RESNET18 = ("basic", (64, 128, 256, 512), (2, 2, 2, 2))
RESNET50 = ("bottleneck", (64, 128, 256, 512), (3, 4, 6, 3))


def resnet_count(spec, size=224, classes=1000):
    """Per-stage (parameters, multiply-adds) of a ResNet with the stride in the 3x3 convolution of the first block of stages 2 to 4.
    Parameters include the two parameters per channel (gain and shift) of every batch normalization and the classifier bias."""
    kind, widths, depths = spec
    exp = 1 if kind == "basic" else 4
    h = cv.out_size(size, 7, 2, 3)
    rows = {}
    p, m = conv_cost(3, 64, 7, h)
    stem = (p + 2 * 64, m)
    h = cv.out_size(h, 3, 2, 1)
    cin = 64
    stages = []
    for si, (w, n) in enumerate(zip(widths, depths)):
        sp = sm = 0
        for b in range(n):
            stride = 2 if (si > 0 and b == 0) else 1
            ho = cv.out_size(h, 3, stride, 1)
            cout = w * exp
            parts, bn = [], 0
            if kind == "basic":
                parts = [(cin, w, 3, ho), (w, w, 3, ho)]
                bn = 2 * (w + w)
            else:
                parts = [(cin, w, 1, h), (w, w, 3, ho), (w, cout, 1, ho)]
                bn = 2 * (w + w + cout)
            if stride != 1 or cin != cout:
                parts.append((cin, cout, 1, ho))
                bn += 2 * cout
            for a, c, k, hh in parts:
                pp, mm = conv_cost(a, c, k, hh)
                sp, sm = sp + pp, sm + mm
            sp += bn
            cin, h = cout, ho
        stages.append((sp, sm, cin, h))
    fc = (cin * classes + classes, cin * classes)
    return stem, stages, fc


def resnet_counts():
    rows = []
    totals = {}
    for name, spec in (("ResNet-18", RESNET18), ("ResNet-50", RESNET50)):
        stem, stages, fc = resnet_count(spec)
        rows.append((name, "stem 7x7, stride 2 (112 x 112)", _int(stem[0]), _int(stem[1])))
        for i, (sp, sm, c, h) in enumerate(stages, 1):
            rows.append(("", f"stage {i}: {RESNET18[2][i - 1] if name == 'ResNet-18' else RESNET50[2][i - 1]} blocks, {c} x {h} x {h}",
                         _int(sp), _int(sm)))
        rows.append(("", "classifier (global average pool, linear)", _int(fc[0]), _int(fc[1])))
        tp = stem[0] + sum(s[0] for s in stages) + fc[0]
        tm = stem[1] + sum(s[1] for s in stages) + fc[1]
        rows.append(("", "total", _int(tp), _int(tm)))
        totals[name] = (tp, tm)
    return ("Parameters and multiply-adds of ResNet-18 (basic blocks) and ResNet-50 (bottleneck blocks) on a 3 by 224 by 224 image, 1000 "
            "classes. The stem is followed by a 3x3 max pool of stride 2 (112 to 56). The first block of stages 2 to 4 has stride 2 in its "
            "3x3 convolution and a 1x1 projection shortcut, as does the first block of stage 1 of ResNet-50 (64 to 256 channels); parameters "
            "count every convolution weight, two numbers (gain, shift) per channel of every batch normalization, and the classifier weights "
            "and bias; multiply-adds count convolutions and the classifier.\n\n"
            + _table(["network", "part", "parameters", "multiply-adds"], rows, "llrr"))


def shortcut_alignment():
    rows = []
    for i in range(4):
        rows.append((i, f"{2 * i - 1} .. {2 * i + 1}", 2 * i, f"{2 * i} .. {2 * i + 1}", _fx(2 * i + 0.5, 1)))
    ramp = list(range(8))
    strided = [ramp[2 * i] for i in range(4)]
    pooled = [(ramp[2 * i] + ramp[2 * i + 1]) / 2 for i in range(4)]
    return ("Which input columns feed output i of a stride-2 shortcut, along one axis. The 3x3 convolution of the block (stride 2, padding 1) "
            "reads columns 2i - 1 .. 2i + 1, centred on column 2i.\n\n"
            + _table(["output i", "3x3 stride 2, pad 1 reads", "1x1 stride 2 reads", "2x2 average pool reads", "centre of the average"], rows)
            + f"\n\nOn the ramp x[j] = j (j = 0 .. 7) the strided 1x1 shortcut (weight 1) gives {strided} and the average pool gives {pooled}: "
            "the same values shifted by half a pixel.")


# ------------------------------------------------------------------------------------------------ batch norm folding

FOLD_GAMMA = [1.5, 0.5, 1.0, 2.0]
FOLD_BETA = [0.1, -0.3, 0.0, 0.2]
FOLD_BIAS = [0.05, -0.05, 0.1, 0.0]
FOLD_EPS = 1e-5


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _conv_values(w, b, img):
    """3 by 3 convolution, padding 1, of one 8 by 8 image: [4][64], taps summed in kernel order."""
    z, _ = conv_fwd(G1, w, b, [img])
    return z


def _conv_values_f32(w, b, img):
    """The same convolution with the weights, the image, every product and every running sum rounded to 32 bits."""
    w32 = [[_f32(v) for v in r] for r in w]
    xp = [_f32(v) for v in G1.pad([list(img)])]
    out = []
    for wo, bo in zip(w32, b):
        o = []
        for ix in G1.idx:
            acc = 0.0
            for wv, t in zip(wo, ix):
                acc = _f32(acc + _f32(wv * xp[t]))
            o.append(_f32(acc + _f32(bo)))
        out.append(o)
    return out


def bn_fold():
    p = toy_init()
    w, b = p["W1"], FOLD_BIAS
    train, test = _train(), _test()
    # per-channel statistics of the convolution output over the 60 training images and their 64 positions (divisor 3840)
    sums, sq = [0.0] * C1, [0.0] * C1
    for img, _ in train:
        z = _conv_values(w, b, img)
        for o in range(C1):
            for v in z[o]:
                sums[o] += v
    cnt = len(train) * 64
    mu = [s / cnt for s in sums]
    for img, _ in train:
        z = _conv_values(w, b, img)
        for o in range(C1):
            for v in z[o]:
                sq[o] += (v - mu[o]) ** 2
    var = [s / cnt for s in sq]
    scale = [FOLD_GAMMA[o] / math.sqrt(var[o] + FOLD_EPS) for o in range(C1)]
    wf = [[scale[o] * v for v in w[o]] for o in range(C1)]
    bf = [scale[o] * (b[o] - mu[o]) + FOLD_BETA[o] for o in range(C1)]
    d64 = d32 = 0.0
    for img, _ in test:
        z = _conv_values(w, b, img)
        zf = _conv_values(wf, bf, img)
        for o in range(C1):
            for q in range(64):
                ref = FOLD_GAMMA[o] * (z[o][q] - mu[o]) / math.sqrt(var[o] + FOLD_EPS) + FOLD_BETA[o]
                d64 = max(d64, abs(ref - zf[o][q]))
        z32 = _conv_values_f32(w, [_f32(v) for v in b], img)
        zf32 = _conv_values_f32(wf, [_f32(v) for v in bf], img)
        for o in range(C1):
            sc32, mu32, bt32 = _f32(scale[o]), _f32(mu[o]), _f32(FOLD_BETA[o])
            for q in range(64):
                ref = _f32(_f32(_f32(z32[o][q] - mu32) * sc32) + bt32)
                d32 = max(d32, abs(ref - zf32[o][q]))
    rows = [(o, _fx(mu[o], 4), _fx(var[o], 4), _fx(scale[o], 4), _fx(bf[o], 4)) for o in range(C1)]
    return (f"The first convolution of the initialized toy network ({INIT_NAME}: 4 filters of 3 by 3 on the 8 by 8 image, padding 1) with the biases "
            f"b = {FOLD_BIAS}, followed by batch normalization with gain gamma = {FOLD_GAMMA}, shift beta = {FOLD_BETA}, epsilon = {FOLD_EPS:g}. "
            "The running mean mu and variance sigma2 (divisor N H W) of each channel are those of the convolution output over the 60 training "
            "images (3,840 values per channel, summed in image order, then row-major). Folded: s = gamma / sqrt(sigma2 + epsilon), "
            "w' = s w, b' = s (b - mu) + beta.\n\n"
            + _table(["channel", "mu", "sigma2", "s", "folded bias b'"], rows)
            + f"\n\nOver the 90 test images and all 4 * 64 outputs of each, the largest difference between the convolution followed by batch "
            f"normalization and the single folded convolution is {'below 1e-13' if d64 < 1e-13 else _sci(d64)} in 64-bit arithmetic "
            f"(criterion 1e-13) and {'below 1e-5' if d32 < 1e-5 else _sci(d32)} when the weights, the image, every product and every running sum are rounded to "
            "32 bits (criterion 1e-5; the printed bound does not depend on the order of the sums).")


def bn_before_conv():
    mu, var, gamma, beta = 0.4, 0.2, 1.2, 0.3
    s = gamma / math.sqrt(var + FOLD_EPS)
    t = beta - mu * s
    p = toy_init()
    w = p["W1"]
    img = list(toy_data()[0][0])
    # unfused: normalize the image, then convolve with zero padding; folded: weights scaled by s, bias b' = t * sum of the weights
    normed = [s * v + t for v in img]
    z_ref, _ = conv_fwd(G1, w, [0.0] * C1, [normed])
    wf = [[s * v for v in r] for r in w]
    bf = [t * sum(r) for r in w]
    z_f, _ = conv_fwd(G1, wf, bf, [img])
    groups = {"interior (36 positions)": [], "edge, not corner (24)": [], "corner (4)": []}
    for q in range(64):
        i, j = divmod(q, 8)
        k = (i in (0, 7)) + (j in (0, 7))
        key = ("interior (36 positions)", "edge, not corner (24)", "corner (4)")[k]
        groups[key].append(q)
    rows = []
    for key, qs in groups.items():
        e = max(abs(z_ref[o][q] - z_f[o][q]) for o in range(C1) for q in qs)
        rows.append((key, "below 1e-13" if e < 1e-13 else _fx(e, 4)))
    return (f"Batch normalization of a single channel (mu = {mu}, sigma2 = {var}, gamma = {gamma}, beta = {beta}, epsilon = {FOLD_EPS:g}, so "
            f"s = {s:.4f} and the shift of the affine map is t = beta - mu s = {t:.4f}) applied to training image 0 and then followed by the four "
            "3 by 3 filters of the initialized network with padding 1; against one convolution with the filters scaled by s and the bias "
            "t * (sum of the 9 taps). Largest difference at each kind of output position.\n\n"
            + _table(["output positions", "largest difference"], rows, "lr"))


# ------------------------------------------------------------------------------------------------ feature pyramids

PYRAMID_INPUT = 640
PYRAMID_C = 256
# ResNet-50 stage outputs (channels) at strides 8, 16, 32 (stages 2, 3, 4)
RESNET50_OUT = {3: 512, 4: 1024, 5: 2048}


def _resnet50_stack():
    """(k, s, d) of the layers on the main path of ResNet-50: only the 3x3 convolutions (and the stem and pool) widen the field."""
    stack = [(7, 2, 1), (3, 2, 1)]
    marks = {}
    for si, n in enumerate(RESNET50[2]):
        for b in range(n):
            stack.append((3, 2 if (si > 0 and b == 0) else 1, 1))
        marks[si + 2] = len(stack)
    return stack, marks


def pyramid_levels():
    stack, marks = _resnet50_stack()
    rec = cv.receptive(stack)
    rows = []
    for lvl in (3, 4, 5):
        r, j = rec[marks[lvl] - 1]
        side = PYRAMID_INPUT // 2 ** lvl
        rows.append((lvl, 2 ** lvl, f"{RESNET50_OUT[lvl]} x {side} x {side}", j, r, _fx(32 / j, 2), _fx(256 / j, 2)))
    return ("The three levels of a ResNet-50 on a 3 by 640 by 640 input: stride 2^level, output shape, the jump and the receptive field along one axis "
            "by the recursion of Chapter 2 along the main path of the blocks (their 3x3 convolutions; the identity shortcut adds nothing), "
            "and how many cells an object 32 and 256 pixels across spans along one axis.\n\n"
            + _table(["level", "stride", "output shape", "jump j", "receptive field r", "cells across a 32 px object", "cells across a 256 px object"], rows))


def fpn_rows():
    """The operations of an FPN (lateral 1x1, nearest upsample, sum, 3x3 smoothing) and the PAN path on the three levels."""
    side = {l: PYRAMID_INPUT // 2 ** l for l in (3, 4, 5)}
    rows = []

    def add(step, shape, params, macs):
        rows.append((step, shape, params, macs))

    c = PYRAMID_C
    for l in (5, 4, 3):
        add(f"lateral 1x1: F{l} ({RESNET50_OUT[l]}) to {c}" + ("  (gives M5)" if l == 5 else ""), f"{c} x {side[l]} x {side[l]}",
            RESNET50_OUT[l] * c + c, side[l] ** 2 * c * RESNET50_OUT[l])
        if l < 5:
            add(f"upsample x2 of M{l + 1}, add  (gives M{l})", f"{c} x {side[l]} x {side[l]}", 0, 0)
        add(f"smooth 3x3: M{l}, {c} to {c}  (gives P{l})", f"{c} x {side[l]} x {side[l]}", c * c * 9 + c, side[l] ** 2 * c * c * 9)
    for l in (3, 4):
        add(f"PAN: downsample 3x3 stride 2 of N{l}, add P{l + 1} (gives N{l + 1})", f"{c} x {side[l + 1]} x {side[l + 1]}", c * c * 9 + c,
            side[l + 1] ** 2 * c * c * 9)
    return rows


def fpn_table():
    rows = fpn_rows()
    tp = sum(r[2] for r in rows)
    tm = sum(r[3] for r in rows)
    top = sum(r[2] for r in rows if not r[0].startswith("PAN"))
    body = [(a, b, _int(c), _int(d)) for a, b, c, d in rows]
    return (f"Feature pyramid on the ResNet-50 levels F3, F4, F5 of a 3 by {PYRAMID_INPUT} by {PYRAMID_INPUT} input, {PYRAMID_C} channels at every "
            "level, parameters with biases. Top-down path: the lateral projection of F5 is M5; each lower level adds the 2x nearest-neighbour "
            "upsampling of the merged map M of the level above to its own lateral projection, and a 3x3 convolution of every merged map M gives P. Bottom-up path (N3 = P3, then N4 and N5) halves the map with a strided "
            "3x3 convolution and adds the pyramid level of the same size.\n\n"
            + _table(["step", "output shape", "parameters", "multiply-adds"], body, "llrr")
            + f"\n\nTop-down path: {_int(top)} parameters; with the bottom-up path: {_int(tp)}; multiply-adds in all: {_int(tm)}.")


YOLO_LEVELS = {  # level: (backbone channels P, neck channels N)
    3: (64, 64), 4: (128, 128), 5: (256, 256)}
YOLO_TOPDOWN = {4: 128}  # the top-down stage at level 4 has 128 channels (neck.0), level 3 uses N3 = 64 (neck.1)


def yolo_neck_rows():
    """The shapes of the PAN-FPN neck with concatenation for the 640-pixel nano configuration of brain's YOLOv8: backbone outputs F3, F4,
    F5 of 64, 128 and 256 channels at strides 8, 16, 32 (named P3, P4, P5 in the code) and stage widths 128, 64, 64, 128, 128, 256."""
    side = {l: PYRAMID_INPUT // 2 ** l for l in (3, 4, 5)}
    f = {3: 64, 4: 128, 5: 256}
    t4 = 128
    n3, n4, n5 = 64, 128, 256
    rows = [
        ("backbone F5", f"{f[5]} x {side[5]} x {side[5]}", "", ""),
        (f"upsample F5, concat F4 ({f[5]} + {f[4]})", f"{f[5] + f[4]} x {side[4]} x {side[4]}", "", ""),
        ("C2f to T4", f"{t4} x {side[4]} x {side[4]}", "", ""),
        (f"upsample T4, concat F3 ({t4} + {f[3]})", f"{t4 + f[3]} x {side[3]} x {side[3]}", "", ""),
        ("C2f to N3 (head input 0)", f"{n3} x {side[3]} x {side[3]}", "", ""),
        (f"conv 3x3 stride 2 on N3, concat T4 ({n3} + {t4})", f"{n3 + t4} x {side[4]} x {side[4]}", _int(9 * n3 * n3), ""),
        ("C2f to N4 (head input 1)", f"{n4} x {side[4]} x {side[4]}", "", ""),
        (f"conv 3x3 stride 2 on N4, concat F5 ({n4} + {f[5]})", f"{n4 + f[5]} x {side[5]} x {side[5]}", _int(9 * n4 * n4), ""),
        ("C2f to N5 (head input 2)", f"{n5} x {side[5]} x {side[5]}", "", ""),
    ]
    return rows


def yolo_neck():
    rows = [(a, b, c) for a, b, c, _ in yolo_neck_rows()]
    return ("The PAN-FPN neck of brain's nano YOLOv8 configuration on a 640 by 640 input (backbone outputs of 64, 128 and 256 channels at strides "
            "8, 16 and 32, called F3, F4 and F5 here and P3, P4 and P5 in the code; stage widths 128, 64, 64, 128, 128, 256; T4 is the top-down stage at stride 16 and N3, N4, N5 are the "
            "outputs that go to the head): channel concatenation instead of addition, nearest-neighbour doubling "
            "for the way up and a 3x3 stride-2 convolution for the way down. The last column is the weight count of each downsampling "
            "convolution (3 * 3 * C_in * C_out, before its batch normalization); a C2f stage is counted by its output only.\n\n"
            + _table(["step", "output shape", "downsampling weights"], rows, "llr"))
