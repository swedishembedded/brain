# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Contrastive Vision-Language Learning".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: clip.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. Earlier modules
are imported, never copied: the exact and the sigmoid-approximated GELU of `mlp`, the stable-digits rule of `stable`.

* a batch of 4 image embeddings and 4 caption embeddings: the 4 by 4 similarity matrix and the symmetric loss, row by row and column
  by column, and what batch size and the logit scale do to the loss;
* a pair of tiny towers (a picture of 6 by 6 pixels and a caption of three words) trained contrastively on synthetic shapes, a gradient
  check of the whole loss, zero-shot classification by class prompts, prompt ensembling, a learned and a fixed logit scale, the batch
  size and the pairwise sigmoid loss;
* the two GELU forms that the CLIP configurations choose between;
* a perceptual distance: pixel error against a distance on the features of a fixed filter bank, and the cost of the AlexNet trunk of
  LPIPS counted from its layer table.

Arithmetic is 64-bit floating point. Random inputs come from `bookrng.generator(name)` with the names, seeds and draw orders printed
beside each table. Sums of products run left to right; the digits of a trained number are printed only when they survive the nudges of
`stable.py`.

Swedish Embedded AB implements solutions for joint image and text embeddings that are measured before they are trusted. If your team
needs expertise in contrastive vision-language models and zero-shot evaluation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import mlp
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


def _fx(v, d=4):
    return f"{v:.{d}f}"


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"seed 0x{s:016x} ({s})"


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return [x / n for x in a]


def _lse(z):
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _below(x, limit):
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else _sci(x)


def _pool_map(fn, args):
    with get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(fn, args)


def _vec(v, d=3):
    return "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"


def _mat_rows(m, d=4, head=None, col_head="j"):
    n = len(m[0])
    heads = [""] + [f"{col_head} = {j}" for j in range(n)] if head is None else head
    return _table(heads, [(f"i = {i}", *[_fx(v, d) for v in row]) for i, row in enumerate(m)])


# ------------------------------------------------------------------------------------------------ the symmetric loss on a printed batch

# Four images and four captions, as raw (unnormalized) vectors of the joint space. Caption 2 is close to image 3, so the batch has one
# real confusion; the rest are well separated.
IMG_RAW = ((3.0, 1.0, 0.0), (0.0, 2.0, 1.0), (1.0, -1.0, 3.0), (1.5, -1.0, 2.0))
TXT_RAW = ((2.0, 1.0, 0.5), (-0.5, 3.0, 1.0), (0.5, -2.0, 2.0), (1.0, -1.5, 3.0))
HAND_SCALE = 10.0


def similarity_matrix(img=IMG_RAW, txt=TXT_RAW):
    """cos(image i, caption j) for the raw vectors, as nested lists, and the two lists of unit vectors."""
    ui = [_unit(list(v)) for v in img]
    ut = [_unit(list(v)) for v in txt]
    return [[_dot(a, b) for b in ut] for a in ui], ui, ut


def symmetric_loss(cs, a):
    """Rows, columns and their mean for logits a * cs: returns (row losses, column losses, mean of all 2B terms)."""
    n = len(cs)
    s = [[a * v for v in row] for row in cs]
    rows = [_lse(s[i]) - s[i][i] for i in range(n)]
    cols = [_lse([s[i][j] for i in range(n)]) - s[j][j] for j in range(n)]
    return rows, cols, (sum(rows) + sum(cols)) / (2 * n)


def hand_batch():
    cs, ui, ut = similarity_matrix()
    s = [[HAND_SCALE * v for v in row] for row in cs]
    p = [_softmax(r) for r in s]
    q_cols = [_softmax([s[i][j] for i in range(4)]) for j in range(4)]
    q = [[q_cols[j][i] for j in range(4)] for i in range(4)]  # q[i][j] = Q_ij
    rows, cols, total = symmetric_loss(cs, HAND_SCALE)
    out = ["Four images and four captions, raw vectors of a joint space of width 3 (row i of each list is the pair i):\n"]
    out.append(_table(["pair", "image vector", "caption vector", "image norm", "caption norm"],
                      [(i, _vec(IMG_RAW[i], 1), _vec(TXT_RAW[i], 1), _fx(_norm(IMG_RAW[i]), 4), _fx(_norm(TXT_RAW[i]), 4)) for i in range(4)]))
    out.append("\nThe cosines cs_ij = (image i . caption j) / (norm * norm), image i down the side and caption j across the top:\n")
    out.append(_mat_rows(cs, 4))
    out.append(f"\nThe logits S_ij = {HAND_SCALE:g} * cs_ij (logit scale a = {HAND_SCALE:g}, temperature 0.1):\n")
    out.append(_mat_rows(s, 3))
    out.append("\nThe softmax along each row of S, P_ij = exp(S_ij) / sum_k exp(S_ik): for image i, which caption belongs to it (image to text):\n")
    out.append(_mat_rows(p, 4))
    out.append("\nThe softmax down each column of S, Q_ij = exp(S_ij) / sum_k exp(S_kj): for caption j, which image belongs to it (text to image):\n")
    out.append(_mat_rows(q, 4))
    out.append("\nThe loss of each direction is -ln of the diagonal entry of its softmax:\n")
    out.append(_table(["pair i", "image to text: -ln P_ii", "text to image: -ln Q_ii", "mean of the two"],
                      [(i, _fx(rows[i], 4), _fx(cols[i], 4), _fx((rows[i] + cols[i]) / 2, 4)) for i in range(4)]
                      + [("mean", _fx(sum(rows) / 4, 4), _fx(sum(cols) / 4, 4), _fx(total, 4))]))
    out.append(f"\nWith every cosine equal the loss would be ln 4 = {math.log(4):.4f}. The symmetric loss of the batch is {total:.4f}.")
    g22 = (p[2][2] + q[2][2] - 2) / 8
    g23 = (p[2][3] + q[2][3]) / 8
    out.append(f"\nThe derivative of the symmetric loss with respect to a logit is (P_ij + Q_ij - 2 [i = j]) / (2B): for the true pair of image 2 "
               f"it is {g22:.4f} (the loss falls if S_22 rises), and for the rival caption 3 of image 2 it is {g23:.4f}.")
    return "\n".join(out)


def hand_batch_data():
    """The cosine matrix, the row softmax P and the column softmax Q of the printed batch (for the figure)."""
    cs, _, _ = similarity_matrix()
    s = [[HAND_SCALE * v for v in row] for row in cs]
    p = [_softmax(r) for r in s]
    q = [[_softmax([s[i][j] for i in range(4)])[k] for j in range(4)] for k in range(4)]
    return cs, p, q


def batch_size_floor():
    """ln N for the batch sizes of the chapter: the loss of a model that cannot tell the candidates apart and the most information
    a batch of N can certify (the InfoNCE bound I >= ln N - L)."""
    rows = []
    for n in (2, 4, 12, 256, 4096, 32768):
        rows.append((_int(n), _int(n - 1), _fx(math.log(n), 4), _fx(math.log2(n), 3)))
    return ("A batch of B pairs gives every image B - 1 negative captions. A model that cannot tell the candidates apart pays ln B per "
            "direction, and since the information the loss can certify is at most ln B minus the loss itself, ln B is also the "
            "ceiling on what a batch of B can measure.\n\n"
            + _table(["B", "negatives per image", "ln B (nats)", "log2 B (bits)"], rows))


def scale_sweep():
    """The loss of the printed batch and of a perfectly ranked one as the logit scale grows, and the best finite scale."""
    cs, _, _ = similarity_matrix()
    good = [[0.9 if i == j else 0.1 for j in range(4)] for i in range(4)]
    scales = (1.0, 5.0, 10.0, 1.0 / 0.07, 30.0, 50.0, 100.0, 1000.0)
    rows = []
    for a in scales:
        l1 = symmetric_loss(cs, a)[2]
        l2 = symmetric_loss(good, a)[2]
        rows.append((f"{a:g}" if a != 1.0 / 0.07 else "1/0.07 = 14.285714", _fx(l1, 6), _fx(l2, 6)))
    # best finite scale of the printed batch by ternary search of the (convex in a) loss
    lo, hi = 0.01, 1000.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if symmetric_loss(cs, m1)[2] < symmetric_loss(cs, m2)[2]:
            hi = m2
        else:
            lo = m1
    best = (lo + hi) / 2
    n = len(cs)
    gaps = [max(max(cs[i]) - cs[i][i], 0.0) for i in range(n)] + [max(max(cs[i][j] for i in range(n)) - cs[j][j], 0.0) for j in range(n)]
    slope = sum(gaps) / (2 * n)
    return ("The symmetric loss as the logit scale a grows, for the printed batch (pair 2 and pair 3 are confused) and for a batch whose "
            "four diagonal cosines are 0.9 and all others 0.1.\n\n"
            + _table(["logit scale a", "printed batch", "diagonal 0.9, others 0.1"], rows)
            + f"\n\nThe printed batch has an interior minimum: a = {best:.3f}, loss {symmetric_loss(cs, best)[2]:.4f}. Past it the "
              f"loss grows linearly in a, with slope {slope:.5f}: the sum of the {sum(g > 0 for g in gaps)} gaps by which a wrong "
              f"cosine beats the diagonal of its row or column ({', '.join(f'{g:.4f}' for g in gaps if g > 0)}), divided by 2B = {2 * n}. "
              "The second batch has no minimum: its loss falls towards 0 for ever.")


# ------------------------------------------------------------------------------------------------ the synthetic world and the two towers

SIDE = 6                       # a picture is SIDE by SIDE pixels, three channels
NPIX = 3 * SIDE * SIDE         # 108 numbers, channel-major: index = 36 c + 6 r + col
COLOURS = ("red", "green", "blue", "yellow")
COLOUR_RGB = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0))
SHAPES = ("square", "frame", "cross", "diagonal")
TEMPLATES = ("a", "photo of a", "drawing of a")
# vocabulary ids: templates 0..2, colours 3..6, shapes 7..10
VOCAB = TEMPLATES + COLOURS + SHAPES
SHAPE_ROWS = (
    ("......", ".###..", ".###..", ".###..", "......", "......"),   # square
    ("......", ".####.", ".#..#.", ".#..#.", ".####.", "......"),   # frame
    ("......", "..##..", ".####.", ".####.", "..##..", "......"),   # cross
    ("#.....", "##....", ".##...", "..##..", "...##.", "....##"),   # diagonal
)
NOISE_SD = 0.1
AMPLITUDE = (0.7, 1.0)        # per-picture brightness drawn uniformly from this range
HELD_OUT = tuple(4 * k + k for k in range(4))   # classes (colour, shape) = (0,0), (1,1), (2,2), (3,3)
TRAIN_CLASSES = tuple(k for k in range(16) if k not in HELD_OUT)
TEST_PER_CLASS = 8
D_IMG_H, D_TXT_E, D_TXT_H, D_OUT = 16, 4, 12, 8
LOGIT_MAX = math.log(100.0)    # CLIP clamps the logit scale to at most 100


def class_name(k):
    return f"{COLOURS[k // 4]} {SHAPES[k % 4]}"


def render(k, amp, g):
    """A picture of class k: the shape's pixels in the colour times `amp`, background 0, then NOISE_SD * normal() added to all 108
    numbers in channel-major order (one draw each, drawn here, after `amp` was drawn by the caller)."""
    rows = SHAPE_ROWS[k % 4]
    rgb = COLOUR_RGB[k // 4]
    return [(rgb[c] * amp if rows[r][col] == "#" else 0.0) + NOISE_SD * g.normal()
            for c in range(3) for r in range(SIDE) for col in range(SIDE)]


def caption_tokens(k, template):
    return (template, 3 + k // 4, 7 + k % 4)


def shape_grids():
    out = ["The four shapes on the 6 by 6 grid (`#` is a pixel of the shape, `.` background); a picture of class (colour, shape) puts "
           "the shape's pixels at colour * brightness in the three channels, the background at 0, and adds noise to every number:\n"]
    out.append(_table(["row", *SHAPES], [(r, *[f"`{SHAPE_ROWS[s][r]}`" for s in range(4)]) for r in range(SIDE)], "lllll"))
    out.append(f"\nColours (R, G, B): " + ", ".join(f"{COLOURS[i]} {tuple(int(v) for v in COLOUR_RGB[i])}" for i in range(4)) +
               f". Pixels per shape: " + ", ".join(f"{SHAPES[s]} {sum(r.count('#') for r in SHAPE_ROWS[s])}" for s in range(4)) +
               f". Noise: normal with standard deviation {NOISE_SD}; brightness uniform in {AMPLITUDE}. Held-out classes (never in a "
               "training batch): " + ", ".join(class_name(k) for k in HELD_OUT) + ". Class index k = 4 * colour + shape.")
    return "\n".join(out)


def _init_name(run):
    return f"clip.init.{run}"


def init_params(run, nudge=0.0):
    """Parameters, drawn in this order from bookrng.generator(f"clip.init.{run}"), `normal() * scale` each, row by row:
    W1 [16, 108] (scale 1/sqrt(108)), Wp [8, 16] (1/sqrt(16)), then the text side: E [11, 4] (0.5), V1 [12, 12] (1/sqrt(12)),
    Vp [8, 12] (1/sqrt(12)). Biases b1 and c1 are 0; psi (the log of the logit scale) is ln(1/0.07) and the sigmoid bias is -10
    (both are only used by the variants that learn them). `nudge` is added to Wp[0][0]."""
    g = bookrng.generator(_init_name(run))

    def mat(r, c, s):
        return [[s * g.normal() for _ in range(c)] for _ in range(r)]
    p = {"W1": mat(D_IMG_H, NPIX, 1 / math.sqrt(NPIX)), "b1": [[0.0] * D_IMG_H], "Wp": mat(D_OUT, D_IMG_H, 1 / math.sqrt(D_IMG_H)),
         "E": mat(len(VOCAB), D_TXT_E, 0.5), "V1": mat(D_TXT_H, 3 * D_TXT_E, 1 / math.sqrt(3 * D_TXT_E)), "c1": [[0.0] * D_TXT_H],
         "Vp": mat(D_OUT, D_TXT_H, 1 / math.sqrt(D_TXT_H))}
    p["Wp"][0][0] += nudge
    return p


def _matvec(m, x):
    return [_dot(row, x) for row in m]


def image_forward(p, x):
    h = [math.tanh(v + b) for v, b in zip(_matvec(p["W1"], x), p["b1"][0])]
    z = _matvec(p["Wp"], h)
    n = _norm(z)
    return [v / n for v in z], (x, h, z, n)


def text_forward(p, toks):
    e = [v for t in toks for v in p["E"][t]]
    h = [math.tanh(v + b) for v, b in zip(_matvec(p["V1"], e), p["c1"][0])]
    z = _matvec(p["Vp"], h)
    n = _norm(z)
    return [v / n for v in z], (toks, e, h, z, n)


def _unit_back(du, u, n):
    d = _dot(du, u)
    return [(a - d * b) / n for a, b in zip(du, u)]


def _add_outer(g, dz, h):
    for row, d in zip(g, dz):
        for j, hv in enumerate(h):
            row[j] += d * hv


def image_backward(p, cache, du, g):
    x, h, z, n = cache
    u = [v / n for v in z]
    dz = _unit_back(du, u, n)
    _add_outer(g["Wp"], dz, h)
    dh = [_dot([p["Wp"][o][j] for o in range(D_OUT)], dz) for j in range(D_IMG_H)]
    dpre = [d * (1.0 - v * v) for d, v in zip(dh, h)]
    _add_outer(g["W1"], dpre, x)
    for j, d in enumerate(dpre):
        g["b1"][0][j] += d


def text_backward(p, cache, du, g):
    toks, e, h, z, n = cache
    u = [v / n for v in z]
    dz = _unit_back(du, u, n)
    _add_outer(g["Vp"], dz, h)
    dh = [_dot([p["Vp"][o][j] for o in range(D_OUT)], dz) for j in range(D_TXT_H)]
    dpre = [d * (1.0 - v * v) for d, v in zip(dh, h)]
    _add_outer(g["V1"], dpre, e)
    for j, d in enumerate(dpre):
        g["c1"][0][j] += d
    de = [_dot([p["V1"][o][j] for o in range(D_TXT_H)], dpre) for j in range(3 * D_TXT_E)]
    for slot, t in enumerate(toks):
        for j in range(D_TXT_E):
            g["E"][t][j] += de[slot * D_TXT_E + j]


def _zeros_like(p):
    return {k: [[0.0] * len(r) for r in v] for k, v in p.items()}


def batch_loss(p, ui, ut, a, kind, bias):
    """Loss of a batch of B unit image vectors and B unit caption vectors, with the gradients with respect to those vectors and to the
    log logit scale psi (a = exp(psi)) and the sigmoid bias. kind "softmax": the symmetric cross-entropy (1 / 2B) sum_i (lse_j S_ij -
    S_ii) + (1 / 2B) sum_j (lse_i S_ij - S_jj); kind "sigmoid": (1 / B) sum_ij softplus(-y_ij (S_ij + bias)) with y_ij = +1 on the
    diagonal and -1 elsewhere. S_ij = a * (image i . caption j)."""
    n = len(ui)
    s = [[a * _dot(ui[i], ut[j]) for j in range(n)] for i in range(n)]
    ds = [[0.0] * n for _ in range(n)]
    dbias = 0.0
    if kind == "softmax":
        loss = 0.0
        for i in range(n):
            pr = _softmax(s[i])
            loss += _lse(s[i]) - s[i][i]
            for j in range(n):
                ds[i][j] += (pr[j] - (1.0 if i == j else 0.0)) / (2 * n)
        for j in range(n):
            col = [s[i][j] for i in range(n)]
            pc = _softmax(col)
            loss += _lse(col) - s[j][j]
            for i in range(n):
                ds[i][j] += (pc[i] - (1.0 if i == j else 0.0)) / (2 * n)
        loss /= 2 * n
    else:
        loss = 0.0
        for i in range(n):
            for j in range(n):
                y = 1.0 if i == j else -1.0
                z = y * (s[i][j] + bias)
                loss += _softplus(-z)
                ds[i][j] = -y * (1.0 - _sigmoid(z)) / n
        loss /= n
        dbias = sum(v for r in ds for v in r)
    dui = [[a * sum(ds[i][j] * ut[j][k] for j in range(n)) for k in range(D_OUT)] for i in range(n)]
    dut = [[a * sum(ds[i][j] * ui[i][k] for i in range(n)) for k in range(D_OUT)] for j in range(n)]
    dpsi = sum(ds[i][j] * s[i][j] for i in range(n) for j in range(n))
    return loss, dui, dut, dpsi, dbias


def loss_and_grad(p, batch, a, kind, bias, grad=True):
    """batch: list of (pixels, tokens). Returns (loss, gradient dict over the tower parameters plus 'psi' and 'bias')."""
    fi = [image_forward(p, x) for x, _ in batch]
    ft = [text_forward(p, t) for _, t in batch]
    loss, dui, dut, dpsi, dbias = batch_loss(p, [u for u, _ in fi], [u for u, _ in ft], a, kind, bias)
    if not grad:
        return loss, None
    g = _zeros_like(p)
    for (_, c), d in zip(fi, dui):
        image_backward(p, c, d, g)
    for (_, c), d in zip(ft, dut):
        text_backward(p, c, d, g)
    g["psi"], g["bias"] = dpsi, dbias
    return loss, g


# ------------------------------------------------------------------------------------------------ training

LR = 0.01
PAIRS_SEEN = 3600              # every configuration sees the same number of (picture, caption) pairs: steps * batch size
STEPS_MAIN = 300               # batch of 12 pairs
PSI0 = math.log(1.0 / 0.07)
BIAS0 = -10.0


def _cfg(kind="softmax", scale="learned", a_fixed=None, batch=12):
    return {"kind": kind, "scale": scale, "a_fixed": a_fixed, "batch": batch, "steps": PAIRS_SEEN // batch}


CONFIGS = {
    "main": _cfg(),
    "fixed_1": _cfg(scale="fixed", a_fixed=1.0),
    "fixed_14": _cfg(scale="fixed", a_fixed=1.0 / 0.07),
    "fixed_100": _cfg(scale="fixed", a_fixed=100.0),
    "sigmoid": _cfg(kind="sigmoid"),
    "batch_4": _cfg(batch=4),
    "batch_2": _cfg(batch=2),
}


@lru_cache(maxsize=None)
def test_set():
    """The test pictures, from bookrng.generator("clip.test"): for class k = 0 .. 15 in order, 8 pictures, each one `amp` =
    0.7 + 0.3 * uniform() and then its 108 noise draws (channel-major). Returns a list of (class, pixels)."""
    g = bookrng.generator("clip.test")
    out = []
    for k in range(16):
        for _ in range(TEST_PER_CLASS):
            amp = AMPLITUDE[0] + (AMPLITUDE[1] - AMPLITUDE[0]) * g.uniform()
            out.append((k, render(k, amp, g)))
    return out


def prompt_vectors(p):
    """Unit vectors of the 16 class prompts under each of the 3 templates: [template][class]."""
    return [[text_forward(p, caption_tokens(k, t))[0] for k in range(16)] for t in range(len(TEMPLATES))]


def ensemble_vectors(pv):
    """Prompt ensembling: per class, the mean of the template vectors, scaled back to length 1."""
    return [_unit([sum(pv[t][k][d] for t in range(len(pv))) / len(pv) for d in range(D_OUT)]) for k in range(16)]


def zero_shot(p):
    """Zero-shot accuracy of the towers p on the test pictures: each picture goes to the class whose prompt vector has the largest dot
    product with it (ties to the first). Returns {method: (all, seen classes, held-out classes)} for each template alone and the
    ensemble."""
    imgs = [(k, image_forward(p, x)[0]) for k, x in test_set()]
    pv = prompt_vectors(p)
    methods = {TEMPLATES[t]: pv[t] for t in range(len(TEMPLATES))}
    methods["ensemble"] = ensemble_vectors(pv)
    res = {}
    for name, vecs in methods.items():
        ok = []
        for k, u in imgs:
            sc = [_dot(u, v) for v in vecs]
            ok.append((k, sc.index(max(sc)) == k))
        seen = [o for k, o in ok if k not in HELD_OUT]
        held = [o for k, o in ok if k in HELD_OUT]
        res[name] = (sum(o for _, o in ok) / len(ok), sum(seen) / len(seen), sum(held) / len(held))
    return res


def template_diagnostics(p):
    """For the test pictures: how far apart the three prompt vectors of a class are, and where the templates disagree. Returns
    {"mean_norm": mean over the 16 classes of the length of the mean of the three unit prompt vectors (1 if they coincide),
    "disagree": pictures on which the three single-template answers are not all the same, "gain": pictures the ensemble gets right and at
    least one single template gets wrong, "loss": pictures the ensemble gets wrong and at least one template gets right}."""
    imgs = [(k, image_forward(p, x)[0]) for k, x in test_set()]
    pv = prompt_vectors(p)
    ens = ensemble_vectors(pv)
    norms = [_norm([sum(pv[t][k][d] for t in range(3)) / 3 for d in range(D_OUT)]) for k in range(16)]
    disagree = gain = loss = 0
    for k, u in imgs:
        ans = []
        for vecs in (*pv, ens):
            sc = [_dot(u, v) for v in vecs]
            ans.append(sc.index(max(sc)))
        singles, e = ans[:3], ans[3]
        disagree += len(set(singles)) > 1
        gain += e == k and any(a != k for a in singles)
        loss += e != k and any(a == k for a in singles)
    return {"mean_norm": sum(norms) / 16, "disagree": disagree, "gain": gain, "loss": loss}


def eval_loss(p, a, kind, bias):
    """The loss of the fixed evaluation batch: the first test picture of each of the 12 training classes, with the caption of
    template 0 ("a red frame")."""
    first = {}
    for k, x in test_set():
        first.setdefault(k, x)
    batch = [(first[k], caption_tokens(k, 0)) for k in TRAIN_CLASSES]
    return loss_and_grad(p, batch, a, kind, bias, grad=False)[0]


def _leaves(p):
    return [r for k in sorted(p) if isinstance(p[k], list) for r in p[k]]


def train(run, variant="main", nudge=0.0):
    """Train the towers of run `run` for the configuration `variant`. Pictures come from bookrng.generator(f"clip.stream.{run}.{variant}"):
    per step one shuffle of the 12 training classes (Fisher-Yates), the first B taken in order, and for each of them in turn the
    template index `integer(3)`, the brightness `amp = 0.7 + 0.3 * uniform()` and the 108 noise draws. Adam (beta_1 0.9, beta_2 0.999,
    epsilon 1e-8, bias correction, learning rate 0.01, no weight decay) on all tower parameters, and on psi (and the sigmoid bias
    when the loss has one); the softmax variant with a learned scale clamps psi to [0, ln 100] after each update. Returns
    {"curve": {step: (a, eval loss, zero-shot dict)}, "params": p}."""
    cfg = CONFIGS[variant]
    p = init_params(run, nudge)
    g = bookrng.generator(f"clip.stream.{run}.{variant}")
    psi = PSI0 if cfg["scale"] == "learned" else math.log(cfg["a_fixed"])
    bias = BIAS0 if cfg["kind"] == "sigmoid" else 0.0
    if cfg["kind"] == "sigmoid":
        psi = math.log(10.0)
    lv = _leaves(p)
    m = [[0.0] * len(r) for r in lv]
    v = [[0.0] * len(r) for r in lv]
    ms = {"psi": 0.0, "bias": 0.0}
    vs = {"psi": 0.0, "bias": 0.0}
    steps, bsz = cfg["steps"], cfg["batch"]
    every = steps // 30
    curve = {}
    for t in range(steps + 1):
        if t % every == 0:
            curve[t] = (math.exp(psi), eval_loss(p, math.exp(psi), cfg["kind"], bias), zero_shot(p))
        if t == steps:
            break
        order = g.shuffle(list(TRAIN_CLASSES))[:bsz]
        batch = []
        for k in order:
            tmpl = g.integer(len(TEMPLATES))
            amp = AMPLITUDE[0] + (AMPLITUDE[1] - AMPLITUDE[0]) * g.uniform()
            batch.append((render(k, amp, g), caption_tokens(k, tmpl)))
        _, gr = loss_and_grad(p, batch, math.exp(psi), cfg["kind"], bias)
        c1, c2 = 1 - 0.9 ** (t + 1), 1 - 0.999 ** (t + 1)
        for r, grr, mr, vr in zip(lv, _leaves(gr), m, v):
            for i in range(len(r)):
                mr[i] = 0.9 * mr[i] + 0.1 * grr[i]
                vr[i] = 0.999 * vr[i] + 0.001 * grr[i] * grr[i]
                r[i] -= LR * (mr[i] / c1) / (math.sqrt(vr[i] / c2) + 1e-8)
        learn = ["psi"] if cfg["scale"] == "learned" else []
        if cfg["kind"] == "sigmoid":
            learn = ["psi", "bias"]
        for name in learn:
            d = gr[name]
            ms[name] = 0.9 * ms[name] + 0.1 * d
            vs[name] = 0.999 * vs[name] + 0.001 * d * d
            step = LR * (ms[name] / c1) / (math.sqrt(vs[name] / c2) + 1e-8)
            if name == "psi":
                psi -= step
                if cfg["kind"] == "softmax":
                    psi = min(max(psi, 0.0), LOGIT_MAX)
            else:
                bias -= step
    return {"curve": curve, "params": p, "psi": psi, "bias": bias, "templates": template_diagnostics(p)}


# ------------------------------------------------------------------------------------------------ the gradient check

def _gc_batch():
    """Four (picture, tokens) pairs for the gradient check: classes 1, 6, 11, 12, each with the draws of a training pair."""
    g = bookrng.generator("clip.gradcheck")
    out = []
    for k in (1, 6, 11, 12):
        tmpl = g.integer(len(TEMPLATES))
        amp = AMPLITUDE[0] + (AMPLITUDE[1] - AMPLITUDE[0]) * g.uniform()
        out.append((render(k, amp, g), caption_tokens(k, tmpl)))
    return out


def _gc_one(kind):
    p = init_params(0)
    batch = _gc_batch()
    psi, bias = 1.3, -0.4
    _, g = loss_and_grad(p, batch, math.exp(psi), kind, bias)
    worst, fails, total = 0.0, 0, 0

    def check(analytic, loss_at):
        nonlocal worst, fails, total
        num = (loss_at(1e-6) - loss_at(-1e-6)) / 2e-6
        err = abs(analytic - num)
        total += 1
        fails += err > 1e-7 + 1e-5 * max(abs(analytic), abs(num))
        worst = max(worst, err)

    for name in sorted(p):
        for r, gr in zip(p[name], g[name]):
            for t in range(len(r)):
                old = r[t]

                def at(h, r=r, t=t, old=old):
                    r[t] = old + h
                    v = loss_and_grad(p, batch, math.exp(psi), kind, bias, grad=False)[0]
                    r[t] = old
                    return v
                check(gr[t], at)
    check(g["psi"], lambda h: loss_and_grad(p, batch, math.exp(psi + h), kind, bias, grad=False)[0])
    if kind == "sigmoid":
        check(g["bias"], lambda h: loss_and_grad(p, batch, math.exp(psi), kind, bias + h, grad=False)[0])
    return worst, fails, total


def gradient_check():
    kinds = ("softmax", "sigmoid")
    res = _pool_map(_gc_one, list(kinds))
    rows = [(k, _int(t), _below(w, 1e-8) if w < 1e-8 else _below(w, 1e-7) if w < 1e-7 else _sci(w), t - f) for k, (w, f, t) in zip(kinds, res)]
    return ("The initialized towers (run 0) on a batch of four pairs (classes 1, 6, 11, 12 with templates drawn from "
            f"`bookrng.generator(\"clip.gradcheck\")`, {_seed('clip.gradcheck')}: for each pair in turn the template `integer(3)`, the "
            "brightness `uniform()` and the 108 noise values), the logit scale a = exp(1.3) and, for the sigmoid loss, the bias -0.4. The "
            "analytic gradient of the loss with respect to every tower weight, to the log scale psi and (sigmoid) to the bias against the "
            "central difference (L(w + h) - L(w - h)) / (2h), h = 1e-6 in 64-bit arithmetic, with the gate |analytic - numeric| <= 1e-7 + "
            "1e-5 max(|analytic|, |numeric|).\n\n"
            + _table(["loss", "parameters checked", "largest absolute difference", "pass the gate"], rows))


# ------------------------------------------------------------------------------------------------ the runs and their tables

RUNS = (0, 1, 2)
MAIN_STEPS = (0, 10, 20, 40, 60, 100, 200, 300)


def _task(args):
    run, variant, nudge = args
    r = train(run, variant, nudge)
    return {"curve": r["curve"], "psi": r["psi"], "bias": r["bias"], "templates": r["templates"]}


@lru_cache(maxsize=None)
def all_runs():
    """Every training run of the chapter, keyed (run, variant, nudge): each configuration for the three runs, each also from the four
    nudged starts of stable.py."""
    jobs = [(run, v, nd) for v in CONFIGS for run in RUNS for nd in (0.0, *stab.NUDGES)]
    return dict(zip(jobs, _pool_map(_task, jobs)))


def _get(run, variant, nudge=0.0):
    return all_runs()[(run, variant, nudge)]


def _twins(f, run, variant):
    return f(_get(run, variant)), [f(_get(run, variant, n)) for n in stab.NUDGES]


def _mean_twins(f, variant):
    base = sum(f(_get(r, variant)) for r in RUNS) / len(RUNS)
    return base, [sum(f(_get(r, variant, n)) for r in RUNS) / len(RUNS) for n in stab.NUDGES]


def _t3(v, ws):
    return stab.text(v, ws, lambda x: f"{x:.3f}")


def _t2(v, ws):
    return stab.text(v, ws, lambda x: f"{x:.2f}")


def _t4(v, ws):
    return stab.text(v, ws, lambda x: f"{x:.4f}")


def training_setup():
    name = _init_name(0)
    cfg = CONFIGS["main"]
    return (f"The towers. Picture tower: the 108 numbers (channel-major) go through a linear layer to 16 units with a bias, tanh, and a "
            f"linear projection without bias to 8 numbers, scaled to length 1. Caption tower: the three words of a caption (a template, a "
            f"colour, a shape; template = `a`, `photo of a` or `drawing of a`, each one token) are looked up in an embedding table of "
            f"{len(VOCAB)} rows of 4 numbers, the three rows are concatenated (12 numbers; the slot of a word is its position), a "
            f"linear layer to 12 units with a bias, tanh, and a linear projection without bias to 8 numbers, scaled to length 1. "
            f"Parameters: {_int(16 * NPIX + 16 + 8 * 16)} in the picture tower and {_int(len(VOCAB) * 4 + 12 * 12 + 12 + 8 * 12)} in the "
            f"caption tower. Initialization of run r from `{_init_name('r')}` (run 0: {_seed(name)}; runs 1 and 2 use the names with 1 and 2), "
            f"`normal() * scale` for every entry, matrix by matrix and row by row, in this order: W1 [16, 108] with scale 1/sqrt(108), "
            f"Wp [8, 16] with 1/sqrt(16), E [11, 4] with 0.5, V1 [12, 12] with 1/sqrt(12), Vp [8, 12] with 1/sqrt(12); both biases 0. "
            f"The logit scale is a = exp(psi) with psi starting at ln(1/0.07) = {PSI0:.4f} (a = {math.exp(PSI0):.3f}) and clamped to "
            f"[0, ln 100] after every update. The classes are the 16 (colour, shape) pairs; the 4 held-out classes never appear in a "
            f"training batch. Each step takes one batch of {cfg['batch']} pairs, one for each of the first {cfg['batch']} classes "
            f"of a fresh `shuffle` of the 12 training classes, so that no two captions of a batch are equal. The training classes, "
            f"before each shuffle, are {list(TRAIN_CLASSES)} in ascending order (the held-out classes {list(HELD_OUT)} removed); the rows of "
            f"the embedding table E are the templates (rows 0 to 2), the colours (3 to 6) and the shapes (7 to 10). Pictures and "
            f"templates come from `bookrng.generator(\"clip.stream.r.main\")` (run 0: {_seed('clip.stream.0.main')}), the same stream that "
            f"does the shuffle: each step first shuffles, then, for each pair in the shuffled order, draws the template `integer(3)`, the "
            f"brightness `0.7 + 0.3 * uniform()` and the 108 noise values (channel-major, one `normal()` times {NOISE_SD} each). Adam "
            f"(beta_1 0.9, beta_2 0.999, epsilon 1e-8, bias correction, learning rate {LR}, no weight decay) on all tower weights and, "
            f"in the settings with a learned scale, on psi = ln a itself (the gradient is taken with respect to psi), {cfg['steps']} steps. "
            f"The 128 test pictures (8 per class) come from `bookrng.generator(\"clip.test\")` ({_seed('clip.test')}): for class 0 to 15 "
            f"in order and 8 pictures each, the brightness and then the 108 noise values. A picture is classified by the class whose prompt "
            f"vector has the largest dot product with its vector (the first on a tie). {stab.RULE} The nudge is added to Wp[0][0], the "
            f"first weight of the picture projection, which every picture uses.")


def training_curve():
    rows = []
    for t in MAIN_STEPS:
        a, wa = _twins(lambda r, t=t: r["curve"][t][0], 0, "main")
        l, wl = _twins(lambda r, t=t: r["curve"][t][1], 0, "main")
        s, ws = _twins(lambda r, t=t: r["curve"][t][2]["ensemble"][1], 0, "main")
        h, wh = _twins(lambda r, t=t: r["curve"][t][2]["ensemble"][2], 0, "main")
        rows.append((t, stab.text(a, wa, lambda x: f"{x:.2f}"), _t4(l, wl), _t3(s, ws), _t3(h, wh)))
    return ("Run 0, the main configuration. The evaluation loss is the symmetric loss of the 12 training classes (the first test picture "
            "of each, template `a`) at the current scale; accuracies are over the 16 class prompts with the ensemble of the three "
            "templates, on the 96 test pictures of the 12 training classes and the 32 of the 4 held-out classes (chance 1/16 = 0.0625). "
            "Numbers by the nudge rule above.\n\n"
            + _table(["step", "logit scale a", "evaluation loss", "accuracy, training classes", "accuracy, held-out classes"], rows))


@lru_cache(maxsize=None)
def main_curve(run):
    """The recorded curve of the main configuration of `run`, trained on its own (the same computation as the unnudged entry of
    all_runs, without the nudged copies; the figure needs only this)."""
    return train(run, "main")["curve"]


def training_curve_data():
    """Runs 0, 1, 2 (unnudged): per recorded step, the scale, the evaluation loss and the ensemble accuracies on training and
    held-out classes of run 0, and the held-out accuracy of every run."""
    c = main_curve(0)
    steps = sorted(c)
    return {"steps": steps, "scale": [c[t][0] for t in steps], "loss": [c[t][1] for t in steps],
            "seen": [c[t][2]["ensemble"][1] for t in steps], "held": [c[t][2]["ensemble"][2] for t in steps],
            "held_runs": [[main_curve(r)[t][2]["ensemble"][2] for t in steps] for r in RUNS]}


def three_runs():
    last = CONFIGS["main"]["steps"]
    rows = []
    for run in RUNS:
        name = f"clip.init.{run}"
        cells = []
        for idx in (1, 2):
            v, ws = _twins(lambda r, idx=idx: r["curve"][last][2]["ensemble"][idx], run, "main")
            cells.append(_t3(v, ws))
        best_h, wb = _twins(lambda r: max(r["curve"][t][2]["ensemble"][2] for t in r["curve"]), run, "main")
        rows.append((run, f"0x{bookrng.seed_of(name):016x}", *cells, _t3(best_h, wb)))
    return (f"The main configuration for the three initializations (the init generator of run r is `clip.init.r`, the stream generator "
            f"`clip.stream.r.main`; seeds below are the init seeds), after {last} steps, ensemble accuracy; the last column is the best "
            f"held-out accuracy at any of the 31 recorded steps (every 10th). Chance on 16 classes: 0.0625.\n\n"
            + _table(["run", "init seed", "training classes", "held-out classes", "best held-out"], rows))


def template_table():
    last = CONFIGS["main"]["steps"]
    names = list(TEMPLATES) + ["ensemble"]
    rows = []
    for run in RUNS:
        cells = []
        for nm in names:
            v, ws = _twins(lambda r, nm=nm: r["curve"][last][2][nm][0], run, "main")
            cells.append(_t3(v, ws))
        rows.append((run, *cells))
    mean = []
    for nm in names:
        v, ws = _mean_twins(lambda r, nm=nm: r["curve"][last][2][nm][0], "main")
        mean.append(_t3(v, ws))
    rows.append(("mean", *mean))
    diag = []
    for run in RUNS:
        cells = []
        for key, f in (("mean_norm", lambda r: r["templates"]["mean_norm"]), ("disagree", lambda r: r["templates"]["disagree"]),
                       ("gain", lambda r: r["templates"]["gain"]), ("loss", lambda r: r["templates"]["loss"])):
            v, ws = _twins(f, run, "main")
            cells.append(stab.text(v, ws, (lambda x: f"{x:.4f}") if key == "mean_norm" else (lambda x: f"{x:.0f}")))
        diag.append((run, *cells))
    return ("Zero-shot accuracy over all 128 test pictures and all 16 classes after the last step, with each template alone and with the "
            "ensemble of the three (the mean of the three prompt vectors per class, scaled back to length 1).\n\n"
            + _table(["run", *[f"`{n}`" if n != "ensemble" else n for n in names]], rows)
            + "\n\nWhere the templates differ, on the same towers: the mean over the 16 classes of the length of the mean of the three "
              "unit prompt vectors (1 if they coincide), the number of the 128 pictures on which the three single-template answers are "
              "not all the same, the number the ensemble gets right while some single template gets it wrong, and the number it gets wrong "
              "while some single template gets it right.\n\n"
            + _table(["run", "length of the mean prompt vector", "pictures with disagreeing templates", "ensemble gains", "ensemble losses"], diag))


def _early(variant, pairs=1200):
    """f(run result): the training-class ensemble accuracy after `pairs` pairs have been seen (a recorded step for every setting)."""
    step = pairs // CONFIGS[variant]["batch"]
    return lambda r: r["curve"][step][2]["ensemble"][1]


def scale_table():
    rows = []
    for key, label in (("fixed_1", "fixed a = 1 (temperature 1)"), ("fixed_14", "fixed a = 14.286 (temperature 0.07)"),
                       ("fixed_100", "fixed a = 100 (temperature 0.01)"), ("main", "learned, start 14.286, clamped at 100"),
                       ("sigmoid", "pairwise sigmoid loss, learned scale (start 10) and bias (start -10)")):
        last = CONFIGS[key]["steps"]
        a, wa = _mean_twins(lambda r, last=last: r["curve"][last][0], key)
        s, ws = _mean_twins(lambda r, last=last: r["curve"][last][2]["ensemble"][1], key)
        h, wh = _mean_twins(lambda r, last=last: r["curve"][last][2]["ensemble"][2], key)
        q, wq = _mean_twins(_early(key), key)
        rows.append((label, stab.text(a, wa, lambda x: f"{x:.2f}"), _t3(q, wq), _t3(s, ws), _t3(h, wh)))
    return ("The same towers, data and 3,600 pairs for each setting, mean over the three runs (0, 1, 2; the stream generators are "
            "`clip.stream.r.<setting>` with the settings fixed_1, fixed_14, fixed_100, main and sigmoid, in the order of the rows), ensemble accuracy after the last step. The scale column is the final a; the next column is the training-class accuracy after "
            "1,200 pairs.\n\n"
            + _table(["setting", "final a", "training accuracy after 1,200 pairs", "accuracy, training classes",
                      "accuracy, held-out classes"], rows, "lrrrr"))


def batch_table():
    rows = []
    for key, bsz in (("batch_2", 2), ("batch_4", 4), ("main", 12)):
        cfg = CONFIGS[key]
        last = cfg["steps"]
        l, wl = _mean_twins(lambda r, last=last: r["curve"][last][1], key)
        s, ws = _mean_twins(lambda r, last=last: r["curve"][last][2]["ensemble"][1], key)
        h, wh = _mean_twins(lambda r, last=last: r["curve"][last][2]["ensemble"][2], key)
        q, wq = _mean_twins(_early(key), key)
        rows.append((bsz, bsz - 1, cfg["steps"], _fx(math.log(bsz), 4), _t4(l, wl), _t3(q, wq), _t3(s, ws), _t3(h, wh)))
    return ("Batch size with the number of pairs seen held at 3,600 (so 1,800, 900 and 300 steps), learned scale, mean over the three "
            "runs (stream generators `clip.stream.r.batch_2`, `clip.stream.r.batch_4` and `clip.stream.r.main`). The evaluation loss is that of "
            "the 12 training classes as one batch of 12 (its chance level is ln 12 = 2.4849), whatever the batch of training.\n\n"
            + _table(["batch size", "negatives per picture", "steps", "ln B", "evaluation loss", "training accuracy after 1,200 pairs",
                      "accuracy, training classes", "accuracy, held-out classes"], rows))


# ------------------------------------------------------------------------------------------------ the two GELU forms of the CLIP configurations

def gelu_forms():
    xs = [i / 1000.0 for i in range(-6000, 6001)]
    diffs = [(abs(mlp._gelu(x) - mlp._quick_gelu(x)), x) for x in xs]
    worst, at = max(diffs)
    rows = [(f"{x:g}", _fx(mlp._gelu(x), 5), _fx(mlp._quick_gelu(x), 5), _fx(mlp._quick_gelu(x) - mlp._gelu(x), 5))
            for x in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)]
    return ("CLIP-L's configuration says `quick_gelu`, x * sigmoid(1.702 x), and OpenCLIP-bigG's says `gelu_erf`, 0.5 x (1 + erf(x / "
            "sqrt 2)). Values at nine inputs (the library `erf`), and the largest absolute difference over x = -6, -5.999, ..., 6.\n\n"
            + _table(["x", "gelu (erf)", "quick gelu", "quick minus erf"], rows)
            + f"\n\nThe largest difference is {worst:.5f}, at x = {at:g}: a systematic deviation of about two hundredths in a single "
              "activation, applied by every hidden unit of every layer, which is why a tower must run the form its weights were trained with.")


# ------------------------------------------------------------------------------------------------ perceptual distance

PN = 12                        # side of the toy picture
BASE_LOW, BASE_HIGH = 0.2, 0.8


def base_picture():
    """Top half: a vertical grating (even columns 0.8, odd 0.2); bottom half: a horizontal grating (even rows 0.8, odd 0.2)."""
    return [[(BASE_HIGH if c % 2 == 0 else BASE_LOW) if r < PN // 2 else (BASE_HIGH if r % 2 == 0 else BASE_LOW) for c in range(PN)]
            for r in range(PN)]


def shift_right(img, k):
    return [[row[(c - k) % PN] for c in range(PN)] for row in img]


def _blur1(v):
    n = len(v)
    return [(v[(i - 1) % n] + 2 * v[i] + v[(i + 1) % n]) / 4 for i in range(n)]


def blur_once(img):
    """Separable [1, 2, 1] / 4 along the rows then the columns, wrapping at the borders."""
    t = [_blur1(r) for r in img]
    cols = [_blur1([t[r][c] for r in range(PN)]) for c in range(PN)]
    return [[cols[c][r] for c in range(PN)] for r in range(PN)]


def scale_contrast(img, f):
    return [[0.5 + f * (v - 0.5) for v in r] for r in img]


NOISE_SD_LIST = (0.05, 0.1, 0.2, 0.3)


def noise_field():
    """144 values `normal()` from bookrng.generator("clip.perceptual.noise"), row by row."""
    g = bookrng.generator("clip.perceptual.noise")
    return [[g.normal() for _ in range(PN)] for _ in range(PN)]


def add_noise(img, sd, z):
    return [[v + sd * zz for v, zz in zip(r, zr)] for r, zr in zip(img, z)]


def pixel_mse(a, b):
    return sum((a[r][c] - b[r][c]) ** 2 for r in range(PN) for c in range(PN)) / (PN * PN)


def _conv_valid(img, k):
    n, m = len(img), len(img[0])
    return [[sum(k[i][j] * img[r + i][c + j] for i in range(3) for j in range(3)) for c in range(m - 2)] for r in range(n - 2)]


MEAN_KERNEL = [[1 / 9] * 3 for _ in range(3)]
SECOND_X = [[0, 0, 0], [-1, 2, -1], [0, 0, 0]]
SECOND_Y = [[0, -1, 0], [0, 2, 0], [0, -1, 0]]
CHANNELS = ("local mean", "horizontal texture", "vertical texture")


def feature_maps(img):
    """Tap 1: three channels on the 10 by 10 valid positions: the 3 by 3 mean, and the absolute values of the second differences along the
    row, (-1, 2, -1) across the window's middle row, and along the column."""
    return [_conv_valid(img, MEAN_KERNEL), [[abs(v) for v in r] for r in _conv_valid(img, SECOND_X)],
            [[abs(v) for v in r] for r in _conv_valid(img, SECOND_Y)]]


def pool2(maps):
    """Tap 2: 2 by 2 average pooling with stride 2 of each channel (10 by 10 becomes 5 by 5)."""
    out = []
    for m in maps:
        out.append([[(m[r][c] + m[r][c + 1] + m[r + 1][c] + m[r + 1][c + 1]) / 4 for c in range(0, len(m[0]) - 1, 2)]
                    for r in range(0, len(m) - 1, 2)])
    return out


LP_EPS = 1e-10


def tap_distance(fa, fb, w=(1.0, 1.0, 1.0)):
    """One tap of the construction: at each position the channel vector is divided by its length plus 1e-10, the squared difference of
    the two unit vectors is weighted by w channel by channel and summed over channels, and the result is averaged over positions."""
    n, m = len(fa[0]), len(fa[0][0])
    total = 0.0
    for r in range(n):
        for c in range(m):
            va = [f[r][c] for f in fa]
            vb = [f[r][c] for f in fb]
            na, nb = _norm(va) + LP_EPS, _norm(vb) + LP_EPS
            total += sum(wc * (x / na - y / nb) ** 2 for wc, x, y in zip(w, va, vb))
    return total / (n * m)


def toy_distance(a, b, w=(1.0, 1.0, 1.0)):
    fa, fb = feature_maps(a), feature_maps(b)
    return [tap_distance(fa, fb, w), tap_distance(pool2(fa), pool2(fb), w)]


def distortions():
    """(series, label, picture) for the distortions of the chapter, in table order."""
    b = base_picture()
    z = noise_field()
    out = [("shift", "shifted right by 1", shift_right(b, 1))]
    out += [("blur", f"blurred {k} time" + ("s" if k > 1 else ""), _repeat(blur_once, b, k)) for k in (1, 2, 3)]
    out += [("noise", f"noise, sd {sd:g}", add_noise(b, sd, z)) for sd in NOISE_SD_LIST]
    out += [("contrast", f"contrast times {f:g}", scale_contrast(b, f)) for f in (0.75, 0.5, 0.25)]
    return out


def _repeat(f, x, k):
    for _ in range(k):
        x = f(x)
    return x


def perceptual_data():
    b = base_picture()
    return [(series, label, pixel_mse(b, img), *toy_distance(b, img)) for series, label, img in distortions()]


def toy_picture_text():
    b = base_picture()
    rows = ["".join("#" if v == BASE_HIGH else "." for v in r) for r in b]
    return ("The reference picture, 12 by 12, `#` = 0.8 and `.` = 0.2: the top six rows are a vertical grating (even columns bright), "
            "the bottom six a horizontal grating (even rows bright).\n\n```text\n" + "\n".join(rows) + "\n```\n")


def pixel_vs_feature():
    rows = []
    for series, label, mse_v, d1, d2 in perceptual_data():
        rows.append((label, _fx(mse_v, 4), _fx(d1, 4), _fx(d2, 4), _fx(d1 + d2, 4)))
    data = {r[1]: r for r in perceptual_data()}
    s1, b1 = data["shifted right by 1"], data["blurred 1 time"]
    return ("Each distortion of the reference picture, scored by the pixel mean squared error (mean of the 144 squared differences) and "
            "by the feature distance of equation (4), with weights 1 on each channel, tap by tap and summed. Noise is "
            "sd * z for one field z of 144 `normal()` values drawn row by row from `bookrng.generator(\"clip.perceptual.noise\")` "
            f"({_seed('clip.perceptual.noise')}), added without clipping. A shift wraps around the border (a shift by 2 returns the reference, a shift by 3 equals a shift by 1); a blur is the separable "
            "kernel [1, 2, 1] / 4 applied to rows then columns, wrapping; a contrast factor f maps v to 0.5 + f (v - 0.5).\n\n"
            + _table(["distortion", "pixel MSE", "feature distance, tap 1", "tap 2", "sum"], rows)
            + f"\n\nBy pixel error the picture shifted by 1 ({s1[2]:.4f}) is further from the original than the picture blurred once "
              f"({b1[2]:.4f}); by feature distance the order reverses ({s1[3] + s1[4]:.4f} against {b1[3] + b1[4]:.4f}).")


def one_position():
    """The construction at one position, step by step: tap 1, position (row 0, column 0), reference against the blurred-once picture."""
    b = base_picture()
    c = blur_once(b)
    fa, fb = feature_maps(b), feature_maps(c)
    va = [f[0][0] for f in fa]
    vb = [f[0][0] for f in fb]
    na, nb = _norm(va) + LP_EPS, _norm(vb) + LP_EPS
    ua = [x / na for x in va]
    ub = [x / nb for x in vb]
    sq = [(x - y) ** 2 for x, y in zip(ua, ub)]
    rows = [(CHANNELS[i], _fx(va[i], 4), _fx(vb[i], 4), _fx(ua[i], 4), _fx(ub[i], 4), _fx(sq[i], 4)) for i in range(3)]
    rows.append(("length / sum", _fx(_norm(va), 4), _fx(_norm(vb), 4), "", "", _fx(sum(sq), 4)))
    return ("Tap 1 at the top-left valid position (the 3 by 3 window of rows 0 to 2, columns 0 to 2), the reference against the picture "
            "blurred once: the three channel values, the same divided by the vector's length (plus 1e-10), and the squared difference of the "
            "unit vectors.\n\n"
            + _table(["channel", "reference", "blurred", "reference, unit", "blurred, unit", "squared difference"], rows))


# AlexNet trunk of LPIPS, from the layer table of `crates/lpips/src/config.rs`: (cin, cout, kernel, stride, pad, max-pool in front)
TRUNK = ((3, 64, 11, 4, 2, False), (64, 192, 5, 1, 2, True), (192, 384, 3, 1, 1, True), (384, 256, 3, 1, 1, False), (256, 256, 3, 1, 1, False))


def trunk_sizes(side):
    s, out = side, []
    for cin, cout, k, stride, pad, pooled in TRUNK:
        if pooled:
            s = (s - 3) // 2 + 1
        s = (s + 2 * pad - k) // stride + 1
        out.append(s)
    return out


def trunk_cost():
    side = 224
    sizes = trunk_sizes(side)
    rows = []
    tot_macs = tot_head = tot_par = 0
    for (cin, cout, k, stride, pad, pooled), s in zip(TRUNK, sizes):
        macs = s * s * cout * cin * k * k
        head = s * s * cout            # the 1 by 1 convolution to a single channel
        par = cout * (cin * k * k + 1)
        tot_macs += macs
        tot_head += head
        tot_par += par
        rows.append((f"{cin} -> {cout}, kernel {k}, stride {stride}" + (", max-pool in front" if pooled else ""), f"{s} x {s}", _int(macs), _int(head), _int(par)))
    rows.append(("total, one picture", "", _int(tot_macs), _int(tot_head), _int(tot_par)))
    heads = sum(c[1] for c in TRUNK)
    sq = sum(s * s * c[1] for s, c in zip(sizes, TRUNK))
    return ("The trunk of the LPIPS metric in brain (layer table of `crates/lpips/src/config.rs`) on a 224 by 224 picture: the five taps, "
            "the multiply-adds of each convolution (output positions * output channels * input channels * kernel^2), of its 1 by 1 "
            f"head (positions * channels), and its parameters ({heads} head weights besides, one per tap channel). The pooling in "
            "front of taps 2 and 3 is a 3 by 3 maximum with stride 2 and no padding.\n\n"
            + _table(["convolution", "tap grid", "multiply-adds", "head multiply-adds", "parameters"], rows)
            + f"\n\nA distance runs the trunk on both pictures: {_int(2 * tot_macs)} multiply-adds ({_int(4 * tot_macs)} FLOPs) for the "
              f"trunk, and {_int(tot_head)} multiply-adds for the heads (one picture's worth: the squared differences of the two "
              f"pictures are one tensor). The normalization, the squaring and the averaging are a few operations per tap value "
              f"({_int(sq)} tap values) and are not counted.")


# ------------------------------------------------------------------------------------------------ the pairwise sigmoid loss

SIGMOID_BIAS = -5.0


def sigmoid_loss_cells(cs, a, b):
    """The B * B terms softplus(-y_ij (a cs_ij + b)) with y_ij = +1 on the diagonal and -1 elsewhere, and their mean over the rows (the sum
    divided by B)."""
    n = len(cs)
    cells = [[_softplus(-(1.0 if i == j else -1.0) * (a * cs[i][j] + b)) for j in range(n)] for i in range(n)]
    return cells, sum(sum(r) for r in cells) / n


def sigmoid_batch():
    cs, _, _ = similarity_matrix()
    a, b = HAND_SCALE, SIGMOID_BIAS
    cells, total = sigmoid_loss_cells(cs, a, b)
    probs = [[_sigmoid(a * cs[i][j] + b) for j in range(4)] for i in range(4)]
    out = [f"The batch of the previous table, logit scale a = {a:g} and bias b = {b:g}. First sigma(a cs_ij + b), the probability the "
           "pair (image i, caption j) is a match:\n"]
    out.append(_mat_rows(probs, 4))
    out.append("\nThen the cell losses softplus(-y_ij (a cs_ij + b)) = -ln sigma(y_ij (a cs_ij + b)), y_ij = +1 on the diagonal and -1 "
               "elsewhere:\n")
    out.append(_mat_rows(cells, 4))
    diag = sum(cells[i][i] for i in range(4))
    off = total * 4 - diag
    out.append(f"\nThe sum of the 16 cells is {total * 4:.4f}: the 4 positive cells contribute {diag:.4f} and the 12 negative cells "
               f"{off:.4f}. Divided by B = 4, the loss is {total:.4f}.")
    # locality: raise S_01 by 1 and see which cell and row terms change
    cs2 = [list(r) for r in cs]
    cs2[0][1] += 1.0 / a
    cells2, _ = sigmoid_loss_cells(cs2, a, b)
    sm1 = _lse([a * v for v in cs[0]]) - a * cs[0][0]
    sm2 = _lse([a * v for v in cs2[0]]) - a * cs2[0][0]
    out.append(f"\nLocality. Raise the logit S_01 by 1 (cs_01 by {1 / a:g}). The softmax loss of row 0 moves from {sm1:.6f} to {sm2:.6f}, "
               f"since S_01 is in its normalizer. In the sigmoid loss only the cell (0, 1) moves, from {cells[0][1]:.6f} to "
               f"{cells2[0][1]:.6f}; the cell (0, 0) stays at {cells[0][0]:.6f} and so does every other cell.")
    return "\n".join(out)


def sigmoid_scale():
    """Entries of the logit matrix a device holds at once, for the global batch of CLIP (32,768 pairs), by number of devices."""
    n = 32768
    rows = []
    for dev in (8, 64, 256):
        local = n // dev
        full = local * n
        chunk = local * local
        rows.append((dev, _int(local), _int(full), _int(chunk), _int(4 * full), _int(4 * chunk), f"{full // chunk}"))
    return (f"A batch of B = {_int(n)} pairs split over a number of devices, each holding B / devices pictures and their captions. A softmax "
            "loss normalizes each picture's row over all B captions, so a device holds (or streams with a running log-sum-exp) its rows "
            "against every caption. A sigmoid loss has no normalizer: a block of (B / devices) by (B / devices) logits is turned into its "
            "cell losses and discarded before the next block arrives.\n\n"
            + _table(["devices", "pictures per device", "logits against all B captions", "logits in one block", "bytes (32-bit), all B",
                      "bytes, one block", "ratio"], rows))


# ------------------------------------------------------------------------------------------------ what the towers cost

def _block_flops(t, d, heads, ffn, gated):
    """The rule of Volume V, Chapter 9 for one block over t tokens: 8 t d^2 for the four projections, the feed-forward network (2 matrices
    of d by ffn: 4 t d ffn; the gated network has 3: 6 t d ffn), 4 t^2 d for the score and average products and 5 heads t^2 for the softmax."""
    mlp = (6 if gated else 4) * t * d * ffn
    return 8 * t * d * d + mlp + 4 * t * t * d + 5 * heads * t * t


def _block_params(d, ffn, gated):
    """Weights and biases of one block: qkv and output projection, two LayerNorms, and the feed-forward network (the EVA block adds its two
    inner LayerNorms, one of width d and one of width ffn, and a third matrix)."""
    attn = 3 * d * d + 3 * d + d * d + d
    mlp = 2 * d * ffn + ffn + d
    norms = 4 * d
    if gated:
        mlp = 3 * d * ffn + 3 * ffn + d   # w1, w2 (biased), w3 (biased)
        norms += 2 * d + 2 * ffn          # inner_ln and ffn_ln
    return attn + mlp + norms


def tower_cost():
    """Parameters (from the tensor manifests) and FLOPs of one forward pass of the towers of OpenAI's CLIP ViT-L/14 pair and of the other
    two configurations of `crates/clip`."""
    rows = []
    # name, tokens, width, heads, ffn, layers, gated, parameter count
    text_l = 49408 * 768 + 77 * 768 + 12 * _block_params(768, 3072, False) + 2 * 768
    text_g = 49408 * 1280 + 77 * 1280 + 32 * _block_params(1280, 5120, False) + 2 * 1280 + 1280 * 1280
    img_l = 1024 + 3 * 14 * 14 * 1024 + 257 * 1024 + 2 * 1024 + 24 * _block_params(1024, 4096, False)
    eva = 1024 + 577 * 1024 + 3 * 14 * 14 * 1024 + 1024 + 24 * _block_params(1024, 2730, True) + 2 * 1024 + 768 * 1024 + 768
    spec = [("CLIP-L text tower (12 x 768)", 77, 768, 12, 3072, 12, False, text_l),
            ("OpenCLIP-bigG text tower (32 x 1280, with projection)", 77, 1280, 20, 5120, 32, False, text_g),
            ("CLIP-L/14 image tower at 224 (24 x 1024)", 257, 1024, 16, 4096, 24, False, img_l),
            ("EVA02-CLIP-L/14 image tower at 336 (24 x 1024, gated)", 577, 1024, 16, 2730, 24, True, eva)]
    flops = {}
    for name, t, d, h, f, n, gated, par in spec:
        fl = n * _block_flops(t, d, h, f, gated)
        flops[name] = fl
        rows.append((name, _int(t), _int(par), f"{fl / 1e9:,.2f}"))
    ratio = flops[spec[2][0]] / flops[spec[0][0]]
    return ("Per forward pass of one text (or one picture): tokens, parameters read off the tensor manifests of `crates/clip/src/config.rs`, "
            "and FLOPs by the rule of "
            "Volume V, Chapter 9 (the four projections 8 T D^2, the feed-forward network 4 T D F or, gated, 6 T D F, the score and average "
            "products 4 T^2 D, the softmax 5 H T^2) summed over the blocks; the patch embedding, norms and activations are under one "
            "percent and left out. Parameters, with D the width and F the feed-forward width: a block has the query, key, value and output "
            "projections with biases (4 D^2 + 4 D), two LayerNorms (4 D) and a feed-forward network of two matrices with biases "
            "(2 D F + F + D); the EVA block has three such matrices (3 D F + 3 F + D) and two more LayerNorms, one of width D and one of "
            "width F (2 D + 2 F). The CLIP-L text tower is the 49,408-row token table, 77 position rows, 12 blocks (F = 3,072) and a final "
            "LayerNorm, without the text projection. The bigG text tower is the same with 1,280-wide rows, 32 blocks (F = 5,120) and the "
            "1,280 by 1,280 projection (no bias), 20 heads. The CLIP-L/14 picture tower is the class token, a 14 by 14 patch convolution "
            "from 3 channels without bias, 257 position rows, the LayerNorm before the blocks and 24 blocks (F = 4,096), without the "
            "final LayerNorm and the projection. The EVA02 tower is the class token, the patch convolution with a bias, 577 position "
            "rows, 24 gated blocks (F = 2,730), the final LayerNorm and the projection to 768 with a bias.\n\n"
            + _table(["tower", "tokens", "parameters", "GFLOPs per pass"], rows)
            + f"\n\nOne 224 by 224 picture through the CLIP-L/14 image tower costs {ratio:.1f} times one caption through the CLIP-L text "
              "tower. A zero-shot classifier pays the caption side once per class, then the picture side once per picture.")
