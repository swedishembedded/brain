# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Vision Transformers": patches, positions on a grid, windows, pyramids and a small ViT.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: vision_transformers.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.
Earlier modules are imported, never copied: the 8 by 8 toy images and the convolutional network of `cnns`, the strided convolution of
`convolution`, the cubic kernel of `images`, the sinusoids and rotations of `positions`, and the normalization, feed-forward network
and attention of `transformer_block` and `self_attention`.

* an 8 by 8 picture cut into 2 by 2 patches, and the patch embedding as a flatten and a matrix product against a strided convolution;
* the number of tokens and the cost of attention at stated resolutions and patch sizes, and the cost of windows;
* position information on a grid: two-axis sinusoids, two-axis rotation, and the resampling of a learned table to a new grid;
* shifted windows: the mask derived on a small grid, and its equality to ragged windows;
* patch merging, a hierarchical pyramid (Swin-T) and the stages of brain's hybrid encoder, with the token budget of a language model;
* a small vision transformer trained on the toy images of the convolutions chapter, against that chapter's network.

Arithmetic is 64-bit floating point. Random inputs come from `bookrng.generator(name)` with the names, seeds and draw orders printed
beside each table. Sums of products run left to right; the digits of a trained number are printed only when they survive the nudges of
`stable.py`.

Swedish Embedded AB implements solutions for image encoders that are counted before they are built. If your team needs expertise in
vision transformers and their cost, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import cnns
import convolution as cv
import images
import positions
import self_attention as sa
import stable as stab
import transformer_block as tb

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(v):
    return f"{v:,}"


def _fx(v, d=3):
    return f"{v:.{d}f}"


def _giga(v, d=3):
    """v / 10^9 with thousands separators: a count of FLOPs or multiply-adds in billions."""
    return f"{v / 1e9:,.{d}f}"


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"seed 0x{s:016x} ({s})"


def _round1(x):
    """x rounded to one decimal, ties away from zero (the pseudocode's `round(x, 1)`)."""
    r = math.floor(abs(x) * 10 + 0.5) / 10
    return -r if x < 0 and r > 0 else r


def _sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _below(x, limit):
    """'below 1e-12' when x is round-off noise under the limit: noise is not reproducible digit by digit."""
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else _sci(x)


def _grid(rows, width=4):
    return "\n".join(" ".join(f"{v:{width}}" for v in r) for r in rows)


def _pool_map(fn, args):
    with get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(fn, args)


# ------------------------------------------------------------------------------------------------ the opening: pixels as tokens

ATT_D, ATT_H, ATT_F = 768, 12, 3072   # width, heads and feed-forward width of the "base" block used for every cost in the chapter


def _score_pairs_flops(pairs, width=ATT_D):
    """FLOPs of the score and weighted-average products when a query reads `pairs` (query, key) pairs: 2 * width for each product."""
    return 4 * pairs * width


def pixel_tokens():
    side = 224
    t = side * side
    pairs = t * t
    pt = (side // 16) ** 2
    rows = [("one token per pixel", _int(t), _int(pairs), _int(pairs * 4), _int(_score_pairs_flops(pairs))),
            ("patches of 16 by 16", _int(pt), _int(pt * pt), _int(pt * pt * 4), _int(_score_pairs_flops(pt * pt)))]
    return (f"A 224 by 224 picture, the width D = {ATT_D} of the base block. Entries of one head's score matrix are T^2 for T tokens, "
            f"stored as 32-bit numbers (4 bytes); the score and average products of one block cost 4 T^2 D FLOPs (Volume V, Chapter 9) "
            f"for all heads together.\n\n"
            + _table(["tokens", "T", "score entries per head", "bytes per head", "FLOPs of the products, one block"], rows)
            + f"\n\nThe ratio of the two token counts is {t // pt}, and of the pair counts {pairs // (pt * pt):,}.")


# ------------------------------------------------------------------------------------------------ the patches of a picture

IMG = 8
PATCH = 2
GRID = IMG // PATCH


def patch_codes():
    """The 8 by 8 picture of the chapter: pixel (i, j) has the code 8 i + j, so every pixel is identifiable."""
    return [[IMG * i + j for j in range(IMG)] for i in range(IMG)]


def flatten_patch(img, a, b, p=PATCH):
    """The patch in row a, column b of a [C][H][W] image as a vector in the order (channel, row, column)."""
    return [img[c][p * a + i][p * b + j] for c in range(len(img)) for i in range(p) for j in range(p)]


def patches():
    img = [patch_codes()]
    rows = []
    for a in range(GRID):
        for b in range(GRID):
            t = a * GRID + b
            rows.append((t, f"({a}, {b})", f"{2 * a}-{2 * a + 1}", f"{2 * b}-{2 * b + 1}", *flatten_patch(img, a, b)))
    pic = _grid(img[0], 3)
    return ("The picture of one channel whose pixel in row i, column j has the code 8 i + j:\n\n"
            f"```text\n{pic}\n```\n\n"
            "Cut into 2 by 2 patches, in row-major order of the patches. A patch is flattened row by row, so its four numbers are "
            "the pixels (2a, 2b), (2a, 2b + 1), (2a + 1, 2b), (2a + 1, 2b + 1).\n\n"
            + _table(["token t = 4a + b", "patch (a, b)", "pixel rows", "pixel columns", "x0", "x1", "x2", "x3"], rows, "lllrrrrr"))


def patch_figure_data():
    img = patch_codes()
    return {"image": img, "patch": PATCH, "grid": GRID, "chosen": (1, 3), "vector": flatten_patch([img], 1, 3)}


# ------------------------------------------------------------------------------------------------ patch embedding as a convolution

EMB_NAME = "vision_transformers.patch_embed"
EMB_C, EMB_D = 3, 4


@lru_cache(maxsize=None)
def _embed_inputs():
    g = bookrng.generator(EMB_NAME)
    x = [[[_round1(g.normal()) for _ in range(IMG)] for _ in range(IMG)] for _ in range(EMB_C)]
    w = [[_round1(0.5 * g.normal()) for _ in range(EMB_C * PATCH * PATCH)] for _ in range(EMB_D)]
    b = [_round1(0.5 * g.normal()) for _ in range(EMB_D)]
    return x, w, b


def embed_by_matrix(x, w, b, p=PATCH, flat_order="channel"):
    """Tokens [n_patches][D]: each patch flattened (channel-major by default, or pixel-major: row, column, channel) times W^T + b."""
    n = len(x[0]) // p
    out = []
    for a in range(n):
        for c in range(len(x[0][0]) // p):
            if flat_order == "channel":
                v = flatten_patch(x, a, c, p)
            else:
                v = [x[ch][p * a + i][p * c + j] for i in range(p) for j in range(p) for ch in range(len(x))]
            out.append([_dot(row, v) + bo for row, bo in zip(w, b)])
    return out


def embed_by_conv(x, w, b, p=PATCH):
    """The same tokens as a convolution with kernel p, stride p and no padding: weights w[o] reshaped to [C][p][p]."""
    c = len(x)
    wk = [[[[row[(ch * p + i) * p + j] for j in range(p)] for i in range(p)] for ch in range(c)] for row in w]
    y = cv.corr2d(x, wk, stride=p)
    return [[y[o][a][bq] + b[o] for o in range(len(w))] for a in range(len(y[0])) for bq in range(len(y[0][0]))]


def _maxdiff(u, v):
    return max(abs(p - q) for ru, rv in zip(u, v) for p, q in zip(ru, rv))


def embedding_equivalence():
    x, w, b = _embed_inputs()
    m = embed_by_matrix(x, w, b)
    c = embed_by_conv(x, w, b)
    wrong = embed_by_matrix(x, w, b, flat_order="pixel")
    d = _maxdiff(m, c)
    wd = _maxdiff(m, wrong)
    wrows = [(f"o = {o}", *[f"{v:.1f}" for v in w[o]], f"{b[o]:.1f}") for o in range(EMB_D)]
    trows = [(t, *[_fx(v, 4) for v in c[t]]) for t in range(len(c))]
    return (f"{_seed(EMB_NAME)}. Draws, in this order: the 3 * 8 * 8 = 192 pixels x[c][i][j] (channel, row, column), each `round(normal(), 1)`; "
            "then the matrix W of 4 rows (output o) and 12 columns, row by row, each `round(0.5 * normal(), 1)`, whose column "
            "index is (c * 2 + i) * 2 + j for the pixel (i, j) of channel c in the patch; then the bias b, 4 numbers, each "
            "`round(0.5 * normal(), 1)`.\n\n"
            + _table(["W", *[f"col {k}" for k in range(12)], "b"], wrows, "l" + "r" * 13)
            + "\n\nThe 16 tokens (patch t = 4a + b is the patch in row a, column b), computed as a matrix product on the "
            "flattened patch, each sum left to right over the 12 numbers, then plus the bias:\n\n"
            + _table(["t", "token[0]", "token[1]", "token[2]", "token[3]"], trows)
            + f"\n\nThe same 64 numbers as a convolution of the [3, 8, 8] picture with the 4 filters W[o] reshaped to [3, 2, 2], stride 2, no "
            f"padding (`corr2d` of Chapter 2), plus the bias: the largest absolute difference is {_below(d, 1e-12)}. "
            f"With the pixels flattened in the other order (row, column, channel) and the same W the largest absolute "
            f"difference from the tokens above is {_fx(wd, 3)}: both orders are matrix products, and only one of them is this "
            f"convolution.")


def merging_equivalence():
    g = bookrng.generator("vision_transformers.patch_merge")
    c, d_out, h = 4, 8, 4
    x = [[[_round1(g.normal()) for _ in range(h)] for _ in range(h)] for _ in range(c)]
    w = [[_round1(0.5 * g.normal()) for _ in range(c * 4)] for _ in range(d_out)]
    zero = [0.0] * d_out
    by_matrix = embed_by_matrix(x, w, zero, p=2)
    by_conv = embed_by_conv(x, w, zero, p=2)
    return (f"{_seed('vision_transformers.patch_merge')}. A feature map of 4 channels on a 4 by 4 grid: the 64 values x[c][i][j] "
            "(channel, row, column), each `round(normal(), 1)`, then a matrix of 8 rows and 16 columns, row by row, each "
            "`round(0.5 * normal(), 1)`, no bias. Merging concatenates the 4 channels of each 2 by 2 group of positions in "
            "the order (channel, row, column) into 16 numbers and multiplies by the matrix.\n\n"
            f"Output: 4 tokens of 8 numbers, the map [8, 2, 2]. The same operation as a convolution with 8 filters of shape "
            f"[4, 2, 2], stride 2: the largest absolute difference over the 32 numbers is {_below(_maxdiff(by_matrix, by_conv), 1e-12)}.")


# ------------------------------------------------------------------------------------------------ tokens and attention cost

COST_ROWS = ((224, 16), (384, 16), (512, 16), (1024, 16), (224, 8), (224, 4))


def _block_total(t):
    parts = tb.block_flops(ATT_D, ATT_H, "gelu", ATT_F, t)
    return sum(parts.values()), parts


def token_costs():
    rows = []
    for side, p in COST_ROWS:
        t = (side // p) ** 2
        total, parts = _block_total(t)
        att = parts["scores"] + parts["softmax"]
        rows.append((side, p, f"{side // p} x {side // p}", _int(t), _fx(t * t * 4 * ATT_H / 2 ** 20, 1), _giga(parts["projections"] + parts["ffn"]),
                     _giga(att), _fx(100.0 * att / total, 1)))
    return (f"One block of width D = {ATT_D}, H = {ATT_H} heads, a GELU feed-forward network of hidden width {ATT_F} (the base size of "
            "the original ViT). Every FLOP count is the rule of Volume V, Chapter 9: 8 T D^2 for the four projections, 2 * 2 T D F "
            "for the feed-forward network, 4 T^2 D for the score and average products and 5 H T^2 for the softmax; the other "
            "terms of that chapter's count (two norms 2 T (4 D + 3), two adds 2 T D and the GELU 5 T F, in all T (10 D + 6 + 5 F) "
            "FLOPs) are under 1 percent, are left out of the first two FLOP columns and are counted in the block total that the "
            "attention share divides by. FLOPs are in billions. The memory is the 32-bit score matrices of all H heads, in MiB, if they are "
            "materialized. A class token would add one to T.\n\n"
            + _table(["side (px)", "patch (px)", "grid", "tokens T", "scores (MiB)", "projections and feed-forward (GFLOPs)",
                      "attention: products and softmax (GFLOPs)", "attention share of the block (%)"], rows))


# ------------------------------------------------------------------------------------------------ windows

def axis_groups(n, m, shift):
    """Cut points of one axis of length n under a cyclic shift of `shift` (< m): [0, shift), then windows of m, then the rest.
    Returns (start, length) pairs."""
    out, p = [], 0
    if shift > 0:
        out.append((0, min(shift, n)))
        p = min(shift, n)
    while p < n:
        out.append((p, min(m, n - p)))
        p += m if p + m <= n else n - p
    return out


def windows_cost():
    rows = []
    for g, m in ((8, 4), (14, 7), (28, 7), (56, 7)):
        t = g * g
        glob = t * t
        win = (g // m) ** 2 * (m * m) ** 2
        rows.append((f"{g} x {g}", f"{m} x {m}", _int(t), _int(glob), _int(win), _fx(glob / win, 1), _int(t * m * m)))
    return ("Pairs of (query, key) per head for global attention (T^2) and for non-overlapping windows of m by m (the grid divides evenly: "
            "T / m^2 windows of m^2 tokens, each with m^4 pairs, so T m^2 in all; the last column recomputes it as T m^2).\n\n"
            + _table(["grid", "window", "tokens T", "global pairs", "window pairs", "global / window", "T m^2"], rows))


# SAM-style tower of brain (`crates/sam1/src/config.rs`, `SamViTConfig::deepseek_ocr`): 12 blocks of width 768, a 64 by 64 grid of
# 16-pixel patches, windows of 14 by 14 with the grid zero-padded to a multiple of 14, blocks 2, 5, 8 and 11 global.
SAM = {"width": 768, "layers": 12, "grid": 64, "window": 14, "global": (2, 5, 8, 11)}


def sam_windows():
    g, m = SAM["grid"], SAM["window"]
    padded = -(-g // m) * m
    nwin = (padded // m) ** 2
    t = g * g
    glob_pairs = t * t
    win_pairs_padded = nwin * (m * m) ** 2
    n_glob = len(SAM["global"])
    n_win = SAM["layers"] - n_glob
    w = SAM["width"]
    all_global = SAM["layers"] * _score_pairs_flops(glob_pairs, w)
    mixed = n_glob * _score_pairs_flops(glob_pairs, w) + n_win * _score_pairs_flops(win_pairs_padded, w)
    glob_total = n_glob * _score_pairs_flops(glob_pairs, w)
    rows = [("tokens of the grid", f"{g} x {g}", _int(t)),
            ("grid padded to a multiple of the window", f"{padded} x {padded}", _int(padded * padded)),
            ("windows (each 14 x 14 = 196 tokens)", f"{padded // m} x {padded // m}", _int(nwin)),
            ("pairs of a global block", "", _int(glob_pairs)),
            ("pairs of a windowed block, padded windows", "", _int(win_pairs_padded)),
            (f"product FLOPs, all {SAM['layers']} blocks global", "", _int(all_global)),
            (f"product FLOPs, {n_win} windowed and {n_glob} global blocks", "", _int(mixed)),
            ("ratio", "", _fx(all_global / mixed, 2)),
            (f"share of the {n_glob} global blocks in the mixed total (%)", "", _fx(100.0 * glob_total / mixed, 1))]
    return ("The tower of `SamViTConfig::deepseek_ocr`: width 768, 12 blocks, a 64 by 64 grid, windows of 14 by 14 on the grid padded "
            "with zeros to a multiple of 14, blocks 2, 5, 8 and 11 global. Products FLOPs are 4 * pairs * 768 per block.\n\n"
            + _table(["quantity", "shape", "value"], rows, "llr"))


# ------------------------------------------------------------------------------------------------ position information on a grid


def sinusoid_2d(r, c, d=8, base=positions.BASE):
    """Two-axis sinusoidal encoding of width d: the sinusoid of the row in the first d/2 channels, of the column in the last d/2."""
    return positions.sinusoid(r, d // 2, base) + positions.sinusoid(c, d // 2, base)


def grid_offsets():
    n, d = 4, 8
    ref = sinusoid_2d(0, 0, d)
    one_d = [positions.sinusoid(k, d) for k in range(n * n)]
    rows2, rows1 = [], []
    for r in range(n):
        rows2.append([_fx(_dot(ref, sinusoid_2d(r, c, d)), 3) for c in range(n)])
        rows1.append([_fx(_dot(one_d[0], one_d[r * n + c]), 3) for c in range(n)])
    head = ["", *[f"col {c}" for c in range(n)]]
    t2 = _table(head, [(f"row {r}", *rows2[r]) for r in range(n)])
    t1 = _table(head, [(f"row {r}", *rows1[r]) for r in range(n)])
    return ("Width D = 8, base 10000 (`positions.sinusoid`, Volume V, Chapter 8). The dot product of the encoding of the patch in row 0, "
            "column 0 with the encoding of every patch of a 4 by 4 grid.\n\n"
            "Two axes: the first four channels are the sinusoid of the row (width 4), the last four of the column.\n\n" + t2
            + "\n\nOne axis: the patches numbered 4 r + c in row-major order and encoded as positions 0 .. 15 of a width-8 sinusoid.\n\n" + t1)


def rope_2d(x, hpos, wpos, theta=10000.0):
    """Two-axis rotary embedding of a vector of width d (d divisible by 4) in the half-split pairing: pair m is (x[m], x[m + d/2]);
    the first d/4 pairs turn by hpos * rate_j and the last d/4 by wpos * rate_j, rate_j = theta^(-4 j / d), j = 0 .. d/4 - 1."""
    d = len(x)
    ws = positions.omegas(d // 2, theta)
    out = list(x)
    for m in range(d // 2):
        pos = hpos if m < d // 4 else wpos
        w = ws[m % (d // 4)]
        cth, sth = math.cos(pos * w), math.sin(pos * w)
        a, b = m, m + d // 2
        out[a] = x[a] * cth - x[b] * sth
        out[b] = x[a] * sth + x[b] * cth
    return out


ROPE_NAME = "vision_transformers.rope2d"


def rope_2d_scores():
    g = bookrng.generator(ROPE_NAME)
    q = [_round1(g.normal()) for _ in range(8)]
    k = [_round1(g.normal()) for _ in range(8)]
    cases = [((0, 0), (1, 2)), ((3, 1), (4, 3)), ((5, 7), (6, 9)), ((1, 2), (0, 0)), ((0, 0), (2, 1)), ((0, 0), (0, 3)), ((0, 0), (3, 0))]
    rows = []
    for (hq, wq), (hk, wk) in cases:
        s = _dot(rope_2d(q, hq, wq), rope_2d(k, hk, wk))
        rows.append((f"({hq}, {wq})", f"({hk}, {wk})", f"({hk - hq}, {wk - wq})", _fx(s, 6)))
    ws = positions.omegas(4)
    return (f"{_seed(ROPE_NAME)}. Draws: the 8 entries of q, then the 8 of k, each `round(normal(), 1)`; "
            f"q = {[f'{v:.1f}' for v in q]}, k = {[f'{v:.1f}' for v in k]}. Head width 8, base 10000, half-split pairs "
            f"(m, m + 4); pairs 0 and 1 turn by the row times the rates {ws[0]:g} and {ws[1]:g}, pairs 2 and 3 by the column times "
            "the same two rates. The score is the dot product of the rotated query at (row, column) with the rotated key.\n\n"
            + _table(["query at", "key at", "offset (rows, columns)", "score"], rows, "lllr"))


# ---- resampling a learned table

POS_NAME = "vision_transformers.pos_table"


def pos_table():
    g = bookrng.generator(POS_NAME)
    return [[_round1(g.normal()) for _ in range(4)] for _ in range(4)]


def taps_keys(n_in, n_out, a=-0.75):
    """Non-antialiased bicubic: for each output o the source coordinate (o + 0.5) n_in / n_out - 0.5 (not clamped), the four taps
    floor(s) - 1 .. floor(s) + 2 with the weights of cubic convolution(a), indices clamped to the edge (no renormalization)."""
    out = []
    for o in range(n_out):
        s = (o + 0.5) * n_in / n_out - 0.5
        f = math.floor(s)
        t = s - f
        ws = [images.cubic(t + 1, a), images.cubic(t, a), images.cubic(1 - t, a), images.cubic(2 - t, a)]
        out.append((s, [(min(max(f + k, 0), n_in - 1), w) for k, w in zip((-1, 0, 1, 2), ws)], [f + k for k in (-1, 0, 1, 2)], ws))
    return out


def taps_aa(n_in, n_out, a=-0.5):
    """Antialiased bicubic (Pillow, and torch's antialias=True): window of 2 * max(scale, 1) around (o + 0.5) * scale, scale =
    n_in / n_out, taps outside the image dropped, the weights renormalized to sum to 1."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 2.0 * fs
    out = []
    for o in range(n_out):
        centre = (o + 0.5) * scale
        lo = max(int(centre - support + 0.5), 0)
        n = min(int(centre + support + 0.5), n_in) - lo
        w = [images.cubic((j + lo - centre + 0.5) / fs, a) for j in range(n)]
        tot = _sum(w)
        out.append([(lo + j, v / tot) for j, v in enumerate(w)])
    return out


def taps_corner_linear(n_in, n_out):
    """Bilinear with the corners aligned: source coordinate o (n_in - 1) / (n_out - 1), taps floor and floor + 1 (clamped)."""
    out = []
    for o in range(n_out):
        s = 0.0 if n_out == 1 else o * (n_in - 1) / (n_out - 1)
        f = math.floor(s)
        fr = s - f
        out.append([(f, 1 - fr), (min(f + 1, n_in - 1), fr)])
    return out


def _apply_axis(rows_taps, v):
    return [_sum([w * v[i] for i, w in taps]) for taps in rows_taps]


def resample_grid(tab, taps_fn, n_out):
    """Separable resampling of a square table: along the columns of every row, then along the rows."""
    n_in = len(tab)
    taps = taps_fn(n_in, n_out)
    rows = [_apply_axis(taps, r) for r in tab]
    cols = [_apply_axis(taps, [rows[i][j] for i in range(n_in)]) for j in range(n_out)]
    return [[cols[j][i] for j in range(n_out)] for i in range(n_out)]


def _clamped(taps_fn):
    def f(n_in, n_out):
        return [t[1] for t in taps_fn(n_in, n_out)]
    return f


def position_resample():
    tab = pos_table()
    keys = taps_keys(4, 6)
    big = resample_grid(tab, _clamped(taps_keys), 6)
    same = resample_grid(tab, _clamped(taps_keys), 4)
    sam = resample_grid(tab, taps_aa, 6)
    lin = resample_grid(tab, taps_corner_linear, 6)
    d_sam = _maxdiff(big, sam)
    d_lin = _maxdiff(big, lin)
    trows = [(o, _fx(s, 4), f"{idx[0]} .. {idx[3]}", *[_fx(w, 4) for w in ws], _fx(_sum(ws), 4)) for o, (s, _, idx, ws) in enumerate(keys)]
    brow = [(f"row {i}", *[_fx(v, 3) for v in r]) for i, r in enumerate(big)]
    trow = [(f"row {i}", *[f"{v:.1f}" for v in r]) for i, r in enumerate(tab)]
    return (f"{_seed(POS_NAME)}. A learned table for a 4 by 4 grid with one channel: 16 values, row by row, each `round(normal(), 1)`.\n\n"
            + _table(["table", "col 0", "col 1", "col 2", "col 3"], trow)
            + "\n\nResampled to 6 by 6 as the CLIP tower does (cubic convolution with a = -0.75, source coordinate "
            "(o + 0.5) * 4 / 6 - 0.5, the four taps floor(s) - 1 .. floor(s) + 2 with indices clamped to the table, no "
            "renormalization), along each row first and then down each column. The taps of one axis:\n\n"
            + _table(["output o", "source s", "taps", "w(-1)", "w(0)", "w(+1)", "w(+2)", "sum"], trows, "rrlrrrrr")
            + "\n\nThe resampled table:\n\n"
            + _table(["", "col 0", "col 1", "col 2", "col 3", "col 4", "col 5"], brow)
            + f"\n\nResampled to its own size the table is reproduced exactly: the largest difference is {_below(_maxdiff(same, tab), 1e-12)}. "
            f"The same table by the antialiased resampler of the SAM tower (a = -0.5, window clipped at the border and renormalized) "
            f"differs from the table above by at most {_fx(d_sam, 3)}, and by bilinear interpolation with the corners aligned "
            f"(source coordinate o * 3 / 5, the convention of the Qwen3-VL tower) by at most {_fx(d_lin, 3)}.")


# ------------------------------------------------------------------------------------------------ shifted windows

SW_N, SW_M, SW_S = 4, 2, 1
SW_NAME = "vision_transformers.shifted_window"


def _axis_region(idx, n, m, s):
    """Swin's region of index idx along an axis of length n, in the rolled map: [0, n - m), [n - m, n - s), [n - s, n)."""
    return 0 if idx < n - m else (1 if idx < n - s else 2)


def shifted_windows_data(n=SW_N, m=SW_M, s=SW_S):
    """Everything about shifted windows on an n by n grid of tokens numbered row-major: the rolled map of token ids, the region
    label of each rolled position, and for each window (row-major) its token ids and the allowed-pair mask."""
    ids = [[n * i + j for j in range(n)] for i in range(n)]
    rolled = [[ids[(i + s) % n][(j + s) % n] for j in range(n)] for i in range(n)]
    label = [[3 * _axis_region(i, n, m, s) + _axis_region(j, n, m, s) for j in range(n)] for i in range(n)]
    wins = []
    for wa in range(n // m):
        for wb in range(n // m):
            pos = [(wa * m + i, wb * m + j) for i in range(m) for j in range(m)]
            tok = [rolled[i][j] for i, j in pos]
            lab = [label[i][j] for i, j in pos]
            mask = [[lab[u] == lab[v] for v in range(m * m)] for u in range(m * m)]
            wins.append({"window": (wa, wb), "positions": pos, "tokens": tok, "labels": lab, "mask": mask})
    return {"n": n, "m": m, "s": s, "rolled": rolled, "label": label, "windows": wins}


def ragged_groups(n=SW_N, m=SW_M, s=SW_S):
    """Groups of token ids of the ragged partition: the product of the cut intervals of the two axes."""
    cuts = axis_groups(n, m, s)
    groups = []
    for r0, rl in cuts:
        for c0, cl in cuts:
            groups.append([n * (r0 + i) + c0 + j for i in range(rl) for j in range(cl)])
    return groups


def _attend(qs, ks, vs, allowed):
    """Single-head attention of one query set over one key set with an allowed-pair mask: softmax(q.k / sqrt(d)) over the allowed."""
    d = len(qs[0])
    out = []
    for i, q in enumerate(qs):
        sc = [(_dot(q, k) / math.sqrt(d)) if allowed[i][j] else -math.inf for j, k in enumerate(ks)]
        mx = max(sc)
        e = [math.exp(v - mx) for v in sc]
        z = _sum(e)
        out.append([_sum([e[j] / z * vs[j][c] for j in range(len(ks))]) for c in range(len(vs[0]))])
    return out


def _qkv():
    g = bookrng.generator(SW_NAME)
    n = SW_N * SW_N
    q = [[_round1(g.normal()) for _ in range(4)] for _ in range(n)]
    k = [[_round1(g.normal()) for _ in range(4)] for _ in range(n)]
    v = [[_round1(g.normal()) for _ in range(4)] for _ in range(n)]
    return q, k, v


def shifted_window_mask():
    d = shifted_windows_data()
    n, m = d["n"], d["m"]
    lab = _grid(d["label"], 2)
    rolled = _grid(d["rolled"], 2)
    wrows = []
    for w in d["windows"]:
        allowed = sum(1 for r in w["mask"] for v in r if v)
        wrows.append((str(w["window"]), " ".join(str(t) for t in w["tokens"]), " ".join(str(l) for l in w["labels"]), allowed))
    w_last = d["windows"][-1]
    mask = "\n".join(" ".join("1" if v else "0" for v in r) for r in w_last["mask"])
    groups = ragged_groups()
    q, k, v = _qkv()
    # (a) roll, window, mask, and put the outputs back
    out_a = [None] * (n * n)
    for w in d["windows"]:
        toks = w["tokens"]
        res = _attend([q[t] for t in toks], [k[t] for t in toks], [v[t] for t in toks], w["mask"])
        for t, r in zip(toks, res):
            out_a[t] = r
    # (b) ragged windows, no mask
    out_b = [None] * (n * n)
    for grp in groups:
        full = [[True] * len(grp) for _ in grp]
        res = _attend([q[t] for t in grp], [k[t] for t in grp], [v[t] for t in grp], full)
        for t, r in zip(grp, res):
            out_b[t] = r
    # unshifted windows and global attention, for contrast
    plain = shifted_windows_data(s=0)
    out_c = [None] * (n * n)
    for w in plain["windows"]:
        toks = w["tokens"]
        res = _attend([q[t] for t in toks], [k[t] for t in toks], [v[t] for t in toks], [[True] * len(toks) for _ in toks])
        for t, r in zip(toks, res):
            out_c[t] = r
    out_g = _attend(q, k, v, [[True] * (n * n) for _ in range(n * n)])
    sizes = sorted((len(g) for g in groups), reverse=True)
    return (f"A 4 by 4 grid of tokens numbered 0 .. 15 row by row, windows of m = {m} by {m}, shift s = {d['s']}. Swin rolls the map by -s "
            "along both axes (the token at row i, column j of the rolled map is the token at row (i + s) mod 4, column (j + s) mod 4 "
            "of the original), cuts it into windows, and labels each position of the rolled map by the region it came from: along "
            "each axis the index i is in region 0 if i < n - m, 1 if n - m <= i < n - s and 2 otherwise, and the label is "
            "3 * (row region) + (column region).\n\n"
            f"Token ids of the rolled map:\n\n```text\n{rolled}\n```\n\nRegion labels of the rolled map:\n\n```text\n{lab}\n```\n\n"
            + _table(["window (row, column)", "token ids", "region labels", "allowed pairs of 16"], wrows, "llll")
            + f"\n\nThe mask of the last window (entry [u][v] is 1 when position u may read position v, i.e. their labels are equal):\n\n"
            f"```text\n{mask}\n```\n\n"
            f"The tokens of one region are exactly the groups of the ragged partition that cuts each axis at [0, s), then every m, "
            f"i.e. at rows and columns {{0}}, {{1, 2}}, {{3}}: its {len(groups)} groups have sizes {sizes}. Single-head attention "
            f"(width 4, scale 1/2) on random q, k, v ({_seed(SW_NAME)}; draws: the 64 entries of q, then of k, then of v, token by "
            f"token, each `round(normal(), 1)`), computed once by the rolled windows with the mask and once by the ragged groups "
            f"with no mask: the largest absolute difference of the 16 outputs is {_below(_maxdiff(out_a, out_b), 1e-12)}. "
            f"Against the unshifted 2 by 2 windows the outputs differ by at most {_fx(_maxdiff(out_a, out_c), 3)}, and against "
            f"global attention by at most {_fx(_maxdiff(out_a, out_g), 3)}.")


def window_pair_data():
    """(label, pairs) of one head on an 8 by 8 grid of tokens, windows of 4 by 4, shift 2 where shifted."""
    n, m, s = 8, 4, 2
    cuts = axis_groups(n, m, s)
    ragged = (sum(l * l for _, l in cuts)) ** 2
    t = n * n
    regular = (n // m) ** 2 * (m * m) ** 2
    return {"global": t * t, "regular": regular, "rolled": regular, "ragged": ragged, "cuts": [l for _, l in cuts]}


def window_pair_counts():
    d = window_pair_data()
    rows = [("global attention", _int(d["global"])), ("regular windows of 4 by 4", _int(d["regular"])),
            ("rolled windows with a mask (every query scans its whole 4 by 4 window)", _int(d["rolled"])),
            (f"ragged windows (sizes {', '.join(str(l) for l in d['cuts'])} on each axis)", _int(d["ragged"]))]
    return ("Pairs of (query, key) of one head on an 8 by 8 grid of tokens, windows of 4 by 4, shift 2 where shifted.\n\n"
            + _table(["variant", "pairs"], rows))


def windows_figure_data():
    n, m, s = 8, 4, 2
    cuts = axis_groups(n, m, s)
    d = shifted_windows_data(n, m, s)
    return {"n": n, "m": m, "s": s, "cuts": cuts, "last": d["windows"][-1], "label": d["label"]}


# ------------------------------------------------------------------------------------------------ hierarchical pyramids

SWIN_T = {"patch": 4, "dims": (96, 192, 384, 768), "depths": (2, 2, 6, 2), "window": 7}


def swin_stages():
    side = 224
    rows = []
    tot_g = tot_w = 0
    for i, (c, dep) in enumerate(zip(SWIN_T["dims"], SWIN_T["depths"])):
        g = side // (SWIN_T["patch"] * 2 ** i)
        m = min(SWIN_T["window"], g)
        t = g * g
        glob = _score_pairs_flops(t * t, c)
        win = _score_pairs_flops((g // m) ** 2 * (m * m) ** 2, c)
        tot_g += dep * glob
        tot_w += dep * win
        rows.append((i + 1, f"{SWIN_T['patch'] * 2 ** i}", f"{g} x {g}", _int(t), c, dep, _giga(glob), _giga(win), _fx(glob / win, 1)))
    return (f"Swin-T on a 224 by 224 picture (Liu and colleagues, 2021): patches of 4 pixels, widths 96, 192, 384, 768, blocks 2, 2, 6, 2, windows of "
            "7 by 7 (the last stage's grid is one window). Patch merging between stages takes the grid and the width to a half and "
            "twice. FLOPs are the score and average products, 4 * pairs * width, of one block of the stage.\n\n"
            + _table(["stage", "stride", "grid", "tokens", "width", "blocks", "global (GFLOPs per block)", "windowed (GFLOPs per block)",
                      "ratio"], rows)
            + f"\n\nSummed over the blocks of all four stages: {_giga(tot_g)} billion FLOPs if every block were global, {_giga(tot_w)} billion windowed, ratio "
            f"{tot_g / tot_w:.2f}.")


# ------------------------------------------------------------------------------------------------ brain's hybrid encoder (FastViTHD)

FV = {"layers": (2, 12, 24, 4, 2), "dims": (96, 192, 384, 768, 1536), "mlp": 4, "head_dim": 32, "attn": (3, 4), "cls": 2}


def _fv_grid(side):
    g = side // 4
    out = []
    for i in range(5):
        out.append(g)
        g //= 2
    return out


def repmixer_macs(c, hw):
    """Multiply-adds of one RepMixer block: the fused 3x3 depthwise mixer, then the ConvFFN (7x7 depthwise, 1x1 to 4c, 1x1 back)."""
    return hw * c * (9 + 49 + 2 * FV["mlp"] * c)


def attention_block_macs(c, hw):
    """One attention block: qkv (3 c^2 hw), output projection (c^2 hw), scores and average (2 hw^2 c) and the ConvFFN."""
    return hw * 4 * c * c + 2 * hw * hw * c + hw * c * (49 + 2 * FV["mlp"] * c)


def fastvithd_counts(side=1024):
    grids = _fv_grid(side)
    dims, layers = FV["dims"], FV["layers"]
    out = {"stem": 0, "stages": [], "down": [], "cpe": [], "exp": 0}
    g0 = side // 2
    out["stem"] = g0 * g0 * dims[0] * 3 * 9 + (side // 4) ** 2 * dims[0] * 9 + (side // 4) ** 2 * dims[0] * dims[0]
    for i in range(5):
        hw = grids[i] ** 2
        c = dims[i]
        if i in FV["attn"]:
            out["cpe"].append(hw * c * 49)
            out["stages"].append(layers[i] * attention_block_macs(c, hw))
        else:
            out["cpe"].append(0)
            out["stages"].append(layers[i] * repmixer_macs(c, hw))
        if i < 4:
            hwo = grids[i + 1] ** 2
            co = dims[i + 1]
            out["down"].append(hwo * co * 49 + hwo * co * co)
    last = grids[4] ** 2
    cexp = dims[4] * FV["cls"]
    se_hidden = cexp // 16
    out["exp"] = last * cexp * 9 + 2 * cexp * se_hidden
    out["total"] = out["stem"] + sum(out["stages"]) + sum(out["down"]) + sum(out["cpe"]) + out["exp"]
    return out


def fastvithd_stages():
    side = 1024
    grids = _fv_grid(side)
    cnt = fastvithd_counts(side)
    rows = []
    for i in range(5):
        hw = grids[i] ** 2
        c = FV["dims"][i]
        is_attn = i in FV["attn"]
        mixer = 2 * hw * hw * c if is_attn else hw * c * 9
        would = 2 * hw * hw * c
        rows.append((i, f"{grids[i]} x {grids[i]}", _int(hw), c, FV["layers"][i],
                     "attention" if is_attn else "RepMixer", _giga(mixer), _giga(would), _giga(cnt["stages"][i])))
    other = cnt["stem"] + sum(cnt["down"]) + sum(cnt["cpe"]) + cnt["exp"]
    return (f"The encoder of `FastVitHdConfig::fastvithd` on a 1024 by 1024 picture: a stem of three convolutions to stride 4, then five "
            "stages at strides 4, 8, 16, 32 and 64 with a stride-2 downsample between them, 44 blocks in all. Multiply-adds "
            "(MACs) are counted from the layer shapes, weights only (no norms, biases or activations). The final expansion `conv_exp` is "
            "a 3 by 3 convolution with groups = 1,536 that doubles the 1,536 channels to 3,072 (9 weights per output channel) and "
            "its squeeze-and-excitation gate of 3,072 -> 192 -> 3,072 on the pooled vector (hidden width 3,072/16); each stage's "
            "RepMixer block is a depthwise 3 by 3 (9 c hw) and a ConvFFN of a depthwise 7 by 7 (49 c hw) and two 1 by 1 "
            "layers c -> 4c -> c (8 c^2 hw), an attention block adds 4 c^2 hw and 2 hw^2 c to the same ConvFFN, and a "
            "downsample is a depthwise 7 by 7 and a 1 by 1 at the output grid; the position convolutions are depthwise 7 by 7 "
            "before the two attention stages. \"Token mixer\" is "
            "what the block spends on letting tokens see each other: the fused depthwise 3 by 3 convolution (9 c hw) in stages 0 to 2 and "
            "the score and average products (2 hw^2 c) in stages 3 and 4; the next column is what global attention would "
            "cost there.\n\n"
            + _table(["stage", "grid", "tokens hw", "width c", "blocks", "token mixer", "mixer (GMACs per block)",
                      "global attention (GMACs per block)", "GMACs of the stage's blocks"], rows, "rrrrrlrrr")
            + f"\n\nIn billions of multiply-adds: stem {_giga(cnt['stem'])}, the four downsamples {_giga(sum(cnt['down']))}, the two "
            f"position convolutions {_giga(sum(cnt['cpe']))}, the final expansion {_giga(cnt['exp'])}; blocks {_giga(sum(cnt['stages']))}. Total "
            f"{_giga(cnt['total'])} billion multiply-adds, {_giga(2 * cnt['total'])} billion FLOPs ({other * 100 / cnt['total']:.1f} percent outside the blocks). "
            f"Exactly, in MACs: one attention block of stage 0 would need {_int(2 * (_fv_grid(side)[0] ** 2) ** 2 * FV['dims'][0])} for global attention, the "
            f"depthwise convolution that is there {_int(9 * FV['dims'][0] * _fv_grid(side)[0] ** 2)}.")


DECODER = {"layers": 24, "hidden": 896, "heads": 14, "kv": 2, "ff": 4864, "vocab": 151936}


def decoder_macs_per_token():
    h = DECODER["hidden"]
    hd = h // DECODER["heads"]
    kvw = DECODER["kv"] * hd
    attn = h * h + 2 * h * kvw + h * h
    mlp = 3 * h * DECODER["ff"]
    return DECODER["layers"] * (attn + mlp)


def token_budget():
    rows = []
    per = decoder_macs_per_token()
    for side in (256, 512, 1024, 1536):
        plain = (side // 16) ** 2
        merged = (side // 32) ** 2
        hyb = (side // 64) ** 2
        cnt = fastvithd_counts(side)
        vit_b = 12 * _block_total(plain)[0]
        rows.append((side, f"{_int(plain)} / {_int(merged)} / {_int(hyb)}", _giga(vit_b), _giga(2 * cnt["total"]),
                     _giga(2 * per * plain), _giga(2 * per * hyb)))
    return ("Tokens a language model reads for one square picture: a plain ViT with 16-pixel patches (T = (s/16)^2), a 16-pixel ViT "
            "whose patches are merged 2 by 2 before the language model (the Qwen3-VL arrangement, (s/32)^2), and the 64-fold reduction "
            "of the hybrid encoder ((s/64)^2). Encoder FLOPs in billions: twelve blocks of the base size (D = 768) at the plain "
            "token count, by the block count of the earlier table (patch embedding left out, but the norms, adds and GELU of the block "
            "counted, T (10 D + 6 + 5 F) FLOPs), against the hybrid encoder's "
            "stage table (2 FLOPs per multiply-add); these are different models, counted alike. The decoder of FastVLM-0.5B has 24 layers "
            "of width 896, 14 query and 2 key-value heads of width 64 and a feed-forward width of 4864; its weight multiply-adds per "
            f"token, summed over the layers (attention 896 * 896 + 2 * 896 * 128 + 896 * 896, feed-forward 3 * 896 * 4864), are {_int(per)}. "
            "The last two columns are twice that times the number of image tokens, in GFLOPs (the attention scores between tokens are "
            "not included).\n\n"
            + _table(["side (px)", "tokens: plain / merged / hybrid", "plain ViT-B/16 encoder", "hybrid encoder",
                      "decoder, plain tokens", "decoder, hybrid tokens"], rows))


def pyramid_series():
    """Tokens and attention FLOPs against the side, for the five designs of the figures."""
    sides = (448, 896, 1344, 1792)
    tok, flops = {}, {}
    names = ("plain ViT, global", "plain ViT, windows 14 x 14", "plain ViT, windows with 4 global blocks", "Swin-T, windows 7 x 7",
             "hybrid encoder (attention in two stages)")
    for n in names:
        tok[n], flops[n] = [], []
    for s in sides:
        g = s // 16
        t = g * g
        tok[names[0]].append(t)
        tok[names[1]].append(t)
        tok[names[2]].append(t)
        glob = _score_pairs_flops(t * t, 768)
        win = _score_pairs_flops(t * 196, 768)
        flops[names[0]].append(12 * glob)
        flops[names[1]].append(12 * win)
        flops[names[2]].append(4 * glob + 8 * win)
        sw = 0
        for i, (c, dep) in enumerate(zip(SWIN_T["dims"], SWIN_T["depths"])):
            gg = s // (4 * 2 ** i)
            sw += dep * _score_pairs_flops(gg * gg * 49, c)
        flops[names[3]].append(sw)
        tok[names[3]].append((s // 4) ** 2)
        hy = 0
        for i in FV["attn"]:
            gg = s // (4 * 2 ** i)
            hy += FV["layers"][i] * 2 * gg ** 2 * gg ** 2 * FV["dims"][i]
        flops[names[4]].append(2 * hy)
        tok[names[4]].append((s // 64) ** 2)
    return sides, names, tok, flops


def pyramid_table():
    sides, names, tok, flops = pyramid_series()
    rows = []
    for n in names:
        rows.append((n, *[_giga(f, 1) for f in flops[n]]))
    return ("Attention FLOPs, in billions, of the score and average products over the whole network (4 * pairs * width per block, projections and "
            "feed-forward networks not counted) at four picture sides. Plain ViT: 12 blocks of width 768, patches of 16 pixels. Windows "
            "14 by 14: every block windowed, then the SAM arrangement of 8 windowed and 4 global blocks (the sides divide by 14 * 16). "
            "Swin-T as in its stage table, windows of 7 by 7. Hybrid encoder: the attention blocks of stages 3 and 4 at global "
            "attention, 4 and 2 blocks, widths 768 and 1536 (the sides divide by 64).\n\n"
            + _table(["design", *[f"side {s}" for s in sides]], rows))


def brain_towers():
    """Token counts of the image encoders of brain's crates, from the constants of their configurations."""
    def g(side, patch):
        return side // patch
    rows = []
    n = g(1024, 16)
    rows.append(("SAM ViT-B tower (`crates/sam1`)", "16", "1024", f"{n} x {n}", _int(n * n), "learned table (bicubic resample for another grid) and a relative bias",
                 "windows 14 x 14, blocks 2, 5, 8, 11 global", _int((n // 4) ** 2)))
    n = g(224, 14)
    rows.append(("CLIP-L/14 body (`crates/clip`)", "14", "224", f"{n} x {n}", _int(n * n + 1), "learned table, bicubic resample", "global", _int(n * n + 1)))
    n = g(336, 14)
    rows.append(("EVA02-CLIP-L/14 (`crates/clip`)", "14", "336", f"{n} x {n}", _int(n * n + 1), "two-axis rotary embedding", "global", _int(n * n + 1)))
    n = g(448, 16)
    rows.append(("Qwen3-VL ViT, a 448 px picture (`crates/qwen3vl`)", "16", "448", f"{n} x {n}", _int(n * n), "learned 48 x 48 table, bilinear resample, two-axis rotary embedding",
                 "global", _int(n * n // 4)))
    n = 1024 // 64
    rows.append(("FastViTHD (`crates/fastvlm`)", "4, then 2 four times", "1024", f"{n} x {n}", _int(n * n), "a depthwise 7 x 7 convolution before each attention stage",
                 "global, in the last two of five stages", _int(n * n)))
    return ("Image encoders of brain's crates: the size of the grid at the point where attention runs last, the tokens the encoder "
            "hands on, how a token learns where it is and where attention is local. The class token of CLIP is counted in its "
            "257 and 577. The last column is what reaches the next stage: the SAM tower's two stride-2 convolutions after its "
            "neck take 4096 tokens to 256, and the Qwen3-VL merger folds each 2 by 2 group of patches into one token.\n\n"
            + _table(["tower", "patch (px)", "input (px)", "grid", "tokens in the encoder", "positions", "attention", "tokens handed on"],
                     rows, "llrrrllr"))


# ------------------------------------------------------------------------------------------------ a small vision transformer

TOY_P, TOY_D, TOY_H, TOY_FF = 2, 8, 2, 16
TOY_T = (cnns.IMG // TOY_P) ** 2          # 16 tokens
TOY_STEPS, TOY_LR = 100, 0.02
TOY_RECORD = (0, 25, 50, 100)
TOY_RUNS = (0, 1, 2)
READOUTS = ("mean", "cls")


def _vit_name(run):
    return f"vision_transformers.vit_init.{run}"


def toy_patches(img):
    """The 16 patches of a flat 64-pixel image, each the 4 pixels (row, column) = (0,0), (0,1), (1,0), (1,1) of its 2 by 2 block."""
    return [[img[(TOY_P * a + i) * cnns.IMG + TOY_P * b + j] for i in range(TOY_P) for j in range(TOY_P)]
            for a in range(cnns.IMG // TOY_P) for b in range(cnns.IMG // TOY_P)]


def vit_init(run, nudge=0.0):
    """Parameters, drawn in this order from bookrng.generator(f"vision_transformers.vit_init.{run}"), `normal() * scale` each, row by
    row: We [4, 8] (scale 1/2), E [16, 8] (0.3), Wq, Wk, Wv, Wo [8, 8] (1/sqrt(8)), W1 [8, 16] (1/sqrt(8)), W2 [16, 8] (1/4), Wh [8, 3]
    (1/sqrt(8)), then the class token [1, 8] (0.3). Biases b_e, b_h are 0; the three gains are 1. `nudge` is added to We[0][0]."""
    g = bookrng.generator(_vit_name(run))

    def mat(r, c, s):
        return [[s * g.normal() for _ in range(c)] for _ in range(r)]
    p = {"We": mat(4, TOY_D, 0.5), "be": [[0.0] * TOY_D], "E": mat(TOY_T, TOY_D, 0.3)}
    for k in ("Wq", "Wk", "Wv", "Wo"):
        p[k] = mat(TOY_D, TOY_D, 1 / math.sqrt(TOY_D))
    p["W1"] = mat(TOY_D, TOY_FF, 1 / math.sqrt(TOY_D))
    p["W2"] = mat(TOY_FF, TOY_D, 0.25)
    p["g1"], p["g2"], p["gf"] = [[1.0] * TOY_D], [[1.0] * TOY_D], [[1.0] * TOY_D]
    p["Wh"] = mat(TOY_D, cnns.NCLS, 1 / math.sqrt(TOY_D))
    p["bh"] = [[0.0] * cnns.NCLS]
    p["cls"] = [[0.3 * g.normal() for _ in range(TOY_D)]]
    p["We"][0][0] += nudge
    return p


def vit_n_params(p, readout):
    return sum(len(r) for k, v in p.items() if not (k == "cls" and readout == "mean") for r in v)


def vit_forward(p, img, readout):
    x = tb.matmul(toy_patches(img), p["We"])
    x = [[v + b + e for v, b, e in zip(row, p["be"][0], er)] for row, er in zip(x, p["E"])]
    if readout == "cls":
        x = [list(p["cls"][0])] + x
    n1, cn1 = tb.rms_forward(x, p["g1"][0])
    a, ca = sa.mha_forward(n1, p["Wq"], p["Wk"], p["Wv"], p["Wo"], TOY_H, None)
    x1 = tb._add(x, a)
    n2, cn2 = tb.rms_forward(x1, p["g2"][0])
    f, cf = tb.gelu_mlp_forward(n2, p["W1"], p["W2"])
    x2 = tb._add(x1, f)
    nf, cnf = tb.rms_forward(x2, p["gf"][0])
    pooled = nf[0] if readout == "cls" else [_sum([r[j] for r in nf]) / len(nf) for j in range(TOY_D)]
    logits = [_dot([pooled[j] for j in range(TOY_D)], [p["Wh"][j][k] for j in range(TOY_D)]) + p["bh"][0][k] for k in range(cnns.NCLS)]
    return logits, (cn1, ca, cn2, cf, cnf, pooled, len(nf))


def _zeros(p):
    return {k: [[0.0] * len(r) for r in v] for k, v in p.items()}


def vit_loss_grad(p, data, readout, grad=True):
    """Mean cross-entropy, accuracy and (when asked) the gradient of the mean loss; images are visited in index order. The backward
    pass chains the backward functions of the previous chapters (`rms_backward`, `gelu_mlp_backward`, `mha_backward`)."""
    g = _zeros(p) if grad else None
    loss, correct = 0.0, 0
    for img, y in data:
        logits, cache = vit_forward(p, img, readout)
        pr = cnns.softmax(logits)
        loss += -math.log(pr[y])
        correct += pr.index(max(pr)) == y
        if not grad:
            continue
        cn1, ca, cn2, cf, cnf, pooled, n = cache
        dl = [pr[k] - (1.0 if k == y else 0.0) for k in range(cnns.NCLS)]
        for j in range(TOY_D):
            for k in range(cnns.NCLS):
                g["Wh"][j][k] += pooled[j] * dl[k]
        for k in range(cnns.NCLS):
            g["bh"][0][k] += dl[k]
        dpool = [_dot(p["Wh"][j], dl) for j in range(TOY_D)]
        dnf = [dpool] + [[0.0] * TOY_D for _ in range(n - 1)] if readout == "cls" else [[v / n for v in dpool] for _ in range(n)]
        dx2, dgf = tb.rms_backward(dnf, cnf)
        dn2_, gm = tb.gelu_mlp_backward(dx2, cf)
        dn2, dg2 = tb.rms_backward(dn2_, cn2)
        dx1 = tb._add(dx2, dn2)
        ga = sa.mha_backward(ca, dx1)
        dn1, dg1 = tb.rms_backward(ga["X"], cn1)
        dx = tb._add(dx1, dn1)
        for name, src in (("Wq", ga), ("Wk", ga), ("Wv", ga), ("Wo", ga), ("W1", gm), ("W2", gm)):
            for r, rr in zip(g[name], src[name]):
                for t in range(len(r)):
                    r[t] += rr[t]
        for t in range(TOY_D):
            g["g1"][0][t] += dg1[t]
            g["g2"][0][t] += dg2[t]
            g["gf"][0][t] += dgf[t]
        off = 1 if readout == "cls" else 0
        if readout == "cls":
            for t in range(TOY_D):
                g["cls"][0][t] += dx[0][t]
        pm = toy_patches(img)
        for tok in range(TOY_T):
            d = dx[tok + off]
            for t in range(TOY_D):
                g["E"][tok][t] += d[t]
                g["be"][0][t] += d[t]
                for i in range(4):
                    g["We"][i][t] += pm[tok][i] * d[t]
    if grad:
        for v in g.values():
            for r in v:
                for t in range(len(r)):
                    r[t] /= len(data)
    return loss / len(data), correct / len(data), g


def _vit_leaves(p):
    return [r for k in sorted(p) for r in p[k]]


def vit_adam(p, data, readout):
    """Full-batch Adam (beta_1 0.9, beta_2 0.999, epsilon 1e-8, bias correction, learning rate TOY_LR) for TOY_STEPS updates, from zero
    moments. Returns {t: (loss, accuracy)} after t updates for t in TOY_RECORD."""
    lv = _vit_leaves(p)
    m = [[0.0] * len(r) for r in lv]
    v = [[0.0] * len(r) for r in lv]
    seen = {}
    for t in range(TOY_STEPS + 1):
        loss, acc, g = vit_loss_grad(p, data, readout)
        if t in TOY_RECORD:
            seen[t] = (loss, acc)
        if t == TOY_STEPS:
            break
        for r, gr, mr, vr in zip(lv, _vit_leaves(g), m, v):
            for i in range(len(r)):
                mr[i] = 0.9 * mr[i] + 0.1 * gr[i]
                vr[i] = 0.999 * vr[i] + 0.001 * gr[i] * gr[i]
                r[i] -= TOY_LR * (mr[i] / (1 - 0.9 ** (t + 1))) / (math.sqrt(vr[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return seen


def _vit_task(args):
    readout, run, nudge = args
    p = vit_init(run, nudge)
    train = [(list(x), y) for x, y in cnns.toy_data()[:cnns.N_TRAIN]]
    test = [(list(x), y) for x, y in cnns.toy_data()[cnns.N_TRAIN:]]
    seen = vit_adam(p, train, readout)
    tl, ta, _ = vit_loss_grad(p, test, readout, grad=False)
    return {"seen": seen, "test": (tl, ta)}


@lru_cache(maxsize=None)
def vit_runs():
    jobs = [(r, run, nudge) for r in READOUTS for run in TOY_RUNS for nudge in (0.0, *stab.NUDGES)]
    return dict(zip(jobs, _pool_map(_vit_task, jobs)))


def _vit_twins(readout, run, f):
    base = vit_runs()[(readout, run, 0.0)]
    return f(base), [f(vit_runs()[(readout, run, n)]) for n in stab.NUDGES]


def _gc_one(readout):
    p = vit_init(0)
    train = [(list(x), y) for x, y in cnns.toy_data()[:cnns.N_TRAIN]][:6]
    _, _, g = vit_loss_grad(p, train, readout)
    worst, fails, total = 0.0, 0, 0
    for name in sorted(p):
        if name == "cls" and readout == "mean":
            continue
        for r, gr in zip(p[name], g[name]):
            for t in range(len(r)):
                old = r[t]
                r[t] = old + 1e-6
                lp = vit_loss_grad(p, train, readout, grad=False)[0]
                r[t] = old - 1e-6
                lm = vit_loss_grad(p, train, readout, grad=False)[0]
                r[t] = old
                num = (lp - lm) / 2e-6
                err = abs(gr[t] - num)
                total += 1
                fails += err > 1e-7 + 1e-5 * max(abs(gr[t]), abs(num))
                worst = max(worst, err)
    return worst, fails, total


def vit_gradient_check():
    res = _pool_map(_gc_one, list(READOUTS))
    rows = [(r, _int(t), "below 1e-7" if w < 1e-7 else _sci(w), t - f) for r, (w, f, t) in zip(READOUTS, res)]
    return ("The initialized transformer (run 0) on the training images n = 0 .. 5 (two of each class): the analytic gradient of the mean "
            "cross-entropy against the central difference (L(w + h) - L(w - h)) / (2h) with h = 1e-6 in 64-bit arithmetic, one "
            "weight at a time, with the gate |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|) of Volume III, Chapter 6.\n\n"
            + _table(["readout", "parameters checked", "largest absolute difference", "pass the gate"], rows)
            + "\n\nThe class-token run checks the class token as well, so it has 8 parameters more.")


def vit_setup():
    rows = []
    for r in READOUTS:
        p = vit_init(0)
        rows.append((r, _int(vit_n_params(p, r)), *[ _seed(_vit_name(run)).split(" (")[0] for run in (0,)]))
    t = TOY_T
    d, ff = TOY_D, TOY_FF
    macs = t * 4 * d + 4 * t * d * d + 2 * TOY_H * t * t * (d // TOY_H) + 2 * t * d * ff + d * cnns.NCLS
    cnn_macs = 5778
    return (f"The images and the split are those of Chapter 3 (`cnns.toy_data`: 150 images of 8 by 8 pixels, class n mod 3 "
            f"for stripes, checkerboards and blobs; the first 60 train, the other 90 test). The model: 2 by 2 patches (16 tokens of 4 "
            f"pixels, flattened row by row), a linear embedding to D = {d} with a bias, a learned position table of 16 rows added, one "
            f"pre-norm block of Volume V (RMS norm, {TOY_H} heads of width {d // TOY_H} with no mask (head h takes columns {d // TOY_H}h to {d // TOY_H}h + {d // TOY_H - 1} of the query, key and value projections, and the head outputs are concatenated in head order before the output projection), residual, RMS norm, a GELU "
            f"feed-forward network of hidden width {ff} without biases, residual), a final RMS norm, a readout, and a linear map "
            f"to the 3 classes. Readout \"mean\": the mean of the 16 token rows. Readout \"cls\": a learned class row placed before the "
            f"patches (no position row of its own), whose output row is read. The RMS norms use epsilon 1e-6 and the GELU is the exact "
            f"one (with `erf`); the initialization is given with the training table below. Multiply-adds per image (patch embedding 16 * 4 * {d}, "
            f"four projections 4 * 16 * {d}^2, the score and average products 2 * {TOY_H} * 16^2 * {d // TOY_H}, the feed-forward network "
            f"2 * 16 * {d} * {ff}, the head {d} * 3): {_int(macs)} for the mean readout, against {_int(cnn_macs)} for the convolutional "
            f"network.")


def vit_training():
    rows = []
    for r in READOUTS:
        for t in TOY_RECORD:
            l0, ws = _vit_twins(r, 0, lambda x, t=t: x["seen"][t][0])
            a0, wa = _vit_twins(r, 0, lambda x, t=t: x["seen"][t][1])
            rows.append((r, t, cnns_text(l0, ws), cnns_text(a0, wa)))
    return (f"Both readouts, run 0 (the generator of the name {_vit_name(0)}, {_seed(_vit_name(0))}; run r uses the name "
            f"vision_transformers.vit_init.r), trained on the 60 training images by full-batch Adam exactly as for the "
            f"network of Chapter 3 (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, learning rate {TOY_LR}, {TOY_STEPS} updates, mean "
            f"cross-entropy, sums over images in index order). Initialization: every entry is `normal() * scale` from that one "
            f"generator, matrix by matrix and row by row, in this order: We [4, 8] with scale 0.5, the position table E [16, 8] with "
            f"0.3, Wq, Wk, Wv and Wo [8, 8] with 1/sqrt(8) each, W1 [8, 16] with 1/sqrt(8), W2 [16, 8] with 1/4, the classifier Wh [8, 3] "
            f"with 1/sqrt(8), and last the class token [1, 8] with 0.3 (drawn, and unused, by the mean readout, so the two readouts "
            f"share every other starting weight). The patch bias and the classifier bias are 0 and the three norm gains are 1. The nudge "
            f"is added to We[0][0], the first weight of the patch embedding, which every image uses. {stab.RULE}\n\n"
            + _table(["readout", "updates t", "training loss", "training accuracy"], rows, "lrrr"))


def cnns_text(v, ws, d=4):
    return stab.text(v, ws, lambda x: f"{x:.{d}f}")


def vit_against_cnn():
    rows = []
    cnn = cnns.toy_run(0.0)
    cws = [cnns.toy_run(n) for n in stab.NUDGES]
    rows.append(("convolutional network (Chapter 3)", "-", _int(cnns.n_params(cnns.toy_init())), cnns_text(cnn["seen"][100][1], [w["seen"][100][1] for w in cws]),
                 cnns_text(cnn["test"][0], [w["test"][0] for w in cws]), cnns_text(cnn["test"][1], [w["test"][1] for w in cws])))
    for r in READOUTS:
        for run in TOY_RUNS:
            tr, wtr = _vit_twins(r, run, lambda x: x["seen"][TOY_STEPS][1])
            tl, wtl = _vit_twins(r, run, lambda x: x["test"][0])
            ta, wta = _vit_twins(r, run, lambda x: x["test"][1])
            rows.append((f"vision transformer, {r}", f"{run} ({_seed(_vit_name(run)).split(' (')[0].replace('seed ', '')})",
                         _int(vit_n_params(vit_init(run), r)), cnns_text(tr, wtr), cnns_text(tl, wtl), cnns_text(ta, wta)))
    return (f"After {TOY_STEPS} updates on the same 60 training images, scored on the same 90 test images (30 of each class), same optimizer "
            f"and rate. The vision transformer is trained from three initializations (runs 0, 1, 2, generators of the printed seeds), "
            f"each with both readouts; the convolutional network is the single run of Chapter 3. {stab.RULE}\n\n"
            + _table(["model", "run (seed)", "parameters", "training accuracy", "test cross-entropy", "test accuracy"], rows, "llrrrr"))
