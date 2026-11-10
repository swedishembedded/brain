# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Convolution".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: convolution.<name> -->` markers;
the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* a 5 by 5 image and a 3 by 3 kernel slid by hand, the same operation as a true convolution with the kernel turned over;
* the output size under kernel size, stride, padding and dilation, against a count of the windows that fit;
* a layer with several channels in and out on the 4 by 4 image of the previous chapter, and the parameter and operation counts of
  some real layer shapes;
* the receptive field of a stack, by the recursion and by tracing the inputs one output reads;
* im2col on the 5 by 5 example, the memory of the unrolled matrix and the panel width of a cache-sized unrolling;
* depthwise, pointwise and grouped counts, and a grouped layer checked against its definition;
* the backward pass by hand, and by three routes (scatter, gather, the matrix products of im2col) with a gradient check;
* the transposed convolution: output sizes, the adjoint of a strided convolution, the overlap counts behind the checkerboard,
  the weight layout that always binds and the output-padding band;
* the cost of a convolution through the Fourier transform against the direct sum, in multiplications and in weight bytes.

Arithmetic is the language's 64-bit floating point unless a table is made of integers. Every table that compares two routes prints
the criterion the comparison meets, not the rounding noise below it. Random inputs come from `bookrng.generator(name)` with the
names, seeds and draw orders printed beside each table.

Swedish Embedded AB implements solutions for convolution kernels that match a reference and run at the speed of the hardware. If
your team needs expertise in convolutional networks and their gradients, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import images as im

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _grid(rows, width=None):
    """A matrix of numbers as a fixed-width text block."""
    width = width or max(len(str(v)) for r in rows for v in r)
    return "\n".join(" ".join(f"{v:{width}}" for v in r) for r in rows)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _zeros(*shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros(*shape[1:]) for _ in range(shape[0])]


def _flat(t):
    out = []
    for v in t:
        out += _flat(v) if isinstance(v, list) else [v]
    return out


def _maxdiff(a, b):
    return max(abs(p - q) for p, q in zip(_flat(a), _flat(b)))


def _indices(t):
    """Every index tuple of a nested list, row-major."""
    if not isinstance(t[0], list):
        return [(i,) for i in range(len(t))]
    return [(i,) + rest for i in range(len(t)) for rest in _indices(t[i])]


def _at(t, idx):
    for i in idx:
        t = t[i]
    return t


def _central(t, idx, loss, step=1e-5):
    """(L(t + step e) - L(t - step e)) / (2 step) for the entry idx of the nested list t, which is restored afterwards."""
    parent = _at(t, idx[:-1]) if len(idx) > 1 else t
    old = parent[idx[-1]]
    parent[idx[-1]] = old + step
    lp = loss()
    parent[idx[-1]] = old - step
    lm = loss()
    parent[idx[-1]] = old
    return (lp - lm) / (2 * step)


def _fmt_int(v):
    return f"{v:,}"


# ------------------------------------------------------------------------------------------------ the core operations


def out_size(h, k, s=1, p=0, d=1):
    """Windows of extent d (k - 1) + 1 that fit in the padded length h + 2 p when they start every s positions."""
    return (h + 2 * p - d * (k - 1) - 1) // s + 1


def corr2d(x, w, stride=1, pad=0, dil=1, groups=1):
    """x[ci][hi][wi], w[co][cl][kh][kw] with cl running over the Cin / groups channels of the group of co; returns y[co][ho][wo].
    Each sum runs over cl, then kh, then kw, in that order; taps outside the image read zero."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout, k = len(w), len(w[0][0])
    cin_g, cout_g = cin // groups, cout // groups
    ho, wo = out_size(h, k, stride, pad, dil), out_size(wd, k, stride, pad, dil)
    y = _zeros(cout, ho, wo)
    for co in range(cout):
        g = co // cout_g
        for i in range(ho):
            for j in range(wo):
                acc = 0.0
                for cl in range(cin_g):
                    for kh in range(k):
                        hi = i * stride - pad + kh * dil
                        if not 0 <= hi < h:
                            continue
                        for kw in range(k):
                            wi = j * stride - pad + kw * dil
                            if 0 <= wi < wd:
                                acc += w[co][cl][kh][kw] * x[g * cin_g + cl][hi][wi]
                y[co][i][j] = acc
    return y


def corr1(x, k, **kw):
    """One channel in, one channel out."""
    return corr2d([x], [[k]], **kw)[0]


def flip(k):
    """The kernel turned half a turn: tap (m, n) goes to (K - 1 - m, K - 1 - n)."""
    return [row[::-1] for row in k[::-1]]


def conv_full(x, h):
    """The true convolution y[i][j] = sum over (m, n) of h[m][n] x[i - m][j - n], every output that touches the image:
    (H + K - 1) by (W + K - 1) of them."""
    ht, wd, k = len(x), len(x[0]), len(h)
    y = _zeros(ht + k - 1, wd + k - 1)
    for i in range(ht + k - 1):
        for j in range(wd + k - 1):
            acc = 0
            for m in range(k):
                for n in range(k):
                    if 0 <= i - m < ht and 0 <= j - n < wd:
                        acc += h[m][n] * x[i - m][j - n]
            y[i][j] = acc
    return y


def im2col(x, k, stride=1, pad=0, dil=1):
    """col[hw][(ci * k + kh) * k + kw] = the input value the window of output position hw reads at tap (ci, kh, kw), 0 outside."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    ho, wo = out_size(h, k, stride, pad, dil), out_size(wd, k, stride, pad, dil)
    col = []
    for i in range(ho):
        for j in range(wo):
            row = []
            for ci in range(cin):
                for kh in range(k):
                    for kw in range(k):
                        hi, wi = i * stride - pad + kh * dil, j * stride - pad + kw * dil
                        row.append(x[ci][hi][wi] if 0 <= hi < h and 0 <= wi < wd else 0)
            col.append(row)
    return col


def col2im(dcol, shape, k, stride=1, pad=0, dil=1):
    """The adjoint of im2col: every entry of dcol is added onto the input position it was read from, so a position read by
    several windows receives the sum."""
    cin, h, wd = shape
    ho, wo = out_size(h, k, stride, pad, dil), out_size(wd, k, stride, pad, dil)
    dx = _zeros(cin, h, wd)
    for i in range(ho):
        for j in range(wo):
            row = dcol[i * wo + j]
            for ci in range(cin):
                for kh in range(k):
                    for kw in range(k):
                        hi, wi = i * stride - pad + kh * dil, j * stride - pad + kw * dil
                        if 0 <= hi < h and 0 <= wi < wd:
                            dx[ci][hi][wi] += row[(ci * k + kh) * k + kw]
    return dx


def matmul(a, b):
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(r) for r in zip(*a)]


# ------------------------------------------------------------------------------------------------ the slide by hand

X5 = [[1, 1, 1, 8, 8],
      [1, 1, 2, 8, 9],
      [1, 2, 8, 9, 9],
      [2, 8, 8, 9, 9],
      [1, 8, 9, 9, 9]]
SOBEL = [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]


def sliding_data():
    """The padded image (zero border of 1) and the output of the 3 by 3 kernel at padding 1: what the sliding figure draws."""
    padded = [[0] * 7] + [[0] + r + [0] for r in X5] + [[0] * 7]
    return padded, [[int(v) for v in r] for r in corr1(X5, SOBEL, pad=1)]


def slide():
    y = corr1(X5, SOBEL)
    terms = [f"({SOBEL[m][n]})({X5[m][n]})" for m in range(3) for n in range(3)]
    y00 = sum(SOBEL[m][n] * X5[m][n] for m in range(3) for n in range(3))
    assert y00 == y[0][0]
    out = ["The 5 by 5 image X (rows top to bottom) and the 3 by 3 kernel K. The output at position (i, j) is the sum over (m, n) "
           "of K[m][n] X[i + m][j + n], with no padding and stride 1.", "", "```text",
           "X:", _grid(X5, 2), "", "K:", _grid(SOBEL, 2), "", "output Y (3 by 3):", _grid([[int(v) for v in r] for r in y], 3), "```", "",
           f"Position (0, 0), term by term (kernel tap times pixel, row by row): {' + '.join(terms)} = {int(y00)}."]
    return "\n".join(out)


def convolution_vs_correlation():
    corr = corr1(X5, SOBEL)
    true_full = conv_full(X5, SOBEL)
    valid = [[true_full[i + 2][j + 2] for j in range(3)] for i in range(3)]
    via_flip = corr1(X5, flip(SOBEL))
    assert _maxdiff(valid, via_flip) == 0
    assert all(valid[i][j] == -corr[i][j] for i in range(3) for j in range(3))   # K is odd under the half turn: flip(K) = -K
    padded = [[0] * 9 for _ in range(2)] + [[0, 0] + r + [0, 0] for r in X5] + [[0] * 9 for _ in range(2)]
    full_by_corr = corr1(padded, flip(SOBEL))
    assert _maxdiff(full_by_corr, true_full) == 0
    g = lambda m: _grid([[int(v) for v in r] for r in m], 3)
    out = ["The same X and K. Correlation is what the previous table computed (sum of K[m][n] X[i + m][j + n]). The true convolution "
           "of Volume I is sum of K[m][n] X[i - m][j - n] over every output that touches the image, 7 by 7 of them; its 3 by 3 block "
           "of outputs whose whole window lies inside the image is rows 2 to 4 and columns 2 to 4.", "", "```text",
           "correlation, valid part (3 by 3):", g(corr), "", "true convolution, valid part (rows 2-4, columns 2-4 of the 7 by 7):",
           g(valid), "", "true convolution, all 7 by 7 outputs:", g(true_full), "```", "",
           "Here K turned half a turn is -K (the kernel is odd under that turn), so the two valid blocks differ by a sign. In general "
           "the valid block of the true convolution with K equals the correlation with the turned kernel (largest difference over "
           "this example: 0), and the whole 7 by 7 convolution equals the correlation of X padded with a border of 2 zeros and the "
           "turned kernel (largest difference: 0)."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ output sizes

SIZE_CASES = [  # (H, k, s, p, d)
    (5, 3, 1, 0, 1), (5, 3, 1, 1, 1), (5, 3, 2, 1, 1), (6, 3, 2, 0, 1), (7, 3, 2, 0, 1), (7, 3, 1, 2, 2),
    (8, 3, 1, 4, 2), (5, 5, 1, 0, 1), (4, 3, 2, 1, 1), (224, 7, 2, 3, 1), (224, 3, 2, 1, 1), (225, 3, 2, 1, 1),
]


def _windows_that_fit(h, k, s, p, d):
    """Count by trying every start: window o covers padded positions o s + d m for m in 0 .. k - 1."""
    n = 0
    while n * s + d * (k - 1) <= h + 2 * p - 1:
        n += 1
    return n


def _unread(h, k, s, p, d):
    read = set()
    for o in range(_windows_that_fit(h, k, s, p, d)):
        for m in range(k):
            i = o * s - p + d * m
            if 0 <= i < h:
                read.add(i)
    return h - len(read)


def output_sizes():
    rows = []
    for h, k, s, p, d in SIZE_CASES:
        f = out_size(h, k, s, p, d)
        c = _windows_that_fit(h, k, s, p, d)
        assert f == c
        rows.append((h, k, s, p, d, d * (k - 1) + 1, f, c, _unread(h, k, s, p, d)))
    return ("Input length H, kernel size k, stride s, padding p, dilation d. The formula is floor((H + 2p - d(k-1) - 1) / s) + 1; "
            "the count column tries window starts 0, 1, 2, ... until the window no longer fits in the padded length H + 2p; the last "
            "column is the number of input positions no window reads.\n\n"
            + _table(["H", "k", "s", "p", "d", "extent d(k-1)+1", "formula", "count", "inputs never read"], rows))


# ------------------------------------------------------------------------------------------------ channels in and out

W2 = [  # w[co][ci][kh][kw] for 3 -> 2 channels, 2 by 2 taps
    [[[0, 0], [0, 0]], [[-1, 1], [-1, 1]], [[0, 0], [0, 0]]],
    [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
]


def channels():
    x = im.image_codes()
    y = corr2d(x, W2)
    out = ["The test image of the previous chapter (3 channels, 4 by 4, codes 20 + 13k, 10 + 9k and 5 + 15k for k = 4 y + x) through a "
           "layer with weight tensor of shape [2, 3, 2, 2] (C_out = 2, C_in = 3, 2 by 2 taps), stride 1, no padding, no bias, so the "
           "output has shape [2, 3, 3]. Output channel 0 has the green channel as its only non-zero input slice, with taps "
           "[[-1, 1], [-1, 1]] (a horizontal difference); output channel 1 has all twelve taps equal to 1 (the sum of the 2 by 2 "
           "window over all three channels).", "", "```text"]
    for co in range(2):
        out += [f"output channel {co}:", _grid([[int(v) for v in r] for r in y[co]], 4), ""]
    out[-1:] = ["```"]
    macs = 2 * 3 * 3 * (3 * 2 * 2)
    out += ["", f"Parameters: 2 * 3 * 2 * 2 = {2 * 3 * 2 * 2}. Multiply-adds: every one of the 2 * 3 * 3 = 18 outputs sums 3 * 2 * 2 = 12 "
            f"products, so {macs} multiply-adds."]
    return "\n".join(out)


LAYERS = [  # name, Cin, H, Cout, k, s, p
    ("stem 7x7 stride 2", 3, 224, 64, 7, 2, 3),
    ("3x3 at 56x56", 64, 56, 64, 3, 1, 1),
    ("3x3 at 14x14", 256, 14, 256, 3, 1, 1),
    ("1x1 at 56x56", 64, 56, 256, 1, 1, 0),
]


def layer_costs():
    rows = []
    for name, cin, h, cout, k, s, p in LAYERS:
        ho = out_size(h, k, s, p)
        params = cout * cin * k * k
        macs = ho * ho * cout * cin * k * k
        rows.append((name, f"[{cin}, {h}, {h}] to [{cout}, {ho}, {ho}]", _fmt_int(params), _fmt_int(macs), _fmt_int(2 * macs), _fmt_int(ho * ho)))
    cin, h, nout = 3, 224, 1000
    dense_params = cin * h * h * nout
    rows.append(("dense on the 3x224x224 image", f"[{cin * h * h}] to [{nout}]", _fmt_int(dense_params), _fmt_int(dense_params),
                 _fmt_int(2 * dense_params), 1))
    return ("Parameters (no bias), multiply-adds (MACs) and FLOPs (2 per multiply-add) of one image through a layer; the last column "
            "is how many times each weight is used, Ho * Wo.\n\n"
            + _table(["layer", "input to output", "parameters", "multiply-adds", "FLOPs", "uses per weight"], rows, "llrrrr"))


def backward_cost():
    cin, h, cout, k = 64, 56, 64, 3
    ho = out_size(h, k, 1, 1)
    f = 2 * ho * ho * cout * cin * k * k
    rows = [("forward", _fmt_int(f)), ("input gradient dX", _fmt_int(f)), ("weight gradient dW", _fmt_int(f)),
            ("backward in all", _fmt_int(2 * f)), ("forward and backward", _fmt_int(3 * f))]
    return ("FLOPs of the 64 to 64 channel 3x3 layer on a 56 by 56 map (stride 1, padding 1) for one image.\n\n"
            + _table(["pass", "FLOPs"], rows, "lr"))


# ------------------------------------------------------------------------------------------------ receptive field

STACK_TRUNK = [  # (kernel, stride, dilation) of seven layers
    (3, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)]
STACK_DILATED = [(3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8), (3, 1, 16)]


def receptive(stack):
    """(r, j) after each layer: r_l = r_(l-1) + d_l (k_l - 1) j_(l-1), j_l = j_(l-1) s_l, starting from r = j = 1."""
    r, j, out = 1, 1, []
    for k, s, d in stack:
        r = r + d * (k - 1) * j
        j = j * s
        out.append((r, j))
    return out


def traced_extent(stack, pad_of=lambda k, d: d * (k - 1) // 2):
    """Walk back from one output position of the last layer to the input and return how many input positions it reads (the number
    of distinct positions) and the length of the interval they span."""
    reach = {0}
    for k, s, d in reversed(stack):
        p = pad_of(k, d)
        reach = {i * s - p + d * m for i in reach for m in range(k)}
    return len(reach), max(reach) - min(reach) + 1


def receptive_field():
    rows = []
    for name, stack in (("trunk", STACK_TRUNK), ("dilated", STACK_DILATED)):
        rec = receptive(stack)
        for l, ((k, s, d), (r, j)) in enumerate(zip(stack, rec), 1):
            rows.append((name if l == 1 else "", l, k, s, d, r, j))
    t_t = traced_extent(STACK_TRUNK)
    t_d = traced_extent(STACK_DILATED)
    assert t_t[1] == receptive(STACK_TRUNK)[-1][0] and t_d[1] == receptive(STACK_DILATED)[-1][0]
    note = (f"Tracing the inputs read by one output of the last layer back through the trunk touches {t_t[0]} input positions spanning "
            f"an interval of {t_t[1]}, equal to r after layer 7; for the dilated stack it touches {t_d[0]} positions (the dilated "
            f"taps leave gaps) spanning {t_d[1]}, equal to r after layer 5.")
    return ("Receptive field r and jump j (input pixels between neighbouring outputs of a layer) after each layer along one axis, "
            "from r = j = 1 at the input. 'trunk' is seven 3x3 layers whose third and fifth have stride 2; 'dilated' is five stride-1 3x3 "
            "layers with dilations 1, 2, 4, 8, 16.\n\n"
            + _table(["stack", "layer l", "k", "s", "d", "r", "j"], rows, "lrrrrrr") + "\n\n" + note)


def receptive_data(n=12):
    """r after each of n layers for three stacks (the figure): plain 3x3, 3x3 with stride 2 on every third layer, and 3x3 with
    dilation doubling at every layer."""
    plain = [(3, 1, 1)] * n
    strided = [(3, 2 if l % 3 == 2 else 1, 1) for l in range(n)]
    dilated = [(3, 1, 2 ** l) for l in range(n)]
    return {name: [r for r, _ in receptive(st)] for name, st in (("plain", plain), ("strided", strided), ("dilated", dilated))}


# ------------------------------------------------------------------------------------------------ im2col


def unrolled():
    col = im2col([X5], 3)
    wrow = [v for r in SOBEL for v in r]
    y = [sum(col[hw][t] * wrow[t] for t in range(9)) for hw in range(9)]
    direct = corr1(X5, SOBEL)
    assert [int(v) for v in y] == [int(direct[hw // 3][hw % 3]) for hw in range(9)]
    out = ["The 5 by 5 image X and kernel K of the first table, unrolled: one row of the matrix per output position hw = 3 i + j, "
           "one column per tap t = 3 m + n, so entry (hw, t) is X[i + m][j + n]. Multiplying by the kernel written as a vector "
           "(taps in the same order) gives the nine outputs.", "", "```text", "col (9 by 9):", _grid(col, 2), "",
           "K as a vector: " + " ".join(str(v) for v in wrow), "col times K:     " + " ".join(str(int(v)) for v in y), "```", "",
           "These nine numbers are the output of the first table read row by row (largest difference: 0)."]
    return "\n".join(out)


MEM_CASES = [  # name, Cin, H, k, s, p
    ("3x3, 64 channels, 56x56", 64, 56, 3, 1, 1),
    ("3x3 stride 2, 128 channels, 56x56", 128, 56, 3, 2, 1),
    ("7x7 stride 2, 3 channels, 224x224", 3, 224, 7, 2, 3),
    ("1x1, 64 channels, 56x56", 64, 56, 1, 1, 0),
]


def im2col_memory():
    rows = []
    for name, cin, h, k, s, p in MEM_CASES:
        ho = out_size(h, k, s, p)
        n_in = cin * h * h
        n_col = ho * ho * cin * k * k
        rows.append((name, _fmt_int(n_in), _fmt_int(n_col), f"{n_col / n_in:.2f}", _fmt_int(4 * n_col)))
    rows2 = []
    for cout in (3, 16, 32, 128, 512):
        rows2.append((cout, f"{4 / cout:.4f}"))
    t1 = _table(["layer", "input elements", "col elements Ho*Wo*Cin*k*k", "col / input", "col bytes (f32)"], rows, "lrrrr")
    t2 = _table(["C_out", "extra bytes per FLOP"], rows2, "rr")
    return ("Elements of one image's input and of its unrolled matrix, and the bytes of the matrix in 32-bit floats.\n\n" + t1 +
            "\n\nThe unrolled matrix is written once and read once by the product: 2 * 4 * Ho*Wo*Cin*k*k bytes beyond the "
            "operands, against 2 * C_out * Ho*Wo*Cin*k*k FLOPs, which is 4 / C_out bytes per FLOP whatever the rest of the "
            "shape:\n\n" + t2)


def panel_widths():
    budget = (192 * 1024) // 4
    rows = []
    for name, cin, h, k, s, p in MEM_CASES[:3]:
        ho = out_size(h, k, s, p)
        kg, psz = cin * k * k, ho * ho
        nc = min(max((budget // kg) // 16 * 16, 16), max(psz, 16))
        rows.append((name, kg, psz, nc, _fmt_int(4 * kg * nc), _fmt_int(4 * kg * psz)))
    return ("A 192 KiB budget is 49,152 floats; the panel width is floor(floor(49152 / (Cin k k)) / 16) * 16 columns, at least 16 and "
            "at most Ho*Wo (the rule of the engine's CPU convolution, which builds the unrolled matrix one cache-sized panel at a "
            "time).\n\n"
            + _table(["layer", "Cin k k", "Ho*Wo", "panel width", "panel bytes", "whole matrix bytes"], rows, "lrrrrr"))


def layouts():
    c, h, w, k = 3, 4, 4, 2
    nchw = sorted((ci * h + kh) * w + kw for ci in range(c) for kh in range(k) for kw in range(k))
    nhwc = sorted((kh * w + kw) * c + ci for ci in range(c) for kh in range(k) for kw in range(k))
    def runs(v):
        out, start = [], v[0]
        for a, b in zip(v, v[1:] + [None]):
            if b != a + 1:
                out.append((start, a))
                start = b
        return out
    ra, rb = runs(nchw), runs(nhwc)
    fmt = lambda rs: ", ".join(f"{a}..{b}" if a != b else str(a) for a, b in rs)
    rows = [("NCHW (planes)", f"(ci * {h} + h) * {w} + w", len(ra), fmt(ra)),
            ("NHWC (pixels)", f"(h * {w} + w) * {c} + ci", len(rb), fmt(rb))]
    return ("The 12 elements of the 2 by 2 window at output position (0, 0) (all three channels) of a [3, 4, 4] image, as offsets from "
            "the start of the buffer, in the two layouts, grouped into runs of consecutive offsets.\n\n"
            + _table(["layout", "offset of (ci, h, w)", "runs", "offsets"], rows, "lrrl"))


# ------------------------------------------------------------------------------------------------ depthwise, pointwise, grouped


def _flops_bytes(macs, n_in, n_out, n_w):
    return 2 * macs, 4 * (n_in + n_out + n_w)


def separable():
    c, h, k = 128, 56, 3
    hw = h * h
    std = (c * c * k * k, hw * c * c * k * k, c * hw, c * hw, c * c * k * k)
    dw = (c * k * k, hw * c * k * k, c * hw, c * hw, c * k * k)
    pw = (c * c, hw * c * c, c * hw, c * hw, c * c)
    rows = []
    for name, (par, macs, ni, no, nw) in (("standard 3x3", std), ("depthwise 3x3", dw), ("pointwise 1x1", pw)):
        fl, by = _flops_bytes(macs, ni, no, nw)
        rows.append((name, _fmt_int(par), _fmt_int(macs), _fmt_int(fl), _fmt_int(by), f"{fl / by:.1f}"))
    par = dw[0] + pw[0]
    macs = dw[1] + pw[1]
    fl = 2 * macs
    by = _flops_bytes(*dw[1:2], dw[2], dw[3], dw[4])[1] + _flops_bytes(pw[1], pw[2], pw[3], pw[4])[1]
    rows.append(("separable pair (unfused)", _fmt_int(par), _fmt_int(macs), _fmt_int(fl), _fmt_int(by), f"{fl / by:.1f}"))
    ratio = f"{par / std[0]:.4f}"
    return ("A 128 to 128 channel layer on a 56 by 56 map, stride 1, padding 1, no bias. Bytes are the least traffic with every "
            "tensor read or written once in 32-bit floats (input, output and weights; the separable pair writes and reads the "
            "intermediate map, so its bytes are the sum of its two layers'). Intensity is FLOPs per byte.\n\n"
            + _table(["layer", "parameters", "multiply-adds", "FLOPs", "bytes", "FLOP / byte"], rows)
            + f"\n\nParameters (and multiply-adds) of the pair over the standard layer: {ratio}, which is 1/128 + 1/9 = "
            f"{1 / 128 + 1 / 9:.4f}.")


def grouped_counts():
    c, h, k = 128, 56, 3
    rows = []
    for g in (1, 2, 4, 8, 128):
        params = c * (c // g) * k * k
        macs = h * h * params
        rows.append((g, f"[{c}, {c // g}, 3, 3]", _fmt_int(params), _fmt_int(macs)))
    return ("A 128 to 128 channel 3x3 layer on a 56 by 56 map (stride 1, padding 1) with G groups: weight shape "
            "[C_out, C_in / G, 3, 3]; G = 128 is the depthwise layer.\n\n" + _table(["groups G", "weight shape", "parameters", "multiply-adds"], rows))


def grouped_check():
    name = "convolution.grouped_check"
    rng = bookrng.generator(name)
    cin, cout, g, h, k = 4, 6, 2, 5, 3
    x = [[[rng.normal() for _ in range(h)] for _ in range(h)] for _ in range(cin)]
    w = [[[[rng.normal() * 0.5 for _ in range(k)] for _ in range(k)] for _ in range(cin // g)] for _ in range(cout)]
    y = corr2d(x, w, pad=1, groups=g)
    # the same layer as G independent dense convolutions on channel slices
    parts = []
    for gi in range(g):
        xs = x[gi * (cin // g):(gi + 1) * (cin // g)]
        ws = w[gi * (cout // g):(gi + 1) * (cout // g)]
        parts += corr2d(xs, ws, pad=1)
    d1 = _maxdiff(y, parts)
    # depthwise: 4 channels, one filter each, against filtering each channel alone
    wd = [[[[rng.normal() * 0.5 for _ in range(k)] for _ in range(k)]] for _ in range(cin)]
    yd = corr2d(x, wd, pad=1, groups=cin)
    solo = [corr2d([x[c]], [wd[c]], pad=1)[0] for c in range(cin)]
    d2 = _maxdiff(yd, solo)
    assert d1 < 1e-12 and d2 < 1e-12
    return (f"Generator `bookrng.generator(\"{name}\")`, seed {_seed(name)}. Draws in order: x of shape [4, 5, 5] (channel, row, "
            "column; every entry normal()), then w of shape [6, 2, 3, 3] (every entry normal() times 0.5), then the depthwise weight "
            "[4, 1, 3, 3] (normal() times 0.5). Stride 1, padding 1, dilation 1.\n\n"
            + _table(["comparison", "largest difference", "criterion"],
                     [("grouped layer G = 2 against two dense layers on channel slices 0-1 and 2-3 of x, output slices 0-2 and 3-5", "below 1e-12", "met"),
                      ("depthwise layer (G = 4) against filtering each channel alone", "below 1e-12", "met")], "lll"))


# ------------------------------------------------------------------------------------------------ backward


def backward_direct(x, w, dy, stride=1, pad=0, dil=1):
    """The definition, as a scatter: every output gradient is added to the weight and input it multiplied in the forward.
    Order of the additions: output position (co, i, j) in row-major order, then (ci, kh, kw)."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout, k = len(w), len(w[0][0])
    dw, dx = _zeros(cout, cin, k, k), _zeros(cin, h, wd)
    for co in range(cout):
        for i in range(len(dy[0])):
            for j in range(len(dy[0][0])):
                g = dy[co][i][j]
                for ci in range(cin):
                    for kh in range(k):
                        hi = i * stride - pad + kh * dil
                        if not 0 <= hi < h:
                            continue
                        for kw in range(k):
                            wi = j * stride - pad + kw * dil
                            if 0 <= wi < wd:
                                dw[co][ci][kh][kw] += g * x[ci][hi][wi]
                                dx[ci][hi][wi] += g * w[co][ci][kh][kw]
    return dw, dx


def backward_gather(x, w, dy, stride=1, pad=0, dil=1):
    """The same two gradients as gathers: one sum per element of the result, with no accumulation into a shared location.
    dx[ci][hi][wi] sums dy over the windows that read (hi, wi): those with i s - p + kh d = hi, an exact division."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout, k = len(w), len(w[0][0])
    ho, wo = len(dy[0]), len(dy[0][0])
    dw, dx = _zeros(cout, cin, k, k), _zeros(cin, h, wd)
    for co in range(cout):
        for ci in range(cin):
            for kh in range(k):
                for kw in range(k):
                    acc = 0.0
                    for i in range(ho):
                        hi = i * stride - pad + kh * dil
                        if not 0 <= hi < h:
                            continue
                        for j in range(wo):
                            wi = j * stride - pad + kw * dil
                            if 0 <= wi < wd:
                                acc += dy[co][i][j] * x[ci][hi][wi]
                    dw[co][ci][kh][kw] = acc
    for ci in range(cin):
        for hi in range(h):
            for wi in range(wd):
                acc = 0.0
                for co in range(cout):
                    for kh in range(k):
                        nh = hi + pad - kh * dil
                        if nh < 0 or nh % stride:
                            continue
                        i = nh // stride
                        if i >= ho:
                            continue
                        for kw in range(k):
                            nw = wi + pad - kw * dil
                            if nw < 0 or nw % stride:
                                continue
                            j = nw // stride
                            if j < wo:
                                acc += dy[co][i][j] * w[co][ci][kh][kw]
                dx[ci][hi][wi] = acc
    return dw, dx


def backward_matmul(x, w, dy, stride=1, pad=0, dil=1):
    """Through the matrix product of im2col: Y = W col^T, so dW = dY col and dcol = dY^T W (Volume III), then dX = col2im(dcol)."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout, k = len(w), len(w[0][0])
    col = im2col(x, k, stride, pad, dil)
    wm = [_flat(w[co]) for co in range(cout)]
    dym = [_flat(dy[co]) for co in range(cout)]
    dwm = matmul(dym, col)
    dcol = matmul(transpose(dym), wm)
    dw = [[[[dwm[co][(ci * k + kh) * k + kw] for kw in range(k)] for kh in range(k)] for ci in range(cin)] for co in range(cout)]
    return dw, col2im(dcol, (cin, h, wd), k, stride, pad, dil)


def backward_full_conv(x, w, dy, pad=0):
    """Stride 1, dilation 1 only: dX = correlation of dY padded with k - 1 - p zeros per side with the turned kernel, input and
    output channels exchanged; dW[co][ci] = correlation of X[ci] with dY[co] (dY as the kernel)."""
    cin, cout, k = len(x), len(w), len(w[0][0])
    q = k - 1 - pad
    dyp = [[[0] * (len(dy[0][0]) + 2 * q) for _ in range(q)] + [[0] * q + r + [0] * q for r in dy[co]] + [[0] * (len(dy[0][0]) + 2 * q) for _ in range(q)]
           for co in range(cout)]
    wt = [[flip(w[co][ci]) for co in range(cout)] for ci in range(cin)]    # [ci][co][k][k]
    dx = corr2d(dyp, wt)
    ho = len(dy[0])
    xp = [[[0] * (len(x[0][0]) + 2 * pad) for _ in range(pad)] + [[0] * pad + r + [0] * pad for r in x[ci]]
          + [[0] * (len(x[0][0]) + 2 * pad) for _ in range(pad)] for ci in range(cin)]
    dw = _zeros(cout, cin, k, k)
    for co in range(cout):
        for ci in range(cin):
            res = corr1(xp[ci], dy[co])
            assert len(res) == k and ho
            dw[co][ci] = res
    return dw, dx


DY3 = [[1, 0, -1],
       [2, 1, 0],
       [0, -1, 1]]


def backward_by_hand():
    x, w = [X5], [[SOBEL]]
    dy = [DY3]
    dw, dx = backward_direct(x, w, dy)
    dw2, dx2 = backward_full_conv(x, w, dy)
    dw3, dx3 = backward_matmul(x, w, dy)
    assert _maxdiff(dw, dw2) == 0 and _maxdiff(dx, dx2) == 0 and _maxdiff(dw, dw3) == 0 and _maxdiff(dx, dx3) == 0
    g = lambda m: _grid([[int(v) for v in r] for r in m], 3)
    out = ["The image X and kernel K of the first table (stride 1, no padding), with an upstream gradient dY of the output's shape "
           "(3 by 3). dW is the correlation of X with dY used as the kernel; dX is the correlation of dY padded with k - 1 = 2 zeros "
           "per side with K turned half a turn.", "", "```text", "dY:", g(DY3), "", "dW = X correlated with dY (3 by 3):", g(dw[0][0]),
           "", "dX (5 by 5):", g(dx[0]), "```", "",
           "The same two arrays come out of the scatter form, the gather form, the matrix products of im2col and the two "
           "correlations above (largest difference between any two: 0, all values being integers)."]
    return "\n".join(out)


def backward_checks():
    name = "convolution.backward_checks"
    rng = bookrng.generator(name)
    cin, cout, h, k = 2, 3, 5, 3
    cases = [("stride 1, padding 1", 1, 1, 1), ("stride 2, padding 1", 2, 1, 1), ("stride 2, padding 0", 2, 0, 1),
             ("stride 1, padding 2, dilation 2", 1, 2, 2)]
    x = [[[rng.normal() for _ in range(h)] for _ in range(h)] for _ in range(cin)]
    w = [[[[rng.normal() * 0.5 for _ in range(k)] for _ in range(k)] for _ in range(cin)] for _ in range(cout)]
    rows = []
    for label, s, p, d in cases:
        ho = out_size(h, k, s, p, d)
        dy = [[[rng.normal() for _ in range(ho)] for _ in range(ho)] for _ in range(cout)]
        a = backward_direct(x, w, dy, s, p, d)
        b = backward_gather(x, w, dy, s, p, d)
        c = backward_matmul(x, w, dy, s, p, d)
        d_ab = max(_maxdiff(a[0], b[0]), _maxdiff(a[1], b[1]))
        d_ac = max(_maxdiff(a[0], c[0]), _maxdiff(a[1], c[1]))
        assert d_ab < 1e-12 and d_ac < 1e-12
        # adjointness: y is bilinear in (x, w), so <y, dy> = <x, dX> = <w, dW>
        y = corr2d(x, w, s, p, d)
        lhs = sum(a_ * b_ for a_, b_ in zip(_flat(y), _flat(dy)))
        r1 = sum(a_ * b_ for a_, b_ in zip(_flat(x), _flat(a[1])))
        r2 = sum(a_ * b_ for a_, b_ in zip(_flat(w), _flat(a[0])))
        assert abs(lhs - r1) < 1e-12 and abs(lhs - r2) < 1e-12
        # finite differences of L = sum(dy * y), central, step 1e-5, every entry of x and w
        def loss():
            return sum(a_ * b_ for a_, b_ in zip(_flat(corr2d(x, w, s, p, d)), _flat(dy)))

        bad, worst, n_checked = 0, 0.0, 0
        for tensor, grad in ((x, a[1]), (w, a[0])):
            for idx in _indices(tensor):
                num = _central(tensor, idx, loss)
                ana = _at(grad, idx)
                gate = 1e-7 + 1e-5 * max(abs(num), abs(ana))
                bad += abs(num - ana) > gate
                worst = max(worst, abs(num - ana))
                n_checked += 1
        assert bad == 0 and worst < 1e-8
        rows.append((label, f"[{cout}, {ho}, {ho}]", "below 1e-12", "below 1e-12", f"{n_checked} of {n_checked} pass", "below 1e-8"))
    return (f"Generator `bookrng.generator(\"{name}\")`, seed {_seed(name)}. Draws in order: x of shape [2, 5, 5] (channel, row, column; "
            "every entry normal()), w of shape [3, 2, 3, 3] (every entry normal() times 0.5), then for each row of the table in "
            "turn the upstream gradient dY of shape [3, Ho, Wo] (every entry normal()). Each sum runs over the loops in the order "
            "of the definition; the comparisons below are to criteria that do not depend on that order. 'routes' compares the "
            "scatter, gather and matrix-product routes; 'adjoint' compares the three numbers <y, dY>, <x, dX> and <w, dW>; the "
            "gradient check perturbs every entry of x and w by +-1e-5, in 64-bit arithmetic, and applies the gate "
            "|a - n| <= 1e-7 + 1e-5 max(|a|, |n|) of Volume III, Chapter 6.\n\n"
            + _table(["case", "dY shape", "routes differ by", "adjoint differs by", "gradient check", "largest |a - n|"], rows, "llllll"))


def backward_stride1_identity():
    name = "convolution.backward_checks"
    rng = bookrng.generator(name + ".stride1")
    cin, cout, h, k = 2, 3, 6, 3
    rows = []
    x = [[[rng.normal() for _ in range(h)] for _ in range(h)] for _ in range(cin)]
    w = [[[[rng.normal() for _ in range(k)] for _ in range(k)] for _ in range(cin)] for _ in range(cout)]
    for p in (0, 1, 2):
        ho = out_size(h, k, 1, p)
        dy = [[[rng.normal() for _ in range(ho)] for _ in range(ho)] for _ in range(cout)]
        a = backward_direct(x, w, dy, 1, p)
        b = backward_full_conv(x, w, dy, p)
        d = max(_maxdiff(a[0], b[0]), _maxdiff(a[1], b[1]))
        assert d < 1e-12
        rows.append((p, f"{k - 1 - p}", f"[{cout}, {ho}, {ho}]", "below 1e-12"))
    return (f"Generator `bookrng.generator(\"{name}.stride1\")`, seed {_seed(name + '.stride1')}. Draws in order: x of shape [2, 6, 6] "
            "(every entry normal()), w of shape [3, 2, 3, 3] (every entry normal()), then dY of shape [3, Ho, Wo] (every entry "
            "normal()) for padding 0, 1 and 2 in turn. dX from the correlation of the padded dY with the turned, channel-exchanged "
            "kernel, and dW from the correlation of the padded X with dY, against the scatter form.\n\n"
            + _table(["forward padding p", "zeros added to dY per side (k - 1 - p)", "dY shape", "largest difference"], rows, "rrll"))


# ------------------------------------------------------------------------------------------------ transposed convolution


def convT_out(h, k, s, p, d=1, out_pad=0):
    return (h - 1) * s - 2 * p + d * (k - 1) + out_pad + 1


def convT2d(x, w, stride=1, pad=0, dil=1, out_pad=0):
    """x[ci][hi][wi], w[ci][co][kh][kw]; every input value stamps the kernel, scaled, onto the output at hi s - p + kh d."""
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout, k = len(w[0]), len(w[0][0][0])
    ho, wo = convT_out(h, k, stride, pad, dil, out_pad), convT_out(wd, k, stride, pad, dil, out_pad)
    y = _zeros(cout, ho, wo)
    for ci in range(cin):
        for hi in range(h):
            for wi in range(wd):
                v = x[ci][hi][wi]
                for co in range(cout):
                    for kh in range(k):
                        oh = hi * stride - pad + kh * dil
                        if not 0 <= oh < ho:
                            continue
                        for kw in range(k):
                            ow = wi * stride - pad + kw * dil
                            if 0 <= ow < wo:
                                y[co][oh][ow] += v * w[ci][co][kh][kw]
    return y


def transposed_sizes():
    rows = []
    for h in range(4, 9):
        n = out_size(h, 3, 2, 1)
        rows.append((h, n, convT_out(n, 3, 2, 1), convT_out(n, 3, 2, 1, out_pad=1), "yes" if n == out_size(convT_out(n, 3, 2, 1), 3, 2, 1) else "no",
                     "yes" if n == out_size(convT_out(n, 3, 2, 1, out_pad=1), 3, 2, 1) else "no"))
    rows2 = [(h, convT_out(h, 2, 2, 0), convT_out(h, 4, 2, 1)) for h in (4, 8, 16)]
    return ("A convolution with k = 3, stride 2, padding 1 maps a length H to n = floor((H - 1) / 2) + 1. The transposed convolution "
            "with the same k, stride and padding maps n to (n - 1) * 2 - 2 + 2 + out_pad + 1 = 2n - 1 + out_pad. The last two "
            "columns check that the length it returns is one the convolution maps back to n.\n\n"
            + _table(["H", "n (convolution)", "transposed, out_pad 0", "transposed, out_pad 1", "conv gives n back (0)", "conv gives n back (1)"], rows)
            + "\n\nThe two common upsampling shapes: k = 2, stride 2, padding 0 gives exactly 2n, and k = 4, stride 2, padding 1 gives 2n.\n\n"
            + _table(["n", "k=2, s=2, p=0", "k=4, s=2, p=1"], rows2))


def transposed_adjoint():
    name = "convolution.transposed_adjoint"
    rng = bookrng.generator(name)
    h, k, s, p = 5, 3, 2, 1
    w = [[[rng.normal() for _ in range(k)] for _ in range(k)]]
    ho = out_size(h, k, s, p)
    n_in, n_out = h * h, ho * ho
    a = []        # the strided convolution as a matrix: column c is the output for the c-th unit input
    for c in range(n_in):
        e = _zeros(1, h, h)
        e[0][c // h][c % h] = 1.0
        a.append(_flat(corr2d(e, [[w[0]]], s, p)))
    a = transpose(a)     # [n_out][n_in]
    y = [[[rng.normal() for _ in range(ho)] for _ in range(ho)]]
    ty = convT2d(y, [[w[0]]], s, p)
    atv = [sum(a[r][c] * _flat(y)[r] for r in range(n_out)) for c in range(n_in)]
    d_mat = max(abs(u - v) for u, v in zip(_flat(ty), atv))
    xx = [[[rng.normal() for _ in range(h)] for _ in range(h)]]
    lhs = sum(u * v for u, v in zip(_flat(corr2d(xx, [[w[0]]], s, p)), _flat(y)))
    rhs = sum(u * v for u, v in zip(_flat(xx), _flat(ty)))
    assert d_mat < 1e-12 and abs(lhs - rhs) < 1e-12
    # the gather form of the transposed convolution (one sum per output, exact division)
    ty2 = _zeros(1, h, h)
    for oh in range(h):
        for ow in range(h):
            acc = 0.0
            for kh in range(k):
                nh = oh + p - kh
                if nh < 0 or nh % s:
                    continue
                for kw in range(k):
                    nw = ow + p - kw
                    if nw < 0 or nw % s:
                        continue
                    i, j = nh // s, nw // s
                    if i < ho and j < ho:
                        acc += y[0][i][j] * w[0][kh][kw]
            ty2[0][oh][ow] = acc
    assert _maxdiff(ty, ty2) < 1e-12
    return (f"Generator `bookrng.generator(\"{name}\")`, seed {_seed(name)}. Draws in order: the kernel w of shape [3, 3] (every entry "
            "normal()), the vector y of shape [3, 3] (every entry normal()), then x of shape [5, 5] (every entry normal()). The "
            "strided convolution (k = 3, stride 2, padding 1) of a 5 by 5 map is a matrix A of shape [9, 25], built column by column "
            "by convolving the 25 unit maps.\n\n"
            + _table(["comparison", "result", "criterion"],
                     [("transposed convolution of y (scatter) against A^T y", "largest difference below 1e-12", "met"),
                      ("transposed convolution of y (scatter) against its gather form", "largest difference below 1e-12", "met"),
                      ("<A x, y> against <x, A^T y>", "difference below 1e-12", "met")], "lll"))


def overlap_1d(n, k, s):
    """How many of the n stamps of a width-k kernel at stride s cover each position of the (n - 1) s + k output positions."""
    cnt = [0] * ((n - 1) * s + k)
    for i in range(n):
        for m in range(k):
            cnt[i * s + m] += 1
    return cnt


def checkerboard_counts(n, k, s):
    """The 2D overlap counts: the outer product of the 1D counts."""
    c = overlap_1d(n, k, s)
    return [[a * b for b in c] for a in c]


def phase_sums():
    cases = [("[1, 1]", [1, 1], 2), ("[1, 1, 1]", [1, 1, 1], 2), ("[1, 2, 1]", [1, 2, 1], 2), ("[1, 3, 3, 1]", [1, 3, 3, 1], 2),
             ("[1, 1, 1, 1]", [1, 1, 1, 1], 2), ("[1, 1, 1, 1, 1, 1]", [1] * 6, 3), ("[1, 2, 3, 2, 1]", [1, 2, 3, 2, 1], 3)]
    rows = []
    for name, ker, s in cases:
        n = 6
        out = [0] * ((n - 1) * s + len(ker))
        for i in range(n):
            for m, v in enumerate(ker):
                out[i * s + m] += v
        sums = [sum(ker[m] for m in range(ph, len(ker), s)) for ph in range(s)]
        interior = out[len(ker) - s:(n - 1) * s + 1]
        rows.append((name, s, " / ".join(map(str, sums)), " ".join(map(str, out)), "uniform" if len(set(interior)) == 1 else "ripple"))
    return ("A constant input of 6 ones through a 1D transposed convolution with the given kernel and stride (no padding). An output "
            "position receives the taps whose index is congruent to its own position modulo the stride, so for constant input the "
            "interior output cycles through the 'phase sums' (the sums of every s-th tap). The 'interior' is the stretch every phase "
            "reaches.\n\n"
            + _table(["kernel", "stride s", "phase sums", "output", "interior"], rows, "lrlll"))


def checkerboard_grids():
    out = []
    for k, s in ((2, 2), (3, 2), (4, 2)):
        c = checkerboard_counts(4, k, s)
        out += [f"k = {k}, stride {s}, a 4 by 4 input of ones through a kernel of ones gives a {len(c)} by {len(c)} output of overlap counts:",
                "", "```text", _grid(c, 2), "```", ""]
    return "\n".join(out[:-1])


def band_and_layout():
    # (a) the output-padding band: k = 3, stride 2, input 4 by 4 of ones, kernel of ones
    ones = [[[1.0] * 4 for _ in range(4)]]
    ker = [[[[1.0] * 3 for _ in range(3)]]]
    rows = []
    for pad in (0, 1):
        y0 = convT2d(ones, ker, 2, pad, out_pad=0)[0]
        y1 = convT2d(ones, ker, 2, pad, out_pad=1)[0]
        last_row = [int(v) for v in y1[-1]]
        rows.append((pad, f"{len(y0)} by {len(y0)}", f"{len(y1)} by {len(y1)}", " ".join(map(str, last_row)),
                     "zeros" if not any(last_row) else "not zeros"))
    band = _table(["padding p", "out_pad 0", "out_pad 1", "last row of the out_pad 1 output", "the extra row is"], rows, "rrrll")
    # (b) the weight layout that always binds: Cin = 2, Cout = 3, k = 2, stride 2; weights 1, 2, 3, ... in storage order
    cin, cout, k = 2, 3, 2
    flat_w = list(range(1, cin * cout * k * k + 1))
    right = [[[[flat_w[((ci * cout + co) * k + kh) * k + kw] for kw in range(k)] for kh in range(k)] for co in range(cout)] for ci in range(cin)]
    wrong = [[[[flat_w[((co * cin + ci) * k + kh) * k + kw] for kw in range(k)] for kh in range(k)] for co in range(cout)] for ci in range(cin)]
    x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    ya, yb = convT2d(x, right, 2), convT2d(x, wrong, 2)
    d = _maxdiff(ya, yb)
    cin = cout = 2
    flat2 = list(range(1, cin * cout * k * k + 1))
    r2 = [[[[flat2[((ci * cout + co) * k + kh) * k + kw] for kw in range(k)] for kh in range(k)] for co in range(cout)] for ci in range(cin)]
    w2 = [[[[flat2[((co * cin + ci) * k + kh) * k + kw] for kw in range(k)] for kh in range(k)] for co in range(cout)] for ci in range(cin)]
    d2 = _maxdiff(convT2d(x, r2, 2), convT2d(x, w2, 2))
    lay = _table(["channels C_in to C_out", "weight elements", "output shape", "largest difference between the two readings"],
                 [("2 to 3", cin * 3 * k * k, "[3, 4, 4]", _fmt_int(int(d))), ("2 to 2", 16, "[2, 4, 4]", _fmt_int(int(d2)))], "lrll")
    return ("Part 1. A 4 by 4 input of ones, kernel of ones, k = 3, stride 2. Without output padding the output is 9 by 9 at p = 0 "
            "and 7 by 7 at p = 1; output padding 1 adds one row and one column at the end.\n\n" + band +
            "\n\nPart 2. A transposed convolution stores its weights as [C_in, C_out, k, k]; a forward convolution stores "
            "[C_out, C_in, k, k]. The input is the [2, 2, 2] map with entries 1 to 8 in row-major order, stride 2, k = 2, "
            "and the weight buffer holds 1, 2, 3, ... in storage order. 'Right' reads element (ci, co, kh, kw) at "
            "((ci * C_out + co) * 2 + kh) * 2 + kw; 'wrong' reads the same buffer as [C_out, C_in, k, k], taking "
            "((co * C_in + ci) * 2 + kh) * 2 + kw.\n\n" + lay)


# ------------------------------------------------------------------------------------------------ the Fourier route


def _fft_mults(length):
    """Real multiplications of one complex FFT of power-of-two length L under the model of Volume I, Chapter 14: (L / 2) log2 L
    complex products at 4 real multiplications each."""
    return 4 * (length // 2) * (length.bit_length() - 1)


def _pow2_at_least(n):
    length = 1
    while length < n:
        length *= 2
    return length


def fft_costs_single(h, k):
    """(direct, FFT) real multiplications to filter one h by h channel with a k by k kernel into the h by h 'same' output:
    direct h^2 k^2; FFT: three 2D transforms of side L >= h + k - 1 (2 L one-dimensional transforms each) and L^2 complex products."""
    length = _pow2_at_least(h + k - 1)
    return h * h * k * k, 3 * 2 * length * _fft_mults(length) + 4 * length * length, length


def fft_single_channel():
    h = 224
    rows = []
    for k in (3, 7, 15, 31, 63, 127):
        d, f, length = fft_costs_single(h, k)
        rows.append((k, _fmt_int(d), length, _fmt_int(f), f"{d / f:.3f}"))
    first = next(k for k in range(1, h + 1, 2) if fft_costs_single(h, k)[0] > fft_costs_single(h, k)[1])
    assert all(fft_costs_single(h, k)[0] > fft_costs_single(h, k)[1] for k in range(first, h + 1, 2))
    return (f"One {h} by {h} channel filtered by a k by k kernel into a {h} by {h} output. Direct: {h}^2 k^2 multiplications. "
            "Through the FFT: three 2D transforms (kernel, image, inverse) of side L, the smallest power of two at least H + k - 1, "
            "each made of 2L one-dimensional transforms of 4 (L/2) log2 L real multiplications, plus L^2 complex products at 4 real "
            f"multiplications each. Smallest odd k at which the FFT route is cheaper: k = {first}, and for every odd k above it.\n\n"
            + _table(["kernel k", "direct", "transform side L", "FFT route", "direct / FFT"], rows))


def fft_layer():
    h, cin, cout, k = 56, 64, 64, 3
    length = _pow2_at_least(h + k - 1)
    direct = h * h * cin * cout * k * k
    per_transform = 2 * length * _fft_mults(length)
    transforms = (cin + cout) * per_transform           # weights already transformed and stored
    products = cin * cout * 4 * length * length
    w_spatial = 4 * cin * cout * k * k
    w_freq = cin * cout * length * length * 8           # a complex value is two 32-bit floats
    rows = [("direct", _fmt_int(direct), "", _fmt_int(w_spatial)),
            ("FFT, weights transformed beforehand", _fmt_int(transforms + products),
             f"{_fmt_int(transforms)} for the {cin + cout} transforms of the activations, {_fmt_int(products)} for the products",
             _fmt_int(w_freq))]
    return (f"The 64 to 64 channel 3x3 layer on a 56 by 56 map (one image; transform side L = {length}). Multiplications are real "
            "multiplications; the FFT route transforms each of the 64 input channels once and inverse-transforms each of the 64 "
            "output channels once (2L one-dimensional transforms of 4 (L/2) log2 L multiplications per 2D transform) and forms "
            "C_in C_out L^2 complex products at 4 real multiplications each; the weights' transforms are computed once, beforehand, "
            "and stored at 8 bytes per complex value. No saving from the symmetry of the transform of a real signal is taken.\n\n"
            + _table(["route", "multiplications", "breakdown", "weight bytes"], rows, "lrll")
            + f"\n\nWeight bytes in the frequency domain over the spatial kernel's: {w_freq / w_spatial:.0f} times.")


def winograd():
    return ("F(2x2, 3x3) computes a 2 by 2 block of outputs of a 3x3 convolution from a 4 by 4 block of inputs with 4 * 4 = 16 "
            "multiplications, where the direct sum needs 2 * 2 * 9 = 36: a factor 36 / 16 = 2.25.")


def padded_slide():
    padded, out = sliding_data()
    g = lambda m: _grid(m, 2)
    return (
        "The image X and kernel K of the first table with a border of one zero on every side (padding 1) and stride 1: the padded "
        "image is 7 by 7 and the output is 5 by 5, the size of the input.\n\n```text\npadded X:\n"
        + g(padded)
        + "\n\noutput Y (5 by 5):\n"
        + _grid(out, 3)
        + "\n```\n\nThe output at the top-left corner, (0, 0), reads the window whose top row and left column are zeros: "
        + f"(-1)(0) + (0)(0) + (1)(0) + (-2)(0) + (0)(1) + (2)(1) + (-1)(0) + (0)(1) + (1)(1) = {out[0][0]}."
    )


def stamping():
    x = [[[1, 2], [3, 4]]]
    k = [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]
    y = convT2d(x, k, 2, 0)[0]
    stamps = []
    for (i, j), v in (((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)):
        stamps.append(f"input {v} at ({i}, {j}) is stamped onto rows {2 * i} to {2 * i + 2}, columns {2 * j} to {2 * j + 2}")
    return (
        "A 2 by 2 input [[1, 2], [3, 4]] through a transposed convolution with k = 3, stride 2, padding 0 and the kernel "
        "[[1, 2, 1], [2, 4, 2], [1, 2, 1]]. Each input value multiplies the whole kernel and lands with its top-left corner at "
        "(2i, 2j) of the output; where stamps overlap they add. The output is (2 - 1) * 2 + 3 = 5 by 5.\n\n```text\n"
        + "\n".join(stamps)
        + "\n\noutput (5 by 5):\n"
        + _grid([[int(v) for v in r] for r in y], 3)
        + "\n```\n\nThe centre entry (2, 2) receives all four stamps: 1 * 1 + 2 * 1 + 3 * 1 + 4 * 1 = "
        + f"{int(y[2][2])}; this kernel has equal sums on its even and odd taps ({1 + 1} and 2 along a row), so a constant input "
        "would give a constant interior."
    )
