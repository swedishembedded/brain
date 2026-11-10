# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Dense Prediction: Segmentation and Depth".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: dense_prediction.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* one 8 by 8 scene with a label map and a depth map, and the resolution ceiling of a predictor that is constant on blocks;
* the ways to double a map (nearest, bilinear under two conventions, two transposed convolutions, the pixel shuffle), and the
  identity between a transposed convolution with k = s and a 1 by 1 convolution followed by a pixel shuffle;
* the confusion matrix, the intersection over union of every class and their mean, computed exactly on a printed prediction;
* a small encoder-decoder segmenter and three simpler relatives trained on a noisy synthetic task, checked by central
  differences, with per-class IoU on a held-out set (digits printed only where they survive the nudges of `stable.py`);
* the pinhole projection of a scene and of the same scene at twice the size, the ground plane's inverse depth, the closed-form
  scale-and-shift alignment, and the depth metrics under three alignments;
* parameter and multiply-add counts of ZipDepth (before and after folding every branch and normalization into one convolution)
  and of the BiSeNet context path, from the layer lists alone.

Arithmetic is the language's 64-bit floating point unless a table says otherwise. Random inputs come from `bookrng.generator(name)`
with the names, seeds and draw orders printed beside each table. Sums over taps run in the order (input channel, kernel row, kernel
column), over positions row-major and over images in index order.

Swedish Embedded AB implements solutions for per-pixel prediction models that are verified against a hand computation. If your
team needs expertise in segmentation and depth estimation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import multiprocessing
from fractions import Fraction
from functools import lru_cache
from operator import mul

import bookrng
import cnns
import convolution as cv
import images as im
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


def _frac(fr, d=4):
    return f"{fr.numerator}/{fr.denominator} = {float(fr):.{d}f}" if fr.denominator != 1 else str(fr.numerator)


def _grid(rows, width=2):
    return "\n".join(" ".join(f"{v:>{width}}" for v in r) for r in rows)


def _block(lines):
    return "```text\n" + "\n".join(lines) + "\n```"


# ------------------------------------------------------------------------------------------------ the scene

H = W = 8
CLASS_NAMES = ("sky", "ground", "box")
N_CLS = 3
HORIZON = 3  # image row on which the horizon lies; rows above are sky, rows from here down are ground
FH = 4.0  # focal length in pixels times the camera's height above the ground in metres (4 pixels, 1 metre)
BOX = (3, 4, 6, 7)  # (x1, y1, x2, y2): columns 3..5, rows 4..6; the box stands on the ground with its base at the row-7 edge


def scene_labels(horizon=HORIZON, box=BOX):
    """Label map lab[y][x] in {0 sky, 1 ground, 2 box}."""
    lab = [[0 if y < horizon else 1 for _ in range(W)] for y in range(H)]
    x1, y1, x2, y2 = box
    for y in range(y1, y2):
        for x in range(x1, x2):
            lab[y][x] = 2
    return lab


def scene_inverse_depth(lab, horizon=HORIZON, box=BOX, fh=FH):
    """Inverse depth 1/Z per pixel, None for the sky. A ground pixel on row y lies on the line of sight (y + 0.5 - horizon)/f below
    the optical axis, so with camera height h its depth is Z = f h / (y + 0.5 - horizon) and 1/Z = (y + 0.5 - horizon)/(f h). The
    box is a vertical plane whose depth is that of the ground at its base row edge y2."""
    nu = [[None] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if lab[y][x] == 1:
                nu[y][x] = (y + 0.5 - horizon) / fh
            elif lab[y][x] == 2:
                nu[y][x] = (box[3] - horizon) / fh
    return nu


def scene():
    lab = scene_labels()
    nu = scene_inverse_depth(lab)
    z = [[None if v is None else 1.0 / v for v in r] for r in nu]
    counts = [sum(r.count(c) for r in lab) for c in range(N_CLS)]
    lab_txt = _grid(lab, 1)
    z_txt = "\n".join(" ".join(f"{'  -  ' if v is None else format(v, '5.2f')}" for v in r) for r in z)
    nu_txt = "\n".join(" ".join(f"{'  -  ' if v is None else format(v, '5.3f')}" for v in r) for r in nu)
    return (f"An 8 by 8 scene, row 0 at the top. The horizon lies on the top edge of row {HORIZON}: rows 0 to {HORIZON - 1} are sky (class 0), "
            f"the others ground (class 1), and a box (class 2) of columns 3 to 5 and rows 4 to 6 stands on the ground with its base on "
            f"the lower edge of row 6. The camera looks along the horizon with focal length 4 pixels, 1 metre above the ground. Ground "
            f"depth on row y is Z = 4 / (y + 0.5 - {HORIZON}) metres; the box is a vertical plane at the depth of its base, 4 / (7 - "
            f"{HORIZON}) = 1 metre. The sky has no depth.\n\n"
            f"Label map (0 sky, 1 ground, 2 box):\n\n" + _block(lab_txt.split("\n"))
            + f"\n\nDepth Z in metres:\n\n" + _block(z_txt.split("\n"))
            + f"\n\nInverse depth 1/Z in 1/metres:\n\n" + _block(nu_txt.split("\n"))
            + f"\n\nPixels per class: sky {counts[0]}, ground {counts[1]}, box {counts[2]} of 64.")


# ------------------------------------------------------------------------------------------------ confusion matrix and IoU


def confusion(truth, pred, n_cls=N_CLS):
    """conf[i][j]: the number of pixels of true class i that are predicted as class j (flat or nested lists, same shape)."""
    conf = [[0] * n_cls for _ in range(n_cls)]
    t = [v for r in truth for v in r] if isinstance(truth[0], list) else truth
    p = [v for r in pred for v in r] if isinstance(pred[0], list) else pred
    for a, b in zip(t, p):
        conf[a][b] += 1
    return conf


def class_iou(conf):
    """IoU_c = TP / (TP + FP + FN) as an exact fraction per class; None for a class absent from truth and prediction."""
    n = len(conf)
    out = []
    for c in range(n):
        tp = conf[c][c]
        fn = sum(conf[c]) - tp
        fp = sum(conf[r][c] for r in range(n)) - tp
        out.append(Fraction(tp, tp + fp + fn) if tp + fp + fn else None)
    return out


def mean_iou(ious):
    """Mean over the classes that occur (None entries are left out)."""
    vals = [v for v in ious if v is not None]
    return sum(vals, Fraction(0)) / len(vals)


def block_vote(lab, f):
    """The best prediction that is constant on every f by f block: each block takes the class of most of its pixels (the lowest class
    index when several tie), then the block is repeated f times along both axes."""
    out = [[0] * W for _ in range(H)]
    for by in range(0, H, f):
        for bx in range(0, W, f):
            cnt = [0] * N_CLS
            for y in range(by, by + f):
                for x in range(bx, bx + f):
                    cnt[lab[y][x]] += 1
            win = cnt.index(max(cnt))
            for y in range(by, by + f):
                for x in range(bx, bx + f):
                    out[y][x] = win
    return out


def ceiling():
    lab = scene_labels()
    rows = []
    for f in (1, 2, 4, 8):
        pred = block_vote(lab, f)
        conf = confusion(lab, pred)
        iou = class_iou(conf)
        acc = Fraction(sum(conf[c][c] for c in range(N_CLS)), 64)
        rows.append((f, f"{8 // f} by {8 // f}", _frac(acc, 4), *[_fx(float(v), 4) if v is not None else "-" for v in iou],
                     _fx(float(mean_iou(iou)), 4)))
    return ("For each output stride f, the best prediction that is constant on f by f blocks of the scene above (each block takes the "
            "class of most of its pixels, the lowest class on a tie) scored against the true label map. No network that predicts at "
            "stride f and repeats the result can do better on this scene.\n\n"
            + _table(["stride f", "grid", "pixel accuracy", "IoU sky", "IoU ground", "IoU box", "mean IoU"], rows))


def iou_example():
    lab = scene_labels()
    pred = block_vote(lab, 2)
    conf = confusion(lab, pred)
    iou = class_iou(conf)
    err = [["." if lab[y][x] == pred[y][x] else "x" for x in range(W)] for y in range(H)]
    rows = []
    for c in range(N_CLS):
        tp = conf[c][c]
        fn = sum(conf[c]) - tp
        fp = sum(conf[r][c] for r in range(N_CLS)) - tp
        rows.append((f"{c} ({CLASS_NAMES[c]})", tp, fp, fn, tp + fp + fn, _frac(iou[c], 4)))
    return ("The prediction is the stride-2 block vote of the scene (the lowest class wins a tie). Rows of the confusion matrix are "
            "the true class, columns the predicted one.\n\n"
            "Truth:\n\n" + _block(_grid(lab, 1).split("\n")) + "\n\nPrediction:\n\n" + _block(_grid(pred, 1).split("\n"))
            + "\n\nWrong pixels (x):\n\n" + _block([" ".join(r) for r in err])
            + "\n\n" + _table(["true \\ predicted", "0 sky", "1 ground", "2 box"], [(f"{c} ({CLASS_NAMES[c]})", *conf[c]) for c in range(N_CLS)])
            + "\n\n" + _table(["class", "TP", "FP", "FN", "TP + FP + FN", "IoU"], rows)
            + f"\n\nMean IoU = ({' + '.join(f'{v.numerator}/{v.denominator}' for v in iou)}) / 3 = {_frac(mean_iou(iou), 4)}. "
            f"Pixel accuracy = {sum(conf[c][c] for c in range(N_CLS))}/64 = {_fx(sum(conf[c][c] for c in range(N_CLS)) / 64, 4)}.")


def figure_data():
    """Truth, the stride-2 block-vote prediction and their confusion data, for the figure of a label map, a prediction and an error map."""
    lab = scene_labels()
    pred = block_vote(lab, 2)
    conf = confusion(lab, pred)
    return lab, pred, conf, class_iou(conf)


# ------------------------------------------------------------------------------------------------ doubling a map


def bowl(y, x):
    """The smooth 8 by 8 test surface: squared distance from the centre (3.5, 3.5) over its largest value 24.5."""
    return ((x - 3.5) ** 2 + (y - 3.5) ** 2) / 24.5


def avg_pool2(plane):
    return [[(plane[2 * i][2 * j] + plane[2 * i][2 * j + 1] + plane[2 * i + 1][2 * j] + plane[2 * i + 1][2 * j + 1]) / 4.0
             for j in range(len(plane[0]) // 2)] for i in range(len(plane) // 2)]


def nearest2(plane):
    return [[plane[y // 2][x // 2] for x in range(2 * len(plane[0]))] for y in range(2 * len(plane))]


def transposed_double(plane, kernel2d, pad, out_pad):
    """Stride-2 transposed convolution of one channel with a fixed kernel (convolution.convT2d), doubling an n by n map."""
    n = len(plane)
    return cv.convT2d([plane], [[kernel2d]], stride=2, pad=pad, out_pad=out_pad)[0]


def pixel_shuffle(x, r):
    """x[c r^2 + a r + b][h][w] to y[c][h r + a][w r + b]: the channels c r^2 .. c r^2 + r^2 - 1 of a pixel become its r by r block."""
    c_in, h, w = len(x), len(x[0]), len(x[0][0])
    c_out = c_in // (r * r)
    y = [[[0.0] * (w * r) for _ in range(h * r)] for _ in range(c_out)]
    for c in range(c_out):
        for i in range(h):
            for j in range(w):
                for a in range(r):
                    for b in range(r):
                        y[c][i * r + a][j * r + b] = x[c * r * r + a * r + b][i][j]
    return y


def doubling():
    truth = [[bowl(y, x) for x in range(8)] for y in range(8)]
    low = avg_pool2(truth)
    const = [[1.0] * 4 for _ in range(4)]
    k3 = [[0.25] * 3 for _ in range(3)]  # a kernel of ones divided by 4
    k4 = [[a * b / 16.0 for b in (1, 3, 3, 1)] for a in (1, 3, 3, 1)]  # bilinear: [1, 3, 3, 1] / 4 along each axis

    methods = [
        ("nearest", nearest2),
        ("bilinear, half-pixel", lambda p: im.resize2d(p, 8, 8, "half")),
        ("bilinear, corners", lambda p: im.resize2d(p, 8, 8, "corners")),
        ("transposed, k = 3, kernel of ones / 4", lambda p: transposed_double(p, k3, 1, 1)),
        ("transposed, k = 4, [1, 3, 3, 1]^2 / 16", lambda p: transposed_double(p, k4, 1, 0)),
    ]
    rows = []
    for name, f in methods:
        c = f(const)
        interior = [c[y][x] for y in range(1, 7) for x in range(1, 7)]
        up = f(low)
        errs = [abs(up[y][x] - truth[y][x]) for y in range(8) for x in range(8)]
        rows.append((name, _fx(min(interior), 4), _fx(max(interior), 4), _fx(sum(errs) / 64, 4), _fx(max(errs), 4)))
    return ("The surface (x - 3.5)^2 + (y - 3.5)^2 over 24.5 on an 8 by 8 grid is averaged over 2 by 2 blocks to a 4 by 4 map and "
            "doubled again by five rules. 'Constant input' is a 4 by 4 map of ones doubled by the same rule: its output should be all "
            "ones, and the smallest and largest value on rows and columns 1 to 6 (the interior) show whether the rule adds a pattern. The error is "
            "against the original 8 by 8 surface, over all 64 pixels. The transposed convolutions use stride 2 and padding 1, "
            "with output padding 1 for k = 3 and 0 for k = 4, and `convolution.convT2d` of the earlier chapter.\n\n"
            + _table(["rule", "constant input: interior min", "interior max", "mean absolute error", "largest error"], rows))


def shuffle_identity():
    name = "dense_prediction.shuffle_identity"
    gen = bookrng.generator(name)
    c_in, c_out, r, n = 3, 2, 2, 3
    x = [[[gen.normal() for _ in range(n)] for _ in range(n)] for _ in range(c_in)]
    w = [[[[gen.normal() for _ in range(r)] for _ in range(r)] for _ in range(c_out)] for _ in range(c_in)]
    y_t = cv.convT2d(x, w, stride=r, pad=0)
    # 1 by 1 convolution with c_out r^2 output channels, weight V[c r^2 + a r + b][ci] = w[ci][c][a][b]; sums over ci in order
    feats = []
    for c in range(c_out):
        for a in range(r):
            for b in range(r):
                feats.append([[sum(x[ci][i][j] * w[ci][c][a][b] for ci in range(c_in)) for j in range(n)] for i in range(n)])
    y_s = pixel_shuffle(feats, r)
    diff = max(abs(y_t[c][i][j] - y_s[c][i][j]) for c in range(c_out) for i in range(n * r) for j in range(n * r))
    # the numbered layout
    lay = pixel_shuffle([[[10 * (k + 1) + 2 * i + j for j in range(2)] for i in range(2)] for k in range(4)], 2)
    return (f"Generator `bookrng.generator(\"{name}\")`, {_seed(name)}. Draws in order: the input x of shape [3, 3, 3] (channel, row, "
            f"column; every entry normal()), then the transposed-convolution weight w of shape [3, 2, 2, 2] (input channel, output "
            f"channel, row, column; every entry normal()). Left: the transposed convolution with k = 2, stride 2, padding 0 "
            f"(`convolution.convT2d`) gives [2, 6, 6]. Right: a 1 by 1 convolution with weight V[c * 4 + m * 2 + n][ci] = w[ci][c][m][n] "
            f"(sums over ci in order) gives 8 channels of 3 by 3, which the pixel shuffle y[c][2i + m][2j + n] = z[4c + 2m + n][i][j] "
            f"rearranges to [2, 6, 6].\n\n"
            + _table(["quantity", "criterion", "result"], [("largest difference of the two [2, 6, 6] outputs", "below 1e-12",
                                                            "met" if diff < 1e-12 else f"NOT met ({_sci(diff)})")])
            + "\n\nThe layout, on four 2 by 2 channels (channel k holds 10 (k + 1) + 2 i + j at "
              "(i, j), so channel 0 is 10 11 / 12 13 and channel 1 is 20 21 / 22 23); the output channel, 4 by 4:\n\n" + _block([" ".join(f"{int(v):>2}" for v in row) for row in lay[0]]))


def convex_demo():
    """One row of low-resolution depth with a step, doubled by bilinear and by a convex combination of the 3 neighbours."""
    row = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    n = len(row)
    bil = [im.lerp_axis(row, o, 2 * n, "half") for o in range(2 * n)]
    # logits per output sub-pixel over the neighbours (left, centre, right), chosen to put the edge between the two output pixels of
    # pixel 2 and 3; the chosen vectors are printed
    logits = {}
    for i in range(n):
        for s in range(2):
            logits[(i, s)] = (0.0, 0.0, 0.0)
    for s in range(2):
        logits[(2, s)] = (0.0, 0.0, -4.0)  # pixel 2 sits left of the step: do not look right
        logits[(3, s)] = (-4.0, 0.0, 0.0)  # pixel 3 sits right of it: do not look left
    out, wts = [], []
    for i in range(n):
        for s in range(2):
            z = logits[(i, s)]
            e = [math.exp(v) for v in z]
            tot = sum(e)
            w3 = [v / tot for v in e]
            nb = [row[max(i - 1, 0)], row[i], row[min(i + 1, n - 1)]]
            out.append(sum(wi * v for wi, v in zip(w3, nb)))
            wts.append(w3)
    rows = []
    for o in range(2 * n):
        i, s = divmod(o, 2)
        rows.append((o, f"({i}, {s})", " ".join(f"{v:.3f}" for v in wts[o]), _fx(bil[o], 3), _fx(out[o], 3)))
    return ("A row of six low-resolution depths [1, 1, 1, 5, 5, 5] doubled by bilinear interpolation (half-pixel rule) and by a convex "
            "combination of each pixel's left, own and right values (the 1-D form of the 3 by 3 rule). Output o is sub-pixel s of "
            "input pixel i; the logits of every sub-pixel are zero (weights 1/3 each) except pixel 2, whose two sub-pixels have the logits (0, 0, -4) "
            "(do not look right), and pixel 3, whose two have (-4, 0, 0) (do not look left). Weights are softmax(logits) with the library exp, "
            "replicate padding at the ends.\n\n"
            + _table(["output o", "(i, s)", "weights (left, own, right)", "bilinear", "convex"], rows, "lllrr")
            + f"\n\nEvery convex output lies in [{min(out):.3f}, {max(out):.3f}], the range of the input; the largest bilinear output is "
            f"{max(bil):.3f}.")


# ------------------------------------------------------------------------------------------------ skip connections and shapes

C_E1, C_E2, C_UP = 4, 8, 4  # channels of the first encoder stage, the bottleneck and the upsampled map

KINDS = ("unet", "noskip", "coarse", "pixel")


def seg_shapes(kind):
    """Rows (name, shape out, parameters, multiply-adds) of the tiny segmenter of each kind."""
    rows = []

    def conv(name, cin, cout, k, h, w):
        rows.append((name, f"[{cout}, {h}, {w}]", cout * cin * k * k + cout, cout * cin * k * k * h * w))

    if kind == "pixel":
        conv("1x1 conv, 1 to 3", 1, N_CLS, 1, 8, 8)
        return rows
    conv("conv 3x3 + ReLU", 1, C_E1, 3, 8, 8)
    rows.append(("max pool 2x2", f"[{C_E1}, 4, 4]", 0, 0))
    conv("conv 3x3 + ReLU", C_E1, C_E2, 3, 4, 4)
    if kind == "coarse":
        conv("1x1 conv head (4x4 logits)", C_E2, N_CLS, 1, 4, 4)
        rows.append(("nearest x2", f"[{N_CLS}, 8, 8]", 0, 0))
        return rows
    rows.append(("transposed conv k2 s2", f"[{C_UP}, 8, 8]", C_E2 * C_UP * 4 + C_UP, C_E2 * C_UP * 4 * 16))
    cat = C_UP + (C_E1 if kind == "unet" else 0)
    if kind == "unet":
        rows.append(("concat with first stage", f"[{cat}, 8, 8]", 0, 0))
    conv("conv 3x3 head", cat, N_CLS, 3, 8, 8)
    return rows


def seg_shape_table():
    out = []
    for kind in KINDS:
        rows = seg_shapes(kind)
        out.append((kind, sum(r[2] for r in rows), sum(r[3] for r in rows)))
    rows_u = seg_shapes("unet")
    rec = cv.receptive([(3, 1, 1), (2, 2, 1), (3, 1, 1)])
    return (_table(["layer (kind unet)", "output [C, H, W]", "parameters", "multiply-adds"], [(a, b, _int(c), _int(d)) for a, b, c, d in rows_u], "llrr")
            + "\n\n" + _table(["kind", "parameters", "multiply-adds per image"], [(k, _int(p), _int(m)) for k, p, m in out], "lrr")
            + f"\n\nReceptive field along one axis of one bottleneck value: after the first convolution {rec[0][0]}, after the pool {rec[1][0]}, "
            f"after the second convolution {rec[2][0]} input pixels (Volume VII, Chapter 2, recursion r = r + d (k - 1) j), so the "
            f"bottleneck of an 8 by 8 image sees all of it.")


# ------------------------------------------------------------------------------------------------ the segmentation data

SIGMA = 0.15
CLASS_MEAN = (0.80, 0.35, 0.65)  # mean intensity of sky, ground, box
DATA_NAME = "dense_prediction.seg_data"
INIT_PREFIX = "dense_prediction.seg_init."
N_TRAIN, N_TEST = 60, 60


def random_scene(gen):
    """One random scene: horizon row h0 = 2 + integer(3); box width bw = 2 + integer(2), height bh = 2 + integer(2), left column
    x1 = integer(8 - bw + 1), lower edge y2 = h0 + 1 + integer(8 - h0); then 64 values normal(), row-major, give the noise. Returns
    (image [[64 values]], labels [64 ints])."""
    h0 = 2 + gen.integer(3)
    bw = 2 + gen.integer(2)
    bh = 2 + gen.integer(2)
    x1 = gen.integer(W - bw + 1)
    y2 = h0 + 1 + gen.integer(H - h0)
    y1 = max(0, y2 - bh)
    lab = scene_labels(h0, (x1, y1, x1 + bw, y2))
    flat = [v for r in lab for v in r]
    img = [CLASS_MEAN[c] + SIGMA * gen.normal() for c in flat]
    return [img], flat


@lru_cache(maxsize=None)
def seg_data():
    gen = bookrng.generator(DATA_NAME)
    train = [random_scene(gen) for _ in range(N_TRAIN)]
    test = [random_scene(gen) for _ in range(N_TEST)]
    return train, test


def class_counts(data):
    cnt = [0] * N_CLS
    for _, y in data:
        for c in y:
            cnt[c] += 1
    return cnt


def class_weights(cnt):
    """Inverse frequency scaled so that equally frequent classes weigh 1: n_pixels / (n_classes * count_c)."""
    tot = sum(cnt)
    return [tot / (N_CLS * c) for c in cnt]


def seg_data_summary():
    train, test = seg_data()
    ctr, cte = class_counts(train), class_counts(test)
    cw = class_weights(ctr)
    rows = [(CLASS_NAMES[c], ctr[c], _fx(ctr[c] / sum(ctr), 4), cte[c], _fx(cte[c] / sum(cte), 4), _fx(cw[c], 4)) for c in range(N_CLS)]
    return (f"Generator `bookrng.generator(\"{DATA_NAME}\")`, {_seed(DATA_NAME)}. One stream makes the {N_TRAIN} training scenes and then "
            f"the {N_TEST} test scenes; a scene draws, in this order, the horizon row h0 = 2 + integer(3), the box width bw = 2 + integer(2) "
            f"and height bh = 2 + integer(2), the left column x1 = integer(8 - bw + 1), the lower edge y2 = h0 + 1 + integer(8 - h0) "
            f"(the top edge is max(0, y2 - bh)), and then 64 values normal() in row-major order. The pixel is "
            f"mean_c + {SIGMA} normal() with the class means sky {CLASS_MEAN[0]}, ground {CLASS_MEAN[1]}, box {CLASS_MEAN[2]}; the box "
            f"overwrites sky and ground. Class weights are n_pixels / (3 count_c) of the training pixels.\n\n"
            + _table(["class", "training pixels", "share", "test pixels", "share", "weight"], rows, "lrrrrr"))


# ------------------------------------------------------------------------------------------------ the segmenters


def _init_shapes(kind):
    """(name, rows, columns, gain) of every weight tensor in draw order; each is drawn row-major as `normal() * gain`."""
    if kind == "pixel":
        return [("Wp", N_CLS, 1, 1.0)]
    base = [("W1", C_E1, 9, math.sqrt(2.0 / 9)), ("W2", C_E2, C_E1 * 9, math.sqrt(2.0 / (C_E1 * 9)))]
    if kind == "coarse":
        return base + [("Wc", N_CLS, C_E2, math.sqrt(1.0 / C_E2))]
    cat = C_UP + (C_E1 if kind == "unet" else 0)
    return base + [("Wu", C_E2 * C_UP, 4, math.sqrt(2.0 / C_E2)), ("Wh", N_CLS, cat * 9, math.sqrt(1.0 / (cat * 9)))]


_BIAS = {"W1": ("b1", C_E1), "W2": ("b2", C_E2), "Wc": ("bc", N_CLS), "Wu": ("bu", C_UP), "Wh": ("bh", N_CLS), "Wp": ("bp", N_CLS)}


def seg_init(kind, nudge=0.0):
    """He-style weights (gain in `_init_shapes`) drawn from `bookrng.generator(INIT_PREFIX + kind)` in the order W1, W2, then Wu, Wh
    (unet, noskip) or Wc (coarse) or Wp (pixel); biases 0. `nudge` is added to the first entry of the first tensor."""
    gen = bookrng.generator(INIT_PREFIX + kind)
    p = {}
    for name, r, c, gain in _init_shapes(kind):
        p[name] = [[gen.normal() * gain for _ in range(c)] for _ in range(r)]
        b, n = _BIAS[name]
        p[b] = [0.0] * n
    first = _init_shapes(kind)[0][0]
    p[first][0][0] += nudge
    return p


def n_params(p):
    return sum(len(r) for r in _leaves(p))


def _leaves(p):
    out = []
    for name in sorted(p):
        v = p[name]
        out += v if isinstance(v[0], list) else [v]
    return out


GEO = {"e1": cnns.Geo(1, 8, 8), "e2": cnns.Geo(C_E1, 4, 4), "head_unet": cnns.Geo(C_UP + C_E1, 8, 8), "head_noskip": cnns.Geo(C_UP, 8, 8)}


def _softmax_pixels(logits):
    """Per-pixel softmax of logits[c][p]: returns probs[c][p]."""
    n = len(logits[0])
    out = [[0.0] * n for _ in logits]
    for q in range(n):
        pr = cnns.softmax([logits[c][q] for c in range(len(logits))])
        for c, v in enumerate(pr):
            out[c][q] = v
    return out


def convt_fwd(a, wu, bu):
    """Transposed convolution with k = 2, stride 2 of a[ci][4*4] (row `ci * 4 + co` of wu holds the 2 by 2 kernel (a, b)): returns
    u[co][8*8]; sums over input channels in order."""
    cin, cout, n = len(a), len(bu), 4
    u = [[bu[co]] * 64 for co in range(cout)]
    for co in range(cout):
        uc = u[co]
        for ci in range(cin):
            ker, ac = wu[ci * cout + co], a[ci]
            for i in range(n):
                for j in range(n):
                    v = ac[i * n + j]
                    for ta in range(2):
                        for tb in range(2):
                            uc[(2 * i + ta) * 8 + 2 * j + tb] += v * ker[ta * 2 + tb]
    return u


def convt_bwd(a, wu, du):
    cin, cout, n = len(a), len(du), 4
    dwu = [[0.0] * 4 for _ in range(cin * cout)]
    da = [[0.0] * 16 for _ in range(cin)]
    dbu = [sum(du[co]) for co in range(cout)]
    for ci in range(cin):
        for co in range(cout):
            ker, dk, duc = wu[ci * cout + co], dwu[ci * cout + co], du[co]
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for ta in range(2):
                        for tb in range(2):
                            g = duc[(2 * i + ta) * 8 + 2 * j + tb]
                            dk[ta * 2 + tb] += a[ci][i * n + j] * g
                            s += ker[ta * 2 + tb] * g
                    da[ci][i * n + j] += s
    return dwu, dbu, da


def _unpool(dp, arg, z):
    """Route the pooled gradient dp[ch][q] to the winners arg[ch][q] of the pre-pool map and apply the ReLU slope of z."""
    d = [[0.0] * len(z[ch]) for ch in range(len(z))]
    for ch in range(len(z)):
        for q, qi in enumerate(arg[ch]):
            d[ch][qi] += dp[ch][q]
    return d


def seg_forward(p, kind, img):
    """Logits [3][64] and a cache for the backward pass."""
    if kind == "pixel":
        x = img[0]
        return [[p["Wp"][c][0] * v + p["bp"][c] for v in x] for c in range(N_CLS)], (x,)
    z1, col1 = cnns.conv_fwd(GEO["e1"], p["W1"], p["b1"], img)
    a1 = cnns.relu_map(z1)
    p1, arg = cnns.maxpool_fwd(a1, C_E1, 8, 8)
    z2, col2 = cnns.conv_fwd(GEO["e2"], p["W2"], p["b2"], p1)
    a2 = cnns.relu_map(z2)
    if kind == "coarse":
        lg4 = [[sum(map(mul, p["Wc"][c], [a2[ci][q] for ci in range(C_E2)])) + p["bc"][c] for q in range(16)] for c in range(N_CLS)]
        logits = [[lg4[c][(q // 8 // 2) * 4 + (q % 8) // 2] for q in range(64)] for c in range(N_CLS)]
        return logits, (z1, col1, arg, z2, col2, a2)
    u = convt_fwd(a2, p["Wu"], p["bu"])
    cat = u + a1 if kind == "unet" else u
    geo = GEO["head_unet" if kind == "unet" else "head_noskip"]
    logits, colh = cnns.conv_fwd(geo, p["Wh"], p["bh"], cat)
    return logits, (z1, col1, arg, z2, col2, a2, a1, u, colh)


def _zeros_like(p):
    return {k: [[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v) for k, v in p.items()}


def _add(g, name, dw, db=None, bname=None):
    for o in range(len(dw)):
        row = g[name][o]
        for t, v in enumerate(dw[o]):
            row[t] += v
    if db is not None:
        for o, v in enumerate(db):
            g[bname][o] += v


def seg_loss_grad(p, kind, data, cw, grad=True):
    """Weighted mean cross-entropy sum_n sum_p cw[y] (-log prob_y) / sum_n sum_p cw[y] over the images in index order, and its gradient."""
    norm = sum(cw[c] * n for c, n in enumerate(class_counts(data)))
    g = _zeros_like(p) if grad else None
    loss = 0.0
    for img, y in data:
        logits, cache = seg_forward(p, kind, img)
        pr = _softmax_pixels(logits)
        for q in range(64):
            loss += -cw[y[q]] * math.log(pr[y[q]][q])
        if not grad:
            continue
        dl = [[cw[y[q]] * (pr[c][q] - (1.0 if c == y[q] else 0.0)) / norm for q in range(64)] for c in range(N_CLS)]
        da1_skip = None
        if kind == "pixel":
            (x,) = cache
            _add(g, "Wp", [[sum(map(mul, dl[c], x))] for c in range(N_CLS)], [sum(dl[c]) for c in range(N_CLS)], "bp")
            continue
        if kind == "coarse":
            z1, col1, arg, z2, col2, a2 = cache
            dl4 = [[sum(dl[c][((q // 4) * 2 + r) * 8 + (q % 4) * 2 + s] for r in range(2) for s in range(2)) for q in range(16)] for c in range(N_CLS)]
            _add(g, "Wc", [[sum(dl4[c][q] * a2[ci][q] for q in range(16)) for ci in range(C_E2)] for c in range(N_CLS)], [sum(dl4[c]) for c in range(N_CLS)], "bc")
            da2 = [[sum(p["Wc"][c][ci] * dl4[c][q] for c in range(N_CLS)) for q in range(16)] for ci in range(C_E2)]
        else:
            z1, col1, arg, z2, col2, a2, a1, u, colh = cache
            geo = GEO["head_unet" if kind == "unet" else "head_noskip"]
            dwh, dbh, dcat = cnns.conv_bwd(geo, p["Wh"], colh, dl)
            _add(g, "Wh", dwh, dbh, "bh")
            du = dcat[:C_UP]
            dwu, dbu, da2 = convt_bwd(a2, p["Wu"], du)
            _add(g, "Wu", dwu, dbu, "bu")
            if kind == "unet":
                da1_skip = dcat[C_UP:]
        dz2 = [[da2[o][q] if z2[o][q] > 0 else 0.0 for q in range(16)] for o in range(C_E2)]
        dw2, db2, dp1 = cnns.conv_bwd(GEO["e2"], p["W2"], col2, dz2)
        _add(g, "W2", dw2, db2, "b2")
        da1 = _unpool(dp1, arg, z1)
        if da1_skip is not None:
            for ch in range(C_E1):
                for q in range(64):
                    da1[ch][q] += da1_skip[ch][q]
        dz1 = [[da1[ch][q] if z1[ch][q] > 0 else 0.0 for q in range(64)] for ch in range(C_E1)]
        dw1, db1, _ = cnns.conv_bwd(GEO["e1"], p["W1"], col1, dz1, need_dx=False)
        _add(g, "W1", dw1, db1, "b1")
    return loss / norm, g


def seg_predict(p, kind, img):
    logits, _ = seg_forward(p, kind, img)
    return [max(range(N_CLS), key=lambda c: (logits[c][q], -c)) for q in range(64)]


def seg_confusion(p, kind, data):
    conf = [[0] * N_CLS for _ in range(N_CLS)]
    for img, y in data:
        for a, b in zip(y, seg_predict(p, kind, img)):
            conf[a][b] += 1
    return conf


def adam_run(p, loss_grad, steps, lr, record, b1=0.9, b2=0.999, eps=1e-8):
    """Adam with bias correction on the leaves of p in place, from zero moments; `loss_grad(p)` returns (loss, gradient with p's layout).
    Returns {t: loss after t updates} for t in `record`."""
    leaves = _leaves(p)
    m = [[0.0] * len(r) for r in leaves]
    v = [[0.0] * len(r) for r in leaves]
    seen = {}
    for t in range(steps + 1):
        loss, g = loss_grad(p)
        if t in record:
            seen[t] = loss
        if t == steps:
            break
        for r, gr, mr, vr in zip(leaves, _leaves(g), m, v):
            for i in range(len(r)):
                mr[i] = b1 * mr[i] + (1 - b1) * gr[i]
                vr[i] = b2 * vr[i] + (1 - b2) * gr[i] * gr[i]
                r[i] -= lr * (mr[i] / (1 - b1 ** (t + 1))) / (math.sqrt(vr[i] / (1 - b2 ** (t + 1))) + eps)
    return seen


SEG_STEPS, SEG_LR = 120, 0.03
SEG_RECORD = (0, 30, 60, 120)


def _seg_task(task):
    kind, weighted, nudge = task
    train, test = seg_data()
    cw = class_weights(class_counts(train)) if weighted else [1.0] * N_CLS
    p = seg_init(kind, nudge)
    seen = adam_run(p, lambda q: seg_loss_grad(q, kind, train, cw), SEG_STEPS, SEG_LR, SEG_RECORD)
    return {"seen": seen, "train": seg_confusion(p, kind, train), "test": seg_confusion(p, kind, test), "params": n_params(p)}


# (kind, weighted) of the five trained models, in table order
SEG_MODELS = (("pixel", False), ("coarse", False), ("noskip", False), ("unet", False), ("unet", True))


def _pool_map(fn, args):
    with multiprocessing.get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(fn, args)


@lru_cache(maxsize=None)
def seg_runs():
    nudges = (0.0, *stab.NUDGES)
    tasks = [(k, w, n) for k, w in SEG_MODELS for n in nudges]
    res = _pool_map(_seg_task, tasks)
    return {t: r for t, r in zip(tasks, res)}


def seg_run(kind, weighted, nudge=0.0):
    return seg_runs()[(kind, weighted, nudge)]


def _iou_floats(conf):
    return [None if v is None else float(v) for v in class_iou(conf)]


def _twin_text(f, kind, weighted, full):
    """Digits of f(run) that survive in the four nudged copies."""
    v = f(seg_run(kind, weighted, 0.0))
    ws = [f(seg_run(kind, weighted, n)) for n in stab.NUDGES]
    return stab.text(v, ws, full)


def seg_training():
    train, _ = seg_data()
    cw = class_weights(class_counts(train))
    rows = []
    for kind, weighted in SEG_MODELS:
        label = kind + (" + class weights" if weighted else "")
        first = lambda r: r["seen"][0]
        last = lambda r: r["seen"][SEG_STEPS]
        rows.append((label, _int(seg_run(kind, weighted)["params"]),
                     _twin_text(first, kind, weighted, lambda x: _fx(x, 4)), _twin_text(last, kind, weighted, lambda x: _fx(x, 4))))
    return (f"Five segmenters trained on the {N_TRAIN} training scenes by full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, "
            f"learning rate {SEG_LR}, bias correction), {SEG_STEPS} updates, from the initializations `{INIT_PREFIX}<kind>` (kinds pixel, "
            f"coarse, noskip, unet; the last model reuses the unet initialization). Loss: the weighted mean cross-entropy of equation (1) of the chapter with "
            f"the weights of the table above for the last row and 1 for the others (a plain mean). Weighted loss at update 0 and "
            f"at update {SEG_STEPS} on the training scenes; sums run over images in index order, then pixels row-major, and over taps in "
            f"the order (input channel, kernel row, kernel column). " + stab.RULE + "\n\n"
            + _table(["model", "parameters", f"loss at 0", f"loss at {SEG_STEPS}"], rows, "lrrr")
            + f"\n\n(Class weights used by the last row: {', '.join(f'{CLASS_NAMES[c]} {cw[c]:.4f}' for c in range(N_CLS))}.)")


def seg_eval():
    rows = []
    for kind, weighted in SEG_MODELS:
        label = kind + (" + class weights" if weighted else "")
        ious = [(lambda c: (lambda r: _iou_floats(r["test"])[c]))(c) for c in range(N_CLS)]
        cells = [_twin_text(f, kind, weighted, lambda x: _fx(x, 4)) for f in ious]
        acc = lambda r: sum(r["test"][c][c] for c in range(N_CLS)) / sum(map(sum, r["test"]))
        miou = lambda r: float(mean_iou(class_iou(r["test"])))
        rows.append((label, _twin_text(acc, kind, weighted, lambda x: _fx(x, 4)), *cells, _twin_text(miou, kind, weighted, lambda x: _fx(x, 4))))
    ref = seg_run("unet", False)["test"]
    return (f"The same five models on the {N_TEST} held-out scenes ({sum(map(sum, ref))} pixels): the prediction of a pixel is the class "
            f"of the largest logit (the lowest class on a tie), the confusion matrices are summed over all test pixels and the IoU of "
            f"each class is TP / (TP + FP + FN) of the summed matrix (not an average of per-image IoUs). Mean IoU is the mean of "
            f"the three. " + stab.RULE + "\n\n"
            + _table(["model", "pixel accuracy", "IoU sky", "IoU ground", "IoU box", "mean IoU"], rows))


def seg_confusion_table():
    r = seg_run("unet", False)["test"]
    texts = []
    for i in range(N_CLS):
        row = []
        for j in range(N_CLS):
            f = (lambda i, j: (lambda run: float(run["test"][i][j])))(i, j)
            v = f(seg_run("unet", False))
            ws = [f(seg_run("unet", False, n)) for n in stab.NUDGES]
            row.append(stab.text(v, ws, lambda x: str(int(round(x)))))
        texts.append((f"{i} ({CLASS_NAMES[i]})", *row))
    return ("Confusion matrix of the unet model on the test scenes (rows: true class, columns: predicted class). " + stab.RULE
            + "\n\n" + _table(["true \\ predicted", "0 sky", "1 ground", "2 box"], texts))


def seg_gradient_check():
    train, _ = seg_data()
    cw = class_weights(class_counts(train))
    sub = train[:3]
    step = 1e-6
    out = []
    for kind in KINDS:
        p = seg_init(kind)
        _, g = seg_loss_grad(p, kind, sub, cw)
        worst, fails, total = 0.0, 0, 0
        for lp, lg in zip(_leaves(p), _leaves(g)):
            for t in range(len(lp)):
                old = lp[t]
                lp[t] = old + step
                lplus = seg_loss_grad(p, kind, sub, cw, grad=False)[0]
                lp[t] = old - step
                lminus = seg_loss_grad(p, kind, sub, cw, grad=False)[0]
                lp[t] = old
                num = (lplus - lminus) / (2 * step)
                err = abs(lg[t] - num)
                total += 1
                fails += err > 1e-7 + 1e-5 * max(abs(lg[t]), abs(num))
                worst = max(worst, err)
        out.append((kind, _int(total), "below 1e-7" if worst < 1e-7 else _sci(worst), f"{total - fails} of {total}"))
    return (f"The initialized models (`{INIT_PREFIX}<kind>`) on the first 3 training scenes with the class weights of the table: analytic "
            f"gradient of the weighted loss against the central difference (L(w + h) - L(w - h)) / (2h), h = {step:g}, in 64-bit "
            f"arithmetic, one parameter at a time, gate |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|).\n\n"
            + _table(["model", "parameters", "largest absolute difference", "pass the gate"], out, "lrrr"))


# ------------------------------------------------------------------------------------------------ depth: the pinhole and its ambiguity

CX, CY = 4.0, 3.0  # the principal point: column 4, and the horizon row 3
F_PX = 4.0


def project(X, Y, Z, f=F_PX):
    """Pinhole projection of a point (X right, Y down, Z ahead, all in metres) to pixel coordinates (x, y): the point at lateral
    offset X and depth Z lies at x = cx + f X / Z, and likewise for y."""
    return CX + f * X / Z, CY + f * Y / Z


SCENE_POINTS = (
    ("box, top left", -0.25, 0.25, 1.0),
    ("box, top right", 0.5, 0.25, 1.0),
    ("box, base right", 0.5, 1.0, 1.0),
    ("ground, centre of row 3", 0.0, 1.0, 8.0),
)


def pinhole_ambiguity():
    rows = []
    for name, X, Y, Z in SCENE_POINTS:
        a = project(X, Y, Z)
        b = project(2 * X, 2 * Y, 2 * Z)
        c = project(X * 3, Y * 3, Z * 3)
        rows.append((name, f"({X:g}, {Y:g}, {Z:g})", f"({2 * X:g}, {2 * Y:g}, {2 * Z:g})", f"({a[0]:g}, {a[1]:g})", f"({b[0]:g}, {b[1]:g})",
                     "yes" if a == b == c else "no"))
    return ("The scene of the first example seen by a camera with focal length f = 4 pixels and principal point (4, 3), which projects the "
            "point (X, Y, Z) (right, down and ahead of the camera, in metres) to x = 4 + 4 X / Z, y = 3 + 4 Y / Z. The camera is 1 metre "
            "above the ground, so the ground lies at Y = 1. Each point is shown in the scene as it is and in the scene made twice as "
            "large about the camera (every coordinate doubled, which doubles the camera height too). The last column also checks a "
            "factor of 3.\n\n"
            + _table(["point", "(X, Y, Z) at scale 1", "(X, Y, Z) at scale 2", "pixel at scale 1", "pixel at scale 2", "same pixel at scales 1, 2, 3"],
                     rows, "lllllr"))


def ground_plane():
    rows = []
    for y in range(3, 8):
        yc = y + 0.5
        z1, z2 = FH / (yc - HORIZON), 2 * FH / (yc - HORIZON)
        rows.append((y, _fx(yc, 1), _fx(z1, 4), _fx(1 / z1, 4), _fx(z2, 4), _fx(1 / z2, 4)))
    ys = [y + 0.5 for y in range(3, 8)]
    nus = [(yc - HORIZON) / FH for yc in ys]
    n = len(ys)
    a = (n * sum(u * v for u, v in zip(ys, nus)) - sum(ys) * sum(nus)) / (n * sum(u * u for u in ys) - sum(ys) ** 2)
    b = (sum(nus) - a * sum(ys)) / n
    worst = max(abs(a * u + b - v) for u, v in zip(ys, nus))
    return ("The ground rows of the first example, row y with centre y + 0.5, for a camera 1 metre and 2 metres above the ground "
            "(f = 4 pixels, horizon at row 3): Z = f h / (y + 0.5 - 3), and its reciprocal.\n\n"
            + _table(["row y", "y + 0.5", "Z (h = 1)", "1/Z (h = 1)", "Z (h = 2)", "1/Z (h = 2)"], rows)
            + f"\n\nA straight line fitted by least squares through the five points (y + 0.5, 1/Z) at h = 1 has slope {a:.4f} = 1/(f h) and "
            f"intercept {b:.4f} = -3/(f h); its largest residual is {'below 1e-12' if worst < 1e-12 else _sci(worst)}.")


# ------------------------------------------------------------------------------------------------ depth: alignment and metrics

PRED_NAME = "dense_prediction.depth_prediction"
PRED_SCALE, PRED_SHIFT, PRED_NOISE = 0.45, 0.2, 0.03


def valid_pixels():
    lab = scene_labels()
    nu = scene_inverse_depth(lab)
    return [(y, x, nu[y][x]) for y in range(H) for x in range(W) if nu[y][x] is not None]


def predicted_inverse_depth():
    """The 'network output' of the examples: 0.45 nu + 0.2 + 0.03 normal() on every non-sky pixel in row-major order (draw order),
    plus 0.1 on the box pixels, a structured error that no scale and shift can remove."""
    gen = bookrng.generator(PRED_NAME)
    lab = scene_labels()
    out = []
    for y, x, v in valid_pixels():
        extra = 0.1 if lab[y][x] == 2 else 0.0
        out.append(PRED_SCALE * v + PRED_SHIFT + PRED_NOISE * gen.normal() + extra)
    return out


def align_scale_shift(u, v):
    """Closed form of min over (a, b) of sum_i (a u_i + b - v_i)^2: a = (n Su,v - Su Sv) / (n Su,u - Su^2), b = (Sv - a Su) / n.
    Sums run in index order. Returns (a, b) and the five sums."""
    n = len(u)
    su, sv = sum(u), sum(v)
    suu = sum(a * a for a in u)
    suv = sum(a * b for a, b in zip(u, v))
    a = (n * suv - su * sv) / (n * suu - su * su)
    return a, (sv - a * su) / n, (n, su, sv, suu, suv)


def align_scale(u, v):
    """Closed form of min over a of sum_i (a u_i - v_i)^2: a = sum u v / sum u u."""
    return sum(a * b for a, b in zip(u, v)) / sum(a * a for a in u)


def sse(u, v, a, b=0.0):
    return sum((a * x + b - y) ** 2 for x, y in zip(u, v))


def alignment():
    nu = [v for _, _, v in valid_pixels()]
    pred = predicted_inverse_depth()
    a, b, (n, su, sv, suu, suv) = align_scale_shift(pred, nu)
    a0 = align_scale(pred, nu)
    best = sse(pred, nu, a, b)
    rows = [("none (a = 1, b = 0)", "1", "0", _fx(sse(pred, nu, 1.0, 0.0), 4)),
            ("scale only (a = sum uv / sum uu)", _fx(a0, 6), "0", _fx(sse(pred, nu, a0), 4)),
            ("scale and shift (closed form)", _fx(a, 6), _fx(b, 6), _fx(best, 4))]
    probes = []
    for da, db in ((0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01), (0.01, 0.01)):
        probes.append((f"({a + da:.6f}, {b + db:.6f})", _fx(sse(pred, nu, a + da, b + db), 6), _fx(sse(pred, nu, a + da, b + db) - best, 6)))
    return (f"Generator `bookrng.generator(\"{PRED_NAME}\")`, {_seed(PRED_NAME)}. The predicted inverse depth u of the {n} non-sky pixels of the "
            f"scene, in row-major order, is {PRED_SCALE} v + {PRED_SHIFT} + {PRED_NOISE} normal() (one draw per pixel in that order) "
            f"plus 0.1 on the 9 box pixels, where v is the true inverse depth. Sums over the pixels in that order: n = {n}, sum u = {su:.6f}, "
            f"sum v = {sv:.6f}, sum u u = {suu:.6f}, sum u v = {suv:.6f}.\n\n"
            + _table(["alignment", "a", "b", "sum of squared residuals"], rows, "lrrr")
            + "\n\nMoving away from the closed-form pair in any direction raises the sum, by exactly the sum over the pixels of (da u + db)^2 for the move (da, db):\n\n"
            + _table(["(a, b)", "sum of squared residuals", "increase over the minimum"], probes))


def depth_metrics(z_hat, z):
    """AbsRel, RMSE and the shares of pixels whose ratio max(z_hat/z, z/z_hat) is below 1.25, 1.25^2, 1.25^3."""
    n = len(z)
    absrel = sum(abs(p - t) / t for p, t in zip(z_hat, z)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(z_hat, z)) / n)
    ratio = [max(p / t, t / p) for p, t in zip(z_hat, z)]
    return absrel, rmse, [sum(r < 1.25 ** k for r in ratio) / n for k in (1, 2, 3)]


def metrics_table():
    nu = [v for _, _, v in valid_pixels()]
    z = [1.0 / v for v in nu]
    pred = predicted_inverse_depth()
    a, b, _ = align_scale_shift(pred, nu)
    a0 = align_scale(pred, nu)
    rows = []
    for name, f in (("none", lambda u: u), ("scale only", lambda u: a0 * u), ("scale and shift", lambda u: a * u + b)):
        aligned = [f(u) for u in pred]
        assert min(aligned) > 0
        absrel, rmse, acc = depth_metrics([1.0 / t for t in aligned], z)
        rows.append((name, _fx(absrel, 4), _fx(rmse, 4), *[_fx(v, 4) for v in acc]))
    return (f"The predictions of the alignment example turned into depths Z-hat = 1 / (aligned inverse depth) (every aligned value is positive) "
            f"and scored on the {len(z)} non-sky pixels against the true depths Z: AbsRel = mean |Z-hat - Z| / Z, RMSE = sqrt(mean (Z-hat - Z)^2) "
            f"in metres, and the share of pixels with max(Z-hat / Z, Z / Z-hat) below 1.25, 1.25^2 = 1.5625 and 1.25^3 = 1.953125.\n\n"
            + _table(["alignment", "AbsRel", "RMSE (m)", "share < 1.25", "share < 1.25^2", "share < 1.25^3"], rows))


def relative_vs_metric():
    nu = [v for _, _, v in valid_pixels()]
    pred = predicted_inverse_depth()
    a, b, _ = align_scale_shift(pred, nu)
    lo, hi = min(pred), max(pred)
    norm = [(u - lo) / (hi - lo) for u in pred]
    an, bn, _ = align_scale_shift(norm, nu)
    s_raw, s_norm = sse(pred, nu, a, b), sse(norm, nu, an, bn)
    # two reference pixels with known metric depth: the nearest pixel and the farthest
    px = valid_pixels()
    i_far = min(range(len(nu)), key=lambda i: nu[i])
    i_near = max(range(len(nu)), key=lambda i: nu[i])
    at = lambda i: f"row {px[i][0]}, column {px[i][1]}"
    a2 = (nu[i_near] - nu[i_far]) / (pred[i_near] - pred[i_far])
    b2 = nu[i_far] - a2 * pred[i_far]
    z = [1.0 / v for v in nu]
    full = depth_metrics([1.0 / (a * u + b) for u in pred], z)
    two = depth_metrics([1.0 / (a2 * u + b2) for u in pred], z)
    rows = [("least squares over all 40 pixels", _fx(a, 6), _fx(b, 6), _fx(full[0], 4), _fx(full[1], 4)),
            ("two reference pixels (nearest, farthest)", _fx(a2, 6), _fx(b2, 6), _fx(two[0], 4), _fx(two[1], 4))]
    return (f"The same predictions after the min-max normalization that brain's depth action returns (u' = (u - min u) / (max u - min u), "
            f"min {lo:.6f}, max {hi:.6f}), aligned to the truth with the closed form: the sum of squared residuals is {s_norm:.6f} for "
            f"the normalized map and {s_raw:.6f} for the raw one, so the normalization (an affine map) changes nothing that a scale and a "
            f"shift can undo (difference {'below 1e-9' if abs(s_norm - s_raw) < 1e-9 else _sci(abs(s_norm - s_raw))}). Metric depth from "
            f"a relative map needs two numbers; here they come from the least-squares fit, or from the true inverse depths at the nearest "
            f"pixel ({at(i_near)}) and the farthest ({at(i_far)}) of the 40 (the first of each in row-major order):\n\n"
            + _table(["how a and b are found", "a", "b", "AbsRel", "RMSE (m)"], rows))


# ------------------------------------------------------------------------------------------------ folding branches into one convolution


def reparam_demo():
    name = "dense_prediction.reparam"
    gen = bookrng.generator(name)
    c, h, w, eps = 2, 4, 4, 1e-5
    x = [[[gen.normal() for _ in range(w)] for _ in range(h)] for _ in range(c)]
    w3 = [[[[gen.normal() for _ in range(3)] for _ in range(3)] for _ in range(c)] for _ in range(c)]
    w1 = [[gen.normal() for _ in range(c)] for _ in range(c)]
    bn3 = [[gen.normal() for _ in range(c)] for _ in range(2)] + [[gen.normal() for _ in range(c)], [0.5 + gen.uniform() for _ in range(c)]]
    bn1 = [[gen.normal() for _ in range(c)] for _ in range(2)] + [[gen.normal() for _ in range(c)], [0.5 + gen.uniform() for _ in range(c)]]

    def conv(x, wt, k):
        pad = k // 2
        out = [[[0.0] * w for _ in range(h)] for _ in range(c)]
        for o in range(c):
            for i in range(h):
                for j in range(w):
                    acc = 0.0
                    for ci in range(c):
                        for a in range(k):
                            for b in range(k):
                                yy, xx = i + a - pad, j + b - pad
                                if 0 <= yy < h and 0 <= xx < w:
                                    acc += x[ci][yy][xx] * (wt[o][ci][a][b] if k == 3 else wt[o][ci])
                    out[o][i][j] = acc
        return out

    def bn(y, p):
        gamma, beta, mean, var = p
        return [[[gamma[o] * (y[o][i][j] - mean[o]) / math.sqrt(var[o] + eps) + beta[o] for j in range(w)] for i in range(h)] for o in range(c)]

    relu = lambda t: [[[max(v, 0.0) for v in r] for r in ch] for ch in t]
    a3, a1 = bn(conv(x, w3, 3), bn3), bn(conv(x, w1, 1), bn1)
    branches = relu([[[a3[o][i][j] + a1[o][i][j] + x[o][i][j] for j in range(w)] for i in range(h)] for o in range(c)])
    # fold: kernel' = w * gamma / sqrt(var + eps), bias' = beta - gamma mean / sqrt(var + eps); 1x1 at the centre of the 3x3; identity adds 1
    k = [[[[w3[o][ci][a][b] * bn3[0][o] / math.sqrt(bn3[3][o] + eps) for b in range(3)] for a in range(3)] for ci in range(c)] for o in range(c)]
    bias = [bn3[1][o] - bn3[0][o] * bn3[2][o] / math.sqrt(bn3[3][o] + eps) + bn1[1][o] - bn1[0][o] * bn1[2][o] / math.sqrt(bn1[3][o] + eps) for o in range(c)]
    for o in range(c):
        for ci in range(c):
            k[o][ci][1][1] += w1[o][ci] * bn1[0][o] / math.sqrt(bn1[3][o] + eps) + (1.0 if o == ci else 0.0)
    y = conv(x, k, 3)
    fused = relu([[[y[o][i][j] + bias[o] for j in range(w)] for i in range(h)] for o in range(c)])
    diff = max(abs(branches[o][i][j] - fused[o][i][j]) for o in range(c) for i in range(h) for j in range(w))
    before = c * c * 9 + c * c + 2 * (2 * c)
    after = c * c * 9 + c
    return (f"Generator `bookrng.generator(\"{name}\")`, {_seed(name)}. Draws in order: x of shape [2, 4, 4] (channel, row, column), the 3 by 3 "
            f"weight w3 of shape [2, 2, 3, 3] (output, input, row, column), the 1 by 1 weight w1 of shape [2, 2], then for the 3 by 3 branch "
            f"and then the 1 by 1 branch the BatchNorm values gamma (2 values), beta (2), running mean (2), all normal(), and running "
            f"variance 0.5 + uniform() (2); epsilon = 1e-5. Left: relu(BN(w3 * x) + BN(w1 * x) + x) with zero padding, from the three "
            f"branches. Right: one 3 by 3 convolution whose kernel is the scaled w3, plus the scaled w1 at its centre tap, plus 1 at the "
            f"centre tap of the diagonal (o = ci), and whose bias is the two folded BatchNorm shifts (the identity adds none); then ReLU.\n\n"
            + _table(["quantity", "criterion", "result"], [("largest difference of the two outputs", "below 1e-12", "met" if diff < 1e-12 else f"NOT met ({_sci(diff)})")])
            + f"\n\nParameters of the block with 2 channels: {before} before (weights of both convolutions and the scale and shift of both "
            f"BatchNorms, 2 channels each), {after} after (one 3 by 3 kernel and a bias per output channel).")


# ------------------------------------------------------------------------------------------------ counts of the two case studies


def pick_groups(cin, cout, max_g=4):
    """The first of max_g, 2, 1 that divides both channel counts (ZipDepth's grouped projections)."""
    for g in (max_g, 2, 1):
        if cin % g == 0 and cout % g == 0:
            return g
    return 1


def _conv_cost(cin, cout, k, hw, groups=1, bias=False, bn=False):
    """(parameters before folding, parameters after folding, multiply-adds) of a convolution, and BatchNorm (scale, shift) when `bn`."""
    wgt = cout * (cin // groups) * k * k
    before = wgt + (cout if bias else 0) + (2 * cout if bn else 0)
    after = wgt + (cout if (bias or bn) else 0)
    return before, after, wgt * hw


def _qarep(cin, cout, hw):
    """A block of a 3 by 3 and a 1 by 1 convolution, each with BatchNorm, and an identity when the shapes agree (stride 1)."""
    b3, _, _ = _conv_cost(cin, cout, 3, hw, bn=True)
    b1, _, _ = _conv_cost(cin, cout, 1, hw, bn=True)
    return b3 + b1, 9 * cin * cout + cout, 9 * cin * cout * hw


def _bn_only(c):
    return 2 * c, 2 * c, 0


def _sum(parts):
    return tuple(sum(p[i] for p in parts) for i in range(3))


def zipdepth_components(side=384):
    """(name, side of the output, channels, parts) for the base ZipDepth at a square input of `side`; the stages of the layer list."""
    d, dep, half = (48, 96, 192, 384), (2, 2, 6, 2), 24
    s2, s4, s8, s16, s32 = side // 2, side // 4, side // 8, side // 16, side // 32
    q = lambda n: n * n
    comp = []
    comp.append(("stem (two stride-2 3x3 convolutions)", s4, d[0],
                 [_conv_cost(3, half, 3, q(s2), bn=True), _conv_cost(half, d[0], 3, q(s4), bn=True)]))
    comp.append(("stage 1: 2 blocks of 48", s4, d[0], [_qarep(d[0], d[0], q(s4)) for _ in range(dep[0])]))
    comp.append(("down 2: block, 48 to 96, stride 2", s8, d[1], [_qarep(d[0], d[1], q(s8))]))
    comp.append(("stage 2: 2 blocks of 96, multi-scale, strip gate", s8, d[1],
                 [_qarep(d[1], d[1], q(s8)) for _ in range(dep[1])]
                 + [_conv_cost(d[1], d[1], 3, q(s8), groups=d[1]), _conv_cost(d[1], d[1], 3, q(s8), groups=d[1]), _bn_only(d[1]),
                    _conv_cost(d[1], d[1], 1, q(s8), groups=d[1]), _bn_only(d[1])]))
    hid, gh = d[2] // 8, d[2] // 4
    comp.append(("down 3: block, 96 to 192, stride 2", s16, d[2], [_qarep(d[1], d[2], q(s16))]))
    comp.append(("stage 3: 6 blocks of 192, channel gate, context block", s16, d[2],
                 [_qarep(d[2], d[2], q(s16)) for _ in range(dep[2])]
                 + [_conv_cost(d[2], hid, 1, 1), _conv_cost(hid, d[2], 1, 1),
                    _conv_cost(d[2], 1, 1, q(s16), bias=True), _conv_cost(d[2], gh, 1, 1, bias=True), _bn_only(gh), _conv_cost(gh, d[2], 1, 1, bias=True)]))
    comp.append(("down 4: block, 192 to 384, stride 2", s32, d[3], [_qarep(d[2], d[3], q(s32))]))
    comp.append(("stage 4: 2 blocks of 384", s32, d[3], [_qarep(d[3], d[3], q(s32)) for _ in range(dep[3])]))
    comp.append(("pyramid pooling (1x1, pools, 1x1)", s32, d[3], [_conv_cost(d[3], d[3] // 4, 1, q(s32), bn=True), _conv_cost(d[3], d[3], 1, q(s32), bn=True)]))
    gl, gh2 = pick_groups(d[3], d[2]), pick_groups(d[2], d[3])
    comp.append(("cross-scale exchange (two grouped 1x1)", s32, d[3],
                 [_conv_cost(d[3], d[2], 1, q(s32), groups=gl), _conv_cost(d[2], d[3], 1, q(s16), groups=gh2)]))
    ch4, ch3, ch2, ch1 = 288, 192, 144, 96
    comp.append(("decoder: project 384 to 288", s32, ch4, [_conv_cost(d[3], ch4, 1, q(s32), bn=True)]))

    def fuse(cin_hi, cin_lo, cout, side_hi):
        return [_conv_cost(cin_hi, cout, 1, q(side_hi), groups=pick_groups(cin_hi, cout)),
                _conv_cost(cin_lo, cout, 1, q(side_hi), groups=pick_groups(cin_lo, cout)), _bn_only(cout)]

    comp.append(("decoder: fuse 192 and 288 at 1/16", s16, ch3, fuse(d[2], ch4, ch3, s16)))
    comp.append(("decoder: fuse 96 and 192 at 1/8", s8, ch2, fuse(d[1], ch3, ch2, s8)))
    comp.append(("decoder: fuse 48 and 144 at 1/4", s4, ch1, fuse(d[0], ch2, ch1, s4)))
    comp.append(("decoder: fuse 24 and 96 at 1/2", s2, 32, fuse(half, ch1, 32, s2)))
    comp.append(("half-resolution depth head (3x3, biased)", s2, 1, [_conv_cost(32, 1, 3, q(s2), bias=True)]))
    gather = (0, 0, 9 * q(side))
    comp.append(("convex upsample (mask 3x3, 1x1 to 36, gather)", side, 1,
                 [_conv_cost(32, 8, 3, q(s2), bn=True), _conv_cost(8, 36, 1, q(s2), bias=True), gather]))
    return [(n, s, c, _sum(parts)) for n, s, c, parts in comp]


def zipdepth_counts():
    comps = zipdepth_components()
    rows = [(n, f"{c} x {s} x {s}", _int(b), _int(a), _int(m)) for n, s, c, (b, a, m) in comps]
    tb, ta, tm = (sum(c[3][i] for c in comps) for i in range(3))
    return ("The base ZipDepth (`dims` 48, 96, 192, 384, `depths` 2, 2, 6, 2, decoder width 96, half-resolution width 32, input 384 by 384) "
            "from its layer list. 'Before' counts the trainable parameters of the trained form: every convolution weight, bias, and the scale "
            "and shift of every BatchNorm (the running statistics and the two normalization constants are not parameters). 'After' is "
            "the form for inference, in which each branch block is one 3 by 3 convolution with a bias, each BatchNorm that follows a "
            "single convolution is folded into it, and a BatchNorm over a sum stays as a scale and a shift. Multiply-adds are those "
            "of the convolutions of the inference form at 384 by 384 (the gather of the convex upsample counts 9 per output pixel); pooling, "
            "softmax, ReLU and the element-wise sums are not counted.\n\n"
            + _table(["stage", "output [C, H, W]", "parameters before", "parameters after", "multiply-adds"], rows, "llrrr")
            + f"\n\nTotal: {_int(tb)} parameters before folding, {_int(ta)} after, {_int(tm)} multiply-adds ({_int(2 * tm)} FLOPs) per 384 by 384 image.")


def zipdepth_totals():
    comps = zipdepth_components()
    return tuple(sum(c[3][i] for c in comps) for i in range(3))


def bisenet_components(side=512):
    """(name, output side, channels, parts) of the face-parsing BiSeNet at a square input of `side`; every convolution is bias-free,
    and those followed by BatchNorm count its scale and shift."""
    s2, s4, s8, s16, s32 = side // 2, side // 4, side // 8, side // 16, side // 32
    q = lambda n: n * n
    comp = [("stem 7x7 stride 2, then max pool", s4, 64, [_conv_cost(3, 64, 7, q(s2), bn=True)])]

    def stage(name, cin, cout, hw_side, first_stride):
        parts = []
        for b in range(2):
            ci = cin if b == 0 else cout
            parts.append(_conv_cost(ci, cout, 3, q(hw_side), bn=True))
            parts.append(_conv_cost(cout, cout, 3, q(hw_side), bn=True))
            if b == 0 and (first_stride != 1 or cin != cout):
                parts.append(_conv_cost(cin, cout, 1, q(hw_side), bn=True))
        return (name, hw_side, cout, parts)

    comp += [stage("layer 1: 2 basic blocks of 64", 64, 64, s4, 1), stage("layer 2: 2 basic blocks of 128", 64, 128, s8, 2),
             stage("layer 3: 2 basic blocks of 256", 128, 256, s16, 2), stage("layer 4: 2 basic blocks of 512", 256, 512, s32, 2)]
    comp.append(("global context: pool, 1x1 to 128", 1, 128, [_conv_cost(512, 128, 1, 1, bn=True)]))
    comp.append(("refinement of layer 4 (3x3, gate)", s32, 128, [_conv_cost(512, 128, 3, q(s32), bn=True), _conv_cost(128, 128, 1, 1, bn=True)]))
    comp.append(("3x3 after doubling to 1/16", s16, 128, [_conv_cost(128, 128, 3, q(s16), bn=True)]))
    comp.append(("refinement of layer 3 (3x3, gate)", s16, 128, [_conv_cost(256, 128, 3, q(s16), bn=True), _conv_cost(128, 128, 1, 1, bn=True)]))
    comp.append(("3x3 after doubling to 1/8", s8, 128, [_conv_cost(128, 128, 3, q(s8), bn=True)]))
    comp.append(("fusion (1x1 on 256, gate 256 to 64 to 256)", s8, 256,
                 [_conv_cost(256, 256, 1, q(s8), bn=True), _conv_cost(256, 64, 1, 1), _conv_cost(64, 256, 1, 1)]))
    comp.append(("output head (3x3 to 256, 1x1 to 19)", s8, 19, [_conv_cost(256, 256, 3, q(s8), bn=True), _conv_cost(256, 19, 1, q(s8))]))
    return [(n, s, c, _sum(parts)) for n, s, c, parts in comp]


BISENET_STACK = ([(7, 2, 1), (3, 2, 1)] + [(3, 1, 1)] * 4 + [(3, 2, 1)] + [(3, 1, 1)] * 3 + [(3, 2, 1)] + [(3, 1, 1)] * 3
                 + [(3, 2, 1)] + [(3, 1, 1)] * 3)


def bisenet_counts():
    comps = bisenet_components()
    rows = [(n, f"{c} x {s} x {s}", _int(b), _int(m)) for n, s, c, (b, a, m) in comps]
    tb = sum(c[3][0] for c in comps)
    tm = sum(c[3][2] for c in comps)
    rec = cv.receptive(BISENET_STACK)
    # indices: layers 0-1 stem+pool; 2-5 layer 1; 6-9 layer 2; 10-13 layer 3; 14-17 layer 4
    r8, r16, r32 = rec[9], rec[13], rec[17]
    r4 = rec[5]
    return ("The face-parsing BiSeNet at 512 by 512 from its layer list (the parameters are the convolution weights and the scale and "
            "shift of the BatchNorm after each convolution; every convolution is bias-free, and the two auxiliary training heads are not part "
            "of the inference graph). Multiply-adds are those of the convolutions at 512 by 512; the gates' pooled 1 by 1 convolutions are "
            "counted at one position.\n\n"
            + _table(["stage", "output [C, H, W]", "parameters", "multiply-adds"], rows, "llrr")
            + f"\n\nTotal: {_int(tb)} parameters and {_int(tm)} multiply-adds ({_int(2 * tm)} FLOPs) per 512 by 512 image. Receptive field of one "
            f"value along one axis through the main path of the backbone (Volume VII, Chapter 2, recursion): {r4[0]} pixels at stride {r4[1]} "
            f"(layer 1), {r8[0]} at stride {r8[1]} (layer 2, the map that goes to the fusion unchanged), {r16[0]} at stride {r16[1]} "
            f"(layer 3) and {r32[0]} at stride {r32[1]} (layer 4); the global pool sees all 512.")


# ------------------------------------------------------------------------------------------------ upsampling labels or logits


def logit_boundary():
    """A two-class decision on one row of 8 pixels from 4 coarse values: the logit difference d = logit(1) - logit(0) is the straight
    line c - 5.2 of the pixel-centre coordinate c = p + 0.5, so the true class of pixel p is 1 exactly when c > 5.2 (p >= 5). The coarse
    map samples d at the centres 1, 3, 5, 7 of its four cells (2 pixels each)."""
    coarse = [c - 5.2 for c in (1.0, 3.0, 5.0, 7.0)]
    truth = [1 if p + 0.5 > 5.2 else 0 for p in range(8)]
    near_labels = [1 if coarse[p // 2] > 0 else 0 for p in range(8)]  # decide at the coarse grid, then repeat the labels
    d_bil = [im.lerp_axis(coarse, p, 8, "half") for p in range(8)]  # repeat nothing: interpolate the logit, then decide
    bil_labels = [1 if v > 0 else 0 for v in d_bil]
    rows = [(p, _fx(p + 0.5 - 5.2, 2), truth[p], near_labels[p], _fx(d_bil[p], 2), bil_labels[p]) for p in range(8)]
    wrong_n = sum(a != b for a, b in zip(truth, near_labels))
    wrong_b = sum(a != b for a, b in zip(truth, bil_labels))
    return ("One row of 8 pixels, pixel p centred at c = p + 0.5. The true logit difference d = logit(1) - logit(0) is c - 5.2, so the "
            "true class is 1 from pixel 5 on. The coarse map holds d at the four cell centres c = 1, 3, 5, 7: " + ", ".join(f"{v:.1f}" for v in coarse)
            + ". 'Label first' takes the sign of each coarse value and repeats it twice; 'logit first' interpolates the coarse values "
            "to the 8 pixel centres with the half-pixel rule (replicating the end values) and then takes the sign.\n\n"
            + _table(["pixel p", "true d", "true class", "label first", "interpolated d", "logit first"], rows)
            + f"\n\nLabel first gets {wrong_n} pixel wrong, logit first {wrong_b}.")
