# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Object Detection".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: detection.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* a 16 by 16 picture with two rectangles, their boxes in the three usual formats, and the intersection over union of pairs;
* the decode of one cell of an anchor-based head (sigmoid offsets, squared scale) and of one point of an anchor-free head;
* the distribution over bins of one box side: its expectation, the distribution focal loss against the two neighbouring bins, the
  gradient, and the Jacobian of the expectation checked by central differences;
* L1, IoU, GIoU, DIoU and CIoU on boxes that overlap, touch and do not touch;
* a task-aligned assignment of anchor points to the two rectangles and the loss of the result;
* greedy non-maximum suppression traced candidate by candidate, class-aware and class-agnostic, and what it costs;
* average precision and mean average precision on a set small enough to read;
* the letterbox and its inverse, and the box errors that a wrong convention makes.

There are no random numbers: every input is a stated rule or a printed list. Arithmetic is the language's 64-bit floating point; the
natural logarithm, the exponential and the arctangent are the correctly rounded library functions. Sums run in the order stated beside
each table (anchors in row-major order of the grid, boxes in the order listed, bins from 0 upwards).

Swedish Embedded AB implements solutions for object detectors whose boxes, losses and evaluation are checked against a hand
computation. If your team needs expertise in detection pipelines, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from fractions import Fraction

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(v, d=4):
    return f"{v:.{d}f}"


def _g(v):
    return f"{v:g}"


def _box(b):
    return "(" + ", ".join(_g(v) for v in b) + ")"


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ------------------------------------------------------------------------------------------------ boxes and IoU

SIZE = 16
# The two rectangles of the toy picture: class, name, box (x1, y1, x2, y2). A box covers the pixels x1 <= x < x2, y1 <= y < y2, so
# its corner coordinates lie on the pixel edges and its area is (x2 - x1) * (y2 - y1) with no "+ 1".
RECT_A = (0, "a", (2, 3, 10, 11))
RECT_B = (1, "b", (7, 6, 14, 14))


def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def inter_wh(a, b):
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0), max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def iou(a, b):
    w, h = inter_wh(a, b)
    inter = w * h
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_fraction(a, b):
    w, h = inter_wh(a, b)
    inter = w * h
    union = area(a) + area(b) - inter
    return inter, union, Fraction(inter, union)


def xyxy_to_xywh(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])


def xywh_to_xyxy(c):
    return (c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2)


def scene_pixels():
    img = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    for _, ch, b in (RECT_A, RECT_B):  # B is drawn second, so it covers A where they overlap
        for y in range(b[1], b[3]):
            for x in range(b[0], b[2]):
                img[y][x] = ch
    return img


def toy_scene():
    img = scene_pixels()
    pic = "\n".join(" ".join(r) for r in img)
    rows = []
    for c, name, b in (RECT_A, RECT_B):
        cx, cy, w, h = xyxy_to_xywh(b)
        norm = tuple(v / SIZE for v in (cx, cy, w, h))
        rows.append([name, c, _box(b), _box((cx, cy, w, h)), "(" + ", ".join(_g(v) for v in norm) + ")", _g(area(b))])
    tab = _table(["rectangle", "class", "xyxy", "xywh (pixels)", "xywh / 16", "area"], rows, "lrllll")
    w, h = inter_wh(RECT_A[2], RECT_B[2])
    i, u, fr = iou_fraction(RECT_A[2], RECT_B[2])
    return (f"A 16 by 16 picture; row 0 is the top. Rectangle a is drawn first, rectangle b second, so b covers the {w} by {h} pixels "
            f"they share. A box (x1, y1, x2, y2) covers the pixels x1 <= x < x2 and y1 <= y < y2.\n\n```text\n{pic}\n```\n\n{tab}\n\n"
            f"The two boxes share {w} * {h} = {i} pixels and together cover {area(RECT_A[2])} + {area(RECT_B[2])} - {i} = {u}, so "
            f"IoU(a, b) = {i}/{u} = {fr}.")


IOU_PREDS = [
    ("the box itself", (2, 3, 10, 11)),
    ("shifted by (1, 1)", (3, 4, 11, 12)),
    ("shifted by (2, 2)", (4, 5, 12, 13)),
    ("its top half", (2, 3, 10, 7)),
    ("the whole picture", (0, 0, 16, 16)),
    ("the box of b", (7, 6, 14, 14)),
    ("to the right, apart", (11, 3, 19, 11)),
]


def iou_pairs():
    g = RECT_A[2]
    rows = []
    for name, p in IOU_PREDS:
        w, h = inter_wh(g, p)
        inter, union, fr = iou_fraction(g, p)
        rows.append([name, _box(p), f"{w} x {h}", inter, union, str(fr), _f(float(fr))])
    return ("Each box against the box of rectangle a, (2, 3, 10, 11), which has area 64. The overlap is the rectangle whose width is "
            "min(x2) - max(x1) and whose height is min(y2) - max(y1), each floored at 0.\n\n" +
            _table(["predicted box", "(x1, y1, x2, y2)", "overlap w x h", "intersection", "union", "IoU", "decimal"], rows, "llrrrrr"))


# ------------------------------------------------------------------------------------------------ head sizes

COCO_ANCHORS = {8: ((10, 13), (16, 30), (33, 23)), 16: ((30, 61), (62, 45), (59, 119)), 32: ((116, 90), (156, 198), (373, 326))}


def head_counts():
    side, nc, reg_max = 640, 80, 16
    rows = []
    cells = 0
    for s in (8, 16, 32):
        g = side // s
        cells += g * g
        rows.append([s, f"{g} x {g}", g * g, 3 * g * g, (reg_max - 1) * s])
    t = _table(["stride", "grid", "cells", "anchor-based candidates (3 per cell)", "largest distance a side can have, pixels"], rows)
    v5_ch, v8_ch = 3 * (5 + nc), nc + 4 * reg_max
    n5, n8 = 3 * cells, cells
    lines = [
        f"Input {side} by {side}, {nc} classes, n_bin = {reg_max} bins per side (`reg_max` in the code). The anchor-based head has 3 anchors per cell, each with "
        f"x, y, w, h, objectness and the {nc} class scores; the anchor-free head has {nc} class scores and {reg_max} bin logits for each of the four sides. "
        "The last column is (n_bin - 1) times the stride, the farthest a side can reach when the distribution is a single bin 15 (n_bin - 1 = 15).",
        "",
        t,
        "",
        _table(["quantity", "anchor-based", "anchor-free"],
               [["outputs per cell", f"3 x (5 + {nc}) = {v5_ch}", f"{nc} + 4 x {reg_max} = {v8_ch}"],
                ["cells over the three levels", cells, cells],
                ["boxes that can be proposed", f"{n5:,}", f"{n8:,}"],
                ["numbers written per image", f"{n5 * (5 + nc):,}", f"{n8 * v8_ch:,}"]], "lrr"),
        "",
        f"Taking the best class of an anchor-free cell on the raw logits needs one sigmoid per cell, {n8:,}, where one sigmoid per class would "
        f"need {n8:,} x {nc} = {n8 * nc:,} exponentials; an anchor-based decoder that reads the objectness of every candidate first "
        f"computes {n5:,} sigmoids and evaluates the other {5 + nc - 1} fields only for the candidates that survive, against "
        f"{n5 * (5 + nc):,} if it computes them all.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ anchor-based decode

V5_CELL = dict(stride=16, gx=3, gy=2, anchor=1, logits=(0.4, -1.1, 0.3, -0.2), obj=1.5, cls=(-0.5, 2.0, 0.1))


def v5_decode(stride, gx, gy, anchor, xy_wh_sig, obj, cls):
    """YOLOv5 v6 decode of one anchor at one cell; xy_wh_sig are the four sigmoid outputs, obj and cls the sigmoid objectness and class scores."""
    aw, ah = COCO_ANCHORS[stride][anchor]
    sx, sy, sw, sh = xy_wh_sig
    cx = (2 * sx - 0.5 + gx) * stride
    cy = (2 * sy - 0.5 + gy) * stride
    w = (2 * sw) ** 2 * aw
    h = (2 * sh) ** 2 * ah
    scores = [obj * c for c in cls]
    k = max(range(len(scores)), key=lambda i: (scores[i], -i))  # the first largest wins a tie
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), scores[k], k, (cx, cy, w, h)


def anchor_decode():
    c = V5_CELL
    s = [sigmoid(z) for z in c["logits"]]
    obj = sigmoid(c["obj"])
    cls = [sigmoid(z) for z in c["cls"]]
    aw, ah = COCO_ANCHORS[c["stride"]][c["anchor"]]
    box, score, k, (cx, cy, w, h) = v5_decode(c["stride"], c["gx"], c["gy"], c["anchor"], s, obj, cls)
    rows = [
        ["x", _f(c["logits"][0], 1), _f(s[0]), f"2 * {_f(s[0])} - 0.5 + {c['gx']} = {_f(2 * s[0] - 0.5 + c['gx'])}", f"x {c['stride']} = {_f(cx)}"],
        ["y", _f(c["logits"][1], 1), _f(s[1]), f"2 * {_f(s[1])} - 0.5 + {c['gy']} = {_f(2 * s[1] - 0.5 + c['gy'])}", f"x {c['stride']} = {_f(cy)}"],
        ["w", _f(c["logits"][2], 1), _f(s[2]), f"(2 * {_f(s[2])})^2 = {_f((2 * s[2]) ** 2)}", f"x {aw} = {_f(w)}"],
        ["h", _f(c["logits"][3], 1), _f(s[3]), f"(2 * {_f(s[3])})^2 = {_f((2 * s[3]) ** 2)}", f"x {ah} = {_f(h)}"],
    ]
    t = _table(["field", "logit", "sigmoid", "offset or scale", "pixels"], rows, "lrrll")
    cls_rows = [[f"class {i}", _f(z, 1), _f(cls[i]), _f(obj * cls[i])] for i, z in enumerate(c["cls"])]
    t2 = _table(["", "logit", "sigmoid", "objectness x class"], cls_rows, "lrrr")
    return (f"One cell of the stride-{c['stride']} level of a 128 by 128 input (a 8 x 8 grid), column g_x = {c['gx']}, row g_y = {c['gy']}, "
            f"anchor {c['anchor']} of that level, which is {aw} x {ah} pixels. The head wrote the logits below; the objectness logit is "
            f"{c['obj']} and sigmoid({c['obj']}) = {_f(obj)}.\n\n{t}\n\n{t2}\n\n"
            f"The best class is {k} with score {_f(score)}. The box is (x1, y1, x2, y2) = ({_f(box[0])}, {_f(box[1])}, {_f(box[2])}, {_f(box[3])}).")


def decode_range():
    rows = []
    for s in (0.01, 0.25, 0.5, 0.75, 0.99):
        rows.append([_g(s), _f(2 * s - 0.5, 3), _f((2 * s) ** 2, 4), _f(-math.log(1 / s - 1), 3).replace('-0.000', '0.000')])
    return ("The sigmoid output o of a coordinate and what the decode makes of it: the centre moves by 2o - 0.5 cells from the cell's "
            "corner, the side is (2o)^2 times the anchor. The last column is the logit that gives that o.\n\n" +
            _table(["o", "centre offset 2o - 0.5, cells", "side factor (2o)^2", "logit"], rows))


# ------------------------------------------------------------------------------------------------ anchor-free decode

V8_POINT = dict(stride=8, gx=5, gy=3, dist=(2.1, 1.6, 3.4, 2.9), logit=1.2)


def dist_to_xyxy(d, ax, ay, stride):
    l, t, r, b = d
    return ((ax - l) * stride, (ay - t) * stride, (ax + r) * stride, (ay + b) * stride)


def anchor_free_decode():
    p = V8_POINT
    ax, ay = p["gx"] + 0.5, p["gy"] + 0.5
    box = dist_to_xyxy(p["dist"], ax, ay, p["stride"])
    names = ["left", "top", "right", "bottom"]
    cols = [(ax - p["dist"][0]), (ay - p["dist"][1]), (ax + p["dist"][2]), (ay + p["dist"][3])]
    rows = [[names[i], _f(p["dist"][i], 1), _f(cols[i], 2), _f(cols[i] * p["stride"], 2)] for i in range(4)]
    return (f"One point of the stride-{p['stride']} level of a 128 by 128 input (a 16 x 16 grid), column {p['gx']}, row {p['gy']}. Its anchor "
            f"point is ({ax}, {ay}) in cell units and ({_g(ax * p['stride'])}, {_g(ay * p['stride'])}) in pixels. The four distances the head "
            f"reports, in cell units, give the box edges below.\n\n" +
            _table(["side", "distance d, cells", "edge, cells", "edge, pixels"], rows, "lrrr") +
            f"\n\nThe box is {_box(tuple(round(v, 2) for v in box))}, {_g(round(box[2] - box[0], 2))} wide and {_g(round(box[3] - box[1], 2))} high. "
            f"The class score of the point is sigmoid({p['logit']}) = {_f(sigmoid(p['logit']))}.")


# ------------------------------------------------------------------------------------------------ distribution focal loss

NBIN = 8
TARGET_D = 3.25


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def expectation(q):
    return sum(k * qk for k, qk in enumerate(q))  # bins from 0 upwards


def two_hot(t, n=NBIN):
    k = math.floor(t)
    tgt = [0.0] * n
    tgt[k] += k + 1 - t
    tgt[min(k + 1, n - 1)] += t - k
    return tgt


def dfl(q, t):
    k = math.floor(t)
    return -((k + 1 - t) * math.log(q[k]) + (t - k) * math.log(q[k + 1]))


def dfl_cases():
    """The five distributions of the chapter, by name: (name, bin probabilities)."""
    uniform = softmax([0.0] * NBIN)
    narrow = softmax([-(k - TARGET_D) ** 2 / (2 * 0.5 ** 2) for k in range(NBIN)])
    wide = softmax([-(k - TARGET_D) ** 2 / (2 * 1.5 ** 2) for k in range(NBIN)])
    spike = softmax([8.0 if k == 3 else 0.0 for k in range(NBIN)])
    exact = two_hot(TARGET_D)
    return [("uniform", uniform), ("narrow bump (sigma 0.5)", narrow), ("wide bump (sigma 1.5)", wide), ("spike on bin 3", spike),
            ("two-hot of 3.25", exact)]


def dfl_data():
    return [(name, q, expectation(q), dfl(q, TARGET_D)) for name, q in dfl_cases()]


def dfl_table():
    cases = dfl_data()
    rows = [[name] + [_f(v, 3) for v in q] for name, q, _, _ in cases]
    t1 = _table(["distribution"] + [f"q{k}" for k in range(NBIN)], rows, "l" + "r" * NBIN)
    t = TARGET_D
    rows2 = []
    for name, q, e, l in cases:
        rows2.append([name, _f(e), _f(abs(e - t)), _f(l)])
    t2 = _table(["distribution", "expectation E", "L1 error abs(E - 3.25)", "DFL loss"], rows2, "lrrr")
    ent = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    u = cases[0][1]
    g = [u[k] - two_hot(t)[k] for k in range(NBIN)]
    return (f"One side of one box with n_bin = {NBIN} bins (numbered 0 to {NBIN - 1}) and a true distance d* = {t} cells, so k = 3 and the "
            f"target puts 0.75 on bin 3 and 0.25 on bin 4. The bumps are softmax(-(k - 3.25)^2 / (2 sigma^2)) over the eight bins; the spike is "
            f"softmax of a logit 8 on bin 3 and 0 elsewhere.\n\n{t1}\n\n{t2}\n\n"
            f"The loss of the two-hot distribution is the entropy of the target, -(0.75 ln 0.75 + 0.25 ln 0.25) = {_f(ent)}, the least any "
            f"distribution can reach. For the uniform distribution the gradient q_k - target_k on the eight logits is "
            f"({', '.join(_f(v, 3) for v in g)}), which sums to {_f(sum(g), 3)}.")


def dfl_jacobian_check():
    """Largest |central difference of E with respect to logit j - q_j (j - E)| over the eight logits of the wide bump, step 1e-5."""
    z = [-(k - TARGET_D) ** 2 / (2 * 1.5 ** 2) for k in range(NBIN)]
    q = softmax(z)
    e = expectation(q)
    worst = 0.0
    h = 1e-5
    for j in range(NBIN):
        zp, zm = list(z), list(z)
        zp[j] += h
        zm[j] -= h
        fd = (expectation(softmax(zp)) - expectation(softmax(zm))) / (2 * h)
        worst = max(worst, abs(fd - q[j] * (j - e)))
    return worst


def dfl_jacobian():
    worst = dfl_jacobian_check()
    return (f"The logits are the wide bump of the table above. The largest difference over the eight logits between the central difference "
            f"(step 1e-5) of E and q_j (j - E) is below 1e-8 (criterion): {'passed' if worst < 1e-8 else 'FAILED'}.")


def dfl_bars_data():
    """The two distributions of the figure: the two-hot optimum and the wide bump, with their expectations and losses."""
    d = {name: (q, e, l) for name, q, e, l in dfl_data()}
    return d["two-hot of 3.25"], d["wide bump (sigma 1.5)"], d["spike on bin 3"]


# ------------------------------------------------------------------------------------------------ box regression losses


def enclosing(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def giou(p, g):
    inter = inter_wh(p, g)
    i = inter[0] * inter[1]
    u = area(p) + area(g) - i
    c = area(enclosing(p, g))
    return i / u - (c - u) / c


def diou(p, g):
    cp = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
    cg = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
    e = enclosing(p, g)
    return iou(p, g) - ((cp[0] - cg[0]) ** 2 + (cp[1] - cg[1]) ** 2) / ((e[2] - e[0]) ** 2 + (e[3] - e[1]) ** 2)


def ciou(p, g):
    v = 4 / math.pi ** 2 * (math.atan2(g[2] - g[0], g[3] - g[1]) - math.atan2(p[2] - p[0], p[3] - p[1])) ** 2
    i = iou(p, g)
    alpha = v / ((1 - i) + v) if (1 - i) + v > 0 else 0.0
    return diou(p, g) - alpha * v


def l1(p, g):
    return sum(abs(a - b) for a, b in zip(p, g))


LOSS_PREDS = [
    ("the box itself", (2, 3, 10, 11)),
    ("shifted by (1, 1)", (3, 4, 11, 12)),
    ("shifted by (4, 4)", (6, 7, 14, 15)),
    ("touching on the right", (10, 3, 16, 11)),
    ("one pixel apart", (11, 3, 16, 11)),
    ("far corner", (12, 12, 16, 16)),
    ("same centre, half the side", (4, 5, 8, 9)),
    ("same centre, other aspect", (1, 5, 11, 9)),
]


def box_losses():
    g = RECT_A[2]
    rows = []
    for name, p in LOSS_PREDS:
        rows.append([name, _box(p), _f(l1(p, g), 1), _f(iou(p, g)), _f(giou(p, g)), _f(diou(p, g)), _f(ciou(p, g))])
    return ("Each box against the box of rectangle a, (2, 3, 10, 11). L1 is the sum of the four absolute coordinate differences; the other "
            "columns are the measures themselves, and the loss of each is 1 minus the measure. The arctangent in the aspect term of CIoU is the "
            "library atan2(w, h).\n\n" +
            _table(["predicted box", "(x1, y1, x2, y2)", "L1", "IoU", "GIoU", "DIoU", "CIoU"], rows, "llrrrrr"))


def _slope(fn, p, g, h=1e-6):
    """Central difference of fn with respect to moving the box p to the right by one unit (both x1 and x2)."""
    pp = (p[0] + h, p[1], p[2] + h, p[3])
    pm = (p[0] - h, p[1], p[2] - h, p[3])
    return (fn(pp, g) - fn(pm, g)) / (2 * h)


def box_slopes():
    g = RECT_A[2]
    rows = []
    for name, p in LOSS_PREDS[4:6]:
        rows.append([name, _box(p)] + [_f(_slope(fn, p, g), 4) for fn in (iou, giou, diou, ciou)])
    return ("The change of each measure per unit of rightward movement of the predicted box (central difference, step 1e-6, four decimals). "
            "Moving the box left, toward the target, is the opposite sign.\n\n" +
            _table(["predicted box", "(x1, y1, x2, y2)", "d IoU", "d GIoU", "d DIoU", "d CIoU"], rows, "llrrrr"))


def scale_blindness():
    rows = []
    for side in (10, 100):
        g = (0, 0, side, side)
        p = (1, 1, side + 1, side + 1)
        rows.append([f"{side} x {side}", _f(l1(p, g), 1), _f(iou(p, g)), _f(1 - iou(p, g))])
    return ("The same error, a prediction one pixel to the right and one down of a square target, on a small and on a large box.\n\n" +
            _table(["target box", "L1", "IoU", "IoU loss"], rows))


# ------------------------------------------------------------------------------------------------ task-aligned assignment

STRIDE = 2
GRID = SIZE // STRIDE
ALPHA_TAL, BETA_TAL, TOPK = 0.5, 6.0, 3
GAINS = (7.5, 0.5, 1.5)
PRED_DIST = (1.0, 1.5, 2.0, 1.0)  # every anchor predicts these distances (l, t, r, b), in cells


def anchors():
    out = []
    for i in range(GRID):
        for j in range(GRID):
            out.append(dict(i=i, j=j, cx=(j + 0.5) * STRIDE, cy=(i + 0.5) * STRIDE, ax=j + 0.5, ay=i + 0.5))
    return out


def pred_box(a):
    return dist_to_xyxy(PRED_DIST, a["ax"], a["ay"], STRIDE)


def class_score(a, c):
    gb = (RECT_A, RECT_B)[c][2]
    gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
    d2 = (a["cx"] - gcx) ** 2 + (a["cy"] - gcy) ** 2
    return 0.02 + 0.9 * math.exp(-d2 / 32)


def assign():
    """Task-aligned assignment of the grid to the two rectangles; returns the per-anchor results and the intermediate tables."""
    A = anchors()
    gts = [RECT_A, RECT_B]
    cand, metric, ovl = {}, {}, {}
    for g, (c, _, gb) in enumerate(gts):
        for n, a in enumerate(A):
            if gb[0] < a["cx"] < gb[2] and gb[1] < a["cy"] < gb[3]:  # strictly inside
                u = max(ciou(pred_box(a), gb), 0.0)
                s = class_score(a, c)
                cand[g, n] = True
                ovl[g, n] = u
                metric[g, n] = s ** ALPHA_TAL * u ** BETA_TAL
    pos = {}
    for g in range(len(gts)):
        ids = sorted((n for (gg, n) in cand if gg == g), key=lambda n: (-metric[g, n], n))
        for n in ids[:TOPK]:
            pos[g, n] = True
    owner = {}
    for n in range(len(A)):
        best = None
        for g in range(len(gts)):
            if (g, n) in pos:
                if best is None or ovl[g, n] > ovl[best, n] or (ovl[g, n] == ovl[best, n] and metric[g, n] > metric[best, n]):
                    best = g
        if best is not None:
            owner[n] = best
    mmax = {g: max(metric[g, n] for n in owner if owner[n] == g) for g in set(owner.values())}
    umax = {g: max(ovl[g, n] for n in owner if owner[n] == g) for g in set(owner.values())}
    target = {n: metric[g, n] / max(mmax[g], 1e-9) * umax[g] for n, g in owner.items()}
    return dict(A=A, gts=gts, cand=cand, metric=metric, ovl=ovl, pos=pos, owner=owner, target=target)


def assign_table():
    r = assign()
    A, gts = r["A"], r["gts"]
    out = [f"Anchor points every {STRIDE} pixels on the 16 by 16 picture (an {GRID} x {GRID} grid, row-major numbering n = {GRID} * row + column), "
           f"each at the pixel centre ((column + 0.5) * {STRIDE}, (row + 0.5) * {STRIDE}). Every anchor predicts the distances (l, t, r, b) = "
           f"{PRED_DIST} cells, that is the box (cx - {_g(PRED_DIST[0] * STRIDE)}, cy - {_g(PRED_DIST[1] * STRIDE)}, cx + {_g(PRED_DIST[2] * STRIDE)}, "
           f"cy + {_g(PRED_DIST[3] * STRIDE)}) around its centre (cx, cy), and its score for class c is 0.02 + 0.9 exp(-d^2 / 32) with d the distance from the anchor to the centre of the "
           f"rectangle of class c. The assigner uses alpha = {ALPHA_TAL}, beta = {BETA_TAL:g} and keeps the top {TOPK} anchors per rectangle."]
    for g, (c, name, gb) in enumerate(gts):
        ids = sorted((n for (gg, n) in r["cand"] if gg == g), key=lambda n: (-r["metric"][g, n], n))
        rows = []
        for rank, n in enumerate(ids):
            a = A[n]
            rows.append([n, f"({_g(a['cx'])}, {_g(a['cy'])})", _f(class_score(a, c)), _f(r["ovl"][g, n]), f"{r['metric'][g, n]:.5f}",
                         "kept" if (g, n) in r["pos"] else ""])
        out.append(f"\nRectangle {name}, box {_box(gb)}: the {len(ids)} anchors whose centre lies strictly inside it, best alignment first.\n\n" +
                   _table(["n", "centre", "score s", "overlap u = max(CIoU, 0)", "t = s^0.5 u^6", "top-%d" % TOPK], rows, "lllrrl"))
    rows = []
    for n in sorted(r["owner"]):
        a, g = A[n], r["owner"][n]
        gb = gts[g][2]
        dist = (a["ax"] - gb[0] / STRIDE, a["ay"] - gb[1] / STRIDE, gb[2] / STRIDE - a["ax"], gb[3] / STRIDE - a["ay"])
        rows.append([n, gts[g][1], _f(r["target"][n]), "(" + ", ".join(_f(v, 2) for v in dist) + ")"])
    out.append("\nThe foreground anchors, with the soft class target (t divided by the largest t of its rectangle among the foreground anchors, times "
               "the largest u there) and the distances (l, t, r, b) in cells from the anchor point to the sides of the rectangle.\n\n" +
               _table(["n", "rectangle", "class target", "target distances (l, t, r, b)"], rows, "llrl"))
    mp = [["." for _ in range(GRID)] for _ in range(GRID)]
    for n, g in r["owner"].items():
        mp[A[n]["i"]][A[n]["j"]] = gts[g][1]
    out.append("\nThe grid, with the foreground anchors marked by the rectangle they belong to:\n\n```text\n" +
               "\n".join(" ".join(row) for row in mp) + "\n```")
    return "\n".join(out)


def bump(d, sigma=1.0, n=NBIN):
    """The distribution over the bins that the head is taken to predict for a distance d: softmax(-(k - d)^2 / (2 sigma^2))."""
    return softmax([-(k - d) ** 2 / (2 * sigma ** 2) for k in range(n)])


def detection_loss():
    r = assign()
    A, gts = r["A"], r["gts"]
    # Binary cross-entropy of all anchors and both classes against the soft targets; the logit of each entry is ln(s / (1 - s)).
    bce = 0.0
    for n, a in enumerate(A):
        for c in range(2):
            s = class_score(a, c)
            t = r["target"][n] if (n in r["owner"] and gts[r["owner"][n]][0] == c) else 0.0
            bce += -(t * math.log(s) + (1 - t) * math.log(1 - s))
    ciou_sum, dfl_sum, rows = 0.0, 0.0, []
    for n in sorted(r["owner"]):
        a, g = A[n], r["owner"][n]
        gb = gts[g][2]
        lc = 1 - ciou(pred_box(a), gb)
        tgt = (a["ax"] - gb[0] / STRIDE, a["ay"] - gb[1] / STRIDE, gb[2] / STRIDE - a["ax"], gb[3] / STRIDE - a["ay"])
        ld = sum(dfl(bump(pd), t) for pd, t in zip(PRED_DIST, tgt))
        ciou_sum += lc
        dfl_sum += ld
        rows.append([n, gts[g][1], _f(lc), _f(ld)])
    norm = max(sum(r["target"].values()), 1.0)
    total = (GAINS[0] * ciou_sum + GAINS[1] * bce + GAINS[2] * dfl_sum) / norm
    t = _table(["n", "rectangle", "1 - CIoU", "DFL (four sides)"], rows, "llrr")
    return (f"The loss of the assignment above. Each side of an anchor's box is held by the distribution softmax(-(k - d)^2 / 2) over the "
            f"{NBIN} bins around its predicted distance d (the four d are {PRED_DIST}), and the DFL of a side is the loss (5) of the chapter against "
            f"the target distance of the table. The class loss is the binary cross-entropy of all {len(A)} x 2 entries with the logit of each "
            f"score as the logit and the soft target on the owner's class channel (0 elsewhere). Sums run over anchors in row-major order.\n\n{t}\n\n"
            f"Sum of (1 - CIoU) = {_f(ciou_sum)}; sum of the DFL terms = {_f(dfl_sum)}; sum of the binary cross-entropy over all {len(A) * 2} "
            f"entries = {_f(bce)}; sum of the class targets = {_f(sum(r['target'].values()))}, so the normaliser is max(sum, 1) = {_f(norm)}.\n\n"
            f"L = ({GAINS[0]} * {_f(ciou_sum)} + {GAINS[1]} * {_f(bce)} + {GAINS[2]} * {_f(dfl_sum)}) / {_f(norm)} = {_f(total)}.")


def contested():
    """With room for every candidate (k = 12) the anchors inside both rectangles are chosen by both; the owner is the one with the larger u."""
    global TOPK
    saved, TOPK = TOPK, 12
    try:
        r = assign()
    finally:
        TOPK = saved
    rows = []
    for n in sorted({n for (g, n) in r["pos"] if (0, n) in r["pos"] and (1, n) in r["pos"]}):
        a = r["A"][n]
        rows.append([n, f"({_g(a['cx'])}, {_g(a['cy'])})", _f(r["ovl"][0, n]), f"{r['metric'][0, n]:.5f}", _f(r["ovl"][1, n]),
                     f"{r['metric'][1, n]:.5f}", r["gts"][r["owner"][n]][1]])
    return (f"The anchors that lie inside both rectangles, when each rectangle may keep all of its candidates (k = 12). Both choose them; the "
            f"anchor goes to the rectangle with the larger overlap u (the larger t decides a tie, then the lower rectangle index).\n\n" +
            _table(["n", "centre", "u for a", "t for a", "u for b", "t for b", "owner"], rows, "llrrrrl"))


# ------------------------------------------------------------------------------------------------ non-maximum suppression

# id, box, confidence, class. Candidates 0-3 are four views of rectangle a (class 0), 4-6 three of rectangle b (class 1),
# 7 is a class-1 box on a, and 8 is a class-0 box that overlaps candidate 0 with IoU exactly 0.5.
CANDIDATES = [
    (0, (2, 3, 10, 11), 0.91, 0),
    (1, (3, 4, 11, 12), 0.84, 0),
    (2, (4, 5, 12, 13), 0.77, 0),
    (3, (7, 6, 14, 14), 0.88, 1),
    (4, (8, 7, 15, 15), 0.80, 1),
    (5, (6, 5, 14, 13), 0.55, 1),
    (6, (1, 2, 9, 10), 0.60, 1),
    (7, (2, 3, 10, 7), 0.40, 0),
]


def nms(dets, thr, class_aware=True, max_det=300, strict=True):
    """Greedy NMS. Returns (kept ids, trace) where the trace has, per candidate in visiting order, the IoU evaluations made."""
    order = sorted(range(len(dets)), key=lambda i: (-dets[i][2], i))  # confidence descending, equal confidences by index
    kept, trace = [], []
    for i in order:
        if len(kept) >= max_det:
            break
        d = dets[i]
        evals = []
        dropped = None
        for k in kept:
            if class_aware and dets[k][3] != d[3]:
                continue
            v = iou(dets[k][1], d[1])
            evals.append((dets[k][0], v))
            if (v > thr) if strict else (v >= thr):
                dropped = dets[k][0]
                break
        trace.append((d[0], evals, dropped))
        if dropped is None:
            kept.append(i)
    return [dets[i][0] for i in kept], trace


def nms_trace():
    out = ["Eight candidates, as (id, box, confidence, class):\n", _table(
        ["id", "box (x1, y1, x2, y2)", "confidence", "class"], [[c[0], _box(c[1]), _f(c[2], 2), c[3]] for c in CANDIDATES], "lllr")]
    for aware in (True, False):
        kept, trace = nms(CANDIDATES, 0.5, aware)
        rows = []
        for cid, evals, dropped in trace:
            ev = ", ".join(f"{k}: {_f(v)}" for k, v in evals) or "none"
            rows.append([cid, ev, "dropped by %d" % dropped if dropped is not None else "kept"])
        total = sum(len(e) for _, e, _ in trace)
        out.append(f"\n{'Class-aware' if aware else 'Class-agnostic'} NMS with threshold 0.5 and a drop only when IoU > 0.5, visiting the "
                   f"candidates by falling confidence. For each candidate: the IoUs against the boxes kept so far that it is compared with, "
                   f"in the order kept, stopping at the first that exceeds the threshold.\n\n" +
                   _table(["candidate", "IoU with the kept boxes compared", "outcome"], rows, "rll") +
                   f"\n\nKept: {kept}; {total} IoU evaluations.")
    return "\n".join(out)


def nms_data():
    """For the figure: the candidates, the class-aware survivors."""
    kept, _ = nms(CANDIDATES, 0.5, True)
    return CANDIDATES, kept


def nms_threshold_sweep():
    rows = []
    for thr in (0.3, 0.5, 0.7):
        ka, _ = nms(CANDIDATES, thr, True)
        kg, _ = nms(CANDIDATES, thr, False)
        rows.append([_g(thr), len(ka), str(ka), len(kg), str(kg)])
    b1, b2 = (0, 0, 10, 10), (0, 0, 10, 5)
    ds = [(0, b1, 0.9, 0), (1, b2, 0.8, 0)]
    strict, _ = nms(ds, 0.5, True, strict=True)
    ge, _ = nms(ds, 0.5, True, strict=False)
    return ("The threshold decides how close two same-class boxes may be. Survivors of the eight candidates:\n\n" +
            _table(["threshold", "class-aware kept", "ids", "class-agnostic kept", "ids"], rows, "rrlrl") +
            f"\n\nAt a threshold of exactly 0.5 the boxes {_box(b1)} and {_box(b2)} have IoU {_f(iou(b1, b2))}; with the test IoU > 0.5 the "
            f"survivors are {strict}, with IoU >= 0.5 they are {ge}.")


def nms_cost():
    n, m = 8400, 300
    worst = sum(min(i, m) for i in range(n))
    return (f"A candidate list of n = {n:,} boxes (one per cell of the three levels of a 640 input) and a cap of {m} kept boxes. Candidate number "
            f"i in visiting order is compared with at most min(i, {m}) kept boxes, so the greatest number of IoU evaluations is the sum over "
            f"i = 0 .. {n - 1}, which is {worst:,}; with no cap it would be n(n - 1)/2 = {n * (n - 1) // 2:,}. One IoU is 8 operations "
            f"for the overlap (a minimum, a maximum, a subtraction and a floor at 0, for each of the width and the height), 1 product for "
            f"the intersection, 3 for each of the two areas, 2 for the union and 1 division, 18 operations, so the worst case is "
            f"{18 * worst:,} operations.")


# ------------------------------------------------------------------------------------------------ average precision

# Class 0: ground truth (image, box); detections (image, box, confidence). Class 1 is a second, smaller set.
MAP_GT = {0: [(0, (2, 3, 10, 11)), (0, (7, 6, 14, 14)), (1, (1, 1, 9, 9)), (2, (4, 4, 12, 12))],
          1: [(1, (11, 2, 15, 6)), (2, (1, 10, 5, 15))]}
MAP_DET = {0: [(2, (12, 12, 16, 16), 0.95), (0, (3, 4, 11, 12), 0.90), (0, (2, 3, 10, 11), 0.85), (1, (1, 2, 9, 10), 0.80),
               (0, (8, 7, 15, 15), 0.70), (2, (6, 6, 14, 14), 0.60), (1, (9, 9, 15, 15), 0.50)],
           1: [(1, (11, 2, 15, 6), 0.92), (0, (4, 4, 8, 8), 0.75), (2, (1, 10, 5, 14), 0.65)]}


def match(c, thr):
    """Rank the detections of class c by confidence (ties by listing order); a detection is a true positive when its best IoU against a
    ground-truth box of the same image that no earlier detection took reaches thr (>=). Returns the ranked rows."""
    dets = MAP_DET[c]
    order = sorted(range(len(dets)), key=lambda i: (-dets[i][2], i))
    used = set()
    rows = []
    for i in order:
        img, box, conf = dets[i]
        best, bi = thr, None
        for gi, (gimg, gb) in enumerate(MAP_GT[c]):
            if gi in used or gimg != img:
                continue
            v = iou(box, gb)
            if v >= best:
                best, bi = v, gi
        # report the best IoU against any same-image box for the table
        anyv = max([iou(box, gb) for gimg, gb in MAP_GT[c] if gimg == img] or [0.0])
        if bi is not None:
            used.add(bi)
        rows.append((i, img, box, conf, anyv, bi is not None))
    return rows


def pr_points(c, thr):
    rows = match(c, thr)
    n_gt = len(MAP_GT[c])
    tp = fp = 0
    pts = [(0.0, 1.0)]
    for r in rows:
        if r[5]:
            tp += 1
        else:
            fp += 1
        pts.append((tp / n_gt, tp / (tp + fp)))
    return rows, pts


def envelope(pts):
    p = [q for _, q in pts]
    for k in range(len(p) - 2, -1, -1):
        p[k] = max(p[k], p[k + 1])
    return p


def average_precision(c, thr):
    _, pts = pr_points(c, thr)
    env = envelope(pts)
    return sum((pts[k][0] - pts[k - 1][0]) * env[k] for k in range(1, len(pts)))


def step_ap(c, thr):
    """The step-wise area of Volume II: the precision of the raw curve with no envelope."""
    _, pts = pr_points(c, thr)
    return sum((pts[k][0] - pts[k - 1][0]) * pts[k][1] for k in range(1, len(pts)))


def pr_data(c=0, thr=0.5):
    _, pts = pr_points(c, thr)
    return pts, envelope(pts), average_precision(c, thr)


def ap_table():
    rows, pts = pr_points(0, 0.5)
    env = envelope(pts)
    n_gt = len(MAP_GT[0])
    out = []
    tp = fp = 0
    for k, (i, img, box, conf, anyv, is_tp) in enumerate(rows):
        tp += is_tp
        fp += not is_tp
        out.append([k + 1, _f(conf, 2), img, _box(box), _f(anyv, 3), "TP" if is_tp else "FP", tp, fp, _f(pts[k + 1][1], 4), _f(pts[k + 1][0], 2),
                    _f(env[k + 1], 4)])
    ap = average_precision(0, 0.5)
    terms = " + ".join(f"{_f(pts[k][0] - pts[k - 1][0], 2)} x {_f(env[k], 4)}" for k in range(1, len(pts)) if pts[k][0] > pts[k - 1][0])
    return (f"Class 0: four ground-truth boxes ({', '.join('image %d %s' % (im, _box(b)) for im, b in MAP_GT[0])}) and seven detections. A detection "
            f"is a true positive when its IoU with a ground-truth box of the same image that no earlier detection took is at least 0.5. "
            f"The column best IoU is the largest IoU with any ground-truth box of the image, taken or not.\n\n" +
            _table(["rank", "confidence", "image", "box", "best IoU", "at 0.5", "TP so far", "FP so far", "precision", "recall", "envelope"],
                   out, "rrrlrlrrrrr") +
            f"\n\nAP@0.5 = {terms} = {_f(ap, 4)}. The step-wise area of Volume II, with no envelope, is {_f(step_ap(0, 0.5), 4)}.")


def map_table():
    ths = [0.3, 0.5, 0.75]
    rows = []
    for c in (0, 1):
        rows.append([c] + [_f(average_precision(c, t), 4) for t in ths])
    rows.append(["mean (mAP)"] + [_f((average_precision(0, t) + average_precision(1, t)) / 2, 4) for t in ths])
    coco = [0.5 + 0.05 * k for k in range(10)]
    m = sum((average_precision(0, t) + average_precision(1, t)) / 2 for t in coco) / 10
    per = ", ".join(_f((average_precision(0, t) + average_precision(1, t)) / 2, 4) for t in coco)
    return (f"Average precision of the two classes at three IoU thresholds, and their mean. Class 1 has two ground-truth boxes "
            f"({', '.join('image %d %s' % (im, _box(b)) for im, b in MAP_GT[1])}) and three detections "
            f"({', '.join('image %d %s %s' % (im, _box(b), _f(cf, 2)) for im, b, cf in MAP_DET[1])}).\n\n" +
            _table(["class", "AP@0.3", "AP@0.5", "AP@0.75"], rows, "lrrr") +
            f"\n\nThe mean over the ten thresholds 0.50, 0.55, ..., 0.95 (mAP@0.5:0.95), whose per-threshold values are "
            f"{per}, is {_f(m, 4)}. A detection matches a ground-truth box when its IoU is at least the threshold. The class-1 detection "
            f"in image 2 has an IoU of exactly 16/20 = 0.8, and the threshold 0.5 + 0.05 * 6, evaluated in double precision, is also exactly "
            f"0.8, so that tie is a match; an evaluator that compared with a strict inequality, or that built the thresholds by repeated "
            f"addition and landed a rounding error above 0.8, would lose it and print {_f((average_precision(0, 0.8000001) + average_precision(1, 0.8000001)) / 2, 4)} "
            f"at that threshold in place of {_f((average_precision(0, 0.8) + average_precision(1, 0.8)) / 2, 4)}.")


# ------------------------------------------------------------------------------------------------ letterbox and conventions


def letterbox(w0, h0, size):
    s = min(size / w0, size / h0)
    nw, nh = round(w0 * s), round(h0 * s)
    return s, nw, nh, (size - nw) / 2, (size - nh) / 2


def apply_lb(b, lb):
    s, _, _, px, py = lb
    return (b[0] * s + px, b[1] * s + py, b[2] * s + px, b[3] * s + py)


def invert_lb(b, lb, w0, h0):
    s, _, _, px, py = lb
    cl = lambda v, hi: min(max(v, 0.0), hi)
    return (cl((b[0] - px) / s, w0), cl((b[1] - py) / s, h0), cl((b[2] - px) / s, w0), cl((b[3] - py) / s, h0))


def unletterbox():
    w0, h0, size = 200, 100, 128
    lb = letterbox(w0, h0, size)
    s, nw, nh, px, py = lb
    truth = (20, 30, 120, 80)
    fwd = apply_lb(truth, lb)
    back = invert_lb(fwd, lb, w0, h0)
    out = [f"A {w0} by {h0} image fitted to a {size} by {size} square: scale = min({size}/{w0}, {size}/{h0}) = {_g(s)}, content {nw} x {nh}, "
           f"padding (pad_x, pad_y) = ({_g(px)}, {_g(py)}). The box {_box(truth)} of the original goes to {_box(tuple(round(v, 4) for v in fwd))} "
           f"in the input, and the inverse (subtract the padding, divide by the scale, clamp to the image) returns {_box(tuple(round(v, 4) for v in back))}."]
    lim = (w0, h0, w0, h0)
    errs = {
        "correct inverse": back,
        "padding not subtracted": tuple(min(max(v / s, 0.0), m) for v, m in zip(fwd, lim)),
        "scale not divided": tuple(min(max(v - o, 0.0), m) for v, o, m in zip(fwd, (px, py, px, py), lim)),
        "padding of the wrong axis": tuple(min(max((v - o) / s, 0.0), m) for v, o, m in zip(fwd, (py, px, py, px), lim)),
    }
    rows = [[k, _box(tuple(round(v, 2) for v in b)), _f(iou(b, truth))] for k, b in errs.items()]
    out.append("\nWhat each mistake in the inverse gives, and its IoU with the true box:\n\n" + _table(["inverse", "box", "IoU with the truth"], rows, "llr"))
    # odd padding: 7 x 4 into 8 x 8 (the case of Volume VII, Chapter 1)
    w1, h1, s1 = 7, 4, 8
    lb1 = letterbox(w1, h1, s1)
    pad = lb1[4]
    out.append(f"\nWith odd padding the pixels and the boxes can disagree. A {w1} by {h1} image fitted to {s1} by {s1} has scale {_f(lb1[0], 6)}, content "
               f"{lb1[1]} x {lb1[2]} and pad_y = {_g(pad)}. Boxes are mapped with the float {_g(pad)}, but a pixel row y of the content is written to "
               f"row y + int(pad_y) = y + {int(pad)}. An edge at row y of the original is therefore drawn by the box rule at y * scale + {_g(pad)} while "
               f"the pixels of that row are written near y * scale + {int(pad)}: the boxes sit {_g(pad - int(pad))} input pixel below the pixels, "
               f"which is {_f((pad - int(pad)) / lb1[0], 4)} pixel of the original.")
    return "\n".join(out)


def convention_mixups():
    truth = (20.0, 30.0, 120.0, 80.0)
    cx, cy, w, h = xyxy_to_xywh(truth)
    img = 200.0
    norm = tuple(v / img for v in (cx, cy, w, h))
    wrong = [
        ("xyxy read as xywh", xywh_to_xyxy(truth)),
        ("xywh read as xyxy", (cx, cy, w, h)),
        ("normalized xywh read as pixels", xywh_to_xyxy(norm)),
    ]
    rows = [[name, _box(tuple(round(v, 3) for v in b)), _f(iou(b, truth))] for name, b in wrong]
    return (f"One box of a 200 by 200 image, (x1, y1, x2, y2) = {_box(truth)}, which is (cx, cy, w, h) = {_box((cx, cy, w, h))} in pixels and "
            f"{_box(tuple(round(v, 3) for v in norm))} divided by 200. What a reader gets when it applies the wrong convention, and the IoU of that box "
            f"with the true one (a box with x2 < x1 or y2 < y1 has area 0 and IoU 0).\n\n" +
            _table(["mistake", "resulting (x1, y1, x2, y2)", "IoU with the truth"], rows, "llr"))
