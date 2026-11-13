# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Faces: Detection, Identity and Parsing": how a model finds faces and recognizes the same person
across photos.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: faces.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. The box
arithmetic of the detection chapter is imported (`detection.iou`, `detection.sigmoid`), never copied.

* the five-point geometry: three photographs as landmark lists, the least-squares similarity that takes a photograph onto the
  template, what it does to the distances, and the inputs it refuses;
* the warp: the destination-to-source map, the grid conventions and what the border does to a sample;
* the detector: the candidates and the cost of the SCRFD layer list, the decode of one row, the two box conventions of the
  suppression, the letterbox of the reference and the two normalizations;
* the embedder: the cost of the IResNet-100 layer list, the loss floor of a normalized classifier and the angle a loss asks for, the
  margin on one sample, the shape of the target logit, its slope, and a simulated threshold on cosines;
* the parser: the two alignment templates compared and the mask built from a class map.

The only randomness is the detector jitter of the three photographs (`faces.landmarks`) and the simulated embeddings
(`faces.verification`); both are drawn from the book's generator in the order the chapter states. Sums run left to right. FLOPs count
2 per multiply-add; biases, activations, pooling and normalizations are left out of every multiply-add count, and each table says so.

Swedish Embedded AB implements solutions for face pipelines whose alignment, normalization and thresholds are checked against a
hand computation. If your team needs expertise in face detection and recognition, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import detection as dt

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(v, d=2):
    return f"{v:.{d}f}"


def _int(v):
    return f"{v:,}"


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"seed 0x{s:016x} ({s})"


def _pt(p, d=1):
    return f"({p[0]:.{d}f}, {p[1]:.{d}f})"


# ------------------------------------------------------------------------------------------------ the five points

# The five landmarks of a face in the order a detector reports them: left eye, right eye, nose tip, left and right corner of the mouth
# ("left" is the left of the picture, the smaller x).
POINTS = ("left eye", "right eye", "nose", "left mouth", "right mouth")
# The canonical template of the 112 by 112 recognition crop, in pixel-centre coordinates (x to the right, y down).
TEMPLATE_112 = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041))
# The canonical template of the 512 by 512 parsing crop.
TEMPLATE_512 = ((192.98138, 239.94708), (318.90277, 240.1936), (256.63416, 314.01935), (201.26117, 371.41043), (313.08905, 371.15118))

# How person A and person B differ from the template, in template pixels, and how each photograph was taken:
# (scale, rotation in degrees from the x axis towards the y axis, which is clockwise on the screen, shift x, shift y).
SHAPE_A = ((-2.0, 1.0), (1.5, -0.8), (0.5, 1.5), (-1.0, -1.5), (1.5, -1.0))
SHAPE_B = ((1.5, -1.2), (-2.0, 0.9), (-0.5, -1.0), (1.8, 1.2), (-2.2, 1.6))
PHOTOS = (("A1", SHAPE_A, 2.0, 10.0, 60.0, 90.0), ("A2", SHAPE_A, 3.0, -12.0, 150.0, 130.0), ("B", SHAPE_B, 2.0, 10.0, 60.0, 90.0))
JITTER = 0.4  # standard deviation of the detector's error, pixels
LANDMARK_NAME = "faces.landmarks"


def similarity_apply(m, p):
    """Apply the similarity m = (a, b, tx, ty), the matrix [[a, -b], [b, a]] and the shift (tx, ty), to the point p."""
    a, b, tx, ty = m[:4]
    return (a * p[0] - b * p[1] + tx, b * p[0] + a * p[1] + ty)


def similarity_fit(src, dst):
    """Least-squares similarity src -> dst. Returns (a, b, tx, ty, det) where det is the determinant of the cross-covariance sums."""
    n = len(src)
    ms = (sum(p[0] for p in src) / n, sum(p[1] for p in src) / n)
    md = (sum(p[0] for p in dst) / n, sum(p[1] for p in dst) / n)
    var = c00 = c01 = c10 = c11 = 0.0
    for (px, py), (qx, qy) in zip(src, dst):
        sx, sy, dx, dy = px - ms[0], py - ms[1], qx - md[0], qy - md[1]
        var += sx * sx + sy * sy
        c00 += dx * sx
        c01 += dx * sy
        c10 += dy * sx
        c11 += dy * sy
    if var <= 0.0:
        raise ValueError("degenerate")
    a, b = (c00 + c11) / var, (c10 - c01) / var
    return (a, b, md[0] - (a * ms[0] - b * ms[1]), md[1] - (b * ms[0] + a * ms[1]), c00 * c11 - c01 * c10)


def similarity_invert(m):
    """The inverse of the similarity (a, b, tx, ty): the matrix is (a, b; -b, a) / (a^2 + b^2), and the shift follows."""
    a, b, tx, ty = m[:4]
    q = a * a + b * b
    ia, ib = a / q, -b / q
    return (ia, ib, -(ia * tx - ib * ty), -(ib * tx + ia * ty))


def rms_distance(p, q):
    return math.sqrt(sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(p, q)) / len(p))


def photo_landmarks():
    """The landmarks of the three photographs, rounded to 0.1 pixel: the canonical shape of the person, moved by the pose of the photo,
    plus the detector's jitter. Draw order: photo A1, A2, B; within a photo the five points in order; per point x then y."""
    g = bookrng.generator(LANDMARK_NAME)
    out = {}
    for name, shape, scale, deg, tx, ty in PHOTOS:
        th = math.radians(deg)
        pts = []
        for t, o in zip(TEMPLATE_112, shape):
            x, y = t[0] + o[0], t[1] + o[1]
            px = scale * (math.cos(th) * x - math.sin(th) * y) + tx + JITTER * g.normal()
            py = scale * (math.sin(th) * x + math.cos(th) * y) + ty + JITTER * g.normal()
            pts.append((round(px, 1), round(py, 1)))
        out[name] = tuple(pts)
    return out


def photos():
    lm = photo_landmarks()
    names = [p[0] for p in PHOTOS]
    rows = [(POINTS[i], *[_pt(lm[n][i]) for n in names]) for i in range(5)]
    pairs = [("A1", "A2"), ("A1", "B"), ("A2", "B")]
    dist = [(f"{a} and {b}", _f(rms_distance(lm[a], lm[b]))) for a, b in pairs]
    return ("Landmarks of three photographs, in pixels of each photograph (x to the right, y down), rounded to 0.1. A1 and A2 are two "
            "photographs of person A, B is person B photographed with the pose of A1. The generator is "
            f"`{LANDMARK_NAME}` ({_seed(LANDMARK_NAME)}): each landmark is the canonical shape of the person moved by the pose "
            "of the photograph plus a jitter. The canonical shape of a person is the 112-pixel template of the recognition network "
            "(printed further on, where the fit is derived) plus a fixed offset for each point, in the order of the table: person A "
            f"{', '.join(_pt(o) for o in SHAPE_A)}; person B {', '.join(_pt(o) for o in SHAPE_B)}. A photograph with scale k, rotation "
            "theta and shift (tx, ty) maps a shape point (x, y) to (k (x cos theta - y sin theta) + tx + 0.4 e_x, "
            "k (x sin theta + y cos theta) + ty + 0.4 e_y), where e_x and e_y are standard normals (`normal()`), drawn in the order "
            "photo (A1, A2, B), point (as in the table), coordinate (x then y), and each coordinate is rounded to 0.1. The poses are "
            "A1: k = 2.0, theta = 10 degrees, shift (60, 90); A2: k = 3.0, theta = -12 degrees, shift (150, 130); B: the pose of A1.\n\n"
            + _table(["landmark", "A1", "A2", "B"], rows)
            + "\n\nRoot-mean-square distance between corresponding landmarks of two photographs (the square root of the mean over "
            "the five points of the squared distance), pixels:\n\n"
            + _table(["pair", "distance"], dist))


def fit_one(name):
    lm = photo_landmarks()[name]
    return lm, similarity_fit(lm, TEMPLATE_112)


def similarity_fit_table():
    lm = photo_landmarks()
    src = lm["A1"]
    n = len(src)
    ms = (sum(p[0] for p in src) / n, sum(p[1] for p in src) / n)
    md = (sum(p[0] for p in TEMPLATE_112) / n, sum(p[1] for p in TEMPLATE_112) / n)
    var = c_dot = c_cross = 0.0
    for (px, py), (qx, qy) in zip(src, TEMPLATE_112):
        sx, sy, dx, dy = px - ms[0], py - ms[1], qx - md[0], qy - md[1]
        var += sx * sx + sy * sy
        c_dot += sx * dx + sy * dy
        c_cross += sx * dy - sy * dx
    a, b, tx, ty, det = similarity_fit(src, TEMPLATE_112)
    steps = [("mean of the photograph's landmarks", _pt(ms, 3)),
             ("mean of the template", _pt(md, 3)),
             ("sum of squared offsets from the mean", _f(var, 3)),
             ("sum of dot products (template offset, photograph offset)", _f(c_dot, 3)),
             ("sum of cross products (photograph offset x template offset)", _f(c_cross, 3)),
             ("a = dot / squared", f"{a:.6f}"),
             ("b = cross / squared", f"{b:.6f}"),
             ("tx = template mean x - (a mean x - b mean y)", _f(tx, 4)),
             ("ty = template mean y - (b mean x + a mean y)", _f(ty, 4))]
    lm_rows = []
    for name in ("A1", "A2", "B"):
        pts, m = fit_one(name)
        al = [similarity_apply(m, p) for p in pts]
        scale, ang = math.hypot(m[0], m[1]), math.degrees(math.atan2(m[1], m[0]))
        lm_rows.append((name, f"{m[0]:.4f}", f"{m[1]:.4f}", _f(m[2], 2), _f(m[3], 2), f"{scale:.4f}", _f(ang, 2), _f(rms_distance(al, TEMPLATE_112), 3)))
    return ("The fit of photograph A1 onto the 112 template (`TEMPLATE_112`, printed in the text), with the sums over the five points "
            "taken left to right:\n\n" + _table(["quantity", "value"], steps)
            + "\n\nThe fit of the three photographs. Scale is the square root of a squared plus b squared and angle is atan2(b, a) in "
            "degrees; the last column is the root-mean-square distance between the five transformed landmarks and the template.\n\n"
            + _table(["photo", "a", "b", "tx", "ty", "scale", "angle", "residual"], lm_rows))


def aligned_data():
    """The aligned landmarks of the three photographs (the template frame), for the figure."""
    lm = photo_landmarks()
    out = {}
    for name in ("A1", "A2", "B"):
        m = similarity_fit(lm[name], TEMPLATE_112)
        out[name] = [similarity_apply(m, p) for p in lm[name]]
    return out


def aligned_distances():
    al = aligned_data()
    pairs = [("A1", "A2"), ("A1", "B"), ("A2", "B")]
    rows = [(f"{a} and {b}", _f(rms_distance(al[a], al[b]), 3)) for a, b in pairs]
    rows += [(f"{n} and the template", _f(rms_distance(al[n], TEMPLATE_112), 3)) for n in ("A1", "A2", "B")]
    return ("Root-mean-square distance in template pixels, after each photograph is taken onto the template by its own fit:\n\n"
            + _table(["pair", "distance"], rows))


def fit_refusals():
    lm = photo_landmarks()["A1"]
    cases = []

    def attempt(label, src, dst=TEMPLATE_112):
        try:
            a, b, tx, ty, det = similarity_fit(src, dst)
            if det < 0.0:
                return (label, "refused: the best fit needs a reflection", "-", "-", "-")
            al = [similarity_apply((a, b, tx, ty), p) for p in src]
            return (label, "fitted", _f(math.hypot(a, b), 4), _f(math.degrees(math.atan2(b, a)), 1), _f(rms_distance(al, dst), 2))
        except ValueError:
            return (label, "refused: every point is the same", "-", "-", "-")

    cases.append(attempt("A1 as detected", lm))
    cases.append(attempt("the two mouth corners swapped", [lm[0], lm[1], lm[2], lm[4], lm[3]]))
    cases.append(attempt("the points in reverse order", [lm[4], lm[3], lm[2], lm[1], lm[0]]))
    cases.append(attempt("eyes and mouth corners swapped", [lm[1], lm[0], lm[2], lm[4], lm[3]]))
    cases.append(attempt("picture mirrored, order kept", [(-p[0], p[1]) for p in lm]))
    cases.append(attempt("five copies of one point", [lm[0]] * 5))
    return ("What the fit does with inputs that are not a face in the expected order (photograph A1, 112 template). Scale is the "
            "square root of a squared plus b squared, angle is atan2(b, a) in degrees, and the residual is in template pixels.\n\n"
            + _table(["input", "outcome", "scale", "angle", "residual"], cases, "lrrrr"))


# ------------------------------------------------------------------------------------------------ the warp

PHOTO_W, PHOTO_H = 320, 400  # the size of the picture that A1 was taken from, for the warp table


def warp_conventions():
    lm = photo_landmarks()["A1"]
    m = similarity_fit(lm, TEMPLATE_112)
    inv = similarity_invert(m)
    rows = []
    for (x, y) in ((0, 0), (111, 0), (56, 56), (111, 111)):
        xs, ys = similarity_apply(inv, (x, y))
        gx, gy = (2 * xs + 1) / PHOTO_W - 1, (2 * ys + 1) / PHOTO_H - 1
        back_x = (gx + 1) * (PHOTO_W - 1) / 2  # the same grid value read with the corner convention
        back_y = (gy + 1) * (PHOTO_H - 1) / 2
        rows.append((f"({x}, {y})", _pt((xs, ys), 3), f"({gx:.5f}, {gy:.5f})", _pt((back_x - xs, back_y - ys), 3)))
    return (f"The destination pixels of the 112 by 112 crop of photograph A1, which is {PHOTO_W} by {PHOTO_H} pixels, mapped back to the "
            "photograph by the inverse of the fit (a, b, tx, ty) of the previous table. gx = (2 xs + 1) / width - 1 and "
            "gy = (2 ys + 1) / height - 1 is the grid value of a sampler whose coordinate -1 is the left edge of the first pixel and "
            "+1 the right edge of the last. The last column is how far (in photograph pixels, x and y) a sampler that reads the "
            "same value with the corner convention (-1 at the centre of the first pixel, +1 at the centre of the last) would "
            "land from the right place.\n\n"
            + _table(["destination pixel", "source (xs, ys)", "grid (gx, gy)", "error of the corner reading"], rows))


def border_fill():
    left, right = 100.0, 120.0   # the values of the pixels at x = 0 and x = 1
    rows = []
    for x in (0.5, 0.0, -0.25, -0.5):
        # linear interpolation between x0 = floor(x) and x0 + 1
        x0 = math.floor(x)
        t = x - x0
        taps = {-1: None, 0: left, 1: right}
        v0, v1 = taps[x0], taps[x0 + 1]
        zero = (1 - t) * (0.0 if v0 is None else v0) + t * v1
        clamp = (1 - t) * (left if v0 is None else v0) + t * v1
        grey = (1 - t) * (128.0 if v0 is None else v0) + t * v1
        rows.append((_f(x, 2), _f(zero, 2), _f(clamp, 2), _f(grey, 2)))
    return ("One row of a picture whose first two pixels are 100 and 120 (pixel i centred at x = i), sampled between them and "
            "beyond the left edge by linear interpolation. The pixel at x = -1 does not exist; the three columns say what it is "
            "taken to be: 0 (zero padding), the edge pixel 100 (clamping) or a constant grey of 128.\n\n"
            + _table(["x", "zero padding", "clamp to the edge", "constant grey 128"], rows))


# ------------------------------------------------------------------------------------------------ the detector

# The layer list of the detector, read off its configuration: a stem of three 3x3 convolutions (the first of stride 2) and a 2x2
# max pool, four stages of basic blocks, a neck, and three heads.
SCRFD_SIZE = 640
SCRFD_STEM = (28, 28, 56)
SCRFD_LAYERS = (3, 4, 2, 3)
SCRFD_CHANNELS = (56, 88, 88, 224)
SCRFD_STAGE_STRIDE = (False, True, True, True)
SCRFD_NECK = 56
SCRFD_HEAD = 80
SCRFD_HEAD_DEPTH = 3
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS = 2


def _conv(cin, cout, k, side):
    """(parameters with a bias, multiply-adds) of a convolution whose output is side x side."""
    return cin * cout * k * k + cout, cin * cout * k * k * side * side


def scrfd_cost(size=SCRFD_SIZE):
    rows, tot_p, tot_m = [], 0, 0

    def stage(name, convs, out):
        nonlocal tot_p, tot_m
        p = sum(c[0] for c in convs)
        m = sum(c[1] for c in convs)
        tot_p += p
        tot_m += m
        rows.append((name, out, _int(p), _int(m)))

    side = size // 2
    stage("stem: three 3x3 convolutions", [_conv(3, SCRFD_STEM[0], 3, side), _conv(SCRFD_STEM[0], SCRFD_STEM[1], 3, side),
                                            _conv(SCRFD_STEM[1], SCRFD_STEM[2], 3, side)], f"{SCRFD_STEM[2]} x {side // 2} x {side // 2}")
    side //= 2
    cin = SCRFD_STEM[2]
    taps = []
    for s in range(4):
        convs = []
        for b in range(SCRFD_LAYERS[s]):
            stride = 2 if (b == 0 and SCRFD_STAGE_STRIDE[s]) else 1
            out = side // stride
            c = SCRFD_CHANNELS[s]
            convs += [_conv(cin, c, 3, out), _conv(c, c, 3, out)]
            if stride != 1 or cin != c:
                convs.append(_conv(cin, c, 1, out))
            cin, side = c, out
        taps.append((cin, side))
        stage(f"stage {s + 1}: {SCRFD_LAYERS[s]} basic blocks", convs, f"{cin} x {side} x {side}")
    nk = SCRFD_NECK
    l3, l4, l5 = taps[1], taps[2], taps[3]
    neck = [_conv(c, nk, 1, h) for c, h in (l3, l4, l5)] + [_conv(nk, nk, 3, h) for _, h in (l3, l4, l5)]
    neck += [_conv(nk, nk, 3, l4[1]), _conv(nk, nk, 3, l5[1])] + [_conv(nk, nk, 3, h) for _, h in (l4, l5)]
    stage("neck: lateral, smoothing, down and merge convolutions", neck, f"{nk} x {l3[1]}, {l4[1]}, {l5[1]}")
    head = []
    for _, h in (l3, l4, l5):
        cin_h = nk
        for _ in range(SCRFD_HEAD_DEPTH):
            head.append(_conv(cin_h, SCRFD_HEAD, 3, h))
            cin_h = SCRFD_HEAD
        head += [_conv(SCRFD_HEAD, SCRFD_ANCHORS * k, 3, h) for k in (1, 4, 10)]
    scale_params = 3
    p = sum(c[0] for c in head) + scale_params
    m = sum(c[1] for c in head)
    tot_p += p
    tot_m += m
    rows.append(("three heads (3 convolutions, then 1, 4 and 10 values per anchor)", f"{SCRFD_ANCHORS * 15} x {l3[1]}, {l4[1]}, {l5[1]}", _int(p), _int(m)))
    return rows, tot_p, tot_m


def scrfd_counts():
    rows = []
    cells = 0
    for s in SCRFD_STRIDES:
        side = SCRFD_SIZE // s
        cand = side * side * SCRFD_ANCHORS
        cells += side * side
        rows.append((str(s), f"{side} x {side}", _int(side * side), _int(cand), _int(cand * 15)))
    tot_c = sum((SCRFD_SIZE // s) ** 2 * SCRFD_ANCHORS for s in SCRFD_STRIDES)
    rows.append(("all", "", _int(cells), _int(tot_c), _int(tot_c * 15)))
    return ("The three levels of the face detector on its 640 by 640 canvas, with two anchors per location and, for each, 1 score, "
            "4 distances and 10 landmark offsets (15 numbers).\n\n"
            + _table(["stride", "grid", "locations", "candidate rows", "numbers written"], rows))


def scrfd_cost_table():
    rows, p, m = scrfd_cost()
    rows.append(("total", "", _int(p), _int(m)))
    vga = m * 480 // 640
    return ("Parameters (weights and biases, the batch normalizations folded into the convolutions, and the three learned scales) "
            "and multiply-adds of the detector's convolutions on a 640 by 640 input, from the layer list of the text. Pooling, the "
            "shortcut average pools, the additions, the activations and the nearest-neighbour doublings are not counted. The layers "
            "counted, in full: the stem convolutions have stride 1 except the first, which has stride 2 (output 320 by 320), followed by the "
            "2 x 2 max pool to 160; a basic block is a 3 x 3 convolution and a second 3 x 3 convolution of the same width, and in a "
            "strided stage (stages 2 to 4) the first block's first convolution has the stride 2, so both its convolutions and its 1 x 1 "
            "shortcut convolution are costed at the halved side; the shortcut also has a 1 x 1 convolution when the width changes. The neck "
            "takes the outputs of stages 2, 3 and 4 (strides 8, 16, 32): three 1 x 1 lateral convolutions to 56 channels, three 3 x 3 "
            "smoothing convolutions on the three levels, two 3 x 3 stride-2 convolutions that bring a finer level down to the next "
            "coarser (costed at the coarser side), and two 3 x 3 merge convolutions on the two coarser levels, each of 56 to 56 channels.\n\n"
            + _table(["part", "output", "parameters", "multiply-adds"], rows, "lrrr")
            + f"\n\nTwice the multiply-adds, the FLOPs, is {_int(2 * m)}. The same layer list on a 640 by 480 input needs "
            f"{_int(vga)} multiply-adds (three quarters of the square canvas).")


def scrfd_decode_row():
    side, stride, a_count = 40, 16, 2
    gx, gy, a = 17, 9, 1
    row = (gy * side + gx) * a_count + a
    ax, ay = gx * stride, gy * stride
    d = (1.4, 2.0, 1.3, 1.9)   # left, top, right, bottom, in strides
    k = ((-0.7, -0.9), (0.6, -0.9), (0.0, 0.2), (-0.5, 0.9), (0.5, 0.9))  # landmark offsets, in strides
    logit = 1.9
    box = (ax - d[0] * stride, ay - d[1] * stride, ax + d[2] * stride, ay + d[3] * stride)
    pts = [(ax + dx * stride, ay + dy * stride) for dx, dy in k]
    det_scale = 0.5
    src_box = tuple(v / det_scale for v in box)
    src_pts = [(x / det_scale, y / det_scale) for x, y in pts]
    rows = [("row of the table", f"({gy} * {side} + {gx}) * {a_count} + {a} = {row}"),
            ("anchor point", f"({gx} * {stride}, {gy} * {stride}) = ({ax}, {ay})"),
            ("score", f"sigmoid({logit}) = {dt.sigmoid(logit):.4f}"),
            ("distances (l, t, r, b)", f"{d} strides = {tuple(v * stride for v in d)} pixels"),
            ("box on the canvas", f"({_f(box[0], 1)}, {_f(box[1], 1)}, {_f(box[2], 1)}, {_f(box[3], 1)})"),
            ("box in the photograph (divide by 0.5)", f"({_f(src_box[0], 1)}, {_f(src_box[1], 1)}, {_f(src_box[2], 1)}, {_f(src_box[3], 1)})")]
    kp = [(POINTS[i], f"{k[i]}", _pt(pts[i]), _pt(src_pts[i])) for i in range(5)]
    return ("One candidate of the stride-16 level (a 40 by 40 grid, two anchors per location), location column 17, row 9, second "
            "anchor. The head's numbers are the distances and offsets in strides, already multiplied by the level's learned scale.\n\n"
            + _table(["quantity", "value"], rows, "ll")
            + "\n\nThe five landmark offsets (dx, dy) in strides, and the points they give on the canvas and in a photograph that "
            "was shrunk by 0.5 to fit it:\n\n"
            + _table(["landmark", "offset", "canvas", "photograph"], kp, "llrr"))


def _iou_plus1(a, b):
    """The intersection over union with the inclusive pixel convention: a side of x1 .. x2 counts x2 - x1 + 1 pixels."""
    iw = max(min(a[2], b[2]) - max(a[0], b[0]) + 1, 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]) + 1, 0.0)
    inter = iw * ih
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def _nms_plus1(boxes, scores, thr, tie_high_index_first=True):
    key = (lambda i: (-scores[i], -i)) if tie_high_index_first else (lambda i: (-scores[i], i))
    order = sorted(range(len(boxes)), key=key)
    keep = []
    for i in order:
        if all(_iou_plus1(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(i)
    return keep


def nms_conventions():
    a, b = (0.0, 0.0, 9.0, 9.0), (5.0, 0.0, 14.0, 9.0)
    half = lambda r: tuple(v / 2 for v in r)  # noqa: E731
    rows = []
    for label, p, q in (("canvas pixels", a, b), ("photograph pixels (divided by 2)", half(a), half(b))):
        rows.append((label, _f(dt.iou(p, q), 4), _f(_iou_plus1(p, q), 4)))
    thr = 0.35
    verdict = [("suppress on the canvas, then scale", len(_nms_plus1([a, b], [0.9, 0.8], thr))),
               ("scale, then suppress", len(_nms_plus1([half(a), half(b)], [0.9, 0.8], thr)))]
    tie_boxes, tie_scores = [(0.0, 0.0, 9.0, 9.0), (1.0, 0.0, 10.0, 9.0)], [0.5, 0.5]
    tie = [("higher index first (the reference)", _nms_plus1(tie_boxes, tie_scores, 0.5, True)),
           ("lower index first (a stable descending sort)", _nms_plus1(tie_boxes, tie_scores, 0.5, False))]
    return ("Two boxes a = (0, 0, 9, 9) and b = (5, 0, 14, 9) with scores 0.9 and 0.8. IoU in the half-open convention of "
            "Chapter 4 (`detection.iou`) and in the inclusive convention (a side from x1 to x2 counts x2 - x1 + 1 pixels).\n\n"
            + _table(["coordinates", "IoU, no +1", "IoU, with +1"], rows)
            + f"\n\nBoxes kept at the threshold {thr} (inclusive convention):\n\n"
            + _table(["order of the two steps", "boxes kept"], verdict)
            + "\n\nTwo boxes (0, 0, 9, 9) and (1, 0, 10, 9) with the same score 0.5 and threshold 0.5 (inclusive IoU "
            f"{_f(_iou_plus1(*tie_boxes), 4)}); indices of the boxes kept, in order:\n\n"
            + _table(["tie rule", "kept"], [(r, str(k)) for r, k in tie], "lr"))


def det_canvas():
    size = SCRFD_SIZE
    rows = []
    for w, h in ((1280, 720), (720, 1280), (640, 640), (1000, 333), (1279, 720)):
        ratio = h / w
        if ratio > 1.0:
            new_w, new_h = max(int(size / ratio), 1), size
        else:
            new_w, new_h = size, max(int(size * ratio), 1)
        det_scale = new_h / h
        true_x = new_w / w
        err = size / det_scale - size / true_x   # where the canvas column 640 lands, against where it should
        rows.append((f"{w} x {h}", f"{new_w} x {new_h}", f"{size - new_w} / {size - new_h}", f"{det_scale:.6f}", f"{true_x:.6f}", _f(err, 2)))
    return ("The detector's canvas for five pictures: the picture is resized to keep its aspect ratio so that its longer side is 640 "
            "(the other side is truncated to a whole number of pixels), placed in the top left corner of the 640 by 640 canvas, and the "
            "rest is zero. det_scale is new height / height. The last column is where the canvas column 640 lands in the picture "
            "when it is divided by det_scale, minus where the picture's own horizontal scale (new width / width) puts it.\n\n"
            + _table(["picture", "resized to", "padding right / bottom", "det_scale", "horizontal scale", "error at the right edge, px"], rows))


def normalisations():
    vals = (0, 64, 128, 255)
    rows = []
    for v in vals:
        det = (v - 127.5) / 128.0
        emb = (v - 127.5) / 127.5
        unit = (v / 255.0 - 127.5 / 255.0) / (128.0 / 255.0)
        rows.append((str(v), f"{det:.5f}", f"{emb:.5f}", f"{unit:.5f}"))
    return ("Pixel value to network input, for the detector (subtract 127.5, divide by 128), for the embedder (subtract 127.5, divide "
            f"by 127.5), and for the detector from a picture stored in [0, 1] (subtract 127.5/255, divide by 128/255). The detector "
            f"constants shrink every deviation from the mean by the factor 127.5/128 = {127.5 / 128.0:.5f}.\n\n"
            + _table(["pixel", "detector", "embedder", "detector from [0, 1]"], rows))


def primary_face():
    faces = [("f0", 0.93, (60.0, 40.0, 120.0, 120.0)), ("f1", 0.88, (300.0, 100.0, 440.0, 270.0)), ("f2", 0.71, (500.0, 60.0, 600.0, 160.0))]
    rows = [(n, _f(s, 2), f"{b[2] - b[0]:.0f} x {b[3] - b[1]:.0f}", _int(int((b[2] - b[0]) * (b[3] - b[1])))) for n, s, b in faces]
    by_score = max(faces, key=lambda f: f[1])[0]
    by_area = max(faces, key=lambda f: (f[2][2] - f[2][0]) * (f[2][3] - f[2][1]))[0]
    return (_table(["face", "score", "box size", "area"], rows)
            + f"\n\nThe highest score is {by_score}; the largest area, the face the recognition path embeds by default, is {by_area}.")


# ------------------------------------------------------------------------------------------------ the embedder

IRES_LAYERS = (3, 13, 30, 3)
IRES_CHANNELS = (64, 128, 256, 512)


def iresnet_cost():
    rows, tot_p, tot_m = [], 0, 0
    side = 112
    p, m = _conv(3, 64, 3, side)
    p += 64   # the stem's per-channel PReLU slopes
    tot_p, tot_m = p, m
    rows.append(("stem: one 3x3 convolution and PReLU", "64 x 112 x 112", _int(p), _int(m)))
    cin = 64
    for s in range(4):
        sp = sm = 0
        c = IRES_CHANNELS[s]
        for b in range(IRES_LAYERS[s]):
            stride = 2 if b == 0 else 1
            out = side // stride
            bp = 4 * cin                              # the entry batch normalization: scale, shift, running mean, running variance
            cp, cm = _conv(cin, c, 3, side)            # the first convolution keeps the size
            bp += cp + c                               # and a PReLU slope per channel
            dp, dm = _conv(c, c, 3, out)               # the second convolution strides
            bp += dp
            sm += cm + dm
            if stride != 1 or cin != c:
                ep, em = _conv(cin, c, 1, out)
                bp += ep
                sm += em
            sp += bp
            cin, side = c, out
        tot_p += sp
        tot_m += sm
        rows.append((f"stage {s + 1}: {IRES_LAYERS[s]} blocks", f"{c} x {side} x {side}", _int(sp), _int(sm)))
    hp = 4 * 512 + 512 * 7 * 7 * 512 + 512 + 4 * 512
    hm = 512 * 7 * 7 * 512
    tot_p += hp
    tot_m += hm
    rows.append(("batch norm, flatten, one 25,088 -> 512 linear layer, batch norm", "512", _int(hp), _int(hm)))
    return rows, tot_p, tot_m


def iresnet_table():
    rows, p, m = iresnet_cost()
    rows.append(("total", "", _int(p), _int(m)))
    n_tensors = 3 + 49 * 9 + 4 * 2 + 4 + 2 + 4
    return ("Parameters and multiply-adds of the embedding network on a 112 by 112 input, from the layer list of the text: 49 blocks "
            "(3, 13, 30 and 3 per stage) of an entry batch normalization (4 numbers per channel), two 3x3 convolutions with biases, "
            "a PReLU with one slope per channel, and in the first block of each stage a 1x1 convolution on the shortcut. The block is "
            "pre-activation: entry batch normalization, then the first 3 x 3 convolution (stride 1, costed at the block's input "
            "resolution, from the input width to the stage width), then the PReLU, then the second 3 x 3 convolution (stage width to "
            "stage width, stride 2 in the first block of each stage, costed at the output resolution), then the addition of the shortcut "
            "(identity, or the strided 1 x 1 convolution); no normalization follows the second convolution. Activations "
            "and additions are not counted.\n\n"
            + _table(["part", "output", "parameters", "multiply-adds"], rows, "lrrr")
            + f"\n\nTwice the multiply-adds, the FLOPs, is {_int(2 * m)}. The tensors of the file, counted as the text counts them "
            f"(a block has 9, the four shortcut convolutions 2 each), number {n_tensors}.")


def scale_floor():
    rows = []
    for s in (1, 4, 16, 64):
        row = [str(s)]
        for n in (10, 1000):
            row.append(f"{math.log1p((n - 1) * math.exp(-s)):.6f}")
        rows.append(tuple(row))
    return ("The smallest loss that a classifier on cosines can reach when the sample points exactly at its own centre (cosine 1) and "
            "every one of the other N - 1 centres is orthogonal to it (cosine 0): ln(1 + (N - 1) e^(-s)), for scale s.\n\n"
            + _table(["scale s", "N = 10", "N = 1000"], rows))


def required_angle():
    eps = 0.01
    m_arc = 0.5
    rows = []
    for s in (4, 16, 64):
        for n in (10, 1000):
            c = math.log((n - 1) / math.expm1(eps)) / s
            if c > 1.0:
                rows.append((str(s), str(n), _f(c, 4), "unreachable", "unreachable"))
            else:
                th = math.degrees(math.acos(c))
                rows.append((str(s), str(n), _f(c, 4), _f(th, 1), _f(th - math.degrees(m_arc), 1)))
    return (f"The largest angle between a sample and its own centre at which the loss is at most {eps}, when the other N - 1 centres are "
            "orthogonal to the sample: cos(theta) must reach c = ln((N - 1) / (e^0.01 - 1)) / s. With an additive angular margin "
            f"of m = 0.5 radian ({_f(math.degrees(m_arc), 1)} degrees) the condition is on theta + m instead, which lowers the "
            "angle by exactly m. Angles in degrees.\n\n"
            + _table(["s", "N", "c needed", "largest angle, plain", "largest angle, margin 0.5"], rows))


def margin_target(c, m):
    """The target logit divided by the scale: cos(theta + m) written with the cosine c (the plain, unclamped form)."""
    sn = math.sqrt(max(1.0 - c * c, 0.0))
    return c * math.cos(m) - sn * math.sin(m)


def _softmax_loss(z, y):
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    s = sum(e)
    p = [v / s for v in e]
    return -math.log(p[y]), p


def margin_logits():
    cos = (0.62, 0.35, 0.10, -0.05, -0.30)
    y, s = 0, 16.0
    rows = []
    for m in (0.0, 0.2, 0.5):
        z = [s * c for c in cos]
        z[y] = s * margin_target(cos[y], m)
        loss, p = _softmax_loss(z, y)
        c = cos[y]
        sn = math.sqrt(1 - c * c)
        slope = s * (math.cos(m) + math.sin(m) * c / sn)
        rows.append((_f(m, 1), _f(z[y], 3), _f(max(z[1:]), 3), _f(p[y], 4), _f(loss, 4), _f(s * (p[y] - 1) * (slope / s), 3)))
    # the derivative of the target logit against a central difference
    c0, m0, h = cos[0], 0.5, 1e-6
    num = s * (margin_target(c0 + h, m0) - margin_target(c0 - h, m0)) / (2 * h)
    sn = math.sqrt(1 - c0 * c0)
    ana = s * (math.cos(m0) + math.sin(m0) * c0 / sn)
    rel = abs(num - ana) / abs(ana)
    return ("One sample, five identities, the cosines of the sample's embedding with the five centres are 0.62, 0.35, 0.10, -0.05 "
            "and -0.30, the sample's identity is the first, and the scale is s = 16. The target logit is s cos(theta + m) written as "
            "s (c cos m - sqrt(1 - c^2) sin m); the others are s c. The last column is the derivative of the loss with respect to the "
            "target cosine c, (p - 1) times d(target logit)/dc, with d/dc = s (cos m + sin m c / sqrt(1 - c^2)).\n\n"
            + _table(["margin m", "target logit", "largest other logit", "probability of the target", "loss", "dloss/dc"], rows)
            + f"\n\nCheck of the derivative of the target logit at c = 0.62, m = 0.5: the formula gives {ana:.6f} and the central difference with "
            f"h = 1e-6 in 64-bit arithmetic gives {num:.6f}; the relative difference is below 1e-7"
            + ("." if rel < 1e-7 else f" (it is {rel:.2e}).") + "\n")


def margin_curve_data():
    """(angle in degrees, plain cos, margin cos, clamped) at 1 degree steps for m = 0.5; used by the figure."""
    m = 0.5
    sw = math.pi - m
    th = list(range(0, 181))
    plain = [math.cos(math.radians(t)) for t in th]
    marg = [margin_target(math.cos(math.radians(t)), m) for t in th]
    clamp = [margin_target(c, m) if math.radians(t) <= sw else c - m * math.sin(m) for t, c in zip(th, plain)]
    return th, plain, marg, clamp


def margin_curve():
    m = 0.5
    sw = math.pi - m
    rows = []
    for t in (0, 30, 60, 90, 120, 140, 150, 160, 170, 180):
        c = math.cos(math.radians(t))
        mg = margin_target(c, m)
        cl = margin_target(c, m) if math.radians(t) <= sw else c - m * math.sin(m)
        rows.append((str(t), _f(c, 4), _f(mg, 4), _f(cl, 4)))
    step = (-math.cos(m)) - m * math.sin(m) - (-1.0)
    return ("The target logit divided by the scale, as a function of the angle theta between the sample and its own centre, for the "
            f"margin m = 0.5 (theta + m reaches pi at theta = pi - m = {_f(math.degrees(sw), 2)} degrees): the plain cosine, the margin "
            "form c cos m - sqrt(1 - c^2) sin m, and the margin form with the clamp c - m sin m for theta beyond pi - m.\n\n"
            + _table(["theta, degrees", "cos(theta)", "margin form", "margin form with clamp"], rows)
            + f"\n\nAt theta = pi - m the clamped branch starts at -cos m - m sin m = {_f(-math.cos(m) - m * math.sin(m), 4)}, while the "
            f"margin form has reached -1 there: the clamp steps the target logit down by {_f(-step, 4)} at the switch.")


def margin_slope():
    s, m = 64.0, 0.5
    floor = 1e-6
    rows = []
    for deg in (60.0, 10.0, 1.0, 0.1, 0.01):
        th = math.radians(deg)
        c = math.cos(th)
        sn = max(math.sqrt(max(1 - c * c, 0.0)), floor)
        slope = s * (math.cos(m) + math.sin(m) * c / sn)
        through = slope * math.sin(th)   # the slope against c times dc/dtheta in magnitude
        rows.append((f"{deg:g}", _sci(slope, 3), _sci(s * math.sin(th + m), 3), _sci(through, 3)))
    slope_one = s * (math.cos(m) + math.sin(m) * 1.0 / floor)
    return ("The slope of the target logit against the cosine c, s (cos m + sin m c / sqrt(1 - c^2)), for s = 64 and m = 0.5, "
            "with the square root floored at 1e-6 as in the forward pass, at several angles theta between the sample and its centre. "
            "The plain logit s c has the slope 64 everywhere. The third column is the slope against the angle itself, "
            "s sin(theta + m); the fourth is the slope against c multiplied by sin(theta), the size of dc/dtheta.\n\n"
            + _table(["theta, degrees", "slope against c", "slope against theta", "slope against c times sin(theta)"], rows)
            + f"\n\nAt c = 1 exactly the floor is active. The formula with the floored sine in the denominator gives "
            f"{_sci(slope_one, 3)} there; the forward is s (c cos m - 1e-6 sin m) on that side of the floor, whose slope is "
            f"s cos m = {s * math.cos(m):.2f}.")


def verification_threshold():
    name = "faces.verification"
    g = bookrng.generator(name)
    d, ids, per = 16, 20, 5
    centres = []
    for _ in range(ids):
        v = [g.normal() for _ in range(d)]
        n = math.sqrt(sum(x * x for x in v))
        centres.append([x / n for x in v])
    noise = [[[g.normal() for _ in range(d)] for _ in range(per)] for _ in range(ids)]
    closest = max(sum(centres[i][k] * centres[j][k] for k in range(d)) for i in range(ids) for j in range(i + 1, ids))
    out = []
    thresholds = (0.2, 0.3, 0.4, 0.5)
    for rho in (0.5, 1.0):
        emb = []
        for i in range(ids):
            for j in range(per):
                v = [centres[i][k] + rho * noise[i][j][k] / math.sqrt(d) for k in range(d)]
                n = math.sqrt(sum(x * x for x in v))
                emb.append((i, [x / n for x in v]))
        gen, imp = [], []
        for p in range(len(emb)):
            for q in range(p + 1, len(emb)):
                c = 0.0
                for k in range(d):
                    c += emb[p][1][k] * emb[q][1][k]
                (gen if emb[p][0] == emb[q][0] else imp).append(c)
        out.append((rho, gen, imp))
    rows = []
    for rho, gen, imp in out:
        rows.append((_f(rho, 1), "genuine", _int(len(gen)), _f(sum(gen) / len(gen), 3), _f(min(gen), 3), _f(max(gen), 3)))
        rows.append((_f(rho, 1), "impostor", _int(len(imp)), _f(sum(imp) / len(imp), 3), _f(min(imp), 3), _f(max(imp), 3)))
    err = []
    for rho, gen, imp in out:
        for t in thresholds:
            fr = sum(1 for c in gen if c < t)
            fa = sum(1 for c in imp if c >= t)
            err.append((_f(rho, 1), _f(t, 1), f"{fr} of {len(gen)}", f"{fa} of {len(imp)}"))
    return (f"Simulated embeddings, not a trained network: {ids} identities, {per} photographs each, vectors of width {d}. "
            f"The generator is `{name}` ({_seed(name)}). Draw order: the {ids} centres (identity 0 to 19, 16 standard normals "
            f"each, each divided by its length), then the noise (identity, photograph, 16 standard normals). A photograph of "
            "identity i is its centre plus rho / sqrt(16) times the noise vector, divided by its length. The same noise is used for "
            "both spreads rho. Cosines are dot products of the unit vectors, summed over the 16 components in order. "
            "Genuine pairs are two photographs of one identity, impostor pairs of two identities. The closest two of the "
            f"{ids} centres have the cosine {_f(closest, 3)}.\n\n"
            + _table(["rho", "pairs", "count", "mean cosine", "smallest", "largest"], rows, "llrrrr")
            + "\n\nWrong rejections (genuine pairs below the threshold) and wrong acceptances (impostor pairs at or above it):\n\n"
            + _table(["rho", "threshold", "genuine rejected", "impostors accepted"], err, "lrrr"))


# ------------------------------------------------------------------------------------------------ the parser

BG_CLASSES = (0, 16, 18, 7, 8, 9, 14, 15)


def template_comparison():
    m = similarity_fit(TEMPLATE_512, TEMPLATE_112)
    al = [similarity_apply(m, p) for p in TEMPLATE_512]
    eye = lambda t: math.hypot(t[1][0] - t[0][0], t[1][1] - t[0][1])  # noqa: E731
    mouth = lambda t: math.hypot((t[3][0] + t[4][0]) / 2 - (t[0][0] + t[1][0]) / 2, (t[3][1] + t[4][1]) / 2 - (t[0][1] + t[1][1]) / 2)  # noqa: E731
    rows = [("crop side, pixels", "112", "512"),
            ("distance between the eye centres, pixels", _f(eye(TEMPLATE_112), 2), _f(eye(TEMPLATE_512), 2)),
            ("that distance as a share of the side", _f(eye(TEMPLATE_112) / 112, 4), _f(eye(TEMPLATE_512) / 512, 4)),
            ("eye midpoint to mouth midpoint, as a share of the side", _f(mouth(TEMPLATE_112) / 112, 4), _f(mouth(TEMPLATE_512) / 512, 4))]
    return ("The recognition template is the five points printed in the text; the parsing template, in pixel-centre coordinates of "
            f"the 512-pixel crop, is {', '.join(_pt(p, 5) for p in TEMPLATE_512)} (the brain constant `FFHQ_DST_512`, the same five "
            "points in the same order). The two distances are the Euclidean distance between the left and right eye points, and the "
            "last row the distance from the midpoint of the eyes to the midpoint of the mouth corners.\n\n"
            + _table(["", "recognition template", "parsing template"], rows, "lrr")
            + f"\n\nThe least-squares similarity that takes the parsing template onto the recognition template has scale "
            f"{math.hypot(m[0], m[1]):.4f} (a plain shrink of 512 to 112 would be {112 / 512:.4f}), angle "
            f"{_f(math.degrees(math.atan2(m[1], m[0])), 4)} degrees and a residual of {_f(rms_distance(al, TEMPLATE_112), 3)} pixels.")


def whiten_and_gray(classes, rgb):
    out = []
    for cls, (r, g, b) in zip(classes, rgb):
        if cls in BG_CLASSES:
            out.append((1.0, 1.0, 1.0))
        else:
            v = 0.299 * r + 0.587 * g + 0.114 * b
            out.append((v, v, v))
    return out


def parsing_mask():
    classes = [0, 17, 1, 10, 16, 14, 1, 11, 18, 15, 12, 1]
    rgb = [(0.20, 0.40, 0.80), (0.10, 0.08, 0.05), (0.85, 0.62, 0.50), (0.78, 0.55, 0.44), (0.30, 0.30, 0.60), (0.80, 0.60, 0.50),
           (0.88, 0.65, 0.52), (0.60, 0.20, 0.20), (0.15, 0.15, 0.15), (0.90, 0.80, 0.30), (0.65, 0.25, 0.25), (0.86, 0.63, 0.51)]
    out = whiten_and_gray(classes, rgb)
    rows = []
    for i, (cls, c, o) in enumerate(zip(classes, rgb, out)):
        rows.append((str(i), str(cls), "whitened" if cls in BG_CLASSES else "kept", "(" + ", ".join(_f(v, 2) for v in c) + ")",
                     "(" + ", ".join(_f(v, 4) for v in o) + ")"))
    return ("A picture of 12 pixels (3 by 4, row by row) after parsing: the class of each pixel, the pixel's colour in [0, 1], and the "
            f"masked colour. The classes {sorted(BG_CLASSES)} are whitened, the others become their gray value "
            "0.299 R + 0.587 G + 0.114 B on all three channels.\n\n"
            + _table(["pixel", "class", "treatment", "colour", "masked colour"], rows, "rllll"))
