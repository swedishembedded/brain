# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume VII project "A Vision Pipeline": detect, segment and track objects through a clip.

Plain Python 3, standard library only, written from the project's own description. The libraries of the earlier chapters are
imported, never copied: the box, IoU, class-aware suppression and the letterbox with its inverse (`detection`, Chapter 4), the
bilinear resize and the ImageNet constants (`images`, Chapter 1), and the memory-slot rule and the FLOPs of the SAM 2 trunk, memory
attention, decoder and memory encoder (`segment_anything`, Chapter 7). What this module writes is what makes them one pipeline:

* a deterministic clip of three coloured rectangles (two of one class, one of another) on a noisy background, with the exact
  visible mask and box of every object in every frame, one object fully hidden for two frames;
* the two views of a frame that the two models read (a letterboxed byte image for the detector, a stretched and normalized tensor
  for the segmenter) and the maps between their coordinates and the frame's;
* a detector stand-in with the interface of an anchor-free head (distances from a point, a class score) computed from the pixels of
  the letterboxed view, then confidence filter, suppression and the inverse letterbox;
* a promptable-segmenter stand-in (a box prompt, a seed, region growing) computed on the stretched view;
* a tracker stand-in with the memory bank of Chapter 7 (slots, an attention over the cells of the remembered frames, an object
  score, an absent memory) and the cost of the real models for the same schedule.

The three stand-ins replace trained networks and nothing else: every interface, every coordinate map and every rule around them
is the pipeline's. Arithmetic is 64-bit floating point; random inputs come from `bookrng.generator(name)` and nothing is trained,
so every printed number is exact arithmetic of its inputs. Sums run in the order the text states.

Swedish Embedded AB implements solutions for video pipelines that detect, segment and track objects with counted cost. If your
team needs expertise in vision pipelines, you can procure our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import math

import bookrng
import detection as det
import images
import segment_anything as sam

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fx(v, d=3):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _box(b, d=2):
    return "(" + ", ".join(_fx(v, d) for v in b) + ")"


def _ibox(b):
    return "(" + ", ".join(str(v) for v in b) + ")"


# ------------------------------------------------------------------------------------------------ the clip

W, H, T = 96, 61, 12            # frame width, height, number of frames
SIZE = 64                       # side of both model inputs
BACKGROUND = (40, 44, 48)
NOISE = 4                       # every sample gets integer(NOISE) added
SCENE = "project_vision_pipeline.scene"
# name, class, colour, (width, height), y0, x0 at frame t as (x_start, x_step), drawn in this order (later ones on top)
OBJECTS = (
    ("A", 0, (200, 60, 50), (10, 8), 22, (8, 4)),
    ("C", 0, (200, 60, 50), (10, 8), 42, (76, -4)),
    ("B", 1, (60, 180, 90), (22, 16), 14, (70, -5)),
)
NAMES = [o[0] for o in OBJECTS]


def object_box(k, t):
    name, cls, colour, (w, h), y0, (xs, dx) = OBJECTS[k]
    x0 = xs + dx * t
    return (x0, y0, x0 + w, y0 + h)


def make_clip():
    """frames[t][y][x] = (r, g, b) as ints; labels[t][y][x] = 0 for background, 1 + k for object k (the last drawn on top).

    Draw order: for each frame, each row, each column, each channel in r, g, b order, one `integer(4)`."""
    g = bookrng.generator(SCENE)
    frames, labels = [], []
    for t in range(T):
        lab = [[0] * W for _ in range(H)]
        for k in range(len(OBJECTS)):
            x1, y1, x2, y2 = object_box(k, t)
            for y in range(y1, y2):
                for x in range(x1, x2):
                    lab[y][x] = 1 + k
        fr = []
        for y in range(H):
            row = []
            for x in range(W):
                base = BACKGROUND if lab[y][x] == 0 else OBJECTS[lab[y][x] - 1][2]
                row.append(tuple(base[c] + g.integer(NOISE) for c in range(3)))
            fr.append(row)
        frames.append(fr)
        labels.append(lab)
    return frames, labels


_CLIP = None


def clip():
    global _CLIP
    if _CLIP is None:
        _CLIP = make_clip()
    return _CLIP


def truth(t):
    """For each object: (visible mask as a set of (y, x), visible box or None, full box)."""
    frames, labels = clip()
    out = []
    for k in range(len(OBJECTS)):
        pix = {(y, x) for y in range(H) for x in range(W) if labels[t][y][x] == 1 + k}
        if pix:
            ys = [p[0] for p in pix]
            xs = [p[1] for p in pix]
            vis = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        else:
            vis = None
        out.append((pix, vis, object_box(k, t)))
    return out


def frame_digest(t):
    """SHA-256 of the frame's bytes: rows top to bottom, columns left to right, r, g, b."""
    frames, _ = clip()
    data = bytes(v for row in frames[t] for px in row for v in px)
    return hashlib.sha256(data).hexdigest()


def clip_table():
    frames, labels = clip()
    rows = []
    for t in range(T):
        tr = truth(t)
        cells = []
        for k in range(len(OBJECTS)):
            pix, vis, full = tr[k]
            cells.append(f"{len(pix)} / {_ibox(vis) if vis else 'hidden'}")
        rows.append([t] + cells + [frame_digest(t)[:12]])
    seed = bookrng.seed_of(SCENE)
    return (f"seed 0x{seed:016x} ({seed}), generator `{SCENE}`. The clip has {T} frames of {W} by {H} pixels. Each object is a "
            "solid rectangle painted over a background of (40, 44, 48); objects are painted in the order A, C, B, so B covers A and C "
            "where they meet. Every one of the 3 * 61 * 96 samples of a frame, in the order frame, row, column, channel, gets one "
            "`integer(4)` added to the base colour of its pixel. Rectangles: A is 10 by 8 at y = 22 with x = 8 + 4t, colour "
            "(200, 60, 50), class 0; C is 10 by 8 at y = 42 with x = 76 - 4t, the same colour, class 0; B is 22 by 16 at y = 14 with "
            "x = 70 - 5t, colour (60, 180, 90), class 1. Each cell shows the number of visible pixels and the box (x1, y1, x2, y2) "
            "of the visible pixels. The last column is the first 12 hex digits of the SHA-256 of the frame's 17,568 bytes.\n\n"
            + _table(["t", "A (class 0)", "C (class 0)", "B (class 1)", "SHA-256 of the frame"], rows, "rlllr"))


# ------------------------------------------------------------------------------------------------ the two views

PAD = 114.0 / 255.0
CLASS_COLOURS = (BACKGROUND, OBJECTS[0][2], OBJECTS[2][2])


def letterbox_plan():
    """Scale, content size and integer padding of the detector's view. The padding is the floor of half the slack, for pixels and
    for boxes alike."""
    s = min(SIZE / W, SIZE / H)
    nw, nh = images.half_up(W * s), images.half_up(H * s)
    return s, nw, nh, (SIZE - nw) // 2, (SIZE - nh) // 2


def src_index(o, s, n_in):
    """Source index of content pixel o: the nearest rule, round half up of (o + 0.5) / s - 0.5, clamped to the image."""
    return min(max(images.half_up((o + 0.5) / s - 0.5), 0), n_in - 1)


def detector_view(t):
    """Letterboxed [3][64][64] floats in [0, 1]: byte / 255, padding 114 / 255, content pasted at (pad_x, pad_y)."""
    frames, _ = clip()
    s, nw, nh, px, py = letterbox_plan()
    out = [[[PAD] * SIZE for _ in range(SIZE)] for _ in range(3)]
    for yo in range(nh):
        ys = src_index(yo, s, H)
        for xo in range(nw):
            xs = src_index(xo, s, W)
            for c in range(3):
                out[c][py + yo][px + xo] = frames[t][ys][xs][c] / 255.0
    return out


def segmenter_view(t):
    """Stretched [3][64][64] floats: each channel's byte / 255 plane resized by the half-pixel bilinear rule of Chapter 1 to 64 by
    64 (each axis on its own scale), then (x - mean) / std with the ImageNet constants."""
    frames, _ = clip()
    out = []
    for c in range(3):
        plane = [[frames[t][y][x][c] / 255.0 for x in range(W)] for y in range(H)]
        r = images.resize2d(plane, SIZE, SIZE, "half")
        m, sd = images.IMAGENET_MEAN[c], images.IMAGENET_STD[c]
        out.append([[(v - m) / sd for v in row] for row in r])
    return out


def to_detector(b):
    s, _, _, px, py = letterbox_plan()
    return (b[0] * s + px, b[1] * s + py, b[2] * s + px, b[3] * s + py)


def from_detector(b):
    s, _, _, px, py = letterbox_plan()
    cl = lambda v, hi: min(max(v, 0.0), hi)
    return (cl((b[0] - px) / s, W), cl((b[1] - py) / s, H), cl((b[2] - px) / s, W), cl((b[3] - py) / s, H))


def to_segmenter(b):
    sx, sy = SIZE / W, SIZE / H
    return (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)


def views_table():
    s, nw, nh, px, py = letterbox_plan()
    v = detector_view(0)
    sv = segmenter_view(0)
    data = bytes(int(images.half_up(v[c][y][x] * 255)) for y in range(SIZE) for x in range(SIZE) for c in range(3))
    sdig = hashlib.sha256(data).hexdigest()[:12]
    ys = [src_index(o, s, H) for o in range(nh)]
    xs_first = [src_index(o, s, W) for o in range(8)]
    pad_rows = [y for y in range(SIZE) if not (py <= y < py + nh)]
    sumd = sum(v[c][y][x] for c in range(3) for y in range(SIZE) for x in range(SIZE))
    sums = sum(sv[c][y][x] for c in range(3) for y in range(SIZE) for x in range(SIZE))
    rows = [
        ["scale min(64/96, 64/61)", _fx(s, 6), "detector view"],
        ["content size (round half up)", f"{nw} x {nh}", "detector view"],
        ["padding (pad_x, pad_y) = floor(slack / 2)", f"({px}, {py})", "detector view"],
        ["padding rows", f"0 to {py - 1} and {py + nh} to {SIZE - 1} ({len(pad_rows)} rows)", "detector view"],
        ["source rows of the first 6 content rows", str(ys[:6]), "detector view"],
        ["source columns of the first 8 content columns", str(xs_first), "detector view"],
        ["SHA-256 (12 hex) of the 12,288 bytes round(255 v), CHW order r, g, b planes of rows then columns, as H W C", sdig, "detector view"],
        ["sum of the 12,288 values (tolerance 1e-9)", _fx(sumd, 6), "detector view"],
        ["scale per axis (64/96, 64/61)", f"({_fx(SIZE / W, 6)}, {_fx(SIZE / H, 6)})", "segmenter view"],
        ["sum of the 12,288 normalized values (tolerance 1e-9)", _fx(sums, 6), "segmenter view"],
        ["normalized value, channel 0, row 0, column 0", _fx(sv[0][0][0], 6), "segmenter view"],
        ["normalized value, channel 2, row 31, column 40", _fx(sv[2][31][40], 6), "segmenter view"],
    ]
    lead = ("Frame 0 of the clip, read two ways. The detector's view: scale = min(64/96, 64/61); content size round half up of "
            "(96 * scale, 61 * scale); padding floor((64 - content) / 2) per axis; content pixel (yo, xo) copies the frame pixel "
            "(floor((yo + 0.5) / scale), floor((xo + 0.5) / scale)) (the nearest rule, round half up of (o + 0.5) / scale - 0.5); values byte / 255, "
            "padding 114 / 255, no normalization. The segmenter's view: each channel's byte / 255 plane resized to 64 by 64 by the "
            "half-pixel bilinear rule of Chapter 1, then (x - mean) / std with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225). "
            "Sums run over channel, row, column with one accumulator.")
    return lead + "\n\n" + _table(["quantity", "value", "view"], rows, "llr")


def box_maps_table():
    tr = truth(0)
    rows = []
    for k in range(len(OBJECTS)):
        b = tr[k][1]
        d = to_detector(b)
        sg = to_segmenter(b)
        back = from_detector(d)
        rows.append([NAMES[k], _ibox(b), _box(d), _box(sg), _box(back)])
    lead = ("The visible box of each object on frame 0 (frame pixels, x1 <= x < x2, y1 <= y < y2) carried into the detector's input "
            "(x s + pad_x, y s + pad_y with the integer padding) and into the segmenter's input (x 64/96, y 64/61); the last column maps the "
            "detector's box back by the inverse letterbox of Chapter 4.")
    return lead + "\n\n" + _table(["object", "box in the frame", "in the detector's input", "in the segmenter's input", "detector input mapped back"], rows, "llllr")


# ------------------------------------------------------------------------------------------------ the detector stand-in

STRIDE = 2
GRID = SIZE // STRIDE
CONF = 0.25
NMS_IOU = 0.45
MAX_DET = 300
REL_ERR = 0.06


def class_map(view):
    """Class of every pixel of the detector's view: the nearest of four prototype colours, in squared Euclidean distance on values in
    [0, 1]; a tie goes to the lower class. Class 0 is the background, 1 the red of A and C, 2 the green of B (each the base colour
    plus 1.5, the mean of integer(4), over 255) and 3 the padding grey 114 / 255. Classes 0 and 3 are not objects."""
    protos = [tuple((c + 1.5) / 255.0 for c in col) for col in CLASS_COLOURS] + [(PAD, PAD, PAD)]
    cm = []
    for y in range(SIZE):
        row = []
        for x in range(SIZE):
            best, bd = 0, None
            for k, p in enumerate(protos):
                d = sum((view[c][y][x] - p[c]) ** 2 for c in range(3))
                if bd is None or d < bd:
                    best, bd = k, d
            row.append(best)
        cm.append(row)
    return cm


def head_candidates(view):
    """The raw output of the stand-in head: one candidate per cell of the 32 by 32 grid whose anchor pixel (2j, 2i) is not
    background. Returns a list of dicts in row-major cell order (i, then j)."""
    cm = class_map(view)
    out = []
    for i in range(GRID):
        for j in range(GRID):
            ay, ax = STRIDE * i, STRIDE * j
            k = cm[ay][ax]
            if k in (0, 3):
                continue
            x1 = ax
            while x1 > 0 and cm[ay][x1 - 1] == k:
                x1 -= 1
            x2 = ax + 1
            while x2 < SIZE and cm[ay][x2] == k:
                x2 += 1
            y1 = ay
            while y1 > 0 and cm[y1 - 1][ax] == k:
                y1 -= 1
            y2 = ay + 1
            while y2 < SIZE and cm[y2][ax] == k:
                y2 += 1
            pxc, pyc = ax + 0.5, ay + 0.5
            d = [pxc - x1, pyc - y1, x2 - pxc, y2 - pyc]          # l, t, r, b: exact distances
            sgn = [1 if (i + j + s) % 2 == 0 else -1 for s in range(4)]
            dp = [d[s] * (1.0 + REL_ERR * sgn[s]) for s in range(4)]
            box = (pxc - dp[0], pyc - dp[1], pxc + dp[2], pyc + dp[3])
            votes = total = 0
            for yy in range(max(ay - 1, 0), min(ay + 2, SIZE)):
                for xx in range(max(ax - 1, 0), min(ax + 2, SIZE)):
                    total += 1
                    votes += cm[yy][xx] == k
            cen = math.sqrt(min(dp[0], dp[2]) / max(dp[0], dp[2]) * min(dp[1], dp[3]) / max(dp[1], dp[3]))
            out.append({"cell": (i, j), "cls": k - 1, "box": box, "score": votes / total * cen, "d": d, "dp": dp, "votes": (votes, total), "cen": cen})
    return out


def head_example():
    """The nine cells of object A on frame 0, as the stand-in head emits them."""
    cands = [c for c in head_candidates(detector_view(0)) if c["cls"] == 0 and c["box"][0] < 30]
    rows = []
    for c in cands:
        d = c["d"]
        rows.append([f"({c['cell'][0]}, {c['cell'][1]})", _box(d, 1), _box(c["dp"], 3), _box(c["box"], 3), f"{c['votes'][0]} of {c['votes'][1]}",
                     _fx(c["cen"], 3), _fx(c["score"], 3), "yes" if c["score"] >= CONF else "no"])
    lead = ("The head's output for the nine cells of grid row 13 to 15 and column 3 to 5, whose anchor pixels (2j, 2i) lie on or at the edge of "
            "object A on frame 0. True l, t, r, b are the distances from the anchor point (2j + 0.5, 2i + 0.5) to the ends of the run of "
            f"pixels of A's class along the anchor's row and column. Predicted distances multiply each by 1 + {REL_ERR} sign, with sign +1 when "
            "i + j + s is even for the side index s = 0, 1, 2, 3 (l, t, r, b) and -1 otherwise. Box: the anchor point minus (l, t) and plus "
            "(r, b), in the detector's input. Votes: pixels of A's class among the 3 by 3 pixels around the anchor pixel. Centerness: "
            "sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b)) of the predicted distances. Score: votes / 9 * centerness.")
    return lead + "\n\n" + _table(["cell (i, j)", "true l, t, r, b", "predicted l, t, r, b", "box (x1, y1, x2, y2)", "votes", "centerness", "score", "kept at 0.25"],
                  rows, "lllllrrr")


def detect(t, conf=CONF, thr=NMS_IOU, class_aware=True, bgr=False, inverse=None):
    """Candidates -> confidence filter (score >= conf) -> class-aware suppression -> inverse letterbox.
    Returns (n_raw, n_filtered, kept list of (box in frame, score, class), candidate list). `bgr` and `inverse` plant mistakes."""
    view = detector_view(t)
    cands = head_candidates(view[::-1] if bgr else view)
    inverse = inverse or from_detector
    kept_in = [c for c in cands if c["score"] >= conf]
    dets = [(n, c["box"], c["score"], c["cls"]) for n, c in enumerate(kept_in)]
    ids, _ = det.nms(dets, thr, class_aware, MAX_DET)
    out = [(inverse(kept_in[n]["box"]), kept_in[n]["score"], kept_in[n]["cls"]) for n in ids]
    return len(cands), len(kept_in), out, cands


def match_detections(t, dets):
    """Greedy match by falling score of each detection to the unmatched visible ground-truth box of its class with the highest IoU,
    at least 0.5. Returns (tp pairs (det index, object index, iou), false positives, missed objects)."""
    tr = truth(t)
    taken = set()
    pairs, fps = [], []
    for n in sorted(range(len(dets)), key=lambda n: (-dets[n][1], n)):
        best, bi = None, 0.0
        for k in range(len(OBJECTS)):
            if k in taken or tr[k][1] is None or OBJECTS[k][1] != dets[n][2]:
                continue
            v = det.iou(dets[n][0], tr[k][1])
            if v >= 0.5 and v > bi:
                best, bi = k, v
        if best is None:
            fps.append(n)
        else:
            taken.add(best)
            pairs.append((n, best, bi))
    missed = [k for k in range(len(OBJECTS)) if tr[k][1] is not None and k not in taken]
    return pairs, fps, missed


def detector_table():
    rows = []
    for t in range(T):
        n_raw, n_f, dets, _ = detect(t)
        pairs, fps, missed = match_detections(t, dets)
        mean_iou = sum(p[2] for p in pairs) / len(pairs) if pairs else 0.0
        rows.append([t, n_raw, n_f, len(dets), len(pairs), len(fps), "".join(NAMES[k] for k in missed) or "-", _fx(mean_iou, 3)])
    lead = (f"The detector of milestone 2 on every frame: confidence threshold {CONF}, class-aware suppression at IoU > {NMS_IOU}, at most "
            f"{MAX_DET} detections. A detection is matched to the unmatched visible object of its class with the largest IoU, at least 0.5; "
            "detections are visited by falling score. Raw candidates are the cells whose anchor is not background or padding.")
    return lead + "\n\n" + _table(["t", "raw candidates", "score >= 0.25", "kept by NMS", "matched", "false positives", "missed", "mean IoU of matches"], rows)


# ------------------------------------------------------------------------------------------------ the segmenter stand-in

TOL = 1.0
MARGIN = 2.0


def colour_dist(a, b):
    return math.sqrt(sum((a[c] - b[c]) ** 2 for c in range(3)))


def pixel(view, y, x):
    return (view[0][y][x], view[1][y][x], view[2][y][x])


def segment_box(view, box_src, tol=TOL, margin=MARGIN, scale_map=None):
    """Box prompt (in frame pixels) -> mask of the segmenter's input as a set of (y, x): the connected component, through
    4-neighbours, of the pixels whose centre lies in the box (mapped to the input, grown by `margin` pixels) and whose colour is
    within `tol` (Euclidean, on the normalized values) of the colour at the seed, the pixel containing the box's centre."""
    b = (scale_map or to_segmenter)(box_src)
    sx, sy = int(math.floor((b[0] + b[2]) / 2)), int(math.floor((b[1] + b[3]) / 2))
    if not (0 <= sx < SIZE and 0 <= sy < SIZE):
        return set()
    ref = pixel(view, sy, sx)

    def ok(y, x):
        return (b[0] - margin <= x + 0.5 <= b[2] + margin and b[1] - margin <= y + 0.5 <= b[3] + margin
                and colour_dist(pixel(view, y, x), ref) <= tol)

    if not ok(sy, sx):
        return set()
    seen = {(sy, sx)}
    stack = [(sy, sx)]
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < SIZE and 0 <= xx < SIZE and (yy, xx) not in seen and ok(yy, xx):
                seen.add((yy, xx))
                stack.append((yy, xx))
    return seen


def mask_to_frame(mask):
    """Nearest rule from the frame to the input: frame pixel (y, x) takes the input pixel (floor((y + 0.5) * 64 / 61),
    floor((x + 0.5) * 64 / 96)) clamped to 63. Returns the set of frame pixels (y, x)."""
    out = set()
    for y in range(H):
        my = min(int(math.floor((y + 0.5) * SIZE / H)), SIZE - 1)
        for x in range(W):
            mx = min(int(math.floor((x + 0.5) * SIZE / W)), SIZE - 1)
            if (my, mx) in mask:
                out.add((y, x))
    return out


def mask_iou(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def frame0_prompts():
    """The detections of frame 0 in the order of falling score, matched to the objects: [(object index, box, score, class)]."""
    _, _, dets, _ = detect(0)
    pairs, fps, missed = match_detections(0, dets)
    by_det = {p[0]: p[1] for p in pairs}
    out = []
    for n in sorted(range(len(dets)), key=lambda n: (-dets[n][1], n)):
        out.append((by_det.get(n), dets[n][0], dets[n][1], dets[n][2]))
    return out


def segmentation_table():
    view = segmenter_view(0)
    tr = truth(0)
    rows = []
    for k, box, score, cls in frame0_prompts():
        m = segment_box(view, box)
        mf = mask_to_frame(m)
        bad = segment_box(view, box, scale_map=to_detector)
        badf = mask_to_frame(bad)
        rows.append([NAMES[k], _box(box), len(m), len(mf), len(tr[k][0]), _fx(mask_iou(mf, tr[k][0]), 3),
                     len(badf), _fx(mask_iou(badf, tr[k][0]), 3)])
    lead = (f"The box prompts of frame 0 (the kept detections, in frame pixels) run through the segmenter of milestone 3: tolerance {TOL} "
            f"on the colour distance, box grown by {MARGIN:g} input pixels. The last two columns carry the prompt to the segmenter by the "
            "detector's letterbox map instead of the per-axis scale, which is the mistake of confusing the two views.")
    return lead + "\n\n" + _table(["object", "prompt box", "pixels in the input", "pixels in the frame", "true pixels", "IoU with the truth",
                   "pixels with the letterbox map", "IoU with the letterbox map"], rows)


# ------------------------------------------------------------------------------------------------ the tracker stand-in

CELL = 4
CG = SIZE // CELL               # 16
TAU = 0.25
RHO = 1.5
ON = 0.5
REF_TOL = 1.0
NULL_SCORE = -4.0


def cell_features(view):
    """[256][3]: the mean of the normalized values over each 4 by 4 block, summed row-major within the block, cells row-major."""
    feats = []
    for cy in range(CG):
        for cx in range(CG):
            acc = [0.0, 0.0, 0.0]
            for y in range(CELL * cy, CELL * cy + CELL):
                for x in range(CELL * cx, CELL * cx + CELL):
                    for c in range(3):
                        acc[c] += view[c][y][x]
            feats.append([a / (CELL * CELL) for a in acc])
    return feats


def mask_cells(mask):
    """[256]: the fraction of each 4 by 4 block that the mask (a set of (y, x) of the input) covers."""
    cnt = [0] * (CG * CG)
    for (y, x) in mask:
        cnt[(y // CELL) * CG + (x // CELL)] += 1
    return [c / (CELL * CELL) for c in cnt]


def read_memory(feats, slabs, rho=RHO, null_score=NULL_SCORE):
    """Memory attention of the stand-in. slabs: list of (features [256][3], mask channel [256]) in slot order. For every cell q
    of the current frame: weights w_k = exp(-|f_q - f_k|^2 / TAU - (dx^2 + dy^2) / (2 RHO^2)) over all cells k of all slabs;
    plus one key that holds nothing (score NULL_SCORE, mask value 0); R(q) = sum(w_k m_k) / (exp(NULL_SCORE) + sum(w_k)).
    Returns the 256 read-backs."""
    out = []
    z0 = math.exp(null_score)
    for q in range(CG * CG):
        qy, qx = divmod(q, CG)
        num, den = 0.0, z0
        fq = feats[q]
        for sf, sm in slabs:
            for k in range(CG * CG):
                ky, kx = divmod(k, CG)
                d2 = (fq[0] - sf[k][0]) ** 2 + (fq[1] - sf[k][1]) ** 2 + (fq[2] - sf[k][2]) ** 2
                w = math.exp(-d2 / TAU - ((qx - kx) ** 2 + (qy - ky) ** 2) / (2 * rho * rho))
                num += w * sm[k]
                den += w
        out.append(num / den)
    return out


def dilate(on):
    """3 by 3 dilation of a [256] 0/1 list on the 16 by 16 cell grid."""
    out = [0] * (CG * CG)
    for c in range(CG * CG):
        if on[c]:
            cy, cx = divmod(c, CG)
            for yy in range(max(cy - 1, 0), min(cy + 2, CG)):
                for xx in range(max(cx - 1, 0), min(cx + 2, CG)):
                    out[yy * CG + xx] = 1
    return out


def mean_colour(view, mask):
    """Mean normalized colour over a mask; the sum runs over the mask's pixels in row-major order."""
    acc = [0.0, 0.0, 0.0]
    for (y, x) in sorted(mask):
        p = pixel(view, y, x)
        for c in range(3):
            acc[c] += p[c]
    return [a / len(mask) for a in acc]


def track_clip(prompt_masks, n_recent=6, rho=RHO):
    """Follow each prompt mask (a set of input pixels on frame 0) through the clip. Returns per track a list over frames of
    (present, mask set, max read-back, margin = min |R - ON| over the cells)."""
    views = [segmenter_view(t) for t in range(T)]
    feats = [cell_features(v) for v in views]
    tracks = []
    for m0 in prompt_masks:
        ref = mean_colour(views[0], m0)
        bank = {0: (feats[0], mask_cells(m0))}
        res = [(True, set(m0), 1.0, None, None)]
        for t in range(1, T):
            slots = sam.frame_memory(t, cond=(0,), num_frames=T)[0]
            slots = [(tp, src) for tp, src in slots if tp == 0 or tp > 6 - n_recent]
            slabs = [bank[src] for tp, src in slots]
            r = read_memory(feats[t], slabs, rho)
            on = [1 if v >= ON else 0 for v in r]
            present = any(on)
            margin = min(abs(v - ON) for v in r)
            mask = set()
            if present:
                dil = dilate(on)
                for cy in range(CG):
                    for cx in range(CG):
                        if dil[cy * CG + cx]:
                            for y in range(CELL * cy, CELL * cy + CELL):
                                for x in range(CELL * cx, CELL * cx + CELL):
                                    if colour_dist(pixel(views[t], y, x), ref) <= REF_TOL:
                                        mask.add((y, x))
            bank[t] = (feats[t], mask_cells(mask) if present else [0.0] * (CG * CG))
            res.append((present, mask, max(r), margin, r))
        tracks.append(res)
    return tracks


_TRACKED = {}


def pipeline_run(n_recent=6, rho=RHO):
    """Detect on frame 0, segment each detection, track all. Returns (prompts, per-track results)."""
    key = (n_recent, rho)
    if key not in _TRACKED:
        view0 = segmenter_view(0)
        prompts = frame0_prompts()
        masks = [segment_box(view0, p[1]) for p in prompts]
        _TRACKED[key] = (prompts, track_clip(masks, n_recent, rho))
    return _TRACKED[key]


def tracking_table(**kw):
    prompts, tracks = pipeline_run(**kw)
    rows = []
    for t in range(T):
        tr = truth(t)
        cells = []
        for (k, box, score, cls), res in zip(prompts, tracks):
            present, mask, mx, margin, _r = res[t]
            mf = mask_to_frame(mask)
            true_pix = tr[k][0]
            if t == 0:
                cells.append(f"prompt  {len(mf)} / {len(true_pix)}  {_fx(mask_iou(mf, true_pix), 3)}")
            elif not present and not true_pix:
                cells.append(f"{_fx(mx, 3)}  absent  correct")
            else:
                cells.append(f"{_fx(mx, 3)}  {len(mf)} / {len(true_pix)}  {_fx(mask_iou(mf, true_pix), 3)}")
        rows.append([t] + cells)
    lead = (f"The three tracks over the clip with the full bank. Read-back is the largest cell read-back of the frame, tau = {TAU:g}, "
            f"rho = {RHO:g} cells, a no-object key of score {NULL_SCORE:g}; a cell is on at 0.5 or more; the mask is the pixels of the "
            f"dilated on-cells within {REF_TOL:g} of the track's reference colour. Pixels / true is the mask on the frame against "
            "the visible pixels of the object; IoU is of those two sets. Frame 0 is the prompt.")
    return lead + "\n\n" + _table(["t"] + [f"track {NAMES[p[0]]}: read-back, pixels / true, IoU" for p in prompts], rows, "rlll")


# ------------------------------------------------------------------------------------------------ checks that tie the stages together


def nms_stats(t, thr):
    """(kept, IoU evaluations, number of candidates entering suppression) of class-aware suppression of frame t at threshold thr."""
    cands = [c for c in head_candidates(detector_view(t)) if c["score"] >= CONF]
    dets = [(n, c["box"], c["score"], c["cls"]) for n, c in enumerate(cands)]
    ids, trace = det.nms(dets, thr, True, MAX_DET)
    return len(ids), sum(len(e) for _, e, _ in trace), len(dets)


def nms_sweep():
    rows = []
    for thr in (0.3, 0.45, 0.7, 0.8, 0.9, 0.99):
        kept, evals, n = nms_stats(0, thr)
        rows.append([f"{thr:g}", n, kept, evals])
    lead = (f"Class-aware suppression of the candidates of frame 0 whose score is at least {CONF}, at several thresholds (a candidate is "
            "dropped when its IoU with a kept box of its class is strictly greater than the threshold). IoU evaluations are the pairs "
            "compared, stopping at the first drop.")
    return lead + "\n\n" + _table(["threshold", "candidates entering", "kept", "IoU evaluations"], rows)


def box_of(mask_frame):
    ys = [q[0] for q in mask_frame]
    xs = [q[1] for q in mask_frame]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def monitor_data():
    """Per frame and track: the largest IoU between the box of the tracked mask (in frame pixels) and a detection of the track's
    class, or None when the track reports the object absent."""
    prompts, tracks = pipeline_run()
    out = []
    for t in range(T):
        _, _, dets, _ = detect(t)
        row = []
        for (k, b, sc, cl), res in zip(prompts, tracks):
            present, mask, mx, mg, _r = res[t]
            if not present:
                row.append(None)
                continue
            bb = box_of(mask_to_frame(mask))
            row.append(max([det.iou(bb, d[0]) for d in dets if d[2] == cl] or [0.0]))
        out.append(row)
    return out


def bank_table():
    rows = []
    for t in (1, 2, 6, 7, 8, 9, 10, 11):
        slabs, ptrs = sam.frame_memory(t, cond=(0,), num_frames=T)
        rows.append([t, ", ".join(str(s) for _, s in slabs), ", ".join(str(tp) for tp, _ in slabs), len(slabs) * CG * CG])
    lead = ("The slabs that the stand-in's tracker reads at some frames of the 12-frame clip, prompted on frame 0 (the rule of Chapter 7 "
            "with 7 slots; each slab holds the 256 cells of a 16 by 16 grid).")
    return lead + "\n\n" + _table(["frame t", "source frames of the slabs", "slots", "keys of the stand-in's attention"], rows, "rlll")


def variant_summary(**kw):
    """Per track: the 12-character presence string, and the mean IoU over the frames where the object is visible."""
    prompts, tracks = pipeline_run(**kw)
    out = []
    for (k, b, sc, cl), res in zip(prompts, tracks):
        pres = "".join("1" if r[0] else "0" for r in res)
        ious = []
        for t in range(T):
            tp = truth(t)[k][0]
            if tp:
                ious.append(mask_iou(mask_to_frame(res[t][1]), tp))
        out.append((NAMES[k], pres, sum(ious) / len(ious)))
    return out


def ablation_table():
    rows = []
    for label, kw in (("prompt frame only", {"n_recent": 0}), ("prompt frame and 1 recent", {"n_recent": 1}),
                      ("prompt frame and 2 recent", {"n_recent": 2}), ("prompt frame and 3 recent", {"n_recent": 3}),
                      ("prompt frame and 6 recent (the full bank)", {"n_recent": 6}),
                      ("full bank, no locality (rho = 1000)", {"rho": 1000.0})):
        rows.append([label] + [f"{p} {_fx(m, 3)}" for _, p, m in variant_summary(**kw)])
    lead = ("The same clip and tracks with different banks. Presence is one character per frame (1 present, 0 absent) from frame 0 "
            "to 11; the number after it is the mean mask IoU over the frames where the object is visible. The last row keeps the full "
            "bank and sets rho to 1000 cells, which removes the preference for nearby cells.")
    return lead + "\n\n" + _table(["memory"] + [f"track {n}: presence, mean IoU" for n, _, _ in variant_summary()], rows, "lrrr")


def detection_quality(**kw):
    """(matched, truth boxes, mean IoU of the matches) summed over the 12 frames."""
    matched = total = 0
    iou_sum = 0.0
    for t in range(T):
        _, _, dets, _ = detect(t, **kw)
        pairs, fps, missed = match_detections(t, dets)
        matched += len(pairs)
        total += sum(1 for _, v, _ in truth(t) if v is not None)
        iou_sum += sum(p[2] for p in pairs)
    return matched, total, (iou_sum / matched if matched else 0.0)


def _no_pad_inverse(b):
    s = letterbox_plan()[0]
    cl = lambda v, hi: min(max(v, 0.0), hi)
    return (cl(b[0] / s, W), cl(b[1] / s, H), cl(b[2] / s, W), cl(b[3] / s, H))


def _float_pad_inverse(b):
    s, _, nh, px, _ = letterbox_plan()
    py = (SIZE - nh) / 2
    cl = lambda v, hi: min(max(v, 0.0), hi)
    return (cl((b[0] - px) / s, W), cl((b[1] - py) / s, H), cl((b[2] - px) / s, W), cl((b[3] - py) / s, H))


def planted_mistakes():
    ok = detection_quality()
    rows = [["none", "all three stages", f"{ok[0]} of {ok[1]} found, mean IoU {_fx(ok[2], 3)}", "baseline"]]
    m = detection_quality(bgr=True)
    rows.append(["the detector's view has its channels in the order b, g, r", "preprocessing",
                 f"{m[0]} of {m[1]} found" + (f", mean IoU {_fx(m[2], 3)}" if m[0] else ""), "found count"])
    m = detection_quality(inverse=_no_pad_inverse)
    rows.append(["the inverse letterbox does not subtract the padding", "detector tail",
                 f"{m[0]} of {m[1]} found" + (f", mean IoU {_fx(m[2], 3)}" if m[0] else ""), "found count"])
    m = detection_quality(inverse=_float_pad_inverse)
    rows.append(["the inverse uses the padding 11.5 where the pixels were pasted at 11", "detector tail",
                 f"{m[0]} of {m[1]} found, mean IoU {_fx(m[2], 3)}", "mean IoU, not the count"])
    view0 = segmenter_view(0)
    tr = truth(0)
    good, bad = [], []
    for k, box, score, cls in frame0_prompts():
        good.append(mask_iou(mask_to_frame(segment_box(view0, box)), tr[k][0]))
        bad.append(mask_iou(mask_to_frame(segment_box(view0, box, scale_map=to_detector)), tr[k][0]))
    rows.append(["the box prompt reaches the segmenter through the letterbox map", "segmenter",
                 f"mask IoU on frame 0: A {_fx(bad[0], 3)}, B {_fx(bad[1], 3)}, C {_fx(bad[2], 3)} "
                 f"(correct: {_fx(good[0], 3)}, {_fx(good[1], 3)}, {_fx(good[2], 3)})", "mask IoU per object"])
    full = variant_summary()
    nol = variant_summary(rho=1000.0)
    rows.append(["attention without locality (rho = 1000)", "tracker",
                 f"mean mask IoU of A {_fx(nol[0][2], 3)} and C {_fx(nol[2][2], 3)} "
                 f"(correct: {_fx(full[0][2], 3)} and {_fx(full[2][2], 3)})", "mean mask IoU of the two look-alikes"])
    short = variant_summary(n_recent=2)
    rows.append(["a bank of the prompt frame and 2 recent frames", "tracker",
                 f"A present {short[0][1]} (correct: {full[0][1]}); C mean IoU {_fx(short[2][2], 3)} (correct: {_fx(full[2][2], 3)})",
                 "presence pattern of A"])
    lead = ("Each planted mistake changes one thing in an otherwise correct pipeline and reports the measurement that sees it. Found is "
            "the number of visible objects matched at IoU 0.5 over the 12 frames (34 are visible).")
    return lead + "\n\n" + _table(["planted mistake", "stage", "what the pipeline reports", "the check that sees it"], rows, "llll")


# ------------------------------------------------------------------------------------------------ the cost of the real models


def _giga(v):
    return f"{v / 1e9:,.2f}"


def ledger_data(k_tracks=3):
    """FLOPs (2 per multiply-add) of the segmenter side of the pipeline on the 12-frame clip with the large SAM 2.1 model: the trunk
    once per frame, then per track a memory attention, a decoder and a memory encoder; frame 0 runs the decoder on a box prompt
    (2 corner tokens and the padding point, so decoder_flops(2)) and the memory encoder, and no memory attention."""
    trunk = sum(sam.hiera_flops(sam.hiera_blocks())[0].values())
    dec_box = sum(sam.decoder_flops(2).values())
    dec_track = sum(sam.decoder_flops(1).values())
    memenc = sam.memory_encoder_flops()
    frames = []
    for t in range(T):
        if t == 0:
            per_track, keys = dec_box + memenc, 0
        else:
            slabs, ptrs = sam.frame_memory(t, cond=(0,), num_frames=T)
            keys = len(slabs) * 4096 + 4 * len(ptrs)
            per_track = sam.memory_attention_flops(keys)[0] + dec_track + memenc
        frames.append((t, keys, trunk, per_track, trunk + k_tracks * per_track, k_tracks * (trunk + per_track)))
    return frames


def ledger():
    rows = []
    fr = ledger_data()
    for t in (0, 1, 6, 11):
        _, keys, trunk, per_track, shared, unshared = fr[t]
        rows.append([t, _int(keys) if keys else "none", _giga(trunk), _giga(per_track), _giga(shared), _giga(unshared)])
    ts = sum(f[4] for f in fr)
    tu = sum(f[5] for f in fr)
    rows.append(["all 12", "", "", "", _giga(ts), _giga(tu)])
    one = sum(f[4] for f in ledger_data(1))
    lead = ("FLOPs (2 per multiply-add) of the segmenter side of the pipeline for the 12-frame clip priced as the large SAM 2.1 model at "
            "1,024 by 1,024 pixels, with the formulas of Chapter 7: the trunk once per frame, and for every track a memory attention "
            "over the keys of the bank (7 slabs of 4,096 cells and one pointer of 4 tokens from each earlier frame, at most 12), a decoder "
            "with the tracking prompt and a memory encoder; frame 0 has no memory attention and a decoder on a box prompt (2 corner "
            "tokens and the padding point). The detector is not counted.\n\n")
    return (lead + _table(["frame", "memory keys per track", "trunk (GFLOPs)", "per track: attention, decoder, encoder", "3 tracks, trunk shared",
                    "3 tracks, trunk per track"], rows)
            + f"\n\nOne track alone costs {_giga(one)} billion FLOPs over the clip; three tracks that share the trunk cost {_giga(ts)} billion, "
              f"{ts / one:.2f} times as much, where three separate pipelines cost {_giga(tu)} billion, {tu / one:.2f} times.")


# ------------------------------------------------------------------------------------------------ the acceptance list


def acceptance_fingerprint():
    rows = []
    add = lambda m, q, v, tol: rows.append([m, q, v, tol])
    add(1, "SHA-256 (first 16 hex) of frames 0, 7 and 11", " / ".join(frame_digest(t)[:16] for t in (0, 7, 11)), "exact")
    add(1, "visible pixels of A at frames 0, 6, 7, 9", " / ".join(str(len(truth(t)[0][0])) for t in (0, 6, 7, 9)), "exact")
    s, nw, nh, px, py = letterbox_plan()
    add(1, "detector view: scale, content, padding", f"{_fx(s, 6)}; {nw} x {nh}; ({px}, {py})", "6 digits; exact")
    v = detector_view(0)
    sv = segmenter_view(0)
    add(1, "sum of the 12,288 values: detector view, segmenter view of frame 0",
        f"{_fx(sum(v[c][y][x] for c in range(3) for y in range(SIZE) for x in range(SIZE)), 6)} / "
        f"{_fx(sum(sv[c][y][x] for c in range(3) for y in range(SIZE) for x in range(SIZE)), 6)}", "1e-6")
    n_raw, n_f, dets, _ = detect(0)
    add(2, "frame 0: candidates, with score at least 0.25, kept", f"{n_raw} / {n_f} / {len(dets)}", "exact")
    add(2, "frame 0: IoU evaluations of suppression at 0.45 and kept at 0.8 and 0.99", f"{nms_stats(0, 0.45)[1]}; {nms_stats(0, 0.8)[0]} and {nms_stats(0, 0.99)[0]}", "exact")
    q = detection_quality()
    add(2, "boxes found, of all visible, over the 12 frames; mean IoU", f"{q[0]} of {q[1]}; {_fx(q[2], 3)}", "exact; 0.001")
    add(2, "frame 0 kept boxes in frame pixels, by falling score, with the object each matches",
        "; ".join(f"{NAMES[k]} {_box(b)} {_fx(sc, 3)}" for k, b, sc, cl in frame0_prompts()), "0.01; 0.001")
    view0 = segmenter_view(0)
    parts, ious = [], []
    for k, box, score, cls in frame0_prompts():
        m = segment_box(view0, box)
        mf = mask_to_frame(m)
        parts.append(f"{NAMES[k]} {len(m)}/{len(mf)}")
        ious.append(_fx(mask_iou(mf, truth(0)[k][0]), 3))
    add(3, "mask pixels in the input / in the frame (A, B, C) and IoU with the truth", "; ".join(parts) + "; IoU " + ", ".join(ious), "exact; 0.001")
    s = variant_summary()
    add(4, "presence string over frames 0 to 11 of A, B, C", ", ".join(f"{n} {p}" for n, p, _ in s), "exact")
    add(4, "mean mask IoU over the visible frames of A, B, C", ", ".join(_fx(m, 3) for _, _, m in s), "0.001")
    _, tracks = pipeline_run()
    add(4, "smallest distance of any read-back from 0.5 over frames 1 to 11, all tracks", _fx(min(r[t][3] for r in tracks for t in range(1, T)), 4), "0.001")
    add(4, "largest read-back of A at frames 7 and 8", f"{_fx(tracks[0][7][2], 4)} / {_fx(tracks[0][8][2], 4)}", "0.001")
    ab = variant_summary(n_recent=2)
    add(4, "bank of the prompt frame and 2 recent: presence of A and mean IoU of C", f"{ab[0][1]}; {_fx(ab[2][2], 3)}", "exact; 0.001")
    fr = ledger_data()
    add(5, "GFLOPs of the clip, 3 tracks sharing the trunk / not sharing it", f"{_giga(sum(f[4] for f in fr))} / {_giga(sum(f[5] for f in fr))}", "0.01")
    lead = "Expected values of the project, computed by this module. A tolerance of 0.001 or 1e-6 is absolute; exact means equal."
    return lead + "\n\n" + _table(["milestone", "quantity", "value", "tolerance"], rows, "rlll")


def iou_series(track_name, **kw):
    """Per frame, the IoU of a track's mask with the truth, or None on the frames where the object is hidden (for the figure)."""
    prompts, tracks = pipeline_run(**kw)
    out = []
    for (k, b, sc, cl), res in zip(prompts, tracks):
        if NAMES[k] != track_name:
            continue
        for t in range(T):
            tp = truth(t)[k][0]
            out.append(mask_iou(mask_to_frame(res[t][1]), tp) if tp else None)
    return out


def cell_in_box(cell, box):
    """Whether the centre of a 4 by 4 cell of the segmenter's input, mapped to frame pixels, lies in the frame box."""
    cy, cx = divmod(cell, CG)
    x = (CELL * cx + CELL / 2) * W / SIZE
    y = (CELL * cy + CELL / 2) * H / SIZE
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def lookalike_table(t=9):
    """At frame t, for each track and bank, how many cells with read-back at least 0.5 lie in the full box of A and in that of C."""
    rows = []
    for label, n in (("prompt frame and 6 recent (full)", 6), ("prompt frame and 3 recent", 3), ("prompt frame and 2 recent", 2)):
        prompts, tracks = pipeline_run(n_recent=n)
        for (k, b, sc, cl), res in zip(prompts, tracks):
            if NAMES[k] == "B":
                continue
            r = res[t][4]
            on = [c for c in range(CG * CG) if r[c] >= ON]
            in_a = sum(cell_in_box(c, object_box(0, t)) for c in on)
            in_c = sum(cell_in_box(c, object_box(1, t)) for c in on)
            rows.append([label, f"track {NAMES[k]}", len(on), in_a, in_c, _fx(max(r), 3)])
    lead = ("Frame 9, where A has come out from behind B. For the tracks of the two red objects, the cells whose read-back is at least "
            "0.5, and how many of them have their centre (mapped to frame pixels) inside the full box of A and of C.")
    return lead + "\n\n" + _table(["bank", "tracker", "cells at least 0.5", "in the box of A", "in the box of C", "largest read-back"], rows, "llrrrr")


def monitor_table():
    rows = []
    for t, row in enumerate(monitor_data()):
        rows.append([t] + [("absent" if v is None else _fx(v, 3)) for v in row])
    prompts, _ = pipeline_run()
    lead = ("Each frame is also run through the detector. For every track that reports its object present, the largest IoU between the "
            "bounding box of its mask (in frame pixels) and a detection of the track's class; absent where the track says so.")
    return lead + "\n\n" + _table(["t"] + [f"track {NAMES[p[0]]}" for p in prompts], rows)
