# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Learning 3D Directly".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: feed_forward_3d.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures.

A network with a billion parameters cannot be trained inside a textbook, so the examples study what surrounds one: the shape of
its outputs and the arithmetic that makes them consistent, scored and expensive. Two small worlds carry them.

* A dense toy: a 48 by 36 image of a room (floor, back wall, a sphere) seen from three cameras, ray cast exactly. Its pointmaps are
  exact; a seeded *stand-in* adds the kind of error a network makes (a smooth bias along the ray of each view, noise per pixel,
  a lateral scatter, and an unknown global scale) so that alignment, camera recovery, fusion and confidence can be measured.
* The sparse scene of the previous chapter (60 points, 4 cameras, half a pixel of noise), with the same stand-in, to compare a
  feed-forward start with the classical pipeline on the same cameras.

The stand-in is not a measurement of brain's model. Its error sizes are choices the chapter states. What the examples measure is
the machinery around such outputs, and brain's model supplies the counts: parameters and arithmetic from its configuration.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, sin, cos, exp, log, atan, tan) unless a block
says otherwise. Sums are accumulated left to right in index order. Singular vectors come from the one-sided Jacobi SVD of the matrix
chapter of Volume I (`decompositions._svd`). Every random number comes from the book's SplitMix64 (`bookrng.generator`, stream
names printed by the chapter).

Swedish Embedded AB implements solutions for feed-forward 3D reconstruction that is checked against geometry it can verify. If your
team needs expertise in learned multi-view geometry, you can procure our services by sending an email to info@swedishembedded.com.
"""
import json
import math
from pathlib import Path

import bookrng
import camera_geometry as cg
import decompositions as dc
import images as im
import structure_from_motion as sfm

_table = im._table
_g = dc._g
mm, mv, tr, dot, norm, unit, cross, sub = cg.mm, cg.mv, cg.tr, cg.dot, cg.norm, cg.unit, cg.cross, cg.sub

_cache = {}


def _once(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def _v(x, d=4):
    return "(" + ", ".join(_g(c, d) for c in x) + ")"


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _scale(k, a):
    return [k * x for x in a]


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ------------------------------------------------------------------------------------------------ the dense toy: a room, ray cast

W, H, FOC = 48, 36, 36.0                  # image size and focal length in pixels
CXD, CYD = W / 2.0, H / 2.0               # principal point
FLOOR_Y, WALL_Z = 1.0, 6.0                # the floor is the plane y = 1 (y points down), the back wall the plane z = 6
SPHERE_C, SPHERE_R = [0.4, 0.3, 3.5], 0.7
EYES = [[0.0, 0.0, 0.0], [0.5, 0.05, 0.0], [-0.5, 0.0, 0.1]]
LOOK, UP = [0.2, 0.3, 3.5], [0.0, -1.0, 0.0]
MATERIAL = {0: "sphere", 1: "floor", 2: "wall"}


def pixel_ray(i, j):
    """The camera-frame ray (x, y, 1) of the centre of pixel (column i, row j)."""
    return [(i + 0.5 - CXD) / FOC, (j + 0.5 - CYD) / FOC, 1.0]


def cast(C, d):
    """The nearest hit of the ray C + t d (t > 0) with the room: (t, material), the floor, the back wall and the sphere."""
    best = (math.inf, -1)
    if d[1] > 0:
        t = (FLOOR_Y - C[1]) / d[1]
        if 0 < t < best[0]:
            best = (t, 1)
    if d[2] > 0:
        t = (WALL_Z - C[2]) / d[2]
        if 0 < t < best[0]:
            best = (t, 2)
    oc = sub(C, SPHERE_C)
    a, b, c = dot(d, d), 2.0 * dot(oc, d), dot(oc, oc) - SPHERE_R ** 2
    disc = b * b - 4 * a * c
    if disc >= 0:
        t = (-b - math.sqrt(disc)) / (2 * a)
        if 0 < t < best[0]:
            best = (t, 0)
    return best


def room():
    """Three cameras and their exact maps: for camera v, `R`, `t` (world to camera), the centre `C`, the camera-to-world rotation
    `Rc` and, per pixel in row-major order, the camera depth z, the world point (the pointmap, in the frame of camera 0) and the
    material. Camera 0 is the identity, so the world is its frame."""
    def make():
        cams = []
        for v, eye in enumerate(EYES):
            R, t = ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0]) if v == 0 else cg.look_at(eye, LOOK, UP)
            Rc = tr(R)
            depth, pts, mat = [], [], []
            for j in range(H):
                for i in range(W):
                    d = mv(Rc, pixel_ray(i, j))
                    tt, m = cast(eye, d)
                    depth.append(tt)
                    pts.append(_add(eye, _scale(tt, d)))
                    mat.append(m)
            cams.append({"R": R, "t": t, "C": list(eye), "Rc": Rc, "z": depth, "X": pts, "m": mat})
        return cams
    return _once("room", make)


def project(cam, X):
    """Pixel (u, v) in continuous coordinates and camera depth of the world point X."""
    c = [a + b for a, b in zip(mv(cam["R"], X), cam["t"])]
    return (CXD + FOC * c[0] / c[2], CYD + FOC * c[1] / c[2]), c[2]


def pointmap_toy():
    cams = room()
    c0, c1 = cams[0], cams[1]
    rows = []
    picks = [(9, 6), (24, 15), (18, 27), (36, 24), (39, 9)]
    for (i, j) in picks:
        k = j * W + i
        X = c0["X"][k]
        (u, v), z1 = project(c1, X)
        inside = 0 <= u < W and 0 <= v < H
        if inside:
            ni, nj = int(u), int(v)
            best = min(range(W * H), key=lambda q: norm(sub(c1["X"][q], X)))
            bi, bj = best % W, best // W
            dist = norm(sub(c1["X"][best], X))
            lands = f"({u:.2f}, {v:.2f})"
            near = f"({bi}, {bj})"
        else:
            lands, near, dist = "outside", "-", float("nan")
        rows.append([f"({i}, {j})", MATERIAL[c0["m"][k]], f"{c0['z'][k]:.4f}", _v(X, 4), lands, near,
                     "-" if not inside else f"{dist:.4f}", f"{c0['z'][k] / FOC:.4f}"])
    # all pixels of view 0: how many are seen by view 1, and how often the nearest pointmap entry is the projected pixel
    seen = same = 0
    for k in range(W * H):
        X = c0["X"][k]
        (u, v), z1 = project(c1, X)
        if not (0 <= u < W and 0 <= v < H):
            continue
        q = int(v) * W + int(u)
        if abs(c1["z"][q] - z1) > 0.05 * z1:
            continue                                   # occluded in view 1: another surface is in front
        seen += 1
        best = min(range(W * H), key=lambda p: norm(sub(c1["X"][p], X)))
        if max(abs(best % W - int(u)), abs(best // W - int(v))) <= 1:
            same += 1
    out = [f"A room seen by three cameras with f = {FOC:g} pixels, principal point ({CXD:g}, {CYD:g}) and a {W} by {H} image. "
           f"The floor is the plane y = {FLOOR_Y:g}, the back wall z = {WALL_Z:g}, and a sphere of radius {SPHERE_R} is centred "
           f"at {_v(SPHERE_C, 3)} (metres, x right, y down, z forward, in the frame of camera 0). Camera 0 is the identity; "
           f"cameras 1 and 2 are placed with `look_at` of Chapter 3 at the eyes {_v(EYES[1], 3)} and {_v(EYES[2], 3)}, both looking "
           f"at {_v(LOOK, 3)} with up = (0, -1, 0). The ray of pixel (i, j) is ((i + 0.5 - c_x) / f, (j + 0.5 - c_y) / f, 1); "
           "the pointmap entry is the nearest hit of the room along it (the floor, then the wall, then the sphere, whichever is "
           "closest), in the frame of camera 0.", "",
           "Five pixels of camera 0, the world point their pointmap entry holds, where that point lands in camera 1, and the pixel of "
           "camera 1 whose pointmap entry is nearest to it in space (with the distance in metres, next to the width z / f of the "
           "pixel's footprint at that depth):", "",
           _table(["pixel (i, j)", "surface", "depth z", "pointmap entry (X, Y, Z)", "lands in camera 1 at", "nearest entry of camera 1",
                   "distance", "z / f"], rows, "llrlrrrr"), "",
           f"Over all {W * H} pixels of camera 0, {seen} are seen by camera 1 (they land inside the image and the depth stored there "
           f"agrees to 5 percent); for {same} of them the pointmap entry of camera 1 that is nearest in space is within one pixel of "
           "the pixel the geometry says they land in."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ the stand-in network

STAND_NAME = "feed_forward_3d.standin"
SCALE_UNKNOWN = 0.37        # the global scale the stand-in's world carries: its units are 0.37 of the room's metres
SIG_BIAS = (0.02, 0.01, 0.01)
SIG_PIX, SIG_HARD, SIG_LAT = 0.01, 0.03, 0.004
HARD_JUMP = 0.5             # a pixel is hard when its depth differs from a 4-neighbour by more than this fraction of the smaller


def hard_pixels(depth):
    """True where the depth differs from a 4-neighbour by more than HARD_JUMP of the smaller depth (the silhouettes)."""
    out = []
    for j in range(H):
        for i in range(W):
            a = depth[j * W + i]
            hard = False
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < W and 0 <= jj < H:
                    b = depth[jj * W + ii]
                    if abs(a - b) / min(a, b) > HARD_JUMP:
                        hard = True
            out.append(hard)
    return out


def standin_rule():
    """The complete recipe of the room's stand-in as page text, from the constants the code uses."""
    nhard = sum(sum(hard_pixels(c["z"])) for c in room())
    return (f"The stand-in (`bookrng.generator(\"{STAND_NAME}\")`, seed {bookrng.seed_of(STAND_NAME)}) is built as follows. Let z be the "
            f"true camera depth of a pixel, (x, y) = ((i + 0.5 - c_x) / f, (j + 0.5 - c_y) / f) its normalized coordinates, "
            f"s = {SCALE_UNKNOWN} the scale of its world, and a pixel *hard* when its true depth differs from that of a 4-neighbour "
            f"by more than {HARD_JUMP:g} of the smaller of the two (the silhouettes: {nhard} of the {3 * W * H} pixels). For camera v "
            f"= 0, 1, 2 in turn, six `normal()` draws are scaled by the standard deviations ({', '.join(f'{x:g}' for x in SIG_BIAS)}) to "
            "give the biases (b0, bx, by) of the depth head, and the next six likewise the biases (b0', bx', by') of the pointmap head. "
            "Then for every pixel in row-major order (row j from 0, column i from 0), five `normal()` draws in this order: e for the depth "
            "head and e' for the pointmap head, each scaled by "
            f"sigma = {SIG_PIX:g} ({SIG_HARD:g} on a hard pixel), and three lateral errors scaled by {SIG_LAT:g} z, one for each world axis. "
            "The depth head's output is s z (1 + b0 + bx x + by y + e). The pointmap head's is s (C + z (1 + b0' + bx' x + by' y + e') r + "
            "lateral), where C is the camera centre and r = R_c (x, y, 1) with R_c the camera-to-world rotation, so that C + z r is the exact "
            "world point; the lateral errors are added in world axes before the scale s. Both heads give the confidence 1 + exp(raw) with "
            f"raw = -2 ln(sigma / {SIG_PIX:g}), which is 2 on an easy pixel and 1 + 1/9 on a hard one. "
            "The sizes are chosen, not measured.")


def standin():
    """Per camera: the depth head's prediction `d` (camera depth, in the stand-in's units), its confidence `cd`, the pointmap head's
    prediction `P` (a world point in the stand-in's units, frame of camera 0) and its confidence `cp`. For camera v = 0, 1, 2 in
    turn: six normal() draws for the biases (b0, bx, by) of the depth head and then of the pointmap head, with standard deviations
    0.02, 0.01, 0.01; then, for each pixel in row-major order, five normal() draws: the depth head's relative error, the pointmap
    head's relative error (both with sigma 0.01, or 0.03 on a hard pixel), and three lateral errors of the pointmap head with
    sigma 0.004 z each. Depth head: s z (1 + b0 + bx x + by y + e) with (x, y) the pixel's normalized coordinates; pointmap head:
    s (C + t d (1 + the same bias form + e') + lateral), d the world ray of the pixel; s = 0.37. The confidence of a pixel is
    1 + exp(raw) with raw = -2 ln(sigma / 0.01), so 2 on an easy pixel and 1 + 1/9 on a hard one."""
    def make():
        cams = room()
        g = bookrng.generator(STAND_NAME)
        s = SCALE_UNKNOWN
        out = []
        for cam in cams:
            bd = [sg * g.normal() for sg in SIG_BIAS]
            bp = [sg * g.normal() for sg in SIG_BIAS]
            hard = hard_pixels(cam["z"])
            d, cd, P, cp = [], [], [], []
            for j in range(H):
                for i in range(W):
                    k = j * W + i
                    x, y, _ = pixel_ray(i, j)
                    sig = SIG_HARD if hard[k] else SIG_PIX
                    ed, ep = sig * g.normal(), sig * g.normal()
                    lat = [SIG_LAT * cam["z"][k] * g.normal() for _ in range(3)]
                    z = cam["z"][k]
                    d.append(s * z * (1.0 + bd[0] + bd[1] * x + bd[2] * y + ed))
                    X = cam["X"][k]
                    ray = mv(cam["Rc"], [x, y, 1.0])
                    P.append(_scale(s, _add(_add(cam["C"], _scale(z * (1.0 + bp[0] + bp[1] * x + bp[2] * y + ep), ray)), lat)))
                    conf = 1.0 + math.exp(-2.0 * math.log(sig / SIG_PIX))
                    cd.append(conf)
                    cp.append(conf)
            out.append({"d": d, "cd": cd, "P": P, "cp": cp, "bd": bd, "bp": bp, "hard": hard})
        return out
    return _once("standin", make)


# ------------------------------------------------------------------------------------------------ cameras from pointmaps

def umeyama(A, B, with_scale=True):
    """The similarity (s, R, t) that best maps the points A onto the points B in the least-squares sense, b = s R a + t. With
    centroids a-bar, b-bar and the cross-covariance S = (1/n) sum (b - b-bar)(a - a-bar)^T, R is the rotation nearest to S (Volume
    I, Chapter 6; reflections are excluded), s = trace(R^T S) / ((1/n) sum |a - a-bar|^2) (1 if not with_scale) and t = b-bar - s R a-bar."""
    n = len(A)
    ca = [sum(a[k] for a in A) / n for k in range(3)]
    cb = [sum(b[k] for b in B) / n for k in range(3)]
    S = [[sum((B[p][r] - cb[r]) * (A[p][c] - ca[c]) for p in range(n)) / n for c in range(3)] for r in range(3)]
    R = cg.nearest_rotation(S)
    var = sum(sum((a[k] - ca[k]) ** 2 for k in range(3)) for a in A) / n
    s = sum(R[r][c] * S[r][c] for r in range(3) for c in range(3)) / var if with_scale else 1.0
    t = sub(cb, _scale(s, mv(R, ca)))
    return s, R, t


def rot_angle_deg(Ra, Rb):
    """The angle in degrees of the rotation Ra^T Rb."""
    return sfm.rot_angle(mm(tr(Ra), Rb))


def pose_from_pointmap(depth, world, pix):
    """The camera-to-world pose (R, C) that carries the camera-frame points z (x, y, 1), with (x, y) the normalized pixel coordinates,
    onto the world points of the pointmap: a rigid Umeyama fit (scale fixed to 1) over the listed pixels."""
    cam_pts = [_scale(depth[k], [(pix[k][0] - CXD) / FOC, (pix[k][1] - CYD) / FOC, 1.0]) for k in range(len(depth))]
    s, R, t = umeyama(cam_pts, world, with_scale=False)
    return R, t


def intrinsics_from_cam_points(pts, pix):
    """The least-squares f and c from x / z = (u - c) / f: the straight line u = c + f (X / Z) through the points of one view
    (ordinary least squares of u on X / Z, by the closed form of Volume II), separately for u and v; returns (fx, cx, fy, cy)."""
    n = len(pts)
    out = []
    for ax, pi in ((0, 0), (1, 1)):
        xs = [p[ax] / p[2] for p in pts]
        us = [q[pi] for q in pix]
        mx, mu = sum(xs) / n, sum(us) / n
        a = sum((x - mx) * (u - mu) for x, u in zip(xs, us)) / sum((x - mx) ** 2 for x in xs)
        out += [a, mu - a * mx]
    return out[0], out[1], out[2], out[3]


def camera_from_pointmap():
    cams = room()
    sd = standin()
    pix = [(i + 0.5, j + 0.5) for j in range(H) for i in range(W)]
    rows = []
    for v in (1, 2):
        cam = cams[v]
        R, C = pose_from_pointmap(cam["z"], cam["X"], pix)
        dR = rot_angle_deg(R, cam["Rc"])
        dC = norm(sub(C, cam["C"]))
        s = sd[v]
        Rn, Cn = pose_from_pointmap(s["d"], s["P"], pix)
        dRn = rot_angle_deg(Rn, cam["Rc"])
        dCn = norm(sub(Cn, _scale(SCALE_UNKNOWN, cam["C"]))) / SCALE_UNKNOWN
        rows.append([v, "below 1e-4" if dR < 1e-4 else f"{dR:.2e}", "below 1e-12" if dC < 1e-12 else f"{dC:.2e}", f"{dRn:.3f}", f"{dCn:.4f}"])
    cam = cams[1]
    cpts = [mv(cam["R"], _add(X, [0.0] * 3)) for X in cam["X"]]
    cpts = [_add(c, cam["t"]) for c in cpts]
    fx, cx, fy, cy = intrinsics_from_cam_points(cpts, pix)
    noisy = [mv(cam["R"], sub(sd[1]["P"][k], _scale(SCALE_UNKNOWN, cam["C"]))) for k in range(W * H)]
    fx2, cx2, fy2, cy2 = intrinsics_from_cam_points(noisy, pix)
    return "\n".join([
        f"Camera v = 1 and 2 of the room. A camera-frame point is z (x, y, 1) with (x, y) = ((i + 0.5 - c_x) / f, (j + 0.5 - c_y) / f); "
        f"the pose (R, C) is the rigid map from these points to the pointmap entries over all {W * H} pixels (the Umeyama fit with "
        "the scale held at 1; the pointmap of the stand-in is in the stand-in's units, so its centre error is divided by the scale "
        f"{SCALE_UNKNOWN} to read in metres). Columns: the rotation error in degrees and the centre error in metres of the pose "
        "from exact maps, and of the pose from the stand-in's depth head and pointmap head (`feed_forward_3d.standin`, seed "
        f"{bookrng.seed_of(STAND_NAME)}).", "", standin_rule(), "",
        _table(["camera", "rotation error, exact (deg)", "centre error, exact (m)", "rotation error, stand-in (deg)",
                "centre error, stand-in (m)"], rows, "rrrrr"), "",
        f"The intrinsics read back from the exact camera-frame points of camera 1 by the line u = c_x + f (X / Z) are f_x = {fx:.6f}, "
        f"c_x = {cx:.6f}, f_y = {fy:.6f}, c_y = {cy:.6f} (true: {FOC:g}, {CXD:g}, {FOC:g}, {CYD:g}); applied to the stand-in's "
        f"pointmap of camera 1 carried into that camera's frame by its true pose, the same fit gives f_x = {fx2:.4f}, c_x = {cx2:.4f}, f_y = {fy2:.4f}, c_y = {cy2:.4f}."])


# ------------------------------------------------------------------------------------------------ scale, shift and the frame

def align_scale_shift(u, v):
    """The least-squares a, b of v ~ a u + b (equation (8) of Volume VII, Chapter 5), the sums taken left to right."""
    n = len(u)
    su, sv = sum(u), sum(v)
    suu, suv = sum(x * x for x in u), sum(x * y for x, y in zip(u, v))
    a = (n * suv - su * sv) / (n * suu - su * su)
    return a, (sv - a * su) / n


def abs_rel(pred, truth):
    return sum(abs(p - t) / t for p, t in zip(pred, truth)) / len(truth)


def scale_alignment():
    cams = room()
    sd = standin()
    truth = [c["z"] for c in cams]
    raw = [s["d"] for s in sd]

    def rows_for(label, npar, preds):
        per = [abs_rel(preds[v], truth[v]) for v in range(3)]
        return [label, npar] + [f"{x:.4f}" for x in per] + [f"{sum(per) / 3:.4f}"]

    out = [rows_for("none", "0", raw)]
    allr = [t / p for v in range(3) for p, t in zip(raw[v], truth[v])]
    gs = _median(allr)
    out.append(rows_for(f"one scale for all three (median ratio {gs:.4f})", "1", [[gs * p for p in raw[v]] for v in range(3)]))
    pf, scales = [], []
    for v in range(3):
        m = _median([t / p for p, t in zip(raw[v], truth[v])])
        scales.append(m)
        pf.append([m * p for p in raw[v]])
    out.append(rows_for("one scale per view (" + ", ".join(f"{m:.4f}" for m in scales) + ")", "3", pf))
    pss = []
    for v in range(3):
        nu = [1.0 / p for p in raw[v]]
        nt = [1.0 / t for t in truth[v]]
        a, b = align_scale_shift(nu, nt)
        pss.append([1.0 / (a * x + b) for x in nu])
    out.append(rows_for("scale and shift in inverse depth per view", "6", pss))

    # pointmaps in an arbitrary frame: the stand-in's world carried by a rotation and a translation as well as the scale
    g_w, g_t = [0.2, -0.4, 0.1], [1.0, -2.0, 3.0]
    Rg = cg.rodrigues(g_w)
    pts_t = [X for c in cams for X in c["X"]]
    pts_s = [P for s in sd for P in s["P"]]
    pts_f = [_add(mv(Rg, P), g_t) for P in pts_s]                       # the frame of camera 0 is lost: R_g P + t_g
    n = len(pts_t)
    cen = [sum(p[k] for p in pts_t) / n for k in range(3)]
    spread = math.sqrt(sum(sum((p[k] - cen[k]) ** 2 for k in range(3)) for p in pts_t) / n)

    def rel_rms(est):
        return math.sqrt(sum(sum((e[k] - t[k]) ** 2 for k in range(3)) for e, t in zip(est, pts_t)) / n) / spread

    def scalar_fit(est):
        ce = [sum(p[k] for p in est) / n for k in range(3)]
        se = math.sqrt(sum(sum((p[k] - ce[k]) ** 2 for k in range(3)) for p in est) / n)
        sc = spread / se
        return [_add(cen, _scale(sc, sub(p, ce))) for p in est], sc          # centroid onto centroid, spread onto spread

    rows2 = []
    rows2.append(["frame of camera 0", "none", "-", f"{rel_rms(pts_s):.4f}"])
    est, sc = scalar_fit(pts_s)
    rows2.append(["frame of camera 0", "one scale (spread ratio)", f"s = {sc:.4f}", f"{rel_rms(est):.4f}"])
    s7, R7, t7 = umeyama(pts_s, pts_t)
    rows2.append(["frame of camera 0", "similarity, 7 numbers", f"s = {s7:.4f}", f"{rel_rms([_add(_scale(s7, mv(R7, p)), t7) for p in pts_s]):.4f}"])
    rows2.append(["rotated and shifted", "none", "-", f"{rel_rms(pts_f):.4f}"])
    est, sc = scalar_fit(pts_f)
    rows2.append(["rotated and shifted", "one scale (spread ratio)", f"s = {sc:.4f}", f"{rel_rms(est):.4f}"])
    s7, R7, t7 = umeyama(pts_f, pts_t)
    best = [_add(_scale(s7, mv(R7, p)), t7) for p in pts_f]
    rows2.append(["rotated and shifted", "similarity, 7 numbers", f"s = {s7:.4f}", f"{rel_rms(best):.4f}"])
    back = mm(R7, Rg)                                  # the fitted rotation undoes the frame change: R7 Rg = the identity
    return "\n".join([
        f"The depth head of the stand-in (`feed_forward_3d.standin`, seed {bookrng.seed_of(STAND_NAME)}) is {SCALE_UNKNOWN} of the "
        "true depth on average, with a smooth bias per view and noise. AbsRel = mean |prediction - truth| / truth over the "
        f"{W * H} pixels of each view, under four ways of aligning it with the truth first. 'One scale for all three' is the median over "
        f"all {3 * W * H} pixels of truth / prediction; 'one scale per view' is the same median per view; the last row fits scale and "
        "shift to the inverse depth of each view separately by the closed form of Volume VII, Chapter 5.", "",
        _table(["alignment", "numbers fitted", "view 0", "view 1", "view 2", "mean"], out, "lrrrrr"), "",
        "The biases that the stand-in gave the depth head at the image centre are " + ", ".join(
            f"{sd[v]['bd'][0]:+.4f} (view {v})" for v in range(3)) + ". A scale per view cannot tell such a bias from a different "
        "size of the room, so it removes them, and the row says the model is better than the single scale says.", "",
        f"The pointmap head's {3 * W * H} points, scored against the true points (relative RMS: the RMS distance of the aligned point to the "
        f"true one divided by the RMS distance of the true points from their centroid, {spread:.4f} m). 'Rotated and shifted' "
        f"carries the stand-in's points by the rotation vector {_v(g_w, 2)} and the translation {_v(g_t, 2)}, as a model that "
        "did not fix its world to the first camera would leave them. The one-scale fit puts the centroid of the points on that "
        "of the truth and the spread on the spread; the similarity is the Umeyama fit.", "",
        _table(["points", "alignment", "scale found (true: 1 / 0.37 = " + f"{1 / SCALE_UNKNOWN:.4f})", "relative RMS"], rows2, "llrr"), "",
        f"The rotation the similarity fit finds undoes the frame change to {rot_angle_deg(back, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):.3f} degrees."])


# ------------------------------------------------------------------------------------------------ tokens for cameras and rays

def quat_xyzw(R):
    """The quaternion (x, y, z, w), scalar last, of a rotation matrix by the branch rule of the pose token: when the trace is
    positive, s = 2 sqrt(1 + trace) and q = ((R21 - R12) / s, (R02 - R20) / s, (R10 - R01) / s, s / 4); otherwise the largest
    diagonal entry (the first on a tie) picks the branch whose own component is s / 4 with s = 2 sqrt(1 + R_kk - the other two). The
    sign is whatever the branch gives: the scalar part is not forced to be non-negative."""
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        return [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s]
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        return [0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s]
    if R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        return [(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s]
    s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
    return [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s]


def pose_vectors(c2w):
    """The 7-vector of each camera's pose token: the quaternion (x, y, z, w) of its camera-to-world rotation, then its centre
    minus the centroid of all the centres, divided by the largest distance of a centre from that centroid (at least 1e-8)."""
    n = len(c2w)
    cen = [sum(C[k] for _, C in c2w) / n for k in range(3)]
    alpha = max(max(norm(sub(C, cen)) for _, C in c2w), 1e-8)
    return [quat_xyzw(R) + [x / alpha for x in sub(C, cen)] for R, C in c2w]


def intrinsic_vector(fx, fy, cx, cy, w, h):
    return [fx / w, fy / h, cx / w, cy / h]


def decode_fov(fx, fy, w, h):
    """What the camera head's output (field of view across, field of view down) says about the intrinsics: 2 atan(size / (2 f)),
    and the decoded focal length 0.5 size / tan(fov / 2) with the principal point at the middle of the image."""
    fu, fv = 2.0 * math.atan(w / (2.0 * fx)), 2.0 * math.atan(h / (2.0 * fy))
    return fu, fv, 0.5 * w / math.tan(0.5 * fu), 0.5 * h / math.tan(0.5 * fv)


def lens_ray_map(k1, w=W, h=H, f=FOC):
    """The unit ray of each pixel centre of a camera whose lens is the radial model of Chapter 3 with coefficient k1 (k2 = 0): the
    normalized distorted point is ((u - c_x) / f, (v - c_y) / f), inverted to the undistorted (x, y) by 12 steps of Newton's
    method from the distorted point, and the ray is (x, y, 1) / |(x, y, 1)|."""
    rays = []
    for j in range(h):
        for i in range(w):
            xd, yd = (i + 0.5 - w / 2.0) / f, (j + 0.5 - h / 2.0) / f
            x, y = cg.invert_newton(xd, yd, 12, k1=k1, k2=0.0)[-1]
            rays.append(unit([x, y, 1.0]))
    return rays


def camera_tokens():
    cams = room()
    c2w = [(c["Rc"], c["C"]) for c in cams]
    pv = pose_vectors(c2w)
    big = [(R, _add(_scale(1000.0, C), [500.0, -300.0, 70.0])) for R, C in c2w]
    pv2 = pose_vectors(big)
    gap = max(abs(a - b) for x, y in zip(pv, pv2) for a, b in zip(x, y))
    rows = [[v, _v(pv[v][:4], 4), _v(pv[v][4:], 4)] for v in range(3)]
    Rb = cg.rodrigues([0.0, math.radians(-150.0), 0.0])
    Rb2 = cg.rodrigues([0.0, math.radians(150.0), 0.0])
    qa, qb = quat_xyzw(Rb), quat_xyzw(Rb2)

    intr = [("the room's camera (48 by 36)", FOC, FOC, 24.0, 18.0, 48, 36), ("Chapter 3's camera (640 by 480)", 500.0, 500.0, 320.0, 240.0, 640, 480),
            ("the same camera at 320 by 240", 250.0, 250.0, 160.0, 120.0, 320, 240), ("a principal point 0.4 pixel off centre (48 by 36)", FOC, FOC, 24.4, 18.0, 48, 36)]
    r2 = []
    for name, fx, fy, cx, cy, w, h in intr:
        iv = intrinsic_vector(fx, fy, cx, cy, w, h)
        fu, fv, dfx, dfy = decode_fov(fx, fy, w, h)
        r2.append([name, _v(iv, 5), f"{math.degrees(fu):.3f}, {math.degrees(fv):.3f}", f"{dfx:.4f}", f"{cx - w / 2.0:+.1f}"])
    rays = lens_ray_map(-0.15)
    xs = [r[0] / r[2] for r in rays]
    ys = [r[1] / r[2] for r in rays]
    us = [i + 0.5 for j in range(H) for i in range(W)]
    n = len(xs)
    mx, mu = sum(xs) / n, sum(us) / n
    a = sum((x - mx) * (u - mu) for x, u in zip(xs, us)) / sum((x - mx) ** 2 for x in xs)
    c0 = mu - a * mx
    res = [abs(c0 + a * x - u) for x, u in zip(xs, us)]
    rms = math.sqrt(sum(r * r for r in res) / n)
    return "\n".join([
        "The room's three cameras as pose tokens: the camera-to-world rotation as a quaternion (x, y, z, w) and the centre, minus the "
        "centroid of the three centres and divided by the largest distance from it.", "",
        _table(["camera", "quaternion (x, y, z, w)", "normalized centre"], rows, "lll"), "",
        "The same three cameras with every centre multiplied by 1000 and moved by (500, -300, 70) give tokens that differ from these "
        f"by {'less than 1e-12' if gap < 1e-12 else 'more than 1e-12'} in any entry. The furthest camera sits at distance 1 from the "
        "centroid by construction.", "",
        "Intrinsics tokens (f_x / w, f_y / h, c_x / w, c_y / h) and what the field-of-view output decodes back to (f_x from "
        "0.5 w / tan(fov_u / 2), with the principal point placed at the middle of the image):", "",
        _table(["camera", "intrinsics token", "field of view across, down (deg)", "decoded f_x", "c_x - w / 2 lost (px)"], r2, "lrrrr"), "",
        f"A rotation of -150 degrees about y has the scalar-last quaternion {_v(qa, 4)} by the branch rule of the pose token, and the "
        f"rotation of +150 degrees {_v(qb, 4)}: the second has w = {qb[3]:.4f}, the first {qa[3]:.4f}, a sign that "
        "follows from the branch and not from the rotation.", "",
        f"A ray map for a barrel lens (k1 = -0.15, {W} by {H} pixels, f = {FOC:g}, the radial model of Chapter 3 inverted by 12 Newton "
        f"steps) is {W * H} unit directions. The pinhole line u = c + f' (x / z) is fitted to the {W * H} pairs of the pixel-centre "
        f"column u = i + 0.5 and x / z of its ray by least squares, in u alone (v is not used), and has f' = {a:.4f}, c = {c0:.4f}; it "
        f"misses the pixel column it was made from by {rms:.3f} px RMS and "
        f"{max(res):.3f} px at the worst, in u; the ray map itself has no such error, since it lists the ray of every pixel."])


# ------------------------------------------------------------------------------------------------ brain's model, counted

D, HEADS, MLP, PATCH, IMG, REG, DEPTH = 1024, 16, 4, 14, 518, 4, 24      # trunk and encoder width, heads, MLP ratio, patch, native size
TAPS, DPT_PROJ, DPT_FEAT, CAM_BLOCKS, CAM_ITERS = (4, 11, 17, 23), (256, 512, 1024, 1024), 256, 4, 4
SPECIALS = 7                                                              # per-frame trunk tokens before the patches
HEADER = Path(__file__).resolve().parents[3] / "crates" / "worldmirror2" / "tests" / "golden" / "header.json"


def _lin(i, o):
    return [(o, i), (o,)]


def _block_shapes(d, qk_norm):
    """Shapes of one pre-norm block: norm1, qkv, [q and k norms over the head width], proj, layer scale 1, norm2, fc1, fc2, layer
    scale 2."""
    sh = [(d,), (d,)] + _lin(d, 3 * d)
    if qk_norm:
        sh += [(d // HEADS,)] * 4
    return sh + _lin(d, d) + [(d,), (d,), (d,)] + _lin(d, MLP * d) + _lin(MLP * d, d) + [(d,)]


def _dpt_shapes(out_ch, merger):
    f, tap = DPT_FEAT, 2 * D
    sh = [(tap,), (tap,)]                                                  # the token norm
    for c in DPT_PROJ:
        sh += [(c, tap, 1, 1), (c,)]
    sh += [(256, 256, 4, 4), (256,), (512, 512, 2, 2), (512,), (1024, 1024, 3, 3), (1024,)]      # resize layers 0, 1 and 3
    sh += [(f, c, 3, 3) for c in DPT_PROJ]                                 # scratch layer_rn, no bias
    for r in (1, 2, 3, 4):
        units = 1 if r == 4 else 2                                         # the deepest refinement has one residual unit
        sh += [(f, f, 3, 3), (f,)] * (2 * units) + [(f, f, 1, 1), (f,)]
    sh += [(f // 2, f, 3, 3), (f // 2,), (f // 8, f // 2, 3, 3), (f // 8,), (out_ch, f // 8, 1, 1), (out_ch,)]
    if merger:
        sh += [(f // 2, 3, 7, 7), (f // 2,)]
    return sh


def group_parts():
    """name -> list of (component, list of tensor shapes): the checkpoint's 1545 tensors in ten groups, each as the named
    components that the page's inventory table lists. The groups are the concatenation of their components in order."""
    pe = (D, 3, PATCH, PATCH)
    c2 = 2 * D
    blocks = lambda d, qk, label: [(f"{label}, block {b}", _block_shapes(d, qk)) for b in range(DEPTH)]
    g = {}
    g["trunk tokens and prior encoders"] = [
        ("camera token, one copy for frame 0 and one for the others", [(1, 2, 1, D)]),
        ("register tokens, same two copies of 4", [(1, 2, REG, D)]),
        ("pose embedder: linear 7 to d, GELU, linear d to d", _lin(7, D) + _lin(D, D)),
        ("intrinsics embedder: linear 4 to d, GELU, linear d to d", _lin(4, D) + _lin(D, D)),
        ("depth embedder: the 14 by 14 values of a depth patch (196) through linear 196 to 4 d, GELU, linear 4 d to d",
         _lin(PATCH * PATCH, MLP * D) + _lin(MLP * D, D))]
    g["DINOv2 patch encoder"] = [
        ("class token", [(1, 1, D)]), ("register tokens", [(1, REG, D)]),
        ("mask token (unused at inference)", [(1, D)]),
        ("position embedding, 1 + 37 * 37 = 1370 positions", [(1, 1 + (IMG // PATCH) ** 2, D)]),
        ("patch projection, a 14 by 14 convolution of 3 channels to d, with bias", [pe, (D,)]),
        ("24 blocks of width d (block table below, 14 tensors each)", sum((_block_shapes(D, False) for _ in range(DEPTH)), [])),
        ("final layer norm", [(D,), (D,)])]
    g["24 frame blocks"] = [
        ("24 blocks of width d with query-key norms (block table below, 18 tensors each)", sum((_block_shapes(D, True) for _ in range(DEPTH)), [])),
        ("the one stored table of rotary periods, head width / 4 = 16 values, shared by all 48 trunk blocks", [(D // HEADS // 4,)])]
    g["24 global blocks"] = [("24 blocks of width d with query-key norms (block table below, 18 tensors each)", sum((_block_shapes(D, True) for _ in range(DEPTH)), []))]
    g["camera head"] = [
        ("initial camera 9-vector", [(1, 1, 9)]), ("parameter embedding: linear 9 to 2 d", _lin(9, c2)),
        ("token norm and output norm over 2 d", [(c2,)] * 4),
        ("modulation generator: linear 2 d to 6 d (read at each of the 4 refinement iterations)", _lin(c2, 3 * c2)),
        ("4 refinement blocks at width 2 d (block table below, 14 tensors each)", sum((_block_shapes(c2, False) for _ in range(CAM_BLOCKS)), [])),
        ("predictor: linear 2 d to d, then linear d to 9", _lin(c2, D) + _lin(D, 9))]
    for name, out_ch, merger in (("depth head (DPT)", 3, False), ("pointmap head (DPT)", 4, False),
                                 ("normal head (DPT)", 4, False), ("gaussian depth head (DPT) and input merger", 3, True)):
        g[name] = [(f"DPT head with {out_ch} output channels" + (" and the 7 by 7 input merger" if merger else "") +
                    " (head table below)", _dpt_shapes(out_ch, merger))]
    g["gaussian parameter convolutions"] = [("3 by 3 convolution 128 to 256 without bias, 1 by 1 convolution 256 to 12 with bias",
                                             [(DPT_FEAT, DPT_FEAT // 2, 3, 3), (12, DPT_FEAT, 1, 1), (12,)])]
    return g


def param_groups():
    """name -> list of tensor shapes (the components of group_parts concatenated)."""
    return {k: sum((sh for _, sh in parts), []) for k, parts in group_parts().items()}


def count_params():
    g = param_groups()
    return {k: (len(v), sum(math.prod(s) for s in v)) for k, v in g.items()}


def _vals(shapes):
    return sum(math.prod(sh) for sh in shapes)


def _block_table():
    """The tensors of one block in checkpoint order as (name, shape expression, values at d = 1024); the four query-key vectors
    are one row."""
    d = D
    rows = [("norm 1 weight, bias", "2 x d", 2 * d), ("fused query-key-value weight", "3d x d", 3 * d * d),
            ("fused query-key-value bias", "3d", 3 * d),
            ("(trunk blocks only) query norm weight and bias, key norm weight and bias", "4 x d/16", 4 * (d // HEADS)),
            ("output weight", "d x d", d * d), ("output bias", "d", d), ("layer scale 1", "d", d),
            ("norm 2 weight, bias", "2 x d", 2 * d), ("feed-forward weight 1", "4d x d", 4 * d * d), ("feed-forward bias 1", "4d", 4 * d),
            ("feed-forward weight 2", "d x 4d", 4 * d * d), ("feed-forward bias 2", "d", d), ("layer scale 2", "d", d)]
    return rows


def _dpt_table():
    """The tensors of a DPT head in checkpoint order as (what, shapes)."""
    f, tap = DPT_FEAT, 2 * D
    rows = [("token norm over the 2 d = 2048 channels of a tap (weight, bias)", [(tap,), (tap,)]),
            ("projections, 1 by 1 convolutions from 2048 to 256, 512, 1024, 1024 channels, with bias",
             sum(([(c, tap, 1, 1), (c,)] for c in DPT_PROJ), [])),
            ("resize 0: transposed convolution 256 to 256, kernel 4, stride 4, with bias", [(256, 256, 4, 4), (256,)]),
            ("resize 1: transposed convolution 512 to 512, kernel 2, stride 2, with bias", [(512, 512, 2, 2), (512,)]),
            ("resize 2: none (the patch grid itself)", []),
            ("resize 3: convolution 1024 to 1024, kernel 3, stride 2, with bias", [(1024, 1024, 3, 3), (1024,)]),
            ("fusion inputs, 3 by 3 convolutions from 256, 512, 1024, 1024 to 256 channels, no bias",
             [(f, c, 3, 3) for c in DPT_PROJ])]
    for r in (1, 2, 3, 4):
        units = 1 if r == 4 else 2
        rows.append((f"refinement block {r}: {units} residual unit{'s' if units > 1 else ''} of two 3 by 3 convolutions 256 to 256 "
                     "with bias, then a 1 by 1 convolution 256 to 256 with bias", [(f, f, 3, 3), (f,)] * (2 * units) + [(f, f, 1, 1), (f,)]))
    rows += [("output convolution 1: 3 by 3, 256 to 128, with bias", [(f // 2, f, 3, 3), (f // 2,)]),
             ("output convolution 2: 3 by 3, 128 to 32, with bias", [(f // 8, f // 2, 3, 3), (f // 8,)]),
             ("output convolution 3: 1 by 1, 32 to the output channels (3, 4, 4 or 3), with bias", [("out", f // 8, 1, 1), ("out",)]),
             ("input merger (gaussian head only): 7 by 7 convolution from the 3 colour channels to 128, with bias", [(f // 2, 3, 7, 7), (f // 2,)])]
    return rows


def inventory():
    """The block table and the DPT head table that make every row of model_counts derivable."""
    d = D
    base, withqk = _vals(_block_shapes(d, False)), _vals(_block_shapes(d, True))
    blk = [[nm, sh, f"{v:,}"] for nm, sh, v in _block_table()]
    out = ["", "", "A block of width d, tensors in checkpoint order (the encoder's and the camera head's blocks have no query-key vectors; "
           "the trunk's have them, 4 tensors of d/16 = 64 values):", "",
           _table(["tensor", "shape", "values at d = 1024"], blk, "llr"), "",
           f"A block without the query-key vectors has 14 tensors and {base:,} values at d = 1024, and one with them 18 tensors and "
           f"{withqk:,}. At d = 2048, the camera head's width, a block has {_vals(_block_shapes(2 * d, False)):,} values.", "",
           "A DPT head has one body for all four dense heads, which differ only in the output channels of the last convolution (depth "
           "3: log depth, confidence and validity logit; pointmap 4: x, y, z and confidence; normals 4: a vector and its confidence; "
           "Gaussian depth 3, with the input merger). Its tensors, in checkpoint order, for the 3-channel head:", ""]
    rows = []
    for what, shp in _dpt_table():
        rows.append([what, len(shp), f"{_vals([tuple(3 if x == 'out' else x for x in sh) for sh in shp]):,}"])
    out.append(_table(["part of a DPT head", "tensors", "values"], rows, "lrr"))
    out += ["", "The 4-channel heads add 33 values (32 weights and 1 bias for the extra channel), and the input merger is the last row, "
            "which only the Gaussian depth head has, so that the 3-channel head without it has 62 tensors and the Gaussian depth head 64."]
    return "\n".join(out)


def _components():
    """What each group is made of, component by component, so that the totals of the table above are derivable."""
    rows = []
    for grp, parts in group_parts().items():
        for label, shapes in parts:
            rows.append([grp, label, len(shapes), f"{_vals([tuple(3 if x == 'out' else x for x in sh) for sh in shapes]):,}"])
    return ("\n\nThe groups, component by component (the encoder and trunk blocks are 24 of the block below; the 16-value table "
            "of rotary periods is stored once, in the first frame block, and is the reason the frame blocks have one tensor and 16 "
            "values more than the global blocks, which are otherwise identical: 24 x 18 = 432 tensors):\n\n" +
            _table(["group", "component", "tensors", "values"], rows, "llrr"))


def model_counts():
    cnt = count_params()
    rows = []
    for k, (nt, npar) in cnt.items():
        rows.append([k, nt, f"{npar:,}", f"{npar * 4 / 2 ** 20:,.1f}"])
    tot_t = sum(v[0] for v in cnt.values())
    tot_p = sum(v[1] for v in cnt.values())
    rows.append(["total", f"{tot_t:,}", f"{tot_p:,}", f"{tot_p * 4 / 2 ** 20:,.1f}"])
    check = ""
    if HEADER.is_file():
        hdr = json.loads(HEADER.read_text())
        ht = sum(math.prod(v) for v in hdr.values())
        by = {}
        for k, v in hdr.items():
            by[k.split(".")[0]] = by.get(k.split(".")[0], 0) + math.prod(v)
        mine = {"visual_geometry_transformer": sum(cnt[k][1] for k in list(cnt)[:4]) - 0, "cam_head": cnt["camera head"][1],
                "depth_head": cnt["depth head (DPT)"][1], "pts_head": cnt["pointmap head (DPT)"][1],
                "norm_head": cnt["normal head (DPT)"][1], "gs_head": cnt["gaussian depth head (DPT) and input merger"][1],
                "gs_renderer": cnt["gaussian parameter convolutions"][1]}
        agree = all(by[k] == mine[k] for k in mine)
        check = (f"\n\nThe tensor shapes of the released checkpoint's header (the committed fixture "
                 f"`crates/worldmirror2/tests/golden/header.json`) number {len(hdr):,} and hold {ht:,} values; grouped by the prefix of their "
                 f"names into the seven groups {', '.join('`' + k + '`' for k in mine)}, every group's count equals the count above "
                 f"({'all seven agree' if agree else 'DISAGREE'}).")
    return "\n".join([
        f"The model of the chapter, from its configuration: width d = {D}, {HEADS} heads of {D // HEADS}, MLP width {MLP} d, patch "
        f"{PATCH}, native size {IMG} (a grid of {IMG // PATCH} by {IMG // PATCH} patches), {REG} register tokens, an encoder of {DEPTH} blocks, "
        f"a trunk of {DEPTH} frame blocks and {DEPTH} global blocks, DPT heads of width {DPT_FEAT}, and a camera head of {CAM_BLOCKS} blocks at "
        f"width 2 d = {2 * D}. Each block has two layer norms, a fused query-key-value matrix, an output matrix, a feed-forward pair and "
        "two layer-scale vectors; a trunk block adds layer norms over the head width for queries and keys. Values are 4-byte floats; "
        "MiB is 2^20 bytes.", "",
        _table(["group", "tensors", "values", "MiB"], rows, "lrrr")]) + check + _components() + inventory()


# ------------------------------------------------------------------------------------------------ arithmetic of a pass

def attn_flops(pairs):
    """FLOPs of attention over `pairs` (query, key) pairs in all heads together: the scores (d multiply-adds per pair) and the
    weighted sum of values (d per pair), two FLOPs each."""
    return 2.0 * 2 * D * pairs


def linear_flops(tokens, d=D):
    """FLOPs of the four matrix products of a block over `tokens` tokens: 3 d^2 + d^2 + 4 d^2 + 4 d^2 = 12 d^2 multiply-adds each."""
    return 2.0 * 12 * d * d * tokens


def trunk_flops(S, hp, wp):
    """(linear FLOPs of the 48 trunk blocks, frame attention FLOPs, global attention FLOPs) for S frames of hp * wp patches."""
    n = SPECIALS + hp * wp
    lin = DEPTH * 2 * linear_flops(S * n)
    fr = DEPTH * attn_flops(S * n * n)
    gl = DEPTH * attn_flops((S * n) ** 2)
    return lin, fr, gl


def encoder_flops(S, hp, wp):
    n = 1 + REG + hp * wp
    conv = 2.0 * S * hp * wp * 3 * PATCH * PATCH * D
    return conv + DEPTH * (linear_flops(S * n) + attn_flops(S * n * n))


def cam_head_flops(S):
    c2 = 2 * D
    per_iter = (9 * c2 + c2 * 3 * c2 + CAM_BLOCKS * 12 * c2 * c2 + c2 * D + D * 9) * S + CAM_BLOCKS * 2 * c2 * S * S      # multiply-adds
    return 2.0 * CAM_ITERS * per_iter


def dpt_macs(hp, wp, out_ch, merger):
    """Multiply-adds of one DPT head on one frame of hp by wp patches (convolutions and transposed convolutions only; each is
    positions x input channels x output channels x kernel area, the positions being those of the output of a convolution and of
    the input of a transposed one)."""
    p, f, h, w = hp * wp, DPT_FEAT, hp * PATCH, wp * PATCH
    q = lambda a, b: ((a + 1) // 2) * ((b + 1) // 2)
    dims = [(4 * hp) * (4 * wp), (2 * hp) * (2 * wp), p, q(hp, wp)]
    m = 0
    m += sum(p * 2 * D * c for c in DPT_PROJ)                                         # the 1 x 1 projections
    m += p * 256 * 256 * 16 + p * 512 * 512 * 4 + q(hp, wp) * 1024 * 1024 * 9          # the three resize layers
    m += sum(dims[i] * DPT_PROJ[i] * f * 9 for i in range(4))                          # layer_rn
    m += 2 * dims[3] * f * f * 9 + dims[2] * f * f                                     # refinenet 4: one unit, then out_conv at scale 2
    for k, (res, out) in enumerate(((dims[2], dims[1]), (dims[1], dims[0]), (dims[0], (8 * hp) * (8 * wp)))):
        m += 4 * res * f * f * 9 + out * f * f                                         # two units of two convolutions, out_conv
    m += (8 * hp) * (8 * wp) * f * (f // 2) * 9                                        # output_conv1
    m += h * w * (f // 2) * (f // 8) * 9 + h * w * (f // 8) * out_ch                   # output_conv2
    if merger:
        m += h * w * 3 * (f // 2) * 49 + h * w * (f // 2) * f * 9 + h * w * f * 12      # input merger, 3 x 3 to 256, 1 x 1 to 12
    return m


def pass_flops(S, hp, wp):
    lin, fr, gl = trunk_flops(S, hp, wp)
    heads = {"depth": dpt_macs(hp, wp, 3, False), "points": dpt_macs(hp, wp, 4, False), "normals": dpt_macs(hp, wp, 4, False),
             "gaussians": dpt_macs(hp, wp, 3, True)}
    return {"encoder": encoder_flops(S, hp, wp), "trunk linear": lin, "frame attention": fr, "global attention": gl,
            "camera head": cam_head_flops(S), "dense heads": 2.0 * S * sum(heads.values()), "per-head": heads}


def trunk_attention():
    hp = wp = IMG // PATCH
    n = SPECIALS + hp * wp
    rows = []
    for S in (1, 2, 4, 8, 10, 16, 24):
        lin, fr, gl = trunk_flops(S, hp, wp)
        tot = lin + fr + gl
        rows.append([S, f"{S * n:,}", f"{S * n * n:,}", f"{(S * n) ** 2:,}", f"{gl / (DEPTH * 2 * linear_flops(S * n) / 2):.2f}",
                     f"{tot / 1e12:.1f}", f"{gl / tot:.2f}"])
    cross = 6 * D
    return "\n".join([
        f"Frames of {IMG} by {IMG} pixels: {hp} by {wp} = {hp * wp} patches, plus {SPECIALS} special tokens, so n = {n} tokens a frame. "
        "For S frames, a frame block scores S n^2 pairs per head and a global block (S n)^2. A block's attention costs "
        f"4 d pairs FLOPs (scores and weighted sum, d = {D} over all heads) and its four matrix products 24 d^2 FLOPs a token; the "
        "ratio column divides the FLOPs of the attention of a global block by those of its matrix products, and the share column "
        f"the FLOPs of all 24 global attentions by those of the whole trunk (48 blocks). The two cost the same where the attended "
        f"span is 6 d = {cross:,} tokens.", "",
        _table(["frames S", "tokens S n", "pairs, frame block", "pairs, global block", "global attention / its matrix products",
                "trunk TFLOP", "share in global attention"], rows, "rrrrrrr")])


def counts_per_pass():
    hp = wp = IMG // PATCH
    rows = []
    for S in (1, 4, 10):
        pf = pass_flops(S, hp, wp)
        tot = sum(v for k, v in pf.items() if k != "per-head")
        rows.append([S] + [f"{pf[k] / 1e12:.2f}" for k in ("encoder", "trunk linear", "frame attention", "global attention", "camera head",
                                                          "dense heads")] + [f"{tot / 1e12:.1f}"])
    pf = pass_flops(1, hp, wp)
    heads = pf["per-head"]
    return "\n".join([
        f"FLOPs of one forward pass (TFLOP, 2 per multiply-add) at {IMG} by {IMG} pixels, by stage: the {DEPTH}-block encoder, the matrix "
        "products of the 48 trunk blocks, the attention of the frame blocks and of the global blocks, the camera head (4 iterations of "
        "4 blocks over the S camera tokens) and the four dense heads, each run on every frame. Normalizations, activations, "
        "softmax and the resampling are not counted. The encoder works on 1 + 4 + 1369 = 1374 tokens a frame (class token, four registers "
        "and the patches), the trunk on 7 + 1369 = 1376 (the seven special tokens and the patches), and the attention of each is "
        "counted at its own token count.", "",
        _table(["frames", "encoder", "trunk matrix products", "frame attention", "global attention", "camera head", "dense heads", "total"],
               rows, "rrrrrrrr"), "",
        "One dense head on one frame costs, in multiply-adds: " + ", ".join(f"{k} {v / 1e9:.1f} G" for k, v in heads.items()) +
        f". The encoder and the heads grow linearly with S; the trunk's global attention grows with S squared."])


# ------------------------------------------------------------------------------------------------ what one pass may hold

MAX_PASS_PATCHES, MIN_PASS_FRAMES, MAX_PASS_FRAMES = 14336, 6, 24
ATTN_BUDGET = 1 << 30


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def input_grid(w, h, cap):
    """The pixel grid a w by h source is run at under a longest-edge cap: the longest side becomes min(longest, cap) floored to a
    multiple of 14 (never below 28), the other side is scaled by the same factor and rounded to a multiple of 14."""
    target = max(min(max(w, h), cap) // PATCH * PATCH, 2 * PATCH)
    if w >= h:
        return target, _round_half_up(h * (target / w) / PATCH) * PATCH
    return _round_half_up(w * (target / h) / PATCH) * PATCH, target


def largest_binding_bytes(S, hp, wp):
    """The largest single buffer of a pass: the larger of 2^30 (the budget of the chunked global scores), one frame's score slab of
    all heads, the fused query-key-value buffer and the MLP hidden buffer, in bytes of 4-byte floats."""
    n = SPECIALS + hp * wp
    rows = S * n
    return max(HEADS * n * n * 4, ATTN_BUDGET, 3 * rows * D * 4, rows * MLP * D * 4)


def frame_budget(w, h):
    """Frames one pass may hold on a w by h grid: 14336 patches divided by the patches of a frame, at least 6 and at most 24."""
    per = max((w // PATCH) * (h // PATCH), 1)
    return min(max(MAX_PASS_PATCHES // per, MIN_PASS_FRAMES), MAX_PASS_FRAMES)


def frame_budget_table():
    rows = []
    for (w, h, cap) in ((1920, 1080, 518), (640, 480, 518), (1080, 1920, 518), (518, 518, 518), (1920, 1080, 952), (800, 600, 952), (4000, 3000, 952),
                        (640, 480, 224)):
        gw, gh = input_grid(w, h, cap)
        per = (gw // PATCH) * (gh // PATCH)
        b = frame_budget(gw, gh)
        n = SPECIALS + per
        rows.append([f"{w} by {h}", cap, f"{gw} by {gh}", per, b, f"{b * per:,}", f"{(b * n) ** 2 / 1e6:,.0f}",
                     f"{largest_binding_bytes(b, gh // PATCH, gw // PATCH) / 2 ** 30:.2f}"])
    return "\n".join([
        "The grid and the frame budget for sources of several sizes under a longest-edge cap of 518 (the model's native size) and of "
        f"952. A frame has (grid width / {PATCH}) (grid height / {PATCH}) patches; the budget is {MAX_PASS_PATCHES} divided by that, "
        f"rounded down, then raised to {MIN_PASS_FRAMES} if lower and cut to {MAX_PASS_FRAMES} if higher (there is no device limit "
        "here). The grid is made as follows: the longest side becomes min(longest side, cap) rounded down to a multiple of 14 (at "
        "least 28), and the other side is multiplied by the same ratio (new longest / old longest) and rounded half up to a multiple "
        "of 14. 'Patches in the pass' is frames times the patches of a frame and does not count the seven special tokens a frame "
        "adds; 'pairs' is the number of (query, key) pairs a global block scores per head at the budget, in millions, and does count "
        "them: (frames x (patches a frame + 7))^2. The last column is the largest single buffer in GiB: the larger of 2^30 bytes, "
        "one frame's scores in all 16 heads, the fused query-key-value buffer and the feed-forward hidden buffer, in 4-byte floats.", "",
        _table(["source", "cap", "grid", "patches a frame", "frames", "patches in the pass", "pairs, global block (M)",
                "largest buffer (GiB)"], rows, "lrrrrrrr")])


# ------------------------------------------------------------------------------------------------ making the views agree

def fuse_depths(depth, conf, rtol):
    """Reconcile per-view depth maps with each other, all views at once from the unfused maps. For pixel k of view i with depth
    z > 0, the world point is p = C_i + z r, r the world ray (x, y, 1) rotated to the world. Each other view j: the camera depth
    of p in j must be > 1e-6; the pixel (u, v) it lands in (continuous) must lie inside the image; the depth d_j stored in the
    pixel containing (u, v) must satisfy |d_j - z_j| / z_j <= rtol, z_j the camera depth of p in j; then j's opinion is
    the point q = C_j + d_j r_j(u, v) (r_j the world ray of the continuous pixel position) written as a distance along r:
    c = (q - C_i) . r / (r . r), kept if c > 0. The fused depth is sum(w c) / sum(w) over the own depth and the opinions kept, with
    the weight of an opinion the confidence stored at the pixel it was read from. Returns (fused depths, support counts)."""
    cams = room()
    S = len(cams)
    fused_all, support_all, dis = [], [], []
    for i in range(S):
        ci = cams[i]
        fused, support = [0.0] * (W * H), [0] * (W * H)
        for py in range(H):
            for px in range(W):
                k = py * W + px
                z = depth[i][k]
                if z <= 0:
                    fused[k] = z
                    continue
                r = mv(ci["Rc"], pixel_ray(px, py))
                p = _add(ci["C"], _scale(z, r))
                acc, wsum, agree = conf[i][k] * z, conf[i][k], 0
                for j in range(S):
                    if j == i:
                        continue
                    cj = cams[j]
                    (u, v), zj = project(cj, p)
                    if zj <= 1e-6 or not (0 <= u < W and 0 <= v < H):
                        continue
                    kj = int(v) * W + int(u)
                    dj = depth[j][kj]
                    if dj <= 0 or abs(dj - zj) / zj > rtol:
                        continue
                    rj = mv(cj["Rc"], [(u - CXD) / FOC, (v - CYD) / FOC, 1.0])
                    q = _add(cj["C"], _scale(dj, rj))
                    cand = dot(sub(q, ci["C"]), r) / dot(r, r)
                    if cand <= 0:
                        continue
                    acc += conf[j][kj] * cand
                    wsum += conf[j][kj]
                    agree += 1
                    dis.append(abs(cand - z) / z)
                fused[k] = acc / wsum
                support[k] = agree
        fused_all.append(fused)
        support_all.append(support)
    return fused_all, support_all, (sum(dis) / len(dis) if dis else float("nan"))


def sd_conf(sd):
    return [s["cd"] for s in sd]


def _metric_depths():
    """The stand-in's depth head scaled by the one global scale of the alignment example (the median of truth / prediction over all
    three views), so that the views are in the room's metres."""
    cams = room()
    sd = standin()
    gs = _median([t / p for v in range(3) for p, t in zip(sd[v]["d"], cams[v]["z"])])
    return gs, [[gs * x for x in sd[v]["d"]] for v in range(3)]


def fusion_table():
    cams = room()
    sd = standin()
    gs, dep = _metric_depths()
    truth = [c["z"] for c in cams]
    edge = [hard_pixels(c["z"]) for c in cams]

    def err(d, mask=None):
        vals = [abs(d[v][k] - truth[v][k]) / truth[v][k] for v in range(3) for k in range(W * H) if mask is None or mask[v][k]]
        return sum(vals) / len(vals)

    nh = sum(sum(e) for e in edge)
    conf_e = [[math.exp(-2.0 * math.log((SIG_HARD if h else SIG_PIX) / SIG_PIX)) for h in sd[v]["hard"]] for v in range(3)]
    flat = [[1.0] * (W * H) for _ in range(3)]
    rows = [["none (the aligned depth head)", "-", "-", "-", f"{err(dep):.4f}", f"{err(dep, edge):.4f}"]]
    for rtol in (0.005, 0.02, 0.05, 0.2, 0.5):
        f, sup, dis = fuse_depths(dep, sd_conf(sd), rtol)
        have = sum(1 for v in range(3) for k in range(W * H) if sup[v][k] > 0) / (3 * W * H)
        rows.append([f"1 + exp(raw), rtol {rtol:g}", f"{have:.3f}", f"{sum(map(sum, sup)) / (3 * W * H):.3f}", f"{dis:.4f}",
                     f"{err(f):.4f}", f"{err(f, edge):.4f}"])
    rows2 = []
    for label, cf in (("flat weights (all 1)", flat), ("1 + exp(raw)", sd_conf(sd)), ("exp(raw) alone", conf_e)):
        f, sup, dis = fuse_depths(dep, cf, 0.05)
        rows2.append([label, f"{err(f):.4f}", f"{err(f, edge):.4f}"])
    f, sup, dis = fuse_depths(dep, sd_conf(sd), 0.05)
    hist = [0, 0, 0]
    for v in range(3):
        for k in range(W * H):
            hist[sup[v][k]] += 1
    # the second head: the pointmap, kept as a depth along the camera's forward axis
    pd = []
    for v in range(3):
        cam = cams[v]
        fwd = cam["R"][2]
        pd.append([dot(fwd, sub(_scale(gs, sd[v]["P"][k]), cam["C"])) for k in range(W * H)])
    blend = [[(sd[v]["cd"][k] * dep[v][k] + sd[v]["cp"][k] * pd[v][k]) / (sd[v]["cd"][k] + sd[v]["cp"][k]) for k in range(W * H)]
             for v in range(3)]
    rows3 = [["depth head", f"{err(dep):.4f}", f"{err(dep, edge):.4f}"], ["pointmap head, kept as a depth", f"{err(pd):.4f}", f"{err(pd, edge):.4f}"],
             ["confidence-weighted blend", f"{err(blend):.4f}", f"{err(blend, edge):.4f}"]]
    fb, supb, disb = fuse_depths(blend, sd_conf(sd), 0.05)
    rows3.append(["blend, then fused across views", f"{err(fb):.4f}", f"{err(fb, edge):.4f}"])
    return "\n".join([
        f"The stand-in's depth head scaled by the single scale {gs:.4f} (so the three views share the room's units), fused across "
        f"the three views by the rule described in the text with the known cameras. {nh} of the {3 * W * H} pixels are silhouette "
        "pixels (a depth more than 1.5 times a neighbour's). Columns: the share of pixels with at least one agreeing "
        "other view, the mean number of agreeing views, the mean relative disagreement |c - z| / z over the agreeing pairs before "
        "fusing, and AbsRel against the true depth over all pixels and over the silhouette pixels after fusing.", "",
        _table(["fusion", "share supported", "mean support", "disagreement", "AbsRel, all", "AbsRel, silhouettes"], rows, "lrrrrr"), "",
        f"Supported views at rtol 0.05: {hist[0]} pixels have none, {hist[1]} one, {hist[2]} two. Weights, at rtol 0.05:", "",
        _table(["weights", "AbsRel, all", "AbsRel, silhouettes"], rows2, "lrr"), "",
        "The second head. The pointmap head's points (times the same scale) written as the depth along the camera's forward axis, "
        "z = f . (X - C) with f the third row of R; its confidence equals the depth head's here, so the blend is their mean:", "",
        _table(["source of the depth", "AbsRel, all", "AbsRel, silhouettes"], rows3, "lrr")])


def confidence_weights():
    """Three estimates of one depth of 4 m, each with the raw confidence a network might emit, fused by three weightings."""
    truth = 4.0
    est = [4.04, 3.97, 4.80]
    raw = [2.0, 2.0, -3.0]
    ex = [math.exp(r) for r in raw]
    cf = [1.0 + e for e in ex]
    rows = []
    for label, w in (("flat", [1.0, 1.0, 1.0]), ("1 + exp(raw)", cf), ("exp(raw) alone", ex)):
        m = sum(a * b for a, b in zip(w, est)) / sum(w)
        sh = [x / sum(w) for x in w]
        rows.append([label, _v(sh, 4), f"{m:.4f}", f"{abs(m - truth):.4f}"])
    return "\n".join([
        "Three views estimate the same depth, truth 4 m: 4.04, 3.97 and 4.80, with raw confidences 2, 2 and -3 (the third view is "
        f"flagged as a guess). exp(raw) = {_v(ex, 4)}, so 1 + exp(raw) = {_v(cf, 4)}.", "",
        _table(["weights", "share of each view", "fused depth", "error"], rows, "lrrr")])


# ------------------------------------------------------------------------------------------------ a start for the classical pipeline

SPARSE_NAME = "feed_forward_3d.sparse"


def sparse_standin():
    """The stand-in on the 60 points of Chapter 4's scene (`structure_from_motion.scene`, four cameras, camera 0 the world). For
    view v = 0 to 3: six normal() draws for the biases of the depth head (b0, bx, by; sigma 0.02, 0.01, 0.01) and of the pointmap
    head; then for each point j = 0 to 59, five normal() draws: the relative error of the depth head and of the pointmap head
    (sigma 0.01 each) and three lateral errors of the pointmap head (sigma 0.004 z). The bias form uses the normalized
    coordinates (x, y) of the exact projection. Returns per view the depth predictions d[v][j] and the pointmap points P[v][j]
    (both in units of 0.37 of a metre)."""
    def make():
        pts, poses, pix0, pix = sfm.scene()
        g = bookrng.generator(SPARSE_NAME)
        s = SCALE_UNKNOWN
        D_, P_ = [], []
        for v, (R, t) in enumerate(poses):
            Rc, C = tr(R), sfm.centre(R, t)
            bd = [sg * g.normal() for sg in SIG_BIAS]
            bp = [sg * g.normal() for sg in SIG_BIAS]
            dv, pv = [], []
            for j, X in enumerate(pts):
                x, y = (pix0[v][j][0] - sfm.CX) / sfm.F, (pix0[v][j][1] - sfm.CY) / sfm.F
                z = _add(mv(R, X), t)[2]
                ed, ep = SIG_PIX * g.normal(), SIG_PIX * g.normal()
                lat = [SIG_LAT * z * g.normal() for _ in range(3)]
                dv.append(s * z * (1.0 + bd[0] + bd[1] * x + bd[2] * y + ed))
                ray = mv(Rc, [x, y, 1.0])
                pv.append(_scale(s, _add(_add(C, _scale(z * (1.0 + bp[0] + bp[1] * x + bp[2] * y + ep), ray)), lat)))
            D_.append(dv)
            P_.append(pv)
        return D_, P_
    return _once("sparse", make)


def seed_or_replace():
    pts, poses, pix0, pix = sfm.scene()
    D_, P_ = sparse_standin()
    s = SCALE_UNKNOWN
    nv, npt = len(poses), len(pts)
    # tracks by position in space: the nearest pointmap entry of view v to that of point j in view 0, if mutual
    obs, wrong = [], 0
    tracks = [[(0, j)] for j in range(npt)]
    for v in range(1, nv):
        nn_f = [min(range(npt), key=lambda m: norm(sub(P_[v][m], P_[0][j]))) for j in range(npt)]
        nn_b = [min(range(npt), key=lambda j: norm(sub(P_[v][m], P_[0][j]))) for m in range(npt)]
        for j in range(npt):
            m = nn_f[j]
            if nn_b[m] == j:
                tracks[j].append((v, m))
            if nn_b[nn_f[j]] == j and m != j:
                wrong += 1
    # a camera from each view's depth and pointmap: rigid fit of the camera-frame points to the world points
    est = [(IDENT_R, [0.0, 0.0, 0.0])]
    for v in range(1, nv):
        cam_pts = [_scale(D_[v][m], sfm.normalized(pix[v][m])) for m in range(npt)]
        Rc, Cc = (lambda r: (r[1], r[2]))(umeyama(cam_pts, P_[v], with_scale=False))
        est.append((Rc, Cc))
    lam = 1.0 / norm(est[1][1])                       # the baseline of cameras 0 and 1 becomes 1, as in the adjustment of Chapter 4
    poses_est = [[tr(Rc), _scale(-lam, mv(tr(Rc), Cc))] for Rc, Cc in est]
    X0, obs = [], []
    for j in range(npt):
        members = tracks[j]
        X0.append(_scale(lam / len(members), [sum(P_[v][m][k] for v, m in members) for k in range(3)]))
        for v, m in members:
            obs.append((v, j, pix[v][m][0], pix[v][m][1]))
    P0 = sfm.Problem(poses_est, X0, obs, ["held", "scale", "free", "free"], anchor=0, scale=1, f=sfm.F, free_f=False, delta=2.0)
    e0, rms0 = sfm.truth_errors(P0), P0.rms()
    hist = P0.solve(60)
    e1, rms1 = sfm.truth_errors(P0), P0.rms()
    # drop what the adjustment cannot explain (a residual above 4 px), the points left with one observation, and adjust again
    Q, ndrop, kept = sfm.drop_unexplained(P0, 4.0)
    histq = Q.solve(60)
    e2, rms2 = sfm.truth_errors(Q), Q.rms()
    # the floor: the same adjustment on all 240 true observations, started at the true cameras and points
    sc = norm(sfm.centre(*poses[1]))
    Pt = sfm.Problem([[R, [x / sc for x in t]] for R, t in poses], [[x / sc for x in p] for p in pts],
                     [(v, j, pix[v][j][0], pix[v][j][1]) for v in range(nv) for j in range(npt)],
                     ["held", "scale", "free", "free"], anchor=0, scale=1, f=sfm.F, free_f=False, delta=2.0)
    histt = Pt.solve(60)
    et, rmst = sfm.truth_errors(Pt), Pt.rms()
    # the classical pipeline of Chapter 4 on the same scene
    Pc, histc, pre, post, truth = sfm.incremental_to_ba()

    def fmt3(e, k):
        return " / ".join(f"{x[k]:.2f}" if k == 0 else f"{x[k]:.3f}" for x in e)

    rows = [
        ["Chapter 4, before its adjustment", f"{len(Pc.points)} / {len(Pc.obs)}", fmt3(pre, 0), fmt3(pre, 1), f"{histc[0][4]:.4f}"],
        ["Chapter 4, after its adjustment", f"{len(Pc.points)} / {len(Pc.obs)}", fmt3(post, 0), fmt3(post, 1), f"{Pc.rms():.4f}"],
        ["network only: cameras and points from the pointmaps", f"{len(P0.points)} / {len(obs)}", fmt3(e0, 0), fmt3(e0, 1), f"{rms0:.4f}"],
        ["network, then the adjustment of Chapter 4", f"{len(P0.points)} / {len(obs)}", fmt3(e1, 0), fmt3(e1, 1), f"{rms1:.4f}"],
        [f"network, adjustment, drop residuals above 4 px ({ndrop}), adjust again", f"{len(Q.points)} / {len(Q.obs)}", fmt3(e2, 0), fmt3(e2, 1),
         f"{rms2:.4f}"],
        ["reference: true cameras and points, all 240 true observations", f"{npt} / {npt * nv}", fmt3(et, 0), fmt3(et, 1), f"{rmst:.4f}"],
    ]
    return "\n".join([
        f"The sparse scene of Chapter 4 (seed {bookrng.seed_of(sfm.SCENE_NAME)} for the points, {bookrng.seed_of(sfm.NOISE_NAME)} for the pixel "
        f"noise) with the stand-in's pointmaps at the 60 keypoints (`feed_forward_3d.sparse`, seed {bookrng.seed_of(SPARSE_NAME)}). "
        "The sparse stand-in follows the rule of the room's (the first worked block of the section on cameras), with the 60 projections of "
        "the noise-free pixels in place of a pixel grid: for view v = 0 to 3 in turn, six `normal()` draws for the biases of the depth head "
        "and the pointmap head (standard deviations 0.02, 0.01, 0.01), then for each point j = 0 to 59 five `normal()` draws "
        "(e, e' with sigma 0.01, three lateral errors with 0.004 z), where (x, y) are the normalized coordinates of the exact projection "
        "of the point in the view, z its true camera depth and no point is hard. Its outputs are in the same units, 0.37 of a metre. "
        "Observed pixels are the noisy ones of Chapter 4. "
        "Tracks: a keypoint of view 0 is matched to the keypoint of view v whose pointmap entry is nearest in space, if the two are mutual "
        f"nearest neighbours; {sum(len(t) for t in tracks)} of the {npt * nv} possible entries land in tracks, {wrong} matches are wrong. "
        "Cameras: for each view v > 0, the rigid fit of its camera-frame points d (x, y, 1) (the depth head and the observed pixel) to its pointmap "
        "entries; camera 0 is the identity. Points: the mean of the track's pointmap entries. Both are divided by the "
        "distance of camera 1 from camera 0, so the baseline is 1 as in Chapter 4's gauge. Errors are those of "
        "Chapter 4's table (rotation in degrees, centre in metres after the true baseline is restored) for cameras 1, 2 and 3.", "",
        _table(["path", "points / observations", "rotation error (deg)", "centre error (m)", "RMS reprojection (px)"],
               rows, "lrrrr"), "",
        "The last row is the floor: the half pixel of noise alone, with no matching, RANSAC or network involved."])


IDENT_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def work_comparison():
    n, nc, dense, reduced, dense_bytes, red_bytes = sfm.work_counts(4, 60, 4)
    hp = wp = IMG // PATCH
    pf = pass_flops(4, hp, wp)
    tot = sum(v for k, v in pf.items() if k != "per-head")
    wbytes = sum(v[1] for v in count_params().values()) * 4
    rows = [["one Levenberg-Marquardt step of the toy, dense normal equations", f"{2 * dense:.2e}", f"{dense_bytes:.2e}"],
            ["the same step by the Schur complement", f"{2 * reduced:.2e}", f"{red_bytes:.2e}"],
            [f"one forward pass of the model, 4 frames of {IMG} by {IMG}", f"{tot:.2e}", f"{wbytes:.2e}"]]
    return "\n".join([
        "Work in FLOPs (2 per multiply-add) and the bytes of the large operand, from the counts of Chapter 4 for a bundle "
        "adjustment of 4 cameras and 60 points seen 4 times each (the normal-equation matrix, 8 bytes a value) and from the counts "
        "of this chapter for the model (its 4-byte weights, each read at least once).", "",
        _table(["computation", "FLOPs", "bytes"], rows, "lrr"), "",
        f"The pass needs {tot / (2 * reduced):.1e} times the arithmetic of the Schur-complement step ({tot / (2 * dense):.1e} times that of "
        f"the dense step), and performs {tot / wbytes:,.0f} FLOPs for every byte of weights it reads."])
