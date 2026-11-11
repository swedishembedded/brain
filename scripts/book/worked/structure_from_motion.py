# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Structure from Motion and Multi-View Stereo".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: structure_from_motion.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree.

One synthetic scene carries the whole chapter: 60 points drawn from a seeded generator, seen by four pinhole cameras (no lens
distortion; that was the previous chapter) with Gaussian pixel noise. On it:

* triangulation and the depth error of a stereo pair as a function of its baseline;
* feature matching by nearest neighbour, the ratio test and the mutual test, scored against the truth;
* the epipolar constraint and the essential matrix, the eight-point algorithm, RANSAC with its adaptive iteration count, the
  four-fold ambiguity of the pose and the triangulation that resolves it;
* registration of a third camera against the points of the first two, by the linear (DLT) absolute pose inside RANSAC;
* bundle adjustment by Levenberg-Marquardt, with the points eliminated by the Schur complement, checked against the dense
  solution, its work counted, the gauge made explicit, and a self-calibrating focal length;
* the similarity that puts two reconstructions of the same cameras into one world, and the gate on its residual;
* a plane sweep and PatchMatch on a small textured, tilted plane seen by three cameras, scored by weighted normalized
  cross-correlation, with the work of each counted.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, sin, cos, exp, log, atan2, acos) unless a
block says otherwise. Sums are accumulated left to right in index order. Singular vectors come from the one-sided Jacobi SVD of
the matrix chapter of Volume I (`decompositions._svd`); a null vector is the right singular vector of the smallest singular
value. Every random number comes from the book's SplitMix64 (`bookrng.generator`, stream names printed by the chapter).

Swedish Embedded AB implements solutions for photogrammetry, camera calibration and dense reconstruction from ordinary photographs.
If your team needs expertise in structure from motion and multi-view stereo, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import camera_geometry as cg
import decompositions as dc
import images as im
import stable

_table = im._table
_g = dc._g
mm, mv, tr, dot, norm, unit, cross, sub, det3 = cg.mm, cg.mv, cg.tr, cg.dot, cg.norm, cg.unit, cg.cross, cg.sub, cg.det3

# ------------------------------------------------------------------------------------------------ the scene

F, CX, CY = 500.0, 320.0, 240.0          # focal length in pixels and principal point of every camera
WIDTH, HEIGHT = 640, 480
NPOINTS, SIGMA = 60, 0.5                 # scene points; pixel noise per coordinate
EYES = [[0.0, 0.0, 0.0], [0.6, 0.08, 0.05], [1.3, -0.05, 0.15], [2.0, 0.06, 0.1]]
TARGET, UP = [0.6, 0.0, 5.5], [0.0, -1.0, 0.0]
SCENE_NAME, NOISE_NAME, FEATURE_NAME = "structure_from_motion.scene", "structure_from_motion.noise", "structure_from_motion.features"
_cache = {}


def _once(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def _fmt(v, d=4):
    return _g(v, d)


def _vec(x, d=4):
    return "(" + ", ".join(_g(c, d) for c in x) + ")"


def project(R, t, X):
    c = [a + b for a, b in zip(mv(R, X), t)]
    return (CX + F * c[0] / c[2], CY + F * c[1] / c[2])


def normalized(px):
    """Pixel (u, v) to the normalized image point (x, y, 1) = K^{-1} (u, v, 1)."""
    return [(px[0] - CX) / F, (px[1] - CY) / F, 1.0]


def scene():
    """points, poses [(R, t)], clean pixels pix0[v][j] and noisy pixels pix[v][j] of the synthetic capture."""
    def make():
        g = bookrng.generator(SCENE_NAME)
        pts = []
        for _ in range(NPOINTS):
            x = 3.6 * g.uniform() - 1.8
            y = 2.4 * g.uniform() - 1.2
            z = 4.0 + 3.0 * g.uniform()
            pts.append([x, y, z])
        poses = [([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])]
        poses += [cg.look_at(e, TARGET, UP) for e in EYES[1:]]
        pix0 = [[project(R, t, X) for X in pts] for R, t in poses]
        n = bookrng.generator(NOISE_NAME)
        pix = []
        for view in pix0:
            row = []
            for (u, v) in view:
                du, dv = SIGMA * n.normal(), SIGMA * n.normal()
                row.append((u + du, v + dv))
            pix.append(row)
        return pts, poses, pix0, pix
    return _once("scene", make)


def centre(R, t):
    """Camera centre -R^T t."""
    return [-x for x in mv(tr(R), t)]


def scene_setup():
    pts, poses, pix0, pix = scene()
    seed_s, seed_n = bookrng.seed_of(SCENE_NAME), bookrng.seed_of(NOISE_NAME)
    rows = []
    for i, (R, t) in enumerate(poses):
        us = [p[0] for p in pix0[i]]
        vs = [p[1] for p in pix0[i]]
        rows.append([i, _vec(EYES[i], 3), "straight ahead (R = I)" if i == 0 else _vec(TARGET, 3),
                     f"{min(us):.1f} .. {max(us):.1f}", f"{min(vs):.1f} .. {max(vs):.1f}"])
    out = [f"Every camera has f = {F:g} pixels, principal point ({CX:g}, {CY:g}) and a {WIDTH} by {HEIGHT} image, with no lens "
           f"distortion. The {NPOINTS} scene points come from `bookrng.generator(\"{SCENE_NAME}\")` (seed {seed_s}): for each "
           "point in turn, three `uniform()` draws give X = 3.6 u - 1.8, Y = 2.4 u - 1.2 and Z = 4 + 3 u (metres, in the frame of "
           "camera 0, x right, y down, z forward). Camera 0 is the world frame; cameras 1 to 3 are placed with the `look_at` of "
           "the previous chapter at the eye positions below, looking at one target, with up = (0, -1, 0). The observed pixels "
           f"are the projections plus noise from `bookrng.generator(\"{NOISE_NAME}\")` (seed {seed_n}), standard deviation "
           f"{SIGMA:g} pixel per coordinate: for view 0 to 3 in turn and point 0 to {NPOINTS - 1} in turn, `normal()` for u "
           "and then `normal()` for v.", "",
           _table(["camera", "eye C (metres)", "looks at", "u of the 60 points", "v of the 60 points"], rows, "lllll"), "",
           "First three points (X, Y, Z) and their noisy pixels in views 0 and 3:", ""]
    r2 = []
    for j in range(3):
        r2.append([j, _vec(pts[j], 4), "(%.2f, %.2f)" % pix[0][j], "(%.2f, %.2f)" % pix[3][j]])
    out.append(_table(["point", "(X, Y, Z)", "view 0 (u, v)", "view 3 (u, v)"], r2, "lrrr"))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ feature matching

DESC = 32            # descriptor length of the toy features
CLUTTER = 40         # keypoints per image that belong to no scene point
TWINS = 10           # points 10 .. 19 are near-copies of points 0 .. 9 (a repeated texture)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def features():
    """keypoints[v] = list of (u, v, descriptor, point index or -1), in shuffled order, for the four views."""
    def make():
        pts, poses, pix0, pix = scene()
        g = bookrng.generator(FEATURE_NAME)
        base = [_unit([g.normal() for _ in range(DESC)]) for _ in range(NPOINTS)]
        for k in range(TWINS):
            base[k + TWINS] = _unit([a + 0.04 * g.normal() for a in base[k]])
        views = []
        for v in range(4):
            kps = [(pix[v][j][0], pix[v][j][1], _unit([a + 0.05 * g.normal() for a in base[j]]), j) for j in range(NPOINTS)]
            for _ in range(CLUTTER):
                u, w = WIDTH * g.uniform(), HEIGHT * g.uniform()
                kps.append((u, w, _unit([g.normal() for _ in range(DESC)]), -1))
            g.shuffle(kps)
            views.append(kps)
        return views
    return _once("features", make)


def best_two(A, B):
    """For each descriptor of A: (index in B of its largest dot product, that value, the second largest); first index wins ties."""
    out = []
    for a in A:
        bi, b1, b2 = -1, -1.0, -1.0
        for j, b in enumerate(B):
            s = sum(p * q for p, q in zip(a, b))
            if s > b1:
                bi, b2, b1 = j, b1, s
            elif s > b2:
                b2 = s
        out.append((bi, b1, b2))
    return out


def _dist(s):
    return math.sqrt(max(0.0, 2.0 - 2.0 * s))


def match(A, B, ratio=None, mutual=True):
    """Index pairs (i, j): nearest neighbour of A[i] in B, kept if mutual and if dist1 < ratio * dist2."""
    ab, ba = best_two(A, B), best_two(B, A)
    out = []
    for i, (j, s1, s2) in enumerate(ab):
        if mutual and ba[j][0] != i:
            continue
        if ratio is not None and not _dist(s1) < ratio * _dist(s2):
            continue
        out.append((i, j))
    return out


def _pair_matches(v0, v1, ratio=None, mutual=True):
    kp = features()
    A, B = [k[2] for k in kp[v0]], [k[2] for k in kp[v1]]
    return match(A, B, ratio, mutual)


def matching_table():
    kp = features()
    seed = bookrng.seed_of(FEATURE_NAME)
    A, B = kp[0], kp[1]
    both = {k[3] for k in A if k[3] >= 0} & {k[3] for k in B if k[3] >= 0}
    rows = []
    for label, ratio, mutual in (("nearest neighbour only", None, False), ("mutual only", None, True), ("ratio 0.9", 0.9, False),
                                 ("ratio 0.9 and mutual", 0.9, True), ("ratio 0.8", 0.8, False), ("ratio 0.8 and mutual", 0.8, True),
                                 ("ratio 0.7 and mutual", 0.7, True)):
        m = _pair_matches(0, 1, ratio, mutual)
        good = [(i, j) for i, j in m if A[i][3] >= 0 and A[i][3] == B[j][3]]
        twin = sum(1 for i, j in good if A[i][3] < 2 * TWINS)
        rows.append([label, len(m), len(good), len(m) - len(good), f"{100.0 * len(good) / max(1, len(m)):.0f}%",
                     f"{100.0 * len(good) / len(both):.0f}%", twin])
    return seed, len(both), rows


def feature_matching():
    seed, nboth, rows = matching_table()
    out = [f"Toy features for the four views: {DESC}-number descriptors with unit length, from `bookrng.generator(\"{FEATURE_NAME}\")` "
           f"(seed {seed}). Draw order: the base descriptors of points 0 to {NPOINTS - 1} ({DESC} `normal()` draws each, divided by "
           f"the length); then, for k = 0 .. {TWINS - 1}, point {TWINS} + k gets the descriptor of point k plus 0.04 `normal()` on "
           "every entry, renormalized (a repeated texture, so that ten pairs of points look alike); then for view 0, 1, 2, 3 in "
           f"turn, the {NPOINTS} keypoints of the scene points in point order (their pixels from the noisy observations; the "
           f"descriptor is the base descriptor plus 0.05 `normal()` on every entry, renormalized), then {CLUTTER} clutter keypoints "
           "(u = 640 `uniform()`, v = 480 `uniform()`, then a fresh unit descriptor from 32 `normal()` draws), and finally "
           "`shuffle` of that list of 100 keypoints.", "",
           f"Matching view 0 against view 1, {nboth} scene points have a keypoint in both. The nearest neighbour of a descriptor is the "
           "one with the largest dot product (unit descriptors, so the distance is sqrt(2 - 2 dot)); the ratio test keeps a match "
           "when its distance is below the ratio times the distance of the second-nearest; the mutual test keeps it when the "
           "nearest neighbour of the match in the other image is the descriptor we started from. A match is correct when both "
           "keypoints belong to the same scene point.", "",
           _table(["rule", "matches", "correct", "wrong", "precision", "recall", "correct matches among points 0 to 19"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ epipolar geometry

def skew(t):
    return [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]


def essential_of(R, t):
    """E = [t]x R for the unit translation direction t: x2^T E x1 = 0 for the images x1, x2 of one point in two cameras."""
    return mm(skew(unit(t)), R)


def outer(u, v):
    return [[a * b for b in v] for a in u]


def matadd(A, B):
    return [[a + b for a, b in zip(r, s)] for r, s in zip(A, B)]


def fro(A):
    return math.sqrt(sum(x * x for r in A for x in r))


def transform_ab(E, a):
    return mv(E, a)


def epipolar_residual(E, a, b):
    """The algebraic epipolar residual b^T E a."""
    return dot(b, mv(E, a))


def sampson_px(E, a, b):
    """First-order distance in pixels of the correspondence a <-> b (normalized points) to the epipolar geometry E:
    |b^T E a| / sqrt((E a)_0^2 + (E a)_1^2 + (E^T b)_0^2 + (E^T b)_1^2), times the focal length."""
    ea = mv(E, a)
    etb = mv(tr(E), b)
    den = ea[0] ** 2 + ea[1] ** 2 + etb[0] ** 2 + etb[1] ** 2
    return F * abs(dot(b, ea)) / math.sqrt(max(den, 1e-300))


def hartley_2d(P):
    """Hartley's normalization of 2D points (chapter 3): the similarity that moves the centroid to the origin and scales the mean
    distance from it to sqrt(2), as a 3 by 3 matrix."""
    n = len(P)
    mx, my = sum(p[0] for p in P) / n, sum(p[1] for p in P) / n
    sc = math.sqrt(2.0) / (sum(math.hypot(p[0] - mx, p[1] - my) for p in P) / n)
    return [[sc, 0.0, -sc * mx], [0.0, sc, -sc * my], [0.0, 0.0, 1.0]]


def nearest_essential(M):
    """The essential matrix nearest to M: its singular values (s1, s2, s3) replaced by (1, 1, 0)."""
    U, _, V = dc._svd(M)
    return matadd(outer(U[0], V[0]), outer(U[1], V[1]))


def eight_point(a, b, essential=True):
    """The eight-point algorithm on n >= 8 correspondences of normalized points a <-> b. Each set is moved by Hartley's similarity
    T_a, T_b (chapter 3); the null vector of the n by 9 matrix whose row i is (b_r a_c) over r, c in row-major order gives E', and
    M = T_b^T E' T_a. With essential=True the nearest essential matrix is returned (singular values 1, 1, 0); otherwise the nearest
    matrix of rank two (singular values kept for the two largest, the third set to 0), the form a RANSAC hypothesis uses."""
    Ta, Tb = hartley_2d(a), hartley_2d(b)
    an = [mv(Ta, x) for x in a]
    bn = [mv(Tb, x) for x in b]
    rows = [[bi[r] * ai[c] for r in range(3) for c in range(3)] for ai, bi in zip(an, bn)]
    v, _ = cg.null_vector(rows, 9)
    En = [v[0:3], v[3:6], v[6:9]]
    U, sg, V = dc._svd(En)
    if essential:
        M = mm(mm(tr(Tb), En), Ta)
        return nearest_essential(M)
    En = matadd(outer([sg[0] * x for x in U[0]], V[0]), outer([sg[1] * x for x in U[1]], V[1]))
    return mm(mm(tr(Tb), En), Ta)


W3 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def decompose_essential(E):
    """The four (R, t) with E proportional to [t]x R: R in {U W V^T, U W^T V^T} (rotations: the signs of U and V are fixed so
    that det U = det V = +1) and t = +-u_3, the last column of U."""
    U, _, V = dc._svd(E)
    Um = [[U[k][i] for k in range(3)] for i in range(3)]
    Vm = [[V[k][i] for k in range(3)] for i in range(3)]
    if det3(Um) < 0:
        Um = [[x if k < 2 else -x for k, x in enumerate(r)] for r in Um]
    if det3(Vm) < 0:
        Vm = [[x if k < 2 else -x for k, x in enumerate(r)] for r in Vm]
    t = [Um[i][2] for i in range(3)]
    R1 = mm(mm(Um, W3), tr(Vm))
    R2 = mm(mm(Um, tr(W3)), tr(Vm))
    return [(R1, t), (R1, [-x for x in t]), (R2, t), (R2, [-x for x in t])]


def triangulate(Ps, xs):
    """Linear triangulation: the point X that makes x p3 . X - p1 . X and y p3 . X - p2 . X small in every view, where P = [R | t]
    has rows p1, p2, p3 and (x, y) is the normalized image point. The null vector of the 2n by 4 matrix is (X, W); X / W."""
    rows = []
    for (R, t), x in zip(Ps, xs):
        P = [list(R[i]) + [t[i]] for i in range(3)]
        rows.append([x[0] * P[2][c] - P[0][c] for c in range(4)])
        rows.append([x[1] * P[2][c] - P[1][c] for c in range(4)])
    h, _ = cg.null_vector(rows, 4)
    return [h[0] / h[3], h[1] / h[3], h[2] / h[3]]


def depth_in(R, t, X):
    return mv(R, X)[2] + t[2]


def rot_angle(R):
    """Angle in degrees of a rotation matrix."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0))))


def rot_error(Ra, Rb):
    """Angle in degrees of the rotation that takes Rb to Ra."""
    return rot_angle(mm(Ra, tr(Rb)))


def dir_error(ta, tb):
    """Angle in degrees between two directions."""
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(unit(ta), unit(tb))))))


def true_pair(v0=0, v1=1):
    """(R, t) taking camera v0's frame to camera v1's frame."""
    pts, poses, _, _ = scene()
    (R0, t0), (R1, t1) = poses[v0], poses[v1]
    R = mm(R1, tr(R0))
    t = sub(t1, mv(R, t0))
    return R, t


def epipole_pixels():
    """The epipoles of cameras 0 and 1: where each camera's centre appears in the other image."""
    R, t = true_pair(0, 1)
    e1 = mv(tr(R), [-x for x in t])          # centre of camera 1 in camera 0's frame is -R^T t
    c0_in_1 = t                               # centre of camera 0 (the origin) in camera 1's frame
    pix = lambda c: (CX + F * c[0] / c[2], CY + F * c[1] / c[2])
    return pix(e1), pix(c0_in_1)


def epipolar_constraint():
    pts, poses, pix0, pix = scene()
    R, t = true_pair(0, 1)
    E = essential_of(R, t)
    U, s, V = dc._svd(E)
    a = [normalized(p) for p in pix0[0]]
    b = [normalized(p) for p in pix0[1]]
    an = [normalized(p) for p in pix[0]]
    bn = [normalized(p) for p in pix[1]]
    rows = []
    for j in range(3):
        wrong = (j + 5)
        rows.append([f"point {j} against point {j}", "%.1e" % abs(epipolar_residual(E, a[j], b[j])),
                     "%.2f" % sampson_px(E, an[j], bn[j]), "yes"])
    for j in range(3):
        k = j + 5
        rows.append([f"point {j} against point {k}", "%.1e" % abs(epipolar_residual(E, a[j], b[k])),
                     "%.2f" % sampson_px(E, an[j], bn[k]), "no"])
    e0p, e1p = epipole_pixels()
    Kinv = [[1 / F, 0.0, -CX / F], [0.0, 1 / F, -CY / F], [0.0, 0.0, 1.0]]
    Fm = mm(mm(tr(Kinv), E), Kinv)
    line = mv(Fm, [pix[0][0][0], pix[0][0][1], 1.0])
    ln = math.hypot(line[0], line[1])
    d = abs(line[0] * pix[1][0][0] + line[1] * pix[1][0][1] + line[2]) / ln
    out = ["The pair (camera 0, camera 1): the relative rotation R = R_1 R_0^T and translation t = t_1 - R t_0 take a point of camera "
           "0's frame to camera 1's, and E = [t]x R with t scaled to unit length.", "",
           "R = " + cg.mat(R, 6), "t = " + _vec(t, 6) + f", |t| = {norm(t):.6f} (the baseline, in metres)",
           "E = " + cg.mat(E, 6),
           "singular values of E: " + ", ".join(_g(x, 6) for x in s) + f"; det E = {det3(E):.1e}; "
           f"the epipole of camera 1 in image 0 is at ({e0p[0]:.1f}, {e0p[1]:.1f}) and that of camera 0 in image 1 at "
           f"({e1p[0]:.1f}, {e1p[1]:.1f}).", "",
           _table(["pair of keypoints", "exact pixels: |b^T E a|", "noisy pixels: Sampson distance (px)", "same scene point"], rows, "lrrl"), "",
           f"The epipolar line of the noisy view-0 pixel of point 0 in image 1 is l = F x with F = K^-T E K^-1: "
           f"l = ({line[0]:.6f}, {line[1]:.6f}, {line[2]:.3f}), and its noisy view-1 pixel ({pix[1][0][0]:.2f}, {pix[1][0][1]:.2f}) "
           f"lies {d:.2f} pixels from it."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ Gauss-Newton and Levenberg-Marquardt

def cholesky_solve(A, b):
    """Solve A x = b for a symmetric positive definite A by A = L L^T and two triangular solves; None if a pivot is not positive."""
    n = len(b)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def tangent_basis(v):
    """Two unit vectors that complete the unit vector v to an orthonormal basis."""
    a = [1.0, 0.0, 0.0] if abs(v[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = unit(cross(v, a))
    return e1, cross(v, e1)


def signed_sampson_px(E, a, b):
    """b^T E a divided by the same denominator as sampson_px, times the focal length: a signed distance in pixels."""
    ea = mv(E, a)
    etb = mv(tr(E), b)
    den = ea[0] ** 2 + ea[1] ** 2 + etb[0] ** 2 + etb[1] ** 2
    return F * dot(b, ea) / math.sqrt(max(den, 1e-300))


def refine_essential(R, t, a, b, iters=20):
    """Levenberg-Marquardt on the five parameters of E = [t]x R: a rotation increment w (R <- exp(w) R) and two steps of the unit
    translation t in the plane across it (t <- unit(t + d1 e1 + d2 e2)), minimizing the sum of squared signed Sampson distances
    (pixels). Jacobian by central differences with step 1e-7; damping lambda starts at 1e-4, is multiplied by 0.3 after a step that
    lowers the cost and by 10 after one that does not (at most 6 tries per iteration); the loop stops when no try lowers the cost.
    Returns (R, t, the cost before, the cost after)."""
    t = unit(t)

    def step(R, t, d):
        e1, e2 = tangent_basis(t)
        Rn = mm(cg.rodrigues(d[0:3]), R)
        tn = unit([t[i] + d[3] * e1[i] + d[4] * e2[i] for i in range(3)])
        return Rn, tn

    def resid(R, t):
        E = essential_of(R, t)
        return [signed_sampson_px(E, x, y) for x, y in zip(a, b)]

    res = resid(R, t)
    cost0 = cost = sum(r * r for r in res)
    lam, h = 1e-4, 1e-7
    for _ in range(iters):
        cols = []
        for k in range(5):
            d = [0.0] * 5
            d[k] = h
            rp = resid(*step(R, t, d))
            d[k] = -h
            rm = resid(*step(R, t, d))
            cols.append([(p - m) / (2 * h) for p, m in zip(rp, rm)])
        JtJ = [[sum(cols[p][i] * cols[q][i] for i in range(len(res))) for q in range(5)] for p in range(5)]
        Jtr = [-sum(cols[p][i] * res[i] for i in range(len(res))) for p in range(5)]
        improved = False
        for _try in range(6):
            damped = [[JtJ[p][q] + (lam * max(JtJ[p][p], 1e-12) if p == q else 0.0) for q in range(5)] for p in range(5)]
            d = cholesky_solve(damped, Jtr)
            if d is None:
                lam *= 10.0
                continue
            Rn, tn = step(R, t, d)
            resn = resid(Rn, tn)
            costn = sum(r * r for r in resn)
            if costn < cost:
                R, t, res, cost = Rn, tn, resn, costn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
    return R, t, cost0, cost


# ------------------------------------------------------------------------------------------------ depth from two views

def disparity_depth():
    f, B = 500.0, 0.6
    rows = []
    for Z in (2.0, 5.0, 10.0, 20.0, 40.0):
        d = f * B / Z
        rows.append([f"{Z:g}", "%.2f" % d, "%.4f" % (f * B / (d + 0.5)), "%.4f" % (f * B / (d - 0.5)) if d > 0.5 else "behind",
                     "%.4f" % (Z * Z * 0.5 / (f * B))])
    return "\n".join([f"Two cameras with f = {f:g} pixels, parallel axes and centres {B:g} m apart along x. A point at depth Z appears "
                      "in the second image d = f B / Z pixels to the left of where it appears in the first (the disparity). The depth "
                      "read back from a disparity that is wrong by half a pixel, and the first-order error Z^2 (0.5) / (f B):", "",
                      _table(["Z (m)", "disparity d (px)", "depth read from d + 0.5", "depth read from d - 0.5",
                              "first-order error for 0.5 px"], rows, "rrrrr")])


TRI_NAME = "structure_from_motion.triangulation"


def refine_point(Ps, pix, X, iters=3):
    """Gauss-Newton on the sum of squared reprojection errors (pixels) of one point seen by the cameras Ps = [(R, t)]."""
    X = X[:]
    for _ in range(iters):
        H = [[0.0] * 3 for _ in range(3)]
        g = [0.0] * 3
        for (R, t), (u, v) in zip(Ps, pix):
            c = [a + b for a, b in zip(mv(R, X), t)]
            r = [CX + F * c[0] / c[2] - u, CY + F * c[1] / c[2] - v]
            Jd = [[F / c[2], 0.0, -F * c[0] / c[2] ** 2], [0.0, F / c[2], -F * c[1] / c[2] ** 2]]
            J = [[sum(Jd[a][m] * R[m][k] for m in range(3)) for k in range(3)] for a in range(2)]
            for a in range(3):
                g[a] += J[0][a] * r[0] + J[1][a] * r[1]
                for b in range(3):
                    H[a][b] += J[0][a] * J[0][b] + J[1][a] * J[1][b]
        d = mv(cg.inv3(H), g)
        X = [X[k] - d[k] for k in range(3)]
    return X


def triangulation_data():
    """Per baseline: (B, angle in degrees, RMS depth error of the linear method, after Gauss-Newton, first-order prediction)."""
    def make():
        trials, Z0 = 200, 5.0
        g = bookrng.generator(TRI_NAME)
        X0 = [0.0, 0.0, Z0]
        out = []
        for B in (0.05, 0.1, 0.25, 0.6, 1.5):
            Ps = [(IDENT, [0.0, 0.0, 0.0]), (IDENT, [-B, 0.0, 0.0])]
            clean = [project(R, t, X0) for R, t in Ps]
            e_dlt, e_gn = [], []
            for _ in range(trials):
                pix = [(u + SIGMA * g.normal(), v + SIGMA * g.normal()) for (u, v) in clean]
                X = triangulate(Ps, [normalized(p) for p in pix])
                e_dlt.append(X[2] - Z0)
                e_gn.append(refine_point(Ps, pix, X)[2] - Z0)
            rms = lambda e: math.sqrt(sum(x * x for x in e) / len(e))
            out.append((B, math.degrees(2 * math.atan(B / (2 * Z0))), rms(e_dlt), rms(e_gn), math.sqrt(2) * SIGMA * Z0 * Z0 / (F * B)))
        return out
    return _once("triangulation_data", make)


def triangulation_error():
    trials, Z0 = 200, 5.0
    rows = [[f"{B:g}", "%.2f" % ang, "%.4f" % a, "%.4f" % b, "%.4f" % p] for B, ang, a, b, p in triangulation_data()]
    seed = bookrng.seed_of(TRI_NAME)
    return "\n".join([f"A point at (0, 0, {Z0:g}) seen by a camera at the origin and one B metres to the right (same orientation, f = {F:g}); "
                      f"{trials} trials per baseline, each adding Gaussian noise of standard deviation {SIGMA:g} pixel to the four pixel "
                      f"coordinates in the order u and v of camera 0, u and v of camera 1 (`bookrng.generator(\"{TRI_NAME}\")`, seed "
                      f"{seed}, one stream through all five baselines in the order printed, B = 0.05 first). Root mean square error of "
                      "the estimated depth, in metres, for the linear method, for the linear method followed by three Gauss-Newton "
                      "steps on the reprojection error, and the first-order prediction sqrt(2) sigma Z^2 / (f B):", "",
                      _table(["baseline B (m)", "angle between the rays (deg)", "linear", "linear + 3 Gauss-Newton steps",
                              "prediction"], rows, "rrrrr")])


def track_building():
    kp = features()
    rows, pair_rows = [], []
    for label, ratio, mutual in (("ratio 0.8 and mutual", 0.8, True), ("ratio 0.9 and mutual", 0.9, True), ("mutual only", None, True)):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        total = good = 0
        for v0 in range(4):
            for v1 in range(v0 + 1, 4):
                m = _pair_matches(v0, v1, ratio, mutual)
                total += len(m)
                good += sum(1 for i, j in m if kp[v0][i][3] >= 0 and kp[v0][i][3] == kp[v1][j][3])
                for i, j in m:
                    a, b = find((v0, i)), find((v1, j))
                    if a != b:
                        parent[a] = b
        groups = {}
        for node in list(parent):
            groups.setdefault(find(node), []).append(node)
        tracks = [sorted(v) for v in groups.values() if len(v) >= 2]
        clean = [t for t in tracks if len({v for v, _ in t}) == len(t)]
        right = [t for t in clean if len({kp[v][i][3] for v, i in t}) == 1 and kp[t[0][0]][t[0][1]][3] >= 0]
        full = [t for t in right if len(t) == 4]
        rows.append([label, total, good, len(tracks), len(tracks) - len(clean), len(clean), len(right), len(full)])
    return "\n".join(["The matches between each of the six pairs of views under three rules are merged into tracks: the nodes are "
                      "(view, keypoint) pairs, the edges are the matches, a track is a connected component of two or more nodes "
                      "(found by union-find), and a component "
                      "that holds two keypoints of one image is dropped. A track is pure when all its keypoints belong to one "
                      "scene point.", "",
                      _table(["rule", "matches (six pairs)", "correct", "components", "dropped: two keypoints in one image",
                              "tracks kept", "pure", "pure, seen in all four views"], rows, "lrrrrrrr")])


# ------------------------------------------------------------------------------------------------ RANSAC and the two-view pose

CONFIDENCE = 0.999


def ransac_iterations(w, s, cap):
    """Samples of size s that contain at least one all-inlier sample with probability CONFIDENCE when a fraction w of the data are
    inliers: ceil(ln(1 - CONFIDENCE) / ln(1 - w^s)), at least 1 and at most cap; the cap when w^s is too small to register."""
    w = min(1.0, max(0.0, w))
    p = w ** s
    if p >= 1.0 - 1e-12:
        return 1
    if p <= 0.0:
        return cap
    n = math.log(1.0 - CONFIDENCE) / math.log1p(-p)
    if not math.isfinite(n):
        return cap
    return max(1, min(cap, math.ceil(n)))


def iteration_bound_rounding():
    """The bound ln(1 - 0.999) / ln(1 - w^8) computed with ln(1 - p) taken from the rounded number 1 - p, and with log1p(-p)."""
    rows = []
    for w in (0.5, 0.1, 0.03, 0.01, 0.003, 1e-9):
        p = w ** 8
        naive = math.log(1.0 - CONFIDENCE)
        d = math.log(1.0 - p)
        n_naive = "division by zero (ln 1 = 0)" if d == 0.0 else "%.4g" % (naive / d)
        n_good = naive / math.log1p(-p)
        rows.append([f"{w:g}", "%.3g" % p, "1" if 1.0 - p == 1.0 else "%.17g" % (1.0 - p), n_naive, "%.4g" % n_good])
    return "\n".join(["The 99.9% bound for samples of 8 correspondences at very low inlier fractions w, computed two ways: "
                      "ln(0.001) / ln(1 - w^8), where 1 - w^8 is first rounded to a 64-bit number, and ln(0.001) / log1p(-w^8), "
                      "which never forms 1 - w^8.", "",
                      _table(["w", "p = w^8", "1 - p as stored", "bound from ln(1 - p)", "bound from log1p(-p)"], rows, "rrrrr")])


def draw_sample(g, n, k):
    """k distinct indices below n: integer(n) draws, a repeated index is discarded and drawn again."""
    out = []
    while len(out) < k:
        c = g.integer(n)
        if c not in out:
            out.append(c)
    return out


RANSAC_NAME = "structure_from_motion.ransac"
THRESH_PX = 1.5


def _matches01():
    kp = features()
    m = _pair_matches(0, 1, None, False)
    a = [normalized(kp[0][i][:2]) for i, _ in m]
    b = [normalized(kp[1][j][:2]) for _, j in m]
    truth = [kp[0][i][3] >= 0 and kp[0][i][3] == kp[1][j][3] for i, j in m]
    return m, a, b, truth


def _front_count(cand, a, b, idx):
    return sum(1 for i in idx if _front(cand, a[i], b[i]))


def pose_from_essential(E, a, b, idx):
    """The one of the four decompositions of E that puts the most of the correspondences idx in front of both cameras, with the
    four counts; the first camera is (I, 0)."""
    cands = decompose_essential(E)
    counts = [_front_count(c, a, b, idx) for c in cands]
    k = max(range(4), key=lambda i: counts[i])
    return cands, counts, k


def ransac_essential(a, b, thresh, cap, gen, size=8):
    """RANSAC with the eight-point algorithm: draw a sample, fit a rank-two matrix, score every correspondence by truncated squared
    Sampson distance (min(d^2, thresh^2), summed; lower is better), keep the best, and shrink the iteration bound with the inlier
    fraction of the best. The winner's inliers then give the essential matrix by the eight-point algorithm, its pose is chosen
    by the four-fold test and refined by Levenberg-Marquardt on the Sampson distances of those inliers; two rounds, the second on
    the inliers of the refined pose. Returns a dict with the refined pose, the inlier mask, the iterations run, the trace of
    improvements (iteration, inliers, bound), and the linear estimate before refinement."""
    n = len(a)
    t2 = thresh * thresh
    best, need, it, trace = None, cap, 0, []
    while it < need:
        it += 1
        idx = draw_sample(gen, n, size)
        M = eight_point([a[i] for i in idx], [b[i] for i in idx], False)
        d = [sampson_px(M, ai, bi) for ai, bi in zip(a, b)]
        score = sum(min(x * x, t2) for x in d)
        count = sum(1 for x in d if x < thresh)
        if best is None or score < best[1]:
            best = (M, score, count)
            need = min(need, ransac_iterations(count / n, size, cap))
            trace.append((it, count, need))
    inl = [i for i in range(n) if sampson_px(best[0], a[i], b[i]) < thresh]
    E_lin = eight_point([a[i] for i in inl], [b[i] for i in inl])
    cands, counts, k = pose_from_essential(E_lin, a, b, inl)
    R, t = cands[k]
    cost = []
    for _ in range(2):
        R, t, c0, c1 = refine_essential(R, t, [a[i] for i in inl], [b[i] for i in inl])
        cost.append((len(inl), c0, c1))
        E = essential_of(R, t)
        inl = [i for i in range(n) if sampson_px(E, a[i], b[i]) < thresh]
    mask = [i in set(inl) for i in range(n)]
    return dict(R=R, t=t, mask=mask, iters=it, trace=trace, E_lin=E_lin, cands=cands, counts=counts, best=k, cost=cost,
                inl_best=[i for i in range(n) if sampson_px(best[0], a[i], b[i]) < thresh])


def two_view():
    """Everything the pair (0, 1) gives: matches, RANSAC result, pose, points. Cached."""
    def make():
        m, a, b, truth = _matches01()
        gen = bookrng.generator(RANSAC_NAME)
        r = ransac_essential(a, b, THRESH_PX, 5000, gen)
        I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        inl = [i for i, ok in enumerate(r["mask"]) if ok]
        pts = {i: triangulate([(I, [0.0, 0.0, 0.0]), (r["R"], r["t"])], [a[i], b[i]]) for i in inl}
        return dict(m=m, a=a, b=b, truth=truth, pts=pts, inl=inl, **r)
    return _once("two_view", make)


def ransac_bounds():
    rows = []
    for w in (0.9, 0.7, 0.5, 0.3):
        rows.append([f"{w:g}"] + [str(ransac_iterations(w, s, 10 ** 9)) for s in (3, 5, 6, 8)])
    return rows


def two_view_ransac():
    tv = two_view()
    m, a, b, truth = tv["m"], tv["a"], tv["b"], tv["truth"]
    R0, t0 = true_pair(0, 1)
    seed = bookrng.seed_of(RANSAC_NAME)
    n = len(m)
    allidx = list(range(n))
    Eall = eight_point(a, b)
    cands, counts, k = pose_from_essential(Eall, a, b, allidx)
    Rall, tall = cands[k]
    inl = tv["inl"]
    kept_true = sum(1 for i in inl if truth[i])
    nb = len(tv["inl_best"])
    gidx = [i for i in inl if truth[i]]
    Eg = eight_point([a[i] for i in gidx], [b[i] for i in gidx])
    cg_, _, kg = pose_from_essential(Eg, a, b, gidx)
    Rg, tg, _, _ = refine_essential(cg_[kg][0], cg_[kg][1], [a[i] for i in gidx], [b[i] for i in gidx])
    trace_rows = [[it, cnt, bound] for it, cnt, bound in tv["trace"]]
    Elin = tv["cands"][tv["best"]]
    out = [f"The {n} matches of the nearest-neighbour-only rule between views 0 and 1 (the first row of the matching table) are the "
           f"input; {sum(truth)} are correct. The sample is 8 distinct matches, drawn with `bookrng.generator(\"{RANSAC_NAME}\")` "
           f"(seed {seed}) as `integer({n})` draws, a repeated index discarded and drawn again; a hypothesis is the rank-two "
           f"matrix of the eight-point algorithm on the sample. The inlier threshold is {THRESH_PX:g} pixel of Sampson distance, the "
           f"score is the sum over matches of min(d^2, {THRESH_PX:g}^2), and the iteration bound is the 99.9% formula of the "
           "chapter with the inlier fraction of the best hypothesis so far (cap 5000).", "",
           "Each time a sample beats the best score so far, the bound is recomputed:", "",
           _table(["iteration", "inliers of the best hypothesis", "bound on the iterations still needed"], trace_rows), "",
           f"The loop ended after {tv['iters']} iterations with {nb} inliers. Their eight-point essential matrix, its pose chosen by "
           "the four-fold test and refined by Levenberg-Marquardt on the Sampson distances of the inliers, twice (the second "
           f"time on the inliers of the first result), leaves {len(inl)} inliers, of which {kept_true} are correct matches and "
           f"{len(inl) - kept_true} are wrong matches that lie within {THRESH_PX:g} pixel of an epipolar line; "
           f"{sum(truth) - kept_true} correct matches were rejected.", "",
           _table(["round", "inliers", "sum of squared Sampson distances before (px^2)", "after (px^2)"],
                  [[i + 1, c[0], "%.2f" % c[1], "%.2f" % c[2]] for i, c in enumerate(tv["cost"])], "rrrr"), "",
           _table(["estimate", "rotation error (degrees)", "translation direction error (degrees)"],
                  [["eight-point on all %d matches" % n, "%.2f" % rot_error(Rall, R0), "%.2f" % dir_error(tall, t0)],
                   [f"eight-point on the {nb} RANSAC inliers", "%.2f" % rot_error(Elin[0], R0), "%.2f" % dir_error(Elin[1], t0)],
                   ["after refinement (the 54 inliers)", "%.2f" % rot_error(tv["R"], R0), "%.2f" % dir_error(tv["t"], t0)],
                   [f"the same procedure on the {kept_true} correct inliers only (needs the truth)", "%.2f" % rot_error(Rg, R0),
                    "%.2f" % dir_error(tg, t0)]], "lrr"), "",
           "Samples needed for 99.9% confidence of drawing at least one all-inlier sample, ceil(ln(0.001) / ln(1 - w^s)) "
           "for inlier fraction w and sample size s:", "",
           _table(["w", "s = 3", "s = 5", "s = 6", "s = 8"], ransac_bounds())]
    return "\n".join(out)


def _front(cand, a, b):
    R, t = cand
    I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    X = triangulate([(I, [0.0, 0.0, 0.0]), (R, t)], [a, b])
    return X[2] > 0 and depth_in(R, t, X) > 0


def pose_ambiguity():
    tv = two_view()
    R0, t0 = true_pair(0, 1)
    rows = []
    for i, ((R, t), c) in enumerate(zip(tv["cands"], tv["counts"])):
        name = ("U W V^T" if i < 2 else "U W^T V^T") + (", +u3" if i % 2 == 0 else ", -u3")
        rows.append([name, "%.2f" % rot_error(R, R0), "%.2f" % dir_error(t, t0), c, len(tv["inl_best"]) - c,
                     "chosen" if i == tv["best"] else ""])
    return "\n".join([f"The {len(tv['inl_best'])} RANSAC inliers triangulated under each of the four decompositions of their "
                      "eight-point essential matrix: how many land in front of both cameras, and how far each candidate is from the "
                      "true rotation and translation direction (degrees).", "",
                      _table(["candidate (R, t)", "rotation error", "direction error", "in front of both", "behind one or both",
                              ""], rows, "lrrrrl")])


MINIMAL_NAME = "structure_from_motion.minimal"


def minimal_samples():
    """How good the hypotheses of a minimal 8-sample are, from 100 draws among the 54 correct matches of the pair: the median and
    largest number of the 54 within 1.5 px of the hypothesis, for the nearest essential matrix and for the nearest rank-two matrix."""
    tv = two_view()
    good = [i for i in range(len(tv["m"])) if tv["truth"][i]]
    a, b = [tv["a"][i] for i in good], [tv["b"][i] for i in good]
    g = bookrng.generator(MINIMAL_NAME)
    ess, rk2 = [], []
    for _ in range(100):
        idx = draw_sample(g, len(good), 8)
        for essential, bucket in ((True, ess), (False, rk2)):
            M = eight_point([a[i] for i in idx], [b[i] for i in idx], essential)
            bucket.append(sum(1 for x, y in zip(a, b) if sampson_px(M, x, y) < THRESH_PX))
    med = lambda v: sorted(v)[len(v) // 2 - 1] / 2.0 + sorted(v)[len(v) // 2] / 2.0
    rows = [["nearest essential matrix (singular values 1, 1, 0)", "%.1f" % med(ess), max(ess)],
            ["nearest rank-two matrix", "%.1f" % med(rk2), max(rk2)]]
    return "\n".join([f"The {len(good)} correct matches of the pair (0, 1) with their noisy pixels, and 100 samples of 8 of them from "
                      f"`bookrng.generator(\"{MINIMAL_NAME}\")` (seed {bookrng.seed_of(MINIMAL_NAME)}; the same `integer` draws "
                      "with repeats discarded). Each sample gives two hypotheses by the eight-point algorithm; the table counts, for "
                      f"each kind, the matches within {THRESH_PX:g} pixel of Sampson distance of it (all {len(good)} are correct, so "
                      "a perfect hypothesis scores 54).", "",
                      _table(["hypothesis", "median inliers", "largest"], rows, "lrr")])


# ------------------------------------------------------------------------------------------------ registering a new camera

PNP_NAME = "structure_from_motion.registration"
PNP_THRESH_PX = 4.0
IDENT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def hartley_3d(P):
    """Similarity of 3D points: centroid to the origin, mean distance from it to sqrt(3); returns the 4 by 4 matrix."""
    n = len(P)
    m = [sum(p[k] for p in P) / n for k in range(3)]
    sc = math.sqrt(3.0) / (sum(norm(sub(p, m)) for p in P) / n)
    return [[sc, 0.0, 0.0, -sc * m[0]], [0.0, sc, 0.0, -sc * m[1]], [0.0, 0.0, sc, -sc * m[2]], [0.0, 0.0, 0.0, 1.0]]


def dlt_pose_rows(X, x):
    """The two rows of the linear absolute-pose system for the point X and the normalized image point x = (x, y, 1): with the camera
    matrix P of rows p1, p2, p3 flattened row-major, x p3.X - p1.X = 0 and y p3.X - p2.X = 0."""
    Xh = list(X) + [1.0]
    z = [0.0] * 4
    return [Xh + z + [-x[0] * c for c in Xh], z + Xh + [-x[1] * c for c in Xh]]


def dlt_pose(Xs, xs):
    """Camera pose (R, t) from n >= 6 points Xs and their normalized images xs: the null vector of the 2n by 12 matrix (in coordinates
    normalized by Hartley's similarity for the 3D points), undone; the left 3 by 3 block is made a rotation by the nearest-rotation
    step of chapter 3 and the translation is scaled by the same factor (the mean of its singular values)."""
    T = hartley_3d(Xs)
    Xn = [mv(T, list(X) + [1.0])[:3] for X in Xs]
    rows = []
    for X, x in zip(Xn, xs):
        rows += dlt_pose_rows(X, x)
    p, _ = cg.null_vector(rows, 12)
    P = mm([p[0:4], p[4:8], p[8:12]], T)                  # 3 by 4 camera matrix of the original coordinates
    A = [r[:3] for r in P]
    if det3(A) < 0:
        P = [[-v for v in r] for r in P]
        A = [r[:3] for r in P]
    U, sg, V = dc._svd(A)
    s = (sg[0] + sg[1] + sg[2]) / 3.0
    R = cg.nearest_rotation(A)
    t = [P[i][3] / s for i in range(3)]
    return R, t


def reproj_px(R, t, X, x_obs_px):
    """Distance in pixels between the projection of X and the observed pixel; infinity if X is not in front of the camera."""
    c = [a + b for a, b in zip(mv(R, X), t)]
    if c[2] <= 1e-9:
        return float("inf")
    return math.hypot(CX + F * c[0] / c[2] - x_obs_px[0], CY + F * c[1] / c[2] - x_obs_px[1])


def ransac_pnp(Xs, xs, pix_obs, thresh, cap, gen):
    """RANSAC with the 6-point DLT: sample 6 distinct correspondences, fit (R, t), score by truncated squared reprojection error,
    adaptive bound as for the essential matrix; returns (R, t, inlier mask, iterations, trace)."""
    n = len(Xs)
    t2 = thresh * thresh
    best, need, it, trace = None, cap, 0, []
    while it < need:
        it += 1
        idx = draw_sample(gen, n, 6)
        R, t = dlt_pose([Xs[i] for i in idx], [xs[i] for i in idx])
        d = [reproj_px(R, t, Xs[i], pix_obs[i]) for i in range(n)]
        score = sum(min(x * x, t2) for x in d)
        count = sum(1 for x in d if x < thresh)
        if best is None or score < best[2]:
            best = (R, t, score, count)
            need = min(need, ransac_iterations(count / n, 6, cap))
            trace.append((it, count, need))
    R, t = best[0], best[1]
    return R, t, it, trace


def register_camera(view):
    """Registration of camera `view` against the 3D points of the two-view reconstruction. Correspondences: for each inlier match of
    the pair (0, 1) the nearest neighbour (no ratio test) of its view-0 descriptor among the keypoints of `view`. Returns a dict."""
    def make():
        tv = two_view()
        kp = features()
        A = [kp[0][ia][2] for ia, _ in (tv["m"][i] for i in tv["inl"])]
        B = [k[2] for k in kp[view]]
        nn = best_two(A, B)
        corr = []
        for (i, (j, _, _)) in zip(tv["inl"], nn):
            ia = tv["m"][i][0]
            corr.append(dict(match=i, kp0=ia, kpv=j, X=tv["pts"][i], x=normalized(kp[view][j][:2]), px=kp[view][j][:2],
                             ok=(kp[0][ia][3] >= 0 and kp[0][ia][3] == kp[view][j][3] and tv["truth"][i])))
        gen = bookrng.generator(PNP_NAME + "." + str(view))
        Xs, xs, pxs = [c["X"] for c in corr], [c["x"] for c in corr], [c["px"] for c in corr]
        R, t, iters, trace = ransac_pnp(Xs, xs, pxs, PNP_THRESH_PX, 2000, gen)
        mask = [reproj_px(R, t, X, p) < PNP_THRESH_PX for X, p in zip(Xs, pxs)]
        return dict(corr=corr, R=R, t=t, iters=iters, trace=trace, mask=mask)
    return _once("register%d" % view, make)


def planar_degeneracy():
    """Smallest singular values of the 2n by 12 absolute-pose system for 8 points: in general position, and on one plane."""
    pts, poses, pix0, pix = scene()
    R, t = poses[2]
    out = []
    for name, P in (("8 points of the scene", [pts[j] for j in range(8)]),
                    ("the same 8 pixels' rays cut by the plane Z = 5.5", None)):
        if P is None:
            P = []
            for j in range(8):
                x = normalized(pix0[0][j])
                P.append([5.5 * x[0], 5.5 * x[1], 5.5])
        xs = [normalized(project(R, t, X)) for X in P]
        T = hartley_3d(P)
        Xn = [mv(T, list(X) + [1.0])[:3] for X in P]
        rows = []
        for X, x in zip(Xn, xs):
            rows += dlt_pose_rows(X, x)
        _, sg = cg.null_vector(rows, 12)
        out.append([name] + ["%.1e" % sg[k] for k in (-1, -2, -3)])
    return out


def absolute_pose():
    reg = {v: register_camera(v) for v in (2, 3)}
    R2, t2 = reg[2]["R"], reg[2]["t"]
    rows = []
    s_true = norm(true_pair(0, 1)[1])
    for v in (2, 3):
        r = reg[v]
        n = len(r["corr"])
        ok = sum(1 for c in r["corr"] if c["ok"])
        inl = sum(r["mask"])
        inl_ok = sum(1 for c, m in zip(r["corr"], r["mask"]) if m and c["ok"])
        Rt, tt = true_pair(0, v)
        rows.append([v, n, ok, ransac_iterations(ok / n, 6, 10 ** 6), r["iters"], inl, inl_ok,
                     "%.2f" % rot_error(r["R"], Rt), "%.2f" % dir_error(r["t"], tt)])
    seed = bookrng.seed_of(PNP_NAME + ".2")
    out = ["For camera v = 2 and 3, each of the inlier matches of the pair (0, 1) gives a 3D point (triangulated with the refined "
           "pose, baseline 1) and a 2D keypoint: the nearest neighbour, among the keypoints of view v, of the match's view-0 "
           "descriptor. The RANSAC sample is 6 distinct correspondences from `integer(n)` draws on `bookrng.generator(\"" + PNP_NAME +
           ".v\")` (v = 2: seed " + str(seed) + "), a hypothesis is the DLT pose of the sample, and the threshold is "
           f"{PNP_THRESH_PX:g} pixels of reprojection error with the same truncated score and adaptive bound as before (cap 2000).",
           "", _table(["camera", "correspondences", "correct", "bound at the true fraction", "iterations run", "inliers",
                       "correct inliers", "rotation error (degrees)", "direction error (degrees)"], rows, "rrrrrrrrr"), "",
           "Smallest singular values of the 16 by 12 matrix of the linear absolute-pose system, 8 points, last three:", "",
           _table(["the 8 points are", "sigma_12", "sigma_11", "sigma_10"], planar_degeneracy(), "lrrr")]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ bundle adjustment

def inv3_spd(V):
    """Inverse of a symmetric positive definite 3 by 3 matrix by cofactors; None if its determinant is not positive."""
    d = det3(V)
    if not d > 0.0:
        return None
    return cg.inv3(V)


def huber_weight(r, delta):
    n = math.hypot(r[0], r[1])
    return 1.0 if delta is None or n <= delta else delta / n


def huber_cost(r, delta):
    n2 = r[0] * r[0] + r[1] * r[1]
    if delta is None or n2 <= delta * delta:
        return n2
    return 2.0 * delta * math.sqrt(n2) - delta * delta


class Problem:
    """A bundle adjustment problem: world-to-camera poses [R, t], points, one shared focal length f (the principal point is fixed at
    (CX, CY)) and observations (camera, point, u, v) in pixels. `slots[i]` is 'held' (the anchor: not a parameter), 'free' (rotation
    increment and translation: 6 columns) or 'scale' (rotation increment and the camera centre moving across the baseline from the
    anchor: 5 columns)."""

    def __init__(self, poses, points, obs, slots, anchor=None, scale=None, f=F, free_f=False, free_points=True, delta=2.0):
        self.poses = [[[r[:] for r in R], t[:]] for R, t in poses]
        self.points = [p[:] for p in points]
        self.obs, self.slots = obs, slots
        self.anchor, self.scale, self.f, self.free_f = anchor, scale, f, free_f
        self.free_points, self.delta = free_points, delta
        self.d0 = None
        if anchor is not None and scale is not None:
            self.d0 = norm(sub(self.cam_centre(scale), self.cam_centre(anchor)))
        self.col0, c = [], 0
        for s in slots:
            self.col0.append(c)
            c += {"held": 0, "free": 6, "scale": 5}[s]
        self.ncam = c                                   # camera columns
        self.fcol = c if free_f else None
        self.nc = c + (1 if free_f else 0)              # columns shared across points (cameras and focal length)
        self.by_point = [[] for _ in points]
        for k, (ci, pj, u, v) in enumerate(obs):
            self.by_point[pj].append(k)

    def cam_centre(self, i):
        R, t = self.poses[i]
        return centre(R, t)

    def project(self, i, X):
        R, t = self.poses[i]
        c = [a + b for a, b in zip(mv(R, X), t)]
        return c, (CX + self.f * c[0] / c[2], CY + self.f * c[1] / c[2])

    def residuals(self):
        out = []
        for ci, pj, u, v in self.obs:
            c, (pu, pv) = self.project(ci, self.points[pj])
            out.append([pu - u, pv - v] if c[2] > 1e-9 else [1e6, 1e6])
        return out

    def cost(self):
        return sum(huber_cost(r, self.delta) for r in self.residuals())

    def rms(self):
        rs = self.residuals()
        return math.sqrt(sum(r[0] ** 2 + r[1] ** 2 for r in rs) / len(rs))

    def snapshot(self):
        return ([[[r[:] for r in R], t[:]] for R, t in self.poses], [p[:] for p in self.points], self.f)

    def restore(self, snap):
        self.poses, self.points, self.f = ([[[r[:] for r in R], t[:]] for R, t in snap[0]], [p[:] for p in snap[1]], snap[2])

    def across(self):
        """Two unit vectors across the baseline from the anchor's centre to the scale camera's centre."""
        b = unit(sub(self.cam_centre(self.scale), self.cam_centre(self.anchor)))
        return tangent_basis(b)

    def linearize(self):
        """Per observation: residual r, weight w, the columns it touches, the Jacobian of r in them (2 rows) and in its point (2 by 3)."""
        lin = []
        basis = self.across() if self.scale is not None else None
        for ci, pj, u, v in self.obs:
            R, t = self.poses[ci]
            X = self.points[pj]
            c, (pu, pv) = self.project(ci, X)
            r = [pu - u, pv - v]
            f, z = self.f, c[2]
            Jd = [[f / z, 0.0, -f * c[0] / (z * z)], [0.0, f / z, -f * c[1] / (z * z)]]
            cols, rows = [], [[], []]
            slot = self.slots[ci]
            if slot != "held":
                q = c if slot == "scale" else [c[k] - t[k] for k in range(3)]     # rotation about the centre / about the origin
                for k in range(3):
                    e = [1.0 if m == k else 0.0 for m in range(3)]
                    d = cross(e, q)
                    cols.append(self.col0[ci] + k)
                    for a in range(2):
                        rows[a].append(sum(Jd[a][m] * d[m] for m in range(3)))
                if slot == "free":
                    for k in range(3):
                        cols.append(self.col0[ci] + 3 + k)
                        for a in range(2):
                            rows[a].append(Jd[a][k])
                else:
                    for k, b in enumerate(basis):
                        d = [-x for x in mv(R, b)]                                # dc/dC = -R
                        cols.append(self.col0[ci] + 3 + k)
                        for a in range(2):
                            rows[a].append(sum(Jd[a][m] * d[m] for m in range(3)))
            if self.free_f:
                cols.append(self.fcol)
                rows[0].append(c[0] / z)
                rows[1].append(c[1] / z)
            Jp = [[sum(Jd[a][m] * R[m][k] for m in range(3)) for k in range(3)] for a in range(2)] if self.free_points else None
            lin.append(dict(r=r, w=huber_weight(r, self.delta), cols=cols, J=rows, Jp=Jp))
        return lin

    def normal_equations(self, lin):
        """U (nc by nc), gc, the per-point blocks V_j (3 by 3) and g_j, and the coupling W of each observation: w J_c^T J_p."""
        nc = self.nc
        U = [[0.0] * nc for _ in range(nc)]
        gc = [0.0] * nc
        V = [[[0.0] * 3 for _ in range(3)] for _ in self.points]
        gp = [[0.0] * 3 for _ in self.points]
        for o, l in zip(self.obs, lin):
            w, r, cols, J, Jp = l["w"], l["r"], l["cols"], l["J"], l["Jp"]
            for a, ca in enumerate(cols):
                gc[ca] += w * (J[0][a] * r[0] + J[1][a] * r[1])
                for b, cb in enumerate(cols):
                    U[ca][cb] += w * (J[0][a] * J[0][b] + J[1][a] * J[1][b])
            if Jp is not None:
                j = o[1]
                for a in range(3):
                    gp[j][a] += w * (Jp[0][a] * r[0] + Jp[1][a] * r[1])
                    for b in range(3):
                        V[j][a][b] += w * (Jp[0][a] * Jp[0][b] + Jp[1][a] * Jp[1][b])
                l["W"] = [[w * (J[0][a] * Jp[0][k] + J[1][a] * Jp[1][k]) for k in range(3)] for a in range(len(cols))]
        return U, gc, V, gp

    def step_schur(self, lin, parts, lam):
        """The damped step by the Schur complement: eliminate the points (3 by 3 blocks), solve the camera system of nc unknowns,
        back-substitute. Returns (dc, dp) or None when a block or the camera system is not positive definite."""
        U, gc, V, gp = parts
        nc = self.nc
        S = [row[:] for row in U]
        for a in range(nc):
            S[a][a] += lam * max(U[a][a], 1e-9)
        rhs = [-g for g in gc]
        Vinv = []
        if self.free_points:
            for j, Vj in enumerate(V):
                Vd = [row[:] for row in Vj]
                for a in range(3):
                    Vd[a][a] += lam * max(Vj[a][a], 1e-9)
                Vi = inv3_spd(Vd)
                if Vi is None:
                    return None
                Vinv.append(Vi)
            for j, Vi in enumerate(Vinv):
                ks = self.by_point[j]
                vg = mv(Vi, gp[j])
                for ko in ks:
                    l = lin[ko]
                    cols, W = l["cols"], l["W"]
                    for a, ca in enumerate(cols):
                        rhs[ca] += sum(W[a][k] * vg[k] for k in range(3))
                        T = mv(Vi, W[a])                                    # V^-1 W_a^T, one column of the product
                        for kp in ks:
                            l2 = lin[kp]
                            for b, cb in enumerate(l2["cols"]):
                                S[cb][ca] -= sum(l2["W"][b][k] * T[k] for k in range(3))
        dc = cholesky_solve(S, rhs) if nc else []
        if dc is None:
            return None
        dp = []
        if self.free_points:
            for j, Vi in enumerate(Vinv):
                b = [-gp[j][k] for k in range(3)]
                for ko in self.by_point[j]:
                    l = lin[ko]
                    for a, ca in enumerate(l["cols"]):
                        for k in range(3):
                            b[k] -= l["W"][a][k] * dc[ca]
                dp.append(mv(Vi, b))
        return dc, dp

    def step_dense(self, lin, parts, lam):
        """The same damped step from the full normal equations of nc + 3 P unknowns, by one dense Cholesky solve."""
        U, gc, V, gp = parts
        nc, P = self.nc, len(self.points)
        n = nc + 3 * P
        H = [[0.0] * n for _ in range(n)]
        g = [0.0] * n
        for a in range(nc):
            for b in range(nc):
                H[a][b] = U[a][b]
            g[a] = gc[a]
            H[a][a] += lam * max(U[a][a], 1e-9)
        for j in range(P):
            for a in range(3):
                g[nc + 3 * j + a] = gp[j][a]
                for b in range(3):
                    H[nc + 3 * j + a][nc + 3 * j + b] = V[j][a][b]
                H[nc + 3 * j + a][nc + 3 * j + a] += lam * max(V[j][a][a], 1e-9)
        for ko in range(len(self.obs)):
            l = lin[ko]
            j = self.obs[ko][1]
            for a, ca in enumerate(l["cols"]):
                for k in range(3):
                    H[ca][nc + 3 * j + k] += l["W"][a][k]
                    H[nc + 3 * j + k][ca] += l["W"][a][k]
        x = cholesky_solve(H, [-v for v in g])
        if x is None:
            return None
        return x[:nc], [x[nc + 3 * j: nc + 3 * j + 3] for j in range(P)]

    def apply(self, dc, dp):
        """Update the state by a step: poses by their increments, points, focal length, then restore the exact baseline."""
        basis = self.across() if self.scale is not None else None
        for i, slot in enumerate(self.slots):
            if slot == "held":
                continue
            c0 = self.col0[i]
            R, t = self.poses[i]
            Rn = mm(cg.rodrigues(dc[c0:c0 + 3]), R)
            if slot == "free":
                self.poses[i] = [Rn, [t[k] + dc[c0 + 3 + k] for k in range(3)]]
            else:
                C = [self.cam_centre(i)[k] + dc[c0 + 3] * basis[0][k] + dc[c0 + 4] * basis[1][k] for k in range(3)]
                self.poses[i] = [Rn, [-x for x in mv(Rn, C)]]
        if self.free_f:
            self.f += dc[self.fcol]
        if self.free_points:
            for j, d in enumerate(dp):
                self.points[j] = [self.points[j][k] + d[k] for k in range(3)]
        if self.d0 is not None:
            self.hold_scale()

    def hold_scale(self):
        """Rescale everything but the anchor about the anchor's centre so that the scale camera is exactly d0 from it (a similarity,
        which changes no reprojection)."""
        Ca = self.cam_centre(self.anchor)
        s = self.d0 / norm(sub(self.cam_centre(self.scale), Ca))
        for i in range(len(self.poses)):
            if i != self.anchor:
                C = [Ca[k] + s * (self.cam_centre(i)[k] - Ca[k]) for k in range(3)]
                R = self.poses[i][0]
                self.poses[i][1] = [-x for x in mv(R, C)]
        self.points = [[Ca[k] + s * (X[k] - Ca[k]) for k in range(3)] for X in self.points]

    def solve(self, iters=50, lam=1e-3, dense=False):
        """Levenberg-Marquardt: damp the diagonal by lam (times the diagonal entry), try the step, accept it if the robust cost falls
        (lam / 3) and otherwise try again with lam * 5, at most 8 tries per iteration; stop after `iters` iterations, when no try
        succeeds or when the gain is below 1e-10 of the cost. Returns the history [(iteration, tries, lam, cost, rms)]."""
        cost = self.cost()
        hist = [(0, 0, lam, cost, self.rms())]
        for it in range(1, iters + 1):
            lin = self.linearize()
            parts = self.normal_equations(lin)
            accepted, tries = False, 0
            for _ in range(8):
                tries += 1
                st = (self.step_dense if dense else self.step_schur)(lin, parts, lam)
                if st is None:
                    lam *= 5.0
                    continue
                snap = self.snapshot()
                self.apply(*st)
                new = self.cost()
                if new < cost:
                    gain = cost - new
                    cost = new
                    lam = max(lam / 3.0, 1e-9)
                    accepted = True
                    break
                self.restore(snap)
                lam *= 5.0
            if not accepted:
                break
            hist.append((it, tries, lam, cost, self.rms()))
            if gain < 1e-10 * max(cost, 1e-300):
                break
        return hist


def refine_registration(view):
    """Gauss-Newton/LM refinement of a registered camera's pose against the fixed 3D points of its inliers (a bundle adjustment with
    one free camera and no free points); returns (R, t, history)."""
    reg = register_camera(view)
    inl = [c for c, m in zip(reg["corr"], reg["mask"]) if m]
    obs = [(0, k, c["px"][0], c["px"][1]) for k, c in enumerate(inl)]
    P = Problem([[reg["R"], reg["t"]]], [c["X"] for c in inl], obs, ["free"], free_points=False, delta=2.0)
    hist = P.solve(30)
    return P.poses[0][0], P.poses[0][1], hist


def initial_problem(delta=2.0, free_f=False, f0=F):
    """The reconstruction as the incremental pipeline leaves it before the global adjustment: camera 0 the anchor at the identity,
    camera 1 from the refined two-view pose (scale camera, baseline 1), cameras 2 and 3 from the refined registration; the
    observations of a point are both keypoints of its inlier match of the pair and the keypoints that were RANSAC inliers of the
    registrations; the point is triangulated from all of them by the linear method, and an observation that lands behind its camera
    is dropped (a point left with one observation is dropped). Returns (problem, truth flags of the observations, match indices)."""
    tv = two_view()
    kp = features()
    pose2 = refine_registration(2)[:2]
    pose3 = refine_registration(3)[:2]
    poses = [[IDENT, [0.0, 0.0, 0.0]], [tv["R"], unit(tv["t"])], list(pose2), list(pose3)]
    seen = {}
    for pj, i in enumerate(tv["inl"]):
        ia, ib = tv["m"][i]
        seen[i] = [(0, kp[0][ia][0], kp[0][ia][1], tv["truth"][i]), (1, kp[1][ib][0], kp[1][ib][1], tv["truth"][i])]
    for v in (2, 3):
        reg = register_camera(v)
        for c, ok in zip(reg["corr"], reg["mask"]):
            if ok and c["match"] in seen:
                seen[c["match"]].append((v, c["px"][0], c["px"][1], c["ok"]))
    points, obs, truth, kept = [], [], [], []
    for i, lst in seen.items():
        X = triangulate([tuple(poses[v]) for v, _, _, _ in lst], [normalized((u, w)) for _, u, w, _ in lst])
        good = [o for o in lst if depth_in(poses[o[0]][0], poses[o[0]][1], X) > 0]
        if len(good) < 2:
            continue
        pj = len(points)
        points.append(X)
        kept.append(i)
        for v, u, w, ok in good:
            obs.append((v, pj, u, w))
            truth.append(ok)
    P = Problem(poses, points, obs, ["held", "scale", "free", "free"], anchor=0, scale=1, f=f0, free_f=free_f, delta=delta)
    return P, truth, kept


def truth_errors(P):
    """Rotation errors (degrees) and camera-centre errors (metres) of the cameras against the truth, after multiplying the estimate's
    lengths by the true baseline of cameras 0 and 1 (both frames have camera 0 as the world, so the scale is all that differs)."""
    pts, poses, _, _ = scene()
    s = norm(centre(*poses[1]))
    out = []
    for i in range(1, len(P.poses)):
        Rt, tt = poses[i]
        Ce = [s * x for x in P.cam_centre(i)]
        out.append((rot_error(P.poses[i][0], Rt), norm(sub(Ce, centre(Rt, tt)))))
    return out


def incremental_to_ba():
    P, truth, inl = initial_problem()
    pre = truth_errors(P)
    rms0 = P.rms()
    hist = P.solve(50)
    post = truth_errors(P)
    return P, hist, pre, post, truth


def drop_unexplained(P, max_px):
    """Remove the observations whose residual is above max_px and the points left with fewer than two (brain's rule after each
    adjustment); returns a new problem with the same state and the number of observations dropped."""
    rs = P.residuals()
    keep = [k for k, r in enumerate(rs) if math.hypot(r[0], r[1]) <= max_px]
    cnt = {}
    for k in keep:
        cnt[P.obs[k][1]] = cnt.get(P.obs[k][1], 0) + 1
    pts = [j for j in range(len(P.points)) if cnt.get(j, 0) >= 2]
    new = {j: n for n, j in enumerate(pts)}
    obs = [(P.obs[k][0], new[P.obs[k][1]], P.obs[k][2], P.obs[k][3]) for k in keep if P.obs[k][1] in new]
    Q = Problem(P.poses, [P.points[j] for j in pts], obs, P.slots, P.anchor, P.scale, P.f, P.free_f, P.free_points, P.delta)
    return Q, len(P.obs) - len(obs), pts


def _truth_start(P_final):
    """The true poses and points in the gauge of the problem (camera 0 at the identity, baseline 1), for the observations of P_final."""
    pts, poses, _, _ = scene()
    s = norm(centre(*poses[1]))
    return [[R, [x / s for x in t]] for R, t in poses], [[x / s for x in X] for X in pts]


def bundle_adjustment():
    P, truth, kept = initial_problem()
    pre = truth_errors(P)
    rms0, cost0 = P.rms(), P.cost()
    start_pts = [X[:] for X in P.points]
    # the first damped step, by the Schur complement and by the dense system
    lin = P.linearize()
    parts = P.normal_equations(lin)
    sd = P.step_dense(lin, parts, 1e-3)
    ss = P.step_schur(lin, parts, 1e-3)
    gap = max(max(abs(a - b) for a, b in zip(sd[0], ss[0])),
              max(abs(a - b) for x, y in zip(sd[1], ss[1]) for a, b in zip(x, y)))
    scale_step = max(abs(v) for v in ss[0])
    hist = P.solve(60)
    post = truth_errors(P)
    n_unexpl = None
    rows = []
    shown = list(range(0, 8)) + [len(hist) - 1]
    for k in shown:
        it, tries, lam, cost, rms = hist[k]
        rows.append([it, tries if it else "", "%.2e" % lam if it else "%.0e" % lam, "%.4f" % cost, "%.4f" % rms])
    # the same problem started at the truth
    poses_t, pts_t = _truth_start(P)
    tv, kp = two_view(), features()
    start = [pts_t[kp[0][tv["m"][i][0]][3]] if tv["truth"][i] else start_pts[j] for j, i in enumerate(kept)]
    Q = Problem([[R, t] for R, t in poses_t], start, P.obs, P.slots, P.anchor, P.scale, P.f, False, True, P.delta)
    histq = Q.solve(60)
    dpose = max(abs(a - b) for i in range(1, 4) for a, b in zip([x for r in P.poses[i][0] for x in r] + P.poses[i][1],
                                                                [x for r in Q.poses[i][0] for x in r] + Q.poses[i][1]))
    n = len(P.obs)
    npar = P.nc + 3 * len(P.points)
    floor = SIGMA * math.sqrt(2.0) * math.sqrt(max(0.0, (2 * n - npar) / (2 * n)))
    out = [f"The problem after registration: {len(P.points)} points and {n} observations in 4 cameras, of which {n - sum(truth)} are "
           f"wrong associations; camera 0 is the anchor (held), camera 1 the scale camera (5 columns, baseline 1), cameras 2 and 3 "
           f"free (6 columns each), so the camera system has {P.nc} unknowns and the points {3 * len(P.points)}. Huber threshold "
           "delta = 2 pixels, lambda starts at 1e-3; an iteration tries the step, accepts it when the robust cost falls "
           "(lambda divided by 3) and otherwise multiplies lambda by 5.", "",
           f"Before the adjustment the root mean square reprojection error is {rms0:.4f} px and the robust cost {cost0:.4f}. The first "
           f"damped step computed from the dense system of {P.nc + 3 * len(P.points)} unknowns and from the Schur complement of the "
           f"{P.nc} camera unknowns differ by at most {gap:.1e} in any entry (the largest camera step is {scale_step:.3f}).", "",
           _table(["iteration", "tries", "lambda after", "robust cost (px^2)", "RMS reprojection (px)"], rows, "rrrrr"),
           f"(iterations 8 to {len(hist) - 2} omitted; the loop stops at iteration {len(hist) - 1}, when the gain is below 1e-10 of "
           "the cost.)", "",
           f"The noise floor: with sigma = {SIGMA:g} pixel per coordinate, {n} observations and {npar} parameters the expected "
           f"RMS error of the fit is sigma sqrt(2) sqrt((2 {n} - {npar}) / (2 {n})) = {floor:.4f} px; the adjustment reaches "
           f"{hist[-1][4]:.4f}.", "",
           _table(["camera", "rotation error before (deg)", "after", "centre error before (m, scaled to the true baseline)", "after"],
                  [[i + 1, "%.2f" % pre[i][0], "%.2f" % post[i][0], "%.3f" % pre[i][1], "%.3f" % post[i][1]] for i in range(3)],
                  "rrrrr"), "",
           f"Started at the true poses and points instead, with the same observations, the same loop reaches the cost "
           f"{histq[-1][3]:.4f} against {hist[-1][3]:.4f}, and the largest difference between any pose entry of the two solutions "
           f"is {dpose:.1e}."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ structure of the problem and its gauge

def dense_jacobian(P):
    """The residual Jacobian of the problem as a dense matrix (2 rows per observation) over its nc camera columns and 3 columns per
    point; unweighted."""
    lin = P.linearize()
    n = P.nc + 3 * len(P.points)
    rows = []
    for o, l in zip(P.obs, lin):
        for a in range(2):
            row = [0.0] * n
            for k, c in enumerate(l["cols"]):
                row[c] = l["J"][a][k]
            if l["Jp"] is not None:
                for k in range(3):
                    row[P.nc + 3 * o[1] + k] = l["Jp"][a][k]
            rows.append(row)
    return rows


def _small_true_problem(slots, anchor=None, scale=None, npts=10):
    pts, poses, pix0, pix = scene()
    obs = [(v, j, pix0[v][j][0], pix0[v][j][1]) for j in range(npts) for v in range(4)]
    return Problem([[R, t] for R, t in poses], pts[:npts], obs, slots, anchor, scale, F, False, True, None)


def singular_values_of_j(P):
    J = dense_jacobian(P)
    _, sg, _ = dc._svd(J)
    return sg


def work_counts(cameras, points, k):
    """Multiply-adds of one Levenberg-Marquardt step: the dense normal equations (Cholesky of n unknowns, n^3 / 6) against the Schur
    complement (per point of k observations of 6-column cameras: 54 k for V^-1 W^T and 108 k^2 for the update of the reduced
    system, then the Cholesky of the 6 (cameras - 2) + 5 camera unknowns)."""
    nc = 6 * (cameras - 2) + 5
    n = nc + 3 * points
    dense = n ** 3 / 6.0
    reduced = points * (54 * k + 108 * k * k) + nc ** 3 / 6.0
    return n, nc, dense, reduced, 8.0 * n * n, 8.0 * (nc * nc + 9 * points)


def ba_structure():
    P, truth, kept = initial_problem()
    lin = P.linearize()
    nnz = sum(2 * (len(l["cols"]) + 3) for l in lin)
    rows_n = 2 * len(P.obs)
    cols_n = P.nc + 3 * len(P.points)
    per_cam = [sum(1 for o in P.obs if o[0] == i) for i in range(4)]
    # the rank of a small problem with and without the gauge removed
    A = _small_true_problem(["free"] * 4)
    sgA = singular_values_of_j(A)
    B = _small_true_problem(["held", "scale", "free", "free"], 0, 1)
    sgB = singular_values_of_j(B)
    tol = 1e-9 * sgA[0]
    rankA = sum(1 for x in sgA if x > tol)
    rankB = sum(1 for x in sgB if x > 1e-9 * sgB[0])
    rows = [["all four cameras free (6 columns each)", len(sgA), rankA, len(sgA) - rankA,
             ", ".join("%.1e" % x for x in sgA[-8:])],
            ["camera 0 held, camera 1 across the baseline", len(sgB), rankB, len(sgB) - rankB,
             ", ".join("%.1e" % x for x in sgB[-8:])]]
    counts = []
    for cams, pts, k, label in ((4, 52, 3, "this chapter's problem"), (16, 8792, 3, "16 photographs, 8,792 points")):
        n, nc, dense, red, mem_d, mem_r = work_counts(cams, pts, k)
        counts.append([label, cams, pts, n, nc, "%.3g" % dense, "%.3g" % red, "%.0f" % (dense / red),
                       "%.3g" % mem_d, "%.3g" % mem_r])
    out = [f"Structure of the problem of the previous table. {len(P.obs)} observations give {rows_n} residuals; the unknowns are the "
           f"{P.nc} camera unknowns and {3 * len(P.points)} point coordinates, {cols_n} columns. Each observation touches the columns "
           f"of its camera (0 for camera 0, 5 for camera 1, 6 for cameras 2 and 3) and the 3 of its point, so the Jacobian has "
           f"{nnz} nonzero entries of {rows_n * cols_n}, {100.0 * nnz / (rows_n * cols_n):.1f}%. The numbers of observations of "
           f"cameras 0 to 3 are {', '.join(str(c) for c in per_cam)}.", "",
           "Singular values of the residual Jacobian at the true parameters, noise-free observations of the first 10 points in all "
           "4 cameras (80 residuals): the eight smallest, and the rank (singular values above 1e-9 times the largest):", "",
           _table(["parametrization", "columns", "rank", "missing", "eight smallest singular values"], rows, "lrrrl"), "",
           "Work of one step, in multiply-adds (dense: Cholesky of all n unknowns, n^3 / 6; Schur: 54 k + 108 k^2 per point with k "
           "observations, here k = 3, plus the Cholesky of the nc camera unknowns), and the bytes of the matrix that is factored "
           "(8 bytes per entry; the Schur column adds the 3 by 3 point blocks):", "",
           _table(["problem", "cameras", "points", "n", "nc", "dense multiply-adds", "Schur multiply-adds", "ratio",
                   "dense matrix bytes", "Schur bytes"], counts, "lrrrrrrrrr")]
    return "\n".join(out)


def apply_similarity(P, s, w, tvec):
    """The same reconstruction in another gauge: X -> s R_g X + t_g, camera centres likewise, rotations R -> R R_g^T."""
    Rg = cg.rodrigues(w)
    pts = [[x + y for x, y in zip(mv(Rg, [s * c for c in X]), tvec)] for X in P.points]
    poses = []
    for i in range(len(P.poses)):
        R, t = P.poses[i]
        C = [x + y for x, y in zip(mv(Rg, [s * c for c in P.cam_centre(i)]), tvec)]
        Rn = mm(R, tr(Rg))
        poses.append([Rn, [-x for x in mv(Rn, C)]])
    return poses, pts


def gauge_freedom():
    P, truth, kept = initial_problem()
    P.solve(60)
    s, w, tv_ = 2.5, [0.3, -0.5, 0.2], [1.0, -2.0, 3.0]
    poses, pts = apply_similarity(P, s, w, tv_)
    Q = Problem(poses, pts, P.obs, ["free"] * 4, free_points=True, delta=None)
    R0 = Problem(P.poses, P.points, P.obs, ["free"] * 4, free_points=True, delta=None)
    r0 = max(math.hypot(*r) for r in R0.residuals())
    r1 = max(math.hypot(*r) for r in Q.residuals())
    d_before = max(abs(a - b) for a, b in zip([x for r in R0.residuals() for x in r], [x for r in Q.residuals() for x in r]))
    rows = [["RMS reprojection (px)", "%.10f" % R0.rms(), "%.10f" % Q.rms()],
            ["largest change of any residual component (px)", "", "%.1e" % d_before],
            ["distance between camera centres 0 and 1", "%.6f" % norm(sub(R0.cam_centre(1), R0.cam_centre(0))),
             "%.6f" % norm(sub(Q.cam_centre(1), Q.cam_centre(0)))],
            ["camera 0 pose", "identity, centre (0, 0, 0)", "centre " + _vec(Q.cam_centre(0), 3)]]
    return "\n".join([f"The solution of the previous section carried into another frame: every point X becomes {s:g} R_g X + t_g, "
                      f"every camera centre likewise and every rotation R becomes R R_g^T, with R_g the rotation vector "
                      f"{_vec(w, 1)} (radians) and t_g = {_vec(tv_, 1)}.", "",
                      _table(["quantity", "solution", "same solution in the new gauge"], rows, "lrr")])


class _focal:
    """Run a block with the module's focal length F replaced (the pixel-to-ray conversion, the projection and the pixel units of the
    Sampson distance all read it); restores it on exit."""

    def __init__(self, f):
        self.f = f

    def __enter__(self):
        global F
        self.old, F = F, self.f

    def __exit__(self, *a):
        global F
        F = self.old


def _pair_with_focal(fg):
    """Pair (0, 1) reconstructed through the focal length fg from the correct inlier matches, then cameras 2 and 3 registered against
    those points: returns (pair RMS of the Sampson distances, rotation error, direction error, RMS reprojection of camera 2, of 3)."""
    tv, kp = two_view(), features()
    good = [i for i in tv["inl"] if tv["truth"][i]]
    R0, t0 = true_pair(0, 1)
    with _focal(fg):
        a = [normalized(kp[0][tv["m"][i][0]][:2]) for i in good]
        b = [normalized(kp[1][tv["m"][i][1]][:2]) for i in good]
        idx = list(range(len(good)))
        cands, counts, k = pose_from_essential(eight_point(a, b), a, b, idx)
        R, t, c0, c1 = refine_essential(cands[k][0], cands[k][1], a, b)
        X = {i: triangulate([(IDENT, [0.0, 0.0, 0.0]), (R, t)], [a[n], b[n]]) for n, i in enumerate(good)}
        rms = []
        for v in (2, 3):
            cs = [c for c in register_camera(v)["corr"] if c["ok"] and c["match"] in X]
            Xs = [X[c["match"]] for c in cs]
            Rp, tp = dlt_pose(Xs, [normalized(c["px"]) for c in cs])
            P = Problem([[Rp, tp]], Xs, [(0, j, c["px"][0], c["px"][1]) for j, c in enumerate(cs)], ["free"], free_points=False,
                        delta=None, f=fg)
            P.solve(40)
            rms.append(P.rms())
    return math.sqrt(c1 / len(good)), rot_error(R, R0), dir_error(t, t0), rms[0], rms[1]


def focal_length_effect():
    rows = []
    for fg in (400.0, 450.0, 500.0, 550.0, 600.0):
        r = _pair_with_focal(fg)
        rows.append([f"{fg:g}", "%.3f" % r[0], "%.2f" % r[1], "%.2f" % r[2], "%.2f" % r[3], "%.2f" % r[4]])
    rows2, prof = [], []
    for f0 in (450.0, 550.0):
        P, truth, kept = initial_problem(free_f=True, f0=f0)
        r0 = P.rms()
        hist = P.solve(80)
        rows2.append([f"{f0:g}", "%.3f" % r0, "%.4f" % P.rms(), "%.2f" % P.f, len(hist) - 1])
    for fv in (400.0, 450.0, 480.0, 500.0, 520.0, 550.0, 600.0):
        P, truth, kept = initial_problem(free_f=False, f0=fv)
        P.solve(80)
        prof.append([f"{fv:g}", "%.3f" % P.cost(), "%.4f" % P.rms()])
    return "\n".join([
        "The inlier matches that are correct (51 of the pair's 54) are turned into rays through a focal length f_given, the pair is "
        "estimated and triangulated as before, and cameras 2 and 3 are registered against the resulting points (DLT, then "
        "Levenberg-Marquardt on the pose alone, projecting through f_given). RMS columns are in pixels.", "",
        _table(["f_given", "pair: RMS Sampson distance", "rotation error (deg)", "direction error (deg)",
                "camera 2: RMS reprojection", "camera 3: RMS reprojection"], rows, "rrrrrr"), "",
        "The adjustment of the previous sections with the focal length as one more unknown (a column shared by every observation), "
        "started at the wrong value:", "",
        _table(["f_given", "RMS before", "RMS after", "focal length found", "iterations"], rows2, "rrrrr"), "",
        "And the adjustment with the focal length held at each value (all poses and points free): the robust cost and the RMS error of "
        "the best reconstruction through it:", "",
        _table(["f held", "robust cost (px^2)", "RMS (px)"], prof, "rrr")])


def camera_matrix_axes(P, i):
    """Camera i's rows of R (its axes in world coordinates) and its centre."""
    return P.poses[i][0], P.cam_centre(i)


def sim3_from_cameras(A, B):
    """The similarity taking world B onto world A, p_A = s R p_B + t, from the same cameras posed in each: A and B are lists of
    (rotation, centre) pairs, rotation world-to-camera. R is the nearest rotation (SVD) to sum over cameras and over the 3 axes of
    a_axis b_axis^T; s is the median over pairs of cameras of |C_A,i - C_A,j| / |C_B,i - C_B,j|; t puts the centroid of the B centres
    onto that of the A centres."""
    S = [[0.0] * 3 for _ in range(3)]
    for (Ra, _), (Rb, _) in zip(A, B):
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    S[i][j] += Ra[k][i] * Rb[k][j]                    # axis k of A against axis k of B: R b_k = a_k
    R = cg.nearest_rotation(S)
    ratios = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            da, db = norm(sub(A[i][1], A[j][1])), norm(sub(B[i][1], B[j][1]))
            if da > 1e-9 and db > 1e-9:
                ratios.append(da / db)
    ratios.sort()
    m = len(ratios)
    s = ratios[m // 2]
    ca = [sum(c[1][k] for c in A) / len(A) for k in range(3)]
    cb = [sum(c[1][k] for c in B) / len(B) for k in range(3)]
    t = sub(ca, [s * x for x in mv(R, cb)])
    return s, R, t


def camera_residual(A, B, sim):
    s, R, t = sim
    return max(norm(sub(a[1], [x + y for x, y in zip([s * v for v in mv(R, b[1])], t)])) for a, b in zip(A, B))


def chunk_registration():
    """Two chunks of the capture, cameras 0 to 2 and cameras 1 to 3 of the adjusted reconstruction, the second carried into another
    gauge; the similarity from the two shared cameras, and the gate on its residual."""
    P, truth, kept = initial_problem()
    P.solve(60)
    poses, pts = apply_similarity(P, 2.5, [0.3, -0.5, 0.2], [1.0, -2.0, 3.0])
    def cam(poses_, i):
        R, t = poses_[i]
        return (R, centre(R, t))
    A = [cam(P.poses, i) for i in (1, 2)]               # the shared cameras 1 and 2 in the world of chunk A
    rows = []
    for label, shift in (("chunk B as solved", 0.0), ("camera 2 of chunk B displaced by 15% of the span", 0.15),
                         ("camera 2 of chunk B displaced by 40% of the span", 0.40)):
        B = [cam(poses, i) for i in (1, 2)]
        span = norm(sub(A[0][1], A[1][1]))
        if shift:
            # a displaced centre in the world of B, along the baseline normal to it, by shift * span / s (B is 2.5 times larger)
            R2, C2 = B[1]
            d = unit(cross(sub(B[1][1], B[0][1]), [0.0, 1.0, 0.0]))
            B[1] = (R2, [C2[k] + shift * span * 2.5 * d[k] for k in range(3)])
        sim = sim3_from_cameras(A, B)
        res = camera_residual(A, B, sim)
        rows.append([label, "%.4f" % sim[0], "%.2e" % res, "%.4f" % (res / span), "yes" if res / span <= 0.1 else "no, refused"])
    return "\n".join([
        "Chunk A holds the adjusted cameras 0 to 2 in their own world; chunk B holds cameras 1 to 3 carried into another gauge "
        "(scale 2.5, rotation vector (0.3, -0.5, 0.2), translation (1, -2, 3)). They share cameras 1 and 2. The similarity from B "
        "to A is the rotation from the cameras' axes, the scale from the median ratio of their centre distances and the translation "
        "from the centroids; the residual is the largest distance between a shared centre in A and the registered one from B, "
        "relative to the distance between the shared centres (the span); the gate is 0.1.", "",
        _table(["chunk B", "scale found (true: 0.4)", "largest centre residual (A's units)", "relative to the span",
                "accepted (gate 0.1)"], rows, "lrrrl")])


# ------------------------------------------------------------------------------------------------ multi-view stereo on a tilted plane

MVS_W, MVS_H, MVS_F = 48, 36, 60.0            # the small cameras of the stereo example: image size and focal length in pixels
MVS_CX, MVS_CY = MVS_W / 2.0, MVS_H / 2.0
MVS_EYES = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.4, 0.1, 0.0]]   # reference, right source, left source; all look along +z
MVS_NORMAL = unit([-0.7, 0.05, -1.0])          # the plane's normal in the reference frame, facing the cameras
MVS_POINT = [0.0, 0.0, 4.0]                    # a point of the plane
MVS_SPACING = 0.25                             # lattice spacing of the texture, metres on the plane
TEXTURE_NAME, IMAGE_NAME, PM_NAME = "structure_from_motion.texture", "structure_from_motion.images", "structure_from_motion.patchmatch"
WIN_RADIUS, WIN_STEP = 5, 2
SIGMA_COLOR, SIGMA_SPATIAL, MIN_VAR, BAD = 0.03, 5.0, 2e-6, 2.0
IMAGE_NOISE = 0.01


def _plane_axes():
    e1 = unit(cross(MVS_NORMAL, [0.0, 1.0, 0.0]))
    e2 = cross(MVS_NORMAL, e1)
    return e1, e2


def _texture(nudge=0.0):
    """Lattice of uniform values, 37 columns (a = -4.5 .. 4.5 in steps of 0.25) by 25 rows (b = -3 .. 3), row by row. `nudge` is added
    to the value of the cell under the middle pixel of the reference image (only the stable-digits check passes a nonzero one)."""
    def make():
        g = bookrng.generator(TEXTURE_NAME)
        L = [[g.uniform() for _ in range(37)] for _ in range(25)]
        if nudge:
            e1, e2 = _plane_axes()
            d = mvs_ray(MVS_W / 2.0, MVS_H / 2.0)
            P = [true_range(MVS_W / 2.0, MVS_H / 2.0) * x for x in d]
            q = sub(P, MVS_POINT)
            L[int((dot(q, e2) + 3.0) / MVS_SPACING)][int((dot(q, e1) + 4.5) / MVS_SPACING)] += nudge
        return L
    return _once(("texture", nudge), make)


def texture_at(a, b, nudge=0.0):
    """Bilinear interpolation of the lattice at plane coordinates (a, b)."""
    L = _texture(nudge)
    x = (a + 4.5) / MVS_SPACING
    y = (b + 3.0) / MVS_SPACING
    i, j = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - i, y - j
    i = min(max(i, 0), 35)
    j = min(max(j, 0), 23)
    top = L[j][i] * (1 - fx) + L[j][i + 1] * fx
    bot = L[j + 1][i] * (1 - fx) + L[j + 1][i + 1] * fx
    return top * (1 - fy) + bot * fy


def mvs_ray(u, v):
    """Unit ray through the continuous pixel (u, v) of the stereo cameras."""
    return unit([(u - MVS_CX) / MVS_F, (v - MVS_CY) / MVS_F, 1.0])


def true_range(u, v, eye=(0.0, 0.0, 0.0)):
    """Distance from the camera at `eye` along the ray through (u, v) to the plane, or None."""
    d = mvs_ray(u, v)
    den = dot(MVS_NORMAL, d)
    if den > -1e-9:
        return None
    return dot(MVS_NORMAL, sub(MVS_POINT, list(eye))) / den


def stereo_images(nudge=0.0):
    """The three gray images (lists of rows of floats in [0, 1]): each pixel centre's ray meets the plane and reads the texture, plus
    Gaussian noise of standard deviation IMAGE_NOISE from the image stream, view 0, 1, 2, row by row, left to right."""
    def make():
        e1, e2 = _plane_axes()
        g = bookrng.generator(IMAGE_NAME)
        imgs = []
        for eye in MVS_EYES:
            img = []
            for y in range(MVS_H):
                row = []
                for x in range(MVS_W):
                    d = mvs_ray(x + 0.5, y + 0.5)
                    r = true_range(x + 0.5, y + 0.5, eye)
                    P = [eye[k] + r * d[k] for k in range(3)] if r is not None else MVS_POINT
                    q = sub(P, MVS_POINT)
                    row.append(min(1.0, max(0.0, texture_at(dot(q, e1), dot(q, e2), nudge) + IMAGE_NOISE * g.normal())))
                img.append(row)
            imgs.append(img)
        return imgs
    return _once(("stereo_images", nudge), make)


def sample_gray(img, u, v):
    """Bilinear gray value at the continuous pixel position (u, v); pixel i covers [i, i + 1) and is centred on i + 0.5, so the
    sample position is (u - 0.5, v - 0.5) in index units, clamped to the image."""
    fx = min(max(u - 0.5, 0.0), MVS_W - 1.0)
    fy = min(max(v - 0.5, 0.0), MVS_H - 1.0)
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, MVS_W - 1), min(y0 + 1, MVS_H - 1)
    ax, ay = fx - x0, fy - y0
    return (img[y0][x0] * (1 - ax) + img[y0][x1] * ax) * (1 - ay) + (img[y1][x0] * (1 - ax) + img[y1][x1] * ax) * ay


_RAYS = None


def _ref_rays():
    global _RAYS
    if _RAYS is None:
        _RAYS = [[mvs_ray(x + 0.5, y + 0.5) for x in range(MVS_W)] for y in range(MVS_H)]
    return _RAYS


def photo_cost(imgs, k, px, py, X0, n):
    """1 - weighted NCC of the reference patch around pixel (px, py) against source k (k = 1, 2), for the plane through X0 with
    normal n. Each window sample (dx, dy) in -5, -3, ..., 5 in both directions, inside the reference image, has the reference ray
    d_q; its point on the plane is d_q (n.X0) / (n.d_q), which the source camera (at MVS_EYES[k], same orientation) images at
    (f x / z + c_x, f y / z + c_y) with (x, y, z) = point - eye. The weight of a sample is exp(-(g_q - g_p)^2 / (2 s_c^2) - (dx^2 +
    dy^2) / (2 s_s^2)) with g from the reference image. A projection outside the source image, a window with no sample, or a weighted
    variance below MIN_VAR in either image costs BAD = 2."""
    ref, src = imgs[0], imgs[k]
    eye = MVS_EYES[k]
    rays = _ref_rays()
    nd = dot(n, X0)
    gp = ref[py][px]
    # the window centre must be imaged inside the source
    c = [X0[i] - eye[i] for i in range(3)]
    if c[2] <= 1e-9:
        return BAD
    uc, vc = MVS_F * c[0] / c[2] + MVS_CX, MVS_F * c[1] / c[2] + MVS_CY
    if not (0.0 <= uc < MVS_W and 0.0 <= vc < MVS_H):
        return BAD
    ic, isg = 1.0 / (2.0 * SIGMA_COLOR ** 2), 1.0 / (2.0 * SIGMA_SPATIAL ** 2)
    sw = sr = ss = srr = sss = srs = 0.0
    for dy in range(-WIN_RADIUS, WIN_RADIUS + 1, WIN_STEP):
        qy = py + dy
        if qy < 0 or qy >= MVS_H:
            continue
        for dx in range(-WIN_RADIUS, WIN_RADIUS + 1, WIN_STEP):
            qx = px + dx
            if qx < 0 or qx >= MVS_W:
                continue
            d = rays[qy][qx]
            den = n[0] * d[0] + n[1] * d[1] + n[2] * d[2]
            if den > -1e-4:
                continue
            s = nd / den
            z = s * d[2] - eye[2]
            if z <= 1e-9:
                return BAD
            gs = sample_gray(src, MVS_F * (s * d[0] - eye[0]) / z + MVS_CX, MVS_F * (s * d[1] - eye[1]) / z + MVS_CY)
            gr = ref[qy][qx]
            dg = gr - gp
            wt = math.exp(-dg * dg * ic - (dx * dx + dy * dy) * isg)
            sw += wt
            sr += wt * gr
            ss += wt * gs
            srr += wt * gr * gr
            sss += wt * gs * gs
            srs += wt * gr * gs
    if sw <= 0.0:
        return BAD
    mr, ms = sr / sw, ss / sw
    vr, vs = srr / sw - mr * mr, sss / sw - ms * ms
    if vr < MIN_VAR or vs < MIN_VAR:
        return BAD
    ncc = (srs / sw - mr * ms) / math.sqrt(vr * vs)
    return min(max(1.0 - ncc, 0.0), BAD)


def mean_cost(imgs, px, py, r, n):
    """The cost of the hypothesis (range r along the ray of pixel (px, py), normal n): the mean of the two sources' costs."""
    d = _ref_rays()[py][px]
    X0 = [r * d[0], r * d[1], r * d[2]]
    return 0.5 * (photo_cost(imgs, 1, px, py, X0, n) + photo_cost(imgs, 2, px, py, X0, n))


def sweep_curve(px, py, normal=(0.0, 0.0, -1.0), samples=64, rmin=2.5, rmax=8.0):
    """Cost of the plane hypotheses with the given normal over `samples` ranges uniform in inverse range between rmin and rmax."""
    imgs = stereo_images()
    out = []
    for i in range(samples):
        inv = 1.0 / rmax + (1.0 / rmin - 1.0 / rmax) * i / (samples - 1)
        r = 1.0 / inv
        out.append((r, mean_cost(imgs, px, py, r, list(normal))))
    return out


def plane_sweep():
    imgs = stereo_images()
    seed_t, seed_i = bookrng.seed_of(TEXTURE_NAME), bookrng.seed_of(IMAGE_NAME)
    e1, e2 = _plane_axes()
    rows = []
    for (px, py) in ((24, 18), (12, 10), (36, 26)):
        rt = true_range(px + 0.5, py + 0.5)
        c_true = mean_cost(imgs, px, py, rt, list(MVS_NORMAL))
        c_front = mean_cost(imgs, px, py, rt, [0.0, 0.0, -1.0])
        sw = sweep_curve(px, py)
        rb, cb = min(sw, key=lambda a: a[1])
        sw_t = sweep_curve(px, py, MVS_NORMAL)
        rb_t, cb_t = min(sw_t, key=lambda a: a[1])
        rows.append([f"({px}, {py})", "%.4f" % rt, "%.4f" % c_true, "%.4f" % c_front, "%.4f" % rb, "%.4f" % cb, "%.4f" % rb_t, "%.4f" % cb_t])
    return "\n".join([
        f"Three {MVS_W} by {MVS_H} gray images from cameras with f = {MVS_F:g} pixels and principal point ({MVS_CX:g}, {MVS_CY:g}) at "
        f"the eyes {MVS_EYES[0]}, {MVS_EYES[1]} and {MVS_EYES[2]} (reference, right source, left source), all looking along +z. The "
        f"scene is the plane through {_vec(MVS_POINT, 1)} with unit normal {_vec(MVS_NORMAL, 6)}, tilted about the vertical axis. Its "
        f"texture is a lattice of 37 by 25 uniform values 0.25 m apart (`bookrng.generator(\"{TEXTURE_NAME}\")`, seed {seed_t}, row "
        "by row, values in the order of increasing a within a row), interpolated bilinearly over the plane coordinates (a, b) along "
        "e1 = unit(n x (0, 1, 0)) and e2 = n x e1 measured from the plane point. Each pixel centre's ray meets the plane and reads the "
        f"texture; Gaussian noise of standard deviation {IMAGE_NOISE:g} is added (`bookrng.generator(\"{IMAGE_NAME}\")`, seed {seed_i}; "
        "views in order, rows top to bottom, left to right, one `normal()` each).", "",
        f"The cost of a hypothesis for the reference pixel is the mean over the two sources of 1 - NCC, the bilaterally weighted "
        f"normalized cross-correlation of the {len(range(-WIN_RADIUS, WIN_RADIUS + 1, WIN_STEP))} by "
        f"{len(range(-WIN_RADIUS, WIN_RADIUS + 1, WIN_STEP))} samples of an 11 by 11 window (spacing {WIN_STEP}), color sigma "
        f"{SIGMA_COLOR:g}, spatial sigma {SIGMA_SPATIAL:g}. A plane sweep tries 64 ranges, uniform in inverse range from 2.5 to 8:", "",
        _table(["pixel", "true range", "cost at the true plane", "cost at the true range, fronto-parallel normal",
                "best range of the fronto-parallel sweep", "its cost", "best range with the true normal", "its cost"], rows, "lrrrrrrr")])


def _rand_dir(g):
    """A uniformly random unit vector: z = 2 u - 1, then phi = 2 pi u, (sqrt(1 - z^2) cos phi, sqrt(1 - z^2) sin phi, z); two draws."""
    z = 2.0 * g.uniform() - 1.0
    phi = 2.0 * math.pi * g.uniform()
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return [s * math.cos(phi), s * math.sin(phi), z]


def _face(n, d):
    """n turned to face a camera looking along d: -n if n . d > 0."""
    return [-x for x in n] if dot(n, d) > 0.0 else n


def _plane_range(n, X, d):
    """Range along the unit ray d to the plane through X with normal n: (n.X) / (n.d); None when n.d > -1e-4 (parallel or facing away)."""
    den = dot(n, d)
    if den > -1e-4:
        return None
    return dot(n, X) / den


PM_RMIN, PM_RMAX = 2.5, 8.0
PM_ITERS = 6


def patchmatch(nudge=0.0, iters=PM_ITERS, name=PM_NAME):
    """Red-black PatchMatch on the reference image, returning (per-iteration statistics, final state). `nudge` is the change added to
    the texture value under the middle pixel by the stable-digits check (see patchmatch_table)."""
    imgs = stereo_images(nudge)
    g = bookrng.generator(name)
    rays = _ref_rays()
    H, W = MVS_H, MVS_W
    state = [[None] * W for _ in range(H)]
    evals = [0]

    def cost_of(px, py, r, n):
        evals[0] += 1
        return mean_cost(imgs, px, py, r, n)

    for y in range(H):
        for x in range(W):
            d = rays[y][x]
            r = 1.0 / (1.0 / PM_RMAX + (1.0 / PM_RMIN - 1.0 / PM_RMAX) * g.uniform())
            n = _face(_rand_dir(g), d)
            state[y][x] = [r, n, cost_of(x, y, r, n)]
    truth = [[true_range(x + 0.5, y + 0.5) for x in range(W)] for y in range(H)]

    def stats(label):
        errs, nerr, within = [], [], 0
        for y in range(H):
            for x in range(W):
                r, n, c = state[y][x]
                e = abs(r - truth[y][x]) / truth[y][x]
                errs.append(e)
                nerr.append(math.degrees(math.acos(max(-1.0, min(1.0, dot(n, MVS_NORMAL))))))
                within += e < 0.01
        errs.sort()
        nerr.sort()
        m = len(errs)
        return [label, float(evals[0]), 0.5 * (errs[m // 2 - 1] + errs[m // 2]), 100.0 * within / m,
                0.5 * (nerr[m // 2 - 1] + nerr[m // 2])]
    rows = [stats("random start")]
    for it in range(iters):
        dr, dn = max(0.1 * 0.5 ** it, 0.002), max(0.5 * 0.5 ** it, 0.02)
        for colour in (0, 1):
            for y in range(H):
                for x in range(W):
                    if (x + y) % 2 != colour:
                        continue
                    d = rays[y][x]
                    best = list(state[y][x])
                    # propagation: the planes of the four neighbours (the other colour), re-intersected with this pixel's ray
                    for (nx, ny) in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < W and 0 <= ny < H:
                            rn, nn, _ = state[ny][nx]
                            dn_ = rays[ny][nx]
                            r = _plane_range(nn, [rn * dn_[k] for k in range(3)], d)
                            if r is not None and 0.0 < r < 4.0 * PM_RMAX:
                                c = cost_of(x, y, r, nn)
                                if c < best[2]:
                                    best = [r, nn, c]
                    # refinement: perturbed and fresh random hypotheses
                    r0, n0 = best[0], best[1]
                    r_pert = r0 * math.exp(dr * (2.0 * g.uniform() - 1.0))
                    rd = _rand_dir(g)
                    n_pert = unit([n0[k] + dn * rd[k] for k in range(3)])
                    r_rand = 1.0 / (1.0 / PM_RMAX + (1.0 / PM_RMIN - 1.0 / PM_RMAX) * g.uniform())
                    n_rand = _face(_rand_dir(g), d)
                    for (r, n) in ((r_pert, n0), (r0, n_pert), (r_pert, n_pert), (r_rand, n0), (r0, n_rand), (r_rand, n_rand)):
                        if dot(n, d) > -0.05:
                            continue
                        c = cost_of(x, y, r, n)
                        if c < best[2]:
                            best = [r, n, c]
                    state[y][x] = best
        rows.append(stats(f"iteration {it + 1}"))
    return rows, state


def patchmatch_table():
    rows, state = patchmatch()
    runs = [patchmatch(nu)[0] for nu in stable.NUDGES]
    seed = bookrng.seed_of(PM_NAME)
    shown = []
    for k, row in enumerate(rows):
        ws = lambda c: [run[k][c] for run in runs]
        shown.append([row[0], stable.text(row[1], ws(1), lambda x: str(int(round(x)))),
                      stable.text(row[2], ws(2), lambda x: "%.4f" % x), stable.text(row[3], ws(3), lambda x: "%.1f%%" % x),
                      stable.text(row[4], ws(4), lambda x: "%.1f" % x)])
    per_iter = int(rows[-1][1])
    nsamp = len(range(-WIN_RADIUS, WIN_RADIUS + 1, WIN_STEP)) ** 2
    n_normals = round(2.0 * math.pi / math.radians(10.0) ** 2)
    n_ranges = round((1.0 / PM_RMIN - 1.0 / PM_RMAX) / (0.01 * 0.25))
    out = [f"PatchMatch on the reference image with the two sources, window and weights of the plane sweep. Generator "
           f"`bookrng.generator(\"{PM_NAME}\")` (seed {seed}). Start: for every pixel in row-major order, three draws: the range "
           f"(1 / (1/8 + (1/2.5 - 1/8) u)), then the two draws of a random direction (z = 2u - 1, phi = 2 pi u), the normal turned "
           f"to face the camera. Then {PM_ITERS} iterations of two half-iterations, colour (x + y) mod 2 = 0 and then 1, pixels of a "
           "colour in row-major order, each in turn: take the planes of its four neighbours (which have the other colour and are "
           "final for this half-iteration), re-intersect each with this pixel's ray, and keep the cheapest of these and the current "
           "hypothesis; then draw six uniform numbers (the range perturbation u, the two numbers of the direction added to the normal, "
           "the fresh range, the two numbers of a fresh normal) and score the six combinations (perturbed range, perturbed normal, "
           "both, fresh range, fresh normal, both fresh), skipping a hypothesis with n.d > -0.05, keeping the cheapest overall. In "
           "iteration t (from 0) the perturbation of the range is a relative factor exp(0.1 * 0.5^t (2u - 1)) and the added direction "
           "is scaled by 0.5 * 0.5^t (floors 0.002 and 0.02). Every number below is printed with only the digits that are the same in this "
           "run and in four copies of it whose texture value under the middle pixel is changed by +1e-12, -1e-12, +3e-12 and -3e-12 "
           "(the stable-digits rule of the book): a decision of the search that depended on rounding would show as a digit that "
           "differs.", "",
           _table(["after", "cost evaluations so far", "median relative range error", "within 1% of the true range",
                   "median normal error (deg)"], shown, "lrrrr"), "",
           f"A cost evaluation is one hypothesis scored against both sources, {nsamp} window samples each. The run used {per_iter} of "
           f"them for {MVS_W * MVS_H} pixels, {per_iter / (MVS_W * MVS_H):.0f} per pixel. An exhaustive search at the same resolution "
           f"needs {n_ranges} ranges (steps of 1% of the inverse range at its middle value) times {n_normals} normals (cells of 10 "
           f"degrees on a hemisphere, 2 pi / (10 degrees in radians)^2) = {n_ranges * n_normals} evaluations per pixel, "
           f"{n_ranges * n_normals / (per_iter / (MVS_W * MVS_H)):.0f} times as many."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ data for the figures

def epipolar_figure_data():
    """Epipolar lines in image 1 of ten matches of the pair (0, 1) under the refined essential matrix: six correct inlier matches and
    four wrong ones (the first of each kind in match order). Each entry: (u0, v0, line (l0, l1, l2) with l0 u + l1 v + l2 = 0, matched
    pixel in image 1, correct)."""
    tv, kp = two_view(), features()
    E = essential_of(tv["R"], tv["t"])
    Kinv = [[1 / F, 0.0, -CX / F], [0.0, 1 / F, -CY / F], [0.0, 0.0, 1.0]]
    Fm = mm(mm(tr(Kinv), E), Kinv)
    good = [i for i in range(len(tv["m"])) if tv["truth"][i] and tv["mask"][i]][:6]
    bad = [i for i in range(len(tv["m"])) if not tv["truth"][i]][:4]
    out = []
    for i in good + bad:
        ia, ib = tv["m"][i]
        x0 = kp[0][ia]
        line = mv(Fm, [x0[0], x0[1], 1.0])
        out.append((x0[0], x0[1], line, (kp[1][ib][0], kp[1][ib][1]), tv["truth"][i]))
    return out


def four_pose_figure_data():
    """For each of the four decompositions: the centre of camera 1 and its optical axis in the frame of camera 0 (x, z), the mean
    triangulated point (x, z) and how many of the inliers are in front of both cameras."""
    tv = two_view()
    out = []
    for i, (R, t) in enumerate(tv["cands"]):
        X = [triangulate([(IDENT, [0.0, 0.0, 0.0]), (R, t)], [tv["a"][j], tv["b"][j]]) for j in tv["inl_best"]]
        mean = [sum(x[k] for x in X) / len(X) for k in range(3)]
        C = centre(R, t)
        axis = R[2]
        out.append(dict(C=C, axis=axis, mean=mean, front=tv["counts"][i], total=len(tv["inl_best"]), chosen=(i == tv["best"])))
    return out
