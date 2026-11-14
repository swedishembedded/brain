# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "3D Gaussian Splatting".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: gaussian_splatting.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures.

The module is a complete small splat renderer with its backward pass:

* a Gaussian is 14 numbers (mean 3, log scales 3, raw quaternion 4, opacity logit, colour 3); its covariance is R diag(s^2) R^T;
* projection to a screen ellipse through the Jacobian of the perspective map, the 0.3 pixel^2 dilation and the truncation radius;
* tile binning: counts, an exclusive scan, 32-bit keys (tile number over the top bits of the depth's float32 pattern), a stable
  least-significant-digit radix sort of 8-bit digits, and the tile ranges by neighbour comparison;
* front-to-back compositing with the opacity cap 0.99, the cut-off 1/255 and the stop at transmittance 1e-4 (the stopping
  Gaussian is left out);
* spherical-harmonic colour of degrees 0 to 2;
* the analytic gradient of a pixel loss with respect to all 14 numbers of every Gaussian, checked against central differences;
* Adam on a three-Gaussian scene seen from four cameras, with the nudges of `stable.py`;
* the split and clone rules of density control, and the opacity and scale correction that makes a relocation image-preserving;
* evaluation along rays through a pinhole and an equidistant fisheye lens, against the linearized footprint;
* the work of the tiled pipeline in pairs, FLOPs and bytes.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, exp, log, sin, cos, atan2, tan, acos) unless a block
says otherwise; the 32-bit patterns of depths are those of IEEE binary32 (`struct`). Sums are accumulated left to right in index
order. Every random number comes from the book's SplitMix64 (`bookrng.generator`, stream names printed by the chapter).

Swedish Embedded AB implements solutions for differentiable rendering and radiance-field reconstruction. If your team needs expertise
in Gaussian splatting and photogrammetry, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import struct
from functools import lru_cache

import bookrng
import camera_geometry as cg
import decompositions as dc
import images as im
import stable as stab

_table = im._table
_g = dc._g
mm, mv, tr, dot, norm, unit, cross, sub = cg.mm, cg.mv, cg.tr, cg.dot, cg.norm, cg.unit, cg.cross, cg.sub

EPS2D = 0.3          # the dilation added to the diagonal of every screen covariance, pixels^2
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_STOP = 1e-4
TILE = 16
EXTEND_MAX = 3.33
BG0 = (0.0, 0.0, 0.0)


def _v(x, d=4):
    return "(" + ", ".join(_g(c, d) for c in x) + ")"


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def logit(p):
    return math.log(p / (1.0 - p))


# ------------------------------------------------------------------------------------------------ the primitive

def gaussian(mu, scale, q, opacity, colour):
    """One Gaussian as its 14 raw numbers: mean (0..2), log scales (3..5), raw quaternion w x y z (6..9), opacity logit (10), colour (11..13)."""
    return list(mu) + [math.log(s) for s in scale] + list(q) + [logit(opacity)] + list(colour)


def rotation(q):
    """The rotation of the quaternion q = (w, x, y, z) after normalising it (the matrix of Chapter 3)."""
    return cg.matrix_of_quat(q)


def covariance(R, s):
    """Sigma = sum_k s_k^2 (column k of R)(column k of R)^T, entry by entry, k from 0 to 2."""
    return [[sum(s[k] * s[k] * R[i][k] * R[j][k] for k in range(3)) for j in range(3)] for i in range(3)]


def covariance_from_scale_rotation():
    q = [1.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0]          # unit: 1/9 + 4/9 + 4/9 = 1
    s = [0.30, 0.10, 0.05]
    R = rotation(q)
    S = covariance(R, s)
    Rs = mm(R, [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]])
    S2 = mm(Rs, tr(Rs))                                  # (R diag(s)) (R diag(s))^T, the same matrix by another route
    asym = max(abs(S[i][j] - S[j][i]) for i in range(3) for j in range(3))
    diff = max(abs(S[i][j] - S2[i][j]) for i in range(3) for j in range(3))
    trace = sum(S[i][i] for i in range(3))
    det = cg.det3(S)
    rtr = mm(tr(R), R)
    orth = max(abs(rtr[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
    # the quadratic form along the axes: v^T Sigma v for the three columns of R gives s_k^2
    along = [dot([R[i][k] for i in range(3)], mv(S, [R[i][k] for i in range(3)])) for k in range(3)]
    # positive definiteness by the leading minors
    m1 = S[0][0]
    m2 = S[0][0] * S[1][1] - S[0][1] ** 2
    rows = [[f"{_g(R[i][0], 6)}", f"{_g(R[i][1], 6)}", f"{_g(R[i][2], 6)}", f"{_g(S[i][0], 6)}", f"{_g(S[i][1], 6)}", f"{_g(S[i][2], 6)}"]
            for i in range(3)]
    return ("A Gaussian with quaternion (w, x, y, z) = (1/3, 2/3, 0, 2/3) and scales s = (0.30, 0.10, 0.05). The matrix is that of the "
            "unit quaternion of Chapter 3, R = [[1 - 2(y^2 + z^2), 2(xy - wz), 2(xz + wy)], [2(xy + wz), 1 - 2(x^2 + z^2), "
            "2(yz - wx)], [2(xz - wy), 2(yz + wx), 1 - 2(x^2 + y^2)]] after dividing the quaternion by its length, and each entry of Sigma is the sum over k = 0, 1, 2 of "
            "s_k^2 R[i][k] R[j][k], added in that order.\n\n"
            + _table(["row", "R column 0", "R column 1", "R column 2", "Sigma column 0", "Sigma column 1", "Sigma column 2"],
                     [[str(i)] + rows[i] for i in range(3)], "lrrrrrr")
            + "\n\nChecks, each with the criterion that the largest difference is below 1e-15 (64-bit arithmetic): "
            + f"R^T R against I {'passes' if orth < 1e-15 else 'FAILS'}; Sigma against its transpose {'passes' if asym < 1e-15 else 'FAILS'}; "
            f"Sigma against (R diag(s))(R diag(s))^T, the same matrix by another route, {'passes' if diff < 1e-15 else 'FAILS'}. The "
            f"trace of Sigma is {trace:.6f} = 0.09 + 0.01 + 0.0025 = {sum(v * v for v in s):.6f} and its determinant "
            f"{det:.6e} = (0.30 * 0.10 * 0.05)^2 = {(s[0] * s[1] * s[2]) ** 2:.6e}. Along the three columns of R the quadratic form "
            f"v^T Sigma v is {_v(along, 6)}, the squares of the scales. The leading minors {m1:.6f} and {m2:.6e} and the determinant are "
            "all positive, so Sigma is positive definite.")


# ------------------------------------------------------------------------------------------------ the camera and the projection

def make_camera(eye, target, f, W, H, up=(0.0, -1.0, 0.0)):
    """A pinhole camera at `eye` looking at `target` (world y points down, so the picture's up is (0, -1, 0)), focal length f in
    pixels for both axes, principal point at the centre of a W by H image. R, t are the world-to-camera pair of Chapter 3."""
    R, t = cg.look_at(list(eye), list(target), list(up))
    return {"R": R, "t": t, "fx": f, "fy": f, "cx": W / 2.0, "cy": H / 2.0, "W": W, "H": H}


def project(p, cam, eps=EPS2D):
    """Everything the compositing needs of one Gaussian seen by one camera, or None when it is culled.

    Camera-frame mean m = R mu + t = (x, y, z); pixel (u, v) = (fx x / z + cx, fy y / z + cy); the Jacobian of that map at m,
    J = [[fx / z, 0, -fx x / z^2], [0, fy / z, -fy y / z^2]]; screen covariance S2 = (J R) Sigma (J R)^T plus eps on the diagonal;
    the conic Lam = S2^-1; the radius ceil(extend sqrt(S2 diagonal)) with extend = min(3.33, sqrt(2 ln(255 opacity))).
    """
    mu, ls, q, lo, c = p[0:3], p[3:6], p[6:10], p[10], p[11:14]
    s = [math.exp(l) for l in ls]
    m = [a + b for a, b in zip(mv(cam["R"], mu), cam["t"])]
    x, y, z = m
    if z <= 0.01:
        return None
    R3 = rotation(q)
    S3 = covariance(R3, s)
    fx, fy = cam["fx"], cam["fy"]
    J = [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]]
    T = mm(J, cam["R"])
    S2 = mm(mm(T, S3), tr(T))
    a, b, d = S2[0][0] + eps, S2[0][1], S2[1][1] + eps
    det = a * d - b * b
    if det <= 0.0:
        return None
    op = sigmoid(lo)
    if 255.0 * op <= 1.0:
        return None
    ext = min(EXTEND_MAX, math.sqrt(2.0 * math.log(255.0 * op)))
    rx, ry = math.ceil(ext * math.sqrt(a)), math.ceil(ext * math.sqrt(d))
    u, v = fx * x / z + cam["cx"], fy * y / z + cam["cy"]
    if u + rx <= 0 or u - rx >= cam["W"] or v + ry <= 0 or v - ry >= cam["H"]:
        return None
    return {"u": u, "v": v, "z": z, "m": m, "J": J, "T": T, "S3": S3, "R3": R3, "s": s, "S2": S2, "La": d / det, "Lb": -b / det,
            "Lc": a / det, "op": op, "rx": rx, "ry": ry, "c": c, "ext": ext}


def conic_matrix(r):
    return [[r["La"], r["Lb"]], [r["Lb"], r["Lc"]]]


def projection_footprint():
    cam = {"R": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "t": [0.0, 0.0, 0.0], "fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 24.0,
           "W": 64, "H": 48}
    q = [1.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0]
    g = gaussian([0.5, -0.2, 4.0], [0.30, 0.10, 0.05], q, 0.8, [0.9, 0.5, 0.2])
    r = project(g, cam, eps=0.0)
    r3 = project(g, cam)
    J = r["J"]
    S2 = r["S2"]
    S3 = r["S3"]
    lam = conic_matrix(r)
    prod = mm(S2, lam)
    # eigenvalues of the 2 by 2 screen covariance: semi-axes of the one-sigma ellipse
    tr2, det2 = S2[0][0] + S2[1][1], S2[0][0] * S2[1][1] - S2[0][1] ** 2
    disc = math.sqrt(tr2 * tr2 / 4 - det2)
    e1, e2 = tr2 / 2 + disc, tr2 / 2 - disc
    ang = 0.5 * math.degrees(math.atan2(2 * S2[0][1], S2[0][0] - S2[1][1]))
    rows = [
        ["camera-frame mean m", _v(r["m"], 6)],
        ["pixel (u, v) = (100 x / z + 32, 100 y / z + 24)", _v([r["u"], r["v"]], 6)],
        ["J row 0 (100 / z, 0, -100 x / z^2)", _v(J[0], 6)],
        ["J row 1 (0, 100 / z, -100 y / z^2)", _v(J[1], 6)],
        ["S2 = J Sigma J^T, entries (00, 01, 11)", _v([S2[0][0], S2[0][1], S2[1][1]], 6)],
        ["one-sigma semi-axes (square roots of the eigenvalues)", _v([math.sqrt(e1), math.sqrt(e2)], 5)],
        ["direction of the long axis, degrees from the u axis", _g(ang, 5)],
        ["S2 + 0.3 I, entries (00, 01, 11)", _v([r3["S2"][0][0] + EPS2D, r3["S2"][0][1], r3["S2"][1][1] + EPS2D], 6)],
        ["conic (S2 + 0.3 I)^-1, entries (a, b, c)", _v([r3["La"], r3["Lb"], r3["Lc"]], 6)],
        ["extend = min(3.33, sqrt(2 ln(255 * 0.8)))", _g(r3["ext"], 6)],
        ["radius (rx, ry) = ceil(extend sqrt(S2 + 0.3 I diagonal))", f"({r3['rx']}, {r3['ry']})"],
        ["opacity compensation sqrt(det S2 / det(S2 + 0.3 I)), not applied by default", _g(math.sqrt(det2 / ((r3["S2"][0][0] + EPS2D) * (r3["S2"][1][1] + EPS2D) - r3["S2"][0][1] ** 2)), 6)],
    ]
    return ("The Gaussian of the previous block (quaternion (1/3, 2/3, 0, 2/3), scales (0.30, 0.10, 0.05)) moved to the mean "
            "(0.5, -0.2, 4.0) and given opacity 0.8, seen by the camera at the origin looking along +z (R = I, t = 0) with "
            "f_x = f_y = 100, principal point (32, 24), a 64 by 48 image. J is the Jacobian of the perspective map at the mean, "
            "S2 = (J R) Sigma (J R)^T with R = I. The first seven rows leave out the dilation; the rest add 0.3 pixel^2 to both "
            "diagonal entries. Matrix products are summed over the inner index in increasing order.\n\n"
            + _table(["quantity", "value"], rows, "lr")
            + "\n\nThe product S2 (S2^-1) (without the dilation) "
            + ("equals I to below 1e-12" if max(abs(prod[i][j] - (1.0 if i == j else 0.0)) for i in range(2) for j in range(2)) < 1e-12 else "FAILS")
            + ". Both radii are rounded up to whole pixels so that a box of that half-width contains the "
            "ellipse at which the Gaussian's opacity falls to 1/255.")


# ------------------------------------------------------------------------------------------------ compositing one pixel

def walk(recs, px, py, bg=BG0):
    """Composite the projected Gaussians `recs` (already sorted front to back) at the pixel centre (px, py).

    Returns (colour, final transmittance, steps, visits). A step is (index in recs, alpha, T before it, sigma); `visits` counts
    the records looked at, including the ones skipped, up to and including the one that stopped the walk.
    """
    T = 1.0
    col = [0.0, 0.0, 0.0]
    steps = []
    visits = 0
    for i, r in enumerate(recs):
        visits += 1
        dx, dy = r["u"] - px, r["v"] - py
        sigma = 0.5 * (r["La"] * dx * dx + r["Lc"] * dy * dy) + r["Lb"] * dx * dy
        if sigma < 0.0:
            continue
        a = min(ALPHA_MAX, r["op"] * math.exp(-sigma))
        if a < ALPHA_MIN:
            continue
        nT = T * (1.0 - a)
        if nT <= T_STOP:
            break
        w = a * T
        for k in range(3):
            col[k] += r["c"][k] * w
        steps.append((i, a, T, sigma))
        T = nT
    return [col[k] + T * bg[k] for k in range(3)], T, steps, visits


def sorted_records(params, cam):
    """Project every Gaussian, drop the culled ones and sort the rest by camera depth (stable). Returns (records, original index)."""
    out = []
    for gi, p in enumerate(params):
        r = project(p, cam)
        if r is not None:
            out.append((r["z"], gi, r))
    out.sort(key=lambda t: t[0])
    return [t[2] for t in out], [t[1] for t in out]


def render(params, cam, bg=BG0):
    recs, _ = sorted_records(params, cam)
    return [[walk(recs, x + 0.5, y + 0.5, bg)[0] for x in range(cam["W"])] for y in range(cam["H"])]


def compositing_pixel():
    """Seven Gaussians along one pixel's ray, front to back: one capped, one skipped, and the last stops the walk."""
    sigma_ = [0.0, 0.0, 6.5, 0.0, 0.0, 0.0, 0.0]
    ops = [0.995, 0.8, 0.9, 0.7, 0.6, 0.5, 0.3]
    cols = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    bg = (0.2, 0.2, 0.2)
    T, col, rows = 1.0, [0.0, 0.0, 0.0], []
    stopped = None
    used = []
    for i, (sg, o, c) in enumerate(zip(sigma_, ops, cols)):
        raw = o * math.exp(-sg)
        a = min(ALPHA_MAX, raw)
        if a < ALPHA_MIN:
            rows.append([str(i), _g(sg), _g(raw, 5), "skipped (below 1/255)", "", "", ""])
            continue
        nT = T * (1.0 - a)
        if nT <= T_STOP:
            rows.append([str(i), _g(sg), _g(raw, 5), _g(a, 5), _g(T, 5), "-", f"stops: T would be {nT:.1e}, at most 1e-4; not added"])
            stopped = i
            break
        w = a * T
        col = [col[k] + w * c[k] for k in range(3)]
        used.append(i)
        rows.append([str(i), _g(sg), _g(raw, 5), _g(a, 5) + (" (capped)" if raw > ALPHA_MAX else ""), _g(T, 5), _g(w, 5), _v(col, 5)])
        T = nT
    final = [col[k] + T * bg[k] for k in range(3)]
    # the same Gaussians in the reverse order (no stop: the order is the point)
    T2, col2 = 1.0, [0.0, 0.0, 0.0]
    for i in reversed(used):
        a = min(ALPHA_MAX, ops[i] * math.exp(-sigma_[i]))
        col2 = [col2[k] + a * T2 * cols[i][k] for k in range(3)]
        T2 *= 1.0 - a
    swapped = [col2[k] + T2 * bg[k] for k in range(3)]
    wsum = 0.0
    Tt = 1.0
    for i in used:
        a = min(ALPHA_MAX, ops[i] * math.exp(-sigma_[i]))
        wsum += a * Tt
        Tt *= 1.0 - a
    return ("One pixel; seven Gaussians in depth order, with kappa (the exponent of the Gaussian at this pixel) 0, 0, 6.5, 0, 0, 0, 0, "
            "peak opacities 0.995, 0.8, 0.9, 0.7, 0.6, 0.5, 0.3, colours red, green, blue, yellow, cyan, magenta, white, and background "
            "(0.2, 0.2, 0.2). The raw alpha is opacity * exp(-kappa), capped at 0.99; a pair below 1/255 is skipped; the walk stops "
            "before a Gaussian that would take T to 1e-4 or below. T is the transmittance before the row, w = alpha T.\n\n"
            + _table(["i", "kappa", "opacity * exp(-kappa)", "alpha", "T before", "w = alpha T", "accumulated colour / note"], rows, "rrrrrrl")
            + f"\n\nThe walk stops at Gaussian {stopped}. The pixel is the accumulated colour plus T * background = {_v(final, 6)} with "
            f"T = {_g(T, 6)}. The weights of the {len(used)} Gaussians used sum to {_g(wsum, 8)} = 1 - T = {_g(1.0 - T, 8)}. Composited "
            f"in the reverse order (the same {len(used)} Gaussians, last used first, each with its own alpha, no stop), the same Gaussians give {_v(swapped, 6)}: compositing is not commutative, which is why the "
            "order is sorted by depth.")


# ------------------------------------------------------------------------------------------------ view-dependent colour

C0 = 0.28209479177387814      # 1 / (2 sqrt(pi)): the constant harmonic
C1 = 0.4886025119029199       # sqrt(3 / (4 pi))
C2 = (1.0925484305920792, 0.31539156525252005, 0.5462742152960396)   # sqrt(15 / (4 pi)), sqrt(5 / (16 pi)), sqrt(15 / (16 pi))


def sh_basis(d):
    """The eight real spherical harmonics of degrees 1 and 2 at the unit direction d, in the order and with the signs of the Inria
    layout (the degree-0 constant C0 is separate). Returns [Y1,-1 .. Y1,1 (3), Y2,-2 .. Y2,2 (5)]."""
    x, y, z = unit(d)
    return [-C1 * y, C1 * z, -C1 * x,
            C2[0] * x * y, -C2[0] * y * z, C2[1] * (2 * z * z - x * x - y * y), -C2[0] * x * z, C2[2] * (x * x - y * y)]


def sh_colour(dc, rest, mean, eye):
    """colour = max(0, C0 dc + 0.5 + sum_t Y_t(d) rest[t]) per channel, d = unit(mean - eye) pointing from the camera to the Gaussian.
    `rest` holds 3 or 8 coefficients per channel (degree 1 or degrees 1 and 2)."""
    y = sh_basis(sub(mean, eye))
    return [max(0.0, C0 * dc[c] + 0.5 + sum(y[t] * rest[c][t] for t in range(len(rest[c])))) for c in range(3)]


def sh_degree_for_views(views):
    """The largest d in 0..3 with 8 (d + 1)^2 <= views, else 0."""
    return next((d for d in (3, 2, 1, 0) if 8 * (d + 1) ** 2 <= views), 0)


def spherical_harmonics():
    n_t, n_p = 90, 180
    gram = [[0.0] * 9 for _ in range(9)]
    for i in range(n_t):
        th = math.pi * (i + 0.5) / n_t
        for j in range(n_p):
            ph = 2.0 * math.pi * (j + 0.5) / n_p
            d = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            w = math.sin(th) * (math.pi / n_t) * (2.0 * math.pi / n_p)
            y = [C0] + sh_basis(d)
            for a in range(9):
                for b in range(9):
                    gram[a][b] += w * y[a] * y[b]
    err = max(abs(gram[a][b] - (1.0 if a == b else 0.0)) for a in range(9) for b in range(9))
    # one Gaussian, three viewpoints
    dc = [0.8, -0.6, -1.4]        # C0 dc + 0.5 = (0.7258, 0.3308, 0.1052)
    rest = [[0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # red: strong y-ish dependence on x (degree 1, coefficient 2)
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    mean = [0.0, 0.0, 4.0]
    eyes = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -3.0, 1.0], [0.0, 3.0, 1.0]]
    rows = []
    for e in eyes:
        d = unit(sub(mean, e))
        col = sh_colour(dc, rest, mean, e)
        rows.append([_v(e, 3), _v(d, 5), _v(col, 5)])
    flat = [max(0.0, C0 * dc[c] + 0.5) for c in range(3)]
    views = (8, 16, 31, 32, 71, 72, 127, 128, 500)
    return ("Check of the basis: the nine functions (the constant C0 = 0.28209479177387814 and the eight of degrees 1 and 2) integrated "
            "against one another over the sphere by the midpoint rule with 90 equal steps in the polar angle theta and 180 in the "
            "azimuth phi (weight sin(theta) d theta d phi). The largest entry of |Gram - I| is "
            f"{'below 1e-3' if err < 1e-3 else 'ABOVE 1e-3'} (criterion 1e-3; the rule is second-order accurate).\n\n"
            f"The eight functions, for the unit direction d = (d_x, d_y, d_z) and C1 = {C1!r}, C2 = ({C2[0]!r}, {C2[1]!r}, {C2[2]!r}) = "
            "(sqrt(15 / (4 pi)), sqrt(5 / (16 pi)), sqrt(15 / (16 pi))), in the order of the coefficients: degree 1: -C1 d_y, C1 d_z, "
            "-C1 d_x; degree 2: C2[0] d_x d_y, -C2[0] d_y d_z, C2[1] (2 d_z^2 - d_x^2 - d_y^2), -C2[0] d_x d_z, C2[2] (d_x^2 - d_y^2).\n\n"
            "A Gaussian at the mean (0, 0, 4) with base coefficients dc = (0.8, -0.6, -1.4), so that the flat colour "
            f"C0 dc + 0.5 is {_v(flat, 5)}, and harmonic coefficients: red 0.30 on the first degree-1 function, green 0.25 on the "
            "third degree-2 function, blue -0.5 on the first degree-1 function, all others 0. Colour = max(0, flat + sum Y_t(d) coefficient_t), "
            "d the unit vector from the eye to the mean.\n\n"
            + _table(["eye", "direction d", "colour"], rows, "lrr")
            + "\n\nThe first degree-1 function is -C1 d_y, so red and blue change only with the height of the eye: "
              "looking from the origin (d_y = 0) leaves the flat colour; from above (the eye at y = -3, the y axis points down) d_y is "
              "positive and red falls while blue rises, clipped at zero where the sum goes negative.\n\n"
            "Degree for a capture of V views: the largest d from 0 to 3 with 8 (d + 1)^2 <= V (the degree is capped at 3, which the "
            "rule alone would exceed from V = 200):\n\n"
            + _table(["views V", "degree"], [[str(v), str(sh_degree_for_views(v))] for v in views], "rr"))


# ------------------------------------------------------------------------------------------------ tiles, keys and the radix sort

TILE_CAM = {"R": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "t": [0.0, 0.0, 0.0], "fx": 40.0, "fy": 40.0, "cx": 24.0, "cy": 16.0,
            "W": 48, "H": 32}
_Q0 = [1.0, 0.0, 0.0, 0.0]
TILE_SCENE = [
    gaussian([-0.5, -0.2, 3.0], [0.20, 0.20, 0.20], _Q0, 0.9, [0.9, 0.2, 0.2]),
    gaussian([0.6, 0.3, 2.0], [0.25, 0.15, 0.15], [0.9238795325, 0.0, 0.0, 0.3826834324], 0.8, [0.2, 0.7, 0.3]),
    gaussian([0.1, -0.3, 6.0], [0.60, 0.30, 0.30], _Q0, 0.7, [0.2, 0.3, 0.9]),
    gaussian([-0.2, 0.5, 4.0], [0.10, 0.10, 0.10], _Q0, 0.95, [0.9, 0.8, 0.1]),
    gaussian([0.5, -0.4, 5.0], [0.30, 0.10, 0.10], _Q0, 0.6, [0.8, 0.3, 0.8]),
    gaussian([-0.9, 0.35, 7.0], [0.40, 0.40, 0.40], _Q0, 0.5, [0.1, 0.7, 0.8]),
]
SORT_BITS = 32


def f32_bits(x):
    """The 32 stored bits of x as a binary32 number (the `bits` of the book's pseudocode)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def tile_layout(W, H):
    tx, ty = -(-W // TILE), -(-H // TILE)
    n = tx * ty
    tile_bits = max(1, (n - 1).bit_length())      # ceil(log2 n), at least 1
    return tx, ty, tile_bits


def tile_span(r, tx, ty):
    """The inclusive range of tile columns and rows the box u +- rx, v +- ry touches, clamped to the grid."""
    t = float(TILE)
    c0 = min(max(math.floor((r["u"] - r["rx"]) / t), 0), tx - 1)
    c1 = min(max(math.floor((r["u"] + r["rx"]) / t), 0), tx - 1)
    r0 = min(max(math.floor((r["v"] - r["ry"]) / t), 0), ty - 1)
    r1 = min(max(math.floor((r["v"] + r["ry"]) / t), 0), ty - 1)
    return c0, c1, r0, r1


def radix_sort_pairs(keys, vals):
    """Stable least-significant-digit radix sort of (key, value) pairs by the 32 bits of the key: four passes of 8-bit digits,
    each a histogram, an exclusive scan and a scatter in input order."""
    keys, vals = list(keys), list(vals)
    for shift in (0, 8, 16, 24):
        counts = [0] * 256
        for k in keys:
            counts[(k >> shift) & 255] += 1
        off, acc = [0] * 256, 0
        for d in range(256):
            off[d], acc = acc, acc + counts[d]
        nk, nv = [0] * len(keys), [0] * len(vals)
        for k, v in zip(keys, vals):
            d = (k >> shift) & 255
            nk[off[d]], nv[off[d]] = k, v
            off[d] += 1
        keys, vals = nk, nv
    return keys, vals


@lru_cache(maxsize=None)
def tile_pipeline():
    """The whole binning of the tile scene: records, counts, offsets, keys, sorted order, ranges, tile lists."""
    cam = TILE_CAM
    recs = [project(g, cam) for g in TILE_SCENE]
    tx, ty, tile_bits = tile_layout(cam["W"], cam["H"])
    depth_bits = 32 - tile_bits
    counts = []
    for r in recs:
        if r is None:
            counts.append(0)
            continue
        c0, c1, r0, r1 = tile_span(r, tx, ty)
        counts.append((c1 - c0 + 1) * (r1 - r0 + 1))
    offsets, acc = [], 0
    for c in counts:
        offsets.append(acc)
        acc += c
    total = acc
    keys, vals, ids = [0] * total, [0] * total, [0] * total
    for i, r in enumerate(recs):
        if r is None:
            continue
        c0, c1, r0, r1 = tile_span(r, tx, ty)
        dq = f32_bits(r["z"]) >> (32 - depth_bits)
        pos = offsets[i]
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                keys[pos] = ((row * tx + col) << depth_bits) | dq
                vals[pos] = pos
                ids[pos] = i
                pos += 1
    skeys, svals = radix_sort_pairs(keys, vals)
    sids = [ids[v] for v in svals]
    ranges = {}
    for k in range(total):
        t = skeys[k] >> depth_bits
        if k == 0 or t != (skeys[k - 1] >> depth_bits):
            ranges.setdefault(t, [k, k])[0] = k
        if k == total - 1 or t != (skeys[k + 1] >> depth_bits):
            ranges.setdefault(t, [k, k])[1] = k + 1
    return {"recs": recs, "tx": tx, "ty": ty, "tile_bits": tile_bits, "depth_bits": depth_bits, "counts": counts, "offsets": offsets,
            "total": total, "keys": keys, "ids": ids, "skeys": skeys, "sids": sids, "ranges": ranges, "unsorted_vals": vals,
            "svals": svals}


def tile_binning():
    P = tile_pipeline()
    cam = TILE_CAM
    recs = P["recs"]
    rows = []
    for i, r in enumerate(recs):
        c0, c1, r0, r1 = tile_span(r, P["tx"], P["ty"])
        rows.append([str(i), f"({r['u']:.2f}, {r['v']:.2f})", f"{r['z']:.1f}", f"0x{f32_bits(r['z']):08x}", f"({r['rx']}, {r['ry']})",
                     f"{c0}..{c1} x {r0}..{r1}", str(P["counts"][i]), str(P["offsets"][i])])
    inst = []
    for k in range(P["total"]):
        key = P["skeys"][k]
        t = key >> P["depth_bits"]
        inst.append([str(k), f"0x{key:08x}", f"{t} (col {t % P['tx']}, row {t // P['tx']})", str(key & ((1 << P['depth_bits']) - 1)),
                     str(P["sids"][k])])
    rng = [[str(t), f"{P['ranges'].get(t, [0, 0])[0]}", f"{P['ranges'].get(t, [0, 0])[1]}", ", ".join(str(P['sids'][k]) for k in range(*P['ranges'].get(t, [0, 0]))) or "empty"]
           for t in range(P["tx"] * P["ty"])]
    # tiled against naive
    visits = 0
    worst = 0.0
    order = sorted(range(len(recs)), key=lambda i: recs[i]["z"])
    naive_recs = [recs[i] for i in order]
    for y in range(cam["H"]):
        for x in range(cam["W"]):
            t = (y // TILE) * P["tx"] + x // TILE
            a, b = P["ranges"].get(t, [0, 0])
            lst = [recs[P["sids"][k]] for k in range(a, b)]
            ct, _, _, vis = walk(lst, x + 0.5, y + 0.5)
            cn = walk(naive_recs, x + 0.5, y + 0.5)[0]
            visits += vis
            worst = max(worst, max(abs(ct[k] - cn[k]) for k in range(3)))
    bound = P["total"] * TILE * TILE
    # truncation of the depth key
    d0 = 2.0
    ulp = struct.unpack("<f", struct.pack("<I", f32_bits(d0) + 1))[0] - d0
    near = [d0, d0 + ulp, d0 + 8 * ulp, d0 + 16 * ulp]
    same = [f32_bits(v) >> (32 - P["depth_bits"]) for v in near]
    scene_rows = [[str(i), _v(g[0:3], 6), _v([math.exp(x) for x in g[3:6]], 6), _v(g[6:10], 10), _g(sigmoid(g[10]), 6), _v(g[11:14], 6)]
                  for i, g in enumerate(TILE_SCENE)]
    return (f"The scene `gaussian_splatting.TILE_SCENE`: six Gaussians, the whole of its input, given by the mean, the three scales, the "
            "quaternion (w, x, y, z; it is normalized on use), the opacity and the colour (the background is black, the Gaussians are "
            "projected by the formulas of this section, with the 0.3 dilation):\n\n"
            + _table(["i", "mean (x, y, z)", "scales", "quaternion", "opacity", "colour"], scene_rows, "rlllrl")
            + f"\n\nThey are seen by the camera at the origin (R = I) with "
            f"f = 40 and principal point (24, 16) in a 48 by 32 image. A tile is 16 by 16 pixels, so the grid is {P['tx']} by {P['ty']} = "
            f"{P['tx'] * P['ty']} tiles, tile numbers need ceil(log2 {P['tx'] * P['ty']}) = {P['tile_bits']} bits and the depth keeps the top "
            f"{P['depth_bits']} bits of its binary32 pattern. Each Gaussian covers the tiles its box u +- rx, v +- ry touches (clamped to the "
            "grid).\n\n"
            + _table(["i", "(u, v)", "depth z", "bits of z", "(rx, ry)", "tile columns x rows", "count", "scanned offset"], rows, "rrrrrrrr")
            + f"\n\nThe counts sum to {P['total']} instances. The key of an instance is the tile number shifted left by {P['depth_bits']} "
              f"or the depth's top {P['depth_bits']} bits (`bits(z) >> {32 - P['depth_bits']}`); the table lists the instances after the "
              "stable radix sort (four passes of 8-bit digits, least significant first) with the Gaussian each came from:\n\n"
            + _table(["rank", "key", "tile", "depth part", "Gaussian"], inst, "rrrrr")
            + "\n\nThe range of each tile in the sorted array (neighbour comparison: a range starts where the tile number differs "
              "from the previous key's and ends where it differs from the next one's), and the Gaussians in it, nearest first:\n\n"
            + _table(["tile", "start", "end", "Gaussians front to back"], rng, "rrrl")
            + f"\n\nCompositing every pixel from its own tile's list gives the same image as compositing all six Gaussians in depth order "
              f"at every pixel: the largest difference over the 1,536 pixels and three channels is {'0' if worst == 0.0 else f'{worst:.1e}'} "
              f"(criterion: below 1e-12). The tiled pass looked at {visits:,} (pixel, instance) pairs, against the bound "
              f"{P['total']} x 256 = {bound:,} (every pixel of a tile looks at every instance of it) and the {6 * cam['W'] * cam['H']:,} of "
              "the untiled walk (every Gaussian at every pixel)"
              + ("; no walk stopped early in this scene.\n\n" if visits == bound else ".\n\n")
            + f"Truncating the depth: the binary32 patterns of 2.0 and of 2.0 plus 1, 8 and 16 units in the last place keep the depth "
              f"parts {same} out of 2^{P['depth_bits']}; the last {32 - P['depth_bits']} bits are dropped, so two depths that differ only in "
              "those bits (a bucket 8 units in the last place wide, under 1e-6 of the depth) tie, and a tie keeps the order in which the "
              "instances were emitted, because the sort is stable.")


# ------------------------------------------------------------------------------------------------ the backward pass

# d (column k of R(q)) / d (w, x, y, z) for a unit quaternion, as [component][w x y z]; read off the entries of the rotation matrix
def _axis_jacobian(q, k):
    w, x, y, z = q
    if k == 0:
        return [[0.0, 0.0, -4 * y, -4 * z], [2 * z, 2 * y, 2 * x, 2 * w], [-2 * y, 2 * z, -2 * w, 2 * x]]
    if k == 1:
        return [[-2 * z, 2 * y, 2 * x, -2 * w], [0.0, -4 * x, 0.0, -4 * z], [2 * x, 2 * w, 2 * z, 2 * y]]
    return [[2 * y, 2 * z, 2 * w, 2 * x], [-2 * x, -2 * w, 2 * z, 2 * y], [0.0, -4 * x, -4 * y, 0.0]]


def pixel_backward(recs, px, py, gcol, bg=BG0):
    """Gradients of  gcol . colour(pixel)  with respect to what each record contributes, walking the pixel's own steps in reverse.

    dC/dalpha_i = T_i c_i - B_i / (1 - alpha_i), with B_i the light that reaches the pixel from behind i (the later Gaussians
    and the background, each already multiplied by the transmittance in front of it). Returns {record index: gradients} with
    gradients du, dv, dLam (2 by 2, symmetric), dopacity, dcolour (3).
    """
    col, T, steps, _ = walk(recs, px, py, bg)
    B = [T * bg[k] for k in range(3)]
    out = {}
    for (i, a, Tb, sigma) in reversed(steps):
        r = recs[i]
        c = r["c"]
        v_alpha = sum(gcol[k] * (Tb * c[k] - B[k] / (1.0 - a)) for k in range(3))
        e = math.exp(-sigma)
        capped = r["op"] * e >= ALPHA_MAX
        v_sigma = 0.0 if capped else -a * v_alpha
        v_op = 0.0 if capped else e * v_alpha
        dx, dy = r["u"] - px, r["v"] - py
        g = out.setdefault(i, {"u": 0.0, "v": 0.0, "Lam": [[0.0, 0.0], [0.0, 0.0]], "op": 0.0, "c": [0.0, 0.0, 0.0]})
        g["u"] += v_sigma * (r["La"] * dx + r["Lb"] * dy)
        g["v"] += v_sigma * (r["Lb"] * dx + r["Lc"] * dy)
        g["Lam"][0][0] += v_sigma * 0.5 * dx * dx
        g["Lam"][0][1] += v_sigma * 0.5 * dx * dy
        g["Lam"][1][0] += v_sigma * 0.5 * dx * dy
        g["Lam"][1][1] += v_sigma * 0.5 * dy * dy
        g["op"] += v_op
        for k in range(3):
            g["c"][k] += a * Tb * gcol[k]
            B[k] += c[k] * a * Tb
    return out


def projection_backward(p, cam, r, g):
    """Carry the gradients of one Gaussian's record (du, dv, dLam, dopacity, dcolour) back to its 14 raw numbers."""
    fx, fy = cam["fx"], cam["fy"]
    x, y, z = r["m"]
    Lam = conic_matrix(r)
    # Lam = (S2 + eps I)^-1:  dL/dS2 = -Lam G Lam
    GS2 = [[-v for v in row] for row in mm(mm(Lam, g["Lam"]), Lam)]
    T, S3, J = r["T"], r["S3"], r["J"]
    GS3 = mm(mm(tr(T), GS2), T)                                  # S2 = T S3 T^T
    GT = [[2.0 * v for v in row] for row in mm(mm(GS2, T), S3)]
    GJ = mm(GT, tr(cam["R"]))                                    # T = J R
    z2, z3 = z * z, z * z * z
    Gm = [GJ[0][2] * (-fx / z2) + g["u"] * fx / z,
          GJ[1][2] * (-fy / z2) + g["v"] * fy / z,
          GJ[0][0] * (-fx / z2) + GJ[0][2] * (2 * fx * x / z3) + GJ[1][1] * (-fy / z2) + GJ[1][2] * (2 * fy * y / z3)
          + g["u"] * (-fx * x / z2) + g["v"] * (-fy * y / z2)]
    gmu = mv(tr(cam["R"]), Gm)                                   # m = R mu + t
    s, R3 = r["s"], r["R3"]
    gls, ga = [], []
    for k in range(3):
        a = [R3[i][k] for i in range(3)]
        Ga = mv(GS3, a)
        gls.append(2.0 * s[k] * s[k] * dot(a, Ga))               # d / d log s_k = s_k * d / d s_k
        ga.append([2.0 * s[k] * s[k] * v for v in Ga])
    q = p[6:10]
    qn = math.sqrt(sum(v * v for v in q))
    qh = [v / qn for v in q]
    gqh = [0.0] * 4
    for k in range(3):
        Jk = _axis_jacobian(qh, k)
        for i in range(3):
            for j in range(4):
                gqh[j] += Jk[i][j] * ga[k][i]
    proj = sum(qh[j] * gqh[j] for j in range(4))
    gq = [(gqh[j] - qh[j] * proj) / qn for j in range(4)]
    glo = r["op"] * (1.0 - r["op"]) * g["op"]
    return list(gmu) + gls + gq + [glo] + list(g["c"])


def view_gradient(params, cam, upstream, bg=BG0):
    """Sum over the pixels of one view of  upstream(x, y, colour) . colour , and its gradient with respect to every Gaussian's 14
    numbers. `upstream(x, y, colour)` returns (loss contribution, gradient with respect to the colour)."""
    recs, order = sorted_records(params, cam)
    acc = {}
    loss = 0.0
    for y in range(cam["H"]):
        for x in range(cam["W"]):
            col = walk(recs, x + 0.5, y + 0.5, bg)[0]
            l, gcol = upstream(x, y, col)
            loss += l
            for i, g in pixel_backward(recs, x + 0.5, y + 0.5, gcol, bg).items():
                a = acc.setdefault(i, {"u": 0.0, "v": 0.0, "Lam": [[0.0, 0.0], [0.0, 0.0]], "op": 0.0, "c": [0.0, 0.0, 0.0]})
                a["u"] += g["u"]
                a["v"] += g["v"]
                a["op"] += g["op"]
                for k in range(3):
                    a["c"][k] += g["c"][k]
                for r_ in range(2):
                    for c_ in range(2):
                        a["Lam"][r_][c_] += g["Lam"][r_][c_]
    grads = [[0.0] * 14 for _ in params]
    for i, g in acc.items():
        gi = order[i]
        grads[gi] = projection_backward(params[gi], cam, recs[i], g)
    return loss, grads


def render_structure(params, cam, bg=BG0):
    """What the render is made of apart from its smooth values: the depth order of the Gaussians, and for every pixel which pairs
    composited, which of them sat at the cap and where the walk ended. A central difference is valid when the structure is the same
    at theta + h and theta - h."""
    recs, order = sorted_records(params, cam)
    sig = [tuple(order)]
    for y in range(cam["H"]):
        for x in range(cam["W"]):
            px, py = x + 0.5, y + 0.5
            _, _, steps, visits = walk(recs, px, py, bg)
            capped = tuple(recs[i]["op"] * math.exp(-sg) >= ALPHA_MAX for i, _, _, sg in steps)
            sig.append((tuple(i for i, _, _, _ in steps), capped, visits))
    return tuple(sig)


def pair_margins(params, cam, bg=BG0):
    """The distances to the discontinuities of the render: how close any contributing alpha is to 1/255 and to the cap 0.99, how close
    any transmittance comes to the stop, and the smallest gap between neighbouring depths. A central difference is meaningful only
    when these exceed the step."""
    recs, _ = sorted_records(params, cam)
    m_lo, m_hi, m_stop = 1e9, 1e9, 1e9
    for y in range(cam["H"]):
        for x in range(cam["W"]):
            px, py = x + 0.5, y + 0.5
            T = 1.0
            for r in recs:
                dx, dy = r["u"] - px, r["v"] - py
                sg = 0.5 * (r["La"] * dx * dx + r["Lc"] * dy * dy) + r["Lb"] * dx * dy
                raw = r["op"] * math.exp(-sg)
                m_lo = min(m_lo, abs(raw - ALPHA_MIN))
                m_hi = min(m_hi, abs(raw - ALPHA_MAX))
                a = min(ALPHA_MAX, raw)
                if a < ALPHA_MIN:
                    continue
                nT = T * (1.0 - a)
                m_stop = min(m_stop, abs(nT - T_STOP))
                if nT <= T_STOP:
                    break
                T = nT
    zs = sorted(r["z"] for r in recs)
    return m_lo, m_hi, m_stop, min(b - a for a, b in zip(zs, zs[1:]))


# ------------------------------------------------------------------------------------------------ the gradient check

GC_CAM_ARGS = ((0.3, -0.2, -1.0), (0.0, 0.0, 3.0), 9.0, 8, 8)
GC_BG = (0.1, 0.15, 0.2)
GC_NAME = "gaussian_splatting.gradient_check"


def _gc_scene():
    return [
        gaussian([0.35, -0.25, 3.0], [0.45, 0.30, 0.20], [0.9, 0.3, -0.2, 0.25], 0.7, [0.9, 0.3, 0.2]),
        gaussian([-0.4, 0.3, 3.8], [0.40, 0.50, 0.30], [0.7, -0.4, 0.5, 0.1], 0.6, [0.2, 0.8, 0.3]),
        gaussian([0.0, 0.1, 5.0], [0.9, 0.7, 0.5], [0.8, 0.1, 0.1, -0.5], 0.5, [0.3, 0.3, 0.9]),
    ]


GROUPS = (("mean", 0, 3), ("log scale", 3, 6), ("quaternion", 6, 10), ("opacity logit", 10, 11), ("colour", 11, 14))


@lru_cache(maxsize=None)
def _gc_result():
    cam = make_camera(*GC_CAM_ARGS)
    scene = _gc_scene()
    rng = bookrng.generator(GC_NAME)
    Wt = [[[rng.normal() for _ in range(3)] for _ in range(cam["W"])] for _ in range(cam["H"])]

    def up(x, y, col):
        w = Wt[y][x]
        return sum(a * b for a, b in zip(w, col)), w

    loss, grads = view_gradient(scene, cam, up, GC_BG)
    h = 1e-6
    errs = []
    base = render_structure(scene, cam, GC_BG)
    same = True
    for gi in range(len(scene)):
        row = []
        for k in range(14):
            sp = [list(p) for p in scene]
            sm = [list(p) for p in scene]
            sp[gi][k] += h
            sm[gi][k] -= h
            same = same and render_structure(sp, cam, GC_BG) == base and render_structure(sm, cam, GC_BG) == base
            num = (view_gradient(sp, cam, up, GC_BG)[0] - view_gradient(sm, cam, up, GC_BG)[0]) / (2 * h)
            row.append(abs(grads[gi][k] - num) / max(abs(grads[gi][k]), 1.0))
        errs.append(row)
    margins = pair_margins(scene, cam, GC_BG)
    return cam, scene, loss, grads, errs, margins, same


def gradient_check():
    cam, scene, loss, grads, errs, margins, same = _gc_result()
    worst = max(max(r) for r in errs)
    rows = []
    for gi in range(3):
        for name, a, b in GROUPS:
            rows.append([f"{gi}", name, ", ".join(_g(grads[gi][k], 6) for k in range(a, b)),
                         "below 1e-6" if max(errs[gi][k] for k in range(a, b)) < 1e-6 else "FAILS"])
    qdot = max(abs(dot(scene[gi][6:10], grads[gi][6:10])) for gi in range(3))
    scene_rows = [[str(i), _v(g[0:3], 4), _v([math.exp(x) for x in g[3:6]], 4), _v(g[6:10], 4), _g(sigmoid(g[10]), 4), _v(g[11:14], 4)]
                  for i, g in enumerate(scene)]
    return (f"A frame of 8 by 8 pixels from the camera at (0.3, -0.2, -1) looking at (0, 0, 3) with up = (0, -1, 0) and f = 9 "
            f"(`make_camera`, the `look_at` of Chapter 3; principal point (4, 4)), background {_v(GC_BG, 3)}, and three Gaussians, the "
            "whole scene (quaternions are normalized on use):\n\n"
            + _table(["i", "mean", "scales", "quaternion (w, x, y, z)", "opacity", "colour"], scene_rows, "rlllrl")
            + f"\n\nThe loss is L = sum over pixels and channels of W C, with 192 weights W drawn in "
            f"order (pixel row, pixel column, channel) from `normal()` of the generator `{GC_NAME}`, seed {bookrng.seed_of(GC_NAME)}, so "
            f"dL/dC = W exactly. L = {_g(loss, 8)}. The analytic gradient is compared with the central difference "
            "(L(theta + h) - L(theta - h)) / 2h, h = 1e-6, for each of the 42 numbers; an entry passes when "
            "|analytic - difference| <= 1e-6 max(|analytic|, 1).\n\n"
            + _table(["Gaussian", "parameters", "analytic gradient", "against central differences"], rows, "llrl")
            + f"\n\nAll 42 entries {'pass' if worst < 1e-6 else 'FAIL'}. The raw quaternion only enters through its direction, so its gradient "
            f"is orthogonal to it: |q . dL/dq| is {'below 1e-12' if qdot < 1e-12 else 'NOT below 1e-12'} for all three. The frame stays away from every "
            f"discontinuity of the render: the nearest pair is {margins[0]:.1e} from the 1/255 cut-off in alpha, the nearest to the cap "
            f"0.99 is {margins[1]:.1f} away, the nearest transmittance to the stop {margins[2]:.1e} away and neighbouring depths "
            f"{margins[3]:.2f} apart, and for each of the 84 displaced scenes the depth order, the pairs composited at every pixel, "
            f"the capped pairs and the end of every walk are {'the same as' if same else 'NOT the same as'} in the undisplaced one.")


def cutoff_jump():
    cam = make_camera((0, 0, 0), (0, 0, 1), 10.0, 1, 1)

    def make(x0):
        return [gaussian([x0, 0.0, 4.0], [0.4, 0.4, 0.4], [1, 0, 0, 0], 0.5, [1.0, 0.5, 0.25])]

    def red(x0):
        return render(make(x0), cam)[0][0][0]

    def alpha(x0):
        recs, _ = sorted_records(make(x0), cam)
        if not recs:
            return 0.0
        r = recs[0]
        dx, dy = r["u"] - 0.5, r["v"] - 0.5
        return r["op"] * math.exp(-(0.5 * (r["La"] * dx * dx + r["Lc"] * dy * dy) + r["Lb"] * dx * dy))

    lo, hi = 0.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if alpha(mid) > ALPHA_MIN * (1.0 + 1e-3):
            lo = mid
        else:
            hi = mid
    x0 = lo
    _, gr = view_gradient(make(x0), cam, lambda x, y, c: (c[0], [1.0, 0.0, 0.0]))
    an = gr[0][0]
    rows = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        fd = (red(x0 + h) - red(x0 - h)) / (2 * h)
        e = abs(fd - an) / abs(an)
        rows.append([f"{h:.0e}", _g(fd, 4) if e >= 1e-6 else _g(an, 4), f"{e:.3g}" if e >= 1e-6 else "below 1e-6"])
    return (f"One pixel (a 1 by 1 image, f = 10, camera at the origin looking along +z, so R = I, with the principal point at the "
            "pixel's own centre (0.5, 0.5) and the pixel sampled at that centre), one Gaussian of opacity 0.5, isotropic scale 0.4 "
            "(quaternion (1, 0, 0, 0)), at (x, 0, 4), colour (1, 0.5, 0.25), background black. Its screen covariance is the full "
            "J Sigma J^T of the projection section, with the z column of J (the mean moves in x, so J changes with it), plus the 0.3 "
            "dilation, and the alpha is opacity times exp(-kappa) at the pixel centre. Its mean x is moved until the alpha at the pixel is 1.001 times 1/255 "
            f"(bisection of 200 halvings on [0, 6]; x = {_g(x0, 8)}). The derivative of the red channel with respect to the mean x is "
            f"{_g(an, 5)} analytically, and the table shows the central difference of the render for five steps h.\n\n"
            + _table(["step h", "central difference", "relative error"], rows)
            + "\n\nThe render jumps by 1/255 times the colour where alpha crosses the cut-off; a step that straddles it adds that jump "
              "divided by 2h to the difference, an error that grows as the step shrinks until the step is smaller than the distance "
              "to the cut-off.")


# ------------------------------------------------------------------------------------------------ fitting a scene to renders

FIT_W = FIT_H = 16
FIT_F = 18.0
FIT_BG = (0.05, 0.05, 0.08)
FIT_STEPS = 300
FIT_RECORD = (0, 10, 50, 100, 200, 300)
FIT_NAME = "gaussian_splatting.fit_start"
# learning rates per parameter: mean 0.02, log scale 0.03, quaternion 0.01, opacity logit 0.1, colour 0.03
FIT_LR = [0.02] * 3 + [0.03] * 3 + [0.01] * 4 + [0.1] + [0.03] * 3
FIT_TARGET = [
    gaussian([0.5, 0.1, 0.0], [0.45, 0.30, 0.25], [0.9, 0.2, -0.3, 0.2], 0.9, [0.9, 0.3, 0.2]),
    gaussian([-0.5, -0.2, 0.3], [0.35, 0.40, 0.30], [0.8, -0.3, 0.4, 0.1], 0.8, [0.2, 0.8, 0.3]),
    gaussian([0.0, 0.4, -0.5], [0.30, 0.30, 0.40], [0.7, 0.3, 0.1, -0.5], 0.85, [0.3, 0.4, 0.9]),
]


def fit_cameras():
    """Four training cameras on a circle of radius 4 about the origin at azimuth 0, 90, 180 and 270 degrees, heights -1.2, -1.5, -1.0,
    -1.7 (y points down, so these are above the scene), and one held-out camera at (2.8, -2.6, -2.6); all look at the origin."""
    train = []
    for deg, h in ((0, -1.2), (90, -1.5), (180, -1.0), (270, -1.7)):
        a = math.radians(deg)
        train.append(make_camera((4.0 * math.sin(a), h, -4.0 * math.cos(a)), (0.0, 0.0, 0.0), FIT_F, FIT_W, FIT_H))
    return train, make_camera((2.8, -2.6, -2.6), (0.0, 0.0, 0.0), FIT_F, FIT_W, FIT_H)


def fit_start():
    """The target with every Gaussian disturbed: 3 mean numbers and 3 log scales and 4 quaternion numbers each get 0.15, 0.2 and 0.15
    times a draw of `normal()`, in the order Gaussian, parameter; the opacity logit is set to 0 (opacity 0.5) and the colour to
    (0.5, 0.5, 0.5)."""
    rng = bookrng.generator(FIT_NAME)
    out = []
    for p in FIT_TARGET:
        q = list(p)
        for k in range(0, 3):
            q[k] += 0.15 * rng.normal()
        for k in range(3, 6):
            q[k] += 0.2 * rng.normal()
        for k in range(6, 10):
            q[k] += 0.15 * rng.normal()
        q[10] = 0.0
        q[11:14] = [0.5, 0.5, 0.5]
        out.append(q)
    return out


def _mse_upstream(img):
    P = len(img) * len(img[0])

    def up(x, y, col):
        d = [col[k] - img[y][x][k] for k in range(3)]
        return sum(v * v for v in d) / (3 * P), [2 * v / (3 * P) for v in d]

    return up


def _mse(a, b):
    return sum((x[k] - y[k]) ** 2 for ra, rb in zip(a, b) for x, y in zip(ra, rb) for k in range(3)) / (3 * len(a) * len(a[0]))


def adam_fit(nudge=0.0, steps=FIT_STEPS, record=FIT_RECORD, trace=False):
    """Full-batch Adam (beta1 0.9, beta2 0.999, epsilon 1e-8, bias correction) on the 42 numbers; the loss of an update is the mean over
    the four training views of the mean squared error against the target's own renders. `nudge` is added to the first mean number of
    the starting scene. Returns {step: (training loss, held-out error)} and the final scene."""
    train, held = fit_cameras()
    tgt = [render(FIT_TARGET, c, FIT_BG) for c in train]
    htgt = render(FIT_TARGET, held, FIT_BG)
    p = [list(x) for x in fit_start()]
    p[0][0] += nudge
    m = [[0.0] * 14 for _ in p]
    v = [[0.0] * 14 for _ in p]
    seen = {}
    curve = []
    for t in range(steps + 1):
        loss = 0.0
        G = [[0.0] * 14 for _ in p]
        for c, im_ in zip(train, tgt):
            l, gr = view_gradient(p, c, _mse_upstream(im_), FIT_BG)
            loss += l / len(train)
            for i in range(len(p)):
                for k in range(14):
                    G[i][k] += gr[i][k] / len(train)
        curve.append(loss)
        if t in record:
            seen[t] = (loss, _mse(render(p, held, FIT_BG), htgt))
        if t == steps:
            break
        for i in range(len(p)):
            for k in range(14):
                m[i][k] = 0.9 * m[i][k] + 0.1 * G[i][k]
                v[i][k] = 0.999 * v[i][k] + 0.001 * G[i][k] * G[i][k]
                p[i][k] -= FIT_LR[k] * (m[i][k] / (1 - 0.9 ** (t + 1))) / (math.sqrt(v[i][k] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return (seen, p, curve) if trace else (seen, p)


def _fit_task(nudge):
    return adam_fit(nudge, trace=True)


@lru_cache(maxsize=None)
def fit_runs():
    nudges = (0.0, *stab.NUDGES)
    with multiprocessing.get_context("fork").Pool(min(4, len(nudges))) as pool:
        res = pool.map(_fit_task, nudges)
    return dict(zip(nudges, res))


def _fit_text(f, full):
    runs = fit_runs()
    v = f(runs[0.0])
    ws = [f(runs[n]) for n in stab.NUDGES]
    return stab.text(v, ws, full)


def fit_toy():
    runs = fit_runs()
    rows = []
    for t in FIT_RECORD:
        rows.append([str(t), _fit_text(lambda r, t=t: r[0][t][0], lambda x: f"{x:.4e}"), _fit_text(lambda r, t=t: r[0][t][1], lambda x: f"{x:.4e}")])
    final = runs[0.0][1]
    dmu = [norm(sub(final[i][0:3], FIT_TARGET[i][0:3])) for i in range(3)]
    dtext = [_fit_text(lambda r, i=i: norm(sub(r[1][i][0:3], FIT_TARGET[i][0:3])), lambda x: f"{x:.3e}") for i in range(3)]
    opac = [_fit_text(lambda r, i=i: sigmoid(r[1][i][10]), lambda x: f"{x:.4f}") for i in range(3)]
    train, held = fit_cameras()
    s0 = fit_start()
    tgt_rows = [[str(i), _v(g[0:3], 4), _v([math.exp(x) for x in g[3:6]], 4), _v(g[6:10], 4), _g(sigmoid(g[10]), 4), _v(g[11:14], 4)]
                for i, g in enumerate(FIT_TARGET)]
    return ("The target is three Gaussians, the whole of it (quaternions normalized on use):\n\n"
            + _table(["i", "mean", "scales", "quaternion (w, x, y, z)", "opacity", "colour"], tgt_rows, "rlllrl")
            + f"\n\nIt is rendered at {FIT_W} by "
            f"{FIT_H} pixels, f = {FIT_F:g}, background {_v(FIT_BG, 3)}, by four training cameras at (4 sin a, h, -4 cos a) for the azimuths a = 0, 90, "
            "180, 270 degrees with the heights h = -1.2, -1.5, -1.0, -1.7, and one held-out camera at (2.8, -2.6, -2.6), all looking at "
            "the origin with up = (0, -1, 0) (`make_camera`, principal point at the middle of the image). "
            f"The start disturbs the target: for each Gaussian in turn, 3 mean numbers by 0.15 z, 3 log scales by 0.2 z and 4 quaternion "
            f"numbers by 0.15 z, with z the next `normal()` of the generator `{FIT_NAME}` (seed {bookrng.seed_of(FIT_NAME)}), in that order, "
            "then every opacity is set to 0.5 and every colour to (0.5, 0.5, 0.5). Full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = "
            f"1e-8, bias correction) with the rates mean 0.02, log scale 0.03, quaternion 0.01, opacity logit 0.1, colour 0.03; the loss of an "
            f"update is the mean over the four training views of the mean squared error over pixels and channels; {FIT_STEPS} updates. The "
            "held-out column is the same error on the fifth camera, which no update saw. Under the stable-digits rule the run is repeated with the first "
            "mean number of the start changed by +1e-12, -1e-12, +3e-12 and -3e-12, and a number is printed with only the digits that "
            "are the same in the run and in the four copies: in full if all agree on it, as 'about x' with the leading digits on which "
            "all agree, or as 'unstable' when not even one digit survives.\n\n"
            + _table(["updates", "training loss", "held-out error"], rows, "rrr")
            + f"\n\nAfter {FIT_STEPS} updates the distance of each fitted mean from the target's is {dtext[0]}, {dtext[1]} and {dtext[2]} "
              f"(0.15 z draws started them at {', '.join(f'{norm(sub(s0[i][0:3], FIT_TARGET[i][0:3])):.4f}' for i in range(3))}), and the fitted "
              f"opacities are {', '.join(opac)} against 0.9, 0.8 and 0.85.")


# ------------------------------------------------------------------------------------------------ density control

DENSIFY_FRAC = 0.25
PRUNE_OPACITY = 0.02
SPLIT_SHRINK = 1.6
DENSE_ROWS = [      # (largest scale, the other two scales, opacity, accumulated gradient magnitude)
    (0.08, (0.08, 0.08), 0.80, 0.20),
    (0.10, (0.10, 0.06), 0.90, 0.90),
    (0.12, (0.12, 0.12), 0.70, 0.10),
    (0.14, (0.10, 0.10), 0.01, 0.05),
    (0.16, (0.16, 0.12), 0.85, 0.30),
    (0.18, (0.18, 0.18), 0.60, 0.70),
    (0.30, (0.30, 0.10), 0.75, 1.20),
    (0.50, (0.20, 0.20), 0.65, 0.40),
]


def densify_step(rows, frac=DENSIFY_FRAC, prune=PRUNE_OPACITY):
    """One round of the threshold-free form of the original rule. `rows` are (largest scale, other scales, opacity, gradient); the
    largest axis of every Gaussian is its x axis. Returns the new list of (origin, kind, x offset, scale factor, opacity) and the
    size threshold `big`.

    want = floor(frac N) Gaussians with the largest gradient are refined; a refined one whose largest scale is at least `big`, the
    value at index floor(0.8 N) of the sorted largest scales, is split into two at +-0.5 s along its axis with all scales divided by
    1.6; a refined smaller one is cloned (two identical copies); a Gaussian with opacity below `prune` is dropped.
    """
    n = len(rows)
    want = int(n * frac)
    order = sorted(range(n), key=lambda i: -rows[i][3])
    chosen = set(order[:want])
    sizes = sorted(r[0] for r in rows)
    big = sizes[int(n * 0.8) % n]
    out = []
    for i, (s, _, o, _) in enumerate(rows):
        if o < prune:
            out.append((i, "pruned", 0.0, 0.0, o))
        elif i in chosen and s >= big:
            out.append((i, "split", +0.5 * s, 1.0 / SPLIT_SHRINK, o))
            out.append((i, "split", -0.5 * s, 1.0 / SPLIT_SHRINK, o))
        elif i in chosen:
            out.append((i, "clone", 0.0, 1.0, o))
            out.append((i, "clone", 0.0, 1.0, o))
        else:
            out.append((i, "kept", 0.0, 1.0, o))
    return out, big


def densify_rules():
    out, big = densify_step(DENSE_ROWS)
    rows = []
    for i, (s, others, o, g) in enumerate(DENSE_ROWS):
        rows.append([str(i), _g(s), f"({others[0]:g}, {others[1]:g})", _g(o), _g(g), "pruned" if o < PRUNE_OPACITY else
                     ("split" if any(k == "split" and a == i for a, k, *_ in out) else "cloned" if any(k == "clone" and a == i for a, k, *_ in out) else "kept")])
    new = [[str(j), str(a), kind, _g(off, 3), _g(sc, 3), _g(o)] for j, (a, kind, off, sc, o) in enumerate(x for x in out if x[1] != "pruned")]
    # the pair of children against the parent: variance along the axis and across it, in units of the parent's variance
    var_along = (1.0 / SPLIT_SHRINK) ** 2 + 0.25
    var_across = (1.0 / SPLIT_SHRINK) ** 2
    return ("Eight Gaussians whose largest axis is their x axis, with the scales, opacities and accumulated gradient magnitudes of the "
            "table (the gradient column is what the backward pass accumulated). One round with refinement fraction 0.25, pruning below "
            f"opacity 0.02 and no cap on the population: the {int(8 * DENSIFY_FRAC)} Gaussians with the largest gradient are refined; the "
            f"largest scales sorted are {', '.join(_g(v) for v in sorted(r[0] for r in DENSE_ROWS))}, and the size threshold is the entry at index "
            f"floor(0.8 * 8) = 6, which is {_g(big)}.\n\n"
            + _table(["id", "largest scale", "other scales", "opacity", "gradient", "outcome"], rows, "rrrrrl")
            + f"\n\nThe new population has {len(new)} Gaussians (8 - 1 pruned + 1 for the split + 1 for the clone), in order:\n\n"
            + _table(["new id", "from", "kind", "x offset", "scale factor", "opacity"], new, "rrlrrr")
            + f"\n\nA split replaces a Gaussian of variance 1 (in units of its own, along its axis) by two of variance (1/1.6)^2 = "
              f"{var_across:.4f} whose centres lie half the old scale to each side of the old centre: the pair has variance {var_across:.4f} + 0.25 = "
              f"{var_along:.4f} along the axis and {var_across:.4f} across it, so it covers a smaller region than the parent did; the "
              "optimizer is expected to finish the job.")


def target_population(start, cap, rnd, rounds):
    """The population a budgeted round aims for: a smoothstep from `start` to `cap` that is reached after ceil(2 rounds / 3) rounds and
    held. t = min(1, (rnd + 1) / ceil(2 rounds / 3)); the population is start + (cap - start) t^2 (3 - 2 t), truncated."""
    full = max(1, math.ceil(rounds * 2.0 / 3.0))
    t = min(1.0, (rnd + 1) / full)
    return int(start + (cap - min(start, cap)) * t * t * (3.0 - 2.0 * t))


def population_schedule():
    start, cap, rounds = 1000, 10000, 20
    rows = [[str(r), f"{min(1.0, (r + 1) / math.ceil(rounds * 2.0 / 3.0)):.4f}", f"{target_population(start, cap, r, rounds):,}"]
            for r in (0, 2, 5, 9, 12, 13, 14, 19)]
    return (f"A budgeted fit that starts with {start:,} Gaussians, may hold {cap:,} and runs {rounds} density rounds reaches the cap after "
            f"ceil(2 * {rounds} / 3) = {math.ceil(rounds * 2.0 / 3.0)} rounds. With t = min(1, (round + 1) / {math.ceil(rounds * 2.0 / 3.0)}) the aim is "
            "the smoothstep start + (cap - start) t^2 (3 - 2t), truncated to a whole number (rounds are numbered from 0).\n\n"
            + _table(["round", "t", "population aimed for"], rows, "rrr"))


def _trapz(f, a, b, n):
    h = (b - a) / n
    return h * (0.5 * f(a) + sum(f(a + i * h) for i in range(1, n)) + 0.5 * f(b))


def relocation_correction(o, n):
    """Opacity o' and scale factor c such that n Gaussians of opacity o' and scale c sigma composite, along a line through their common
    centre, to the same total alpha integral as one Gaussian of opacity o and scale sigma. o' makes the peak alpha equal: 1 - (1 - o')^n = o;
    c = o / sum_{k=1..n} (-1)^(k+1) C(n, k) o'^k / sqrt(k)."""
    op = 1.0 - (1.0 - o) ** (1.0 / n)
    denom = sum((-1) ** (k + 1) * math.comb(n, k) * op ** k / math.sqrt(k) for k in range(1, n + 1))
    return op, o / denom


def split_clone_relocate():
    o = 0.8
    sh = 1.0 / SPLIT_SHRINK

    def prof_parent(x):
        return o * math.exp(-0.5 * x * x)

    def two(a, b):
        return lambda x: 1.0 - (1.0 - a(x)) * (1.0 - b(x))

    def g(op, mu, sd):
        return lambda x: op * math.exp(-0.5 * ((x - mu) / sd) ** 2)

    o2, c2 = relocation_correction(o, 2)
    profiles = [
        ("the parent, opacity 0.8, sigma 1", prof_parent),
        ("split: two children, opacity 0.8, sigma 1/1.6, centres at -0.5 and +0.5", two(g(o, -0.5, sh), g(o, 0.5, sh))),
        ("clone: two copies of the parent", two(g(o, 0.0, 1.0), g(o, 0.0, 1.0))),
        (f"relocated pair: opacity {o2:.5f}, sigma {c2:.5f} each", two(g(o2, 0.0, c2), g(o2, 0.0, c2))),
    ]
    base = _trapz(prof_parent, -12.0, 12.0, 24000)
    rows = []
    for name, f in profiles:
        area = _trapz(f, -12.0, 12.0, 24000)
        peak = max(f(i * 0.001) for i in range(-3000, 3001))
        rows.append([name, f"{peak:.5f}", f"{area:.5f}", f"{area / base:.5f}"])
    corr = []
    for n in (2, 3, 4, 5):
        on, cn = relocation_correction(o, n)
        corr.append([str(n), f"{on:.5f}", f"{cn:.5f}"])
    return ("The alpha along a line through the common centre of a Gaussian's pixels, in units of its sigma: parent opacity 0.8. The area is the "
            "integral over x in [-12, 12] by the trapezoid rule with 24,000 steps; the peak is the largest value on a grid of step 0.001 on "
            "[-3, 3]. Two Gaussians at the same depth composite to 1 - (1 - a_1)(1 - a_2).\n\n"
            + _table(["profile", "peak alpha", "area", "area / parent's"], rows, "lrrr")
            + "\n\nThe correction for n Gaussians sharing one site, opacity 0.8: o' = 1 - (1 - 0.8)^(1/n) and c = 0.8 / "
              "sum_{k=1..n} (-1)^(k+1) C(n, k) o'^k / sqrt(k), applied to all n.\n\n"
            + _table(["n", "opacity o'", "scale factor c"], corr, "rrr"))


# ------------------------------------------------------------------------------------------------ evaluating along rays

def ray_response(m, Lam, d):
    """The response of a Gaussian (mean m, precision Lam, both in the camera frame) along the ray through the origin with direction d: the
    largest value of exp(-q/2) along the ray, at t* = d^T Lam m / d^T Lam d, with q = e^T Lam e at e = m - t* d. Returns (exp(-q/2), t*)."""
    Ld = mv(Lam, d)
    t = dot(Ld, m) / dot(Ld, d)
    e = [m[i] - t * d[i] for i in range(3)]
    q = dot(e, mv(Lam, e))
    return math.exp(-0.5 * q), t


def ewa_response(m, sigma, f, u, v):
    """The linearized footprint of an isotropic Gaussian of standard deviation `sigma` at camera-frame mean m, evaluated at pixel offset
    (u, v) from the principal point, with no dilation: exp(-sigma_q) for S2 = J (sigma^2 I) J^T."""
    x, y, z = m
    J = [[f / z, 0.0, -f * x / (z * z)], [0.0, f / z, -f * y / (z * z)]]
    S2 = [[sigma * sigma * dot(J[i], J[j]) for j in range(2)] for i in range(2)]
    det = S2[0][0] * S2[1][1] - S2[0][1] ** 2
    dx, dy = u - f * x / z, v - f * y / z
    q = (S2[1][1] * dx * dx - 2 * S2[0][1] * dx * dy + S2[0][0] * dy * dy) / det
    return math.exp(-0.5 * q)


def unscented_trace(m, sigma, f):
    """Trace of the 2 by 2 covariance of the six sigma points m +- sqrt(3) sigma e_k, projected by the pinhole map and taken about the
    projection of m (the bound used for tiles under any lens)."""
    c0 = (f * m[0] / m[2], f * m[1] / m[2])
    acc = 0.0
    for k in range(3):
        for sgn in (1.0, -1.0):
            x = list(m)
            x[k] += sgn * math.sqrt(3.0) * sigma
            du, dv = f * x[0] / x[2] - c0[0], f * x[1] / x[2] - c0[1]
            acc += (du * du + dv * dv) / 6.0
    return acc


def ewa_trace(m, sigma, f):
    x, y, z = m
    J = [[f / z, 0.0, -f * x / (z * z)], [0.0, f / z, -f * y / (z * z)]]
    return sigma * sigma * (dot(J[0], J[0]) + dot(J[1], J[1]))


def _ewa_vs_ray_row(z, sigma=0.5, f=100.0, dirs=(0.25, 0.1)):
    m = [dirs[0] * z, dirs[1] * z, z]
    Lam = [[1.0 / (sigma * sigma) if i == j else 0.0 for j in range(3)] for i in range(3)]
    spx = f * sigma / z
    half = int(min(3.5 * spx, 150))
    cu, cv = f * m[0] / m[2], f * m[1] / m[2]
    worst, at = 0.0, (0, 0)
    for du in range(-half, half + 1):
        for dv in range(-half, half + 1):
            u, v = math.floor(cu) + du + 0.5, math.floor(cv) + dv + 0.5
            ray = ray_response(m, Lam, [u / f, v / f, 1.0])[0]
            ew = ewa_response(m, sigma, f, u, v)
            if abs(ray - ew) > worst:
                worst, at = abs(ray - ew), (du, dv)
    return z, sigma / z, spx, worst, at, unscented_trace(m, sigma, f) / ewa_trace(m, sigma, f)


def _ewa_vs_ray_curve():
    return [_ewa_vs_ray_row(z) for z in (24.0, 16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0)]


def ewa_vs_ray():
    rows = []
    for z in (16.0, 8.0, 4.0, 2.0, 1.0):
        z_, ratio, spx, worst, at, ut = _ewa_vs_ray_row(z)
        rows.append([_g(z_), _g(ratio, 3), _g(spx, 4), f"{worst:.4f}", f"({at[0]:+d}, {at[1]:+d})", f"{ut:.5f}"])
    return ("An isotropic Gaussian of standard deviation 0.5 (opacity 1, no dilation) whose mean is at camera-frame direction "
            "(0.25, 0.1, 1) times z, seen by a pinhole camera with f = 100; z is the distance along the axis. The ray column evaluates the "
            "exact maximum response along each pixel's ray (`ray_response`, with the ray (u/f, v/f, 1) for the pixel centre at "
            "offset (u, v) from the principal point); the linearized column is exp(-q/2) of the Gaussian whose covariance is J Sigma J^T "
            "with J taken at the mean (the full 2 by 3 Jacobian including its z column). The principal point is the origin of the pixel "
            "offsets (u, v), so a pixel centre is a half-integer offset: with (c_u, c_v) = (f x / z, f y / z) the projected mean, the "
            "centres tried are (floor(c_u) + d_u + 1/2, floor(c_v) + d_v + 1/2) for the integers d_u, d_v in [-h, h], a square, "
            "with h = the integer part of min(3.5 projected sigma, 150). The error is the largest absolute difference over those pixels, "
            "with the pixel where it occurs given as (d_u, d_v) relative to the pixel containing the projected mean. The last column "
            "is the trace of the covariance of six sigma points m +- sqrt(3) sigma e_k projected through the same pinhole map, each "
            "squared offset taken about the projection of m itself (not about the mean of the projected points) and divided by 6, "
            "over the trace of J Sigma J^T.\n\n"
            + _table(["z", "sigma / z", "projected sigma (pixels)", "largest difference", "at pixel", "sigma-point trace / linearized trace"],
                     rows, "rrrrrr"))


def _lens_radius(kind, f, theta):
    return f * math.tan(theta) if kind == "pinhole" else f * theta


def _lens_ray(kind, f, r):
    """The direction of the pixel at distance r from the principal point along the row."""
    if kind == "pinhole":
        return [r / f, 0.0, 1.0]
    th = r / f
    return [math.sin(th), 0.0, math.cos(th)]


def _half_width(kind, f, theta0, sigma=0.3, R=4.0):
    m = [R * math.sin(theta0), 0.0, R * math.cos(theta0)]
    Lam = [[1.0 / (sigma * sigma) if i == j else 0.0 for j in range(3)] for i in range(3)]
    r0 = _lens_radius(kind, f, theta0)
    step = 0.05
    edges = []
    for sgn in (1.0, -1.0):
        r = r0
        while True:
            r += sgn * step
            if kind == "fisheye" and not 0.0 <= r / f < math.pi:
                break
            if kind == "pinhole" and abs(r) > 1e5:
                break
            if ray_response(m, Lam, _lens_ray(kind, f, r))[0] < 0.5:
                break
        edges.append(abs(r - r0))
    return r0, 0.5 * (edges[0] + edges[1])


def lens_rays():
    f = 200.0
    rows = []
    for deg in (10, 40, 70, 80):
        th = math.radians(deg)
        rp, wp = _half_width("pinhole", f, th)
        rf, wf = _half_width("fisheye", f, th)
        rows.append([str(deg), f"{rf:.2f}", f"{rp:.2f}", f"{wf:.2f}", f"{wp:.2f}", f"{wp / wf:.3f}", f"{1.0 / math.cos(th) ** 2:.3f}"])
    return ("An isotropic Gaussian of standard deviation 0.3, opacity 1, at distance 4 from the camera, in the plane of the row through the "
            "principal point at angle theta from the optical axis. Two cameras with f = 200: a pinhole (a pixel at distance r from the "
            "principal point looks along (r/f, 0, 1)) and an equidistant fisheye ((sin(r/f), 0, cos(r/f))). The response of the Gaussian "
            "at each pixel of the row is the exact ray response (`ray_response`), tried every 0.05 pixel outward and inward from the "
            "pixel of the Gaussian's centre until it falls below one half; the half-width is the mean of the two distances. The last column "
            "is 1 / cos^2(theta), the ratio of the radial rates of the two lenses (f / cos^2 theta against f).\n\n"
            + _table(["theta (degrees)", "centre radius, fisheye f theta", "centre radius, pinhole f tan theta", "half-width at one half, fisheye",
                      "half-width at one half, pinhole", "ratio pinhole / fisheye", "1 / cos^2 theta"], rows, "rrrrrrr"))


# ------------------------------------------------------------------------------------------------ the work of a frame

PAIR_FLOPS = 21          # per (pixel, instance) pair: 2 + 9 (sigma) + 1 (alpha) + 2 (T) + 1 (w) + 6 (colour), plus one exponential
SLOT_WORDS = 11          # channels of one pixel's partials in the backward (v_xy 2, v_conic 3, v_opacity 1, v_rgb 3, v_depth 1, |v_xy| 1)
SORT_BYTES_PER_KEY = 80  # four passes, each: histogram reads a 4-byte key; scatter reads key and value and writes both
PROJECT_FLOPS = 260      # a hand count of the lines of the projection, per Gaussian


def expected_instances(n, radius_px):
    """Expected tile instances of n Gaussians whose boxes are 2 radius wide and tall, at uniformly random positions on a grid of 16 pixel
    tiles: an interval of length L meets L / 16 + 1 tiles on average, in each axis."""
    return n * (2.0 * radius_px / TILE + 1.0) ** 2


def work_budget():
    N, W, H = 1_000_000, 1920, 1080
    tiles = -(-W // TILE) * -(-H // TILE)
    rows = []
    for r in (2, 4, 8, 16):
        inst = expected_instances(N, r)
        rows.append([str(r), f"{inst / N:.2f}", f"{inst:.3e}", f"{inst * SORT_BYTES_PER_KEY:.2e}", f"{inst * 256:.2e}", f"{inst * 256 * PAIR_FLOPS:.2e}",
                     f"{N * W * H / (inst * 256):.0f}"])
    back = []
    for r in (2, 4, 8, 16):
        inst = expected_instances(N, r)
        slot = inst * SLOT_WORDS * 256 * 4
        back.append([str(r), f"{inst:.3e}", f"{slot:.2e}", f"{2 * slot:.2e}", f"{inst * SLOT_WORDS * 4:.2e}"])
    view = []
    for div in (1, 2, 4):
        w, h = W // div, H // div
        r = 8.0 / div
        t = -(-w // TILE) * -(-h // TILE)
        inst = expected_instances(N, r)
        view.append([f"1/{div}", f"{w} x {h}", str(t), _g(r), f"{inst:.3e}", f"{inst * 256:.3e}", f"{inst * 256 / (expected_instances(N, 8.0) * 256):.3f}"])
    return (f"One million Gaussians in a {W} by {H} frame: {tiles:,} tiles of 16 by 16 pixels. Projection costs about {PROJECT_FLOPS} FLOPs per "
            f"Gaussian (a hand count of the lines of the projection), {PROJECT_FLOPS * N:.2e} in all, whatever the image. The instances are "
            "N (2r/16 + 1)^2 for boxes of half-width r pixels (the expected number of tiles a box meets, at random positions). The sort "
            f"moves {SORT_BYTES_PER_KEY} bytes per instance (four passes; each reads the key to count it, then reads and writes a 4-byte key "
            "and a 4-byte value), and the compositing looks at each instance with every one of the 256 pixels of its tile: at most "
            f"256 pairs per instance, each costing {PAIR_FLOPS} FLOPs and one exponential. The last column is how many times fewer pairs "
            f"this is than the {N * W * H:.2e} of looking at every Gaussian from every pixel.\n\n"
            + _table(["r (pixels)", "tiles per Gaussian", "instances", "sort bytes", "pairs (at most)", "pair FLOPs", "untiled pairs / tiled pairs"],
                     rows, "rrrrrrr")
            + f"\n\nThe backward pass of the same frame writes, for every instance, {SLOT_WORDS} partial derivatives for each of the 256 pixels of its "
              f"tile (a slot grid of {SLOT_WORDS} * 256 * 4 = {SLOT_WORDS * 256 * 4:,} bytes per instance), reads the grid back to sum the pixels of the instance into "
              f"one record of {SLOT_WORDS} words ({SLOT_WORDS * 4} bytes), and sums an instance's record into its Gaussian's gradient:\n\n"
            + _table(["r (pixels)", "instances", "slot grid written (bytes)", "written and read (bytes)", "records (bytes)"], back, "rrrrr")
            + "\n\nThe same scene drawn at a fraction of the window (the viewer's quality levels), boxes of half-width 8 pixels at full size and "
              "half as many at each halving:\n\n"
            + _table(["render size", "pixels", "tiles", "r", "instances", "pairs (at most)", "pairs / full-size pairs"], view, "rrrrrrr"))


# ------------------------------------------------------------------------------------------------ data of the figures

def _lens_curves():
    """(theta in degrees, half-width at one half for the fisheye, for the pinhole) at theta = 5, 10, ..., 80 degrees."""
    out = []
    for deg in range(5, 85, 5):
        th = math.radians(deg)
        out.append((deg, _half_width("fisheye", 200.0, th)[1], _half_width("pinhole", 200.0, th)[1]))
    return out


def _fit_curve():
    """The training loss of the unnudged fit after each of the 300 updates, from update 0."""
    return fit_runs()[0.0][2]
