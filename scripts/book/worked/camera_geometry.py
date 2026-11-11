# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Cameras and Projective Geometry".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: camera_geometry.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree.

* the ray behind one pixel of a pinhole camera, and the points on it;
* the same projection written with homogeneous coordinates, and the point at infinity of a direction;
* a camera placed in a world with `look_at`: the world-to-camera pair (R, t), the centre recovered from it, and the pixel a
  transposed rotation gives instead;
* one rotation in four forms (matrix, axis-angle, quaternion, Euler angles) and every conversion between them, the order of
  two rotations, and the Euler angles that coincide at gimbal lock;
* how an image resize moves the principal point under the corner and the centre convention;
* radial distortion of a normalized radius, the fold of a strong barrel lens, and what it does to a point past the fold;
* the inversion of a distorted point by fixed-point iteration and by Newton's method, iteration by iteration;
* the image radius of a ray at angle theta for the pinhole, the equidistant fisheye and a Kannala-Brandt polynomial, and the pixels
  per radian of each;
* the homography of a plane by the direct linear transform, with and without Hartley's normalization;
* calibration from planar views by Zhang's closed form, on exact data and on data with Gaussian noise, and the reprojection error.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, sin, cos, atan2, acos) unless a block says
otherwise. Sums are accumulated left to right in index order. Singular vectors come from the one-sided Jacobi SVD of the matrix
chapter of Volume I (`decompositions._svd`); a null vector is the right singular vector of the smallest singular value, its sign
fixed by the rule stated where it is used. The only random numbers are the noise of the calibration views, from the book's
SplitMix64 (`bookrng.generator`, stream name printed by the chapter).

Swedish Embedded AB implements solutions for camera calibration and lens modelling in photogrammetry, robotics and embedded vision.
If your team needs expertise in camera geometry and calibration, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import decompositions as dc
import images as im

_table = im._table
_g = dc._g

# ------------------------------------------------------------------------------------------------ small linear algebra

I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def mm(A, B):
    """Matrix product, each entry a left-to-right sum over the inner index."""
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def mv(A, x):
    return [sum(A[i][k] * x[k] for k in range(len(x))) for i in range(len(A))]


def tr(A):
    return [list(r) for r in zip(*A)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return [x / n for x in a]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def det3(A):
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))


def inv3(A):
    """Inverse of a 3 by 3 matrix by cofactors."""
    d = det3(A)
    c = [[A[(j + 1) % 3][(i + 1) % 3] * A[(j + 2) % 3][(i + 2) % 3] - A[(j + 1) % 3][(i + 2) % 3] * A[(j + 2) % 3][(i + 1) % 3]
          for j in range(3)] for i in range(3)]
    return [[c[i][j] / d for j in range(3)] for i in range(3)]


def vec(x, digits=4):
    return "(" + ", ".join(_g(c, digits) for c in x) + ")"


def mat(A, digits=4):
    return "[" + "; ".join(", ".join(_g(c, digits) for c in row) for row in A) + "]"


def sci(v):
    return f"{v:.1e}"


def null_vector(rows, n):
    """The unit vector of length n that the rows nearly annihilate: the right singular vector of the smallest singular value.

    A matrix with fewer rows than columns is padded with zero rows to n by n, which changes no singular vector. The sign is fixed
    so that the entry of largest magnitude is positive.
    """
    A = [list(r) for r in rows] + [[0.0] * n for _ in range(max(0, n - len(rows)))]
    _, sig, V = dc._svd(A)
    v = list(V[-1])
    k = max(range(n), key=lambda i: abs(v[i]))
    return [x if v[k] > 0 else -x for x in v], sig


# ------------------------------------------------------------------------------------------------ the pinhole

F0, CX0, CY0 = 500.0, 320.0, 240.0


def project_camera_point(P, f=F0, cx=CX0, cy=CY0):
    """u = cx + f X / Z, v = cy + f Y / Z for a point in the camera frame."""
    return (cx + f * P[0] / P[2], cy + f * P[1] / P[2])


def pixel_ray():
    """One pixel of a 640 by 480 camera with f = 500, c = (320, 240), and the points that land on it."""
    u, v = 400.0, 300.0
    d = [(u - CX0) / F0, (v - CY0) / F0, 1.0]
    rows = []
    for Z in (0.5, 1.0, 2.0, 5.0, 20.0):
        P = [Z * d[0], Z * d[1], Z]
        pu, pv = project_camera_point(P)
        rows.append([_g(Z), vec(P), f"({pu:.6f}, {pv:.6f})"])
    n = norm(d)
    out = ["The pixel (u, v) = (400, 300) of a camera with f_x = f_y = 500 and principal point (320, 240). Its ray is the set of "
           "points Z (x, y, 1) with x = (u - c_x) / f_x and y = (v - c_y) / f_y; the table projects five of them with "
           "u = c_x + f_x X / Z, v = c_y + f_y Y / Z.", "",
           _table(["Z", "(X, Y, Z)", "projects to"], rows, "lrr"), "",
           f"The ray direction is (x, y, 1) = {vec(d, 6)}; its length is {n:.6f}, so the unit ray is {vec([c / n for c in d], 6)}.",
           "", "The image is 640 by 480, so the angle between the rays through the left and right edges, 2 atan(320 / 500), is "
           f"{2 * math.degrees(math.atan(320 / F0)):.2f} degrees, and 2 atan(240 / 500) = {2 * math.degrees(math.atan(240 / F0)):.2f} "
           "degrees from top to bottom."]
    return "\n".join(out)


def homogeneous_projection():
    """K applied to camera-frame points gives (uZ, vZ, Z); a point at infinity projects to a finite pixel."""
    K = [[F0, 0.0, CX0], [0.0, F0, CY0], [0.0, 0.0, 1.0]]
    rows = []
    for P in ([0.0, 0.0, 4.0], [0.8, -0.4, 2.0], [-1.0, 0.5, 5.0], [0.16, 0.12, 1.0]):
        h = mv(K, P)
        rows.append([vec(P), vec(h), _g(h[2]), f"({h[0] / h[2]:.4f}, {h[1] / h[2]:.4f})"])
    out = ["The matrix K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]] applied to camera-frame points. Column 3 is the third entry, "
           "which equals Z; column 4 is the first two entries divided by it.", "",
           _table(["(X, Y, Z)", "K (X, Y, Z)", "third entry", "pixel (u, v)"], rows, "lrrr")]
    # a direction (a point at infinity) seen by a camera turned 30 degrees about the vertical (y) axis
    a = math.radians(30)
    R = [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    d = [0.0, 0.0, 1.0]
    h = mv(K, mv(R, d))
    out += ["", "A direction d = (0, 0, 1) of the world, seen by a camera whose rotation is R = turn by 30 degrees about the y axis "
            "(R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]] with cos = cos 30 degrees and sin = sin 30 degrees), is the point "
            "(d, 0) at infinity. Then K R d = " + vec(h, 6) + ", whose third entry is cos 30 degrees = 0.866025, so the pixel "
            f"is ({h[0] / h[2]:.4f}, {h[1] / h[2]:.4f}) = (c_x + f tan 30 degrees, c_y) with f tan 30 degrees = "
            f"{F0 * math.tan(a):.4f}."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ extrinsics

K0 = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]


def look_at(eye, target, up):
    """World-to-camera (R, t) of a camera at `eye` looking at `target`; camera axes +x right, +y down, +z forward.

    forward f = unit(target - eye), right r = unit(f x up), down d = unit(f x r); the rows of R are r, d, f and t = -R eye.
    `up` is the world direction that is up in the picture.
    """
    f = unit(sub(target, eye))
    r = unit(cross(f, up))
    d = unit(cross(f, r))
    R = [r, d, f]
    return R, [-x for x in mv(R, eye)]


EYE, TARGET, UP = [4.0, -3.0, 2.5], [0.2, 0.1, 0.4], [0.0, 0.0, 1.0]


def world_to_pixel(Pw, R, t, K=K0):
    c = [a + b for a, b in zip(mv(R, Pw), t)]
    h = mv(K, c)
    return c, (h[0] / h[2], h[1] / h[2])


def extrinsics_table():
    R, t = look_at(EYE, TARGET, UP)
    rows = []
    for name, Pw in (("origin", [0.0, 0.0, 0.0]), ("x = 1", [1.0, 0.0, 0.0]), ("x = y = 1", [1.0, 1.0, 0.0]),
                     ("top of the post", [0.0, 0.0, 1.0])):
        c, (u, v) = world_to_pixel(Pw, R, t)
        wrong = [a + b for a, b in zip(mv(tr(R), Pw), t)]  # R transposed by mistake
        hw = mv(K0, wrong)
        rows.append([name, vec(Pw), vec(c), f"({u:.3f}, {v:.3f})",
                     f"({hw[0] / hw[2]:.1f}, {hw[1] / hw[2]:.1f})" if hw[2] > 0 else "behind the camera"])
    RRt = mm(R, tr(R))
    orth = max(abs(RRt[i][j] - I3[i][j]) for i in range(3) for j in range(3))
    centre = [-x for x in mv(tr(R), t)]
    out = ["A camera with f_x = f_y = 800 and principal point (320, 240), at the world point (4, -3, 2.5) looking at (0.2, 0.1, 0.4), "
           "with the world's z axis up. The rows of R are the camera's right, down and forward axes written in world "
           "coordinates; t = -R C.", "",
           "R = " + mat(R, 6), f"t = {vec(t, 6)}", f"det R = {det3(R):.6f}; largest entry of R R^T - I: below 1e-15 "
           f"({'true' if orth < 1e-15 else 'false'}).",
           f"The centre recovered as -R^T t is {vec(centre, 6)}.", "",
           _table(["point", "world (X, Y, Z)", "camera (x, y, z)", "pixel (u, v)", "pixel with R^T used instead of R"], rows,
                  "llrrr")]
    return "\n".join(out)


def _look_at_columns():
    R, t = look_at(EYE, TARGET, UP)
    return R, t


# ------------------------------------------------------------------------------------------------ rotations

def rodrigues(w):
    """Rotation matrix of the rotation vector w: angle |w| about w / |w| (identity for w = 0)."""
    th = norm(w)
    if th < 1e-12:
        return [r[:] for r in I3]
    k = [c / th for c in w]
    K = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    KK = mm(K, K)
    s, c = math.sin(th), 1.0 - math.cos(th)
    return [[I3[i][j] + s * K[i][j] + c * KK[i][j] for j in range(3)] for i in range(3)]


def _axis_from_diagonal(R):
    """Unit axis of a half turn: R = 2 a a^T - I, so the largest diagonal entry gives the largest axis component."""
    d = [(R[i][i] + 1.0) / 2.0 for i in range(3)]
    i = max(range(3), key=lambda j: d[j])
    ax = [0.0] * 3
    ax[i] = math.sqrt(max(d[i], 0.0))
    for j in range(3):
        if j != i:
            ax[j] = R[i][j] / (2.0 * ax[i])
    return unit(ax)


def rotvec_of(R):
    """The rotation vector of a rotation matrix, angle in [0, pi].

    The skew part of R is w = 2 sin(th) a, so sin(th) = |w| / 2 and cos(th) = (tr R - 1) / 2; the angle is atan2(sin, cos),
    which stays accurate at every angle. At a half turn sin(th) is zero and the axis comes from the diagonal.
    """
    w = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    s = norm(w) / 2.0
    c = (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0
    th = math.atan2(s, c)
    if s < 1e-9:
        return [x / 2.0 for x in w] if c > 0 else [th * x for x in _axis_from_diagonal(R)]
    return [x * th / (2.0 * s) for x in w]


def rotvec_of_acos(R):
    """The same conversion with the angle taken as acos((tr R - 1) / 2), which loses digits close to a half turn."""
    c = max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0))
    th = math.acos(c)
    w = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    if th < 1e-9:
        return [x / 2.0 for x in w]
    if math.pi - th < 1e-6:
        return [th * x for x in _axis_from_diagonal(R)]
    return [x * th / (2.0 * math.sin(th)) for x in w]


def quat_of_rotvec(w):
    th = norm(w)
    if th < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    s = math.sin(th / 2) / th
    return [math.cos(th / 2)] + [s * x for x in w]


def matrix_of_quat(q):
    n = math.sqrt(sum(x * x for x in q))
    w, x, y, z = [c / n for c in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_of_matrix(R):
    """Unit quaternion (w, x, y, z) with w >= 0 of a rotation matrix: the largest of w, x, y, z is found from the diagonal."""
    t = R[0][0] + R[1][1] + R[2][2]
    cands = [t, R[0][0] - R[1][1] - R[2][2], -R[0][0] + R[1][1] - R[2][2], -R[0][0] - R[1][1] + R[2][2]]
    k = max(range(4), key=lambda i: cands[i])
    s = 2.0 * math.sqrt(1.0 + cands[k])
    if k == 0:
        q = [s / 4, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s]
    elif k == 1:
        q = [(R[2][1] - R[1][2]) / s, s / 4, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s]
    elif k == 2:
        q = [(R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, s / 4, (R[1][2] + R[2][1]) / s]
    else:
        q = [(R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, s / 4]
    return q if q[0] >= 0 else [-x for x in q]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def matrix_of_euler(yaw, pitch, roll):
    """R = Rz(yaw) Ry(pitch) Rx(roll): the roll is applied first, the yaw last."""
    return mm(rz(yaw), mm(ry(pitch), rx(roll)))


def euler_of_matrix(R):
    """(yaw, pitch, roll) of R = Rz Ry Rx. At pitch = +-90 degrees the yaw is set to 0 and the roll carries the rest."""
    s = -R[2][0]
    pitch = math.asin(max(-1.0, min(1.0, s)))
    if abs(s) > 1.0 - 1e-12:
        return 0.0, pitch, math.atan2(s * R[0][1], R[1][1])
    return math.atan2(R[1][0], R[0][0]), pitch, math.atan2(R[2][1], R[2][2])


def _maxdiff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def rotation_forms():
    """A rotation of 120 degrees about (1, 1, 1) and one of 150 degrees about (1, 2, 2), through every form."""
    cases = [("120 degrees about (1, 1, 1)", [x * math.radians(120) for x in unit([1.0, 1.0, 1.0])]),
             ("150 degrees about (1, 2, 2)", [x * math.radians(150) for x in unit([1.0, 2.0, 2.0])]),
             ("179.9999 degrees about z", [0.0, 0.0, math.radians(179.9999)])]
    out = ["Each rotation is given as a rotation vector w (angle |w| about w / |w|). The matrix is Rodrigues' formula "
           "I + sin(th) S + (1 - cos(th)) S^2 with S the cross-product matrix of the unit axis; the quaternion is "
           "(cos(th/2), sin(th/2) axis); the matrix is taken back to a quaternion by the largest-component rule and to a "
           "rotation vector by the formula of the chapter. The last column is the largest entry-wise difference between "
           "the original matrix and the matrix rebuilt from each recovered form.", ""]
    rows = []
    for name, w in cases:
        R = rodrigues(w)
        q = quat_of_rotvec(w)
        w_back = rotvec_of(R)
        q_back = quat_of_matrix(R)
        err = max(_maxdiff(R, rodrigues(w_back)), _maxdiff(R, matrix_of_quat(q_back)), _maxdiff(R, matrix_of_quat(q)))
        rows.append([name, mat(R, 5), vec(q, 5), vec(w_back, 5), vec(q_back, 5), sci(err)])
    out.append(_table(["rotation", "matrix", "quaternion (w, x, y, z)", "rotation vector back", "quaternion back", "round trip"],
                      rows, "lrrrrr"))
    # q and -q
    q = quat_of_rotvec(cases[0][1])
    out += ["", f"The quaternion {vec([-x for x in q], 4)} (every sign flipped) gives the same matrix: largest difference "
            f"{sci(_maxdiff(matrix_of_quat(q), matrix_of_quat([-x for x in q])))}."]
    return "\n".join(out)


def half_turn_loss():
    """Rotation vectors of rotations a distance delta short of a half turn about (1, 2, 2) / 3, by acos and by atan2."""
    rows = []
    for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6, 2e-6, 1.5e-6, 1e-6):
        w = [x * (math.pi - delta) for x in unit([1.0, 2.0, 2.0])]
        R = rodrigues(w)
        e1 = norm(sub(rotvec_of_acos(R), w))
        e2 = norm(sub(rotvec_of(R), w))
        rows.append([f"{delta:.1e}", sci(e1), sci(e2)])
    return ("Rotations of angle pi - delta about the axis (1, 2, 2) / 3, built by Rodrigues' formula and read back. The error is the "
            "length of the difference between the recovered and the original rotation vector. 'acos' takes the angle as "
            "acos((tr R - 1) / 2) and, within 1e-6 of pi, the axis from the diagonal; 'atan2' takes it as atan2(|w| / 2, "
            "(tr R - 1) / 2) with w the skew part of R.\n\n"
            + _table(["delta", "error with acos", "error with atan2"], rows))


def rotation_order():
    """A quarter turn about z then one about x, in both orders, as matrices and as quaternion products."""
    a = math.radians(90)
    Rz, Rx = rz(a), rx(a)
    qz, qx = quat_of_rotvec([0, 0, a]), quat_of_rotvec([a, 0, 0])
    A, B = mm(Rx, Rz), mm(Rz, Rx)   # first z then x ; first x then z
    qa, qb = qmul(qx, qz), qmul(qz, qx)
    p = [1.0, 0.0, 0.0]
    out = ["R_z and R_x are quarter turns about the z and the x axis. 'z then x' applies R_z first: the matrix is R_x R_z and the "
           "quaternion is q_x q_z (Hamilton product, the right factor acts first).", "",
           _table(["order", "matrix", "quaternion", "image of (1, 0, 0)", "image of (0, 0, 1)"],
                  [["z then x", mat(A, 3), vec(qa, 4), vec(mv(A, p), 3), vec(mv(A, [0, 0, 1.0]), 3)],
                   ["x then z", mat(B, 3), vec(qb, 4), vec(mv(B, p), 3), vec(mv(B, [0, 0, 1.0]), 3)]], "lrrrr"), "",
           f"Matrix from the quaternion product against the product of matrices: {sci(_maxdiff(matrix_of_quat(qa), A))} and "
           f"{sci(_maxdiff(matrix_of_quat(qb), B))}."]
    return "\n".join(out)


def gimbal_lock():
    """Two different (yaw, roll) at pitch 90 degrees give one matrix; at 89 degrees they do not."""
    rows = []
    for pitch in (89.0, 90.0):
        for yaw, roll in ((30.0, 10.0), (35.0, 15.0), (40.0, 20.0)):
            R = matrix_of_euler(math.radians(yaw), math.radians(pitch), math.radians(roll))
            rows.append([_g(pitch), f"({_g(yaw)}, {_g(roll)})", _g(yaw - roll), f"{R[0][1]:.6f}", f"{R[1][1]:.6f}",
                         f"{R[2][2]:.6f}"])
    R1 = matrix_of_euler(math.radians(30), math.radians(90), math.radians(10))
    R2 = matrix_of_euler(math.radians(40), math.radians(90), math.radians(20))
    yw, pt, rl = euler_of_matrix(R1)
    out = ["R = R_z(yaw) R_y(pitch) R_x(roll), angles in degrees. At a pitch of 90 degrees only yaw - roll reaches the matrix; "
           "three (yaw, roll) pairs with the same difference of 20 degrees are tried at 89 and at 90 degrees, and the "
           "entries R[0][1], R[1][1] and R[2][2] are shown.", "",
           _table(["pitch", "(yaw, roll)", "yaw - roll", "R[0][1]", "R[1][1]", "R[2][2]"], rows, "lrrrrr"), "",
           f"At pitch 90 degrees, (yaw, roll) = (30, 10) and (40, 20) give matrices that differ by {sci(_maxdiff(R1, R2))}. "
           f"Reading the angles back from that matrix gives (yaw, pitch, roll) = ({math.degrees(yw):.4f}, "
           f"{math.degrees(pt):.4f}, {math.degrees(rl):.4f}) degrees, with the yaw set to 0 by the rule of the chapter."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ pixel convention

def resize_table():
    """cx = 322.5 under the corner convention, and what resizing does to it under each rule."""
    cx_corner = 322.5          # the principal point of the 640-wide camera of the lens tests, corner convention
    cx_centre = cx_corner - 0.5   # the same physical point if pixel i has its centre at coordinate i
    rows = []
    for s in (0.5, 0.25, 2.0):
        right = cx_corner * s
        # the centre-convention rule: c' = (c + 1/2) s - 1/2, expressed back in the corner convention (+1/2)
        scaled_centre = cx_centre * s          # applying the corner rule to a centre-convention number
        wrong_in_corner = scaled_centre + 0.5
        rows.append([_g(s), f"{right:.4f}", f"{(cx_centre + 0.5) * s - 0.5:.4f}", f"{scaled_centre:.4f}", f"{wrong_in_corner:.4f}",
                     f"{wrong_in_corner - right:+.4f}"])
    out = ["The principal point of a 640-pixel-wide camera is 322.5 when pixel i covers [i, i + 1) and is centred on i + 1/2 "
           "(the corner convention); in the convention where pixel i is centred on i it is 322.0. The image is resampled by "
           "a factor s. Columns: the corner rule c' = s c; the centre rule c' = s (c + 1/2) - 1/2; the corner rule applied to "
           "the centre-convention number 322.0 by mistake; that mistaken number read back in the corner convention (+ 1/2); "
           "and its error in pixels of the new image.", "",
           _table(["s", "corner rule", "centre rule", "mistake (centre number, corner rule)", "mistake in corner terms",
                   "error"], rows, "lrrrrr")]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ lens distortion

def radial_factor(r2, k1, k2):
    return 1.0 + k1 * r2 + k2 * r2 * r2


def distort(x, y, k1, k2, p1=0.0, p2=0.0):
    """Brown-Conrady on normalized coordinates: radial 1 + k1 r^2 + k2 r^4 and decentering p1, p2."""
    r2 = x * x + y * y
    g = radial_factor(r2, k1, k2)
    return (x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * g + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def distort_jacobian(x, y, k1, k2, p1=0.0, p2=0.0):
    r2 = x * x + y * y
    g = radial_factor(r2, k1, k2)
    dg = k1 + 2 * k2 * r2     # d g / d(r^2)
    return [[g + 2 * x * x * dg + 2 * p1 * y + 6 * p2 * x, 2 * x * y * dg + 2 * p1 * x + 2 * p2 * y],
            [2 * x * y * dg + 2 * p1 * x + 2 * p2 * y, g + 2 * y * y * dg + 6 * p1 * y + 2 * p2 * x]]


def radial_table():
    """rd = r (1 - 0.35 r^2): the map from a normalized radius to the distorted one, its slope, and the fold."""
    k1 = -0.35
    fold = 1.0 / math.sqrt(3 * 0.35)
    rows = []
    for r in (0.25, 0.5, 0.75, 0.9, fold, 1.1, 1.3, 1.6):
        rd = r * (1 + k1 * r * r)
        slope = 1 + 3 * k1 * r * r
        rows.append([f"{r:.4f}", f"{rd:.4f}", f"{slope:+.4f}", "peak" if r == fold else ("folded" if slope < 0 else "")])
    return ("The radial map r_d = r (1 + k_1 r^2) with k_1 = -0.35 and k_2 = 0. Its slope is 1 + 3 k_1 r^2 and vanishes at "
            f"r = 1 / sqrt(3 * 0.35) = {fold:.6f}, where r_d reaches its largest value {fold * (1 + k1 * fold * fold):.6f}.\n\n"
            + _table(["r", "r_d", "slope d r_d / d r", ""], rows, "rrrl"))


BARREL = dict(k1=-0.2, k2=0.05, p1=0.002, p2=-0.001)


def invert_fixed_point(xd, yd, steps, **c):
    """x <- (xd - tangential(x)) / radial(x), starting from (xd, yd)."""
    x, y = xd, yd
    errs = []
    for _ in range(steps):
        r2 = x * x + y * y
        g = radial_factor(r2, c["k1"], c["k2"])
        tx = 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x)
        ty = c["p1"] * (r2 + 2 * y * y) + 2 * c["p2"] * x * y
        x, y = (xd - tx) / g, (yd - ty) / g
        errs.append((x, y))
    return errs


def invert_newton(xd, yd, steps, **c):
    """Newton on distort(x) = xd with the 2 by 2 Jacobian, starting from (xd, yd)."""
    x, y = xd, yd
    errs = []
    for _ in range(steps):
        fx, fy = distort(x, y, **c)
        J = distort_jacobian(x, y, **c)
        ex, ey = fx - xd, fy - yd
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        x -= (J[1][1] * ex - J[0][1] * ey) / det
        y -= (-J[1][0] * ex + J[0][0] * ey) / det
        errs.append((x, y))
    return errs


def inversion_table():
    c = BARREL
    xt, yt = 0.6, -0.45     # the true undistorted point: r = 0.75
    xd, yd = distort(xt, yt, **c)
    fp = invert_fixed_point(xd, yd, 12, **c)
    nw = invert_newton(xd, yd, 5, **c)

    def err(p):
        return math.hypot(p[0] - xt, p[1] - yt)

    def show(e):
        return "below 1e-14" if e < 1e-14 else f"{e:.1e}"
    rows = [["0 (start at the distorted point)", show(math.hypot(xd - xt, yd - yt)), show(math.hypot(xd - xt, yd - yt))]]
    for i in range(12):
        rows.append([str(i + 1), show(err(fp[i])), show(err(nw[i])) if i < 5 else ""])
    out = ["The lens k_1 = -0.2, k_2 = 0.05, p_1 = 0.002, p_2 = -0.001 applied to the normalized point (0.6, -0.45), whose radius "
           f"is 0.75, gives the distorted point ({xd:.9f}, {yd:.9f}). Both inversions start at the distorted point and "
           "report the distance of the iterate from (0.6, -0.45). Fixed point: x <- (x_d - T(x)) / (1 + k_1 r^2 + k_2 r^4), "
           "T the tangential part. Newton: x <- x - J^{-1} (distort(x) - x_d) with the analytic 2 by 2 Jacobian. Errors are "
           "printed to two digits; 'below 1e-14' is rounding noise.", "",
           _table(["iteration", "fixed-point error", "Newton error"], rows)]
    return "\n".join(out)


def fold_projection():
    """A point on the same ray as one inside the lens, past the fold of the k1 = -0.35 lens."""
    k1 = -0.35
    f = 400.0
    rows = []
    for Xz in (0.5, 0.9, 0.9759, 1.1, 1.3):
        r = Xz
        rd = r * (1 + k1 * r * r)
        rows.append([f"{Xz}", f"{f * Xz:.2f}", f"{f * rd:.2f}"])
    return ("A pinhole with f = 400 behind the lens k_1 = -0.35 (k_2 = 0). A point (X, 0, 1) lands at u - c_x = f r (1 + k_1 r^2) "
            "with r = X. Past the fold X = 0.9759 the pixel offset decreases again, so (1.3, 0, 1) would land closer to the "
            "centre than (0.9, 0, 1).\n\n"
            + _table(["X (at Z = 1)", "offset without the lens (pixels)", "offset with the lens (pixels)"], rows))


# ------------------------------------------------------------------------------------------------ fisheye

def kb_radius(theta, k):
    """theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), evaluated innermost first."""
    t2 = theta * theta
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def kb_slope(theta, k):
    """d theta_d / d theta of the Kannala-Brandt polynomial."""
    t2 = theta * theta
    return 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])))


def kb_fold(k, limit=math.pi, steps=512):
    """The first angle in (0, limit] where the slope stops being positive: walk out in `steps` equal steps, then bisect 60 times."""
    lo = 0.0
    for i in range(1, steps + 1):
        r = limit * i / steps
        if kb_slope(r, k) <= 0.0:
            hi = r
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if kb_slope(mid, k) > 0.0:
                    lo = mid
                else:
                    hi = mid
            return lo
        lo = r
    return None


def pinhole_vs_fisheye():
    f = 200.0
    kb = [0.02, -0.01, 0.003, -0.0005]
    fold = kb_fold(kb)
    rows = []
    for deg in (10, 30, 60, 80, 89, 100, 150):
        th = math.radians(deg)
        pin = f * math.tan(th) if deg < 90 else None
        eq = f * th
        kbv = f * kb_radius(th, kb)
        kb_cell = f"{kbv:.2f}" + (" (past the fold)" if th > fold else "")
        # pixels per radian: radial slope and the tangential rate r / sin(theta), geometric mean
        if pin is not None:
            rad, tang = f / math.cos(th) ** 2, f * math.tan(th) / math.sin(th)
            ppr_pin = f"{math.sqrt(rad * tang):.2f}"
        else:
            ppr_pin = "none"
        ppr_eq = math.sqrt(f * (f * th / math.sin(th)))
        rows.append([str(deg), f"{pin:.2f}" if pin is not None else "none", f"{eq:.2f}", kb_cell, ppr_pin, f"{ppr_eq:.2f}"])
    return ("Radius in pixels of a ray at angle theta from the optical axis, with f = 200: the pinhole f tan(theta), the "
            "equidistant fisheye f theta, and the Kannala-Brandt polynomial with k = (0.02, -0.01, 0.003, -0.0005). The last two "
            "columns are the pixels per radian (geometric mean of the radial rate d r / d theta and the tangential rate "
            "r / sin(theta)) of the pinhole and of the equidistant model. The slope of the Kannala-Brandt polynomial is positive up to "
            f"theta = {math.degrees(fold):.4f} degrees, found by walking out in 512 equal steps from 0 to pi and bisecting the first step "
            "that ends non-positive 60 times; beyond it the model is not a lens.\n\n"
            + _table(["theta (degrees)", "pinhole", "equidistant", "Kannala-Brandt", "pinhole pixels/rad", "equidistant pixels/rad"],
                     rows))


# ------------------------------------------------------------------------------------------------ the homography

GRID = [(i * 0.1, j * 0.1) for j in range(3) for i in range(4)]   # 4 by 3 points on Z = 0, spacing 0.1


def plane_homography(K, R, t):
    """K [r1 r2 t]: the map from plane coordinates (X, Y) of Z = 0 to pixels, a 3 by 3 matrix."""
    cols = [[R[i][0] for i in range(3)], [R[i][1] for i in range(3)], t]
    return mm(K, tr(cols))


def apply_h(H, X, Y):
    h = mv(H, [X, Y, 1.0])
    return h[0] / h[2], h[1] / h[2]


def hartley(pts):
    """Similarity T taking the points to centroid 0 and mean distance sqrt(2). Sums run in point order."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    d = sum(math.hypot(p[0] - mx, p[1] - my) for p in pts) / n
    s = math.sqrt(2.0) / d
    return [[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]]


def dlt_homography(src, dst, normalize=True):
    """H with dst ~ H src from n >= 4 pairs: two rows per pair, the null vector of the 2n by 9 matrix."""
    Ts = hartley(src) if normalize else [r[:] for r in I3]
    Td = hartley(dst) if normalize else [r[:] for r in I3]
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        a = mv(Ts, [x, y, 1.0])
        b = mv(Td, [u, v, 1.0])
        rows.append([0.0, 0.0, 0.0] + [-b[2] * c for c in a] + [b[1] * c for c in a])
        rows.append([b[2] * c for c in a] + [0.0, 0.0, 0.0] + [-b[0] * c for c in a])
    h, sig = null_vector(rows, 9)
    Hn = [h[0:3], h[3:6], h[6:9]]
    H = mm(inv3(Td), mm(Hn, Ts))
    return H, sig


def normalize_h(H):
    """Scale to unit Frobenius norm with H[2][2] > 0 (the sign of a homography is free)."""
    n = math.sqrt(sum(c * c for r in H for c in r))
    s = n if H[2][2] > 0 else -n
    return [[c / s for c in r] for r in H]


def homography_table():
    R, t = look_at(EYE, TARGET, UP)
    # the plane target lies on Z = 0, seen by the camera of the extrinsics example
    Htrue = plane_homography(K0, R, t)
    dst = [apply_h(Htrue, X, Y) for X, Y in GRID]
    rows = [[f"({x:.1f}, {y:.1f})", f"({u:.3f}, {v:.3f})"] for (x, y), (u, v) in zip(GRID, dst)]
    Hn, sig_n = dlt_homography(GRID, dst, True)
    Hr, sig_r = dlt_homography(GRID, dst, False)
    Ht = normalize_h(Htrue)
    en = _maxdiff(normalize_h(Hn), Ht)
    er = _maxdiff(normalize_h(Hr), Ht)
    out = ["The twelve points of a 4 by 3 grid with spacing 0.1 on the plane Z = 0, imaged by the camera of the extrinsics "
           "example (so that H = K [r_1 r_2 t]).", "", _table(["plane point (X, Y)", "pixel (u, v)"], rows, "lr"), "",
           "H = K [r_1 r_2 t], scaled to unit Frobenius norm with a positive last entry:", "", "`" + mat(Ht, 6) + "`", "",
           _table(["", "singular values of the 24 by 9 matrix: sigma_1, sigma_8, sigma_9", "sigma_1 / sigma_8",
                   "largest entry error of H"],
                  [["raw coordinates", f"{sig_r[0]:.4g}, {sig_r[7]:.4g}, below 1e-6" if sig_r[8] < 1e-6 else "?",
                    f"{sig_r[0] / sig_r[7]:.4g}", sci(er)],
                   ["Hartley-normalized", f"{sig_n[0]:.4g}, {sig_n[7]:.4g}, below 1e-6" if sig_n[8] < 1e-6 else "?",
                    f"{sig_n[0] / sig_n[7]:.4g}", sci(en)]], "lrrr")]
    return "\n".join(out)


HOM_NOISE_NAME = "camera_geometry.homography_noise"


def homography_noise():
    """Median and largest transfer error of the DLT with and without normalization, over 20 noisy copies of the twelve grid pixels."""
    R, t = look_at(EYE, TARGET, UP)
    H = plane_homography(K0, R, t)
    clean = [apply_h(H, X, Y) for X, Y in GRID]
    g = bookrng.generator(HOM_NOISE_NAME)
    trials = [[g.normal() for _ in range(2 * len(GRID))] for _ in range(20)]
    rows = []
    for sigma in (0.25, 1.0):
        errs = {False: [], True: []}
        for z in trials:
            noisy = [(u + sigma * z[2 * i], v + sigma * z[2 * i + 1]) for i, (u, v) in enumerate(clean)]
            for norm_ in (False, True):
                Hh, _ = dlt_homography(GRID, noisy, norm_)
                e = sum((a - u) ** 2 + (b - v) ** 2 for (X, Y), (u, v) in zip(GRID, clean) for a, b in [apply_h(Hh, X, Y)])
                errs[norm_].append(math.sqrt(e / len(clean)))
        med = {k: sorted(v)[9] / 2 + sorted(v)[10] / 2 for k, v in errs.items()}
        rows.append([_g(sigma), f"{med[False]:.3f}", _g(max(errs[False]), 3), f"{med[True]:.3f}", _g(max(errs[True]), 3)])
    return (f"The twelve grid pixels of the previous table with Gaussian noise of standard deviation sigma added to each coordinate, "
            f"20 trials. The generator is `bookrng.generator(\"{HOM_NOISE_NAME}\")` (seed {bookrng.seed_of(HOM_NOISE_NAME)}); trial by "
            "trial, 24 draws of `normal()`, in the order of the grid, u before v, are taken once and multiplied by sigma for each row. "
            "The error of a trial is the root mean square, over the twelve grid points, of the distance in pixels between H applied "
            "to the grid point and the noise-free pixel; the table gives the median (mean of the 10th and 11th smallest) and the largest of "
            "the 20 trials.\n\n"
            + _table(["sigma (pixels)", "raw: median", "raw: largest", "normalized: median", "normalized: largest"], rows))


def inversion_strength():
    """Iterations of each inversion to reach an error below 1e-10 as the lens is made two and three times as strong."""
    xt, yt = 0.6, -0.45
    rows = []
    for m in (1, 2, 3):
        c = dict(k1=-0.2 * m, k2=0.05 * m, p1=0.002 * m, p2=-0.001 * m)
        xd, yd = distort(xt, yt, **c)

        def count(path):
            for i, (x, y) in enumerate(path):
                if math.hypot(x - xt, y - yt) < 1e-10:
                    return i + 1
            return None
        fp = count(invert_fixed_point(xd, yd, 300, **c))
        nw = count(invert_newton(xd, yd, 30, **c))
        r2 = xt * xt + yt * yt
        slope = 1 + 3 * c["k1"] * r2 + 5 * c["k2"] * r2 * r2
        e0 = math.hypot(xd - xt, yd - yt)
        rows.append([f"{m}", f"{slope:.4f}", f"{e0:.3f}", str(fp), str(nw)])
    return ("The lens of the previous table with every coefficient multiplied by m, the same point (0.6, -0.45), both iterations "
            "started at the distorted point. The slope is that of the radial map at r = 0.75, 1 + 3 k_1 r^2 + 5 k_2 r^4; the "
            "counts are the first iteration whose error is below 1e-10.\n\n"
            + _table(["m", "radial slope at r = 0.75", "error at the start", "fixed-point iterations", "Newton iterations"], rows))


# ------------------------------------------------------------------------------------------------ calibration from a plane

KTRUE = dict(fx=820.0, fy=815.0, cx=322.5, cy=238.0)
BOARD = [(i * 0.04, j * 0.04) for j in range(5) for i in range(7)]       # 7 by 5 corners, spacing 0.04 m, on Z = 0
VIEWS = [  # (rotation vector in degrees about each axis as a vector, translation) of the board in the camera frame
    ([-0.3, 0.2, 0.1], [-0.12, -0.08, 0.60]),
    ([0.4, -0.25, 0.05], [-0.10, -0.06, 0.55]),
    ([-0.2, -0.45, -0.15], [-0.14, -0.07, 0.65]),
    ([0.35, 0.3, 0.2], [-0.11, -0.09, 0.58]),
    ([0.0, 0.5, -0.1], [-0.13, -0.05, 0.62]),
]
SIGMA_PX = 0.25
CAL_NAME = "camera_geometry.calibration_noise"


def kmat(fx, fy, cx, cy, gamma=0.0):
    return [[fx, gamma, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def board_pixels(K, R, t, grid=BOARD):
    H = plane_homography(K, R, t)
    return [apply_h(H, X, Y) for X, Y in grid]


def synthetic_views(noise=0.0, name=CAL_NAME):
    """Pixel corners of the board in each view; noise > 0 adds Gaussian offsets drawn view by view, point by point, u then v."""
    K = kmat(KTRUE["fx"], KTRUE["fy"], KTRUE["cx"], KTRUE["cy"])
    g = bookrng.generator(name)
    views = []
    for w, t in VIEWS:
        R = rodrigues(w)
        pix = board_pixels(K, R, t)
        if noise > 0:
            pix = [(u + noise * g.normal(), v + noise * g.normal()) for u, v in pix]
        views.append(pix)
    return views


def _vij(H, i, j):
    """Row of the constraint matrix: h_i^T B h_j written as v . b with b = (B11, B12, B22, B13, B23, B33)."""
    hi = [H[r][i] for r in range(3)]
    hj = [H[r][j] for r in range(3)]
    return [hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1],
            hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]]


def zhang_intrinsics(Hs):
    """The intrinsic matrix from n >= 3 plane homographies: two constraints per view, the null vector b of the 2n by 6 matrix."""
    rows = []
    for H in Hs:
        rows.append(_vij(H, 0, 1))
        v11, v22 = _vij(H, 0, 0), _vij(H, 1, 1)
        rows.append([a - b for a, b in zip(v11, v22)])
    b, sig = null_vector(rows, 6)
    if b[0] < 0:
        b = [-x for x in b]                      # B must be positive definite, so B11 > 0
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
    lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = math.sqrt(lam / B11)
    beta = math.sqrt(lam * B11 / (B11 * B22 - B12 * B12))
    gamma = -B12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha * alpha / lam
    return kmat(alpha, beta, u0, v0, gamma), sig


def nearest_rotation(M):
    """The rotation closest to M in the Frobenius norm: U V^T from the SVD M = U S V^T, with det fixed to +1."""
    U, _, V = dc._svd(M)        # lists of columns, largest singular value first
    signs = [1.0, 1.0, 1.0]
    for _ in range(2):
        R = [[sum(signs[k] * U[k][i] * V[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        if det3(R) > 0:
            return R
        signs[2] = -1.0         # flip the direction of the smallest singular value
    return R


def zhang_pose(K, H):
    """R, t of one view from K and its homography: columns K^{-1} h_1, K^{-1} h_2 of unit length, third column their cross product."""
    Ki = inv3(K)
    h1, h2, h3 = ([H[r][c] for r in range(3)] for c in range(3))
    a1, a2, a3 = mv(Ki, h1), mv(Ki, h2), mv(Ki, h3)
    s = 2.0 / (norm(a1) + norm(a2))
    if a3[2] * s < 0:
        s = -s                                   # the board is in front of the camera
    r1, r2 = [s * c for c in a1], [s * c for c in a2]
    r3 = cross(r1, r2)
    R = nearest_rotation(tr([r1, r2, r3]))
    return R, [s * c for c in a3]


def reprojection_rms(K, poses, views):
    """Root mean square distance in pixels between the corners and the board projected with (K, R, t): sqrt(mean of u and v errors squared sum)."""
    s, n = 0.0, 0
    for (R, t), pix in zip(poses, views):
        for (u, v), (pu, pv) in zip(pix, board_pixels(K, R, t)):
            s += (u - pu) ** 2 + (v - pv) ** 2
            n += 1
    return math.sqrt(s / n)


def _calibrate(views):
    Hs = [normalize_h(dlt_homography(BOARD, pix, True)[0]) for pix in views]
    K, sig = zhang_intrinsics(Hs)
    poses = [zhang_pose(K, H) for H in Hs]
    return K, poses, sig


def calibration_exact():
    views = synthetic_views(0.0)
    K, poses, sig = _calibrate(views)
    Ktrue = kmat(KTRUE["fx"], KTRUE["fy"], KTRUE["cx"], KTRUE["cy"])
    rows = [[n, _g(KTRUE[key]), f"{K[i][j]:.6f}"]
            for n, key, (i, j) in (("f_x", "fx", (0, 0)), ("f_y", "fy", (1, 1)), ("c_x", "cx", (0, 2)), ("c_y", "cy", (1, 2)))]
    rows.append(["skew", "0", f"{K[0][1]:.2e}"])
    prow = []
    for v, ((w, t), (R, tt)) in enumerate(zip(VIEWS, poses)):
        prow.append([str(v), vec(t, 4), vec(tt, 6),
                     f"{math.degrees(norm(rotvec_of(mm(tr(rodrigues(w)), R)))):.2e}"])
    out = [f"The board: 7 by 5 corners with spacing 0.04 m on Z = 0, seen in {len(VIEWS)} views by the camera "
           f"f_x = 820, f_y = 815, (c_x, c_y) = (322.5, 238), no skew, no noise. View v has rotation vector w_v (radians) and "
           "translation t_v below (the pose maps board coordinates to camera coordinates, x_c = R x_w + t).", "",
           _table(["view", "w (radians)", "t (metres)"], [[str(i), vec(w, 3), vec(t, 3)] for i, (w, t) in enumerate(VIEWS)], "lrr"),
           "", "Intrinsics recovered by the closed form, from the singular values of the 10 by 6 constraint matrix:", "",
           _table(["parameter", "true", "recovered"], rows, "lrr"), "",
           f"The singular values of the constraint matrix are {', '.join(f'{x:.4g}' for x in sig[:5])} and a last one below 1e-9. "
           f"The reprojection RMS of the recovered parameters is {reprojection_rms(K, poses, views):.2e} pixels.", "",
           "Recovered poses (translation, and the angle in degrees of the rotation that takes the true R to the recovered one):", "",
           _table(["view", "true t", "recovered t", "rotation error (degrees)"], prow, "lrrr")]
    return "\n".join(out)


def calibration_noisy():
    views = synthetic_views(SIGMA_PX)
    K, poses, sig = _calibrate(views)
    Ktrue = kmat(KTRUE["fx"], KTRUE["fy"], KTRUE["cx"], KTRUE["cy"])
    truth = [(rodrigues(w), t) for w, t in VIEWS]
    rms_fit = reprojection_rms(K, poses, views)
    rms_true = reprojection_rms(Ktrue, truth, views)
    rows = [[n, _g(KTRUE[key]), f"{K[i][j]:.2f}"]
            for n, key, (i, j) in (("f_x", "fx", (0, 0)), ("f_y", "fy", (1, 1)), ("c_x", "cx", (0, 2)), ("c_y", "cy", (1, 2)))]
    rows.append(["skew", "0", f"{K[0][1]:.2f}"])
    return (f"The same five views with Gaussian noise of standard deviation {SIGMA_PX} pixel added to every corner coordinate. The "
            f"generator is `bookrng.generator(\"{CAL_NAME}\")` (seed {bookrng.seed_of(CAL_NAME)}); the draws are taken view by view, "
            "corner by corner in the order of the board (row by row, the column index fastest), the u coordinate before the v "
            "coordinate, one `normal()` call each (70 a view, 350 in all). "
            "Values are printed to two decimals; the runs of the closed form are stable to that.\n\n"
            + _table(["parameter", "true", "closed form"], rows, "lrr") + "\n\n"
            f"Reprojection RMS over the {sum(len(v) for v in views)} corners: {rms_fit:.4f} pixels for the recovered parameters "
            f"and {rms_true:.4f} pixels for the true parameters on the same noisy corners (the noise floor: "
            f"{SIGMA_PX} sqrt(2) = {SIGMA_PX * math.sqrt(2):.4f} for the expectation).")


# ------------------------------------------------------------------------------------------------ data of the figures

def radial_curves(n=161, rmax=1.6):
    """Distorted radius r (1 + k1 r^2 + k2 r^4) for three lenses on r in [0, rmax], and the fold radius of the last."""
    rs = [rmax * i / (n - 1) for i in range(n)]
    lenses = [("k1 = 0.15 (pincushion)", 0.15, 0.0), ("k1 = -0.2, k2 = 0.05 (mild barrel)", -0.2, 0.05),
              ("k1 = -0.35 (folds)", -0.35, 0.0)]
    curves = [(name, [r * radial_factor(r * r, k1, k2) for r in rs]) for name, k1, k2 in lenses]
    return rs, curves, 1.0 / math.sqrt(3 * 0.35)


def radius_curves(n=151):
    """r / f against the angle theta in degrees (0..150) for the pinhole (up to 85 degrees), the equidistant model and the KB polynomial."""
    kb = [0.02, -0.01, 0.003, -0.0005]
    degs = [150.0 * i / (n - 1) for i in range(n)]
    pin = [(d, math.tan(math.radians(d))) for d in degs if d <= 85.0]
    eq = [math.radians(d) for d in degs]
    kbv = [kb_radius(math.radians(d), kb) for d in degs]
    return degs, pin, eq, kbv, math.degrees(kb_fold(kb))
