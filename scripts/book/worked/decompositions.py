# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Eigenvalues, Rotations and Low-Rank Structure".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: decompositions.<name> -->` markers. Functions whose names start
with an underscore (or that take arguments) are shared with the chapter's
figures, so a picture and the table beside it cannot disagree.

Matrices are plain lists of rows of floats (or Fractions where the chapter
works by hand). The only eigen-solver is cyclic Jacobi rotation for symmetric
matrices; singular values come either from the eigenvalues of A^T A (the
chapter's derivation) or from one-sided Jacobi on A itself (the accurate way).

Swedish Embedded AB implements solutions for compressing and analysing
neural-network weights. If your team needs expertise in matrix decompositions,
low-rank adaptation and numerical parity testing, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    cell = lambda c: str(c).replace("|", "\\|")  # a literal pipe inside a cell must not split the column
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def _g(v, digits=4):
    """Compact decimal: up to `digits` significant digits, no trailing zeros."""
    if isinstance(v, str):
        return v
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator)
        if abs(v.denominator) <= 64:
            return f"{v.numerator}/{v.denominator}"
        v = float(v)
    if abs(v) < 1e-12:
        return "0"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def _v(x, digits=4):
    return "(" + ", ".join(_g(c, digits) for c in x) + ")"


def _m(A, digits=4):
    return "[" + ", ".join("[" + ", ".join(_g(c, digits) for c in row) + "]" for row in A) + "]"


def _sci(v):
    return f"{v:.2e}"


# ------------------------------------------------------ matrix operations

def _matmul(A, B):
    assert len(A[0]) == len(B)
    return [[sum(A[i][l] * B[l][j] for l in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def _T(A):
    return [list(col) for col in zip(*A)]


def _eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _fro(A):
    return math.sqrt(sum(x * x for row in A for x in row))


def _sub(A, B):
    return [[a - b for a, b in zip(r, s)] for r, s in zip(A, B)]


def _scale_cols(A, s):
    return [[a * s[j] for j, a in enumerate(row)] for row in A]


def _det2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _det3(A):
    """Cofactor expansion along the first row."""
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))


def _polygon_area(pts):
    """Signed shoelace area of a polygon given as a list of (x, y)."""
    s = 0
    for k in range(len(pts)):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % len(pts)]
        s += x0 * y1 - x1 * y0
    return s / 2


def _eigh(S, sweeps=60):
    """Eigenvalues (descending) and eigenvector columns of a symmetric matrix, by cyclic Jacobi rotations.

    Each rotation zeroes one off-diagonal pair; sweeping over all pairs until
    the off-diagonal is negligible leaves the eigenvalues on the diagonal and
    the accumulated rotations as the eigenvectors.
    """
    n = len(S)
    A = [list(map(float, r)) for r in S]
    V = _eye(n)
    scale = sum(x * x for row in A for x in row)  # squared Frobenius size: unchanged by rotations
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off <= 1e-30 * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                J = _eye(n)
                J[p][p] = J[q][q] = c
                J[p][q] = s
                J[q][p] = -s
                A = _matmul(_T(J), _matmul(A, J))
                V = _matmul(V, J)
    order = sorted(range(n), key=lambda i: -A[i][i])
    vals = [A[i][i] for i in order]
    vecs = [[V[r][i] for r in range(n)] for i in order]  # each eigenvector as a list
    for v in vecs:  # a sign convention: the largest-magnitude entry is positive... first nonzero positive
        k = next(i for i, x in enumerate(v) if abs(x) > 1e-12)
        if v[k] < 0:
            v[:] = [-x for x in v]
    return vals, vecs


def _svd(A, sweeps=60):
    """Thin SVD by one-sided Jacobi: returns (U columns, sigma descending, V columns).

    Columns of A are rotated pairwise until all are orthogonal. Then A V = U
    Sigma: the column lengths are the singular values and the normalised
    columns are the u_i.
    """
    m, n = len(A), len(A[0])
    if m < n:
        Ut, s, Vt = _svd(_T(A))
        return Vt, s, Ut
    U = [[float(x) for x in row] for row in A]
    V = _eye(n)
    for _ in range(sweeps):
        rotated = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = sum(U[k][i] ** 2 for k in range(m))
                b = sum(U[k][j] ** 2 for k in range(m))
                g = sum(U[k][i] * U[k][j] for k in range(m))
                if abs(g) <= 1e-15 * math.sqrt(a * b) or g == 0.0:
                    continue
                rotated = True
                zeta = (b - a) / (2 * g)
                t = (1 if zeta >= 0 else -1) / (abs(zeta) + math.sqrt(1 + zeta * zeta))
                c = 1 / math.sqrt(1 + t * t)
                s = c * t
                for M in (U, V):
                    for row in M:
                        x, y = row[i], row[j]
                        row[i], row[j] = c * x - s * y, s * x + c * y
        if not rotated:
            break
    sig = [math.sqrt(sum(U[k][j] ** 2 for k in range(m))) for j in range(n)]
    order = sorted(range(n), key=lambda j: -sig[j])
    sigma = [sig[j] for j in order]
    ucols = [[U[k][j] / sig[j] if sig[j] > 1e-300 else 0.0 for k in range(m)] for j in order]
    vcols = [[V[k][j] for k in range(n)] for j in order]
    return ucols, sigma, vcols


def _outer(u, v):
    return [[a * b for b in v] for a in u]


def _truncate(U, s, V, r):
    """sum_{i<r} s_i u_i v_i^T as a matrix."""
    m, n = len(U[0]), len(V[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(r):
        for a in range(m):
            for b in range(n):
                out[a][b] += s[i] * U[i][a] * V[i][b]
    return out


def _spectral_norm(A, iters=500):
    """Largest stretch max ||Ax||/||x||, by power iteration on A^T A (the chapter's algorithm)."""
    G = _matmul(_T(A), A)
    x = [1.0 / math.sqrt(len(G))] * len(G)
    lam = 0.0
    for _ in range(iters):
        y = _matvec(G, x)
        nrm = _norm(y)
        if nrm == 0:
            return 0.0
        x = [c / nrm for c in y]
        lam = _dot(x, _matvec(G, x))
    return math.sqrt(lam)


def _gram_schmidt(cols):
    out = []
    for c in cols:
        v = list(c)
        for q in out:
            d = _dot(v, q)
            v = [a - d * b for a, b in zip(v, q)]
        n = _norm(v)
        if n > 1e-12:
            out.append([a / n for a in v])
    return out


def _project_columns(A, basis):
    """Q Q^T A for orthonormal columns Q given as a list of vectors."""
    m, n = len(A), len(A[0])
    out = [[0.0] * n for _ in range(m)]
    for q in basis:
        for j in range(n):
            coef = sum(q[i] * A[i][j] for i in range(m))
            for i in range(m):
                out[i][j] += q[i] * coef
    return out


# ------------------------------------------------ the chapter's test data

SQUARE_A = [[Fraction(2), Fraction(1)], [Fraction(1, 2), Fraction(3, 2)]]  # chapter 4's parallelogram

SYM2 = [[2.0, 1.0], [1.0, 2.0]]
NONSYM2 = [[4.0, 1.0], [2.0, 3.0]]
TRI3 = [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
SVD_A = [[3.0, 0.0], [4.0, 5.0]]


def spectrum_matrix():
    """The 8 x 6 table used for low-rank approximation: three smooth patterns plus a little noise.

    M[i][j] = 6 sin(1 (i+1/2) pi/8) cos(1 (j+1/2) pi/6)
            + 3 sin(2 (i+1/2) pi/8) cos(2 (j+1/2) pi/6)
            + 1.5 sin(3 (i+1/2) pi/8) cos(3 (j+1/2) pi/6)
            + 0.05 ((7 i + 13 j) mod 11 - 5) / 5.
    """
    m, n = 8, 6
    M = []
    for i in range(m):
        row = []
        for j in range(n):
            v = 0.0
            for k, w in ((1, 6.0), (2, 3.0), (3, 1.5)):
                v += w * math.sin(k * (i + 0.5) * math.pi / m) * math.cos(k * (j + 0.5) * math.pi / n)
            v += 0.05 * (((7 * i + 13 * j) % 11) - 5) / 5
            row.append(v)
        M.append(row)
    return M


# ------------------------------------------------------------ determinant

def determinant_area():
    cases = [
        ("stretch", [[3, 0], [0, 2]]),
        ("chapter 4 parallelogram", [[Fraction(2), Fraction(1)], [Fraction(1, 2), Fraction(3, 2)]]),
        ("shear", [[1, 1], [0, 1]]),
        ("swap axes", [[0, 1], [1, 0]]),
        ("collapse", [[1, 2], [Fraction(1, 2), 1]]),
        ("generic", [[1, 2], [3, 4]]),
    ]
    rows = []
    for name, A in cases:
        A = [[Fraction(x) for x in r] for r in A]
        imgs = [(A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        area = _polygon_area(imgs)
        d = _det2(A)
        rows.append((name, _m(A), f"{_g(A[0][0])}*{_g(A[1][1])} - {_g(A[0][1])}*{_g(A[1][0])}", _g(d), _g(area)))
    return ("Unit square (0,0), (1,0), (1,1), (0,1) mapped by each matrix; the signed area of the image polygon, "
            "walked in that order, by the shoelace formula, next to ad - bc.\n\n"
            + _table(["map", "matrix", "ad - bc", "det", "signed area of image"], rows, "lllrr"))


def determinant_rules():
    A = [[Fraction(x) for x in r] for r in [[4, 7], [2, 6]]]
    B = [[Fraction(x) for x in r] for r in [[1, 2], [3, 4]]]

    def mul(P, Q):
        return [[sum(P[i][l] * Q[l][j] for l in range(2)) for j in range(2)] for i in range(2)]

    AB = mul(A, B)
    d = _det2(A)
    inv = [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]
    rows = [
        ("det A", _g(_det2(A))),
        ("det B", _g(_det2(B))),
        ("det(A) * det(B)", _g(_det2(A) * _det2(B))),
        ("A B", _m(AB)),
        ("det(A B)", _g(_det2(AB))),
        ("det(A^T)", _g(_det2(_T(A)))),
        ("A^-1", _m(inv)),
        ("det(A^-1)", _g(_det2(inv))),
        ("1 / det A", _g(1 / _det2(A))),
        ("det(3 A), A is 2 x 2", _g(_det2([[3 * x for x in r] for r in A]))),
        ("3^2 det A", _g(9 * _det2(A))),
    ]
    return "A = [[4, 7], [2, 6]], B = [[1, 2], [3, 4]].\n\n" + _table(["quantity", "value"], rows)


def determinant_3x3():
    A = [[2, 0, 1], [1, 3, 2], [1, 1, 2]]
    terms = []
    total = 0
    for j, sign in ((0, 1), (1, -1), (2, 1)):
        minor = [[A[r][c] for c in range(3) if c != j] for r in (1, 2)]
        md = _det2(minor)
        terms.append((f"column {j}", A[0][j], _m(minor), md, sign * A[0][j] * md))
        total += sign * A[0][j] * md
    rows = [(a, b, c, d, e) for a, b, c, d, e in terms]
    lines = ["A = [[2, 0, 1], [1, 3, 2], [1, 1, 2]], expanded along row 0 with signs +, -, +.", "",
             _table(["term", "entry A[0][j]", "minor (delete row 0 and column j)", "det of minor", "signed contribution"],
                    rows, "lrlrr"),
             "", f"det A = {total}; the cofactor routine gives {_det3(A)}.", ""]
    S = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    lines.append(f"The singular matrix of chapter 4, {_m(S)}: det = {_det3(S)}.")
    # volume of the image of the unit cube: |det| equals the scalar triple product
    cols = [[A[r][c] for r in range(3)] for c in range(3)]
    cx = [cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1],
          cols[1][2] * cols[2][0] - cols[1][0] * cols[2][2],
          cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0]]
    lines.append(f"Columns of A: a1 = {_v(cols[0])}, a2 = {_v(cols[1])}, a3 = {_v(cols[2])}; "
                 f"a2 x a3 = {_v(cx)}, and a1 . (a2 x a3) = {_dot(cols[0], cx)}, the volume of the parallelepiped they span.")
    return "\n".join(lines)


def determinant_pivots():
    """det by elimination: row operations that add a multiple of one row to another leave it unchanged."""
    A = [[Fraction(x) for x in r] for r in [[2, 0, 1], [1, 3, 2], [1, 1, 2]]]
    M = [r[:] for r in A]
    n = 3
    swaps = 0
    rows = []
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            swaps += 1
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        rows.append((f"pivot {c}", _g(M[c][c]), _m(M)))
    prod = Fraction(1)
    for c in range(n):
        prod *= M[c][c]
    prod *= (-1) ** swaps
    return (_table(["step", "pivot", "matrix after clearing the column"], rows, "lrl")
            + f"\n\nProduct of the pivots: {_g(prod)} (row swaps used: {swaps}). Cofactor expansion: {_g(_det3(A))}.")


def determinant_scale():
    """det(c I_n) = c^n : exactly invertible, yet the determinant leaves the floating-point range."""
    rows = []
    for n in (10, 100, 300, 400):
        c = 0.1
        d = 1.0
        for _ in range(n):
            d *= c
        logd = n * math.log(c)
        rows.append((n, _sci(d) if d != 0 else "0.0 (underflow)", _g(logd)))
    return ("A = 0.1 I_n multiplies every vector by 0.1: invertible for every n. Its determinant is 0.1^n.\n\n"
            + _table(["n", "det computed as a float64 product", "log |det| = n log 0.1"], rows))


# ------------------------------------------------------------ eigenvalues

def eigen_directions():
    A = SYM2
    rows = []
    for deg in (0, 30, 45, 60, 90, 135):
        t = math.radians(deg)
        x = [math.cos(t), math.sin(t)]
        y = _matvec(A, x)
        a = math.degrees(math.atan2(y[1], y[0]))
        rows.append((deg, _v(x), _v(y), _g(a), _g(_norm(y))))
    return ("A = [[2, 1], [1, 2]] applied to unit vectors at several angles.\n\n"
            + _table(["angle of x", "x", "A x", "angle of A x", "length of A x"], rows))


def eigen_by_hand():
    A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    tr = A[0][0] + A[1][1]
    d = _det2(A)
    disc = tr * tr - 4 * d
    lam = [(tr + math.sqrt(disc)) / 2, (tr - math.sqrt(disc)) / 2]
    lines = [
        f"A = {_m(A)}: trace = {_g(tr)}, det = {_g(d)}.",
        f"det(A - lambda I) = (2 - lambda)(2 - lambda) - 1 = lambda^2 - {_g(tr)} lambda + {_g(d)} = 0.",
        f"Discriminant {_g(disc)}; lambda = ({_g(tr)} +- {_g(math.sqrt(disc))}) / 2 = {_g(lam[0])} or {_g(lam[1])}.",
        "",
    ]
    rows = []
    for l in lam:
        # (A - lambda I) v = 0: first row (2 - l) v0 + v1 = 0 gives v = (1, l - 2)
        v = [1.0, l - 2]
        Av = _matvec(A, v)
        rows.append((_g(l), f"(2 - {_g(l)}) v0 + v1 = 0, so v = (1, {_g(v[1])})", _v(Av), _v([l * c for c in v])))
    lines.append(_table(["lambda", "solve (A - lambda I) v = 0", "A v", "lambda v"], rows, "rlll"))
    lines += ["", f"Sum of eigenvalues {_g(sum(lam))} = trace {_g(tr)}; product {_g(lam[0] * lam[1])} = det {_g(d)}."]
    return "\n".join(lines)


def eigen_nonsymmetric():
    A = NONSYM2
    tr, d = A[0][0] + A[1][1], _det2(A)
    disc = tr * tr - 4 * d
    lam = [(tr + math.sqrt(disc)) / 2, (tr - math.sqrt(disc)) / 2]
    rows = []
    vecs = []
    for l in lam:
        # row 0: (4 - l) v0 + v1 = 0
        v = [1.0, l - 4]
        vecs.append(v)
        rows.append((_g(l), _v(v), _v(_matvec(A, v)), _v([l * c for c in v])))
    return (f"A = {_m(A)}: trace {_g(tr)}, det {_g(d)}, lambda^2 - {_g(tr)} lambda + {_g(d)} = 0.\n\n"
            + _table(["lambda", "eigenvector v", "A v", "lambda v"], rows, "rlll")
            + f"\n\nDot product of the two eigenvectors: {_g(_dot(vecs[0], vecs[1]))}; "
              f"angle between them {_g(math.degrees(math.acos(_dot(vecs[0], vecs[1]) / (_norm(vecs[0]) * _norm(vecs[1])))))} degrees.")


def rotation_has_no_real_eigenvalue():
    A = [[0, -1], [1, 0]]
    tr, d = 0, 1
    disc = tr * tr - 4 * d
    return (f"The quarter turn A = {_m(A)}: trace {tr}, det {d}, characteristic polynomial lambda^2 + 1; "
            f"discriminant {disc} < 0, so no real lambda.\n\n"
            + _table(["x", "A x", "angle between x and A x"],
                     [(_v(x), _v(_matvec(A, x)), "90 degrees") for x in ([1, 0], [1, 1], [-2, 3])]))


def matrix_power():
    A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    # eigenpairs 3 with (1, 1), 1 with (1, -1); V = [[1, 1], [1, -1]], V^-1 = V / 2
    V = [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(-1)]]
    Vi = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(-1, 2)]]

    def mul(P, Q):
        return [[sum(P[i][l] * Q[l][j] for l in range(2)) for j in range(2)] for i in range(2)]

    rows = []
    P = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    for k in range(1, 11):
        P = mul(P, A)
        if k in (1, 2, 5, 10):
            D = [[Fraction(3) ** k, Fraction(0)], [Fraction(0), Fraction(1)]]
            Q = mul(mul(V, D), Vi)
            rows.append((k, _m(P), _m(Q), "yes" if P == Q else "NO"))
    return ("A = [[2, 1], [1, 2]], A = V diag(3, 1) V^-1 with V = [[1, 1], [1, -1]] (eigenvectors as columns).\n\n"
            + _table(["k", "A^k by k multiplications", "V diag(3^k, 1^k) V^-1", "equal"], rows, "rllr"))


def power_iteration_trace():
    A = TRI3
    vals, vecs = _eigh(A)
    v1 = vecs[0]
    x = [1.0, 0.0, 0.0]
    rows = []
    for k in range(0, 11):
        Ax = _matvec(A, x)
        lam = _dot(x, Ax)
        err = _norm([a - b for a, b in zip(x, v1)])
        rows.append((k, _v(x, 5), _g(lam, 6), _sci(vals[0] - lam), _sci(err)))
        nrm = _norm(Ax)
        x = [c / nrm for c in Ax]
    return (f"A = {_m(A)}, x_0 = (1, 0, 0). The true top eigenpair is lambda_1 = 2 + sqrt 2 = {_g(vals[0], 6)}, "
            f"v_1 = (1, sqrt 2, 1)/2 = {_v(v1, 5)}; the other eigenvalues are {_g(vals[1], 6)} and {_g(vals[2], 6)}.\n\n"
            + _table(["k", "x_k", "Rayleigh quotient x_k . A x_k", "lambda_1 - Rayleigh", "|x_k - v_1|"], rows, "rlrrr"))


def _power_errors(A, x0, steps):
    """Distance from each normalised power-iteration iterate to the dominant eigenvector."""
    vals, vecs = _eigh(A)
    v1 = vecs[0]
    if _dot(x0, v1) < 0:
        v1 = [-c for c in v1]
    x = [c / _norm(x0) for c in x0]
    errs = []
    for _ in range(steps + 1):
        errs.append(_norm([a - b for a, b in zip(x, v1)]))
        y = _matvec(A, x)
        nrm = _norm(y)
        x = [c / nrm for c in y]
    return errs, vals


def _shifted(A, s):
    return [[A[i][j] + (s if i == j else 0.0) for j in range(len(A))] for i in range(len(A))]


SHIFTS = (-1.0, 0.0, 2.0)


def power_iteration_rates():
    rows = []
    for s in SHIFTS:
        A = _shifted(TRI3, s)
        errs, vals = _power_errors(A, [1.0, 0.0, 0.0], 40)
        # measured per-step ratio between iterations 10 and 20
        measured = (errs[20] / errs[10]) ** (1 / 10)
        # the second ratio uses the larger of |lambda_2|, |lambda_3|
        ratio = max(abs(vals[1]), abs(vals[2])) / abs(vals[0])
        rows.append((_g(s), ", ".join(_g(v, 4) for v in vals), _g(ratio, 4), _g(measured, 4),
                     _g(math.log(1e-8) / math.log(ratio), 3)))
    return ("A + s I for the tridiagonal matrix above, started at x_0 = (1, 0, 0).\n\n"
            + _table(["shift s", "eigenvalues of A + s I", "|lambda_2| / |lambda_1|", "measured error ratio per step",
                      "steps to reach error 1e-8 (predicted)"], rows, "rlrrr"))


# --------------------------------------------- symmetric matrices, orthogonality

def symmetric_orthogonal():
    A = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
    vals, vecs = _eigh(A)
    Q = _T(vecs)  # eigenvectors as columns
    QtQ = _matmul(_T(Q), Q)
    D = [[vals[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    rec = _matmul(_matmul(Q, D), _T(Q))
    lines = [f"A = {_m(A)} (symmetric).", ""]
    rows = []
    for l, v in zip(vals, vecs):
        rows.append((_g(l, 5), _v(v, 4), _v(_matvec(A, v), 4), _v([l * c for c in v], 4)))
    lines.append(_table(["lambda", "unit eigenvector v", "A v", "lambda v"], rows, "rlll"))
    dots = [(i, j, _dot(vecs[i], vecs[j])) for i in range(3) for j in range(i + 1, 3)]
    lines += ["", "Pairwise dot products of the eigenvectors: "
              + ", ".join(f"v{i}.v{j} = {_g(d, 3)}" for i, j, d in dots) + ".",
              f"Q^T Q = {_m(QtQ, 3)}.",
              f"Largest entry of |Q diag(lambda) Q^T - A|: {_sci(max(abs(rec[i][j] - A[i][j]) for i in range(3) for j in range(3)))}."]
    return "\n".join(lines)


def orthogonal_matrix():
    F = Fraction
    Q = [[F(1, 3), F(2, 3), F(2, 3)], [F(2, 3), F(1, 3), F(-2, 3)], [F(2, 3), F(-2, 3), F(1, 3)]]
    Qt = _T(Q)
    QtQ = _matmul(Qt, Q)
    x, y = [F(3), F(0), F(4)], [F(1), F(2), F(-1)]
    Qx, Qy = _matvec(Q, x), _matvec(Q, y)
    rows = [
        ("column norms squared", ", ".join(_g(sum(Q[i][j] ** 2 for i in range(3))) for j in range(3))),
        ("column dot products (1,2), (1,3), (2,3)",
         ", ".join(_g(sum(Q[i][a] * Q[i][b] for i in range(3))) for a, b in ((0, 1), (0, 2), (1, 2)))),
        ("Q^T Q", _m(QtQ)),
        ("Q Q^T", _m(_matmul(Q, Qt))),
        ("x", _v(x)),
        ("Q x", _v(Qx)),
        ("|x|^2 and |Q x|^2", f"{_g(_dot(x, x))} and {_g(_dot(Qx, Qx))}"),
        ("y", _v(y)),
        ("x . y and (Q x) . (Q y)", f"{_g(_dot(x, y))} and {_g(_dot(Qx, Qy))}"),
        ("det Q", _g(_det3(Q))),
    ]
    return "Q = (1/3) [[1, 2, 2], [2, 1, -2], [2, -2, 1]].\n\n" + _table(["quantity", "value"], rows, "ll")


def rotation_matrix_table():
    rows = []
    for deg in (0, 30, 45, 90, 180):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        R = [[c, -s], [s, c]]
        e1, p = _matvec(R, [1, 0]), _matvec(R, [2, 1])
        rows.append((deg, _m(R, 4), _v(e1, 4), _v(p, 4), _g(_det2(R), 4), _g(_norm(p), 5)))
    return ("R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]], applied to e1 = (1, 0) and p = (2, 1), |p| = "
            + _g(math.sqrt(5), 5) + ".\n\n"
            + _table(["theta (degrees)", "R(theta)", "R e1", "R p", "det R", "|R p|"], rows, "rlllrr"))


def rotation_composition():
    def R(deg):
        t = math.radians(deg)
        return [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]

    a, b = 20, 25
    P = _matmul(R(a), R(b))
    Q = _matmul(R(b), R(a))
    W = R(a + b)
    err = max(abs(P[i][j] - W[i][j]) for i in range(2) for j in range(2))
    t1, t2 = math.radians(a), math.radians(b)
    rows = [
        ("R(20) R(25)", _m(P, 6)),
        ("R(25) R(20)", _m(Q, 6)),
        ("R(45)", _m(W, 6)),
        ("cos 20 cos 25 - sin 20 sin 25", _g(math.cos(t1) * math.cos(t2) - math.sin(t1) * math.sin(t2), 6)),
        ("cos 45", _g(math.cos(math.radians(45)), 6)),
        ("sin 20 cos 25 + cos 20 sin 25", _g(math.sin(t1) * math.cos(t2) + math.cos(t1) * math.sin(t2), 6)),
        ("sin 45", _g(math.sin(math.radians(45)), 6)),
        ("largest |R(20) R(25) - R(45)| entry", _sci(err)),
        ("R(30)^T", _m(_T(R(30)), 5)),
        ("R(-30)", _m(R(-30), 5)),
        ("R(30)^T R(30)", _m(_matmul(_T(R(30)), R(30)), 5)),
    ]
    return _table(["expression", "value"], rows, "ll")


def rotation_relative_position():
    def R(t):
        return [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]

    def rotate_pairs(x, pos, omegas):
        out = []
        for k, w in enumerate(omegas):
            out += _matvec(R(pos * w), x[2 * k:2 * k + 2])
        return out

    omegas = (0.5, 0.05)
    q = [0.6, 0.8, 1.0, -0.5]
    k = [1.0, 0.5, -0.4, 0.9]
    rows = []
    for p, s in ((2, 5), (10, 13), (100, 103), (7, 3), (50, 50)):
        d = _dot(rotate_pairs(q, p, omegas), rotate_pairs(k, s, omegas))
        rows.append((p, s, s - p, _g(d, 6)))
    # relative form: sum over pairs of q_k^T R((s - p) w_k) k_k
    rel = []
    for delta in (3, -4, 0):
        tot = 0.0
        for idx, w in enumerate(omegas):
            tot += _dot(q[2 * idx:2 * idx + 2], _matvec(R(delta * w), k[2 * idx:2 * idx + 2]))
        rel.append((delta, _g(tot, 6)))
    return ("q = (0.6, 0.8, 1.0, -0.5), k = (1.0, 0.5, -0.4, 0.9) in two coordinate pairs; pair 0 is rotated by "
            "position * 0.5 radians, pair 1 by position * 0.05 radians. The dot product of the rotated vectors:\n\n"
            + _table(["position of q", "position of k", "offset s - p", "rotated q . rotated k"], rows)
            + "\n\nThe same quantity computed with a single rotation by the offset, "
              "sum over pairs of q_pair . R(offset * omega) k_pair:\n\n"
            + _table(["offset", "value"], rel))


# --------------------------------------------------------------------- SVD

def svd_by_hand():
    A = SVD_A
    G = _matmul(_T(A), A)
    vals, vecs = _eigh(G)
    sig = [math.sqrt(v) for v in vals]
    U = []
    for s, v in zip(sig, vecs):
        Av = _matvec(A, v)
        U.append([c / s for c in Av])
    lines = [f"A = {_m(A)}, A^T A = {_m(G)}.",
             f"det(A^T A - lambda I) = (25 - lambda)^2 - 400 = lambda^2 - 50 lambda + 225 = 0, so lambda = (50 +- 40)/2 = "
             f"{_g(vals[0], 5)} and {_g(vals[1], 5)}.",
             ""]
    rows = []
    for i in range(2):
        rows.append((i + 1, _g(vals[i], 5), _g(sig[i], 5), _v(vecs[i], 4), _v(_matvec(A, vecs[i]), 4), _v(U[i], 4),
                     _g(_norm(U[i]), 4)))
    lines.append(_table(["i", "lambda_i", "sigma_i = sqrt lambda_i", "v_i", "A v_i", "u_i = A v_i / sigma_i", "|u_i|"],
                        rows, "rrrlllr"))
    rec = _matmul(_matmul(_T(U), [[sig[0], 0], [0, sig[1]]]), _T(vecs))
    lines += ["", f"u_1 . u_2 = {_g(_dot(U[0], U[1]), 3)}; v_1 . v_2 = {_g(_dot(vecs[0], vecs[1]), 3)}.",
              f"U Sigma V^T = {_m(rec, 4)}.",
              f"sigma_1 sigma_2 = {_g(sig[0] * sig[1], 5)}, |det A| = {_g(abs(_det2(A)), 5)}."]
    return "\n".join(lines)


def _svd_circle_stages(A=None, n=24):
    """For points on the unit circle: x, V^T x, Sigma V^T x, U Sigma V^T x (the last equals A x)."""
    A = A or SVD_A
    G = _matmul(_T(A), A)
    vals, vecs = _eigh(G)
    sig = [math.sqrt(v) for v in vals]
    U = [[c / s for c in _matvec(A, v)] for s, v in zip(sig, vecs)]
    stages = []
    for k in range(n + 1):
        t = 2 * math.pi * k / n
        x = [math.cos(t), math.sin(t)]
        z1 = [_dot(vecs[0], x), _dot(vecs[1], x)]
        z2 = [sig[0] * z1[0], sig[1] * z1[1]]
        z3 = [U[0][r] * z2[0] + U[1][r] * z2[1] for r in range(2)]
        stages.append((x, z1, z2, z3))
    return sig, vecs, U, stages


def svd_rotate_stretch_rotate():
    sig, vecs, U, stages = _svd_circle_stages(n=8)
    rows = []
    for x, z1, z2, z3 in stages[:8]:
        rows.append((_v(x, 3), _v(z1, 3), _v(z2, 3), _v(z3, 3), _v(_matvec(SVD_A, x), 3)))
    return ("Unit vectors x pushed through the three factors of A = U Sigma V^T for A = [[3, 0], [4, 5]]: "
            "V^T x are the coordinates of x along v_1, v_2; Sigma stretches them; U places them along u_1, u_2.\n\n"
            + _table(["x", "V^T x", "Sigma V^T x", "U Sigma V^T x", "A x"], rows, "lllll"))


def stretch_scan():
    A = SVD_A
    best = (0, 0)
    worst = (None, 1e9)
    for deg in range(0, 360):
        t = math.radians(deg)
        n = _norm(_matvec(A, [math.cos(t), math.sin(t)]))
        if n > best[1]:
            best = (deg, n)
        if n < worst[1]:
            worst = (deg, n)
    sig = [math.sqrt(v) for v in _eigh(_matmul(_T(A), A))[0]]
    return (_table(["", "angle (degrees)", "|A x|", "singular value"],
                   [("longest image", best[0], _g(best[1], 5), _g(sig[0], 5)),
                    ("shortest image", worst[0], _g(worst[1], 5), _g(sig[1], 5))])
            + "\n\nScan of |A x| over x at every whole degree from 0 to 359. The longest image lies along v_1 (45 degrees); the shortest lies along the line of v_2, whose upper end is at 135 degrees.")


def svd_shapes_and_rank():
    out = []
    for name, A in (("A = [[1, 0], [0, 1], [1, 1]] (shape [3, 2])", [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                    ("B = [[1, 2], [2, 4]] (shape [2, 2], second row twice the first)", [[1.0, 2.0], [2.0, 4.0]]),
                    ("C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] (chapter 4's singular matrix)",
                     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])):
        U, s, V = _svd(A)
        rank = sum(1 for x in s if x > 1e-9 * s[0])
        out.append((name, _v(s, 5), rank))
    G = _matmul(_T([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    U, s, V = _svd([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    shapes = _table(["factor", "shape (full SVD)", "shape (thin SVD)"],
                    [("U", "[3, 3]", "[3, 2]"), ("Sigma", "[3, 2]", "[2, 2]"), ("V^T", "[2, 2]", "[2, 2]")], "lll")
    return (_table(["matrix", "singular values", "rank = number of nonzero singular values"], out, "lll")
            + f"\n\nFor A the Gram matrix A^T A = {_m(G)} has eigenvalues 3 and 1, so sigma = (sqrt 3, 1) = {_v(s, 5)}; "
              f"u_1 = {_v(U[0], 4)}, u_2 = {_v(U[1], 4)}.\n\n" + shapes)


def lauchli_squaring():
    eps = 1e-8
    A = [[1.0, 1.0], [eps, 0.0], [0.0, eps]]
    G = _matmul(_T(A), A)
    vals, _ = _eigh(G)
    via_gram = [math.sqrt(max(v, 0.0)) for v in vals]
    _, direct, _ = _svd(A)
    return (f"A = [[1, 1], [1e-8, 0], [0, 1e-8]]. The exact singular values are sqrt(2 + 1e-16) and 1e-8.\n\n"
            + _table(["route", "sigma_1", "sigma_2"],
                     [("eigenvalues of A^T A (float64)", _g(via_gram[0], 8), _sci(via_gram[1])),
                      ("one-sided Jacobi on A itself", _g(direct[0], 8), _sci(direct[1]))], "lrr")
            + f"\n\nIn float64, A^T A = {_m(G, 17)}: the 1 + 1e-16 on its diagonal rounds to 1, the two rows become equal, "
              "and the small singular value is lost.")


# ------------------------------------------------------- low-rank structure

def rank_one_pieces():
    A = SVD_A
    U, s, V = _svd(A)
    A1 = _truncate(U, s, V, 1)
    piece2 = _truncate([U[1]], [s[1]], [V[1]], 1)
    diff = _sub(A, A1)
    return (f"A = {_m(A)} = sigma_1 u_1 v_1^T + sigma_2 u_2 v_2^T.\n\n"
            + _table(["piece", "matrix", "rank"],
                     [("sigma_1 u_1 v_1^T", _m(A1, 5), 1), ("sigma_2 u_2 v_2^T", _m(piece2, 5), 1),
                      ("sum", _m([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(A1, piece2)], 5), 2),
                      ("A - A_1", _m(diff, 5), 1)], "lrr")
            + f"\n\n|A - A_1|_F = {_g(_fro(diff), 5)} = sigma_2 = {_g(s[1], 5)}; the largest stretch of A - A_1 is also {_g(_spectral_norm(diff), 5)}.")


def spectrum_table():
    M = spectrum_matrix()
    U, s, V = _svd(M)
    total = sum(x * x for x in s)
    rows = []
    cum = 0.0
    for i, x in enumerate(s):
        cum += x * x
        rows.append((i + 1, _g(x, 5), _g(100 * x * x / total, 4), _g(100 * cum / total, 5)))
    return ("The 8 x 6 table M (three smooth patterns plus noise, defined in the text).\n\n"
            + _table(["i", "sigma_i", "share of |M|_F^2 (%)", "cumulative share (%)"], rows)
            + f"\n\n|M|_F = {_g(_fro(M), 5)}, sqrt(sum sigma_i^2) = {_g(math.sqrt(total), 5)}.")


def truncation_error():
    M = spectrum_matrix()
    U, s, V = _svd(M)
    m, n = len(M), len(M[0])
    rows = []
    for r in range(0, 7):
        Ar = _truncate(U, s, V, r)
        E = _sub(M, Ar)
        fro = _fro(E)
        tail = math.sqrt(sum(x * x for x in s[r:]))
        spec = _spectral_norm(E) if r < 6 else 0.0
        nxt = s[r] if r < 6 else 0.0
        rows.append((r, r * (m + n), _g(fro, 5), _g(tail, 5), _g(spec, 5), _g(nxt, 5)))
    return (f"M is {m} x {n}: storing it takes m n = {m * n} numbers; a rank-r approximation as two factors takes r (m + n) = r * {m + n}.\n\n"
            + _table(["r", "numbers stored r(m+n)", "|M - M_r|_F", "sqrt(sum of dropped sigma^2)",
                      "largest stretch of M - M_r", "sigma_(r+1)"], rows))


def eckart_young_check():
    M = spectrum_matrix()
    m, n = len(M), len(M[0])
    U, s, V = _svd(M)
    name = "decompositions.eckart_young_check"
    rng = bookrng.generator(name)  # one generator, never reseeded, shared by every r
    rows = []
    for r in (1, 2, 3, 4):
        opt = _fro(_sub(M, _truncate(U, s, V, r)))
        # (a) project onto the span of the r longest columns of M
        cols = [[M[i][j] for i in range(m)] for j in range(n)]
        order = sorted(range(n), key=lambda j: -_norm(cols[j]))[:r]
        a = _fro(_sub(M, _project_columns(M, _gram_schmidt([cols[j] for j in order]))))
        # (b) truncated SVD of M plus noise of size 0.3
        N = [[M[i][j] + 0.6 * (rng.uniform() - 0.5) for j in range(n)] for i in range(m)]
        Un, sn, Vn = _svd(N)
        b = _fro(_sub(M, _truncate(Un, sn, Vn, r)))
        # (c) best of 2000 random r-dimensional column spaces
        best = 1e18
        for _ in range(2000):
            basis = _gram_schmidt([[rng.uniform() - 0.5 for _ in range(m)] for _ in range(r)])
            best = min(best, _fro(_sub(M, _project_columns(M, basis))))
        rows.append((r, _g(opt, 5), _g(a, 5), _g(b, 5), _g(best, 5)))
    seed = bookrng.seed_of(name)
    return (f"Frobenius error |M - X|_F for rank-r matrices X built four ways (smaller is better), printed to 5 significant digits.\n\n"
            f"Column 2: orthonormalise (Gram-Schmidt, in order of decreasing column length) the r columns of M with the largest Euclidean norm, "
            f"and project M onto their span in R^8. Columns 3 and 4 draw from ONE SplitMix64 generator with seed {seed} "
            f"(0x{seed:016X}), seeded once and never reseeded, used for r = 1, 2, 3, 4 in turn. For each r: first the 48 noise values "
            f"0.6 * (uniform() - 0.5), row by row (row 0 first), added to M for column 3; then 2000 random subspaces for column 4, each "
            f"made of r vectors of 8 entries (uniform() - 0.5), drawn vector by vector and orthonormalised by Gram-Schmidt in the order drawn; "
            f"M is projected on the span and the smallest error of the 2000 is kept.\n\n"
            + _table(["r", "truncated SVD of M", "project on the r longest columns", "truncated SVD of M + noise",
                      "best of 2000 random subspaces"], rows))


def lora_parameter_counts():
    rows = []
    for d_out, d_in, r in ((4096, 4096, 8), (4096, 4096, 16), (4096, 4096, 64), (4096, 11008, 16), (1024, 1024, 8)):
        full = d_out * d_in
        low = r * (d_out + d_in)
        rows.append((f"[{d_out}, {d_in}]", r, f"{full:,}", f"{low:,}", _g(full / low, 4), _g(100 * low / full, 3)))
    return _table(["W shape", "r", "m n", "r (m + n)", "ratio m n / r (m + n)", "adapter size (% of W)"], rows)


def lora_zero_start():
    """W + B A with B = 0 is exactly W, and B A never has rank above r."""
    W = [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [2.0, 0.0, 1.0, 0.0]]  # [3, 4]
    A = [[0.5, -1.0, 0.0, 1.0], [1.0, 0.0, 0.5, 0.0]]  # [r=2, 4]
    B0 = [[0.0, 0.0] for _ in range(3)]  # [3, 2]
    B = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    x = [1.0, 2.0, 3.0, 4.0]
    base = _matvec(W, x)
    with0 = _matvec([[w + d for w, d in zip(rw, rd)] for rw, rd in zip(W, _matmul(B0, A))], x)
    D = _matmul(B, A)
    withB = _matvec([[w + d for w, d in zip(rw, rd)] for rw, rd in zip(W, D)], x)
    split = [a + b for a, b in zip(base, _matvec(B, _matvec(A, x)))]
    _, s, _ = _svd(D)
    rank = sum(1 for v in s if v > 1e-9 * s[0])
    rows = [("W x", _v(base)), ("(W + B_0 A) x with B_0 = 0", _v(with0)), ("B A", _m(D)),
            ("(W + B A) x", _v(withB)), ("W x + B (A x), two thin products", _v(split)),
            ("singular values of B A", _v(s, 4)), ("rank of B A (r = 2)", rank)]
    return "W [3, 4], A [2, 4], B [3, 2], x = (1, 2, 3, 4).\n\n" + _table(["quantity", "value"], rows, "ll")
