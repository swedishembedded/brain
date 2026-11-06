# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Matrices as Linear Maps".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: matrices.<name> -->` markers. Functions whose names start with an
underscore (or that take arguments) are shared with the chapter's figures, so
a picture and the table beside it cannot disagree.

A matrix here is a list of rows, each row a list of numbers. Exact examples use
`fractions.Fraction`, so an elimination step prints 3/2 and not 1.5000000000000002.

Swedish Embedded AB implements solutions for trustworthy linear algebra in
numerical software. If your team needs expertise in matrix computation and
its cost, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from fractions import Fraction

F = Fraction


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _q(v, digits=4):
    """A number as text: exact fractions as a/b, floats compactly."""
    if isinstance(v, str):
        return v
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, int):
        return str(v)
    if abs(v) < 1e-12:
        return "0"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def _vec(x):
    return "(" + ", ".join(_q(c) for c in x) + ")"


def _mat(A):
    """A matrix on one line: [[1, 2], [3, 4]]."""
    return "[" + ", ".join("[" + ", ".join(_q(c) for c in row) + "]" for row in A) + "]"


def _rows_text(A, labels=None, sep_col=None):
    """A matrix as aligned text rows; a bar before column `sep_col` marks an augmented system."""
    cells = [[_q(c) for c in row] for row in A]
    width = max(len(c) for row in cells for c in row)
    lines = []
    for i, row in enumerate(cells):
        parts = []
        for j, c in enumerate(row):
            if sep_col is not None and j == sep_col:
                parts.append("|")
            parts.append(c.rjust(width))
        lines.append(("R%d  " % (i + 1) if labels else "") + "[ " + " ".join(parts) + " ]")
    return "\n".join(lines)


def _shape(A):
    return (len(A), len(A[0]))


def _frac(A):
    return [[F(c) for c in row] for row in A]


# ---------------------------------------------------- the four operations

def _matvec(A, x):
    """Row view: entry i is the dot product of row i with x."""
    assert len(A[0]) == len(x), "matrix-vector shape mismatch"
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def _matvec_columns(A, x):
    """Column view: the sum of the columns of A, each weighted by an entry of x."""
    m, n = _shape(A)
    assert n == len(x)
    out = [0] * m
    for j in range(n):
        for i in range(m):
            out[i] += x[j] * A[i][j]
    return out


def _matmul(A, B):
    """C[i][j] = sum_l A[i][l] * B[l][j]; the column j of C is A times column j of B."""
    assert len(A[0]) == len(B), "matmul shape mismatch"
    return [[sum(A[i][l] * B[l][j] for l in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _identity(n):
    return [[F(1) if i == j else F(0) for j in range(n)] for i in range(n)]


def _matmul_counted(A, B):
    """Matrix product that also counts its multiply-adds."""
    count = 0
    C = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            acc = 0
            for l in range(len(B)):
                acc += A[i][l] * B[l][j]
                count += 1
            row.append(acc)
        C.append(row)
    return C, count


def _lcg_matrix(m, n, seed):
    """Deterministic small integers in [-4, 4] (no random module, so the example is repeatable)."""
    s = seed
    out = []
    for _ in range(m):
        row = []
        for _ in range(n):
            s = (s * 1103515245 + 12345) % (2 ** 31)
            row.append((s >> 16) % 9 - 4)
        out.append(row)
    return out


# the small matrices reused through the chapter
LAYER_W = [[2, -1, 1], [1, 3, 2]]  # [2, 3]: three numbers in, two out
LAYER_X = [1, 2, 3]
MAP_A = [[1, 2], [0, 1], [3, -1]]  # [3, 2]: R^2 -> R^3
MAP_B = [[2, 0, 1], [1, 1, -1]]    # [2, 3]: R^3 -> R^2
SQUARE_A = [[2, 1], [F(1, 2), F(3, 2)]]  # the 2-D map of the figure; columns (2, 1/2) and (1, 3/2)
SQUASH_A = [[1, 2], [F(1, 2), 1]]  # the 2-D map that collapses the square onto a line (column 2 is 2 times column 1)


# ------------------------------------------------------------ worked blocks

def matvec_two_views():
    A, x = LAYER_W, LAYER_X
    rows = [(f"row {i}", _vec(A[i]), " + ".join(f"{_q(a)}*{_q(b)}" for a, b in zip(A[i], x)),
             _q(sum(a * b for a, b in zip(A[i], x)))) for i in range(len(A))]
    out = [f"A = {_mat(A)}, shape {list(_shape(A))}; x = {_vec(x)}.", "",
           "Row view: entry i of Ax is row i of A dotted with x.", "",
           _table(["entry", "row of A", "dot with x", "value"], rows, "lllr"), ""]
    cols = [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
    scaled = [[x[j] * c for c in cols[j]] for j in range(len(cols))]
    rows2 = [(f"column {j}", _vec(cols[j]), _q(x[j]), _vec(scaled[j])) for j in range(len(cols))]
    total = [sum(s[i] for s in scaled) for i in range(len(A))]
    out += ["Column view: Ax is the sum of the columns of A, column j weighted by x[j].", "",
            _table(["term", "column of A", "weight x[j]", "weighted column"], rows2, "lllr"), "",
            f"Sum of the weighted columns: {_vec(total)}. Row view: {_vec(_matvec(A, x))}."]
    assert total == _matvec(A, x) == _matvec_columns(A, x)
    return "\n".join(out)


def linearity_check():
    A = LAYER_W
    x, y, c = [1, 2, 3], [0, -1, 4], 3
    xy = [a + b for a, b in zip(x, y)]
    cx = [c * a for a in x]
    rows = [
        ("A(x + y)", _vec(_matvec(A, xy))),
        ("Ax + Ay", _vec([a + b for a, b in zip(_matvec(A, x), _matvec(A, y))])),
        (f"A({c}x)", _vec(_matvec(A, cx))),
        (f"{c}(Ax)", _vec([c * a for a in _matvec(A, x)])),
    ]
    out = [f"x = {_vec(x)}, y = {_vec(y)}, c = {c}.", "", _table(["expression", "value"], rows, "lr"), ""]
    b = [10, -5]

    def affine(v):
        return [a + s for a, s in zip(_matvec(A, v), b)]

    def relu(v):
        return [max(0, a) for a in v]

    f_x, f_y, f_xy = affine(x), affine(y), affine(xy)
    out += [f"Add a shift b = {_vec(b)}, f(v) = Av + b:", "",
            _table(["expression", "value"], [
                ("f(x + y)", _vec(f_xy)),
                ("f(x) + f(y)", _vec([a + s for a, s in zip(f_x, f_y)])),
                ("f(0)", _vec(affine([0, 0, 0]))),
            ], "lr"), ""]
    u, w = [1, -2], [-3, 1]
    uw = [a + s for a, s in zip(u, w)]
    out += [f"ReLU (negative entries become 0) on u = {_vec(u)}, w = {_vec(w)}: "
            f"relu(u + w) = {_vec(relu(uw))}, relu(u) + relu(w) = {_vec([a + s for a, s in zip(relu(u), relu(w))])}."]
    assert _matvec(A, xy) == [a + b_ for a, b_ in zip(_matvec(A, x), _matvec(A, y))]
    return "\n".join(out)


def _unit_square_images(A):
    """Images of the corners of the unit square, and the area of the image by the shoelace formula."""
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    imgs = [_matvec(A, c) for c in corners]
    area2 = sum(imgs[i][0] * imgs[(i + 1) % 4][1] - imgs[(i + 1) % 4][0] * imgs[i][1] for i in range(4))
    return corners, imgs, F(area2) / 2


def columns_are_images():
    A = SQUARE_A
    corners, imgs, area = _unit_square_images(A)
    cols = [[A[0][j], A[1][j]] for j in range(2)]
    rows = [(_vec(c), f"{_q(c[0])}*{_vec(cols[0])} + {_q(c[1])}*{_vec(cols[1])}", _vec(i)) for c, i in zip(corners, imgs)]
    for c, i in zip(corners, imgs):
        assert i == [c[0] * cols[0][r] + c[1] * cols[1][r] for r in range(2)]
    return "\n".join([
        f"A = {_mat(A)}. A e1 = {_vec(cols[0])} and A e2 = {_vec(cols[1])} are its columns.", "",
        _table(["corner x", "x[0] A e1 + x[1] A e2", "A x"], rows, "lll"), "",
        f"The image of the unit square is a parallelogram of area {_q(abs(area))} "
        f"(shoelace formula over the four images); the unit square has area 1.",
    ])


def composition_is_product():
    A, B = MAP_A, MAP_B  # f(x) = A x : R^2 -> R^3 ; g(y) = B y : R^3 -> R^2
    e = [[1, 0], [0, 1]]
    steps = []
    for j, ej in enumerate(e):
        fx = _matvec(A, ej)
        steps.append((f"e{j + 1} = {_vec(ej)}", _vec(fx), _vec(_matvec(B, fx))))
    C = _matmul(B, A)
    out = [f"f(x) = Ax with A = {_mat(A)} (shape [3, 2]); g(y) = By with B = {_mat(B)} (shape [2, 3]).", "",
           "Feed the unit vectors through f, then g. The results are the columns of the matrix of g after f.", "",
           _table(["input", "f(input) = A input", "g(f(input))"], steps, "lll"), "",
           f"So that matrix has columns (5, -2) and (3, 4): C = {_mat(C)}, shape [2, 2], and C[i][j] = sum over l of B[i][l] A[l][j].", ""]
    x = [2, -1]
    two = _matvec(B, _matvec(A, x))
    one = _matvec(C, x)
    out += [f"Check on x = {_vec(x)}: g(f(x)) = {_vec(two)}, Cx = {_vec(one)}."]
    assert two == one
    # entry-by-entry formula, one entry shown with its terms
    i, j = 0, 1
    terms = " + ".join(f"{B[i][l]}*{A[l][j]}" for l in range(3))
    out += ["", f"Entry C[0][1] = {terms} = {_q(C[i][j])}."]
    assert C[i][j] == sum(B[i][l] * A[l][j] for l in range(3))
    return "\n".join(out)


def shapes_and_cost():
    shapes = [(2, 3, 4), (1, 512, 512), (512, 1, 512), (512, 512, 512), (128, 4096, 4096)]
    rows = []
    for m, k, n in shapes:
        ma = m * n * k
        rows.append((f"[{m}, {k}] x [{k}, {n}]", f"[{m}, {n}]", f"{m * n:,}", f"{k:,}", f"{ma:,}", f"{2 * ma:,}"))
    A, B = _lcg_matrix(2, 3, 7), _lcg_matrix(3, 4, 11)
    _, counted = _matmul_counted(A, B)
    assert counted == 2 * 3 * 4
    return "\n".join([
        _table(["product", "result shape", "outputs m*n", "terms per output k", "multiply-adds m*n*k", "FLOP 2*m*n*k"], rows),
        "",
        f"Counting the inner-loop iterations of a plain triple loop on a [2, 3] x [3, 4] example gives {counted} "
        f"multiply-adds, the value of m*n*k = 2*3*4.",
    ])


def association_costs():
    def cost(shapes_left, shapes_right):
        (a, b), (c, d) = shapes_left, shapes_right
        assert b == c
        return a * b * d, (a, d)

    n = 512
    # (AB)x versus A(Bx)
    ab, ab_shape = cost((n, n), (n, n))
    abx, _ = cost(ab_shape, (n, 1))
    bx, bx_shape = cost((n, n), (n, 1))
    a_bx, _ = cost((n, n), bx_shape)
    rows = [("(A B) x", f"A B: [{n},{n}]x[{n},{n}] = {ab:,}", f"(AB) x: [{n},{n}]x[{n},1] = {abx:,}", f"{ab + abx:,}"),
            ("A (B x)", f"B x: [{n},{n}]x[{n},1] = {bx:,}", f"A (Bx): [{n},{n}]x[{n},1] = {a_bx:,}", f"{bx + a_bx:,}")]
    out = [f"A and B are [{n}, {n}], x is [{n}, 1]. Cost in multiply-adds:", "",
           _table(["order", "first product", "second product", "total"], rows, "llll"),
           "", f"The ratio of the two totals is {(ab + abx) / (bx + a_bx):.1f}.", ""]
    # a chain of three matrices
    s = [(10, 100), (100, 5), (5, 50)]
    c1, sh1 = cost(s[0], s[1])
    c2, _ = cost(sh1, s[2])
    d1, sh2 = cost(s[1], s[2])
    d2, _ = cost(s[0], sh2)
    rows2 = [("(A B) C", f"A B -> [10, 5]: {c1:,}", f"(AB) C -> [10, 50]: {c2:,}", f"{c1 + c2:,}"),
             ("A (B C)", f"B C -> [100, 50]: {d1:,}", f"A (BC) -> [10, 50]: {d2:,}", f"{d1 + d2:,}")]
    out += ["A is [10, 100], B is [100, 5], C is [5, 50]:", "",
            _table(["order", "first product", "second product", "total"], rows2, "llll"), ""]
    # the two orders give the same answer, checked exactly on small integer matrices
    A, B, C = _lcg_matrix(3, 4, 1), _lcg_matrix(4, 2, 2), _lcg_matrix(2, 5, 3)
    left, right = _matmul(_matmul(A, B), C), _matmul(A, _matmul(B, C))
    assert left == right
    out += [f"On integer matrices of shapes [3,4], [4,2], [2,5] the two orders produce the same [3, 5] result "
            f"(first row {_vec(left[0])}), as associativity promises."]
    return "\n".join(out)


def noncommutative():
    A, B = [[1, 2], [3, 4]], [[0, 1], [1, 0]]
    ab, ba = _matmul(A, B), _matmul(B, A)
    out = [f"A = {_mat(A)}, B = {_mat(B)} (B swaps the two coordinates).", "",
           _table(["product", "value", "what happened"], [
               ("A B", _mat(ab), "columns of A swapped"),
               ("B A", _mat(ba), "rows of A swapped")], "lll"), ""]
    P, Q = MAP_B, MAP_A  # [2,3] and [3,2]
    out += [f"With P = {_mat(P)} of shape {list(_shape(P))} and Q = {_mat(Q)} of shape {list(_shape(Q))}: "
            f"P Q has shape {list(_shape(_matmul(P, Q)))} and Q P has shape {list(_shape(_matmul(Q, P)))}."]
    assert ab != ba
    return "\n".join(out)


def transpose_rules():
    A, B = MAP_A, MAP_B  # [3,2] and [2,3]
    AB = _matmul(A, B)   # [3,3]
    lhs = _transpose(AB)
    rhs = _matmul(_transpose(B), _transpose(A))
    wrong = _matmul(_transpose(A), _transpose(B))
    assert lhs == rhs
    x, y = [1, -1, 2], [3, 0]  # (Ax)... use a [2,3] matrix M: M x in R^2, so y in R^2
    M = B
    mx, mty = _matvec(M, x), _matvec(_transpose(M), y)
    lhs_dot = sum(a * b for a, b in zip(mx, y))
    rhs_dot = sum(a * b for a, b in zip(x, mty))
    assert lhs_dot == rhs_dot
    return "\n".join([
        f"A = {_mat(A)} [3, 2], B = {_mat(B)} [2, 3].", "",
        _table(["expression", "shape", "value"], [
            ("A B", "[3, 3]", _mat(AB)),
            ("(A B)^T", "[3, 3]", _mat(lhs)),
            ("B^T A^T", "[3, 3]", _mat(rhs)),
            ("A^T B^T", "[2, 2]", _mat(wrong)),
        ], "llr"), "",
        f"Adjoint rule for M = B, x = {_vec(x)}, y = {_vec(y)}: (Mx).y = {_vec(mx)}.{_vec(y)} = {lhs_dot}; "
        f"x.(M^T y) = {_vec(x)}.{_vec(mty)} = {rhs_dot}.",
    ])


def inverse_2x2():
    A = [[4, 7], [2, 6]]
    a, b, c, d = (F(v) for v in (A[0][0], A[0][1], A[1][0], A[1][1]))
    det = a * d - b * c
    inv = [[d / det, -b / det], [-c / det, a / det]]
    prod = _matmul(A, inv)
    prod2 = _matmul(inv, A)
    assert prod == prod2 == _identity(2)
    rhs = [1, 2]
    x = _matvec(inv, rhs)
    assert _matvec(A, x) == [F(v) for v in rhs]
    B = [[1, 1], [0, 1]]
    binv = [[1, -1], [0, 1]]
    ba = _matmul(B, A)
    right_order = _matmul(inv, binv)  # A^-1 B^-1
    wrong_order = _matmul(binv, inv)  # B^-1 A^-1
    assert _matmul(ba, right_order) == _identity(2)
    assert _matmul(ba, wrong_order) != _identity(2)
    return "\n".join([
        f"A = {_mat(A)}: ad - bc = {_q(a)}*{_q(d)} - {_q(b)}*{_q(c)} = {_q(det)}.", "",
        f"A^-1 = (1/{_q(det)}) {_mat([[d, -b], [-c, a]])} = {_mat(inv)}.", "",
        _table(["check", "value"], [
            ("A A^-1", _mat(prod)),
            ("A^-1 A", _mat(prod2)),
            ("I A", _mat(_matmul(_identity(2), A))),
            (f"x = A^-1 {_vec(rhs)}", _vec(x)),
            (f"A x", _vec(_matvec(A, x))),
        ], "lr"), "",
        f"Inverse of a product: with B = {_mat(B)} and B^-1 = {_mat(binv)}, (B A) times A^-1 B^-1 is "
        f"{_mat(_matmul(ba, right_order))}, but (B A) times B^-1 A^-1 is {_mat(_matmul(ba, wrong_order))}.",
    ])


# --------------------------------------------------------------- elimination

def _eliminate(M, ncols=None, pivoting="first", trace=None):
    """Forward elimination to row echelon form, in place on a list of rows of Fractions or floats.

    Only the first `ncols` columns are searched for pivots (the rest are an augmented right-hand side).
    pivoting: "first" takes the first nonzero entry in the column, "partial" the largest in magnitude.
    Returns the list of pivot columns; its length is the rank of the leading block.
    """
    m, n = len(M), len(M[0])
    ncols = n if ncols is None else ncols
    r, pivots = 0, []
    for c in range(ncols):
        if r == m:
            break
        cands = [i for i in range(r, m) if M[i][c] != 0]
        if not cands:
            continue
        p = cands[0] if pivoting == "first" else max(cands, key=lambda i: abs(M[i][c]))
        if p != r:
            M[r], M[p] = M[p], M[r]
            if trace is not None:
                trace.append((f"swap R{r + 1} and R{p + 1}", [row[:] for row in M]))
        for i in range(r + 1, m):
            if M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
                if trace is not None:
                    sign = "-" if f > 0 else "+"
                    trace.append((f"R{i + 1} <- R{i + 1} {sign} {_q(abs(f))} R{r + 1}", [row[:] for row in M]))
        pivots.append(c)
        r += 1
    return pivots


def _back_substitute(M, n):
    """Solve the square system held as an echelon augmented matrix [U | c] with n unknowns."""
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def elimination_trace():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    M = [[F(v) for v in row] + [F(bv)] for row, bv in zip(A, b)]
    trace = []
    shown = [("start", [row[:] for row in M])]
    piv = _eliminate(M, ncols=3, trace=trace)
    shown += trace
    out = ["The system 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3 as an augmented matrix [A | b]:", ""]
    for label, mat in shown:
        out += [f"{label}", "```text", _rows_text(mat, labels=True, sep_col=3), "```", ""]
    x = _back_substitute(M, 3)
    out += [f"Pivot columns {piv}: three pivots, so the system has a unique solution. "
            f"Back substitution from the last row up gives x = {_vec(x)}.", ""]
    check = _matvec(A, x)
    out += [f"Check: A x = {_vec(check)}, equal to b = {_vec(b)}."]
    assert check == [F(v) for v in b]
    return "\n".join(out)


def singular_example():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    M = [[F(v) for v in row] for row in A]
    trace = []
    start = [row[:] for row in M]
    piv = _eliminate(M, trace=trace)
    out = ["A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].", "", "start", "```text", _rows_text(start, labels=True), "```", ""]
    for label, mat in trace:
        out += [label, "```text", _rows_text(mat, labels=True), "```", ""]
    out += [f"Pivot columns {piv}: only {len(piv)} pivots, the last row is all zeros, rank = {len(piv)}.", ""]
    v1, v2 = [1, 1, 1], [1, -2, 1]
    rows = [("A (1, 1, 1)", _vec(_matvec(A, v1))), ("A (1, -2, 1)", _vec(_matvec(A, v2)))]
    t = 2
    vt = [a + t * b for a, b in zip(v1, v2)]
    rows.append((f"A ((1, 1, 1) + {t} (1, -2, 1)) = A {_vec(vt)}", _vec(_matvec(A, vt))))
    assert _matvec(A, v2) == [0, 0, 0]
    out += [_table(["product", "value"], rows, "lr"), "",
            "Reachable right-hand sides are those with b1 - 2 b2 + b3 = 0 (the elimination leaves "
            "b3 - 2 b2 + b1 in the last row):", ""]
    res = []
    for bv in ([6, 15, 24], [1, 2, 4]):
        Mb = [[F(v) for v in row] + [F(w)] for row, w in zip(A, bv)]
        _eliminate(Mb, ncols=3)
        last = Mb[2][3]
        res.append((_vec(bv), _q(bv[0] - 2 * bv[1] + bv[2]), _q(last),
                    "no solution" if last != 0 else "infinitely many solutions"))
    out.append(_table(["b", "b1 - 2 b2 + b3", "last row after elimination: 0 = ?", "outcome"], res, "lrrl"))
    return "\n".join(out)


def rank_survey():
    outer = [[a * b for b in (1, 2, -1)] for a in (1, 2, 3)]
    cases = [
        ("I (3 x 3)", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("[[1,2,3],[4,5,6],[7,8,9]]", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ("outer product (1,2,3)(1,2,-1)^T", outer),
        ("[[1,0,2,1],[0,1,1,3]]", [[1, 0, 2, 1], [0, 1, 1, 3]]),
        ("[[1,2],[2,4],[3,6]]", [[1, 2], [2, 4], [3, 6]]),
        ("zero (2 x 3)", [[0, 0, 0], [0, 0, 0]]),
    ]
    rows = []
    for name, A in cases:
        r1 = len(_eliminate(_frac(A)))
        r2 = len(_eliminate(_frac(_transpose(A))))
        assert r1 == r2 <= min(_shape(A))
        rows.append((name, f"{_shape(A)[0]} x {_shape(A)[1]}", r1, r2, min(_shape(A))))
    return _table(["matrix", "shape", "rank of A", "rank of A^T", "min(m, n)"], rows, "llrrr")


def layer_batch():
    W, b = LAYER_W, [F(1, 2), F(-1)]
    X = [[1, 2, 3], [0, 1, -1]]  # a batch of two examples, shape [2, 3]
    per_example = [[w + bb for w, bb in zip(_matvec(W, x), b)] for x in X]
    Y = _matmul(X, _transpose(W))
    Y = [[y + bb for y, bb in zip(row, b)] for row in Y]
    assert Y == per_example
    # homogeneous coordinates: append a 1 to x and the column b to W
    Wb = [row + [bb] for row, bb in zip(W, b)]
    aug = [_matvec(Wb, x + [1]) for x in X]
    assert aug == per_example
    return "\n".join([
        f"W = {_mat(W)} [2, 3], b = {_vec(b)}, batch X = {_mat(X)} [2, 3].", "",
        _table(["example", "x", "W x", "W x + b"], [
            (i, _vec(x), _vec(_matvec(W, x)), _vec(per_example[i])) for i, x in enumerate(X)], "lllr"), "",
        f"Batched: X W^T = {_mat(_matmul(X, _transpose(W)))}; adding b to every row gives Y = {_mat(Y)} [2, 2], "
        f"the same two rows.", "",
        f"One matrix instead: [W | b] = {_mat(Wb)} applied to (x, 1) gives {_mat(aug)}, again the same rows.",
    ])


def stacking_layers():
    A, B = MAP_A, MAP_B
    C = _matmul(B, A)
    xs = [[1, 0], [0, 1], [2, -1], [3, 5]]
    rows = []
    for x in xs:
        rows.append((_vec(x), _vec(_matvec(B, _matvec(A, x))), _vec(_matvec(C, x))))
    out = [f"Two linear layers x -> Ax -> B(Ax), with A = {_mat(A)}, B = {_mat(B)}, against the single matrix "
           f"C = BA = {_mat(C)}:", "",
           _table(["x", "B(A x)", "C x"], rows, "lll"), ""]

    def relu(v):
        return [max(0, a) for a in v]

    x, y = [1, 0], [-1, 0]
    xy = [a + c for a, c in zip(x, y)]
    f = lambda v: _matvec(B, relu(_matvec(A, v)))  # noqa: E731
    out += [f"Put a ReLU between them, h(v) = B relu(A v). For x = {_vec(x)}, y = {_vec(y)}: "
            f"h(x + y) = {_vec(f(xy))}, but h(x) + h(y) = {_vec([a + c for a, c in zip(f(x), f(y))])}. "
            f"The stack is no longer a single matrix.", ""]
    d = 1024
    rows2 = []
    for h in (16, 64, 256, 1024):
        rows2.append((h, f"{2 * d * h:,}", f"{d * d:,}", f"{2 * d * h / (d * d):.3f}", f"at most {h}"))
    out += [f"Factored as [{d}, h] times [h, {d}] instead of one [{d}, {d}] matrix:", "",
            _table(["h", "entries 2*d*h", "entries d*d", "ratio", "rank of the product"], rows2)]
    return "\n".join(out)


def elimination_cost():
    def run(n):
        # a diagonally dominant matrix needs no row swaps; count multiply-adds in elimination and back substitution
        A = [[float((i * 7 + j * 3) % 5 - 2) + (2.0 * n if i == j else 0.0) for j in range(n)] + [1.0] for i in range(n)]
        count = 0
        for c in range(n):
            for i in range(c + 1, n):
                f = A[i][c] / A[c][c]
                for j in range(c, n + 1):
                    A[i][j] -= f * A[c][j]
                    count += 1
        for i in range(n - 1, -1, -1):
            count += n - 1 - i
        return count

    rows = []
    for n in (3, 10, 100):
        cnt = run(n)
        rows.append((n, f"{cnt:,}", f"{n ** 3 / 3:,.0f}", f"{n ** 3:,}"))
    return _table(["n", "multiply-adds counted", "n^3 / 3", "n^3 (one n x n product)"], rows)


def pivot_choice():
    eps = 1e-17
    A = [[eps, 1.0], [1.0, 1.0]]
    b = [1.0, 2.0]
    out = []
    rows = []
    for mode in ("first", "partial"):
        M = [A[0] + [b[0]], A[1] + [b[1]]]
        _eliminate(M, ncols=2, pivoting=mode)
        x = _back_substitute(M, 2)
        res = [sum(a * xi for a, xi in zip(row, x)) - bi for row, bi in zip(A, b)]
        rows.append((mode, _vec(x), f"{max(abs(r) for r in res):.1e}"))
    out.append(_table(["pivot rule", "computed x", "largest residual |Ax - b|"], rows, "lrr"))
    out.append("")
    out.append("System: 1e-17 x1 + x2 = 1 and x1 + x2 = 2, in 64-bit floating point. The exact solution is "
               "x1 = 1/(1 - 1e-17), x2 = 1 - 1e-17 x1, both within 1e-16 of 1.")
    return "\n".join(out)
