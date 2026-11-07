# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Arrays, Shapes, Strides and Broadcasting".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: arrays.<name> -->` markers. Names that start with an underscore
(and the `Arr` class) are shared with the chapter's figures, so a picture and
the table beside it cannot disagree.

The model is the chapter's: an array *view* is a piece of flat storage plus an
offset, a shape and one stride per axis. Element `index` lives at
`offset + sum(index[k] * strides[k])`. Everything else (transpose, slicing,
reshape, broadcasting) is a function that builds a new view of the same storage.

Swedish Embedded AB implements solutions for tensor layout and indexing in
numerical software. If your team needs expertise in array memory layout and
the kernels that read it, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from itertools import product

# ------------------------------------------------------------------ the model


def row_major_strides(shape):
    """Strides (in elements) of a contiguous row-major array: last axis fastest."""
    strides = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    return tuple(strides)


def col_major_strides(shape):
    """Strides of a contiguous column-major array: first axis fastest."""
    strides, run = [], 1
    for n in shape:
        strides.append(run)
        run *= n
    return tuple(strides)


def numel(shape):
    return math.prod(shape)


def offset_of(index, strides, base=0):
    """offset = base + sum_k index[k] * strides[k]."""
    return base + sum(i * s for i, s in zip(index, strides))


def unravel(offset, shape):
    """The row-major index whose offset is `offset` (repeated divmod from the left)."""
    index = []
    for stride in row_major_strides(shape):
        index.append(offset // stride)
        offset %= stride
    return tuple(index)


class Arr:
    """A view: shared flat `data`, plus `shape`, `strides` and a base `offset`."""

    def __init__(self, data, shape, strides=None, offset=0):
        self.data = data
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else row_major_strides(self.shape)
        self.offset = offset
        assert len(self.shape) == len(self.strides)

    def indices(self):
        """Every index in row-major order (last axis fastest)."""
        return product(*[range(n) for n in self.shape])

    def at(self, index):
        return self.data[offset_of(index, self.strides, self.offset)]

    def flat(self):
        """The elements read in row-major order of the view."""
        return [self.at(i) for i in self.indices()]

    def nested(self, flat=None, shape=None):
        vals = self.flat() if flat is None else flat
        shape = self.shape if shape is None else shape
        if len(shape) == 0:
            return vals[0]
        if len(shape) == 1:
            return list(vals)
        step = numel(shape[1:])
        return [self.nested(vals[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]

    def permute(self, axes):
        """A view with the axes reordered; transposing a matrix is permute((1, 0))."""
        assert sorted(axes) == list(range(len(self.shape)))
        return Arr(self.data, [self.shape[a] for a in axes], [self.strides[a] for a in axes], self.offset)

    def slice(self, *spec):
        """A view of `spec[k] = (start, stop, step)` per axis, with no copy."""
        shape, strides, off = [], [], self.offset
        for k, (start, stop, step) in enumerate(spec):
            shape.append(max(0, -(-(stop - start) // step)))
            strides.append(self.strides[k] * step)
            off += start * self.strides[k]
        return Arr(self.data, shape, strides, off)

    def is_contiguous(self):
        """True when the strides are the row-major strides of the shape.

        An axis of size 1 is never stepped along, so its stride does not matter.
        """
        want = row_major_strides(self.shape)
        return all(n == 1 or s == w for n, s, w in zip(self.shape, self.strides, want))

    def copy(self):
        return Arr(self.flat(), self.shape)

    def reshape(self, shape):
        """Same elements in row-major order under a new shape. Returns (array, was_a_view)."""
        assert numel(shape) == numel(self.shape), "element counts differ"
        if self.is_contiguous():
            return Arr(self.data, shape, None, self.offset), True
        return Arr(self.flat(), shape), False

    def broadcast_to(self, shape):
        """A view stretched to `shape`: new and size-1 axes get stride 0."""
        lead = len(shape) - len(self.shape)
        assert lead >= 0
        strides = [0] * lead
        for n, s, want in zip(self.shape, self.strides, shape[lead:]):
            if n == want:
                strides.append(s)
            else:
                assert n == 1, f"cannot stretch {self.shape} to {tuple(shape)}"
                strides.append(0)
        return Arr(self.data, shape, strides, self.offset)


def from_nested(rows):
    """A contiguous row-major array from a nested list."""
    shape, probe = [], rows
    while isinstance(probe, list):
        shape.append(len(probe))
        probe = probe[0]
    flat = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            flat.append(x)
    walk(rows)
    return Arr(flat, shape)


def broadcast_shape(a, b):
    """The broadcast of two shapes: align from the right; sizes must match or one be 1."""
    out = []
    for k in range(1, max(len(a), len(b)) + 1):
        x = a[-k] if k <= len(a) else 1
        y = b[-k] if k <= len(b) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError(f"axis {k} from the right: {x} vs {y}")
        out.append(max(x, y) if min(x, y) != 0 else 0)
    return tuple(reversed(out))


def add(a, b):
    """Elementwise sum with the explicit broadcast of both operands."""
    shape = broadcast_shape(a.shape, b.shape)
    a2, b2 = a.broadcast_to(shape), b.broadcast_to(shape)
    return Arr([a2.at(i) + b2.at(i) for i in a2.indices()], shape)


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    cell = lambda c: str(c).replace("|", "\\|")  # a literal pipe inside a cell must not split the column
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def _t(x):
    """A tuple or list of numbers as text: (1, 2, 3)."""
    return "(" + ", ".join(str(v) for v in x) + ")"


def _sh(x):
    """A shape as text: [2, 3, 4]."""
    return "[" + ", ".join(str(v) for v in x) + "]"


def _labelled_matrix(rows, cols):
    """A rows x cols matrix whose entry (i, j) is the number 10 i + j: the value names its place."""
    return from_nested([[10 * i + j for j in range(cols)] for i in range(rows)])


# ------------------------------------------------------------------ worked blocks

def layout_3x4():
    A = _labelled_matrix(3, 4)
    rows, cols = A.shape
    row_major = [0] * 12
    col_major = [0] * 12
    for i, j in A.indices():
        row_major[i * cols + j] = A.at((i, j))
        col_major[i + rows * j] = A.at((i, j))
    out = [f"A = {A.nested()} {_sh(A.shape)}; entry (i, j) holds the number 10 i + j.", "",
           f"Row-major memory: {row_major}", "", f"Column-major memory: {col_major}", ""]
    picks = [(0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3)]
    out.append(_table(["(i, j)", "value", "row-major i*4 + j", "column-major i + 3*j"],
                      [(f"({i}, {j})", A.at((i, j)), i * cols + j, i + rows * j) for i, j in picks]))
    out += ["", f"Memory distance from (i, j) to (i, j+1): {1} row-major, {rows} column-major. "
                f"From (i, j) to (i+1, j): {cols} row-major, {1} column-major."]
    return "\n".join(out)


def nd_offsets():
    shape = (2, 3, 4)
    strides = row_major_strides(shape)
    picks = [(0, 0, 0), (0, 0, 3), (0, 1, 0), (0, 2, 3), (1, 0, 0), (1, 1, 1), (1, 2, 3)]
    rows = []
    for i, j, k in picks:
        nested = (i * shape[1] + j) * shape[2] + k
        rows.append((f"({i}, {j}, {k})", f"(({i}*3 + {j})*4 + {k}) = {nested}",
                     f"{i}*{strides[0]} + {j}*{strides[1]} + {k}*{strides[2]} = {offset_of((i, j, k), strides)}"))
    out = [f"Shape {_sh(shape)} has {numel(shape)} elements and row-major strides {_t(strides)}.", "",
           _table(["index", "nested (Horner) form", "stride form"], rows, "lll")]
    offs = sorted(offset_of(i, strides) for i in product(*[range(n) for n in shape]))
    out += ["", f"All {numel(shape)} offsets are distinct and cover exactly 0 to {numel(shape) - 1}: "
                f"{offs == list(range(numel(shape)))}."]
    off = 17
    q1, r1 = divmod(off, strides[0])
    q2, r2 = divmod(r1, strides[1])
    out += ["", f"Going back: offset {off} = {q1}*{strides[0]} + {r1}, then {r1} = {q2}*{strides[1]} + {r2}, "
                f"so the index is ({q1}, {q2}, {r2}); the function agrees: {unravel(off, shape)}.",
            "", f"Column-major strides of the same shape are {_t(col_major_strides(shape))}, so index (1, 2, 3) "
                f"sits at offset {offset_of((1, 2, 3), col_major_strides(shape))} there and at "
                f"{offset_of((1, 2, 3), strides)} row-major (both are the last element, as they must be)."]
    return "\n".join(out)


def strides_table():
    cases = [("vector", [12]), ("matrix", [3, 4]), ("row of a batch", [1, 12]), ("three axes", [2, 3, 4]),
             ("images [B, C, H, W]", [2, 3, 4, 5]), ("sequences [B, T, D]", [2, 5, 8]),
             ("a size-1 axis", [3, 1, 4])]
    rows = []
    for name, shape in cases:
        rows.append((name, _sh(shape), numel(shape), _t(row_major_strides(shape)), _t(col_major_strides(shape))))
    big = [(32, 3, 224, 224), (8, 512, 4096)]
    out = [_table(["kind", "shape", "elements", "row-major strides", "column-major strides"], rows, "llrll"), "",
           "Each row-major stride is the product of all the sizes to its right; the last is always 1.", ""]
    out.append(_table(["shape", "elements", "bytes at 4 bytes each", "MiB"],
                      [(_sh(s), f"{numel(s):,}", f"{numel(s) * 4:,}", f"{numel(s) * 4 / 2**20:.3f}") for s in big]))
    return "\n".join(out)


def _colmajor_gemm(m, n, k, a, lda, b, ldb, trans_a=False):
    """C = op(A) B with every buffer column-major, C[i + j*m] = sum_l op(A)[i, l] B[l, j].

    A is stored with leading dimension lda: A[i, l] = a[i + l*lda], or a[l + i*lda] when trans_a.
    """
    def A(i, l):
        return a[l + i * lda] if trans_a else a[i + l * lda]
    return [sum(A(i, l) * b[l + j * ldb] for l in range(k)) for j in range(n) for i in range(m)]


def misread_and_swap():
    A = from_nested([[1, 2, 3], [4, 5, 6]])
    buf = A.data
    # The same six numbers read as a column-major matrix of shape [3, 2]: column j is buf[3j : 3j + 3].
    as_col = from_nested([[buf[i + 3 * j] for j in range(2)] for i in range(3)])
    A_t = from_nested([[A.at((i, j)) for i in range(2)] for j in range(3)])
    out = [f"A = {A.nested()} {_sh(A.shape)}, stored row-major: {buf}.", "",
           f"Read that buffer as a column-major matrix of shape [3, 2] (column j is buf[3j:3j+3]): {as_col.nested()}. "
           f"That is A^T = {A_t.nested()}: {as_col.nested() == A_t.nested()}. Same bytes, other convention, transposed matrix.", ""]
    X = from_nested([[1, 2, 3], [4, 5, 6]])                         # [m = 2, k = 3]
    W = from_nested([[1, 0, 2], [0, 1, 1], [1, 1, 0], [2, 0, 1]])   # [n = 4, k = 3]
    m, k = X.shape
    n = W.shape[0]
    want = [sum(X.at((i, l)) * W.at((j, l)) for l in range(k)) for i in range(m) for j in range(n)]
    # Column-major routine for C' [n, m] = op(W) X^T: W's buffer has leading dimension k and is transposed
    # on read, X's buffer read column-major is X^T with leading dimension k.
    got = _colmajor_gemm(n, m, k, W.data, k, X.data, k, trans_a=True)
    out += [f"Row-major product Y = X W^T with X {_sh(X.shape)} = {X.nested()} and W {_sh(W.shape)} = {W.nested()}: "
            f"Y flat = {want}.", "",
            f"The column-major routine gemm of the text is asked for the [M, N] = [{n}, {m}] product W X^T, given W's buffer "
            f"and X's buffer unchanged: M = {n}, N = {m}, K = {k}, A = W's buffer with trans_a = true and lda = K = {k}, "
            f"B = X's buffer with ldb = K = {k}, and ldc = M = {n}. Its output buffer is {got}.", "",
            f"Equal to Y's row-major buffer: {got == want}. No operand was copied or transposed."]
    return "\n".join(out)


def transpose_view():
    A = _labelled_matrix(3, 4)
    T = A.permute((1, 0))
    rows = []
    for j, i in T.indices():
        rows.append((f"({j}, {i})", f"{j}*{T.strides[0]} + {i}*{T.strides[1]} = {offset_of((j, i), T.strides)}",
                     T.at((j, i)), f"A[{i}][{j}] = {A.at((i, j))}"))
    out = [f"A = {A.nested()} {_sh(A.shape)}, strides {_t(A.strides)}, storage {A.data}.", "",
           f"B = transpose view of A: shape {_sh(T.shape)}, strides {_t(T.strides)}, same storage, "
           "no element moved. Every element of B:", "",
           _table(["index in B", "offset", "B value", "which element of A"], rows, "lrrl"), "",
           f"B read in its own row-major order: {T.flat()}. The offsets it visits: "
           f"{[offset_of(i, T.strides) for i in T.indices()]}.", ""]
    mat = T.copy()
    out.append(f"Copy of B, materialised: storage {mat.data}, strides {_t(mat.strides)}; "
               f"contiguous: {mat.is_contiguous()} (the view itself: {T.is_contiguous()}).")
    return "\n".join(out)


def slices_are_views():
    A = _labelled_matrix(3, 4)
    cases = [("A", A), ("A[:, 1:3]", A.slice((0, 3, 1), (1, 3, 1))), ("A[1:3, :]", A.slice((1, 3, 1), (0, 4, 1))),
             ("A[::2, :]", A.slice((0, 3, 2), (0, 4, 1))), ("A[:, ::2]", A.slice((0, 3, 1), (0, 4, 2))),
             ("A[1:2, :] (a row)", A.slice((1, 2, 1), (0, 4, 1))), ("A[:, 2:3] (a column)", A.slice((0, 3, 1), (2, 3, 1))),
             ("transpose(A)", A.permute((1, 0)))]
    rows = [(name, _sh(v.shape), _t(v.strides), v.offset, v.flat(), "yes" if v.is_contiguous() else "no")
            for name, v in cases]
    out = [f"Storage of A (row-major [3, 4]): {A.data}.", "",
           _table(["view", "shape", "strides", "start offset", "elements in order", "contiguous"], rows, "llllll"), "",
           "Writing through a view changes the original:", ""]
    col = A.slice((0, 3, 1), (2, 3, 1))
    before = A.nested()
    A.data[offset_of((1, 0), col.strides, col.offset)] = -1
    out.append(f"after setting element 1 of the column view A[:, 2:3] to -1, A = {A.nested()} (was {before}).")
    return "\n".join(out)


def reshape_vs_transpose():
    A = from_nested([[1, 2, 3], [4, 5, 6]])
    R, r_view = A.reshape((3, 2))
    T = A.permute((1, 0))
    Tc = T.copy()
    out = [f"A = {A.nested()} {_sh(A.shape)}, storage {A.data}.", "",
           _table(["operation", "shape", "result", "storage order", "work"],
                  [("reshape A to [3, 2]", _sh(R.shape), R.nested(), R.flat(), "view" if r_view else "copy"),
                   ("transpose A", _sh(T.shape), T.nested(), T.flat(), "view (strides " + _t(T.strides) + ")")],
                  "lllll"), "",
           f"The two results have the same shape and different contents: {R.nested() != T.nested()}.", ""]
    flat, was_view = T.reshape((6,))
    out.append(f"Flattening the transposed view needs a copy ({'view' if was_view else 'copy'}): reshape(transpose(A), [6]) "
               f"= {flat.flat()}, whereas reshape(A, [6]) = {A.reshape((6,))[0].flat()}.")
    v = [1, 10]
    out += ["", f"Multiply each [3, 2] result by v = {_t(v)}: reshape gives "
                f"{[sum(R.at((i, j)) * v[j] for j in range(2)) for i in range(3)]}, "
                f"transpose gives {[sum(Tc.at((i, j)) * v[j] for j in range(2)) for i in range(3)]}."]
    return "\n".join(out)


def split_heads():
    T, H, Dh = 2, 2, 3
    D = H * Dh
    x = Arr([10 * t + c for t in range(T) for c in range(D)], (T, D))
    split, v1 = x.reshape((T, H, Dh))
    heads = split.permute((1, 0, 2))
    wrong, v2 = x.reshape((H, T, Dh))
    restored = heads.permute((1, 0, 2))           # back to [T, H, Dh]
    merged, v3 = restored.reshape((T, D))
    out = [f"x is a sequence of T = {T} positions of width D = {D}; entry (t, c) holds 10 t + c: x = {x.nested()}, "
           f"storage {x.data}. Split D into H = {H} heads of Dh = {Dh}.", "",
           f"Right: reshape to [T, H, Dh] = {_sh(split.shape)} ({'a view' if v1 else 'a copy'}, strides "
           f"{_t(split.strides)}), then swap the first two axes to [H, T, Dh] (a view, strides {_t(heads.strides)}).", "",
           _table(["head", "rows of the head (one row per position)"],
                  [(h, heads.slice((h, h + 1, 1), (0, T, 1), (0, Dh, 1)).nested()[0]) for h in range(H)], "rl"), "",
           f"Wrong: reshape x directly to [H, T, Dh] = {_sh(wrong.shape)} ({'a view' if v2 else 'a copy'}, strides {_t(wrong.strides)}):", "",
           _table(["head", "rows of the head"], [(h, wrong.nested()[h]) for h in range(H)], "rl"), "",
           "The wrong version hands head 0 the first two groups of position 0 (entries 0 to 5 split in threes) "
           "instead of the first group of every position.", "",
           f"Merging back: swap the axes back to [T, H, Dh] (strides {_t(restored.strides)}, which is the row-major "
           f"strides of [T, H, Dh]: {restored.is_contiguous()}), then reshape to [T, D] ({'a view' if v3 else 'a copy'}); "
           f"the result equals x: {merged.nested() == x.nested()}. Reshaping the [H, T, Dh] view itself would "
           f"be {'a view' if heads.is_contiguous() else 'a copy'} (contiguous {heads.is_contiguous()}), and its "
           f"elements {heads.reshape((T, D))[0].nested()} are not x."]
    return "\n".join(out)


def split_axis_rule():
    # Splitting one axis of size n and stride s into (a, b) with a*b = n gives strides (b*s, s).
    T, D, H, Dh = 4, 6, 2, 3
    base = Arr(list(range(T * D)), (T, D))
    every_other = base.slice((0, T, 2), (0, D, 1))     # rows 0 and 2: strides (12, 1), not contiguous
    sub = Arr(every_other.data, (every_other.shape[0], H, Dh),
              (every_other.strides[0], Dh * every_other.strides[1], every_other.strides[1]), every_other.offset)
    copy_split = Arr(every_other.flat(), (every_other.shape[0], H, Dh))
    out = [f"A non-contiguous view W = base[::2, :] of a [{T}, {D}] array: shape {_sh(every_other.shape)}, "
           f"strides {_t(every_other.strides)}, contiguous: {every_other.is_contiguous()}.", "",
           f"Splitting its last axis (size {D}, stride 1) into ({H}, {Dh}) needs no copy. The new strides are "
           f"(Dh*1, 1) = ({Dh}, 1) for the pieces, the first axis keeps stride {every_other.strides[0]}: "
           f"shape {_sh(sub.shape)}, strides {_t(sub.strides)}.", "",
           f"Its elements equal those of a copied-and-reshaped array: {sub.flat() == copy_split.flat()}. "
           f"Element (1, 1, 2) sits at offset {offset_of((1, 1, 2), sub.strides, sub.offset)} of the original storage."]
    return "\n".join(out)


def broadcast_shapes():
    cases = [((3, 4), (4,)), ((3, 4), (3, 1)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 4)), ((2, 3, 4), (2, 1, 4)),
             ((8, 1, 6, 1), (7, 1, 5)), ((), (2, 3)), ((3, 4), (3,)), ((2, 3), (4, 3)), ((5, 1), (5,))]
    rows = []
    for a, b in cases:
        try:
            res = _sh(broadcast_shape(a, b))
            why = "ok"
        except ValueError as e:
            res, why = "error", str(e)
        rows.append((_sh(a), _sh(b), res, why))
    return _table(["shape a", "shape b", "result", "note"], rows, "llll")


def broadcast_values():
    X = from_nested([[1, 2, 3], [4, 5, 6]])
    bias = from_nested([10, 20, 30])
    col = from_nested([[100], [200]])
    row = from_nested([[1, 2, 3, 4]])
    colv = from_nested([[10], [20], [30]])
    bv = bias.broadcast_to((2, 3))
    out = [f"X = {X.nested()} {_sh(X.shape)} and bias = {bias.flat()} {_sh(bias.shape)}. The bias aligned to the "
           f"right axis of X is stretched over rows: as a view of shape {_sh(bv.shape)} its strides are {_t(bv.strides)} "
           "(stride 0 on the stretched axis, the original stride 1 on the other), over the same three numbers of storage.", "",
           _table(["index (i, j)", "offset into bias", "bias value", "X value", "sum"],
                  [(f"({i}, {j})", offset_of((i, j), bv.strides), bv.at((i, j)), X.at((i, j)), bv.at((i, j)) + X.at((i, j)))
                   for i, j in bv.indices()], "lrrrr"), "",
           f"X + bias = {add(X, bias).nested()} {_sh(add(X, bias).shape)}.", "",
           f"Column [2, 1] = {col.flat()} added to X: {add(X, col).nested()}. Mixing the two kinds, "
           f"shapes {_sh(col.shape)} and {_sh(bias.shape)} broadcast to "
           f"{_sh(broadcast_shape(col.shape, bias.shape))}: {add(col, bias).nested()}.", "",
           f"Outer sum: [3, 1] + [1, 4] gives {_sh(add(colv, row).shape)}: {add(colv, row).nested()}."]
    return "\n".join(out)


def broadcast_cost():
    # The stretched view reads the same stored numbers: count how many distinct storage cells it touches.
    rows = []
    for name, a, target in [("bias [D] over [T, D]", Arr(list(range(5)), (5,)), (7, 5)),
                            ("column [T, 1] over [T, D]", Arr(list(range(7)), (7, 1)), (7, 5)),
                            ("scalar [] over [T, D]", Arr([0], ()), (7, 5)),
                            ("[B, 1, T] over [B, T, T]", Arr(list(range(2 * 4)), (2, 1, 4)), (2, 4, 4))]:
        v = a.broadcast_to(target)
        touched = {offset_of(i, v.strides, v.offset) for i in v.indices()}
        rows.append((name, _t(v.strides), numel(target), len(touched)))
    return ("Symbol values: T = 7 and D = 5 in the first three rows; B = 2 and T = 4 in the last row.\n\n"
            + _table(["broadcast", "view strides", "elements seen", "stored numbers touched"], rows, "lrrr"))


def image_layouts():
    B, C, H, W = 2, 3, 2, 2
    nchw = Arr(list(range(B * C * H * W)), (B, C, H, W))
    # The same picture stored channels-last: NHWC[b, h, w, c] = NCHW[b, c, h, w].
    nhwc_data = [nchw.at((b, c, h, w)) for b in range(B) for h in range(H) for w in range(W) for c in range(C)]
    nhwc = Arr(nhwc_data, (B, H, W, C))
    px = (1, 1, 0)  # image 1, row 1, column 0
    chans = []
    for c in range(C):
        o1 = offset_of((px[0], c, px[1], px[2]), nchw.strides)
        o2 = offset_of((px[0], px[1], px[2], c), nhwc.strides)
        chans.append((c, o1, nchw.data[o1], o2, nhwc.data[o2]))
    view = nchw.permute((0, 2, 3, 1))
    out = [f"Image batch with B = {B}, C = {C}, H = {H}, W = {W}; element value = its NCHW offset.", "",
           f"NCHW: shape {_sh(nchw.shape)}, strides {_t(nchw.strides)}. NHWC: shape {_sh(nhwc.shape)}, strides {_t(nhwc.strides)}.", "",
           "The NHWC array is the same picture stored channels-last: its element (b, h, w, c) is the NCHW element (b, c, h, w). "
           f"The three channel values of the pixel at image 1, row 1, column 0:", "",
           _table(["channel", "NCHW offset", "NCHW value", "NHWC offset", "NHWC value"], chans), "",
           f"In NCHW the three channels of one pixel are {nchw.strides[1]} apart; in NHWC they are adjacent.", "",
           f"Viewing the NCHW storage as NHWC without moving anything is the permuted view with axes (0, 2, 3, 1): "
           f"shape {_sh(view.shape)}, strides {_t(view.strides)}, contiguous {view.is_contiguous()}; "
           f"its elements equal the NHWC array's: {view.flat() == nhwc.flat()}. "
           f"So a real NHWC layout needs a copy ({nhwc.is_contiguous()} contiguous afterwards).", ""]
    Bs, T, D = 2, 3, 4
    seq = Arr(list(range(Bs * T * D)), (Bs, T, D))
    flat, was_view = seq.reshape((Bs * T, D))
    rows = [(b, t, b * T + t, offset_of((b, t, 0), seq.strides), flat.at((b * T + t, 0)))
            for b in range(Bs) for t in range(T)]
    out += [f"Sequence batch B = {Bs}, T = {T}, D = {D}: shape {_sh(seq.shape)}, strides {_t(seq.strides)}. "
            f"Merging the first two axes to [B*T, D] = {_sh(flat.shape)} is a {'view' if was_view else 'copy'}; "
            "the row of token (b, t) is b*T + t:", "",
            _table(["b", "t", "row b*T + t", "offset of (b, t, 0)", "value of row's first entry"], rows)]
    return "\n".join(out)


def cache_lines():
    line_bytes, elem_bytes = 64, 4
    rows = []
    for s in (1, 2, 4, 8, 16, 17, 64):
        lines = {(s * i * elem_bytes) // line_bytes for i in range(64)}
        rows.append((s, len(lines), f"{64 / len(lines):.1f}"))
    r = {(i * elem_bytes) // line_bytes for i in range(64)}
    c = {(i * 64 * elem_bytes) // line_bytes for i in range(64)}
    out = [f"Reading 64 numbers of 4 bytes starting at a line boundary, with memory fetched in lines of {line_bytes} bytes "
           f"({line_bytes // elem_bytes} numbers). Distinct lines touched, by stride in elements:", "",
           _table(["stride", "lines touched", "useful numbers per line"], rows), "",
           f"One row of a [64, 64] matrix (stride 1) touches {len(r)} lines; one column (stride 64) touches {len(c)}."]
    return "\n".join(out)


def storage_view_gather():
    storage = [0, 10, 20, 30, 40, 50, 60]
    V = Arr(storage, (2, 3), (1, 2), 1)
    rows = [(f"({i}, {j})", f"1 + {i}*1 + {j}*2 = {offset_of((i, j), V.strides, V.offset)}", V.at((i, j)))
            for i, j in V.indices()]
    return "\n".join([f"Storage {storage}, offset 1, shape [2, 3], strides (1, 2):", "",
                      _table(["index", "offset", "value"], rows), "",
                      f"Materialised in row-major order: {V.flat()}."])


def alignment_arithmetic():
    rows = []
    for d in (20, 48, 64, 96, 100, 128):
        g = math.gcd(64, d)
        need = 64 // g
        rows.append((d, need, need * d * 4, 6 * d * 4, (6 * d * 4) % 256))
    out = ["A byte offset into a buffer must be a multiple of 256 bytes = 64 four-byte numbers. A row boundary r of a "
           "matrix with D columns is at r*D numbers; the smallest r > 0 that works is 64 / gcd(64, D).", "",
           _table(["D", "smallest aligned r", "bytes at that r", "bytes at r = 6", "remainder mod 256"], rows), "",
           "With D = 20 a tile boundary at row 6 sits 480 bytes in, which is 224 bytes past the last 256-byte boundary; "
           f"row {64 // math.gcd(64, 20)} sits at {64 // math.gcd(64, 20) * 20 * 4} = "
           f"{64 // math.gcd(64, 20) * 20 * 4 // 256} * 256 bytes."]
    return "\n".join(out)


def kernel_permute():
    n, c, hw = 2, 3, 4
    nchw = Arr(list(range(n * c * hw)), (n, c, hw))
    gathered = []
    for idx in range(n * hw * c):
        nn = idx // (hw * c)
        r0 = idx % (hw * c)
        l = r0 // c
        ch = r0 % c
        gathered.append(nchw.data[(nn * c + ch) * hw + l])
    view = nchw.permute((0, 2, 1))
    rows = []
    for idx in range(c * hw + 3):
        nn = idx // (hw * c)
        r0 = idx % (hw * c)
        rows.append((idx, nn, r0 // c, r0 % c, (nn * c + r0 % c) * hw + r0 // c))
    out = [f"Layout change [N, C, HW] to [N, HW, C] with N = {n}, C = {c}, HW = {hw}. One output element per index "
           "idx, which decomposes as n = idx / (HW C), r = idx mod (HW C), l = r / C, ch = r mod C and is read from "
           "(n C + ch) HW + l of the input. The first 15 outputs:", "",
           _table(["idx", "n", "l", "ch", "source offset"], rows), "",
           f"The gathered output equals the row-major reading of the permuted view with axes (0, 2, 1): "
           f"{gathered == view.flat()}. The permuted view's strides are {_t(view.strides)}."]
    return "\n".join(out)


def broadcast_hw_kernel():
    N, C, H, W = 1, 2, 2, 3
    a = Arr([1, 2, 3, 4], (N, C, H, 1))
    b = Arr([10, 20, 30, 40, 50, 60], (N, C, 1, W))
    shape = broadcast_shape(a.shape, b.shape)
    via_views = add(a, b)
    kernel = []
    for idx in range(N * C * H * W):
        w = idx % W
        t1 = idx // W
        h = t1 % H
        nc = t1 // H
        kernel.append(a.data[nc * H + h] + b.data[nc * W + w])
    av, bv = a.broadcast_to(shape), b.broadcast_to(shape)
    return "\n".join([f"a {_sh(a.shape)} = {a.data}, b {_sh(b.shape)} = {b.data}; the broadcast shape is {_sh(shape)}.", "",
                      f"As stretched views: a has strides {_t(av.strides)}, b has strides {_t(bv.strides)}.", "",
                      f"Result, flat: {via_views.data}.", "",
                      f"The index formula y[idx] = a[nc*H + h] + b[nc*W + w] gives {kernel}; equal: {kernel == via_views.data}."])


def merge_rule():
    """Two adjacent axes of sizes >= 2 merge without a copy exactly when s_k == d_{k+1} * s_{k+1};
    an axis of size 1 is dropped first, and the other axis keeps its stride."""
    def merged_stride(v, k):
        """The stride of the merged axis, or None when a copy is needed."""
        if v.shape[k] == 1:
            return v.strides[k + 1]
        if v.shape[k + 1] == 1:
            return v.strides[k]
        return v.strides[k + 1] if v.strides[k] == v.shape[k + 1] * v.strides[k + 1] else None

    A = _labelled_matrix(3, 4)
    x = Arr(list(range(24)), (2, 3, 4))
    heads = x.permute((1, 0, 2))
    cases = [("contiguous [2, 3, 4]", x, 0), ("contiguous [2, 3, 4]", x, 1),
             ("swapped axes [3, 2, 4]", heads, 0), ("swapped axes [3, 2, 4]", heads, 1),
             ("A[:, 0:2] of [3, 4]", A.slice((0, 3, 1), (0, 2, 1)), 0),
             ("A[::2, :] of [3, 4]", A.slice((0, 3, 2), (0, 4, 1)), 0),
             ("A[:, 2:3] of [3, 4]", A.slice((0, 3, 1), (2, 3, 1)), 0)]
    rows = []
    for name, v, k in cases:
        ms = merged_stride(v, k)
        if 1 in (v.shape[k], v.shape[k + 1]):
            cmp = "an axis has size 1: drop it"
        else:
            cmp = f"{v.strides[k]} vs {v.shape[k + 1]}*{v.strides[k + 1]} = {v.shape[k + 1] * v.strides[k + 1]}"
        rows.append((name, _sh(v.shape), _t(v.strides), f"{k}, {k + 1}", cmp,
                     f"view, merged stride {ms}" if ms is not None else "copy"))
    # Check the rule against brute force: a merged view must reproduce the row-major reading.
    agree = True
    for name, v, k in cases:
        shape = v.shape[:k] + (v.shape[k] * v.shape[k + 1],) + v.shape[k + 2:]
        ms = merged_stride(v, k)
        if ms is not None:
            strides = v.strides[:k] + (ms,) + v.strides[k + 2:]
            agree &= Arr(v.data, shape, strides, v.offset).flat() == v.flat()
    return _table(["array", "shape", "strides", "axes merged", "s_k vs d_(k+1) s_(k+1)", "result"], rows, "llllll") + \
        f"\n\nEvery merge the rule allows reproduces the row-major reading exactly: {agree}."


def silent_outer_diff():
    pred = from_nested([[1], [2], [3], [4]])       # [4, 1]: a column, one prediction per row
    target = from_nested([1, 2, 3, 4])             # [4]
    wrong = pred.broadcast_to(broadcast_shape(pred.shape, target.shape))
    tb = target.broadcast_to(wrong.shape)
    diff = [[wrong.at((i, j)) - tb.at((i, j)) for j in range(4)] for i in range(4)]
    mse_wrong = sum(d * d for r in diff for d in r) / 16
    flat_pred, _ = pred.reshape((4,))
    mse_right = sum((flat_pred.at((i,)) - target.at((i,))) ** 2 for i in range(4)) / 4
    return "\n".join([f"Predictions p of shape [4, 1] = {pred.flat()} and targets t of shape [4] = {target.flat()}; "
                      "they agree exactly.", "",
                      f"The shapes broadcast to {_sh(wrong.shape)}, so p - t is the table of all 16 differences:", "",
                      f"{diff}", "",
                      f"Mean squared error computed that way: {mse_wrong}. With p reshaped to [4] first: {mse_right}."])
