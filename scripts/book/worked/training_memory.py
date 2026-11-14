# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Where Training Memory Goes".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: training_memory.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared
with the chapter's figures so a picture and the table beside it cannot disagree.

The residual block of the chapter is written here from its description (the forward is Z = X W1^T, A = tanh(Z),
Y = X + A W2^T; the backward reads X and A); the number formats come from `number_formats` and the counter-based draw
of the mixed-precision chapter from `mixed_precision` (imported, never copied).

Swedish Embedded AB implements solutions for memory-bounded training of large models for its clients. If your team
needs expertise in fitting a training run into a fixed device then you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct

import bookrng
import mixed_precision as mp
import number_formats as nf
import stable

_table = nf._table

# ------------------------------------------------------------------ the block stack the chapter counts

D, F, L, N = 4096, 16384, 32, 8192          # width, hidden width, blocks, tokens of one step
P = L * 2 * D * F                           # trainable numbers of the stack: 2^32
GIB = 2 ** 30


def _gib(b):
    return f"{b / GIB:.2f}"


def _num(v, d=4):
    return nf._num(v, d)


# ------------------------------------------------------------------ a tiny stack, executed

def _mm(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def _t(M):
    return [list(r) for r in zip(*M)]


def _forward_block(X, W1, W2):
    """Z = X W1^T, A = tanh(Z), Y = X + A W2^T. Returns (A, Y)."""
    A = [[math.tanh(z) for z in r] for r in _mm(X, _t(W1))]
    Y = [[x + y for x, y in zip(rx, ry)] for rx, ry in zip(X, _mm(A, _t(W2)))]
    return A, Y


def _backward_block(dY, W1, W2, X, A):
    """The gradients of one block. Only X and A come from the forward pass; W1 and W2 are parameters."""
    dW2 = _mm(_t(dY), A)
    dA = _mm(dY, W2)
    dZ = [[da * (1.0 - a * a) for da, a in zip(rda, ra)] for rda, ra in zip(dA, A)]
    dW1 = _mm(_t(dZ), X)
    dX = [[d + e for d, e in zip(rd, re)] for rd, re in zip(dY, _mm(dZ, W1))]
    return dX, dW1, dW2


def _loss_of(Y):
    return 0.5 * sum(y * y for r in Y for y in r)


def _stack_loss(Ws, X0):
    X = X0
    for W1, W2 in Ws:
        _, X = _forward_block(X, W1, W2)
    return _loss_of(X)


def _draw_stack(rng, blocks, n, d, f, scale=0.5):
    """Per block W1 ([f, d], row-major) then W2 ([d, f], row-major), each entry normal() * scale; then X ([n, d])."""
    Ws = []
    for _ in range(blocks):
        W1 = [[rng.normal() * scale for _ in range(d)] for _ in range(f)]
        W2 = [[rng.normal() * scale for _ in range(f)] for _ in range(d)]
        Ws.append((W1, W2))
    X0 = [[rng.normal() for _ in range(d)] for _ in range(n)]
    return Ws, X0


def _round_trip(x, kind):
    """Rows of x stored in a narrow format and read back: 'f32' leaves them, 'bf16' rounds each element to nearest even,
    'fp8' divides a row by s = amax / 448, rounds to E4M3 (saturating) and multiplies by s."""
    if kind == "f32":
        return [list(r) for r in x]
    if kind == "bf16":
        return [[mp.rn_bf16(v) if v != 0.0 else 0.0 for v in r] for r in x]
    out = []
    for r in x:
        amax = max(abs(v) for v in r)
        s = amax / 448.0 if amax > 0 else 1.0
        out.append([s * nf.quantize(v / s, "E4M3", True) for v in r])
    return out


def _run_stack(Ws, X0, policy="keep", k=1, boundary="f32", perturb=None):
    """One forward and backward through the stack. policy 'keep' holds every block's (X, A) after the forward;
    'recompute' holds the input of every k-th block and rebuilds a segment of k blocks before its backward.
    `boundary` is the format the held inputs are stored in (the forward continues from the stored value).
    `perturb` is called with the weights between the forward and the backward (an update made too early).
    Returns (loss, [(dW1, dW2) per block], most block caches alive at once)."""
    nb = len(Ws)
    held, caches = [], []
    X = X0
    for l, (W1, W2) in enumerate(Ws):
        keep_input = policy == "keep" or l % k == 0
        if keep_input:
            X = _round_trip(X, boundary)
        held.append(X if keep_input else None)
        A, Y = _forward_block(X, W1, W2)
        caches.append((X, A) if policy == "keep" else None)
        X = Y
    loss = _loss_of(X)
    dY = [list(r) for r in X]
    if perturb:
        perturb(Ws)
    grads = [None] * nb
    alive = nb if policy == "keep" else 0
    if policy == "keep":
        for l in range(nb - 1, -1, -1):
            Xl, A = caches[l]
            dY, dW1, dW2 = _backward_block(dY, Ws[l][0], Ws[l][1], Xl, A)
            grads[l] = (dW1, dW2)
    else:
        for s in range(((nb - 1) // k) * k, -1, -k):
            seg, Xc = [], held[s]
            for l in range(s, min(s + k, nb)):
                A, Y = _forward_block(Xc, Ws[l][0], Ws[l][1])
                seg.append((Xc, A))
                Xc = Y
            alive = max(alive, len(seg))
            for off in range(len(seg) - 1, -1, -1):
                l = s + off
                Xl, A = seg[off]
                dY, dW1, dW2 = _backward_block(dY, Ws[l][0], Ws[l][1], Xl, A)
                grads[l] = (dW1, dW2)
    return loss, grads, alive


def _copy_ws(Ws):
    return [([list(r) for r in W1], [list(r) for r in W2]) for W1, W2 in Ws]


def _flat(grads):
    return [v for g in grads for M in g for r in M for v in r]


def _bits(x):
    return struct.pack("<d", x)


# ------------------------------------------------------------------ 1. what a block keeps for its backward

def saved_by_backward():
    blocks, n, d, f = 2, 3, 4, 8
    name = "training_memory.saved_by_backward"
    rng = bookrng.generator(name)
    Ws, X0 = _draw_stack(rng, blocks, n, d, f)
    # forward with a tape: the only way the backward reaches the forward pass is tape.read
    tape, reads = {}, []
    X = X0
    for l, (W1, W2) in enumerate(Ws):
        A, Y = _forward_block(X, W1, W2)
        tape[("X", l)], tape[("A", l)] = X, A
        X = Y
    dY = [list(r) for r in X]
    grads = [None] * blocks
    for l in range(blocks - 1, -1, -1):
        Xl, A = tape[("X", l)], tape[("A", l)]
        reads += [("X", l), ("A", l)]
        dY, dW1, dW2 = _backward_block(dY, Ws[l][0], Ws[l][1], Xl, A)
        grads[l] = (dW1, dW2)
    # gradient check against central differences, every weight
    h, worst = 1e-5, 0.0
    for l in range(blocks):
        for which in (0, 1):
            W = Ws[l][which]
            for i in range(len(W)):
                for j in range(len(W[0])):
                    old = W[i][j]
                    W[i][j] = old + h
                    up = _stack_loss(Ws, X0)
                    W[i][j] = old - h
                    dn = _stack_loss(Ws, X0)
                    W[i][j] = old
                    num = (up - dn) / (2 * h)
                    ana = grads[l][which][i][j]
                    worst = max(worst, abs(ana - num) / max(abs(num), 1e-3))
    assert worst < 1e-6, worst
    rows = []
    for key in sorted(set(reads), key=lambda k: (k[1], k[0])):
        arr = tape[key]
        rows.append([f"block {key[1]}", key[0], f"[{len(arr)}, {len(arr[0])}]", len(arr) * len(arr[0]),
                     "dW1 (reads X)" if key[0] == "X" else "dW2 and dZ = dA (1 - A^2) (read A)"])
    total = sum(len(tape[k]) * len(tape[k][0]) for k in set(reads))
    assert total == blocks * n * (d + f)
    seed = bookrng.seed_of(name)
    head = (f"A stack of {blocks} residual blocks with D = {d}, F = {f} on n = {n} tokens (rows). The forward is Z = X W1^T "
            "(W1 is [F, D]), A = tanh(Z) (the correctly rounded library tanh), Y = X + A W2^T (W2 is [D, F]); the loss is half the sum "
            "of the squares of the last output, so the gradient entering the last block is that output. The weights are drawn from one "
            f"generator seeded {seed} (seed_of '{name}'), as normal() times 0.5 in this order: for each block W1 row by row, then W2 row "
            "by row; then X row by row with normal() times 1. The backward of a block is written to take only the saved arrays X and A "
            "(and the weights, which are parameters); the table lists what it read. All "
            f"{sum(len(W1) * len(W1[0]) + len(W2) * len(W2[0]) for W1, W2 in Ws)} weights' gradients agree with central differences "
            f"(step 1e-5) to a relative difference below 1e-6: {'yes' if worst < 1e-6 else 'NO'}.\n\n")
    foot = (f"\n\nSaved numbers: {total} = blocks x n x (D + F) = {blocks} x {n} x {d + f}. The weights of a block, 2 D F = {2 * d * f}, "
            "are not counted here: they exist whether or not a backward pass follows.")
    return head + _table(["block", "array", "shape", "numbers", "read by"], rows, "llrrl") + foot


# ------------------------------------------------------------------ 2. activations against state

def activations_against_state():
    rows = []
    wb = 4 * P
    for n in (512, 2048, 8192, 32768, 131072):
        act = 4 * L * n * (D + F)
        rows.append([f"{n:,}", _gib(wb), _gib(wb), _gib(2 * wb), _gib(act), _gib(4 * wb + act), f"{(4 * wb + act) / wb:.2f}"])
    cross = 2 * D * F / (D + F)
    head = (f"The stack: L = {L} blocks with D = {D} and F = {F:,}, so P = L x 2 D F = {P:,} trainable numbers "
            "(2^32), no embedding and no attention. Every number is fp32, 4 bytes: a weight, its gradient, and the two AdamW moments "
            f"(16 bytes per parameter in all). Kept activations are L x n x (D + F) numbers (the count of the previous table, per block "
            f"and token). Gibibytes are 2^30 bytes, rounded to two decimals. The kept activations equal the parameter count at "
            f"n = 2 D F / (D + F) = {cross:,.1f} tokens.\n\n")
    return head + _table(["tokens n", "weights", "gradients", "two moments", "kept activations", "total", "total / weights"], rows)


# ------------------------------------------------------------------ 3. recomputing in segments

def _seg_numbers(k):
    """Numbers per token held at the peak of the backward when the input of every k-th block is kept: the L / k kept inputs
    (D each) and the full (X, A) of the k blocks of the segment being rebuilt."""
    return (L // k) * D + k * (D + F)


def segment_choices():
    keep = L * (D + F)
    rows = [["keep every block", "-", f"{keep:,}", _gib(4 * N * keep), "1.000", f"{12 * N * D * F * L / 1e12:.1f}"]]
    for k in (1, 2, 4, 8, 16, 32):
        num = _seg_numbers(k)
        rows.append([f"recompute, segments of {k}", k, f"{num:,}", _gib(4 * N * num), f"{num / keep:.3f}",
                     f"{16 * N * D * F * L / 1e12:.1f}"])
    kstar = math.sqrt(L * D / (D + F))
    head = (f"The stack of the previous tables (L = {L}, D = {D}, F = {F:,}) on n = {N:,} tokens, fp32. A segment of k blocks keeps "
            "its input; the backward rebuilds the segment (k blocks of X and A), runs its backward and drops it. The peak held at the "
            "backward is (L / k) D + k (D + F) numbers per token. FLOPs count 2 per multiply-add of a matrix product and nothing for "
            "tanh or additions: a forward is two products of 2 n D F FLOPs each per block, the backward four such products, so a step "
            f"that keeps everything is 12 n D F L and a step that rebuilds every block once is 16 n D F L, whatever k is. The real minimum "
            f"of the peak over k is at k = sqrt(L D / (D + F)) = {kstar:.3f}; the table lists the divisors of L.\n\n")
    return head + _table(["policy", "k", "numbers per token", f"GiB at n = {N:,}", "share of keep", "10^12 FLOPs per step"], rows,
                         "lrrrrr")


# ------------------------------------------------------------------ 4. the rebuilt block is the kept block

def recompute_exact():
    blocks, n, d, f = 6, 3, 4, 8
    name = "training_memory.recompute_exact"
    rng = bookrng.generator(name)
    Ws, X0 = _draw_stack(rng, blocks, n, d, f)
    ref_loss, ref_g, ref_alive = _run_stack(Ws, X0, "keep")
    rf = _flat(ref_g)
    rows = [["keep every block", ref_alive, "-", "0 of %d" % len(rf)]]
    for k in (1, 2, 3):
        loss, g, alive = _run_stack(Ws, X0, "recompute", k)
        diff = sum(1 for a, b in zip(rf, _flat(g)) if _bits(a) != _bits(b))
        assert loss == ref_loss and diff == 0
        rows.append([f"recompute, segments of {k}", alive, "yes" if _bits(loss) == _bits(ref_loss) else "no", f"{diff} of {len(rf)}"])
    # the weights change between the forward and the backward
    def update(Ws_):
        Ws_[3][0][1][2] += 0.05

    def distance(a, b):
        va, vb = _flat(a), _flat(b)
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)) / sum(y * y for y in vb))

    Wp = _copy_ws(Ws)
    Wp[3][0][1][2] += 0.05
    _, fresh, _ = _run_stack(Wp, X0, "keep")      # the gradient at the changed weights
    mixed = []
    for pol in ("keep", "recompute"):
        _, g, _ = _run_stack(_copy_ws(Ws), X0, pol, 1, perturb=update)
        mixed.append((pol, distance(g, ref_g), distance(g, fresh)))
    seed = bookrng.seed_of(name)
    head = (f"A stack of {blocks} blocks (D = {d}, F = {f}, n = {n}) drawn as in the first table from the generator seeded {seed} "
            f"(seed_of '{name}'); the loss is half the sum of squares of the last output, in float64. 'caches alive' is the most blocks "
            "whose X and A exist at once. 'bit-identical loss' compares the 64-bit patterns; the last column counts gradient entries, "
            "of both weight matrices of every block, whose 64-bit pattern differs from the keep run.\n\n")
    t1 = _table(["policy", "caches alive", "bit-identical loss", "gradient entries that differ"], rows, "lrrr")
    mid = ("\n\nThe same stack when an update has already changed one weight (W1 of block 3, entry [1][2], by +0.05) after the forward "
           "and before the backward, in both rows with recompute in segments of k = 1 (the keep row holds every block's X and A): the relative distance, in the Euclidean norm of all gradient entries, of the resulting gradient "
           "from the gradient of the unchanged weights, and from the gradient a fresh forward and backward at the changed weights "
           "would give.\n\n")
    t2 = _table(["backward after the early update", "distance from the old point's gradient", "distance from the new point's gradient"],
                [[("keep: kept X and A" if p == "keep" else "recompute: kept inputs, rebuilt A"), f"{a:.3g}", f"{b:.3g}"]
                 for p, a, b in mixed], "lrr")
    return head + t1 + mid + t2


# ------------------------------------------------------------------ 4b. the logits of the output head

HEAD_V, HEAD_D, HEAD_N = 248_320, 5_120, 4_096


def head_chunks():
    rows = []
    products = 2 * HEAD_N * HEAD_D * HEAD_V          # one [n, d] x [d, V] product, FLOPs
    for label, chunk in (("whole (every token's logits kept)", HEAD_N), ("chunks of 1,024 tokens", 1024), ("chunks of 512 tokens", 512)):
        by = 4 * chunk * HEAD_V
        flops = (3 if chunk == HEAD_N else 4) * products
        rows.append([label, f"{by:,}", _gib(by), f"{flops / 1e12:.1f}"])
    head = (f"The logits of an output head over a vocabulary of V = {HEAD_V:,} with hidden width d = {HEAD_D:,} on n = {HEAD_N:,} tokens, "
            "in fp32. The logits of t tokens are 4 t V bytes. FLOPs count 2 per multiply-add of a product and nothing else: the forward "
            "is one [n, d] by [d, V] product of 2 n d V FLOPs, the backward two (the gradient of the hidden state and the gradient of the "
            "head), so keeping the logits costs 3 products and recomputing each chunk's logits in the backward 4.\n\n")
    return head + _table(["logits held at once", "bytes", "GiB", "10^12 FLOPs of the head per step"], rows, "lrrr")


# ------------------------------------------------------------------ 5. the ladder of saved-activation policies

def _bytes_f32(rows, width):
    return 4 * rows * width


def _bytes_bf16(rows, width):
    return 4 * -(-(rows * width) // 2)


def _bytes_fp8(rows, width):
    return 4 * -(-(rows * width) // 4) + 4 * rows


FORMAT_BYTES = {"f32": _bytes_f32, "bf16": _bytes_bf16, "fp8": _bytes_fp8}


def _activation_rungs(blocks, rows=N, d=D, f=F):
    """(label, bytes) of the rungs of the ladder for a stack of `blocks` blocks, before the rule that a rung must be smaller."""
    keep = 4 * rows * blocks * (d + f)
    rec = 4 * rows * blocks * d + 4 * rows * (d + f)          # f32 boundaries and one block's X and A
    work = 2 * _bytes_f32(rows, d)                            # two f32 working buffers a narrow stream decodes into
    out = [("keep", keep), ("recompute", rec)]
    for fmt in ("bf16", "fp8"):
        out.append((f"recompute+{fmt}", rec + blocks * FORMAT_BYTES[fmt](rows, d) + work - blocks * _bytes_f32(rows, d)))
    return out


def _offered(blocks, rows=N, d=D, f=F):
    """The rungs the planner is offered: each one only if strictly smaller than the rung before it."""
    rungs = _activation_rungs(blocks, rows, d, f)
    offered = rungs[:2]
    for r in rungs[2:]:
        if r[1] < offered[-1][1]:
            offered.append(r)
    return rungs, offered


def boundary_ladder():
    rungs, offered = _offered(L)
    names = {r[0] for r in offered}
    rows = [[lab, f"{b:,}", _gib(b), f"{b / rungs[0][1]:.4f}", "yes" if lab in names else "no"] for lab, b in rungs]
    t1 = _table(["rung", "bytes", "GiB", "share of keep", "offered"], rows, "lrrrl")
    rows2 = []
    for blocks in (1, 2, 3, 4, 5, 6, 8, 16, 32):
        r, o = _offered(blocks)
        rows2.append([blocks, f"{r[2][1] - r[1][1]:+,}", f"{r[3][1] - r[1][1]:+,}", ", ".join(x[0] for x in o[2:]) or "none"])
    t2 = _table(["blocks (boundaries)", "bf16 rung minus recompute, bytes", "fp8 rung minus recompute, bytes", "narrow rungs offered"],
                rows2, "lrrl")
    head = (f"The stack of the previous tables on n = {N:,} tokens: keep holds L n (D + F) fp32 numbers; recompute holds the L fp32 "
            "boundaries and the X and A of one block; a narrow rung replaces the L fp32 boundaries by L boundaries in the format "
            "(bf16: 4 x ceil(n D / 2) bytes; fp8, E4M3 with one fp32 scale per row: 4 x ceil(n D / 4) + 4 n) and adds two fp32 "
            "working buffers of n D numbers that the narrow stream decodes into. A rung is offered only if it is smaller than the one "
            f"before it. First table: L = {L}. Second table: the same quantities for other L.\n\n")
    return head + t1 + "\n\n" + t2


# ------------------------------------------------------------------ 5b. narrow boundaries change the loss a little

def boundary_rounding():
    name = "training_memory.boundary_rounding"
    rng = bookrng.generator(name)
    rows_in = [[rng.normal() for _ in range(8)] for _ in range(3)]
    rows_in[2][3] *= 20.0
    t1rows = []
    for r_i, row in enumerate(rows_in):
        row_out = []
        for kind in ("bf16", "fp8"):
            back = _round_trip([row], kind)[0]
            rel = max(abs(a - b) / abs(a) for a, b in zip(row, back))
            amax = max(abs(a) for a in row)
            rowmax = max(abs(a - b) for a, b in zip(row, back)) / amax
            row_out += [f"{rel:.3g}", f"{rowmax:.3g}"]
        t1rows.append([r_i, f"{max(abs(a) for a in row):.3f}"] + row_out)
    # the stack with narrow boundaries
    blocks, n, d, f = 6, 64, 16, 32
    seed_stack = bookrng.generator(name + ".stack")
    Ws, X0 = _draw_stack(seed_stack, blocks, n, d, f, 0.25)
    loss0, g0, _ = _run_stack(Ws, X0, "recompute", 1, "f32")
    v0 = _flat(g0)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    t2rows = []
    for kind in ("bf16", "fp8"):
        loss, g, _ = _run_stack(Ws, X0, "recompute", 1, kind)
        mincos = min(cos(_flat([gl]), _flat([g0l])) for gl, g0l in zip(g, g0))
        # minimum over the 2L weight matrices
        mats = [(gl[w], g0l[w]) for gl, g0l in zip(g, g0) for w in (0, 1)]
        mincos = min(cos([v for r in a for v in r], [v for r in b for v in r]) for a, b in mats)
        t2rows.append([kind, f"{abs(loss - loss0) / abs(loss0):.3g}", f"{cos(_flat(g), v0):.6f}", f"{mincos:.6f}"])
    head = (f"Three rows of 8 numbers, drawn from the generator seeded {bookrng.seed_of(name)} (seed_of '{name}') as normal() in order, row "
            "by row; entry 3 of row 2 (counting from 0) is then multiplied by 20, so it is the largest entry of its row. Each row is stored "
            "in bf16 (nearest even) or fp8 (the row's own scale s = amax / 448, then E4M3 rounding to nearest even with a subnormal "
            "step of 2^-9 and saturation at 448, then multiplication by s) and read back. 'relative error' is the largest |error| / |x| over the "
            "row; 'error / row max' is the largest |error| / amax.\n\n")
    t1 = _table(["row", "amax", "bf16, relative error", "bf16, error / row max", "fp8, relative error", "fp8, error / row max"],
                t1rows, "lrrrrr")
    mid = (f"\n\nA stack of {blocks} blocks (D = {d}, F = {f}, n = {n}), drawn as in the first table from the generator seeded "
           f"{bookrng.seed_of(name + '.stack')} (seed_of '{name}.stack') with entries of W1 and W2 scaled by 0.25 in place of 0.5 (each entry normal() times 0.25, W1 then W2 of block 0, then of block 1, and so on, then the {n} x {d} entries of X as normal() times 1 in row-major order), run with recompute in segments of 1. Each block's input is "
           "stored in the format and the block reads it back, in the forward and in the rebuild, so the forward continues from the "
           "rounded value. Compared with the fp32 boundaries: the relative change of the loss, the cosine between the two complete "
           "gradients, and the smallest cosine over the 12 weight matrices.\n\n")
    t2 = _table(["boundary format", "relative change of the loss", "cosine of the whole gradient", "smallest cosine of a matrix"],
                t2rows, "lrrr")
    return head + t1 + mid + t2


# ------------------------------------------------------------------ 6. the planner

def _plan(demands, cap_dev, cap_host, authorized):
    """demands: list of (category, [variant dicts]); the first variant is the base. Levers are pulled in the fixed order,
    each category in the listed order, only while the total does not fit, never to a variant whose host bytes overflow."""
    chosen = [0] * len(demands)

    def totals(c):
        return (sum(d[1][i]["dev"] for d, i in zip(demands, c)), sum(d[1][i]["host"] for d, i in zip(demands, c)))

    def fits(c):
        dv, h = totals(c)
        return dv <= cap_dev and h <= cap_host

    decisions = []
    order = ["state", "recompute", "precision", "offload"]
    done = False
    for lever in order:
        if lever not in authorized or done:
            continue
        for ci, (cat, variants) in enumerate(demands):
            while True:
                if fits(chosen):
                    done = True
                    break
                nxt = next((i for i in range(chosen[ci] + 1, len(variants)) if variants[i]["lever"] == lever), None)
                if nxt is None:
                    break
                v = variants[nxt]
                if lever == "precision" and not v["validated"]:
                    break
                trial = list(chosen)
                trial[ci] = nxt
                if totals(trial)[1] > cap_host:
                    break
                decisions.append((lever, cat, variants[chosen[ci]]["label"], v["label"]))
                chosen = trial
            if done:
                break
    return chosen, decisions, totals(chosen), fits(chosen)


def _training_demands():
    state_f32 = 2 * 4 * P
    state_bf16 = 2 * 2 * P
    state_i8 = 2 * (P + 4 * -(-P // 256))
    _, offered = _offered(L)
    names = {o[0] for o in offered}
    rungs = _activation_rungs(L)
    act = [dict(lever="base", label="keep", dev=rungs[0][1], host=0, validated=True),
           dict(lever="recompute", label="recompute", dev=rungs[1][1], host=0, validated=True)]
    for lab, b in rungs[2:]:
        if lab in names:
            act.append(dict(lever="precision", label=lab, dev=b, host=0, validated=lab != "recompute+fp8"))
    return [
        ("weights", [dict(lever="base", label="fp32 values", dev=4 * P, host=0, validated=True)]),
        ("gradients", [dict(lever="base", label="fp32 gradients", dev=4 * P, host=0, validated=True)]),
        ("optimizer state", [dict(lever="base", label="f32 moments", dev=state_f32, host=0, validated=True),
                             dict(lever="state", label="bf16 moments", dev=state_bf16, host=0, validated=True),
                             dict(lever="state", label="int8 blocks of 256", dev=state_i8, host=0, validated=True),
                             dict(lever="offload", label="f32 moments on the host", dev=0, host=state_f32, validated=True)]),
        ("activations", act),
    ]


PLAN_CAPS = (96, 80, 64, 48, 40, 32, 24)
PLAN_HOST = 64


def memory_plans():
    demands = _training_demands()
    auth = {"state", "recompute", "precision", "offload"}
    rows = []

    def row(cap, authorized, label=None):
        chosen, decisions, (dv, h), ok = _plan(demands, cap * GIB, PLAN_HOST * GIB, authorized)
        pick = {cat: variants[i]["label"] for (cat, variants), i in zip(demands, chosen)}
        rows.append([label or cap, pick["optimizer state"], pick["activations"], len(decisions), _gib(dv), _gib(h),
                     "fits" if ok else "refused"])

    for cap in PLAN_CAPS:
        row(cap, auth)
    row(40, {"recompute"}, "40, only recompute authorized")
    head = (f"The stack of the previous tables (P = {P:,}, L = {L}, n = {N:,}) in fp32 with AdamW. Categories and variants, in the "
            f"order the planner visits them: weights, 4 P = {_gib(4 * P)} GiB (fixed); gradients, {_gib(4 * P)} GiB (fixed); optimizer "
            f"state, f32 moments {_gib(2 * 4 * P)} GiB, then bf16 moments {_gib(2 * 2 * P)}, then int8 blocks of 256 "
            f"{_gib(2 * (P + 4 * -(-P // 256)))} (code plus one fp32 scale per 256 numbers, both moments), then the f32 moments held on "
            f"the host ({_gib(8 * P)} GiB of host bytes, 0 on the device); activations, the rungs of the previous table (the fp8 rung "
            "is not one the planner may choose on its own). The planner pulls the levers in the order state, recompute, activation "
            "precision, offload; within a lever it visits the categories in the order listed, takes the next variant of that lever, and "
            f"stops the moment the device total fits. Host capacity is {PLAN_HOST} GiB; device capacities are in GiB. 'decisions' counts "
            "the variants taken. When nothing fits, the row is the smallest plan the authorized levers reach.\n\n")
    return head + _table(["device GiB", "optimizer state", "activations", "decisions", "device GiB used", "host GiB used", "outcome"],
                         rows, "lllrrrl")


# ------------------------------------------------------------------ 7. optimizer state in bytes

def optimizer_bytes():
    def state_bytes(slots, fmt, block=None):
        per = {"f32": 4 * P, "bf16": 2 * P, "int8": P + 4 * -(-P // block) if block else None}[fmt]
        return slots * per

    cases = [("AdamW (2 slots)", 2, "f32", None), ("AdamW", 2, "bf16", None), ("AdamW", 2, "int8", 256), ("AdamW", 2, "int8", 64),
             ("Lion or SGD with momentum (1 slot)", 1, "f32", None), ("Lion or SGD with momentum", 1, "int8", 64)]
    rows = []
    for name, slots, fmt, block in cases:
        b = state_bytes(slots, fmt, block)
        label = fmt if fmt != "int8" else f"int8, blocks of {block}"
        rows.append([name, label, f"{b:,}", f"{b / GIB:.3f}", f"{b / P:.4f}"])
    # Adafactor: rows + columns of every matrix of the stack
    per_block = (F + D) + (D + F)
    af = 4 * per_block * L
    rows.append(["Adafactor (factored, no first moment)", "rows + columns, fp32", f"{af:,}", f"{af / GIB:.3f}", f"{af / P:.6f}"])
    head = (f"The optimizer state of the stack, P = {P:,} numbers, counted physically: an fp32 slot is 4 bytes a number, "
            "a bf16 slot 2, an int8 slot 1 byte a number plus one fp32 scale (4 bytes) per block (the last block counted whole, "
            "here P divides evenly). Adafactor keeps, for each of the 2 L matrices of shape [F, D] and [D, F], one fp32 number per row "
            "and one per column.\n\n")
    t1 = _table(["rule", "format of a slot", "bytes", "GiB", "bytes per parameter"], rows, "llrrr")
    return head + t1


def factored_state():
    a, b = [1.0, 4.0, 9.0], [1.0, 2.0, 3.0, 4.0]
    cases = [("squares g^2 of rank one, G2_ij = a_i b_j with a = (1, 4, 9), b = (1, 2, 3, 4)",
              [[x * y for y in b] for x in a]),
             ("squares of the identity pattern, G2 = [[4, 0], [0, 4]]", [[4.0, 0.0], [0.0, 4.0]])]
    out = []
    for title, G2 in cases:
        n, k = len(G2), len(G2[0])
        R = [sum(r) / k for r in G2]
        C = [sum(G2[i][j] for i in range(n)) / n for j in range(k)]
        mR = sum(R) / n
        V = [[R[i] * C[j] / mR for j in range(k)] for i in range(n)]
        err = max(abs(V[i][j] - G2[i][j]) for i in range(n) for j in range(k))
        rows = [["R (row means)", ", ".join(f"{x:g}" for x in R)], ["C (column means)", ", ".join(f"{x:g}" for x in C)],
                ["mean of R", f"{mR:g}"],
                ["V = R C^T / mean(R), row by row", " | ".join(", ".join(f"{x:.6g}" for x in r) for r in V)],
                ["largest |V - G2|", f"{err:.3g}" if err > 1e-9 else "below 1e-12"], ["numbers kept, rows + columns, against rows x columns", f"{n + k} against {n * k}"]]
        out.append(f"{title}:\n\n" + _table(["quantity", "value"], rows, "ll"))
    head = ("Adafactor's second moment of a matrix of squared gradients G2 with n rows and k columns, kept as the row means R and the "
            "column means C of G2 and rebuilt as V = R C^T / mean(R), where mean(R) is the mean of the n row means (the mean of all "
            "entries of G2). The statistics of a single step, with no averaging over steps, are shown.\n\n")
    return head + "\n\n".join(out)


# ------------------------------------------------------------------ 8. eight-bit codes with a scale per block

def _enc(map_, x, s, u=0.5):
    """The code of x in a block whose scale is s, for the named map; u = 0.5 is rounding to nearest (floor(y + u)),
    a uniform draw in [0, 1) is stochastic rounding. Signed maps return a signed integer magnitude in -127 .. 127,
    non-negative maps 0 .. 255."""
    if s <= 0.0:
        return 0
    a = min(abs(x) / s, 1.0)
    if map_ == "linear-signed":
        q = min(max(math.floor(a * 127 + u), 0), 127)
        return -q if x < 0 else q
    if map_ == "sqrt":
        q = min(max(math.floor(math.sqrt(a) * 127 + u), 0), 127)
        return -q if x < 0 else q
    if map_ == "log-signed":
        if a <= 0.0:
            return 0
        q = min(max(math.floor(127 + math.log2(a) * 8 + u), 0), 127)
        return -q if x < 0 else q
    if map_ == "linear":
        return min(max(math.floor(a * 255 + u), 0), 255)
    if map_ == "fourth-root":
        if x <= 0.0:
            return 0
        return min(max(math.floor(a ** 0.25 * 255 + u), 1), 255)
    if map_ == "log":
        if x <= 0.0 or a <= 0.0:
            return 0
        return min(max(math.floor(255 + math.log2(a) * 16 + u), 1), 255)
    raise ValueError(map_)


def _dec(map_, q, s):
    if map_ == "linear-signed":
        return math.copysign(abs(q) / 127 * s, q) if q else 0.0
    if map_ == "sqrt":
        return math.copysign((abs(q) / 127) ** 2 * s, q) if q else 0.0
    if map_ == "log-signed":
        return math.copysign(s * 2.0 ** ((abs(q) - 127) / 8), q) if q else 0.0
    if map_ == "linear":
        return q / 255 * s
    if map_ == "fourth-root":
        return (q / 255) ** 4 * s
    if map_ == "log":
        return s * 2.0 ** ((q - 255) / 16) if q else 0.0
    raise ValueError(map_)


def _quantise_block(map_, xs, u=0.5):
    s = max(abs(x) for x in xs)
    return [_dec(map_, _enc(map_, x, s, u), s) for x in xs], s


RATIOS = (1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-6, 1e-8)


def _worst_relative(map_, rho):
    """The largest relative error of rounding to nearest over 400 values evenly spaced in the logarithm between rho and
    2 rho, in a block whose maximum (the scale) is 1, and whether any of them decoded as zero."""
    worst, zero = 0.0, False
    for i in range(400):
        x = min(rho * 2.0 ** (i / 399), 1.0)
        y = _dec(map_, _enc(map_, x, 1.0), 1.0)
        zero = zero or y == 0.0
        worst = max(worst, abs(y - x) / x)
    return worst, zero


def state_maps():
    def cell(map_, rho):
        w, zero = _worst_relative(map_, rho)
        if zero:
            return "100 (zero)"
        return f"{100 * w:,.3g}" + (" (raised)" if w > 1.0 else "")

    non = [[f"{rho:g}"] + [cell(m, rho) for m in ("linear", "fourth-root", "log")] for rho in RATIOS]
    sig = [[f"{rho:g}"] + [cell(m, rho) for m in ("linear-signed", "sqrt", "log-signed")] for rho in RATIOS]
    head = ("The largest relative error, in percent, of encoding a value x with ratio rho = x / (block maximum) to nearest and "
            "decoding it, searched over 400 values evenly spaced in the logarithm from rho to 2 rho. A block's scale is its maximum, "
            "so rho <= 1. 'zero' marks a value that decodes as 0 (error 100 percent); 'raised' marks a value below the smallest level "
            "that the encoder raises to it. A value above the scale is capped at it, and the 400 values include both ends, x = rho 2^(j / 399) for j = 0 .. 399. "
            "The fourth-root and log maps of the non-negative slots have 255 positive levels (codes 1 to 255; their encoder clamps a "
            "positive value up to code 1, so a decode is never zero); the linear map is the unclamped baseline, whose code 0 decodes as "
            "zero, so a value below half a level is lost; signed slots have a sign and 127 magnitude levels (code 0 is zero). "
            "'linear' puts level q at q / 255 (or q / 127) of the scale; 'fourth-root' puts it at (q / 255)^4, 'sqrt' at (q / 127)^2, "
            "and 'log' at 2^((q - 255) / 16) (16 levels an octave; signed: 2^((q - 127) / 8), 8 levels an octave).\n\n")
    return (head + "Second-moment slots (non-negative):\n\n" + _table(["rho", "linear", "fourth-root", "log"], non, "lrrr")
            + "\n\nFirst-moment slots (signed):\n\n" + _table(["rho", "linear", "sqrt", "log"], sig, "lrrr"))


BLOCK_N = 64


def block_codes():
    name = "training_memory.block_codes"
    rng = bookrng.generator(name)
    g = []
    for _ in range(BLOCK_N):
        a = rng.normal()
        b = rng.normal()
        g.append(a * math.exp(2.0 * b))
    m, v = list(g), [x * x for x in g]
    eps = 1e-8
    gmax, gmin = max(abs(x) for x in g), min(abs(x) for x in g)

    def stats(mmap, vmap):
        mq, _ = _quantise_block(mmap, m)
        vq, _ = _quantise_block(vmap, v)
        u = [mq_i / (math.sqrt(vq_i) + eps) for mq_i, vq_i in zip(mq, vq)]
        zeros_v = sum(1 for x in vq if x == 0.0)
        zeros_m = sum(1 for x in mq if x == 0.0)
        wrong = sum(1 for x, mi in zip(u, m) if abs(x - math.copysign(1.0, mi)) > 0.1)
        return zeros_m, zeros_v, wrong, max(abs(x) for x in u)

    cases = [("linear 8-bit for both", "linear-signed", "linear"), ("sqrt map for m, fourth-root map for v", "sqrt", "fourth-root"),
             ("log map for both", "log-signed", "log")]
    rows = []
    for label, mm_, vm_ in cases:
        zm, zv, wrong, big = stats(mm_, vm_)
        rows.append([label, zm, zv, wrong, f"{big:.3g}"])
    head = (f"One block of {BLOCK_N} gradients g_i = a_i exp(2 b_i), a_i and b_i from normal() of the generator seeded "
            f"{bookrng.seed_of(name)} (seed_of '{name}'), in the order a_0, b_0, a_1, b_1, ...; they span |g| from {gmin:.3g} to {gmax:.3g}. "
            "After one Adam step from zero (bias-corrected) the first moment is m = g and the second moment v = g^2, so the exact update "
            "m / (sqrt(v) + epsilon) is sign(g), of size 1 (epsilon = 1e-8). Each slot is encoded to nearest with its own block scale "
            "(the block maximum) and decoded, and the update is formed from the decoded values. The linear rows are the unclamped baseline of the previous table: a value that "
            "rounds to code 0 decodes as 0, while the fourth-root and log maps raise a positive value to code 1.\n\n")
    return head + _table(["codes", "m decoded as 0", "v decoded as 0", "updates farther than 0.1 from sign(g)",
                          "largest |update|"], rows, "lrrrr")


EMA_STEPS, EMA_BETA = 1000, 0.999


def ema_decay():
    name = "training_memory.ema_decay"
    rng = bookrng.generator(name)
    n = 63
    start = _dec("fourth-root", 214, 1.0)
    v_nearest = [start] * n
    v_stoch = [start] * n
    marks = {100, 300, 600, 1000}
    rows = []

    def code(x, u):
        return _enc("fourth-root", x, 1.0, u)

    def mean(vs):
        return sum(vs) / len(vs)

    def snap(t):
        mu = mean(v_stoch)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v_stoch) / (len(v_stoch) - 1))
        rows.append([t, f"{start * EMA_BETA ** t:.5f}", f"{mean(v_nearest):.5f}", f"{mu:.5f}", f"{sd:.5f}",
                     f"{sd / math.sqrt(len(v_stoch)):.5f}"])

    snap(0)
    for t in range(1, EMA_STEPS + 1):
        v_nearest = [_dec("fourth-root", code(EMA_BETA * x, 0.5), 1.0) for x in v_nearest]
        v_stoch = [_dec("fourth-root", code(EMA_BETA * x, rng.uniform()), 1.0) for x in v_stoch]
        if t in marks:
            snap(t)
    head = (f"A block of 64 second moments: element 0 is held at 1.0 by a large gradient every step, so the block's scale stays 1.0; "
            f"the other {n} elements start at the value of code 214, (214 / 255)^4 = {start:.5f} (the code nearest to 0.5), and receive a "
            f"gradient of zero, so the exact update is v <- {EMA_BETA} v. "
            "Each element is stored with the fourth-root map, code = clamp(floor(255 v^(1/4) + u), 1, 255) and v = (code / 255)^4, "
            "and re-encoded after every step: u = 0.5 is rounding to nearest, and u = uniform() from the generator seeded "
            f"{bookrng.seed_of(name)} (seed_of '{name}') is stochastic rounding, one draw per element per step, step by step, the elements "
            "in order 1 to 63 (element 0 is not rounded). The columns are the exact value, start x 0.999^step, the mean over the 63 "
            "elements, and for the stochastic run the standard deviation of the 63 (divisor 62) and the standard error of their mean "
            "(that deviation over sqrt(63)).\n\n")
    return head + _table(["step", "exact start x 0.999^step", "mean, nearest", "mean, stochastic", "std of the 63", "std error of the mean"], rows, "rrrrrr")


# ------------------------------------------------------------------ 9. a short training run with eight-bit moments

Q_N, Q_STEPS, Q_LR, Q_B1, Q_B2, Q_EPS = 256, 300, 0.02, 0.9, 0.999, 1e-8
Q_BLOCK = 64


def _q_run(mode, nudge=0.0):
    """AdamW (no weight decay) on L = sum_i h_i (w_i - t_i)^2 / 2 from w = 0 for Q_STEPS steps. mode 'f32': fp32 moments (float64
    here); 'sr': both moments in 8-bit blocks of 64 with stochastic rounding drawn by the counter draw; 'nearest': the same,
    rounded to nearest. Returns the loss after steps 10, 50 and 300."""
    rng = bookrng.generator("training_memory.int8_training")
    t = [rng.normal() for _ in range(Q_N)]
    h = [10.0 ** (-1.0 + 2.0 * i / (Q_N - 1)) for i in range(Q_N)]
    w = [0.0] * Q_N
    w[0] += nudge
    m, v = [0.0] * Q_N, [0.0] * Q_N
    out = {}
    for step in range(1, Q_STEPS + 1):
        g = [h[i] * (w[i] - t[i]) for i in range(Q_N)]
        for i in range(Q_N):
            m[i] = Q_B1 * m[i] + (1 - Q_B1) * g[i]
            v[i] = Q_B2 * v[i] + (1 - Q_B2) * g[i] * g[i]
            w[i] -= Q_LR * (m[i] / (1 - Q_B1 ** step)) / (math.sqrt(v[i] / (1 - Q_B2 ** step)) + Q_EPS)
        if mode != "f32":
            for slot, arr, map_ in ((0, m, "sqrt"), (1, v, "fourth-root")):
                key = mp.mix32(2 * step + slot)
                for b0 in range(0, Q_N, Q_BLOCK):
                    s = max(abs(x) for x in arr[b0:b0 + Q_BLOCK])
                    for i in range(b0, b0 + Q_BLOCK):
                        u = mp.counter_uniform(key, i) if mode == "sr" else 0.5
                        arr[i] = _dec(map_, _enc(map_, arr[i], s, u), s)
        if step in (10, 50, 100, Q_STEPS):
            out[step] = 0.5 * sum(h[i] * (w[i] - t[i]) ** 2 for i in range(Q_N))
    return out


def int8_training():
    rows = []
    for label, mode in (("fp32 moments", "f32"), ("8-bit blocks of 64, stochastic rounding", "sr"),
                        ("8-bit blocks of 64, nearest", "nearest")):
        base = _q_run(mode)
        ws = [_q_run(mode, d) for d in stable.NUDGES]
        cells = [stable.text(base[s], [x[s] for x in ws], lambda x: f"{x:.4g}") for s in (10, 50, 100, Q_STEPS)]
        rows.append([label] + cells)
    seed = bookrng.seed_of("training_memory.int8_training")
    head = (f"AdamW (beta1 = {Q_B1}, beta2 = {Q_B2}, epsilon = {Q_EPS:g}, rate {Q_LR}, no weight decay, bias-corrected) on "
            f"L = sum_i h_i (w_i - t_i)^2 / 2 over {Q_N} weights from w = 0, with curvature h_i = 10^(-1 + 2 i / {Q_N - 1}) and targets "
            f"t_i = normal() in order i = 0 .. {Q_N - 1} from one generator seeded {seed} (seed_of 'training_memory.int8_training'). "
            "In the 8-bit rows both moments are re-encoded after every update, the first with the square-root map and the second with "
            f"the fourth-root map, each in blocks of {Q_BLOCK} weights with the block maximum as scale; the stochastic rounding uses "
            "the counter draw of Chapter 2, u = counter_uniform(key, i) for weight i, with key = mix32(2 t + slot) at step t = 1, 2, ... and "
            "slot 0 for the first moment, 1 for the second. The cells are the loss after steps 10, 50, 100 and 300. "
            + stable.RULE + " In the nudged runs weight 0 starts at the stated tiny amount instead of 0.\n\n")
    g0 = bookrng.generator("training_memory.int8_training")
    start = 0.5 * sum((10.0 ** (-1.0 + 2.0 * i / (Q_N - 1))) * g0.normal() ** 2 for i in range(Q_N))
    return head + _table(["moments", "loss after step 10", "after step 50", "after step 100", "after step 300"], rows, "lrrrr") + \
        f"\n\nThe loss at w = 0, sum_i h_i t_i^2 / 2, is {start:.2f}."


# ------------------------------------------------------------------ 10. the update in tiles

T_N, T_BLOCK, T_STEPS, T_LR = 1000, 64, 5, 0.05


def _effective_tile(requested, block=T_BLOCK):
    """The requested tile rounded up to the next multiple of the block (`optim::streamed::effective_tile`)."""
    return -(-requested // block) * block


def _streamed_run(tile):
    """T_STEPS AdamW steps over T_N weights with both moments in 8-bit blocks of 64 and stochastic rounding keyed on the
    absolute index. The update walks the tensor in tiles of `tile` elements: decode the tile's moments (each under the scale
    stored for its block), apply the rule, and encode the tile: every block of the grid that the tile touches is encoded from
    the part of it inside the tile and its scale is written, so a tile that starts or ends inside a block writes a scale
    computed from a piece of it."""
    rng = bookrng.generator("training_memory.tile_independence")
    t = [rng.normal() for _ in range(T_N)]
    hh = [1.0 + (i % 7) for i in range(T_N)]
    w = [0.0] * T_N
    nb = -(-T_N // T_BLOCK)
    codes = {0: [0] * T_N, 1: [0] * T_N}
    scales = {0: [0.0] * nb, 1: [0.0] * nb}
    maps = {0: "sqrt", 1: "fourth-root"}
    for step in range(1, T_STEPS + 1):
        g = [hh[i] * (w[i] - t[i]) for i in range(T_N)]
        for a in range(0, T_N, tile):
            b = min(a + tile, T_N)
            mv = {slot: [_dec(maps[slot], codes[slot][i], scales[slot][i // T_BLOCK]) for i in range(a, b)] for slot in (0, 1)}
            for j, i in enumerate(range(a, b)):
                m = 0.9 * mv[0][j] + 0.1 * g[i]
                v = 0.999 * mv[1][j] + 0.001 * g[i] * g[i]
                w[i] -= T_LR * (m / (1 - 0.9 ** step)) / (math.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
                mv[0][j], mv[1][j] = m, v
            for slot in (0, 1):
                key = mp.mix32(2 * step + slot)
                for bi in range(a // T_BLOCK, (b - 1) // T_BLOCK + 1):
                    s0, e0 = max(bi * T_BLOCK, a), min((bi + 1) * T_BLOCK, b)
                    vals = mv[slot][s0 - a:e0 - a]
                    sc = max(abs(x) for x in vals)
                    scales[slot][bi] = sc
                    for off, x in enumerate(vals):
                        codes[slot][s0 + off] = _enc(maps[slot], x, sc, mp.counter_uniform(key, s0 + off))
    return w


def tile_independence():
    whole = _streamed_run(T_N)
    rows = []
    for req in (64, 192, 256, 1000, 100):
        eff = _effective_tile(req)
        w = _streamed_run(min(eff, T_N))
        diff = sum(1 for a, b in zip(w, whole) if _bits(a) != _bits(b))
        rows.append([req, f"{eff:,}", f"{20 * eff:,}", f"{diff} of {T_N}"])
    cut = _streamed_run(100)
    diff = sum(1 for a, b in zip(cut, whole) if _bits(a) != _bits(b))
    rows.append(["100, used as requested", "100", f"{20 * 100:,}", f"{diff} of {T_N}"])
    head = (f"{T_STEPS} AdamW steps (beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, rate {T_LR}, no weight decay, bias-corrected) over "
            f"{T_N} weights from w = 0 on L = sum_i h_i (w_i - t_i)^2 / 2 with h_i = 1 + (i mod 7) and t_i = normal() in order from "
            f"the generator seeded {bookrng.seed_of('training_memory.tile_independence')} (seed_of 'training_memory.tile_independence'). Both moments are held in "
            f"8-bit blocks of {T_BLOCK} (square-root map for the first, fourth-root map for the second) with stochastic rounding by the "
            "counter draw, u = counter_uniform(key, i) with i the absolute index and key = mix32(2 t + slot) (t the step from 1, slot 0 "
            "or 1). The update walks the weights in tiles: decode the tile's moments, apply the rule, encode them. The requested tile is "
            "rounded up to the next multiple of the block, which is the tile used; the staging column is 5 fp32 tiles, "
            "20 bytes an element of the tile. The last column counts weights after step 5 whose 64-bit pattern differs from the run that "
            "takes the whole tensor as one tile.\n\n")
    return head + _table(["tile requested", "tile used", "staging bytes of the five tiles", "weights that differ"], rows, "rrrr")


def staging_bytes():
    rows = []
    for n in (10 ** 6, 10 ** 9, 7 * 10 ** 9):
        state = 2 * (n + 4 * -(-n // 256)) + 4 * n
        cells = [f"{n:,}", f"{state:,}"]
        for tile in (1 << 16, 1 << 26):
            st = 5 * tile * 4 + 4 * tile + 4 * -(-tile // 256)
            cells.append(f"{st:,}")
        rows.append(cells)
    head = ("The bytes of a streamed AdamW step for a tensor of n numbers whose two moments are in 8-bit blocks of 256 and whose "
            "master copy is fp32: the state, 2 (n + 4 ceil(n / 256)) + 4 n bytes, and the staging, which is five fp32 tiles (value or "
            "master, gradient, two moments: 20 bytes a tile element) plus the storage's byte buffer of 4 bytes a tile element and one "
            "fp32 scale per 256 elements of the tile. Tiles of 2^16 and 2^26 elements.\n\n")
    return head + _table(["n", "state bytes", "staging bytes, tile 2^16", "staging bytes, tile 2^26"], rows, "rrrr")


# ------------------------------------------------------------------ shared with the figures

def segment_curve():
    """(k, numbers per token) for every k from 1 to L, with the keep line, for the figure."""
    return [(k, (L / k) * D + k * (D + F)) for k in range(1, L + 1)], L * (D + F)
