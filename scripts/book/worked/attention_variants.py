# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Attention Variants: Sharing and Compressing Keys": how to shrink the key-value cache.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds between
`<!-- worked: attention_variants.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a
picture and the table beside it cannot disagree.

* the bytes of cache per token of multi-head, grouped-query, multi-query and multi-head latent attention at one model shape,
  and the parameters of their key and value paths;
* a tiny multi-head latent attention layer (a latent, up-projections, a decoupled rotary key) run explicitly and run with
  the up-projections absorbed into the query and the output, and the two compared;
* why a rotation between the up-projection and the query breaks absorption, and what the decoupled key does about it;
* FLOPs and bytes per cached key of every variant, and the arithmetic intensity that follows;
* the receptive field of stacked sliding windows, a ring buffer's slack, and the cache of a hybrid of windowed and global layers;
* a softmax row that has nowhere to put its mass, and a stream where the first token leaves the window.

The rotary rotation is the one of Volume V, Chapter 8 (`rotary`, implemented in `positions`), imported and not copied. Every random quantity comes
from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products are plain left-to-right
additions, the order a C loop would use (never Python's `sum`, which may compensate).

Swedish Embedded AB implements solutions for serving language models within a memory budget. If your team needs expertise in
key-value cache design for long contexts, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import modern_decoder as md
import positions as pos_mod

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _below(x, limit=1e-12):
    """'below 1e-12' when x is round-off noise under the limit, else the value to three decimals."""
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else _fx(x, 3)


def _vec(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _matvec(W, x):
    """W x for W of shape [m, n]: row by row, each sum left to right."""
    return [_dot(row, x) for row in W]


def _matvec_t(W, v):
    """W^T v for W of shape [m, n]: entry c is the sum over the rows r = 0 .. m-1 of W[r][c] v[r]."""
    out = []
    for c in range(len(W[0])):
        s = 0.0
        for r in range(len(W)):
            s += W[r][c] * v[r]
        out.append(s)
    return out


def _matmul(A, B):
    """A B for A of shape [m, k] and B of shape [k, n]; each entry a sum over p = 0 .. k-1 left to right."""
    cols = list(zip(*B))
    return [[_dot(row, col) for col in cols] for row in A]


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = 0.0
    for x in e:
        s += x
    return [x / s for x in e]


def _weighted_sum(p, rows):
    """sum_j p[j] rows[j], accumulated over j ascending."""
    out = [0.0] * len(rows[0])
    for w, row in zip(p, rows):
        for d, x in enumerate(row):
            out[d] += w * x
    return out


def _max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _gib(b):
    return b / 2 ** 30


# ------------------------------------------------------------------------------------------------ cache of one layer, four ways

SHAPE = md.QWEN3_8B                    # 36 layers, 32 query heads, 8 key-value heads, head width 128, width 4,096
BYTES = md.BYTES                       # bf16
LATENT = 4 * SHAPE["dh"]               # d_c = 4 d_h, the setting reported for DeepSeek-V2
ROPE_KEY = SHAPE["dh"] // 2            # d_R = d_h / 2, likewise
CONTEXT = 131072                       # 128 Ki tokens


def numbers_per_layer(kind, s=SHAPE):
    """Numbers one token adds to the cache of one layer."""
    dh = s["dh"]
    return {"MHA": 2 * s["H"] * dh, "GQA": 2 * s["hkv"] * dh, "MQA": 2 * dh, "MLA": LATENT + ROPE_KEY}[kind]


def kv_parameters(kind, s=SHAPE):
    """Parameters of the key and value path of one layer (the matrices that produce what is cached or reconstructed)."""
    D, H, hkv, dh = s["D"], s["H"], s["hkv"], s["dh"]
    if kind == "MLA":
        # W_DKV [d_c, D], W_KR [d_R, D], W_UK and W_UV for every head [d_h, d_c] and [d_v, d_c] (d_v = d_h here)
        return LATENT * D + ROPE_KEY * D + 2 * H * dh * LATENT
    heads = {"MHA": H, "GQA": hkv, "MQA": 1}[kind]
    return 2 * D * heads * dh


KINDS = (("MHA", "multi-head", "32 key-value heads"), ("GQA", "grouped-query", "8 key-value heads"),
         ("MQA", "multi-query", "1 key-value head"), ("MLA", "multi-head latent", "latent of 512 and a rotary key of 64"))


def cache_sizes_data():
    """(label, bytes of cache per token) for the four variants at the shape of SHAPE."""
    return [(name, SHAPE["L"] * numbers_per_layer(k) * BYTES) for k, name, _ in KINDS]


def _cache_rows(kinds, with_parameters):
    L = SHAPE["L"]
    mha_bytes = L * numbers_per_layer("MHA") * BYTES
    rows = []
    for k, name, stores in KINDS:
        if k not in kinds:
            continue
        per_layer = numbers_per_layer(k)
        per_token = L * per_layer * BYTES
        row = [f"{name} ({k})", stores, _int(per_layer), _int(per_token), f"{_gib(per_token * CONTEXT):.4f}",
               f"{per_token / mha_bytes:.4f}"]
        if with_parameters:
            row.append(_int(kv_parameters(k)))
        rows.append(row)
    return rows


def cache_sizes():
    weights = md.parameters(SHAPE) * BYTES
    return (f"The decoder shape of Qwen3 8B (Chapter 1): {SHAPE['L']} layers, {SHAPE['H']} query heads of width {SHAPE['dh']}, "
            f"model width {_int(SHAPE['D'])}, bf16 ({BYTES} bytes). The rows differ only in key-value heads. Numbers per layer: "
            f"2 x key-value heads x head width. Bytes per token = layers x numbers x {BYTES}. The GiB column is a context of "
            f"{_int(CONTEXT)} tokens.\n\n"
            + _table(["variant", "what a layer stores", "numbers per token and layer", "bytes per token", "GiB at 131,072 tokens",
                      "fraction of the multi-head cache"], _cache_rows(("MHA", "GQA", "MQA"), False), "llrrrr")
            + f"\n\nFor scale: the {_int(md.parameters(SHAPE))} parameters of this shape are {_int(weights)} bytes in bf16, "
            f"{_gib(weights):.2f} GiB.")


def cache_sizes_latent():
    return ("The same shape with multi-head latent attention added, with the settings reported for DeepSeek-V2: a latent of "
            f"$d_c = 4 d_h = {LATENT}$ numbers and a rotary key of $d_R = d_h / 2 = {ROPE_KEY}$. This chapter counts this layer and "
            "does not train it. Numbers per layer: $d_c + d_R$ for the latent variant, 2 x key-value heads x head width for "
            "the others. The last column counts the parameters of the matrices that produce keys and values: $2 D h_{kv} d_h$, "
            "and for the latent variant $d_c D + d_R D + 2 H d_h d_c$ (the down-projection, the rotary key and the two "
            "up-projections of all heads).\n\n"
            + _table(["variant", "what a layer stores", "numbers per token and layer", "bytes per token", "GiB at 131,072 tokens",
                      "fraction of the multi-head cache", "key-value path parameters per layer"], _cache_rows(("MHA", "GQA", "MQA", "MLA"), True),
                     "llrrrrr"))


# ------------------------------------------------------------------------------------------------ a tiny latent attention layer

NAME_MLA = "attention_variants.mla"
T, D, H, DH, DV, DC, DR = 8, 7, 3, 4, 6, 5, 2   # tokens, width, heads, key width, value width, latent width, rotary key width
WS_R = [1.0]                                     # the one rotary pair of the decoupled key: 1.0 radian per position
WS_C = [1.0, 0.5]                                # the two pairs of a rotated content key (for the section on why absorption breaks)
SCALE = 1.0 / math.sqrt(DH + DR)


def _draw(g):
    return md._round1(0.5 * g.normal())


def _draw_matrix(g, rows, cols):
    return [[_draw(g) for _ in range(cols)] for _ in range(rows)]


def mla_setup():
    """All tensors of the tiny layer, drawn in the order the chapter states."""
    g = bookrng.generator(NAME_MLA)
    m = {"X": _draw_matrix(g, T, D)}
    m["Wdkv"] = _draw_matrix(g, DC, D)
    m["Wuk"] = [_draw_matrix(g, DH, DC) for _ in range(H)]
    m["Wuv"] = [_draw_matrix(g, DV, DC) for _ in range(H)]
    m["Wq"] = [_draw_matrix(g, DH, D) for _ in range(H)]
    m["Wqr"] = [_draw_matrix(g, DR, D) for _ in range(H)]
    m["Wkr"] = _draw_matrix(g, DR, D)
    m["Wo"] = _draw_matrix(g, D, H * DV)
    return m


def mla_explicit(m):
    """The layer as the training forward computes it: reconstruct every key and value, then ordinary attention.
    Returns (outputs [T][D], scores of the last token per head [H][T])."""
    X = m["X"]
    C = [_matvec(m["Wdkv"], x) for x in X]
    KR = [pos_mod.rotate(_matvec(m["Wkr"], x), j, WS_R) for j, x in enumerate(X)]
    per_head, last = [], []
    for h in range(H):
        K = [_matvec(m["Wuk"][h], c) for c in C]
        V = [_matvec(m["Wuv"][h], c) for c in C]
        rows = []
        for i in range(T):
            q = _matvec(m["Wq"][h], X[i])
            qr = pos_mod.rotate(_matvec(m["Wqr"][h], X[i]), i, WS_R)
            s = [(_dot(q, K[j]) + _dot(qr, KR[j])) * SCALE for j in range(i + 1)]
            if i == T - 1:
                last.append(s)
            rows.append(_weighted_sum(_softmax(s), V[: i + 1]))
        per_head.append(rows)
    Y = [_matvec(m["Wo"], sum((per_head[h][i] for h in range(H)), [])) for i in range(T)]
    return Y, last


def absorbed_output_matrices(m):
    """W_OV^h = W_O[:, h-th block of d_v columns] W_UV^h, of shape [D, d_c]: the value up-projection folded into the output."""
    out = []
    for h in range(H):
        block = [row[h * DV:(h + 1) * DV] for row in m["Wo"]]
        out.append(_matmul(block, m["Wuv"][h]))
    return out


def mla_absorbed_last(m):
    """Decode of the last token from the cache alone: the latents C [T][d_c] and the rotary keys KR [T][d_R].
    Returns (output row with W_UV folded into the output matrix, output row with W_UV applied after the average,
    scores per head [H][T])."""
    X = m["X"]
    i = T - 1
    C = [_matvec(m["Wdkv"], x) for x in X]
    KR = [pos_mod.rotate(_matvec(m["Wkr"], x), j, WS_R) for j, x in enumerate(X)]
    Wov = absorbed_output_matrices(m)
    y = [0.0] * D
    after = [0.0] * (H * DV)                                 # the other way: W_UV applied to the weighted latent, then W_O
    scores = []
    for h in range(H):
        q = _matvec(m["Wq"][h], X[i])
        qr = pos_mod.rotate(_matvec(m["Wqr"][h], X[i]), i, WS_R)
        qt = _matvec_t(m["Wuk"][h], q)                       # q~ = W_UK^T q, d_c numbers
        s = [(_dot(qt, C[j]) + _dot(qr, KR[j])) * SCALE for j in range(T)]
        scores.append(s)
        ctx = _weighted_sum(_softmax(s), C)                  # the weighted latent, d_c numbers
        part = _matvec(Wov[h], ctx)
        y = [a + b for a, b in zip(y, part)]
        after[h * DV:(h + 1) * DV] = _matvec(m["Wuv"][h], ctx)
    return y, _matvec(m["Wo"], after), scores


def absorption():
    m = mla_setup()
    Y, s_exp = mla_explicit(m)
    y_abs, y_after, s_abs = mla_absorbed_last(m)
    explicit_cache = T * (H * DH + DR + H * DV)
    latent_cache = T * (DC + DR)
    rows = [(j, _fx(s_exp[0][j], 5), _fx(s_abs[0][j], 5)) for j in range(T)]
    d_scores = max(_max_abs(s_exp[h], s_abs[h]) for h in range(H))
    d_out = max(_max_abs(Y[T - 1], y_abs), _max_abs(Y[T - 1], y_after))
    return (_seed_line(NAME_MLA) + f" Shapes: T = {T} tokens, D = {D}, H = {H} heads, content key width d_h = {DH}, value width "
            f"d_v = {DV}, latent d_c = {DC}, rotary key d_R = {DR}. Each entry of every tensor is `round(0.5 * normal(), 1)` "
            "(ties away from zero), drawn row by row in this order: X [8, 7]; W_DKV [5, 7]; W_UK for heads 0, 1, 2 (each [4, 5]); "
            "W_UV for heads 0, 1, 2 (each [6, 5]); W_Q for heads 0, 1, 2 (each [4, 7]); W_QR for heads 0, 1, 2 (each [2, 7]); "
            "W_KR [2, 7]; W_O [7, 18]. The latent of token j is c_j = W_DKV x_j; the rotary parts are rotated by "
            "the `rotary` of Volume V, Chapter 8 with one pair of rate 1.0 radian per position, the pair being the two channels; scores are "
            f"divided by sqrt(d_h + d_R) = sqrt(6); the mask is causal; the head outputs are concatenated in head order. The "
            "explicit run reconstructs every key and value from the latents. The absorbed run starts from the cache alone "
            "(the 8 latents and the 8 rotary keys), forms q~ = W_UK^T q for each head, scores it against the latents, "
            "averages the latents with the weights, and finishes in either of two ways: multiplying the weighted latent of each head by "
            "W_OV = (the head's 7 by 6 block of W_O) times W_UV and adding the head contributions in head order; or applying W_UV to "
            "the weighted latent of each head, concatenating, and multiplying by W_O.\n\n"
            "Scores of head 0 for the last token (position 7) against positions 0 to 7:\n\n"
            + _table(["position j", "explicit: q . k + q_R . k_R, scaled", "absorbed: q~ . c + q_R . k_R, scaled"], rows)
            + "\n\nOutput row of the last token (7 numbers):\n\n"
            + _table(["run", "output row"], [("explicit", _vec(Y[T - 1], 4)), ("absorbed, W_UV folded into W_O", _vec(y_abs, 4)),
                                                      ("absorbed, W_UV applied to the averaged latent", _vec(y_after, 4))], "ll")
            + f"\n\nLargest absolute difference over the {H * T} scores of all three heads: {_below(d_scores)}. Largest absolute "
            f"difference over the {D} output numbers: {_below(d_out)}. Numbers cached after {T} tokens: "
            f"{_int(explicit_cache)} for the explicit layer (per token {H} x {DH} content keys, {DR} rotary key and {H} x {DV} values) "
            f"against {_int(latent_cache)} for the latent one (per token {DC} + {DR}).")


# ------------------------------------------------------------------------------------------------ rotation between W_UK and the query


def rope_breaks_absorption():
    m = mla_setup()
    X = m["X"]
    i = T - 1
    C = [_matvec(m["Wdkv"], x) for x in X]
    W = m["Wuk"][0]
    q = _matvec(m["Wq"][0], X[i])
    qi = pos_mod.rotate(q, i, WS_C)
    q_fixed = _matvec_t(W, qi)                            # absorb the query as rotated by its own position only
    rows, vec_rows, worst_off, worst_fixed = [], [], 0.0, 0.0
    for delta in range(4):
        j = i - delta
        k = _matvec(W, C[j])
        true = _dot(qi, pos_mod.rotate(k, j, WS_C))      # both rotated after the up-projection
        q_off = _matvec_t(W, pos_mod.rotate(q, delta, WS_C))
        by_offset = _dot(q_off, C[j])
        fixed = _dot(q_fixed, C[j])
        worst_off = max(worst_off, abs(true - by_offset))
        worst_fixed = max(worst_fixed, abs(true - fixed))
        rows.append((delta, j, _fx(true, 5), _fx(by_offset, 5), _fx(fixed, 5)))
        vec_rows.append((delta, _vec(q_off, 4)))
    return (_seed_line(NAME_MLA) + f" The same tensors as before (the block above); head 0 only, query at position {i}, the "
            "rotation now applied to the whole content key and query of width 4 (two pairs, rates 1.0 and 0.5 radian per position, "
            "adjacent channels; the `rotary` of Volume V, Chapter 8 with base 4, because 4^(-2m/4) is 1.0 for m = 0 and 0.5 for m = 1) instead of to a separate rotary slice. Scores in this block are raw dot products, without the division by sqrt(d_h + d_R) = sqrt(6) that the scores of the block above carry (the three columns are compared with each other, so a common factor changes none of the comparisons). The true score rotates the query by its position and "
            "the up-projected key W_UK c_j by its own. 'By offset' absorbs the query as q~_delta = W_UK^T R(delta) q, with "
            "delta = i - j; 'fixed' absorbs it once, as W_UK^T R(i) q, and scores that against the unrotated latent.\n\n"
            + _table(["offset delta = i - j", "j", "true score", "by offset: q~_delta . c_j", "fixed: q~ . c_j"], rows)
            + f"\n\nLargest absolute difference from the true score: by offset {_below(worst_off)}; fixed {_fx(worst_fixed, 3)}. "
            "The absorbed query of each offset (5 numbers; no two are equal):\n\n"
            + _table(["offset delta", "q~_delta = W_UK^T R(delta) q"], vec_rows, "rl"))


# ------------------------------------------------------------------------------------------------ FLOPs and bytes per cached key

DECODE_KEYS = 32768


def per_key_costs(s=SHAPE):
    """(label, numbers cached per key per layer, FLOPs per cached key per layer, bytes per cached key per layer)."""
    H, hkv, dh = s["H"], s["hkv"], s["dh"]
    dv, dc, dr = dh, LATENT, ROPE_KEY
    out = []
    for k, name in (("MHA", "multi-head"), ("GQA", "grouped-query (8 key-value heads)"), ("MQA", "multi-query")):
        numbers = numbers_per_layer(k, s)
        out.append((name, numbers, 4 * H * dh, numbers * BYTES))
    cached = dc + dr
    reconstruct = 2 * dc * H * (dh + dv) + 2 * H * (dh + dr) + 2 * H * dv
    out.append(("latent, keys and values rebuilt from the latent at every step", cached, reconstruct, cached * BYTES))
    absorbed = 2 * H * (dc + dr) + 2 * H * dc
    out.append(("latent, absorbed", cached, absorbed, cached * BYTES))
    expanded = H * dh + dr + H * dv
    out.append(("latent weights, expanded keys and values cached", expanded, 2 * H * (dh + dr) + 2 * H * dv, expanded * BYTES))
    return out


def decode_cost_data():
    return per_key_costs()


def decode_cost():
    L, n = SHAPE["L"], DECODE_KEYS
    rows = []
    for label, numbers, flops, nbytes in per_key_costs():
        rows.append((label, _int(numbers), _int(flops), _int(nbytes), f"{flops / nbytes:,.2f}",
                     f"{L * n * flops / 1e9:,.2f}", f"{_gib(L * n * nbytes):.3f}"))
    H, dh, dc, dr = SHAPE["H"], SHAPE["dh"], LATENT, ROPE_KEY
    return (f"The decoder shape of the previous table ($H = {H}$ query heads, $d_h = d_v = {dh}$, $d_c = {dc}$, $d_R = {dr}$, "
            f"{L} layers, bf16). Per cached key and layer: multi-head, grouped and multi-query attention score and average with "
            f"$2 H d_h + 2 H d_h = 4 H d_h$ FLOPs whatever $h_{{kv}}$ is; rebuilding keys and values from the latent adds "
            f"$2 d_c H (d_h + d_v)$ for the up-projections, then $2 H (d_h + d_R)$ for the scores and $2 H d_v$ to average; the "
            f"absorbed form scores against the latent ($2 H (d_c + d_R)$) and averages the latent ($2 H d_c$); the last row "
            f"stores the up-projected keys, the rotary key and the values ($H d_h + d_R + H d_v$ numbers), as brain's "
            f"GLM decoder does. Bytes per key = numbers x {BYTES}. The last two columns are one decoded token on a "
            f"context of {_int(n)} keys across all {L} layers: billions of FLOPs, and GiB of cache read. The matrix products with the "
            "weights, the same for every row up to the per-token terms of the next paragraph, are not counted here.\n\n"
            + _table(["attention", "numbers per key and layer", "FLOPs per key and layer", "bytes per key and layer",
                      "FLOPs per byte", "billions of FLOPs for 32,768 keys", "GiB read"], rows, "lrrrrrr")
            + f"\n\nPer decoded token and layer, outside the sum over keys, the absorbed form also computes $\\tilde{{\\mathbf{{q}}}}^h = "
            f"\\mathbf{{W}}_{{\\mathrm{{UK}}}}^{{h\\top}} \\mathbf{{q}}^h$ for every head ($2 H d_h d_c$ = {_int(2 * H * dh * dc)} FLOPs) "
            f"and then either applies $\\mathbf{{W}}_{{\\mathrm{{UV}}}}^h$ to each weighted latent ($2 H d_v d_c$ = {_int(2 * H * dh * dc)} FLOPs) "
            f"before the ordinary output matrix ($2 H d_v D$ = {_int(2 * H * dh * SHAPE['D'])} FLOPs), or folds it into "
            f"$\\mathbf{{W}}_{{\\mathrm{{OV}}}}^h$ and multiplies the weighted latents by that ($2 H d_c D$ = "
            f"{_int(2 * H * dc * SHAPE['D'])} FLOPs). Either is a fixed cost per token, whatever the number of keys.")


# ------------------------------------------------------------------------------------------------ sliding windows

def window_mask(t, n_win, n_sink=0):
    """mask[i][j] is 'w' (inside the window), 's' (a sink outside the window), 'x' (masked) or '.' (future), 0-indexed."""
    rows = []
    for i in range(t):
        row = []
        for j in range(t):
            if j > i:
                row.append(".")
            elif i - j < n_win:
                row.append("w")
            elif j < n_sink:
                row.append("s")
            else:
                row.append("x")
        rows.append(row)
    return rows


def mask_data(t=12, n_win=4, n_sink=2):
    return window_mask(t, n_win, n_sink)


def receptive_field():
    t, n_win = 16, 4
    mask = window_mask(t, n_win)
    last = t - 1
    reach = {j for j in range(t) if mask[last][j] == "w"}
    rows = []
    for layers in range(1, 6):
        if layers > 1:
            reach = {j for r in reach for j in range(t) if mask[r][j] == "w"} | reach
        lo, hi = min(reach), max(reach)
        rows.append((layers, f"{lo} to {hi}", len(reach), layers * (n_win - 1) + 1))
    return (f"T = {t} positions (0 to {t - 1}), window $n_{{win}} = {n_win}$: position $i$ sees $j$ when $0 \\le i - j < {n_win}$. "
            f"The set of input positions that can change the output at position {last} after $L$ layers is found by applying "
            "the mask to the set $L$ times (the positions reached by layer $l$ are all the positions that a reached position "
            "sees).\n\n"
            + _table(["layers L", "input positions that reach position 15", "count", "L (n_win - 1) + 1"], rows))


def ring_slack():
    n_win, chunk, start = 4, 3, 8                   # window, rows of one prefill chunk, first position of the chunk
    chunk_pos = list(range(start, start + chunk))
    rows = []
    for ring in (4, 5, 6, 7):
        slots = {}                                   # slot -> position held after the earlier history
        for p in range(start):
            slots[p % ring] = p
        for p in chunk_pos:                          # the chunk writes all its rows before any row attends
            slots[p % ring] = p
        lost = []
        for q in chunk_pos:
            need = range(q - n_win + 1, q + 1)
            gone = [p for p in need if slots.get(p % ring) != p]
            lost.append(f"row {q}: " + (", ".join(str(p) for p in gone) if gone else "none"))
        rows.append((ring, "; ".join(lost), "yes" if all("none" in x for x in lost) else "no"))
    return (f"A ring of $R$ rows holds position $p$ in row $p \\bmod R$. Window $n_{{win}} = {n_win}$; positions 0 to "
            f"{start - 1} are already written; a prefill chunk of {chunk} rows (positions {chunk_pos[0]}, {chunk_pos[1]}, "
            f"{chunk_pos[2]}) writes all three before any of them attends, and row $q$ then needs positions $q - 3$ to $q$. "
            "Positions each row needs and no longer finds in the ring:\n\n"
            + _table(["ring rows R", "missing positions", "every row served"], rows, "rll"))


RIPPLE = {"L": 80, "global_every": 5, "kv_dim": 1024, "window": 1024}


def hybrid_cache():
    r = RIPPLE
    n_global = sum(1 for i in range(r["L"]) if i % r["global_every"] == r["global_every"] - 1)
    n_slide = r["L"] - n_global
    per_layer_token = 2 * r["kv_dim"] * BYTES
    global_per_token = n_global * per_layer_token
    ring = n_slide * per_layer_token * r["window"]
    all_global = r["L"] * per_layer_token
    rows = []
    for n in (8192, 131072, 1048576):
        rows.append((_int(n), f"{_gib(all_global * n):.3f}", f"{_gib(global_per_token * n + ring):.3f}",
                     f"{(all_global * n) / (global_per_token * n + ring):.2f}"))
    return (f"The layer pattern of brain's Ripple decoder (`crates/ricursive`): {r['L']} layers, every {r['global_every']}th one "
            f"global (layers 4, 9, ..., 79), the rest sliding with a window of {_int(r['window'])} keys; each layer has "
            f"$h_{{kv}} d_h = {_int(r['kv_dim'])}$ key numbers and as many value numbers per token, in bf16. Global layers: "
            f"{n_global}, so {_int(global_per_token)} bytes per token. Sliding layers: {n_slide}, each holding at most "
            f"{_int(r['window'])} tokens, so a fixed {_int(ring)} bytes ({ring / 2 ** 20:.0f} MiB) per sequence once it is "
            f"longer than the window (the slack rows of a prefill chunk are not counted). All {r['L']} layers global would "
            f"cost {_int(all_global)} bytes per token.\n\n"
            + _table(["tokens in the sequence", "all layers global, GiB", "4 sliding to 1 global, GiB", "ratio"], rows))


# ------------------------------------------------------------------------------------------------ a softmax row has to sum to one


def mass_needs_a_place():
    g = bookrng.generator("attention_variants.mass")
    n = 4
    values = [[md._round1(g.normal()) for _ in range(3)] for _ in range(n)]
    base = [-6.0, -6.2, -5.9, -6.1]
    rows = []
    for shift in (0.0, -10.0, -100.0):
        p = _softmax([x + shift for x in base])
        out = _weighted_sum(p, values)
        rows.append((f"{shift:+.0f}", _vec(p, 6), f"{sum(p):.6f}", _fx(math.sqrt(_dot(out, out)), 4)))
    mean = _weighted_sum([1.0 / n] * n, values)
    # one key added that every query scores 5 higher, with a value of zero
    with_sink = _softmax([x for x in base] + [-1.0])
    out_sink = _weighted_sum(with_sink, values + [[0.0, 0.0, 0.0]])
    return (_seed_line("attention_variants.mass") + " Draws: the 4 value vectors of 3 numbers each, `round(normal(), 1)`, "
            f"in order: {', '.join(_vec(v, 1) for v in values)}. Scores of one query against four keys, all low: "
            f"(-6.0, -6.2, -5.9, -6.1), shifted by a constant before the softmax.\n\n"
            + _table(["shift added to every score", "attention weights", "sum of weights", "length of the output"], rows, "rlrr")
            + f"\n\nThe unweighted mean of the four values has length {_fx(math.sqrt(_dot(mean, mean)), 4)}. A fifth key "
            "with score -1.0 and a value of zeros, five above the others, takes the weights "
            f"{_vec(with_sink, 6)}, and the output has length {_fx(math.sqrt(_dot(out_sink, out_sink)), 4)}.")


STREAM_T, STREAM_WIN = 12, 4
SINK_SCORE = 5.0


def _stream_setup():
    g = bookrng.generator("attention_variants.sink")
    a = [None] + [md._round1(g.normal()) for _ in range(1, STREAM_T)]          # query component of tokens 1 .. 9
    b = [None] + [md._round1(g.normal()) for _ in range(1, STREAM_T)]          # key component of tokens 1 .. 9
    v = [[0.0, 0.0, 0.0]] + [[md._round1(g.normal()) for _ in range(3)] for _ in range(1, STREAM_T)]
    return a, b, v


def _stream_scores(a, b, t, keys):
    """Score of query t against key j: q_t = (a_t, 1), k_j = (b_j, -0.5), and k_0 = (0, 4) for the first token."""
    out = []
    for j in keys:
        out.append(SINK_SCORE if j == 0 else a[t] * b[j] - 0.5)
    return out


def _norm(v):
    return math.sqrt(_dot(v, v))


def sink_stream():
    a, b, v = _stream_setup()
    last = STREAM_T - 1
    rows, worst_sink, best_win = [], 0.0, None
    for t in range(1, STREAM_T):
        full_keys = list(range(t + 1))
        p_full = _softmax(_stream_scores(a, b, t, full_keys))
        ref = _weighted_sum(p_full, [v[j] for j in full_keys])
        win_keys = list(range(max(0, t - STREAM_WIN + 1), t + 1))
        out_win = _weighted_sum(_softmax(_stream_scores(a, b, t, win_keys)), [v[j] for j in win_keys])
        sink_keys = sorted(set(win_keys) | {0})
        out_sink = _weighted_sum(_softmax(_stream_scores(a, b, t, sink_keys)), [v[j] for j in sink_keys])
        err_win = _norm([x - y for x, y in zip(out_win, ref)])
        err_sink = _norm([x - y for x, y in zip(out_sink, ref)])
        dropped = sum(p_full[j] for j in full_keys if j not in sink_keys)
        rows.append((t, f"{p_full[0]:.4f}", f"{_norm(ref):.4f}", f"{err_win:.4f}", f"{err_sink:.4f}", f"{dropped:.5f}"))
        if t >= STREAM_WIN:
            worst_sink = max(worst_sink, err_sink)
            best_win = err_win if best_win is None else min(best_win, err_win)
    mean_len = sum(_norm(x) for x in v[1:]) / (STREAM_T - 1)
    return (_seed_line("attention_variants.sink") + f" A stream of {STREAM_T} tokens (positions 0 to {last}); query $t$ is "
            "the pair $(a_t, 1)$ and key $j$ the pair $(b_j, -0.5)$, scored by their dot product with no scaling, except key 0, "
            f"which is $(0, {SINK_SCORE})$ so that every query scores it {SINK_SCORE}. The value of token 0 is zeros; tokens 1 to "
            f"{last} have three numbers each. Draws, in this order, each `round(normal(), 1)`: $a_1, \\dots, a_{{{last}}}$; "
            f"$b_1, \\dots, b_{{{last}}}$; then the values of tokens 1 to {last}, three numbers each, token by token. "
            "Reference: query $t$ attends to every position 0 to $t$. 'Window' keeps only the last "
            f"{STREAM_WIN} positions (itself included), 'sink and window' keeps those and position 0. Errors are the Euclidean "
            "distance of the output from the reference output; 'weight dropped' is the reference attention weight of the positions "
            "that 'sink and window' leaves out. For scale, the mean length of the values of tokens 1 to "
            f"{last} is {_fx(mean_len, 3)}.\n\n"
            + _table(["query t", "reference weight on position 0", "reference output length", "error, window",
                      "error, sink and window", "weight dropped"], rows)
            + f"\n\nFrom query {STREAM_WIN}, the first at which position 0 is outside the window, the error of the window alone "
            f"is at least {_fx(best_win, 4)} and that of sink and window at most {_fx(worst_sink, 4)}.")


# ------------------------------------------------------------------------------------------------ the caches of brain's decoders

GLM = {"name": "GLM-5.2 shape", "L": 78, "H": 64, "nope": 192, "rope": 64, "v": 256, "kv_lora": 512, "D": 6144}
OCR = {"name": "DeepSeek-OCR decoder", "L": 12, "H": 10, "dh": 128, "D": 1280}
POOL = 131072


def brain_caches():
    g, o = GLM, OCR
    ocr_numbers = 2 * o["H"] * o["dh"]
    glm_expanded = g["H"] * g["nope"] + g["rope"] + g["H"] * g["v"]
    glm_latent = g["kv_lora"] + g["rope"]
    rows = [
        ("DeepSeek-OCR decoder, `deepseek2`", "multi-head, 10 heads of 128", _int(ocr_numbers), "bf16", _int(o["L"] * ocr_numbers * 2),
         f"{_gib(o['L'] * ocr_numbers * 2 * POOL):,.3f}"),
        ("DeepSeek-OCR decoder, `deepseek2`", "multi-head, 10 heads of 128", _int(ocr_numbers), "f32", _int(o["L"] * ocr_numbers * 4),
         f"{_gib(o['L'] * ocr_numbers * 4 * POOL):.3f}"),
        ("GLM-5.2 shape, `glmdsa`, as it caches", "expanded: 64 x 192 keys, 64 rotary, 64 x 256 values", _int(glm_expanded), "f32",
         _int(g["L"] * glm_expanded * 4), f"{_gib(g['L'] * glm_expanded * 4 * POOL):.3f}"),
        ("GLM-5.2 shape, latent and rotary key", "512 latent and 64 rotary", _int(glm_latent), "f32",
         _int(g["L"] * glm_latent * 4), f"{_gib(g['L'] * glm_latent * 4 * POOL):.3f}"),
        ("GLM-5.2 shape, latent and rotary key", "512 latent and 64 rotary", _int(glm_latent), "bf16",
         _int(g["L"] * glm_latent * 2), f"{_gib(g['L'] * glm_latent * 2 * POOL):.3f}"),
    ]
    return (f"Widths read from the presets of the crates (`DeepseekV2Config::deepseek_ocr`: {o['L']} layers, "
            f"{o['H']} heads and {o['H']} key-value heads of width {o['dh']}; `GlmConfig::glm5_2`: {g['L']} layers, {g['H']} heads, "
            f"qk_nope_head_dim {g['nope']}, qk_rope_head_dim {g['rope']}, v_head_dim {g['v']}, kv_lora_rank {g['kv_lora']}). "
            "Numbers per token and layer: 2 x heads x head width for multi-head attention; $H d_{nope} + d_{rope} + H d_v$ for the "
            "expanded cache of the GLM decoder, which stores the up-projected keys, the one shared rotary key and the values; "
            "$d_c + d_R$ for a latent cache. The last column is a pool of 131,072 tokens. At equal precision the expanded cache of "
            f"the GLM shape is {glm_expanded / glm_latent:.2f} times the latent one.\n\n"
            + _table(["decoder", "what a layer stores", "numbers per token and layer", "precision", "bytes per token", "GiB at 131,072 tokens"],
                     rows, "llrlrr"))
