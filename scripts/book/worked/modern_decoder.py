# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Anatomy of a Modern Decoder": what changed between the first GPT and today's open-weight models.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds between
`<!-- worked: modern_decoder.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture
and the table beside it cannot disagree.

* the recipe of GPT-2 small beside that of Qwen3, with every parameter counted, and the ladder of changes from one to the other;
* grouped-query attention against multi-head attention with repeated key-value heads, at four query heads and two key-value heads;
* the bytes of the key-value cache as the number of key-value heads falls;
* attention logits with and without a per-head normalization of queries and keys, on weights that grow;
* the order of the normalization and the rotary rotation, and what it does to the dependence on the offset;
* a tiny decoder that has every choice of the recipe: forward pass, decode on a cache equal to the forward pass, and a gradient
  check of its hand-written backward pass against central differences, with wrong backward passes that the check must catch.

Everything the chapter reuses from Volume V is imported, never copied: RMS normalization and SwiGLU with their backward passes
(`transformer_block`), multi-head attention (`self_attention`) and the rotary rotation (`positions`). Every random quantity comes
from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products are plain left-to-right
additions, the order a C loop would use (never Python's `sum`, which may compensate).

Swedish Embedded AB implements solutions for deploying open-weight language models, from the architecture to the checkpoint. If
your team needs expertise in decoder architectures and their cost, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import positions as pos_mod
import self_attention as sa
import transformer_block as tb

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


def _round1(x):
    """x rounded to one decimal, ties away from zero (the pseudocode's `round(x, 1)`)."""
    r = math.floor(abs(x) * 10 + 0.5) / 10
    return -r if x < 0 else r


def _below(x, limit=1e-12):
    """'below 1e-12' when x is round-off noise under the limit, else the value to three decimals."""
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else _fx(x, 3)


def _gib(b):
    return b / 2 ** 30


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _vec(v, d=2):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


BYTES = 2   # bf16: bytes per stored cache value

# ------------------------------------------------------------------------------------------------ the shapes of real decoders


def shape(name, V, D, L, H, hkv, dh, F, mlp, bias, shifted_norm, learned_positions, tied, qk_norm, context):
    """The choices of one decoder. `mlp` is the number of feed-forward matrices (2 for a GELU network, 3 for SwiGLU), `bias`
    whether every linear map has a bias, `shifted_norm` whether the normalizations have a shift as well as a gain (LayerNorm),
    `learned_positions` the length of a table of learned position rows (0 for rotary embeddings)."""
    return {"name": name, "V": V, "D": D, "L": L, "H": H, "hkv": hkv, "dh": dh, "F": F, "mlp": mlp, "bias": bias,
            "shifted": shifted_norm, "pos": learned_positions, "tied": tied, "qkn": qk_norm, "ctx": context}


# GPT-2 small as published (Radford and colleagues, 2019): 12 heads of width 64, a GELU network of 4 D, biases everywhere,
# LayerNorm, 1,024 learned positions, a head tied to the embedding table.
GPT2 = shape("GPT-2 small", 50257, 768, 12, 12, 12, 64, 3072, 2, True, True, 1024, True, False, 1024)
# The Qwen3 sizes as brain's presets `QwenConfig::qwen3_0_6b` .. `qwen3_8b` give them (and the published context of the 0.6B).
QWEN3_0_6B = shape("Qwen3 0.6B", 151936, 1024, 28, 16, 8, 128, 3072, 3, False, False, 0, True, True, 40960)
QWEN3_1_7B = shape("Qwen3 1.7B", 151936, 2048, 28, 16, 8, 128, 6144, 3, False, False, 0, True, True, 40960)
QWEN3_4B = shape("Qwen3 4B", 151936, 2560, 36, 32, 8, 128, 9728, 3, False, False, 0, True, True, 40960)
QWEN3_8B = shape("Qwen3 8B", 151936, 4096, 36, 32, 8, 128, 12288, 3, False, False, 0, False, True, 40960)
# Llama-2 13B, brain's preset `QwenConfig::llama2_13b`: plain multi-head attention, no QK-norm, untied head.
LLAMA2_13B = shape("Llama-2 13B", 32000, 5120, 40, 40, 40, 128, 13824, 3, False, False, 0, False, False, 4096)


def parameter_groups(s):
    """Parameters of a decoder by group: (name, count) pairs. Layer-wise groups are summed over the L layers; a norm counts D
    numbers (gain) or 2 D (gain and shift); the head counts only when it is not tied."""
    D, L, H, hkv, dh, F = s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]
    norm = (2 if s["shifted"] else 1) * D
    q_dim, kv_dim = H * dh, hkv * dh
    attn = D * q_dim + 2 * D * kv_dim + q_dim * D
    attn_bias = (q_dim + 2 * kv_dim + D) if s["bias"] else 0
    mlp = s["mlp"] * D * F
    mlp_bias = ((s["mlp"] - 1) * F + D) if s["bias"] else 0
    groups = [("token embedding table", s["V"] * D),
              ("learned position table", s["pos"] * D),
              ("attention matrices W_Q, W_K, W_V, W_O", L * attn),
              ("attention biases", L * attn_bias),
              ("QK-norm gains", L * 2 * dh if s["qkn"] else 0),
              ("feed-forward matrices", L * mlp),
              ("feed-forward biases", L * mlp_bias),
              ("normalizations (two per layer and the last)", (2 * L + 1) * norm),
              ("separate output matrix", 0 if s["tied"] else s["V"] * D)]
    return groups


def parameters(s):
    return sum(c for _, c in parameter_groups(s))


def cache_bytes_per_token(s, b=BYTES):
    """2 (keys and values) x layers x key-value heads x head width x bytes per value (Volume V, Chapter 12)."""
    return 2 * s["L"] * s["hkv"] * s["dh"] * b


def decode_flops(s, c):
    """FLOPs of the matrix products of one decoded token on a cache of c tokens: 2 per weight of the blocks, 4 H dh per visible
    key and layer (c + 1 keys including the token itself), and the head 2 D V (Volume V, Chapter 12)."""
    D, L, H, hkv, dh, F = s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]
    blocks = L * (D * H * dh + 2 * D * hkv * dh + H * dh * D + s["mlp"] * D * F)
    return 2 * blocks + 4 * L * H * dh * (c + 1) + 2 * D * s["V"]


def side_by_side():
    a, b, c = GPT2, QWEN3_0_6B, QWEN3_8B

    def tie(s):
        return "tied to the embedding" if s["tied"] else "separate matrix"

    def att(s):
        g = s["H"] // s["hkv"]
        return (f"{s['H']} heads, one key-value head each" if g == 1 else
                f"{s['H']} query heads, {s['hkv']} key-value heads (groups of {g})")

    def pos(s):
        return f"learned table, {_int(s['pos'])} rows" if s["pos"] else "rotary, base 1,000,000, no table"

    def norm(s):
        return "LayerNorm (gain and shift)" if s["shifted"] else "RMS norm (gain only)"

    def mlp(s):
        return (f"GELU network, {s['D']} to {_int(s['F'])} to {s['D']}" if s["mlp"] == 2 else
                f"SwiGLU, {_int(s['D'])} to {_int(s['F'])} to {_int(s['D'])}, three matrices")

    rows = [("positions", pos(a), pos(b), pos(c)),
            ("normalization", norm(a), norm(b), norm(c)),
            ("feed-forward network", mlp(a), mlp(b), mlp(c)),
            ("biases on linear maps", "all of them", "none", "none"),
            ("attention", att(a), att(b), att(c)),
            ("head width", str(a["dh"]), str(b["dh"]) + f" (H x head width = {_int(b['H'] * b['dh'])}, width D = {_int(b['D'])})",
             str(c["dh"]) + f" (H x head width = {_int(c['H'] * c['dh'])}, width D = {_int(c['D'])})"),
            ("QK-norm", "none", "per head, before rotation", "per head, before rotation"),
            ("output head", tie(a), tie(b), tie(c)),
            ("vocabulary", _int(a["V"]), _int(b["V"]), _int(c["V"])),
            ("context", f"{_int(a['ctx'])} (trained)", f"{_int(b['ctx'])} (published)", ""),
            ("layers", str(a["L"]), str(b["L"]), str(c["L"])),
            ("parameters", _int(parameters(a)), _int(parameters(b)), _int(parameters(c))),
            ("cache bytes per token (bf16)", _int(cache_bytes_per_token(a)), _int(cache_bytes_per_token(b)), _int(cache_bytes_per_token(c)))]
    names = [x["name"] for x in (a, b, c)]
    return ("The choices of three decoders. GPT-2 small is the published model of Volume V, Chapter 10. Qwen3 0.6B and 8B are the "
            "shapes of brain's presets `QwenConfig::qwen3_0_6b` and `qwen3_8b` (`crates/qwen3/src/config.rs`), with the published "
            "context of the 0.6B checkpoint (`max_position_embeddings`) in the context row (the 8B's is not pinned by brain's sources and is "
            "left blank). Parameters are counted by "
            "`parameter_groups` from the shapes (every tensor listed in the next table); the cache row is "
            "2 x layers x key-value heads x head width x 2 bytes.\n\n"
            + _table(["choice"] + names, rows, "llll"))


def parameter_table():
    cols = [GPT2, QWEN3_0_6B, QWEN3_8B]
    names = [g for g, _ in parameter_groups(GPT2)]
    rows = []
    for i, n in enumerate(names):
        rows.append((n, *[_int(parameter_groups(s)[i][1]) for s in cols]))
    rows.append(("**total**", *[_int(parameters(s)) for s in cols]))
    emb = [parameter_groups(s)[0][1] for s in cols]
    rows.append(("embedding table as a share of the total", *[f"{100 * e / parameters(s):.1f}%" for e, s in zip(emb, cols)]))
    return ("Every tensor of the three decoders, counted from the shapes of the previous table (zero where the decoder has none). "
            "A layer's attention holds W_Q (D by H x head width), W_K and W_V (D by key-value heads x head width) and W_O "
            "(H x head width by D); a layer's network holds 2 (GELU) or 3 (SwiGLU) matrices of D by F; the QK-norm of Qwen3 has "
            "a gain of head-width numbers for queries and again for keys in every layer.\n\n"
            + _table(["group"] + [s["name"] for s in cols], rows))


# ------------------------------------------------------------------------------------------------ the ladder of changes

LADDER_CONTEXT = 1023   # tokens already cached when the decoded token is counted: it sees 1,024 keys


def _step(s, **kw):
    t = dict(s)
    t.update(kw)
    return t


def ladder_steps():
    """GPT-2 small with the changes applied one at a time, at GPT-2's own width and depth (not Qwen3's)."""
    s0 = GPT2
    s1 = _step(s0, pos=0)                                   # rotary embeddings instead of a learned table
    s2 = _step(s1, shifted=False)                           # RMS norm instead of LayerNorm
    s3 = _step(s2, mlp=3, F=2048)                           # SwiGLU, F chosen so the matrices hold as many numbers as before
    s4 = _step(s3, bias=False)                              # no biases
    s5 = _step(s4, hkv=4)                                   # grouped-query attention, groups of three
    s6 = _step(s5, qkn=True)                                # QK-norm
    s7 = _step(s6, tied=False)                              # a separate output matrix
    return [("GPT-2 small", s0),
            ("rotary embeddings, no position table", s1),
            ("RMS norm instead of LayerNorm", s2),
            ("SwiGLU, hidden width 2,048", s3),
            ("no biases", s4),
            ("4 key-value heads for 12 query heads", s5),
            ("QK-norm", s6),
            ("separate output matrix", s7)]


def ladder():
    steps = ladder_steps()
    rows, prev = [], None
    for label, s in steps:
        p, cb, fl = parameters(s), cache_bytes_per_token(s), decode_flops(s, LADDER_CONTEXT)
        if prev is None:
            delta = ["", "", ""]
        else:
            delta = [f"{p - prev[0]:+,}", f"{cb - prev[1]:+,}", f"{fl - prev[2]:+,}"]
        rows.append((label, _int(p), delta[0], _int(cb), delta[1], _int(fl), delta[2]))
        prev = (p, cb, fl)
    return ("Changes applied one after another to GPT-2 small at its own width (768), depth (12) and vocabulary. The first row is "
            "GPT-2; each later row keeps all earlier changes. SwiGLU takes F = 2,048 so that its three matrices "
            "(3 x 768 x 2,048) hold exactly as many numbers as the two matrices of the GELU network (2 x 768 x 3,072). "
            "Cache bytes are per token in bf16. FLOPs are those of the matrix products of one decoded token that sees 1,024 "
            "keys (cache of 1,023): 2 FLOPs for every matrix weight of the blocks (the biases, the norms and the rotation add "
            "FLOPs that are not counted), 4 x layers x H x head width for every visible key, and 2 x D x V for the head. "
            "Until the row \"no biases\" the SwiGLU network keeps biases: one on each of gate and up (F = 2,048 numbers each) "
            "and one on down (D = 768 numbers), in every layer.\n\n"
            + _table(["step", "parameters", "change", "cache bytes", "change", "FLOPs per decoded token", "change"], rows))


# ------------------------------------------------------------------------------------------------ the Qwen3 family and GQA's bill


def family():
    shapes = [QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, LLAMA2_13B]
    rows = []
    for s in shapes:
        mha = _step(s, hkv=s["H"])
        total = parameters(s)
        emb = s["V"] * s["D"]
        rows.append((s["name"], s["L"], _int(s["D"]), f"{s['H']} / {s['hkv']}", s["dh"], _int(s["F"]),
                     "tied" if s["tied"] else "untied", _int(total), f"{100 * emb / total:.1f}%",
                     _int(cache_bytes_per_token(s)), _int(cache_bytes_per_token(mha))))
    return ("Five decoders from brain's presets (`crates/qwen3/src/config.rs`). Parameters are counted by `parameter_groups`; the "
            "embedding share is V x D over the total, the table that an untied head repeats. Cache bytes per token are in bf16 "
            "for the key-value heads the model has and for the same model with one key-value head per query head.\n\n"
            + _table(["decoder", "layers", "D", "H / h_kv", "head width", "F", "head", "parameters", "embedding share",
                      "cache bytes per token", "same with h_kv = H"], rows))


# ------------------------------------------------------------------------------------------------ the cache against the key-value heads

CACHE_CONTEXT = 32768


def cache_vs_kv_heads_data(s):
    """(h_kv, cache bytes per token, GiB for CACHE_CONTEXT tokens) for every divisor h_kv of the query heads of s."""
    out = []
    for hkv in range(1, s["H"] + 1):
        if s["H"] % hkv == 0:
            t = _step(s, hkv=hkv)
            b = cache_bytes_per_token(t)
            out.append((hkv, b, _gib(b * CACHE_CONTEXT)))
    return out


def cache_vs_kv_heads():
    a, b = QWEN3_0_6B, QWEN3_8B
    da, db = cache_vs_kv_heads_data(a), cache_vs_kv_heads_data(b)
    lookup_a = {h: (bt, g) for h, bt, g in da}
    lookup_b = {h: (bt, g) for h, bt, g in db}
    rows = []
    for h in sorted(set(lookup_a) | set(lookup_b)):
        ra = lookup_a.get(h)
        rb = lookup_b.get(h)
        rows.append((h, _int(ra[0]) if ra else "", f"{ra[1]:.2f}" if ra else "", _int(rb[0]) if rb else "", f"{rb[1]:.2f}" if rb else ""))
    return (f"Cache bytes per token (bf16) and GiB for a context of {_int(CACHE_CONTEXT)} tokens as the key-value heads of one "
            "layer fall from one per query head to one in all, for the layer counts and head widths of Qwen3 0.6B "
            "(16 query heads, 28 layers, width 128) and Qwen3 8B (32 query heads, 36 layers, width 128). The row of the "
            "published choice, 8 key-value heads, is in both.\n\n"
            + _table(["h_kv", "0.6B bytes per token", "0.6B GiB", "8B bytes per token", "8B GiB"], rows))


# ------------------------------------------------------------------------------------------------ GQA equals MHA with repeated heads

NAME_GQA = "modern_decoder.gqa"
GQA_T, GQA_D, GQA_H, GQA_HKV, GQA_DH = 3, 8, 4, 2, 2


def _normal_matrix(g, rows, cols):
    return [[_round1(g.normal()) for _ in range(cols)] for _ in range(rows)]


def _columns(M, a, b):
    return [row[a:b] for row in M]


def _repeat_heads(W, hkv, dh, mapping):
    """The matrix with, for query head i = 0 .. len(mapping) - 1, the columns of key-value head mapping[i] of W (width dh each)."""
    blocks = [_columns(W, mapping[i] * dh, (mapping[i] + 1) * dh) for i in range(len(mapping))]
    return [sum(rows, []) for rows in zip(*blocks)]


def gqa_forward(X, Wq, Wk, Wv, Wo, H, hkv, dh):
    """Grouped-query attention written directly: query head i reads key-value head i // (H / hkv), causal. Returns (Y, weights)
    with weights[i] the attention matrix of query head i."""
    T, group = len(X), H // hkv
    scale = 1.0 / math.sqrt(dh)
    Q, K, V = sa.matmul(X, Wq), sa.matmul(X, Wk), sa.matmul(X, Wv)
    outs, weights = [], []
    for h in range(H):
        kv = h // group
        q = _columns(Q, h * dh, (h + 1) * dh)
        k = _columns(K, kv * dh, (kv + 1) * dh)
        v = _columns(V, kv * dh, (kv + 1) * dh)
        P = []
        for i in range(T):
            s = [_dot(q[i], k[j]) * scale for j in range(i + 1)]
            m = max(s)
            e = [math.exp(x - m) for x in s]
            z = 0.0
            for x in e:
                z += x
            P.append([x / z for x in e] + [0.0] * (T - i - 1))
        weights.append(P)
        outs.append(sa.matmul(P, v))
    O = [sum((outs[h][i] for h in range(H)), []) for i in range(T)]
    return sa.matmul(O, Wo), weights


def gqa_example_data():
    g = bookrng.generator(NAME_GQA)
    X = _normal_matrix(g, GQA_T, GQA_D)
    Wq = _normal_matrix(g, GQA_D, GQA_H * GQA_DH)
    Wk = _normal_matrix(g, GQA_D, GQA_HKV * GQA_DH)
    Wv = _normal_matrix(g, GQA_D, GQA_HKV * GQA_DH)
    Wo = _normal_matrix(g, GQA_H * GQA_DH, GQA_D)
    return X, Wq, Wk, Wv, Wo


def _max_diff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def gqa_equals_mha():
    X, Wq, Wk, Wv, Wo = gqa_example_data()
    H, hkv, dh = GQA_H, GQA_HKV, GQA_DH
    group = H // hkv
    Y, weights = gqa_forward(X, Wq, Wk, Wv, Wo, H, hkv, dh)
    mapping = [h // group for h in range(H)]
    Wk_rep, Wv_rep = _repeat_heads(Wk, hkv, dh, mapping), _repeat_heads(Wv, hkv, dh, mapping)
    Ym, cache = sa.mha_forward(X, Wq, Wk_rep, Wv_rep, Wo, H, 0)
    wrong = [h % hkv for h in range(H)]   # the layout that cycles through the key-value heads instead of grouping neighbours
    Wk_w, Wv_w = _repeat_heads(Wk, hkv, dh, wrong), _repeat_heads(Wv, hkv, dh, wrong)
    Yw, _ = sa.mha_forward(X, Wq, Wk_w, Wv_w, Wo, H, 0)
    Y1, _ = gqa_forward(X, Wq, _columns(Wk, 0, dh), _columns(Wv, 0, dh), Wo, H, 1, dh)
    Wk1, Wv1 = _repeat_heads(Wk, hkv, dh, [0] * H), _repeat_heads(Wv, hkv, dh, [0] * H)
    Y1m, _ = sa.mha_forward(X, Wq, Wk1, Wv1, Wo, H, 0)
    head_rows = []
    for h in range(H):
        w = weights[h][GQA_T - 1]
        wm = cache["heads"][h]["P"][GQA_T - 1]
        head_rows.append((h, h // group, _vec(w, 3), _vec(wm, 3)))
    out_rows = []
    for i in range(GQA_T):
        out_rows.append((i, _vec(Y[i], 2), _vec(Ym[i], 2)))
    sizes = (("multi-head, 4 key-value heads", H), ("grouped-query, 2 key-value heads", hkv), ("multi-query, 1 key-value head", 1))
    size_rows = [(n, k, 2 * k * dh) for n, k in sizes]
    return (_seed_line(NAME_GQA) + f" Shapes: T = {GQA_T} tokens, D = {GQA_D}, H = {H} query heads of width {dh}, {hkv} key-value "
            "heads. Draws, in this order, each as `round(normal(), 1)` and row by row: X [3, 8], W_Q [8, 8], W_K [8, 4], W_V [8, 4], "
            "W_O [8, 8]. The mask is causal; scores are divided by sqrt(2).\n\n"
            "Which key-value head each query head reads, and the attention weights of the last token over positions 0, 1, 2 "
            "(grouped-query attention written directly; multi-head attention of Volume V, Chapter 7 run on W_K and W_V with "
            "each key-value head's columns repeated for the query heads of its group):\n\n"
            + _table(["query head", "key-value head", "weights, grouped-query", "weights, multi-head with repeated heads"],
                     head_rows, "rlll")
            + "\n\nOutput rows (one per token, 8 numbers; first two shown):\n\n"
            + _table(["token", "grouped-query, columns 0-1", "multi-head with repeated heads, columns 0-1"],
                     [(i, _vec(Y[i][:2], 3), _vec(Ym[i][:2], 3)) for i in range(GQA_T)], "rll")
            + f"\n\nLargest absolute difference over all {GQA_T * GQA_D} output numbers and all weights: "
            f"{_below(max(_max_diff(Y, Ym), max(_max_diff(weights[h], cache['heads'][h]['P']) for h in range(H))))}. With the layout that cycles through the key-value heads instead (query head i reads head i mod 2, "
            f"neighbours do not share) the multi-head run differs from grouped-query attention by {_fx(_max_diff(Y, Yw), 3)} at most. "
            "Multi-query attention, one key-value head (head 0 of W_K and W_V), equals multi-head attention with that head "
            f"repeated four times (difference {_below(_max_diff(Y1, Y1m))}) and differs from the grouped-query output by "
            f"{_fx(_max_diff(Y, Y1), 3)} at most.\n\n"
            "Numbers cached for one token in one layer (a key and a value row per key-value head):\n\n"
            + _table(["attention", "key-value heads", "numbers cached per token and layer"], size_rows, "lrr"))


# ------------------------------------------------------------------------------------------------ logits that grow with the weights

NAME_LOGITS = "modern_decoder.logits"
LOGIT_T, LOGIT_D, LOGIT_DH = 6, 8, 8
LOGIT_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)


def logit_inputs():
    """X [6, 8], W0_Q [8, 8], W0_K [8, 8]: X entries `round(normal(), 1)`; weights `normal() / sqrt(8)` (unrounded)."""
    g = bookrng.generator(NAME_LOGITS)
    X = [[_round1(g.normal()) for _ in range(LOGIT_D)] for _ in range(LOGIT_T)]
    W0q = [[g.normal() / math.sqrt(LOGIT_D) for _ in range(LOGIT_DH)] for _ in range(LOGIT_D)]
    W0k = [[g.normal() / math.sqrt(LOGIT_D) for _ in range(LOGIT_DH)] for _ in range(LOGIT_D)]
    return X, W0q, W0k


def _logits(Q, K, dh):
    """Causal scaled scores s_ij = q_i . k_j / sqrt(dh), j <= i, as a list of rows."""
    return [[_dot(Q[i], K[j]) / math.sqrt(dh) for j in range(i + 1)] for i in range(len(Q))]


def _softmax_row(s):
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = 0.0
    for v in e:
        z += v
    return [v / z for v in e]


def logit_growth_data(scales=LOGIT_SCALES):
    """Per factor: (factor, largest |logit| without normalization, with it, largest weight of the last row without, with)."""
    X, W0q, W0k = logit_inputs()
    ones = [1.0] * LOGIT_DH
    out = []
    for a in scales:
        Q = sa.matmul(X, [[a * w for w in r] for r in W0q])
        K = sa.matmul(X, [[a * w for w in r] for r in W0k])
        plain = _logits(Q, K, LOGIT_DH)
        Qn, _ = tb.rms_forward(Q, ones)
        Kn, _ = tb.rms_forward(K, ones)
        normed = _logits(Qn, Kn, LOGIT_DH)
        out.append((a, max(abs(v) for r in plain for v in r), max(abs(v) for r in normed for v in r),
                    max(_softmax_row(plain[-1])), max(_softmax_row(normed[-1]))))
    return out


def logit_growth():
    data = logit_growth_data()
    bound = math.sqrt(LOGIT_DH)
    rows = [(f"{a:g}", _fx(p, 3), _fx(n, 4), _fx(wp, 4), _fx(wn, 4)) for a, p, n, wp, wn in data]
    return (_seed_line(NAME_LOGITS) + f" Six tokens (T = {LOGIT_T}) of width D = {LOGIT_D}, one head of width {LOGIT_DH}. Draws, in this "
            "order: X [6, 8] as `round(normal(), 1)` row by row, then W_Q and W_K [8, 8] as `normal() / sqrt(8)` (unrounded), row by row. "
            "The weights of a row are the factor in its first column times these, for queries and keys alike; the mask is causal and the "
            "logit of a pair is the query-key dot product divided by sqrt(8). The normalized columns apply the RMS norm of Volume V, "
            "Chapter 9 (epsilon 1e-6, gain 1) to every query row and every key row before the dot product. Largest absolute logit over "
            "the 21 allowed pairs, and the largest attention weight in the last row (the weight of the sharpest position out of 6; a "
            f"uniform row has {_fx(1 / LOGIT_T, 4)}):\n\n"
            + _table(["factor", "largest absolute logit, no norm", "largest absolute logit, QK-norm", "largest weight, no norm", "largest weight, QK-norm"], rows)
            + f"\n\nThe bound for unit gains is sqrt(8) = {_fx(bound, 4)}: a normalized row has Euclidean length sqrt(8 x mean square / "
            "(mean square + epsilon)), below sqrt(8), so a dot product of two of them is below 8 and the scaled logit below sqrt(8).")


# ------------------------------------------------------------------------------------------------ the order of the norm and the rotation

NAME_ORDER = "modern_decoder.order"
ORDER_D, ORDER_BASE, ORDER_OFFSET = 8, 100.0, 3
ORDER_PAIRS = ((2, 5), (10, 13), (40, 43))


def order_inputs():
    """q, k [8]: `round(normal(), 1)`; gains for queries and keys [8]: `1 + 0.5 * normal()`, unrounded. Draw order: q, k, gq, gk."""
    g = bookrng.generator(NAME_ORDER)
    q = [_round1(g.normal()) for _ in range(ORDER_D)]
    k = [_round1(g.normal()) for _ in range(ORDER_D)]
    gq = [1.0 + 0.5 * g.normal() for _ in range(ORDER_D)]
    gk = [1.0 + 0.5 * g.normal() for _ in range(ORDER_D)]
    return q, k, gq, gk


def _rms_gain(x, gain):
    return tb.rms_forward([x], gain)[0][0]


def _score_norm_then_rotate(q, k, i, j, gq, gk, ws):
    return _dot(pos_mod.rotate(_rms_gain(q, gq), i, ws, "half"), pos_mod.rotate(_rms_gain(k, gk), j, ws, "half"))


def _score_rotate_then_norm(q, k, i, j, gq, gk, ws):
    return _dot(_rms_gain(pos_mod.rotate(q, i, ws, "half"), gq), _rms_gain(pos_mod.rotate(k, j, ws, "half"), gk))


def norm_rope_order():
    q, k, gq, gk = order_inputs()
    ws = pos_mod.omegas(ORDER_D, ORDER_BASE)
    ones = [1.0] * ORDER_D
    rows, spread = [], {}
    for label, f, a, b in (("norm (with gains), then rotate", _score_norm_then_rotate, gq, gk),
                           ("rotate, then norm (with gains)", _score_rotate_then_norm, gq, gk),
                           ("norm with gains all 1, then rotate", _score_norm_then_rotate, ones, ones),
                           ("rotate, then norm with gains all 1", _score_rotate_then_norm, ones, ones)):
        s = [f(q, k, i, j, a, b, ws) for i, j in ORDER_PAIRS]
        spread[label] = max(s) - min(s)
        rows.append((label, *[_fx(v, 6) for v in s], _below(spread[label])))
    return (_seed_line(NAME_ORDER) + f" One query and one key of width {ORDER_D}. Draws, in this order: q and k, each as 8 times "
            "`round(normal(), 1)`; then the gain vectors for queries and keys, each 8 times `1 + 0.5 * normal()` (unrounded). The "
            f"rotation is the half-split rotary embedding of Volume V, Chapter 8 with base {ORDER_BASE:g} (rates "
            + ", ".join(_fx(w, 4) for w in ws) + f"), the norm the RMS norm of Volume V, Chapter 9 with epsilon 1e-6. The score is the dot "
            "product of the processed query at position i and key at position j, for three pairs of positions with the same offset "
            f"{ORDER_OFFSET}:\n\n"
            + _table(["order", *[f"i = {i}, j = {j}" for i, j in ORDER_PAIRS], "largest difference"], rows)
            + "\n\nWith the gains applied before the rotation the score is the same for every pair of positions with that offset; "
            "applied after it, the gains act on rotated channels and the score depends on where the pair sits.")


# ------------------------------------------------------------------------------------------------ a tiny decoder with every choice

NAME_TINY = "modern_decoder.tiny"
CFG = {"V": 11, "D": 6, "L": 2, "H": 4, "HKV": 2, "DH": 4, "F": 10, "BASE": 100.0}
# Every width differs (D 6, H x DH 16, HKV x DH 8, F 10) so that a transpose or a swap of any two cannot hide (knowledge #4).
IDS = (3, 9, 1, 7, 4, 4, 10, 2)   # the text the decoder reads; the targets are the same ids one place to the left


def cfg_with(**kw):
    c = dict(CFG)
    c.update(kw)
    return c


def init_decoder(cfg, tied=False, name=NAME_TINY):
    """Weights, drawn from `bookrng.generator(name)` (Chapter 1 uses NAME_TINY) in this order, each matrix row by row:
    the embedding table W_emb [V, D] as `normal()`; then for each layer l = 0 .. L-1: the gain of norm 1 (D numbers, each
    `1 + 0.1 * normal()`), W_Q [D, H DH], W_K [D, HKV DH], W_V [D, HKV DH] (`normal() / sqrt(D)`), the QK-norm gains of queries
    and then keys (DH numbers each, `1 + 0.1 * normal()`), W_O [H DH, D] (`normal() / sqrt(H DH)`), the gain of norm 2 (D),
    W_gate and W_up [D, F] (`normal() / sqrt(D)`), W_down [F, D] (`normal() / sqrt(F)`); then the last gain (D) and, unless the
    head is tied, the output matrix W_out [V, D] (`normal() / sqrt(D)`)."""
    g = bookrng.generator(name)
    V, D, L, H, HKV, DH, F = (cfg[k] for k in ("V", "D", "L", "H", "HKV", "DH", "F"))

    def mat(r, c, s):
        return [[s * g.normal() for _ in range(c)] for _ in range(r)]

    def gain(n):
        return [1.0 + 0.1 * g.normal() for _ in range(n)]

    p = {"emb": mat(V, D, 1.0), "layers": []}
    for _ in range(L):
        lay = {"g1": gain(D)}
        lay["Wq"], lay["Wk"], lay["Wv"] = mat(D, H * DH, 1 / math.sqrt(D)), mat(D, HKV * DH, 1 / math.sqrt(D)), mat(D, HKV * DH, 1 / math.sqrt(D))
        lay["qn"], lay["kn"] = gain(DH), gain(DH)
        lay["Wo"] = mat(H * DH, D, 1 / math.sqrt(H * DH))
        lay["g2"] = gain(D)
        lay["Wg"], lay["Wu"], lay["Wd"] = mat(D, F, 1 / math.sqrt(D)), mat(D, F, 1 / math.sqrt(D)), mat(F, D, 1 / math.sqrt(F))
        p["layers"].append(lay)
    p["gn"] = gain(D)
    if not tied:
        p["out"] = mat(V, D, 1 / math.sqrt(D))
    return p


def tensors(p):
    """Every parameter tensor as (name, list of rows or [row]) in draw order; vectors are one-row matrices."""
    out = [("W_emb", p["emb"])]
    for l, lay in enumerate(p["layers"]):
        for key, name in (("g1", "gain 1"), ("Wq", "W_Q"), ("Wk", "W_K"), ("Wv", "W_V"), ("qn", "QK-norm gain, queries"),
                          ("kn", "QK-norm gain, keys"), ("Wo", "W_O"), ("g2", "gain 2"), ("Wg", "W_gate"), ("Wu", "W_up"),
                          ("Wd", "W_down")):
            v = lay[key]
            out.append((f"{name} (layer {l})", v if isinstance(v[0], list) else [v]))
    out.append(("last gain", [p["gn"]]))
    if "out" in p:
        out.append(("W_out", p["out"]))
    return out


def parameter_count(p):
    return sum(len(r) for _, rows in tensors(p) for r in rows)


def new_cache(cfg):
    return [{"K": [], "V": []} for _ in range(cfg["L"])]


def cache_numbers(cfg, cache):
    """How many numbers the cache holds: keys and values, per layer, per position, per key-value head, DH each."""
    return sum(len(x) * cfg["HKV"] * cfg["DH"] for lay in cache for x in (lay["K"], lay["V"]))


def extend(p, cfg, ids, cache, trace=None):
    """Run the tokens `ids` on top of a cache holding c earlier positions; return one row of logits for every new token and append
    the new keys (normalized and rotated) and values to the cache. Empty cache and the whole text is the forward pass; a cache
    and one token is a decode step. With a list `trace`, the intermediates the backward pass needs are appended to it."""
    D, L, H, HKV, DH = (cfg[k] for k in ("D", "L", "H", "HKV", "DH"))
    group, scale = H // HKV, 1.0 / math.sqrt(DH)
    ws = pos_mod.omegas(DH, cfg["BASE"])
    c0 = len(cache[0]["K"])
    n = len(ids)
    X = [list(p["emb"][t]) for t in ids]
    for l, lay in enumerate(p["layers"]):
        N1, cn1 = tb.rms_forward(X, lay["g1"])
        Q0, K0, V0 = sa.matmul(N1, lay["Wq"]), sa.matmul(N1, lay["Wk"]), sa.matmul(N1, lay["Wv"])
        qrows = [Q0[t][h * DH:(h + 1) * DH] for t in range(n) for h in range(H)]
        krows = [K0[t][h * DH:(h + 1) * DH] for t in range(n) for h in range(HKV)]
        Q1, cq = tb.rms_forward(qrows, lay["qn"])
        K1, ck = tb.rms_forward(krows, lay["kn"])
        q2 = [[pos_mod.rotate(Q1[t * H + h], c0 + t, ws, "half") for h in range(H)] for t in range(n)]
        k2 = [[pos_mod.rotate(K1[t * HKV + h], c0 + t, ws, "half") for h in range(HKV)] for t in range(n)]
        v = [[V0[t][h * DH:(h + 1) * DH] for h in range(HKV)] for t in range(n)]
        cache[l]["K"].extend(k2)
        cache[l]["V"].extend(v)
        Kc, Vc = cache[l]["K"], cache[l]["V"]
        O, Pall = [], []
        for t in range(n):
            i = c0 + t
            row, prow = [], []
            for h in range(H):
                kv = h // group
                pr = _softmax_row([_dot(q2[t][h], Kc[j][kv]) * scale for j in range(i + 1)])
                o = [0.0] * DH
                for j, w in enumerate(pr):
                    for d in range(DH):
                        o[d] += w * Vc[j][kv][d]
                row += o
                prow.append(pr)
            O.append(row)
            Pall.append(prow)
        A = sa.matmul(O, lay["Wo"])
        X1 = [[x + a for x, a in zip(rx, ra)] for rx, ra in zip(X, A)]
        N2, cn2 = tb.rms_forward(X1, lay["g2"])
        Fo, cf = tb.swiglu_forward(N2, lay["Wg"], lay["Wu"], lay["Wd"])
        X_out = [[x + f for x, f in zip(rx, rf)] for rx, rf in zip(X1, Fo)]
        if trace is not None:
            trace.append({"cn1": cn1, "N1": N1, "cq": cq, "ck": ck, "q2": q2, "k2": k2, "v": v, "P": Pall, "O": O,
                          "cn2": cn2, "cf": cf})
        X = X_out
    Hn, cnf = tb.rms_forward(X, p["gn"])
    W = p["out"] if "out" in p else p["emb"]
    Z = [[_dot(h, W[r]) for r in range(cfg["V"])] for h in Hn]
    if trace is not None:
        trace.append({"cnf": cnf, "Hn": Hn})
    return Z


def forward(p, cfg, ids):
    return extend(p, cfg, ids, new_cache(cfg))


def loss(p, cfg, ids):
    """Mean cross-entropy of predicting ids[t + 1] from the logits at position t, for t = 0 .. len(ids) - 2."""
    Z = forward(p, cfg, ids)
    total = 0.0
    for t in range(len(ids) - 1):
        m = max(Z[t])
        lse = m + math.log(sum(math.exp(z - m) for z in Z[t]))
        total += lse - Z[t][ids[t + 1]]
    return total / (len(ids) - 1)


def _add_into(dst, src):
    for i, row in enumerate(src):
        for j, x in enumerate(row):
            dst[i][j] += x


def _zeros_like(M):
    return [[0.0] * len(r) for r in M]


def gradients(p, cfg, ids, bug=None):
    """The hand-written backward pass: the gradient of `loss` for every tensor of p (same layout as p). `bug` names a wrong
    backward pass the gradient check must catch: "group_first" (a key-value head receives the gradient of the first query head
    of its group only, not the sum over the group), "rope_forward_angle" (the rotary rotation of the gradient turns by +angle,
    not -angle), "qk_norm_constant" (the QK-norm's scale factor is treated as a constant) and, for a tied head,
    "tied_lookup_only" (the table receives the gradient of the lookup but not that of the head)."""
    V, D, L, H, HKV, DH = (cfg[k] for k in ("V", "D", "L", "H", "HKV", "DH"))
    group, scale = H // HKV, 1.0 / math.sqrt(DH)
    ws = pos_mod.omegas(DH, cfg["BASE"])
    n = len(ids)
    trace = []
    Z = extend(p, cfg, ids, new_cache(cfg), trace)
    tl = trace[-1]
    # logits, head and last norm
    dZ = [[0.0] * V for _ in range(n)]
    for t in range(n - 1):
        pr = _softmax_row(Z[t])
        for r in range(V):
            dZ[t][r] = (pr[r] - (1.0 if r == ids[t + 1] else 0.0)) / (n - 1)
    W = p["out"] if "out" in p else p["emb"]
    dW = [[0.0] * D for _ in range(V)]
    dHn = [[0.0] * D for _ in range(n)]
    for t in range(n):
        for r in range(V):
            z = dZ[t][r]
            if z != 0.0:
                for d in range(D):
                    dW[r][d] += z * tl["Hn"][t][d]
                    dHn[t][d] += z * W[r][d]
    g = {"layers": [None] * L}
    nbug = "norm_constant" if bug == "qk_norm_constant" else None
    dX, dgn = tb.rms_backward(dHn, tl["cnf"])
    g["gn"] = dgn
    sign = 1.0 if bug == "rope_forward_angle" else -1.0
    for l in reversed(range(L)):
        lay, tr = p["layers"][l], trace[l]
        gl = {}
        # X_out = X1 + F(norm2(X1))
        dN2, gf = tb.swiglu_backward(dX, tr["cf"])
        dX1n, gl["g2"] = tb.rms_backward(dN2, tr["cn2"])
        dX1 = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(dX, dX1n)]
        gl["Wg"], gl["Wu"], gl["Wd"] = gf["Wg"], gf["Wu"], gf["Wd"]
        # X1 = X + O W_O
        gl["Wo"] = sa.matmul(sa.transpose(tr["O"]), dX1)
        dO = sa.matmul(dX1, sa.transpose(lay["Wo"]))
        q2, k2, v, P = tr["q2"], tr["k2"], tr["v"], tr["P"]
        dq2 = [[[0.0] * DH for _ in range(H)] for _ in range(n)]
        dk2 = [[[0.0] * DH for _ in range(HKV)] for _ in range(n)]
        dv = [[[0.0] * DH for _ in range(HKV)] for _ in range(n)]
        for h in range(H):
            kv = h // group
            for i in range(n):
                doh = dO[i][h * DH:(h + 1) * DH]
                pr = P[i][h]
                dp = [_dot(doh, v[j][kv]) for j in range(i + 1)]
                rowsum = _dot(pr, dp)
                for j in range(i + 1):
                    ds = pr[j] * (dp[j] - rowsum)
                    take = not (bug == "group_first" and h % group != 0)
                    for d in range(DH):
                        dq2[i][h][d] += scale * ds * k2[j][kv][d]
                        if take:
                            dk2[j][kv][d] += scale * ds * q2[i][h][d]
                            dv[j][kv][d] += pr[j] * doh[d]
        dq1 = [pos_mod.rotate(dq2[t][h], sign * t, ws, "half") for t in range(n) for h in range(H)]
        dk1 = [pos_mod.rotate(dk2[t][h], sign * t, ws, "half") for t in range(n) for h in range(HKV)]
        dq0, gl["qn"] = tb.rms_backward(dq1, tr["cq"], nbug)
        dk0, gl["kn"] = tb.rms_backward(dk1, tr["ck"], nbug)
        dQ0 = [sum((dq0[t * H + h] for h in range(H)), []) for t in range(n)]
        dK0 = [sum((dk0[t * HKV + h] for h in range(HKV)), []) for t in range(n)]
        dV0 = [sum((dv[t][h] for h in range(HKV)), []) for t in range(n)]
        N1t = sa.transpose(tr["N1"])
        gl["Wq"], gl["Wk"], gl["Wv"] = sa.matmul(N1t, dQ0), sa.matmul(N1t, dK0), sa.matmul(N1t, dV0)
        dN1 = sa.matmul(dQ0, sa.transpose(lay["Wq"]))
        _add_into(dN1, sa.matmul(dK0, sa.transpose(lay["Wk"])))
        _add_into(dN1, sa.matmul(dV0, sa.transpose(lay["Wv"])))
        dXn, gl["g1"] = tb.rms_backward(dN1, tr["cn1"])
        dX = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(dX1, dXn)]
        g["layers"][l] = gl
    demb = _zeros_like(p["emb"])
    for t, tok in enumerate(ids):
        for d in range(D):
            demb[tok][d] += dX[t][d]
    if "out" in p:
        g["out"] = dW
    elif bug != "tied_lookup_only":
        _add_into(demb, dW)    # a tied table also receives the gradient of the head
    g["emb"] = demb
    return g


def _flat(p, key_path):
    """The row list at key_path ('emb', 'out', ('layers', l, 'Wq') ...); vectors are wrapped into one row."""
    if isinstance(key_path, str):
        v = p[key_path]
    else:
        v = p[key_path[0]][key_path[1]][key_path[2]]
    return v if isinstance(v[0], list) else [v]


def _paths(p):
    out = [("W_emb", "emb")]
    for l in range(len(p["layers"])):
        for key, name in (("g1", "gain 1"), ("Wq", "W_Q"), ("Wk", "W_K"), ("Wv", "W_V"), ("qn", "QK-norm gain, queries"),
                          ("kn", "QK-norm gain, keys"), ("Wo", "W_O"), ("g2", "gain 2"), ("Wg", "W_gate"), ("Wu", "W_up"),
                          ("Wd", "W_down")):
            out.append((f"{name}", ("layers", l, key)))
    out.append(("last gain", "gn"))
    if "out" in p:
        out.append(("W_out", "out"))
    return out


CHECK_STEP = 1e-5
CRITERION = 1e-6


def check_tensors(p, cfg, ids, bug=None):
    """Compare `gradients` with central differences of `loss` (step CHECK_STEP) entry by entry. Returns {tensor name:
    (entries, largest |gradient|, largest |analytic - numeric|)} with the layers' tensors of one kind merged."""
    g = gradients(p, cfg, ids, bug)
    res = {}
    for name, path in _paths(p):
        rows = _flat(p, path)
        grows = _flat(g, path)
        n = big = worst = 0
        for i, row in enumerate(rows):
            for j in range(len(row)):
                old = row[j]
                row[j] = old + CHECK_STEP
                lp = loss(p, cfg, ids)
                row[j] = old - CHECK_STEP
                lm = loss(p, cfg, ids)
                row[j] = old
                num = (lp - lm) / (2 * CHECK_STEP)
                ana = grows[i][j]
                n += 1
                big = max(big, abs(ana))
                worst = max(worst, abs(ana - num))
        e = res.get(name, (0, 0.0, 0.0))
        res[name] = (e[0] + n, max(e[1], big), max(e[2], worst))
    return res


def tiny_setup():
    cfg = CFG
    p = init_decoder(cfg)
    pt = init_decoder(cfg, tied=True)
    rows = [("token embedding W_emb", f"[{cfg['V']}, {cfg['D']}]", "once"),
            ("norm 1, norm 2 gains", f"[{cfg['D']}]", "per layer"),
            ("W_Q", f"[{cfg['D']}, {cfg['H'] * cfg['DH']}]", "per layer"),
            ("W_K, W_V", f"[{cfg['D']}, {cfg['HKV'] * cfg['DH']}]", "per layer"),
            ("QK-norm gains (queries, keys)", f"[{cfg['DH']}] each", "per layer"),
            ("W_O", f"[{cfg['H'] * cfg['DH']}, {cfg['D']}]", "per layer"),
            ("W_gate, W_up", f"[{cfg['D']}, {cfg['F']}]", "per layer"),
            ("W_down", f"[{cfg['F']}, {cfg['D']}]", "per layer"),
            ("last gain", f"[{cfg['D']}]", "once"),
            ("output matrix W_out", f"[{cfg['V']}, {cfg['D']}]", "once, untied")]
    return (_seed_line(NAME_TINY) + f" The tiny decoder has V = {cfg['V']} token ids, width D = {cfg['D']}, L = {cfg['L']} layers, "
            f"H = {cfg['H']} query heads and {cfg['HKV']} key-value heads of width {cfg['DH']} (so groups of {cfg['H'] // cfg['HKV']}, "
            f"and the query projection is {cfg['H'] * cfg['DH']} wide while the width is {cfg['D']}), a SwiGLU network of hidden width "
            f"F = {cfg['F']}, rotary base {cfg['BASE']:g} on the half-split pairs, RMS norms with epsilon 1e-6, no biases, QK-norm "
            "before the rotation and an untied output matrix. Draws, in this order, each matrix row by row: W_emb as `normal()`; then "
            "for each layer the gain of norm 1 (each `1 + 0.1 * normal()`), W_Q, W_K, W_V (each `normal() / sqrt(D)`), the QK-norm "
            "gains of queries then keys (each `1 + 0.1 * normal()`), W_O (`normal() / sqrt(H x width)`), the gain of norm 2, W_gate "
            "and W_up (`normal() / sqrt(D)`), W_down (`normal() / sqrt(F)`); then the last gain and W_out (`normal() / sqrt(D)`). "
            f"That is {_int(parameter_count(p))} parameters ({_int(parameter_count(pt))} with the head tied to W_emb, which is not "
            "drawn again). The text is the ids "
            + ", ".join(str(i) for i in IDS) + "; the loss predicts each id from the ones before it.\n\n"
            + _table(["tensor", "shape", "how often"], rows, "lll"))


def decode_equals_forward():
    rows = []
    ids = list(IDS)
    for hkv in (4, 2, 1):
        cfg = cfg_with(HKV=hkv)
        p = init_decoder(cfg)
        full = forward(p, cfg, ids)
        cache = new_cache(cfg)
        got = extend(p, cfg, ids[:3], cache)           # prefill of the first 3 tokens
        for t in range(3, len(ids)):                    # then one decode step per token
            got += extend(p, cfg, [ids[t]], cache)
        diff = max(abs(a - b) for ra, rb in zip(full, got) for a, b in zip(ra, rb))
        rows.append((hkv, cfg["H"] // hkv, _int(parameter_count(p)), cache_numbers(cfg, cache), _below(diff)))
    return ("The decoder of the previous table at three numbers of key-value heads (each drawn with the same generator and the same "
            "order, so a different number of draws for W_K and W_V). The 8 tokens are run once as a forward pass with an empty cache; "
            "and again as a prefill of the first 3 tokens followed by 5 decode steps of one token each, on the cache the prefill "
            "left. Largest absolute difference between the two sets of 8 rows of logits (11 numbers each):\n\n"
            + _table(["key-value heads", "group size", "parameters", "numbers in the cache after 8 tokens", "largest difference of logits"],
                     rows))


def gradient_check():
    cfg, ids = CFG, list(IDS)
    out = []
    for label, tied in (("untied head", False), ("head tied to W_emb", True)):
        p = init_decoder(cfg, tied)
        res = check_tensors(p, cfg, ids)
        rows = [(name, _int(n), f"{big:.3f}", _below(worst, CRITERION)) for name, (n, big, worst) in res.items()]
        total = sum(n for n, _, _ in res.values())
        worst_all = max(w for _, _, w in res.values())
        out.append(f"{label}: loss {loss(p, cfg, ids):.4f} on the text; {_int(total)} entries checked, largest difference over all "
                   f"tensors {_below(worst_all, CRITERION)} (criterion: below 1e-6).\n\n"
                   + _table(["tensor (both layers merged)", "entries", "largest absolute gradient", "largest difference from central difference"], rows))
    return ("Gradient check of the hand-written backward pass against central differences, entry by entry, with step "
            f"{CHECK_STEP:g}, in double precision: for every entry w of every tensor, (loss(w + h) - loss(w - h)) / 2h. The loss is the "
            "mean cross-entropy of the text of the previous table. The column of largest gradients is there to show that no tensor "
            "was checked against a gradient of zero.\n\n" + "\n\n".join(out))


def wrong_backward_passes():
    cfg, ids = CFG, list(IDS)
    rows = []
    for label, bug, tied in (("the correct backward pass", None, False),
                             ("a key-value head takes the gradient of the first query head of its group only", "group_first", False),
                             ("the rotation of the gradient turns by +angle instead of -angle", "rope_forward_angle", False),
                             ("the QK-norm's scale factor is treated as a constant", "qk_norm_constant", False),
                             ("the correct backward pass", None, True),
                             ("the tied table receives the gradient of the lookup but not that of the head", "tied_lookup_only", True)):
        p = init_decoder(cfg, tied)
        res = check_tensors(p, cfg, ids, bug)
        bad = [(n, w) for n, (_, _, w) in res.items() if w >= CRITERION]
        worst_name, worst = max(((n, w) for n, (_, _, w) in res.items()), key=lambda x: x[1])
        rows.append((label, "tied" if tied else "untied", f"{len(bad)} of {len(res)}", worst_name if bad else "none",
                     _below(worst, CRITERION) if bad else "all below 1e-6"))
    return ("The same check on backward passes with one deliberate mistake each, on the weights and the text of the previous "
            "tables. A check passes when every tensor is below the criterion of 1e-6; the table lists how many kinds of tensor "
            "fail (a kind merges the two layers) and the one that fails most.\n\n"
            + _table(["backward pass", "head", "kinds of tensor failing", "worst tensor", "largest difference"], rows, "llrll"))
