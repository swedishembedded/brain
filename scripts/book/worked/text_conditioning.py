# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Conditioning on Text".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: text_conditioning.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. The pre-norm block, the masks and the initialisation of
a small encoder are those of Volume V (`training_gpt`, imported, not copied); the cost rule of an encoder block is that of the CLIP
chapter (`clip`), and the denoiser costs are those of the previous chapter (`denoiser_architectures`).

* what each of brain's image models hands its denoiser: rows, widths, the pooled vector, and how much of the window a prompt fills;
* a four-block toy encoder with random weights, read under a causal and a bidirectional mask: which rows two prompts that differ
  in one word share, which pooled vectors tell them apart, which layer is read and how three layers are laid out in one row;
* padding as a convention: the four combinations of a causal or bidirectional encoder with and without a key padding mask, what
  changes in the rows when the window grows, and how much attention a random reader puts on the padded rows;
* the T5-XXL encoder counted from its configuration, the cost of encoding against the cost of denoising, and the weight-byte count
  that a hand-written formula once got wrong.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its stream names and seeds. The
toy encoder is evaluated, not trained, so its numbers are stable to the digits printed; sums are plain left-to-right additions.

Swedish Embedded AB implements solutions for text-conditioned generators whose prompt interface is specified, counted and checked
against exact references. If your team needs expertise in text encoders and prompt conditioning for image generation, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import clip
import denoiser_architectures as da
import self_attention as sa
import training_gpt as tg
import transformer_block as tb

_table, _seed, _f = da._table, da._seed, da._f
_sum = sa._sum


def _int(n):
    return f"{n:,}"


def _pct(x, d=1):
    return f"{100.0 * x:.{d}f}%"


def _dist(a, b):
    return math.sqrt(_sum([(x - y) ** 2 for x, y in zip(a, b)]))


def _below(x, limit=1e-12):
    return f"below {limit:.0e}" if x < limit else _f(x, 4)


# ------------------------------------------------------------------ what each model hands its denoiser

# name, tokens in the window, row width, pooled width, latent numbers at 1024 x 1024, where the rows are read
INTERFACES = (
    ("SDXL", 77, 768 + 1280, 1280, 4 * 128 * 128, "cross-attention keys and values, 70 sites"),
    ("FLUX.1 dev", 512, 4096, 768, 16 * 128 * 128, "txt_in, then joint attention"),
    ("FLUX.2 klein-4b", 512, 3 * 2560, 0, 4096 * 128, "txt_in, then joint attention"),
    ("FLUX.2 klein-9b", 512, 3 * 4096, 0, 4096 * 128, "txt_in, then joint attention"),
    ("Z-Image Turbo", 512, 2560, 0, 16 * 128 * 128, "cap_embedder, then joint attention"),
)


def interface_table():
    rows = []
    for name, w, d, pooled, latent, where in INTERFACES:
        seq = w * d
        rows.append([name, _int(w), _int(d), _int(seq), _int(pooled) if pooled else "none", _int(seq + pooled), _int(latent),
                     f"{seq / latent:.1f}"])
    return (
        "What leaves the text side of each pipeline for one prompt, from the configurations of the crates (SDXL: the 77-position "
        "windows of its two CLIP towers, concatenated along the feature axis, 768 + 1280; FLUX.1: the 512-row T5-XXL window, the "
        "largest `max_len` it accepts, and the pooled CLIP-L vector; FLUX.2: three Qwen3 hidden states per token; Z-Image: one "
        "Qwen3-4B hidden state per token and the default caption capacity of 512). The latent is what the denoiser steers at 1024 "
        "by 1024 pixels: 4 x 128 x 128 for SDXL, 16 x 128 x 128 for FLUX.1 and Z-Image, and 4096 tokens of 128 values for FLUX.2.\n\n"
        + _table(["pipeline", "rows", "row width", "sequence numbers", "pooled numbers", "text numbers", "latent numbers",
                  "sequence / latent"], rows, "lrrrrrrr")
    )


def window_use_table(used=(12, 40)):
    """The share of a window that is padding when a prompt fills `used` rows, and for the transformers the share of the attention
    score elements of the joint sequence (4096 image tokens and the window) that touch a padded row."""
    rows = []
    for name, w, d, pooled, latent, where in INTERFACES:
        row = [name, _int(w)]
        for u in used:
            pad = w - u
            row.append(_pct(pad / w))
            if name == "SDXL":
                row.append("-")
            else:
                n = 4096 + w
                row.append(_pct(1 - ((n - pad) / n) ** 2))
        rows.append(row)
    heads = ["pipeline", "rows"]
    for u in used:
        heads += [f"padding, {u} rows used", f"score elements on padding, {u} used"]
    return (
        "A prompt that fills `u` rows of the window leaves the other rows to padding. For the transformers the joint sequence has "
        "N = 4096 + rows tokens at 1024 by 1024, and with p padded rows the score elements that involve a padded query or key are "
        "1 - ((N - p) / N)^2 of the N^2 (SDXL's cross-attention has no such sequence).\n\n"
        + _table(heads, rows, "lr" + "rr" * len(used))
    )


# ------------------------------------------------------------------ a small encoder, read under two masks

TOY = "text_conditioning.toy"
TOY_CFG = {"V": 9, "D": 8, "H": 2, "FF": 12, "L": 4, "T": 16}
TOY_STD = 0.5
WORDS = ["<pad>", "<s>", "</s>", "a", "red", "blue", "cube", "sphere", "on"]
ID = {w: i for i, w in enumerate(WORDS)}
PAD = ID["<pad>"]
PROMPT_1 = "a red cube on a blue sphere"
PROMPT_2 = "a blue cube on a red sphere"
SHORT = "a red cube"

CAUSAL = lambda i, j: j <= i                                         # noqa: E731
FULL = lambda i, j: True                                              # noqa: E731


def _toy():
    return tg.init_gpt(TOY, TOY_CFG, std=TOY_STD)


def _ids(text, window=None):
    ids = [ID["<s>"]] + [ID[w] for w in text.split()] + [ID["</s>"]]
    n = len(ids)
    if window is not None:
        ids = ids + [PAD] * (window - n)
    return ids, n


def _mask(kind, n):
    """kind: "causal" or "full", with the key padding mask `masked` removing every key at or after position n."""
    base, masked = kind
    inner = CAUSAL if base == "causal" else FULL
    return (lambda i, j: inner(i, j) and j < n) if masked else inner


def _streams(p, ids, allowed):
    """The residual stream after the embedding (index 0) and after each block (index k: block k - 1), and the final normalised rows."""
    x = [[a + b for a, b in zip(p["E"][t], p["P"][i])] for i, t in enumerate(ids)]
    out = [x]
    for b in p["blocks"]:
        x, _ = tg.block(b, x, TOY_CFG["H"], allowed)
        out.append(x)
    final, _ = tb.rms_forward(x, p["gf"][0])
    return out, final


def toy_setup():
    p = _toy()
    n_params = tg.parameter_count(p)
    gains = (2 * len(p["blocks"]) + 1) * TOY_CFG["D"]
    rows = []
    for text in (PROMPT_1, PROMPT_2, SHORT):
        ids, n = _ids(text)
        rows.append([f"`{text}`", str(n), " ".join(str(i) for i in ids)])
    vocab = ", ".join(f"{i} `{w}`" for i, w in enumerate(WORDS))
    return (
        f"The toy encoder is `init_gpt` of Volume V (stream `{TOY}`, seed 0x{bookrng.seed_of(TOY):016x}) with V = 9, D = 8, "
        "2 heads of width 4, SwiGLU width 12, L = 4 blocks and 16 learned positions, every matrix entry `0.5 * normal()` (instead "
        "of the 0.02 of a language model, so that attention has something to do), drawn row by row in the order embedding table, "
        "position table, then per block W_Q, W_K, W_V, W_O, W_gate, W_up, W_down (W_O and W_down multiplied by 1 / sqrt(2 L)); all "
        f"norm gains are 1 (two a block and a final one, {_int(gains)} numbers, not drawn). That is {_int(n_params)} parameters, "
        f"{_int(n_params - gains)} of them drawn. The pre-norm block is the one of Volume V, Chapter 9, the epsilon "
        "of every RMS norm is 1e-6, and a text is `<s>`, its words, `</s>`, then `<pad>` up to the window. Vocabulary: "
        + vocab + ".\n\n"
        + _table(["text", "tokens", "ids"], rows, "lrl")
    )


def _pair(kind):
    p = _toy()
    a, n = _ids(PROMPT_1)
    b, _ = _ids(PROMPT_2)
    allowed = CAUSAL if kind == "causal" else FULL
    return _streams(p, a, allowed)[1], _streams(p, b, allowed)[1], a, b, p


def order_data():
    """Per position, the distance between the final rows of the two prompts under the causal and the full mask."""
    ca, cb, a, b, _ = _pair("causal")
    fa, fb, _, _, _ = _pair("full")
    return [(i, WORDS[a[i]], WORDS[b[i]], _dist(ca[i], cb[i]), _dist(fa[i], fb[i])) for i in range(len(a))]


def order_example():
    data = order_data()
    rows = [[str(i), f"`{x}`", f"`{y}`", "0" if c == 0.0 else _f(c, 4), _f(f, 4)] for i, x, y, c, f in data]
    ca, cb, a, b, p = _pair("causal")
    fa, fb, _, _, _ = _pair("full")
    bag = lambda ids: [_sum([p["E"][t][j] for t in ids]) / len(ids) for j in range(TOY_CFG["D"])]   # noqa: E731
    mean = lambda rows_: [_sum([r[j] for r in rows_]) / len(rows_) for j in range(TOY_CFG["D"])]    # noqa: E731
    pool = [
        ("mean of the token embeddings (no encoder)", _dist(bag(a), bag(b))),
        ("causal encoder, mean of the rows", _dist(mean(ca), mean(cb))),
        ("causal encoder, row of `</s>`", _dist(ca[-1], cb[-1])),
        ("bidirectional encoder, mean of the rows", _dist(mean(fa), mean(fb))),
        ("bidirectional encoder, row of `<s>`", _dist(fa[0], fb[0])),
    ]
    prow = [[name, _below(d) if d < 1e-12 else _f(d, 4)] for name, d in pool]
    return (
        f"Two prompts of nine tokens with no padding, `{PROMPT_1}` and `{PROMPT_2}` (the two colours exchanged), through the toy "
        "encoder of `text_conditioning.toy_setup`, once with the causal mask and once with the full mask. A row is the final "
        "RMS-normalised vector of a position (its Euclidean norm is about 2.83 = sqrt(8)). Distance between the two prompts' rows "
        "at each position:\n\n"
        + _table(["position", "token, prompt 1", "token, prompt 2", "causal", "bidirectional"], rows, "lllrr")
        + "\n\nDistance between the two prompts' single vectors for five ways of pooling:\n\n"
        + _table(["pooled vector", "distance"], prow, "lr")
    )


def taps_example():
    """RMS of the rows at each depth, and the layout of three taps in one row of a conditioning buffer."""
    p = _toy()
    ids, n = _ids(PROMPT_1)
    streams, final = _streams(p, ids, CAUSAL)
    d = TOY_CFG["D"]
    rms = lambda row: math.sqrt(_sum([v * v for v in row]) / d)                                  # noqa: E731
    cos = lambda u, v: _sum([x * y for x, y in zip(u, v)]) / math.sqrt(_sum([x * x for x in u]) * _sum([y * y for y in v]))  # noqa: E731
    pen = streams[TOY_CFG["L"] - 1]
    rows = []
    names = ["embedding + position", "after block 0", "after block 1", "after block 2 (the penultimate)", "after block 3 (the last)"]
    for k, nm in enumerate(names):
        s = streams[k]
        rows.append([nm, _f(_sum([rms(r) for r in s]) / n, 4), _f(min(rms(r) for r in s), 4), _f(max(rms(r) for r in s), 4),
                     _f(_sum([cos(r, q) for r, q in zip(s, pen)]) / n, 4)])
    rows.append(["final RMS norm of the last block", _f(_sum([rms(r) for r in final]) / n, 4), _f(min(rms(r) for r in final), 4),
                 _f(max(rms(r) for r in final), 4), _f(_sum([cos(r, q) for r, q in zip(final, pen)]) / n, 4)])
    first = (
        f"The prompt `{PROMPT_1}` (nine tokens, causal mask) through the toy encoder. The residual stream is not normalised "
        "between blocks, so the length of a row is something the blocks add to; the table gives the RMS of the rows at each depth "
        "(mean, smallest, largest over the nine positions) and the mean cosine similarity of each depth's rows with the penultimate rows.\n\n"
        + _table(["depth", "mean RMS", "smallest RMS", "largest RMS", "mean cosine similarity with the penultimate rows"], rows, "lrrrr")
    )

    # the layout of three taps (the outputs of blocks 0, 1 and 2: a quarter, a half and three quarters of the depth)
    taps = (1, 2, 3)
    t, w = n, d
    right, wrong = [], []
    flat = [(k, tok) for k in taps for tok in range(t) for _ in range(w)]          # the buffer [3, T, D], tap-major, one entry per number
    for r in (0, 1, t - 1):
        right.append(", ".join(f"tap {k} token {r}" for k in taps))
        cells = flat[r * 3 * w:(r + 1) * 3 * w]
        seen = []
        for c in cells:
            if c not in seen:
                seen.append(c)
        wrong.append(", ".join(f"tap {k} token {tok}" for k, tok in seen))
    lay = [[str(r), right[i], wrong[i]] for i, r in enumerate((0, 1, t - 1))]
    return (
        first + "\n\nThree taps (the streams after 1, 2 and 3 blocks, a quarter, a half and three quarters of the depth as 9, 18 and 27 "
        "are of 36, written tap 1, 2 and 3 in the table) of the same nine "
        "positions have to enter a conditioning row of 24 numbers per position. A buffer `taps[3, 9, 8]` read as 9 rows of 24 "
        "numbers without moving anything fills each row from other positions' states:\n\n"
        + _table(["row", "right layout: the row holds", "buffer reshaped as it lies: the row holds"], lay, "lll")
    )


# ------------------------------------------------------------------ padding as a convention

CONVENTIONS = (
    ("bidirectional, pad keys readable", ("full", False)),
    ("bidirectional, pad keys masked", ("full", True)),
    ("causal, pad keys readable", ("causal", False)),
    ("causal, pad keys masked", ("causal", True)),
)
WINDOWS = (8, 12, 16)


def _rows(kind, window, text=SHORT):
    p = _toy()
    ids, n = _ids(text, window)
    return _streams(p, ids, _mask(kind, n))[1], n


def padding_matrix():
    n = _ids(SHORT)[1]
    a_rows, b_rows = [], []
    for name, kind in CONVENTIONS:
        r8, _ = _rows(kind, 8)
        r12, _ = _rows(kind, 12)
        content = max(abs(x - y) for i in range(n) for x, y in zip(r8[i], r12[i]))
        pad = max(abs(x - y) for i in range(n, 8) for x, y in zip(r8[i], r12[i]))
        a_rows.append([name, _below(content), _below(pad) if pad < 1e-12 else _f(pad, 4)])
    for base in ("full", "causal"):
        r_open, _ = _rows((base, False), 8)
        r_mask, _ = _rows((base, True), 8)
        content = max(abs(x - y) for i in range(n) for x, y in zip(r_open[i], r_mask[i]))
        pad = max(abs(x - y) for i in range(n, 8) for x, y in zip(r_open[i], r_mask[i]))
        b_rows.append([base + " encoder", _below(content), _below(pad) if pad < 1e-12 else _f(pad, 4)])
    return (
        f"The prompt `{SHORT}` (five tokens with `<s>` and `</s>`) padded with `<pad>` to a window of 8 positions (3 pads) and to "
        "12 (7 pads), through the toy encoder of `text_conditioning.toy_setup`; the pad keys are removed by the mask `j < 5` where "
        "stated. Largest absolute difference in any coordinate of the final normalised rows between the two windows, at the five "
        "content rows and at the three pad rows (positions 5 to 7) that both windows have:\n\n"
        + _table(["encoder", "content rows, window 8 against 12", "pad rows 5 to 7, window 8 against 12"], a_rows, "lrr")
        + "\n\nAnd at the one window of 8, the largest difference between the encoder with its pad keys readable and the same "
        "encoder with them masked:\n\n"
        + _table(["encoder", "content rows", "pad rows 5 to 7"], b_rows, "lrr")
    )


READER = "text_conditioning.reader"


def reader_mass():
    g = bookrng.generator(READER)
    d = TOY_CFG["D"]
    n_q = 6
    queries = [[g.normal() for _ in range(d)] for _ in range(n_q)]
    wk = [[0.5 * g.normal() for _ in range(d)] for _ in range(d)]
    n = _ids(SHORT)[1]
    rows = [["share of the window that is padding"] + [_pct((w - n) / w) for w in WINDOWS]]
    for name, kind in CONVENTIONS:
        line = [name]
        for w in WINDOWS:
            r, _ = _rows(kind, w)
            keys = sa.matmul(r, wk)
            mass = 0.0
            for q in queries:
                s = [sa._dot(q, k) / math.sqrt(d) for k in keys]
                pr = sa._softmax(s)
                mass += _sum(pr[n:])
            line.append(_pct(mass / n_q, 2))
        rows.append(line)
    return (
        f"A reader of the encoder's rows that never masks (as the cross-attention of SDXL and the joint attention of the "
        f"transformers do not): `{SHORT}` in windows of 8, 12 and 16 through the four encoders of `text_conditioning.padding_matrix`, "
        f"read by {n_q} queries of one head of width 8. Stream `{READER}` (seed 0x{bookrng.seed_of(READER):016x}); draws in this "
        f"order: the {n_q} queries, each 8 values of `normal()`, then W_K (8 by 8, row by row, `0.5 * normal()`). The keys are the "
        "final normalised rows times W_K, the scores are the dot products over sqrt(8), the softmax runs over every row of the "
        "window, and the table gives the mean over the queries of the weight that falls on the pad rows (positions 5 and up).\n\n"
        + _table(["encoder"] + [f"window {w}" for w in WINDOWS], rows, "lrrr")
    )


# ------------------------------------------------------------------ the cost of encoding

def block_flops(t, d, heads, head_dim, kv_heads, ffn, gated):
    """FLOPs of one encoder block over t tokens by the rule of Volume V, Chapter 9: the projections (queries and output over
    heads * head_dim, keys and values over kv_heads * head_dim), the feed-forward network (3 matrices if gated, else 2), the score
    and average products 4 t^2 heads head_dim, and the softmax 5 heads t^2."""
    proj = 2 * t * d * heads * head_dim + 2 * 2 * t * d * kv_heads * head_dim + 2 * t * heads * head_dim * d
    mlp = (6 if gated else 4) * t * d * ffn
    return proj + mlp + 4 * t * t * heads * head_dim + 5 * heads * t * t


assert block_flops(77, 768, 12, 64, 12, 3072, False) == clip._block_flops(77, 768, 12, 3072, False)
assert block_flops(77, 1280, 20, 64, 20, 5120, False) == clip._block_flops(77, 1280, 20, 5120, False)

T5_XXL = dict(vocab=32128, d=4096, ff=10240, dkv=64, layers=24, heads=64, buckets=32)
UMT5_XXL = dict(T5_XXL, vocab=256384)
QWEN3_4B = dict(d=2560, layers=36, heads=32, kv_heads=8, head_dim=128, ff=9728)
QWEN3_8B = dict(d=4096, layers=36, heads=32, kv_heads=8, head_dim=128, ff=12288)
FLUX2_TAPS = (9, 18, 27)


def t5_groups(c, per_block_bias=False):
    d, ff, inner = c["d"], c["ff"], c["heads"] * c["dkv"]
    bias = c["buckets"] * c["heads"]
    g = {
        "token embedding": c["vocab"] * d,
        "attention (fused q, k, v and o)": c["layers"] * (3 * inner * d + d * inner),
        "feed-forward (wi_0, wi_1, wo)": c["layers"] * 3 * ff * d,
        "block norms (two per block)": c["layers"] * 2 * d,
        "relative-position bias": bias * (c["layers"] if per_block_bias else 1),
        "final norm": d,
    }
    return g


def t5_count():
    rows = []
    for name, c, per in (("T5-XXL (FLUX.1)", T5_XXL, False), ("umT5-XXL (Wan)", UMT5_XXL, True)):
        g = t5_groups(c, per)
        total = sum(g.values())
        for k, v in g.items():
            rows.append([name if k == "token embedding" else "", k, _int(v), _pct(v / total, 2)])
        rows.append(["", "total", _int(total), "100%"])
        rows.append(["", "bytes in 32-bit floats", _int(4 * total), ""])
    return (
        "The encoder stacks of `crates/t5encoder`, counted from the configuration: width 4096, 24 blocks, 64 heads of width 64 "
        "(so the attention inner width is 4096), gated feed-forward width 10240, 32 relative-position buckets, vocabulary 32128 "
        "(umT5: 256384 and one bias table per block). The attention holds q, k, v and the output matrix (4 x 4096 x 4096 per block) "
        "and the feed-forward network three matrices of 10240 by 4096.\n\n"
        + _table(["encoder", "part", "parameters", "share"], rows, "llrr")
    )


QWEN_VOCAB = 151936
CLIP_L_PARAMS = 123_060_480      # clip.tower_cost: 49408 x 768 token rows, 77 positions, 12 blocks, final norm
CLIP_G_PARAMS = 694_659_840      # the same with 1280-wide rows, 32 blocks and the 1280 x 1280 projection


def qwen_params(c, blocks):
    """Parameters the text encoder reads when it runs `blocks` blocks: the token table and per block the query, key, value and
    output matrices (queries and output over heads * head_dim, keys and values over kv_heads * head_dim), the gated feed-forward
    network, two norms of width d and the two per-head norms of width head_dim."""
    d, hd = c["d"], c["head_dim"]
    layer = (d * c["heads"] * hd + 2 * d * c["kv_heads"] * hd + c["heads"] * hd * d + 3 * d * c["ff"] + 2 * d + 2 * hd)
    return QWEN_VOCAB * d + blocks * layer


def flux1_params():
    """FLUX.1 dev from the manifest of `crates/flux1/src/config.rs`: every linear layer biased, 19 double-stream blocks (two streams
    of modulation 6d, qkv 3d, projection d, a feed-forward network 4d and back, two query-key gains of 128) and 38 single-stream."""
    d, mlp = 3072, 4 * 3072
    lin = lambda o, i: o * i + o                       # noqa: E731
    emb = lin(d, 64) + lin(d, 4096) + 2 * (lin(d, 256) + lin(d, d)) + (lin(d, 768) + lin(d, d))
    stream = lin(6 * d, d) + lin(3 * d, d) + 2 * 128 + lin(d, d) + lin(mlp, d) + lin(d, mlp)
    single = lin(3 * d, d) + lin(3 * d + mlp, d) + lin(d, d + mlp) + 2 * 128
    out = lin(2 * d, d) + lin(64, d)
    return emb + 19 * 2 * stream + 38 * single + out


def encoder_work_data():
    """(name, encoder parameters, denoiser parameters, FLOPs of one prompt's encoding, FLOPs of one image's denoising) for the five
    pipelines at their default passes."""
    sd_n = da.unet_count(da.SDXL_UNET, 128, 128)[0]
    clip_l = 12 * block_flops(77, 768, 12, 64, 12, 3072, False)
    clip_g = 32 * block_flops(77, 1280, 20, 64, 20, 5120, False)
    t5 = T5_XXL["layers"] * block_flops(512, 4096, 64, 64, 64, 10240, True)

    def qwen(c, blocks, t=512):
        return blocks * block_flops(t, c["d"], c["heads"], c["head_dim"], c["kv_heads"], c["ff"], True)

    zl, za = da.zimage_pass(1024)
    k4l, k4a = da.flux2_pass(da.FLUX2["klein-4b"], 1024)
    k9l, k9a = da.flux2_pass(da.FLUX2["klein-9b"], 1024)
    deep = max(FLUX2_TAPS)
    return [
        ("SDXL, 30 steps with guidance", CLIP_L_PARAMS + CLIP_G_PARAMS, sd_n.total_params(), clip_l + clip_g, 60 * sd_n.total_flops()),
        ("FLUX.1 dev, 50 steps", sum(t5_groups(T5_XXL).values()) + CLIP_L_PARAMS, flux1_params(), t5 + clip_l, 50 * flux1_pass_flops(1024, 512)),
        ("FLUX.2 klein-4b, 4 steps", qwen_params(QWEN3_4B, deep), da.flux2_params(da.FLUX2["klein-4b"])["total"], qwen(QWEN3_4B, deep), 4 * (k4l + k4a)),
        ("FLUX.2 klein-9b, 4 steps", qwen_params(QWEN3_8B, deep), da.flux2_params(da.FLUX2["klein-9b"])["total"], qwen(QWEN3_8B, deep), 4 * (k9l + k9a)),
        ("Z-Image Turbo, 8 steps", qwen_params(QWEN3_4B, 35), da.zimage_total_params(), qwen(QWEN3_4B, 35), 8 * (zl + za)),
    ]


def flux1_pass_flops(side, txt):
    """One FLUX.1 dev forward pass in FLOPs (multiply-adds twice): per token and block the linear layers hold 12 d^2 weights in
    a double-stream block (qkv 3 d^2, projection d^2, a feed-forward network of 2 x 4 d^2) and 12 d^2 in a single-stream block
    (linear1 d x 7 d, linear2 5 d x d); attention is 4 n^2 d per block; the embedders are img_in (64 to d), txt_in (4096 to d)
    and the output layer. Norms, softmax and the modulation are left out."""
    d, dbl, sgl = 3072, 19, 38
    n_img = (side // 16) ** 2
    n = n_img + txt
    lin = 2 * 12 * d * d * (dbl * n + sgl * n)
    att = (dbl + sgl) * 4 * n * n * d
    emb = 2 * n_img * 64 * d + 2 * txt * 4096 * d + 2 * n_img * d * 64
    return lin + att + emb


def encoder_work():
    rows = []
    for name, pe, pd, fe, fd in encoder_work_data():
        rows.append([name, _int(pe), _int(pd), f"{pe / pd:.2f}", f"{fe / 1e9:,.1f}", f"{fd / 1e12:,.1f}", _pct(fe / fd, 3)])
    return (
        "The text side against the denoising side of one 1024 by 1024 image at the default step counts of Chapter 7, from the "
        "configurations: the parameters the text encoders read, the parameters of the denoiser, the work (a multiply-add is two "
        "operations; lookups, norms and activations are left out) of encoding one prompt by the rule of Volume V, Chapter 9, and "
        "the work of all the denoising passes. Encoders: SDXL's two CLIP towers over 77 tokens; FLUX.1's T5-XXL over 512 tokens "
        "and CLIP-L over 77; FLUX.2's Qwen3 over 512 tokens and the first 27 of its 36 blocks, which is as deep as its deepest tap "
        "reads; Z-Image's Qwen3-4B over 512 tokens and the first 35 of 36. Denoisers: `denoiser_architectures` "
        "(FLUX.1 by this rule: 19 double and 38 single blocks of width d = 3072, n = (side / 16)^2 image rows and t text rows, "
        "N = n + t; a double-stream or a single-stream block has 12 d^2 linear weights per row (a double block: the query-key-value "
        "3 d^2, the projection d^2 and a feed-forward network of 2 x 4 d^2; a single block: 7 d^2 in and 5 d^2 out), so the "
        "linear layers cost 2 x 12 d^2 N per block, attention 4 N^2 d per block, and the embedders 2 n 64 d (image rows in), "
        "2 t 4096 d (text rows in) and 2 n d 64 (out); norms, softmax and modulation are left out).\n\n"
        + _table(["pipeline", "encoder params", "denoiser params", "ratio", "encode, GFLOP", "denoise, TFLOP", "work ratio"],
                 rows, "lrrrrrr")
    )


def te_bytes_story():
    """The weight bytes of T5-XXL: the manifest's count, and a formula that counted two feed-forward matrices a block."""
    c = T5_XXL
    d, ff, layers, heads, vocab = c["d"], c["ff"], c["layers"], c["heads"], c["vocab"]
    manifest = 4 * sum(t5_groups(c).values())
    by_hand = (2 * d * ff + 4 * 64 * heads * d + 2 * d) * layers * 4 + d * vocab * 4
    miss = manifest - by_hand
    one_matrix = d * ff * layers * 4
    gib = 2 ** 30
    rows = [
        ["the formula with two feed-forward matrices a block", _int(by_hand), f"{by_hand / gib:.2f}"],
        ["the manifest count (three matrices, bias table, final norm)", _int(manifest), f"{manifest / gib:.2f}"],
        ["the difference", _int(miss), f"{miss / gib:.2f}"],
        ["one matrix of 10240 by 4096 in each of 24 blocks", _int(one_matrix), f"{one_matrix / gib:.2f}"],
    ]
    return (
        "T5-XXL's weight bytes in 32-bit floats, as `te_bytes` of `crates/flux1/src/pipeline.rs` once computed them "
        "((2 d d_ff + 4 x 64 x heads x d + 2 d) x layers x 4 + d x vocab x 4, with d = 4096, d_ff = 10240, 64 heads and 24 layers) "
        "and as the tensor manifest counts them (three feed-forward matrices a block, the bias table of 32 x 64 and the final norm "
        f"of 4096 as well). One GiB is 2^30 bytes. The difference is the missing matrix in each block plus {_int(miss - one_matrix)} "
        "bytes for the bias table and the final norm.\n\n"
        + _table(["quantity", "bytes", "GiB"], rows, "lrr")
    )


# ------------------------------------------------------------------ how SDXL reads its text

def sdxl_sites():
    """The spatial-transformer blocks of the SDXL U-Net by channel width, from its configuration: the blocks per level times the
    resnet slots that carry attention (2 per down level, 3 per up level, the middle block once)."""
    cfg = da.SDXL_UNET
    sites = {}
    layers, levels = cfg["layers"], len(cfg["ch"])
    for i in range(levels):
        if cfg["down"][i] == "attn":
            sites[cfg["ch"][i]] = sites.get(cfg["ch"][i], 0) + layers * cfg["tf"][i]
    sites[cfg["ch"][-1]] = sites.get(cfg["ch"][-1], 0) + cfg["tf"][-1]
    for i in range(levels):
        level = levels - 1 - i
        if cfg["up"][i] == "attn":
            sites[cfg["ch"][level]] = sites.get(cfg["ch"][level], 0) + (layers + 1) * cfg["tf"][level]
    return sites


def sdxl_reading():
    cfg = da.SDXL_UNET
    x = cfg["cross_dim"]
    sites = sdxl_sites()
    rows, params = [], 0
    for c in sorted(sites):
        per = 2 * c * x
        rows.append([str(c), str(sites[c]), _int(per), _int(per * sites[c]), _int(2 * 77 * per * sites[c])])
        params += per * sites[c]
    total_sites = sum(sites.values())
    flops = 2 * 77 * params
    clip_l = 12 * block_flops(77, 768, 12, 64, 12, 3072, False)
    clip_g = 32 * block_flops(77, 1280, 20, 64, 20, 5120, False)
    rows.append(["total", str(total_sites), "", _int(params), _int(flops)])
    add_in = 1280 + 6 * 256
    add_p = add_in * 1280 + 1280 + 1280 * 1280 + 1280
    vec_p = 768 * 3072 + 3072 + 3072 * 3072 + 3072
    return (
        "How SDXL's U-Net reads the 77 x 2048 sequence, from the configuration of `denoiser_architectures` (levels of 320, 640 and 1280 channels, 1, 2 and 10 "
        "transformer blocks per resnet slot, attention at the two coarser levels, the middle block once, 2 resnet slots per down "
        "level and 3 per up level). Every block projects the 77 rows to keys and values with one weight of 2c by 2048, "
        "2 x c x 2048 numbers, which costs 2 x 77 x (that many) FLOPs in a pass; the table sums it over the blocks of each width.\n\n"
        + _table(["channels c", "blocks", "k and v weights per block", "k and v weights in all", "FLOPs of those projections per pass"],
                 rows, "rrrrr")
        + f"\n\nThe two CLIP towers cost {_int(clip_l + clip_g)} FLOPs for a prompt (`text_conditioning.encoder_work`), once; the key "
        f"and value projections of the same rows cost {_int(flops)} in every pass, {flops / (clip_l + clip_g):.2f} times as much, and "
        f"{60 * flops / (clip_l + clip_g):.1f} times as much over the 60 passes of 30 steps with guidance.\n\n"
        f"The pooled vector enters elsewhere: the added embedding reads {_int(add_in)} numbers (1280 pooled and six 256-wide "
        f"sinusoids, so 1280 of them text), through two layers of {_int(add_p)} parameters. FLUX.1's `vector_in` "
        f"reads 768 numbers through two layers of {_int(vec_p)} parameters and adds the result to the timestep's vector."
    )


def readers():
    """Parameters of the first layer that reads the rows, per pipeline."""
    rows = [
        ["FLUX.1 dev", "txt_in", "4096 to 3072, biased", 4096 * 3072 + 3072],
        ["FLUX.2 klein-4b", "txt_in", "7680 to 3072, no bias", 7680 * 3072],
        ["FLUX.2 klein-9b", "txt_in", "12288 to 4096, no bias", 12288 * 4096],
        ["Z-Image Turbo", "cap_embedder", "RMS norm of 2560, then 2560 to 3840, biased", 2560 + 2560 * 3840 + 3840],
    ]
    return (
        "The first layer that reads the rows of the transformers' prompt window, from the configurations "
        "(`crates/flux1/src/config.rs`, `crates/flux2/src/config.rs`, `crates/s3dit/src/model.rs`):\n\n"
        + _table(["pipeline", "layer", "shape", "parameters"], [[a, b, c, _int(d)] for a, b, c, d in rows], "lllr")
    )
