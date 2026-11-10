# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Vision-Language Models": how a language model is given eyes.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: vlms.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. Earlier
modules are imported, never copied: the prefill FLOPs and the cache bytes of a decoder shape (`kv_cache`), and the position
triples of a prompt with a picture (`positions`).

* a prompt with a picture in it, position by position: a printed vocabulary, a 4 by 4 picture cut into four 2 by 2 patches, a
  tower reduced to its patch embedding, a linear projector, and the embedding rows after the splice, with the gradient that the
  splice routes back;
* the vision projectors of the crates (linear, two-layer, patch-merging, split, resampler) with parameter counts and FLOPs;
* the rounding rule of a dynamic resolution (`smart_resize`) on several aspect ratios and two pixel budgets, the slicing rule of a
  tiled design (the closest grid in log space) and the overlap crops of a design that pools back to a fixed count;
* image tokens against resolution for fixed, tiled and dynamic designs, and what they cost the decoder in prefill FLOPs and cache
  bytes;
* one-dimensional positions against the triples of a multi-axis rotary embedding, and how far a picture moves the counter;
* quantized location tokens of a box, their decoding error and the parser that reads them back.

Everything is deterministic arithmetic on printed constants: no random number is drawn, so no generator or seed is involved. Sums
of products run left to right. Python's `round` is round half to even, the rule of the reference rounding functions the chapter
reproduces.

Swedish Embedded AB implements solutions for multimodal models whose image-token budget is counted before it is paid. If your team
needs expertise in vision-language model integration, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import kv_cache as kvc
import positions as pos_mod

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(v):
    return f"{v:,}"


def _fx(v, d=2):
    return f"{v:.{d}f}"


def _vec(v, d=2):
    return "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"


def _gflops(v, d=1):
    return f"{v / 1e9:,.{d}f}"


def _tflops(v, d=2):
    return f"{v / 1e12:,.{d}f}"


def _mib(b):
    return f"{b / 2 ** 20:,.1f}"


def _matvec(w, x):
    out = []
    for row in w:
        s = 0.0
        for a, b in zip(row, x):
            s += a * b
        out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ a prompt with a picture in it

VOCAB = ("<bos>", "<image>", "what", "is", "this", "?", "cat", "<eos>")
EMBED = (
    (0.0, 0.1, 0.0),   # <bos>
    (0.5, 0.5, 0.5),   # <image>: a row whose value will never be read
    (1.0, 0.0, 0.5),   # what
    (0.0, 1.0, 0.5),   # is
    (0.5, 0.5, 1.0),   # this
    (0.2, 0.2, 0.2),   # ?
    (1.0, 1.0, 0.0),   # cat
    (0.0, 0.0, 0.1),   # <eos>
)
PIXELS = (
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.5, 1.0, 0.5),
    (1.0, 0.0, 0.0, 0.5),
    (1.0, 1.0, 0.5, 1.0),
)
PATCH = 2
W_EMBED = ((0.5, 0.5, 0.5, 0.5), (1.0, -1.0, 1.0, -1.0))   # the tower, reduced to a patch embedding: 2 x 4
W_PROJ = ((2.0, 0.0), (0.0, 2.0), (1.0, 1.0))               # the projector: 3 x 2
B_PROJ = (0.1, -0.1, 0.0)
PROMPT_IDS = (1, 2, 3, 4, 5)                                # "<image> what is this ?" ; <bos> is put in front
GRAD_ROWS = tuple((0.1 * t, -0.1 * t, 0.05) for t in range(9))


def patches():
    """The 2 by 2 patches of PIXELS in row-major order, each flattened row by row."""
    out = []
    for a in range(0, len(PIXELS), PATCH):
        for b in range(0, len(PIXELS[0]), PATCH):
            out.append(tuple(PIXELS[a + i][b + j] for i in range(PATCH) for j in range(PATCH)))
    return out


def image_features():
    """Tower output (the patch embedding alone) and projector output, one row per patch."""
    f = [_matvec(W_EMBED, p) for p in patches()]
    rows = []
    for v in f:
        out = _matvec(W_PROJ, v)
        rows.append([x + b for x, b in zip(out, B_PROJ)])
    return f, rows


def spliced_sequence():
    """(ids, source labels, rows): the input embeddings after the placeholder is expanded and overwritten."""
    _, img = image_features()
    ids = [0] + [PROMPT_IDS[0]] * len(img) + list(PROMPT_IDS[1:])
    rows, source, k = [], [], 0
    for t in ids:
        if t == 1:
            rows.append(img[k])
            source.append(f"projector row {k}")
            k += 1
        else:
            rows.append(list(EMBED[t]))
            source.append(f"table row {t}")
    return ids, source, rows


def splice_setup():
    f, img = image_features()
    ps = patches()
    vocab_rows = [(i, w, _vec(EMBED[i], 1)) for i, w in enumerate(VOCAB)]
    pix_rows = [(f"row {r}", *[_fx(v, 1) for v in row]) for r, row in enumerate(PIXELS)]
    patch_rows = [(f"patch {k} (rows {2 * (k // 2)}-{2 * (k // 2) + 1}, columns {2 * (k % 2)}-{2 * (k % 2) + 1})", _vec(p, 1),
                   _vec(f[k], 3), _vec(img[k], 3)) for k, p in enumerate(ps)]
    return ("A decoder with the vocabulary of eight tokens below and width D = 3. Its embedding table E has one row of three numbers "
            "for each token.\n\n" + _table(["id", "token", "row of E"], vocab_rows, "rll")
            + "\n\nThe picture is 4 by 4 pixels, one channel, brightness 0 to 1.\n\n"
            + _table(["", "column 0", "column 1", "column 2", "column 3"], pix_rows, "lrrrr")
            + "\n\nIt is cut into four patches of 2 by 2 pixels, each flattened row by row. The tower is reduced to its patch "
            "embedding (the blocks of Volume VII, Chapter 6 are left out so that the numbers stay printable): a 2 by 4 matrix "
            f"W_v = {[list(r) for r in W_EMBED]} with no bias gives each patch a vector of width D_v = 2. The projector is one "
            f"linear map from 2 to 3 numbers, W_p = {[list(r) for r in W_PROJ]} and b_p = {list(B_PROJ)}, so that "
            "p = W_p f + b_p.\n\n"
            + _table(["patch", "flattened pixels", "tower output f", "projector output p"], patch_rows, "llll"))


def splice_sequence():
    ids, source, rows = spliced_sequence()
    body = []
    for t, (i, s, r) in enumerate(zip(ids, source, rows)):
        body.append((t, i, VOCAB[i] if i != 1 else "<image>", s, _vec(r, 3)))
    return ("The prompt `<image> what is this ?` is tokenized to ids (1, 2, 3, 4, 5), a `<bos>` (id 0) goes in front, and the "
            "single `<image>` is expanded to four copies, one per patch. The embedding gather reads row 1 of E four times; the "
            "splice then overwrites those four rows with the four projector outputs, in order. The decoder reads the right-hand "
            "column.\n\n"
            + _table(["position", "id", "token", "where the row comes from", "row after the splice"], body, "rrllr")
            + "\n\nThe sequence has 9 rows: 5 from the table and 4 from the picture. The values `(0.5, 0.5, 0.5)` of E's row 1 are "
            "gathered and then overwritten, so nothing in the output depends on them.")


def splice_backward():
    ids, _, _ = spliced_sequence()
    n = len(ids)
    img_rows = [t for t, i in enumerate(ids) if i == 1]
    d_img = [GRAD_ROWS[t] for t in img_rows]
    d_res = [list(g) for g in GRAD_ROWS]
    for t in img_rows:
        d_res[t] = [0.0, 0.0, 0.0]
    d_e_plain = [[0.0] * 3 for _ in VOCAB]
    d_e_spliced = [[0.0] * 3 for _ in VOCAB]
    for t in range(n):
        for c in range(3):
            d_e_plain[ids[t]][c] += GRAD_ROWS[t][c]
            d_e_spliced[ids[t]][c] += d_res[t][c]
    f, _ = image_features()
    d_w = [[0.0, 0.0] for _ in range(3)]
    d_b = [0.0] * 3
    for k, g in enumerate(d_img):
        for i in range(3):
            d_b[i] += g[i]
            for j in range(2):
                d_w[i][j] += g[i] * f[k][j]
    rows = [(i, VOCAB[i], _vec(d_e_plain[i], 2), _vec(d_e_spliced[i], 2)) for i in range(len(VOCAB))]
    return ("Suppose the gradient of the loss with respect to the residual-stream row at position t is "
            "g_t = (0.1 t, -0.1 t, 0.05), for t = 0 to 8. The splice routes the gradient of the four picture rows (positions "
            "1 to 4) to the projector output, and zeroes those rows so that the gather's backward pass does not add them to E.\n\n"
            "Gradient of E, row by row, if the picture rows were not zeroed and if they are:\n\n"
            + _table(["id", "token", "without the zeroing", "with it"], rows, "rlll")
            + "\n\nThe gradient that reaches the projector is the four rows "
            + ", ".join(_vec(g, 2) for g in d_img) + f", and with the tower outputs f of the previous table it gives "
            f"dW_p = {[[round(x, 3) for x in r] for r in d_w]} and db_p = {[round(x, 3) for x in d_b]} (the sums over the four "
            "picture rows of g f^T and of g).")


# ------------------------------------------------------------------------------------------------ projectors


def linear_count(n_in, n_out, bias=True):
    return n_in * n_out + (n_out if bias else 0)


def projector_rows():
    """(label, shape text, tokens in, tokens out, parameters, matmul parameters) for each projector."""
    rows = []

    def add(label, shape, tin, tout, params, mm, note):
        rows.append((label, shape, tin, tout, params, mm, note))

    # LLaVA-1.5-13B: two linear layers, 1024 -> 5120 -> 5120
    p = linear_count(1024, 5120) + linear_count(5120, 5120)
    add("LLaVA-1.5-13B, two layers", "1024 -> 5120 -> 5120", 576, 576, p, 1024 * 5120 + 5120 * 5120, "crates/llava")
    # Qwen3-VL-4B: patch merger, LayerNorm(1024) per patch, 2x2 concat to 4096, 4096 -> 4096 -> 2560
    mrg = 4 * 1024
    ln_pre = 2 * 1024
    ln_post = 2 * mrg
    p = ln_pre + linear_count(mrg, mrg) + linear_count(mrg, 2560)
    mm = mrg * mrg + mrg * 2560
    add("Qwen3-VL-4B, main merger", "1024 x 4 -> 4096 -> 2560", 4, 1, p, mm, "crates/qwen3vl")
    p_ds = ln_post + linear_count(mrg, mrg) + linear_count(mrg, 2560)
    add("Qwen3-VL-4B, one DeepStack merger", "1024 x 4 -> 4096 -> 2560", 4, 1, p_ds, mm, "norm over the 4096")
    # Qwen3-VL-30B-A3B: tower 1152, out 2048
    mrg2 = 4 * 1152
    p = 2 * 1152 + linear_count(mrg2, mrg2) + linear_count(mrg2, 2048)
    add("Qwen3-VL-30B-A3B, main merger", "1152 x 4 -> 4608 -> 2048", 4, 1, p, mrg2 * mrg2 + mrg2 * 2048, "crates/qwen3vlmoe")
    # DeepSeek-VL: split aligner, two streams of 1024, each to 2048, concatenated to 4096, then 4096 -> 4096
    p = 2 * linear_count(1024, 2048) + linear_count(4096, 4096)
    add("DeepSeek-VL-7B, split aligner", "(1024, 1024) -> (2048 + 2048) -> 4096", 2, 1, p, 2 * 1024 * 2048 + 4096 * 4096, "crates/deepseekvl")
    # Janus-Pro: understanding aligner 1024 -> 4096 -> 4096
    p = linear_count(1024, 4096) + linear_count(4096, 4096)
    add("Janus-Pro-7B, understanding aligner", "1024 -> 4096 -> 4096", 1, 1, p, 1024 * 4096 + 4096 * 4096, "crates/januspro")
    p = linear_count(8, 4096) + linear_count(4096, 4096)
    add("Janus-Pro-7B, generation aligner", "8 -> 4096 -> 4096", 1, 1, p, 8 * 4096 + 4096 * 4096, "Volume VIII")
    # Moondream 3: connector 2304 -> 8192 -> 2048
    p = linear_count(2304, 8192) + linear_count(8192, 2048)
    add("Moondream 3 preview, connector", "2304 -> 8192 -> 2048", 1, 1, p, 2304 * 8192 + 8192 * 2048, "crates/moondream3")
    # MiniCPM-V 4.6: window merger after block 6 (attention block over 4 patches, then LN(4608), 4608 -> 17216 -> 1152, + mean)
    c, wc, wm = 1152, 4608, 17216
    attn = 2 * c + linear_count(c, 3 * c) + linear_count(c, c)
    p = attn + 2 * wc + linear_count(wc, wm) + linear_count(wm, c)
    add("MiniCPM-V 4.6, window merger (inside the tower)", "1152 x 4 -> 4608 -> 17216 -> 1152", 4, 1, p, 4 * c * c + wc * wm + wm * c,
        "attention block over the 4 patches, then the MLP")
    return rows


def resampler_count(n_q, d, d_v):
    """One cross-attention layer of a resampler: n_q learned queries of width d, W_q d x d, W_k and W_v d_v x d, W_o d x d, no biases."""
    return n_q * d + 2 * d * d + 2 * d_v * d


def projector_table():
    rows = projector_rows()
    body = []
    for label, shape, tin, tout, p, mm, note in rows:
        body.append((label, shape, _int(p), _int(mm), note))
    q = resampler_count(64, 2048, 1152)
    return ("Parameters of the vision-to-language maps of the crates, counted from the tensor shapes of their configurations "
            "(weights and biases, norm gains and offsets included; `matmul` is the weights that multiply a row, the number that "
            "sets the FLOPs per row: 2 per weight). A DeepStack merger differs from the main merger only in its norm, over the "
            "4096 concatenated numbers instead of the 1024 of one patch.\n\n"
            + _table(["projector", "widths", "parameters", "matmul weights", "where"], body, "llrrl")
            + f"\n\nFor comparison, one cross-attention layer of a resampler with 64 learned queries of width 2048 reading "
            f"towers of width 1152 (queries 64 x 2048, W_q and W_o 2048 x 2048, W_k and W_v 1152 x 2048, no biases) has "
            f"{_int(q)} parameters and always returns 64 tokens, whatever the number of patches it reads.")


def projector_flops():
    """(label, rows per image, GFLOPs per image) at the typical token counts of the crates."""
    rows = projector_rows()
    pick = [
        ("LLaVA-1.5-13B, two layers", 576, 576),
        ("Qwen3-VL-4B, main merger", 4096, 1024),     # a 1024 x 1024 picture: 4096 patches, 1024 merged rows
        ("Qwen3-VL-4B, one DeepStack merger", 4096, 1024),
        ("DeepSeek-VL-7B, split aligner", 576, 576),
        ("Moondream 3 preview, connector", 729, 729),
    ]
    body = []
    for label, patches_in, rows_out in pick:
        for lab, shape, tin, tout, p, mm, note in rows:
            if lab == label:
                body.append((label, _int(patches_in), _int(rows_out), _gflops(2 * rows_out * mm, 2)))
    return ("FLOPs of the projector for one picture: 2 FLOPs per matmul weight per row it maps (the merger maps one row per "
            "merged block of patches). The pictures are the usual size of each design, see the next section.\n\n"
            + _table(["projector", "patches from the tower", "rows mapped", "GFLOPs"], body, "lrrr"))


# ------------------------------------------------------------------------------------------------ dynamic resolution


def round_by(x, f):
    return round(x / f) * f              # Python round: half to even


def smart_resize(h, w, factor, min_px, max_px):
    hb = max(factor, round_by(h, factor))
    wb = max(factor, round_by(w, factor))
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


QWEN_PATCH, QWEN_MERGE = 16, 2
QWEN_FACTOR = QWEN_PATCH * QWEN_MERGE
QWEN_MIN = 56 * 56
QWEN_MAX = 3584 * 3584
QWEN_SERVE_MAX = 1024 * 1024
QWEN_CAPTION_MAX = 1280 * 1280


def qwen_tokens(hb, wb):
    return (hb // QWEN_PATCH) * (wb // QWEN_PATCH) // (QWEN_MERGE * QWEN_MERGE)


SIZES = ((448, 448), (480, 640), (1080, 1920), (3000, 2000), (100, 1500), (40, 40), (4000, 6000))


def smart_resize_table():
    rows = []
    for h, w in SIZES:
        a = smart_resize(h, w, QWEN_FACTOR, QWEN_MIN, QWEN_MAX)
        b = smart_resize(h, w, QWEN_FACTOR, QWEN_MIN, QWEN_SERVE_MAX)
        rows.append((f"{h} x {w}", f"{w / h:.3f}", f"{a[0]} x {a[1]}", f"{a[1] / a[0]:.3f}", _int(qwen_tokens(*a)),
                     f"{b[0]} x {b[1]}", _int(qwen_tokens(*b))))
    return ("`smart_resize` with factor 32 (patch 16 times merge 2), minimum area 56 x 56 = 3,136 pixels and maximum area "
            "3584 x 3584 = 12,845,056 pixels (the released limits), and with the maximum 1024 x 1024 = 1,048,576 of brain's served "
            "default. Sizes are height x width in pixels; a token is one 32 x 32 block of the resized picture.\n\n"
            + _table(["picture h x w", "w / h", "resized (released limits)", "w / h after", "tokens", "resized (served limit)",
                      "tokens"], rows, "lrrrrrr"))


def smart_resize_branches():
    """The three branches of the rule on three pictures, with the intermediate numbers."""
    f, lo, hi = QWEN_FACTOR, QWEN_MIN, QWEN_MAX
    rows = []
    for h, w in ((1080, 1920), (4000, 6000), (40, 40)):
        rh, rw = max(f, round_by(h, f)), max(f, round_by(w, f))
        area = rh * rw
        if area > hi:
            beta = math.sqrt(h * w / hi)
            branch = f"{area:,} > {hi:,}: shrink by beta = sqrt({h} {w} / {hi:,}) = {beta:.4f}"
            end = (math.floor(h / beta / f) * f, math.floor(w / beta / f) * f)
        elif area < lo:
            beta = math.sqrt(lo / (h * w))
            branch = f"{area:,} < {lo:,}: grow by beta = sqrt({lo:,} / ({h} {w})) = {beta:.4f}"
            end = (math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f)
        else:
            branch = f"{lo:,} <= {area:,} <= {hi:,}: keep"
            end = (rh, rw)
        rows.append((f"{h} x {w}", f"{h / f:.3f}, {w / f:.3f}", f"{rh} x {rw}", branch, f"{end[0]} x {end[1]}", _int(end[0] * end[1])))
    return ("The three branches of the rule at factor 32 and the released limits: first both sides are rounded to the nearest "
            "multiple of 32 (halves to the even multiple) and at least 32; if the area is then over the maximum both original "
            "sides are divided by beta and rounded down to a multiple of 32, and if it is under the minimum both are multiplied "
            "by beta and rounded up.\n\n"
            + _table(["picture h x w", "h / 32, w / 32", "rounded", "area test", "result", "area"], rows, "lrrlrr"))


def smart_resize_halves():
    """Where the rounding rule breaks ties: sides that are an odd multiple of 16, and the zero a very wide picture reaches."""
    rows = []
    for side in (48, 80, 112, 144, 176):
        rows.append((side, side / 32, round_by(side, 32), math.floor(side / 32) * 32, math.ceil(side / 32) * 32))
    tie = _table(["side", "side / 32", "round half to even", "floor", "ceil"], rows)
    h, w = 31, 600000
    hb = max(QWEN_FACTOR, round_by(h, QWEN_FACTOR))
    wb = max(QWEN_FACTOR, round_by(w, QWEN_FACTOR))
    beta = math.sqrt(h * w / QWEN_MAX)
    hz = math.floor(h / beta / QWEN_FACTOR) * QWEN_FACTOR
    return ("Sides that are exactly half way between two multiples of 32 go to the even multiple (Python's `round`, and the "
            "Rust `round_ties_even` that ports it):\n\n" + tie
            + f"\n\nAnd a picture {h} pixels high and {w:,} wide: rounded it is {hb} x {wb} = {hb * wb:,} pixels, over the maximum, "
            f"so beta = sqrt({h} * {w:,} / {QWEN_MAX:,}) = {beta:.4f} and the height becomes floor({h} / {beta:.4f} / 32) * 32 = {hz}.")


# ------------------------------------------------------------------------------------------------ slicing (MiniCPM-V)

MC_PATCH, MC_SCALE, MC_MAX_SLICES, MC_DIV = 14, 448, 9, 16


def ensure_divide(length, divisor):
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(h, w, scale, patch, allow_upscale):
    if h * w > scale * scale or allow_upscale:
        aspect = w / h
        nh = int(scale / math.sqrt(aspect))
        nw = int(nh * aspect)
        h, w = nh, nw
    return ensure_divide(h, patch * 4), ensure_divide(w, patch * 4)


def refine_size(h, w, grid, scale, patch):
    gy, gx = grid
    rw = ensure_divide(w, gx)
    rh = ensure_divide(h, gy)
    bh, bw = find_best_resize(rh / gy, rw / gx, scale, patch, True)
    return bh * gy, bw * gx


def sliced_grid(h, w, max_slices, scale):
    log_ratio = math.log(w / h)
    ratio = (w * h) / (scale * scale)
    multiple = min(math.ceil(ratio), max_slices)
    if multiple <= 1:
        return None
    best, min_error = (1, 1), float("inf")
    for slices in (multiple - 1, multiple, multiple + 1):
        if slices == 1 or slices > max_slices:
            continue
        for rows in range(1, slices + 1):
            if slices % rows == 0:
                cols = slices // rows
                err = abs(log_ratio - math.log(cols / rows))
                if err < min_error:
                    best, min_error = (rows, cols), err
                elif err == min_error and rows > best[0]:
                    best = (rows, cols)
    return best


def minicpm_views(h, w, mode_div=MC_DIV):
    """(grid, source (h, w), source tokens, slice (h, w) or None, slice tokens, total visual tokens)."""
    grid = sliced_grid(h, w, MC_MAX_SLICES, MC_SCALE)
    sh, sw = find_best_resize(h, w, MC_SCALE, MC_PATCH, grid is None)
    src_tokens = (sh // MC_PATCH) * (sw // MC_PATCH) // mode_div
    if grid is None:
        return None, (sh, sw), src_tokens, None, 0, src_tokens
    rh, rw = refine_size(h, w, grid, MC_SCALE, MC_PATCH)
    ph, pw = rh // grid[0], rw // grid[1]
    sl_tokens = (ph // MC_PATCH) * (pw // MC_PATCH) // mode_div
    return grid, (sh, sw), src_tokens, (ph, pw), sl_tokens, src_tokens + grid[0] * grid[1] * sl_tokens


SLICE_SIZES = ((448, 448), (768, 1024), (1080, 1920), (3000, 2000), (200, 1600), (4000, 6000))


def slicing_table():
    rows = []
    for h, w in SLICE_SIZES:
        g, src, st, sl, slt, tot = minicpm_views(h, w)
        g4 = minicpm_views(h, w, 4)[5]
        rows.append((f"{h} x {w}", "none" if g is None else f"{g[0]} x {g[1]}", f"{src[0]} x {src[1]}", st,
                     "-" if sl is None else f"{sl[0]} x {sl[1]}", slt if sl else "-", tot, g4))
    return ("The image processor of MiniCPM-V 4.6 as `preprocess.rs` states it (patch 14, slice scale 448, at most 9 slices; "
            "Python rounding and truncation): whether the picture is sliced and into how many rows x columns, the source view "
            "(the whole picture resized to a multiple of 56 on each side), the size of one slice, and the visual tokens at the "
            "default 16x downsampling (one token per 16 patches), in total and at 4x (one token per 4 patches).\n\n"
            + _table(["picture h x w", "grid rows x cols", "source view", "source tokens", "one slice", "tokens per slice",
                      "visual tokens (16x)", "visual tokens (4x)"], rows, "lrrrrrrr"))


def slicing_candidates():
    """The candidate grids of one picture and their log-aspect errors."""
    h, w = 1080, 1920
    ratio = (w * h) / (MC_SCALE * MC_SCALE)
    multiple = min(math.ceil(ratio), MC_MAX_SLICES)
    rows = []
    for slices in (multiple - 1, multiple, multiple + 1):
        if slices == 1 or slices > MC_MAX_SLICES:
            continue
        for r in range(1, slices + 1):
            if slices % r == 0:
                c = slices // r
                rows.append((slices, f"{r} x {c}", f"{c / r:.3f}", f"{abs(math.log(w / h) - math.log(c / r)):.4f}"))
    best = sliced_grid(h, w, MC_MAX_SLICES, MC_SCALE)
    return (f"A {h} x {w} picture: area / 448^2 = {ratio:.3f}, so multiple = ceil({ratio:.3f}) = {multiple} and the candidates "
            f"are the slice counts {multiple - 1}, {multiple} and {multiple + 1}. The aspect ratio of the picture is "
            f"{w / h:.3f}, ln = {math.log(w / h):.4f}. The error of a grid is |ln(w / h) - ln(cols / rows)|; the smallest "
            f"wins and a tie goes to more rows.\n\n"
            + _table(["slices", "rows x cols", "cols / rows", "error"], rows) + f"\n\nThe winner is {best[0]} x {best[1]}.")


def window_merger_saving():
    """MiniCPM-V 4.6's tower at a 448 x 448 picture (32 x 32 = 1,024 patches): 27 blocks of width 1152, feed-forward 4304, a
    window merger after block 6 (the 7 blocks 0 to 6 run on 1,024 rows, the other 20 on 256)."""
    c, f, depth, before, rows = 1152, 4304, 27, 7, 1024
    after_rows = rows // 4

    def block(r):
        return 2 * r * (4 * c * c + 2 * c * f), 4 * r * r * c       # weight FLOPs, score and average FLOPs

    w_full, a_full = block(rows)
    w_cut, a_cut = block(after_rows)
    plain_w, plain_a = depth * w_full, depth * a_full
    merged_w, merged_a = before * w_full + (depth - before) * w_cut, before * a_full + (depth - before) * a_cut
    wc, wm = 4 * c, 17216
    wm_attn_w = 2 * rows * 4 * c * c                                    # q, k, v and output projections of its own attention block
    wm_mlp_w = 2 * after_rows * (wc * wm + wm * c)
    wm_attn_a = after_rows * 4 * 4 * 4 * c                              # 256 windows, 4 queries x 4 keys, width c: 4 x 4 x 4 x c each
    extra = wm_attn_w + wm_mlp_w + wm_attn_a
    rows_out = [
        ("no merger: 27 blocks on 1,024 rows", _gflops(plain_w), _gflops(plain_a), _gflops(plain_w + plain_a)),
        ("window merger after block 6: 7 blocks on 1,024 rows, 20 on 256", _gflops(merged_w), _gflops(merged_a), _gflops(merged_w + merged_a)),
        ("the window merger itself", _gflops(wm_attn_w + wm_mlp_w), _gflops(wm_attn_a), _gflops(extra)),
        ("merged tower plus its window merger", _gflops(merged_w + wm_attn_w + wm_mlp_w), _gflops(merged_a + wm_attn_a),
         _gflops(merged_w + merged_a + extra)),
    ]
    saving = 1 - (merged_w + merged_a + extra) / (plain_w + plain_a)
    return ("FLOPs of the MiniCPM-V 4.6 tower for a 448 x 448 picture (1,024 patches of 14 pixels), width 1152, feed-forward 4304. "
            "A block costs 2 x rows x (4 x 1152^2 + 2 x 1152 x 4304) FLOPs for its matrix products and 4 x rows^2 x 1152 for the "
            "score and average products. The window merger runs one attention block over each 2 x 2 window "
            "(projections on 1,024 rows, scores over 4 keys), concatenates the four rows to 4,608 numbers and applies "
            "4608 -> 17216 -> 1152 on 256 rows. Billions of FLOPs:\n\n"
            + _table(["tower", "matrix products", "score and average", "total"], rows_out)
            + f"\n\nThe merged tower, window merger included, needs {100 * (1 - saving):.1f} percent of the FLOPs of the tower "
            f"without it, a saving of {100 * saving:.1f} percent, and its later blocks attend over 256 rows instead of 1,024.")


# ------------------------------------------------------------------------------------------------ overlap crops (Moondream 3)

MD_GRID, MD_PATCH, MD_MARGIN, MD_MAX_CROPS, MD_CROP = 27, 14, 4, 12, 378


def moondream_select_tiling(h, w, crop, max_crops):
    if h <= crop or w <= crop:
        return 1, 1
    min_h, min_w = math.ceil(h / crop), math.ceil(w / crop)
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio))
    ht = max(math.floor(math.sqrt(max_crops * h / w)), min_h)
    wt = max(math.floor(math.sqrt(max_crops * w / h)), min_w)
    if ht * wt > max_crops:
        if wt > ht:
            wt = math.floor(max_crops / ht)
        else:
            ht = math.floor(max_crops / wt)
    return max(1, ht), max(1, wt)


def moondream_plan(h, w):
    stride_px = (MD_GRID - 2 * MD_MARGIN) * MD_PATCH
    margin_px = MD_MARGIN * MD_PATCH
    ht, wt = moondream_select_tiling(max(1, h - 2 * margin_px), max(1, w - 2 * margin_px), stride_px, MD_MAX_CROPS)
    return ht, wt, stride_px * ht + 2 * margin_px, stride_px * wt + 2 * margin_px


MD_SIZES = ((378, 378), (600, 800), (768, 1024), (500, 2000), (3000, 4000))


def overlap_crop_table():
    rows = []
    for h, w in MD_SIZES:
        ht, wt, rh, rw = moondream_plan(h, w)
        oh = (MD_GRID - 2 * MD_MARGIN) * ht + 2 * MD_MARGIN
        ow = (MD_GRID - 2 * MD_MARGIN) * wt + 2 * MD_MARGIN
        passes = 1 + ht * wt
        rows.append((f"{h} x {w}", f"{ht} x {wt}", f"{rh} x {rw}", f"{oh} x {ow}", f"{oh * ow / (MD_GRID * MD_GRID):.2f}", passes,
                     _int(passes * MD_GRID * MD_GRID), 729))
    return ("The overlap multi-crop of Moondream 3 (crop 378 pixels = 27 x 27 patches of 14, margin 4 patches, at most 12 "
            "crops): a crop origin every (27 - 8) x 14 = 266 pixels, a resized picture of 266 x tiles + 112 pixels per side, "
            "local feature maps stitched into (27 - 8) x tiles + 8 patches per side, and one more pass for the global "
            "crop. Whatever the tiling, the stitched map is pooled to 27 x 27, so the decoder reads 729 rows.\n\n"
            + _table(["picture h x w", "tiles rows x cols", "resized to", "stitched patches", "pooling ratio (area)",
                      "tower passes", "patches through the tower", "image tokens"], rows, "lrrrrrrr"))


# ------------------------------------------------------------------------------------------------ token budget and the cost on the decoder

LLAVA_SHAPE = kvc.shape(32000, 5120, 40, 40, 40, 128, 13824, 3)    # Vicuna-1.5-13B (Llama-2-13B): multi-head attention
QWEN4B_SHAPE = kvc.shape(151936, 2560, 36, 32, 8, 128, 9728, 3)    # Qwen3-4B text: 8 key-value heads for 32 query heads
TEXT_TOKENS = 24


def llava_tokens(h, w):
    return 576                                  # resize shortest edge to 336, centre crop 336 x 336, 14-pixel patches


def deepseek_tokens(h, w):
    return 576                                  # padded square: SigLIP-L/16 at 384 and SAM-B at 1024 both give 24 x 24


def moondream_tokens(h, w):
    return 729


def qwen_tokens_for(h, w, max_px=QWEN_MAX):
    return qwen_tokens(*smart_resize(h, w, QWEN_FACTOR, QWEN_MIN, max_px))


def minicpm_tokens(h, w):
    return minicpm_views(h, w)[5]


BUDGET_SIZES = ((336, 336), (448, 448), (480, 640), (1080, 1920), (3000, 2000))


def token_budget():
    rows = []
    for h, w in BUDGET_SIZES:
        rows.append((f"{h} x {w}", _int(llava_tokens(h, w)), _int(moondream_tokens(h, w)), _int(minicpm_tokens(h, w)),
                     _int(qwen_tokens_for(h, w)), _int(qwen_tokens_for(h, w, QWEN_SERVE_MAX))))
    return ("Image tokens that reach the decoder for one picture of the given height x width, under the rule of each design: "
            "a fixed square (LLaVA-1.5, and DeepSeek-VL and Janus-Pro, which also give 576), a fixed count after pooling "
            "(Moondream 3), slices with a source view (MiniCPM-V 4.6 at 16x), and dynamic resolution with the released limits and "
            "with brain's served limit of 1024 x 1024 pixels (Qwen3-VL).\n\n"
            + _table(["picture h x w", "fixed square (LLaVA-1.5)", "fixed after pooling (Moondream 3)", "sliced (MiniCPM-V)",
                      "dynamic (Qwen3-VL), released limits", "dynamic, served limit"], rows))


def merge_ratio_rows():
    rows = []
    for side in (448, 1024, 1920):
        patches_ = (side // QWEN_PATCH) ** 2
        rows.append((side, _int(patches_), _int(patches_ // 4), _int(patches_ // 16)))
    return ("A square picture of the given side through a tower with patches of 16 pixels. Merging each 2 x 2 block of patches "
            "into one token divides the count by 4; merging 4 x 4 blocks divides it by 16.\n\n"
            + _table(["side", "patches", "tokens, 2 x 2 merge", "tokens, 4 x 4 merge"], rows))


def decoder_cost():
    """The marginal cost of the image tokens in a prompt with TEXT_TOKENS text tokens."""
    cases = [
        ("LLaVA-1.5-13B", LLAVA_SHAPE, "336 x 336, fixed", llava_tokens(336, 336)),
        ("Qwen3-VL-4B", QWEN4B_SHAPE, "448 x 448", qwen_tokens_for(448, 448)),
        ("Qwen3-VL-4B", QWEN4B_SHAPE, "1080 x 1920", qwen_tokens_for(1080, 1920)),
        ("Qwen3-VL-4B", QWEN4B_SHAPE, "1080 x 1920, served limit", qwen_tokens_for(1080, 1920, QWEN_SERVE_MAX)),
        ("Qwen3-VL-4B", QWEN4B_SHAPE, "3000 x 2000", qwen_tokens_for(3000, 2000)),
    ]
    rows = []
    for name, s, pic, n in cases:
        base = kvc.flops_new(s, 0, TEXT_TOKENS)
        full = kvc.flops_new(s, 0, TEXT_TOKENS + n)
        bytes_ = kvc.cache_bytes_per_token(s) * n
        rows.append((name, pic, _int(n), _tflops(full - base), _int(kvc.cache_bytes_per_token(s)), _mib(bytes_)))
    return ("What the image tokens add to a prompt of 24 text tokens, with the counting rules of Volume V (Chapter 12): the "
            "prefill FLOPs of the whole prompt minus those of the 24 text tokens alone (weights at 2 FLOPs each, plus the score "
            "and average products over the keys each row sees; the output head counted once in both, so it cancels), and the "
            "cache the image rows occupy, at 2 bytes per stored number. Shapes: LLaVA-1.5's decoder has 40 layers, width 5120, "
            "40 query and 40 key-value heads of width 128, feed-forward width 13824, gated; Qwen3-VL-4B's has 36 layers, width "
            "2560, 32 query and 8 key-value heads of width 128, feed-forward width 9728, gated.\n\n"
            + _table(["decoder", "picture", "image tokens", "prefill TFLOPs added", "cache bytes per token", "cache of the image (MiB)"],
                     rows))


def tower_attention():
    """Score and average products of the Qwen3-VL-4B tower (24 blocks, width 1024) against the patches it attends over."""
    rows = []
    for h, w in ((448, 448), (1080, 1920), (3000, 2000)):
        hb, wb = smart_resize(h, w, QWEN_FACTOR, QWEN_MIN, QWEN_MAX)
        t = (hb // QWEN_PATCH) * (wb // QWEN_PATCH)
        per_block = 4 * t * t * 1024
        rows.append((f"{h} x {w}", _int(t), _int(t // 4), _tflops(24 * per_block)))
    return ("The tower attends over every patch of the picture at once (no windows), 24 blocks of width 1024; one block's score "
            "and average products cost 4 T^2 x 1024 FLOPs for T patches (Volume VII, Chapter 6). The decoder sees T / 4 tokens.\n\n"
            + _table(["picture h x w", "patches T", "tokens T / 4", "tower attention TFLOPs"], rows))


def tokens_series():
    """Data of the figure: image tokens against the side of a square picture for the three designs."""
    sides = [224, 336, 448, 672, 896, 1344, 1792, 2688]
    fixed = [576 for _ in sides]
    tiled = [minicpm_tokens(s, s) for s in sides]
    dyn = [qwen_tokens_for(s, s) for s in sides]
    pooled = [729 for _ in sides]
    return sides, {"fixed square (LLaVA-1.5)": fixed, "fixed after pooling (Moondream 3)": pooled,
                   "sliced (MiniCPM-V 4.6)": tiled, "dynamic (Qwen3-VL)": dyn}


# ------------------------------------------------------------------------------------------------ positions

TOY_PIECES = (("<bos>", 1), ("image", 4), ("what", 1), ("is", 1), ("this", 1), ("?", 1))
TOY_GRID = (1, 2, 2)


def toy_positions():
    ids, _, _ = spliced_sequence()
    trip = pos_mod.mrope_index(TOY_PIECES, TOY_GRID)
    rows = []
    for t, (name, p) in enumerate(trip):
        rows.append((t, name if name != "image" else "image patch", t, f"({p[0]}, {p[1]}, {p[2]})"))
    return ("The spliced sequence of the first example: `<bos>`, four picture rows in a 2 x 2 grid (one frame, so "
            "(t, h, w) = (1, 2, 2)), and four words. One-dimensional RoPE gives each row its index. The multi-axis rule of Volume "
            "V, Chapter 8 gives a word (c, c, c) and advances the counter c by 1, and gives a picture row (c + t, c + h, c + w) "
            "for its frame t, row h and column w, then advances c by max(1, 2, 2) = 2.\n\n"
            + _table(["index", "token", "one-dimensional position", "(t, h, w)"], rows, "llrr"))


def position_advance():
    rows = []
    cases = (("toy picture", (2, 2)), ("LLaVA-1.5 / DeepSeek-VL grid", (24, 24)))
    for label, (h, w) in cases:
        rows.append(_advance_row(label, h, w))
    for h0, w0 in ((448, 448), (1080, 1920), (3000, 2000)):
        hb, wb = smart_resize(h0, w0, QWEN_FACTOR, QWEN_MIN, QWEN_MAX)
        rows.append(_advance_row(f"{h0} x {w0} in Qwen3-VL", hb // QWEN_FACTOR, wb // QWEN_FACTOR))
    return ("How far a picture moves the position counter. A grid of h x w merged tokens takes h w rows of the sequence; "
            "one-dimensional RoPE advances the counter by h w, the multi-axis rule by max(h, w). The text that follows is that "
            "many positions closer to the text before the picture (and the cached rows are still one per token).\n\n"
            + _table(["picture", "grid h x w", "rows in the sequence", "advance, one-dimensional", "advance, multi-axis",
                      "positions saved"], rows, "llrrrr"))


def _advance_row(label, h, w):
    n = h * w
    return (label, f"{h} x {w}", _int(n), _int(n), _int(max(h, w)), _int(n - max(h, w)))


def axis_ownership():
    half = 64
    own = pos_mod.axis_owner((24, 20, 20), half, "interleaved")
    names = "thw"
    count = [own.count(a) for a in range(3)]
    t_ch = [m for m in range(half) if own[m] == 0]
    h_ch = [m for m in range(half) if own[m] == 1]
    w_ch = [m for m in range(half) if own[m] == 2]
    return (f"Qwen3-VL-4B: head width 128, so 64 pairs, sections (t, h, w) = (24, 20, 20) of the interleaved layout. The pairs "
            f"owned by each axis (counts {count[0]}, {count[1]}, {count[2]}):\n\n"
            + _table(["axis", "pairs", "first, last"],
                     [("t", len(t_ch), f"{t_ch[0]} to {t_ch[-1]} (every third, then 60 to 63)"), ("h", len(h_ch), f"{h_ch[0]}, {h_ch[1]}, ... {h_ch[-1]}"),
                      ("w", len(w_ch), f"{w_ch[0]}, {w_ch[1]}, ... {w_ch[-1]}")], "lrl")
            + f"\n\nThe pairs of a pattern `{''.join(names[a] for a in own[:12])}...` rotate through all three axes, so each axis "
            "keeps both the fast and the slow end of the frequency range.")


# ------------------------------------------------------------------------------------------------ location tokens

BINS = 1000


def quantize(x, size):
    return min(BINS - 1, max(0, math.floor(BINS * x / size)))


BOX = (100.0, 60.0, 340.0, 420.0)           # x1, y1, x2, y2 in pixels
IMG_W, IMG_H = 640, 480


def box_tokens(box=BOX, w=IMG_W, h=IMG_H):
    bins = [quantize(box[0], w), quantize(box[1], h), quantize(box[2], w), quantize(box[3], h)]
    return bins


def location_table():
    bins = box_tokens()
    sizes = (IMG_W, IMG_H, IMG_W, IMG_H)
    names = ("x1", "y1", "x2", "y2")
    rows = []
    for n, v, s, b in zip(names, BOX, sizes, bins):
        low = b / BINS * s
        mid = (b + 0.5) / BINS * s
        rows.append((n, f"{v:g}", f"{BINS} x {v:g} / {s} = {BINS * v / s:.3f}", b, f"<loc_{b}>", f"{low:.2f}", f"{abs(low - v):.2f}",
                     f"{mid:.2f}", f"{abs(mid - v):.2f}"))
    text = "cat" + "".join(f"<loc_{b}>" for b in bins)
    return (f"A box in a {IMG_W} x {IMG_H} picture: corners (x1, y1) = ({BOX[0]:g}, {BOX[1]:g}) and (x2, y2) = ({BOX[2]:g}, "
            f"{BOX[3]:g}) pixels. Each coordinate is divided by the side it belongs to, multiplied by {BINS} and rounded down "
            f"(at most {BINS - 1}); the four bins become four tokens after the phrase. Decoded as bin / {BINS} of the side "
            "(the lower edge of the bin) or as (bin + 0.5) / 1000 (its centre):\n\n"
            + _table(["coordinate", "pixels", "1000 x pixels / side", "bin", "token", "decoded, lower edge", "error",
                      "decoded, centre", "error"], rows, "lrrrlrrrr")
            + f"\n\nThe text the decoder writes is `{text}`.")


def parse_boxes(text):
    """Pull `phrase<loc_a><loc_b><loc_c><loc_d>` groups out of text; a trailing run of fewer than four tags is dropped."""
    out, pos, bins, phrase_start, run_start = [], 0, [], 0, 0
    while True:
        i = text.find("<loc_", pos)
        if i < 0:
            break
        j = text.find(">", i)
        digits = text[i + 5:j] if j >= 0 else ""
        if not digits.isdigit():
            pos = i + 5
            continue
        if not bins:
            run_start = i
        bins.append(int(digits))
        pos = j + 1
        if len(bins) == 4:
            out.append((text[phrase_start:run_start].strip(), [b / BINS for b in bins]))
            bins = []
            phrase_start = pos
    return out


def parse_example():
    text = "a cat<loc_156><loc_125><loc_531><loc_875>the dog<loc_10><loc_20><loc_30><loc_40><loc_7><loc_8>"
    rows = [(p, _vec(b, 3)) for p, b in parse_boxes(text)]
    return (f"The text `{text}` has two complete groups of four tags and a trailing pair, which is dropped (a generation cut "
            "short in the middle of a box).\n\n" + _table(["phrase", "[x1, y1, x2, y2] as fractions of the sides"], rows, "ll"))


def bin_width_table():
    rows = []
    for side in (384, 640, 768, 1920, 4000):
        rows.append((side, f"{side / BINS:.3f}", f"{side / BINS / 2:.3f}"))
    return ("The width of one bin in pixels on a side of the given length, and the largest error of the centre decoding (half a "
            "bin; the lower-edge decoding errs by up to a whole bin).\n\n" + _table(["side (pixels)", "pixels per bin", "centre decoding, largest error"], rows))


def florence_tokens():
    return ("The tokens of Florence-2-base: the picture is resized to 768 x 768 and cut by the tower into a 24 x 24 grid "
            f"({768 // 32} per side, stride 32), so 576 tokens of width 1024; one more token is their average; a projection "
            "maps each to width 768. The 577 vision tokens come before the prompt tokens in the encoder input. The vocabulary is "
            "50,265 + 1,024 = 51,289 ids: the base byte-level BPE, 24 task markers and the 1,000 location tokens. "
            f"Parameters of the projection: 1024 x 768 + 2 x 768 = {_int(1024 * 768 + 2 * 768)}.")


# ------------------------------------------------------------------------------------------------ the tower's attention slab and the bounds story


def tower_slab():
    """Bytes of one head-by-query-by-key score slab of the Qwen3-VL-4B tower (16 heads) in 32-bit numbers, against the pixel budget."""
    rows = []
    for name, budget in (("served default 1024 x 1024", QWEN_SERVE_MAX), ("captioning default 1280 x 1280", QWEN_CAPTION_MAX),
                         ("released limit 3584 x 3584", QWEN_MAX)):
        side = int(math.sqrt(budget))
        hb, wb = smart_resize(side, side, QWEN_FACTOR, QWEN_MIN, budget)
        t = (hb // QWEN_PATCH) * (wb // QWEN_PATCH)
        slab = 16 * t * t * 4
        chunk = min(4096, max(64, (256 * 2 ** 20) // (16 * t * 4)))
        rows.append((name, f"{hb} x {wb}", _int(t), _int(t // 4), f"{slab / 2 ** 30:,.2f}", _int(chunk), _int(-(-t // chunk))))
    return ("A square picture at each pixel budget through the Qwen3-VL-4B tower: the resized side, the patches T the tower "
            "attends over, the tokens T / 4 the decoder reads, one slab of scores `[16 heads, T queries, T keys]` in 32-bit "
            "numbers (GiB), and the query rows per chunk that keeps one slab within 256 MiB (the chunk is 256 MiB / (16 x T x 4 bytes), at "
            "least 64 and at most 4,096 rows) with the number of chunks that makes.\n\n"
            + _table(["budget", "resized", "patches T", "decoder tokens", "score slab (GiB)", "query rows per chunk", "chunks"], rows,
                     "llrrrrr"))


def bounds_story():
    old = (256 * 256, 4096 * 4096)
    new = (QWEN_MIN, QWEN_MAX)
    rows = []
    for h, w in ((503, 697), (100, 100), (200, 200), (3500, 4000), (4096, 4096)):
        a = smart_resize(h, w, QWEN_FACTOR, *old)
        b = smart_resize(h, w, QWEN_FACTOR, *new)
        rows.append((f"{h} x {w}", f"{a[0]} x {a[1]}", _int(qwen_tokens(*a)), f"{b[0]} x {b[1]}", _int(qwen_tokens(*b))))
    return ("The same rule with two pairs of limits: the constants the crate carried before commit `70665529a` (minimum area "
            "256 x 256, maximum 4096 x 4096) and the released ones (56 x 56 and 3584 x 3584). Pictures in the wide middle of "
            "the range cannot tell them apart.\n\n"
            + _table(["picture h x w", "resized, old limits", "tokens", "resized, released limits", "tokens"], rows, "lrrrr"))


# ------------------------------------------------------------------------------------------------ the nine crates side by side

# (crate, tower, native input, rows into the projector, projector, tokens to the decoder, decoder)
def _crate_rows():
    def sq(side, patch):
        return (side // patch) ** 2
    return [
        ("llava", "CLIP-L/14, 24 x 1024, tapped at the penultimate block, class token dropped", "336 x 336", _int(sq(336, 14)),
         "two layers 1024 -> 5120 -> 5120", sq(336, 14), "Vicuna-1.5-13B, 40 layers, width 5120, no grouping"),
        ("qwen3vl", "ViT 24 x 1024, two-axis RoPE, DeepStack taps after blocks 5, 11, 17 (from 0)", "448 x 448", _int(sq(448, 16)),
         "merge 2 x 2, then 4096 -> 4096 -> 2560", sq(448, 16) // 4, "Qwen3 4B preset, 36 layers, width 2560, 32 / 8 heads"),
        ("qwen3vlmoe", "the same tower at 27 x 1152, taps after blocks 8, 16, 24 (from 0)", "448 x 448", _int(sq(448, 16)),
         "merge 2 x 2, then 4608 -> 4608 -> 2048", sq(448, 16) // 4, "48-layer sparse decoder, 128 experts, 8 per token"),
        ("minicpmv", "NaViT SigLIP 27 x 1152, window merger after block 6", "448 x 448", f"{_int(sq(448, 14) // 4)} (of {_int(sq(448, 14))} patches)",
         "merge 2 x 2 (MLP), after the window merger", sq(448, 14) // 16, "Qwen3.5-0.8B, hybrid layers"),
        ("moondream3", "SigLIP 27 x 1152 on 378-pixel crops", "378 x 378", _int(sq(378, 14)),
         "global and pooled local maps joined, 2304 -> 8192 -> 2048", sq(378, 14), "24 layers, width 2048, sparse from layer 4"),
        ("deepseekvl", "SAM-B at 1024 and SigLIP-L at 384, 24 x 24 each", "1024 and 384", f"2 x {_int(sq(384, 16))}",
         "split: (1024, 1024) -> 2 x 2048 -> 4096", sq(384, 16), "Llama, 30 layers, width 4096"),
        ("januspro", "SigLIP-L/16 at 384", "384 x 384", _int(sq(384, 16)), "two layers 1024 -> 4096 -> 4096", sq(384, 16),
         "the same Llama decoder"),
        ("florence2", "DaViT, four stages, 24 x 24 grid at 768", "768 x 768", _int(sq(768, 32)), "one linear 1024 -> 768 and a norm",
         sq(768, 32) + 1, "BART encoder-decoder, 6 + 6 layers, width 768"),
    ]


def crate_towers():
    body = [(c, t, inp, rows, proj, _int(tok), dec) for c, t, inp, rows, proj, tok, dec in _crate_rows()]
    return ("The tower, the projector and the token count at each model's native input, from the configurations and constructors "
            "of the crates. `Rows into the projector` is what the tower hands on (DeepSeek-VL's two towers each give 576, which "
            "the split aligner joins row by row; MiniCPM-V's window merger has already cut its 1,024 patches to 256); the "
            "last but one column is what the decoder reads for one picture (the encoder, for Florence-2, whose 577 is the 576 "
            "grid cells and one pooled token).\n\n"
            + _table(["crate", "tower", "native input", "rows into the projector", "projector", "image tokens", "decoder"],
                     body, "lllrlrl"))


def crate_entry():
    rows = [
        ("llava", "shortest edge to 336, centre crop", "id `-200`, expanded to 576 rows", "one-dimensional"),
        ("qwen3vl", "dynamic: multiples of 32, area 56^2 to 3584^2", "`vision_start`, N pad ids, `vision_end`",
         "multi-axis, sections 24, 20, 20"),
        ("qwen3vlmoe", "as qwen3vl", "as qwen3vl (host-side gather, then row splice)", "multi-axis, sections 24, 20, 20"),
        ("minicpmv", "source view and up to 9 slices of about 448", "`image` marker, pad ids, then a `slice` marker and pad run per slice",
         "one-dimensional"),
        ("moondream3", "global crop and up to 12 overlapping crops, pooled to 27 x 27", "`bos`, then 729 rows as a prefix attended bidirectionally",
         "one-dimensional, 32 of 64 channels rotated"),
        ("deepseekvl", "pad to a square, 1024 and 384", "`image_placeholder` expanded to 576 pad ids", "one-dimensional"),
        ("januspro", "pad to a square, 384", "`begin_of_image`, 576 pad ids, `end_of_image`", "one-dimensional"),
        ("florence2", "direct resize to 768 x 768", "no placeholder: the rows are the first of the encoder input",
         "two learned axes on the rows, learned absolute positions in the encoder"),
    ]
    return ("How each crate fits a picture to its tower, how the rows enter the sequence and what position they carry.\n\n"
            + _table(["crate", "resolution policy", "how the rows enter", "positions of the rows"], rows, "llll"))
