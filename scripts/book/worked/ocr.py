# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Reading Documents": how a model reads a page of text from pixels, and how few tokens a page can
be squeezed into.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: ocr.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. Earlier
modules are imported, never copied: the output size of a convolution (`convolution.out_size`), the cache of a decoder shape
(`kv_cache`), the decoder shape and latent widths of the attention chapter (`attention_variants`) and the image-token rules of the
previous chapter (`vlms`).

* the quantities of a page: what a stroke becomes after a resize, and what the three readings of a page (one small view, one
  global view, a mosaic of tiles and a global view) do to the size of the text;
* the DeepEncoder as arithmetic: grids after each stage, attention pairs of windowed and global blocks, FLOPs by stage;
* the layout of the image block in the decoder's sequence (rows, newlines, separator), the tile-grid policy and the rows of a page;
* the learned-query encoder of the second generation: its mask, its rows and its FLOPs;
* the decoder behind it: parameters, active parameters, cache per token and per page;
* serving: the order in which pages are encoded, the pages a cache pool holds, where the two halves are placed, and the
  anti-repetition filter;
* the word error rate as an edit distance.

Everything is deterministic arithmetic on printed constants: no random number is drawn, so no generator or seed is involved. Sums
of products run left to right. FLOPs count 2 per multiply-add; the layer normalizations, activations, biases and the relative-position
bias of the SAM blocks are left out of every count, and each table says so.

Swedish Embedded AB implements solutions for document-reading models whose token budget is counted before it is paid. If your team
needs expertise in document OCR serving, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math

import attention_variants as av
import convolution as cv
import kv_cache as kvc
import vlms

# ------------------------------------------------------------------------------------------------ formatting


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


def _gflops(v, d=1):
    return f"{v / 1e9:,.{d}f}"


def _tflops(v, d=2):
    return f"{v / 1e12:,.{d}f}"


def _mib(b, d=1):
    return f"{b / 2 ** 20:,.{d}f}"


# ------------------------------------------------------------------------------------------------ the shapes of the two generations

SAM = {"patch": 16, "d": 768, "layers": 12, "ffn": 3072, "window": 14, "global_blocks": (2, 5, 8, 11),
       "neck": 256, "mid": 512, "out": 1024}
CLIP = {"d": 1024, "layers": 24, "ffn": 4096}
DEC = {"V": 129280, "D": 1280, "L": 12, "H": 10, "dh": 128, "dense": 1, "dense_ffn": 6848, "experts": 64, "topk": 6,
       "moe_ffn": 896, "shared": 2}
V2 = {"d": 896, "layers": 24, "heads": 14, "kv_heads": 2, "head": 64, "ffn": 4864, "q_local": 144, "q_global": 256, "sam_out": 896}

GLOBAL_SIDE = 1024
V1_TILE, V1_MAX_TILES = 640, 9
V2_TILE, V2_MAX_TILES = 768, 6
INSTRUCTION_TOKENS = 9      # "\n<|grounding|>Convert the document to markdown.", as the reference documentation counts it


def conv_side(n, k=3, s=2, p=1):
    return cv.out_size(n, k, s, p)


def token_side(side_px):
    """Tokens per side of a view of side_px pixels after the patch embedding and the two stride-2 convolutions."""
    return conv_side(conv_side(side_px // SAM["patch"]))


# ------------------------------------------------------------------------------------------------ what a resize does to a stroke


def _on_length(x, period):
    """Length of the 'ink' of a square wave (ink on the first half of every period) in [0, x]."""
    full, rest = divmod(x, period)
    return full * period / 2 + min(rest, period / 2)


def box_row(period, scale, n_out=200):
    """Row of area averages: each output pixel averages the 1 / scale input pixels it covers of a square wave of the given period."""
    out = []
    for k in range(n_out):
        a, b = k / scale, (k + 1) / scale
        out.append((_on_length(b, period) - _on_length(a, period)) * scale)
    return out


PAGE_W, PAGE_H = 1654, 2339          # A4 at 200 pixels per inch
PAGE_300 = (2480, 3508)              # A4 at 300 pixels per inch
STEMS = (4.0, 7.5)                   # periods, in pixels, of a run of stems and of letters of 15-pixel text


def legibility_views():
    """(name, scale) of the text on the 1654 x 2339 page in each reading."""
    return [
        ("native pixels", 1.0),
        ("v2 tile mosaic, across", V2_TILE * 2 / PAGE_W),
        ("v2 tile mosaic, down", V2_TILE * 3 / PAGE_H),
        ("v1 tile mosaic, across", V1_TILE * 2 / PAGE_W),
        ("v1 tile mosaic, down", V1_TILE * 3 / PAGE_H),
        ("global view, 1654 x 2339", GLOBAL_SIDE / PAGE_H),
        ("global view, 2480 x 3508", GLOBAL_SIDE / PAGE_300[1]),
    ]


def legibility():
    rows = []
    for name, s in legibility_views():
        cells = [name, f"{s:.4f}", _fx(1 / s, 3)]
        for p in STEMS:
            row = box_row(p, s)
            period = p * s
            cells += [_fx(period, 2), _fx(max(row) - min(row), 3), "yes" if period > 2 else "no"]
        rows.append(cells)
    return ("A row of stems (ink for half of every period, ink = 1) is averaged over the 1 / scale input pixels that each output pixel "
            "covers (an exact area average, not the antialiased bicubic of the crates), and 200 output pixels are taken, "
            "output pixel k covering the input interval from k / scale to (k + 1) / scale. 'Period out' is the period times the "
            "scale; the stems can only be told apart when it exceeds 2 output pixels (the Nyquist limit of Volume I, Chapter 14). 'Swing' is "
            "the largest output minus the smallest.\n\n"
            + _table(["text seen at", "scale", "box width", "period out, 4 px", "swing", "above 2", "period out, 7.5 px", "swing", "above 2"],
                     rows))


# ------------------------------------------------------------------------------------------------ the grids of the DeepEncoder


def grids_data():
    """(name, side, patches per side, side after conv 1, tokens per side, tokens, windows per side, window padding)."""
    out = []
    for name, side in (("global view", 1024), ("v1 tile", 640), ("v2 tile", 768)):
        n = side // SAM["patch"]
        n1 = conv_side(n)
        n2 = conv_side(n1)
        wn = -(-n // SAM["window"])
        out.append((name, side, n, n1, n2, n2 * n2, wn, wn * SAM["window"]))
    return out


def grids():
    rows = []
    for name, side, n, n1, n2, tokens, wn, padded in grids_data():
        rows.append((name, f"{side} x {side}", f"{n} x {n} = {_int(n * n)}", f"{n1} x {n1}", f"{n2} x {n2} = {_int(tokens)}",
                     f"{wn} x {wn} of 14 x 14 (padded to {padded})"))
    return ("Patches of 16 pixels (a convolution of kernel 16 and stride 16), then two convolutions of kernel 3, stride 2, padding 1, "
            "whose output side is `out_size` of Volume VII, Chapter 2: (n + 2 - 3) // 2 + 1. The last column is the window "
            "partition of the SAM blocks, which pads the patch grid on the bottom and the right to a multiple of 14.\n\n"
            + _table(["view", "pixels", "patch grid", "after conv 1", "after conv 2 (tokens)", "windows of a windowed block"], rows,
                     "lllllll"[:6]))


def attention_pairs_data(n):
    """Query-key pairs of one windowed and one global SAM block on an n x n patch grid: a windowed block attends inside each of the
    ceil(n / 14)^2 windows of the padded grid (14 x 14 queries and keys each), a global block over the n x n real tokens."""
    wn = -(-n // SAM["window"])
    return wn * wn * SAM["window"] ** 4, (n * n) ** 2


def attention_pairs():
    rows = []
    for name, side, n, *_ in grids_data():
        w, g = attention_pairs_data(n)
        n_g = len(SAM["global_blocks"])
        n_w = SAM["layers"] - n_g
        total = n_w * w + n_g * g
        all_global = SAM["layers"] * g
        rows.append((name, _int(n * n), _int(w), _int(g), _fx(g / w, 1), _int(total), _int(all_global), _fx(100 * total / all_global, 1)))
    return ("Pairs of one query and one key in the attention of a SAM block. A windowed block's pairs are the windows of the padded "
            "grid times `14^2 x 14^2` (padding rows included, as in Chapter 7); a global block's are `tokens x tokens`. The tower has "
            "8 windowed blocks and 4 global ones (blocks 2, 5, 8 and 11), and the last two columns compare its total with a tower of "
            "12 global blocks.\n\n"
            + _table(["view", "patches", "pairs, windowed", "pairs, global", "global / windowed", "pairs, SAM tower", "pairs, 12 global",
                      "share (%)"], rows))


# ------------------------------------------------------------------------------------------------ FLOPs by stage


def sam_parts(n):
    """FLOPs of the SAM tower and neck on an n x n patch grid, by the counting rule of Chapter 7: the query, key, value and output
    products of a windowed block run over the padded grid, its feed-forward over the real tokens, its attention over the windows;
    a global block runs everything over the real tokens (2 per multiply-add; norms, activations, biases and the relative-position
    bias are not counted)."""
    d, ffn, m = SAM["d"], SAM["ffn"], SAM["window"]
    t = n * n
    padded = -(-n // m) * m
    tp = padded * padded
    parts = {"patch": 2 * t * SAM["patch"] ** 2 * 3 * d, "qkv": 0, "attn": 0, "proj": 0, "mlp": 0}
    for l in range(SAM["layers"]):
        glob = l in SAM["global_blocks"]
        rows = t if glob else tp
        parts["qkv"] += 2 * rows * d * 3 * d
        parts["attn"] += 4 * d * (t * t if glob else (padded // m) ** 2 * (m * m) ** 2)
        parts["proj"] += 2 * rows * d * d
        parts["mlp"] += 2 * 2 * t * d * ffn
    parts["neck"] = 2 * t * d * SAM["neck"] + 2 * t * SAM["neck"] ** 2 * 9
    return parts


def compressor_flops(n, out):
    n1 = conv_side(n)
    n2 = conv_side(n1)
    return 2 * n1 * n1 * 9 * SAM["neck"] * SAM["mid"] + 2 * n2 * n2 * 9 * SAM["mid"] * out


def sam_flops(n):
    p = sam_parts(n)
    return {"patch embedding": p["patch"], "SAM blocks, products": p["qkv"] + p["proj"] + p["mlp"],
            "SAM blocks, attention": p["attn"], "neck": p["neck"], "compressor": compressor_flops(n, SAM["out"])}


def clip_flops(tokens):
    d, ffn = CLIP["d"], CLIP["ffn"]
    T = tokens + 1
    return CLIP["layers"] * (2 * T * (4 * d * d + 2 * d * ffn) + 4 * d * T * T)


def projector_flops_v1(tokens):
    return 2 * tokens * (CLIP["d"] + SAM["out"]) * DEC["D"]


def prefix_pairs(p):
    """Allowed (query, key) pairs of the prefix mask on 2p rows whose first p rows are the prefix."""
    return p * p + sum(i + 1 for i in range(p, 2 * p))


def resampler_flops(tokens):
    d, ffn, kv = V2["d"], V2["ffn"], V2["kv_heads"] * V2["head"]
    T = 2 * tokens
    layer = 2 * T * (2 * d * d + 2 * d * kv + 3 * d * ffn) + 4 * d * prefix_pairs(tokens)
    return V2["layers"] * layer


def projector_flops_v2(tokens):
    return 2 * tokens * V2["d"] * DEC["D"]


def v2_sam_flops(n):
    """The SAM part of the second generation: the compressor ends in 896 channels."""
    f = sam_flops(n)
    f["compressor"] = compressor_flops(n, V2["sam_out"])
    return f


def view_flops_v1(side):
    n = side // SAM["patch"]
    tokens = token_side(side) ** 2
    f = sam_flops(n)
    f["CLIP-L"] = clip_flops(tokens)
    f["projector"] = projector_flops_v1(tokens)
    return f


def view_flops_v2(side):
    n = side // SAM["patch"]
    tokens = token_side(side) ** 2
    f = v2_sam_flops(n)
    f["resampler"] = resampler_flops(tokens)
    f["projector"] = projector_flops_v2(tokens)
    return f


def encoder_flops():
    cols = [("v1, 1024", view_flops_v1(1024)), ("v1, 640", view_flops_v1(640)), ("v2, 1024", view_flops_v2(1024)),
            ("v2, 768", view_flops_v2(768))]
    stages = ["patch embedding", "SAM blocks, products", "SAM blocks, attention", "neck", "compressor", "CLIP-L", "resampler", "projector"]
    rows = []
    for st in stages:
        rows.append([st] + [(_gflops(f[st]) if st in f else "-") for _, f in cols])
    rows.append(["total"] + [_gflops(sum(f.values())) for _, f in cols])
    return ("GFLOPs of the vision half of one view (2 FLOPs per multiply-add; layer norms, activations, biases and the SAM "
            "relative-position bias left out). SAM, counted as in Chapter 7: patch embedding of 768 outputs per 16 x 16 x 3 patch; "
            "the query, key, value and output products of a windowed block over the padded grid and of a global block over the "
            "real tokens, the feed-forward over the real tokens (d = 768, f = 3072), and `4 d` per attention pair "
            "(the pairs of the previous table); the neck is a 1 x 1 convolution 768 to 256 and a 3 x 3 convolution 256 to 256; "
            "the compressor is a 3 x 3 convolution 256 to 512 and a 3 x 3 convolution 512 to 1024 (896 in v2) at stride 2. "
            "CLIP-L: 24 blocks of width 1024 and 4096 over the tokens and a class row. Projector v1: 2048 to 1280. "
            "Resampler: 24 blocks of width 896, 14 heads of 64, 2 key-value heads, feed-forward 4864, over twice the tokens "
            "(the tokens and as many queries), attention pairs of the prefix mask, projector 896 to 1280. The width of the "
            "first convolution of the compressor in v2 (512) is assumed equal to v1's.\n\n"
            + _table(["stage", "v1, 1024 px", "v1, 640 px", "v2, 1024 px", "v2, 768 px"], rows))


def encoder_total(version, side):
    return sum((view_flops_v1 if version == 1 else view_flops_v2)(side).values())


def native_patches():
    rows = []
    base = 64 ** 4
    for w, h in ((PAGE_W, PAGE_H), PAGE_300):
        gw, gh = w // SAM["patch"], h // SAM["patch"]
        t = gw * gh
        rows.append((f"{w} x {h}", f"{gw} x {gh}", _int(t), _int(t * t), _fx(t * t / base, 1)))
    return ("The patch grid of a page at its own pixels (whole patches of 16 pixels), the query-key pairs of one global block over "
            "it, and that count against the 4,096 x 4,096 = 16,777,216 pairs of a global block of the 64 x 64 grid of the global "
            "view.\n\n" + _table(["page", "patch grid", "patches", "pairs, one global block", "against the 64 x 64 grid"], rows))


def compression_payoff():
    """The same global view with and without the compressor: tokens into CLIP and into the decoder."""
    rows = []
    for name, t, r in (("patches straight on (hypothetical)", 4096, 4096), ("after the 16x compressor", 256, 273)):
        n = prompt_tokens(r)
        rows.append((name, _int(t), _int((t + 1) ** 2), _tflops(clip_flops(t)), _int(n), _tflops(prefill_flops(n)), _mib(n * kv_per_token())))
    return ("The global view of the first generation, with its 4,096 patch tokens given to CLIP-L and then to the decoder as they are "
            "(a count, not a model that exists: its position table and weights would have to be trained for it) and as the compressor "
            "gives them. CLIP's attention pairs are those of one block over the tokens and the class row; the decoder's rows are "
            "the prompt (BOS, the image rows, the 9 tokens of the instruction).\n\n"
            + _table(["variant", "tokens", "pairs, one CLIP block", "CLIP TFLOPs", "prompt tokens", "decoder prefill TFLOPs", "cache (MiB)"], rows))


# ------------------------------------------------------------------------------------------------ the image block in the sequence

NEWLINE, SEPARATOR = "N", "S"


def row_plan(g, t, cols, rows):
    """The image block of one page: g tokens per side of the global view, t per side of a tile, cols x rows tiles.

    Entries are ('P', view, token) for a projector row (view 0 the global view, view 1 + k the k-th tile in row-major order),
    'N' for the learned newline row and 'S' for the learned separator row."""
    out = []
    for ty in range(rows):
        for y in range(t):
            for tx in range(cols):
                view = 1 + ty * cols + tx
                out.extend(("P", view, y * t + x) for x in range(t))
            out.append(NEWLINE)
    for y in range(g):
        out.extend(("P", 0, y * g + x) for x in range(g))
        out.append(NEWLINE)
    out.append(SEPARATOR)
    return out


def row_count(g, t, cols, rows):
    return g * (g + 1) + 1 + rows * (t * cols + 1) * t


def projector_index(entry, g, t):
    """Index of a projector row in the concatenation of the views (global view first, then the tiles)."""
    _, view, token = entry
    return token if view == 0 else g * g + (view - 1) * t * t + token


def toy_plan():
    return row_plan(3, 2, 2, 1)


def toy_layout():
    g, t = 3, 2
    plan = toy_plan()
    rows = []
    for i, e in enumerate(plan):
        if isinstance(e, tuple):
            _, view, token = e
            what = "global view" if view == 0 else f"tile {view - 1}"
            rows.append((i, "projector row", projector_index(e, g, t), what, token))
        else:
            rows.append((i, "newline" if e == NEWLINE else "separator", "-", "-", "-"))
    assert len(plan) == row_count(g, t, 2, 1)
    return ("A page read as a mosaic of 2 tiles in one row (2 columns, 1 row), each tile 2 x 2 tokens, plus a global view of "
            "3 x 3 tokens. The projector rows are numbered as the encoder produces them: the 9 rows of the global view first "
            "(0 to 8), then 4 rows for tile 0 (9 to 12) and 4 for tile 1 (13 to 16). The image block has 3 x (3 + 1) + 1 = 13 "
            f"rows for the global view and (2 x 2 + 1) x 2 = 10 for the mosaic, {len(plan)} in all.\n\n"
            + _table(["row of the block", "what stands there", "projector row", "from", "token of that view"], rows))


def toy_runs():
    """Contiguous runs of projector rows: (row of the block, projector row, length)."""
    g, t = 3, 2
    runs = []
    for i, e in enumerate(toy_plan()):
        if not isinstance(e, tuple):
            continue
        p = projector_index(e, g, t)
        if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == p:
            runs[-1][2] += 1
        else:
            runs.append([i, p, 1])
    return runs


def toy_runs_text():
    runs = toy_runs()
    return ("Contiguous runs of the toy block, written (row of the block, first projector row, length): "
            + ", ".join(f"({a}, {b}, {c})" for a, b, c in runs) + f". There are {len(runs)} of them: a newline breaks a run on the block's side and a change of tile breaks it on the "
            "encoder's.")


def block_rows_table():
    g, t = 16, 10
    rows = []
    for cols, rws in ((0, 0), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (9, 1), (1, 9)):
        if cols == 0:
            n = row_count(g, t, 0, 0)
            rows.append(("none (global view only)", "0", _int(g * g), _int(n)))
        else:
            n = row_count(g, t, cols, rws)
            rows.append((f"{cols} x {rws}", str(cols * rws), _int(g * g + cols * rws * t * t), _int(n)))
    return ("Rows of the image block of the first generation (global view 16 x 16 tokens, tile 10 x 10) for several tile grids, "
            "written columns x rows. The rows are `16 x 17 + 1 = 273` for the global view plus `rows x (10 x columns + 1) x 10` for "
            "the mosaic: a mosaic with more tile rows has more newlines. The second column is the number of tiles and the third "
            "the projector rows the encoder produces.\n\n"
            + _table(["tile grid", "tiles", "projector rows", "rows of the block"], rows))


# ------------------------------------------------------------------------------------------------ the tile-grid policy


def tile_grid_candidates(w, h, tile, max_tiles, min_tiles=2):
    """(i, j, |w / h - i / j|) for every grid i x j with min_tiles <= i j <= max_tiles, in the order (i j, i, j)."""
    aspect = w / h
    cands = [(i, j) for i in range(1, max_tiles + 1) for j in range(1, max_tiles + 1) if min_tiles <= i * j <= max_tiles]
    cands.sort(key=lambda c: (c[0] * c[1], c[0], c[1]))
    return [(i, j, abs(aspect - i / j)) for i, j in cands]


def internvl_grid(w, h, tile, max_tiles, min_tiles=2):
    area = w * h
    best, best_diff = (1, 1), math.inf
    for i, j, diff in tile_grid_candidates(w, h, tile, max_tiles, min_tiles):
        if diff < best_diff:
            best, best_diff = (i, j), diff
        elif diff == best_diff and area > 0.5 * tile * tile * (i * j):
            best = (i, j)
    return best


def tile_grid(w, h, tile, max_tiles):
    """None when the page fits one tile; otherwise (columns, rows) of the closest grid of at least two tiles."""
    if w <= tile and h <= tile:
        return None
    return internvl_grid(w, h, tile, max(max_tiles, 2))


def reading(w, h, version=1, max_tiles=None):
    """('small' | 'global' | 'tiles', columns, rows)."""
    tile = V1_TILE if version == 1 else V2_TILE
    cap = V1_MAX_TILES if version == 1 else V2_MAX_TILES
    if max_tiles == 0:
        return ("global", 0, 0)
    grid = tile_grid(w, h, tile, cap if max_tiles is None else max_tiles)
    return ("small", 0, 0) if grid is None else ("tiles", grid[0], grid[1])


def block_rows(w, h, version=1, max_tiles=None):
    kind, c, r = reading(w, h, version, max_tiles)
    if version == 1:
        g = token_side(GLOBAL_SIDE)
        t = token_side(V1_TILE)
        if kind == "small":
            return t * (t + 1) + 1
        if kind == "global":
            return g * (g + 1) + 1
        return row_count(g, t, c, r)
    if kind == "small":
        return V2["q_local"] + 1
    if kind == "global":
        return V2["q_global"] + 1
    return c * r * V2["q_local"] + V2["q_global"] + 1


def tie_example():
    """A 16:9 screenshot: two grids have the same error, and the pixel rule decides."""
    w, h = 1920, 1080
    cs = tile_grid_candidates(w, h, V1_TILE, V1_MAX_TILES)
    best = min(d for _, _, d in cs)
    area = w * h
    rows = []
    for i, j, d in cs:
        if d == best:
            need = 0.5 * V1_TILE * V1_TILE * i * j
            rows.append((f"{i} x {j}", i * j, _int(int(need)), "yes" if area > need else "no"))
    chosen = internvl_grid(w, h, V1_TILE, V1_MAX_TILES)
    return (f"A {w} x {h} page (aspect ratio {_fx(w / h, 4)}), tiles of 640 pixels, at most 9. The grids 2 x 1 and 4 x 2 are both "
            f"{_fx(best, 4)} away from it (columns / rows = 2). A later candidate of the same error replaces an earlier one only "
            f"if the page has more than half of `640^2 x tiles` pixels; the page has {_int(area)}.\n\n"
            + _table(["grid", "tiles", "half the pixels of the tiles", "page has more"], rows)
            + f"\n\nThe grid chosen is {chosen[0]} x {chosen[1]}.")


def candidates_table():
    w, h = PAGE_W, PAGE_H
    cs = tile_grid_candidates(w, h, V1_TILE, V1_MAX_TILES)
    best = sorted(cs, key=lambda c: c[2])[:5]
    chosen = internvl_grid(w, h, V1_TILE, V1_MAX_TILES)
    rows = [(f"{i} x {j}", i * j, _fx(i / j, 4), _fx(d, 4), "chosen" if (i, j) == chosen else "") for i, j, d in best]
    return (f"The page of {w} x {h} pixels has aspect ratio {w}/{h} = {_fx(w / h, 4)}. The five grids of 2 to 9 tiles whose "
            "columns / rows is closest to it:\n\n" + _table(["grid (columns x rows)", "tiles", "columns / rows", "difference", ""], rows))


PAGES = ((300, 40), (640, 640), (1024, 1024), (1240, 1754), (PAGE_W, PAGE_H), PAGE_300, (1920, 1080))


def readings_data():
    out = []
    for w, h in PAGES:
        out.append(((w, h), reading(w, h, 1), block_rows(w, h, 1), reading(w, h, 2), block_rows(w, h, 2)))
    return out


def _reading_name(r):
    kind, c, rw = r
    return {"small": "one small view", "global": "global view"}.get(kind, f"{c} x {rw} tiles and global")


def readings():
    rows = []
    for (w, h), r1, n1, r2, n2 in readings_data():
        rows.append((f"{w} x {h}", _reading_name(r1), _int(n1), _reading_name(r2), _int(n2)))
    return ("The reading each generation gives a page of the given width x height, and the rows of the image block it puts in "
            "the sequence. First generation: tiles of 640 pixels, at most 9, 273 rows for the global view, 111 for one small "
            "view. Second generation: tiles of 768 pixels, at most 6, 257 rows for the global view, 145 for one small view, "
            "144 rows per tile.\n\n"
            + _table(["page (w x h)", "v1 reading", "v1 rows", "v2 reading", "v2 rows"], rows, "llrlr"))


def reading_scales():
    """Scale (output pixels per page pixel) across and down of each reading, or the one scale of a fitted view."""
    rows = []
    for w, h in ((PAGE_W, PAGE_H), PAGE_300, (1920, 1080)):
        cells = [f"{w} x {h}"]
        for version, tile in ((1, V1_TILE), (2, V2_TILE)):
            kind, c, r = reading(w, h, version)
            cells.append(f"{GLOBAL_SIDE / max(w, h):.4f}")
            if kind == "tiles":
                cells.append(f"{tile * c / w:.4f} / {tile * r / h:.4f}")
            else:
                cells.append("-")
        rows.append(cells)
    return ("Scale of the page in the pixels the encoder sees. 'Global' is the aspect-fitted view in a 1024 x 1024 square, "
            "1024 / max(width, height); 'tiles' is the mosaic of the chosen grid, tile x columns / width across and tile x rows "
            "/ height down.\n\n"
            + _table(["page", "v1 global", "v1 tiles (across / down)", "v2 global", "v2 tiles (across / down)"], rows))


# ------------------------------------------------------------------------------------------------ the learned-query encoder


def prefix_allow(p, total):
    return [[(i < p and j < p) or j <= i for j in range(total)] for i in range(total)]


def mask_toy():
    p = 3
    m = prefix_allow(p, 2 * p)
    labels = [f"image {i}" for i in range(p)] + [f"query {i}" for i in range(p)]
    rows = []
    for i, lab in enumerate(labels):
        rows.append([lab] + ["1" if x else "0" for x in m[i]] + [str(sum(m[i]))])
    total = sum(sum(1 for x in r if x) for r in m)
    return ("The mask of the resampler for 3 image rows and 3 queries (6 rows in all), `allow(i, j) = (i < 3 and j < 3) or j <= i`: "
            "row i may read column j when the entry is 1. The last column counts the keys of the row.\n\n"
            + _table(["row"] + [f"{j}" for j in range(2 * p)] + ["keys"], rows)
            + f"\n\nThe mask has {total} entries equal to 1 of 36, which is 3^2 + (4 + 5 + 6) = {prefix_pairs(3)}.")


def v2_param_count():
    d, ffn, kv = V2["d"], V2["ffn"], V2["kv_heads"] * V2["head"]
    per_layer = (d * d + d) + 2 * (kv * d + kv) + d * d + 3 * d * ffn + 2 * d
    banks = (V2["q_local"] + V2["q_global"]) * d
    proj = d * DEC["D"] + DEC["D"]
    return per_layer, V2["layers"] * per_layer + d + banks, proj + DEC["D"]


def clip_param_count():
    d, ffn = CLIP["d"], CLIP["ffn"]
    per_layer = 4 * d * d + 4 * d + 2 * d * ffn + ffn + d + 4 * d
    return per_layer, CLIP["layers"] * per_layer


def v2_tower():
    per_layer, tower, glue = v2_param_count()
    c_layer, c_tower = clip_param_count()
    rows = [
        ("CLIP-L tower of v1 (24 blocks, width 1024)", _int(c_layer), _int(c_tower), "-"),
        ("learned-query tower of v2 (24 blocks, width 896)", _int(per_layer), _int(tower), "-"),
        ("projector of v1: Linear(2048, 1280)", "-", _int(2048 * DEC["D"] + DEC["D"]), "-"),
        ("projector and separator of v2: Linear(896, 1280), 1280 numbers", "-", _int(glue), "-"),
    ]
    return ("Parameters of the second tower of each generation, counted from the shapes: v1's CLIP block is `4 d^2 + 4 d` (attention "
            "with biases) `+ 2 d f + f + d` (feed-forward with biases) `+ 4 d` (two layer norms with gain and offset); v2's block "
            "has a query matrix `d^2 + d`, key and value matrices `128 d + 128` each (2 key-value heads of 64), an output matrix "
            "`d^2`, three feed-forward matrices `3 d f` of f = 4864, and two gains `2 d`; v2's tower also holds the final gain, "
            "and the two query banks (144 and 256 rows of 896). The patch embedding and class row of CLIP, which v1 bypasses, are "
            "not counted.\n\n"
            + _table(["part", "per block", "parameters"], [(a, b, c) for a, b, c, _ in rows]))


# ------------------------------------------------------------------------------------------------ the decoder


def decoder_params():
    D, V, L = DEC["D"], DEC["V"], DEC["L"]
    attn = 4 * D * D
    dense = 3 * D * DEC["dense_ffn"]
    experts = DEC["experts"] * 3 * D * DEC["moe_ffn"]
    shared = 3 * D * DEC["shared"] * DEC["moe_ffn"]
    router = DEC["experts"] * D
    moe_layers = L - DEC["dense"]
    norms = L * 2 * D + D
    total = 2 * V * D + L * attn + DEC["dense"] * dense + moe_layers * (experts + shared + router) + norms
    active_moe = DEC["topk"] * 3 * D * DEC["moe_ffn"] + shared + router
    body = L * attn + DEC["dense"] * dense + moe_layers * active_moe
    head = V * D
    return {"attn": attn, "dense": dense, "experts": experts, "shared": shared, "router": router, "total": total,
            "body": body, "head": head, "active_moe": active_moe}


def kv_per_token():
    return kvc.cache_bytes_per_token(kvc.shape(DEC["V"], DEC["D"], DEC["L"], DEC["H"], DEC["H"], DEC["dh"], DEC["dense_ffn"], 3))


def mla_per_token():
    """Bytes per token of the latent cache of the attention chapter (a latent of 4 d_h and a rotary key of d_h / 2), bf16."""
    return DEC["L"] * (av.LATENT + av.ROPE_KEY) * kvc.BYTES


def decoder_table():
    p = decoder_params()
    D = DEC["D"]
    rows = [
        ("attention, per block (q, k, v, o)", f"4 x {D}^2", _int(p["attn"])),
        ("dense feed-forward, block 0", f"3 x {D} x {DEC['dense_ffn']}", _int(p["dense"])),
        ("64 routed experts, per MoE block", f"64 x 3 x {D} x {DEC['moe_ffn']}", _int(p["experts"])),
        ("2 shared experts, per MoE block", f"3 x {D} x 2 x {DEC['moe_ffn']}", _int(p["shared"])),
        ("router, per MoE block", f"64 x {D}", _int(p["router"])),
        ("embedding table and output head (untied), each", f"{DEC['V']:,} x {D}", _int(DEC["V"] * D)),
        ("all parameters (and 25 gains of width 1280)", "", _int(p["total"])),
        ("applied to every prompt token (6 of 64 experts)", "", _int(p["body"])),
        ("output head, applied once per scored row", "", _int(p["head"])),
    ]
    kv = kv_per_token()
    mla = mla_per_token()
    return ("The decoder of both generations: 12 blocks of width 1280, 10 heads of 128 with 10 key-value heads (multi-head attention), "
            "block 0 dense with feed-forward width 6848, blocks 1 to 11 mixing 64 routed experts of width 896 (6 chosen per token) "
            "and 2 shared experts, vocabulary 129,280, output head not tied to the embedding table. Counts are products of the "
            "shapes in the second column.\n\n"
            + _table(["part", "shape", "parameters"], rows)
            + f"\n\nCache per token: 2 x 12 x 10 x 128 x 2 bytes = {_int(kv)} bytes in bf16. A latent cache of the kind of "
            f"Volume VI, Chapter 3 (a latent of {av.LATENT} and a rotary key of {av.ROPE_KEY} per layer) would store 12 x "
            f"{av.LATENT + av.ROPE_KEY} x 2 = {_int(mla)} bytes per token, {kv / mla:.2f} times fewer; this decoder does not have "
            "that layer.")


def prompt_tokens(rows):
    return 1 + rows + INSTRUCTION_TOKENS


def prefill_flops(n):
    p = decoder_params()
    return 2 * n * p["body"] + 4 * DEC["L"] * DEC["H"] * DEC["dh"] * (n * (n + 1) // 2) + 2 * DEC["D"] * DEC["V"]


def page_costs():
    rows = []
    cases = [("one small view", block_rows(300, 40, 1), None), ("global view", 273, None),
             ("2 x 3 tiles and global", block_rows(PAGE_W, PAGE_H, 1), None), ("3 x 3 tiles and global", row_count(16, 10, 3, 3), None)]
    kv = kv_per_token()
    for name, r, _ in cases:
        n = prompt_tokens(r)
        tiles = {"one small view": 1, "global view": 0, "2 x 3 tiles and global": 6, "3 x 3 tiles and global": 9}[name]
        vision = (encoder_total(1, 1024) if name != "one small view" else 0) + tiles * encoder_total(1, 640)
        rows.append((name, _int(r), _int(n), _tflops(vision), _tflops(prefill_flops(n)), _mib(n * kv)))
    return ("What one page costs the first generation before the first word is written: the vision half (the encoder over every view, "
            "at the FLOPs of the table above), the decoder's prefill of the prompt (BOS, the image block and the 9 tokens of the "
            "default instruction) at 2 FLOPs per active parameter per token, `4 L H d_h` per attention pair and the output head once, "
            "and the cache the prompt occupies at 61,440 bytes per token.\n\n"
            + _table(["reading", "rows", "prompt tokens", "vision TFLOPs", "decoder prefill TFLOPs", "cache (MiB)"], rows))


# ------------------------------------------------------------------------------------------------ squeezing a page


WORDS = 930
CHARS_PER_WORD = 6          # including the space after the word
CHARS_PER_TOKEN = 4


def text_tokens(words=WORDS):
    return words * CHARS_PER_WORD / CHARS_PER_TOKEN


def squeeze():
    toks = text_tokens()
    rows = []
    for name, r in (("global view", 273), ("v1, 2 x 3 tiles and global", block_rows(PAGE_W, PAGE_H, 1)),
                    ("v2, 2 x 3 tiles and global", block_rows(PAGE_W, PAGE_H, 2))):
        rows.append((name, _int(r), _fx(toks / r, 2)))
    cap = []
    for f in (15, 30):
        for s_name, sx, sy in (("global view", GLOBAL_SIDE / PAGE_H, GLOBAL_SIDE / PAGE_H),
                               ("v1 tiles", V1_TILE * 2 / PAGE_W, V1_TILE * 3 / PAGE_H),
                               ("v2 tiles", V2_TILE * 2 / PAGE_W, V2_TILE * 3 / PAGE_H)):
            area = (64 / sx) * (64 / sy)
            chars = area / (0.6 * f * f)
            cap.append((f"{f}", s_name, _int(round(area)), _fx(chars, 1), _fx(chars / CHARS_PER_TOKEN, 1)))
    return (f"A page of {WORDS} words (the dense synthetic page of the commit cited below), at {CHARS_PER_WORD} "
            f"characters a word including the space and {CHARS_PER_TOKEN} characters a text token, is {_int(round(toks))} text tokens. "
            "Divided by the rows of the image block:\n\n"
            + _table(["reading of the 1654 x 2339 page", "rows", "text tokens / rows"], rows)
            + "\n\nWhat one projector token can carry. A token stands for a block of 64 x 64 pixels of the view (a 16-pixel patch "
            "times the 4 of the compressor), which is `64 / s` by `64 / s'` pixels of the page when the view shows the page at "
            "scales s across and s' down (those of the previous tables). Text of font size f is taken to need `0.5 f` of width "
            "per character and `1.2 f` per line, so `0.6 f^2` pixels of page, and the table assumes the block is full of it.\n\n"
            + _table(["font size (px)", "seen as", "page pixels per token", "characters", "text tokens"], cap)
            + f"\n\nA row of the block is {DEC['D']:,} numbers of 16 bits, {DEC['D'] * 16:,} bits. The 158.3 characters of the first "
            f"row, at log2(70) = {math.log2(70):.2f} bits for a character drawn from 70 equally likely symbols (an upper bound), "
            f"are {158.3 * math.log2(70):,.0f} bits.")


def row_capacity():
    """Bits a row of the block can nominally hold against the bits of the characters it stands for."""
    bits_row = DEC["D"] * 16
    bits_char = math.log2(70)
    return bits_row, bits_char


# ------------------------------------------------------------------------------------------------ serving


def encode_order_data():
    """The pages of the chapter's example, as (pixels, wait in seconds)."""
    return [("2000 x 3000", 2000 * 3000), ("600 x 400", 600 * 400), ("1200 x 1700", 1200 * 1700)]


def encode_order():
    pages = encode_order_data()
    remaining = [(name, px, 0.0) for name, px in pages]
    order = []
    t = 0.0
    while remaining:
        costs = [(px / (1 + w), name, px, w) for name, px, w in remaining]
        best = min(costs, key=lambda c: c[0])
        order.append((best[1], f"{best[2] / (1 + best[3]):,.0f}"))
        remaining = [(n, p, w) for n, p, w in remaining if n != best[1]]
    rows = [(i + 1, name, c) for i, (name, c) in enumerate(order)]
    aged = [("2000 x 3000", 2000 * 3000 / (1 + 120)), ("600 x 400", 600 * 400 / 1.0)]
    wait_big = 2000 * 3000 / (600 * 400) - 1
    return ("The encoder takes the waiting page of the smallest `pixels / (1 + seconds waited)`. Three pages that arrived together "
            "(wait 0) are taken in this order:\n\n" + _table(["taken", "page", "cost when taken"], rows)
            + f"\n\nA 2000 x 3000 page that has waited 120 seconds costs {aged[0][1]:,.0f} against the {aged[1][1]:,.0f} of a "
            f"fresh 600 x 400 crop, and goes first. The wait at which it passes a fresh crop is `6,000,000 / 240,000 - 1` = "
            f"{wait_big:.0f} seconds.")


POOL_TOKENS = 128 * 1024
BLOCK = 64
MAX_BATCH = 64


def pool_pages(prompt, new):
    blocks = -(-(prompt + new) // BLOCK)
    return blocks, min(MAX_BATCH, (POOL_TOKENS // BLOCK) // blocks)


def pool_capacity():
    rows = []
    for name, r in (("one small view", 111), ("global view", 273), ("2 x 3 tiles and global", 903), ("3 x 3 tiles and global", 1203)):
        n = prompt_tokens(r)
        cells = [name, _int(n)]
        for new in (512, 2048):
            blocks, pages = pool_pages(n, new)
            cells += [_int(blocks), _int(pages)]
        rows.append(cells)
    kv = kv_per_token()
    return (f"The cache pool holds {_int(POOL_TOKENS)} tokens in blocks of {BLOCK} ({_int(POOL_TOKENS // BLOCK)} blocks, "
            f"{_fx(POOL_TOKENS * kv / 2 ** 30, 2)} GiB at {_int(kv)} bytes a token), and at most {MAX_BATCH} pages are decoded together. "
            "A page that generates its whole budget of new tokens holds `ceil((prompt + new) / 64)` blocks; the table gives that "
            "and the number of such pages the pool holds (capped at 64).\n\n"
            + _table(["reading", "prompt tokens", "blocks, 512 new", "pages", "blocks, 2048 new", "pages"], rows))


GIB = 1
VISION_GIB, DECODER_GIB = 15, 18


def _last_max(items, key):
    best = None
    for it in items:
        if best is None or key(it) >= key(best):
            best = it
    return best


def plan_placement(cards, vision=VISION_GIB, decoder=DECODER_GIB):
    fit = [c for c in cards if c[1] >= vision]
    if not fit:
        return (None, "host")
    v = _last_max(fit, lambda c: c[1])
    other = [c for c in cards if c[0] != v[0] and c[1] >= decoder]
    if other:
        return (v[0], f"card {_last_max(other, lambda c: c[1])[0]}")
    if v[1] >= vision + decoder:
        return (v[0], "same card")
    return (v[0], "host")


def plan_replicas(cards, capacities, vision=VISION_GIB, decoder=DECODER_GIB):
    need = vision + decoder
    cap = dict(capacities)
    whole = [(i, "same card") for i, free in cards if max(cap.get(i, free), free) >= need]
    return whole if len(whole) >= 2 else [plan_placement(cards, vision, decoder)]


def _fmt_place(p):
    return "no card" if p[0] is None else f"encoder on card {p[0]}"


def placement():
    cases = [
        ("one card of 24", [(0, 24)], None),
        ("cards of 24 and 30", [(0, 24), (1, 30)], None),
        ("two cards of 90, one replica planned", [(0, 90), (1, 90)], "single"),
        ("one card of 90", [(0, 90)], None),
        ("two cards of 90", [(0, 90), (1, 90)], "replicas"),
        ("cards of 90 and 12", [(0, 90), (1, 12)], "replicas"),
        ("one card of 10", [(0, 10)], None),
        ("cards 90 and 12 free, 96 and 96 in all", [(0, 90), (1, 12)], "held"),
    ]
    rows = []
    for name, cards, mode in cases:
        if mode in (None, "single"):
            ps = [plan_placement(cards)]
        elif mode == "replicas":
            ps = plan_replicas(cards, cards)
        else:
            ps = plan_replicas(cards, [(0, 96), (1, 96)])
        rows.append((name, str(len(ps)), "; ".join(f"{_fmt_place(p)}, decoder on {p[1]}" for p in ps)))
    return (f"The encoders of one replica need {VISION_GIB} GiB and the decoder {DECODER_GIB} GiB (the figures the crates budget). "
            "Cards are written (usable GiB). The encoder goes on the card with the most room that holds it (the last of equals); "
            "the decoder on another card that holds it, else on the encoder's card if it holds both, else on the host. With two "
            "or more cards that each hold the whole model, a replica runs on each, judged by the card's total capacity.\n\n"
            + _table(["offered", "replicas", "placement"], rows, "lrl"))


# ------------------------------------------------------------------------------------------------ the anti-repetition filter

NGRAM_TOY = {"n": 4, "window": 13, "allow": (11,)}
NGRAM_TOKENS = [7, 8, 9, 3, 5, 7, 8, 9, 11, 2, 7, 8, 9]


def ngram_followers(generated, n, window):
    if n == 0 or len(generated) < n:
        return []
    prefix = generated[len(generated) - (n - 1):]
    start = max(len(generated) - window, 0)
    end = len(generated) - n + 1
    out = []
    for i in range(start, end):
        nxt = generated[i + n - 1]
        if generated[i:i + n - 1] == prefix and nxt not in out:
            out.append(nxt)
    return out


def ngram_banned(generated, n, window, allow):
    return [t for t in ngram_followers(generated, n, window) if t not in allow]


def ngram_filter():
    n, w, allow = NGRAM_TOY["n"], NGRAM_TOY["window"], NGRAM_TOY["allow"]
    g = NGRAM_TOKENS
    prefix = g[len(g) - (n - 1):]
    start = max(len(g) - w, 0)
    end = len(g) - n + 1
    rows = []
    for i in range(start, end):
        match = g[i:i + n - 1] == prefix
        rows.append((i, str(g[i:i + n - 1]), g[i + n - 1], "yes" if match else "no"))
    followers = ngram_followers(g, n, w)
    banned = ngram_banned(g, n, w, allow)
    by_window = [(win, max(len(g) - win, 0), str(ngram_followers(g, n, win)), str(ngram_banned(g, n, win, allow))) for win in (13, 12, 7)]
    return (f"Generated so far: {g} (13 tokens). n-gram size n = {n}, window {w}, allowed token {list(allow)}. The trailing "
            f"{n - 1} tokens are {prefix}. Start of the scan = max(13 - {w}, 0) = {start}; end (exclusive) = 13 - {n} + 1 = {end}.\n\n"
            + _table(["start i", "tokens i .. i + 2", "token i + 3", "equals the trailing three"], rows)
            + f"\n\nTokens that would complete a repeated 4-gram: {followers}. After removing the allowed token {list(allow)}: "
            f"{banned}. The same history under three windows:\n\n"
            + _table(["window", "scan starts at", "followers", "banned"], by_window, "lrll"))


# ------------------------------------------------------------------------------------------------ word error rate

WER_REF = "the page reads left to right".split()
WER_HYP = "the page read left right now".split()


def edit_table(ref, hyp):
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            sub = d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    return d


def wer():
    d = edit_table(WER_REF, WER_HYP)
    rows = [[("start" if i == 0 else WER_REF[i - 1])] + [str(x) for x in d[i]] for i in range(len(WER_REF) + 1)]
    dist = d[-1][-1]
    long_hyp = ["a"] * 12
    d2 = edit_table(WER_REF, long_hyp)
    return (f"Reference: {' '.join(WER_REF)} ({len(WER_REF)} words). Hypothesis: {' '.join(WER_HYP)}. Cell (i, j) is the fewest "
            "insertions, deletions and substitutions turning the first i reference words into the first j hypothesis words; "
            "the columns are the empty hypothesis and then the words of the hypothesis in order.\n\n"
            + _table(["reference word", "-"] + WER_HYP, rows)
            + f"\n\nThe edit distance is {dist}, and the word error rate is {dist} / {len(WER_REF)} = {_fx(dist / len(WER_REF), 3)}. "
            f"A hypothesis of 12 unrelated words against the same reference has distance {d2[-1][-1]} and a rate of "
            f"{_fx(d2[-1][-1] / len(WER_REF), 3)}, above 1.")


# ------------------------------------------------------------------------------------------------ figure data


def rows_series(heights=None):
    """(page heights, v1 rows, v2 rows, dynamic-resolution rows of the previous chapter) for portrait pages of aspect 1 / sqrt 2."""
    heights = heights or list(range(300, 4001, 50))
    v1, v2, dyn = [], [], []
    for h in heights:
        w = round(h / math.sqrt(2))
        v1.append(block_rows(w, h, 1))
        v2.append(block_rows(w, h, 2))
        dyn.append(vlms.qwen_tokens_for(h, w, vlms.QWEN_SERVE_MAX))
    return heights, v1, v2, dyn


def pipeline_data():
    """Numbers drawn in the encoder figure."""
    g = {name: (side, n, n2) for name, side, n, n1, n2, _, _, _ in grids_data()}
    per_layer, tower, _ = v2_param_count()
    return {"patches": g["global view"][1] ** 2, "tokens": g["global view"][2] ** 2, "rows_v1": 273, "rows_v2": 257,
            "tile_tokens": g["v1 tile"][2] ** 2, "tile_tokens_v2": g["v2 tile"][2] ** 2, "tower": tower}
