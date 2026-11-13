# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Promptable Segmentation and Video Tracking".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: segment_anything.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. Earlier modules are imported, never copied: the matrix
product and sums of `self_attention`, the exact GELU of `transformer_block`.

* a toy promptable segmenter (region growing from a click) that shows what a prompt is, why one click is ambiguous, and why a
  model answers with several masks and a predicted overlap for each;
* random Fourier features of a coordinate: the encoding of a click, and the kernel that the encoding implements;
* the prompt tokens of a click, a pair of clicks and a box, with the learned type embeddings;
* a tiny two-way mask decoder with random weights: token and image attention in both directions, the transposed-convolution
  upscaling and the per-token dot product that makes a mask, and the overlap head;
* the decomposed relative-position bias of the image encoder, and the stages and the counted FLOPs of SAM 2's Hiera trunk, of the
  ViT-B tower of the first SAM, and of the decoder per prompt;
* the memory bank of the video path: which frames and object pointers frame t reads, the cost of reading them, and a small
  memory attention with a rotary position encoding on a 3 by 3 grid.

Arithmetic is 64-bit floating point. Random inputs come from `bookrng.generator(name)` with the names, seeds and draw orders printed
beside each table. Sums of products run left to right.

Swedish Embedded AB implements solutions for interactive segmentation and video tracking whose cost is counted before it is paid.
If your team needs expertise in promptable segmentation models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import self_attention as sa
import transformer_block as tb

# ------------------------------------------------------------------------------------------------ helpers

_sum = sa._sum
matmul = sa.matmul
transpose = sa.transpose


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fx(v, d=3):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _vec(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"seed 0x{s:016x} ({s})"


def _round1(x):
    r = math.floor(abs(x) * 10 + 0.5) / 10
    return -r if x < 0 and r > 0 else r


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _giga(v, d=2):
    return f"{v / 1e9:,.{d}f}"


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = _sum(e)
    return [v / s for v in e]


# ------------------------------------------------------------------------------------------------ the toy promptable segmenter

TOY_IMAGE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 6, 6, 6, 0, 0, 0],
    [0, 6, 9, 9, 6, 3, 3, 0],
    [0, 6, 9, 9, 6, 3, 3, 0],
    [0, 6, 6, 6, 6, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
TOLERANCES = (0, 3, 6)


def grow(image, seed_rc, tol):
    """The pixels reachable from seed_rc through 4-neighbours whose code differs from the seed's code by at most tol."""
    r0, c0 = seed_rc
    ref = image[r0][c0]
    n = len(image)
    seen = {(r0, c0)}
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in seen and abs(image[rr][cc] - ref) <= tol:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return seen


def _rows_of(mask, n=8):
    return ["".join("#" if (r, c) in mask else "." for c in range(n)) for r in range(n)]


def _iou(a, b):
    return len(a & b) / len(a | b)


def toy_masks_data():
    click = (2, 2)
    cands = [grow(TOY_IMAGE, click, t) for t in TOLERANCES]
    negative = grow(TOY_IMAGE, (3, 6), 0)           # a click on object B, grown with tolerance 0
    refined = cands[2] - negative
    core = cands[0]
    obj_a = cands[1]
    return {"click": click, "cands": cands, "negative": negative, "refined": refined, "core": core, "obj_a": obj_a}


def toy_prompts():
    d = toy_masks_data()
    img_rows = ["".join(str(v) for v in row) for row in TOY_IMAGE]
    cols = [img_rows] + [_rows_of(m) for m in d["cands"]] + [_rows_of(d["refined"])]
    rows = [(r,) + tuple(c[r] for c in cols) for r in range(8)]
    sizes = ("pixels", "", *[str(len(m)) for m in d["cands"]], str(len(d["refined"])))
    rows.append(("count",) + sizes[1:])
    return ("The image is 8 by 8 with the codes shown (0 is background). The toy segmenter takes a click (row 2, column 2, the code 9) "
            "and a tolerance, and grows the set of pixels reachable through 4-neighbours whose code differs from the clicked code by "
            "at most the tolerance. The last column adds a second, negative click at (row 3, column 6) on the right-hand object, grown "
            "with tolerance 0 and removed from the tolerance-6 mask.\n\n"
            + _table(["row", "image", "tolerance 0", "tolerance 3", "tolerance 6", "tolerance 6 minus the negative click"], rows, "llllll"))


def toy_readings():
    d = toy_masks_data()
    readings = (("the bright core", d["core"]), ("the whole left object", d["obj_a"]))
    rows = []
    for name, m in zip(("tolerance 0", "tolerance 3", "tolerance 6", "tolerance 6, negative click"), d["cands"] + [d["refined"]]):
        rows.append((name, str(len(m))) + tuple(_fx(_iou(m, r), 3) for _, r in readings))
    # the best single mask for a user who is equally likely to mean either reading, under a per-pixel squared error
    p = {}
    for (r, c) in d["obj_a"]:
        p[(r, c)] = 0.5 * (1.0 if (r, c) in d["core"] else 0.0) + 0.5
    halo = sorted(k for k in p if p[k] != 1.0)
    return ("Overlap (intersection over union of pixel sets, as in Chapter 5) of each candidate with the two readings of the click.\n\n"
            + _table(["candidate", "pixels", "IoU with the core (4 pixels)", "IoU with the whole object (16 pixels)"], rows, "lrrr")
            + f"\n\nIf the user means either reading with probability 1/2, the per-pixel average of the two truth masks is 1 on the "
              f"{len(d['core'])} core pixels, 1/2 on the {len(halo)} other pixels of the object and 0 elsewhere: the squared-error "
              "optimum of one mask is a blurred mask that neither reading would draw.")


# ------------------------------------------------------------------------------------------------ random Fourier features

TWO_PI = 2.0 * math.pi
NAME_FOURIER = "segment_anything.fourier"


def fourier_matrix(gen, features):
    """Z[2][features]: row 0 (the x frequencies) then row 1 (the y frequencies), each entry one `normal()` draw."""
    return [[gen.normal() for _ in range(features)] for _ in range(2)]


def fourier_encode(G, x, y):
    """phi(x, y) for a point already mapped to [-1, 1]^2: the sines then the cosines of 2 pi (x G[0][j] + y G[1][j])."""
    f = len(G[0])
    arg = [TWO_PI * (x * G[0][j] + y * G[1][j]) for j in range(f)]
    return [math.sin(a) for a in arg] + [math.cos(a) for a in arg]


def fourier_encoding():
    g = bookrng.generator(NAME_FOURIER)
    G = fourier_matrix(g, 4)
    pts = [(0.3, 0.7), (0.3, 0.8), (0.9, 0.1)]
    rows = []
    for (u, v) in pts:
        x, y = 2 * u - 1, 2 * v - 1
        rows.append((f"({u}, {v})", f"({_fx(x, 1)}, {_fx(y, 1)})") + tuple(_fx(e, 3) for e in fourier_encode(G, x, y)))
    sim = lambda a, b: _dot(fourier_encode(G, 2 * a[0] - 1, 2 * a[1] - 1), fourier_encode(G, 2 * b[0] - 1, 2 * b[1] - 1)) / 4.0
    return (f"{_seed(NAME_FOURIER)}. Draws, in this order: the 4 entries of row 0 of G (the x frequencies), then the 4 of row 1 "
            "(the y frequencies), each one `normal()` (scale 1). "
            f"G row 0 = {_vec(G[0])}, row 1 = {_vec(G[1])}.\n\n"
            + _table(["point (u, v) in the picture", "(x, y) = (2u - 1, 2v - 1)", "sin 0", "sin 1", "sin 2", "sin 3", "cos 0", "cos 1",
                      "cos 2", "cos 3"], rows, "ll" + "r" * 8)
            + f"\n\nDot product of the encodings divided by 4: (0.3, 0.7) with (0.3, 0.8) is {_fx(sim(pts[0], pts[1]), 3)}, "
              f"(0.3, 0.7) with (0.9, 0.1) is {_fx(sim(pts[0], pts[2]), 3)}, a point with itself is {_fx(sim(pts[0], pts[0]), 3)}.")


def fourier_kernel_rows(features=128, sigmas=(1.0, 4.0), deltas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    g = bookrng.generator(NAME_FOURIER + ".kernel")
    Z = fourier_matrix(g, features)
    rows = []
    for dl in deltas:
        row = [_fx(dl, 2)]
        for s in sigmas:
            G = [[s * z for z in zr] for zr in Z]
            a = fourier_encode(G, -0.2, 0.1)
            b = fourier_encode(G, -0.2 + dl, 0.1)
            est = _dot(a, b) / features
            exact = math.exp(-0.5 * (TWO_PI * s * dl) ** 2)
            row += [_fx(est, 3), _fx(exact, 3)]
        rows.append(tuple(row))
    return rows


def fourier_kernel():
    features = 128
    rows = fourier_kernel_rows(features)
    return (f"{_seed(NAME_FOURIER + '.kernel')}. Draws: the 128 entries of row 0 of Z, then the 128 of row 1, each one `normal()`; "
            "G = sigma Z for sigma = 1 and 4. The two points are (x, y) = (-0.2, 0.1) and (-0.2 + delta, 0.1), "
            "estimate = (phi(p) . phi(p')) / 128 with the 256-channel encoding, exact = exp(-(2 pi sigma delta)^2 / 2) "
            "(library exp).\n\n"
            + _table(["delta (in the [-1, 1] frame)", "sigma 1, estimate", "sigma 1, exact", "sigma 4, estimate", "sigma 4, exact"],
                     rows))


# ------------------------------------------------------------------------------------------------ prompt tokens

D = 8                      # width of the toy decoder
FEATS = D // 2             # random Fourier features per coordinate pair
NAME_TOY = "segment_anything.toy_sam"
IMG = 8                    # side of the toy image in pixels
EMB = 4                    # side of the toy image embedding in tokens (the image is cut 2 pixels per token)
TYPES = ("background click", "foreground click", "box corner 1", "box corner 2")


class ToySam:
    """The learned tensors of the toy prompt encoder and decoder, drawn in the order printed by `toy_setup`."""

    def __init__(self):
        g = bookrng.generator(NAME_TOY)
        self.g = g
        n = lambda r, c, s: [[s * g.normal() for _ in range(c)] for _ in range(r)]
        lin = lambda i, o: n(i, o, 1.0 / math.sqrt(i))        # input-major: y = x W, so W is [in, out]
        self.G = fourier_matrix(g, FEATS)
        self.point_emb = n(4, D, 1.0)
        self.not_a_point = n(1, D, 1.0)[0]
        self.iou_token = n(1, D, 1.0)[0]
        self.mask_tokens = n(4, D, 1.0)
        self.no_mask = n(1, D, 1.0)[0]
        self.image = [[_round1(g.normal()) for _ in range(D)] for _ in range(EMB * EMB)]
        io = D // 2
        self.layers = []
        for layer in range(2):
            self.layers.append({
                "sa": [lin(D, D) for _ in range(4)],                       # q, k, v, out
                "t2i": [lin(D, io), lin(D, io), lin(D, io), lin(io, D)],
                "mlp": [lin(D, 2 * D), lin(2 * D, D)],
                "i2t": [lin(D, io), lin(D, io), lin(D, io), lin(io, D)],
            })
        self.final = [lin(D, io), lin(D, io), lin(D, io), lin(io, D)]
        self.convt = [[[[g.normal() / math.sqrt(D) for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(D)]
        self.hyper = [[lin(D, D), lin(D, D // 4)] for _ in range(4)]
        self.iou_head = [lin(D, D), lin(D, 4)]


_TOY = None


def toy():
    global _TOY
    if _TOY is None:
        _TOY = ToySam()
    return _TOY


def embed_points(t, points, labels, size=IMG):
    """The sparse prompt tokens of the points (x, y in pixels) with their labels, and the padding point appended last.

    A point with label l in 0..3 is the Fourier encoding of ((x + 0.5)/size, (y + 0.5)/size) mapped to [-1, 1], plus the learned
    embedding number l. The padding point (label -1) is the learned `not_a_point` row alone."""
    rows = []
    for (x, y), lab in zip(points, labels):
        u, v = (x + 0.5) / size, (y + 0.5) / size
        enc = fourier_encode(t.G, 2 * u - 1, 2 * v - 1)
        rows.append([a + b for a, b in zip(enc, t.point_emb[lab])])
    rows.append(list(t.not_a_point))
    return rows


def prompt_tokens():
    t = toy()
    cases = [
        ("one foreground click at (x 2, y 2)", [(2, 2)], [1], ["click (label 1)", "padding (label -1)"]),
        ("a foreground click and a background click", [(2, 2), (6, 3)], [1, 0], ["click (label 1)", "click (label 0)", "padding"]),
        ("a box from (x 1, y 1) to (x 4, y 4)", [(1, 1), (4, 4)], [2, 3], ["corner (label 2)", "corner (label 3)", "padding"]),
    ]
    out = [f"The first two cases use the click (x 2, y 2) and the background click (x 6, y 3), the box uses the corners (x 1, y 1) and "
           "(x 4, y 4); x is the column and y the row of the 8 by 8 picture, and a pixel is encoded at its centre, "
           "((x + 0.5)/8, (y + 0.5)/8) mapped to [-1, 1] by `2u - 1`, then Fourier-encoded with G and added to the type embedding "
           "of its label."]
    for title, pts, labs, names in cases:
        rows = embed_points(t, pts, labs)
        out.append(f"{title}:\n\n" + _table(["token"] + [f"d{i}" for i in range(D)],
                                           [(n,) + tuple(_fx(v, 3) for v in r) for n, r in zip(names, rows)], "l" + "r" * D))
    return "\n\n".join(out)


def prompt_shapes():
    s_img, emb, d, mc = 1024, 64, 256, 16
    rows = [
        ("mask prompt", f"1 x {4 * emb} x {4 * emb}", "input, a logit map at 4 x the embedding side"),
        ("conv 2 x 2, stride 2, LayerNorm2d, GELU", f"{mc // 4} x {2 * emb} x {2 * emb}", f"1 -> {mc // 4} channels"),
        ("conv 2 x 2, stride 2, LayerNorm2d, GELU", f"{mc} x {emb} x {emb}", f"{mc // 4} -> {mc} channels"),
        ("conv 1 x 1", f"{d} x {emb} x {emb}", f"{mc} -> {d} channels, added to the image embedding"),
        ("no mask given", f"{d} x {emb} x {emb}", "the one learned vector no_mask_embed, copied to every cell"),
    ]
    weights = 1 * (mc // 4) * 4 + (mc // 4) * mc * 4 + mc * d
    return (f"The image is {s_img} x {s_img} pixels and the embedding {emb} x {emb} ({s_img // emb} pixels per cell).\n\n"
            + _table(["stage", "output shape", "note"], rows, "lll")
            + f"\n\nWeights of the three convolutions (without biases and norms): {_int(weights)}.")


def toy_setup():
    t = toy()
    io = D // 2
    items = [
        ("G, the Fourier matrix", f"2 x {FEATS}", "1"),
        ("point_emb, the 4 type embeddings", f"4 x {D}", "1"),
        ("not_a_point", f"{D}", "1"),
        ("iou_token", f"{D}", "1"),
        ("mask_tokens", f"4 x {D}", "1"),
        ("no_mask, the dense embedding of no mask", f"{D}", "1"),
        ("image embedding, 4 x 4 tokens (row-major), each `round(normal(), 1)`", f"{EMB * EMB} x {D}", "1"),
        ("per layer (2 layers): self-attention q, k, v, out", f"4 x ({D} x {D})", f"1/sqrt({D})"),
        ("per layer: token-to-image q, k, v, out", f"3 x ({D} x {io}), {io} x {D}", f"1/sqrt(fan-in)"),
        ("per layer: MLP up, down", f"{D} x {2 * D}, {2 * D} x {D}", "1/sqrt(fan-in)"),
        ("per layer: image-to-token q, k, v, out", f"3 x ({D} x {io}), {io} x {D}", "1/sqrt(fan-in)"),
        ("final token-to-image q, k, v, out", f"3 x ({D} x {io}), {io} x {D}", "1/sqrt(fan-in)"),
        ("transposed convolution W[c][c'][a][b]", f"{D} x 2 x 2 x 2", f"1/sqrt({D})"),
        ("hypernetwork of mask token i = 0..3: layer 1, layer 2", f"{D} x {D}, {D} x {D // 4}", "1/sqrt(fan-in)"),
        ("IoU head: layer 1, layer 2", f"{D} x {D}, {D} x 4", "1/sqrt(fan-in)"),
    ]
    return (f"{_seed(NAME_TOY)}. One generator draws every tensor, each entry one `normal()` times the scale shown, tensors in the "
            "order of the table, each row-major (a weight [in, out] is drawn row by row; `y = x W`). All biases are 0, "
            "all LayerNorm gains 1 and offsets 0 (epsilon 1e-5).\n\n"
            + _table(["tensor (draw order, top to bottom)", "shape", "scale"], items, "lll")
            + "\n\nWhat the tensors do, so that the forward pass can be repeated. A point (x, y) in pixels becomes the Fourier encoding "
              "of ((x + 0.5)/8, (y + 0.5)/8) mapped to [-1, 1] by `2u - 1`, plus the type embedding of its label (label 0 is a "
              "background click, 1 a foreground click, 2 and 3 the corners of a box); the padding point is `not_a_point` alone. The "
              "token sequence is the iou token, the four mask tokens, then the prompt tokens, and this initial sequence is also the "
              "position code `tok_pe` that is added to the queries and keys of the token side in every layer. The cells carry the "
              "image embedding plus `no_mask` (copied to every cell) as their content and, as their position code, the Fourier encoding "
              "of the cell centre ((j + 0.5)/4, (i + 0.5)/4) mapped to [-1, 1]. Every attention has one head, scores q . k / sqrt(w) "
              "with w the projected width (8 in the self-attention, 4 in the cross-attentions), a softmax over the keys, the values "
              "projected by v and the result by out. In layer 0 the self-attention takes no position code and has no residual; in "
              "layer 1 it compares `q + tok_pe`, copies `q` and has a residual. Each attention and the MLP are followed by a residual "
              "add and a LayerNorm; the MLP is `relu(x W_up) W_down`. The upscaling is the transposed convolution of kernel 2 and "
              "stride 2 from 8 to 2 channels, followed by the exact GELU (of Volume III, Chapter 2), taking the 4 x 4 cells to "
              "8 x 8. A hypernetwork is `relu(h W_1) W_2` on the mask token's output, giving 2 numbers; the IoU head is "
              "`relu(h W_1) W_2` on the overlap token followed by a sigmoid. The large model's hypernetworks have three layers (256, 256, "
              "256 and 32 numbers, a ReLU between layers); the toy keeps two.")


# ------------------------------------------------------------------------------------------------ the two-way decoder


def _linear(X, W, b=None):
    out = matmul(X, W)
    return out if b is None else [[v + bb for v, bb in zip(r, b)] for r in out]


def _layernorm(X, eps=1e-5):
    out = []
    for r in X:
        m = _sum(r) / len(r)
        v = _sum([(x - m) ** 2 for x in r]) / len(r)
        inv = 1.0 / math.sqrt(v + eps)
        out.append([(x - m) * inv for x in r])
    return out


def _add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def attention(Qin, Kin, Vin, W):
    """One head: Q = Qin W_q, K = Kin W_k, V = Vin W_v, softmax(Q K^T / sqrt(width)) V W_out; returns (output, weights)."""
    Wq, Wk, Wv, Wo = W
    Q, K, V = matmul(Qin, Wq), matmul(Kin, Wk), matmul(Vin, Wv)
    w = len(Q[0])
    P = [_softmax([_dot(q, k) / math.sqrt(w) for k in K]) for q in Q]
    return matmul(matmul(P, V), Wo), P


def two_way(t, tokens, image, pe):
    """The two-way transformer: queries start as `tokens` (also their position encoding), keys as `image` with position `pe`.

    Layer 0 skips the positional add and the residual of its self-attention. Returns (hs, keys, taps)."""
    q, k = [list(r) for r in tokens], [list(r) for r in image]
    taps = []
    for li, L in enumerate(t.layers):
        if li == 0:
            q, _ = attention(q, q, q, L["sa"])
        else:
            qp = _add(q, tokens)
            o, _ = attention(qp, qp, q, L["sa"])
            q = _add(q, o)
        q = _layernorm(q)
        o, w_t2i = attention(_add(q, tokens), _add(k, pe), k, L["t2i"])
        q = _layernorm(_add(q, o))
        m = _linear([[max(0.0, v) for v in r] for r in _linear(q, L["mlp"][0])], L["mlp"][1])
        q = _layernorm(_add(q, m))
        o, w_i2t = attention(_add(k, pe), _add(q, tokens), q, L["i2t"])
        k = _layernorm(_add(k, o))
        taps.append({"t2i": w_t2i, "i2t": w_i2t})
    o, w_fin = attention(_add(q, tokens), _add(k, pe), k, t.final)
    hs = _layernorm(_add(q, o))
    taps.append({"final": w_fin})
    return hs, k, taps


def upscale(t, keys):
    """The image tokens as a [D, 4, 4] map, a transposed convolution (kernel 2, stride 2, D -> D/4 channels), then GELU."""
    co = D // 4
    out = [[[0.0] * (2 * EMB) for _ in range(2 * EMB)] for _ in range(co)]
    for i in range(EMB):
        for j in range(EMB):
            tok = keys[i * EMB + j]
            for cp in range(co):
                for a in range(2):
                    for b in range(2):
                        s = 0.0
                        for c in range(D):
                            s += tok[c] * t.convt[c][cp][a][b]
                        out[cp][2 * i + a][2 * j + b] = tb.gelu(s)
    return out


def decode(t, tokens_sparse):
    """Everything the toy decoder computes for a prompt whose sparse tokens are `tokens_sparse`."""
    tokens = [list(t.iou_token)] + [list(r) for r in t.mask_tokens] + [list(r) for r in tokens_sparse]
    # the dense embedding is no_mask copied to every cell; the keys are image + dense, the key position is the Fourier encoding
    # of the cell centres
    image = [_add([row], [t.no_mask])[0] for row in t.image]
    pe = []
    for i in range(EMB):
        for j in range(EMB):
            u, v = (j + 0.5) / EMB, (i + 0.5) / EMB
            pe.append(fourier_encode(t.G, 2 * u - 1, 2 * v - 1))
    hs, keys, taps = two_way(t, tokens, image, pe)
    up = upscale(t, keys)
    hyper = []
    for i in range(4):
        h1 = [max(0.0, v) for v in _linear([hs[1 + i]], t.hyper[i][0])[0]]
        hyper.append(_linear([h1], t.hyper[i][1])[0])
    flat = [[up[c][r][cc] for r in range(2 * EMB) for cc in range(2 * EMB)] for c in range(D // 4)]
    masks = matmul(hyper, flat)                       # [4, 64]: one dot product per mask token and pixel
    h1 = [max(0.0, v) for v in _linear([hs[0]], t.iou_head[0])[0]]
    iou = [1.0 / (1.0 + math.exp(-v)) for v in _linear([h1], t.iou_head[1])[0]]
    return {"tokens": tokens, "hs": hs, "keys": keys, "taps": taps, "up": up, "hyper": hyper, "flat": flat, "masks": masks,
            "iou": iou, "pe": pe}


def _grid_text(vals, side, d=2):
    return "\n".join("  ".join(f"{v:7.{d}f}" for v in vals[r * side:(r + 1) * side]) for r in range(side))


def decoder_shapes():
    t = toy()
    toks = embed_points(t, [(2, 2)], [1])
    r = decode(t, toks)
    rows = [
        ("tokens: iou, 4 masks, click, padding", f"{len(r['tokens'])} x {D}"),
        ("image tokens (4 x 4 cells) with position", f"{len(r['keys'])} x {D}"),
        ("token-to-image weights, one layer", f"{len(r['taps'][0]['t2i'])} x {len(r['taps'][0]['t2i'][0])}"),
        ("image-to-token weights, one layer", f"{len(r['taps'][0]['i2t'])} x {len(r['taps'][0]['i2t'][0])}"),
        ("tokens leaving the transformer, hs", f"{len(r['hs'])} x {D}"),
        ("upscaled image features", f"{len(r['up'])} x {len(r['up'][0])} x {len(r['up'][0][0])}"),
        ("hypernetwork outputs, one per mask token", f"{len(r['hyper'])} x {len(r['hyper'][0])}"),
        ("mask logits", f"{len(r['masks'])} x {len(r['masks'][0])} (= 4 maps of 8 x 8)"),
        ("predicted overlaps", f"{len(r['iou'])}"),
    ]
    return ("The prompt is the one foreground click of the previous tensors. Shapes (rows x columns) through the decoder.\n\n"
            + _table(["quantity", "shape"], rows, "ll"))


def decoder_attention():
    t = toy()
    r = decode(t, embed_points(t, [(2, 2)], [1]))
    out = []
    for name, i in (("token-to-image, layer 0", 0), ("token-to-image, layer 1", 1)):
        w = r["taps"][i]["t2i"][5]           # token 5 is the click
        out.append(f"{name}, the click token's weights over the 4 x 4 cells (row-major; each row of the map sums to 1):\n\n```text\n"
                   + _grid_text(w, EMB, 3) + f"\n```\n\nsum = {_fx(_sum(w), 6)}")
    w = r["taps"][2]["final"][5]
    out.append("final token-to-image, the click token:\n\n```text\n" + _grid_text(w, EMB, 3) + "\n```")
    w = r["taps"][0]["i2t"]
    col = [row[5] for row in w]
    out.append("image-to-token, layer 0: the weight that each of the 16 cells gives to the click token (each cell's row over the 7 "
               "tokens sums to 1; the column is not normalized):\n\n```text\n" + _grid_text(col, EMB, 3) + "\n```")
    return "\n\n".join(out)


def decoder_masks():
    t = toy()
    r = decode(t, embed_points(t, [(2, 2)], [1]))
    out = []
    for i in range(4):
        out.append(f"mask token {i}, logits at the 8 x 8 output positions:\n\n```text\n" + _grid_text(r["masks"][i], 2 * EMB, 2) + "\n```")
    p = 3 * 2 * EMB + 5                      # output pixel (row 3, column 5)
    terms = [r["hyper"][1][c] * r["flat"][c][p] for c in range(D // 4)]
    s = 0.0
    for v in terms:
        s += v
    out.append(f"The logit of mask token 1 at pixel (row 3, column 5) is the dot product of its hypernetwork output "
               f"{_vec(r['hyper'][1], 4)} with the upscaled features at that pixel {_vec([r['flat'][c][p] for c in range(D // 4)], 4)}: "
               f"{' + '.join(_fx(v, 4) for v in terms)} = {_fx(s, 4)}; the table above has {_fx(r['masks'][1][p], 4)}.")
    out.append("Predicted overlaps of the four mask tokens (sigmoid of the IoU head): " + _vec(r["iou"], 4) + ". With several clicks the "
               "model uses token 0 alone; with one click it returns tokens 1, 2 and 3, and the one with the highest predicted overlap "
               f"is token {1 + max(range(3), key=lambda k: r['iou'][1 + k])} here.")
    return "\n\n".join(out)


def decoder_prompts():
    t = toy()
    rows = []
    cases = [("click", [(2, 2)], [1]), ("click and background click", [(2, 2), (6, 3)], [1, 0]), ("box", [(1, 1), (4, 4)], [2, 3])]
    base = decode(t, embed_points(t, *cases[0][1:]))
    for name, pts, labs in cases:
        r = decode(t, embed_points(t, pts, labs))
        diff = max(abs(a - b) for a, b in zip(r["masks"][1], base["masks"][1]))
        pos = sum(1 for v in r["masks"][1] if v > 0)
        rows.append((name, str(len(r["tokens"])), str(pos), _fx(diff, 3)))
    return ("The same weights and the same image embedding; only the prompt changes. Mask token 1. The click is (x 2, y 2) with label 1; "
            "the background click is (x 6, y 3) with label 0; the box has corners (x 1, y 1) with label 2 and (x 4, y 4) with label 3. "
            "Every prompt gets the padding point, and the difference is the largest absolute difference over the 64 logits.\n\n"
            + _table(["prompt", "tokens", "pixels with logit > 0", "largest difference of logit from the one-click map"], rows))


# ------------------------------------------------------------------------------------------------ relative position in the encoder

NAME_RELPOS = "segment_anything.relpos"


def relative_bias():
    g = bookrng.generator(NAME_RELPOS)
    w, hd = 3, 4
    Rh = [[_round1(g.normal()) for _ in range(hd)] for _ in range(2 * w - 1)]
    Rw = [[_round1(g.normal()) for _ in range(hd)] for _ in range(2 * w - 1)]
    q = [_round1(g.normal()) for _ in range(hd)]
    qh, qw = 1, 1
    rel_h = [_dot(q, Rh[qh - kh + w - 1]) for kh in range(w)]
    rel_w = [_dot(q, Rw[qw - kw + w - 1]) for kw in range(w)]
    bias = [[rel_h[kh] + rel_w[kw] for kw in range(w)] for kh in range(w)]
    full = [[_dot(q, Rh[qh - kh + w - 1]) + _dot(q, Rw[qw - kw + w - 1]) for kw in range(w)] for kh in range(w)]
    grid = lambda M: "```text\n" + "\n".join("  ".join(f"{v:6.2f}" for v in r) for r in M) + "\n```"
    t_glob = 64 * 64
    slab = 12 * t_glob * t_glob * 4
    hoisted = 12 * t_glob * (64 + 64) * 4
    return (f"{_seed(NAME_RELPOS)}. Window of {w} by {w} tokens, head width {hd}. Draws, in this order: the table Rh "
            f"({2 * w - 1} rows of {hd}), the table Rw ({2 * w - 1} rows of {hd}), then the query q ({hd}), each entry `round(normal(), 1)`. "
            f"Rh = {[list(map(float, r)) for r in Rh]}, Rw = {[list(map(float, r)) for r in Rw]}, q = {list(map(float, q))}. "
            f"The query sits at row {qh}, column {qw} of the window. The bias it adds to the score of the key at (kh, kw) is "
            "q . Rh[qh - kh + 2] + q . Rw[qw - kw + 2].\n\n"
            f"The two small vectors, one entry per key row and per key column: rel_h = {_vec(rel_h, 2)}, rel_w = {_vec(rel_w, 2)}.\n\n"
            "The bias map over the 3 by 3 keys, as the sum rel_h[kh] + rel_w[kw]:\n\n" + grid(bias) +
            f"\n\nEvaluating the formula pair by pair, 9 times, gives the same map; the largest difference is "
            f"{max(abs(a - b) for ra, rb in zip(bias, full) for a, b in zip(ra, rb)):.1f}.\n\n"
            "Storage at the scale of a global block of the first SAM's ViT-B tower (64 by 64 tokens, 12 heads, 4-byte numbers):\n\n"
            + _table(["quantity", "entries", "bytes"], [
                ("bias slab, one number per head and (query, key) pair", _int(12 * t_glob * t_glob), _int(slab)),
                ("two small tables per head and query, (64 + 64) numbers", _int(12 * t_glob * 128), _int(hoisted)),
                ("ratio", "", _fx(slab / hoisted, 1)),
                ("learned tables of the block (127 rows of 64 for each axis, 12 heads share them)", _int(2 * 127 * 64), _int(2 * 127 * 64 * 4)),
            ]))


NAME_PAD = "segment_anything.pad"


def pad_keys():
    g = bookrng.generator(NAME_PAD)
    r1 = lambda: _round1(g.normal())
    x = [[r1(), r1()] for _ in range(3)]                       # three real tokens (after the norm), width 2
    q = [r1(), r1()]
    Wk = [[_round1(0.5 * g.normal()) for _ in range(2)] for _ in range(2)]
    bk = [r1(), r1()]
    Wv = [[_round1(0.5 * g.normal()) for _ in range(2)] for _ in range(2)]
    bv = [r1(), r1()]
    keys = [[_dot(xi, [Wk[a][c] for a in range(2)]) + bk[c] for c in range(2)] for xi in x]
    vals = [[_dot(xi, [Wv[a][c] for a in range(2)]) + bv[c] for c in range(2)] for xi in x]
    keys_pad = keys + [list(bk)]                               # the pad token enters the projection as zeros: its key is bk
    vals_pad = vals + [list(bv)]
    sc = lambda ks: [_dot(q, k) / math.sqrt(2) for k in ks]
    w_real = _softmax(sc(keys))
    w_pad = _softmax(sc(keys_pad))
    out = lambda w, vs: [_sum([w[i] * vs[i][c] for i in range(len(w))]) for c in range(2)]
    rows = [("masked pad (3 keys)", _vec(w_real, 3) + ", 0", _vec(out(w_real, vals), 3)),
            ("pad as the layer computes it (4 keys)", _vec(w_pad, 3), _vec(out(w_pad, vals_pad), 3))]
    return (f"{_seed(NAME_PAD)}. Draws, in this order: the 6 entries of the three tokens x (row by row) and the 2 of the query q, each `round(normal(), 1)`; "
            "Wk (2 x 2) as `round(0.5 * normal(), 1)`, then bk (2) as `round(normal(), 1)`; Wv as Wk, then bv as bk. "
            f"x = {[list(map(float, r)) for r in x]}, q = {list(map(float, q))}, Wk = {[list(map(float, r)) for r in Wk]}, "
            f"bk = {list(map(float, bk))}, Wv = {[list(map(float, r)) for r in Wv]}, bv = {list(map(float, bv))}. A key is "
            "x Wk + bk, a value x Wv + bv, the score q . k / sqrt(2). The pad token's input is the zero vector, so its key is bk and "
            "its value bv.\n\n"
            + _table(["window", "weights on the 3 real tokens and the pad", "output"], rows, "lll"))


# ------------------------------------------------------------------------------------------------ the encoder trunks and the counted work

HIERA_L = {"embed": 144, "heads": 2, "stages": (2, 6, 36, 4), "global": (23, 33, 43), "windows": (8, 4, 16, 8), "image": 1024,
           "patch": 7, "stride": 4, "pool": 3, "mlp": 4}


def hiera_blocks(cfg=HIERA_L):
    """The per-block table by the rule of the chapter: widths and heads double and the grid halves at the first block of stages
    2 to 4, a block's window is the previous stage's at its first block, and the global blocks have no window."""
    ends, tot = [], 0
    for s in cfg["stages"]:
        tot += s
        ends.append(tot - 1)
    pools = [e + 1 for e in ends[:-1]][:cfg["pool"]]
    dim, heads, stage, h = cfg["embed"], cfg["heads"], 1, cfg["image"] // cfg["stride"]
    out = []
    for i in range(tot):
        dim_out = dim
        ws = cfg["windows"][stage - 1]
        if i in cfg["global"]:
            ws = 0
        if i > 0 and (i - 1) in ends:
            dim_out = dim * 2
            heads *= 2
            stage += 1
        pooled = i in pools
        h_out = h // 2 if pooled else h
        out.append({"i": i, "dim": dim, "dim_out": dim_out, "heads": heads, "ws": ws, "pool": pooled, "h_in": h, "h_out": h_out,
                    "stage": stage})
        dim, h = dim_out, h_out
    return out


def hiera_flops(blocks, cfg=HIERA_L):
    """FLOPs (a multiply-add is 2) of the trunk: the patch embedding and, per block, the optional shortcut projection, the
    qkv projection, the score and weighted-average products, the output projection and the MLP. Norms, activations, the
    pooling and the rotation-free relative terms are not counted."""
    out = cfg["image"] // cfg["stride"]
    total = {"patch": 2 * out * out * cfg["patch"] ** 2 * 3 * cfg["embed"], "proj": 0, "qkv": 0, "attn": 0, "out": 0, "mlp": 0}
    per_block = []
    for b in blocks:
        t_in, t_out, c, co = b["h_in"] ** 2, b["h_out"] ** 2, b["dim"], b["dim_out"]
        if b["ws"]:
            assert b["h_in"] % b["ws"] == 0, "the Hiera-L windows divide every grid, so there is no padding to count"
            n_win = (b["h_in"] // b["ws"]) ** 2
            keys = b["ws"] ** 2
            queries_per = keys // 4 if b["pool"] else keys
            attn = 4 * n_win * queries_per * keys * co
        else:
            attn = 4 * t_out * t_in * co
        parts = {"proj": 2 * t_in * c * co if c != co else 0, "qkv": 2 * t_in * c * 3 * co, "attn": attn,
                 "out": 2 * t_out * co * co, "mlp": 2 * 2 * t_out * co * cfg["mlp"] * co}
        for k, v in parts.items():
            total[k] += v
        per_block.append(sum(parts.values()))
    return total, per_block


def hiera_stages():
    blocks = hiera_blocks()
    total, per_block = hiera_flops(blocks)
    rows = []
    for s in range(1, 5):
        bs = [b for b in blocks if b["stage"] == s]       # the first block of stages 2-4 belongs to the stage whose width it outputs
        flo = sum(per_block[b["i"]] for b in bs)
        wins = sorted({b["ws"] for b in bs})
        rows.append((str(s), str(len(bs)), f"{bs[-1]['h_out']} x {bs[-1]['h_out']}", str(bs[-1]["dim_out"]), str(bs[-1]["heads"]),
                     ", ".join("global" if w == 0 else str(w) for w in wins), _giga(flo)))
    return ("The Hiera trunk of the large SAM 2.1 model (`sam2.1_hiera_l`): patch embedding a 7 by 7 convolution of stride 4 and "
            "3 -> 144 channels on the 1,024 by 1,024 picture; 48 blocks in stages of 2, 6, 36 and 4; the width and the number of "
            "heads double, and the grid halves (a 2 by 2 max pool of the queries), at the first block of stages 2, 3 and 4; the "
            "windows of the stages are 8, 4, 16 and 8 and a block's window at the first block of a stage is that of the stage before; "
            "blocks 23, 33 and 43 (numbering the blocks from 0) are global; an MLP ratio 4. FLOPs count 2 per multiply-add. Per block, "
            "with N_in cells at the block's input, N_out at its output (a quarter of N_in at the first block of stages 2 to 4), c the "
            "input width and c' the output width: the shortcut projection (only where c' differs from c) 2 N_in c c'; the qkv projection "
            "2 N_in c (3c'), taken on the input grid and already at the output width; the output projection 2 N_out c' c'; the MLP "
            "2 * 2 N_out c' (4c'); the products of a windowed block 4 (N_in/w^2) (queries per window) (w^2) c', with w^2 queries per "
            "window and w^2/4 at the first block of a stage (the pooled queries), and of a global block 4 N_out N_in c'. The window w "
            "of a stage's first block is that of the stage before, and every window divides its grid, so nothing is padded.\n\n"
            + _table(["stage", "blocks", "output grid", "width", "heads", "window(s)", "GFLOPs"], rows, "lrrrrrr")
            + f"\n\nPatch embedding: {_giga(total['patch'])} billion FLOPs. Trunk in all: {_giga(sum(total.values()), 4)} billion FLOPs "
              f"(shortcut projections {_giga(total['proj'])}, qkv {_giga(total['qkv'])}, attention products {_giga(total['attn'])}, "
              f"output projections {_giga(total['out'])}, MLPs {_giga(total['mlp'])}).")


def vit_b_flops():
    cfg = {"d": 768, "layers": 12, "grid": 64, "win": 14, "global": (2, 5, 8, 11), "ffn": 3072}
    d, g, m = cfg["d"], cfg["grid"], cfg["win"]
    t = g * g
    padded = -(-g // m) * m
    tp = padded * padded
    n_win = (padded // m) ** 2
    total = 2 * t * 16 * 16 * 3 * d
    parts = {"patch": total, "qkv": 0, "attn": 0, "proj": 0, "mlp": 0}
    for l in range(cfg["layers"]):
        glob = l in cfg["global"]
        rows = t if glob else tp
        parts["qkv"] += 2 * rows * d * 3 * d
        parts["attn"] += 4 * (t * t if glob else n_win * (m * m) ** 2) * d
        parts["proj"] += 2 * rows * d * d
        parts["mlp"] += 2 * 2 * t * d * cfg["ffn"]
    neck = 2 * t * d * 256 + 2 * t * 256 * 256 * 9
    parts["neck"] = neck
    return parts, tp, n_win


def decoder_flops(n_prompt=1, width=256, io=128, mlp=2048, grid=64, obj_score=True):
    """FLOPs of one call of the SAM 2 mask decoder with `n_prompt` prompt tokens (a click gives 1 point plus the padding point).

    Counts the projections, scores and weighted averages of the 2 two-way layers and the final attention, the token MLPs,
    the two transposed convolutions of the upscaling, the hypernetworks and the dot products of the mask logits."""
    d, n = width, grid * grid
    t = (1 if obj_score else 0) + 1 + 4 + n_prompt + 1
    proj = lambda rows, i, o: 2 * rows * i * o
    self_attn = 4 * proj(t, d, d) + 2 * 2 * t * t * d
    t2i = proj(t, d, io) + 2 * proj(n, d, io) + proj(t, io, d) + 2 * 2 * t * n * io
    i2t = proj(n, d, io) + 2 * proj(t, d, io) + proj(n, io, d) + 2 * 2 * n * t * io
    layer = self_attn + t2i + 2 * proj(t, d, mlp) + i2t
    transformer = 2 * layer + t2i
    up1 = 2 * n * d * (d // 4) * 4                       # 64 x 64 inputs, 4 taps each, 256 -> 64 channels
    up2 = 2 * (4 * n) * (d // 4) * (d // 8) * 4          # 128 x 128 inputs, 64 -> 32 channels
    out_px = 16 * n
    hyper = 4 * (proj(1, d, d) * 2 + proj(1, d, d // 8))
    dots = 2 * 4 * (d // 8) * out_px
    heads = proj(1, d, d) * 2 + proj(1, d, 4) + proj(1, d, d) * 2 + proj(1, d, 1)
    return {"transformer": transformer, "upscaling": up1 + up2, "hypernetworks": hyper, "mask dot products": dots, "heads": heads}


def encoder_vs_decoder():
    blocks = hiera_blocks()
    tot, _ = hiera_flops(blocks)
    hiera = sum(tot.values())
    parts, tp, n_win = vit_b_flops()
    vitb = sum(parts.values())
    dec = decoder_flops(1)
    decsum = sum(dec.values())
    mem_bytes = 4 * (256 * 64 * 64 + 32 * 256 * 256 + 64 * 128 * 128)
    rows = [
        ("Hiera-L trunk (SAM 2.1), once per image", _giga(hiera, 4), "1.00"),
        ("ViT-B tower with neck (the SAM 1 tower of brain), once per image", _giga(vitb, 4), _fx(vitb / hiera, 2)),
        ("mask decoder of SAM 2, one click, once per prompt", _giga(decsum, 4), _fx(decsum / hiera, 4)),
    ]
    drows = [(k, f"{v / 1e6:,.1f}", _fx(100 * v / decsum, 1)) for k, v in dec.items()]
    n_prompts = 8
    return ("FLOPs count 2 per multiply-add; norms, activations and the relative-position terms are not counted. The ViT-B tower: a "
            "16 by 16 patch embedding of stride 16 on 1,024 by 1,024 (4,096 tokens, width 768), 12 blocks of 12 heads and an MLP of "
            "3,072, global attention in blocks 2, 5, 8 and 11 (numbered from 0) and windows of 14 elsewhere; the qkv and output "
            "projections run on the rows the block holds (4,900 padded rows in a windowed block, 4,096 in a global one), the MLP "
            "on the 4,096 real tokens; the neck is a 1 x 1 convolution 768 -> 256 and a 3 x 3 convolution 256 -> 256 on 64 by 64. "
            "The decoder, with 8 tokens and the 4,096 cells: the self-attention (four 256 x 256 projections and the products), the "
            "token-to-image and image-to-token attentions (projections 256 -> 128 of the tokens or the cells, 128 -> 256 back, and "
            "their products at width 128), a token MLP 256 -> 2,048 -> 256, two layers and the final token-to-image attention; the two "
            "transposed convolutions 256 -> 64 on 64 by 64 and 64 -> 32 on 128 by 128; four hypernetworks of three layers (256, 256, "
            "256 and 32); the dot products with the 32 channels at 256 by 256; and the IoU and object-score heads.\n\n"
            + _table(["quantity", "GFLOPs", "relative to the Hiera-L trunk"], rows, "lrr")
            + "\n\nThe decoder by part (one click: 1 point token and the padding token, 4 mask tokens, the IoU token and the object-score "
              "token, 8 tokens in all):\n\n" + _table(["part", "MFLOPs", "share (%)"], drows, "lrr")
            + f"\n\nThe ViT-B tower pads its 64 by 64 grid to {int(math.sqrt(tp))} by {int(math.sqrt(tp))} for windows of 14 "
              f"({tp:,} rows in {n_win} windows, against {64 * 64:,}). Features the decoder needs from one image, in 32-bit numbers: "
              f"the image embedding 256 x 64 x 64, the two high-resolution maps 32 x 256 x 256 and 64 x 128 x 128: {mem_bytes:,} bytes "
              f"({mem_bytes / 2 ** 20:.0f} MiB).\n\n"
              + _table(["prompts on one image", "work (GFLOPs), encoder once", "work (GFLOPs), encoder per prompt"],
                       [(str(k), _giga(hiera + k * decsum), _giga(k * (hiera + decsum))) for k in (1, 2, n_prompts)]))


# ------------------------------------------------------------------------------------------------ the video memory bank

NUM_MASKMEM = 7
MAX_PTRS = 16
MEM_DIM = 64
D_MODEL = 256


def frame_memory(frame, cond=(0,), num_frames=64):
    """The memories frame `frame` reads (stride 1): (t_pos, source frame) per slab, and (frame difference) per object pointer.

    Frame 0..: the conditioning frames always sit at t_pos 0; the recent non-conditioning frames fill t_pos = 1 .. 6 as the frames
    `frame - 6 .. frame - 1` that exist and are not conditioning frames. A pointer comes from each conditioning frame at or before
    the frame and from each of the frames `frame - 1 .. frame - 15` that has one."""
    slabs = [(0, c) for c in cond]
    for t_pos in range(1, NUM_MASKMEM):
        t_rel = NUM_MASKMEM - t_pos
        prev = frame - t_rel
        if prev >= 0 and prev not in cond:
            slabs.append((t_pos, prev))
    max_ptrs = min(MAX_PTRS, num_frames)
    ptrs = [frame - c for c in cond if c <= frame]
    for dt in range(1, max_ptrs):
        t = frame - dt
        if t < 0:
            break
        if t not in cond:
            ptrs.append(dt)
    return slabs, ptrs


def memory_selection():
    rows = []
    for f in (1, 2, 3, 7, 8, 12, 20):
        slabs, ptrs = frame_memory(f)
        tk = len(slabs) * 4096 + 4 * len(ptrs)
        rows.append((str(f), ", ".join(f"{s}" for _, s in slabs), ", ".join(str(tp) for tp, _ in slabs), str(len(ptrs)),
                     _int(tk)))
    return ("A clip of 64 frames, one prompt on frame 0 (the only conditioning frame), the large model: 7 memory slots, 16 object "
            "pointers at most, 4,096 cells per slab (a 64 by 64 grid) and 4 tokens per pointer (256 / 64). Frame t reads the pointer of "
            "the prompted frame 0 and of each of the frames t - 1 down to t - 15 that exists and is not prompted, so it reads "
            "min(t, 16) pointers; the keys are 4,096 times the slabs plus 4 times the pointers.\n\n"
            + _table(["frame t", "source frames of the slabs", "temporal positions", "pointers", "keys of the cross-attention"], rows, "lllrr"))


def memory_attention_flops(keys, n=4096, d=256, kv=64, layers=4, ff=2048):
    """FLOPs of the 4 memory-attention layers on n query cells and `keys` memory tokens (single head of width d)."""
    sa_ = 4 * 2 * n * d * d + 2 * 2 * n * n * d
    ca = 2 * n * d * d + 2 * 2 * keys * kv * d + 2 * n * d * d + 2 * 2 * n * keys * d
    mlp = 2 * 2 * n * d * ff
    return layers * (sa_ + ca + mlp), layers * sa_, layers * ca, layers * mlp


def memory_encoder_flops(grid=64, d=256, image=1024, mem=64):
    chans = (1, 4, 16, 64, 256)
    total, side = 0, image
    for i in range(4):
        side //= 2
        total += 2 * side * side * chans[i] * chans[i + 1] * 9
    total += 2 * grid * grid * d * d                                       # the 1 x 1 convolution after the downsampler
    total += 2 * grid * grid * d * d                                       # pix_feat_proj
    total += 2 * (2 * grid * grid * d * 49 + 2 * 2 * grid * grid * d * 4 * d)   # two ConvNeXt blocks: depthwise 7 x 7, two pointwise
    total += 2 * grid * grid * d * mem
    return total


def frame_costs():
    blocks = hiera_blocks()
    trunk = sum(hiera_flops(blocks)[0].values())
    dec = sum(decoder_flops(1).values())
    memenc = memory_encoder_flops()
    rows = []
    for f in (1, 2, 8, 20):
        slabs, ptrs = frame_memory(f)
        tk = len(slabs) * 4096 + 4 * len(ptrs)
        mem, sa_, ca, _ = memory_attention_flops(tk)
        tot = trunk + mem + dec + memenc
        rows.append((str(f), _int(tk), _giga(trunk, 3), _giga(mem, 3), _giga(dec, 3), _giga(memenc, 3), _giga(tot, 3),
                     _fx(100 * mem / tot, 1)))
    return ("GFLOPs for one tracked frame of the large model (2 per multiply-add). The decoder runs with the tracking prompt: no click, "
            "the padding point twice (8 tokens, as for one click). Memory attention: 4 layers on 4,096 cells, a single head of width 256 "
            "whose cross-attention keys and values come from 64-wide memory tokens; the memory encoder: a mask downsampler of 4 "
            "stride-2 convolutions (1, 4, 16, 64, 256 channels, kernel 3, on 1,024 by 1,024 down to 64 by 64), a 1 x 1 convolution 256 -> 256 "
            "after the downsampler and a 1 x 1 projection 256 -> 256 of the frame's features, two ConvNeXt blocks "
            "(depthwise 7 x 7 and pointwise layers 256 -> 1,024 -> 256) and a 1 x 1 convolution to 64. In each memory-attention layer "
            "the self-attention has four 256 x 256 projections, the cross-attention has 256 x 256 projections of the queries and of "
            "the output, 64 -> 256 projections of the keys and the values of the memory tokens, and the feed-forward network is "
            "256 -> 2,048 -> 256. Totals are the sum of the unrounded counts, so the printed columns can differ from the total by 0.001.\n\n"
            + _table(["frame", "memory keys", "trunk", "memory attention", "decoder", "memory encoder", "total", "memory attention (%)"],
                     rows))


def mask_to_memory():
    rows = []
    for logit in (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0):
        s = 1.0 / (1.0 + math.exp(-logit))
        rows.append((_fx(logit, 1), _fx(s, 4), _fx(s * 20.0 - 10.0, 3)))
    return ("A mask logit enters the memory encoder as 20 * sigmoid(logit) - 10 (library exp).\n\n"
            + _table(["mask logit", "sigmoid", "value given to the memory encoder"], rows))


def occlusion_rules():
    rows = [
        ("object score > 0", "the best mask's logits", "the pointer projected from the output token", "none"),
        ("object score <= 0", "every logit replaced by -1024", "the learned no_obj_ptr", "the learned no_obj_embed_spatial added to every cell"),
    ]
    sigma = tb.sigmoid(-1024.0)
    return (_table(["object score", "mask", "object pointer", "memory features"], rows, "llll")
            + f"\n\nA logit of -1024 gives the probability sigmoid(-1024) = {sigma:g} (the stable form of the sigmoid, "
              "`exp(z) / (1 + exp(z))` for negative z, underflows to 0), and 20 * 0 - 10 = -10 in the memory: the same value as the most confident background.")


# ------------------------------------------------------------------------------------------------ memory attention on a 3 x 3 grid

GRID = 3
HEAD = 8


def axial_rope(x, px, py, theta=10000.0):
    """Axial rotary embedding of a vector of width 8 in the interleaved pairing (2j, 2j + 1): the first 2 pairs turn by px times
    the rates theta^(-4j/8), j = 0, 1 (1 and 0.01), the last 2 pairs by py times the same rates."""
    out = list(x)
    rates = [theta ** (-4.0 * j / HEAD) for j in range(HEAD // 4)]
    for pair in range(HEAD // 2):
        pos = px if pair < HEAD // 4 else py
        w = rates[pair % (HEAD // 4)]
        c, s = math.cos(pos * w), math.sin(pos * w)
        a, b = x[2 * pair], x[2 * pair + 1]
        out[2 * pair] = a * c - b * s
        out[2 * pair + 1] = a * s + b * c
    return out


def _cell_features(obj_cell, masked_cell, with_mask):
    feats = []
    for t in range(GRID * GRID):
        a = 1.4 if t == obj_cell else -1.4
        m = 2.0 if (with_mask and t == masked_cell) else 0.0
        feats.append([a, 0.0, m, 0.0, a, 0.0, 0.0, 0.0])
    return feats


def memory_toy_data(use_rope):
    mem_obj, cur_obj = 4, 5                 # token t sits at x = t mod 3, y = t div 3: the object moves one cell to the right
    mem = _cell_features(mem_obj, mem_obj, True)
    cur = _cell_features(cur_obj, None, False)
    pos = lambda t: (t % GRID, t // GRID)
    weights, out = [], []
    for qt in range(GRID * GRID):
        q = axial_rope(cur[qt], *pos(qt)) if use_rope else cur[qt]
        sc = []
        for kt in range(GRID * GRID):
            k = axial_rope(mem[kt], *pos(kt)) if use_rope else mem[kt]
            sc.append(_dot(q, k) / math.sqrt(HEAD))
        w = _softmax(sc)
        weights.append(w)
        out.append(_sum([w[kt] * mem[kt][2] for kt in range(GRID * GRID)]))
    return weights, out


def memory_toy():
    parts = []
    for use_rope in (False, True):
        w, o = memory_toy_data(use_rope)
        title = "with the axial rotary embedding on queries and keys" if use_rope else "without position (content only)"
        parts.append(f"Attention {title}. Weights of the query at the object's new cell (token 5), over the 9 memory cells:\n\n```text\n"
                     + _grid_text(w[5], GRID, 3) + "\n```\n\nThe mask channel read back at each of the 9 current cells:\n\n```text\n"
                     + _grid_text(o, GRID, 3) + "\n```")
    return ("A 3 by 3 grid, token t at column x = t mod 3 and row y = t div 3. A cell's feature has width 8: "
            "(a, 0, m, 0, a, 0, 0, 0) with the appearance a = 1.4 on the object and -1.4 elsewhere in both the x and y channels, and "
            "the mask channel m = 2 on the memory cell that the earlier mask covered and 0 everywhere else. Memory: the object (and its "
            "mask) at token 4, column 1 and row 1. Current frame: the object has moved to token 5 (column 2, row 1), and the current "
            "cells carry no mask channel. The query, key and value projections are the identity, the score is q . k / sqrt(8), and the "
            "rotation (when used) is applied to queries and keys by the cell's (x, y); the values are the unrotated memory features. "
            "The rotation pairs adjacent channels (0, 1), (2, 3), (4, 5), (6, 7), unlike the half-split pairing of Chapter 6's "
            "toy; pairs 0 and 1 turn by x times the rates 1 and 0.01 (10000^(-4j/8) for j = 0, 1) and pairs 2 and 3 turn by y times "
            "the same two rates, a pair (a, b) going to (a cos - b sin, a sin + b cos).\n\n" + "\n\n".join(parts))


# ------------------------------------------------------------------------------------------------ figure data


def attention_flops_split():
    """(label, GFLOPs) of one frame's work by part at frame 20, for a figure of the cost of tracking."""
    blocks = hiera_blocks()
    trunk = sum(hiera_flops(blocks)[0].values())
    slabs, ptrs = frame_memory(20)
    tk = len(slabs) * 4096 + 4 * len(ptrs)
    mem = memory_attention_flops(tk)[0]
    return [("trunk", trunk), ("memory attention", mem), ("decoder", sum(decoder_flops(1).values())),
            ("memory encoder", memory_encoder_flops())]


def architecture_data():
    """GFLOPs (and bytes) shown on the architecture figure: the Hiera-L trunk once per image, the decoder per click."""
    trunk = sum(hiera_flops(hiera_blocks())[0].values())
    dec = sum(decoder_flops(1).values())
    return {"trunk": trunk, "decoder": dec, "cache_bytes": 4 * (256 * 64 * 64 + 32 * 256 * 256 + 64 * 128 * 128)}


def bank_figure_data(frame=12):
    """The memories frame `frame` reads: (frames, slab sources with their temporal positions, frames that contribute a pointer)."""
    slabs, ptrs = frame_memory(frame)
    ptr_frames = sorted(frame - dt for dt in ptrs)
    return {"frame": frame, "slabs": slabs, "ptr_frames": ptr_frames, "keys": len(slabs) * 4096 + 4 * len(ptrs), "n_ptrs": len(ptrs)}
