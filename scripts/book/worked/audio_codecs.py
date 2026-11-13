# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Neural Audio Codecs".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: audio_codecs.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree. Earlier modules are imported, never copied: the table helper of `sound` and the stable-digits rule
of `stable`.

* the bit arithmetic of a token stream against raw samples;
* a stack of strided convolutions: lengths, left padding, the rate at which frames come out, the receptive field, and the identity
  that turns five nested ceilings into one;
* a residual vector quantizer small enough to follow by hand, and the same quantizer trained stage by stage on a stated data set
  with the error after each stage, set beside one flat codebook with the same number of cells;
* the cost of the nearest-code search against the number of cells it addresses;
* the parameters and the multiply-adds of one second of audio through the encoder and the decoder of the codec brain runs, counted
  from the layer list of its configuration, and the state a streaming decoder must carry;
* the sizes and the loss of the reconstruction trainer in brain's codec crate.

Arithmetic is the language's 64-bit floating point. The only random numbers are those of the residual quantizer's data and of the
initial centres, from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. A trained
number is printed with the digits that agree between the run and four copies whose first centre is changed by the nudges of
`stable.py`. The ten trainings are run in at most four worker processes.

Swedish Embedded AB implements solutions for turning audio into discrete tokens that a language model can read and write, with
the bit budget counted before anything is trained. If your team needs expertise in neural audio codecs and residual quantization,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache

import bookrng
import sound
import stable

_table = sound._table

FS = 24000                       # sampling rate of the codec, Hz

# ---------------------------------------------------------------- the codec's configuration (the fields the code reads)
STRIDES = (4, 5, 6, 8)           # encoder downsampling stages, in the order they are applied
FILTERS = 64                     # width of the first encoder convolution
K_HEAD, K_RES, K_LAST = 7, 3, 3  # kernels of the first, the residual and the last encoder convolution
COMPRESS = 2                     # the residual branch is this many times narrower than its input
HIDDEN = 512                     # width after the encoder convolutions
DOWN_STRIDE, DOWN_K = 2, 4       # frame-rate match: stride, and kernel = 2 * int(25 / 12.5)
ENC_LAYERS, ENC_HEADS, ENC_HEAD_DIM, ENC_FFN = 8, 8, 64, 2048
CODE_DIM, BINS, NQ, N_SEM = 256, 2048, 16, 1   # codebook width, entries per codebook, codebooks, semantic codebooks

LAT = 512                        # decoder: width after the quantizer's output projections
LATENT = 1024                    # decoder: width of the pre-convolution and of the upsampling stages
DEC_HIDDEN, DEC_FFN, DEC_LAYERS, DEC_HEADS, DEC_HEAD_DIM, WINDOW = 512, 1024, 8, 16, 64, 72
ENC_WINDOW = 250
UP = (2, 2)                      # transposed convolutions between the frame rate and the SEANet decoder
RATES = (8, 5, 4, 3)             # SEANet decoder upsampling rates
DEC_DIM = 1536                   # width of the first SEANet decoder convolution

HOP = 1
for _s in STRIDES:
    HOP *= _s
HOP *= DOWN_STRIDE               # 1920 samples per frame
FRAME_RATE = FS / HOP            # 12.5 frames per second
CODE_BITS = int(math.log2(BINS))  # 11 bits per code


def _f(x, d=4):
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _i(n):
    return f"{n:,}"


def _seed(name):
    return f"`{name}` (seed 0x{bookrng.seed_of(name):016x})"


def _ceil_div(a, b):
    return -(-a // b)


# ================================================================= the bit arithmetic

def token_rates():
    rows = []
    for name, rate, per, bits in [("raw samples, 24 kHz, 16 bit", FS, 1, 16), ("raw samples, 16 kHz, 16 bit", 16000, 1, 16),
                                  ("brain's codec: 16 codebooks of 2048", FRAME_RATE, NQ, CODE_BITS),
                                  ("the same frames with 8 codebooks (illustration)", FRAME_RATE, 8, CODE_BITS),
                                  ("the same frames with 1 codebook (illustration)", FRAME_RATE, 1, CODE_BITS)]:
        bits_s = rate * per * bits
        tokens = rate * per
        rows.append((name, f"{rate:g}", per, bits, f"{bits_s:,.1f}".rstrip("0").rstrip("."), f"{tokens:,.1f}",
                     f"{FS * 16 / bits_s:.1f}"))
    return ("Bits per second = frames (or samples) per second x numbers per frame x bits per number; a codebook of 2048 entries costs "
            "log2 2048 = 11 bits per code. The last column divides the bits of 24 kHz, 16-bit samples by the row's bits.\n\n"
            + _table(["stream", "frames or samples per second", "numbers per frame", "bits per number", "bits per second",
                      "numbers per second", "24 kHz PCM bits / row bits"], rows, "lrrrrrr"))


# ================================================================= strided convolutions

def encoder_layers():
    """The convolutions of the encoder, in order: (name, cin, cout, k, stride, bias, in_branch)."""
    out = [("head", 1, FILTERS, K_HEAD, 1, True, False)]
    c = FILTERS
    for i, r in enumerate(STRIDES):
        out.append((f"stage {i + 1} residual a", c, c // COMPRESS, K_RES, 1, True, True))
        out.append((f"stage {i + 1} residual b", c // COMPRESS, c, 1, 1, True, True))
        out.append((f"stage {i + 1} down by {r}", c, 2 * c, 2 * r, r, True, False))
        c *= 2
    out.append(("tail", c, HIDDEN, K_LAST, 1, True, False))
    return out


def _conv_out_len(length, stride):
    """Causal convolution of the codec: left padding k - stride (dilation 1), output ceil(length / stride)."""
    return _ceil_div(length, stride)


def encoder_lengths():
    L1, L2 = FS, FS // 2
    rows = []
    rf, jump = 1, 1
    l1, l2 = L1, L2
    for name, cin, cout, k, s, bias, branch in encoder_layers():
        rf += (k - 1) * jump
        if s > 1:
            l1, l2 = _conv_out_len(l1, s), _conv_out_len(l2, s)
            jump *= s
        rows.append((name, f"{cin} -> {cout}", k, s, k - s if s > 1 else k - 1, f"{l1:,}", f"{l2:,}", jump, f"{rf:,}"))
    rows.append(("after the transformer: down by 2", f"{HIDDEN} -> {HIDDEN}", DOWN_K, DOWN_STRIDE, DOWN_K - DOWN_STRIDE,
                 f"{_conv_out_len(l1, DOWN_STRIDE):,}", f"{_conv_out_len(l2, DOWN_STRIDE):,}", jump * DOWN_STRIDE, "(attention)"))
    return ("Each convolution is causal: zeros are laid on the left only, k - stride of them for a strided convolution and (k - 1) for "
            "a stride-1 one, and the output has ceil(length / stride) positions. 'hop' is the number of input samples between two "
            "neighbouring outputs. The receptive field is the number of samples of the clip that can change one output, "
            "rf += (k - 1) * hop before the layer, a residual branch counted once.\n\n"
            + _table(["layer", "channels", "kernel k", "stride", "zeros on the left", "length for 24,000 samples",
                      "length for 12,000 samples", "hop (samples)", "receptive field (samples)"], rows, "llrrrrrrr"))


def frame_alignment():
    rows = []
    for n in (1, 1920, 1921, 12000, 24000, 24001, 48000, 240000):
        t = n
        for s in (*STRIDES, DOWN_STRIDE):
            t = _conv_out_len(t, s)
        assert t == _ceil_div(n, HOP)
        out = t * HOP
        rows.append((f"{n:,}", f"{n / FS:.4f}", t, f"{out:,}", f"{out - n:,}", f"{(out - n) / FS * 1000:.1f}"))
    return ("A clip of n samples is cut by the five strides 4, 5, 6, 8 and 2 in turn, each output having ceil(length / stride) "
            "positions; the decoder returns 1920 samples for every frame.\n\n"
            + _table(["n (samples)", "seconds", "frames T", "decoded 1920 T", "extra samples", "extra (ms)"], rows, "rrrrrr"))


# ================================================================= residual vector quantization

def _nearest(C, x):
    best, bd = 0, float("inf")
    for k, c in enumerate(C):
        d = 0.0
        for a, b in zip(x, c):
            e = a - b
            d += e * e
        if d < bd:
            best, bd = k, d
    return best


def _sq(v):
    s = 0.0
    for a in v:
        s += a * a
    return s


def rvq_trace():
    base = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    books = [[(a * 0.5 ** q, b * 0.5 ** q) for a, b in base] for q in range(3)]
    out = ["Stage q has the four codes (+-0.5, +-0.5) times 0.5^q, listed in the order (-,-), (+,-), (-,+), (+,+); each stage "
           "takes the nearest code of the residual left by the stages before it (a tie goes to the lower index).", ""]
    for x in [(0.62, -0.35), (-0.18, 0.41)]:
        rows, r, recon = [], x, (0.0, 0.0)
        for q, C in enumerate(books):
            k = _nearest(C, r)
            c = C[k]
            r = (r[0] - c[0], r[1] - c[1])
            recon = (recon[0] + c[0], recon[1] + c[1])
            rows.append((q + 1, k, f"({_f(c[0], 3)}, {_f(c[1], 3)})", f"({_f(recon[0], 3)}, {_f(recon[1], 3)})",
                         f"({_f(r[0], 3)}, {_f(r[1], 3)})", _f(_sq(r), 5)))
        out.append(f"Vector x = ({x[0]}, {x[1]}), indices (" + ", ".join(str(r[1]) for r in rows) + ").")
        out.append("")
        out.append(_table(["stage q", "index", "code", "sum of codes so far", "residual", "squared error"], rows, "rrrrrr"))
        out.append("")
    return "\n".join(out).rstrip()


# --- the trained quantizer

D, N_TRAIN, N_TEST, Q, K, ITERS, K_FLAT = 6, 1024, 512, 4, 8, 20, 64
DATA_TRAIN, DATA_TEST, INIT_RVQ, INIT_FLAT = "audio_codecs.rvq_train", "audio_codecs.rvq_test", "audio_codecs.rvq_init", "audio_codecs.rvq_flat"
SCALES = tuple(1.0 / (1 + j) for j in range(D))


def _data(name, n):
    g = bookrng.generator(name)
    return [[g.normal() * SCALES[j] for j in range(D)] for _ in range(n)]


@lru_cache(maxsize=None)
def _sets():
    return _data(DATA_TRAIN, N_TRAIN), _data(DATA_TEST, N_TEST)


def _kmeans(X, k, perm, nudge):
    C = [list(X[i]) for i in perm[:k]]
    C[0][0] += nudge
    for _ in range(ITERS):
        sums = [[0.0] * D for _ in range(k)]
        counts = [0] * k
        for x in X:
            j = _nearest(C, x)
            counts[j] += 1
            s = sums[j]
            for a in range(D):
                s[a] += x[a]
        for j in range(k):
            if counts[j]:
                C[j] = [v / counts[j] for v in sums[j]]
    return C


def _mse(R):
    t = 0.0
    for r in R:
        t += _sq(r)
    return t / len(R)


def _usage(codes, k):
    counts = [0] * k
    for c in codes:
        counts[c] += 1
    n = len(codes)
    h = 0.0
    for c in counts:
        if c:
            h -= c / n * math.log(c / n)
    return sum(1 for c in counts if c), math.exp(h)


def _residuals(books, X):
    """Residual vectors after each stage, and the codes of the last pass, for the data X."""
    R = [list(x) for x in X]
    out = [_mse(R)]
    used = []
    for C in books:
        codes = [_nearest(C, r) for r in R]
        R = [[r[a] - C[c][a] for a in range(D)] for r, c in zip(R, codes)]
        out.append(_mse(R))
        used.append(_usage(codes, len(C)))
    return out, used


def _train_job(nudge):
    tr, te = _sets()
    g = bookrng.generator(INIT_RVQ)
    R = [list(x) for x in tr]
    books = []
    for _ in range(Q):
        perm = list(range(N_TRAIN))
        g.shuffle(perm)
        C = _kmeans(R, K, perm, nudge)
        books.append(C)
        R = [[r[a] - C[_nearest(C, r)][a] for a in range(D)] for r in R]
    rv_tr, used = _residuals(books, tr)
    rv_te, _ = _residuals(books, te)
    perm = list(range(N_TRAIN))
    bookrng.generator(INIT_FLAT).shuffle(perm)
    flat = _kmeans(tr, K_FLAT, perm, nudge)
    fl_tr, _ = _residuals([flat], tr)
    fl_te, _ = _residuals([flat], te)
    return {"rv_tr": rv_tr, "rv_te": rv_te, "used": used, "fl_tr": fl_tr[1], "fl_te": fl_te[1]}


@lru_cache(maxsize=None)
def _runs():
    with multiprocessing.get_context("fork").Pool(4) as pool:
        return pool.map(_train_job, [0.0, *stable.NUDGES])


def _cell(getter, d=4):
    rs = _runs()
    vals = [getter(r) for r in rs]
    return stable.text(vals[0], vals[1:], lambda x: _f(x, d))


def rvq_data_check():
    tr, te = _sets()
    rows = [(j, _f(SCALES[j], 4)) for j in range(D)]
    e_tr, e_te = _mse(tr), _mse(te)
    return ("Each of the six coordinates of a vector is a draw of normal() times the scale of its position; a vector takes its six "
            "draws in coordinate order and the vectors follow one another.\n\n"
            + _table(["coordinate j", "scale 1 / (1 + j)"], rows, "rr")
            + f"\n\nMean squared norm of a vector: {_f(e_tr)} on the {N_TRAIN} training vectors, {_f(e_te)} on the {N_TEST} held-out "
              f"ones (the expected value is {_f(sum(s * s for s in SCALES))}).")


def rvq_stages():
    rs = _runs()
    rows = [("0 (nothing sent)", 0, _cell(lambda r: r["rv_tr"][0]), "", _cell(lambda r: r["rv_te"][0]), "", "", "")]
    for q in range(1, Q + 1):
        ratio = _cell(lambda r, q=q: r["rv_tr"][q] / r["rv_tr"][q - 1])
        gain = _cell(lambda r, q=q: 10 * math.log10(r["rv_tr"][0] / r["rv_tr"][q]), 2)
        used = f"{rs[0]['used'][q - 1][0]} / {K}"
        ppl = _cell(lambda r, q=q: r["used"][q - 1][1], 2)
        rows.append((q, 3 * q, _cell(lambda r, q=q: r["rv_tr"][q]), ratio, _cell(lambda r, q=q: r["rv_te"][q]), gain, used, ppl))
    return (f"Training vectors from {_seed(DATA_TRAIN)}, held-out vectors from {_seed(DATA_TEST)}; d = {D}, {Q} stages of K = {K} codes "
            f"(3 bits each). Stage q is trained by {ITERS} rounds of Lloyd's algorithm on the residuals the first q - 1 stages leave, "
            f"its starting centres the first {K} rows of the residual set in the order of one shuffle of 0 .. {N_TRAIN - 1} from "
            f"{_seed(INIT_RVQ)} (one shuffle per stage, in stage order, the stream continuing); a round assigns every row to its "
            "nearest centre (lower index on a tie, squared distance summed over the coordinates in order), then replaces each "
            "centre that has rows by their mean, summed in row order, and leaves a centre with no rows where it is. The codes "
            "are the assignments to the final centres. Errors are mean squared norms of the residual; the gain is "
            "10 log10 of the mean squared norm of the data over the error, on the training set; codes used and the perplexity "
            "are those of the stage's training assignments. Numbers are printed with the digits that agree across the run and "
            "four nudged copies (first coordinate of the first starting centre of every stage and of the flat run changed by +-1e-12 "
            "and +-3e-12).\n\n"
            + _table(["stages sent", "bits per vector", "training error", "error / previous", "held-out error", "gain (dB)",
                      "codes used", "perplexity of the codes"], rows, "lrrrrrrr"))


def rvq_flat():
    rows = []
    rows.append(("residual, 2 stages of 8", "2 x 8 = 16", 64, _cell(lambda r: r["rv_tr"][2]), _cell(lambda r: r["rv_te"][2])))
    rows.append(("one flat codebook", K_FLAT, K_FLAT, _cell(lambda r: r["fl_tr"]), _cell(lambda r: r["fl_te"])))
    rows.append(("residual, 4 stages of 8", "4 x 8 = 32", 4096, _cell(lambda r: r["rv_tr"][4]), _cell(lambda r: r["rv_te"][4])))
    return (f"The flat codebook is {K_FLAT} centres trained by the same {ITERS} rounds on the same training vectors, its starting "
            f"centres the first {K_FLAT} rows of one shuffle from {_seed(INIT_FLAT)}. 'entries searched' is the number of centres "
            "one vector is compared with; 'cells' is the number of different reconstructions the quantizer can give.\n\n"
            + _table(["quantizer", "entries searched", "cells", "training error", "held-out error"], rows, "lrrrr"))


def search_vs_cells():
    rows = []
    for name, d, stages, k in [("the toy, 2 stages", D, 2, K), ("the toy, 4 stages", D, 4, K),
                               ("brain's codec, 1 stage", CODE_DIM, 1, BINS), ("brain's codec, 16 stages", CODE_DIM, NQ, BINS)]:
        searched = stages * k
        cells = k ** stages
        cells_text = _i(cells) if cells < 10 ** 7 else f"2^{stages * int(math.log2(k))} (about {cells:.2e})"
        rows.append((name, d, stages, k, _i(searched), cells_text, _i(d * searched), _i(stages * k * d * 4)))
    return ("A residual quantizer compares a vector with K entries at each of its stages; a flat codebook with the same cells would "
            "compare it with all of them. Distances are counted as d multiply-adds per entry; the table of one stage is K d 32-bit "
            "floats.\n\n"
            + _table(["quantizer", "width d", "stages", "entries per stage K", "entries searched", "cells K^stages",
                      "multiply-adds of one search", "bytes of the tables"], rows, "lrrrrrrr"))


# ================================================================= the codec's size

def _conv_params(cin, cout, k, groups=1, bias=True):
    return cout * (cin // groups) * k + (cout if bias else 0)


def _encoder_rows():
    """(component, parameters, multiply-adds per second of audio)."""
    rows = []
    # convolutions
    p = m = 0
    jump = 1
    for name, cin, cout, k, s, bias, branch in encoder_layers():
        if s > 1:
            jump *= s
        pos = FS / jump
        p += _conv_params(cin, cout, k, bias=bias)
        m += cout * cin * k * pos
    rows.append(("encoder convolutions (head, 4 stages, tail)", p, m))
    # transformer: 4 attention projections + 2 feed-forward matrices, two LayerNorms (weight, bias), two layer scales
    d = HIDDEN
    per = 4 * d * ENC_HEADS * ENC_HEAD_DIM + 2 * d * ENC_FFN + 2 * 2 * d + 2 * d
    rate25 = FS / (HOP // DOWN_STRIDE)
    macs = ENC_LAYERS * (4 * d * ENC_HEADS * ENC_HEAD_DIM + 2 * d * ENC_FFN) * rate25
    rows.append(("encoder transformer, without attention scores", ENC_LAYERS * per, macs))
    rows.append(("frame-rate match (512 -> 512, k 4, stride 2)", _conv_params(HIDDEN, HIDDEN, DOWN_K, bias=False),
                 HIDDEN * HIDDEN * DOWN_K * FRAME_RATE))
    rows.append(("quantizer input projections (2 x 512 -> 256)", 2 * HIDDEN * CODE_DIM, 2 * HIDDEN * CODE_DIM * FRAME_RATE))
    rows.append(("quantizer tables (16 x 2048 x 256) and search", NQ * BINS * CODE_DIM, NQ * BINS * CODE_DIM * FRAME_RATE))
    return rows


def _decoder_rows():
    rows = []
    rows.append(("dequantizer tables (16 x 2048 x 256)", NQ * BINS * CODE_DIM, 0))
    rows.append(("dequantizer output projections (2 x 256 -> 512)", 2 * CODE_DIM * LAT, 2 * CODE_DIM * LAT * FRAME_RATE))
    rows.append(("pre-convolution (512 -> 1024, k 3)", _conv_params(LAT, LATENT, 3), LATENT * LAT * 3 * FRAME_RATE))
    d, hq, ff = DEC_HIDDEN, DEC_HEADS * DEC_HEAD_DIM, DEC_FFN
    per = 4 * d * hq + 3 * d * ff + 2 * d + 2 * d
    projections = (LATENT * d + d) + (d * LATENT + LATENT) + d
    macs = (DEC_LAYERS * (4 * d * hq + 3 * d * ff) + LATENT * d + d * LATENT) * FRAME_RATE
    rows.append(("pre-transformer without attention scores (8 layers) and projections", DEC_LAYERS * per + projections, macs))
    # upsample stages
    p = m = 0
    rate = FRAME_RATE
    for u in UP:
        p += LATENT * LATENT * u + LATENT
        m += LATENT * LATENT * u * rate               # transposed convolution: cin cout k per input position
        rate *= u
        c = LATENT
        p += _conv_params(c, c, 7, groups=c) + 2 * c + _conv_params(c, 4 * c, 1) + _conv_params(4 * c, c, 1) + c
        m += (c * 7 + c * 4 * c * 2) * rate
    rows.append(("upsampling stages (2 x transposed convolution + ConvNeXt block)", p, m))
    # SEANet
    p = _conv_params(LATENT, DEC_DIM, 7)
    m = LATENT * DEC_DIM * 7 * rate
    rows.append(("SEANet decoder: first convolution", p, m))
    cin = DEC_DIM
    for i, r in enumerate(RATES):
        cout = DEC_DIM >> (i + 1)
        p = 2 * cin + (cin * cout * 2 * r + cout)
        m = cin * cout * 2 * r * rate               # per input position
        rate *= r
        for _ in range(3):
            p += 2 * (2 * cout) + _conv_params(cout, cout, 7) + _conv_params(cout, cout, 1)
            m += (cout * cout * 7 + cout * cout) * rate
        rows.append((f"SEANet block {i + 1}: up by {r}, {cin} -> {cout} channels", p, m))
        cin = cout
    p = 2 * cin + _conv_params(cin, 1, 7)
    m = cin * 7 * rate
    rows.append(("SEANet tail (96 -> 1, k 7)", p, m))
    return rows


def _fmt_m(x):
    return f"{x / 1e6:,.1f}" if x else "0"


def codec_layout():
    """The layer layout behind every row of codec_counts, printed from the constants the counts are computed from."""
    d, hq, ff = DEC_HIDDEN, DEC_HEADS * DEC_HEAD_DIM, DEC_FFN
    c = LATENT
    rows = [
        ("encoder convolutions", f"head 1 -> {FILTERS} (k {K_HEAD}); four stages with strides {', '.join(map(str, STRIDES))}, "
         f"each a residual branch c -> c/{COMPRESS} (k {K_RES}) then c/{COMPRESS} -> c (k 1) and a strided convolution "
         f"c -> 2c (k 2 x stride); tail {FILTERS * 2 ** len(STRIDES)} -> {HIDDEN} (k {K_LAST})",
         "weight and bias on every convolution; the activations carry nothing"),
        ("encoder transformer", f"{ENC_LAYERS} layers of width {HIDDEN}; attention with {ENC_HEADS} heads of width "
         f"{ENC_HEAD_DIM} (query, key, value and output matrices, {HIDDEN} x {ENC_HEADS * ENC_HEAD_DIM}); feed-forward "
         f"{HIDDEN} -> {ENC_FFN} -> {HIDDEN} with two matrices",
         f"no matrix has a bias; per layer two LayerNorms (weight and bias of width {HIDDEN} each, {2 * 2 * HIDDEN:,} numbers) and two layer scales "
         f"(one number per channel, {2 * HIDDEN:,} numbers), so {2 * 2 * HIDDEN + 2 * HIDDEN:,} small numbers"),
        ("frame-rate match", f"one convolution {HIDDEN} -> {HIDDEN}, k {DOWN_K}, stride {DOWN_STRIDE}", "no bias"),
        ("quantizer projections", f"two of {HIDDEN} -> {CODE_DIM}, one for the semantic codebook and one for the acoustic ones",
         "no bias"),
        ("decoder pre-transformer", f"input projection {LATENT} -> {d}, {DEC_LAYERS} layers of width {d}, output projection "
         f"{d} -> {LATENT}; attention with {DEC_HEADS} heads of width {DEC_HEAD_DIM} (matrices {d} x {hq}); feed-forward gated: "
         f"three matrices of {d} x {ff} (gate, up, down)",
         f"both projections have a bias; no matrix in a layer has one; per layer two RMSNorms (weight only, 2 x {d}) and two "
         f"layer scales (2 x {d}); one final RMSNorm ({d})"),
        ("decoder upsampling stages", f"two stages: a transposed convolution {c} -> {c} with k = stride = {UP[0]}, then a "
         f"ConvNeXt block of width {c}: depthwise convolution k 7, LayerNorm, pointwise {c} -> {4 * c} -> {c}",
         f"transposed convolution and all three convolutions with bias; LayerNorm weight and bias; one layer scale ({c})"),
        ("SEANet decoder", f"first convolution {c} -> {DEC_DIM} (k 7); four blocks of rates {', '.join(map(str, RATES))}, each "
         f"a SnakeBeta, a transposed convolution c_in -> c_out with k = 2 x rate, and three residual units (dilations 1, 3, 9) of "
         f"SnakeBeta, convolution c_out -> c_out (k 7), SnakeBeta, convolution c_out -> c_out (k 1); a last SnakeBeta and a "
         f"convolution 96 -> 1 (k 7)",
         "weight and bias on every convolution; every SnakeBeta has an alpha and a beta per channel (2 x channels of its input)"),
    ]
    return ("The layers behind the counts below: widths, heads, kernels, and which tensors are counted. Where two matrices or "
            "norms are named they are the tensors the code reads; nothing else carries a parameter.\n\n"
            + _table(["component", "layout", "tensors counted besides the weights named"], rows, "lll"))


def codec_counts():
    out = []
    totals = {}
    for title, rows in (("Encoder", _encoder_rows()), ("Decoder", _decoder_rows())):
        body = [(n, _i(p), _fmt_m(m)) for n, p, m in rows]
        tp, tm = sum(p for _, p, _ in rows), sum(m for _, _, m in rows)
        body.append((f"**{title} total**", _i(tp), _fmt_m(tm)))
        out.append(_table(["component", "parameters", "multiply-adds per second of audio (millions)"], body, "lrr"))
        totals[title] = (tp, tm)
    return ("Parameters are counted from the tensors the code reads (weights and the biases it adds; the quantizer tables at 2048 "
            "entries of 256 numbers each). A convolution costs c_out c_in k / groups multiply-adds per output position and a "
            "transposed convolution c_in c_out k per input position; a matrix product costs its input width times its output "
            "width per row; the nearest-code search is counted as d multiply-adds per entry. The squared distances of the "
            "attention scores, the activations and the normalisations are not counted. Frames arrive at 12.5 per second, "
            "25 per second inside the encoder, 50 per second after the upsampling stages. Each total is the sum of the unrounded "
            "rows, so a printed total can differ by 0.1 from the sum of the printed rows.\n\n"
            + out[0] + "\n\n" + out[1])


def attention_work():
    rows = []
    for secs in (1, 5, 10, 30):
        t25 = _ceil_div(secs * FS, HOP // DOWN_STRIDE)
        t12 = _ceil_div(secs * FS, HOP)
        enc_keys = sum(min(i + 1, ENC_WINDOW) for i in range(t25))
        enc = 2 * ENC_HEADS * ENC_HEAD_DIM * ENC_LAYERS * enc_keys
        keys = sum(min(i + 1, WINDOW) for i in range(t12))
        dec = 2 * DEC_HEADS * DEC_HEAD_DIM * DEC_LAYERS * keys
        rows.append((secs, t25, _i(enc), t12, _i(dec)))
    return ("Per query and key a layer spends 2 x heads x head width multiply-adds (one product for the score, one for the weighted "
            f"value). The encoder has {ENC_LAYERS} layers of {ENC_HEADS} heads of width {ENC_HEAD_DIM} (2 x {ENC_HEADS * ENC_HEAD_DIM} "
            f"per query and key per layer) and sees at most the {ENC_WINDOW} most recent frames, the sum over frames i of "
            f"min(i + 1, {ENC_WINDOW}) pairs; the decoder has {DEC_LAYERS} layers of {DEC_HEADS} heads of width {DEC_HEAD_DIM} "
            f"(2 x {DEC_HEADS * DEC_HEAD_DIM}) and sees at most the {WINDOW} most recent frames, the sum over frames i of "
            f"min(i + 1, {WINDOW}) pairs. Frames are ceil(seconds x 24000 / 960) at 25 Hz and ceil(seconds x 24000 / 1920) at "
            f"12.5 Hz.\n\n"
            + _table(["seconds", "encoder frames (25 Hz)", "encoder attention multiply-adds", "decoder frames (12.5 Hz)",
                      "decoder attention multiply-adds"], rows, "rrrrr"))


def streaming_state():
    rows = []
    total = 0

    def add(name, floats):
        nonlocal total
        total += floats
        rows.append((name, _i(floats), _i(4 * floats)))
    for u in range(len(UP)):
        add(f"upsample {u + 1} ConvNeXt depthwise (c 1024, k 7)", LATENT * 6)
    add("first SEANet convolution (1024 channels in, k 7)", LATENT * 6)
    cin = DEC_DIM
    for i, r in enumerate(RATES):
        cout = DEC_DIM >> (i + 1)
        add(f"block {i + 1} transposed convolution carry (c_out {cout} x {r})", cout * r)
        for dil in (1, 3, 9):
            add(f"block {i + 1} residual conv dilation {dil} (c {cout}, k 7)", cout * 6 * dil)
        cin = cout
    add("last convolution (96 channels in, k 7)", cin * 6)
    rows.append(("**total**", _i(total), _i(4 * total)))
    return ("A stride-1 causal convolution keeps its last dilation x (k - 1) input columns per input channel; a transposed convolution "
            "keeps the k - stride output columns that later frames still add to, per output channel. The two upsampling "
            "transposed convolutions have k = stride and keep nothing. 32-bit floats.\n\n"
            + _table(["stage of the streamed part", "floats carried", "bytes"], rows, "lrr"))


# ================================================================= the reconstruction trainer

def _demo_codebook(bins, dim):
    return [[((b * 31 + d * 17) % 200) / 100.0 - 1.0 for d in range(dim)] for b in range(bins)]


def demo_codebook():
    C = _demo_codebook(8, 4)
    rows = [(b, *(_f(v, 2) for v in C[b])) for b in range(8)]
    return ("Entry (b, j) of the codebook the trainer starts and keeps, for 8 codes of width 4: ((31 b + 17 j) mod 200) / 100 - 1.\n\n"
            + _table(["code b", "j = 0", "j = 1", "j = 2", "j = 3"], rows, "rrrrr"))


def recon_loss():
    pred, target = [0.5, -0.2, 0.1, 0.9], [0.4, 0.1, 0.1, -0.1]
    n = len(pred)
    d = [p - t for p, t in zip(pred, target)]
    sign = [1.0 if v >= 0 else -1.0 for v in d]
    loss = sum(abs(v) + 0.5 * v * v for v in d) / n
    grad = [(s + v) / n for s, v in zip(sign, d)]
    rows = [(i, f"{pred[i]}", f"{target[i]}", _f(d[i], 2), _f(abs(d[i]) + 0.5 * d[i] * d[i], 4), _f(grad[i], 4)) for i in range(n)]
    return ("Loss mean(|d|) + 0.5 mean(d^2) with d = prediction - target and the book's sign (+1 at 0): gradient (sign(d) + d) / n.\n\n"
            + _table(["i", "prediction", "target", "d", "|d| + d^2 / 2", "gradient (sign(d) + d) / 4"], rows, "rrrrrr")
            + f"\n\nLoss: {_f(loss, 4)}.")


def trainer_sizes():
    rows = []
    for c, k, test in [(4, 3, "gradient check"), (6, 3, "training test")]:
        rows.append((test, c, k, 2 * c * k, 8 * c, 2 * (k - 1) + 1, 1, "3", f"{3 * FS:,}"))
    return ("The trainer's encoder is one causal convolution 1 -> c channels (kernel k, stride 1), its quantizer is the frozen "
            "table of the previous block (8 codes of width c), and its decoder one causal convolution c -> 1. Weights are c k for "
            "each convolution; the table is 8 c numbers that are never trained.\n\n"
            + _table(["configuration", "c", "k", "trained weights", "table entries (fixed)", "receptive field (samples)",
                      "samples per frame", "bits per frame", "bits per second at 24 kHz"], rows, "lrrrrrrrr"))


# ================================================================= data of the figures

def rvq_curve():
    """([bits], [training error], [held-out error]) after 0 .. Q stages, and the flat codebook's (bits, training, held-out)."""
    rs = _runs()
    r0 = rs[0]
    return ([3 * q for q in range(Q + 1)], r0["rv_tr"], r0["rv_te"], (math.log2(K_FLAT), r0["fl_tr"], r0["fl_te"]))


def pipeline_rates():
    """The (rate in Hz, width) pairs the pipeline figure labels."""
    return {"samples": FS, "convolutions": FS / (HOP // DOWN_STRIDE), "frames": FRAME_RATE, "up": FRAME_RATE * UP[0] * UP[1] * 1.0}
