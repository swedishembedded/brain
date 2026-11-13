# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Who Is Speaking": how a model captures the identity of a voice in a single vector.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: speaker_embeddings.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree. Earlier modules are imported, never copied: the frame count of `spectrograms` and the
receptive-field recursion of `convolution`.

* what two front ends make of clips of different lengths, and the shortest clip each encoder can read;
* a time-delay layer by hand: a dilated convolution over time with reflect padding, against zero padding;
* the receptive field of both encoders by the recursion and by tracing the frames one output reads through the real wiring, and
  how the ladder of a Res2Net block gives every channel group its own depth;
* a squeeze-and-excitation gate and a context gate, each on a few numbers;
* attentive statistics pooling by hand, with the plain statistics beside it;
* the parameters and the work of both encoders, counted from the shapes of their configurations;
* cosine scoring of a synthetic population of speakers, with the equal error rate for clips of four lengths;
* why a cosine alone cannot verify a vector that is used as it is.

Arithmetic is the language's 64-bit floating point, sums running left to right in the order the text states; exp, tanh and sqrt are
the correctly rounded library functions. The only random numbers are those of the scoring population, from the book's SplitMix64
(`bookrng.generator(name)`), whose name and draw order the chapter prints. Nothing is trained, so no number is chaotic and every
digit printed is reproducible. Operation counts use 2 per multiply-add and leave out activations, softmaxes, normalisations and
pooling sums; each table says so.

Swedish Embedded AB implements solutions for recognising who is speaking from a few seconds of audio, with the cost of the encoder
counted before it is deployed. If your team needs expertise in speaker verification and voice-conditioned speech synthesis, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import convolution as cv
import spectrograms as sp

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(v):
    return f"{v:,}"


def _f(v, d=4):
    text = f"{v:.{d}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _mega(v, d=1):
    return f"{v / 1e6:,.{d}f}"


def _seed(name):
    return f"seed 0x{bookrng.seed_of(name):016x} ({bookrng.seed_of(name)})"


def _row(v, d=3):
    return "(" + ", ".join(_f(x, d) for x in v) + ")"


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ------------------------------------------------------------------------------------------------ the two front ends

ECAPA_RATE, ECAPA_N, ECAPA_HOP, ECAPA_PAD, ECAPA_MELS = 24000, 1024, 256, 384, 128
CAM_RATE, CAM_N, CAM_HOP, CAM_MELS = 16000, 400, 160, 80
ECAPA_DILATIONS = (2, 3, 4)  # of the three Res2Net blocks; the first convolution has k = 5, d = 1


def ecapa_frames(samples):
    """Frames of the reflect-padded log-mel front end: 1 + (L + 2 * 384 - 1024) // 256 when that is at least 1, else 0."""
    return sp.frame_count(samples + 2 * ECAPA_PAD, ECAPA_N, ECAPA_HOP)


def cam_frames(samples):
    """Frames of the 25 millisecond, 10 millisecond kaldi-style filterbank with no padding (frames that fit)."""
    return sp.frame_count(samples, CAM_N, CAM_HOP)


def cam_rows(frames):
    """Rows after the stride-2 convolution of kernel 5 and padding 2: (T + 4 - 5) // 2 + 1."""
    return (frames + 4 - 5) // 2 + 1 if frames >= 1 else 0


def ecapa_readable(frames):
    """The largest same-padding reflect pad of the encoder is d (k - 1) / 2 = 4 (k = 3, d = 4), and a reflect pad of p needs
    p < T; the first convolution needs 2 < T. So T >= 5."""
    pads = [2] + [d for d in ECAPA_DILATIONS]
    return all(p < frames for p in pads)


def cam_readable(frames):
    """The statistics pool divides by L - 1, so it needs at least two rows after the stride-2 convolution."""
    return cam_rows(frames) >= 2


def _shortest(frames_of, readable, rate):
    n = 1
    while not readable(frames_of(n)):
        n += 1
    return n, n / rate


def clip_geometry():
    rows = []
    for sec in (0.05, 0.1, 1.0, 2.5, 7.0):
        e_n, c_n = round(sec * ECAPA_RATE), round(sec * CAM_RATE)
        e_t, c_t = ecapa_frames(e_n), cam_frames(c_n)
        rows.append((f"{sec:g}", _int(e_n), e_t, "yes" if ecapa_readable(e_t) else "no", _int(c_n), c_t, cam_rows(c_t),
                     "yes" if cam_readable(c_t) else "no"))
    e_min, e_sec = _shortest(ecapa_frames, ecapa_readable, ECAPA_RATE)
    c_min, c_sec = _shortest(cam_frames, cam_readable, CAM_RATE)
    return ("Frames of the two front ends for clips of five lengths. Front end 1 (the 24 kHz log-mel): 1024-sample frames at a hop of "
            "256 after a reflect pad of 384 on each side, 128 values per frame. Front end 2 (the 16 kHz filterbank): 400-sample frames "
            "at a hop of 160, no padding, 80 values per frame, then one convolution of kernel 5, stride 2 and padding 2 makes "
            "(T + 4 - 5) // 2 + 1 rows. 'Readable' applies the rules of the text: encoder 1 needs T >= 5 frames, encoder 2 needs "
            "at least 2 rows.\n\n"
            + _table(["seconds", "samples at 24 kHz", "frames T", "readable", "samples at 16 kHz", "frames T", "rows", "readable"],
                     rows, "lrrlrrrl")
            + f"\n\nThe shortest readable clip is {_int(e_min)} samples ({e_sec * 1000:.2f} milliseconds) for encoder 1 and "
              f"{_int(c_min)} samples ({c_sec * 1000:.1f} milliseconds) for encoder 2.")


# ------------------------------------------------------------------------------------------------ a time-delay layer by hand

TDNN_X = [[1.0, 3.0, 2.0, 5.0, 4.0, 2.0], [0.5, -1.0, 1.0, 0.0, 2.0, 1.0]]  # 2 channels, 6 frames
TDNN_W = [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]  # one output channel: a kernel of 3 taps for each input channel
TDNN_B = 0.25
TDNN_D = 2


def _tdnn_out(padded, p):
    """Output t = bias + sum over input channel c, tap m of W[c][m] * padded[c][t + d m], for t = 0..T-1 (padded has p extra on
    each side and the taps sit d frames apart, tap m at frame t - d + d m of the unpadded signal)."""
    T = len(padded[0]) - 2 * p
    out = []
    for t in range(T):
        acc = TDNN_B
        for c in range(len(padded)):
            for m in range(3):
                acc += TDNN_W[c][m] * padded[c][t + TDNN_D * m]
        out.append(acc)
    return out


def tdnn_by_hand():
    p = TDNN_D * (3 - 1) // 2
    refl = [sp.reflect_pad(row, p) for row in TDNN_X]
    zero = [sp.zero_pad(row, p) for row in TDNN_X]
    y_r, y_z = _tdnn_out(refl, p), _tdnn_out(zero, p)
    rows = [("input, channel 0", *[_f(v, 2) for v in TDNN_X[0]]), ("input, channel 1", *[_f(v, 2) for v in TDNN_X[1]]),
            ("channel 0 reflect-padded", *[_f(v, 2) for v in refl[0]]), ("channel 1 reflect-padded", *[_f(v, 2) for v in refl[1]])]
    rows2 = [("reflect padding", *[_f(v, 3) for v in y_r]), ("zero padding", *[_f(v, 3) for v in y_z])]
    return (f"One output channel from two input channels over T = 6 frames, kernel 3, dilation {TDNN_D}, so the same padding is "
            f"{p} frames on each side. Weights of channel 0: {_row(TDNN_W[0], 1)}; of channel 1: {_row(TDNN_W[1], 1)}; bias "
            f"{TDNN_B}. Output t is the bias plus the sum over both channels and the three taps (frames t - 2, t, t + 2 of the "
            f"padded signal's own positions) of weight times value, accumulated in the order channel 0 tap 0 to 2, then channel 1.\n\n"
            + _table(["row", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7"], [_pad_row(r) for r in rows], "l" + "r" * 10)
            + "\n\nOutputs for t = 0 to 5 (a pad outside the signal changes only the frames within 2 of an end):\n\n"
            + _table(["padding", "t=0", "t=1", "t=2", "t=3", "t=4", "t=5"], rows2, "lrrrrrr"))


def _pad_row(r):
    """A table row placed under the column heads -2 .. 7: an unpadded row leaves the pad columns empty."""
    label, cells = r[0], list(r[1:])
    if "padded" in label:
        return (label, *cells)
    return (label, "", "", *cells, "", "")


# ------------------------------------------------------------------------------------------------ receptive fields

ECAPA_STACK = [(5, 1, 1)] + [(3, 1, d) for d in ECAPA_DILATIONS for _ in range(7)]  # the deepest chunk of each block
CAM_STEM = [(3, 1, 1)] * 10
CAM_BLOCKS = ((12, 1), (24, 2), (16, 2))  # layers, dilation of CAMLayer's local convolution
CAM_STACK = CAM_STEM + [(5, 2, 1)] + [(3, 1, d) for n, d in CAM_BLOCKS for _ in range(n)]


def _need(i, S, d):
    """Positions of a Res2Net block's input that chunk i's output at positions S reads: chunk 0 passes through; chunk i >= 1 reads its
    own chunk and, for i >= 2, the output of chunk i - 1, through a convolution of kernel 3 and dilation d."""
    if i == 0:
        return set(S)
    wide = {p + d * m for p in S for m in (-1, 0, 1)}
    return wide if i == 1 else wide | _need(i - 1, wide, d)


def ecapa_traced(centre=0):
    """The frames of the first convolution's input that one output of the feature aggregation reads, by tracing the wiring; also
    the spans after each block."""
    spans = []
    for upto in range(1, 4):
        S = {centre}
        for d in reversed(ECAPA_DILATIONS[:upto]):
            nxt = set(S)  # the residual connection
            for i in range(8):
                nxt |= _need(i, S, d)
            S = nxt
        S = {p + m for p in S for m in range(-2, 3)}  # the first convolution, kernel 5, dilation 1
        spans.append((len(S), max(S) - min(S) + 1))
    return spans


def cam_traced(centre=600):
    """Input frames read by one row of the last CAM++ layer: back through 52 layers (kernel 3, dilation d, each one reading the
    previous layer's output through the dense concatenation), the stride-2 convolution, and ten stem convolutions."""
    S = {centre}
    for n, d in reversed(CAM_BLOCKS):
        for _ in range(n):
            S = {p + d * m for p in S for m in (-1, 0, 1)}
    S = {2 * p - 2 + m for p in S for m in range(5)}
    for _ in range(10):
        S = {p + m for p in S for m in (-1, 0, 1)}
    return len(S), max(S) - min(S) + 1


def receptive_fields():
    e_rec = cv.receptive(ECAPA_STACK)
    c_rec = cv.receptive(CAM_STACK)
    e_after = [e_rec[0][0], e_rec[7][0], e_rec[14][0], e_rec[21][0]]
    e_traced = ecapa_traced()
    c_after = [c_rec[9][0], c_rec[10][0], c_rec[22][0], c_rec[46][0], c_rec[-1][0]]
    c_traced = cam_traced()
    assert [s for _, s in e_traced] == e_after[1:] and c_traced[1] == c_after[-1]
    e_samples = lambda r: (r - 1) * ECAPA_HOP + ECAPA_N
    c_samples = lambda r: (r - 1) * CAM_HOP + CAM_N
    rows1 = [("first convolution (k 5, d 1)", e_after[0]), ("block 1 (d 2, chain of 7)", e_after[1]),
             ("block 2 (d 3, chain of 7)", e_after[2]), ("block 3 (d 4, chain of 7)", e_after[3])]
    rows2 = [("ten stem convolutions (k 3)", c_after[0]), ("stride-2 convolution (k 5)", c_after[1]),
             ("block 1: 12 layers, d 1", c_after[2]), ("block 2: 24 layers, d 2", c_after[3]),
             ("block 3: 16 layers, d 2", c_after[4])]
    t1 = _table(["stage of encoder 1", "r (frames)", "samples", "seconds"],
                [(n, r, _int(e_samples(r)), _f(e_samples(r) / ECAPA_RATE, 3)) for n, r in rows1], "lrrr")
    t2 = _table(["stage of encoder 2", "r (frames)", "samples", "seconds"],
                [(n, r, _int(c_samples(r)), _f(c_samples(r) / CAM_RATE, 3)) for n, r in rows2], "lrrr")
    return ("Receptive field r in frames of the front end along time, by the recursion r = r + d (k - 1) j with the jump j (1 until "
            "the stride-2 layer of encoder 2, then 2); a convolution of kernel 1 adds nothing. Samples are (r - 1) hop + frame "
            "length, seconds at the clip's own rate. Encoder 1 follows the longest chain of a Res2Net block (7 convolutions), "
            "encoder 2 follows the 52 dense layers one after another, since each reads the output of the one before it.\n\n"
            + t1 + "\n\n" + t2
            + f"\n\nTracing the frames one output reads through the real wiring (every chunk, the residual connection, the stride) "
              f"gives, for the three blocks of encoder 1, spans of {e_traced[0][1]}, {e_traced[1][1]} and {e_traced[2][1]} frames "
              f"made of {e_traced[0][0]}, {e_traced[1][0]} and {e_traced[2][0]} distinct frames (every frame of the span is read, because the chunks of a block fill the gaps one chain would leave), and "
              f"for encoder 2 a span of {c_traced[1]} frames made of {c_traced[0]} distinct frames. Both equal the recursion.")


def ladder():
    d = 2
    rows = []
    for i in range(8):
        depth = 0 if i == 0 else i
        rows.append((i, "none" if i == 0 else ("the chunk alone" if i == 1 else f"the chunk plus the output of {i - 1}"), depth,
                     1 + 2 * d * depth))
    dense = 512 * 512 * 3
    lad = 7 * 64 * 64 * 3
    return ("The Res2Net block of encoder 1 splits its 512 channels into 8 chunks of 64. Chunk 0 is passed on as it is; chunk i is "
            "added to the output of chunk i - 1 (for i >= 2) and goes through one convolution of 64 channels, kernel 3, "
            "dilation d. Here d = 2.\n\n"
            + _table(["chunk i", "input of its convolution", "convolutions on its path", "frames it reads"], rows, "llrr")
            + f"\n\nOne dense convolution of 512 channels would hold 512 x 512 x 3 = {_int(dense)} weights; the seven convolutions "
              f"of the ladder hold 7 x 64 x 64 x 3 = {_int(lad)}, which is {dense / lad:.2f} times fewer.")


# ------------------------------------------------------------------------------------------------ gates

SE_X = [[2.0, 4.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 3.0, 2.0], [0.0, 0.0, 6.0]]  # 4 channels, 3 frames
SE_W1 = [[0.5, 0.25, -0.5, 0.0], [-0.25, 0.5, 0.25, 0.25]]  # 2 x 4
SE_B1 = [0.0, -0.25]
SE_W2 = [[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]  # 4 x 2
SE_B2 = [0.0, 0.5, 0.0, 0.25]


def squeeze_excitation():
    C, T = len(SE_X), len(SE_X[0])
    mean = [sum(row) / T for row in SE_X]
    h = [max(0.0, sum(SE_W1[s][c] * mean[c] for c in range(C)) + SE_B1[s]) for s in range(2)]
    z = [sum(SE_W2[c][s] * h[s] for s in range(2)) + SE_B2[c] for c in range(C)]
    g = [sigmoid(v) for v in z]
    y = [[g[c] * v for v in SE_X[c]] for c in range(C)]
    rows = [(c, _row(SE_X[c], 1), _f(mean[c], 4), _f(z[c], 4), _f(g[c], 4), _row(y[c], 4)) for c in range(C)]
    return (f"Squeeze and excitation on 4 channels and 3 frames. Squeeze: the mean over the 3 frames of each channel. Excite: "
            f"h = relu(W1 m + b1) with W1 = {_row(SE_W1[0], 2)} and {_row(SE_W1[1], 2)}, b1 = {_row(SE_B1, 2)}; z = W2 h + b2 with "
            f"the rows of W2 {', '.join(_row(r, 1) for r in SE_W2)} and b2 = {_row(SE_B2, 2)}; g = sigmoid(z) (the stable form of "
            f"the book's pseudocode). Output: each channel times its gate. The hidden vector is h = {_row(h, 4)}.\n\n"
            + _table(["channel", "frames", "mean m", "z", "gate g", "output"], rows, "lllrrl"))


CTX_X = [[1.0, 2.0, 3.0, 6.0, 5.0, 4.0, 9.0], [0.0, 1.0, 0.0, 2.0, 4.0, 3.0, 1.0]]  # 2 channels, 7 frames
CTX_SEG = 3
CTX_LOCAL = [2.0, -1.0, 1.0, 0.5, 3.0, 2.0, -2.0]  # pretend output of the local convolution of one channel
CTX_W1 = [[0.25, 0.0], [0.0, 0.5]]  # 2 x 2 (context bottleneck of width 2)
CTX_B1 = [-0.5, 0.0]
CTX_W2 = [0.5, -1.0]  # 1 x 2
CTX_B2 = 0.25


def context_gate():
    C, T = len(CTX_X), len(CTX_X[0])
    nseg = -(-T // CTX_SEG)
    mean = [sum(r) / T for r in CTX_X]
    seg_means = []
    for r in CTX_X:
        seg_means.append([sum(r[s * CTX_SEG:(s + 1) * CTX_SEG]) / len(r[s * CTX_SEG:(s + 1) * CTX_SEG]) for s in range(nseg)])
    ctx = [[mean[c] + seg_means[c][t // CTX_SEG] for t in range(T)] for c in range(C)]
    gate = []
    for t in range(T):
        h = [max(0.0, sum(CTX_W1[s][c] * ctx[c][t] for c in range(C)) + CTX_B1[s]) for s in range(2)]
        gate.append(sigmoid(sum(CTX_W2[s] * h[s] for s in range(2)) + CTX_B2))
    out = [gate[t] * CTX_LOCAL[t] for t in range(T)]
    rows = [("channel 0", *[_f(v, 2) for v in CTX_X[0]]), ("channel 1", *[_f(v, 2) for v in CTX_X[1]]),
            ("context, channel 0", *[_f(v, 4) for v in ctx[0]]), ("context, channel 1", *[_f(v, 4) for v in ctx[1]]),
            ("gate", *[_f(v, 4) for v in gate]), ("local output", *[_f(v, 2) for v in CTX_LOCAL]),
            ("gated output", *[_f(v, 4) for v in out])]
    return (f"A context gate on 2 channels and 7 frames with segments of {CTX_SEG} frames (so 3, 3 and 1 frames). The context of "
            f"frame t is the mean of the channel over all 7 frames plus the mean over the segment that holds t (the last segment "
            f"has one frame and is divided by 1). Channel means {_row(mean, 4)}; segment means of channel 0 {_row(seg_means[0], 4)} "
            f"and of channel 1 {_row(seg_means[1], 4)}. Gate of frame t: sigmoid(W2 relu(W1 context + b1) + b2) with W1 = "
            f"{_row(CTX_W1[0], 2)} and {_row(CTX_W1[1], 2)}, b1 = {_row(CTX_B1, 2)}, W2 = {_row(CTX_W2, 2)}, b2 = {CTX_B2}.\n\n"
            + _table(["row", *[f"t={t}" for t in range(T)]], rows, "l" + "r" * T))


# ------------------------------------------------------------------------------------------------ attentive statistics pooling

ASP_X = [[1.0, 3.0, 2.0, 0.0], [2.0, -1.0, 1.0, 0.0]]  # 2 channels, 4 frames; the last frame is a silence
ASP_WA = [[0.5, 0.0, 0.0, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.5], [0.25, 0.25, -0.25, -0.25, 0.0, 0.0]]  # 3 x 6
ASP_BA = [0.0, 0.0, 0.0]
ASP_WC = [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]  # 2 x 3
ASP_BC = [0.0, 0.0]
ASP_EPS = 1e-12


def _stats(x, w):
    """Weighted mean and standard deviation of each channel for time weights w that add to 1."""
    mean = [sum(w[t] * r[t] for t in range(len(r))) for r in x]
    var = [sum(w[t] * (r[t] - m) ** 2 for t in range(len(r))) for r, m in zip(x, mean)]
    return mean, [math.sqrt(max(v, ASP_EPS)) for v in var]


def asp_data():
    C, T = len(ASP_X), len(ASP_X[0])
    mean, std = _stats(ASP_X, [1.0 / T] * T)
    stack = ASP_X + [[m] * T for m in mean] + [[s] * T for s in std]  # 6 x 4
    hidden = [[max(0.0, sum(ASP_WA[a][i] * stack[i][t] for i in range(6)) + ASP_BA[a]) for t in range(T)] for a in range(3)]
    tanh = [[math.tanh(v) for v in row] for row in hidden]
    score = [[sum(ASP_WC[c][a] * tanh[a][t] for a in range(3)) + ASP_BC[c] for t in range(T)] for c in range(C)]
    weight = []
    for row in score:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        weight.append([v / s for v in e])
    amean, astd = [], []
    for c in range(C):
        m = sum(weight[c][t] * ASP_X[c][t] for t in range(T))
        v = sum(weight[c][t] * (ASP_X[c][t] - m) ** 2 for t in range(T))
        amean.append(m)
        astd.append(math.sqrt(max(v, ASP_EPS)))
    unbiased = []
    for r, m in zip(ASP_X, mean):
        unbiased.append(math.sqrt(sum((v - m) ** 2 for v in r) / (T - 1)))
    return dict(mean=mean, std=std, stack=stack, hidden=hidden, tanh=tanh, score=score, weight=weight, amean=amean, astd=astd,
                unbiased=unbiased)


def attentive_pooling():
    d = asp_data()
    T = len(ASP_X[0])
    rows1 = [("x, channel 0", *[_f(v, 2) for v in ASP_X[0]]), ("x, channel 1", *[_f(v, 2) for v in ASP_X[1]])]
    rows2 = [(f"channel {c}", *[_f(v, 4) for v in d["score"][c]], *[_f(v, 4) for v in d["weight"][c]]) for c in range(2)]
    rows3 = [("plain mean and standard deviation (weights 1/4)", _f(d["mean"][0], 4), _f(d["mean"][1], 4), _f(d["std"][0], 4),
              _f(d["std"][1], 4)),
             ("the same with the divisor 3 (unbiased)", _f(d["mean"][0], 4), _f(d["mean"][1], 4), _f(d["unbiased"][0], 4),
              _f(d["unbiased"][1], 4)),
             ("attentive mean and standard deviation", _f(d["amean"][0], 4), _f(d["amean"][1], 4), _f(d["astd"][0], 4),
              _f(d["astd"][1], 4))]
    return (f"Attentive statistics pooling of 2 channels over T = {T} frames; the last frame is a silence. Step 1: the plain mean and "
            f"population standard deviation of each channel (weights 1/{T}, divisor {T}): means {_row(d['mean'], 4)}, deviations "
            f"{_row(d['std'], 4)}. Step 2: the 6 x {T} array of x with its channel means and deviations repeated at every frame. "
            f"Step 3: a convolution of kernel 1 with weights W_a (3 x 6) = {', '.join(_row(r, 2) for r in ASP_WA)}, no bias, then "
            f"relu, then tanh (the library function), then a kernel-1 convolution W_c (2 x 3) = "
            f"{', '.join(_row(r, 1) for r in ASP_WC)}, no bias, the score. Step 4: a softmax over the {T} frames of each channel. "
            f"Step 5: weighted mean and weighted standard deviation with those weights, the variance floored at {ASP_EPS:g}.\n\n"
            + _table(["row", "t=0", "t=1", "t=2", "t=3"], rows1, "lrrrr")
            + "\n\nScores and attention weights (each weight row adds to 1):\n\n"
            + _table(["channel", "score 0", "score 1", "score 2", "score 3", "weight 0", "weight 1", "weight 2", "weight 3"], rows2,
                     "lrrrrrrrr")
            + "\n\nThe pooled vector of 4 numbers is the mean of channel 0, the mean of channel 1, then the two deviations:\n\n"
            + _table(["pooling", "mean 0", "mean 1", "std 0", "std 1"], rows3, "lrrrr"))


# ------------------------------------------------------------------------------------------------ counts

ECAPA = dict(mel=128, enc=1024, channels=(512, 512, 512, 512, 1536), kernels=(5, 3, 3, 3, 1), scale=8, att=128, se=128)
CAM = dict(feat=80, emb=192, fcm=32, tdnn=128, growth=32, mid=64, out=32, layers=(12, 24, 16), dilation=(1, 2, 2))


def _conv(cin, cout, k, groups=1):
    """(weights, biases) of a convolution with a bias."""
    return cin // groups * cout * k, cout


def ecapa_counts():
    """(stage, weights, biases, tensors, macs per frame, macs per utterance): the per-frame convolutions run at every frame, the
    squeeze-excitation convolutions and the last projection once on a vector."""
    e = ECAPA
    c0, cm = e["channels"][0], e["channels"][4]
    rows = []
    w, b = _conv(e["mel"], c0, e["kernels"][0])
    rows.append(("first convolution, 128 to 512, k 5", w, b, 2, w, 0))
    sub = c0 // e["scale"]
    w1, b1 = _conv(c0, c0, 1)
    wr, br = _conv(sub, sub, 3)
    ws1, bs1 = _conv(c0, e["se"], 1)
    ws2, bs2 = _conv(e["se"], c0, 1)
    per_block_w = 2 * w1 + (e["scale"] - 1) * wr + ws1 + ws2
    per_block_b = 2 * b1 + (e["scale"] - 1) * br + bs1 + bs2
    for i in (1, 2, 3):
        rows.append((f"SE-Res2Net block {i}", per_block_w, per_block_b, 2 + 2 * (e["scale"] - 1) + 2 + 4,
                     2 * w1 + (e["scale"] - 1) * wr, ws1 + ws2))
    w, b = _conv(3 * c0, cm, 1)
    rows.append(("aggregation convolution, 1536 to 1536", w, b, 2, w, 0))
    wa, ba = _conv(3 * cm, e["att"], 1)
    wc, bc = _conv(e["att"], cm, 1)
    rows.append(("attention of the pooling (4608 to 128 to 1536)", wa + wc, ba + bc, 4, wa + wc, 0))
    w, b = _conv(2 * cm, e["enc"], 1)
    rows.append(("projection, 3072 to 1024", w, b, 2, 0, w))
    return rows


def cam_counts():
    """Rows (stage, weights, biases, buffers, tensors, macs per second) of the second encoder, in the order of the data flow.
    The stem runs at 100 frames a second; every later stage at 50 rows a second."""
    c = CAM
    fc, ts, gr, md, ou = c["fcm"], c["tdnn"], c["growth"], c["mid"], c["out"]
    rows = []
    # stem: frequency 80 -> 40 -> 20 -> 10, time unchanged; macs per time column, times 100 columns a second
    freq = c["feat"]
    w = b = tensors = macs = 0
    w += 1 * fc * 9; b += fc; tensors += 2; macs += freq * 1 * fc * 9
    for li in range(2):
        for bi in range(2):
            stride = 2 if bi == 0 else 1
            fo = freq // stride
            w += fc * fc * 9 * 2; b += 2 * fc; tensors += 4
            macs += fo * fc * fc * 9 * 2
            if bi == 0:
                w += fc * fc; b += fc; tensors += 2
                macs += fo * fc * fc
            freq = fo
    freq //= 2
    w += fc * fc * 9; b += fc; tensors += 2; macs += freq * fc * fc * 9
    rows.append(("stem: 10 2-D convolutions and 2 shortcuts", w, b, 0, tensors, macs * 100))
    stem_out = fc * freq
    w1, b1 = _conv(stem_out, ts, 5)
    rows.append(("time-delay convolution, 320 to 128, k 5, stride 2", w1, b1, 0, 2, w1 * 50))
    cin0 = ts
    for bi, (n, dil) in enumerate(zip(c["layers"], c["dilation"])):
        w = b = buf = tensors = macs = 0
        cin = cin0
        for _ in range(n):
            w += 2 * cin  # batch-norm gain and shift
            buf += 2 * cin  # running mean, running variance
            tensors += 4
            lw, lb = _conv(cin, ts, 1)
            w += lw; b += lb; tensors += 2; macs += lw
            lw, _ = _conv(ts, ou, 3)
            w += lw; tensors += 1; macs += lw
            lw, lb = _conv(ts, md, 1)
            w += lw; b += lb; tensors += 2; macs += lw
            lw, lb = _conv(md, ou, 1)
            w += lw; b += lb; tensors += 2; macs += lw
            cin += gr
        rows.append((f"dense block {bi + 1}: {n} layers, d {dil}", w, b, buf, tensors, macs * 50))
        tin, tout = cin, cin // 2
        tw, tb = _conv(tin, tout, 1)
        tb = tb if bi == 2 else 0  # the last transit absorbed a normalisation into a bias
        rows.append((f"transit {bi + 1}: {tin} to {tout}", tw + 2 * tin, tb, 2 * tin, 5 if bi < 2 else 6, tw * 50))
        cin0 = tout
    dw, _ = _conv(2 * cin0, c["emb"], 1)
    rows.append(("projection 1024 to 192, final normalization", dw, 0, 2 * c["emb"], 3, 0))
    return rows


def layout():
    """The layers behind every row of parameter_counts, printed from the constants the counts use."""
    e, c = ECAPA, CAM
    c0, cm, sc = e["channels"][0], e["channels"][4], e["scale"]
    sub = c0 // sc
    fc, ts, gr, md, ou = c["fcm"], c["tdnn"], c["growth"], c["mid"], c["out"]
    layers, dil = c["layers"], c["dilation"]
    widths, cin = [], ts
    for n in layers:
        widths.append((cin + n * gr, (cin + n * gr) // 2))
        cin = widths[-1][1]
    rows1 = [
        ("first convolution", f"{e['mel']} -> {c0}, k {e['kernels'][0]}", "weight and bias"),
        ("SE-Res2Net block (three, dilations " + ", ".join(map(str, ECAPA_DILATIONS)) + ")",
         f"1 x 1 convolution {c0} -> {c0}; the {c0} channels split into {sc} chunks of {sub}, chunk 0 passed on, chunks 1 to "
         f"{sc - 1} each a convolution {sub} -> {sub}, k 3, at the block's dilation; 1 x 1 convolution {c0} -> {c0}; "
         f"squeeze-excitation {c0} -> {e['se']} -> {c0} (1 x 1, run once on the clip's mean)",
         "weight and bias on every convolution; no normalization tensor is stored"),
        ("aggregation convolution", f"{3 * c0} -> {cm}, k 1 (the three block outputs side by side)", "weight and bias"),
        ("attention of the pooling", f"{3 * cm} -> {e['att']} -> {cm}, k 1 (the channels, their mean and their deviation side by side)",
         "weight and bias on both"),
        ("projection", f"{2 * cm} -> {e['enc']} (the mean and the deviation of the {cm} channels)", "weight and bias"),
    ]
    rows2 = [
        ("stem", f"{fc} channels, 3 x 3 over frequency and time: a convolution 1 -> {fc}; two groups of two residual blocks, each "
         f"block two convolutions {fc} -> {fc}, the first block of each group striding by 2 over frequency (80 -> 40 -> 20) and "
         f"carrying a 1 x 1 shortcut {fc} -> {fc}; a last convolution {fc} -> {fc}, stride 2 over frequency (-> 10): 1 + 8 + 1 = "
         f"10 convolutions and 2 shortcuts, and {fc} x 10 = {fc * 10} channels leave it", "weight and bias on all 12"),
        ("time-delay convolution", f"{fc * 10} -> {ts}, k 5, stride 2 over time", "weight and bias"),
        ("dense layer", f"input of c_in channels (c_in is {ts} at the first layer of the first block and grows by {gr} with each "
         f"layer): normalization of the input (gain, shift, running mean, running variance), 1 x 1 convolution c_in -> {ts}, local "
         f"convolution {ts} -> {ou}, k 3 at the block's dilation, and the context gate, two 1 x 1 convolutions {ts} -> {md} -> {ou}; "
         f"the {ou} output channels are appended to the input; 11 tensors", "bias on the 1 x 1 convolutions and the gate, none on the local convolution"),
        ("dense blocks", f"{layers[0]}, {layers[1]} and {layers[2]} layers at dilations {dil[0]}, {dil[1]} and {dil[2]}", ""),
        ("transit", f"normalization of the block's c channels (gain, shift, running mean and variance), then 1 x 1 convolution "
         f"c -> c / 2; c is {widths[0][0]}, {widths[1][0]} and {widths[2][0]} for the three, and the channels left are "
         f"{widths[0][1]}, {widths[1][1]} and {widths[2][1]}",
         f"no bias on the first two; the third has a bias of {widths[2][1]} (the normalization after it is folded into the "
         "convolution in the released graph, which is why it has 6 tensors and the others 5)"),
        ("projection", f"statistics pooling of the {widths[2][1]} channels (mean and deviation) -> {2 * widths[2][1]}, 1 x 1 convolution "
         f"{2 * widths[2][1]} -> {c['emb']}, then a normalization with running mean and variance only (no gain or shift)",
         "no bias; 2 x 192 buffers"),
    ]
    return ("The layers behind the counts below: widths, kernels, chunking and which tensors exist. 'k 1' is a 1 x 1 "
            "convolution, a matrix applied to every frame.\n\nEncoder 1\n\n"
            + _table(["part", "layout", "biases and norms"], rows1, "lll")
            + "\n\nEncoder 2\n\n" + _table(["part", "layout", "biases and norms"], rows2, "lll"))


def parameter_counts():
    e = ecapa_counts()
    c = cam_counts()
    tot_e = sum(w + b for _, w, b, _, _, _ in e)
    t_e = sum(t for _, _, _, t, _, _ in e)
    tot_c = sum(w + b for _, w, b, _, _, _ in c)
    buf_c = sum(r[3] for r in c)
    t_c = sum(r[4] for r in c)
    assert t_e == 76 and t_c == 617
    rows1 = [(n, _int(w), _int(b), t, _int(2 * pf), _int(2 * pu)) for n, w, b, t, pf, pu in e]
    rows1.append(("total", _int(sum(r[1] for r in e)), _int(sum(r[2] for r in e)), t_e, _int(2 * sum(r[4] for r in e)),
                  _int(2 * sum(r[5] for r in e))))
    rows2 = [(n, _int(w), _int(b), _int(buf), t, _int(2 * m)) for n, w, b, buf, t, m in c]
    rows2.append(("total", _int(sum(r[1] for r in c)), _int(sum(r[2] for r in c)), _int(buf_c), t_c, _int(2 * sum(r[5] for r in c))))
    return ("Stored values and operations of the two encoders, counted from their configurations. Weights are the entries of "
            "kernels and matrices (and, for encoder 2, the gain and shift of its normalizations); biases are the entries of bias "
            "vectors; buffers are the running means and variances of encoder 2's normalizations, which are stored but not trained. "
            "Operations count 2 per multiply-add of the convolutions and projections only.\n\n"
            "Encoder 1 (24 kHz log-mel, 93.75 frames a second). 'Operations per frame' is for the convolutions that run at every "
            "frame; 'operations per clip' for the vector convolutions that run once, whatever the length.\n\n"
            + _table(["stage", "weights", "biases", "tensors", "operations per frame", "operations per clip"], rows1, "lrrrrr")
            + f"\n\nOne second of audio costs encoder 1 {_mega(2 * sum(r[4] for r in e) * 93.75, 1)} million operations in the "
              f"per-frame part (93.75 frames), and the once-per-clip part adds {_mega(2 * sum(r[5] for r in e), 1)} million whatever "
              f"the length. Total stored values: {_int(tot_e)}.\n\n"
              "Encoder 2 (16 kHz filterbank at 100 frames a second; the stem at that rate, the rest at 50 rows a second):\n\n"
            + _table(["stage", "weights", "biases", "buffers", "tensors", "operations per second of audio"], rows2, "lrrrrr")
            + f"\n\nTotal stored values: {_int(tot_c + buf_c)}, of which {_int(buf_c)} are buffers. The tensor counts, 76 and 617, are "
              f"the numbers of tensors in the two released checkpoints that the engine verifies.")


# ------------------------------------------------------------------------------------------------ scoring a population

SC_NAME = "speaker_embeddings.scoring"
SC_D, SC_SPEAKERS, SC_UTT, SC_RHO = 16, 40, 4, 2.0
SC_SECONDS = (1, 2, 4, 8)


def _population():
    g = bookrng.generator(SC_NAME)
    centres = []
    for _ in range(SC_SPEAKERS):
        v = [g.normal() for _ in range(SC_D)]
        n = math.sqrt(sum(x * x for x in v))
        centres.append([x / n for x in v])
    noise = [[[g.normal() for _ in range(SC_D)] for _ in range(SC_UTT)] for _ in range(SC_SPEAKERS)]
    return centres, noise


def _embed(centres, noise, seconds):
    emb = []
    for s in range(SC_SPEAKERS):
        for u in range(SC_UTT):
            v = [centres[s][k] + SC_RHO * noise[s][u][k] / math.sqrt(SC_D * seconds) for k in range(SC_D)]
            n = math.sqrt(sum(x * x for x in v))
            emb.append((s, [x / n for x in v]))
    return emb


def _scores(emb):
    gen, imp = [], []
    for p in range(len(emb)):
        for q in range(p + 1, len(emb)):
            c = 0.0
            for k in range(SC_D):
                c += emb[p][1][k] * emb[q][1][k]
            (gen if emb[p][0] == emb[q][0] else imp).append(c)
    return gen, imp


def rates(gen, imp, tau):
    """False rejection rate (genuine below tau) and false acceptance rate (impostor at or above tau)."""
    return sum(1 for c in gen if c < tau) / len(gen), sum(1 for c in imp if c >= tau) / len(imp)


def equal_error(gen, imp):
    """Over every observed score as a threshold, the one where |FAR - FRR| is least (the lowest threshold on a tie); returns
    (threshold, error rate), the error rate being the mean of FAR and FRR there."""
    best = None
    for tau in sorted(set(gen) | set(imp)):
        frr, far = rates(gen, imp, tau)
        key = abs(far - frr)
        if best is None or key < best[0]:
            best = (key, tau, (far + frr) / 2)
    return best[1], best[2]


def scoring_data():
    centres, noise = _population()
    out = {}
    for sec in SC_SECONDS:
        out[sec] = _scores(_embed(centres, noise, sec))
    return out


def scoring():
    data = scoring_data()
    g0, i0 = data[SC_SECONDS[0]]
    rows, rows2 = [], []
    taus = {}
    for sec in SC_SECONDS:
        gen, imp = data[sec]
        tau, eer = equal_error(gen, imp)
        taus[sec] = tau
        rows.append((sec, _f(sum(gen) / len(gen), 3), _f(sum(imp) / len(imp), 3), _f(tau, 3), f"{100 * eer:.1f}%"))
    tau8 = taus[SC_SECONDS[-1]]
    for sec in SC_SECONDS:
        gen, imp = data[sec]
        frr, far = rates(gen, imp, tau8)
        rows2.append((sec, _f(tau8, 3), f"{100 * frr:.1f}%", f"{100 * far:.1f}%"))
    return (f"Simulated embeddings, not a trained network: {SC_SPEAKERS} speakers, {SC_UTT} utterances each, vectors of width {SC_D}. "
            f"The generator is `{SC_NAME}` ({_seed(SC_NAME)}). Draw order: the {SC_SPEAKERS} centres (speaker 0 to {SC_SPEAKERS - 1}, "
            f"{SC_D} standard normals each, each divided by its length), then the noise (speaker, utterance, {SC_D} standard "
            f"normals). An utterance of s seconds of speaker i is its centre plus {SC_RHO:g} times its noise vector divided by "
            f"sqrt({SC_D} * s), the whole divided by its length; the same noise is used for all four lengths. Cosines are dot products "
            f"of the unit vectors summed over the {SC_D} components in order. Genuine pairs are two utterances of one speaker "
            f"({_int(len(g0))} pairs), impostor pairs of two speakers ({_int(len(i0))}). A pair is accepted when its cosine is at or "
            f"above the threshold. The equal error rate is found by trying every observed cosine as the threshold and keeping the one "
            f"where the false rejection rate and the false acceptance rate are closest (the lowest on a tie); the error rate "
            f"printed is the mean of the two there.\n\n"
            + _table(["seconds", "mean genuine cosine", "mean impostor cosine", "threshold at the equal error rate",
                      "equal error rate"], rows, "rrrrr")
            + f"\n\nOne threshold, that of the {SC_SECONDS[-1]} second clips ({_f(tau8, 3)}), applied to all four lengths:\n\n"
            + _table(["seconds", "threshold", "genuine rejected", "impostors accepted"], rows2, "rrrr"))


# ------------------------------------------------------------------------------------------------ scale and parity

PAR_V = [3.0, 4.0, 0.0, 12.0]
PAR_A = [[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.5, -0.5]]  # 2 x 4, a stand-in for the affine layer after a normalisation


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (_norm(a) * _norm(b))


def _rel(a, b):
    return _norm([x - y for x, y in zip(a, b)]) / _norm(b)


def _affine(v):
    n = _norm(v)
    u = [x / n for x in v]
    return [sum(PAR_A[r][c] * u[c] for c in range(4)) for r in range(2)]


def parity_metrics():
    rows = []
    cases = (("the reference vector", [1.0 * x for x in PAR_V]), ("scaled by 1.05", [1.05 * x for x in PAR_V]),
             ("scaled by 2", [2.0 * x for x in PAR_V]), ("one entry off by 0.001", [PAR_V[0] + 0.001] + PAR_V[1:]),
             ("one entry off by 1", [PAR_V[0] + 1.0] + PAR_V[1:]))
    base = _affine(PAR_V)
    for name, v in cases:
        after = _affine(v)
        rows.append((name, _f(_cos(v, PAR_V), 6), f"{_rel(v, PAR_V):.6f}", f"{_rel(after, base):.6f}"))
    return (f"The reference vector is {_row(PAR_V, 1)} (length {_norm(PAR_V):g}); 'one entry off' adds the stated amount to its first entry (the 3.0), "
            f"with a plus sign. 'Cosine' and 'relative L2' compare a candidate with it "
            f"(relative L2 is the length of the difference over the length of the reference). The last column passes both vectors "
            f"through the same stand-in for a consumer that normalizes first, divides each by its length and multiplies by "
            f"{', '.join(_row(r, 1) for r in PAR_A)}, and reports the relative L2 of the two results.\n\n"
            + _table(["candidate", "cosine", "relative L2", "relative L2 after normalizing and multiplying"], rows, "lrrr"))


# ------------------------------------------------------------------------------------------------ figures


def cam_final_width():
    """Channels leaving the last transit layer: each block adds growth x layers to its input and its transit halves the sum."""
    cin = CAM["tdnn"]
    for n in CAM["layers"]:
        cin = (cin + n * CAM["growth"]) // 2
    return cin


def figure_encoders():
    """Numbers behind the architecture figure: rates, widths and stage counts of both encoders."""
    e, c = ecapa_counts(), cam_counts()
    return dict(
        e_frames=ECAPA_RATE / ECAPA_HOP, e_mel=ECAPA_MELS, e_width=ECAPA["channels"][0], e_agg=ECAPA["channels"][4],
        e_pool=2 * ECAPA["channels"][4], e_emb=ECAPA["enc"], e_params=sum(w + b for _, w, b, _, _, _ in e),
        c_frames=CAM_RATE / CAM_HOP, c_mel=CAM_MELS, c_stem=CAM["fcm"] * (CAM["feat"] // 8), c_width=CAM["tdnn"],
        c_rows=CAM_RATE / CAM_HOP / 2, c_layers=sum(CAM["layers"]), c_emb=CAM["emb"],
        c_pool=2 * cam_final_width(),
        c_params=sum(w + b + buf for _, w, b, buf, _, _ in c))


def figure_scoring():
    """FAR and FRR against the threshold for the 1 second and the 8 second clips, and the equal error points."""
    data = scoring_data()
    taus = [i / 50 - 0.2 for i in range(0, 61)]
    out = {}
    for sec in (SC_SECONDS[0], SC_SECONDS[-1]):
        gen, imp = data[sec]
        far = [rates(gen, imp, t)[1] for t in taus]
        frr = [rates(gen, imp, t)[0] for t in taus]
        out[sec] = (far, frr, equal_error(gen, imp))
    return taus, out
