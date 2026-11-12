# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Foundation Models for Time Series".

Plain Python 3, standard library only, written from the chapter's own description: a series cut into
patches and what a patch embedding sees, the scalers a pretrained forecaster applies to a context and
inverts on its output (whole-context standardisation with arcsinh, the first patch, running statistics
merged patch by patch, a clipped z-score), a decoder over patches with a quantile head, the two ways of
turning values into tokens (uniform bins after a scale, and the sign code of a binary spherical
quantizer), group attention across a target and its covariates, the layout of a covariate beside a
target in the input of a patch, the cross-fade that stitches overlapping output patches into a long
horizon, the exact parameter and work counts of the four forecasters of brain from their layer
inventories, a shared linear map pretrained on a corpus of series and applied to series it was not fitted
on, and the gates by which a ported checkpoint is compared with its reference. Each public function
without a leading underscore that takes no arguments returns a markdown string that the chapter embeds
between `<!-- worked: foundation_forecasters.<name> -->` markers. Underscore functions and the constants are
shared with the chapter's figures, so a picture and the table beside it cannot disagree. The series and
the baselines of Chapter 1 (`time_series`) and the transformer block of Volume V (`transformer_block`)
are imported, never copied.

Swedish Embedded AB implements solutions for forecasting with pretrained models whose behaviour is checked
rather than assumed. If your team needs expertise in time-series foundation models and their evaluation,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng
import time_series as ts
import transformer_block as tb
from generalization import _solve
from time_series import _mean, _seed_text, _table

# ------------------------------------------------------------------ small helpers


def _fx(v, d=3):
    return f"{v:.{d}f}"


def _int(n):
    return f"{n:,}"


def _sci(x, d=2):
    """x as a power of ten for a table cell: 1.23e-05."""
    return f"{x:.{d}e}"


def _flops(x):
    """A count of floating-point operations as $m \\times 10^{e}$ (three significant digits)."""
    e = int(math.floor(math.log10(x)))
    return f"${x / 10 ** e:.2f} \\times 10^{{{e}}}$"


def _draws(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} ({bookrng.seed_of(name)})."


# ------------------------------------------------------------------ patching a series into tokens

PATCH_NAME = "foundation_forecasters.patching"
PATCH_SERIES = (12.0, 12.4, 13.1, 12.8, 13.6, None, 14.2, 14.0, 14.9, 15.3)   # None: bar 5 was not observed
PATCH_LEN, PATCH_WIDTH, PATCH_HIDDEN, PATCH_SCALE = 4, 3, 4, 0.5


def _standardise_observed(series):
    """(mean, population sd) over the observed values, and the series standardised with None kept."""
    obs = [v for v in series if v is not None]
    m = sum(obs) / len(obs)
    s = math.sqrt(sum((v - m) ** 2 for v in obs) / len(obs))
    return m, s, [None if v is None else (v - m) / s for v in series]


def _left_pad(series, patch):
    pad = (patch - len(series) % patch) % patch
    return [None] * pad + list(series), pad


def _patch_features(series, patch):
    """Rows [values (0 where missing) | mask (1 observed)] of the left-padded, patched series."""
    padded, pad = _left_pad(series, patch)
    rows = []
    for p in range(len(padded) // patch):
        chunk = padded[p * patch:(p + 1) * patch]
        rows.append([0.0 if v is None else v for v in chunk] + [0.0 if v is None else 1.0 for v in chunk])
    return rows, pad


def _residual_block(x, w_hidden, w_out, w_res):
    """out = W_out relu(W_hidden x) + W_res x (no biases)."""
    h = [max(0.0, sum(w * v for w, v in zip(row, x))) for row in w_hidden]
    return [sum(w * v for w, v in zip(ro, h)) + sum(w * v for w, v in zip(rr, x)) for ro, rr in zip(w_out, w_res)]


def _embedding_weights():
    g = bookrng.generator(PATCH_NAME)
    n_in = 2 * PATCH_LEN
    w_hidden = [[PATCH_SCALE * g.normal() for _ in range(n_in)] for _ in range(PATCH_HIDDEN)]
    w_res = [[PATCH_SCALE * g.normal() for _ in range(n_in)] for _ in range(PATCH_WIDTH)]
    w_out = [[PATCH_SCALE * g.normal() for _ in range(PATCH_HIDDEN)] for _ in range(PATCH_WIDTH)]
    return w_hidden, w_out, w_res


def patching():
    m, s, z = _standardise_observed(PATCH_SERIES)
    rows, pad = _patch_features(z, PATCH_LEN)
    w_hidden, w_out, w_res = _embedding_weights()
    shown = ["missing" if v is None else _fx(v, 1) for v in PATCH_SERIES]
    head = (f"The ten bars {', '.join(shown)} (bar 5 was not observed). The observed values have mean {m:.4f} and population standard "
            f"deviation {s:.4f}; each observed bar is standardised as (y - mean) / sd. The series is padded on the left with "
            f"{pad} missing bars to {len(rows) * PATCH_LEN}, which are cut into {len(rows)} patches of {PATCH_LEN}. A patch is the "
            f"vector [4 values, 4 mask bits] with a missing value written as 0 and the mask 1 for an observed bar. The embedding is "
            f"out = W_out relu(W_hidden x) + W_res x with hidden width {PATCH_HIDDEN} and token width {PATCH_WIDTH}, no biases. "
            f"{_draws(PATCH_NAME)} Draws in this order, each entry {PATCH_SCALE} normal(), each matrix row by row: W_hidden "
            f"({PATCH_HIDDEN} by {2 * PATCH_LEN}), W_res ({PATCH_WIDTH} by {2 * PATCH_LEN}), W_out ({PATCH_WIDTH} by {PATCH_HIDDEN}).\n\n")
    feat = [(p, *(_fx(v, 3) for v in r[:PATCH_LEN]), *(f"{int(v)}" for v in r[PATCH_LEN:])) for p, r in enumerate(rows)]
    t1 = _table(["patch", "v0", "v1", "v2", "v3", "m0", "m1", "m2", "m3"], feat)
    emb = [(p, *(_fx(v, 4) for v in _residual_block(r, w_hidden, w_out, w_res))) for p, r in enumerate(rows)]
    t2 = _table(["patch", "token 0", "token 1", "token 2"], emb)
    n_par = PATCH_HIDDEN * 2 * PATCH_LEN + PATCH_WIDTH * 2 * PATCH_LEN + PATCH_WIDTH * PATCH_HIDDEN
    tail = (f"\n\nThe embedding has {n_par} weights and turns the {len(PATCH_SERIES)} bars into {len(rows)} tokens of {PATCH_WIDTH} numbers. "
            f"The first patch holds {pad} padded bars and {PATCH_LEN - pad} observed ones, and its mask bits say which; "
            f"bar 5 sits inside patch 1, whose mask bit m3 is 0.")
    return head + t1 + "\n\n" + t2 + tail


PATCH_CONTEXTS = (512, 8192)
PATCH_LENGTHS = (1, 16, 32)


def patch_counts():
    rows = []
    for c in PATCH_CONTEXTS:
        for p in PATCH_LENGTHS:
            n = math.ceil(c / p)
            rows.append((c, p, n, _int(n * n), f"{(c * c) / (n * n):.0f}"))
    return ("Tokens n = ceil(C / P) of a context of C bars cut into patches of P bars, and the n^2 pairs of positions that bidirectional "
            "attention scores in one layer for one head. 'ratio' is the pair count at P = 1 divided by the pair count at the row's P.\n\n"
            + _table(["context C", "patch P", "tokens n", "pairs n^2", "ratio"], rows))


# ------------------------------------------------------------------ the scalers of the four forecasters

SCALE_CTX, SCALE_P = 64, 8
EPS_KRONOS, CLIP_KRONOS = 1e-5, 5.0
TOL_REVIN = 1e-6


def _pop_stats(x):
    m = sum(x) / len(x)
    return m, math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def _norm_whole(x, arcsinh=True):
    """Whole-context standardisation (loc = mean, scale = population sd, 1e-5 for a constant row), then arcsinh."""
    loc, scale = _pop_stats(x)
    if scale == 0.0:
        scale = 1e-5
    z = [(v - loc) / scale for v in x]
    return [math.asinh(v) for v in z] if arcsinh else z


def _norm_zscore_clip(x):
    """Kronos: (x - mean) / (sd + 1e-5), clipped to [-5, 5]."""
    m, s = _pop_stats(x)
    return [max(-CLIP_KRONOS, min(CLIP_KRONOS, (v - m) / (s + EPS_KRONOS))) for v in x]


def _norm_first_patch(x, patch=SCALE_P):
    """FinCast: loc and scale of the first patch with at least 3 valid values (here every value is valid, so patch 0)."""
    m, s = _pop_stats(x[:patch])
    s = max(s, TOL_REVIN)
    return [(v - m) / s for v in x]


def _merge(n, mu, sigma, n2, mu2, sigma2):
    """The parallel-variance merge of two groups with biased (divisor n) standard deviations."""
    nn = n + n2
    m = (n * mu + n2 * mu2) / nn
    var = (n * sigma ** 2 + n2 * sigma2 ** 2 + n * (mu - m) ** 2 + n2 * (mu2 - m) ** 2) / nn
    return nn, m, math.sqrt(var)


def _running_stats(x, patch=SCALE_P):
    """(n, mu, sigma) after each patch, by merging one patch at a time."""
    out, state = [], (0, 0.0, 0.0)
    for p in range(len(x) // patch):
        m, s = _pop_stats(x[p * patch:(p + 1) * patch])
        state = (patch, m, s) if state[0] == 0 else _merge(*state, patch, m, s)
        out.append(state)
    return out


def _norm_running(x, patch=SCALE_P):
    """TimesFM-3: patch p is standardised with the statistics of patches 0 .. p; a deviation below 1e-6 counts as 1."""
    stats = _running_stats(x, patch)
    out = []
    for p, (_, m, s) in enumerate(stats):
        s = 1.0 if s < TOL_REVIN else s
        out.extend((v - m) / s for v in x[p * patch:(p + 1) * patch])
    return out


SCHEMES = (("whole context, then arcsinh", _norm_whole), ("z-score and clip to +-5", _norm_zscore_clip),
           ("first patch", _norm_first_patch), ("running, patch by patch", _norm_running))


def _scale_series():
    return ts.sample_values()[:SCALE_CTX]


def scalers():
    x = _scale_series()
    x2 = [10.0 * v + 100.0 for v in x]
    bumped = list(x)
    for i in range(SCALE_CTX - SCALE_P, SCALE_CTX):
        bumped[i] += 50.0
    early = SCALE_CTX - SCALE_P
    rows = []
    for name, f in SCHEMES:
        a, b, c = f(x), f(x2), f(bumped)
        inv = max(abs(u - v) for u, v in zip(a, b))
        changed = sum(1 for i in range(early) if abs(a[i] - c[i]) > 1e-12)
        rows.append((name, "below 1e-9" if inv < 1e-9 else _sci(inv), f"{changed} of {early}"))
    stats = _running_stats(x)
    worst = 0.0
    for p, (n, m, s) in enumerate(stats):
        dm, ds = _pop_stats(x[:(p + 1) * SCALE_P])
        worst = max(worst, abs(m - dm), abs(s - ds))
    head = (f"The first {SCALE_CTX} bars of the series of Chapter 1 (seed {bookrng.seed_of(ts.SERIES_NAME):#018x}), cut into {SCALE_CTX // SCALE_P} "
            f"patches of {SCALE_P}. 'invariance' is the largest absolute difference between the scaled values of the series x and of "
            f"10 x + 100. 'earlier values changed' adds 50 to the {SCALE_P} bars of the last patch and counts how many of the "
            f"{early} scaled values of the earlier patches then move by more than 1e-12. The arcsinh is the library function; the "
            f"z-score divides by the population standard deviation plus 1e-5; the first patch uses the population standard "
            f"deviation, floored at 1e-6; the running scheme merges the statistics of patch p into those of patches 0 .. p - 1 by "
            f"n = n1 + n2, mu = (n1 mu1 + n2 mu2) / n and n sigma^2 = n1 sigma1^2 + n2 sigma2^2 + n1 (mu1 - mu)^2 + n2 (mu2 - mu)^2, "
            f"and a standard deviation below 1e-6 is replaced by 1.\n\n")
    tail = (f"\n\nThe merged running mean and standard deviation after each patch differ from the mean and standard deviation of "
            f"the whole prefix by at most {_sci(worst, 1)}, which is rounding.")
    return head + _table(["scaler", "invariance", "earlier values changed"], rows) + tail


TREND_SLOPE, TREND_BASE, TREND_WAVE = 0.5, 10.0, 0.3
ARCSINH_Z = (0.5, 2.0, 5.0, 20.0, 50.0)


def scalers_trend():
    x = [TREND_BASE + TREND_SLOPE * t + TREND_WAVE * math.sin(2.0 * math.pi * t / 12.0) for t in range(SCALE_CTX)]
    rows = []
    for name, f in SCHEMES:
        z = f(x)
        last = z[-SCALE_P:]
        rows.append((name, _fx(min(last), 2), _fx(max(last), 2), _fx(max(abs(v) for v in z), 2)))
    head = (f"The series y_t = {TREND_BASE} + {TREND_SLOPE} t + {TREND_WAVE} sin(2 pi t / 12) for t = 0 .. {SCALE_CTX - 1}, cut into patches of "
            f"{SCALE_P}, scaled by each scheme. The columns give the smallest and the largest scaled value in the last patch and the "
            f"largest absolute scaled value anywhere.\n\n")
    t1 = _table(["scaler", "last patch, smallest", "last patch, largest", "largest anywhere"], rows)
    back = []
    for z in ARCSINH_Z:
        a = math.asinh(z)
        back.append((_fx(z, 1), _fx(a, 4), _fx(math.sinh(a), 4)))
    t2 = _table(["scaled value z", "arcsinh(z)", "sinh(arcsinh(z))"], back)
    return (head + t1 + "\n\nThe squashing of arcsinh (library functions `asinh` and `sinh`) on a standardised value z, and its inverse:\n\n" + t2)


# ------------------------------------------------------------------ a decoder over patches with a quantile head

DEC_NAME = "foundation_forecasters.decoder"
DEC_P, DEC_D, DEC_HEADS, DEC_FF, DEC_HIDDEN = 2, 4, 2, 6, 4
DEC_LEVELS = (0.1, 0.5, 0.9)
DEC_BARS = (10.0, 10.8, 11.5, 11.0, 12.2, 12.9, 12.4, 13.6)      # 8 bars -> 4 patches of 2


def _dec_weights():
    g = bookrng.generator(DEC_NAME)
    n_in = 2 * DEC_P
    w_h = [[0.5 * g.normal() for _ in range(n_in)] for _ in range(DEC_HIDDEN)]
    w_r = [[0.5 * g.normal() for _ in range(n_in)] for _ in range(DEC_D)]
    w_o = [[0.5 * g.normal() for _ in range(DEC_HIDDEN)] for _ in range(DEC_D)]
    block = tb.init_block(g, DEC_D, DEC_FF)
    head = [[g.normal() / math.sqrt(DEC_D) for _ in range(DEC_D)] for _ in range(DEC_P * len(DEC_LEVELS))]
    return w_h, w_o, w_r, block, head


def _dec_forward(bars):
    """Standardise the context, cut it into patches, embed, one causal block, head: per position the P x K outputs, step-major."""
    m, s = _pop_stats(bars)
    z = [(v - m) / s for v in bars]
    w_h, w_o, w_r, block, head = _dec_weights()
    toks = []
    for p in range(len(z) // DEC_P):
        x = z[p * DEC_P:(p + 1) * DEC_P] + [1.0] * DEC_P
        toks.append(_residual_block(x, w_h, w_o, w_r))
    out, _ = tb.block_forward(block, toks, "pre", DEC_HEADS)
    return m, s, z, [[sum(w * v for w, v in zip(row, h)) for row in head] for h in out]


def _pin(q, y, tau):
    e = y - q
    return max(tau * e, (tau - 1.0) * e)


def decoder_toy():
    m, s, z, heads = _dec_forward(DEC_BARS)
    k = len(DEC_LEVELS)
    rows = []
    for p in range(len(heads) - 1):
        target = z[(p + 1) * DEC_P:(p + 2) * DEC_P]
        out = heads[p]
        loss = _mean([_pin(out[o * k + j], target[o], DEC_LEVELS[j]) for o in range(DEC_P) for j in range(k)])
        crossing = sum(1 for o in range(DEC_P) if not out[o * k] <= out[o * k + 1] <= out[o * k + 2])
        rows.append((p, f"{target[0]:+.3f}, {target[1]:+.3f}", *(_fx(v, 3) for v in out), _fx(loss, 4), crossing))
    t1 = _table(["position", "target bars", "o0 q0.1", "o0 q0.5", "o0 q0.9", "o1 q0.1", "o1 q0.5", "o1 q0.9", "pinball", "crossed"], rows)
    w_h, w_o, w_r, block, head = _dec_weights()
    toks = []
    for p in range(len(z) // DEC_P):
        x = z[p * DEC_P:(p + 1) * DEC_P] + [1.0] * DEC_P
        toks.append(_residual_block(x, w_h, w_o, w_r))
    toks2 = [list(t) for t in toks]
    toks2[-1] = [v + 1.0 for v in toks2[-1]]
    o1, _ = tb.block_forward(block, toks, "pre", DEC_HEADS)
    o2, _ = tb.block_forward(block, toks2, "pre", DEC_HEADS)
    change = [max(abs(a - b) for a, b in zip(r1, r2)) for r1, r2 in zip(o1, o2)]
    n_par = (DEC_HIDDEN * 2 * DEC_P + DEC_D * 2 * DEC_P + DEC_D * DEC_HIDDEN + 4 * DEC_D * DEC_D + 3 * DEC_D * DEC_FF + 2 * DEC_D
             + DEC_P * len(DEC_LEVELS) * DEC_D)
    head = (f"The eight bars {', '.join(str(v) for v in DEC_BARS)}, standardised with their own mean {m:.4f} and population standard deviation {s:.4f}, "
            f"cut into 4 patches of {DEC_P}. Each patch is embedded as in the previous example (the vector [2 values, 2 mask bits], all mask bits 1) by a "
            f"residual block, then one pre-norm block of Volume V (RMS norm, causal attention with {DEC_HEADS} heads of width {DEC_D // DEC_HEADS}, SwiGLU "
            f"of hidden width {DEC_FF}, no position information), then a linear head of {DEC_P * k} outputs per position, laid out step by step "
            f"(for each of the {DEC_P} output bars the levels {', '.join(str(v) for v in DEC_LEVELS)}). {_draws(DEC_NAME)} Draws in this order: "
            f"W_hidden (4 by 4), W_res (4 by 4) and W_out (4 by 4) of the embedding, each entry 0.5 normal(), each matrix row by row; the block "
            f"by `init_block` of Volume V (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, each normal() / sqrt(fan_in), gains 1); the head "
            f"({DEC_P * k} by {DEC_D}), each entry normal() / sqrt({DEC_D}). The {n_par} weights are random, so the pinball losses below "
            f"are those of an untrained net and say nothing about skill. Position p predicts the standardised bars of patch p + 1.\n\n")
    tail = (f"\n\nOne pass gives {len(heads) - 1} supervised positions from {len(DEC_BARS)} bars; the fourth position predicts the two bars after the "
            f"context, for which training has no target in this window. Adding 1 to every entry of the last token and running the block again "
            f"changes the outputs of positions 0, 1 and 2 by {', '.join(_fx(c, 1) for c in change[:3])} (largest absolute change) and the output "
            f"of position 3 by {_fx(change[3], 3)}.")
    return head + t1 + tail


# ------------------------------------------------------------------ turning values into tokens: uniform bins

BIN_B, BIN_LIMIT = 64, 4.0
BIN_LEVELS = (5.0, 50.0, 500.0)
BIN_N = 16


def _bin_series(level):
    return [level * (1.0 + 0.25 * math.sin(2.0 * math.pi * t / 12.0) + 0.004 * t) for t in range(BIN_N)]


def _to_token(x, scale):
    w = 2.0 * BIN_LIMIT / BIN_B
    return max(0, min(BIN_B - 1, math.floor((x / scale + BIN_LIMIT) / w)))


def _from_token(k, scale):
    w = 2.0 * BIN_LIMIT / BIN_B
    return (-BIN_LIMIT + (k + 0.5) * w) * scale


def value_tokens():
    w = 2.0 * BIN_LIMIT / BIN_B
    rows = []
    ref = None
    for lv in BIN_LEVELS:
        y = _bin_series(lv)
        scale = sum(abs(v) for v in y) / len(y)
        toks = [_to_token(v, scale) for v in y]
        back = [_from_token(k, scale) for k in toks]
        err = max(abs(a - b) for a, b in zip(y, back))
        ref = ref or toks
        rows.append((lv, _fx(scale, 3), "same" if toks == ref else "different", _fx(err, 4), _fx(err / scale, 4)))
    t1 = _table(["level L", "scale s", "tokens vs level 5", "largest error", "error / s"], rows)
    first = [_to_token(v, sum(abs(u) for u in _bin_series(BIN_LEVELS[0])) / BIN_N) for v in _bin_series(BIN_LEVELS[0])]
    y = _bin_series(10.0)
    spike = list(y)
    spike[-1] = 12.0 * _mean(y)
    scale = sum(abs(v) for v in spike) / len(spike)
    clipped = _from_token(_to_token(spike[-1], scale), scale)
    s0 = sum(abs(v) for v in y) / len(y)
    t2 = [("without the spike", _fx(s0, 3), _fx(abs(y[-1] - _from_token(_to_token(y[-1], s0), s0)), 3)),
          ("last bar replaced by 12 times the mean", _fx(scale, 3), _fx(abs(spike[-1] - clipped), 3))]
    t2 = _table(["context", "scale s", "error of the last bar after the round trip"], t2)
    # a categorical forecast read as quantiles
    mu, sd = 0.3, 0.5
    cdf = [0.0]
    nd = __import__("statistics").NormalDist(mu, sd)
    edges = [-BIN_LIMIT + k * w for k in range(BIN_B + 1)]
    probs = [(nd.cdf(edges[k + 1]) if k < BIN_B - 1 else 1.0) - (nd.cdf(edges[k]) if k > 0 else 0.0) for k in range(BIN_B)]
    acc, cum = 0.0, []
    for p in probs:
        acc += p
        cum.append(acc)
    rd = []
    for tau in (0.1, 0.5, 0.9):
        k = next(i for i, c in enumerate(cum) if c >= tau)
        q_bin = -BIN_LIMIT + (k + 0.5) * w
        q_true = nd.inv_cdf(tau)
        rd.append((tau, k, _fx(q_bin, 4), _fx(q_true, 4), _fx(abs(q_bin - q_true), 4)))
    t3 = _table(["level", "first bin with cumulative probability >= level", "centre of that bin", "exact quantile", "difference"], rd)
    head = (f"A context of {BIN_N} bars y_t = L (1 + 0.25 sin(2 pi t / 12) + 0.004 t), t = 0 .. {BIN_N - 1}, at the three levels L, "
            f"scaled by its mean absolute value s and cut into {BIN_B} bins of width {w} between -{BIN_LIMIT} and {BIN_LIMIT}: "
            f"the token of x is clamp(floor((x / s + {BIN_LIMIT}) / {w}), 0, {BIN_B - 1}), and a token k is read back as the centre "
            f"(-{BIN_LIMIT} + (k + 0.5) {w}) s of its bin.\n\n")
    mid = ("\n\nThe same code with the last bar of the level-10 context replaced by 12 times the mean of that context (the scale is recomputed "
           f"from the changed context, and a value beyond {BIN_LIMIT} scales falls into the last bin):\n\n")
    end = (f"\n\nA forecast of the next scaled bar as a categorical distribution over the {BIN_B} bins: the probability of bin k is the Gaussian "
           f"N({mu}, {sd}^2) probability of the interval between its edges (the first and last bins take the tails). Read as quantiles by "
           f"taking the centre of the first bin whose cumulative probability reaches the level:\n\n")
    return head + t1 + mid + t2 + end + t3


# ------------------------------------------------------------------ the sign code of a binary spherical quantizer

BSQ_NAME = "foundation_forecasters.bsq"
BSQ_KS = (2, 4, 6, 10, 20)
BSQ_DRAWS = 2000
BSQ_EXAMPLES = ((0.8, -0.3, 1.5, -2.0, 0.1, -0.7), (-1.1, 0.4, -0.2, 0.9, 1.3, -0.5), (0.2, 0.2, -0.2, -1.5, 0.6, 2.2))
BSQ_S1, BSQ_S2 = 3, 3


def _bsq_bits(z):
    return [1 if v > 0 else 0 for v in z]


def _bsq_index(bits):
    return sum(b << j for j, b in enumerate(bits))


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def bsq():
    k = BSQ_S1 + BSQ_S2
    rows = []
    for z in BSQ_EXAMPLES:
        bits = _bsq_bits(z)
        s1, s2 = _bsq_index(bits[:BSQ_S1]), _bsq_index(bits[BSQ_S1:])
        full = [(2 * b - 1) / math.sqrt(k) for b in bits]
        coarse = [(2 * b - 1) / math.sqrt(k) for b in bits[:BSQ_S1]] + [0.0] * BSQ_S2
        rows.append((", ".join(f"{v:+.1f}" for v in z), "".join(str(b) for b in bits), s1, s2, _fx(_cosine(z, coarse), 3), _fx(_cosine(z, full), 3)))
    t1 = _table(["latent z", "bits", "s1", "s2", "cosine, s1 only", "cosine, s1 and s2"], rows)
    g = bookrng.generator(BSQ_NAME)
    cs = []
    for kk in BSQ_KS:
        vals = []
        for _ in range(BSQ_DRAWS):
            z = [g.normal() for _ in range(kk)]
            vals.append(_cosine(z, [(2 * b - 1) for b in _bsq_bits(z)]))
        cs.append((kk, _int(2 ** kk), _fx(_mean(vals), 4)))
    t2 = _table(["bits k", "distinct codes 2^k", "mean cosine of z with its corner"], cs)
    limit = math.sqrt(2.0 / math.pi)
    d = 512
    flat = (2 ** 20) * d
    hier = 2 * (2 ** 10) * d
    t3 = _table(["output head over the 20 bits", "rows", "weights at width 512"], [("one softmax over all 2^20 codes", _int(2 ** 20), _int(flat)),
                                                                                      ("s1 and s2, two softmaxes of 2^10", _int(2 * 2 ** 10), _int(hier))])
    head = (f"The sign code of a latent z in R^{k}: bit j is 1 when z_j > 0 and 0 otherwise (so z_j = 0 gives 0), the first {BSQ_S1} bits make the "
            f"coarse token s1 and the last {BSQ_S2} the fine token s2, each as an integer with bit j worth 2^j (the first bit is the least "
            f"significant), and the code is read back as the vector with entries (2 bit - 1) / sqrt({k}), of length 1. 'cosine, s1 only' keeps the "
            f"coarse bits and puts 0 where the fine bits were.\n\n")
    mid = (f"\n\nHow much of a direction survives the sign? For z with independent standard Gaussian entries, the cosine between z and "
           f"its corner (2 bit - 1) is the sum of |z_j| over sqrt(k) times the length of z. {_draws(BSQ_NAME)} For k in "
           f"{', '.join(str(v) for v in BSQ_KS)} in this order, {BSQ_DRAWS} vectors of k calls of normal() each; the limit for large k is "
           f"sqrt(2 / pi) = {limit:.4f}.\n\n")
    end = ("\n\nRows and weights of the output matrix that maps a hidden state of 512 numbers to the logits over the 20-bit code of the "
           "next bar (biases not counted):\n\n")
    return head + t1 + mid + t2 + end + t3


# ------------------------------------------------------------------ group attention across a target and its covariates

GROUP_NAME = "foundation_forecasters.group"
GROUP_STATES = ((0.9, -0.4, 0.2, 1.1), (1.0, -0.5, 0.1, 1.0), (-0.8, 0.9, -1.2, 0.1))
GROUP_NAMES = ("target", "covariate A", "covariate B")
GROUP_SCALE = 0.7
SHARP_NAME = "foundation_forecasters.sharpness"
SHARP_DIMS = (4, 16, 64)
SHARP_DRAWS = 2000


def _rms_row(x, eps=1e-6):
    r = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * r for v in x]


def _mv(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]


def _softmax_row(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    t = sum(e)
    return [v / t for v in e]


def _group_weights():
    g = bookrng.generator(GROUP_NAME)
    return [[[GROUP_SCALE * g.normal() for _ in range(4)] for _ in range(4)] for _ in range(4)]   # Wq, Wk, Wv, Wo


def _group_attention(states, scaled):
    wq, wk, wv, wo = _group_weights()
    xs = [_rms_row(h) for h in states]
    q, k, v = ([_mv(w, x) for x in xs] for w in (wq, wk, wv))
    d = len(q[0])
    scale = 1.0 / math.sqrt(d) if scaled else 1.0
    attn = [_softmax_row([scale * sum(a * b for a, b in zip(q[i], k[j])) for j in range(len(xs))]) for i in range(len(xs))]
    mixed = [[sum(attn[i][j] * v[j][c] for j in range(len(xs))) for c in range(d)] for i in range(len(xs))]
    return attn, [_mv(wo, m) for m in mixed], _mv(wo, v[0])


def group_attention():
    rows_u, upd, alone = _group_attention(GROUP_STATES, scaled=False)
    rows_s, _, _ = _group_attention(GROUP_STATES, scaled=True)
    t1 = _table(["row (query)", *(f"to {n}" for n in GROUP_NAMES)],
                [(f"{GROUP_NAMES[i]}, unscaled", *(_fx(w, 3) for w in rows_u[i])) for i in range(3)]
                + [(f"{GROUP_NAMES[i]}, divided by sqrt(4)", *(_fx(w, 3) for w in rows_s[i])) for i in range(3)])
    diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(upd[0], alone)))
    norm = math.sqrt(sum(a * a for a in upd[0]))
    t2 = _table(["update of the target token", "c0", "c1", "c2", "c3"],
                [("with the two covariates in the group", *(_fx(v, 3) for v in upd[0])), ("alone in its group (weight 1 on itself)", *(_fx(v, 3) for v in alone))])
    g = bookrng.generator(SHARP_NAME)
    sh = []
    for d in SHARP_DIMS:
        mu = mx = 0.0
        for _ in range(SHARP_DRAWS):
            q = [g.normal() for _ in range(d)]
            ks = [[g.normal() for _ in range(d)] for _ in range(3)]
            s = [sum(a * b for a, b in zip(q, k)) for k in ks]
            mu += max(_softmax_row(s))
            mx += max(_softmax_row([v / math.sqrt(d) for v in s]))
        sh.append((d, _fx(mu / SHARP_DRAWS, 3), _fx(mx / SHARP_DRAWS, 3)))
    t3 = _table(["key width d_k", "mean largest weight, unscaled", "mean largest weight, divided by sqrt(d_k)"], sh)
    head = (f"Three token states at one patch position, one per series of a group (width 4, one head of width 4): "
            + "; ".join(f"{n} {tuple(h)}" for n, h in zip(GROUP_NAMES, GROUP_STATES))
            + f". Each is RMS-normalised (gain 1, epsilon 1e-6), projected by W_Q, W_K and W_V, the three rows of scores q_i . k_j go through a softmax "
              f"with no position information, the mixed values go through W_O. {_draws(GROUP_NAME)} Draws: W_Q, W_K, W_V, W_O in this order, each 4 by 4, "
              f"each entry {GROUP_SCALE} normal(), row by row. The attention weights, once with the scores as they are and once divided by sqrt(4):\n\n")
    mid = (f"\n\nThe update the target token receives from the group sublayer is W_O (sum_j a_0j v_j); the euclidean distance between it and the update "
           f"of a target alone in its group, W_O v_0, is {diff:.3f} against a length of {norm:.3f}.\n\n")
    end = (f"\n\nWhy the scale of a score matters: for q and three keys with independent standard Gaussian entries of width d_k, the largest of the "
           f"three weights averaged over {SHARP_DRAWS} draws. {_draws(SHARP_NAME)} Per draw: q (d_k calls of normal()), then the three keys in order.\n\n")
    return head + t1 + mid + t2 + end + t3


# ------------------------------------------------------------------ the input of a patch for a target and for a covariate

COV_P, COV_POUT, COV_CTX, COV_HOR = 2, 4, 6, 4
COV_TARGET = (3, 4, 5, 6, 7, 8)
COV_KNOWN = (0, 0, 0, 1, 1, 0, 0, 1, 1, 1)           # 6 context bars and 4 known future bars


def _layout_rows(target_like, values, P, rolls, n_patches, n_ctx):
    """Rows [values (P) | next rolls patches (rolls P) | mask (P) | mask of the next patches (rolls P)], mask 1 = masked."""
    flat = list(values) + [None] * (n_patches * P - len(values))
    rows = []
    for p in range(n_patches):
        own = flat[p * P:(p + 1) * P]
        fut = []
        for k in range(rolls * P):
            src = p * P + P + k
            fut.append(None if (src >= n_patches * P or target_like) else flat[src])
        rows.append((own, fut))
    return rows


def covariate_layout():
    P, rolls = COV_P, COV_POUT // COV_P
    extract = min(2 * P, COV_POUT)
    overlap = extract - P
    forecast_patches = max(-(-(COV_HOR - overlap) // P), 1)
    horizon_patches = forecast_patches + rolls - 1
    n_ctx = COV_CTX // P
    n_patches = n_ctx + horizon_patches
    tgt = _layout_rows(True, list(COV_TARGET) + [None] * (horizon_patches * P), P, rolls, n_patches, n_ctx)
    cov = _layout_rows(False, list(COV_KNOWN), P, rolls, n_patches, n_ctx)

    def cell(v):
        return "0" if v is None else str(v)

    def mask(v):
        return "1" if v is None else "0"

    rows = []
    for p in range(n_patches):
        for name, r in (("target", tgt), ("known-future covariate", cov)):
            own, fut = r[p]
            rows.append((p, name, " ".join(cell(v) for v in own), " ".join(cell(v) for v in fut), " ".join(mask(v) for v in own), " ".join(mask(v) for v in fut)))
    head = (f"Input patches P = {P}, output patches {COV_POUT} (so {rolls} input patches ahead), a context of {COV_CTX} bars and a horizon of {COV_HOR}. "
            f"The stitched output keeps min(2 P, {COV_POUT}) = {extract} bars per forecast patch, so the overlap is {overlap} bars and the number of "
            f"forecast patches is max(ceil(({COV_HOR} - {overlap}) / {P}), 1) = {forecast_patches}; the model reads {horizon_patches} = {forecast_patches} + {rolls} - 1 "
            f"patches beyond the {n_ctx} of the context, {n_patches} in all. The target is {', '.join(str(v) for v in COV_TARGET)}; the covariate is "
            f"{', '.join(str(v) for v in COV_KNOWN)} over the context and the horizon. Each row is [values of the patch ({P}) | values of the next {rolls} patches "
            f"({rolls * P}) | mask of the patch | mask of the next patches] with a mask bit of 1 for a value that is missing, a bar beyond the last patch or "
            f"hidden from a target; values are shown before the standardisation.\n\n")
    return head + _table(["patch", "variate", "values", "next patches", "mask", "next-patch mask"], rows, "llllll")


# ------------------------------------------------------------------ stitching overlapping output patches

STITCH_NAME = "foundation_forecasters.stitching"
STITCH_P, STITCH_EXTRACT, STITCH_H, STITCH_REPS = 4, 8, 20, 400
STITCH_OFFSET, STITCH_SLOPE = 0.30, 0.05


def _stitch_weights(overlap):
    step = 1.0 / (max(overlap, 2) - 1)
    return [1.0 - k * step for k in range(overlap)]


def _stitch(preds, P, extract):
    """The cross-fade of the TimesFM-3 postprocess: the first P of patch 0, a cross-fade over each overlap, the rest of each next patch,
    and the overlap tail of the last patch."""
    F = len(preds)
    overlap = extract - P
    if F == 1:
        return list(preds[0])
    w = _stitch_weights(overlap)
    out = list(preds[0][:P])
    for f in range(F - 1):
        prev, nxt = preds[f], preds[f + 1]
        out.extend(w[k] * prev[P + k] + (1.0 - w[k]) * nxt[k] for k in range(overlap))
        out.extend(nxt[overlap:P])
    out.extend(preds[-1][P:P + overlap])
    return out


def _hard(preds, P, extract):
    out = []
    for f, p in enumerate(preds):
        out.extend(p[:P])
    out.extend(preds[-1][P:extract])
    return out


def _stitch_truth(n):
    return [10.0 + 0.3 * s + 2.0 * math.sin(2.0 * math.pi * s / 12.0) for s in range(n)]


def _stitch_first():
    """Truth, the cross-fade and the hard switch of the first repetition of the experiment below (the first four draw pairs)."""
    P, ex = STITCH_P, STITCH_EXTRACT
    overlap = ex - P
    F = max(-(-(STITCH_H - overlap) // P), 1)
    truth = _stitch_truth(F * P + overlap + ex)
    g = bookrng.generator(STITCH_NAME)
    preds = []
    for f in range(F):
        a, b = STITCH_OFFSET * g.normal(), STITCH_SLOPE * g.normal()
        preds.append([truth[P * f + j] + a + b * j for j in range(ex)])
    return truth[:STITCH_H], _stitch(preds, P, ex)[:STITCH_H], _hard(preds, P, ex)[:STITCH_H], preds


def stitching():
    P, ex = STITCH_P, STITCH_EXTRACT
    overlap = ex - P
    F = max(-(-(STITCH_H - overlap) // P), 1)
    out_len = F * P + overlap
    truth = _stitch_truth(out_len + ex)
    g = bookrng.generator(STITCH_NAME)
    se = {"cross-fade": 0.0, "hard switch": 0.0}
    jump = {"cross-fade": 0.0, "hard switch": 0.0}
    seams = [P * (f + 1) for f in range(F - 1)]
    cf_cover = [P * (f + 1) + k for f in range(F - 1) for k in range(overlap)]
    for _ in range(STITCH_REPS):
        preds = []
        for f in range(F):
            a, b = STITCH_OFFSET * g.normal(), STITCH_SLOPE * g.normal()
            preds.append([truth[P * f + j] + a + b * j for j in range(ex)])
        for name, fn in (("cross-fade", _stitch), ("hard switch", _hard)):
            out = fn(preds, P, ex)
            se[name] += sum((out[s] - truth[s]) ** 2 for s in range(STITCH_H)) / STITCH_H
            jump[name] += sum(abs((out[s] - out[s - 1]) - (truth[s] - truth[s - 1])) for s in seams) / len(seams)
    w = _stitch_weights(overlap)
    t1 = _table(["step k of the overlap", "weight of the earlier patch", "weight of the later patch"], [(k, _fx(w[k], 4), _fx(1.0 - w[k], 4)) for k in range(overlap)])
    t2 = _table(["rule", "mean squared error", "mean jump at the seams"], [(n, _fx(se[n] / STITCH_REPS, 4), _fx(jump[n] / STITCH_REPS, 4)) for n in se])
    head = (f"Output patches of {ex} bars from input patches of {P}, so consecutive output patches overlap in {overlap} bars and a horizon of {STITCH_H} "
            f"needs max(ceil(({STITCH_H} - {overlap}) / {P}), 1) = {F} forecast patches, which together cover {F} * {P} + {overlap} = {out_len} bars. "
            f"Forecast patch f covers bars {P} f .. {P} f + {ex - 1}. Over its overlap with the next patch the earlier patch gets the weight "
            f"1 - k / ({overlap} - 1) at step k = 0 .. {overlap - 1} and the later patch the rest:\n\n")
    mid = (f"\n\nThe truth is y_s = 10 + 0.3 * s + 2 * sin(2 pi s / 12). Each forecast patch is the truth plus its own error a + b j at its j-th bar "
           f"(j = 0 .. {ex - 1}), with a = {STITCH_OFFSET} normal() and b = {STITCH_SLOPE} normal(). {_draws(STITCH_NAME)} {STITCH_REPS} repetitions; in each, for f = 0 .. {F - 1}, "
           f"a then b. 'hard switch' takes bars {P} f .. {P} f + {P - 1} from patch f and the tail of the last patch. 'jump' is the absolute "
           f"difference between the step of the stitched output and the step of the truth across a seam, |(out_s - out_(s-1)) - (y_s - y_(s-1))|, averaged over the seams "
           f"s = {', '.join(str(s) for s in seams)} (the first bar of each later patch).\n\n")
    return head + t1 + mid + t2


# ------------------------------------------------------------------ four ways to a long horizon

LONG_CONTEXT = 448
LONG_HORIZONS = (64, 256, 1024)
FB_PHI, FB_BLOCK = 0.99, 128
FB_STEPS = (1, 64, 128, 129, 192, 256)


def _long_rows(h):
    c2_out = -(-h // 16)
    c2_tokens = -(-LONG_CONTEXT // 16) + 1 + c2_out
    tf_fp = max(-(-(h - 32) // 32), 1)
    tf_tokens = LONG_CONTEXT // 32 + tf_fp + 1
    fc_passes = -(-h // 128)
    return c2_out, c2_tokens, tf_fp, tf_tokens, fc_passes


def long_horizon():
    rows = []
    for h in LONG_HORIZONS:
        c2_out, c2_tokens, tf_fp, tf_tokens, fc_passes = _long_rows(h)
        rows.append((h, f"1 ({c2_tokens} tokens)", f"1 ({tf_tokens} tokens)", f"{fc_passes} (16 tokens each)", f"1 prefill and {h} token steps"))
    t1 = _table(["horizon H", "Chronos-2", "TimesFM-3", "FinCast", "Kronos"], rows)
    fb = []
    for h in FB_STEPS:
        true_sd = math.sqrt((1.0 - FB_PHI ** (2 * h)) / (1.0 - FB_PHI ** 2))
        j = h if h <= FB_BLOCK else h - FB_BLOCK
        fed_sd = math.sqrt((1.0 - FB_PHI ** (2 * j)) / (1.0 - FB_PHI ** 2))
        fb.append((h, _fx(true_sd, 3), _fx(fed_sd, 3), _fx(fed_sd / true_sd, 3)))
    t2 = _table(["step h", "true standard deviation", "standard deviation after feeding back the mean", "ratio"], fb)
    head = (f"Forward passes (and the tokens each pass reads) to forecast H bars from a context of {LONG_CONTEXT} bars. Chronos-2 reads ceil({LONG_CONTEXT} / 16) context "
            f"patches, one register token and ceil(H / 16) patches of horizon; TimesFM-3 reads {LONG_CONTEXT} / 32 patches of context and max(ceil((H - 32) / 32), 1) + 1 "
            f"patches of horizon; FinCast pads or truncates every context to 512 bars (16 patches of 32) and decodes 128 bars per pass; Kronos runs the decoder once over "
            f"the {LONG_CONTEXT} context tokens and then once per generated bar.\n\n")
    mid = (f"\n\nWhat feeding the mean back does to the spread. Take a process x_t = {FB_PHI} x_(t-1) + e_t with unit-variance innovations, observed up to t = 0, and a "
           f"forecaster whose quantiles are exactly the conditional quantiles given what it is shown. For horizons up to {FB_BLOCK} it is shown the true past, "
           f"and the standard deviation at step h is sqrt((1 - {FB_PHI}^(2 h)) / (1 - {FB_PHI}^2)). For the next {FB_BLOCK} bars it is shown the first block's "
           f"mean path as if it had been observed, so its spread at step h is that of a forecast h - {FB_BLOCK} steps ahead of the last bar of that mean path:\n\n")
    return head + t1 + mid + t2


# ------------------------------------------------------------------ parameter and work counts of the four forecasters

def _res_block(i, h, o, bias=True):
    """Weights of out = W_out act(W_hidden x) + W_res x with hidden width h; with biases each of the three linears has one."""
    n = h * i + o * h + o * i
    return n + (h + 2 * o if bias else 0)


def _timesfm3_inventory(d=1280, hidden=1280, hd=80, layers=20, p_in=32, p_out=64, q=9):
    width = 2 * (p_in + p_out)
    attn = 4 * d * d + 2 * d + 3 * hd                  # pre norm, q k v o, query norm, key norm, per-dim scale, post norm
    ff = 2 * d * hidden + 2 * d                        # pre norm, ff0, ff1, post norm
    layer = 2 * attn + ff
    items = [("patch embedding (3 linears, no bias)", _res_block(width, d, d, bias=False)),
             (f"sequence attention, {layers} layers", layers * attn), (f"variate attention, {layers} layers", layers * attn),
             (f"feed-forward, {layers} layers", layers * ff), ("output head (weight and bias)", p_out * q * d + p_out * q)]
    return items, layers * layer


def _chronos2_inventory(d=768, f=3072, layers=12, vocab=2, patch=16, q=21, inner=768):
    attn = 4 * d * inner + d
    mlp = 2 * d * f + d
    items = [("special-token table [PAD], [REG]", vocab * d), ("patch embedding (residual block)", _res_block(3 * patch, f, d)),
             (f"time attention, {layers} layers", layers * attn), (f"group attention, {layers} layers", layers * attn),
             (f"feed-forward, {layers} layers", layers * mlp), ("final norm", d), ("quantile head (residual block)", _res_block(d, f, q * patch))]
    return items


def _fincast_inventory(d=1280, f=1280, hd=80, heads=16, layers=50, experts=4, patch=32, horizon=128, q=9):
    inner = heads * hd
    qkv = (heads + 2 * heads) * hd
    attn = d + hd + qkv * d + qkv + d * inner + d      # input norm, per-dim scale, qkv, o
    expert = 2 * (d * d + d) + 2 * d                   # gate_proj, down_proj with biases, layer norm weight and bias
    moe = d + experts * d + experts * expert           # prenorm gamma, gate matrix, experts
    items = [("patch embedding (residual block)", _res_block(2 * patch, f, d)), ("frequency embedding", 3 * d),
             (f"attention, {layers} layers", layers * attn), (f"experts, {layers} layers of {experts}", layers * experts * expert),
             (f"gates and norms of the mixture, {layers} layers", layers * (moe - experts * expert)),
             ("horizon head (residual block)", _res_block(d, f, horizon * (1 + q)))]
    return items


def _kronos_block(d, ff):
    return 4 * (d * d + d) + 2 * d + 3 * d * ff


def _kronos_inventory(d=512, layers=8, ff=1024, bits=10):
    v = 1 << bits
    items = [("token tables emb_s1, emb_s2", 2 * v * d), ("fusion projection (weight and bias)", d * 2 * d + d),
             ("calendar tables (60, 24, 7, 32, 13 rows)", (60 + 24 + 7 + 32 + 13) * d), (f"transformer blocks, {layers}", layers * _kronos_block(d, ff)),
             ("final norm", d), ("dependency layer (4 projections with bias, norm)", 4 * (d * d + d) + d),
             ("heads proj_s1, proj_s2 (weights and biases)", 2 * (v * d + v))]
    return items


def _kronos_tokenizer_inventory(d=256, ff=512, d_in=6, bits=10, blocks=3):
    k = 2 * bits
    items = [("embed and head", d_in * d + d + d * d_in + d_in), ("quant_embed", k * d + k), ("post_quant_embed_pre and post_quant_embed", d * bits + d + d * k + d),
             (f"encoder blocks, {blocks}", blocks * _kronos_block(d, ff)), (f"decoder blocks, {blocks}", blocks * _kronos_block(d, ff))]
    return items


def parameter_counts():
    tf, tf_total = _timesfm3_inventory()
    tf_items = tf
    c2 = _chronos2_inventory()
    fc = _fincast_inventory()
    kd = _kronos_inventory()
    kt = _kronos_tokenizer_inventory()
    sections = [("TimesFM-3", tf_items, 330710976), ("Chronos-2", c2, 119477664), ("FinCast", fc, 991436960),
                ("Kronos decoder (small)", kd, None), ("Kronos tokenizer (base)", kt, None)]
    out = []
    for name, items, published in sections:
        total = sum(n for _, n in items)
        rows = [(label, _int(n)) for label, n in items] + [("**total**", f"**{_int(total)}**")]
        note = "" if published is None else f" The count of the crate's own test is {_int(published)}: {'equal' if published == total else 'DIFFERENT'}."
        out.append(f"{name}.{note}\n\n" + _table(["part", "weights"], rows))
    return ("The weights of each forecaster by part, from the shapes of its published configuration (every linear counts its weight matrix, and "
            "its bias where the model has one; every norm its gain vector and, for a layer norm, its bias). Hidden widths and block counts are those "
            "of the layer lists in the text.\n\n" + "\n\n".join(out))


WORK_CONTEXT, WORK_HORIZON = 448, 64


def _causal_pairs(n):
    return n * (n + 1) // 2


def _work_timesfm3(c=WORK_CONTEXT, h=WORK_HORIZON, d=1280, hidden=1280, layers=20, p=32, p_out=64, q=9):
    forecast_patches = max(-(-(h - min(2 * p, p_out) + p) // p), 1)
    n = c // p + forecast_patches + p_out // p - 1
    width = 2 * (p + p_out)
    matrix = 2 * n * (2 * width * d + d * d) + layers * (2 * n * (8 * d * d + 2 * d * hidden)) + 2 * n * d * p_out * q
    attention = layers * 4 * d * (_causal_pairs(n) + n)
    return n, matrix, attention, _causal_pairs(n) + n


def _work_chronos2(c=WORK_CONTEXT, h=WORK_HORIZON, d=768, f=3072, inner=768, layers=12, patch=16, q=21):
    n_ctx, n_out = -(-c // patch), -(-h // patch)
    n = n_ctx + 1 + n_out
    embed = 2 * (n_ctx + n_out) * (3 * patch * f + f * d + 3 * patch * d)
    per_layer = 2 * n * (4 * d * inner + 2 * d * inner + 2 * d * f)         # time q k v o, group v o (one series), feed-forward
    head = 2 * n_out * (d * f + f * q * patch + d * q * patch)
    attention = layers * 4 * inner * n * n
    return n, embed + layers * per_layer + head, attention, n * n


def _work_fincast(c=WORK_CONTEXT, d=1280, f=1280, hd=80, heads=16, layers=50, patch=32, horizon=128, q=9, experts=4, top=2, routed=True):
    n = 512 // patch
    inner = heads * hd
    qkv = 3 * heads * hd
    embed = 2 * n * (2 * patch * f + f * d + 2 * patch * d)
    per_layer = 2 * n * (d * qkv + inner * d + d * experts) + 2 * n * (top if routed else experts) * 2 * d * d
    head = 2 * n * (d * f + f * horizon * (1 + q) + d * horizon * (1 + q))
    attention = layers * 4 * inner * _causal_pairs(n)
    return n, embed + layers * per_layer + head, attention, _causal_pairs(n)


def _work_kronos(c=WORK_CONTEXT, h=WORK_HORIZON, d=512, ff=1024, layers=8, vocab=1024, dt=256, fft=512, blocks=3, k=20, d_in=6, max_ctx=512):
    block_dense = lambda dd, f_: 2 * (4 * dd * dd + 3 * dd * f_)
    # tokenizer: encode the context, decode the window of context and generated bars
    n_w = min(c + h, max_ctx)
    tok_matrix = (2 * c * d_in * dt + blocks * c * block_dense(dt, fft) + 2 * c * dt * k
                  + 2 * n_w * k * dt + blocks * n_w * block_dense(dt, fft) + 2 * n_w * dt * d_in)
    tok_pairs = _causal_pairs(c) + _causal_pairs(n_w)
    tok_attention = blocks * 4 * dt * tok_pairs
    # decoder prefill of the context, then one token step and one dependency step per generated bar
    per_token = 2 * (2 * d) * d + layers * block_dense(d, ff) + 2 * d * vocab           # fusion, blocks, proj_s1
    dep_kv = 2 * 2 * d * d
    prefill_matrix = c * (per_token + dep_kv)
    prefill_pairs = _causal_pairs(c)
    step_matrix = h * (per_token + dep_kv + 2 * d * d + 2 * d * d + 2 * d * vocab)       # token step, dep kv, dep q, dep o, proj_s2
    step_pairs = sum(c + i + 1 for i in range(h))
    dep_pairs = sum(c + i for i in range(h))
    attention = tok_attention + layers * 4 * d * (prefill_pairs + step_pairs) + 4 * d * dep_pairs
    matrix = tok_matrix + prefill_matrix + step_matrix
    return c, matrix, attention, tok_pairs + prefill_pairs + step_pairs + dep_pairs


def forecast_work():
    rows = []
    items = (("Chronos-2", _work_chronos2()), ("TimesFM-3", _work_timesfm3()), ("FinCast", _work_fincast()),
             ("FinCast, all four experts run", _work_fincast(routed=False)), ("Kronos", _work_kronos()))
    for name, (n, matrix, attention, pairs) in items:
        total = matrix + attention
        rows.append((name, n, _int(pairs), _flops(matrix), _flops(attention), _flops(total), _flops(total / WORK_HORIZON)))
    head = (f"Floating-point operations (a multiply and an add count as 2) of one forecast of {WORK_HORIZON} bars from a context of {WORK_CONTEXT} bars of one series, "
            f"counting only matrix products and the two products of attention (scores and weighted sum, 4 d per scored pair of positions), and no norms, "
            f"activations, softmaxes, table lookups or sampling. 'tokens' is the length of the longest sequence the backbone reads (for Kronos the context "
            f"tokens of the prefill); 'pairs' is the number of (query, key) pairs scored, causal pairs counted once, bidirectional ones in both directions. "
            f"Chronos-2 runs the group sublayer of a series alone as its value and output projections only; FinCast pads its context to 512 bars; "
            f"for Kronos the rows include the tokenizer (encode the context, decode the 512-bar window of context and forecast), the prefill, and "
            f"{WORK_HORIZON} steps each of one token step and one dependency step.\n\n")
    return head + _table(["forecaster", "tokens", "pairs scored", "matrix FLOPs", "attention FLOPs", "total FLOPs", "per forecast bar"], rows)


# ------------------------------------------------------------------ a shared map pretrained on a corpus, applied to series it has never seen

ZS_CTX, ZS_HOR, ZS_LEN, ZS_ORIGIN = 48, 12, 120, 96
ZS_M_IN, ZS_M_OUT, ZS_M_NEW = 12, 8, 10
ZS_MARKS = (5, 20, 100, 400)
ZS_RIDGE = 1.0
ZS_NAMES = {"corpus": "foundation_forecasters.corpus", "corpus8": "foundation_forecasters.corpus_period8",
            "in": "foundation_forecasters.test_in", "out": "foundation_forecasters.test_out", "new": "foundation_forecasters.test_new"}
ZS_TEST = 200


def _family_series(g, m, n=ZS_LEN):
    """One series of the family: level L = 5 * 100^u, slope, amplitude, phase and shock size drawn in this order (one uniform() each),
    then y_t = L (1 + slope t + amp sin(2 pi t / m + phase) + v_t), v_t = 0.7 v_(t-1) + shock normal()."""
    level = 5.0 * 100.0 ** g.uniform()
    slope = -0.002 + 0.010 * g.uniform()
    amp = 0.05 + 0.25 * g.uniform()
    phase = 2.0 * math.pi * g.uniform()
    shock = 0.02 + 0.04 * g.uniform()
    v, y = 0.0, []
    for t in range(n):
        v = 0.7 * v + shock * g.normal()
        y.append(level * (1.0 + slope * t + amp * math.sin(2.0 * math.pi * t / m + phase) + v))
    return y


def _scaled_window(ctx, tgt):
    m, s = _pop_stats(ctx)
    d = s + ts.SCALE_EPS
    return m, d, [(v - m) / d for v in ctx], [(v - m) / d for v in tgt]


def _windows(y):
    for o in range(ZS_CTX, ZS_LEN - ZS_HOR + 1):
        yield _scaled_window(y[o - ZS_CTX:o], y[o:o + ZS_HOR])[2:]


class _Gram:
    """Accumulates X^T X and X^T Y over rows [1, x ...] with 12 targets; windows in order, features in order."""

    def __init__(self, k, n_out):
        self.k, self.n_out = k, n_out
        self.a = [[0.0] * k for _ in range(k)]
        self.b = [[0.0] * n_out for _ in range(k)]

    def copy(self):
        c = _Gram(self.k, self.n_out)
        c.a = [list(r) for r in self.a]
        c.b = [list(r) for r in self.b]
        return c

    def add(self, row, ys):
        for i in range(self.k):
            ri = row[i]
            ai = self.a[i]
            for j in range(i, self.k):
                ai[j] += ri * row[j]
            bi = self.b[i]
            for o in range(self.n_out):
                bi[o] += ri * ys[o]

    def solve(self, ridge, k=None):
        """One weight vector per output, using the first k features (all by default)."""
        k = k or self.k
        a = [[self.a[min(i, j)][max(i, j)] + (ridge if i == j and i > 0 else 0.0) for j in range(k)] for i in range(k)]
        return [_solve(a, [self.b[i][o] for i in range(k)]) for o in range(self.n_out)]


def _predict(weights, row):
    return [sum(w * v for w, v in zip(wo, row)) for wo in weights]


@lru_cache(maxsize=1)
def _zero_shot_results():
    g = bookrng.generator(ZS_NAMES["corpus"])
    gram = _Gram(ZS_CTX + 1, ZS_HOR)
    maps = {}
    snap200 = None
    for i in range(1, max(ZS_MARKS) + 1):
        y = _family_series(g, ZS_M_IN)
        for x, t in _windows(y):
            gram.add([1.0] + x, t)
        if i in ZS_MARKS:
            maps[f"{i} series"] = gram.solve(ZS_RIDGE)
        if i == 200:
            snap200 = gram.copy()
    g8 = bookrng.generator(ZS_NAMES["corpus8"])
    for i in range(200):
        y = _family_series(g8, ZS_M_OUT)
        for x, t in _windows(y):
            snap200.add([1.0] + x, t)
    maps["200 series of period 12 and 200 of period 8"] = snap200.solve(ZS_RIDGE)
    results = {}
    for fam, m in (("in", ZS_M_IN), ("out", ZS_M_OUT), ("new", ZS_M_NEW)):
        gt = bookrng.generator(ZS_NAMES[fam])
        res = {}
        for _ in range(ZS_TEST):
            y = _family_series(gt, m)
            ctx, tgt = y[ZS_ORIGIN - ZS_CTX:ZS_ORIGIN], y[ZS_ORIGIN:ZS_ORIGIN + ZS_HOR]
            mu, d, x, _t = _scaled_window(ctx, tgt)
            preds = {"seasonal naive": ts._fc_snaive(ctx, ZS_HOR, m), "AR(1) on seasonal differences, fitted to the 48 bars": ts._fc_ar(ctx, ZS_HOR, 1, m)}
            for name, w in maps.items():
                preds[f"shared map, {name}"] = [mu + d * v for v in _predict(w, [1.0] + x)]
            for name, f in preds.items():
                res.setdefault(name, []).append(_mean([abs(a - b) for a, b in zip(f, tgt)]) / d)
        results[fam] = res
    return results


def zero_shot():
    res = _zero_shot_results()
    names = list(res["in"])
    rows = [(n, *(_fx(_mean(res[f][n]), 3) for f in ("in", "out", "new"))) for n in names]
    seeds = ", ".join(f"{k}: {bookrng.seed_of(v):#018x}" for k, v in ZS_NAMES.items())
    head = (f"A family of series: y_t = L (1 + s t + a sin(2 pi t / m + phi) + v_t) for t = 0 .. {ZS_LEN - 1}, with L = 5 * 100^u, s = -0.002 + 0.010 u, a = 0.05 + 0.25 u, "
            f"phi = 2 pi u and a shock size c = 0.02 + 0.04 u (each u one uniform() call, in this order), and then for t = 0 .. {ZS_LEN - 1} one normal() call each "
            f"for v_t = 0.7 v_(t-1) + c normal(), v_(-1) = 0. The period m is {ZS_M_IN} in the corpus and in the first test family, {ZS_M_OUT} in the second test family "
            f"(and in the second corpus stream) and {ZS_M_NEW} in the third. Generators (SplitMix64, seeds): {seeds}; the series are drawn one after another from each "
            f"stream. A window is {ZS_CTX} context bars and {ZS_HOR} target bars; both are scaled by the mean and population standard deviation (plus 1e-5) of the "
            f"context. The shared map is the ridge regression (penalty {ZS_RIDGE} on every weight but the constant) from [1, the {ZS_CTX} scaled context bars] to the "
            f"{ZS_HOR} scaled target bars, fitted over every window of every corpus series with its context starting at bar 0 .. {ZS_LEN - ZS_HOR - ZS_CTX} (origins "
            f"{ZS_CTX} .. {ZS_LEN - ZS_HOR}, {ZS_LEN - ZS_HOR - ZS_CTX + 1} windows per series), accumulating the normal equations window by window and feature by "
            f"feature, solved by elimination with partial pivoting. The corpus is the first 5, 20, 100 and 400 series of the first stream (all of period {ZS_M_IN}); "
            f"the last row adds 200 series from the second stream to the first 200 of the first. Each test family has {ZS_TEST} fresh series, one window each, context "
            f"bars {ZS_ORIGIN - ZS_CTX} .. {ZS_ORIGIN - 1} and target bars {ZS_ORIGIN} .. {ZS_ORIGIN + ZS_HOR - 1}. The two baselines are told the period of the series "
            f"they forecast. 'scaled MAE' is the mean absolute error over the {ZS_HOR} target bars, in units of the context's standard deviation, averaged over the "
            f"series.\n\n")
    return head + _table(["forecaster", f"period {ZS_M_IN}", f"period {ZS_M_OUT}", f"period {ZS_M_NEW}"], rows)


# ------------------------------------------------------------------ the same map with a covariate whose future is known

COV_NAMES = {"corpus": "foundation_forecasters.cov_corpus", "test": "foundation_forecasters.cov_test", "shuffle": "foundation_forecasters.cov_shuffle"}
COV_CORPUS, COV_TEST, COV_STEP, COV_PROMO = 150, 200, 3, 0.2


def _promo_series(g, m=ZS_M_IN, n=ZS_LEN):
    """Draws: L, s, a, phi, c, d (six uniform() calls: d = 0.10 + 0.20 u is the size of a promotion), then for each bar one normal() (the shock) and one
    uniform() (a promotion on the bar when below 0.2). y_t = L (1 + s t + a sin(2 pi t / m + phi) + v_t + d x_t)."""
    level = 5.0 * 100.0 ** g.uniform()
    slope = -0.002 + 0.010 * g.uniform()
    amp = 0.05 + 0.25 * g.uniform()
    phase = 2.0 * math.pi * g.uniform()
    shock = 0.02 + 0.04 * g.uniform()
    delta = 0.10 + 0.20 * g.uniform()
    v, y, x = 0.0, [], []
    for t in range(n):
        v = 0.7 * v + shock * g.normal()
        promo = 1.0 if g.uniform() < COV_PROMO else 0.0
        x.append(promo)
        y.append(level * (1.0 + slope * t + amp * math.sin(2.0 * math.pi * t / m + phase) + v + delta * promo))
    return y, x


def _cov_row(xn, cov, o):
    return [1.0] + xn + cov[o - ZS_CTX:o] + cov[o:o + ZS_HOR]


@lru_cache(maxsize=1)
def _covariate_results():
    g = bookrng.generator(COV_NAMES["corpus"])
    gram = _Gram(1 + ZS_CTX + ZS_CTX + ZS_HOR, ZS_HOR)
    for _ in range(COV_CORPUS):
        y, x = _promo_series(g)
        for o in range(ZS_CTX, ZS_LEN - ZS_HOR + 1, COV_STEP):
            _, _, xn, tn = _scaled_window(y[o - ZS_CTX:o], y[o:o + ZS_HOR])
            gram.add(_cov_row(xn, x, o), tn)
    w_plain = gram.solve(ZS_RIDGE, 1 + ZS_CTX)
    w_cov = gram.solve(ZS_RIDGE)
    gt = bookrng.generator(COV_NAMES["test"])
    tests = [_promo_series(gt) for _ in range(COV_TEST)]
    order = list(range(COV_TEST))
    bookrng.generator(COV_NAMES["shuffle"]).shuffle(order)
    res = {}
    for i, (y, x) in enumerate(tests):
        o = ZS_ORIGIN
        ctx, tgt = y[o - ZS_CTX:o], y[o:o + ZS_HOR]
        mu, d, xn, _ = _scaled_window(ctx, tgt)
        other = tests[order[i]][1]
        variants = {"seasonal naive": ts._fc_snaive(ctx, ZS_HOR, ZS_M_IN),
                    "shared map without covariate": [mu + d * v for v in _predict(w_plain, [1.0] + xn)],
                    "with the true future promotions": [mu + d * v for v in _predict(w_cov, _cov_row(xn, x, o))],
                    "with another series' future promotions": [mu + d * v for v in _predict(w_cov, _cov_row(xn, x[:o] + other[o:o + ZS_HOR], o))],
                    "with the future promotions set to none": [mu + d * v for v in _predict(w_cov, _cov_row(xn, x[:o] + [0.0] * ZS_HOR, o))]}
        for name, f in variants.items():
            res.setdefault(name, []).append(_mean([abs(a - b) for a, b in zip(f, tgt)]) / d)
    return res, sum(sum(x[ZS_ORIGIN:ZS_ORIGIN + ZS_HOR]) for _, x in tests) / COV_TEST


def covariate_gain():
    res, mean_promos = _covariate_results()
    rows = [(n, _fx(_mean(v), 3)) for n, v in res.items()]
    seeds = ", ".join(f"{k}: {bookrng.seed_of(v):#018x}" for k, v in COV_NAMES.items())
    head = (f"The family of the previous example (period {ZS_M_IN}) with a promotion indicator x_t: each series draws L, s, a, phi, c and d = 0.10 + 0.20 u (six uniform() "
            f"calls in this order), and then for each of the {ZS_LEN} bars one normal() (the shock of v_t) and one uniform() (a promotion, x_t = 1, when it is below "
            f"{COV_PROMO}); y_t = L (1 + s t + a sin(2 pi t / 12 + phi) + v_t + d x_t). Generators (seeds): {seeds}; the corpus is the first {COV_CORPUS} series of its stream, "
            f"the test is {COV_TEST} fresh series, and the third stream shuffles the list 0 .. {COV_TEST - 1} (Fisher-Yates of the book) to pick, for series i, the future "
            f"promotions of series list[i]. A window is as before, with origins {ZS_CTX}, {ZS_CTX + COV_STEP}, .. {ZS_LEN - ZS_HOR} for the corpus, so {len(range(ZS_CTX, ZS_LEN - ZS_HOR + 1, COV_STEP))} windows "
            f"per series. The map with a covariate is the ridge regression (penalty {ZS_RIDGE}) from [1, {ZS_CTX} scaled context bars, the {ZS_CTX} promotion bits of the context, "
            f"the {ZS_HOR} promotion bits of the horizon] to the {ZS_HOR} scaled target bars; the map without one uses only the first {ZS_CTX + 1} of these inputs. "
            f"Test windows are as in the previous example (target bars {ZS_ORIGIN} .. {ZS_ORIGIN + ZS_HOR - 1}); on average a test series has {mean_promos:.3f} promotions among them.\n\n")
    return head + _table(["forecaster", "scaled MAE"], rows)


# ------------------------------------------------------------------ comparing a port with its reference

PARITY_WANT = (1.0, -2.0, 3.0, 0.5)
PARITY_CASES = (("1.05 times the reference", tuple(1.05 * v for v in PARITY_WANT)),
                ("the reference plus 0.1 on every entry", tuple(v + 0.1 for v in PARITY_WANT)),
                ("the last entry wrong (0.9 for 0.5)", PARITY_WANT[:3] + (0.9,)),
                ("first two entries exchanged", (PARITY_WANT[1], PARITY_WANT[0]) + PARITY_WANT[2:]))
PARITY_GATES = (("TimesFM-3 core", "cosine > 0.9999 and relative L2 < 0.01"),
                ("Chronos-2 forecast", "cosine > 0.99, Pearson > 0.99, largest absolute error over the range of the reference < 0.05"),
                ("FinCast forecast", "cosine > 0.99, Pearson > 0.99, relative L2 < 0.02"))
EPS_VALUES = (("32-bit machine epsilon, 1.1920929e-7 (TimesFM-3)", 1.1920929e-7), ("1e-6 (a literal in the kernel)", 1e-6), ("1e-5 (Kronos)", 1e-5))
EPS_INPUTS = (("entries of size 1", (1.0, -1.0, 1.0, -1.0)), ("entries of size 0.001", (0.001, -0.001, 0.001, -0.001)))


def parity_gates():
    rows = []
    for name, got in PARITY_CASES:
        want = PARITY_WANT
        cos = _cosine(got, want)
        rel = math.sqrt(sum((a - b) ** 2 for a, b in zip(got, want))) / math.sqrt(sum(b * b for b in want))
        rows.append((name, _fx(cos, 4), _fx(rel, 4), _fx(max(abs(a - b) for a, b in zip(got, want)), 3)))
    t1 = _table(["output under test", "cosine", "relative L2", "largest absolute difference"], rows)
    floor = [(c, _fx(math.sqrt(1.0 - c * c), 4)) for c in (0.99, 0.999, 0.9999)]
    t2 = _table(["cosine", "relative L2 that a wrong direction alone can cause"], floor)
    t3 = _table(["gate", "criterion"], PARITY_GATES, "ll")
    eps_rows = []
    for iname, x in EPS_INPUTS:
        ms = sum(v * v for v in x) / len(x)
        base = x[0] / math.sqrt(ms + 1e-6)
        for ename, e in EPS_VALUES:
            y = x[0] / math.sqrt(ms + e)
            eps_rows.append((iname, ename, _fx(y, 4), _fx(y / base, 4)))
    t4 = _table(["input", "epsilon", "first output entry", "ratio to the output at 1e-6"], eps_rows, "llrr")
    head = (f"The reference output is {PARITY_WANT}. Cosine is the dot product over the product of lengths, relative L2 is the length of the difference over the "
            f"length of the reference, and the last column is the largest absolute difference of an entry.\n\n")
    m2 = "\n\nIf the scale is free, a direction that is wrong by the angle t leaves a relative L2 of at least sin t = sqrt(1 - cosine^2):\n\n"
    m3 = "\n\nThe gates of the three parity tests of the crates:\n\n"
    m4 = ("\n\nAn RMS norm y = x / sqrt(mean(x^2) + epsilon) of four numbers (gain 1) at three epsilons. The same code that cannot be told apart at entries of size 1 "
          "differs by a third, or by more than a half, at entries of size 0.001:\n\n")
    return head + t1 + m2 + t2 + m3 + t3 + m4 + t4
