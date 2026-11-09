# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "A Research Architecture: The Spectral Language Model".

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. It builds the idea of `crates/spectrallm` at toy size and tests it the way the chapter teaches:

* the real Fourier basis, its truncation, and the synthesis computed by a radix-2 FFT against the dense product;
* the arithmetic of the mixer, counted per token for a toy and for the paper's shape;
* a sequence-axis transform leaks the future into the past, a feed-forward transform over the feature axis cannot;
* a tiny decoder (an embedding of the last two tokens, one pre-norm block, a tied head) with four feed-forward sublayers
  that differ in that sublayer only, trained on a source whose entropy is known, with its backward pass checked
  against central differences;
* the registered comparison below, eight seeds per arm, effect sizes with paired 95 percent intervals.

REGISTRATION. The three hypotheses, their margin (0.02 nats) and the decision rule below were written into this file before
the comparison was first run, and have not been changed since. The unit is the population cross-entropy of a trained
model (nats per token, computed exactly from the source) and a difference is "arm minus reference", so a negative number
favours the arm. Each hypothesis is judged by the paired 95 percent interval of the difference over the seeds: it is
SUPPORTED when the interval lies wholly on the predicted side, REFUTED when it lies wholly on the other side, and
UNDECIDED when it straddles the margin.

  H1  the spectral mixer (Fourier basis) is not worse than the SwiGLU mixer with the same parameter count by more than
      the margin: S minus Q_par <= +0.02. (Prediction: supported.)
  H2  a SwiGLU mixer with the arithmetic of the spectral mixer is better than it by more than the margin:
      S minus Q_flop >= +0.02. (Prediction: supported.)
  H3  the Fourier basis itself does not matter: replacing it by a random orthonormal basis of the same shape changes the
      result by less than the margin: |S minus R| < 0.02, an equivalence needing the whole interval inside the margin.
      (Prediction: supported.)

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seeds. Sums of
products are plain left-to-right additions (never Python's `sum`, which may compensate). Numbers of trained models are
printed by the stable-digits rule of `stable.py`.

Swedish Embedded AB implements solutions for testing new model architectures against matched baselines. If your team needs
expertise in experiment design for neural architectures, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import hashlib
import json
import math
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import bookrng
import experiments as ex
import stable

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=3):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sg(v, d=3):
    """Signed fixed decimals; a value that rounds to zero is written without a sign."""
    t = _fx(abs(v), d)
    return t if t.strip("0.") == "" else ("+" if v >= 0 else "-") + t


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


# ------------------------------------------------------------------------------------------------ the basis and its transform

D_TOY, M_TOY, C_TOY = 16, 8, 4       # signal length, retained modes, channels of the toy model


def real_basis(d, m):
    """S[j][i] = phi_i(j), the lowest m functions of the real orthonormal Fourier basis of R^d (d even), in frequency
    order: 1/sqrt(d); then for k = 1 .. d/2-1 the pair sqrt(2/d) cos(2 pi k j/d), sqrt(2/d) sin(2 pi k j/d); then the
    alternation (-1)^j / sqrt(d)."""
    s = [[0.0] * m for _ in range(d)]
    for j in range(d):
        for i in range(m):
            if i == 0:
                v = 1.0 / math.sqrt(d)
            elif i == d - 1:
                v = (1.0 if j % 2 == 0 else -1.0) / math.sqrt(d)
            else:
                k = (i + 1) // 2
                th = 2.0 * math.pi * k * j / d
                v = math.sqrt(2.0 / d) * (math.cos(th) if i % 2 == 1 else math.sin(th))
            s[j][i] = v
    return s


def _fft(x, cnt):
    """Radix-2 decimation-in-time FFT of a list of complex numbers (length a power of two) by the even/odd recursion,
    counting complex multiplications by twiddles and complex additions in `cnt`."""
    n = len(x)
    if n == 1:
        return list(x)
    ev, od = _fft(x[0::2], cnt), _fft(x[1::2], cnt)
    out = [0j] * n
    for k in range(n // 2):
        t = complex(math.cos(2 * math.pi * k / n), -math.sin(2 * math.pi * k / n)) * od[k]
        cnt["mul"] += 1
        out[k] = ev[k] + t
        out[k + n // 2] = ev[k] - t
        cnt["add"] += 2
    return out


def synth_by_fft(z, d, cnt):
    """S z for the real basis by one complex FFT of length d. With a_k = z[2k-1] and b_k = z[2k], the pair contributes
    sqrt(2/d) (a_k cos t + b_k sin t) = sqrt(2/d) Re[(a_k - i b_k) e^{i t}], so a conjugate-symmetric spectrum
    X[0] = z[0] sqrt(d)... is built, and x[j] = (1/sqrt(d)) sum_k X[k] e^{+2 pi i k j / d} is real. The inverse transform is
    the forward one on the conjugate, conjugated back."""
    m = len(z)
    spec = [0j] * d
    spec[0] = complex(z[0], 0.0)
    for k in range(1, d // 2):
        a = z[2 * k - 1] if 2 * k - 1 < m else 0.0
        b = z[2 * k] if 2 * k < m else 0.0
        spec[k] = complex(a, -b) / math.sqrt(2.0)
        spec[d - k] = complex(a, b) / math.sqrt(2.0)
    if m == d:
        spec[d // 2] = complex(z[d - 1], 0.0)
    conj = [v.conjugate() for v in spec]
    y = _fft(conj, cnt)
    return [v.conjugate().real / math.sqrt(d) for v in y]


def basis_check():
    s = real_basis(D_TOY, M_TOY)
    worst_gram = 0.0
    for i in range(M_TOY):
        for j in range(M_TOY):
            g = 0.0
            for r in range(D_TOY):
                g += s[r][i] * s[r][j]
            worst_gram = max(worst_gram, abs(g - (1.0 if i == j else 0.0)))
    rng = bookrng.generator("spectral_lm.basis_check")
    z = [rng.normal() for _ in range(M_TOY)]
    dense = [0.0] * D_TOY
    for j in range(D_TOY):
        a = 0.0
        for i in range(M_TOY):
            a += s[j][i] * z[i]
        dense[j] = a
    cnt = {"mul": 0, "add": 0}
    fast = synth_by_fft(z, D_TOY, cnt)
    diff = max(abs(u - v) for u, v in zip(dense, fast))
    nz, nx = _dot(z, z), _dot(dense, dense)
    rows = [(j, _fx(z[j], 4) if j < M_TOY else "", _fx(dense[j], 4), _fx(fast[j], 4)) for j in range(D_TOY)]
    note = ("D = 16, m = 8: the retained functions are the constant, the cosine and sine at frequencies 1, 2 and 3, and the cosine at "
            "frequency 4 (its sine partner, index 8, is cut off). "
            f"Draw order: z[0] .. z[7] from `spectral_lm.basis_check` ({_seed('spectral_lm.basis_check')}), z[i] = rng.normal(). "
            f"Largest deviation of S^T S from the identity: below 1e-12 ({'yes' if worst_gram < 1e-12 else 'NO'}). "
            f"Largest difference between the dense product S z and the FFT synthesis: below 1e-12 ({'yes' if diff < 1e-12 else 'NO'}). "
            f"|z|^2 = {_fx(nz, 4)}, |S z|^2 = {_fx(nx, 4)} (Parseval). The FFT used {cnt['mul']} complex multiplications "
            f"and {cnt['add']} complex additions (the dense product: {D_TOY * M_TOY} real multiplications).")
    return note + "\n\n" + _table(["j", "z[j]", "(S z)[j] dense", "(S z)[j] by FFT"], rows)


# ------------------------------------------------------------------------------------------------ the arithmetic


def fft_flops(n):
    """The analytic count of one real transform row of length n, as brain's cost model states it: 5 n log2 n (a radix-2
    butterfly is a complex multiplication, 6 operations, and two complex additions, 4 operations, and there are
    n/2 log2 n butterflies)."""
    return 5 * n * int(math.log2(n))


def dense_flops(n, m):
    return 2 * n * m


def mixer_flops(m, d, c, fft):
    """Spectral mixer per token and block: two filters and the output weighting (4 c m), the gated product (5 c d), and 3 c
    transforms (two syntheses and one analysis per channel)."""
    per = fft_flops(d) if fft else dense_flops(d, m)
    return 4 * c * m + 5 * c * d + 3 * c * per


def swiglu_flops(m, f):
    return 6 * m * f


def transform_cost():
    rows = []
    for d in (16, 32, 64, 128, 256, 512, 1024, 2048):
        m = d // 2
        dn, ff = dense_flops(d, m), fft_flops(d)
        rows.append((d, m, _int(dn), _int(ff), _fx(dn / ff, 2), "dense" if dn <= ff else "FFT"))
    t1 = _table(["D", "m = D/2", "dense product 2 D m", "FFT 5 D log2 D", "dense / FFT", "fewer operations"], rows)
    # the counted FFT against the formula
    cnt_rows = []
    for d in (16, 64, 512):
        cnt = {"mul": 0, "add": 0}
        _fft([0j] * d, cnt)
        cnt_rows.append((d, _int(cnt["mul"]), _int(cnt["add"]), _int(6 * cnt["mul"] + 2 * cnt["add"]), _int(fft_flops(d))))
    t2 = _table(["D", "twiddle multiplications", "complex additions", "6 x mult + 2 x add", "5 D log2 D"], cnt_rows)
    shapes = [("the toy of this chapter", M_TOY, D_TOY, C_TOY, 4, 72), ("the paper's shape", 256, 512, 24, 24, 1024)]
    rows3 = []
    for name, m, d, c, f_par, f_big in shapes:
        rows3.append((name, f"m = {m}, D = {d}, C = {c}", _int(3 * c * m), _int(mixer_flops(m, d, c, False)),
                      _int(mixer_flops(m, d, c, True)), f"f = {f_par}: {_int(3 * m * f_par)} / {_int(swiglu_flops(m, f_par))}",
                      f"f = {f_big}: {_int(3 * m * f_big)} / {_int(swiglu_flops(m, f_big))}"))
    t3 = _table(["shape", "spectral mixer", "its parameters", "FLOPs, dense transforms", "FLOPs, FFT transforms",
                 "SwiGLU of equal parameters (params / FLOPs)", "SwiGLU of larger width (params / FLOPs)"], rows3)
    return ("First table: operations of one transform row of length D with m = D/2 retained modes, by the definition "
            "(2 D m: D outputs, m multiply-adds each) and by the FFT count 5 D log2 D.\n\n" + t1 +
            "\n\nSecond table: the recursion of this module run on D zeros and its operations counted (a complex multiplication is 6 real "
            "operations, a complex addition 2); the total equals 5 D log2 D exactly.\n\n" + t2 +
            "\n\nThird table: per token and block, FLOPs as the chapter counts them (filters 4 C m, gated product 5 C D, 3 C transforms; "
            "SwiGLU 6 m f), parameters 3 C m against 3 m f.\n\n" + t3)


# ------------------------------------------------------------------------------------------------ causality on a toy

T_LEAK = 8


def _seq_filter_matrix(t, gains):
    """The matrix of y = Re IDFT(g * DFT(x)) along the sequence axis for real x of length t, with real gains g[0..t-1]
    (g[k] = g[t-k] for a real output): y_i = sum_j J[i][j] x_j with J[i][j] = (1/t) sum_k g[k] cos(2 pi k (i-j) / t)."""
    return [[sum_cos(gains, t, i - j) for j in range(t)] for i in range(t)]


def sum_cos(gains, t, lag):
    s = 0.0
    for k in range(t):
        s += gains[k] * math.cos(2.0 * math.pi * k * lag / t)
    return s / t


def _causal_filter_matrix(t, kernel):
    """The same linear map for a causal convolution y_i = sum_{j<=i} kernel[i-j] x_j, computed through zero-padded FFTs of
    length 2t: pad x and kernel to 2t, multiply the spectra, invert, keep the first t outputs. Read out as a matrix."""
    n = 2 * t
    cnt = {"mul": 0, "add": 0}
    kp = [complex(kernel[i], 0.0) if i < t else 0j for i in range(n)]
    kf = _fft(kp, cnt)
    mat = [[0.0] * t for _ in range(t)]
    for j in range(t):
        xp = [0j] * n
        xp[j] = 1.0 + 0j
        xf = _fft(xp, cnt)
        prod = [a * b for a, b in zip(kf, xf)]
        y = _fft([v.conjugate() for v in prod], cnt)
        for i in range(t):
            mat[i][j] = y[i].conjugate().real / n
    return mat


def leak_table():
    rng = bookrng.generator("spectral_lm.leak")
    gains = [0.0] * T_LEAK
    for k in range(T_LEAK // 2 + 1):
        gains[k] = rng.uniform() + 0.5
    for k in range(1, T_LEAK // 2):
        gains[T_LEAK - k] = gains[k]
    kernel = [rng.uniform() for _ in range(T_LEAK)]
    j_seq = _seq_filter_matrix(T_LEAK, gains)
    j_causal = _causal_filter_matrix(T_LEAK, kernel)
    # the feature-axis spectral mixer of the toy model on T_LEAK tokens: each token's features pass through the mixer alone
    p = init_params("S", 0)
    toks = [[rng.normal() for _ in range(M_TOY)] for _ in range(T_LEAK)]

    def run(rows):
        return [mixer_forward(p, rmsnorm(r)[0])[0] for r in rows]

    base = run(toks)
    bumped = [list(r) for r in toks]
    bumped[T_LEAK - 1] = [v + 1.0 for v in bumped[T_LEAK - 1]]
    after = run(bumped)
    rows = []
    for i in range(T_LEAK):
        feat = max(abs(u - v) for u, v in zip(after[i], base[i]))
        causal = abs(j_causal[i][T_LEAK - 1])
        rows.append((i, _fx(abs(j_seq[i][T_LEAK - 1]), 4), _fx(causal, 4) if causal > 1e-12 else "0", _fx(feat, 4) if feat > 0.0 else "0"))
    worst_upper = max(abs(j_seq[i][j]) for i in range(T_LEAK) for j in range(i + 1, T_LEAK))
    worst_upper_c = max(abs(j_causal[i][j]) for i in range(T_LEAK) for j in range(i + 1, T_LEAK))
    leaked = sum(1 for i in range(T_LEAK) for j in range(i + 1, T_LEAK) if abs(j_seq[i][j]) > 1e-12)
    pairs = T_LEAK * (T_LEAK - 1) // 2
    return (f"Sequence length T = {T_LEAK}. Generator `spectral_lm.leak` ({_seed('spectral_lm.leak')}), in this order: the gains g[0] .. g[4] are rng.uniform() + 0.5, "
            f"and g[5], g[6], g[7] are copies of g[3], g[2], g[1]; then the causal kernel k[0] .. k[7] is eight rng.uniform() draws; then the token vectors, "
            f"64 rng.normal() draws, row by row (the mixer is the spectral mixer of this chapter's toy model with the initial weights of "
            f"run 0, applied to the RMS-normalised vector of each token: the RMS norm has no gain and its epsilon is {EPS_NORM:g}, inside the root). 'Sequence transform' is y = Re IDFT(g * DFT(x)) over the sequence axis of a scalar "
            f"feature; 'causal kernel' is y_i = sum over j <= i of k[i - j] x_j computed with FFTs of length 16 on zero-padded inputs; 'feature-axis mixer' maps each "
            f"token's own 8 features. Each entry is the largest change of output position i when the input at the last position, 7, is increased by 1 (for the "
            f"feature-axis mixer: every one of that token's 8 raw features by 1, before the RMS norm, and the change is measured on the largest absolute "
            f"difference over the 8 outputs of the mixer alone, without the residual connection).\n\n"
            + _table(["output position i", "sequence transform", "causal kernel, zero-padded", "feature-axis mixer"], rows)
            + f"\n\nEntries above the diagonal of the sequence transform's matrix (output i reading input j > i): {leaked} of {pairs} are "
            f"nonzero, the largest has magnitude {_fx(worst_upper, 4)}; for the zero-padded causal kernel the largest is below 1e-12 "
            f"({'yes' if worst_upper_c < 1e-12 else 'NO'}).")


# ------------------------------------------------------------------------------------------------ the toy language model

V = 6                    # vocabulary
BETA = 1.5               # sharpness of the source's conditional distributions
STEPS = 240              # full-batch Adam steps of a comparison run
SEL_STEPS = 120          # steps of a learning-rate selection run
EVAL_EVERY = 20
FLOOR = 0.1              # the annealed rate ends at this fraction of its peak
EPS_NORM = 1e-6
BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-8
INIT_STD = 0.5           # embeddings
DENSE_STD = 0.25         # SwiGLU matrices; the spectral filters get DENSE_STD * sqrt(D)
N_RUNS = 8
MARGIN = 0.02
F_PAR = C_TOY            # SwiGLU width with the spectral mixer's parameter count: 3 C m = 3 m f
F_FLOP = 72              # SwiGLU width with (about) the spectral mixer's dense-transform arithmetic: 6 m f = 3456 against 3520

ARMS = ("S", "R", "Q_par", "Q_flop")
ARM_LABELS = {"S": "spectral, Fourier basis", "R": "spectral, random orthonormal basis",
              "Q_par": "SwiGLU, equal parameters", "Q_flop": "SwiGLU, equal arithmetic"}


def silu(a):
    return a / (1.0 + math.exp(-a))


def silu_grad(a):
    sg = 1.0 / (1.0 + math.exp(-a))
    return sg * (1.0 + a * (1.0 - sg))


def source():
    """The data-generating source: an order-2 Markov chain on V symbols. p(c | a, b) = softmax_c(BETA * g[a][b][c]) with
    g ~ Normal(0, 1) drawn from `spectral_lm.source` in the order a, b, c (c fastest). Returns the conditional table, the
    stationary distribution over (a, b) pairs and the entropy rate (nats per token)."""
    rng = bookrng.generator("spectral_lm.source")
    p = [[None] * V for _ in range(V)]
    for a in range(V):
        for b in range(V):
            lg = [BETA * rng.normal() for _ in range(V)]
            mx = max(lg)
            e = [math.exp(v - mx) for v in lg]
            z = 0.0
            for v in e:
                z += v
            p[a][b] = [v / z for v in e]
    pi = [[1.0 / (V * V)] * V for _ in range(V)]
    for _ in range(5000):
        nxt = [[0.0] * V for _ in range(V)]
        for a in range(V):
            for b in range(V):
                for c in range(V):
                    nxt[b][c] += pi[a][b] * p[a][b][c]
        delta = max(abs(nxt[a][b] - pi[a][b]) for a in range(V) for b in range(V))
        pi = nxt
        if delta < 1e-16:
            break
    h = 0.0
    for a in range(V):
        for b in range(V):
            hc = 0.0
            for c in range(V):
                hc -= p[a][b][c] * math.log(p[a][b][c])
            h += pi[a][b] * hc
    return p, pi, h


@lru_cache(maxsize=None)
def _source():
    return source()


def sample_counts(seed_idx, n_tok):
    """The training counts n[a][b][c] of a sequence of n_tok tokens: the generator `spectral_lm.run<i>.data` draws the first
    two tokens with rng.integer(V) and each next token with one rng.uniform(), taking the first c whose cumulative
    probability exceeds it (the last c if rounding leaves none)."""
    p, _, _ = _source()
    rng = bookrng.generator(f"spectral_lm.run{seed_idx}.data")
    toks = [rng.integer(V), rng.integer(V)]
    for _ in range(n_tok - 2):
        u, cum, nxt = rng.uniform(), 0.0, V - 1
        for c in range(V):
            cum += p[toks[-2]][toks[-1]][c]
            if u < cum:
                nxt = c
                break
        toks.append(nxt)
    counts = [[[0] * V for _ in range(V)] for _ in range(V)]
    for i in range(n_tok - 2):
        counts[toks[i]][toks[i + 1]][toks[i + 2]] += 1
    return counts


# ---- the model: parameters are a dict of lists of rows ----------------------------------------------------------


def gram_schmidt_columns(mat, d, m):
    """Orthonormalise the m columns of a d x m matrix (list of d rows), in order, by classical Gram-Schmidt."""
    cols = [[mat[j][i] for j in range(d)] for i in range(m)]
    out = []
    for c in cols:
        v = list(c)
        for u in out:
            proj = _dot(v, u)
            v = [x - proj * y for x, y in zip(v, u)]
        nrm = math.sqrt(_dot(v, v))
        out.append([x / nrm for x in v])
    return [[out[i][j] for i in range(m)] for j in range(d)]


def init_params(arm, seed_idx):
    """Tensors drawn from `spectral_lm.run<i>.init`, rng.normal() each, in this order: E1 [V, m] and E2 [V, m] (std 0.5, row
    major); then the sublayer. SwiGLU arms: Wg [f, m], Wu [f, m], Wd [m, f], std 0.25. Spectral arms: G [C, m], U [C, m],
    W [C, m], std 0.25 sqrt(D) = 1 (so that S(G h) has the variance of a dense gate: a row of S has squared length m/D).
    The R arm then draws a [D, m] matrix (std 1, row major) and orthonormalises its columns by Gram-Schmidt; the other
    spectral arm uses the Fourier basis."""
    rng = bookrng.generator(f"spectral_lm.run{seed_idx}.init")
    m = M_TOY

    def mat(r, c, std):
        return [[std * rng.normal() for _ in range(c)] for _ in range(r)]

    p = {"E1": mat(V, m, INIT_STD), "E2": mat(V, m, INIT_STD)}
    if arm in ("Q_par", "Q_flop"):
        f = F_PAR if arm == "Q_par" else F_FLOP
        p["Wg"], p["Wu"], p["Wd"] = mat(f, m, DENSE_STD), mat(f, m, DENSE_STD), mat(m, f, DENSE_STD)
        p["basis"] = None
    else:
        std = DENSE_STD * math.sqrt(D_TOY)
        p["G"], p["U"], p["W"] = mat(C_TOY, m, std), mat(C_TOY, m, std), mat(C_TOY, m, std)
        p["basis"] = real_basis(D_TOY, m) if arm == "S" else gram_schmidt_columns(mat(D_TOY, m, 1.0), D_TOY, m)
    return p


TRAINABLE = ("E1", "E2", "Wg", "Wu", "Wd", "G", "U", "W")


def rmsnorm(x):
    ms = 0.0
    for v in x:
        ms += v * v
    r = math.sqrt(ms / len(x) + EPS_NORM)
    return [v / r for v in x], r


def rmsnorm_back(y, r, dy):
    mdy = _dot(dy, y) / len(y)
    return [(g - yi * mdy) / r for g, yi in zip(dy, y)]


def mixer_forward(p, h):
    m = M_TOY
    if p["basis"] is None:
        gate = [_dot(row, h) for row in p["Wg"]]
        up = [_dot(row, h) for row in p["Wu"]]
        s = [silu(a) * b for a, b in zip(gate, up)]
        y = [_dot(row, s) for row in p["Wd"]]
        return y, (gate, up, s)
    sb = p["basis"]
    cache = []
    y = [0.0] * m
    for c in range(C_TOY):
        hg = [p["G"][c][i] * h[i] for i in range(m)]
        hu = [p["U"][c][i] * h[i] for i in range(m)]
        ag = [_dot(sb[j], hg) for j in range(D_TOY)]
        au = [_dot(sb[j], hu) for j in range(D_TOY)]
        s = [silu(a) * b for a, b in zip(ag, au)]
        t = [0.0] * m
        for j in range(D_TOY):
            sj, sv = sb[j], s[j]
            for i in range(m):
                t[i] += sj[i] * sv
        for i in range(m):
            y[i] += p["W"][c][i] * t[i]
        cache.append((ag, au, s, t))
    return y, cache


def mixer_backward(p, h, cache, dy, g):
    """Accumulate the sublayer's parameter gradients into g and return dh."""
    m = M_TOY
    dh = [0.0] * m
    if p["basis"] is None:
        gate, up, s = cache
        for o in range(m):
            for k in range(len(s)):
                g["Wd"][o][k] += dy[o] * s[k]
        ds = [0.0] * len(s)
        for o in range(m):
            for k in range(len(s)):
                ds[k] += p["Wd"][o][k] * dy[o]
        for k in range(len(s)):
            dgate = ds[k] * up[k] * silu_grad(gate[k])
            dup = ds[k] * silu(gate[k])
            for i in range(m):
                g["Wg"][k][i] += dgate * h[i]
                g["Wu"][k][i] += dup * h[i]
                dh[i] += p["Wg"][k][i] * dgate + p["Wu"][k][i] * dup
        return dh
    sb = p["basis"]
    for c in range(C_TOY):
        ag, au, s, t = cache[c]
        dt = [dy[i] * p["W"][c][i] for i in range(m)]
        for i in range(m):
            g["W"][c][i] += dy[i] * t[i]
        ds = [_dot(sb[j], dt) for j in range(D_TOY)]
        dag = [ds[j] * au[j] * silu_grad(ag[j]) for j in range(D_TOY)]
        dau = [ds[j] * silu(ag[j]) for j in range(D_TOY)]
        dhg, dhu = [0.0] * m, [0.0] * m
        for j in range(D_TOY):
            sj = sb[j]
            a_, b_ = dag[j], dau[j]
            for i in range(m):
                dhg[i] += sj[i] * a_
                dhu[i] += sj[i] * b_
        for i in range(m):
            g["G"][c][i] += dhg[i] * h[i]
            g["U"][c][i] += dhu[i] * h[i]
            dh[i] += p["G"][c][i] * dhg[i] + p["U"][c][i] * dhu[i]
    return dh


def context_forward(p, a, b):
    x = [u + v for u, v in zip(p["E2"][a], p["E1"][b])]
    h, r1 = rmsnorm(x)
    y, cache = mixer_forward(p, h)
    x2 = [u + v for u, v in zip(x, y)]
    hf, r2 = rmsnorm(x2)
    logits = [_dot(row, hf) for row in p["E1"]]
    mx = max(logits)
    e = [math.exp(v - mx) for v in logits]
    z = 0.0
    for v in e:
        z += v
    probs = [v / z for v in e]
    return probs, (h, r1, cache, hf, r2)


def zero_grads(p):
    return {k: [[0.0] * len(r) for r in p[k]] for k in TRAINABLE if k in p}


def empirical_loss_grad(p, counts, total):
    """Mean cross-entropy of the training sequence (nats per prediction) and its gradient, by the counts: each of the V x V
    contexts is run once and weighted by the number of times it occurs with each next token."""
    g = zero_grads(p)
    loss = 0.0
    for a in range(V):
        for b in range(V):
            n = counts[a][b]
            tot = 0
            for v in n:
                tot += v
            if tot == 0:
                continue
            probs, (h, r1, cache, hf, r2) = context_forward(p, a, b)
            for c in range(V):
                if n[c]:
                    loss -= n[c] * math.log(probs[c]) / total
            dlogits = [(probs[c] * tot - n[c]) / total for c in range(V)]
            dhf = [0.0] * M_TOY
            for c in range(V):
                row = p["E1"][c]
                for i in range(M_TOY):
                    g["E1"][c][i] += dlogits[c] * hf[i]
                    dhf[i] += dlogits[c] * row[i]
            dx2 = rmsnorm_back(hf, r2, dhf)
            dh = mixer_backward(p, h, cache, dx2, g)
            dx = [u + v for u, v in zip(dx2, rmsnorm_back(h, r1, dh))]
            for i in range(M_TOY):
                g["E2"][a][i] += dx[i]
                g["E1"][b][i] += dx[i]
    return loss, g


def population_loss(p):
    """The exact cross-entropy (nats per token) of the model against the source: sum over contexts of the stationary
    probability times the cross-entropy of the true conditional against the model's."""
    tp, pi, _ = _source()
    tot = 0.0
    for a in range(V):
        for b in range(V):
            probs, _ = context_forward(p, a, b)
            ce = 0.0
            for c in range(V):
                ce -= tp[a][b][c] * math.log(probs[c])
            tot += pi[a][b] * ce
    return tot


def rate_at(peak, step, steps, annealed):
    """The learning rate of step 1 .. steps: the peak when constant; when annealed a cosine from the peak at step 1 to FLOOR times
    the peak, peak (FLOOR + (1 - FLOOR) (1 + cos(pi (step - 1) / steps)) / 2)."""
    if not annealed:
        return peak
    return peak * (FLOOR + (1.0 - FLOOR) * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / steps)))


def train(p, counts, steps, lr, annealed, eval_every=None):
    """Adam (beta 0.9, 0.999, epsilon 1e-8, no weight decay, no clipping) on every tensor, full batch, at the rate `rate_at`.
    Returns the final empirical loss and, every `eval_every` steps, the population loss."""
    total = 0
    for a in range(V):
        for b in range(V):
            for v in counts[a][b]:
                total += v
    mom = zero_grads(p)
    var = zero_grads(p)
    curve = {}
    if eval_every:
        curve[0] = population_loss(p)
    loss = None
    for s in range(1, steps + 1):
        loss, g = empirical_loss_grad(p, counts, total)
        c1, c2 = 1.0 - BETA1 ** s, 1.0 - BETA2 ** s
        rate = rate_at(lr, s, steps, annealed)
        for k in g:
            for ri, row in enumerate(p[k]):
                gr, mr, vr = g[k][ri], mom[k][ri], var[k][ri]
                for j in range(len(row)):
                    gv = gr[j]
                    mr[j] = BETA1 * mr[j] + (1.0 - BETA1) * gv
                    vr[j] = BETA2 * vr[j] + (1.0 - BETA2) * gv * gv
                    row[j] -= rate * (mr[j] / c1) / (math.sqrt(vr[j] / c2) + EPS_ADAM)
        if eval_every and s % eval_every == 0:
            curve[s] = population_loss(p)
    final_empirical, _ = empirical_loss_grad(p, counts, total)
    return final_empirical, curve


def n_params(arm):
    m = M_TOY
    emb = 2 * V * m
    block = 3 * C_TOY * m if arm in ("S", "R") else 3 * m * (F_PAR if arm == "Q_par" else F_FLOP)
    return emb, block


# ------------------------------------------------------------------------------------------------ the runs

LR_RUN = 0   # the seed index of the learning-rate selection runs; comparison seeds are 1 .. N_RUNS

# A configuration of the comparison: the length of the training sequence, whether the rate is annealed, the rates tried, and the nudges
# of the starting weight under which the run is repeated (the stable-digits rule needs four; the first attempt, whose digits are
# never printed, needs one to show how much it moves).
CONFIGS = {
    "first": {"n_tok": 1200, "annealed": False, "grid": (0.01, 0.03, 0.1), "nudges": (1e-12,)},
    "main": {"n_tok": 1200, "annealed": True, "grid": (0.01, 0.03, 0.1), "nudges": stable.NUDGES},
    "more": {"n_tok": 12000, "annealed": True, "grid": (0.01, 0.03, 0.1), "nudges": stable.NUDGES},
}


def _job(spec):
    cfg_name, kind, arm, seed_idx, lr, nudge = spec
    cfg = CONFIGS[cfg_name]
    p = init_params(arm, seed_idx)
    p["E1"][0][0] += nudge
    counts = sample_counts(seed_idx, cfg["n_tok"])
    if kind == "select":
        emp, _ = train(p, counts, SEL_STEPS, lr, cfg["annealed"])
        return {"cfg": cfg_name, "kind": kind, "arm": arm, "lr": lr, "nudge": nudge, "emp": emp}
    emp, curve = train(p, counts, STEPS, lr, cfg["annealed"], EVAL_EVERY)
    return {"cfg": cfg_name, "kind": kind, "arm": arm, "seed": seed_idx, "lr": lr, "nudge": nudge, "emp": emp,
            "pop": curve[STEPS], "curve": curve}


MEMO = Path(__file__).resolve().parents[3] / "build" / "book" / "spectral_lm-results.json"


END_OF_RUNS = "# ---- end of the computation that is kept"


def _digest():
    """A hash of the sources the kept results depend on: this module up to the marker line (the tables after it only format them) and its imports."""
    h = hashlib.sha256()
    src = Path(__file__).read_text(encoding="utf-8")
    h.update(src[:src.index("\n" + END_OF_RUNS)].encode("utf-8"))
    for path in (Path(ex.__file__), Path(stable.__file__), Path(bookrng.__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()


def _select_from(sel):
    """arm -> the learning rate with the lowest final empirical loss in the (unnudged) selection runs."""
    best = {}
    for arm in ARMS:
        runs = [r for r in sel if r["arm"] == arm and r["nudge"] == 0.0]
        best[arm] = min(runs, key=lambda r: (r["emp"], r["lr"]))["lr"]
    return best


def _compute(pool, cfg_name):
    cfg = CONFIGS[cfg_name]
    nudges = (0.0, *cfg["nudges"])
    sel = pool.map(_job, [(cfg_name, "select", arm, LR_RUN, lr, nd) for arm in ARMS for lr in cfg["grid"] for nd in nudges], chunksize=1)
    rates = _select_from(sel)
    main = pool.map(_job, [(cfg_name, "main", arm, s, rates[arm], nd) for arm in ARMS for s in range(1, N_RUNS + 1) for nd in nudges], chunksize=1)
    return {"select": sel, "main": main}


def _pack(res):
    return {"select": res["select"], "main": [dict(r, curve={str(k): v for k, v in r["curve"].items()}) for r in res["main"]]}


def _unpack(raw):
    return {"select": raw["select"], "main": [dict(r, curve={int(k): v for k, v in r["curve"].items()}) for r in raw["main"]]}


@lru_cache(maxsize=None)
def results():
    """{config: {'select': [records], 'main': [records]}}; the unnudged selection of a configuration decides the rates of its main runs,
    which (with their nudged twins) are spread over four worker processes and kept in MEMO under a digest of the sources."""
    digest = _digest()
    if MEMO.is_file():
        try:
            memo = json.loads(MEMO.read_text(encoding="utf-8"))
            if memo.get("digest") == digest:
                return {k: _unpack(v) for k, v in memo["results"].items()}
        except (ValueError, KeyError):
            pass
    with get_context("fork").Pool(4) as pool:
        out = {name: _compute(pool, name) for name in CONFIGS}
    MEMO.parent.mkdir(parents=True, exist_ok=True)
    MEMO.write_text(json.dumps({"digest": digest, "results": {k: _pack(v) for k, v in out.items()}}), encoding="utf-8")
    return out


def rates(cfg):
    return _select_from(results()[cfg]["select"])


def _runs(cfg, arm, seed_idx):
    """The unnudged run and its nudged twins, in the order of the configuration's nudges."""
    rs = [r for r in results()[cfg]["main"] if r["arm"] == arm and r["seed"] == seed_idx]
    base = [r for r in rs if r["nudge"] == 0.0][0]
    twins = [next(r for r in rs if r["nudge"] == nd) for nd in CONFIGS[cfg]["nudges"]]
    return base, twins


def _vals(cfg, arm, key, k=None):
    """Per-seed values of `key` for an arm: the unnudged runs, or those of the k-th nudged copy."""
    return [(_runs(cfg, arm, s)[0] if k is None else _runs(cfg, arm, s)[1][k])[key] for s in range(1, N_RUNS + 1)]


def mean_sd(v):
    n = len(v)
    mu = 0.0
    for x in v:
        mu += x
    mu /= n
    ss = 0.0
    for x in v:
        ss += (x - mu) ** 2
    return mu, math.sqrt(ss / (n - 1))


def paired_interval(a, b):
    """Mean of a_i - b_i, its sample sd (divisor n - 1), standard error and the 95 percent t interval."""
    d = [x - y for x, y in zip(a, b)]
    mu, sd = mean_sd(d)
    se = sd / math.sqrt(len(d))
    half = ex._t95(len(d) - 1) * se
    return mu, sd, se, mu - half, mu + half


def stable_text(cfg, f, digits=3, signed=False):
    """f(k) -> a float computed from the per-seed values of the run (k None) or of its k-th nudged copy; text by the stable-digits rule."""
    n = len(CONFIGS[cfg]["nudges"])
    fmt = (lambda x: _sg(x, digits)) if signed else (lambda x: _fx(x, digits))
    return stable.text(f(None), [f(k) for k in range(n)], fmt)


# ---- end of the computation that is kept

# ------------------------------------------------------------------------------------------------ tables


def source_table():
    p, pi, h = _source()
    ent_uniform = math.log(V)
    mx = max(max(pi[a][b] for b in range(V)) for a in range(V))
    mn = min(min(pi[a][b] for b in range(V)) for a in range(V))
    rows = []
    for a in range(2):
        for b in range(2):
            rows.append((f"({a}, {b})", _fx(pi[a][b], 4), *[_fx(x, 3) for x in p[a][b]]))
    return (f"The source: an order-2 Markov chain on V = {V} symbols, p(c | a, b) = softmax over c of {BETA} g[a][b][c] with g[a][b][c] drawn "
            f"from `spectral_lm.source` ({_seed('spectral_lm.source')}) by rng.normal(), in the order a, b, c with c fastest. The stationary distribution over "
            f"the pair (a, b) (probability that two consecutive tokens are a then b) is found by iterating the chain from the uniform pair distribution until "
            f"it moves by less than 1e-16 or 5000 times. First four of the 36 contexts:\n\n"
            + _table(["context (a, b)", "stationary probability", "p(0)", "p(1)", "p(2)", "p(3)", "p(4)", "p(5)"], rows)
            + f"\n\nStationary probabilities range from {_fx(mn, 4)} to {_fx(mx, 4)} (uniform would be {_fx(1 / (V * V), 4)}). The entropy rate of the source, "
            f"the lowest cross-entropy any model can reach, is {_fx(h, 4)} nats per token; a model that ignores the context and predicts uniformly "
            f"gets ln {V} = {_fx(ent_uniform, 4)}.")


def arms_table():
    rows = []
    for arm in ARMS:
        emb, block = n_params(arm)
        m = M_TOY
        if arm in ("S", "R"):
            fl = mixer_flops(m, D_TOY, C_TOY, False)
            flf = mixer_flops(m, D_TOY, C_TOY, True)
            detail = f"C = {C_TOY}, D = {D_TOY}"
        else:
            f = F_PAR if arm == "Q_par" else F_FLOP
            fl = flf = swiglu_flops(m, f)
            detail = f"f = {f}"
        rows.append((arm, ARM_LABELS[arm], detail, _int(emb), _int(block), _int(emb + block), _int(fl), _int(flf)))
    return ("Embedding parameters: two tables of V x m = 6 x 8 (the tied head reads E1). Mixer FLOPs per token as in the arithmetic section: "
            "spectral rows with dense transforms and with FFT counts, SwiGLU rows 6 m f. No biases, no learnable norm gains.\n\n"
            + _table(["arm", "sublayer", "shape", "embedding parameters", "sublayer parameters", "all parameters", "FLOPs (dense)", "FLOPs (FFT)"], rows))


def gradient_check():
    """Central differences of the empirical loss against the analytic gradient, every tensor entry, for three sublayer types."""
    rows = []
    h = 1e-5
    counts = sample_counts(LR_RUN, 1200)
    total = 0
    for a in range(V):
        for b in range(V):
            for v in counts[a][b]:
                total += v
    for arm in ("S", "R", "Q_par"):
        p = init_params(arm, LR_RUN)
        _, g = empirical_loss_grad(p, counts, total)
        worst, n = 0.0, 0
        for k in g:
            for ri, row in enumerate(p[k]):
                for j in range(len(row)):
                    old = row[j]
                    row[j] = old + h
                    lp, _ = empirical_loss_grad(p, counts, total)
                    row[j] = old - h
                    lm, _ = empirical_loss_grad(p, counts, total)
                    row[j] = old
                    num = (lp - lm) / (2 * h)
                    ana = g[k][ri][j]
                    worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-3))
                    n += 1
        rows.append((arm, ARM_LABELS[arm], _int(n), "below 1e-6" if worst < 1e-6 else f"{worst:.1e}"))
    return ("Every entry of every trainable tensor, central difference with step 1e-5 of the empirical loss (the mean cross-entropy of the "
            f"1200-token training sequence of run 0, `spectral_lm.run0.data` {_seed('spectral_lm.run0.data')}) at the initial parameters of run 0 "
            f"(`spectral_lm.run0.init` {_seed('spectral_lm.run0.init')}). Relative error |numerical - analytic| / max(|numerical|, |analytic|, 1e-3); "
            "the criterion is below 1e-6. The SwiGLU arm of equal arithmetic runs the same code as `Q_par` with f = 72 instead of 4.\n\n"
            + _table(["arm", "sublayer", "entries checked", "largest relative error"], rows))


def seeds_table():
    rows = []
    for s in range(1, N_RUNS + 1):
        rows.append((s, _seed(f"spectral_lm.run{s}.data"), _seed(f"spectral_lm.run{s}.init")))
    return _table(["run", "data seed (`spectral_lm.run<i>.data`)", "init seed (`spectral_lm.run<i>.init`)"], rows)


def _lr_table(cfg):
    sel = results()[cfg]["select"]
    grid = CONFIGS[cfg]["grid"]
    r = rates(cfg)
    rows = []
    for arm in ARMS:
        cells = []
        for lr in grid:
            runs = {x["nudge"]: x["emp"] for x in sel if x["arm"] == arm and x["lr"] == lr}
            cells.append(stable.text(runs[0.0], [runs[nd] for nd in CONFIGS[cfg]["nudges"]], lambda x: _fx(x, 3)))
        rows.append((arm, *cells, r[arm]))
    return _table(["arm", *[f"rate {x}" for x in grid], "chosen"], rows)


SHORT = ("Digits by the stable-digits rule stated under the table of learning rates: a number is printed only with the digits that agree in the run "
         "and in its four nudged copies.")


def lr_selection():
    return (f"Learning-rate selection of the amended recipe: {SEL_STEPS} full-batch Adam steps from the initial parameters of run 0, the rate annealed "
            f"over those steps (step s = 1 .. {SEL_STEPS} runs at peak (0.1 + 0.9 (1 + cos(pi (s - 1) / {SEL_STEPS})) / 2), so the first step is at the peak and "
            f"the last at {_fx(FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * (SEL_STEPS - 1) / SEL_STEPS)), 4)} of it), one run per arm and rate, scored by the final training loss (mean "
            f"cross-entropy of the 1200-token training sequence, whose {1200 - 2} predictions are made from the 2 tokens before each of positions 2 .. 1199; nats per "
            "prediction); the lowest decides the arm's peak rate. " + stable.RULE + "\n\n" + _lr_table("main"))


def lr_selection_first():
    return _lr_table("first")


def _final_table(cfg):
    r = rates(cfg)
    rows = [("peak rate", *[r[arm] for arm in ARMS])]
    for s in range(1, N_RUNS + 1):
        cells = []
        for arm in ARMS:
            base, twins = _runs(cfg, arm, s)
            cells.append(stable.text(base["pop"], [t["pop"] for t in twins], lambda x: _fx(x, 3)))
        rows.append((f"run {s}", *cells))
    rows.append(("mean", *[stable_text(cfg, lambda k, arm=arm: mean_sd(_vals(cfg, arm, "pop", k))[0], 3) for arm in ARMS]))
    rows.append(("sd", *[stable_text(cfg, lambda k, arm=arm: mean_sd(_vals(cfg, arm, "pop", k))[1], 3) for arm in ARMS]))
    return _table(["", *ARMS], rows)


def final_losses():
    _, _, h = _source()
    return (f"Population cross-entropy (nats per token, computed exactly against the source) after {STEPS} full-batch Adam steps, the rate annealed from the arm's "
            f"peak rate over the {STEPS} steps, for runs 1 to {N_RUNS} (seeds in the table above); the source's entropy rate is {_fx(h, 4)}. "
            "Mean and sd (divisor n - 1) are over the eight runs. " + SHORT + "\n\n" + _final_table("main"))


def final_losses_more():
    _, _, h = _source()
    n = CONFIGS["more"]["n_tok"]
    return (f"The same eight runs trained on {n}-token sequences (the 1200-token sequence of each run is the first 1200 tokens of its {n}-token one; "
            f"{n - 2} predictions), rates selected afresh. First the selection, as before: for each arm and rate, one run from the initial parameters of run 0 of "
            f"{SEL_STEPS} annealed full-batch steps, scored by the final training loss (mean cross-entropy of the {n}-token sequence of run 0, nats per prediction). "
            f"Then, for each arm at its chosen peak rate, the population cross-entropy after {STEPS} steps (rate annealed over those {STEPS} steps), nats per token; "
            f"the entropy rate is {_fx(h, 4)}. " + SHORT
            + "\n\n" + _lr_table("more") + "\n\n" + _final_table("more"))


def _verdict(lo, hi, side, margin=MARGIN):
    """side 'le': claim is difference <= +margin. 'ge': claim is difference >= +margin. 'eq': |difference| < margin."""
    if side == "le":
        return "supported" if hi < margin else ("refuted" if lo > margin else "undecided")
    if side == "ge":
        return "supported" if lo > margin else ("refuted" if hi < margin else "undecided")
    return "supported" if (lo > -margin and hi < margin) else ("refuted" if (lo > margin or hi < -margin) else "undecided")


HYPOTHESES = (
    ("H1", "S", "Q_par", "le", "S - Q_par <= +0.02"),
    ("H2", "S", "Q_flop", "ge", "S - Q_flop >= +0.02"),
    ("H3", "S", "R", "eq", "|S - R| < 0.02"),
)


def _ci(cfg, a, b, k=None):
    return paired_interval(_vals(cfg, a, "pop", k), _vals(cfg, b, "pop", k))


def _effect_table(cfg):
    n = len(CONFIGS[cfg]["nudges"])
    rows = []
    for name, a, b, side, claim in HYPOTHESES:
        mu = stable_text(cfg, lambda k, a=a, b=b: _ci(cfg, a, b, k)[0], 3, signed=True)
        lo = stable_text(cfg, lambda k, a=a, b=b: _ci(cfg, a, b, k)[3], 3, signed=True)
        hi = stable_text(cfg, lambda k, a=a, b=b: _ci(cfg, a, b, k)[4], 3, signed=True)
        vs = [_verdict(_ci(cfg, a, b, k)[3], _ci(cfg, a, b, k)[4], side) for k in [None, *range(n)]]
        verdict = vs[0] if all(v == vs[0] for v in vs) else "unstable"
        rows.append((name, claim, mu, f"{lo} to {hi}", verdict))
    return _table(["hypothesis", "claim", "mean difference", "95% interval", "verdict"], rows)


EFFECT_NOTE = ("Paired differences of the population cross-entropy over the same eight runs (the arms of a run share the data sequence), first arm minus second in "
               "nats per token, so a negative difference favours the first arm. Mean difference and the 95 percent t interval of the mean (7 degrees of freedom, "
               f"multiplier {_fx(ex._t95(N_RUNS - 1), 4)}, the Student t value of Volume II, Chapter 8). Verdicts by the registered rule: supported when the interval lies "
               "wholly on the predicted side of the margin 0.02, refuted when wholly on the other side, undecided otherwise; a verdict that differs between a run and "
               "its nudged copies would print 'unstable'. ")


def effect_sizes():
    return EFFECT_NOTE + SHORT + "\n\n" + _effect_table("main")


def effect_sizes_more():
    return "The same contrasts on the 12000-token sequences. " + EFFECT_NOTE + SHORT + "\n\n" + _effect_table("more")


def first_attempt():
    """The recipe written down before the first run: constant rate, the rates 0.01, 0.03, 0.1. Its verdicts by the same rule, and how far a
    1e-12 change of one weight moves each run."""
    cfg = "first"
    r = rates(cfg)
    rows = []
    for name, a, b, side, claim in HYPOTHESES:
        v0 = _ci(cfg, a, b)
        v1 = _ci(cfg, a, b, 0)
        rows.append((name, claim, _verdict(v0[3], v0[4], side), _verdict(v1[3], v1[4], side)))
    t1 = _table(["hypothesis", "claim", "verdict, original runs", "verdict, copies with one weight changed by 1e-12"], rows)
    mv = []
    for arm in ARMS:
        moves = [abs(x - y) for x, y in zip(_vals(cfg, arm, "pop"), _vals(cfg, arm, "pop", 0))]
        mv.append((arm, r[arm], _bound(max(moves)), _bound(sum(moves) / len(moves))))
    t2 = _table(["arm", "peak rate", "largest change of a run's final loss", "mean change"], mv)
    sds = [mean_sd(_vals(cfg, arm, "pop"))[1] for arm in ARMS]
    return (f"The recipe written into the module before its first run: a constant rate chosen from {CONFIGS['first']['grid']} by the same selection "
            f"(one run per arm and rate from the initial parameters of run 0, {SEL_STEPS} steps at the constant rate, the lowest final training loss), 1200-token "
            f"sequences, {STEPS} steps, the same eight runs. Each run was repeated once with the embedding weight E1[0][0] increased by 1e-12. Verdicts of the original runs and "
            "of the repeats by the registered rule:\n\n" + t1 + "\n\nHow far the repeat of a run ends from the run, in population cross-entropy (nats per token). A run at a "
            "constant rate is chaotic at the end, so the change itself cannot be reproduced; each cell is a bound that is at least ten times the value found in "
            "the module (the smallest of 0.05, 0.1, 0.5 that does so):\n\n" + t2
            + f"\n\nFor scale, the standard deviation over the eight runs of one arm lies between {_fx(math.floor(min(sds) * 100) / 100, 2)} and "
            f"{_fx(math.ceil(max(sds) * 100) / 100, 2)} (to the nearest hundredth outward, for the same reason).")


def _bound(x):
    for b in (0.05, 0.1, 0.5):
        if 10 * x < b:
            return f"below {b:g}"
    return f"above {0.5:g}"


# ------------------------------------------------------------------------------------------------ data for the figures


def effect_data(cfg):
    """(label, mean, lo, hi) of the three registered contrasts, unnudged runs."""
    out = []
    for name, a, b, side, claim in HYPOTHESES:
        mu, sd, se, lo, hi = _ci(cfg, a, b)
        out.append((f"{a} - {b}", mu, lo, hi))
    return out


def curve_data(cfg="main"):
    """{arm: [(step, [per-run population losses])]} for the learning-curve figure (unnudged runs)."""
    steps = sorted(_runs(cfg, "S", 1)[0]["curve"])
    return {arm: [(s, [_runs(cfg, arm, sd)[0]["curve"][s] for sd in range(1, N_RUNS + 1)]) for s in steps] for arm in ARMS}


def source_entropy():
    return _source()[2]


def seeds_needed():
    """What the paired spread of S - R (the contrast whose interval is widest relative to its margin) says about the number of runs: the half-width
    t(n - 1) sd / sqrt(n) of the 95 percent interval for n runs, with sd the standard deviation of the eight observed differences."""
    cfg = "main"
    sd_t = stable_text(cfg, lambda k: paired_interval(_vals(cfg, "S", "pop", k), _vals(cfg, "R", "pop", k))[1], 3)
    sd = paired_interval(_vals(cfg, "S", "pop"), _vals(cfg, "R", "pop"))[1]
    rows = []
    for n in (3, 4, 6, 8, 12, 16, 24, 32):
        half = ex._t95(n - 1) * sd / math.sqrt(n)
        rows.append((n, _fx(ex._t95(n - 1), 3), _fx(half, 3), "yes" if half < MARGIN else "no"))
    return (f"The standard deviation of the eight paired differences S - R (divisor n - 1) is {sd_t} nats per token. If that spread held for n runs, the half-width of "
            "the 95 percent interval of the mean would be t sd / sqrt(n), with t the Student multiplier for n - 1 degrees of freedom. An equivalence at the margin 0.02 "
            "needs the half-width below the margin even when the true difference is zero.\n\n"
            + _table(["runs n", "multiplier t", "half-width", "below the margin 0.02"], rows))


def fewer_runs():
    """The three registered contrasts judged from the first three runs only, beside all eight."""
    cfg = "main"
    n = len(CONFIGS[cfg]["nudges"])
    rows = []
    for name, a, b, side, claim in HYPOTHESES:
        cells = []
        for runs in (3, N_RUNS):
            def iv(k, runs=runs, a=a, b=b):
                return paired_interval(_vals(cfg, a, "pop", k)[:runs], _vals(cfg, b, "pop", k)[:runs])
            lo = stable_text(cfg, lambda k: iv(k)[3], 3, signed=True)
            hi = stable_text(cfg, lambda k: iv(k)[4], 3, signed=True)
            vs = [_verdict(iv(k)[3], iv(k)[4], side) for k in [None, *range(n)]]
            cells.append(f"{lo} to {hi}: {vs[0] if all(v == vs[0] for v in vs) else 'unstable'}")
        rows.append((name, claim, *cells))
    return ("The three contrasts of the main comparison judged from runs 1 to 3 alone and from all eight: the 95 percent interval (7 degrees of freedom for eight runs, 2 for "
            "three, multiplier " + _fx(ex._t95(2), 3) + ") and the verdict by the registered rule. " + SHORT + "\n\n"
            + _table(["hypothesis", "claim", "runs 1 to 3", "all eight runs"], rows))


def fit_gap():
    """Training loss (mean cross-entropy of the training sequence) and population cross-entropy per arm, for both lengths of sequence."""
    rows = []
    for cfg, n_tok in (("main", 1200), ("more", 12000)):
        for arm in ARMS:
            emp = stable_text(cfg, lambda k, cfg=cfg, arm=arm: mean_sd(_vals(cfg, arm, "emp", k))[0], 3)
            pop = stable_text(cfg, lambda k, cfg=cfg, arm=arm: mean_sd(_vals(cfg, arm, "pop", k))[0], 3)
            gap = stable_text(cfg, lambda k, cfg=cfg, arm=arm: mean_sd([b - a for a, b in zip(_vals(cfg, arm, "emp", k), _vals(cfg, arm, "pop", k))])[0], 3, signed=True)
            rows.append((_int(n_tok), arm, _int(sum(n_params(arm))), emp, pop, gap))
    return ("Mean over the eight runs of the loss on the training sequence and of the population cross-entropy after the last step, and the mean of their difference "
            "(population minus training; positive when the model does better on the sequence it was trained on than on the source). " + SHORT + "\n\n"
            + _table(["training tokens", "arm", "all parameters", "training loss", "population cross-entropy", "population minus training"], rows))


def edge_check():
    """The selection runs of the main recipe, repeated at the rate 0.3, one above the largest rate of the grid: is the training loss still falling?"""
    cfg = CONFIGS["main"]
    sel = results()["main"]["select"]
    counts = sample_counts(LR_RUN, cfg["n_tok"])
    rows = []
    for arm in ARMS:
        at_top = [r for r in sel if r["arm"] == arm and r["lr"] == 0.1 and r["nudge"] == 0.0][0]["emp"]
        p = init_params(arm, LR_RUN)
        beyond, _ = train(p, counts, SEL_STEPS, 0.3, True)
        rows.append((arm, _fx(at_top, 2), _fx(beyond, 2), "lower" if beyond < at_top - 0.005 else ("higher" if beyond > at_top + 0.005 else "within 0.005")))
    return ("The selection runs of the main recipe (run 0, 120 annealed steps) at the largest rate of the grid, 0.1, and at 0.3, scored by the final training loss "
            "(two decimals; the runs are not repeated under nudges, so only the comparison is read).\n\n"
            + _table(["arm", "training loss at 0.1", "training loss at 0.3", "0.3 compared with 0.1"], rows))


def absorption():
    """A dense (SwiGLU) model and the same model written in the Fourier basis give the same predictions: with Q the square orthonormal real
    basis of R^8 (D = m = 8) the stream is z = Q^T x, the embeddings become E Q, the matrices that read the stream W Q, the matrix that writes it
    Q^T W; the RMS norm (no gain) is unchanged by an orthogonal map."""
    m = M_TOY
    q = real_basis(m, m)
    p = init_params("Q_par", LR_RUN)

    def right(w):   # w Q: rows of w expressed in the new basis
        return [[sum_left(w[r][j] * q[j][i] for j in range(m)) for i in range(m)] for r in range(len(w))]

    def left_t(w):  # Q^T w
        return [[sum_left(q[j][i] * w[j][k] for j in range(m)) for k in range(len(w[0]))] for i in range(m)]

    p2 = dict(p)
    p2["E1"], p2["E2"], p2["Wg"], p2["Wu"], p2["Wd"] = right(p["E1"]), right(p["E2"]), right(p["Wg"]), right(p["Wu"]), left_t(p["Wd"])
    worst, changed = 0.0, 0.0
    for a in range(V):
        for b in range(V):
            u, _ = context_forward(p, a, b)
            w, _ = context_forward(p2, a, b)
            worst = max(worst, max(abs(x - y) for x, y in zip(u, w)))
    changed = max(abs(p["Wg"][0][j] - p2["Wg"][0][j]) for j in range(m))
    return (f"The dense model of arm Q_par at the initial parameters of run 0 (`spectral_lm.run0.init`, {_seed('spectral_lm.run0.init')}), and the model whose "
            f"E1, E2, Wg, Wu are multiplied on the right by the 8 x 8 real Fourier basis Q (the full basis, D = m = 8) and whose Wd is multiplied on the left by Q^T. "
            f"Largest difference between the two models' predicted probabilities over all 36 contexts and 6 next tokens: below 1e-12 "
            f"({'yes' if worst < 1e-12 else 'NO'}). The weights themselves are different (the first row of Wg changes by up to {_fx(changed, 3)} in an entry), "
            "and so is every coordinate of the stream; the function is not.")


def sum_left(it):
    s = 0.0
    for v in it:
        s += v
    return s


def product_couples_modes():
    """The pointwise product of two signals, each a single mode, analysed back into modes: D = 16, the full basis."""
    d = 16
    s = real_basis(d, d)
    a, b = 1, 3   # cos at frequency 1 and cos at frequency 2
    xa = [s[j][a] for j in range(d)]
    xb = [s[j][b] for j in range(d)]
    prod = [u * v for u, v in zip(xa, xb)]
    z = [sum_left(s[j][i] * prod[j] for j in range(d)) for i in range(d)]
    rows = []
    names = {0: "constant"}
    for i in range(1, d - 1):
        names[i] = f"{'cos' if i % 2 == 1 else 'sin'}, k = {(i + 1) // 2}"
    names[d - 1] = "alternating"
    for i in range(d):
        if abs(z[i]) > 1e-12:
            rows.append((i, names[i], _fx(z[i], 5)))
    return (f"D = 16 and the full basis. Mode 1 (cosine at frequency 1) and mode 3 (cosine at frequency 2), each as a signal S e_i, multiplied sample by sample; "
            f"the product analysed with S^T. Entries with magnitude below 1e-12 are left out; 1 / sqrt(2 D) = {_fx(1 / math.sqrt(2 * d), 5)}.\n\n"
            + _table(["mode i", "basis function", "coefficient of the product"], rows))
