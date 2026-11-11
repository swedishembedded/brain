# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume IX project "From Waveform to Speech and Back".

Plain Python 3, standard library only, written from the project's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: project_speech_pipeline.<name> -->`
markers. What the project reuses from the chapters of the volume is imported, never copied: the resamplers of Chapter 1 (`sound`), the
window, the reflecting pad, the filterbank and the frame count of Chapter 2 (`spectrograms`), the radix-2 transform of Volume I
(`fourier`), the nearest-codeword search and squared error of Chapter 3 (`audio_codecs`), the CTC lattice and the word-error
alignment of Chapter 4 (`speech_recognition`, against which the scaled training recursion written here is checked), the equal error
rate of Chapter 5 (`speaker_embeddings`), the book's generator and the stable-digits rule, the AdamW step of Volume III
(`first_models`) and the elimination of Volume II (`project_predictor`).

The world is a closed loop on a synthetic language:

* four letters, each a sum of two sinusoids ("formants") whose frequencies a speaker parameter scales, a space that is silence, six
  words, sentences of two or three words, rendered at 12 kHz with a random gain and white noise;
* a Kaiser-tapered resampler to 8 kHz, a 16-filter Slaney log-mel front end of 50 frames a second, standardized per band;
* a two-layer perceptron over five frames with a CTC head of six outputs (blank, space and the four letters), trained by AdamW with
  the scaled forward-backward recursion and a hand-written backward pass, decoded greedily and scored by word and character error;
* a residual quantizer of the frames and the recogniser's score on what it returns;
* a speaker embedding by pooled statistics, a ridge map from it to the speaker parameter, and a voice-transfer loop that recognises
  a reference clip, estimates the voice and speaks the transcript again in it;
* three planted mistakes, each measured by a number, and the cost of every stage as operations and bytes.

Trained numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the nudges of
`stable.py`. The ten runs (two kinds of five) are trained in four worker processes.

Swedish Embedded AB implements solutions for speech pipelines whose rates, front ends and error budgets are specified, counted and
tested end to end. If your team needs expertise in speech recognition and synthesis engineering, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import operator
from functools import lru_cache

import audio_codecs as ac
import bookrng
import first_models as fm
import fourier as fo
import project_predictor as pp
import sound
import speaker_embeddings as se
import spectrograms as sp
import speech_recognition as sr
import stable

mul = operator.mul

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or "l" * len(headers)
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(("---:" if a == "r" else "---") for a in align) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _f(v, d=3):
    t = f"{v:.{d}f}"
    return t[1:] if t.startswith("-") and float(t) == 0 else t


def _i(n):
    return f"{n:,}"


def _seed(name):
    return f"0x{bookrng.seed_of(name):016x}"


# ------------------------------------------------------------------------------------------------ the world

FS_SYN = 12000          # rate of the synthesiser, Hz
FS_REC = 8000           # rate of the recogniser, Hz
RAMP = 120              # samples of the raised-cosine ramp at each end of a letter (10 ms)
PAD = 1200              # samples of silence before and after a sentence (100 ms)
NOISE = 0.02            # standard deviation of the white noise added to every sample
PHONES = {"a": ((350.0, 0.5), (1000.0, 0.4)), "b": ((350.0, 0.5), (2200.0, 0.4)),
          "c": ((700.0, 0.5), (1400.0, 0.4)), "d": ((700.0, 0.5), (5200.0, 0.4))}
WORDS = ("ab", "cad", "bd", "dca", "bca", "adb")
SPEAKERS = (0.88, 0.94, 1.00, 1.06, 1.12, 1.03)   # speaker parameter g of speakers 0 to 4 (heard in training) and 5 (never heard)
N_SEEN = 5
UNSEEN = N_SEEN
N_TRAIN, N_TEST = 120, 180
SYMBOLS = "-" + " abcd"               # id 0 is the blank, 1 the space, 2 to 5 the letters
SHIFTED = "-abcd?"                    # the planted mistake: ids one lower, the space on the blank's own id 0 (a blank when read back)
ID = {c: i for i, c in enumerate(SYMBOLS)}
K = len(SYMBOLS)


def stream_name(kind, n):
    return f"project_speech_pipeline.{kind}.{n}"


def draw_text(gen):
    """The sentence: 2 + integer(2) words, each WORDS[integer(6)], in order."""
    return " ".join(WORDS[gen.integer(len(WORDS))] for _ in range(2 + gen.integer(2)))


def symbol_samples(ch, d):
    """Samples of a symbol whose duration code is d = integer(3): a letter lasts 60 + 20 d ms, a space 40 + 20 d ms."""
    ms = 40 + 20 * d if ch == " " else 60 + 20 * d
    return ms * FS_SYN // 1000


def synthesise(text, g, gen):
    """Waveform of `text` for speaker parameter g at 12 kHz. Draws, in order: integer(3) for every symbol of the text, one
    uniform() for the gain 0.5 + uniform(), then one normal() for every sample (the noise, standard deviation NOISE)."""
    durs = [gen.integer(3) for _ in text]
    gain = 0.5 + gen.uniform()
    sizes = [symbol_samples(ch, d) for ch, d in zip(text, durs)]
    x = [0.0] * (2 * PAD + sum(sizes))
    pos = PAD
    for ch, n in zip(text, sizes):
        if ch != " ":
            for i in range(n):
                u = min(1.0, (i + 0.5) / RAMP, (n - i - 0.5) / RAMP)
                env = 0.5 - 0.5 * math.cos(math.pi * u)
                s = 0.0
                for f, a in PHONES[ch]:
                    s += a * math.sin(2.0 * math.pi * g * f * i / FS_SYN)
                x[pos + i] = gain * env * s
        pos += n
    return [v + NOISE * gen.normal() for v in x]


# ------------------------------------------------------------------------------------------------ the front end

N_FFT, HOP, N_MELS = 256, 160, 16
WINDOW = sp.hann(N_FFT)
BANK = sp.filterbank(FS_REC, N_FFT, N_MELS, 0.0, 4000.0, "slaney", True)
GUARD = 1e-6
CTX = 2                      # frames on each side of the centre frame the recogniser reads
D_IN = N_MELS * (2 * CTX + 1)


def power_frames(y):
    """Reflect-pad by N_FFT / 2 on each side, frames of N_FFT at hop HOP, periodic Hann, power of bins 0 .. N_FFT / 2."""
    p = sp.reflect_pad(y, N_FFT // 2)
    out = []
    for t in range(sp.frame_count(len(p), N_FFT, HOP)):
        X = fo._fft([p[t * HOP + i] * WINDOW[i] for i in range(N_FFT)])
        out.append([X[k].real ** 2 + X[k].imag ** 2 for k in range(N_FFT // 2 + 1)])
    return out


def log_mel(power, bank=BANK, base_ten=False):
    """ln(E + 1e-6) of the filterbank energies, the largest value of the whole sentence subtracted, divided by 4. With base_ten the
    natural logarithm is replaced by log10 (the planted mistake)."""
    lg = math.log10 if base_ten else math.log
    mel = [[lg(sum(w * p for w, p in zip(row, fr)) + GUARD) for row in bank] for fr in power]
    top = max(max(r) for r in mel)
    return [[(v - top) / 4.0 for v in r] for r in mel]


@lru_cache(maxsize=None)
def _sentence(kind, n):
    """(text, speaker, power frames of the Kaiser-resampled sentence, power frames of the linearly resampled one)."""
    gen = bookrng.generator(stream_name(kind, n))
    text = draw_text(gen)
    speaker = n % (N_SEEN if kind == "train" else len(SPEAKERS))
    x = synthesise(text, SPEAKERS[speaker], gen)
    return (text, speaker, power_frames(sound.resample_kaiser(x, FS_SYN, FS_REC)),
            power_frames(sound.resample_linear(x, FS_SYN, FS_REC)) if kind == "test" else None)


def _build(spec):
    kind, n = spec
    return spec, _sentence.__wrapped__(kind, n)


@lru_cache(maxsize=None)
def data():
    """Train and test sentences, built in four worker processes (each sentence has its own stream)."""
    specs = [("train", n) for n in range(N_TRAIN)] + [("test", n) for n in range(N_TEST)]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        built = dict(pool.map(_build, specs, chunksize=8))
    return ([built[("train", n)] for n in range(N_TRAIN)], [built[("test", n)] for n in range(N_TEST)])


@lru_cache(maxsize=None)
def standardiser():
    """Mean and standard deviation (divisor n) of every band of the log-mel frames of the training sentences."""
    rows = [r for _, _, p, _ in data()[0] for r in log_mel(p)]
    mu = [sum(r[j] for r in rows) / len(rows) for j in range(N_MELS)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in rows) / len(rows)) for j in range(N_MELS)]
    return mu, sd


def standardise(frames):
    mu, sd = standardiser()
    return [[(r[j] - mu[j]) / sd[j] for j in range(N_MELS)] for r in frames]


def stack(frames):
    """Frame t becomes the concatenation of frames t - CTX .. t + CTX, indices clamped to 0 .. T - 1."""
    T = len(frames)
    return [sum((frames[min(max(t + o, 0), T - 1)] for o in range(-CTX, CTX + 1)), []) for t in range(T)]


def features(power, **kw):
    return stack(standardise(log_mel(power, **kw)))


# ------------------------------------------------------------------------------------------------ the recogniser

HID = 24
N_PARAMS = HID * D_IN + HID + K * HID + K
INIT, ORDER = "project_speech_pipeline.init", "project_speech_pipeline.order"
EPOCHS, BATCH, LR = 20, 4, 0.01


def init_params(gen):
    """W1 [HID, D_IN] row-major, each N(0, 1 / D_IN), then b1 = 0, then W2 [K, HID], each N(0, 1 / HID), then b2 = 0."""
    s1, s2 = math.sqrt(1.0 / D_IN), math.sqrt(1.0 / HID)
    th = [s1 * gen.normal() for _ in range(HID * D_IN)] + [0.0] * HID
    th += [s2 * gen.normal() for _ in range(K * HID)] + [0.0] * K
    return th


def _unpack(th):
    W1 = [th[j * D_IN:(j + 1) * D_IN] for j in range(HID)]
    o = HID * D_IN
    b1 = th[o:o + HID]
    o += HID
    W2 = [th[o + k * HID:o + (k + 1) * HID] for k in range(K)]
    o += K * HID
    return W1, b1, W2, th[o:o + K]


def forward(th, X):
    """Hidden activations (tanh) and softmax rows of every frame of X."""
    W1, b1, W2, b2 = _unpack(th)
    hs, ps = [], []
    for x in X:
        h = [math.tanh(b + sum(map(mul, w, x))) for w, b in zip(W1, b1)]
        z = [b + sum(map(mul, w, h)) for w, b in zip(W2, b2)]
        m = max(z)
        ex = [math.exp(v - m) for v in z]
        s = sum(ex)
        hs.append(h)
        ps.append([v / s for v in ex])
    return hs, ps


def ctc_scaled(P, labels, blank=0):
    """The CTC loss -ln P(labels) and the soft alignment gamma [T][K] by the recursion of Chapter 4 with each frame of alpha and of
    beta divided by its own sum (the logarithms of the alpha sums are added back into the loss), so that no product underflows."""
    e = [blank]
    for l in labels:
        e += [l, blank]
    S, T = len(e), len(P)
    skip = [s >= 2 and e[s] != blank and e[s] != e[s - 2] for s in range(S)]
    al = [[0.0] * S for _ in range(T)]
    al[0][0], al[0][1] = P[0][e[0]], P[0][e[1]]
    c = sum(al[0])
    al[0] = [a / c for a in al[0]]
    ln = math.log(c)
    for t in range(1, T):
        p, prev, row = P[t], al[t - 1], al[t]
        for s in range(S):
            a = prev[s] + (prev[s - 1] if s >= 1 else 0.0) + (prev[s - 2] if skip[s] else 0.0)
            row[s] = a * p[e[s]]
        c = sum(row)
        al[t] = [a / c for a in row]
        ln += math.log(c)
    loss = -(ln + math.log(al[T - 1][S - 1] + al[T - 1][S - 2]))
    be = [[0.0] * S for _ in range(T)]
    be[T - 1][S - 1], be[T - 1][S - 2] = P[T - 1][e[S - 1]], P[T - 1][e[S - 2]]
    c = sum(be[T - 1])
    be[T - 1] = [b / c for b in be[T - 1]]
    for t in range(T - 2, -1, -1):
        p, nxt, row = P[t], be[t + 1], be[t]
        for s in range(S):
            b = nxt[s] + (nxt[s + 1] if s + 1 < S else 0.0) + (nxt[s + 2] if s + 2 < S and skip[s + 2] else 0.0)
            row[s] = b * p[e[s]]
        c = sum(row)
        be[t] = [b / c for b in row]
    gamma = []
    for t in range(T):
        g = [0.0] * len(P[0])
        for s in range(S):
            g[e[s]] += al[t][s] * be[t][s] / P[t][e[s]]
        z = sum(g)
        gamma.append([x / z for x in g])
    return loss, gamma


def loss_grad(th, X, labels, grad=None):
    """CTC loss of one sentence; when `grad` (a list of N_PARAMS) is given, the gradient is added into it. Backward pass by hand:
    dz = P - gamma at every frame, dW2 += dz h^T, dh = W2^T dz, dpre = dh (1 - h^2), dW1 += dpre x^T."""
    hs, ps = forward(th, X)
    loss, gamma = ctc_scaled(ps, labels)
    if grad is None:
        return loss
    _, _, W2, _ = _unpack(th)
    o1 = HID * D_IN
    o2 = o1 + HID
    o3 = o2 + K * HID
    for t, x in enumerate(X):
        dz = [p - g for p, g in zip(ps[t], gamma[t])]
        h = hs[t]
        for k in range(K):
            d, base = dz[k], o2 + k * HID
            grad[base:base + HID] = [g + d * hv for g, hv in zip(grad[base:base + HID], h)]
            grad[o3 + k] += d
        for j in range(HID):
            dp = sum(W2[k][j] * dz[k] for k in range(K)) * (1.0 - h[j] * h[j])
            base = j * D_IN
            grad[base:base + D_IN] = [g + dp * xi for g, xi in zip(grad[base:base + D_IN], x)]
            grad[o1 + j] += dp
    return loss


def decode(ps, symbols=SYMBOLS):
    """Greedy: the first-largest entry of every frame, collapse repeats, drop the blank."""
    path = [max(range(len(p)), key=lambda k: p[k]) for p in ps]
    return "".join(symbols[i] for i in sr._collapse(path))


def edits(ref, hyp, words=True):
    d, _ = sr._wer_ops(ref.split() if words else list(ref), hyp.split() if words else list(hyp))
    return d


def error_rates(refs, hyps):
    """(word error rate, character error rate): total edits over total reference words, and over total reference characters."""
    ew = sum(edits(r, h) for r, h in zip(refs, hyps))
    ec = sum(edits(r, h, False) for r, h in zip(refs, hyps))
    return ew / sum(len(r.split()) for r in refs), ec / sum(len(r) for r in refs)


# ------------------------------------------------------------------------------------------------ the residual quantizer

RVQ_K, RVQ_Q, RVQ_ITERS, RVQ_EVERY = 16, 4, 10, 3
RVQ_NAME = "project_speech_pipeline.rvq"


def _kmeans(X, k, perm, nudge):
    """Lloyd's rounds of Chapter 3: starting centres the first k rows in the order of perm, ties to the lower index."""
    d = len(X[0])
    C = [list(X[i]) for i in perm[:k]]
    C[0][0] += nudge
    for _ in range(RVQ_ITERS):
        sums, counts = [[0.0] * d for _ in range(k)], [0] * k
        for x in X:
            j = ac._nearest(C, x)
            counts[j] += 1
            s = sums[j]
            for a in range(d):
                s[a] += x[a]
        for j in range(k):
            if counts[j]:
                C[j] = [v / counts[j] for v in sums[j]]
    return C


def rvq_train(X, nudge=0.0):
    """Q stages. Stage q is Lloyd on the residuals left by stages before it, its start taken from one shuffle of range(len(X)) drawn
    from the stream RVQ_NAME (one shuffle per stage, in stage order)."""
    g = bookrng.generator(RVQ_NAME)
    R = [list(x) for x in X]
    books = []
    for _ in range(RVQ_Q):
        perm = list(range(len(R)))
        g.shuffle(perm)
        C = _kmeans(R, RVQ_K, perm, nudge)
        books.append(C)
        R = [[r[a] - C[ac._nearest(C, r)][a] for a in range(len(r))] for r in R]
    return books


def rvq_reconstruct(books, x, stages):
    """Sum of the chosen codewords of the first `stages` stages (each stage quantizes what the earlier ones left)."""
    r, out = list(x), [0.0] * len(x)
    for C in books[:stages]:
        c = C[ac._nearest(C, r)]
        r = [a - b for a, b in zip(r, c)]
        out = [a + b for a, b in zip(out, c)]
    return out


# ------------------------------------------------------------------------------------------------ the speaker

ACTIVE = -0.5          # a frame is voiced when its largest log-mel value (before standardizing) is above this
EMB_RIDGE = 1e-3


def embed_raw(power):
    """Mean of the standardized log-mel frames whose largest value before standardizing exceeds ACTIVE (zero vector when none)."""
    lm = log_mel(power)
    st = standardise(lm)
    rows = [s for l, s in zip(lm, st) if max(l) > ACTIVE]
    return [sum(r[j] for r in rows) / len(rows) for j in range(N_MELS)] if rows else [0.0] * N_MELS


def _unit(v):
    n = math.sqrt(sum(a * a for a in v))
    return [a / n for a in v]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@lru_cache(maxsize=None)
def embedding_centre():
    """The mean of the raw embeddings of the training sentences, subtracted from every embedding before it is made a unit vector."""
    raws = [embed_raw(p) for _, _, p, _ in data()[0]]
    return [sum(r[j] for r in raws) / len(raws) for j in range(N_MELS)]


def embed(power):
    c = embedding_centre()
    return _unit([a - b for a, b in zip(embed_raw(power), c)])


@lru_cache(maxsize=None)
def voice_map():
    """Ridge regression w, b with ln g ~ w . embedding + b over the training sentences: the normal equations
    (E^T E + EMB_RIDGE I) w = E^T y on centred columns, b the mean of ln g minus w . mean(E)."""
    E = [embed(p) for _, _, p, _ in data()[0]]
    y = [math.log(SPEAKERS[s]) for _, s, _, _ in data()[0]]
    n = len(E)
    me = [sum(e[j] for e in E) / n for j in range(N_MELS)]
    my = sum(y) / n
    A = [[sum((e[a] - me[a]) * (e[b] - me[b]) for e in E) + (EMB_RIDGE if a == b else 0.0) for b in range(N_MELS)] for a in range(N_MELS)]
    r = [sum((e[a] - me[a]) * (yy - my) for e, yy in zip(E, y)) for a in range(N_MELS)]
    w = pp.solve(A, r)
    return w, my - sum(a * b for a, b in zip(w, me))


def estimate_g(powers):
    """The speaker parameter heard in a set of clips: exp of the mean over the clips of w . embedding + b."""
    w, b = voice_map()
    return math.exp(sum(sum(x * y for x, y in zip(w, embed(p))) + b for p in powers) / len(powers))


# ------------------------------------------------------------------------------------------------ one training run

ENROL, VOICE_CLIPS = 8, 16   # clips of the unseen speaker that enrol the voice, and clips the voice-transfer loop speaks again


def _banks():
    return {"htk": sp.filterbank(FS_REC, N_FFT, N_MELS, 0.0, 4000.0, "htk", True),
            "norm": sp.filterbank(FS_REC, N_FFT, N_MELS, 0.0, 4000.0, "slaney", False)}


def train(kind, nudge):
    """Train one recogniser. kind 'clean': labels use the ids of SYMBOLS. kind 'shifted': the planted label-shift, every id one lower
    (the space is id 0, the blank's own, the letters 1 to 4), the ids never offset past the blank. The first weight (W1[0][0]) is moved
    by `nudge` before the first step. Returns the parameters and the mean training loss of each epoch."""
    tr = data()[0]
    shift = 1 if kind == "shifted" else 0
    X = [features(p) for _, _, p, _ in tr]
    Y = [[ID[c] - shift for c in text] for text, _, _, _ in tr]
    th = init_params(bookrng.generator(INIT))
    th[0] += nudge
    m, v = [0.0] * N_PARAMS, [0.0] * N_PARAMS
    order = list(range(len(tr)))
    shuf = bookrng.generator(ORDER)
    step, curve = 0, []
    for _ in range(EPOCHS):
        shuf.shuffle(order)
        total = 0.0
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            g = [0.0] * N_PARAMS
            for i in idx:
                total += loss_grad(th, X[i], Y[i], g)
            g = [x / len(idx) for x in g]
            step += 1
            fm.adamw_step(th, g, m, v, step, LR, 0.0)
        curve.append(total / len(tr))
    return th, curve


def recognise(th, power, symbols=SYMBOLS, **kw):
    return decode(forward(th, features(power, **kw))[1], symbols)


def _score(th, sets, symbols=SYMBOLS, **kw):
    refs = [t for t, _, _, _ in sets]
    return error_rates(refs, [recognise(th, p, symbols, **kw) for _, _, p, _ in sets])


def voice_loop(th):
    """Speaker 3 is never heard in training. Its first ENROL test clips estimate g_hat and a reference embedding (the unit vector of
    the mean of their embeddings). For each of the next VOICE_CLIPS clips: recognise it, speak the transcript again (the stream
    `voice.<i>`, i the clip's index among speaker 3's clips, so only g differs between the voices) with g_hat and with each trained
    speaker's g, and recognise the hat version once more."""
    mine = [s for s in data()[1] if s[1] == UNSEEN]
    enrol, clips = mine[:ENROL], mine[ENROL:ENROL + VOICE_CLIPS]
    g_hat = estimate_g([e[2] for e in enrol])
    ref = _unit([sum(c) for c in zip(*[embed(e[2]) for e in enrol])])
    rows = []
    for i, (text, _, power, _) in enumerate(clips):
        said = recognise(th, power)
        scores, again = {}, None
        for tag, g in (("hat", g_hat), *[(f"s{k}", SPEAKERS[k]) for k in range(N_SEEN)]):
            gen = bookrng.generator(stream_name("voice", ENROL + i))
            gp = power_frames(sound.resample_kaiser(synthesise(said, g, gen), FS_SYN, FS_REC))
            scores[tag] = _cos(ref, embed(gp))
            if tag == "hat":
                again = recognise(th, gp)
        rows.append((text, said, scores, again))
    return g_hat, rows


def job(spec):
    kind, nudge = spec
    th, curve = train(kind, nudge)
    train_sets, test_sets = data()
    out = {"curve": curve}
    if kind == "shifted":
        out["wer"], out["cer"] = _score(th, test_sets, SHIFTED)
        out["spaces"] = sum(recognise(th, p, SHIFTED).count(" ") for _, _, p, _ in test_sets)
        return out
    out["wer"], out["cer"] = _score(th, test_sets)
    out["train_wer"], out["train_cer"] = _score(th, train_sets)
    for s in range(len(SPEAKERS)):
        out[f"wer_s{s}"], out[f"cer_s{s}"] = _score(th, [e for e in test_sets if e[1] == s])
    refs = [t for t, _, _, _ in test_sets]
    out["lin_wer"], out["lin_cer"] = error_rates(refs, [recognise(th, e[3]) for e in test_sets])
    banks = _banks()
    out["htk_wer"], out["htk_cer"] = _score(th, test_sets, bank=banks["htk"])
    out["norm_wer"], out["norm_cer"] = _score(th, test_sets, bank=banks["norm"])
    out["log10_wer"], out["log10_cer"] = _score(th, test_sets, base_ten=True)
    # the quantizer, on the standardized frames of the training sentences (every RVQ_EVERY-th frame) and the test sentences
    frames = [r for _, _, p, _ in train_sets for r in standardise(log_mel(p))][::RVQ_EVERY]
    books = rvq_train(frames, nudge)
    test_frames = [standardise(log_mel(p)) for _, _, p, _ in test_sets]
    allf = [r for F in test_frames for r in F]
    for q in range(0, RVQ_Q + 1):
        rec = [[rvq_reconstruct(books, r, q) for r in F] for F in test_frames]
        out[f"rvq_mse{q}"] = sum(sum((a - b) ** 2 for a, b in zip(r, o)) for F, G in zip(test_frames, rec) for r, o in zip(F, G)) / len(allf)
        hyps = [decode(forward(th, stack(G))[1]) for G in rec]
        out[f"rvq_wer{q}"], out[f"rvq_cer{q}"] = error_rates(refs, hyps)
    # the voice loop
    out["g_hat"], vl = voice_loop(th)
    for tag in ("hat", *[f"s{k}" for k in range(N_SEEN)]):
        out[f"cos_{tag}"] = sum(r[2][tag] for r in vl) / len(vl)
    out["wins"] = sum(1 for r in vl if r[2]["hat"] > max(r[2][f"s{k}"] for k in range(N_SEEN))) / len(vl)
    out["said_wer"] = error_rates([r[0] for r in vl], [r[1] for r in vl])[0]
    out["again_wer"] = error_rates([r[1] for r in vl], [r[3] for r in vl])[0]
    return out


@lru_cache(maxsize=None)
def runs():
    """Five clean and five shifted runs (the nominal one and the four nudged copies), four worker processes."""
    data()
    standardiser()
    embedding_centre()
    voice_map()
    specs = [(k, n) for n in (0.0, *stable.NUDGES) for k in ("clean", "shifted")]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        res = pool.map(job, specs, chunksize=1)
    return ([r for (k, _), r in zip(specs, res) if k == "clean"], [r for (k, _), r in zip(specs, res) if k == "shifted"])


def cell(key, fmt, kind="clean"):
    rs = runs()[0 if kind == "clean" else 1]
    vals = [r[key] for r in rs]
    return stable.text(vals[0], vals[1:], fmt)


def f3(x):
    return f"{x:.3f}"


def curve_cell(epoch, kind="clean"):
    rs = runs()[0 if kind == "clean" else 1]
    vals = [r["curve"][epoch - 1] for r in rs]
    return stable.text(vals[0], vals[1:], lambda x: f"{x:.3f}")


# ------------------------------------------------------------------------------------------------ the opening

def opening():
    rows = [("the pipeline as specified", "clean", "wer", "cer"),
            ("test audio resampled by linear interpolation", "clean", "lin_wer", "lin_cer"),
            ("test frames from an HTK-scale filterbank", "clean", "htk_wer", "htk_cer"),
            ("test frames from a filterbank without area normalisation", "clean", "norm_wer", "norm_cer"),
            ("test frames with log10 in place of ln", "clean", "log10_wer", "log10_cer"),
            ("trained with ids not offset past the blank", "shifted", "wer", "cer"),
            ("the four-stage quantizer between front end and recogniser", "clean", "rvq_wer4", "rvq_cer4")]
    body = [(a, cell(w, f3, k), cell(c, f3, k)) for a, k, w, c in rows]
    return (f"The test set is the {N_TEST} sentences `project_speech_pipeline.test.<n>`, n = 0 .. {N_TEST - 1}, of milestone 1. 'Word "
            "error' is the total of word edits over the reference words of all 180 sentences, 'character error' the total of "
            "character edits (spaces included) over their reference characters. Each row is one recogniser trained from "
            "`project_speech_pipeline.init` and fed in the way the row says; every number carries only the digits on which the run "
            "and its four nudged copies agree.\n\n"
            + _table(["how the recogniser was fed", "word error", "character error"], body, "lrr"))


# ------------------------------------------------------------------------------------------------ milestone 1: the world

def _wave(kind, n):
    gen = bookrng.generator(stream_name(kind, n))
    text = draw_text(gen)
    speaker = n % (N_SEEN if kind == "train" else len(SPEAKERS))
    return text, speaker, synthesise(text, SPEAKERS[speaker], gen)


def world():
    letter_rows = []
    for c, ((f1, a1), (f2, a2)) in PHONES.items():
        letter_rows.append((f"`{c}`", ID[c], f"{f1:g} Hz x {a1}", f"{f2:g} Hz x {a2}"))
    spk_rows = [(k, f"{g:g}", "heard in training" if k < N_SEEN else "never heard", f"{350 * g:.1f} / {1000 * g:.1f}")
                for k, g in enumerate(SPEAKERS)]
    sent_rows = []
    for n in range(4):
        text, s, x = _wave("train", n)
        sent_rows.append((n, f"`{text}`", s, len(x), len(sound.resample_kaiser(x, FS_SYN, FS_REC)),
                          1 + (-(-len(x) * 2 // 3)) // HOP, _f(sum(x), 6), _f(sum(v * v for v in x), 6)))
    return (_table(["letter", "id", "first sinusoid", "second sinusoid"], letter_rows, "lrll")
            + "\n\nEvery sinusoid of a letter is amplitude times sin(2 pi g f i / 12000) for sample i of the letter, g the speaker parameter. "
            "The ids of the other two symbols are 0 for the blank and 1 for the space. The six words, in the order of the index "
            "`integer(6)` draws: " + ", ".join(f"`{w}`" for w in WORDS) + ".\n\n"
            + _table(["speaker", "g", "role", "letter `a` first / second frequency (Hz)"], spk_rows, "rrll")
            + "\n\nThe first four training sentences, each from its own stream "
            + f"`project_speech_pipeline.train.<n>` (the seed of n = 0 is {_seed(stream_name('train', 0))}): the sentence, the speaker "
            "(n mod 5 for the training sentences), the samples at 12 kHz, the samples after the resampler of milestone 2, the frames "
            "of milestone 2, the sum of the samples and the sum of their squares (6 decimals, tolerance 1e-6).\n\n"
            + _table(["n", "sentence", "speaker", "samples at 12 kHz", "samples at 8 kHz", "frames", "sum", "sum of squares"],
                     sent_rows, "rlrrrrrr"))


# ------------------------------------------------------------------------------------------------ milestone 2: resampler and front end

TONES = (1000.0, 3000.0, 3900.0, 4400.0, 5200.0)


def _amp(y, f):
    return sound.amplitude_at(y, f, FS_REC, 500, 2000)


@lru_cache(maxsize=None)
def tone_amplitudes():
    """For each tone of TONES, the amplitudes (Kaiser, linear) of the resampled second at the frequency it should have."""
    out = {}
    for f in TONES:
        x = sound.tone(f, FS_SYN, FS_SYN)
        out_f = f if f < 4000.0 else FS_REC - f
        out[f] = (_amp(sound.resample_kaiser(x, FS_SYN, FS_REC), out_f), _amp(sound.resample_linear(x, FS_SYN, FS_REC), out_f))
    return out


def resampler():
    rows = []
    for f in TONES:
        k, l = tone_amplitudes()[f]
        out_f = f if f < 4000.0 else FS_REC - f
        role = "must go" if f >= 4000.0 else ("kept, in the transition band" if f > 3500.0 else "kept")
        rows.append((f"{f:g}", f"{out_f:g}", role, f"{k:.2e}" if f >= 4000.0 else _f(k, 4), _f(l, 4)))
    up, down = sound.reduced(FS_SYN, FS_REC)
    ker = sound.KaiserSinc(FS_SYN, FS_REC)
    return (f"The reduced ratio of {FS_SYN} to {FS_REC} is {up} / {down}; the Kaiser design of Chapter 1 has beta = "
            f"{ker.beta:.4f}, a reach of {ker.reach} units (so {2 * ker.reach // up + 1} input samples at most under one output) and "
            f"a cutoff at {ker.cut:.4f} of the input's Nyquist rate, which is the output's. One second of cos(2 pi f i / 12000), amplitude 1; the amplitude "
            "of the output at the frequency it should have after resampling (the frequency itself below 4000 Hz, 8000 minus it "
            "above), measured over output samples 500 to 2499, a whole number of cycles in each case.\n\n"
            + _table(["input tone (Hz)", "output tone (Hz)", "the tone should be", "Kaiser sinc", "linear interpolation"], rows, "rrlrr"))


def _centres(scale):
    e = sp.band_edges(N_MELS, 0.0, 4000.0, scale)
    return e[1:-1]


def front_end():
    edges = sp.band_edges(N_MELS, 0.0, 4000.0, "slaney")
    nz = [sum(1 for w in row if w > 0) for row in BANK]
    rows = [(j, _f(edges[j], 2), _f(edges[j + 1], 2), _f(edges[j + 2], 2), nz[j], f"{sum(BANK[j]):.5f}") for j in range(N_MELS)]
    mu, sd = standardiser()
    text, _, x = _wave("train", 0)
    y = sound.resample_kaiser(x, FS_SYN, FS_REC)
    frame = [y[i] for i in range(N_FFT)]
    X = fo._fft([a * w for a, w in zip(frame, WINDOW)])
    direct = sp.power_bins([a * w for a, w in zip(frame, WINDOW)])
    diff = max(abs(X[k].real ** 2 + X[k].imag ** 2 - direct[k]) / max(direct[k], 1e-12) for k in range(N_FFT // 2 + 1))
    p = _sentence("train", 0)[2]
    F = features(p)
    stats = [(j, _f(mu[j], 4), _f(sd[j], 4)) for j in range(N_MELS)]
    return (f"Filterbank: {N_MELS} Slaney triangles (Chapter 2) over 0 to 4000 Hz at 8000 Hz and N = {N_FFT} (bins at 31.25 Hz), "
            "area normalised (each row multiplied by 2 / (right edge - left edge)); columns are the three edges in hertz, the "
            "number of nonzero weights in the row and the row's sum.\n\n"
            + _table(["filter", "left", "centre", "right", "nonzero weights", "row sum"], rows, "rrrrrr")
            + f"\n\nThe radix-2 transform of the first frame of training sentence 0, against the transform by definition "
            f"(Chapter 2's `power_bins`): largest relative difference of the 129 powers {diff:.1e}, below 1e-9. The log of Chapter 2's "
            f"recipe A with the guard {GUARD:g}, the sentence's largest value subtracted and a division by 4, then per-band "
            f"standardisation with the mean and standard deviation (divisor n) over every frame of the {N_TRAIN} training "
            "sentences:\n\n" + _table(["band", "mean", "standard deviation"], stats, "rrr")
            + f"\n\nTraining sentence 0 (`{text}`) has {len(p)} frames of {N_MELS} values; the sum of its {len(p) * N_MELS} "
            f"standardized values is {_f(sum(v for r in standardise(log_mel(p)) for v in r), 6)} and the first four entries of the "
            f"stacked vector of frame 0 are {', '.join(_f(v, 6) for v in F[0][:4])} (frame 0's own frames -2 and -1 are copies of "
            "frame 0, tolerance 1e-6).")


# ------------------------------------------------------------------------------------------------ milestone 3: the recogniser

@lru_cache(maxsize=None)
def grad_check():
    """(initial softmax row of frame 0, scaled loss, lattice loss, row-sum gap of gamma, worst gradient difference) of training sentence 0."""
    p = _sentence("train", 0)[2]
    X = features(p)
    th = init_params(bookrng.generator(INIT))
    _, ps = forward(th, X[:1])
    seg, labels = X[3:17], [ID["a"], ID["b"]]
    _, P = forward(th, seg)
    loss, gamma = ctc_scaled(P, labels)
    alpha, _ = sr._ctc_lattice(P, tuple(labels))
    ref = -math.log(alpha[-1][-1] + alpha[-1][-2])
    g = [0.0] * N_PARAMS
    loss_grad(th, seg, labels, g)
    worst = 0.0
    h = 1e-5
    for i in range(N_PARAMS):
        th[i] += h
        a = loss_grad(th, seg, labels)
        th[i] -= 2 * h
        b = loss_grad(th, seg, labels)
        th[i] += h
        worst = max(worst, abs((a - b) / (2 * h) - g[i]))
    row_gap = max(abs(sum(r) - 1.0) for r in gamma)
    return ps[0], loss, ref, row_gap, worst


def recogniser():
    th = init_params(bookrng.generator(INIT))
    row0, loss, ref, row_gap, worst = grad_check()
    rows = [("W1", f"{HID} x {D_IN}", HID * D_IN, "N(0, 1 / 80)"), ("b1", f"{HID}", HID, "0"),
            ("W2", f"{K} x {HID}", K * HID, "N(0, 1 / 24)"), ("b2", f"{K}", K, "0"), ("all", "", N_PARAMS, "")]
    return (_table(["parameter", "shape", "values", "initial value"], rows, "lrrl")
            + f"\n\nInitialised from `{INIT}` ({_seed(INIT)}): the {HID * D_IN} values of W1 in row-major order, each scale times one "
            f"normal(), then the {K * HID} of W2. The first three of W1 are {', '.join(_f(v, 6) for v in th[:3])}, the first three of W2 "
            f"are {', '.join(_f(v, 6) for v in th[HID * D_IN + HID:HID * D_IN + HID + 3])}. The softmax row of frame 0 of training sentence 0 at "
            f"this start is {', '.join(_f(v, 6) for v in row0)} (tolerance 1e-6).\n\n"
            f"The scaled recursion of the CTC loss on the 14 frames 3 .. 16 of that sentence with the label `ab` gives "
            f"{_f(loss, 10)}; Chapter 4's recursion in probabilities, summed over the two last states, gives {_f(ref, 10)}; they "
            f"agree to within 1e-12, and every row of the soft alignment sums to 1 to within 1e-12. The "
            f"hand-written backward pass against central differences with step 1e-5 on each of the {N_PARAMS} parameters: the "
            f"largest absolute difference is below 1e-6.")


# ------------------------------------------------------------------------------------------------ milestone 4: training and the round trip

def training():
    ep = (1, 2, 3, 5, 10, 20)
    rows = [(e, e * (N_TRAIN // BATCH), curve_cell(e, "clean"), curve_cell(e, "shifted")) for e in ep]
    return (f"{EPOCHS} epochs of {N_TRAIN // BATCH} steps at batch {BATCH}; each epoch `shuffle`s `range({N_TRAIN})` with the stream "
            f"`{ORDER}` ({_seed(ORDER)}) and takes four consecutive sentences per step. AdamW (0.9, 0.999, 1e-8), rate {LR:g}, no "
            "decay, no clipping, the gradient the mean over the batch. The first weight W1[0][0] is the one the nudges move. The "
            "table is the mean over the epoch's sentences of the CTC loss each had when its batch was formed (so the weights moved "
            "during the epoch). 'Clean' uses the ids of the alphabet; 'shifted' is the planted mistake of milestone 7.\n\n"
            + _table(["epoch", "steps so far", "clean: mean loss", "shifted: mean loss"], rows, "rrrr"))


def round_trip():
    rows = []
    for k, g in enumerate(SPEAKERS):
        n = sum(1 for e in data()[1] if e[1] == k)
        rows.append((k, f"{g:g}", "heard" if k < N_SEEN else "never heard", n, cell(f"wer_s{k}", f3), cell(f"cer_s{k}", f3)))
    rows.append(("training set", "", "heard", N_TRAIN, cell("train_wer", f3), cell("train_cer", f3)))
    rows.append(("test set", "", "", N_TEST, cell("wer", f3), cell("cer", f3)))
    words = sum(len(t.split()) for t, _, _, _ in data()[1])
    chars = sum(len(t) for t, _, _, _ in data()[1])
    return (f"The test set holds {words} reference words and {chars} reference characters. Text to waveform to text: each sentence is synthesised (milestone 1), resampled, framed, recognised greedily (the first "
            "largest entry of each frame, repeats collapsed, blanks dropped) and compared with the text it was made from. The test "
            "sentences are `project_speech_pipeline.test.<n>` with speaker n mod 6.\n\n"
            + _table(["speaker", "g", "in training", "sentences", "word error", "character error"], rows, "rrlrrr"))


# ------------------------------------------------------------------------------------------------ milestone 5: the representation

def rvq_table():
    frames_per_s = FS_REC / HOP
    rows = []
    for q in range(RVQ_Q + 1):
        bits = q * int(math.log2(RVQ_K))
        rows.append((q, bits, _i(int(bits * frames_per_s)), cell(f"rvq_mse{q}", lambda x: f"{x:.2f}"), cell(f"rvq_wer{q}", f3),
                     cell(f"rvq_cer{q}", f3)))
    n_fit = len([r for _, _, p, _ in data()[0] for r in p][::RVQ_EVERY])
    return (f"A residual quantizer of the standardized 16-value frames: {RVQ_Q} stages of {RVQ_K} codewords, trained by {RVQ_ITERS} Lloyd "
            f"rounds on every {RVQ_EVERY}rd frame of the training sentences ({n_fit} frames), starting centres from one `shuffle` of "
            f"the rows per stage with `{RVQ_NAME}` ({_seed(RVQ_NAME)}). The test frames are quantized and summed back, stacked, and read "
            "by the recogniser of milestone 4, unchanged. Bits are per frame; the mean squared error is the sum over the 16 values "
            "of the squared difference, averaged over the test frames (q = 0 returns zeros).\n\n"
            + _table(["stages", "bits per frame", "bits per second", "squared error per frame", "word error", "character error"], rows,
                     "rrrrrr"))


# ------------------------------------------------------------------------------------------------ milestone 6: the voice

def speaker_scores():
    E = [(s, embed(p)) for _, s, p, _ in data()[1]]
    gen, imp = [], []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            (gen if E[i][0] == E[j][0] else imp).append(_cos(E[i][1], E[j][1]))
    tau, eer = se.equal_error(gen, imp)
    return gen, imp, tau, eer


def speaker_block():
    gen, imp, tau, eer = speaker_scores()
    w, b = voice_map()
    rows = []
    for k, g in enumerate(SPEAKERS):
        clips = [e[2] for e in data()[1] if e[1] == k]
        single = [estimate_g([p]) for p in clips]
        mu = sum(single) / len(single)
        sd = math.sqrt(sum((x - mu) ** 2 for x in single) / len(single))
        rows.append((k, f"{g:g}", "heard" if k < N_SEEN else "never heard", len(clips), _f(mu, 4), _f(sd, 4), _f(estimate_g(clips[:ENROL]), 4)))
    mu_g = sum(gen) / len(gen)
    mu_i = sum(imp) / len(imp)
    head = (f"The embedding of a clip: the mean of its standardized frames whose largest value before standardizing is above {ACTIVE}, "
            "minus the mean of that quantity over the training sentences, made a unit vector. Over the 180 test clips, "
            f"{_i(len(gen))} same-speaker and {_i(len(imp))} different-speaker pairs scored by the dot product of the unit vectors "
            f"(summed over the 16 values in order): the mean same-speaker cosine is {_f(mu_g, 3)}, the mean different-speaker "
            f"cosine {_f(mu_i, 3)}, and the equal error rate (Chapter 5's rule) is {100 * eer:.1f}% at the threshold {_f(tau, 3)}.\n\n")
    tail = (f"\n\nThe map from embedding to speaker parameter is ridge regression of ln g on the embedding over the {N_TRAIN} training "
            f"sentences (the five heard speakers), penalty {EMB_RIDGE:g}, solved by Gaussian elimination with partial pivoting "
            f"on the 16 normal equations; its intercept is {_f(b, 4)} and its largest weight in magnitude {_f(max(abs(x) for x in w), 3)}. "
            "'One clip' averages g over each clip estimated alone and gives the spread; 'eight clips' estimates g from the first "
            f"{ENROL} clips of the speaker together (the mean of ln g over them).")
    return head + _table(["speaker", "true g", "role", "clips", "mean g, one clip", "standard deviation", "g, eight clips"], rows, "rrlrrrr") + tail


def voice_transfer():
    rows = []
    order = sorted([(SPEAKERS[k], f"trained speaker {k}", f"s{k}") for k in range(N_SEEN)] + [(None, "estimated", "hat")],
                   key=lambda r: (r[0] is None, r[0] or 0))
    rs = runs()[0]
    ghat = [r["g_hat"] for r in rs]
    est = stable.text(ghat[0], ghat[1:], lambda x: f"{x:.3f}")
    for g, label, tag in order:
        rows.append((label, est if g is None else f"{g:g}", cell(f"cos_{tag}", f3)))
    return (f"The unseen speaker {UNSEEN} (g = {SPEAKERS[UNSEEN]:g}). Its first {ENROL} test clips fix a reference embedding (the unit "
            f"vector of the sum of their embeddings) and the estimate g_hat. For each of the next {VOICE_CLIPS} clips: the recogniser "
            "writes its transcript, and the transcript is spoken again from the stream `project_speech_pipeline.voice.<n>` (n the clip's "
            f"index among the speaker's clips, {ENROL} .. {ENROL + VOICE_CLIPS - 1}), with the estimated g and with each trained speaker's g, "
            "so that only g differs between the voices. The table is the mean over those clips of the cosine between the "
            "reference and the embedding of the re-spoken clip, in the order of g.\n\n"
            + _table(["voice", "g", "mean cosine to the reference"], rows, "lrr")
            + f"\n\nWord error of the transcripts of the {VOICE_CLIPS} clips against their text: {cell('said_wer', f3)}. Word error "
            f"of the recogniser on the re-spoken audio (estimated voice) against the transcript it was made from: {cell('again_wer', f3)}.")


# ------------------------------------------------------------------------------------------------ milestones 7 and 8: mistakes and the bill

def mistakes():
    sl, hk = _centres("slaney"), _centres("htk")
    shift = max(abs(a - b) for a, b in zip(sl, hk))
    sums = [sum(r) for r in BANK]
    flat = sp.filterbank(FS_REC, N_FFT, N_MELS, 0.0, 4000.0, "slaney", False)
    fsums = [sum(r) for r in flat]
    alias_k, alias_l = tone_amplitudes()[5200.0]
    rows = [("resampling by linear interpolation", f"a 5200 Hz tone arrives at 2800 Hz with amplitude {alias_l:.3f}, not {alias_k:.1e}",
             cell("lin_wer", f3), cell("lin_cer", f3)),
            ("the HTK scale for the Slaney one", f"the filter centres move by up to {shift:.0f} Hz", cell("htk_wer", f3), cell("htk_cer", f3)),
            ("no area normalisation of the triangles", f"the row sums span a factor {max(fsums) / min(fsums):.1f}, not {max(sums) / min(sums):.1f}",
             cell("norm_wer", f3), cell("norm_cer", f3)),
            ("log10 in place of ln", f"every value is multiplied by {1 / math.log(10):.4f}", cell("log10_wer", f3), cell("log10_cer", f3)),
            ("label ids not offset past the blank", f"the mean training loss reaches {curve_cell(EPOCHS, 'shifted')} (a loss of -ln p is never below 0); spaces written: "
             f"{cell('spaces', lambda v: f'{v:.0f}', 'shifted')}, where the references hold {sum(t.count(' ') for t, _, _, _ in data()[1])}",
             cell("wer", f3, "shifted"), cell("cer", f3, "shifted"))]
    return ("Each mistake is planted alone in an otherwise correct pipeline (the first four at test time only, the recogniser being "
            "the one trained correctly; the last in training) and measured twice: by a number that needs no recogniser, and by "
            "the recogniser's error on the 180 test sentences.\n\n"
            + _table(["mistake", "what it does, measured directly", "word error", "character error"], rows, "llrr"))


def _taps_per_second():
    ker = sound.KaiserSinc(FS_SYN, FS_REC)
    n = FS_SYN
    return sum(len(sound.taps_of(ker, j, n)) for j in range(sound.out_length(n, FS_SYN, FS_REC)))


def bill():
    fps = FS_REC // HOP
    macs_net = D_IN * HID + HID * K
    fft = sp._fft_flops(N_FFT)
    nz = sum(1 for r in BANK for w in r if w > 0)
    taps = _taps_per_second()
    stage_ops = RVQ_Q * 3 * RVQ_K * N_MELS
    frames_total = sum(len(p) for _, _, p, _ in data()[0])
    train_fwd = 2 * macs_net
    train_bwd = 2 * (D_IN * HID + HID * K + HID * K)
    steps = EPOCHS * (N_TRAIN // BATCH)
    train_total = EPOCHS * frames_total * (train_fwd + train_bwd) + steps * N_PARAMS * 16
    text_chars = sum(len(t) for t, _, _, _ in data()[0]) / (sum(len(p) for _, _, p, _ in data()[0]) * HOP / FS_REC)
    rows = [("resampler, 12 kHz to 8 kHz", _i(2 * taps), _i(4 * FS_SYN), _i(4 * FS_REC),
             f"{taps / FS_REC:.1f} taps per output, each a multiply-add"),
            ("window and transform", _i(fps * (N_FFT + fft)), _i(4 * FS_REC), _i(4 * fps * (N_FFT // 2 + 1)),
             f"{N_FFT} multiplies and {fft:,} for the radix-2 transform per frame"),
            ("power and mel", _i(fps * (3 * (N_FFT // 2 + 1) + 2 * nz)), _i(4 * fps * (N_FFT // 2 + 1)), _i(4 * fps * N_MELS),
             f"3 per bin, then 2 per nonzero weight ({nz} of {N_MELS * (N_FFT // 2 + 1)})"),
            ("recogniser, forward", _i(fps * train_fwd), _i(4 * fps * N_MELS), _i(4 * fps * K), f"{N_PARAMS * 4:,} bytes of weights read once per frame"),
            ("quantizer, four stages", _i(fps * stage_ops), _i(4 * fps * N_MELS), _i(fps * RVQ_Q * 4 // 8), "3 per value per codeword, all stages")]
    train = (f"Training: {EPOCHS} epochs over {_i(frames_total)} frames is {_i(EPOCHS * frames_total)} frame passes of "
             f"{_i(train_fwd)} forward and {_i(train_bwd)} backward operations, plus {_i(steps)} AdamW steps of 16 operations on each "
             f"of {N_PARAMS:,} parameters: {_i(train_total)} operations in all, before the lattice (about 8 S per frame for a label of "
             f"S states, a few percent of the network's).")
    return ("Operations (a multiply-add is 2) and bytes (32-bit values) for one second of audio through each stage, counted from the "
            "shapes. The 'bytes in' and 'bytes out' columns are the arrays read and written.\n\n"
            + _table(["stage", "operations per second", "bytes in", "bytes out", "count"], rows, "lrrrl")
            + f"\n\nThe whole forward chain is {_i(2 * taps + fps * (N_FFT + fft + 3 * (N_FFT // 2 + 1) + 2 * nz + train_fwd))} operations per "
            f"second of audio, of which the resampler is {100 * 2 * taps / (2 * taps + fps * (N_FFT + fft + 3 * (N_FFT // 2 + 1) + 2 * nz + train_fwd)):.0f}%. "
            f"One frame of the recogniser reads {N_PARAMS * 4:,} bytes of weights for {train_fwd:,} operations: "
            f"{train_fwd / (N_PARAMS * 4):.2f} operations per byte, the 2 operations per 4-byte weight of any vector-matrix product. "
            f"The speech is {text_chars:.1f} characters a second.\n\n" + train)


# ------------------------------------------------------------------------------------------------ acceptance tests

def acceptance_tests():
    text0, s0, x0 = _wave("train", 0)
    p0 = _sentence("train", 0)[2]
    F0 = features(p0)
    kz = tone_amplitudes()
    row0, loss, ref, row_gap, worst = grad_check()
    gen, imp, tau, eer = speaker_scores()
    th = init_params(bookrng.generator(INIT))
    rs = runs()[0]
    ghat = stable.text(rs[0]["g_hat"], [r["g_hat"] for r in rs[1:]], lambda x: f"{x:.3f}")
    rows = [
        (1, "training sentence 0: text; samples at 12 kHz; sum; sum of squares", f"`{text0}`; {len(x0)}; {_f(sum(x0), 6)}; {_f(sum(v * v for v in x0), 6)}", "exact; exact; 1e-6; 1e-6"),
        (2, "tones 1000 and 3000 Hz through Kaiser sinc / linear interpolation", f"{_f(kz[1000.0][0], 4)}, {_f(kz[3000.0][0], 4)} / {_f(kz[1000.0][1], 4)}, {_f(kz[3000.0][1], 4)}", "0.0001"),
        (2, "tones 4400 and 5200 Hz, amplitude at 3600 and 2800 Hz: Kaiser sinc / linear", f"{kz[4400.0][0]:.1e}, {kz[5200.0][0]:.1e} / {_f(kz[4400.0][1], 4)}, {_f(kz[5200.0][1], 4)}", "below 1e-3; 0.0001"),
        (2, "frames of sentence 0; sum of its 544 standardized values", f"{len(p0)}; {_f(sum(v for r in standardise(log_mel(p0)) for v in r), 6)}", "exact; 1e-6"),
        (2, "first four entries of the stacked vector of frame 0", ", ".join(_f(v, 6) for v in F0[0][:4]), "1e-6"),
        (3, "parameters; first W1 value; softmax row 0 at the start", f"{N_PARAMS}; {_f(th[0], 6)}; " + ", ".join(_f(v, 6) for v in row0), "exact; 1e-6; 1e-6"),
        (3, "CTC loss of frames 3 .. 16 with label `ab` at the start", _f(loss, 8), "1e-8"),
        (3, "largest gradient difference against central differences", "below 1e-6", "the criterion"),
        (4, "mean training loss after epochs 1, 5 and 20", f"{curve_cell(1)}, {curve_cell(5)}, {curve_cell(20)}", "the digits shown"),
        (4, "word and character error: test set, training set", f"{cell('wer', f3)}, {cell('cer', f3)}; {cell('train_wer', f3)}, {cell('train_cer', f3)}", "the digits shown"),
        (4, "word error on the unseen speaker 5", cell("wer_s5", f3), "the digits shown"),
        (5, "word error after 1, 2, 3 and 4 quantizer stages", ", ".join(cell(f"rvq_wer{q}", f3) for q in (1, 2, 3, 4)), "the digits shown"),
        (5, "bits per second after 4 stages", f"{RVQ_Q * int(math.log2(RVQ_K)) * (FS_REC // HOP)}", "exact"),
        (6, "equal error rate of the embedding; same / different cosine", f"{100 * eer:.1f}%; {_f(sum(gen) / len(gen), 3)} / {_f(sum(imp) / len(imp), 3)}", "0.1; 0.001"),
        (6, f"g estimated for the unseen speaker (true {SPEAKERS[UNSEEN]:g}) from {ENROL} clips", ghat, "the digits shown"),
        (6, "mean cosine to the reference: estimated voice, g = 0.88, 1.12", f"{cell('cos_hat', f3)}, {cell('cos_s0', f3)}, {cell('cos_s4', f3)}", "the digits shown"),
        (7, "word error: linear resampler; HTK scale; no area norm; log10", "; ".join(cell(k, f3) for k in ("lin_wer", "htk_wer", "norm_wer", "log10_wer")), "the digits shown"),
        (7, "label shift: word error; character error; spaces written", f"{cell('wer', f3, 'shifted')}; {cell('cer', f3, 'shifted')}; {cell('spaces', lambda v: f'{v:.0f}', 'shifted')}", "the digits shown"),
        (7, "operations per second of audio through the resampler", _i(2 * _taps_per_second()), "exact"),
    ]
    return ("Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the "
            "last column, absolute; 'the digits shown' means the digits on which the run and its four nudged copies agree, and "
            "no more can be asked of another implementation.\n\n"
            + _table(["milestone", "quantity", "value", "tolerance"], rows, "rllr"))


# ------------------------------------------------------------------------------------------------ figure data

def figure_mels():
    """Log-mel frames (before standardizing) of test sentence 0 by the Kaiser resampler and by linear interpolation, and its text."""
    text, _, p_k, p_l = _sentence("test", 0)
    return text, log_mel(p_k), log_mel(p_l)
