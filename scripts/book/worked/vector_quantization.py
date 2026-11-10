# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Discrete Latents and Vector Quantization".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: vector_quantization.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.
Earlier modules are imported, never copied: the curved sensor readings of `first_models`, the curve and the off-curve test of `vae`,
the toy perceptual distance of `clip`, the stable-digits rule of `stable`.

* the budget of a token grid: positions, bits and the factor by which a codebook shrinks an image, for two presets of brain;
* the nearest-code rule on a handful of points, and the staircase it makes of a loss, with the straight-through gradient beside it;
* the three terms of the quantizer's loss, their gradients, and the two places the weight beta can sit;
* a tokenizer small enough to check: 16 sensor readings, an encoder that returns four 2-vectors, a codebook of K entries, a decoder;
  its gradient check, its training log, the price of the codes against an autoencoder without them, the codes that die and the
  recipes that revive them, the straight-through estimator against its absence, and the two places for beta trained;
* a sequence model over the four tokens: the bits it needs and the readings it generates;
* why a decoder that cannot tell two pictures apart blurs under squared error, and what a feature distance does instead;
* what the codebook search costs, counted in multiply-adds and bytes.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. Trained
numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the nudges of
`stable.py`. The five runs of each model are trained in at most four worker processes.

Swedish Embedded AB implements solutions for image tokenizers whose codebooks are trained, checked and kept alive. If your team needs
expertise in vector quantization and discrete latent image models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache
from operator import mul

import bookrng
import clip
import first_models as fm
import stable
import vae

# ------------------------------------------------------------------ constants

D = fm.D                                   # sensors
T = 4                                      # latent positions: four 2-vectors per reading
DZ = 2                                     # entries of one latent vector (the codebook dimension)
HID = 12                                   # hidden width of encoder and decoder
BETA = 0.25                                # weight of one of the two quantizer terms
EPOCHS, BATCH, PEAK = 200, 32, 1e-2        # 8 steps per epoch, 1600 steps; warmup 160 steps, cosine to PEAK / 100
INIT, CODES, BATCHES, RESET = "vector_quantization.init", "vector_quantization.codebook", "vector_quantization.batches", "vector_quantization.reset"
SAMPLE, GRADCHECK = "vector_quantization.sample", "vector_quantization.gradcheck"
LOG_EPOCHS = (1, 5, 25, 100, 200)
NOISE_FLOOR = fm.SENSOR_NOISE ** 2         # the mean squared error that sensor noise alone leaves
N_NET = HID * D + HID + T * DZ * HID + T * DZ + HID * T * DZ + HID + D * HID + D        # 624 weights and biases outside the codebook
N_SAMPLES = 300
WIDE = 4.0                                 # half-width of the 'wide' codebook initialisation


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed(name):
    return f"`{name}` (seed 0x{bookrng.seed_of(name):016x})"


def _f(x, d=4):
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _g(x, d=3):
    return f"{x:.{d}g}"


# ------------------------------------------------------------------ the tokenizer: parameters

class Config:
    """One training recipe. K codes; `beta_on` is 'commitment' (the paper's placement) or 'codebook' (the other placement);
    `beta_on` 'none' drops both terms (a gradient check of the reconstruction alone); `ste` False cuts the gradient at the quantizer; `quant` False removes the quantizer (a plain autoencoder); `init` names the
    codebook initialisation; `reset` revives codes that an epoch never used."""

    def __init__(self, K=16, beta_on="commitment", ste=True, quant=True, init="data", reset=False, epochs=EPOCHS):
        self.K, self.beta_on, self.ste, self.quant, self.init, self.reset, self.epochs = K, beta_on, ste, quant, init, reset, epochs

    @property
    def key(self):
        return (self.K, self.beta_on, self.ste, self.quant, self.init, self.reset, self.epochs)

    @property
    def weights(self):
        """(weight of the codebook term, weight of the commitment term)."""
        if self.beta_on == "none":
            return 0.0, 0.0
        return (BETA, 1.0) if self.beta_on == "codebook" else (1.0, BETA)


def n_params(cfg):
    return N_NET + cfg.K * DZ


def unpack(th, cfg):
    """Slices of the flat vector, in this order, each matrix row-major: W1 [12,16], b1 [12], W2 [8,12], b2 [8], W3 [12,8], b3 [12],
    W4 [16,12], b4 [16], then the codebook C [K,2]."""
    o = 0

    def take(n):
        nonlocal o
        r = th[o:o + n]
        o += n
        return r
    W1 = [take(D) for _ in range(HID)]
    b1 = take(HID)
    W2 = [take(HID) for _ in range(T * DZ)]
    b2 = take(T * DZ)
    W3 = [take(T * DZ) for _ in range(HID)]
    b3 = take(HID)
    W4 = [take(HID) for _ in range(D)]
    b4 = take(D)
    C = [take(DZ) for _ in range(cfg.K)] if cfg.quant else []
    return W1, b1, W2, b2, W3, b3, W4, b4, C


def init_network(gen):
    """Weights N(0, 1/n_in) layer by layer (16->12, 12->8, 8->12, 12->16) in row-major order, each layer's biases 0. Draws: weights only."""
    th = []
    for n_in, n_out in ((D, HID), (HID, T * DZ), (T * DZ, HID), (HID, D)):
        s = 1.0 / math.sqrt(n_in)
        th.extend(gen.normal() * s for _ in range(n_out * n_in))
        th.extend([0.0] * n_out)
    return th


def encode_rows(th, cfg, x):
    """The T encoder rows z_e[t] (each 2 numbers) of one reading, plus the hidden activations."""
    W1, b1, W2, b2 = unpack(th, cfg)[:4]
    p1 = [sum(map(mul, w, x)) + b for w, b in zip(W1, b1)]
    a1 = [fm.gelu(v) for v in p1]
    out = [sum(map(mul, w, a1)) + b for w, b in zip(W2, b2)]
    return [out[DZ * t:DZ * t + DZ] for t in range(T)], p1, a1


def nearest(C, z):
    """Index of the code at the smallest squared distance to z; on a tie the lowest index wins."""
    best, arg = None, 0
    for k, c in enumerate(C):
        d = (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2
        if best is None or d < best:
            best, arg = d, k
    return arg


def decode_flat(th, cfg, zq):
    """The reconstruction from the flat latent zq (8 numbers), and the decoder's hidden layer."""
    W3, b3, W4, b4 = unpack(th, cfg)[4:8]
    u = [sum(map(mul, w, zq)) + b for w, b in zip(W3, b3)]
    a3 = [fm.gelu(v) for v in u]
    return [sum(map(mul, w, a3)) + b for w, b in zip(W4, b4)], u, a3


def init_codebook(cfg, th_net, X):
    """The starting codebook, drawn from the stream CODES. 'tiny': both coordinates of every code uniform in (-1/K, 1/K), the
    initialisation of the original VQ-VAE code (draws: K x 2 uniforms, code by code). 'data': every code is the encoder row z_e[t]
    of a reading chosen by integer(len(X)) and a position chosen by integer(T) (draws: reading then position, code by code), at
    the initial encoder weights."""
    gen = bookrng.generator(CODES)
    if cfg.init == "tiny":
        return [(2.0 * gen.uniform() - 1.0) / cfg.K for _ in range(cfg.K * DZ)]
    if cfg.init == "wide":
        return [WIDE * (2.0 * gen.uniform() - 1.0) for _ in range(cfg.K * DZ)]
    out = []
    for _ in range(cfg.K):
        i = gen.integer(len(X))
        t = gen.integer(T)
        out.extend(encode_rows(th_net, cfg, X[i])[0][t])
    return out


# ------------------------------------------------------------------ loss and gradient

def forward_loss(th, cfg, X, latch):
    """Mean loss over X with the code assignment, z0 and q0 pinned to `latch` (a list per example of T (k, z0, q0) triples). The loss
    is rec + w_cb * mean((C[k] - z0)^2) + w_com * mean((z_e - q0)^2) with the decoder fed z_e + (q0 - z0). At the point where the
    latch was taken this is the objective of training; as a function of the parameters it is smooth, so a central difference
    checks the gradient that a training step applies."""
    w_cb, w_com = cfg.weights
    C = unpack(th, cfg)[8]
    total = 0.0
    for x, lt in zip(X, latch):
        rows = encode_rows(th, cfg, x)[0]
        zq = []
        cb = com = 0.0
        for t in range(T):
            k, z0, q0 = lt[t]
            for j in range(DZ):
                zq.append(rows[t][j] + (q0[j] - z0[j]))
                cb += (C[k][j] - z0[j]) ** 2
                com += (rows[t][j] - q0[j]) ** 2
        xh = decode_flat(th, cfg, zq)[0]
        rec = sum((a - b) ** 2 for a, b in zip(xh, x)) / D
        total += rec + w_cb * cb / (T * DZ) + w_com * com / (T * DZ)
    return total / len(X)


def take_latch(th, cfg, X):
    """The latch of the current parameters: per example and position the code k, z0 = z_e and q0 = C[k]."""
    C = unpack(th, cfg)[8]
    out = []
    for x in X:
        rows = encode_rows(th, cfg, x)[0]
        lt = []
        for t in range(T):
            k = nearest(C, rows[t])
            lt.append((k, list(rows[t]), list(C[k])))
        out.append(lt)
    return out


def loss_grad(th, cfg, X):
    """(mean loss, mean reconstruction, mean codebook term, mean commitment term, gradient, codes) over the examples X.

    Examples are accumulated in order; the gradient is divided by the number of examples at the end. With cfg.quant False the
    decoder is fed z_e and only the reconstruction counts. With cfg.ste False the quantizer is a plain lookup: the decoder's
    gradient goes to the codebook entry it read and nothing reaches the encoder through the quantizer."""
    W1, b1, W2, b2, W3, b3, W4, b4, C = unpack(th, cfg)
    w_cb, w_com = cfg.weights
    B = len(X)
    gW1 = [[0.0] * D for _ in range(HID)]
    gb1 = [0.0] * HID
    gW2 = [[0.0] * HID for _ in range(T * DZ)]
    gb2 = [0.0] * (T * DZ)
    gW3 = [[0.0] * (T * DZ) for _ in range(HID)]
    gb3 = [0.0] * HID
    gW4 = [[0.0] * HID for _ in range(D)]
    gb4 = [0.0] * D
    gC = [[0.0] * DZ for _ in range(len(C))]
    rec_s = cb_s = com_s = 0.0
    codes = []
    for x in X:
        rows, p1, a1 = encode_rows(th, cfg, x)
        zq, ks = [], []
        for t in range(T):
            if cfg.quant:
                k = nearest(C, rows[t])
                ks.append(k)
                zq.extend(C[k])
            else:
                zq.extend(rows[t])
        codes.append(ks)
        xh, u, a3 = decode_flat(th, cfg, zq)
        rec_s += sum((a - b) ** 2 for a, b in zip(xh, x)) / D
        dxh = [2.0 * (a - b) / D for a, b in zip(xh, x)]
        da3 = [sum(dxh[k] * W4[k][j] for k in range(D)) for j in range(HID)]
        for k in range(D):
            gb4[k] += dxh[k]
            for j in range(HID):
                gW4[k][j] += dxh[k] * a3[j]
        du = [da3[j] * fm.gelu_slope(u[j]) for j in range(HID)]
        for j in range(HID):
            gb3[j] += du[j]
            for i in range(T * DZ):
                gW3[j][i] += du[j] * zq[i]
        dzq = [sum(du[j] * W3[j][i] for j in range(HID)) for i in range(T * DZ)]
        dz = [0.0] * (T * DZ)
        if cfg.quant:
            for t in range(T):
                k = ks[t]
                for j in range(DZ):
                    i = DZ * t + j
                    diff = C[k][j] - rows[t][j]
                    cb_s += diff * diff / (T * DZ)
                    com_s += diff * diff / (T * DZ)
                    if cfg.ste:
                        dz[i] += dzq[i]
                    else:
                        gC[k][j] += dzq[i]
                    gC[k][j] += w_cb * 2.0 * diff / (T * DZ)       # the codebook term reaches the codebook only
                    dz[i] += w_com * 2.0 * (-diff) / (T * DZ)      # the commitment term reaches the encoder only
        else:
            dz = dzq
        for i in range(T * DZ):
            gb2[i] += dz[i]
            for j in range(HID):
                gW2[i][j] += dz[i] * a1[j]
        dp1 = [sum(dz[i] * W2[i][j] for i in range(T * DZ)) * fm.gelu_slope(p1[j]) for j in range(HID)]
        for j in range(HID):
            gb1[j] += dp1[j]
            for i in range(D):
                gW1[j][i] += dp1[j] * x[i]
    flat = []
    for r in gW1:
        flat += r
    flat += gb1
    for r in gW2:
        flat += r
    flat += gb2
    for r in gW3:
        flat += r
    flat += gb3
    for r in gW4:
        flat += r
    flat += gb4
    if cfg.quant:
        for r in gC:
            flat += r
    loss = (rec_s + w_cb * cb_s + w_com * com_s) / B
    return loss, rec_s / B, cb_s / B, com_s / B, [g / B for g in flat], codes


# ------------------------------------------------------------------ training

def usage(codes, K):
    """(number of codes used at least once, perplexity exp(-sum p ln p) of the code frequencies) over a list of per-example token lists."""
    counts = [0] * K
    n = 0
    for ks in codes:
        for k in ks:
            counts[k] += 1
            n += 1
    ent = -sum(c / n * math.log(c / n) for c in counts if c)
    return sum(1 for c in counts if c), math.exp(ent)


def train(cfg, nudge=0.0):
    """Train for cfg.epochs epochs; returns {'theta', 'log'} where log[epoch] = (mean loss, reconstruction, codebook term, commitment
    term, codes used, perplexity) over the epoch's 8 batches (the usage over the whole epoch's tokens). Per epoch one shuffle of
    0 .. 255 from the stream BATCHES; Adam (0.9, 0.999, 1e-8), no weight decay, no clipping, warmup of a tenth of the steps then a
    cosine from PEAK to PEAK / 100. With cfg.reset, at the end of every epoch but the last, each code that the epoch never used is
    replaced by the encoder row z_e[t] of a reading chosen by integer(256) and a position by integer(4) (stream RESET, drawn
    reading then position, dead codes in index order), and its Adam moments are zeroed. The first weight starts `nudge` away."""
    tr, _, _, _ = fm.curved_data()
    th = init_network(bookrng.generator(INIT))
    if cfg.quant:
        th = th + init_codebook(cfg, th, tr)
    th[0] += nudge
    n = len(th)
    m, v = [0.0] * n, [0.0] * n
    shuf, rst = bookrng.generator(BATCHES), bookrng.generator(RESET)
    order = list(range(len(tr)))
    per_epoch = len(tr) // BATCH
    total = cfg.epochs * per_epoch
    warmup = total // 10
    c0 = N_NET
    log, t = {}, 0
    for epoch in range(1, cfg.epochs + 1):
        shuf.shuffle(order)
        acc = [0.0, 0.0, 0.0, 0.0]
        epoch_codes = []
        for k in range(per_epoch):
            rows = order[k * BATCH:(k + 1) * BATCH]
            loss, rec, cb, com, g, codes = loss_grad(th, cfg, [tr[i] for i in rows])
            acc[0] += loss
            acc[1] += rec
            acc[2] += cb
            acc[3] += com
            epoch_codes.extend(codes)
            fm.adamw_step(th, g, m, v, t + 1, fm.schedule_rate(t, PEAK, PEAK / 100.0, warmup, total), 0.0)
            t += 1
        if cfg.quant:
            used, ppl = usage(epoch_codes, cfg.K)
            if cfg.reset and epoch < cfg.epochs:
                counts = [0] * cfg.K
                for ks in epoch_codes:
                    for k in ks:
                        counts[k] += 1
                for k in range(cfg.K):
                    if counts[k] == 0:
                        i = rst.integer(len(tr))
                        p = rst.integer(T)
                        z = encode_rows(th, cfg, tr[i])[0][p]
                        for j in range(DZ):
                            th[c0 + DZ * k + j] = z[j]
                            m[c0 + DZ * k + j] = 0.0
                            v[c0 + DZ * k + j] = 0.0
        else:
            used, ppl = 0, 0.0
        if epoch in LOG_EPOCHS:
            log[epoch] = tuple(a / per_epoch for a in acc) + (used, ppl)
    return {"theta": th, "log": log}


def _train_job(job):
    return train(Config(*job[0]), job[1])


def _cfg_from_key(key):
    return Config(*key)


@lru_cache(maxsize=None)
def runs(key):
    """The run for the configuration with this key and its four nudged copies, trained in four worker processes: a list of 5."""
    cfg = Config(*key)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_train_job, [(key, d) for d in (0.0, *stable.NUDGES)])


def test_mse(cfg, th):
    """Mean squared error per sensor on the 128 test readings, each reading encoded, quantized (when the configuration has a
    codebook) and decoded."""
    _, _, te, _ = fm.curved_data()
    C = unpack(th, cfg)[8]
    err = 0.0
    for x in te:
        rows = encode_rows(th, cfg, x)[0]
        zq = []
        for t in range(T):
            zq.extend(C[nearest(C, rows[t])] if cfg.quant else rows[t])
        xh = decode_flat(th, cfg, zq)[0]
        err += sum((a - b) ** 2 for a, b in zip(xh, x)) / D
    return err / len(te)


# ------------------------------------------------------------------ tokens of a trained model

def tokens(cfg, th, X):
    """The T codes of every reading of X."""
    C = unpack(th, cfg)[8]
    return [[nearest(C, row) for row in encode_rows(th, cfg, x)[0]] for x in X]


# ------------------------------------------------------------------ the recipes that are trained

RECIPES = {
    "main": Config(K=16),
    "autoencoder": Config(quant=False),
    "k4": Config(K=4), "k8": Config(K=8), "k32": Config(K=32), "k64": Config(K=64),
    "wide": Config(K=64, init="wide"),
    "wide_reset": Config(K=64, init="wide", reset=True),
    "no_ste": Config(K=16, ste=False),
    "on_codebook": Config(K=16, beta_on="codebook"),
}


def _job(job):
    return train(Config(*job[0]), job[1])


@lru_cache(maxsize=None)
def all_runs():
    """name -> the 5 runs (the run and its four nudged copies) of every recipe, trained in four worker processes."""
    jobs = [(RECIPES[n].key, d) for n in RECIPES for d in (0.0, *stable.NUDGES)]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    return {n: out[5 * i:5 * i + 5] for i, n in enumerate(RECIPES)}


@lru_cache(maxsize=None)
def evaluation(name, i):
    """Summary numbers of run i of the recipe: test error, codes used and perplexity on the 256 training readings, final log."""
    cfg, run = RECIPES[name], all_runs()[name][i]
    th = run["theta"]
    tr, _, _, _ = fm.curved_data()
    used, ppl = usage(tokens(cfg, th, tr), cfg.K) if cfg.quant else (0, 0.0)
    return {"test": test_mse(cfg, th), "used": used, "ppl": ppl, "final": run["log"][cfg.epochs], "log": run["log"]}


def _cells(name, getter, fmt):
    """The digits of getter(evaluation) that the run and its four nudged copies share."""
    vals = [getter(evaluation(name, i)) for i in range(5)]
    return stable.text(vals[0], vals[1:], fmt)


NOTE = ("Every trained number is printed with the digits that the run and four copies of it, started with the first weight changed by +1e-12, "
        "-1e-12, +3e-12 and -3e-12, share (the rule of `stable.py`); here all five agree on every digit shown.")


def _note(table):
    return stable.RULE if ("about " in table or "unstable" in table) else NOTE


# ------------------------------------------------------------------ the budget of a token grid

# (name, image side in pixels, channel multipliers of the encoder, codebook size K, codebook width d) of two presets of brain.
PRESETS = (("CodeFormer tokenizer", 512, (1, 2, 2, 4, 4, 8), 1024, 256),
           ("LlamaGen VQ-16 tokenizer", 384, (1, 1, 2, 2, 4), 16384, 8))


def token_budget():
    rows = []
    for name, side, mult, K, d in PRESETS:
        down = 2 ** (len(mult) - 1)
        grid = side // down
        pos = grid * grid
        bits_pos = int(round(math.log2(K)))
        bits = pos * bits_pos
        raw = 3 * side * side
        rows.append((name, f"{side} x {side}", len(mult) - 1, down, f"{grid} x {grid} = {pos}", f"{K:,}", bits_pos, f"{bits:,}", f"{bits // 8:,}",
                     f"{raw:,}", _f(raw / (bits / 8), 1)))
    return ("An encoder halves the side once for each level but the last, so the side shrinks by 2 raised to (number of levels - 1); a "
            "position holds one index into the codebook, which costs log2 K bits; the image is stored as 8-bit RGB.\n\n"
            + _table(["tokenizer", "image", "halvings", "side divided by", "positions", "codebook size K", "bits per position",
                      "bits per image", "bytes per image", "bytes of the 8-bit image", "image bytes per code byte"], rows))


# ------------------------------------------------------------------ the nearest-code rule

POINTS = ((0.2, 0.1), (0.9, 0.2), (0.5, 0.0), (0.4, 0.8), (1.2, 1.1), (-0.3, 0.4))
CODEBOOK4 = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def quantize_points():
    rows, total = [], 0.0
    for z in POINTS:
        d2 = [(z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2 for c in CODEBOOK4]
        k = min(range(4), key=lambda i: (d2[i], i))           # smallest distance, lowest index on a tie
        total += d2[k]
        rows.append((f"({z[0]:g}, {z[1]:g})", *[_f(d, 2) for d in d2], k, f"({CODEBOOK4[k][0]:g}, {CODEBOOK4[k][1]:g})", _f(d2[k], 2)))
    return ("Codebook: c0 = (0, 0), c1 = (1, 0), c2 = (0, 1), c3 = (1, 1). Each point is replaced by the code at the smallest squared "
            "distance; when two codes are equally near, the lower index wins.\n\n"
            + _table(["point z", "dist to c0", "dist to c1", "dist to c2", "dist to c3", "index k", "code q", "squared error"], rows)
            + f"\n\nMean squared error of the six points: {_f(total / len(POINTS), 4)}.")


# ------------------------------------------------------------------ the staircase

STAIR_CODES = (-1.0, 0.0, 1.0)
STAIR_GAIN, STAIR_TARGET, STAIR_H = 2.0, 0.7, 1e-4


def _stair_q(z):
    return STAIR_CODES[min(range(3), key=lambda i: ((z - STAIR_CODES[i]) ** 2, i))]


def _stair_loss(z):
    return (STAIR_GAIN * _stair_q(z) - STAIR_TARGET) ** 2


def staircase_data():
    """(z, loss through the codebook, loss without it, literal derivative by central difference) on a fine grid of [-1.4, 1.4]."""
    zs = [-1.4 + 0.001 * k for k in range(2801)]
    return zs, [_stair_loss(z) for z in zs], [(STAIR_GAIN * z - STAIR_TARGET) ** 2 for z in zs]


def staircase():
    rows = []
    for z in (-0.8, -0.4, 0.0, 0.3, 0.45, 0.5, 0.55, 0.9):
        q = _stair_q(z)
        fd = (_stair_loss(z + STAIR_H) - _stair_loss(z - STAIR_H)) / (2.0 * STAIR_H)
        ste = 2.0 * (STAIR_GAIN * q - STAIR_TARGET) * STAIR_GAIN             # dL/dq of the loss at q, handed to z
        free = 2.0 * (STAIR_GAIN * z - STAIR_TARGET) * STAIR_GAIN            # dL/dz if there were no codebook
        rows.append((_f(z, 2), _f(q, 1), _f(_stair_loss(z), 2), _f(fd, 1), _f(ste, 2), _f(free, 2)))
    return ("Codebook {-1, 0, 1}, decoder g(q) = 2q, target 0.7, loss L = (g(q) - 0.7)^2, the lower index winning a tie (z = 0.5 is "
            "equally near 0 and 1). The derivative of the loss through the codebook is a central difference with step 0.0001.\n\n"
            + _table(["z", "code q", "loss L", "central difference dL/dz", "dL/dq (straight-through)", "dL/dz with no codebook"], rows))


# ------------------------------------------------------------------ the three terms and where beta sits

def _surrogate(ze, c, g_target, w_cb, w_com, z0, q0):
    """Loss of the stand-in used by the gradient examples: reconstruction (zq - target)^2 / 2 fed z_e + (q0 - z0), plus the two terms."""
    n = len(ze)
    rec = 0.5 * sum((z + (qq - zz) - t) ** 2 for z, qq, zz, t in zip(ze, q0, z0, g_target))
    cb = sum((a - b) ** 2 for a, b in zip(c, z0)) / n
    com = sum((a - b) ** 2 for a, b in zip(ze, q0)) / n
    return rec + w_cb * cb + w_com * com


def term_gradients():
    ze, q, target = [0.30, -0.20], [0.50, 0.10], [0.10, 0.45]
    n = len(ze)
    g = [qq - t for qq, t in zip(q, target)]                    # d rec / d zq at zq = q
    rows, checks = [], []
    for label, (w_cb, w_com) in (("beta on the commitment term (VQ-VAE paper)", (1.0, BETA)), ("beta on the codebook term (CodeFormer's reference)", (BETA, 1.0))):
        enc_rec = g
        enc_com = [w_com * 2.0 * (a - b) / n for a, b in zip(ze, q)]
        cb_cb = [w_cb * 2.0 * (b - a) / n for a, b in zip(ze, q)]
        enc = [a + b for a, b in zip(enc_rec, enc_com)]
        # central differences of the smooth surrogate with z0 = z_e and q0 = q held fixed
        h, worst = 1e-6, 0.0
        for j in range(n):
            def f_e(d):
                z = list(ze)
                z[j] += d
                return _surrogate(z, q, target, w_cb, w_com, ze, q)

            def f_c(d):
                c = list(q)
                c[j] += d
                return _surrogate(ze, c, target, w_cb, w_com, ze, q)
            worst = max(worst, abs((f_e(h) - f_e(-h)) / (2 * h) - enc[j]), abs((f_c(h) - f_c(-h)) / (2 * h) - cb_cb[j]))
        pull_c = math.sqrt(sum(a * a for a in cb_cb))
        pull_e = math.sqrt(sum(a * a for a in enc_com))
        rows.append((label, f"({_f(enc_rec[0], 3)}, {_f(enc_rec[1], 3)})", f"({_f(enc_com[0], 4)}, {_f(enc_com[1], 4)})",
                     f"({_f(cb_cb[0], 4)}, {_f(cb_cb[1], 4)})", _f(pull_c / pull_e, 4), "below 1e-6" if worst < 1e-6 else f"{worst:.1e}"))
        checks.append(pull_c / pull_e)
    return ("One position with a 2-vector: z_e = (0.30, -0.20), nearest code q = (0.50, 0.10), n = 2 entries, beta = 0.25. The decoder is "
            "stood in for by rec = (1/2)||z_q - t||^2 with t = (0.10, 0.45), so that the gradient it hands back at z_q = q is q - t = "
            "(0.40, -0.35). The gradient that the encoder receives is that vector (straight-through) plus the commitment gradient; the "
            "codebook entry receives the codebook gradient alone. The last column is the largest difference between each formula and a "
            "central difference (step 1e-6) of the smooth loss with the stop-gradient values held fixed.\n\n"
            + _table(["placement of beta", "straight-through into the encoder", "commitment into the encoder", "codebook term into the code",
                      "pull on the code / pull on the encoder", "formulas against finite differences"], rows)
            + f"\n\nMoving beta from the codebook term to the commitment term multiplies the ratio of the pull on the code to the pull on the encoder by "
              f"{_f(checks[0] / checks[1], 1)}, which is 1/beta^2.")


# ------------------------------------------------------------------ the gradient check of the tokenizer

def _initial(cfg):
    tr, _, _, _ = fm.curved_data()
    th = init_network(bookrng.generator(INIT))
    return th + init_codebook(cfg, th, tr) if cfg.quant else th


def _groups(cfg):
    sizes = [("W1", HID * D), ("b1", HID), ("W2", T * DZ * HID), ("b2", T * DZ), ("W3", HID * T * DZ), ("b3", HID), ("W4", D * HID), ("b4", D)]
    if cfg.quant:
        sizes.append(("codebook C", cfg.K * DZ))
    out, o = [], 0
    for name, n in sizes:
        out.append((name, o, o + n))
        o += n
    return out


FD_STEP = 1e-5


def gradient_check():
    tr, _, _, _ = fm.curved_data()
    X = tr[:8]
    rows = []
    for placement in ("commitment", "codebook"):
        cfg = Config(K=16, beta_on=placement)
        th = _initial(cfg)
        latch = take_latch(th, cfg, X)
        grad = loss_grad(th, cfg, X)[4]
        fails, worst = 0, {}
        for name, lo, hi in _groups(cfg):
            w = 0.0
            for i in range(lo, hi):
                up, dn = list(th), list(th)
                up[i] += FD_STEP
                dn[i] -= FD_STEP
                num = (forward_loss(up, cfg, X, latch) - forward_loss(dn, cfg, X, latch)) / (2.0 * FD_STEP)
                diff = abs(grad[i] - num)
                w = max(w, diff)
                if diff > 1e-6 + 1e-6 * max(abs(grad[i]), abs(num)):
                    fails += 1
            worst[name] = w
        rows.append((f"beta on the {placement} term", len(th), fails,
                     "below 1e-6" if all(w < 1e-6 for w in worst.values()) else f"{max(worst.values()):.1e}"))
    # the reconstruction term alone, differentiated as written (the codes recomputed at every perturbed point)
    cfg = Config(K=16, beta_on="none")
    th = _initial(cfg)
    ste = loss_grad(th, cfg, X)[4]
    n_enc = sum(hi - lo for name, lo, hi in _groups(cfg)[:4])
    C = unpack(th, cfg)[8]

    def literal(t):
        total = 0.0
        for x in X:
            zq = []
            for row in encode_rows(t, cfg, x)[0]:
                zq.extend(C[nearest(C, row)])
            total += sum((a - b) ** 2 for a, b in zip(decode_flat(t, cfg, zq)[0], x)) / D
        return total / len(X)
    nonzero_fd = 0
    for i in range(n_enc):
        up, dn = list(th), list(th)
        up[i] += FD_STEP
        dn[i] -= FD_STEP
        if literal(up) != literal(dn):
            nonzero_fd += 1
    nonzero_ste = sum(1 for i in range(n_enc) if ste[i] != 0.0)
    head = (f"The first 8 training readings, the initial parameters ({_seed(INIT)}, then the codebook from {_seed(CODES)}, initialisation "
            f"'data'), K = 16. The assignments, z_0 and q_0 are pinned (the latch) and each of the parameters is moved by plus and minus "
            f"{FD_STEP:g} in turn; an entry fails when |analytic - numerical| exceeds 1e-6 + 1e-6 max(|analytic|, |numerical|).\n\n")
    table = _table(["objective", "parameters checked", "entries failing the gate", "worst absolute difference"], rows)
    second = (f"\n\nThe reconstruction term alone, written as the network computes it, with the nearest code looked up again at every perturbed "
              f"point, has the same {n_enc} encoder parameters (W1, b1, W2, b2): a central difference of step {FD_STEP:g} is nonzero for "
              f"{nonzero_fd} of them, while the straight-through gradient is nonzero for {nonzero_ste}.")
    return head + table + second


# ------------------------------------------------------------------ the recipe and what it learns

def _recipe_text():
    return (f"Data: the 256 training readings of `first_models.curved_data` (16 sensors, noise sd {fm.SENSOR_NOISE}), tested on its 128 test readings. "
            f"Encoder 16-12-8 with GELU after the first layer, whose 8 outputs are read as {T} rows of {DZ} numbers; decoder 8-12-16 with GELU; "
            f"{N_NET} weights and biases outside the codebook. Weights N(0, 1/n_in) layer by layer in row-major order, biases 0, from {_seed(INIT)} "
            f"(weights only, layer by layer). Batch {BATCH} (8 steps per epoch), {EPOCHS} epochs, one shuffle of 0 .. 255 per epoch from {_seed(BATCHES)}; "
            f"Adam (0.9, 0.999, 1e-8), no decay and no clip, rate warming up over {EPOCHS * 8 // 10} steps to {PEAK:g} and following a cosine to {PEAK / 100:g}; "
            f"the codebook is trained by the same optimiser as the weights. Loss per reading: rec + codebook term + commitment term, rec the mean over the 16 sensors "
            f"of the squared error and each term the mean over the {T * DZ} entries of the squared distance, with beta = {BETA}.")


def training_log():
    rows = []
    for epoch in LOG_EPOCHS:
        cells = [str(epoch)]
        for k, fmt in ((0, lambda x: _f(x, 4)), (1, lambda x: _f(x, 4)), (2, lambda x: _f(x, 4))):
            cells.append(_cells("main", lambda e, k=k: e["log"][epoch][k], fmt))
        cells.append(_cells("main", lambda e: e["log"][epoch][4], lambda x: f"{int(round(x))}"))
        cells.append(_cells("main", lambda e: e["log"][epoch][5], lambda x: _f(x, 2)))
        rows.append(cells)
    table = _table(["epoch", "loss", "reconstruction", "mean (z_e - q)^2", "codes used (of 16)", "perplexity of the codes"], rows)
    return (_recipe_text() + " K = 16; the codebook starts at 16 encoder rows (reading by `integer(256)`, then position by `integer(4)`, "
            f"code by code, from {_seed(CODES)}). Each cell is the mean over the epoch's 8 batches (the codes used and the perplexity "
            "count the epoch's 1024 tokens).\n\n" + table + "\n\n" + _note(table))


def rate_table():
    rows = []
    for label, name, bits in (("autoencoder, no codebook", "autoencoder", "8 numbers"), ("K = 4", "k4", "8"), ("K = 8", "k8", "12"), ("K = 16", "main", "16"),
                              ("K = 32", "k32", "20"), ("K = 64", "k64", "24")):
        cfg = RECIPES[name]
        rows.append((label, bits, _cells(name, lambda e: e["final"][1], lambda x: _f(x, 4)), _cells(name, lambda e: e["test"], lambda x: _f(x, 4)),
                     "" if not cfg.quant else _cells(name, lambda e: e["used"], lambda x: f"{int(round(x))}"),
                     "" if not cfg.quant else _cells(name, lambda e: e["ppl"], lambda x: _f(x, 2))))
    table = _table(["model", "bits per reading", "training reconstruction", "test reconstruction", "codes used", "perplexity"], rows)
    return ("The recipe of the previous table with the codebook size K changed (initialisation 'data', all else equal) and, in the first row, "
            f"the same network without a quantizer. The sensor noise has variance {_f(NOISE_FLOOR, 4)}. Codes used and perplexity count the 1024 "
            "tokens of the 256 training readings after training.\n\n" + table + "\n\n" + _note(table))


def straight_through_and_beta():
    rows = []
    for label, name in (("straight-through, beta on the commitment term", "main"), ("straight-through, beta on the codebook term", "on_codebook"),
                        ("no straight-through (a plain lookup)", "no_ste")):
        rows.append((label, _cells(name, lambda e: e["final"][1], lambda x: _f(x, 4)), _cells(name, lambda e: e["test"], lambda x: _f(x, 4)),
                     _cells(name, lambda e: e["final"][2], lambda x: _f(x, 4)), _cells(name, lambda e: e["ppl"], lambda x: _f(x, 2))))
    table = _table(["variant", "training reconstruction", "test reconstruction", "mean (z_e - q)^2", "perplexity"], rows)
    return ("The recipe of the first table, K = 16, three ways of training the same network. In the last row the decoder's gradient goes to "
            "the codebook entry it read (the lookup is differentiable in the codebook), the encoder is trained by the commitment term alone, "
            "and the loss is otherwise the same.\n\n" + table + "\n\n" + _note(table))


def dead_codes():
    rows = []
    for label, name in (("encoder rows (initialisation 'data')", "k64"), (f"uniform in (-{WIDE:g}, {WIDE:g}) ('wide')", "wide"),
                        ("'wide', and dead codes revived each epoch", "wide_reset")):
        rows.append((label, _cells(name, lambda e: e["log"][5][4], lambda x: f"{int(round(x))}"), _cells(name, lambda e: e["used"], lambda x: f"{int(round(x))}"),
                     _cells(name, lambda e: e["ppl"], lambda x: _f(x, 2)), _cells(name, lambda e: e["test"], lambda x: _f(x, 4))))
    table = _table(["codebook of 64 starts as", "used in epoch 5", "used at the end", "perplexity", "test reconstruction"], rows)
    return (f"K = 64, the recipe of the first table. 'wide' draws both coordinates of every code as {WIDE:g} * (2 u - 1) with u from `uniform()`, "
            f"code by code, from {_seed(CODES)}. The revival rule (stream {_seed(RESET)}): at the end of every epoch but the last, each code "
            "that none of the epoch's 1024 tokens used becomes the encoder row of a reading chosen with `integer(256)` and a position chosen "
            "with `integer(4)` (in that order, dead codes in index order), and its Adam moments are set to 0.\n\n" + table + "\n\n" + _note(table))


# ------------------------------------------------------------------ a sequence model over the tokens

ALPHA = 1.0                                # Laplace pseudo-count of the count tables


def _fit_counts(seqs, K):
    """Tables of the chain p(t_0) p(t_1 | t_0) p(t_2 | t_1) p(t_3 | t_2) with every count starting at ALPHA, and of the position
    marginals p(t_i) alone."""
    first = [ALPHA] * K
    trans = [[[ALPHA] * K for _ in range(K)] for _ in range(T - 1)]
    marg = [[ALPHA] * K for _ in range(T)]
    for s in seqs:
        first[s[0]] += 1
        for i in range(T):
            marg[i][s[i]] += 1
        for i in range(T - 1):
            trans[i][s[i]][s[i + 1]] += 1

    def norm(v):
        z = sum(v)
        return [a / z for a in v]
    return norm(first), [[norm(r) for r in m] for m in trans], [norm(m) for m in marg]


def _bits(s, first, trans, marg, chain):
    if chain:
        b = -math.log2(first[s[0]])
        for i in range(T - 1):
            b -= math.log2(trans[i][s[i]][s[i + 1]])
        return b
    return -sum(math.log2(marg[i][s[i]]) for i in range(T))


def _draw(gen, p):
    """Index k of the first code whose running total of p exceeds u = gen.uniform() (the last code if rounding leaves none)."""
    u, acc = gen.uniform(), 0.0
    for k, x in enumerate(p):
        acc += x
        if u < acc:
            return k
    return len(p) - 1


def _sample_stats(run_cfg, th, tables, kind):
    """(fraction of the N_SAMPLES decoded samples whose nearest point of the curve is under 0.03 rms away per sensor, under 0.1)
    for tokens drawn as 'chain', 'independent' or 'uniform'."""
    first, trans, marg = tables
    K = run_cfg.K
    C = unpack(th, run_cfg)[8]
    gen = bookrng.generator(f"{SAMPLE}.{kind}")
    near = far = 0
    for _ in range(N_SAMPLES):
        if kind == "chain":
            s = [_draw(gen, first)]
            for i in range(T - 1):
                s.append(_draw(gen, trans[i][s[-1]]))
        elif kind == "independent":
            s = [_draw(gen, marg[i]) for i in range(T)]
        else:
            s = [gen.integer(K) for _ in range(T)]
        zq = []
        for k in s:
            zq.extend(C[k])
        miss = vae.nearest_on_curve(decode_flat(th, run_cfg, zq)[0])[0]
        near += miss < 0.03
        far += miss < vae.OFF_CURVE
    return near / N_SAMPLES, far / N_SAMPLES


@lru_cache(maxsize=None)
def sequence_eval(i):
    cfg, th = RECIPES["main"], all_runs()["main"][i]["theta"]
    tr, _, te, _ = fm.curved_data()
    ttr, tte = tokens(cfg, th, tr), tokens(cfg, th, te)
    tables = _fit_counts(ttr, cfg.K)
    out = {"chain_test": sum(_bits(s, *tables, True) for s in tte) / len(tte), "indep_test": sum(_bits(s, *tables, False) for s in tte) / len(tte)}
    for kind in ("chain", "independent", "uniform"):
        out[kind] = _sample_stats(cfg, th, tables, kind)
    return out


def sequence_model():
    cfg = RECIPES["main"]
    bits = [("uniform over 16^4 sequences", _f(T * math.log2(cfg.K), 2)),
            ("the four positions independent", _cells_seq(lambda e: e["indep_test"], 2)),
            ("a chain, each token given the one before", _cells_seq(lambda e: e["chain_test"], 2))]
    t1 = _table(["model of the four tokens", "bits per test reading"], bits)
    rows = []
    for label, kind in (("tokens drawn uniformly", "uniform"), ("positions drawn independently", "independent"), ("chain", "chain")):
        rows.append((label, _cells_seq(lambda e: e[kind][0], 3), _cells_seq(lambda e: e[kind][1], 3)))
    t2 = _table(["token model", "decoded within 0.03 of the curve", f"within {vae.OFF_CURVE:g}"], rows)
    return ("The tokenizer of the first table (K = 16). Counts: every entry of the tables starts at 1 (Laplace), the 256 training readings add their "
            "tokens; the chain has one table for the first token and one 16 x 16 table for each of the three steps, the independent model one "
            "table per position. Bits are the mean of -log2 of the model's probability of the 4 tokens of each of the 128 test readings.\n\n"
            + t1 + "\n\n" + f"Then {N_SAMPLES} readings are sampled from each model and decoded; the distance is the root-mean-square miss per sensor "
            "of the nearest point of the noiseless curve (`vae.nearest_on_curve`). Draws: for each sample, one `uniform()` per token walked "
            f"through the cumulative probabilities in code order (a token is the first code whose running total exceeds the draw), or one "
            f"`integer(16)` per token for the uniform model, from {_seed(SAMPLE + '.chain')}, {_seed(SAMPLE + '.independent')} and "
            f"{_seed(SAMPLE + '.uniform')}.\n\n" + t2 + "\n\n" + _note(t1 + t2))


def _cells_seq(getter, d):
    vals = [getter(sequence_eval(i)) for i in range(5)]
    return stable.text(vals[0], vals[1:], lambda x: _f(x, d))


# ------------------------------------------------------------------ what a squared error prefers

def _mean_picture(a, b):
    return [[(x + y) / 2.0 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _l1(a, b):
    return sum(abs(a[r][c] - b[r][c]) for r in range(clip.PN) for c in range(clip.PN)) / (clip.PN * clip.PN)


def _fd(a, b):
    return sum(clip.toy_distance(a, b))


def blur_and_feature_distance():
    a = clip.base_picture()
    b = clip.shift_right(a, 1)
    mean = _mean_picture(a, b)
    rows = []
    for label, p in (("picture A", a), ("picture B", b), ("the mean of A and B", mean)):
        vals = (0.5 * clip.pixel_mse(p, a) + 0.5 * clip.pixel_mse(p, b), 0.5 * _l1(p, a) + 0.5 * _l1(p, b), 0.5 * _fd(p, a) + 0.5 * _fd(p, b))
        rows.append((label, *[_f(v, 4) for v in vals]))
    return ("The reference picture A of `clip.base_picture` (12 x 12, a vertical grating in the top six rows and a horizontal one below) and "
            "picture B, A shifted right by one pixel with wraparound (the top half inverts, the bottom half is unchanged). The code cannot "
            "tell which of the two was encoded, so each is the truth with probability 1/2. Each row is a candidate output; the cells are the "
            "expected loss, 1/2 loss(P, A) + 1/2 loss(P, B), under the pixel mean squared error, the pixel mean absolute error and the toy "
            "feature distance of `clip.toy_distance` (both taps summed).\n\n"
            + _table(["decoder output P", "expected squared error", "expected absolute error", "expected feature distance"], rows))


# ------------------------------------------------------------------ what the codebook search costs

def search_cost():
    rows = []
    for name, side, mult, K, d in PRESETS:
        grid = side // 2 ** (len(mult) - 1)
        M = grid * grid
        ops = 3 * M * K * d                       # one subtraction, one multiplication and one addition per entry of every distance
        table_bytes = 4 * K * d                   # the codebook in 32-bit floats
        rows.append((name, f"{M}", f"{K:,}", d, f"{ops:,}", f"{K * d:,}", f"{table_bytes:,}", f"{M * table_bytes:,}"))
    return ("The kernel sketched in the chapter evaluates, for each of the M positions, the squared distance to each of the K codes of width d: "
            "per entry one subtraction, one multiplication and one addition. Each position reads the whole table once; 32-bit floats.\n\n"
            + _table(["tokenizer", "positions M", "codes K", "width d", "operations of the search 3 M K d", "codebook entries K d", "codebook bytes",
                      "bytes requested by all M positions"], rows))


# ------------------------------------------------------------------ data of the figures

def latent_map_data():
    """(test encoder rows [(position, x, y)], codebook [(x, y)]) of the base run of the main recipe."""
    cfg, th = RECIPES["main"], all_runs()["main"][0]["theta"]
    _, _, te, _ = fm.curved_data()
    pts = []
    for x in te:
        for t, row in enumerate(encode_rows(th, cfg, x)[0]):
            pts.append((t, row[0], row[1]))
    return pts, [tuple(c) for c in unpack(th, cfg)[8]]
