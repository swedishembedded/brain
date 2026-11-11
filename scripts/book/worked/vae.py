# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Variational Autoencoders".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: vae.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

The data are the curved sensor readings of Volume III, chapter 13 (`first_models.curved_data`): sixteen sensors, one hidden cause
s uniform on [-1, 1), readings x = f(s) + 0.05 n. They are a latent-variable model whose truth is known, so every quantity of the
chapter that is an integral over the latent (the marginal likelihood, the exact evidence lower bound, the gap between them) is a
one-dimensional quadrature that this module carries out, on a grid, rather than estimates.

* an autoencoder whose code is rescaled: reconstruction does not change, what a random code decodes to does;
* the evidence lower bound in closed form for a linear Gaussian model, and its gap as a KL divergence;
* the reparameterization trick against the score-function estimator, on a draw sequence of 1000;
* a variational autoencoder 16-12-(mean, log-variance)-12-16 with a one-number latent, its hand-written backward pass, its
  gradient check, its training log, and the exact test-set bound, marginal likelihood and true likelihood;
* where the KL term goes, generation from the prior against the plain autoencoder, interpolation, and the rescaling of the code
  under the bound;
* the boundary between a trained autoencoder and a denoiser in brain: the latent packing, compression ratios, pooled
  statistics of a tiled decode and the decoder's byte budget.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.
Trained numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the
nudges of `stable.py`. The five runs of each model are trained in at most four worker processes.

Swedish Embedded AB implements solutions for generative models whose bound on likelihood is computed, checked and trusted. If your
team needs expertise in variational autoencoders and latent image models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache
from operator import mul

import bookrng
import first_models as fm
import stable

# ------------------------------------------------------------------ constants

D = fm.D                                   # sensors
HID = 12                                   # hidden width of encoder and decoder
SIGMA_X = fm.SENSOR_NOISE                  # the decoder's fixed standard deviation, the true sensor noise
LOG_NORM = 0.5 * D * math.log(2.0 * math.pi * SIGMA_X ** 2)     # (D/2) log(2 pi sigma_x^2)
EPOCHS, BATCH, PEAK = 800, 32, 3e-2        # 8 steps per epoch, 6400 steps; warmup 640 steps, cosine to PEAK / 100
INIT, BATCHES, NOISE = "vae.init", "vae.batches", "vae.noise"
PRIOR_DRAWS, N_DRAWS = "vae.prior", 300
REPARAM = "vae.reparam"
GRADCHECK = "vae.gradcheck"
LOG_EPOCHS = (1, 5, 25, 100, 300, 800)
STEP = 0.002                               # quadrature step of the latent grid
ZLO, ZHI = -6.0, 6.0
GRID = [ZLO + STEP * k for k in range(int(round((ZHI - ZLO) / STEP)) + 1)]          # 6001 points
SGRID_STEP, SLIM = 0.005, 1.25                                                       # nearest-point search on the curve
S_GRID = [-SLIM + SGRID_STEP * k for k in range(int(round(2 * SLIM / SGRID_STEP)) + 1)]
OFF_CURVE = 0.1                            # root-mean-square miss, per sensor, beyond which a sample is off the curve
RESCALES = (0.25, 1.0, 4.0)

N_PARAMS = HID * D + HID + 2 * HID + 2 + HID + HID + D * HID + D        # 462


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


def _pct(x):
    return f"{100.0 * x:.1f}%"


def _lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(a - m) for a in v))


def gelu(z):
    return fm.gelu(z)


def curve(s):
    """The noiseless reading f(s) = s pat1 + 1.5 (s^2 - 1/3) pat2 of the curved data."""
    return [s * fm.PATTERNS[0][j] + fm.CURVE_QUAD * (s * s - 1.0 / 3.0) * fm.PATTERNS[1][j] for j in range(D)]


S_CURVE = [curve(s) for s in S_GRID]


def nearest_on_curve(x):
    """(root-mean-square miss per sensor, s) of the point of the curve nearest to x, searched on the grid S_GRID."""
    best, arg = None, 0.0
    for s, c in zip(S_GRID, S_CURVE):
        d = 0.0
        for a, b in zip(x, c):
            d += (a - b) * (a - b)
        if best is None or d < best:
            best, arg = d, s
    return math.sqrt(best / D), arg


# ------------------------------------------------------------------ the network

def unpack(th):
    """Slices of the flat vector: W1 [12,16], b1, W2 [2,12] (row 0 the mean, row 1 the log-variance), b2, W3 [12] (1 -> 12), b3,
    W4 [16,12], b4, in this order, each matrix row-major."""
    o = 0

    def take(n):
        nonlocal o
        r = th[o:o + n]
        o += n
        return r
    W1 = [take(D) for _ in range(HID)]
    b1 = take(HID)
    W2 = [take(HID) for _ in range(2)]
    b2 = take(2)
    W3 = take(HID)
    b3 = take(HID)
    W4 = [take(HID) for _ in range(D)]
    b4 = take(D)
    return W1, b1, W2, b2, W3, b3, W4, b4


def init_theta(gen):
    """Weights N(0, 1/n_in) layer by layer in row-major order (layers 16->12, 12->2, 1->12, 12->16), biases 0. Draws: weights only."""
    th = []
    for n_in, n_out in ((D, HID), (HID, 2), (1, HID), (HID, D)):
        s = 1.0 / math.sqrt(n_in)
        th.extend(gen.normal() * s for _ in range(n_out * n_in))
        th.extend([0.0] * n_out)
    return th


def encode(th, x):
    """(mean, log-variance) of the encoder distribution of one reading."""
    W1, b1, W2, b2, *_ = unpack(th)
    a1 = [gelu(sum(map(mul, w, x)) + b) for w, b in zip(W1, b1)]
    return sum(map(mul, W2[0], a1)) + b2[0], sum(map(mul, W2[1], a1)) + b2[1]


def decode(th, z):
    """The decoder's mean reading for the latent z."""
    *_, W3, b3, W4, b4 = unpack(th)
    a3 = [gelu(W3[j] * z + b3[j]) for j in range(HID)]
    return [sum(map(mul, w, a3)) + b for w, b in zip(W4, b4)]


def loss_grad(th, X, xi, vae):
    """(mean loss, mean reconstruction term, mean KL term, gradient) over the examples X.

    vae=True: z = m + exp(v/2) xi with the given draws xi, loss = ||x - g(z)||^2/(2 s^2) + (D/2) log(2 pi s^2) + KL.
    vae=False: z = m, loss = ||x - g(z)||^2/(2 s^2) + (D/2) log(2 pi s^2), the plain autoencoder (row 1 of W2 gets no gradient).
    Examples are accumulated in order; the gradient is divided by the number of examples at the end."""
    W1, b1, W2, b2, W3, b3, W4, b4 = unpack(th)
    B = len(X)
    gW1 = [[0.0] * D for _ in range(HID)]
    gb1 = [0.0] * HID
    gW2 = [[0.0] * HID for _ in range(2)]
    gb2 = [0.0, 0.0]
    gW3, gb3 = [0.0] * HID, [0.0] * HID
    gW4 = [[0.0] * HID for _ in range(D)]
    gb4 = [0.0] * D
    rec_sum = kl_sum = 0.0
    for n, x in enumerate(X):
        p1 = [sum(map(mul, w, x)) + b for w, b in zip(W1, b1)]
        a1 = [gelu(v) for v in p1]
        m = sum(map(mul, W2[0], a1)) + b2[0]
        v = sum(map(mul, W2[1], a1)) + b2[1]
        if vae:
            s = math.exp(0.5 * v)
            z = m + s * xi[n]
        else:
            z = m
        u = [W3[j] * z + b3[j] for j in range(HID)]
        a3 = [gelu(w) for w in u]
        xh = [sum(map(mul, w, a3)) + b for w, b in zip(W4, b4)]
        rec_sum += sum((a - b) ** 2 for a, b in zip(xh, x)) / (2.0 * SIGMA_X ** 2)
        if vae:
            kl_sum += 0.5 * (m * m + math.exp(v) - 1.0 - v)
        dxh = [(a - b) / SIGMA_X ** 2 for a, b in zip(xh, x)]
        da3 = [sum(dxh[k] * W4[k][j] for k in range(D)) for j in range(HID)]
        for k in range(D):
            gb4[k] += dxh[k]
            for j in range(HID):
                gW4[k][j] += dxh[k] * a3[j]
        du = [da3[j] * fm.gelu_slope(u[j]) for j in range(HID)]
        for j in range(HID):
            gW3[j] += du[j] * z
            gb3[j] += du[j]
        dz = sum(du[j] * W3[j] for j in range(HID))
        if vae:
            dm = dz + m
            dv = dz * xi[n] * s * 0.5 + 0.5 * (math.exp(v) - 1.0)
        else:
            dm, dv = dz, 0.0
        gb2[0] += dm
        gb2[1] += dv
        for j in range(HID):
            gW2[0][j] += dm * a1[j]
            gW2[1][j] += dv * a1[j]
        dp1 = [(W2[0][j] * dm + W2[1][j] * dv) * fm.gelu_slope(p1[j]) for j in range(HID)]
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
    flat += gb2 + gW3 + gb3
    for r in gW4:
        flat += r
    flat += gb4
    return (rec_sum + kl_sum) / B + LOG_NORM, rec_sum / B + LOG_NORM, kl_sum / B, [g / B for g in flat]


# ------------------------------------------------------------------ training

def train(vae, nudge=0.0):
    """Train for EPOCHS epochs; returns {'theta', 'log'} where log[epoch] = (mean loss, reconstruction, KL) over the epoch's 8
    batches. Per epoch one shuffle of 0 .. 255 from the stream VAE_BATCHES; per step, for the vae, one normal per example in batch
    order from the stream NOISE. Adam (0.9, 0.999, 1e-8), no weight decay, no clipping, warmup of a tenth of the steps then a cosine
    from PEAK to PEAK / 100. The first weight starts `nudge` away (the stable-digits rule)."""
    tr, _, _, _ = fm.curved_data()
    th = init_theta(bookrng.generator(INIT))
    th[0] += nudge
    m, v = [0.0] * N_PARAMS, [0.0] * N_PARAMS
    shuf, noise = bookrng.generator(BATCHES), bookrng.generator(NOISE)
    order = list(range(len(tr)))
    per_epoch = len(tr) // BATCH
    total = EPOCHS * per_epoch
    warmup = total // 10
    log, t = {}, 0
    for epoch in range(1, EPOCHS + 1):
        shuf.shuffle(order)
        acc = [0.0, 0.0, 0.0]
        for k in range(per_epoch):
            rows = order[k * BATCH:(k + 1) * BATCH]
            xb = [tr[i] for i in rows]
            xi = [noise.normal() for _ in rows] if vae else None
            loss, rec, kl, g = loss_grad(th, xb, xi, vae)
            acc[0] += loss
            acc[1] += rec
            acc[2] += kl
            fm.adamw_step(th, g, m, v, t + 1, fm.schedule_rate(t, PEAK, PEAK / 100.0, warmup, total), 0.0)
            t += 1
        if epoch in LOG_EPOCHS:
            log[epoch] = tuple(a / per_epoch for a in acc)
    return {"theta": th, "log": log}


def _train_job(job):
    return train(*job)


@lru_cache(maxsize=None)
def base_run(vae):
    return train(vae, 0.0)


@lru_cache(maxsize=None)
def nudged_runs(vae):
    """The four copies of the run whose first weight starts at each of stable.NUDGES, trained in four worker processes."""
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_train_job, [(vae, d) for d in stable.NUDGES])


def all_runs(vae):
    return [base_run(vae), *nudged_runs(vae)]


def run_i(vae, i):
    """Run i of all_runs, training the nudged copies only when one of them is asked for."""
    return base_run(vae) if i == 0 else nudged_runs(vae)[i - 1]


# ------------------------------------------------------------------ evaluation

def sample_stats(th, scale):
    """For N_DRAWS draws z ~ N(0, 1) (stream PRIOR_DRAWS, in order) decoded by the model whose code is multiplied by `scale`
    (decoder input z / scale): the share more than OFF_CURVE off the curve, the share whose nearest point of the curve has
    |s| >= 0.5 (the data's own share is 1/2), and the mean miss."""
    g = bookrng.generator(PRIOR_DRAWS)
    off = outer = 0
    tot = 0.0
    for _ in range(N_DRAWS):
        miss, s = nearest_on_curve(decode(th, g.normal() / scale))
        off += miss > OFF_CURVE
        outer += abs(s) >= 0.5
        tot += miss
    return off / N_DRAWS, outer / N_DRAWS, tot / N_DRAWS


@lru_cache(maxsize=None)
def _ae_eval(key):
    """key = (vae?, index of the run in all_runs): code statistics and test error of a trained model, cached."""
    vae, i = key
    th = run_i(vae, i)["theta"]
    tr, _, te, _ = fm.curved_data()
    codes = [encode(th, x)[0] for x in tr]
    mean = sum(codes) / len(codes)
    sd = math.sqrt(sum((c - mean) ** 2 for c in codes) / len(codes))
    mse = sum((a - b) ** 2 for x in te for a, b in zip(decode(th, encode(th, x)[0]), x)) / (len(te) * D)
    return {"min": min(codes), "max": max(codes), "mean": mean, "sd": sd, "mse": mse}


NOTE = ("Every trained number is printed with the digits that the run and four copies of it, started with the first weight changed by +1e-12, "
        "-1e-12, +3e-12 and -3e-12, share (the rule of `stable.py`); here all five agree on every digit shown.")


ABOUT = ("'About x' means x rounded to three, two or one significant digits, as many as the five values round to alike while lying within half a "
         "unit of that last digit, so 'about 0.003' stands for a value anywhere from 0.0025 to 0.0035.")


def _note(table):
    """The reproducibility note of a table of trained numbers: short when every cell is in full, the full rule when some digits did not survive."""
    return stable.RULE + " " + ABOUT if ("about " in table or "unstable" in table) else NOTE


def _cell(vals, fmt):
    """The digits of vals[0] that survive in all runs (stable.text)."""
    return stable.text(vals[0], vals[1:], fmt)


def _stable_all(vae, getter, fmt):
    return _cell([getter(i) for i in range(5)], fmt)


# ------------------------------------------------------------------ the opening: a rescaled code

def rescaled_codes():
    rows = []
    for c in RESCALES:

        def ev(i, c=c):
            e = _ae_eval((False, i))
            sa = sample_stats(all_runs(False)[i]["theta"], c)
            return e, sa
        evs = [ev(i) for i in range(5)]
        lo = _cell([e["min"] * c for e, _ in evs], lambda x: _f(x, 2))
        hi = _cell([e["max"] * c for e, _ in evs], lambda x: _f(x, 2))
        mse = _cell([e["mse"] for e, _ in evs], lambda x: _f(x, 5))
        off = _cell([sa[0] for _, sa in evs], _pct)
        outer = _cell([sa[1] for _, sa in evs], _pct)
        cells = [f"{c:g}", mse, f"[{lo}, {hi}]", off, outer]
        rows.append(cells)
    head = (f"The plain autoencoder 16-12-1-12-16 of this chapter's training recipe (below), trained on the 256 training readings of the "
            f"curved data. Row c: the code multiplied by c and the decoder's input divided by c, which changes no output. Test error: mean "
            f"squared error per entry on the 128 test readings. The code range is that of the 256 training readings. Draws: {N_DRAWS} values "
            f"z ~ N(0, 1) from the generator {_seed(PRIOR_DRAWS)}, drawn in order and used for every row, decoded as g(z / c). A decoded point is "
            f"off the curve when its root-mean-square miss per sensor from the nearest point f(s) exceeds {OFF_CURVE} (the sensor noise is 0.05); "
            f"the nearest s is searched on a grid of step {SGRID_STEP} over [-{SLIM}, {SLIM}]. The data's own share with |s| >= 0.5 is 50%.\n\n")
    body = head + _table(["c", "test error", "range of the codes", "draws off the curve", "draws with |s| >= 0.5"], rows)
    return body + "\n\n" + _note(body)


# ------------------------------------------------------------------ the bound in closed form

def _gauss_logpdf(x, mean, var):
    return -0.5 * math.log(2.0 * math.pi * var) - (x - mean) ** 2 / (2.0 * var)


def _kl_gauss(m1, v1, m2, v2):
    """KL( N(m1, v1) || N(m2, v2) ) in nats."""
    return 0.5 * (math.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)


def linear_gaussian_bound():
    w, sig, x = 2.0, 0.5, 1.5
    ev_var = w * w + sig * sig
    log_evidence = _gauss_logpdf(x, 0.0, ev_var)
    pm, pv = w * x / ev_var, sig * sig / ev_var
    cases = [("the prior", 0.0, 1.0), ("the exact posterior", pm, math.sqrt(pv)), ("posterior mean, std 0.5", pm, 0.5),
             ("posterior mean, std 0.1", pm, 0.1), ("mean 0.4, posterior std", 0.4, math.sqrt(pv))]
    rows = []
    for name, m, s in cases:
        # E_q log p(x | z) for p(x | z) = N(x; w z, sig^2) and z ~ N(m, s^2)
        exp_ll = -0.5 * math.log(2.0 * math.pi * sig * sig) - ((x - w * m) ** 2 + w * w * s * s) / (2.0 * sig * sig)
        kl_prior = _kl_gauss(m, s * s, 0.0, 1.0)
        elbo = exp_ll - kl_prior
        kl_post = _kl_gauss(m, s * s, pm, pv)
        rows.append([name, _f(m), _f(s), _f(exp_ll), _f(kl_prior), _f(elbo), _f(kl_post), _f(log_evidence - elbo)])
    head = (f"Model: z ~ N(0, 1), x | z ~ N(2 z, 0.5^2), observed x = 1.5. The evidence is N(1.5; 0, 2^2 + 0.5^2), so log p(x) = {_f(log_evidence, 6)}; "
            f"the posterior is N({_f(pm, 6)}, {_f(pv, 6)}). Each row is a distribution q = N(m, std^2) for z. Expected log-likelihood and KL to the "
            f"prior are the closed forms of the chapter, KL to the posterior is the closed Gaussian KL, and the last column is log p(x) minus the bound.\n\n")
    return head + _table(["q", "m", "std", "E_q log p(x|z)", "KL(q || prior)", "bound", "KL(q || posterior)", "log p(x) - bound"], rows)


# ------------------------------------------------------------------ the reparameterization trick

def reparam_gradients():
    gen = bookrng.generator(REPARAM)
    xis = [gen.normal() for _ in range(1000)]
    rows = []
    for mu, sig in ((1.0, 0.5), (5.0, 0.5)):
        zs = [mu + sig * xi for xi in xis]
        est = {
            "reparameterized": ([2.0 * z for z in zs], [2.0 * z * xi for z, xi in zip(zs, xis)]),
            "score function": ([z * z * xi / sig for z, xi in zip(zs, xis)], [z * z * (xi * xi - 1.0) / sig for z, xi in zip(zs, xis)]),
        }
        for name, (gm_, gs) in est.items():
            row = [f"{mu:g}", f"{sig:g}", name]
            for exact, g in ((2.0 * mu, gm_), (2.0 * sig, gs)):
                mean = sum(g) / len(g)
                sd = math.sqrt(sum((a - mean) ** 2 for a in g) / len(g))
                row += [_f(exact, 3), _f(mean, 3), _f(sd, 3)]
            rows.append(row)
    head = (f"f(z) = z^2 and z ~ N(mu, sigma^2), so E f = mu^2 + sigma^2 and the exact gradients are 2 mu and 2 sigma. One sequence of 1000 draws xi ~ N(0, 1) "
            f"from {_seed(REPARAM)} (normal calls in order, Box-Muller with the held sine partner) serves every row; z = mu + sigma xi. Reparameterized: the "
            f"per-draw gradients 2 z (for mu) and 2 z xi (for sigma). Score function: f(z) (z - mu)/sigma^2 = z^2 xi/sigma and f(z) ((z - mu)^2/sigma^3 - 1/sigma) = "
            f"z^2 (xi^2 - 1)/sigma. Means and standard deviations (divisor 1000) are over the 1000 per-draw values, summed in draw order.\n\n")
    return head + _table(["mu", "sigma", "estimator", "d/dmu exact", "mean", "std", "d/dsigma exact", "mean", "std"], rows,
                         align="rrlrrrrrr")


# ------------------------------------------------------------------ the gradient check

def vae_gradcheck():
    tr, _, _, _ = fm.curved_data()
    gen = bookrng.generator(GRADCHECK)
    xi = [gen.normal() for _ in range(4)]
    th0 = init_theta(bookrng.generator(INIT))
    X = tr[:4]
    h, atol, rtol = 1e-5, 1e-6, 1e-6
    rows = []
    for vae, name in ((True, "variational autoencoder"), (False, "plain autoencoder")):
        _, _, _, g = loss_grad(th0, X, xi, vae)
        fails = 0
        for i in range(N_PARAMS):
            tp, tm = list(th0), list(th0)
            tp[i] += h
            tm[i] -= h
            fd = (loss_grad(tp, X, xi, vae)[0] - loss_grad(tm, X, xi, vae)[0]) / (2.0 * h)
            if abs(g[i] - fd) > atol + rtol * max(abs(g[i]), abs(fd)):
                fails += 1
        rows.append([name, N_PARAMS, fails])
    head = (f"Initial parameters (stream {_seed(INIT)}), the first 4 training readings, draws xi from {_seed(GRADCHECK)} (4 normals in order, held fixed "
            f"for every perturbation), central differences with step h = 1e-5 in double precision. An entry passes when |analytic - numeric| <= 1e-6 + 1e-6 max(|analytic|, |numeric|).\n\n")
    return head + _table(["model", "parameters checked", "entries failing the gate"], rows)


# ------------------------------------------------------------------ the training log

def training_log():
    rows = []
    for epoch in LOG_EPOCHS:
        cells = [str(epoch)]
        for k, d in ((0, 3), (1, 3), (2, 3)):
            cells.append(_stable_all(True, lambda i, k=k: all_runs(True)[i]["log"][epoch][k], lambda x, d=d: _f(x, d)))
        rows.append(cells)
    ae = ["autoencoder, epoch 800 (no KL term)"]
    ae.append(_stable_all(False, lambda i: all_runs(False)[i]["log"][EPOCHS][0], lambda x: _f(x, 3)))
    ae += ["", ""]
    steps = EPOCHS * 8
    head = (f"Training readings 0 .. 255 of `first_models.curved_data`, batch 32 (8 steps per epoch, {steps} steps), Adam (0.9, 0.999, 1e-8) with no decay and no clip, "
            f"rate warming up over {steps // 10} steps to {PEAK:g} and following a cosine to {PEAK / 100:g}. Initial parameters from {_seed(INIT)}, one in-place shuffle "
            f"per epoch from {_seed(BATCHES)} of the array 0 .. 255, which is set to the identity once, before epoch 1, and left in the order each epoch leaves it; the noise is one normal per example, batch by batch, from {_seed(NOISE)}. Each cell is the mean over the epoch's 8 batches "
            f"of the single-draw loss, in nats per reading.\n\n")
    table = _table(["epoch", "loss = -bound", "reconstruction", "KL"], rows + [ae])
    body = head + table
    return body + "\n\n" + _note(table)


# ------------------------------------------------------------------ exact quantities by quadrature

@lru_cache(maxsize=None)
def exact_eval(i):
    """Exact (quadrature) quantities of the variational autoencoder run i on the 128 test readings, per reading."""
    th = run_i(True, i)["theta"]
    _, _, te, _ = fm.curved_data()
    gz = [decode(th, z) for z in GRID]
    log_prior = [-0.5 * z * z - 0.5 * math.log(2.0 * math.pi) + math.log(STEP) for z in GRID]
    s_mid = [-1.0 + 0.001 * (k + 0.5) for k in range(2000)]
    fs = [curve(s) for s in s_mid]
    out = []
    qs = []
    for x in te:
        m, v = encode(th, x)
        sd = math.exp(0.5 * v)
        d2 = [sum((a - b) ** 2 for a, b in zip(g, x)) for g in gz]
        q = [math.exp(-0.5 * ((z - m) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi)) * STEP for z in GRID]
        exp_ll = -sum(a * b for a, b in zip(q, d2)) / (2.0 * SIGMA_X ** 2) - LOG_NORM
        kl = 0.5 * (m * m + math.exp(v) - 1.0 - v)
        log_marg = _lse([lp - d / (2.0 * SIGMA_X ** 2) for lp, d in zip(log_prior, d2)]) - LOG_NORM
        d_true = [sum((a - b) ** 2 for a, b in zip(f, x)) for f in fs]
        log_true = _lse([math.log(0.001 / 2.0) - d / (2.0 * SIGMA_X ** 2) for d in d_true]) - LOG_NORM
        out.append({"exp_ll": exp_ll, "kl": kl, "elbo": exp_ll - kl, "log_marg": log_marg, "log_true": log_true, "m": m, "v": v,
                    "qmass": sum(q)})
        qs.append(q)
    return out, qs


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def bound_test():
    def get(key):
        return lambda i: _mean(exact_eval(i)[0], key)

    def get_gap(i):
        rows = exact_eval(i)[0]
        return sum(r["log_marg"] - r["elbo"] for r in rows) / len(rows)

    def get_min_gap(i):
        return min(r["log_marg"] - r["elbo"] for r in exact_eval(i)[0])

    def get_kl_true(i):
        rows = exact_eval(i)[0]
        return sum(r["log_true"] - r["log_marg"] for r in rows) / len(rows)

    fm4 = lambda x: _f(x, 4)  # noqa: E731
    rows = [
        ["E_q log p(x | z), the reconstruction term", _stable_all(True, get("exp_ll"), fm4)],
        ["KL(q || prior)", _stable_all(True, get("kl"), fm4)],
        ["evidence lower bound", _stable_all(True, get("elbo"), fm4)],
        ["log q_theta(x), the model's own log-likelihood", _stable_all(True, get("log_marg"), fm4)],
        ["gap: log q_theta(x) - bound = mean KL(q || posterior)", _stable_all(True, get_gap, fm4)],
        ["log p*(x), the data's true log-density", _stable_all(True, get("log_true"), fm4)],
        ["log p*(x) - log q_theta(x)", _stable_all(True, get_kl_true, fm4)],
        ["smallest single-reading gap (never below 0)", _stable_all(True, get_min_gap, lambda x: _f(x, 4))],
    ]
    head = (f"The 128 test readings (`first_models.curved_data`, readings 384 .. 511 of the stream), the variational autoencoder after epoch {EPOCHS}. All "
            f"integrals over z are sums on the grid z = -6, -5.998, ..., 6 (step {STEP}, 6001 points): E_q by the grid weights of q(z | x), q_theta(x) by "
            f"the log-sum-exp of log p(z) + log p(x | z) + log {STEP}, and p*(x) = (1/2) integral over s of N(x; f(s), 0.05^2 I) by the midpoint rule with step 0.001 on "
            f"[-1, 1). Values are nats per reading, means over the 128 readings.\n\n")
    body = head + _table(["quantity", "value"], rows)
    return body + "\n\n" + _note(body)


def where_kl_goes():
    def parts(i):
        rows, qs = exact_eval(i)
        n = len(rows)
        agg = [sum(q[k] for q in qs) / n for k in range(len(GRID))]
        log_agg = [math.log(a) if a > 0.0 else None for a in agg]
        log_prior = [-0.5 * z * z - 0.5 * math.log(2.0 * math.pi) + math.log(STEP) for z in GRID]
        kl_agg = sum(a * (la - lp) for a, la, lp in zip(agg, log_agg, log_prior) if la is not None)
        info = 0.0
        for q in qs:
            for qk, la in zip(q, log_agg):
                if qk > 0.0 and la is not None:
                    info += qk * (math.log(qk) - la)
        info /= n
        return _mean(rows, "kl"), kl_agg, info

    fm4 = lambda x: _f(x, 4)  # noqa: E731
    row = [_stable_all(True, lambda i: parts(i)[0], fm4), _stable_all(True, lambda i: parts(i)[1], fm4),
           _stable_all(True, lambda i: parts(i)[2], fm4), _stable_all(True, lambda i: parts(i)[1] + parts(i)[2], fm4)]

    def moments(i):
        rows = exact_eval(i)[0]
        n = len(rows)
        mean = sum(r["m"] for r in rows) / n
        var = sum((r["m"] - mean) ** 2 for r in rows) / n + sum(math.exp(r["v"]) for r in rows) / n
        return mean, var, sum(math.exp(r["v"]) for r in rows) / n
    mrow = [_stable_all(True, lambda i: moments(i)[0], fm4), _stable_all(True, lambda i: moments(i)[1], fm4),
            _stable_all(True, lambda i: moments(i)[2], lambda x: _f(x, 5))]
    head = ("The same 128 test readings and the same grid. The aggregate posterior is the equal mixture of the 128 encoder distributions q(z | x_n), its density "
            "taken on the grid; KL(aggregate || prior) and the mutual information I = mean over n of KL(q(z | x_n) || aggregate) are sums over the grid.\n\n")
    t1 = _table(["mean KL(q || prior)", "KL(aggregate || prior)", "I", "KL(aggregate || prior) + I"], [row])
    t2 = _table(["mean of the encoder means", "variance of the aggregate posterior", "mean encoder variance e^v"], [mrow])
    body = head + t1 + "\n\n" + t2
    return body + "\n\n" + _note(t1 + t2)


# ------------------------------------------------------------------ generation

def generation():
    def row(name, vae, scale=1.0):
        def stats(i):
            return sample_stats(all_runs(vae)[i]["theta"], scale)
        sts = [stats(i) for i in range(5)]
        return [name, _cell([s[0] for s in sts], _pct), _cell([s[1] for s in sts], _pct), _cell([s[2] for s in sts], lambda x: _f(x, 4))]
    rows = [row("plain autoencoder, z ~ N(0, 1)", False), row("variational autoencoder, z ~ N(0, 1)", True)]
    head = (f"{N_DRAWS} draws z ~ N(0, 1) from {_seed(PRIOR_DRAWS)} (the same draws for both models, in order), decoded. Off the curve: root-mean-square miss per "
            f"sensor above {OFF_CURVE}. The share with |s| >= 0.5 is 50% in the data. Miss: the mean root-mean-square miss per sensor to the nearest point of the curve "
            f"(the sensor noise itself is 0.05).\n\n")
    body = head + _table(["model", "draws off the curve", "draws with |s| >= 0.5", "mean miss"], rows)
    return body + "\n\n" + _note(body)


def generation_bins():
    """Where the decoded draws land on the curve: counts of the nearest s in the five bins [-1, -0.6), ..., [0.6, 1), and outside."""
    def counts(vae, i):
        g = bookrng.generator(PRIOR_DRAWS)
        bins, outside = [0] * 5, 0
        for _ in range(N_DRAWS):
            _, s = nearest_on_curve(decode(all_runs(vae)[i]["theta"], g.normal()))
            if -1.0 <= s < 1.0:
                bins[int((s + 1.0) / 0.4)] += 1
            else:
                outside += 1
        return bins + [outside]
    rows = []
    for name, vae in (("plain autoencoder", False), ("variational autoencoder", True)):
        cs = [counts(vae, i) for i in range(5)]
        rows.append([name] + [_cell([c[k] for c in cs], lambda x: str(int(round(x)))) for k in range(6)])
    rows.append(["the data (uniform s)"] + [f"{N_DRAWS / 5:g}"] * 5 + ["0"])
    head = (f"The {N_DRAWS} draws z ~ N(0, 1) of {_seed(PRIOR_DRAWS)}, decoded by each model; counts of the nearest point of the curve in five bins of s of width 0.4, and "
            f"the count that falls outside [-1, 1) (the training data never do). The nearest point is searched on the grid s = -{SLIM} + {SGRID_STEP} k, k = 0 .. {len(S_GRID) - 1}, "
            f"with s computed in 64-bit floats as written and the first minimum of the squared distance kept on a tie; a draw is put in bin floor((s + 1) / 0.4) if -1 <= s < 1, "
            f"so the edges -0.2, 0.2 and 0.6, which are nodes of the grid, go by the float: -0.2 (stored as -0.19999999999999996) and 0.6 (0.6000000000000001) fall in the bin on their right, 0.2 (0.19999999999999996) in the bin on its left.\n\n")
    body = head + _table(["draws from", "[-1, -0.6)", "[-0.6, -0.2)", "[-0.2, 0.2)", "[0.2, 0.6)", "[0.6, 1)", "outside"], rows)
    return body + "\n\n" + _note(body)


# ------------------------------------------------------------------ interpolation

def interpolation():
    _, _, te, s_te = fm.curved_data()
    lo, hi = min(range(len(te)), key=lambda i: s_te[i]), max(range(len(te)), key=lambda i: s_te[i])
    rows = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        row = [f"{t:g}"]
        for vae in (True, False):
            th = base_run(vae)["theta"]
            za, zb = encode(th, te[lo])[0], encode(th, te[hi])[0]
            z = (1.0 - t) * za + t * zb
            miss, s = nearest_on_curve(decode(th, z))
            row += [_f(z, 3), _f(s, 3), _f(miss, 4)]
        rows.append(row)
    head = (f"The test readings with the smallest and the largest hidden cause: reading {lo} (s = {_f(s_te[lo], 4)}) and reading {hi} (s = {_f(s_te[hi], 4)}). "
            f"Each model encodes both to the mean of its encoder distribution (the code, for the plain autoencoder), z(t) = (1 - t) z_a + t z_b is decoded, and the "
            f"nearest point of the curve is searched on a grid of step {SGRID_STEP}. Both columns are from the base run of each model.\n\n")
    return head + _table(["t", "VAE z", "VAE s", "VAE miss", "AE z", "AE s", "AE miss"], rows)


# ------------------------------------------------------------------ rescaling the code under the bound

def elbo_rescale():
    def stats(i):
        rows = exact_eval(i)[0]
        n = len(rows)
        a = sum(r["m"] ** 2 + math.exp(r["v"]) for r in rows) / n
        return rows, a

    def kl_c(i, c):
        rows, _ = stats(i)
        return sum(0.5 * (c * c * (r["m"] ** 2 + math.exp(r["v"])) - 1.0 - math.log(c * c * math.exp(r["v"]))) for r in rows) / len(rows)

    def cstar(i):
        return 1.0 / math.sqrt(stats(i)[1])
    cs = [("0.25", lambda i: 0.25), ("0.5", lambda i: 0.5), ("1", lambda i: 1.0), ("c*", cstar), ("2", lambda i: 2.0), ("4", lambda i: 4.0)]
    rows = []
    for name, cf in cs:
        ell = _stable_all(True, lambda i: _mean(exact_eval(i)[0], "exp_ll"), lambda x: _f(x, 4))
        kl = _stable_all(True, lambda i, cf=cf: kl_c(i, cf(i)), lambda x: _f(x, 4))
        elbo = _stable_all(True, lambda i, cf=cf: _mean(exact_eval(i)[0], "exp_ll") - kl_c(i, cf(i)), lambda x: _f(x, 4))
        if name == "c*":
            cval = _stable_all(True, cstar, lambda x: _f(x, 4))
            label = f"{cval} (c*)"
            sa = ["", ""]
        else:
            label = name
            c = float(name)
            sts = [sample_stats(all_runs(True)[i]["theta"], c) for i in range(5)]
            sa = [_cell([s[0] for s in sts], _pct), _cell([s[1] for s in sts], _pct)]
        rows.append([label, ell, kl, elbo] + sa)
    head = (f"The variational autoencoder of the training log, test readings. Row c: the encoder's mean and standard deviation (e^(v/2)) are multiplied by c and the decoder's input "
            f"divided by c, so the decoder sees exactly what it saw, the reconstruction term does not move, and only the KL term changes: "
            f"KL_c = (1/2)(c^2 (m^2 + e^v) - 1 - log(c^2 e^v)), averaged over the readings. c* = 1/sqrt(mean(m^2 + e^v)) makes the derivative of the mean "
            f"KL_c with respect to c zero. The last two columns repeat the generation test of the chapter, decoding g(z / c) for the {N_DRAWS} draws of {_seed(PRIOR_DRAWS)}.\n\n")
    body = head + _table(["c", "reconstruction term", "KL", "bound", "draws off the curve", "draws with |s| >= 0.5"], rows)
    return body + "\n\n" + _note(body)


# ------------------------------------------------------------------ data for the figures

def latent_map_data():
    """(grid z, nearest s of g(z)) of the base variational autoencoder; the test points (m, s); the prior density on the grid."""
    th = base_run(True)["theta"]
    _, _, te, s_te = fm.curved_data()
    zs = [-4.0 + 0.05 * k for k in range(161)]
    ss = [nearest_on_curve(decode(th, z))[1] for z in zs]
    pts = [(encode(th, x)[0], s) for x, s in zip(te, s_te)]
    dens = [math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) for z in zs]
    return zs, ss, pts, dens


def rescale_figure_data():
    """(log2 c, bound) for c in a geometric range, for the base variational autoencoder; and the plain autoencoder's
    reconstruction-only value E log p(x | z) on the test readings, which does not depend on c."""
    rows = exact_eval(0)[0]
    out = []
    for k in range(-12, 13):
        c = 2.0 ** (k / 4.0)
        kl = sum(0.5 * (c * c * (r["m"] ** 2 + math.exp(r["v"])) - 1.0 - math.log(c * c * math.exp(r["v"]))) for r in rows) / len(rows)
        out.append((k / 4.0, _mean(rows, "exp_ll") - kl))
    ae_level = -D * _ae_eval((False, 0))["mse"] / (2.0 * SIGMA_X ** 2) - LOG_NORM
    return out, ae_level


# ------------------------------------------------------------------ brain: the latent boundary

def pack(mean, c, h, w, bn_mean, bn_var, eps):
    """2x2 pixel-unshuffle then per-channel (z - mean)/sqrt(var + eps): packed channel ci*4 + pi*2 + pj at (i, j) holds
    mean[ci, 2i + pi, 2j + pj]. `mean` is a flat [c, h, w] list."""
    oh, ow = h // 2, w // 2
    out = [0.0] * (4 * c * oh * ow)
    for ci in range(c):
        for pi in range(2):
            for pj in range(2):
                oc = ci * 4 + pi * 2 + pj
                s = math.sqrt(bn_var[oc] + eps)
                for i in range(oh):
                    for j in range(ow):
                        out[(oc * oh + i) * ow + j] = (mean[(ci * h + 2 * i + pi) * w + 2 * j + pj] - bn_mean[oc]) / s
    return out


def unpack_latent(z, c, h, w, bn_mean, bn_var, eps):
    oh, ow = h // 2, w // 2
    out = [0.0] * (c * h * w)
    for ci in range(c):
        for pi in range(2):
            for pj in range(2):
                oc = ci * 4 + pi * 2 + pj
                s = math.sqrt(bn_var[oc] + eps)
                for i in range(oh):
                    for j in range(ow):
                        out[(ci * h + 2 * i + pi) * w + 2 * j + pj] = z[(oc * oh + i) * ow + j] * s + bn_mean[oc]
    return out


def latent_packing():
    # A 1-channel 2x2 latent with four different statistics: each pixel of the patch becomes a channel.
    mean = [1.0, 2.0, 3.0, 4.0]
    bn_mean, bn_var, eps = [0.5, 0.0, -1.0, 2.0], [3.0, 8.0, 0.0, 15.0], 1.0
    z = pack(mean, 1, 2, 2, bn_mean, bn_var, eps)
    back = unpack_latent(z, 1, 2, 2, bn_mean, bn_var, eps)
    rows = [[oc, f"({oc // 2}, {oc % 2})", f"{mean[oc]:g}", f"{bn_mean[oc]:g}", f"{bn_var[oc]:g}", f"{math.sqrt(bn_var[oc] + eps):g}",
             _f(z[oc], 6), f"{back[oc]:g}"] for oc in range(4)]
    t1 = _table(["packed channel", "patch position (row, col)", "value", "mean", "variance", "sqrt(variance + 1)", "packed", "unpacked"], rows)
    # A 1-channel 4x4 latent with the numbers 0 .. 15 and identity statistics: where each number goes.
    m4 = [float(k) for k in range(16)]
    z4 = pack(m4, 1, 4, 4, [0.0] * 4, [1.0] * 4, 0.0)
    grids = []
    for oc in range(4):
        g = z4[oc * 4:(oc + 1) * 4]
        grids.append([oc, f"({oc // 2}, {oc % 2})", f"{g[0]:g} {g[1]:g} / {g[2]:g} {g[3]:g}"])
    t2 = _table(["packed channel", "patch position", "packed 2 x 2 map (top row / bottom row)"], grids, align="llr")
    head1 = ("One channel, a 2 x 2 latent holding 1, 2, 3, 4 (one patch), eps = 1; the four packed channels take the four values of the patch in row-major order and "
             "are normalized by their own statistics. Unpacking multiplies and adds back.\n\n")
    head2 = ("One channel, a 4 x 4 latent holding 0 .. 15 in row-major order, statistics (0, 1) and eps = 0: the four packed channels are the four 2 x 2 maps that take "
             "every second pixel.\n\n")
    return head1 + t1 + "\n\n" + head2 + t2


CONFIGS = (("a 4-channel SDXL-style autoencoder", 4), ("a 16-channel autoencoder", 16), ("a 32-channel autoencoder (packed to 128 channels at half the side)", 32))


def compression_table():
    side = 1024
    pixels = 3 * side * side
    rows = []
    for name, c in CONFIGS:
        n = c * (side // 8) ** 2
        rows.append([name, f"{c}", f"{c} x {side // 8} x {side // 8}", f"{n:,}", f"{pixels / n:g}"])
    head = f"A {side} x {side} RGB image has {pixels:,} numbers; each autoencoder halves the side three times (a factor of 8 per side, 64 positions per latent position).\n\n"
    return head + _table(["autoencoder", "latent channels", "latent shape", "latent numbers", "numbers per latent number"], rows, align="lrrrr")


# ------------------------------------------------------------------ brain: pooled statistics of a tiled decode

def pooled_statistics():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    eps = 1e-6

    def stats(v):
        n = len(v)
        mean = sum(v) / n
        return n, mean, sum((x - mean) ** 2 for x in v) / n
    na, ma, va = stats(a)
    nb, mb, vb = stats(b)
    nw, mw, vw = stats(a + b)
    wsum = na + nb
    mean_p = (na * ma + nb * mb) / wsum
    second_p = (na * (va + ma * ma) + nb * (vb + mb * mb)) / wsum
    var_p = second_p - mean_p * mean_p
    rows = [["tile A", na, _f(ma, 4), _f(va, 4)], ["tile B", nb, _f(mb, 4), _f(vb, 4)],
            ["whole signal", nw, _f(mw, 4), _f(vw, 4)], ["pooled from the tiles", wsum, _f(mean_p, 4), _f(var_p, 4)]]

    def norm(x, m, v):
        return (x - m) / math.sqrt(v + eps)
    rows2 = [[x, _f(norm(x, ma, va), 4), _f(norm(x, mw, vw), 4)] for x in a]
    rows2 += [[x, _f(norm(x, mb, vb), 4), _f(norm(x, mw, vw), 4)] for x in b]
    head = ("A signal of 10 values cut into tile A = (1, 2, 3, 4) and tile B = (3, 6, 9, 12, 15, 18). A norm treats the values it is handed as one group: mean, "
            "variance with divisor n, then (x - mean)/sqrt(variance + 10^-6). The pooled mean is the count-weighted mean; the pooled variance is the weighted "
            "mean of (variance + mean^2) minus the square of the pooled mean.\n\n")
    return (head + _table(["group", "count", "mean", "variance"], rows) + "\n\n" +
            _table(["value", "normalized by its own tile", "normalized by the whole signal"], rows2))


# ------------------------------------------------------------------ brain: the decoder's byte budget

def _conv(ci, co, k):
    return co * ci * k * k + co


def _resnet(ci, co):
    n = 2 * ci + _conv(ci, co, 3) + 2 * co + _conv(co, co, 3)
    return n + (_conv(ci, co, 1) if ci != co else 0)


def _attention(c):
    return 2 * c + 4 * _conv(c, c, 1)


def decoder_parameters(channels=(128, 256, 512, 512), latent=32, out=3, layers=2):
    """Parameters of the decoder described in the chapter: 1x1 mixing conv, 3x3 conv in, mid block (resnet, attention, resnet),
    four up blocks of layers + 1 resnets each (the first three followed by a 3x3 conv after the nearest-neighbour doubling),
    a group norm and a 3x3 conv out."""
    rc = list(reversed(channels))
    n = _conv(latent, latent, 1) + _conv(latent, rc[0], 3) + _resnet(rc[0], rc[0]) + _attention(rc[0]) + _resnet(rc[0], rc[0])
    prev = rc[0]
    for i, c in enumerate(rc):
        for r in range(layers + 1):
            n += _resnet(prev if r == 0 else c, c)
        if i < len(rc) - 1:
            n += _conv(c, c, 3)
        prev = c
    return n + 2 * prev + _conv(prev, out, 3)


def decoder_footprint():
    channels = (128, 256, 512, 512)
    per_pixel = sum(c * 4 // 4 ** j for j, c in enumerate(channels))
    params = decoder_parameters()
    live, scratch = 11, 768 * 2 ** 20
    rows = []
    for side in (512, 1024, 2048):
        px = side * side
        act = live * per_pixel * px
        total = params * 4 + act + scratch
        rows.append([f"{side} x {side}", f"{px:,}", f"{act / 2 ** 30:.2f}", f"{total / 2 ** 30:.2f}"])
    head = (f"Channels (128, 256, 512, 512), so one live 32-bit buffer at every resolution level costs 4 (128 + 256/4 + 512/16 + 512/64) = {per_pixel} bytes per output pixel; "
            f"11 such sets are live at once; the decoder has {params:,} parameters ({params * 4 / 2 ** 20:.1f} MiB in 32-bit floats); a fixed scratch of 768 MiB is added. "
            f"GiB = 2^30 bytes. The parameter count is that of this decoder, input to output: a 1x1 convolution on the 32 latent channels, a 3x3 convolution from 32 to 512 "
            f"channels, a middle block (resnet, attention, resnet) at 512 channels, then four up levels with channels 512, 512, 256, 128 (the channel list reversed), each of "
            f"3 resnets (layers + 1, layers = 2), the first three levels followed by a nearest-neighbour doubling and a 3x3 convolution at the level's channels, and finally a "
            f"group norm and a 3x3 convolution from 128 to 3 channels. A resnet from ci to co channels is group norm (2 ci scale and shift), 3x3 convolution, group norm "
            f"(2 co), 3x3 convolution, plus a 1x1 shortcut convolution when ci differs from co. An attention block is a group norm (2 c) and four 1x1 convolutions (query, key, "
            f"value, output) on c channels, single head. A convolution from ci to co channels with a k x k kernel has co ci k^2 weights and co biases. "
            f"These terms are summed to the total above; nothing else carries parameters.\n\n")
    return head + _table(["output image", "output pixels", "activations (GiB)", "total (GiB)"], rows)
