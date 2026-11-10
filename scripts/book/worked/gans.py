# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Adversarial Training".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: gans.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. The
Adam update is `first_models.adamw_step` (with its weight decay at 0 and beta1 = 0.5), the stable-digits rule is `stable`.

* the best possible discriminator for two Gaussians, in closed form and by quadrature, and the value of the game it leaves;
* the three generator losses (saturating, non-saturating, least squares) as gradients on the discriminator's logit;
* the smallest game with no minimum (min over x, max over y of x y), played simultaneously and alternately;
* a generator and a discriminator on a line: three narrow clusters at -3, 0 and 3. Their hand-written backward passes and the
  gradient check against central differences; two training schedules (one generator step per discriminator step, and twenty)
  followed over 400 discriminator steps; five runs of each, counted for collapse; the generator against a frozen discriminator;
  the generator's start far from the data under the three losses;
* the formulas of a discriminator loss, a generator loss and a feature-matching loss on a few numbers (the least-squares losses
  of `crates/minimaxmusic3/src/discriminator.rs`, written from their definitions);
* the shapes, parameters and multiply-adds of a patch discriminator over a spectrogram and of a convolutional image generator
  and discriminator, counted from the layer rules.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.
Trained numbers are printed with the digits that agree between the run and four copies whose first generator weight is
changed by the nudges of `stable.py`. The runs are trained in at most four worker processes.

Swedish Embedded AB implements solutions for generative models whose adversarial training is specified, checked and made
reproducible. If your team needs expertise in generative adversarial networks and image synthesis, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache

import bookrng
import first_models as fm
import stable

# ------------------------------------------------------------------ constants

MODES = (-3.0, 0.0, 3.0)                  # centres of the three clusters of the data
SD = 0.25                                 # standard deviation of each cluster
NEAR = 3 * SD                             # a sample is 'in' a mode when it is within three standard deviations of its centre
DATA_STD = math.sqrt(sum(m * m for m in MODES) / 3 + SD * SD)
DZ, HG, HD = 2, 16, 16                    # noise entries, hidden units of the generator and of the discriminator
SLOPE = 0.1                               # leaky ReLU slope of the discriminator
BATCH, LR = 32, 2e-3
ADAM_B1 = 0.5                             # beta1 of Adam in both networks (beta2 0.999, epsilon 1e-8, no weight decay)
STEPS, EVERY = 400, 10                    # discriminator steps per run; the generator is read out every EVERY of them
LOG_STEPS = (10, 50, 100, 200, 400)
SCHEDULES = {"one": 1, "twenty": 20}      # generator steps per discriminator step
RUN_NAMES = tuple(f"gans.run{i}" for i in range(1, 6))
EVAL_NAME, EVAL_N = "gans.evaluation", 1000
SETTLED_AFTER = 100                       # the checkpoint counts of the five-run table start after this many discriminator steps
COVERED = 100                             # a mode is covered when it holds at least this many of the 1000 evaluation samples
FROZEN, FROZEN_STEPS = "gans.frozen", (0, 25, 100, 400, 1600)
DARK_NAME, DARK_GEN, DARK_EVAL = "gans.dark", "gans.dark.generator", "gans.dark.evaluation"
DARK_SHIFT, DARK_D_STEPS, DARK_RATE, DARK_STEPS, DARK_LOG = -8.0, 600, 0.05, 100, (0, 25, 100)
GRADCHECK, FD_STEP = "gans.gradcheck", 1e-5


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


def sigmoid(a):
    """The logistic function, evaluated so that neither tail overflows."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    e = math.exp(a)
    return e / (1.0 + e)


def softplus(a):
    """log(1 + e^a) without overflow: -log(1 - sigmoid(a)), and -log(sigmoid(a)) is softplus(-a)."""
    return a + math.log1p(math.exp(-a)) if a > 0 else math.log1p(math.exp(a))


# ------------------------------------------------------------------ the best discriminator for two Gaussians

QUAD_STEP = 0.001


def _quadrature(mu):
    """The Jensen-Shannon divergence in nats of N(mu, 1) and N(0, 1), the value of the game at the best discriminator, and the
    accuracy of that discriminator, by the midpoint rule with step 0.001 on [-14, mu + 14]."""
    lo, hi = -14.0, mu + 14.0
    n = int(round((hi - lo) / QUAD_STEP))
    js = value = 0.0
    for i in range(n):
        x = lo + (i + 0.5) * QUAD_STEP
        p = math.exp(-0.5 * (x - mu) ** 2) / math.sqrt(2.0 * math.pi)
        q = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        a = mu * x - 0.5 * mu * mu                      # log p/q
        value += (p * -softplus(-a) + q * -softplus(a)) * QUAD_STEP
        m = 0.5 * (p + q)
        if p > 0.0 and q > 0.0:
            js += 0.5 * (p * math.log(p / m) + q * math.log(q / m)) * QUAD_STEP
    return js, value, 0.5 * (1.0 + math.erf(mu / 2.0 / math.sqrt(2.0)))


def optimal_discriminator():
    mu = 2.0
    rows = []
    for x in (-2, -1, 0, 1, 2, 3, 4):
        a = mu * x - 0.5 * mu * mu
        rows.append((x, _f(a, 1), _f(sigmoid(a), 4)))
    return ("Data p_data = N(2, 1) and a generator whose output is q = N(0, 1), both densities of one real number, each equally likely "
            "to have produced the number shown. D*(x) = p_data(x) / (p_data(x) + q(x)) = sigmoid(log p_data(x) / q(x)).\n\n"
            + _table(["x", "log p_data(x) / q(x) = 2x - 2", "best discriminator D*(x)"], rows))


def game_value():
    rows, worst = [], 0.0
    for mu in (0.0, 1.0, 2.0, 4.0, 6.0):
        js, value, acc = _quadrature(mu)
        worst = max(worst, abs(value - (2.0 * js - math.log(4.0))))
        rows.append((f"{mu:g}", _f(mu * mu / 2.0, 4), _f(js, 4), _f(value, 4), _f(acc, 4)))
    table = _table(["shift mu", "KL(p_data || q) = mu^2 / 2", "JS(p_data, q)", "game value V(D*, q)", "accuracy of D*"], rows)
    return (f"Data N(mu, 1) against a generator output N(0, 1), for five values of mu. JS is the Jensen-Shannon divergence in nats, "
            f"JS(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m) with m = (p + q) / 2; the game value is V = E_p log D*(x) + E_q log(1 - D*(x)); "
            f"the accuracy is the probability that D* labels a sample correctly when data and generated samples are equally likely (it "
            f"says 'data' when D*(x) > 1/2), which is Phi(mu / 2) for the standard normal distribution function Phi, computed with the "
            f"library erf. JS and V are midpoint sums with step {QUAD_STEP:g} over [-14, mu + 14]. log 2 = {math.log(2.0):.4f} and "
            f"log 4 = {math.log(4.0):.4f}.\n\n{table}\n\n"
            + ("The largest difference between V and 2 JS - log 4 over the five rows is below 1e-9." if worst < 1e-9 else "CHECK FAILED"))


# ------------------------------------------------------------------ the generator's loss as a gradient on the logit

def generator_gradient(kind, l):
    """d(loss)/d(logit) of the generator's loss on a fake whose discriminator logit is l.
    'saturating': log(1 - sigmoid(l)); 'non-saturating': -log sigmoid(l); 'least-squares': (l - 1)^2."""
    if kind == "saturating":
        return -sigmoid(l)
    if kind == "non-saturating":
        return -(1.0 - sigmoid(l))
    return 2.0 * (l - 1.0)


LOGITS = (-8, -4, -2, 0, 2, 4)


def gradient_curves_data():
    ls = [-8.0 + 0.1 * i for i in range(121)]
    return ls, {k: [abs(generator_gradient(k, l)) for l in ls] for k in ("saturating", "non-saturating", "least-squares")}


def generator_gradients():
    rows = []
    for l in LOGITS:
        d = sigmoid(l)
        rows.append((l, _f(d, 5), _f(generator_gradient("saturating", l), 5), _f(generator_gradient("non-saturating", l), 5),
                     _f(generator_gradient("least-squares", l), 1)))
    return ("A fake sample on which the discriminator has logit l, so D = sigmoid(l). Each column is the derivative of the generator's "
            "loss for that sample with respect to l.\n\n"
            + _table(["logit l", "D = sigmoid(l)", "log(1 - D)", "-log D", "(l - 1)^2"], rows))


# ------------------------------------------------------------------ a game with no minimum

ETA, START, T_MAX = 0.1, (1.0, 1.0), 200


def _play(alternating, steps=T_MAX):
    x, y = START
    out = [(x, y)]
    for _ in range(steps):
        if alternating:
            x = x - ETA * y
            y = y + ETA * x
        else:
            x, y = x - ETA * y, y + ETA * x
        out.append((x, y))
    return out


def bilinear_data():
    return _play(False), _play(True)


def bilinear_game():
    sim, alt = bilinear_data()
    rows = []
    for t in (0, 1, 2, 5, 10, 50, 100, 200):
        rs, ra = math.hypot(*sim[t]), math.hypot(*alt[t])
        inv = alt[t][0] ** 2 - ETA * alt[t][0] * alt[t][1] + alt[t][1] ** 2
        rows.append((t, _f(rs, 4), _f(math.sqrt(2.0) * (1.0 + ETA * ETA) ** (t / 2.0), 4), _f(ra, 4), _f(inv, 4)))
    lo = min(math.hypot(*p) for p in alt)
    hi = max(math.hypot(*p) for p in alt)
    return (f"The game: x chooses to make x y small and y to make it large, each by a gradient step of size eta = {ETA:g} on "
            f"its own objective, from (x, y) = (1, 1). Simultaneous: both steps use the old pair. Alternating: x steps first and y "
            f"steps from the new x. Radius = sqrt(x^2 + y^2).\n\n"
            + _table(["step t", "radius, simultaneous", "sqrt(2) (1 + eta^2)^(t/2)", "radius, alternating", "x^2 - eta x y + y^2, alternating"], rows)
            + f"\n\nOver all {T_MAX} alternating steps the radius stays between {lo:.4f} and {hi:.4f}.")


# ------------------------------------------------------------------ the two networks

def n_gen():
    return HG * DZ + HG + HG + 1


def n_disc():
    return HD + HD + HD + 1


def init_net(gen, n_in, n_hid):
    """[W1 (n_hid x n_in, row-major), b1 (n_hid), W2 (n_hid), b2 (1)]: both weight matrices drawn normal() / sqrt(fan_in), W1 first;
    biases 0."""
    w1 = [gen.normal() / math.sqrt(n_in) for _ in range(n_hid * n_in)]
    w2 = [gen.normal() / math.sqrt(n_hid) for _ in range(n_hid)]
    return w1 + [0.0] * n_hid + w2 + [0.0]


def gen_forward(G, z):
    """x = W2 . tanh(W1 z + b1) + b2; returns (x, hidden)."""
    o2 = HG * DZ + HG
    h = [math.tanh(sum(G[i * DZ + j] * z[j] for j in range(DZ)) + G[HG * DZ + i]) for i in range(HG)]
    return sum(G[o2 + i] * h[i] for i in range(HG)) + G[o2 + HG], h


def gen_backward(G, z, h, dx, grad):
    """Adds the gradient of the loss to grad given dx = d loss / d x for one sample."""
    o2 = HG * DZ + HG
    grad[o2 + HG] += dx
    for i in range(HG):
        grad[o2 + i] += dx * h[i]
        da = dx * G[o2 + i] * (1.0 - h[i] * h[i])
        grad[HG * DZ + i] += da
        for j in range(DZ):
            grad[i * DZ + j] += da * z[j]


def disc_forward(D, x):
    """logit = W2 . leaky_relu(W1 x + b1) + b2 with slope 0.1 below zero; returns (logit, pre-activations, hidden)."""
    o2 = HD + HD
    a = [D[i] * x + D[HD + i] for i in range(HD)]
    h = [v if v > 0.0 else SLOPE * v for v in a]
    return sum(D[o2 + i] * h[i] for i in range(HD)) + D[o2 + HD], a, h


def disc_backward(D, x, a, h, dl, grad):
    """Adds the gradient to grad given dl = d loss / d logit; returns d loss / d x."""
    o2 = HD + HD
    grad[o2 + HD] += dl
    dx = 0.0
    for i in range(HD):
        grad[o2 + i] += dl * h[i]
        da = dl * D[o2 + i] * (1.0 if a[i] > 0.0 else SLOPE)
        grad[HD + i] += da
        grad[i] += da * x
        dx += da * D[i]
    return dx


def disc_logit(D, x):
    return disc_forward(D, x)[0]


def _adam(theta, g, m, v, t):
    fm.adamw_step(theta, g, m, v, t, LR, 0.0, ADAM_B1)


def real_sample(gen):
    """One draw of the data: the cluster index integer(3), then the offset normal()."""
    k = gen.integer(3)
    return MODES[k] + SD * gen.normal()


def noise(gen):
    return [gen.normal() for _ in range(DZ)]


# ------------------------------------------------------------------ the gradient check

def gradient_check():
    gen = bookrng.generator(GRADCHECK)
    G = [v + 0.1 * gen.normal() for v in init_net(gen, DZ, HG)]
    D = [v + 0.1 * gen.normal() for v in init_net(gen, 1, HD)]
    n = 4
    reals = [real_sample(gen) for _ in range(n)]
    zs = [noise(gen) for _ in range(n)]

    def d_loss(Dp):
        tot = 0.0
        for xr, z in zip(reals, zs):
            tot += softplus(-disc_logit(Dp, xr)) + softplus(disc_logit(Dp, gen_forward(G, z)[0]))
        return tot / n

    def g_loss(Gp):
        return sum(softplus(-disc_logit(D, gen_forward(Gp, z)[0])) for z in zs) / n

    gd = [0.0] * n_disc()
    for xr, z in zip(reals, zs):
        l, a, h = disc_forward(D, xr)
        disc_backward(D, xr, a, h, -(1.0 - sigmoid(l)) / n, gd)
        xf = gen_forward(G, z)[0]
        l, a, h = disc_forward(D, xf)
        disc_backward(D, xf, a, h, sigmoid(l) / n, gd)
    gg = [0.0] * n_gen()
    for z in zs:
        xf, hid = gen_forward(G, z)
        l, a, h = disc_forward(D, xf)
        dx = disc_backward(D, xf, a, h, -(1.0 - sigmoid(l)) / n, [0.0] * n_disc())
        gen_backward(G, z, hid, dx, gg)
    rows = []
    for name, theta, grad, loss in (("discriminator, its own loss", D, gd, d_loss), ("generator, through the discriminator", G, gg, g_loss)):
        fails, worst = 0, 0.0
        for i in range(len(theta)):
            up, dn = list(theta), list(theta)
            up[i] += FD_STEP
            dn[i] -= FD_STEP
            num = (loss(up) - loss(dn)) / (2.0 * FD_STEP)
            diff = abs(grad[i] - num)
            worst = max(worst, diff)
            if diff > 1e-6 + 1e-6 * max(abs(grad[i]), abs(num)):
                fails += 1
        rows.append((name, len(theta), fails, "below 1e-6" if worst < 1e-6 else f"{worst:.1e}"))
    return (f"Both networks start from their {_seed(GRADCHECK)} initialisation (init_net, then 0.1 normal() added to every entry in "
            f"order, generator first), a batch of {n} reals and {n} noise vectors drawn real, real, ..., then noise, noise, ..., and "
            f"central differences of step {FD_STEP:g}. An entry fails when |analytic - numerical| exceeds 1e-6 + 1e-6 max(|analytic|, "
            f"|numerical|).\n\n"
            + _table(["gradient", "parameters checked", "entries failing the gate", "worst absolute difference"], rows))


# ------------------------------------------------------------------ training

@lru_cache(maxsize=None)
def evaluation_noise():
    gen = bookrng.generator(EVAL_NAME)
    return tuple(tuple(noise(gen)) for _ in range(EVAL_N))


def read_out(G):
    """(samples in each of the three modes, standard deviation of the 1000 evaluation samples)."""
    xs = [gen_forward(G, z)[0] for z in evaluation_noise()]
    counts = tuple(sum(1 for x in xs if abs(x - m) < NEAR) for m in MODES)
    mean = sum(xs) / len(xs)
    return counts, math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def train(name, steps_per_d, nudge=0.0):
    """Alternating training. Setup: init_net for the generator (W1 [16, 2], then W2 [16]), then for the discriminator (W1 [16], then
    W2 [16]), all from the stream `name`; the first generator weight starts `nudge` away. One discriminator step: for each of 32 items
    in turn draw a real sample (integer(3), then normal()) and a noise vector (2 normal()), compute -log sigmoid(logit(real)) -
    log(1 - sigmoid(logit(G(z)))), average, Adam. Then `steps_per_d` generator steps, each drawing 32 fresh noise vectors, loss
    -log sigmoid(logit(G(z))) averaged, Adam with its own moments and step count. Returns {'log': {discriminator step: read_out}, 'G', 'D'}."""
    gen = bookrng.generator(name)
    G = init_net(gen, DZ, HG)
    D = init_net(gen, 1, HD)
    G[0] += nudge
    mg, vg, md, vd = [0.0] * len(G), [0.0] * len(G), [0.0] * len(D), [0.0] * len(D)
    tg = 0
    log = {}
    for t in range(1, STEPS + 1):
        gd = [0.0] * len(D)
        for _ in range(BATCH):
            xr = real_sample(gen)
            l, a, h = disc_forward(D, xr)
            disc_backward(D, xr, a, h, -(1.0 - sigmoid(l)) / BATCH, gd)
            xf = gen_forward(G, noise(gen))[0]
            l, a, h = disc_forward(D, xf)
            disc_backward(D, xf, a, h, sigmoid(l) / BATCH, gd)
        _adam(D, gd, md, vd, t)
        for _ in range(steps_per_d):
            gg = [0.0] * len(G)
            for _ in range(BATCH):
                z = noise(gen)
                xf, hid = gen_forward(G, z)
                l, a, h = disc_forward(D, xf)
                dx = disc_backward(D, xf, a, h, -(1.0 - sigmoid(l)) / BATCH, [0.0] * len(D))
                gen_backward(G, z, hid, dx, gg)
            tg += 1
            _adam(G, gg, mg, vg, tg)
        if t % EVERY == 0:
            log[t] = read_out(G)
    return {"log": log, "G": G, "D": D}


def _job(job):
    name, sched, nudge = job
    return train(name, SCHEDULES[sched], nudge)


@lru_cache(maxsize=None)
def all_runs():
    """(name, schedule) -> the run and its four nudged copies, trained in four worker processes (50 runs)."""
    keys = [(n, s) for s in SCHEDULES for n in RUN_NAMES]
    jobs = [(n, s, d) for n, s in keys for d in (0.0, *stable.NUDGES)]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    return {k: out[5 * i:5 * i + 5] for i, k in enumerate(keys)}


def _shared(vals, fmt):
    return stable.text(float(vals[0]), [float(v) for v in vals[1:]], fmt)


def _int(x):
    return f"{x:.0f}"


def training_run():
    out = []
    for sched, title in (("one", "one generator step per discriminator step"), ("twenty", "twenty generator steps per discriminator step")):
        runs = all_runs()[(RUN_NAMES[0], sched)]
        rows = []
        for t in LOG_STEPS:
            cells = [t]
            for m in range(3):
                cells.append(_shared([r["log"][t][0][m] for r in runs], _int))
            other = [EVAL_N - sum(r["log"][t][0]) for r in runs]
            cells.append(_shared(other, _int))
            cells.append(_shared([r["log"][t][1] for r in runs], lambda x: f"{x:.3f}"))
            rows.append(cells)
        out.append(f"{title}:\n\n" + _table(["discriminator steps", "near -3", "near 0", "near +3", "elsewhere", "spread of samples"], rows))
    head = (f"Stream {_seed(RUN_NAMES[0])}, initialisation and every draw in the order of the chapter. The generator is read out on the "
            f"1000 evaluation noise vectors ({_seed(EVAL_NAME)}, {DZ} normal() each, drawn in order and used for every read-out): the "
            f"columns count the samples within {NEAR:g} of each centre, 'elsewhere' is the rest, and the spread is the standard deviation "
            f"of all 1000 samples (the data's is {DATA_STD:.3f}). Digits are those shared by the run and four copies with the first "
            f"generator weight changed by +1e-12, -1e-12, +3e-12 and -3e-12.")
    return head + "\n\n" + "\n\n".join(out)


@lru_cache(maxsize=None)
def nominal_logs():
    """The read-out logs of the first stream under both schedules, trained without the nudged copies (for the figure)."""
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(2) as pool:
        out = pool.map(_job, [(RUN_NAMES[0], s, 0.0) for s in SCHEDULES], chunksize=1)
    return {s: r["log"] for s, r in zip(SCHEDULES, out)}


def spread_data():
    """Steps and the spread of the samples at every read-out, for the two schedules of the first stream."""
    logs = nominal_logs()
    steps = sorted(logs["one"])
    return steps, {s: [logs[s][t][1] for t in steps] for s in SCHEDULES}


def collapse_counts():
    rows = []
    for name in RUN_NAMES:
        cells = [f"`{name}` (0x{bookrng.seed_of(name):016x})"]
        for sched in SCHEDULES:
            runs = all_runs()[(name, sched)]
            late = [t for t in sorted(runs[0]["log"]) if t > SETTLED_AFTER]
            collapsed = [sum(1 for t in late if sum(1 for c in r["log"][t][0] if c >= COVERED) <= 1) for r in runs]
            full = [sum(1 for t in late if all(c >= COVERED for c in r["log"][t][0])) for r in runs]
            cells.append(_shared(collapsed, _int))
            cells.append(_shared(full, _int))
        rows.append(cells)
    n_late = len([t for t in range(EVERY, STEPS + 1, EVERY) if t > SETTLED_AFTER])
    return (f"Five streams, each trained under both schedules for {STEPS} discriminator steps and read out every {EVERY} (the same "
            f"1000 evaluation noise vectors). Of the {n_late} read-outs after step {SETTLED_AFTER}, the table counts those at which at most "
            f"one mode holds {COVERED} or more of the 1000 samples (collapsed) and those at which all three do (all three covered). "
            f"Counts keep the digits shared with the four nudged copies.\n\n"
            + _table(["stream", "one: collapsed", "one: all three covered", "twenty: collapsed", "twenty: all three covered"], rows))


# ------------------------------------------------------------------ the generator against a frozen discriminator

def _frozen(G0, D, steps):
    """Generator-only training against the frozen D: Adam, 32 fresh noise vectors per step from the stream FROZEN, non-saturating loss.
    Returns {step: (mean D logit of the evaluation samples, mean x, spread, samples near a mode)}."""
    G = list(G0)
    gen = bookrng.generator(FROZEN)
    m, v = [0.0] * len(G), [0.0] * len(G)
    out = {}

    def summary():
        xs = [gen_forward(G, z)[0] for z in evaluation_noise()]
        mean = sum(xs) / len(xs)
        spread = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
        near = sum(1 for x in xs if any(abs(x - c) < NEAR for c in MODES))
        return (sum(disc_logit(D, x) for x in xs) / len(xs), mean, spread, near)
    out[0] = summary()
    for t in range(1, max(steps) + 1):
        gg = [0.0] * len(G)
        for _ in range(BATCH):
            z = noise(gen)
            xf, hid = gen_forward(G, z)
            l, a, h = disc_forward(D, xf)
            dx = disc_backward(D, xf, a, h, -(1.0 - sigmoid(l)) / BATCH, [0.0] * len(D))
            gen_backward(G, z, hid, dx, gg)
        _adam(G, gg, m, v, t)
        if t in steps:
            out[t] = summary()
    return out


def _frozen_job(run):
    return _frozen(run["G"], run["D"], FROZEN_STEPS)


@lru_cache(maxsize=None)
def frozen_runs():
    runs = all_runs()[(RUN_NAMES[0], "one")]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_frozen_job, runs)


def frozen_judge():
    res = frozen_runs()
    rows = []
    for t in FROZEN_STEPS:
        rows.append((t, _shared([r[t][0] for r in res], lambda x: f"{x:.3f}"), _shared([r[t][1] for r in res], lambda x: f"{x:.2f}"),
                     _shared([r[t][2] for r in res], lambda x: f"{x:.3f}"), _shared([r[t][3] for r in res], _int)))
    return (f"The discriminator and generator of the run `{RUN_NAMES[0]}` after {STEPS} steps of one generator step per discriminator step "
            f"(first table of the previous example). The discriminator is now frozen and only the generator trains: Adam with the same "
            f"settings and a fresh state, 32 noise vectors per step from {_seed(FROZEN)} ({DZ} normal() each, in order), the "
            f"non-saturating loss. Read out on the 1000 evaluation noise vectors: the mean discriminator logit of the samples, their "
            f"mean and spread, and how many lie within {NEAR:g} of a centre. Shared digits with the four nudged copies.\n\n"
            + _table(["generator-only steps", "mean logit of the samples", "mean of the samples", "spread of the samples", "samples near a centre"], rows))


# ------------------------------------------------------------------ a generator that starts far from the data

@lru_cache(maxsize=None)
def _dark_start():
    gen = bookrng.generator(DARK_NAME)
    G = init_net(gen, DZ, HG)
    D = init_net(gen, 1, HD)
    G[HG * DZ + HG + HG] += DARK_SHIFT
    md, vd = [0.0] * len(D), [0.0] * len(D)
    for t in range(1, DARK_D_STEPS + 1):
        gd = [0.0] * len(D)
        for _ in range(BATCH):
            xr = real_sample(gen)
            l, a, h = disc_forward(D, xr)
            disc_backward(D, xr, a, h, -(1.0 - sigmoid(l)) / BATCH, gd)
            xf = gen_forward(G, noise(gen))[0]
            l, a, h = disc_forward(D, xf)
            disc_backward(D, xf, a, h, sigmoid(l) / BATCH, gd)
        _adam(D, gd, md, vd, t)
    return G, D


def dark_data(kind):
    """{step: (norm of the generator's gradient, mean x, mean logit)} for plain gradient descent of the given loss."""
    G0, D = _dark_start()
    G = list(G0)
    ev = bookrng.generator(DARK_EVAL)
    Z = [noise(ev) for _ in range(256)]
    gen = bookrng.generator(DARK_GEN)
    out = {}

    def summary(gnorm):
        xs = [gen_forward(G, z)[0] for z in Z]
        return (gnorm, sum(xs) / len(xs), sum(disc_logit(D, x) for x in xs) / len(xs))
    gnorm = None
    for t in range(1, DARK_STEPS + 1):
        gg = [0.0] * len(G)
        for _ in range(BATCH):
            z = noise(gen)
            xf, hid = gen_forward(G, z)
            l, a, h = disc_forward(D, xf)
            dx = disc_backward(D, xf, a, h, generator_gradient(kind, l) / BATCH, [0.0] * len(D))
            gen_backward(G, z, hid, dx, gg)
        if t == 1:
            gnorm = math.sqrt(sum(g * g for g in gg))
            out[0] = summary(gnorm)
        for i in range(len(G)):
            G[i] -= DARK_RATE * gg[i]
        if t in DARK_LOG:
            out[t] = summary(gnorm)
    return out


def dark_start():
    rows = []
    for kind in ("saturating", "non-saturating", "least-squares"):
        d = dark_data(kind)
        rows.append((kind, _f(d[0][0], 4), _f(d[0][1], 2), _f(d[25][1], 2), _f(d[100][1], 2), _f(d[100][2], 2)))
    start = dark_data("saturating")[0]
    return (f"A generator and a discriminator from the stream {_seed(DARK_NAME)} (init_net for each, generator first), with {DARK_SHIFT:g} "
            f"added to the generator's output bias so that its samples start near {DARK_SHIFT:g}. The discriminator trains alone "
            f"for {DARK_D_STEPS} Adam steps against that generator (each step: 32 items, a real draw then a noise draw each, from the same "
            f"stream) and is then frozen; its mean logit on the generator's samples is {start[2]:.2f}. The generator then takes {DARK_STEPS} steps of plain "
            f"gradient descent, rate {DARK_RATE:g}, on each loss in turn, from the same start and the same noise ({DARK_GEN}: 32 vectors "
            f"per step), read out on 256 noise vectors from {_seed(DARK_EVAL)}.\n\n"
            + _table(["generator loss", "norm of its gradient at step 1", "mean sample at start", "after 25 steps", "after 100 steps",
                      "mean logit after 100 steps"], rows))


# ------------------------------------------------------------------ the least-squares losses of a patch discriminator

REAL_LOGITS = (0.8, 0.3, 1.2, 0.5)
FAKE_LOGITS = (-0.2, 0.4, 0.1, -0.6)
FEAT_REAL = (0.2, -0.1, 0.5, 0.0)
FEAT_FAKE = (0.1, 0.3, 0.6, -0.4)


def least_squares_losses():
    nr, nf = len(REAL_LOGITS), len(FAKE_LOGITS)
    loss_r = sum((l - 1.0) ** 2 for l in REAL_LOGITS) / nr
    loss_f = sum(l * l for l in FAKE_LOGITS) / nf
    d_r = [2.0 * (l - 1.0) / nr for l in REAL_LOGITS]
    d_f = [2.0 * l / nf for l in FAKE_LOGITS]
    g_loss = sum((l - 1.0) ** 2 for l in FAKE_LOGITS) / nf
    g_grad = [2.0 * (l - 1.0) / nf for l in FAKE_LOGITS]
    n = len(FEAT_REAL)
    fm_loss = sum(abs(f - r) for r, f in zip(FEAT_REAL, FEAT_FAKE)) / n
    fm_grad = [(1.0 if f > r else -1.0) / n for r, f in zip(FEAT_REAL, FEAT_FAKE)]
    fmt = lambda xs, d=4: "(" + ", ".join(_f(x, d) for x in xs) + ")"
    rows = [
        ("discriminator loss, mean((D(real) - 1)^2) + mean(D(fake)^2)", _f(loss_r, 4) + " + " + _f(loss_f, 4) + " = " + _f(loss_r + loss_f, 4)),
        ("its gradient on the real logits", fmt(d_r)),
        ("its gradient on the fake logits", fmt(d_f)),
        ("generator loss, mean((D(fake) - 1)^2)", _f(g_loss, 4)),
        ("its gradient on the fake logits", fmt(g_grad)),
        ("feature-matching loss, mean(|feature(fake) - feature(real)|)", _f(fm_loss, 4)),
        ("its gradient on the fake features", fmt(fm_grad)),
    ]
    return (f"Real logits {REAL_LOGITS}, fake logits {FAKE_LOGITS}; features of the real sample {FEAT_REAL} and of the fake {FEAT_FAKE} "
            f"(no pair equal, so the absolute value has a slope everywhere). Means are over the entries of the array they name.\n\n"
            + _table(["quantity", "value"], rows, "ll"))


# ------------------------------------------------------------------ counting a patch discriminator

def _conv_out(h, k, pad, stride=1):
    return (h + 2 * pad - k) // stride + 1


def patch_discriminator():
    rows = []
    for label, n_fft, hop, clip, c1, c2, k in (("the crate's training-test configuration", 8, 4, 24, 2, 2, 3),
                                              ("an audio-sized configuration (illustration)", 1024, 256, 24000, 16, 32, 3)):
        bins, frames = n_fft // 2 + 1, (clip - n_fft) // hop + 1
        h1, w1 = _conv_out(bins, k, 1), _conv_out(frames, k, 1)
        h2, w2 = _conv_out(h1, k, 1), _conv_out(w1, k, 1)
        h3, w3 = _conv_out(h2, k, 1), _conv_out(w2, k, 1)
        params = (c1 * k * k + c1) + (c2 * c1 * k * k + c2) + (c2 * k * k + 1)
        macs = c1 * k * k * h1 * w1 + c2 * c1 * k * k * h2 * w2 + c2 * k * k * h3 * w3
        dft = bins * frames * n_fft * 2
        rf = 1 + 3 * (k - 1)
        rows.append((label, f"{n_fft} / {hop} / {clip:,}", f"{bins} x {frames}", f"{h3} x {w3} = {h3 * w3:,}", f"{rf} x {rf}",
                     f"{c1}, {c2}", f"{params:,}", f"{macs:,}", f"{dft:,}"))
    return ("A discriminator of three stride-1 convolutions of kernel k = 3 and padding 1 over a magnitude spectrogram (frequency bins "
            "x frames), LeakyReLU after the first two, as in `crates/minimaxmusic3/src/discriminator.rs`: 1 -> c1 -> c2 -> 1 channels. "
            "Frames = (clip - n_fft) / hop + 1 rounded down, bins = n_fft / 2 + 1. A convolution layer with c_in -> c_out channels has "
            "c_out c_in k^2 weights and c_out biases and costs c_out c_in k^2 multiply-adds per output position; the direct "
            "transform costs 2 n_fft multiply-adds per bin and frame (a real and an imaginary sum). The second row's sizes are chosen "
            "here for the count and are not a brain model.\n\n"
            + _table(["configuration", "n_fft / hop / samples", "spectrogram", "logit map (patches)", "receptive field of one logit",
                      "c1, c2", "parameters", "multiply-adds of D", "multiply-adds of the transform"], rows))


# ------------------------------------------------------------------ the shapes of a convolutional image generator and discriminator

def conv_shapes():
    rows = []
    gen_layers = ((4, 512, 256), (8, 256, 128), (16, 128, 64), (32, 64, 3))
    total_w = total_m = 0
    for side, cin, cout in gen_layers:
        out = (side - 1) * 2 - 2 * 1 + 4
        w = cin * cout * 16 + cout
        m = cin * cout * 16 * side * side
        total_w += w
        total_m += m
        rows.append(("generator", "transposed conv 4 x 4, stride 2, padding 1", f"{cin} x {side} x {side}", f"{cout} x {out} x {out}", f"{w:,}", f"{m:,}"))
    rows.append(("generator", "total after the 4 x 4 x 512 start", "", "", f"{total_w:,}", f"{total_m:,}"))
    total_w = total_m = 0
    dis_layers = ((64, 3, 64), (32, 64, 128), (16, 128, 256), (8, 256, 512))
    for side, cin, cout in dis_layers:
        out = _conv_out(side, 4, 1, 2)
        w = cin * cout * 16 + cout
        m = cin * cout * 16 * out * out
        total_w += w
        total_m += m
        rows.append(("discriminator", "conv 4 x 4, stride 2, padding 1", f"{cin} x {side} x {side}", f"{cout} x {out} x {out}", f"{w:,}", f"{m:,}"))
    side, cin = 4, 512
    w, m = cin * 1 * 9 + 1, cin * 1 * 9 * side * side
    total_w += w
    total_m += m
    rows.append(("discriminator", "conv 3 x 3, stride 1, padding 1, to one channel", "512 x 4 x 4", "1 x 4 x 4 (16 patch logits)", f"{w:,}", f"{m:,}"))
    rows.append(("discriminator", "total", "", "", f"{total_w:,}", f"{total_m:,}"))
    return ("A layout for 64 x 64 colour images chosen here to count with: a transposed convolution of output size (in - 1) stride - "
            "2 padding + kernel, a convolution of output size floor((in + 2 padding - kernel) / stride) + 1, weights c_in c_out k^2 plus "
            "c_out biases, and c_in c_out k^2 multiply-adds per input position of a transposed convolution or per output position of a "
            "convolution. Activations and the linear layer that maps the noise to 4 x 4 x 512 are not counted.\n\n"
            + _table(["network", "layer", "input", "output", "parameters", "multiply-adds"], rows, "lllrrr"))
