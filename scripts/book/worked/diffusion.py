# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Diffusion Models".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: diffusion.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. The data are
the three clusters of `gans` (imported, not copied), the Adam update is `first_models.adamw_step`, the stable-digits rule is `stable`.

* a noising chain of 50 steps on one real number, its closed form, and the check of the closed form against the step-by-step moments;
* the noised marginals of the three clusters in closed form: the density, the number of modes and the distance from a standard
  normal at several steps;
* three schedules of brain (linear, scaled linear, cosine), their terminal signal-to-noise ratio, and the order of the products
  that build the cumulative product of the SDXL schedule;
* the posterior of the reverse step, three ways, and the exact denoiser (posterior mean, noise estimate, score) of the clusters,
  with the score checked against a central difference of the log of the density;
* the least squared error any noise predictor can reach at each step, from the exact denoiser;
* the three targets (clean value, noise, velocity), the conversions among them, the factor by which each amplifies an error, and the
  weight each loss puts on the error of the clean value;
* a two-hidden-layer network that predicts the noise from the noisy value and the step: its hand-written backward pass and the
  gradient check; its training over 5000 steps against the exact denoiser; the table of its error per band of steps; and samples drawn
  with its predictions, with the exact denoiser's, and with one jump.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.
Trained numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the
nudges of `stable.py`. The runs are trained in at most four worker processes.

Swedish Embedded AB implements solutions for generative models whose noising schedules, targets and samplers are specified,
checked and made reproducible. If your team needs expertise in diffusion models and image synthesis, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import struct
from functools import lru_cache

import bookrng
import first_models as fm
import gans
import stable

# ------------------------------------------------------------------ constants

MODES, SD = gans.MODES, gans.SD           # the three clusters of data: centres -3, 0, 3, standard deviation 0.25, equal weights
NEAR, DATA_STD = gans.NEAR, gans.DATA_STD
T = 50                                    # steps of the chain
BETA_FIRST, BETA_LAST = 0.001, 0.25       # beta_t is linspace(BETA_FIRST, BETA_LAST, T)[t - 1]
H = 32                                    # hidden units of each of the two hidden layers
STEPS, BATCH, LR = 5000, 64, 3e-3         # training: Adam, LR * (1 - (s - 1) / STEPS) at step s = 1 .. STEPS
LOG_STEPS = (500, 1000, 2000, 3000, 4000, 5000)
RUN, EVAL, SAMPLE = "diffusion.run", "diffusion.evaluation", "diffusion.sample"
EVAL_N, SAMPLE_N = 1000, 1000
GRADCHECK, FD_STEP = "diffusion.gradcheck", 1e-5
VCHECK = "diffusion.targets"
BANDS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50))
SDXL = dict(n=1000, first=0.00085, last=0.012)


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


def _g(x, d=4):
    return f"{x:.{d}g}"


# ------------------------------------------------------------------ the chain

def betas():
    """beta_t for t = 1 .. T as a list (index t - 1): linspace(BETA_FIRST, BETA_LAST, T)."""
    return [BETA_FIRST + (BETA_LAST - BETA_FIRST) * i / (T - 1) for i in range(T)]


@lru_cache(maxsize=None)
def alpha_bars():
    """abar_0 = 1 and abar_t = abar_{t-1} (1 - beta_t) for t = 1 .. T, as a list of T + 1 numbers."""
    out = [1.0]
    for b in betas():
        out.append(out[-1] * (1.0 - b))
    return out


def ab_coeffs(t):
    """(a_t, b_t) = (sqrt(abar_t), sqrt(1 - abar_t))."""
    ab = alpha_bars()[t]
    return math.sqrt(ab), math.sqrt(1.0 - ab)


def bound_weights():
    """The weight of the squared noise error in the step-t term of the bound, beta_t^2 / (2 tilde-beta_t alpha_t (1 - abar_t)), which
    simplifies to beta_t / (2 alpha_t (1 - abar_(t-1))); defined for t >= 2 (tilde-beta_1 is 0)."""
    ab, bs = alpha_bars(), betas()
    rows = []
    for t in (2, 5, 10, 20, 30, 40, 50):
        beta, al = bs[t - 1], 1.0 - bs[t - 1]
        tilde = (1.0 - ab[t - 1]) / (1.0 - ab[t]) * beta
        direct = beta * beta / (2.0 * tilde * al * (1.0 - ab[t]))
        simple = beta / (2.0 * al * (1.0 - ab[t - 1]))
        assert abs(direct - simple) < 1e-9 * simple
        rows.append((t, _g(beta, 3), _g(tilde, 3), _g(simple, 4)))
    return ("The weight w_t of (noise - estimate)^2 in the t-th term of the bound, w_t = beta_t^2 / (2 tilde-beta_t alpha_t (1 - abar_t)), "
            "for the chain of the chapter (tilde-beta_t = (1 - abar_(t-1)) beta_t / (1 - abar_t)).\n\n"
            + _table(["t", "beta_t", "tilde-beta_t", "w_t"], rows))


def schedule_table():
    ab, bs = alpha_bars(), betas()
    rows = []
    for t in (1, 2, 5, 10, 20, 30, 40, 50):
        a, b = ab_coeffs(t)
        rows.append((t, _f(bs[t - 1], 4), _f(ab[t], 4), _f(a, 4), _f(b, 4), _g(ab[t] / (1.0 - ab[t]), 4)))
    mean, var, worst = 3.0, SD * SD, 0.0
    for t in range(1, T + 1):
        mean, var = math.sqrt(1.0 - bs[t - 1]) * mean, (1.0 - bs[t - 1]) * var + bs[t - 1]
        a, b = ab_coeffs(t)
        worst = max(worst, abs(mean - a * 3.0), abs(var - (a * a * SD * SD + b * b)))
    check = ("The largest difference over the 50 steps between the mean and variance of a cluster (centre 3, standard deviation 0.25) "
             "carried step by step and the closed form is " + ("below 1e-12." if worst < 1e-12 else f"{worst:.1e} (CHECK FAILED)."))
    return (f"The chain of {T} steps with beta_t = {BETA_FIRST:g} + ({BETA_LAST:g} - {BETA_FIRST:g}) (t - 1) / {T - 1} for t = 1 .. {T}, "
            f"abar_t = (1 - beta_1) ... (1 - beta_t), a_t = sqrt(abar_t), b_t = sqrt(1 - abar_t) and the signal-to-noise ratio "
            f"abar_t / (1 - abar_t). The products are taken in increasing t.\n\n"
            + _table(["t", "beta_t", "abar_t", "a_t", "b_t", "abar_t / (1 - abar_t)"], rows) + "\n\n" + check)


# ------------------------------------------------------------------ the noised marginals in closed form

def _normal_pdf(x, mean, var):
    return math.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def marginal_density(x, t):
    """p_t(x) of the three clusters noised t steps: the mixture of N(a_t m_k, a_t^2 SD^2 + b_t^2) with weights 1/3."""
    a, b = ab_coeffs(t)
    var = a * a * SD * SD + b * b
    return sum(_normal_pdf(x, a * m, var) for m in MODES) / 3.0


def _count_modes(t):
    """Local maxima of p_t on the grid -8, -7.99, ..., 8 (a point higher than its left neighbour and not lower than its right)."""
    xs = [-8.0 + 0.01 * i for i in range(1601)]
    p = [marginal_density(x, t) for x in xs]
    return sum(1 for i in range(1, 1600) if p[i] > p[i - 1] and p[i] >= p[i + 1])


def _kl_to_standard_normal(t):
    """KL(p_t || N(0, 1)) in nats by the midpoint rule with step 0.002 on [-12, 12]."""
    h, tot = 0.002, 0.0
    for i in range(12000):
        x = -12.0 + (i + 0.5) * h
        p = marginal_density(x, t)
        if p > 0.0:
            tot += p * (math.log(p) - math.log(_normal_pdf(x, 0.0, 1.0))) * h
    return tot


MARGINAL_STEPS = (0, 5, 10, 14, 18, 25, 35, 50)


def noised_marginals():
    rows = []
    for t in MARGINAL_STEPS:
        p0, pv = marginal_density(0.0, t), marginal_density(1.5, t)
        rows.append((t, _f(p0, 4), _f(pv, 4), _g(pv / p0, 3), _count_modes(t), _f(_kl_to_standard_normal(t), 4)))
    return ("The three clusters (centres -3, 0, 3, standard deviation 0.25, weight 1/3 each) after t steps of the chain: "
            "p_t(x) = (1/3) sum_k N(x; a_t m_k, a_t^2 0.25^2 + b_t^2). The valley is x = 1.5, between the clusters at 0 and 3, and 'valley / peak' is p_t(1.5) / p_t(0), the peak being the density at the centre of the middle cluster. "
            "Modes are local maxima on the grid -8, -7.99, ..., 8. The distance is KL(p_t || N(0, 1)) in nats by the midpoint rule with "
            "step 0.002 on [-12, 12].\n\n"
            + _table(["t", "p_t(0)", "p_t(1.5)", "valley / peak", "modes", "KL(p_t \\|\\| N(0, 1))"], rows))


def marginal_curves_data():
    xs = [-6.0 + 0.05 * i for i in range(241)]
    steps = (0, 10, 14, 18, 25, 50)
    return xs, {t: [marginal_density(x, t) for x in xs] for t in steps}


# ------------------------------------------------------------------ other schedules, and the order of a product

def _linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def schedule_betas(kind, n=1000):
    """The three beta schedules of brain's DiscreteConfig, in float64: linear, scaled linear (linspace of the square roots, squared)
    and the cosine schedule abar(s) = cos((s + 0.008) / 1.008 pi / 2)^2 with beta_i = min(1 - abar((i + 1) / n) / abar(i / n), 0.999)."""
    if kind == "linear":
        return _linspace(1e-4, 0.02, n)
    if kind == "scaled linear":
        return [v * v for v in _linspace(math.sqrt(SDXL["first"]), math.sqrt(SDXL["last"]), n)]
    bar = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2.0) ** 2
    return [min(1.0 - bar((i + 1) / n) / bar(i / n), 0.999) for i in range(n)]


def cumulative(bs):
    out, acc = [], 1.0
    for b in bs:
        acc *= 1.0 - b
        out.append(acc)
    return out


KINDS = ("linear", "scaled linear", "cosine")


def schedules_compared():
    rows = []
    for kind in KINDS:
        ab = cumulative(schedule_betas(kind))
        half = next(i + 1 for i, v in enumerate(ab) if v < 0.5)
        hund = next(i + 1 for i, v in enumerate(ab) if v < 0.01)
        end = ab[-1]
        rows.append((kind, _g(ab[249], 3), _g(ab[499], 3), _g(ab[749], 3), _g(end, 3), _g(end / (1.0 - end), 3), half, hund))
    toy = alpha_bars()
    return ("Each schedule has 1000 steps; abar_t = (1 - beta_1) ... (1 - beta_t) in float64, t counted from 1 (the entry of brain's "
            "table with index t - 1). Linear: beta from 0.0001 to 0.02. Scaled linear (SDXL): the square root of beta from sqrt(0.00085) "
            "to sqrt(0.012) in 1000 equal steps, squared. Cosine: abar(s) = cos((s + 0.008) / 1.008 * pi / 2)^2, beta_i = min(1 - "
            "abar((i + 1) / 1000) / abar(i / 1000), 0.999) for i = 0 .. 999. 'First t below' is the smallest t at which abar_t falls "
            "below the threshold.\n\n"
            + _table(["schedule", "abar_250", "abar_500", "abar_750", "abar_1000", "SNR at t = 1000", "first t below 1/2", "first t below 0.01"], rows)
            + f"\n\nThe chain of this chapter ends at abar_{T} = {toy[T]:.3e}, a signal-to-noise ratio of {toy[T] / (1 - toy[T]):.3e}.")


def schedule_curves_data():
    """log-SNR = ln(abar / (1 - abar)) against t / n for each schedule of schedules_compared, and for the 50-step chain."""
    out = {}
    for kind in KINDS:
        ab = cumulative(schedule_betas(kind))
        n = len(ab)
        out[kind] = ([(i + 1) / n for i in range(n)], [math.log(v / (1.0 - v)) for v in ab])
    ab = alpha_bars()
    out["chain of this chapter"] = ([t / T for t in range(1, T + 1)], [math.log(ab[t] / (1.0 - ab[t])) for t in range(1, T + 1)])
    return out


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def product_order():
    """The SDXL abar_1000 by two orders of rounding. beta is the float64 value rounded to float32 in both."""
    b32 = [_f32(b) for b in schedule_betas("scaled linear")]
    acc32, seq32 = 1.0, []
    for b in b32:
        acc32 = _f32(acc32 * _f32(1.0 - b))
        seq32.append(acc32)
    acc64, wide = 1.0, []
    for b in b32:
        acc64 *= 1.0 - b
        wide.append(_f32(acc64))
    rel = max(abs(s - w) / w for s, w in zip(seq32, wide))
    return (f"The scaled-linear schedule of the SDXL, beta computed in float64 and rounded to float32 (round to nearest). Left: every "
            f"factor 1 - beta_t is rounded to float32 and every partial product too. Right: the product is accumulated in float64 "
            f"from the same float32 betas and rounded to float32 once, at the end of each entry.\n\n"
            + _table(["quantity", "float32 partial products", "float64 accumulator"],
                     [("abar_1000", f"{seq32[-1]:.10f}", f"{wide[-1]:.10f}"), ("abar_500", f"{seq32[499]:.10f}", f"{wide[499]:.10f}"),
                      ("largest relative difference over the 1000 entries", f"{rel:.3e}", "reference")]))


def start_noise():
    """The two candidates for the standard deviation of the starting noise of a 20-step run of the SDXL schedule."""
    ab = cumulative(schedule_betas("scaled linear"))
    k = 19 * 50 + 1                                  # the first sampling step of 20 (steps_offset 1): table index 951
    sigma = math.sqrt((1.0 - ab[k]) / ab[k])
    lifted = math.sqrt(sigma * sigma + 1.0)
    return (f"The SDXL schedule (scaled linear, 1000 steps, float64) at table index {k}, the product of the first {k + 1} factors "
            f"(index counted from 0, so t = {k + 1} in this chapter's counting), where a 20-step sampler with offset 1 starts. "
            f"abar = {ab[k]:.10f}, a = sqrt(abar) = {math.sqrt(ab[k]):.10f}.\n\n"
            + _table(["quantity", "value"], [("sigma = b / a = sqrt((1 - abar) / abar)", f"{sigma:.4f}"),
                                             ("sqrt(sigma^2 + 1) = 1 / a", f"{lifted:.4f}"),
                                             ("the second over the first, minus 1", f"{lifted / sigma - 1.0:.4%}")]))


# ------------------------------------------------------------------ the reverse step, and the exact denoiser

def reverse_posterior():
    t, x0, eps = 20, 2.9, 0.7
    ab, bs = alpha_bars(), betas()
    a_t, b_t = ab_coeffs(t)
    x_t = a_t * x0 + b_t * eps
    beta, al, ab_t, ab_p = bs[t - 1], 1.0 - bs[t - 1], ab[t], ab[t - 1]
    # (1) two Gaussians multiplied: p(x_{t-1} | x0) = N(sqrt(abar_{t-1}) x0, 1 - abar_{t-1}) times p(x_t | x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t)
    prec = al / beta + 1.0 / (1.0 - ab_p)
    var1 = 1.0 / prec
    mean1 = var1 * (math.sqrt(al) * x_t / beta + math.sqrt(ab_p) * x0 / (1.0 - ab_p))
    # (2) the closed form in x_0 and x_t
    mean2 = math.sqrt(ab_p) * beta / (1.0 - ab_t) * x0 + math.sqrt(al) * (1.0 - ab_p) / (1.0 - ab_t) * x_t
    var2 = (1.0 - ab_p) / (1.0 - ab_t) * beta
    # (3) in x_t and the noise
    mean3 = (x_t - beta / math.sqrt(1.0 - ab_t) * eps) / math.sqrt(al)
    rows = [("product of the two Gaussians", f"{mean1:.10f}", f"{var1:.10f}"), ("closed form in x_0 and x_t", f"{mean2:.10f}", f"{var2:.10f}"),
            ("closed form in x_t and the noise", f"{mean3:.10f}", f"{var2:.10f}")]
    ok = max(abs(mean1 - mean2), abs(mean1 - mean3), abs(var1 - var2)) < 1e-12
    return (f"t = {t}, x_0 = {x0}, noise = {eps}, so x_t = a_t x_0 + b_t noise = {x_t:.10f}, beta_t = {beta:.6f}, abar_t = {ab_t:.10f}, "
            f"abar_(t-1) = {ab_p:.10f}. The three ways of writing the mean and variance of p(x_(t-1) | x_t, x_0):\n\n"
            + _table(["route", "mean", "variance"], rows) + "\n\n"
            + ("The three means and the two variances agree to below 1e-12." if ok else "CHECK FAILED"))


def exact_denoiser(x, t):
    """(E[x_0 | x_t = x], E[noise | x_t = x]) of the three clusters. For cluster k, with v = a^2 SD^2 + b^2, the posterior mean of x_0
    is m_k + a SD^2 / v (x - a m_k) and the cluster's weight is proportional to N(x; a m_k, v); the noise estimate is (x - a x0hat) / b."""
    a, b = ab_coeffs(t)
    v = a * a * SD * SD + b * b
    lw = [-0.5 * (x - a * m) ** 2 / v for m in MODES]          # the components share v, so the weights are softmax(lw)
    top = max(lw)
    w = [math.exp(l - top) for l in lw]
    z = sum(w)
    x0 = sum(wk * (m + a * SD * SD / v * (x - a * m)) for wk, m in zip(w, MODES)) / z
    return x0, (x - a * x0) / b


def _log_marginal(x, t):
    return math.log(marginal_density(x, t))


def denoiser_table():
    rows = []
    for t in (5, 20, 35):
        a, b = ab_coeffs(t)
        for x in (-1.0, 0.25, 1.0, 2.5):
            x0, e = exact_denoiser(x, t)
            rows.append((t, _f(x, 4), _f(x0, 4), _f(e, 4), _f(-e / b, 4)))
    return ("The exact denoiser of the three clusters at three steps and four noisy values x_t each: the "
            "posterior mean E[x_0 | x_t], the noise estimate E[noise | x_t] = (x_t - a_t E[x_0 | x_t]) / b_t and the score "
            "-E[noise | x_t] / b_t.\n\n"
            + _table(["t", "x_t", "E[x_0 given x_t]", "E[noise given x_t]", "-E[noise given x_t] / b_t"], rows)
            + "\n\nAt 24 further values of x_t (at each of the three steps, x_t = (k - 4) a_t + 0.37 (k mod 3) for k = 0 .. 7) the score "
              "-E[noise | x_t] / b_t differs from the central difference of ln p_t(x_t) (step 1e-5) by "
            + ("less than 1e-6." if _score_worst() < 1e-6 else "more than 1e-6 (CHECK FAILED)."))


def _score_worst():
    worst = 0.0
    for t in (5, 20, 35):
        a, b = ab_coeffs(t)
        for k in range(8):
            x = (-4.0 + k) * a + 0.37 * (k % 3)
            e = exact_denoiser(x, t)[1]
            h = 1e-5
            fd = (_log_marginal(x + h, t) - _log_marginal(x - h, t)) / (2.0 * h)
            worst = max(worst, abs(fd - (-e / b)))
    return worst


@lru_cache(maxsize=None)
def floor_losses():
    """For t = 1 .. T the least value of E (noise - g(x_t))^2 over all functions g: 1 - E [E(noise | x_t)^2], by the midpoint rule with
    step 0.005 on [-14, 14]."""
    h, out = 0.005, []
    for t in range(1, T + 1):
        tot = 0.0
        for i in range(5600):
            x = -14.0 + (i + 0.5) * h
            tot += marginal_density(x, t) * exact_denoiser(x, t)[1] ** 2 * h
        out.append(1.0 - tot)
    return out


def floor_table():
    fl = floor_losses()
    rows = [(t, _f(fl[t - 1], 4)) for t in (1, 2, 5, 10, 20, 30, 40, 50)]
    return ("The least mean squared error of any predictor of the noise from (x_t, t), computed from the exact denoiser: "
            "1 - E[E(noise | x_t)^2], the expectation by the midpoint rule with step 0.005 on [-14, 14]. The weights of the three clusters in the exact denoiser are computed as a softmax in the log domain, the largest log-weight subtracted before exponentiating, because at t = 1 and 2 the density of a far cluster underflows otherwise.\n\n"
            + _table(["t", "least error"], rows) + f"\n\nThe mean over the {T} steps (the least value of the training loss when t is uniform) is {sum(fl) / T:.4f}.")


# ------------------------------------------------------------------ the three targets

def targets_table():
    x0, eps = 2.9, 0.7
    rows, worst = [], 0.0
    for t in (1, 10, 25, 50):
        a, b = ab_coeffs(t)
        x = a * x0 + b * eps
        v = a * eps - b * x0
        x0_e, x0_v = (x - b * eps) / a, a * x - b * v
        e_x, e_v = (x - a * x0) / b, b * x + a * v
        worst = max(worst, abs(x0_e - x0), abs(x0_v - x0), abs(e_x - eps), abs(e_v - eps))
        rows.append((t, _f(x, 6), _f(v, 6), _f(x0_e, 6), _f(x0_v, 6), _f(e_x, 6), _f(e_v, 6)))
    return (f"x_0 = {x0}, noise = {eps}. x_t = a_t x_0 + b_t noise and v = a_t noise - b_t x_0. The clean value recovered "
            f"from (x_t, noise) as (x_t - b_t noise) / a_t and from (x_t, v) as a_t x_t - b_t v; the noise recovered from (x_t, x_0) "
            f"as (x_t - a_t x_0) / b_t and from (x_t, v) as b_t x_t + a_t v.\n\n"
            + _table(["t", "x_t", "v", "x_0 from noise", "x_0 from v", "noise from x_0", "noise from v"], rows)
            + "\n\nThe largest difference from the true x_0 or noise over the four rows is "
            + ("below 1e-12." if worst < 1e-12 else f"{worst:.1e} (CHECK FAILED)."))


def error_table():
    rows = []
    for t in (1, 5, 10, 20, 30, 40, 50):
        a, b = ab_coeffs(t)
        snr = a * a / (b * b)
        rows.append((t, _g(snr, 4), _g(b / a, 4), _g(b, 4), _g(1.0, 1), _g(snr, 4), _g(snr + 1.0, 4)))
    gen = bookrng.generator(VCHECK)
    worst = 0.0
    for _ in range(1000):
        t = gen.integer(T) + 1
        a, b = ab_coeffs(t)
        snr = a * a / (b * b)
        x0, eps, d = gen.normal(), gen.normal(), 0.1 * gen.normal()
        x = a * x0 + b * eps
        # the noise estimate is off by d; the clean value and the velocity it implies
        x0_hat = (x - b * (eps + d)) / a
        v_hat = (a * (eps + d) - b * x0_hat)
        err_x0, err_v = (x0_hat - x0) ** 2, (v_hat - (a * eps - b * x0)) ** 2
        worst = max(worst, abs(d * d - snr * err_x0) / (d * d), abs(err_v - (snr + 1.0) * err_x0) / err_v)
    return (f"For each step: the signal-to-noise ratio SNR = a_t^2 / b_t^2; the error of the clean value x_0 implied by a noise "
            f"estimate that is off by 1 (it is b_t / a_t); the error of x_0 implied by a velocity estimate that is off by 1 "
            f"(it is b_t); and the weight each loss puts on the squared error of the implied clean value: 1 for the loss on x_0, SNR for "
            f"the loss on the noise, SNR + 1 for the loss on the velocity.\n\n"
            + _table(["t", "SNR", "x_0 error per unit noise error", "x_0 error per unit velocity error", "weight, x_0 loss", "weight, noise loss", "weight, velocity loss"], rows)
            + f"\n\nCheck on 1000 draws from {_seed(VCHECK)} (per draw: t = integer(50) + 1, then x_0, noise and an error d = 0.1 normal(), "
              f"each by normal(), in this order): the squared noise error equals SNR times the squared x_0 error, and the squared "
              f"velocity error equals (SNR + 1) times it, with a largest relative difference "
            + ("below 1e-9." if worst < 1e-9 else f"{worst:.1e} (CHECK FAILED)."))


# ------------------------------------------------------------------ the network

O2 = 3 * H            # offset of W2 (H x H)
O3 = O2 + H * H + H   # offset of W3 (H)
N_PARAMS = O3 + H + 1


def init_net(gen):
    """[W1 (H x 2, row-major: the weight on x then the weight on tau), b1 (H), W2 (H x H, row-major), b2 (H), W3 (H), b3 (1)]:
    each weight matrix drawn normal() / sqrt(fan_in) in the order W1, W2, W3, each in row-major order; all biases 0."""
    w1 = [gen.normal() / math.sqrt(2.0) for _ in range(2 * H)]
    w2 = [gen.normal() / math.sqrt(H) for _ in range(H * H)]
    w3 = [gen.normal() / math.sqrt(H) for _ in range(H)]
    return w1 + [0.0] * H + w2 + [0.0] * H + w3 + [0.0]


def net_forward(P, x, tau):
    """out = W3 . tanh(W2 tanh(W1 [x, tau] + b1) + b2) + b3; the sums over the input index run in increasing order."""
    h1 = [math.tanh(P[2 * i] * x + P[2 * i + 1] * tau + P[2 * H + i]) for i in range(H)]
    h2 = []
    for j in range(H):
        s = P[O2 + H * H + j]
        base = O2 + j * H
        for i in range(H):
            s += P[base + i] * h1[i]
        h2.append(math.tanh(s))
    out = P[O3 + H]
    for j in range(H):
        out += P[O3 + j] * h2[j]
    return out, h1, h2


def net_backward(P, x, tau, h1, h2, d, G):
    """Adds the gradient of the loss to G given d = d loss / d out for one sample."""
    G[O3 + H] += d
    d2 = [0.0] * H
    for j in range(H):
        G[O3 + j] += d * h2[j]
        d2[j] = d * P[O3 + j] * (1.0 - h2[j] * h2[j])
    d1 = [0.0] * H
    for j in range(H):
        G[O2 + H * H + j] += d2[j]
        base = O2 + j * H
        for i in range(H):
            G[base + i] += d2[j] * h1[i]
            d1[i] += d2[j] * P[base + i]
    for i in range(H):
        da = d1[i] * (1.0 - h1[i] * h1[i])
        G[2 * H + i] += da
        G[2 * i] += da * x
        G[2 * i + 1] += da * tau


def draw_item(gen):
    """One training item: t = integer(T) + 1, cluster k = integer(3), x_0 = m_k + SD normal(), noise = normal(); returns (t, x_t, noise)."""
    t = gen.integer(T) + 1
    k = gen.integer(3)
    x0 = MODES[k] + SD * gen.normal()
    eps = gen.normal()
    a, b = ab_coeffs(t)
    return t, a * x0 + b * eps, eps


def gradient_check():
    gen = bookrng.generator(GRADCHECK)
    P = [v + 0.1 * gen.normal() for v in init_net(gen)]
    items = [draw_item(gen) for _ in range(4)]

    def loss(theta):
        return sum((net_forward(theta, x, t / T)[0] - e) ** 2 for t, x, e in items) / len(items)

    G = [0.0] * N_PARAMS
    for t, x, e in items:
        o, h1, h2 = net_forward(P, x, t / T)
        net_backward(P, x, t / T, h1, h2, 2.0 * (o - e) / len(items), G)
    fails, worst = 0, 0.0
    for i in range(N_PARAMS):
        up, dn = list(P), list(P)
        up[i] += FD_STEP
        dn[i] -= FD_STEP
        num = (loss(up) - loss(dn)) / (2.0 * FD_STEP)
        diff = abs(G[i] - num)
        worst = max(worst, diff)
        if diff > 1e-6 + 1e-6 * max(abs(G[i]), abs(num)):
            fails += 1
    return (f"The network from the stream {_seed(GRADCHECK)}: init_net, then 0.1 normal() added to every entry in order; a batch of four "
            f"items drawn item after item (t, cluster, x_0 and noise as in training); the loss is the mean of (out - noise)^2 with "
            f"tau = t / {T}; central differences of step {FD_STEP:g}. An entry fails when |analytic - numerical| exceeds 1e-6 + 1e-6 "
            f"max(|analytic|, |numerical|).\n\n"
            + _table(["parameters checked", "entries failing the gate", "worst absolute difference"],
                     [(N_PARAMS, fails, "below 1e-6" if worst < 1e-6 else f"{worst:.1e}")]))


# ------------------------------------------------------------------ training and sampling

@lru_cache(maxsize=None)
def evaluation_items():
    gen = bookrng.generator(EVAL)
    return tuple(draw_item(gen) for _ in range(EVAL_N))


def distances(P, items=None):
    """(mean (net - noise)^2, mean (exact - noise)^2, mean (net - exact)^2) over the items, the noise estimate being the network's output
    at (x_t, t / T) and the exact denoiser's."""
    items = items or evaluation_items()
    a = b = c = 0.0
    for t, x, e in items:
        n = net_forward(P, x, t / T)[0]
        s = exact_denoiser(x, t)[1]
        a += (n - e) ** 2
        b += (s - e) ** 2
        c += (n - s) ** 2
    k = len(items)
    return a / k, b / k, c / k


def sample_chain(predict, name=SAMPLE, n=SAMPLE_N):
    """Ancestral sampling. x_T = normal() for each of the n samples in order; then for t = T, T - 1, ..., 1: for each sample in order
    the noise estimate e = predict(x, t), x = (x - beta_t / b_t e) / sqrt(1 - beta_t) + s_t z, where s_t^2 = (1 - abar_(t-1)) beta_t /
    (1 - abar_t) and z = normal() is drawn (for every sample, in order) only when t > 1."""
    gen = bookrng.generator(name)
    ab, bs = alpha_bars(), betas()
    xs = [gen.normal() for _ in range(n)]
    for t in range(T, 0, -1):
        beta = bs[t - 1]
        coef = beta / math.sqrt(1.0 - ab[t])
        inv = 1.0 / math.sqrt(1.0 - beta)
        sd = math.sqrt((1.0 - ab[t - 1]) / (1.0 - ab[t]) * beta)
        for i in range(n):
            e = predict(xs[i], t)
            xs[i] = (xs[i] - coef * e) * inv + (sd * gen.normal() if t > 1 else 0.0)
    return xs


def one_jump(name=SAMPLE, n=SAMPLE_N):
    """x_T = normal() as in sample_chain, then x_0 = E[x_0 | x_T] at once."""
    gen = bookrng.generator(name)
    return [exact_denoiser(gen.normal(), T)[0] for _ in range(n)]


def read_out(xs):
    """(samples within NEAR of each centre, standard deviation of all samples)."""
    cnt = tuple(sum(1 for x in xs if abs(x - m) < NEAR) for m in MODES)
    mu = sum(xs) / len(xs)
    return cnt, math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def train(nudge=0.0):
    """Setup: init_net from the stream `RUN` with the first weight (the weight of x in hidden unit 0 of the first layer) moved by `nudge`.
    Step s = 1 .. STEPS: BATCH items drawn one after another with draw_item from the same stream, loss the mean of (out - noise)^2 with
    tau = t / T, the gradient of that mean, Adam (beta1 0.9, beta2 0.999, epsilon 1e-8, no weight decay) with rate LR (1 - (s - 1) / STEPS).
    Returns the log of (loss, optimum, distance) on the evaluation items at the LOG_STEPS, the parameters and the samples."""
    gen = bookrng.generator(RUN)
    P = init_net(gen)
    P[0] += nudge
    m, v = [0.0] * N_PARAMS, [0.0] * N_PARAMS
    log = {}
    for s in range(1, STEPS + 1):
        G = [0.0] * N_PARAMS
        for _ in range(BATCH):
            t, x, e = draw_item(gen)
            o, h1, h2 = net_forward(P, x, t / T)
            net_backward(P, x, t / T, h1, h2, 2.0 * (o - e) / BATCH, G)
        fm.adamw_step(P, G, m, v, s, LR * (1.0 - (s - 1) / STEPS), 0.0)
        if s in LOG_STEPS:
            log[s] = distances(P)
    xs = sample_chain(lambda x, t: net_forward(P, x, t / T)[0])
    return {"log": log, "P": P, "samples": xs}


def _job(nudge):
    return train(nudge)


@lru_cache(maxsize=None)
def all_runs():
    """The run and its four nudged copies, trained in four worker processes."""
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_job, [0.0, *stable.NUDGES], chunksize=1)


@lru_cache(maxsize=None)
def nominal_run():
    """The run without nudged copies (for the figures)."""
    return train(0.0)


def _shared(vals, fmt):
    return stable.text(float(vals[0]), [float(v) for v in vals[1:]], fmt)


def _int(x):
    return f"{x:.0f}"


def _fixed(d):
    return lambda x: f"{x:.{d}f}"


def training_table():
    runs = all_runs()
    rows = []
    for s in LOG_STEPS:
        rows.append((s, _shared([r["log"][s][0] for r in runs], _fixed(4)), _shared([r["log"][s][1] for r in runs], _fixed(4)),
                     _shared([r["log"][s][2] for r in runs], _fixed(4))))
    return (f"One run: the network of init_net from the stream {_seed(RUN)}, then batches of {BATCH} items drawn from the same stream in "
            f"the order of the chapter, Adam with beta1 0.9, beta2 0.999, epsilon 1e-8 and rate {LR:g} (1 - (s - 1) / {STEPS}) at step s. "
            f"After each listed step the network and the exact denoiser are scored on {EVAL_N} evaluation items ({_seed(EVAL)}, each drawn "
            f"as in training): the mean of (network - noise)^2, the mean of (exact - noise)^2, and the mean of (network - exact)^2. "
            f"Digits are those shared by the run and four copies with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12.\n\n"
            + _table(["step", "loss of the network", "loss of the exact denoiser", "mean (network - exact)^2"], rows))


def band_table():
    runs = all_runs()
    items = evaluation_items()
    rows = []
    for lo, hi in BANDS:
        sub = tuple(it for it in items if lo <= it[0] <= hi)
        per = [distances(r["P"], sub) for r in runs]
        rows.append((f"{lo} .. {hi}", len(sub), _shared([p[1] for p in per], _fixed(4)), _shared([p[0] for p in per], _fixed(4)),
                     _shared([p[2] for p in per], _fixed(4))))
    return ("The evaluation items of the table above, split by the step t of the item, after the last training step.\n\n"
            + _table(["steps t", "items", "loss of the exact denoiser", "loss of the network", "mean (network - exact)^2"], rows))


def sample_table():
    runs = all_runs()
    exact = read_out(sample_chain(lambda x, t: exact_denoiser(x, t)[1]))
    jump = read_out(one_jump())
    rows = []
    for name, ro in (("50 steps, exact denoiser", exact), ("one jump, exact denoiser", jump)):
        rows.append((name, ro[0][0], ro[0][1], ro[0][2], EVAL_N - sum(ro[0]), f"{ro[1]:.3f}"))
    reads = [read_out(r["samples"]) for r in runs]
    rows.append(("50 steps, trained network", *[_shared([rd[0][m] for rd in reads], _int) for m in range(3)],
                 _shared([SAMPLE_N - sum(rd[0]) for rd in reads], _int), _shared([rd[1] for rd in reads], _fixed(3))))
    return (f"{SAMPLE_N} samples from the stream {_seed(SAMPLE)}: first the {SAMPLE_N} starting values x_T = normal(), then, for t = {T}, "
            f"{T - 1}, ..., 2, one normal() for each sample in order (none at t = 1). The columns count the samples within {NEAR:g} of "
            f"each centre; 'elsewhere' is the rest; the spread is the population standard deviation of all samples (divisor the number of samples, not one less; the data's is {DATA_STD:.3f}). "
            f"The one jump replaces each x_T by E[x_0 | x_T] and stops. The network row keeps the digits shared with the four nudged copies.\n\n"
            + _table(["sampler", "near -3", "near 0", "near +3", "elsewhere", "spread"], rows))


# ------------------------------------------------------------------ data of the figures

def denoiser_curves_data():
    """x_t grid and the exact noise estimate and the nominal network's at t = 5, 20, 45, each on the grid of its own scale."""
    P = nominal_run()["P"]
    out = {}
    for t in (5, 20, 45):
        a, b = ab_coeffs(t)
        lim = 3.4 * a + 1.2 * b
        xs = [-lim + 2.0 * lim * i / 200 for i in range(201)]
        out[t] = (xs, [exact_denoiser(x, t)[1] for x in xs], [net_forward(P, x, t / T)[0] for x in xs])
    return out


def sample_histogram_data():
    """Bin edges (width 0.25 on [-5, 5]), the share of the nominal network's samples per bin, and the true share per bin (midpoint rule)."""
    xs = nominal_run()["samples"]
    edges = [-5.0 + 0.25 * i for i in range(41)]
    got = [0.0] * 40
    for x in xs:
        i = int((x + 5.0) // 0.25)
        if 0 <= i < 40:
            got[i] += 1.0 / len(xs)
    true = []
    for i in range(40):
        s = sum(marginal_density(edges[i] + (j + 0.5) * 0.005, 0) * 0.005 for j in range(50))
        true.append(s)
    return edges, got, true
