# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Samplers and Guidance".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: samplers_and_guidance.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. The three clusters, their exact denoiser, the 1000
starting values of the stream `flow_matching.sample` and the transport map T are those of `flow_matching` (imported, not copied), so
that every sampler of this chapter is scored against the same exact answer as the Euler sampler of the last one. The network and the
Adam update of the distillation example are those of `diffusion` and `first_models`.

* the deterministic step written two ways (a denoiser's guess and Euler on a rescaled variable), on the straight path and on the
  variance-preserving chain of `diffusion`;
* the family of steps that put fresh noise back, the split of the noise into a part that is removed and a part that is re-added, and
  its error against the deterministic step, with an exact denoiser and with one that is wrong;
* the log signal-to-noise ratio, in which the timestep shift is a translation, the coefficients of the exponential-integrator steps
  (exactness checks against numerical integrals), and the orders that the first and last steps of a multistep solver really run at;
* four solvers (Euler, Heun, DPM-Solver++(2M), UniPC) against the exact transport map, per evaluation of the field, on two grids, and
  with a field that is wrong by a stated amount;
* a two-class problem with exact conditional fields: the identities behind classifier-free guidance, the samples it produces against
  the density it is said to sample, a guidance ramp, the amplification of an error in one branch, and a network distilled to give the
  guided field in one evaluation.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.

Swedish Embedded AB implements solutions for sampling schedules and guidance of generative models that are specified, checked against
exact references and made reproducible. If your team needs expertise in diffusion and flow-matching samplers, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import struct
from functools import lru_cache

import bookrng
import diffusion
import first_models as fm
import flow_matching as fl
import stable

# ------------------------------------------------------------------ constants

MODES, SD = fl.MODES, fl.SD
NOISE = fl.sample_noise                         # the 1000 starting values of the stream `flow_matching.sample`
TARGET = fl.sample_targets                      # T(noise): the exact end point of every starting value
SAMPLE = fl.SAMPLE
START = "samplers_and_guidance.start"           # the 4000 starting values of the stochastic samplers
FRESH = "samplers_and_guidance.fresh"           # the fresh noise of the stochastic steps
ETA_N = 4000
GUIDE = "samplers_and_guidance.guidance"        # the starting values of the two-class problem
STUDENT = "samplers_and_guidance.student"       # the distilled network: initialisation and training items
STUDENT_EVAL = "samplers_and_guidance.student_eval"
NFES = (4, 8, 16, 32, 64)                       # evaluations of the field per sample
ETAS = (0.0, 0.5, 1.0)
ETA_STEPS = (4, 8, 16, 32, 64, 128)
UNIPC_TOP = struct.unpack("f", struct.pack("f", 1.0 - 1.0 / 1000.0))[0]     # 1 - 1/1000 rounded to a 32-bit float, as a number
MODEL_ERROR = (0.3, 0.6)                        # amplitudes of the stated error of a wrong denoiser
MU_G, S_G = (-1.0, 1.0), 0.6                    # the two classes: N(-1, 0.6^2) and N(+1, 0.6^2), equal weights
G_VALUES = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
GUIDE_N, GUIDE_RK = 1000, 100                   # samples of the guidance tables; steps of the Runge-Kutta reference
G_STUDENT = 3.0
STU_STEPS, STU_BATCH, STU_LR = 3000, 64, 3e-3
STU_NFES = (4, 8, 16, 32)
STU_EVAL_N = 1000

_table, _seed, _f, _g = diffusion._table, diffusion._seed, diffusion._f, diffusion._g


# ------------------------------------------------------------------ the field in the form of the denoiser's guess

def lam(s):
    """Log signal-to-noise ratio ln((1 - s)/s) of the straight path; -infinity at s = 1 and +infinity at s = 0."""
    if s >= 1.0:
        return -math.inf
    if s <= 0.0:
        return math.inf
    return math.log((1.0 - s) / s)


def data_guess(x, s):
    """E[x_0 | x_s = x] for the three clusters: the exact denoiser of `flow_matching`."""
    return fl.posterior_mean(x, 1.0 - s, s)


def velocity(guess):
    """The velocity field belonging to a denoiser `guess(x, s)`: (x - guess) / s, evaluated at s >= 1e-9 (0/0 at s = 0)."""
    return lambda x, s: (x - guess(x, max(s, 1e-9))) / max(s, 1e-9)


def endpoint(x, guess, n=200):
    """The accurate end point of the starting value x under the field of `guess`: fourth-order Runge-Kutta, n steps (`flow_matching.rk4`)."""
    return fl.rk4(x, n, velocity(guess))


def uniform_grid(n, top=1.0):
    return [top * (1.0 - i / n) for i in range(n + 1)]


def shifted(k, s):
    return k * s / (1.0 + (k - 1.0) * s)


def mean_distance(xs, ref):
    return sum(abs(a - b) for a, b in zip(xs, ref)) / len(xs)


# ------------------------------------------------------------------ the solvers, in terms of the denoiser's guess

def euler(x, grid, guess):
    """The Euler step of the last chapter, written with the guess: x <- (s'/s) x + (1 - s'/s) guess(x, s)."""
    for i in range(len(grid) - 1):
        s, t = grid[i], grid[i + 1]
        r = t / s
        x = r * x + (1.0 - r) * guess(x, s)
    return x


def heun(x, grid, guess):
    """Heun's method of Volume I on the velocity: an Euler step, the velocity at its end (at 1e-9 when the end is 0), the average."""
    u = velocity(guess)
    for i in range(len(grid) - 1):
        s, t = grid[i], grid[i + 1]
        k1 = u(x, s)
        k2 = u(x + (t - s) * k1, t)
        x = x + (t - s) * 0.5 * (k1 + k2)
    return x


def dpm2m(x, grid, guess):
    """DPM-Solver++(2M), midpoint form: one evaluation per step; the first step and the last are first order, and a difference
    of two guesses across an infinite gap in log signal-to-noise ratio is 0."""
    n = len(grid) - 1
    prev = None
    for i in range(n):
        s, t = grid[i], grid[i + 1]
        m0 = guess(x, s)
        if t == 0.0:
            return m0
        h = lam(t) - lam(s)
        e = math.exp(-h) - 1.0
        xn = (t / s) * x - (1.0 - t) * e * m0
        if prev is not None and i < n - 1:
            h0 = lam(s) - lam(prev[1])
            r0 = h0 / h
            d1 = (m0 - prev[0]) / r0 if math.isfinite(r0) else 0.0
            xn += -0.5 * (1.0 - t) * e * d1
        prev = (m0, s)
        x = xn
    return x


def phi_integrals(h, p):
    """I_j(h) = integral over u in [0, h] of e^-(h-u) u^j du for j = 0 .. p, by I_0 = 1 - e^-h and I_j = h^j - j I_(j-1)."""
    out = [1.0 - math.exp(-h)]
    for j in range(1, p + 1):
        out.append(h ** j - j * out[-1])
    return out


def b_vector(h, p):
    """b_j = I_j / (h^j (1 - e^-h)), j = 1 .. p: the right-hand sides of the exactness conditions."""
    big = phi_integrals(h, p)
    return [big[j] / (h ** j * big[0]) for j in range(1, p + 1)]


def solve(a, b):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(b)
    m = [list(row) + [bb] for row, bb in zip(a, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(m[i][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def unipc(x, grid, guess, order=2, correct=True):
    """UniPC (B(h) = e^h - 1 variant, data prediction), one evaluation per step. At step i: the guess at the current point is the
    new evaluation; a corrector of the order the last predictor ran at refines the current point; a predictor of order
    min(order, n - i, i + 1) moves to the next; the last step lands on the guess."""
    n = len(grid) - 1
    guesses, last_x, last_order = [], None, 0
    for i in range(n):
        s, t = grid[i], grid[i + 1]
        m = guess(x, s)
        if i > 0 and correct:
            s0 = grid[i - 1]
            h = lam(s) - lam(s0)
            p = last_order
            rks = [(lam(grid[i - 1 - k]) - lam(s0)) / h for k in range(1, p)] + [1.0]
            if not all(math.isfinite(r) for r in rks):
                # a history point at lambda = -infinity (sigma = 1) has weight 0, and the system leaves the new guess the weight b_1
                p, rks = 1, [1.0]
                rho = b_vector(h, 1)
            else:
                rho = [0.5] if p == 1 else solve([[r ** j for r in rks] for j in range(p)], b_vector(h, p))
            m0 = guesses[-1]
            c = (1.0 - s) * (1.0 - math.exp(-h))
            x = (s / s0) * last_x + c * m0
            for k in range(p - 1):
                x += c * rho[k] * (guesses[-2 - k] - m0) / rks[k]
            x += c * rho[p - 1] * (m - m0)
        guesses.append(m)
        if len(guesses) > order:
            guesses.pop(0)
        p = min(order, n - i, i + 1)
        if t == 0.0:
            xn = m
        else:
            h = lam(t) - lam(s)
            c = (1.0 - t) * (1.0 - math.exp(-h))
            xn = (t / s) * x + c * guesses[-1]
            if p == 2:
                r1 = (lam(grid[i - 1]) - lam(s)) / h
                xn += c * 0.5 * (guesses[-2] - guesses[-1]) / r1
        last_x, last_order = x, p
        x = xn
    return x


SOLVERS = (("Euler", euler, 1), ("Heun", heun, 2), ("DPM-Solver++(2M)", dpm2m, 1), ("UniPC", unipc, 1))


def run_solver(solver, nfe, guess, shift=None):
    """The 1000 end points of solver (a row of SOLVERS) with nfe evaluations of the field; the grid is uniform in sigma from 1 to 0,
    shifted by `shift` when given."""
    _, f, per_step = solver
    grid = uniform_grid(nfe // per_step)
    if shift is not None:
        grid = [shifted(shift, s) for s in grid]
    return [f(e, grid, guess) for e in NOISE()]


@lru_cache(maxsize=None)
def solver_errors():
    """{(name, nfe): mean distance to T(noise)} on the uniform grids, exact field."""
    ref = TARGET()
    return {(sv[0], nfe): mean_distance(run_solver(sv, nfe, data_guess), ref) for sv in SOLVERS for nfe in NFES}


# ------------------------------------------------------------------ cost of a request

def forward_counts():
    rows = [
        ("SDXL text2image", 30, 5.0, True, "`crates/sdxlunet/src/caps.rs`: steps 30, guidance 5.0 (1.0 disables guidance)"),
        ("Wan text-to-video", 50, 5.0, True, "`crates/wan/src/config.rs`: sample_steps 50, guide_scale 5.0"),
        ("FLUX.2 base", 50, 4.0, True, "`crates/flux2/src/pipeline.rs`: 50 steps, guidance 4.0"),
        ("FLUX.2 Klein (distilled)", 4, 1.0, False, "`resolved_steps`: fixed 4, no guidance branch"),
        ("FLUX.1 dev", 50, 3.5, False, "one forward; the guidance scalar is an input of the network"),
        ("Z-Image Turbo", 8, 0.0, False, "`crates/s3dit/src/caps.rs`: steps 8, guidance 0"),
    ]
    out = []
    for name, steps, g, two, note in rows:
        out.append((name, steps, f"{g:g}", 2 if two else 1, steps * (2 if two else 1), note))
    return ("Network evaluations (forward passes of the denoiser) per image, from each pipeline's default step count and the rule that "
            "decides whether a second, unconditional pass is made: evaluations = steps x passes per step.\n\n"
            + _table(["model", "steps", "guidance scale", "passes per step", "evaluations", "source"], out, "lrrrrl"))


# ------------------------------------------------------------------ the deterministic step, written two ways

def ddim_is_euler():
    rows = []
    worst = 0.0
    for x, s, t in ((2.5, 0.7, 0.5), (-1.0, 0.4, 0.3), (0.25, 0.9, 0.6), (1.0, 0.5, 0.1), (2.9, 0.2, 0.0)):
        x0 = data_guess(x, s)
        e = (x - (1.0 - s) * x0) / s
        ddim = (1.0 - t) * x0 + t * e
        eul = x + (t - s) * (x - x0) / s
        worst = max(worst, abs(ddim - eul))
        rows.append((_f(x, 2), _f(s, 1), _f(t, 1), _f(x0, 6), _f(e, 6), _f(ddim, 8), _f(eul, 8)))
    text = ("The deterministic step on the straight path (a = 1 - sigma, b = sigma), for the three clusters: from x at noise fraction sigma to "
            "sigma', with the denoiser's guess x0 = E[x_0 | x], the noise it implies e = (x - (1 - sigma) x0) / sigma, the step "
            "(1 - sigma') x0 + sigma' e, and the Euler step x + (sigma' - sigma)(x - x0)/sigma of the last chapter.\n\n"
            + _table(["x", "sigma", "sigma'", "guess x0", "implied e", "step from the guess", "Euler step"], rows)
            + f"\n\nThe two steps differ by {_below(worst, 1e-12)} over the five rows (a rounding error).\n\n")
    chain = []
    worst = 0.0
    for x, t, u in ((1.3, 50, 40), (-0.8, 40, 30), (2.1, 30, 20), (0.4, 20, 10), (2.9, 10, 0)):
        a, b = diffusion.ab_coeffs(t)
        a2, b2 = diffusion.ab_coeffs(u)
        x0, e = diffusion.exact_denoiser(x, t)
        ddim = a2 * x0 + b2 * e
        y, r, r2 = x / a, b / a, (b2 / a2)
        eul = a2 * (y + (r2 - r) * e)
        worst = max(worst, abs(ddim - eul))
        chain.append((t, u, _f(x, 2), _f(r, 4), _f(r2, 4), _f(ddim, 8), _f(eul, 8)))
    return text + ("The same on the variance-preserving chain of the diffusion chapter (T = 50, `diffusion.ab_coeffs` and "
                   "`diffusion.exact_denoiser`): the step a' x0 + b' e, against one Euler step of the variable y = x / a in "
                   "the noise level sigma_t = b / a, y' = y + (sigma_t' - sigma_t) e, and then x' = a' y'.\n\n"
                   + _table(["t", "t'", "x_t", "sigma_t = b/a", "sigma_t'", "step from the guess", "Euler in (y, sigma_t), times a'"], chain)
                   + f"\n\nThe two differ by {_below(worst, 1e-12)} over the five rows.")


# ------------------------------------------------------------------ the steps that put noise back

def nu_post(s, t):
    """Standard deviation of the exact reverse step of the straight path from sigma = s to t < s (at eta = 1)."""
    return t * math.sqrt(max(0.0, 1.0 - ((1.0 - s) * t / ((1.0 - t) * s)) ** 2))


def _posterior_gap():
    ab, bs = diffusion.alpha_bars(), diffusion.betas()
    worst = 0.0
    for t in range(2, diffusion.T + 1):
        a, b = diffusion.ab_coeffs(t)
        a2, b2 = diffusion.ab_coeffs(t - 1)
        nu2 = (b2 * math.sqrt(1.0 - (a * b2 / (a2 * b)) ** 2)) ** 2
        tilde = (1.0 - ab[t - 1]) / (1.0 - ab[t]) * bs[t - 1]
        worst = max(worst, abs(nu2 - tilde))
    return worst


def noise_split():
    s, t = 0.6, 0.4
    bt, bs = s, t
    top = nu_post(s, t)
    rows = []
    for eta in ETAS:
        nu = eta * top
        c = math.sqrt(bs * bs - nu * nu) / bt
        rows.append((f"{eta:g}", _f(nu, 6), _f(c, 6), _f(c * c * bt * bt + nu * nu, 10), _f(bs * bs, 10)))
    out = (f"The noise split of one step of the straight path from sigma = {s} to {t} (b = sigma): the standard deviation nu = eta nu_1 of the "
           f"fresh noise, with nu_1 = {top:.6f}; the weight kappa = sqrt(b'^2 - nu^2)/b on the noise the denoiser implies; and the variance "
           "kappa^2 b^2 + nu^2 of the new value given x_0, which must equal b'^2.\n\n"
           + _table(["eta", "nu", "kappa", "kappa^2 b^2 + nu^2", "b'^2"], rows) + "\n\n")
    rows, worst = [], 0.0
    for t0, t1 in ((50, 40), (40, 30), (30, 20), (20, 10), (10, 1)):
        a, b = diffusion.ab_coeffs(t0)
        a2, b2 = diffusion.ab_coeffs(t1)
        eta1 = b2 * math.sqrt(1.0 - (a * b2 / (a2 * b)) ** 2)
        r, r2 = b / a, b2 / a2
        up = r2 * math.sqrt((r * r - r2 * r2) / (r * r))
        worst = max(worst, abs(eta1 - a2 * up))
        down = math.sqrt(r2 * r2 - up * up)
        rows.append((t0, t1, _f(eta1, 8), _f(a2 * up, 8), _f(down, 6), _f(r2, 6)))
    return out + ("On the chain of the diffusion chapter: the noise of the eta = 1 step, nu_1 = b' sqrt(1 - (a b' / (a' b))^2), against a' "
                  "sigma_up with sigma_up = sigma_t' sqrt((sigma_t^2 - sigma_t'^2)/sigma_t^2) the 'up' of the ancestral Euler step in the noise "
                  "level sigma_t = b/a (and sigma_down = sqrt(sigma_t'^2 - sigma_up^2), the level it lands on before the noise is added).\n\n"
                  + _table(["t", "t'", "nu_1", "a' sigma_up", "sigma_down", "sigma_t'"], rows)
                  + f"\n\nThe two noise levels differ by {_below(worst, 1e-12)} over the five rows. For single steps (t' = t - 1, t = 2 .. 50) "
                    f"nu_1^2 differs from the variance tilde-beta_t = (1 - abar_(t-1)) beta_t / (1 - abar_t) of the exact reverse step of the "
                    f"diffusion chapter by {_below(_posterior_gap(), 1e-12)}.")


@lru_cache(maxsize=None)
def eta_noise():
    gen = bookrng.generator(START)
    return tuple(gen.normal() for _ in range(ETA_N))


def w1_quantiles():
    n = ETA_N
    out = []
    for k in range(n):
        p = (k + 0.5) / n
        lo, hi = -fl.X_MAX, fl.X_MAX
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            if fl.data_cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


@lru_cache(maxsize=None)
def quantiles():
    return tuple(w1_quantiles())


def w1(xs):
    """Mean absolute difference between the sorted samples and the quantiles of the data at (k + 1/2)/N."""
    return sum(abs(a - b) for a, b in zip(sorted(xs), quantiles())) / len(xs)


def wrong_guess(amp):
    """The denoiser with a stated error: the exact guess plus amp sigma sin(2 x + 3 sigma)."""
    return lambda x, s: data_guess(x, s) + amp * s * math.sin(2.0 * x + 3.0 * s)


def eta_sample(x, grid, guess, eta, gen):
    """Steps of the eta family from the guess and the noise it implies; one normal() per step except the last (which lands on the guess)."""
    for i in range(len(grid) - 1):
        s, t = grid[i], grid[i + 1]
        m = guess(x, s)
        if t == 0.0:
            return m
        e = (x - (1.0 - s) * m) / s
        nu = eta * nu_post(s, t)
        z = gen.normal() if nu > 0.0 else 0.0
        x = (1.0 - t) * m + math.sqrt(t * t - nu * nu) * e + nu * z
    return x


def eta_w1(n, eta, guess):
    gen = bookrng.generator(FRESH)
    grid = uniform_grid(n)
    return w1([eta_sample(e, grid, guess, eta, gen) for e in eta_noise()])


@lru_cache(maxsize=None)
def eta_errors():
    out = {}
    for amp in (0.0, *MODEL_ERROR):
        guess = data_guess if amp == 0.0 else wrong_guess(amp)
        for n in ETA_STEPS:
            for eta in ETAS:
                if amp > 0.0 and eta == 0.5:
                    continue
                out[(amp, n, eta)] = eta_w1(n, eta, guess)
    return out


def eta_table():
    err = eta_errors()
    floor = w1([fl.transport(e) for e in eta_noise()])
    rows = [(n, *[_g(err[(0.0, n, eta)], 3) for eta in ETAS]) for n in ETA_STEPS]
    first = (f"The eta family on the straight path with the exact denoiser: {ETA_N} samples, the starting values one normal() each from "
             f"{_seed(START)}, n uniform steps from sigma = 1 to 0 (n evaluations). The fresh noise is one normal() per sample and per "
             f"step except the last, in sample order and step order, from {_seed(FRESH)} (restarted for every cell). The score is the mean "
             f"of |sorted sample - quantile of the data| at the probabilities (k + 1/2)/N; the {ETA_N} exact end points T(noise) themselves "
             f"score {floor:.4f}, which is the noise of the score, not an error of the sampler.\n\n"
             + _table(["steps n", "eta = 0", "eta = 0.5", "eta = 1"], rows))
    rows = [(n, *[_g(err[(amp, n, eta)], 3) for amp in MODEL_ERROR for eta in (0.0, 1.0)]) for n in ETA_STEPS]
    second = ("The same score with a wrong denoiser: the exact guess plus amp sigma sin(2 x + 3 sigma) (library sin, radians), amp = "
              "0.3 and 0.6, for eta = 0 and 1. The noise stream and its order are those of the table above.\n\n"
              + _table(["steps n", "amp 0.3, eta = 0", "amp 0.3, eta = 1", "amp 0.6, eta = 0", "amp 0.6, eta = 1"], rows))
    return first + "\n\n" + second


def churn_table():
    rows = []
    for churn, n in ((5.0, 50), (5.0, 10), (20.0, 10)):
        gamma = min(churn / (n - 1.0), math.sqrt(2.0) - 1.0)
        rows.append((f"{churn:g}", n, _f(gamma, 5), _f(1.0 + gamma, 5), _f(1.01 * math.sqrt((1.0 + gamma) ** 2 - 1.0), 5)))
    return ("Churn, the other way to put noise back: before the step at noise level sigma the sampler raises it to sigma_hat = sigma (1 + gamma), with "
            "gamma = min(s_churn / (n - 1), sqrt(2) - 1) for n steps, by adding noise of standard deviation s_noise sqrt(sigma_hat^2 - sigma^2) "
            "to the variance-exploding latent, and then denoises from sigma_hat. For s_noise = 1.01:\n\n"
            + _table(["s_churn", "steps n", "gamma", "sigma_hat / sigma", "added noise / sigma"], rows))


# ------------------------------------------------------------------ the shift is a translation of the log signal-to-noise ratio

def log_snr_shift():
    rows, worst = [], 0.0
    for s in (0.1, 0.3, 0.5, 0.7, 0.9):
        t = shifted(3.0, s)
        d = lam(t) - (lam(s) - math.log(3.0))
        worst = max(worst, abs(d))
        rows.append((_f(s, 1), _f(t, 6), _f(lam(s), 6), _f(lam(t), 6), _f(lam(s) - lam(t), 6)))
    comp = max(abs(shifted(5.0, shifted(5.0, s)) - shifted(25.0, s)) for s in (0.1, 0.3, 0.5, 0.7, 0.9))
    top = UNIPC_TOP
    return ("The timestep shift with k = 3, sigma' = 3 sigma / (1 + 2 sigma), seen in the log signal-to-noise ratio lambda = ln((1 - sigma)/sigma) "
            "(natural logarithm).\n\n"
            + _table(["sigma", "sigma'", "lambda(sigma)", "lambda(sigma')", "lambda(sigma) - lambda(sigma')"], rows)
            + f"\n\nThe last column is ln 3 = {math.log(3.0):.6f} in every row (lambda(sigma') differs from lambda(sigma) - ln 3 by "
              f"{_below(worst, 1e-12)}). Shifting twice by k = 5 is shifting once by 25 (the two differ by {_below(comp, 1e-12)} at the five sigma). "
              f"At sigma = 1, lambda is -infinity and at sigma = 0 it is +infinity. The value 1 - 1/1000 rounded to a 32-bit float is "
              f"{top:.9f}, which a timestep rule trunc(1000 sigma) turns into 999, where sigma = 1 gives 1000.")


# ------------------------------------------------------------------ the coefficients of the exponential-integrator steps

def simpson(f, a, b, n=4000):
    h = (b - a) / n
    tot = f(a) + f(b)
    for k in range(1, n):
        tot += (4 if k % 2 else 2) * f(a + k * h)
    return tot * h / 3.0


def step_coefficients():
    rows = []
    worst = 0.0
    for h in (0.05, 0.3, 1.2):
        big = phi_integrals(h, 3)
        for j in range(4):
            q = simpson(lambda u: math.exp(-(h - u)) * u ** j, 0.0, h)
            worst = max(worst, abs(q - big[j]))
        b = b_vector(h, 3)
        rows.append((_f(h, 2), _f(b[0], 6), _f(b[1], 6), _f(b[2], 6)))
    text = ("The right-hand sides b_j = I_j / (h^j (1 - e^-h)) of the exactness conditions, with I_j the integral over [0, h] of e^-(h - u) u^j "
            "du computed by I_0 = 1 - e^-h, I_j = h^j - j I_(j-1); b_1 tends to 1/2 as h tends to 0.\n\n"
            + _table(["h", "b_1", "b_2", "b_3"], rows)
            + f"\n\nThe I_j agree with Simpson's rule (4000 panels) for j = 0 .. 3 and these three h to {_below(worst, 1e-10)}.\n\n")
    s1, s0, t = 0.8, 0.6, 0.45
    l1, l0, lt = lam(s1), lam(s0), lam(t)
    h = lt - l0
    x0 = 2.0
    out = []
    for name, m in (("linear", lambda l: 0.3 + 0.7 * l), ("quadratic", lambda l: 0.3 + 0.7 * l - 0.2 * l * l)):
        exact = (t / s0) * x0 + (1.0 - t) * simpson(lambda l: math.exp(-(lt - l)) * m(l), l0, lt)
        m0, m1, mt = m(l0), m(l1), m(lt)
        r1 = (l1 - l0) / h
        c = (1.0 - t) * (1.0 - math.exp(-h))
        base = (t / s0) * x0 + c * m0
        rho2 = solve([[1.0, 1.0], [r1, 1.0]], b_vector(h, 2))
        out.append((name, base - exact,
                    base + c * 0.5 * (m1 - m0) / r1 - exact,
                    base + c * b_vector(h, 1)[0] * (m1 - m0) / r1 - exact,
                    base + c * 0.5 * (mt - m0) - exact,
                    base + c * (rho2[0] * (m1 - m0) / r1 + rho2[1] * (mt - m0)) - exact))
    rows = [(n, *[f"{v:+.2e}" if abs(v) > 1e-13 else "below 1e-13" for v in vals]) for n, *vals in out]
    return text + (f"One step of each kind from sigma = 0.6 (x = 2) to 0.45 (h = {h:.4f}), with the history point at sigma = 0.8 and a guess that depends only on "
                   "lambda (so the exact step is (sigma'/sigma) x + (1 - sigma') I and the integral I is Simpson's rule with 4000 panels): "
                   "the error of the step against it. 'first order' keeps the guess; 'predictor 1/2' adds the difference with weight 1/2; "
                   "'predictor b_1' with weight b_1; 'corrector 1' uses the new guess with weight 1/2; 'corrector 2' solves the 2 x 2 system.\n\n"
                   + _table(["guess", "first order", "predictor 1/2", "predictor b_1", "corrector 1", "corrector 2"], rows))


def step_orders():
    n = 8
    rows = []
    gd = uniform_grid(n)
    gu = uniform_grid(n, UNIPC_TOP)
    lon = 0
    for i in range(n):
        s, t = gd[i], gd[i + 1]
        h = lam(t) - lam(s)
        if i == 0:
            eff = "1 (no earlier guess)"
        elif t == 0.0:
            eff = "1 (lands on the guess)"
        else:
            h0 = lam(s) - lam(gd[i - 1])
            eff = "1 (difference is 0: r0 infinite)" if not math.isfinite(h0) else "2"
        pu = min(2, n - i, i + 1)
        cu = "none" if i == 0 else str(min(2, n - (i - 1), i))
        rows.append((i, _f(s, 4), "-inf" if not math.isfinite(lam(s)) else _f(lam(s), 3), "inf" if not math.isfinite(h) else _f(h, 3), eff,
                     _f(gu[i], 6), pu, cu))
    return (f"The orders that two multistep solvers really run at on {n} steps. Left: DPM-Solver++(2M) on the grid sigma_i = 1 - i/{n} "
            "(lambda = ln((1 - sigma)/sigma), h = lambda_(i+1) - lambda_i); the order is 2 only where an earlier guess exists, the last step is "
            f"first order, and a guess at sigma = 1 sits at lambda = -infinity. Right: UniPC on sigma_i = (1 - 1/1000 in 32-bit) (1 - i/{n}), "
            "with the order of the predictor, min(2, n - i, i + 1), and of the corrector, the order the previous predictor ran at.\n\n"
            + _table(["i", "sigma_i", "lambda_i", "h_i", "effective order of DPM++(2M)", "UniPC sigma_i", "predictor order", "corrector order"],
                     rows, "lrrrlrrr"))


# ------------------------------------------------------------------ solvers against the exact map

def solver_table():
    err = solver_errors()
    rows = []
    for nfe in NFES:
        rows.append((nfe, *[_g(err[(sv[0], nfe)], 3) for sv in SOLVERS]))
    ref = TARGET()
    rk = mean_distance([fl.rk4(e, 200, fl.field) for e in NOISE()], ref)
    ratios = []
    for sv in SOLVERS:
        ratios.append(f"{sv[0]} " + ", ".join(f"{err[(sv[0], a)] / err[(sv[0], b)]:.1f}" for a, b in zip(NFES[:-1], NFES[1:])))
    return (f"The 1000 starting values of {_seed(SAMPLE)} carried from sigma = 1 to 0 with the exact field by four solvers, each given the stated "
            "number of evaluations of the field. Euler and DPM-Solver++(2M) take one step per evaluation on sigma_i = 1 - i/n; Heun takes "
            "n/2 steps of two evaluations (the second at 1e-9 on the last step); UniPC takes one step per evaluation, its "
            "history point at sigma = 1 (lambda = -infinity) given weight 0. Distance is the mean of |x_0 - T(noise)|, to three significant digits.\n\n"
            + _table(["evaluations", *[sv[0] for sv in SOLVERS]], rows)
            + f"\n\nFor comparison, 200 steps of classical Runge-Kutta (800 evaluations) score {rk:.1e}. The factor by which the distance falls when the "
              "evaluations double (4 to 8, 8 to 16, 16 to 32, 32 to 64): " + "; ".join(ratios) + ".")


def tail_table():
    ref = TARGET()
    rows = []
    for n in (16, 32, 64):
        errs = sorted(abs(a - b) for a, b in zip(run_solver(SOLVERS[3], n, data_guess), ref))
        tot = sum(errs)
        rows.append((n, _g(tot / len(errs), 3), _g(errs[len(errs) // 2], 3), f"{sum(errs[-50:]) / tot:.2f}", _g(errs[-1], 3)))
    return ("Where UniPC's error sits: for the evaluations n of the table above, the mean over the 1000 starting values, the median (the 501st "
            "smallest of the 1000 errors), the share of the total carried by the 50 worst starting values, and the largest error.\n\n"
            + _table(["evaluations", "mean", "median", "share of the 50 worst", "largest"], rows))


def predictor_check():
    worst = {}
    for n in (8, 16, 32):
        grid = uniform_grid(n)
        worst[n] = max(abs(unipc(e, grid, data_guess, correct=False) - dpm2m(e, grid, data_guess)) for e in NOISE())
    return ("UniPC with its corrector switched off against DPM-Solver++(2M) on the same grid and the same field, for the 1000 starting values of "
            f"{_seed(SAMPLE)}: the largest difference between end points is {_below(max(worst.values()), 1e-12)} at 8, 16 and 32 evaluations.")


def grid_table():
    rows = []
    ref = TARGET()
    for sv in SOLVERS:
        cells = []
        for nfe in (8, 16):
            for sh in (None, 3.0):
                cells.append(_g(mean_distance(run_solver(sv, nfe, data_guess, sh), ref), 3))
        rows.append((sv[0], *cells))
    return ("The same four solvers, exact field, on the uniform grid and on the grid shifted by k = 3 (sigma' = 3 sigma / (1 + 2 sigma) applied to "
            "every grid point), at 8 and 16 evaluations. Mean distance to T(noise).\n\n"
            + _table(["solver", "8, uniform", "8, shifted", "16, uniform", "16, shifted"], rows))


def model_error_table():
    ref = TARGET()
    rows = []
    for amp in MODEL_ERROR:
        guess = wrong_guess(amp)
        end = [endpoint(e, guess) for e in NOISE()]
        floor = mean_distance(end, ref)
        for nfe in (8, 16, 32, 64):
            cells = []
            for sv in (SOLVERS[0], SOLVERS[3]):
                xs = run_solver(sv, nfe, guess)
                cells += [_g(mean_distance(xs, end), 3), _g(mean_distance(xs, ref), 3)]
            rows.append((f"{amp:g}", nfe, _g(floor, 3), *cells))
    return ("Solver error against model error. The denoiser is the exact guess plus amp sigma sin(2 x + 3 sigma) with amp 0.3 and 0.6. 'Floor' "
            "is the mean distance between T(noise) and the accurate end point of the wrong field (Runge-Kutta, 200 steps): what no solver "
            "can remove. For Euler and UniPC (grids of the last table, uniform), the mean distance to that accurate end point ('solver') and to "
            "T(noise) ('total').\n\n"
            + _table(["amp", "evaluations", "floor", "Euler: solver", "Euler: total", "UniPC: solver", "UniPC: total"], rows))


def solver_curves_data():
    err = solver_errors()
    return list(NFES), {sv[0]: [err[(sv[0], n)] for n in NFES] for sv in SOLVERS}


# ------------------------------------------------------------------ two classes with exact conditional fields

S2 = S_G * S_G


def class_guess(x, s, c):
    """E[x_0 | x_s = x, class c]: x_0 ~ N(mu_c, S_G^2), so the posterior mean is mu_c + a S^2 / v (x - a mu_c), v = a^2 S^2 + s^2."""
    a = 1.0 - s
    v = a * a * S2 + s * s
    return MU_G[c] + a * S2 / v * (x - a * MU_G[c])


def class_posterior(x, s):
    """p(c | x_s = x) for the two equally likely classes, a softmax of the log densities of x_s given c (shared variance)."""
    a = 1.0 - s
    v = a * a * S2 + s * s
    lw = [-0.5 * (x - a * m) ** 2 / v for m in MU_G]
    top = max(lw)
    w = [math.exp(q - top) for q in lw]
    z = sum(w)
    return [q / z for q in w]


def plain_guess(x, s):
    """The unconditional denoiser: the class guesses weighted by p(c | x_s)."""
    w = class_posterior(x, s)
    return sum(wc * class_guess(x, s, c) for c, wc in enumerate(w))


def guided_guess(c, g):
    """Classifier-free guidance on the guess: plain + g (class - plain)."""
    def f(x, s):
        u = plain_guess(x, s)
        return u + g * (class_guess(x, s, c) - u)
    return f


def log_density(x, s, c=None):
    """ln p_s(x | c), or the unconditional ln p_s(x) when c is None."""
    a = 1.0 - s
    v = a * a * S2 + s * s
    lg = [-0.5 * (x - a * m) ** 2 / v - 0.5 * math.log(2.0 * math.pi * v) for m in MU_G]
    if c is not None:
        return lg[c]
    top = max(lg)
    return top + math.log(sum(0.5 * math.exp(q - top) for q in lg))


def central(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def guidance_identities():
    worst_avg = worst_score = worst_cls = 0.0
    for s in (0.2, 0.5, 0.8):
        for k in range(8):
            x = (k - 4) * (1.0 - s) * 0.5 + 0.37 * (k % 3)
            w = class_posterior(x, s)
            u = lambda guess: (x - guess(x, s)) / s
            avg = sum(wc * (x - class_guess(x, s, c)) / s for c, wc in enumerate(w))
            worst_avg = max(worst_avg, abs(avg - u(plain_guess)))
            sc = [central(lambda y: log_density(y, s, c), x) for c in (0, 1)]
            su = central(lambda y: log_density(y, s), x)
            for g in (3.0,):
                sg = su + g * (sc[1] - su)
                ug = u(guided_guess(1, g))
                worst_score = max(worst_score, abs(ug + (x + s * sg) / (1.0 - s)))
            cls = central(lambda y: math.log(class_posterior(y, s)[1]), x)
            worst_cls = max(worst_cls, abs((sc[1] - su) - cls))
    return ("Identities of the two-class problem (classes N(-1, 0.6^2) and N(+1, 0.6^2), equal weights; noised classes N((1 - sigma) mu_c, "
            "(1 - sigma)^2 0.36 + sigma^2)), at the 24 points sigma in {0.2, 0.5, 0.8}, x = 0.5 (k - 4)(1 - sigma) + 0.37 (k mod 3) for "
            "k = 0 .. 7, with scores taken as central differences of the log densities (step 1e-5):\n\n"
            "| identity | largest difference |\n|---|---:|\n"
            f"| unconditional velocity = sum over c of p(c given x) times the class velocity | {_crit(worst_avg, 1e-12)} |\n"
            f"| guided velocity (class 1, g = 3) = -(x + sigma s_g)/(1 - sigma) with s_g = s + g (s_1 - s) | {_crit(worst_score, 1e-6)} |\n"
            f"| s_1 - s = d/dx ln p(class 1 given x) | {_crit(worst_cls, 1e-6)} |")


def _below(v, bound):
    """'less than <bound>' when the check passes, otherwise the value and a failure marker (a failed check must not print quietly)."""
    return f"less than {bound:.0e}" if v < bound else f"{v:.1e} (CHECK FAILED)"


def _crit(v, bound):
    return f"below {bound:.0e}" if v < bound else f"{v:.1e} (CHECK FAILED)"


def post_one(x):
    """p(class 1 | x_0 = x) at the data end: a logistic function of the two class densities."""
    return class_posterior(x, 0.0)[1]


@lru_cache(maxsize=None)
def guide_noise():
    gen = bookrng.generator(GUIDE)
    return tuple(gen.normal() for _ in range(GUIDE_N))


def sample_guided(g):
    """End points for class 1 under the guided field with a constant scale g, or with a scale function g(sigma)."""
    scale = g if callable(g) else (lambda s: g)

    def field(x, s):
        gg = scale(s)
        u = plain_guess(x, s)
        return (x - (u + gg * (class_guess(x, s, 1) - u))) / s
    return [fl.rk4(e, GUIDE_RK, field) for e in guide_noise()]


@lru_cache(maxsize=None)
def guided_samples():
    return {g: tuple(sample_guided(g)) for g in G_VALUES}


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    return m, sd, sum(1 for x in xs if x < 0.0) / n, sum(post_one(x) for x in xs) / n


def tilted_stats(g):
    """Mean, standard deviation, share below 0 and mean p(class 1 | x) of the density proportional to p(x | 1)^g p(x)^(1 - g), by the
    Riemann sum over 4001 points of [-5, 5]."""
    xs = [-5.0 + 10.0 * k / 4000 for k in range(4001)]
    ps = [math.exp(g * log_density(x, 0.0, 1) + (1.0 - g) * log_density(x, 0.0)) for x in xs]
    z = sum(ps)
    m = sum(p * x for p, x in zip(ps, xs)) / z
    sd = math.sqrt(sum(p * (x - m) ** 2 for p, x in zip(ps, xs)) / z)
    return m, sd, sum(p for p, x in zip(ps, xs) if x < 0.0) / z


def guidance_table():
    rows = []
    for g in G_VALUES:
        m, sd, wrong, cls = stats(guided_samples()[g])
        tm, tsd, twrong = tilted_stats(g)
        rows.append((f"{g:g}", _f(m, 3), _f(sd, 3), _f(wrong, 3), _f(cls, 4), _f(tm, 3), _f(tsd, 3), _f(twrong, 3)))
    xs = [fl.rk4(e, 2 * GUIDE_RK, lambda x, s: (x - guided_guess(1, 8.0)(x, max(s, 1e-9))) / max(s, 1e-9)) for e in guide_noise()]
    gap = max(abs(a - b) for a, b in zip(xs, guided_samples()[8.0]))
    return (f"Class 1 (N(+1, 0.6^2)) under the guided field u_g = u + g (u_1 - u) with the exact fields of both branches: {GUIDE_N} starting values "
            f"(one normal() each, from {_seed(GUIDE)}) carried from sigma = 1 to 0 by {GUIDE_RK} steps of classical Runge-Kutta. 'Wrong side' is the "
            "share of end points below 0; 'class' is the mean of p(class 1 | x_0), the probability a classifier with the exact densities "
            "gives to the right class. The last three columns are the same statistics for the density proportional to "
            "p(x | class 1)^g p(x)^(1 - g), computed by a Riemann sum over 4001 points of [-5, 5]. The class itself has mean 1, standard "
            "deviation 0.6 and wrong-side share 0.048.\n\n"
            + _table(["g", "mean", "std", "wrong side", "class", "density: mean", "density: std", "density: wrong side"], rows)
            + f"\n\nDoubling the steps of the reference (200) moves no end point of g = 8 by more than {gap:.0e}.")


def ramp_table():
    big = 5.0
    variants = (("constant 5", lambda s: big),
                ("1 at noise, 5 at data: 1 + 4 (1 - sigma)", lambda s: 1.0 + (big - 1.0) * (1.0 - s)),
                ("5 at noise, 1 at data: 1 + 4 sigma", lambda s: 1.0 + (big - 1.0) * s),
                ("5 only below sigma = 0.5", lambda s: big if s < 0.5 else 1.0),
                ("5 only above sigma = 0.5", lambda s: big if s > 0.5 else 1.0))
    rows = []
    for name, sc in variants:
        m, sd, wrong, cls = stats(sample_guided(sc))
        rows.append((name, _f(m, 3), _f(sd, 3), _f(wrong, 3), _f(cls, 4)))
    return ("Guidance on a schedule, class 1, the starting values and the Runge-Kutta reference of the last table. The scale at noise fraction "
            "sigma is given in the first column; a scale of 1 is the plain class field.\n\n"
            + _table(["scale as a function of sigma", "mean", "std", "wrong side", "class"], rows, "lrrrr"))


def error_amplification():
    amp = 0.1
    base = {}
    rows = []
    for g in (1.0, 2.0, 3.0, 5.0, 8.0):
        clean = guided_samples().get(g)
        if clean is None:
            clean = tuple(sample_guided(g))
        def field(x, s, g=g):
            u = plain_guess(x, s)
            c = class_guess(x, s, 1) + amp * s * math.sin(3.0 * x + 2.0 * s)
            return (x - (u + g * (c - u))) / s
        pert = [fl.rk4(e, GUIDE_RK, field) for e in guide_noise()]
        d = sum(abs(a - b) for a, b in zip(pert, clean)) / len(pert)
        base.setdefault("d1", d)
        rows.append((f"{g:g}", _g(d, 3), f"{d / base['d1']:.1f}"))
    return ("An error in one branch. The class-1 guess is replaced by the exact one plus 0.1 sigma sin(3 x + 2 sigma) (library sin); the plain guess "
            "stays exact. For each g, the mean distance between the end points of the wrong field and the right one (the same starting values "
            "and reference as the guidance table), and that distance relative to g = 1.\n\n"
            + _table(["g", "mean displacement", "relative to g = 1"], rows))


def guidance_curves_data():
    """Histograms of the end points (bin width 0.25 on [-2.5, 5]) for g = 0, 1, 3 and 8, and the density of class 1."""
    bins = [-2.5 + 0.25 * k for k in range(31)]
    mids = [0.5 * (a + b) for a, b in zip(bins[:-1], bins[1:])]
    curves = {}
    for g in (0.0, 1.0, 3.0, 8.0):
        xs = guided_samples()[g]
        cnt = [0] * 30
        for x in xs:
            k = int((x + 2.5) // 0.25)
            if 0 <= k < 30:
                cnt[k] += 1
        curves[g] = [c / (len(xs) * 0.25) for c in cnt]
    cls = [math.exp(log_density(x, 0.0, 1)) for x in mids]
    return mids, curves, cls


# ------------------------------------------------------------------ distilling the guided field into one network

def teacher_velocity(x, s):
    return (x - guided_guess(1, G_STUDENT)(x, s)) / s


def student_item(gen):
    s = gen.uniform()
    x0 = MU_G[1] + S_G * gen.normal()
    e = gen.normal()
    x = (1.0 - s) * x0 + s * e
    return s, x, teacher_velocity(x, s)


@lru_cache(maxsize=None)
def student_eval_items():
    gen = bookrng.generator(STUDENT_EVAL)
    return tuple(student_item(gen) for _ in range(STU_EVAL_N))


def student_train(nudge=0.0):
    """Setup: diffusion.init_net from the stream STUDENT with the first weight moved by `nudge`; step i = 1 .. STU_STEPS: STU_BATCH items from
    the same stream (sigma, x_0, noise by uniform(), normal(), normal(), in this order), the loss the mean of (output - teacher velocity)^2,
    Adam (beta1 0.9, beta2 0.999, epsilon 1e-8) with rate STU_LR (1 - (i - 1) / STU_STEPS)."""
    gen = bookrng.generator(STUDENT)
    P = diffusion.init_net(gen)
    P[0] += nudge
    m, v = [0.0] * diffusion.N_PARAMS, [0.0] * diffusion.N_PARAMS
    for i in range(1, STU_STEPS + 1):
        G = [0.0] * diffusion.N_PARAMS
        for _ in range(STU_BATCH):
            s, x, y = student_item(gen)
            o, h1, h2 = diffusion.net_forward(P, x, s)
            diffusion.net_backward(P, x, s, h1, h2, 2.0 * (o - y) / STU_BATCH, G)
        fm.adamw_step(P, G, m, v, i, STU_LR * (1.0 - (i - 1) / STU_STEPS), 0.0)
    return P


def _job(nudge):
    return student_measure(student_train(nudge))


def teacher_states():
    """The states of 20 Euler steps of the guided teacher from the 1000 starting values, at sigma = 0.9, 0.5 and 0.1 (steps 2, 10, 18)."""
    states = {2: [], 10: [], 18: []}
    for e in guide_noise():
        x = e
        for i in range(20):
            s, t = 1.0 - i / 20, 1.0 - (i + 1) / 20
            if i in states:
                states[i].append((s, x))
            x = x + (t - s) * teacher_velocity(x, s)
    return states


def student_measure(P):
    f = lambda x, s: diffusion.net_forward(P, x, s)[0]
    out = {"fit": sum((f(x, s) - y) ** 2 for s, x, y in student_eval_items()) / STU_EVAL_N}
    st = teacher_states()
    out["visited"] = {i: sum((f(x, s) - teacher_velocity(x, s)) ** 2 for s, x in st[i]) / len(st[i]) for i in st}
    ref = [fl.rk4(e, GUIDE_RK, teacher_velocity) for e in guide_noise()]
    out["dist"] = {}
    out["stats"] = {}
    for nfe in STU_NFES:
        xs = [fl.euler(e, fl.uniform_grid(nfe), f) for e in guide_noise()]
        out["dist"][nfe] = mean_distance(xs, ref)
        if nfe == STU_NFES[-1]:
            out["stats"] = stats(xs)
    return out


@lru_cache(maxsize=None)
def student_runs():
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_job, [0.0, *stable.NUDGES], chunksize=1)


def _shared(vals, fmt):
    return stable.text(float(vals[0]), [float(v) for v in vals[1:]], fmt)


def _fixed(d):
    return lambda x: f"{x:.{d}f}"


def student_table():
    runs = student_runs()
    ref = [fl.rk4(e, GUIDE_RK, teacher_velocity) for e in guide_noise()]
    rows = []
    for nfe in STU_NFES:
        teacher = [fl.euler(e, fl.uniform_grid(nfe // 2), teacher_velocity) for e in guide_noise()]
        rows.append((nfe, nfe // 2, _g(mean_distance(teacher, ref), 3), nfe, _shared([r["dist"][nfe] for r in runs], _fixed(3))))
    t_stats = stats(ref)
    s_stats = [r["stats"] for r in runs]
    fit = [_shared([r["fit"] for r in runs], _fixed(4))]
    vis = [_shared([r["visited"][i] for r in runs], _fixed(4)) for i in (2, 10, 18)]
    return (f"Guidance distilled into one network. Teacher: the guided field of class 1 with g = {G_STUDENT:g} (two evaluations per step). Student: the "
            f"network of `diffusion.init_net` from the stream {_seed(STUDENT)}, input (x_sigma, sigma), trained {STU_STEPS} steps of {STU_BATCH} items "
            f"(sigma = uniform(), x_0 = 1 + 0.6 normal(), noise = normal(), per item in this order, then x_sigma = (1 - sigma) x_0 + sigma noise) "
            f"on the teacher's velocity at x_sigma, Adam with beta1 0.9, beta2 0.999, epsilon 1e-8 and rate 0.003 (1 - (i - 1)/{STU_STEPS}). Digits "
            "are those shared by the run and four copies with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12. The score is the "
            "mean of |x_0 - reference| over the 1000 starting values of the guidance tables, the reference being the teacher with 100 "
            "Runge-Kutta steps; Euler steps on the uniform grid for both.\n\n"
            + _table(["evaluations", "teacher: steps", "teacher: distance", "student: steps", "student: distance"], rows)
            + f"\n\nAfter 32 evaluations the teacher's end points have mean {t_stats[0]:.3f}, standard deviation {t_stats[1]:.3f} and the student's "
              f"{_shared([s[0] for s in s_stats], _fixed(3))} and {_shared([s[1] for s in s_stats], _fixed(3))}. The student's mean squared error "
              f"against the teacher's velocity is {fit[0]} on {STU_EVAL_N} fresh items of the training distribution ({_seed(STUDENT_EVAL)}), and on the "
              f"states the teacher's own 20-step trajectories visit it is {vis[0]} at sigma = 0.9, {vis[1]} at 0.5 and {vis[2]} at 0.1.")
