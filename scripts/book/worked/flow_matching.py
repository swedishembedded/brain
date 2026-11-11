# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Flow Matching and Rectified Flow".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: flow_matching.<name> -->` markers;
the `*_data` functions supply the numbers of the chapter's figures. The data are the three clusters of `gans` and the network, its
backward pass, its initialisation and the Adam update are those of `diffusion` and `first_models` (imported, not copied), so that the
velocity network of this chapter and the noise network of the last one differ only in what they are asked for.

* two straight lines that cross, and the one velocity a network can give at the crossing;
* the exact marginal velocity field of the three clusters, its three identities (the posterior mean, the score, the continuity
  equation) checked numerically, and the noised clusters along the straight path;
* Euler sampling with the exact field, against the exact transport map of one dimension, and a fourth-order reference;
* the straightness of the paths, the crossings among pairs, and the second round of rectified flow (reflow) with its exact field;
* the conversions between the velocity and the clean value, and the weight the velocity loss puts on the error of the clean value;
* the same network as the last chapter trained to predict the velocity, its error against the exact field, and the samples it gives;
* the timestep shift derived from redundant copies of a value, a toy in which the exact field is known for any number of copies,
  and the schedules of brain's flow-matching scheduler (static shift, empirical mu).

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.
Trained numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the
nudges of `stable.py`. The runs are trained in at most four worker processes.

Swedish Embedded AB implements solutions for generative models whose paths, velocity targets and sampling schedules are specified,
checked and made reproducible. If your team needs expertise in flow matching and image synthesis, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import struct
from functools import lru_cache

import bookrng
import diffusion
import first_models as fm
import gans
import stable

# ------------------------------------------------------------------ constants

MODES, SD, NEAR, DATA_STD = gans.MODES, gans.SD, gans.NEAR, gans.DATA_STD
H = diffusion.H                             # hidden units of each of the two hidden layers
STEPS, BATCH, LR = 5000, 64, 3e-3           # training: Adam, LR * (1 - (s - 1) / STEPS) at step s = 1 .. STEPS
LOG_STEPS = (500, 1000, 2000, 3000, 4000, 5000)
RUN, EVAL, SAMPLE = "flow_matching.run", "flow_matching.evaluation", "flow_matching.sample"
PAIRS, TARGETS, SHIFT = "flow_matching.pairs", "flow_matching.targets", "flow_matching.shift"
EVAL_N, SAMPLE_N, PAIR_N = 1000, 1000, 1000
SAMPLE_STEPS = (1, 2, 4, 8, 16, 32, 64)     # Euler steps of the sampling tables
NET_STEPS = (1, 2, 4, 8, 16, 32)            # the same for the trained network (a forward pass per step)
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
X_MAX = 4.6                                 # |data| < 4.6: the clusters are at 0, +-3 with standard deviation 0.25
E_ENERGY = 1.0 + (sum(m * m for m in MODES) / 3.0 + SD * SD)     # E (noise - x_0)^2 = 1 + E x_0^2 (independent, E noise = 0)
SHIFT_DIMS, SHIFT_FACTORS, SHIFT_STEPS = (1, 4, 16, 64), (1, 2, 4, 8), 8
KLEIN = dict(a1=8.73809524e-05, b1=1.89833333, a2=0.00016927, b2=0.45666666, knee=4300)

_table, _seed, _f, _g = diffusion._table, diffusion._seed, diffusion._f, diffusion._g


# ------------------------------------------------------------------ the exact field of the three clusters

def posterior_mean(x, a, b):
    """E[x_0 | x] when x = a x_0 + b noise and x_0 is one of the three clusters: for cluster k with v = a^2 SD^2 + b^2 the posterior
    mean of x_0 is m_k + a SD^2 / v (x - a m_k), weighted by N(x; a m_k, v); the weights are a softmax because v is shared."""
    v = a * a * SD * SD + b * b
    lw = [-0.5 * (x - a * m) ** 2 / v for m in MODES]
    top = max(lw)
    w = [math.exp(l - top) for l in lw]
    z = sum(w)
    return sum(wk * (m + a * SD * SD / v * (x - a * m)) for wk, m in zip(w, MODES)) / z


def field(x, s):
    """The marginal velocity u(x, s) = E[noise - x_0 | x_s = x] = (x - E[x_0 | x_s = x]) / s, for 0 < s <= 1."""
    return (x - posterior_mean(x, 1.0 - s, s)) / s


def density(x, s, d=1):
    """p_s(x): the mixture of N((1 - s) m_k, (1 - s)^2 SD^2 + s^2 / d) with weights 1/3; d = 1 for a single value, and d > 1 for the mean
    of d copies that each carry their own noise."""
    a, v = 1.0 - s, (1.0 - s) ** 2 * SD * SD + s * s / d
    return sum(math.exp(-0.5 * (x - a * m) ** 2 / v) for m in MODES) / (3.0 * math.sqrt(2.0 * math.pi * v))


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def data_cdf(x):
    return sum(_phi((x - m) / SD) for m in MODES) / 3.0


def transport(e):
    """T(e) = F^-1(Phi(e)): the increasing map that carries N(0, 1) onto the data, by 46 bisection steps on [-4.6, 4.6]."""
    p = _phi(e)
    lo, hi = -X_MAX, X_MAX
    for _ in range(46):
        mid = 0.5 * (lo + hi)
        if data_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reflow_field(x, s):
    """The marginal velocity of the second round: the pairs are (T(e), e), the line is x = (1 - s) T(e) + s e, which is increasing in e,
    so e is found by 52 bisection steps and the velocity is e - T(e). At s = 1 the line starts at e = x."""
    if s >= 1.0:
        return x - transport(x)
    lo, hi = (x - X_MAX * (1.0 - s)) / s, (x + X_MAX * (1.0 - s)) / s
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if (1.0 - s) * transport(mid) + s * mid < x:
            lo = mid
        else:
            hi = mid
    e = 0.5 * (lo + hi)
    return e - transport(e)


def euler(x, grid, f):
    """Explicit Euler along the sigma grid grid[0] > grid[1] > ...: x <- x + (grid[i + 1] - grid[i]) f(x, grid[i])."""
    for i in range(len(grid) - 1):
        x = x + (grid[i + 1] - grid[i]) * f(x, grid[i])
    return x


def uniform_grid(n):
    return [1.0 - i / n for i in range(n + 1)]


def rk4(x, n, f):
    """Classical fourth-order Runge-Kutta from sigma = 1 to 0 in n equal steps; the one evaluation at sigma = 0 (the last stage of
    the last step) is made at 1e-9 because the exact field is 0/0 there."""
    h = 1.0 / n
    for i in range(n):
        s = 1.0 - i * h
        k1 = f(x, s)
        k2 = f(x - 0.5 * h * k1, s - 0.5 * h)
        k3 = f(x - 0.5 * h * k2, s - 0.5 * h)
        k4 = f(x - h * k3, max(s - h, 1e-9))
        x -= h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


@lru_cache(maxsize=None)
def sample_noise():
    gen = bookrng.generator(SAMPLE)
    return tuple(gen.normal() for _ in range(SAMPLE_N))


@lru_cache(maxsize=None)
def sample_targets():
    return tuple(transport(e) for e in sample_noise())


def _counts(xs):
    near = tuple(sum(1 for x in xs if abs(x - m) < NEAR) for m in MODES)
    return near + (len(xs) - sum(near),)


def _distance(xs, ref):
    return sum(abs(a - b) for a, b in zip(xs, ref)) / len(xs)


# ------------------------------------------------------------------ two lines that cross

def two_lines():
    pairs = {"A": (3.0, -1.0), "B": (0.0, 2.0)}
    rows = []
    for s in (1.0, 0.75, 0.5, 0.25, 0.0):
        xa, xb = ((1.0 - s) * x0 + s * e for x0, e in pairs.values())
        rows.append((s, f"{xa:.2f}", f"{xb:.2f}", f"{xa - xb:.2f}"))
    return ("Two pairs (x_0, noise): A = (3, -1) and B = (0, 2). The point of a pair at noise fraction sigma is x_sigma = (1 - sigma) x_0 + sigma noise, "
            "and it moves with velocity noise - x_0: -4 for A and +2 for B.\n\n"
            + _table(["sigma", "x_sigma of A", "x_sigma of B", "difference"], rows)
            + "\n\nThe lines meet at sigma = 0.5, at x = 1.00. A network given (x, sigma) = (1.00, 0.5) must give one velocity; if the two pairs "
              "are equally likely the squared error is smallest at the mean of -4 and 2, which is -1.")


# ------------------------------------------------------------------ the marginal field and its identities

def _check_points():
    """24 points (s, x): s in {0.2, 0.5, 0.8}, x = (k - 4) (1 - s) + 0.37 (k mod 3) for k = 0 .. 7."""
    return [(s, (k - 4.0) * (1.0 - s) + 0.37 * (k % 3)) for s in (0.2, 0.5, 0.8) for k in range(8)]


def _identity_worst():
    """Largest differences over the 24 points: field against -(x + s score) / (1 - s) (score by a central difference of ln p_s, step
    1e-5), and the residual of the continuity equation d p / d s + d (p u) / d x (central differences, step 1e-5)."""
    h, w_score, w_cont = 1e-5, 0.0, 0.0
    for s, x in _check_points():
        score = (math.log(density(x + h, s)) - math.log(density(x - h, s))) / (2.0 * h)
        w_score = max(w_score, abs(field(x, s) + (x + s * score) / (1.0 - s)))
        ds = (density(x, s + h) - density(x, s - h)) / (2.0 * h)
        dx = (density(x + h, s) * field(x + h, s) - density(x - h, s) * field(x - h, s)) / (2.0 * h)
        w_cont = max(w_cont, abs(ds + dx))
    return w_score, w_cont


def field_table():
    rows = []
    for s in (0.1, 0.4, 0.7):
        for x in (-1.0, 0.25, 1.0, 2.5):
            x0 = posterior_mean(x, 1.0 - s, s)
            rows.append((s, _f(x, 2), _f(density(x, s), 4), _f(x0, 4), _f(field(x, s), 4)))
    ch5 = max(abs(posterior_mean(x, *diffusion.ab_coeffs(t)) - diffusion.exact_denoiser(x, t)[0])
              for t in (5, 20, 35) for x in (-1.0, 0.25, 1.0, 2.5))
    w_score, w_cont = _identity_worst()
    return ("The three clusters (centres -3, 0, 3, standard deviation 0.25, weight 1/3 each) along the straight path x_sigma = (1 - sigma) x_0 + "
            "sigma noise. For each sigma and x: the density p_sigma(x), the posterior mean E[x_0 | x_sigma = x] and the velocity "
            "u(x, sigma) = (x - E[x_0 | x_sigma = x]) / sigma.\n\n"
            + _table(["sigma", "x", "p_sigma(x)", "E[x_0 given x]", "u(x, sigma)"], rows)
            + "\n\nAt the coefficients (a_t, b_t) of the last chapter's chain the posterior mean agrees with `diffusion.exact_denoiser` "
              f"to {'below 1e-12' if ch5 < 1e-12 else f'{ch5:.1e} (CHECK FAILED)'} (t = 5, 20, 35 and the four values of x). At 24 further points "
              "(sigma in {0.2, 0.5, 0.8}, x = (k - 4)(1 - sigma) + 0.37 (k mod 3) for k = 0 .. 7) the velocity differs from -(x + sigma score)/(1 - sigma), "
              "with the score the central difference (step 1e-5) of ln p_s, by "
            + ("less than 1e-6" if w_score < 1e-6 else f"{w_score:.1e} (CHECK FAILED)")
            + ", and the continuity residual dp/ds + d(p u)/dx, by central differences of step 1e-5, is "
            + ("below 1e-6." if w_cont < 1e-6 else f"{w_cont:.1e} (CHECK FAILED)."))


def _count_modes(s, d=1):
    xs = [-6.0 + 0.01 * i for i in range(1201)]
    p = [density(x, s, d) for x in xs]
    return sum(1 for i in range(1, 1200) if p[i] > p[i - 1] and p[i] >= p[i + 1])


def merge_point(d=1):
    """(lo, hi): the sigma interval of width 1e-4 in which the number of modes of p_s falls from 3 to 1, by bisection from [0.3, 0.97]."""
    lo, hi = 0.3, 0.97
    assert _count_modes(lo, d) == 3 and _count_modes(hi, d) == 1
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        if _count_modes(mid, d) == 3:
            lo = mid
        else:
            hi = mid
    return lo, hi


def merge_shift_table():
    base = 0.5 * sum(merge_point(1))
    rows = []
    for d in SHIFT_DIMS:
        lo, hi = merge_point(d)
        rows.append((d, f"{0.5 * (lo + hi):.4f}", _f(shift(math.sqrt(d), base), 4)))
    return ("Where the three modes of the mean coordinate of d copies become one (the d copies share one data value and carry their own noise, so "
            "the mean coordinate is a mixture of N((1 - sigma) m_k, (1 - sigma)^2 0.25^2 + sigma^2 / d); modes on the grid -6, -5.99, ..., 6, "
            "bisection to a width of 1e-4, midpoint printed), against the shift with k = sqrt(d) of the single-value merge point "
            f"{base:.4f}.\n\n"
            + _table(["copies d", "merge point", "shifted single-value merge point"], rows))


def path_marginals():
    rows = []
    for s in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        a, sd = 1.0 - s, math.sqrt((1.0 - s) ** 2 * SD * SD + s * s)
        peak, valley = density(3.0 * a, s), density(1.5 * a, s)
        rows.append((s, _f(a, 2), _f(sd, 4), _f(3.0 * a / sd, 3), _count_modes(s), _g(valley / peak, 3)))
    lo, hi = merge_point()
    return ("The three clusters after the straight path to noise fraction sigma: each cluster is N((1 - sigma) m_k, (1 - sigma)^2 0.25^2 + sigma^2). "
            "'Spread' is the standard deviation of one cluster, 'gap / spread' the distance 3 (1 - sigma) between neighbouring centres over it, "
            "'valley / peak' the density halfway between two centres, 1.5 (1 - sigma), over the density at the centre 3 (1 - sigma). Modes are local "
            "maxima on the grid -6, -5.99, ..., 6.\n\n"
            + _table(["sigma", "1 - sigma", "spread", "gap / spread", "modes", "valley / peak"], rows)
            + f"\n\nBy bisection on the mode count (from the bracket [0.3, 0.97], halving until the interval is narrower than 1e-4) the three modes become one between sigma = {lo:.5f} and sigma = {hi:.5f}.")


# ------------------------------------------------------------------ sampling with the exact field

def euler_exact():
    eps, ref = sample_noise(), sample_targets()
    rows = [("exact map T(noise)", "0", *_counts(ref))]
    for n in SAMPLE_STEPS:
        xs = [euler(e, uniform_grid(n), field) for e in eps]
        rows.append((n, _g(_distance(xs, ref), 3), *_counts(xs)))
    worst = max(abs(rk4(e, 200, field) - t) for e, t in zip(eps[:20], ref[:20]))
    return (f"{SAMPLE_N} starting values x_1 = noise from the stream {_seed(SAMPLE)} (one normal() each, in order), carried from sigma = 1 to 0 "
            f"by n Euler steps on the grid s_i = 1 - i/n with the exact velocity u at the left end of each step. 'Distance' is the mean of "
            f"|x_0 - T(noise)|, with T = F^-1(Phi(noise)) the exact transport map of one dimension (F the cumulative distribution of the data, "
            f"46 bisection steps on the bracket [-4.6, 4.6], keeping the lower half when F(mid) < Phi(noise), and the midpoint of the last bracket as the answer); the counts are the samples within {NEAR:g} of each centre and the rest.\n\n"
            + _table(["Euler steps n", "distance", "near -3", "near 0", "near +3", "elsewhere"], rows)
            + "\n\nFor the first 20 starting values, 200 steps of classical fourth-order Runge-Kutta on the same field (the exact velocity is 0/0 at sigma = 0, so the one evaluation that would fall there, the last stage of the last step, is made at sigma = 1e-9) end within "
            + ("1e-6" if worst < 1e-6 else f"{worst:.1e} (CHECK FAILED)") + " of T(noise).")


# ------------------------------------------------------------------ straight paths and the second round

@lru_cache(maxsize=None)
def pair_data():
    gen = bookrng.generator(PAIRS)
    out = []
    for _ in range(PAIR_N):
        k = gen.integer(3)
        x0 = MODES[k] + SD * gen.normal()
        out.append((x0, gen.normal()))
    return tuple(out)


def _discordant(pairs):
    """The number of pairs of lines that cross inside 0 < s < 1: (x_0i - x_0j) (noise_i - noise_j) < 0."""
    n, c = len(pairs), 0
    for i in range(n):
        xi, ei = pairs[i]
        for j in range(i + 1, n):
            if (xi - pairs[j][0]) * (ei - pairs[j][1]) < 0.0:
                c += 1
    return c


def _field_energy(f, n_s=200):
    """int_0^1 E u(x_s, s)^2 ds for the field f's own marginals: midpoint rule with n_s values of s and, at each, step 0.01 on [-6, 6]."""
    tot = 0.0
    for j in range(n_s):
        s = (j + 0.5) / n_s
        acc = 0.0
        for i in range(1200):
            x = -6.0 + (i + 0.5) * 0.01
            acc += density(x, s) * f(x, s) ** 2 * 0.01
        tot += acc / n_s
    return tot


def _transport_energy():
    """E (noise - T(noise))^2 by the midpoint rule with step 0.005 on [-8, 8]."""
    h, tot = 0.005, 0.0
    for i in range(3200):
        e = -8.0 + (i + 0.5) * h
        tot += math.exp(-0.5 * e * e) / math.sqrt(2.0 * math.pi) * (e - transport(e)) ** 2 * h
    return tot


@lru_cache(maxsize=None)
def energies():
    return _field_energy(field), _transport_energy()


def straightness():
    pairs = pair_data()
    mono = tuple((transport(e), e) for _, e in pairs)
    cross_ind, cross_mono = _discordant(pairs), _discordant(mono)
    first, second = energies()
    total = PAIR_N * (PAIR_N - 1) // 2
    # the second-round field reproduces the straight chord: velocity noise - T(noise) at five values of s for every pair of the first 100
    worst = 0.0
    for x0, e in mono[:100]:
        for s in (0.1, 0.3, 0.5, 0.7, 0.9):
            worst = max(worst, abs(reflow_field((1.0 - s) * x0 + s * e, s) - (e - x0)))
    rows = [("independent (round 1)", cross_ind, _f(E_ENERGY, 4), _f(first, 4), _f(E_ENERGY - first, 4)),
            ("increasing map (round 2)", cross_mono, _f(second, 4), _f(second, 4), _f(0.0, 4))]
    return (f"{PAIR_N} pairs (x_0, noise) from the stream {_seed(PAIRS)}: for each pair k = integer(3), then x_0 = m_k + 0.25 normal(), then "
            f"noise = normal(). 'Round 2' keeps each noise and replaces its partner by T(noise). A pair of lines crosses inside the path when "
            f"(x_0i - x_0j)(noise_i - noise_j) < 0; there are {total} pairs of pairs. 'Chord energy' is E (noise - x_0)^2, for the independent "
            f"coupling 1 + E x_0^2 = {E_ENERGY:.4f} exactly, and for round 2 the midpoint rule with step 0.005 on [-8, 8]. 'Field energy' is the "
            f"integral over sigma of E u^2 for the field of that coupling's own marginals (round 1: midpoint rule over 200 values of sigma and "
            f"step 0.01 in x on [-6, 6]; round 2: equal to the chord energy because the chord is determined by x). Their difference is "
            f"the straightness S, the part of the chord's energy that no function of (x, sigma) can explain.\n\n"
            + _table(["coupling", "crossing pairs of lines", "chord energy", "field energy", "S"], rows)
            + "\n\nThe second-round field, found by bisection on the line, reproduces the chord's velocity at sigma = 0.1, 0.3, 0.5, 0.7, 0.9 for the "
              "first 100 pairs to " + ("below 1e-9." if worst < 1e-9 else f"{worst:.1e} (CHECK FAILED)."))


def reflow_table():
    eps, ref = sample_noise(), sample_targets()
    rows = []
    for n in (1, 2, 4, 8):
        a = [euler(e, uniform_grid(n), field) for e in eps]
        b = [euler(e, uniform_grid(n), reflow_field) for e in eps]
        d2 = _distance(b, ref)
        rows.append((n, _g(_distance(a, ref), 3), _counts(a)[3], _g(d2, 3) if d2 >= 1e-9 else "below 1e-9", _counts(b)[3]))
    return (f"The {SAMPLE_N} starting values of `{SAMPLE}` carried by n Euler steps on s_i = 1 - i/n, with the exact field of round 1 and "
            f"of round 2 (the second by bisection on the line, so a distance below 1e-9 is the solver's tolerance, not a trend). 'Distance' is the mean of "
            f"|x_0 - T(noise)| and 'elsewhere' the number of samples farther than {NEAR:g} from every centre; the map T itself leaves 3.\n\n"
            + _table(["steps n", "round 1: distance", "round 1: elsewhere", "round 2: distance", "round 2: elsewhere"], rows))


# ------------------------------------------------------------------ targets, conversions and weights

def targets_table():
    x0, e = 2.9, 0.7
    rows, worst = [], 0.0
    for s in (0.1, 0.5, 0.9):
        x = (1.0 - s) * x0 + s * e
        u = e - x0
        x0_u, e_u = x - s * u, x + (1.0 - s) * u
        worst = max(worst, abs(x0_u - x0), abs(e_u - e))
        rows.append((s, _f(x, 6), _f(u, 6), _f(x0_u, 6), _f(e_u, 6)))
    wrows = []
    for s in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        snr = ((1.0 - s) / s) ** 2
        wrows.append((s, _g(snr, 4), _g(snr, 4), _g(snr + 1.0, 4), _g(1.0 / (s * s), 4)))
    gen = bookrng.generator(TARGETS)
    rel = 0.0
    for _ in range(1000):
        s = gen.uniform()
        x0_, e_, d = gen.normal(), gen.normal(), 0.1 * gen.normal()
        x = (1.0 - s) * x0_ + s * e_
        err = (x - s * (e_ - x0_ + d) - x0_) ** 2          # error of the clean value implied by a velocity that is off by d
        rel = max(rel, abs(err - s * s * d * d) / (s * s * d * d))
    return (f"x_0 = {x0}, noise = {e}; x_sigma = (1 - sigma) x_0 + sigma noise and u = noise - x_0. The clean value recovered from (x_sigma, u) as x_sigma - sigma u "
            f"and the noise as x_sigma + (1 - sigma) u.\n\n"
            + _table(["sigma", "x_sigma", "u", "x_0 from u", "noise from u"], rows)
            + f"\n\nThe largest difference from the true x_0 or noise over the three rows is {'below 1e-12' if worst < 1e-12 else f'{worst:.1e} (CHECK FAILED)'}.\n\n"
              "The weight each loss puts on the squared error of the clean value it implies. Signal-to-noise ratio SNR = ((1 - sigma)/sigma)^2; the loss "
              "on the noise weighs by SNR, the last chapter's loss on the velocity of the circular path by SNR + 1, and the loss on the velocity u "
              "of the straight path by 1/sigma^2 = (1 + sqrt(SNR))^2.\n\n"
            + _table(["sigma", "SNR", "noise loss", "circular velocity loss", "straight velocity loss"], wrows)
            + f"\n\nCheck on 1000 draws from {_seed(TARGETS)} (per draw: sigma = uniform(), then x_0, noise and an error d = 0.1 normal(), each by "
              f"normal(), in this order): the squared error of the clean value implied by a velocity off by d equals sigma^2 d^2 with a largest "
              f"relative difference " + ("below 1e-9." if rel < 1e-9 else f"{rel:.1e} (CHECK FAILED)."))


# ------------------------------------------------------------------ the network

def draw_item(gen):
    """One training item: s = uniform(), cluster k = integer(3), x_0 = m_k + SD normal(), noise = normal(); returns (s, x_s, target) with
    x_s = (1 - s) x_0 + s noise and target = noise - x_0."""
    s = gen.uniform()
    k = gen.integer(3)
    x0 = MODES[k] + SD * gen.normal()
    e = gen.normal()
    return s, (1.0 - s) * x0 + s * e, e - x0


@lru_cache(maxsize=None)
def evaluation_items():
    gen = bookrng.generator(EVAL)
    return tuple(draw_item(gen) for _ in range(EVAL_N))


def distances(P, items=None):
    """(mean (net - target)^2, mean (exact - target)^2, mean (net - exact)^2): the network's output at (x_s, s) and the exact field's."""
    items = items or evaluation_items()
    a = b = c = 0.0
    for s, x, y in items:
        n = diffusion.net_forward(P, x, s)[0]
        u = field(x, s)
        a += (n - y) ** 2
        b += (u - y) ** 2
        c += (n - u) ** 2
    k = len(items)
    return a / k, b / k, c / k


def train(nudge=0.0):
    """Setup: diffusion.init_net from the stream `RUN` with the first weight moved by `nudge`; step i = 1 .. STEPS: BATCH items drawn one
    after another with draw_item from the same stream, loss the mean of (out - target)^2 with the network input (x_s, s), Adam (beta1 0.9,
    beta2 0.999, epsilon 1e-8, no weight decay) with rate LR (1 - (i - 1) / STEPS). Returns the log at LOG_STEPS, the parameters and the
    samples of the Euler sampler at NET_STEPS from the starting values of `SAMPLE`."""
    gen = bookrng.generator(RUN)
    P = diffusion.init_net(gen)
    P[0] += nudge
    m, v = [0.0] * diffusion.N_PARAMS, [0.0] * diffusion.N_PARAMS
    log = {}
    for i in range(1, STEPS + 1):
        G = [0.0] * diffusion.N_PARAMS
        for _ in range(BATCH):
            s, x, y = draw_item(gen)
            o, h1, h2 = diffusion.net_forward(P, x, s)
            diffusion.net_backward(P, x, s, h1, h2, 2.0 * (o - y) / BATCH, G)
        fm.adamw_step(P, G, m, v, i, LR * (1.0 - (i - 1) / STEPS), 0.0)
        if i in LOG_STEPS:
            log[i] = distances(P)
    return {"log": log, "P": P, "samples": net_samples(P)}


def net_samples(P):
    f = lambda x, s: diffusion.net_forward(P, x, s)[0]
    return {n: [euler(e, uniform_grid(n), f) for e in sample_noise()] for n in NET_STEPS}


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
    """The run without nudged copies (for the figures); the same computation as the first entry of all_runs."""
    return train(0.0)


def _shared(vals, fmt):
    return stable.text(float(vals[0]), [float(v) for v in vals[1:]], fmt)


def _fixed(d):
    return lambda x: f"{x:.{d}f}"


def _int(x):
    return f"{x:.0f}"


def training_table():
    runs = all_runs()
    rows = []
    for s in LOG_STEPS:
        rows.append((s, _shared([r["log"][s][0] for r in runs], _fixed(4)), _shared([r["log"][s][1] for r in runs], _fixed(4)),
                     _shared([r["log"][s][2] for r in runs], _fixed(4))))
    return (f"One run: the network of `diffusion.init_net` from the stream {_seed(RUN)}, input (x_sigma, sigma), then batches of {BATCH} items drawn from "
            f"the same stream in the order of the chapter, Adam with beta1 0.9, beta2 0.999, epsilon 1e-8 and rate {LR:g} (1 - (i - 1) / {STEPS}) "
            f"at step i. After each listed step the network and the exact field are scored on {EVAL_N} evaluation items ({_seed(EVAL)}, each "
            f"drawn as in training): the mean of (network - target)^2, the mean of (exact - target)^2, and the mean of (network - exact)^2. "
            f"Digits are those shared by the run and four copies with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12.\n\n"
            + _table(["step", "loss of the network", "loss of the exact field", "mean (network - exact)^2"], rows))


def band_table():
    runs = all_runs()
    items = evaluation_items()
    rows = []
    for lo, hi in BANDS:
        sub = tuple(it for it in items if lo <= it[0] < hi)
        per = [distances(r["P"], sub) for r in runs]
        rows.append((f"{lo:g} .. {hi:g}", len(sub), _shared([p[1] for p in per], _fixed(4)), _shared([p[0] for p in per], _fixed(4)),
                     _shared([p[2] for p in per], _fixed(4))))
    return ("The evaluation items of the table above, split by their noise fraction sigma (lower end included, upper end excluded), after the last "
            "training step.\n\n"
            + _table(["sigma", "items", "loss of the exact field", "loss of the network", "mean (network - exact)^2"], rows))


def net_sampling_table():
    runs = all_runs()
    eps, ref = sample_noise(), sample_targets()
    rows = []
    for n in NET_STEPS:
        ex = [euler(e, uniform_grid(n), field) for e in eps]
        cnt = [_counts(r["samples"][n]) for r in runs]
        dist = [_distance(r["samples"][n], ref) for r in runs]
        rows.append((n, _g(_distance(ex, ref), 3), _shared(dist, _fixed(3)), *[_shared([c[m] for c in cnt], _int) for m in range(4)]))
    return (f"The {SAMPLE_N} starting values of `{SAMPLE}` carried from sigma = 1 to 0 by n Euler steps on sigma_i = 1 - i/n, with the exact field and "
            f"with the trained network (input (x, s_i) at the left end of each step). 'Distance' is the mean of |x_0 - T(noise)| as before; the "
            f"counts are the network's samples within {NEAR:g} of each centre and the rest. The network columns keep the digits shared "
            f"with the four nudged copies.\n\n"
            + _table(["steps n", "exact: distance", "network: distance", "near -3", "near 0", "near +3", "elsewhere"], rows))


# ------------------------------------------------------------------ the timestep shift

def shift(k, s):
    """The static shift sigma' = k s / (1 + (k - 1) s): 0 stays 0, 1 stays 1, and a factor k > 1 moves every other value toward 1."""
    return k * s / (1.0 + (k - 1.0) * s)


def shift_exp(mu, s):
    """The exponential form e^mu / (e^mu + 1/s - 1) for 0 < s <= 1 and 0 at s = 0."""
    return 0.0 if s <= 0.0 else math.exp(mu) / (math.exp(mu) + 1.0 / s - 1.0)


def shift_map():
    rows = []
    for s in (0.1, 0.25, 0.5, 0.75, 0.9):
        rows.append((s, *[_f(shift(k, s), 4) for k in (2.0, 3.0, 4.0)]))
    snr = lambda s: ((1.0 - s) / s) ** 2
    d_err = max(abs(16.0 * snr(shift(4.0, s)) - snr(s)) / snr(s) for s in (0.1, 0.25, 0.5, 0.75, 0.9))
    e_err = max(abs(shift_exp(math.log(3.0), s) - shift(3.0, s)) for s in [i / 100.0 for i in range(101)])
    lam = lambda s: math.log(snr(s))
    l_err = max(abs(lam(shift(4.0, s)) - lam(s) + 2.0 * math.log(4.0)) for s in (0.1, 0.25, 0.5, 0.75, 0.9))
    return ("The shifted noise fraction sigma' = k sigma / (1 + (k - 1) sigma).\n\n"
            + _table(["sigma", "sigma', k = 2", "sigma', k = 3", "sigma', k = 4"], rows)
            + f"\n\nWith k = 4 and 16 redundant copies the effective signal-to-noise ratio 16 SNR(sigma') equals SNR(sigma) with a largest relative "
              f"difference {'below 1e-12' if d_err < 1e-12 else f'{d_err:.1e} (CHECK FAILED)'}; the logarithm of SNR moves by exactly -2 ln 4 = -2.7726 "
              f"(largest deviation {'below 1e-12' if l_err < 1e-12 else f'{l_err:.1e} (CHECK FAILED)'}); and the exponential form with mu = ln 3 equals the static "
              f"shift 3 on sigma = 0, 0.01, ..., 1 to {'below 1e-15' if e_err < 1e-15 else f'{e_err:.1e} (CHECK FAILED)'}.")


def _shift_grid(k, n):
    return [shift(k, s) for s in uniform_grid(n)]


@lru_cache(maxsize=None)
def shift_toy_samples():
    """For each d in SHIFT_DIMS, in order, SAMPLE_N values z = (d draws summed in draw order) / sqrt(d), standard normal, from `SHIFT`."""
    gen = bookrng.generator(SHIFT)
    out = {}
    for d in SHIFT_DIMS:
        zs = []
        for _ in range(SAMPLE_N):
            tot = 0.0
            for _ in range(d):
                tot += gen.normal()
            zs.append(tot / math.sqrt(d))
        out[d] = zs
    return out


def toy_field(d):
    """The velocity of the mean coordinate xbar of d copies of x_0 that each carry their own noise: xbar = (1 - s) x_0 + s noise_bar with
    noise_bar of variance 1/d, so the posterior mean of x_0 is posterior_mean(xbar, 1 - s, s / sqrt(d)) and the velocity is
    (xbar - E[x_0 | xbar]) / s. The other coordinates shrink as s (their velocity is their deviation over s)."""
    return lambda x, s: (x - posterior_mean(x, 1.0 - s, s / math.sqrt(d))) / s


def shift_toy():
    zs = shift_toy_samples()
    rows = []
    for k in SHIFT_FACTORS:
        row = [k]
        for d in SHIFT_DIMS:
            f, grid = toy_field(d), _shift_grid(float(k), SHIFT_STEPS)
            xs = [euler(z / math.sqrt(d), grid, f) for z in zs[d]]
            row.append(_g(_distance(xs, [transport(z) for z in zs[d]]), 3))
        rows.append(row)
    # the reduction to the mean coordinate against the full vector: d = 4, k = 2, the first sample of the d = 4 block
    d, grid = 4, _shift_grid(2.0, SHIFT_STEPS)
    gen = bookrng.generator(SHIFT)
    for _ in range(SAMPLE_N):
        gen.normal()                                           # the d = 1 block
    draws = [gen.normal() for _ in range(d)]                   # the noise of the four coordinates: x = noise at s = 1
    x = list(draws)
    for i in range(SHIFT_STEPS):
        s, sn = grid[i], grid[i + 1]
        x0hat = posterior_mean(sum(x) / d, 1.0 - s, s / math.sqrt(d))
        x = [c + (sn - s) * ((c - (1.0 - s) * x0hat) / s - x0hat) for c in x]
    reduced = euler(sum(draws) / d, grid, toy_field(d))
    worst = max(abs(c - reduced) for c in x)
    return (f"{SAMPLE_N} samples for each number d of copies, d = 1, 4, 16, 64 in this order, from the stream {_seed(SHIFT)}: for each sample d "
            f"normal() draws, summed in draw order and divided by sqrt(d), which gives z with a standard normal distribution; the mean noise "
            f"of the d copies is z / sqrt(d). The data x_0 is one value shared by all d coordinates, each coordinate has its own noise, so "
            f"the mean coordinate satisfies x = (1 - sigma) x_0 + sigma z / sqrt(d) and its exact velocity is (x - E[x_0 | x]) / sigma with the posterior mean taken for "
            f"noise of standard deviation sigma / sqrt(d). Eight Euler steps on the grid sigma_i' = k sigma_i / (1 + (k - 1) sigma_i), sigma_i = 1 - i/8 (the "
            f"factor k = 1 is the uniform grid). The entry is the mean of |x_0 - T(z)| over the samples, T the transport map of the data.\n\n"
            + _table(["k \\ d"] + [str(d) for d in SHIFT_DIMS], rows)
            + "\n\nFor d = 4, k = 2 and the first sample of that block, carrying all four coordinates with the exact velocity of each (each "
              "coordinate's velocity is (c - (1 - sigma) E[x_0 | mean]) / sigma - E[x_0 | mean]) ends at the value of the one-dimensional run on the "
              "mean coordinate, in all four coordinates, to " + ("below 1e-12." if worst < 1e-12 else f"{worst:.1e} (CHECK FAILED)."))


# ------------------------------------------------------------------ brain's schedules

def empirical_mu(tokens, steps):
    """The exponential shift strength of FLUX.2: above 4300 tokens the 200-step line a2 n + b2; below, the line through the 10-step value
    a1 n + b1 and the 200-step value, evaluated at the number of steps."""
    m200 = KLEIN["a2"] * tokens + KLEIN["b2"]
    if tokens > KLEIN["knee"]:
        return m200
    m10 = KLEIN["a1"] * tokens + KLEIN["b1"]
    a = (m200 - m10) / 190.0
    return a * steps + (m200 - 200.0 * a)


def klein_sigmas(steps, tokens):
    """linspace(1, 1/steps, steps) shifted by e^mu / (e^mu + 1/s - 1), mu = empirical_mu(tokens, steps), and a terminal 0."""
    mu = empirical_mu(tokens, steps)
    base = [1.0 - i * (1.0 - 1.0 / steps) / max(steps - 1, 1) for i in range(steps)]
    return [shift_exp(mu, s) for s in base] + [0.0]


def static_schedule(steps, k):
    """Z-Image's static schedule: linspace(1, 1/steps, steps) shifted by k, then a terminal 0."""
    base = [1.0 - i * (1.0 - 1.0 / steps) / max(steps - 1, 1) for i in range(steps)]
    return [shift(k, s) for s in base] + [0.0]


def _f32(x):
    """x rounded to the nearest 32-bit float (ties to even), returned as a Python float; applied after every operation below."""
    return struct.unpack("f", struct.pack("f", x))[0]


def ramp_spellings():
    """The ramp linspace(1, 1/n, n) in 32-bit arithmetic, two ways, for n = 1 .. 2000: (A) 1 - i * (1 - 1/n) / (max(n, 2) - 1) and (B) with
    step = (1/n - 1) / (n - 1), 1 + step * i. Every operation is rounded to 32 bits. Returns (count of n at which they differ, largest difference,
    the agreeing n below 60)."""
    differ, worst, agree = 0, 0.0, []
    for n in range(1, 2001):
        inv = _f32(1.0 / n)
        one_minus = _f32(1.0 - inv)
        den = float(max(n, 2) - 1)
        step = _f32(_f32(inv - 1.0) / _f32(n - 1.0)) if n > 1 else 0.0
        gap = 0.0
        for i in range(n):
            a = _f32(1.0 - _f32(_f32(_f32(float(i)) * one_minus) / den))
            b = 1.0 if n == 1 else _f32(1.0 + _f32(step * _f32(float(i))))
            gap = max(gap, abs(a - b))
        if gap > 0.0:
            differ += 1
            worst = max(worst, gap)
        elif n < 60:
            agree.append(n)
    return differ, worst, agree


def ramp_table():
    differ, worst, agree = ramp_spellings()
    return ("The first sigma of a 4-step or 50-step schedule is the same however the ramp linspace(1, 1/n, n) is spelled; in 32-bit arithmetic "
            "the two spellings, (A) 1 - ((i * (1 - 1/n)) / (n - 1)) (the product of i and the rounded 1 - 1/n first, then the quotient, then the subtraction) and "
            "(B) 1 + (step * i) with step = (1/n - 1) / (n - 1) rounded once and then multiplied by i, each operation rounded to nearest 32-bit "
            "float (i converted to 32 bits first), are compared over i = 0 .. n - 1 for each n = 1 .. 2000 (for n = 1 the single entry is 1; the maximum with 1 keeps n - 1 from being 0 in (A)).\n\n"
            + _table(["quantity", "value"], [("step counts n at which some entry differs", f"{differ} of 2000"),
                                             ("largest difference", f"{worst:.3e} ({worst / 2.0 ** -24:.0f} times 2^-24 = 5.96e-8)"),
                                             ("step counts below 60 at which they agree", ", ".join(str(n) for n in agree))]))


def brain_schedules():
    z = static_schedule(8, 3.0)
    z_exp = [shift_exp(math.log(3.0), 1.0 - i * (1.0 - 1.0 / 8) / 7) for i in range(8)] + [0.0]
    zdiff = max(abs(a - b) for a, b in zip(z, z_exp))
    rows = [(n, *[_f(empirical_mu(n, st), 4) for st in (4, 10, 50, 200)]) for n in (256, 1024, 2304, 4096, 4300, 4301, 8192)]
    k_rows = []
    for size in (512, 1024):
        tokens = (size // 16) ** 2
        s4 = klein_sigmas(4, tokens)
        k_rows.append((f"{size} x {size}", tokens, _f(math.exp(empirical_mu(tokens, 4)), 4), *[_f(v, 4) for v in s4[:4]]))
    return ("Z-Image's static schedule, 8 steps and shift 3: linspace(1, 1/8, 8) shifted by sigma' = 3 sigma / (1 + 2 sigma), then a terminal 0.\n\n"
            + _table(["step"] + [str(i) for i in range(9)], [("sigma", *[_f(v, 4) for v in z])])
            + f"\n\nThe exponential form with mu = ln 3 gives the same nine values to {'below 1e-15' if zdiff < 1e-15 else f'{zdiff:.1e} (CHECK FAILED)'}.\n\n"
              "The strength mu of FLUX.2's exponential shift as a function of the number of image tokens and the number of steps, "
              f"mu = a_2 n + b_2 above {KLEIN['knee']} tokens and the line through (10, a_1 n + b_1) and (200, a_2 n + b_2) below, with "
              f"a_1 = {KLEIN['a1']:.9g}, b_1 = {KLEIN['b1']:.9g}, a_2 = {KLEIN['a2']:.9g}, b_2 = {KLEIN['b2']:.9g}.\n\n"
            + _table(["tokens n", "4 steps", "10 steps", "50 steps", "200 steps"], rows)
            + "\n\nThe 4-step schedule of FLUX.2 Klein (linspace(1, 1/4, 4) shifted by e^mu / (e^mu + 1/sigma - 1), a terminal 0 after the fourth), with "
              "one token per 16 x 16 pixels, computed in float64; the first four entries are the values the model sees.\n\n"
            + _table(["image", "tokens", "e^mu", "sigma 0", "sigma 1", "sigma 2", "sigma 3"], k_rows))


# ------------------------------------------------------------------ data of the figures

def crossing_lines_data():
    """The first 12 pairs of `PAIRS` as lines (s, x_s) for s = 0 .. 1."""
    return [([0.0, 1.0], [x0, e]) for x0, e in pair_data()[:12]]


def trajectory_data():
    """For eight starting values, the path of the exact field (classical Runge-Kutta, 200 steps, recorded every 5) as (s, x), and the
    straight chord from (1, noise) to (0, T(noise))."""
    out = []
    for e in (-2.0, -1.2, -0.6, -0.2, 0.2, 0.6, 1.2, 2.0):
        xs, x, h = [e], e, 1.0 / 200
        for i in range(200):
            x = rk4_step(x, 1.0 - i * h, h)
            if (i + 1) % 5 == 0:
                xs.append(x)
        out.append(([1.0 - 5 * j / 200 for j in range(len(xs))], xs, [0.0, 1.0], [transport(e), e]))
    return out


def rk4_step(x, s, h):
    k1 = field(x, s)
    k2 = field(x - 0.5 * h * k1, s - 0.5 * h)
    k3 = field(x - 0.5 * h * k2, s - 0.5 * h)
    k4 = field(x - h * k3, max(s - h, 1e-9))
    return x - h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_curves_data():
    """x grid, exact field and nominal network's output at s = 0.2, 0.5, 0.8, each on the grid of its own scale."""
    P = nominal_run()["P"]
    out = {}
    for s in (0.2, 0.5, 0.8):
        lim = 3.4 * (1.0 - s) + 1.2 * s
        xs = [-lim + 2.0 * lim * i / 200 for i in range(201)]
        out[s] = (xs, [field(x, s) for x in xs], [diffusion.net_forward(P, x, s)[0] for x in xs])
    return out


def error_curves_data():
    """Number of Euler steps against the mean distance to T(noise): exact field, trained network (nominal run), second-round field."""
    eps, ref = sample_noise(), sample_targets()
    exact = [_distance([euler(e, uniform_grid(n), field) for e in eps], ref) for n in SAMPLE_STEPS]
    net = nominal_run()["samples"]
    return SAMPLE_STEPS, exact, NET_STEPS, [_distance(net[n], ref) for n in NET_STEPS]


def shift_curves_data():
    """s' against s for k = 1, 2, 3, 4, 8 on 0 .. 1, and the four step sigmas of the 512 x 512 FLUX.2 Klein schedule against 1 - i/4."""
    xs = [i / 100.0 for i in range(101)]
    curves = {k: [shift(float(k), s) for s in xs] for k in (1, 2, 3, 4, 8)}
    return xs, curves, [1.0 - i / 4 for i in range(4)], klein_sigmas(4, 1024)[:4]
