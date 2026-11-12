# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "World Models".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: world_models.<name> -->`
markers; the functions with a leading underscore supply the numbers of the chapter's figures, so a picture and the table beside it
cannot disagree.

* a world model that is two integers: the seed and the walk of a toroidal square, with the 64-bit FNV-1a hash that seeds it;
* a learned linear dynamics model of a cart on a track, its one-step error against its error after many free-running steps;
* a latent action recovered by clustering the displacements between consecutive frames, and how sensor noise blurs the codes;
* exact probabilities of a valid frame when six tokens that depend on each other are decoded in one pass, in a few, and one by one,
  and the schedule of tokens that a cosine mask ratio leaves to predict;
* the conditioners of an EDM denoiser at the noise levels of a three-step sampler;
* a denoiser that knows its world exactly (two futures, one more likely), run with one to twenty evaluations from the two starts;
* parameter counts and the work of one forward pass of a UNet world model and of a tokenizer plus an action-conditioned
  transformer, counted from their configs;
* the integer frame budget of a fixed-timestep loop and the limit cycle of its quality controller;
* a policy searched inside the learned model of the cart, and what it scores in the real cart.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, exp, log, tanh, cos, erf) unless a block says
otherwise. Sums are accumulated left to right in index order. Random numbers come from the book's SplitMix64 (`bookrng.generator`);
every stream name and seed is printed by the block that uses it. Nothing here is a chaotic training run: the fits are solved
directly, the clusters are a discrete procedure and the policy is a grid search, so the digits printed are the digits of any correct
implementation. A FLOP is one multiply or one add, so a multiply-add is two.

Swedish Embedded AB implements solutions for learned simulators that can be steered in real time. If your team needs expertise in
world models, action-conditioned video generation and interactive inference, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng
import images as im

_table = im._table

M64 = (1 << 64) - 1


def _seed(name):
    return f"0x{bookrng.seed_of(name):016x}"


def _int(n):
    return f"{n:,}"


# ------------------------------------------------------------------------------------------- a world model that is two integers

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def _fnv_seed(frames_bits, actions):
    """FNV-1a (64-bit) of: the number of frame values (8 bytes, little endian), each frame value's 32-bit pattern (4 bytes, little
    endian), the number of actions (8 bytes), each action (4 bytes). Returns (hash, px, py) with px = h & 63 and py = (h >> 32) & 63."""
    h = FNV_OFFSET

    def eat(data):
        nonlocal h
        for b in data:
            h = ((h ^ b) * FNV_PRIME) & M64

    eat(len(frames_bits).to_bytes(8, "little"))
    for f in frames_bits:
        eat(f.to_bytes(4, "little"))
    eat(len(actions).to_bytes(8, "little"))
    for a in actions:
        eat(a.to_bytes(4, "little"))
    return h, h & 63, (h >> 32) & 63


_MOVES = {0: (0, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}
_NAMES = {0: "noop", 1: "up", 2: "left", 3: "down", 4: "right"}


def _walk(px, py, actions):
    out = [(px, py)]
    for a in actions:
        dx, dy = _MOVES[a]
        px, py = (px + dx) % 64, (py + dy) % 64
        out.append((px, py))
    return out


def fake_model_start():
    seeds = [("empty context", [], []), ("no frame, actions 3 3", [], [3, 3]), ("no frame, actions 3 4", [], [3, 4])]
    rows = []
    for name, fr, ac in seeds:
        h, px, py = _fnv_seed(fr, ac)
        rows.append((name, f"0x{h:016x}", px, py))
    acts = [4, 4, 1, 1, 2, 3, 0, 1]
    h, px, py = _fnv_seed([], [])
    path = _walk(px, py, acts)
    walk_rows = [("start", "-", f"({path[0][0]}, {path[0][1]})")]
    walk_rows += [(f"step {i + 1}", f"{a} ({_NAMES[a]})", f"({path[i + 1][0]}, {path[i + 1][1]})") for i, a in enumerate(acts)]
    corner = _walk(0, 0, [1, 2, 4, 3])
    corner_rows = [("start", "-", "(0, 0)")]
    corner_rows += [(f"step {i + 1}", f"{a} ({_NAMES[a]})", f"({corner[i + 1][0]}, {corner[i + 1][1]})") for i, a in enumerate([1, 2, 4, 3])]
    out = [
        "The context bytes hashed by FNV-1a (offset 0xcbf29ce484222325, prime 0x100000001b3, 64-bit wrap-around): the number of frame "
        "values as 8 bytes little endian, then the 4 bytes of each frame value's float32 pattern, then the number of actions as 8 "
        "bytes, then each action as 4 bytes little endian. The square's left column is the low six bits of the hash, its top row the "
        "six bits above bit 32.",
        "",
        _table(["context", "hash", "column px", "row py"], rows, "lrrr"),
        "",
        "Walks on the 64 by 64 torus (a step moves the square one cell, mod 64). First from the start of the empty context:",
        "",
        _table(["after", "action", "(px, py)"], walk_rows, "lll"),
        "",
        "and from the corner (0, 0), where the wrap shows:",
        "",
        _table(["after", "action", "(px, py)"], corner_rows, "lll"),
    ]
    return "\n".join(out)


# ----------------------------------------------------------------------------------- a learned dynamics model and its free-running error

TRACK_LEN = 10.0
TRACK_DRAG = 0.9
TRACK_THRUST = 0.5
N_FIT = 2000
ROLL_N = 200
ROLL_K = 32
FIT_NAME = "world_models.track_data"
ROLL_NAME = "world_models.track_rollouts"


def _track_step(x, v, a):
    """The real cart: v' = 0.9 v + 0.5 a, x' = x + v'; a wall at 0 and at 10 stops the cart (x' clamped, v' = 0)."""
    v2 = TRACK_DRAG * v + TRACK_THRUST * a
    x2 = x + v2
    if x2 < 0.0:
        return 0.0, 0.0
    if x2 > TRACK_LEN:
        return TRACK_LEN, 0.0
    return x2, v2


def _solve(a, b):
    """Gauss-Jordan elimination with partial pivoting (the pivot is the largest magnitude in the column, the first on a tie)."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(m[r][c]) > abs(m[p][c]):
                p = r
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [m[r][k] - f * m[c][k] for k in range(n + 1)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _fit_track():
    """Least squares for x' and v' on the features (x, v, a, 1); returns (rows, wx, wv, wall_hits)."""
    g = bookrng.generator(FIT_NAME)
    rows = []
    for _ in range(N_FIT):
        x = 10.0 * g.uniform()
        v = -2.0 + 4.0 * g.uniform()
        a = g.integer(3) - 1
        x2, v2 = _track_step(x, v, a)
        rows.append((x, v, a, x2, v2))
    feats = [[r[0], r[1], float(r[2]), 1.0] for r in rows]

    def fit(col):
        gram = []
        for i in range(4):
            line = []
            for j in range(4):
                s = 0.0
                for f in feats:
                    s += f[i] * f[j]
                line.append(s)
            gram.append(line)
        rhs = []
        for i in range(4):
            s = 0.0
            for k, f in enumerate(feats):
                s += f[i] * rows[k][col]
            rhs.append(s)
        return _solve(gram, rhs)

    hits = sum(1 for r in rows if r[3] == 0.0 or r[3] == TRACK_LEN)
    return rows, fit(3), fit(4), hits


_FIT = None


def _fit():
    global _FIT
    if _FIT is None:
        _FIT = _fit_track()
    return _FIT


def _model_step(x, v, a):
    _, wx, wv, _ = _fit()
    f = (x, v, float(a), 1.0)
    nx = wx[0] * f[0] + wx[1] * f[1] + wx[2] * f[2] + wx[3] * f[3]
    nv = wv[0] * f[0] + wv[1] * f[1] + wv[2] * f[2] + wv[3] * f[3]
    return nx, nv


def _rollout_stats():
    """Mean absolute position error after k free-running steps, the fraction of model rollouts that have left the track by step k,
    and the mean one-step error when the model is fed the true state each step."""
    g = bookrng.generator(ROLL_NAME)
    err = [0.0] * (ROLL_K + 1)
    out_of_track = [0] * (ROLL_K + 1)
    teacher = 0.0
    for _ in range(ROLL_N):
        x0 = 10.0 * g.uniform()
        xt, vt = x0, 0.0
        xm, vm = x0, 0.0
        for k in range(1, ROLL_K + 1):
            a = g.integer(3) - 1
            px, _ = _model_step(xt, vt, a)
            teacher += abs(px - _track_step(xt, vt, a)[0])
            xt, vt = _track_step(xt, vt, a)
            xm, vm = _model_step(xm, vm, a)
            err[k] += abs(xm - xt)
            if xm < 0.0 or xm > TRACK_LEN:
                out_of_track[k] += 1
    return [e / ROLL_N for e in err], [c / ROLL_N for c in out_of_track], teacher / (ROLL_N * ROLL_K)


def _error_curve():
    err, _, teacher = _rollout_stats()
    return err, teacher


def track_fit():
    rows, wx, wv, hits = _fit()
    err, outside, teacher = _rollout_stats()
    coef = [
        ("x'", f"{wx[0]:.4f}", f"{wx[1]:.4f}", f"{wx[2]:.4f}", f"{wx[3]:.4f}", "1", "1", "0.5", "0"),
        ("v'", f"{wv[0]:.4f}", f"{wv[1]:.4f}", f"{wv[2]:.4f}", f"{wv[3]:.4f}", "0", "0.9", "0.5", "0"),
    ]
    tr = wx[0] + wv[1]
    det = wx[0] * wv[1] - wx[1] * wv[0]
    disc = tr * tr - 4.0 * det
    assert disc < 0.0
    eig = (f"The 2 by 2 matrix of the weights on (x, v) has trace {tr:.4f} and determinant {det:.4f}, so its eigenvalues are the "
           f"pair {tr / 2:.4f} +- {math.sqrt(-disc) / 2:.4f} i, of modulus {math.sqrt(det):.4f}; the real cart's are 1 and 0.9.")
    ks = [1, 2, 4, 8, 16, 32]
    roll = [(k, f"{err[k]:.3f}", f"{100 * outside[k]:.1f}%") for k in ks]
    return "\n".join([
        f"The real cart: v' = 0.9 v + 0.5 a, x' = x + v' on a track [0, 10], and a wall stops it (when x' < 0 the cart is put at x' = 0, and when x' > 10 at 10, with v' = 0 in both cases). "
        f"Data: {N_FIT} transitions from the stream `{FIT_NAME}` (seed {_seed(FIT_NAME)}); for each, in order, x = 10 u, "
        f"v = -2 + 4 u (two `uniform()` draws) and a = `integer(3)` - 1; {hits} of the {N_FIT} end on a wall. The model is the "
        "least-squares solution of the normal equations (Gauss-Jordan, partial pivoting) for x' and for v' on the features "
        "(x, v, a, 1); each sum runs over the transitions in order.",
        "",
        _table(["output", "w on x", "w on v", "w on a", "bias", "true x", "true v", "true a", "true bias"], coef, "lrrrrrrrr"),
        "",
        eig,
        "",
        f"Free-running error: {ROLL_N} rollouts of {ROLL_K} steps from the stream `{ROLL_NAME}` (seed {_seed(ROLL_NAME)}); "
        "for each rollout, in order, x = 10 u (v starts at 0), then for each step a = `integer(3)` - 1. The real cart and the model "
        "take the same actions; the model is fed its own output and is never clamped (it knows no walls), and 'off the track' counts the model rollouts whose position is below 0 or above 10 at step k. The error is |x_model - x_real|, averaged over the rollouts. "
        f"Fed the true state at every step instead, the model's mean one-step position error over all {ROLL_N * ROLL_K} steps is "
        f"{teacher:.3f}.",
        "",
        _table(["step k", "mean position error", "model rollouts off the track"], roll, "lrr"),
    ])


# ------------------------------------------------------------------------------------------------- a latent action from video alone

LAT_N = 400
LAT_NAME = "world_models.latent_action"


def _wrap8(d):
    return d - 8.0 * math.floor(d / 8.0 + 0.5)


def _kmeans(points, k):
    """Farthest-point start (first point, then the point with the largest distance to its nearest chosen centre, the first on a tie),
    then Lloyd rounds (assign to the nearest centre, the lowest index on a tie; centre = mean, an empty cluster keeps its centre)
    until the assignment stops changing (at most 100 rounds). Returns (centres, labels)."""
    cs = [points[0]]
    while len(cs) < k:
        best, bi = -1.0, 0
        for i, p in enumerate(points):
            d = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in cs)
            if d > best:
                best, bi = d, i
        cs.append(points[bi])
    labels = None
    for _ in range(100):
        new = []
        for p in points:
            bj, bd = 0, None
            for j, c in enumerate(cs):
                d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if bd is None or d < bd:
                    bj, bd = j, d
            new.append(bj)
        if new == labels:
            break
        labels = new
        for j in range(k):
            sx = sy = 0.0
            n = 0
            for p, lab in zip(points, labels):
                if lab == j:
                    sx += p[0]
                    sy += p[1]
                    n += 1
            if n:
                cs[j] = (sx / n, sy / n)
    return cs, labels


def _latent_run(sd, k):
    """The displacements of a square wandering a 8 by 8 torus under 5 random actions, seen through a localiser with error sd."""
    ga = bookrng.generator(LAT_NAME + ".actions")
    gn = bookrng.generator(LAT_NAME + ".noise")
    acts = [ga.integer(5) for _ in range(LAT_N)]
    px = py = 0
    seen = []
    pos = [(0, 0)]
    for a in acts:
        dx, dy = _MOVES[a]
        px, py = (px + dx) % 8, (py + dy) % 8
        pos.append((px, py))
    for (x, y) in pos:
        seen.append((x + sd * gn.normal(), y + sd * gn.normal()))
    disp = [(_wrap8(seen[t + 1][0] - seen[t][0]), _wrap8(seen[t + 1][1] - seen[t][1])) for t in range(LAT_N)]
    cs, labels = _kmeans(disp, k)
    counts = {}
    for lab, a in zip(labels, acts):
        counts.setdefault(lab, [0] * 5)[a] += 1
    pure = sum(max(c) for c in counts.values())
    return acts, cs, labels, counts, pure / LAT_N


def latent_action_toy():
    sds = [0.05, 0.15, 0.25, 0.35, 0.45]
    rows = []
    for sd in sds:
        for k in (5,):
            _, _, _, counts, pur = _latent_run(sd, k)
            rows.append((f"{sd:.2f}", k, len(counts), f"{pur:.3f}"))
    _, _, _, counts3, pur3 = _latent_run(0.05, 3)
    rows.append(("0.05", 3, len(counts3), f"{pur3:.3f}"))
    _, _, _, counts6, pur6 = _latent_run(0.05, 6)
    rows.append(("0.05", 6, len(counts6), f"{pur6:.3f}"))
    _, cs, labels, counts, _ = _latent_run(0.05, 5)
    cent = []
    for j in range(5):
        c = counts[j]
        major = max(range(5), key=lambda a: (c[a], -a))
        cent.append((j, f"({cs[j][0]:+.3f}, {cs[j][1]:+.3f})", sum(c), f"{_NAMES[major]} ({c[major]} of {sum(c)})"))
    return "\n".join([
        f"A square wanders an 8 by 8 torus for {LAT_N} steps under random actions 0 .. 4 (0 noop, 1 up, 2 left, 3 down, 4 right, "
        f"the order of the first block), drawn as `integer(5)` from the stream `{LAT_NAME}.actions` (seed "
        f"{_seed(LAT_NAME + '.actions')}), starting at (0, 0). A localiser reports each of the {LAT_N + 1} positions plus normal noise "
        f"of the stated spread on both coordinates, two `normal()` draws per position (x then y) from `{LAT_NAME}.noise` (seed "
        f"{_seed(LAT_NAME + '.noise')}). All {LAT_N} actions are drawn first, then the noise. An action moves the square by (dx, dy) = noop (0, 0), up (0, -1), left (-1, 0), down (0, +1), right (+1, 0), "
        "with the coordinates taken mod 8 (y grows downward). The displacement of a step is the difference of two reports with each coordinate wrapped into "
        "[-4, 4) by d - 8 floor(d / 8 + 1/2). The displacements are clustered by k-means with a farthest-point start (the first point "
        "and then, k - 1 times, the point farthest from its nearest chosen centre, the first on a tie) and Lloyd rounds (assign each point to "
        "the nearest centre, the lowest-numbered on a tie; a centre becomes the mean of its points, an empty cluster keeps its centre) until the "
        "assignment is unchanged (at most 100 rounds); codes are numbered in the order the centres were chosen; purity is the fraction of steps whose cluster's most common true action is their own.",
        "",
        _table(["localiser error", "codes k", "codes used", "purity"], rows, "rrrr"),
        "",
        "The five codes at error 0.05, with the true action each one stands for:",
        "",
        _table(["code", "centre (dx, dy)", "steps", "most common true action"], cent, "rlrl"),
    ])


# -------------------------------------------------------------------------------------------- masked decoding of dependent tokens


def _valid_probability(slots, batches):
    """Exact probability that decoding `slots` binary tokens in the given batch sizes ends with exactly one 1. The world puts one
    marker in one of the slots, uniformly. A model that knows this exactly gives, with u slots still masked and no 1 yet seen, the
    probability 1/u of a 1 to every masked slot; once a 1 is seen, the probability 0. A batch of m slots is sampled independently."""
    zero, one = Fraction(1), Fraction(0)
    u = slots
    for m in batches:
        assert 0 < m <= u
        p = Fraction(1, u)
        pk = [Fraction(math.comb(m, k)) * p**k * (1 - p) ** (m - k) for k in range(m + 1)]
        zero, one = zero * pk[0], one + zero * pk[1]
        u -= m
    assert u == 0
    return one


def masked_decoding():
    plans = [("one pass, sampled", [6]), ("two passes of 3", [3, 3]), ("three passes of 2", [2, 2, 2]), ("six passes of 1", [1] * 6)]
    rows = []
    for name, b in plans:
        p = _valid_probability(6, b)
        rows.append((name, len(b), f"{p.numerator}/{p.denominator}", f"{float(p):.4f}"))
    return "\n".join([
        "A world drops one marker into one of six slots, uniformly, so a frame is valid when exactly one of its six tokens is 1. A "
        "masked model that is exact gives each masked slot the probability 1/u of being 1, where u is the number of slots still "
        "masked and no 1 has been revealed yet, and 0 once a 1 is revealed. The batch of slots revealed in a pass are sampled "
        "independently from those probabilities. The probability of ending with a valid frame is exact (rational arithmetic):",
        "",
        _table(["decoding", "network passes", "P(valid) as a fraction", "P(valid)"], rows, "lrrr"),
        "",
        "Taking the most probable value of each slot instead (argmax) gives 0 for every slot while u > 1, and the single pass "
        "produces the all-zero frame with probability 1.",
    ])


def mask_schedule():
    n = 256
    rows = []
    for steps in (4, 8):
        counts = [n]
        for s in range(1, steps):
            ratio = (s + 1) / steps
            r = max(math.cos(ratio * math.pi / 2.0), 1e-6)
            counts.append(int(r * n))
        rows.append((steps, ", ".join(str(c) for c in counts)))
    return "\n".join([
        "Tokens still masked at the start of each pass, for the 256 tokens of one frame: the first pass masks all of them; pass s "
        "(s = 1, 2, ...) masks floor(max(cos(pi/2 (s + 1)/S), 1e-6) * 256) of them, the least confident, where S is the number of "
        "passes.",
        "",
        _table(["passes S", "tokens masked at the start of passes 0, 1, 2, ..."], rows, "rl"),
    ])


# ------------------------------------------------------------------------------------------------- the conditioners of a denoiser

SIGMA_DATA = 0.5
SIGMA_OFFSET = 0.3


def _karras(n, smin=2e-3, smax=5.0, rho=7.0):
    mi, ma = smin ** (1.0 / rho), smax ** (1.0 / rho)
    out = []
    for i in range(n):
        l = 0.0 if n == 1 else i / (n - 1)
        out.append((ma + l * (mi - ma)) ** rho)
    out.append(0.0)
    return out


def edm_conditioners():
    sg = _karras(3)
    rows = []
    for s in sg[:3] + [1.0]:
        s2 = s * s + SIGMA_OFFSET**2
        sp = math.sqrt(s2)
        sd2 = SIGMA_DATA**2
        c_in = 1.0 / math.sqrt(s2 + sd2)
        c_skip = sd2 / (s2 + sd2)
        c_out = sp * math.sqrt(c_skip)
        c_noise = math.log(sp) / 4.0
        rows.append((f"{s:.6f}", f"{sp:.6f}", f"{c_in:.6f}", f"{c_skip:.6f}", f"{c_out:.6f}", f"{c_noise:.6f}",
                     f"{c_in * c_in * (s2 + sd2):.6f}"))
    return "\n".join([
        "The conditioners at the noise levels of the three-step schedule (sigma_max = 5, sigma_min = 0.002, rho = 7: the levels "
        "(5^(1/7) + l (0.002^(1/7) - 5^(1/7)))^7 for l = 0, 1/2, 1, then 0), and at sigma = 1. sigma_data = 0.5, offset 0.3, "
        "s = sqrt(sigma^2 + 0.3^2); c_in = 1/sqrt(s^2 + sigma_data^2), c_skip = sigma_data^2/(s^2 + sigma_data^2), "
        "c_out = s sqrt(c_skip), c_noise = ln(s)/4. The last column is the variance of the scaled network input, which the choice of "
        "c_in makes 1.",
        "",
        _table(["sigma", "s", "c_in", "c_skip", "c_out", "c_noise", "input variance"], rows, "rrrrrrr"),
    ])


# --------------------------------------------------------------------------------------- a denoiser that knows two futures exactly


def _denoise_two(x, s, bias):
    """The exact denoiser of a world whose next 'frame' is +1 with probability p and -1 otherwise: E[y | x = y + s z]
    = tanh(x / s^2 + bias), bias = ln(p / (1 - p)) / 2."""
    return math.tanh(x / (s * s) + bias)


def _euler_sample(x0, sg, bias):
    x = x0
    for i in range(len(sg) - 1):
        s, sn = sg[i], sg[i + 1]
        d = _denoise_two(x, s, bias)
        x = x + (x - d) / s * (sn - s)
    return x


def _phi(t):
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _threshold(f, level, lo, hi):
    """The x0 at which the increasing map f crosses `level` (bisection, 200 halvings); None if it does not cross inside [lo, hi]."""
    if f(lo) > level or f(hi) < level:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) > level:
            hi = mid
        else:
            lo = mid
    return hi


def _mode_stats(n, p, std):
    """For n evaluations of the Euler sampler and a start x0 ~ N(0, std^2): the probability that the sample ends within 0.1 of +1
    ('plus'), within 0.1 of -1 ('minus'), and elsewhere (a frame that is neither future)."""
    bias = 0.5 * math.log(p / (1.0 - p))
    sg = _karras(n)

    def f(x0):
        return _euler_sample(x0, sg, bias)

    lo, hi = -12.0 * std, 12.0 * std
    # P(f(x0) > 0.9) and P(f(x0) < -0.9) for the increasing map f; when f is already beyond the level at the left end of the search
    # interval (12 standard deviations) every start is on that side, and when it never reaches the level at the right end none is.
    t_hi = _threshold(f, 0.9, lo, hi)
    t_lo = _threshold(f, -0.9, lo, hi)
    plus = (1.0 if f(lo) > 0.9 else 0.0) if t_hi is None else 1.0 - _phi(t_hi / std)
    minus = (1.0 if f(hi) < -0.9 else 0.0) if t_lo is None else _phi(t_lo / std)
    return plus, minus, 1.0 - plus - minus


def _share_curve(std, p=0.8, ns=range(1, 21)):
    out = []
    for n in ns:
        plus, minus, _ = _mode_stats(n, p, std)
        out.append(plus / (plus + minus) if 1.0 - plus - minus < 0.01 else float("nan"))
    return out


def diffusion_two_futures():
    p = 0.8
    rows = []
    for n in (1, 2, 3, 4, 6, 10, 20):
        cells = [n]
        for std in (5.0, 1.0):
            plus, minus, other = _mode_stats(n, p, std)
            other = max(other, 0.0)
            share = plus / (plus + minus) if other < 0.01 else float("nan")
            cells += [f"{100 * other:.2f}%", "-" if math.isnan(share) else f"{share:.4f}"]
        rows.append(tuple(cells))
    return "\n".join([
        "The next 'frame' is a single number y: +1 with probability 0.8, -1 with probability 0.2. The denoiser is exact, "
        "D(x, s) = tanh(x/s^2 + (1/2) ln 4), and the sampler is the Euler sampler of the chapter with the Karras levels of "
        "n evaluations (sigma_max 5, sigma_min 0.002, rho 7): x <- x + (x - D(x, s_i)) (s_(i+1) - s_i)/s_i. The start is x ~ N(0, 25) "
        "(unit noise scaled to sigma_max) or x ~ N(0, 1) (unit noise, unscaled). The probabilities are exact integrals of the Gaussian "
        "start over the set of starts the sampler maps to the stated place, found by bisection on the (increasing) map. 'Neither' is "
        "the probability of ending further than 0.1 from both +1 and -1; 'share of +1' is P(within 0.1 of +1) divided by P(within "
        "0.1 of either), shown when 'neither' is below 1%.",
        "",
        _table(["n", "neither, start N(0, 25)", "share of +1, N(0, 25)", "neither, start N(0, 1)", "share of +1, N(0, 1)"], rows, "rrrrr"),
        "",
        "The target share is 0.8000.",
    ])


# ------------------------------------------------------------------------------------------------------- the counts of the models


def _diamond_params(c=64, cc=256, nsc=4, ic=3, na=4, levels=4, depth=2):
    """Parameters by group of a UNet with `levels` levels of `c` channels, `depth` residual blocks a level on the way down and
    depth + 1 on the way up, two blocks and self-attention in the middle, conditioning width cc, nsc context frames."""
    groups = {}

    def add(g, n):
        groups[g] = groups.get(g, 0) + n

    add("conditioning (noise, actions, MLP)", c * 2 + na * (cc // nsc) + 2 * (cc * cc + cc))
    add("input convolution", c * (nsc + 1) * ic * 9 + c)

    def res(cin, cout, where, attn=False):
        if cin != cout:
            add("residual-block convolutions", cout * cin + cout)
        add("conditioned norms (scale and shift linears)", 2 * cin * cc + 2 * cin + 2 * cout * cc + 2 * cout)
        add("residual-block convolutions", cout * cin * 9 + cout + cout * cout * 9 + cout)
        if attn:
            add("self-attention (norm, qkv, output)", 2 * cout + 3 * cout * cout + 3 * cout + cout * cout + cout)

    for _ in range(levels * depth):
        res(c, c, "down")
    for _ in range(2):
        res(c, c, "mid", True)
    for _ in range(levels * (depth + 1)):
        res(2 * c, c, "up")
    add("down and up sampling convolutions", 2 * (levels - 1) * (c * c * 9 + c))
    add("output norm and convolution", 2 * c + ic * c * 9 + ic)
    return groups


def _conv_flops(cin, cout, k, ho, wo):
    return 2 * cin * cout * k * k * ho * wo


def _diamond_flops(c=64, hw=64, nsc=4, ic=3, levels=4, depth=2):
    """FLOPs (multiplies and adds of the convolutions and attention matmuls) of one forward pass; the elementwise work (normalisation,
    SiLU, additions, nearest-neighbour upsampling) and the host-side conditioning linears are not counted."""
    f = {}

    def add(g, n):
        f[g] = f.get(g, 0) + n

    add("input convolution", _conv_flops((nsc + 1) * ic, c, 3, hw, hw))
    sizes = [hw >> i for i in range(levels)]
    for i, s in enumerate(sizes):
        if i > 0:
            add("sampling convolutions", _conv_flops(c, c, 3, s, s))
        for _ in range(depth):
            add("residual-block convolutions", 2 * _conv_flops(c, c, 3, s, s))
    s = sizes[-1]
    n = s * s
    for _ in range(2):
        add("residual-block convolutions", 2 * _conv_flops(c, c, 3, s, s))
        add("self-attention", 2 * n * c * 3 * c + 2 * n * n * c + 2 * n * n * c + 2 * n * c * c)
    for j in range(levels):
        s = sizes[levels - 1 - j]
        if j > 0:
            add("sampling convolutions", _conv_flops(c, c, 3, s, s))
        for _ in range(depth + 1):
            add("residual-block convolutions", _conv_flops(2 * c, c, 3, s, s) + _conv_flops(c, c, 3, s, s) + _conv_flops(2 * c, c, 1, s, s))
    add("output convolution", _conv_flops(c, ic, 3, hw, hw))
    return f


def diamond_counts():
    params = _diamond_params()
    flops = _diamond_flops()
    tp = sum(params.values())
    tf = sum(flops.values())
    rows = [(g, _int(n), f"{n / tp:.3f}") for g, n in sorted(params.items(), key=lambda kv: -kv[1])]
    rows.append(("total", _int(tp), "1.000"))
    frows = [(g, f"{n / 1e9:.4f}", f"{n / tf:.3f}") for g, n in sorted(flops.items(), key=lambda kv: -kv[1])]
    frows.append(("total, one pass", f"{tf / 1e9:.4f}", "1.000"))
    # trainable = every convolution, the 1x1 projections and the attention projections (not the norms, not the conditioning)
    c = 64
    att = 2 * (3 * c * c + 3 * c + c * c + c)
    train = (params["input convolution"] + params["residual-block convolutions"] + params["down and up sampling convolutions"]
             + (3 * c * 9 + 3) + att)
    per_step = [(n, f"{n * tf / 1e9:.3f}", f"{15 * n * tf / 1e12:.4f}") for n in (1, 2, 3)]
    weight_bytes = 4 * tp
    return "\n".join([
        "The UNet of the Atari model, built from its config: 64 by 64 frames of 3 channels, the 4 context frames and the noisy frame "
        "stacked on the channels (5 frames of 3 channels: 15 input channels in all), 4 levels of 64 channels with 2 residual blocks "
        "a level on the way down and 3 on the way up, the sampling convolutions between levels (stride 2 down, nearest-neighbour "
        "doubling and a 3 by 3 convolution up), two residual blocks with self-attention at 8 by 8 in the middle (64 positions, "
        "head width 8), conditioning width 256, 4 actions. A convolution costs 2 c_in c_out k^2 per output pixel; attention costs "
        "2 n c (3 c) for the projections, 2 n^2 c for the scores, 2 n^2 c for the weighted sum and 2 n c^2 for the output "
        "projection, for n = 64 positions. Inventory of the parameters: a residual block from c_in to c_out channels has two conditioned "
        "norms, the first over c_in channels and the second over c_out, each a linear layer from the 256-vector to a scale and a shift "
        "per channel (2 c_in * 256 + 2 c_in, and 2 c_out * 256 + 2 c_out); a 3 by 3 convolution c_in to c_out and one c_out to c_out, both "
        "with biases; and, only when c_in differs from c_out (the blocks of the way up, whose input is the 64 channels of the path "
        "joined to the 64 of the skip connection, 128 to 64), a 1 by 1 shortcut convolution with a bias. The middle's two blocks each "
        "add a norm (gain and bias), a 1 by 1 query-key-value convolution 64 to 192 with a bias and a 1 by 1 output convolution with a "
        "bias. 'Conditioning' is 384 + 2 * (256 * 256 + 256) = 131,968 parameters: 128 for the noise-level embedding (2 * 64), 256 "
        "for the action table (4 actions of 64 values, so that the 4 context frames' actions fill the 256-vector) and two 256 by 256 "
        "linear layers with biases. The sampling convolutions are 3 by 3, 64 to 64 with a bias, three down and three up. The FLOP "
        "table counts the up blocks' 128 to 64 shortcuts, and each block's first convolution, at the resolution of the block's "
        "output; a sampling convolution on the way up runs after the doubling, so at the larger resolution.",
        "",
        _table(["parameters", "count", "share"], rows, "lrr"),
        "",
        f"Trainable in the fine-tuning of the chapter (every convolution including the 1 by 1 projections, and the attention "
        f"projections; the norms and the whole conditioning path stay frozen): {_int(train)} of {_int(tp)} parameters. "
        f"Stored as float32 the weights are {weight_bytes / 1e6:.2f} MB. The scale-and-shift linears of the conditioned norms "
        f"({_int(params['conditioned norms (scale and shift linears)'])} parameters, {2 * params['conditioned norms (scale and shift linears)'] / 1e6:.2f} "
        "MFLOP per evaluation) are applied to one 256-vector, not to an image.",
        "",
        _table(["FLOPs of one forward pass", "billions", "share"], frows, "lrr"),
        "",
        "Work of one generated frame is the pass times the number of evaluations; the last column is the work of one second of play "
        "at 15 frames a second, in teraFLOPs (a requirement of the model, not a measurement of any machine).",
        "",
        _table(["evaluations per frame", "GFLOP per frame", "TFLOP per second of play at 15 frames a second"], per_step, "rrr"),
    ])


def _attn_params(d, inner, hd):
    """to_q, to_kv (fused k and v) and to_out without bias, two per-head-dimension scales, the norm gain and its (unused) bias buffer,
    and the unused context-norm gain and bias buffer."""
    return 2 * hd + 4 * d + d * inner + 2 * inner * d + inner * d


def _ff_params(d, inner):
    return 2 * d + 2 * inner * d + d * inner


def genie_inventory():
    """The shape rules behind genie_counts, with the numbers computed from the same functions."""
    d, inner, hd, ffi = 512, 512, 64, 1365
    peg = 27 * d + d
    att = _attn_params(d, inner, hd)
    ff = _ff_params(d, ffi)
    half = _stblock_params(d, inner, hd, ffi) // 2
    return (
        f"A block is two halves (spatial, then temporal) of the same shape. A half holds a depthwise 3 by 3 by 3 convolution over the channels (PEG: 27 weights "
        f"and a bias per channel, {_int(peg)} at width {d}); an attention ({_int(att)}: two vectors of {hd} per-head-dimension scales for the queries and keys (2 * {hd}), "
        f"a norm gain and its unused bias buffer and the unused context-norm gain and bias buffer (4 * {d}), a query matrix {d} to {inner}, a fused key-value "
        f"matrix {d} to 2 * {inner}, and an output matrix {inner} to {d}, none with a bias); and a gated feed-forward ({_int(ff)}: a norm gain and bias buffer (2 * {d}), an input matrix {d} to "
        f"2 * {ffi} whose two halves are the value and the gate of a GEGLU, and an output matrix {ffi} to {d}, without biases). That is {_int(half)} for a half, {_int(2 * half)} "
        f"for a block, and 15 tensors a half (2 for the PEG, 9 for the attention, 4 for the feed-forward). The output gain of a stack is a gain and a bias buffer (2 * width). "
        f"A position-bias MLP maps a 2-vector to the width h, then h to h, then h to 8 heads, each linear with a bias. A patch embedding is a norm over the 48 values of a 4 by 4 patch of 3 channels "
        f"(gain and bias), a linear layer 48 to {d} with a bias and a norm over {d} (gain and bias); a pixel head is a linear layer {d} to 48 with a bias. "
        f"The quantizer is a linear layer {d} to 32 with a bias, a codebook of 1024 vectors of 32, a linear layer 32 to {d} with a bias, and the buffers the importer "
        "reads: one initialised flag, 1024 cluster counts and 1024 vectors of 32 for the moving average. The dynamics works at width 519 (512 plus 7 action channels) with feed-forward width 1384; "
        "its token embedding has 1025 rows of 512 (1024 codes and a mask), its position embedding 4000 rows of 512, and its logits are a linear layer 519 to 1024 with a bias. ")


def _stblock_params(d, inner_attn, hd, ffi):
    peg = 27 * d + d
    half = peg + _attn_params(d, inner_attn, hd) + _ff_params(d, ffi)
    return 2 * half


def _genie_counts():
    d, heads, hd, ffi = 512, 8, 64, 1365
    inner = heads * hd
    pf = 3 * 4 * 4
    cpb = lambda hidden: 2 * hidden + hidden + hidden * hidden + hidden + heads * hidden + heads
    tok = {
        "encoder: 8 blocks and output gain": 8 * _stblock_params(d, inner, hd, ffi) + 2 * d,
        "decoder: 8 blocks and output gain": 8 * _stblock_params(d, inner, hd, ffi) + 2 * d,
        "spatial position-bias MLP": cpb(512),
        "patch embeddings (first frame, rest)": 2 * (2 * pf + pf * d + d + 2 * d),
        "pixel heads (first frame, rest)": 2 * (pf * d + pf),
        "quantizer (in, codebook, out, EMA buffers)": d * 32 + 32 + 1024 * 32 + 32 * d + d + 1 + 1024 + 1024 * 32,
    }
    d2, ffi2 = 519, 1384
    dyn = {
        "12 blocks and output gain": 12 * _stblock_params(d2, inner, hd, ffi2) + 2 * d2,
        "token embedding (1024 codes and a mask)": 1025 * d,
        "position embedding (4000 positions)": 4000 * d,
        "spatial position-bias MLP": cpb(64),
        "logits": 1024 * d2 + 1024,
    }
    return tok, dyn


def genie_counts():
    tok, dyn = _genie_counts()
    tt, td = sum(tok.values()), sum(dyn.values())
    rows = [("tokenizer: " + k, _int(v)) for k, v in tok.items()] + [("tokenizer total", _int(tt))]
    rows += [("dynamics: " + k, _int(v)) for k, v in dyn.items()] + [("dynamics total", _int(td))]
    per_block = 2 * (2 + 9 + 4)
    dyn_tensors = 12 * per_block + 2 + 2 + 6 + 2
    return "\n".join([
        "Parameters of the two networks of the GenieRedux checkpoints brain imports, counted from their shapes: width 512 (519 in the "
        "dynamics: 512 plus 7 action channels), 8 heads of width 64, feed-forward inner width 1365 (1384), a 3 by 3 by 3 depthwise "
        "convolution (PEG) and an attention and a feed-forward in each of the spatial and temporal halves of a block. The attention has "
        "no biases; the tensors the importer reads and discards (the unused context norm and the quantizer's exponential-moving-average (EMA) "
        "buffers) are counted because they are in the file. " + genie_inventory(),
        "",
        _table(["part", "parameters"], rows, "lr"),
        "",
        f"Tensors per checkpoint: {dyn_tensors} in the dynamics (12 blocks of 30, 2 output-gain tensors, the two embeddings, "
        f"6 for the position-bias MLP, 2 for the logits).",
    ])


def _genie_macs_per_token_block():
    """Multiply-adds per token of the linear parts of one dynamics block (q, kv, out, feed-forward in and out, and the PEG), the
    spatial and the temporal half together."""
    d, inner, ffi = 519, 512, 1384
    half = d * inner + d * 2 * inner + inner * d + d * 2 * ffi + ffi * d + 27 * d
    return 2 * half


def _genie_pairs(t, hw=256):
    return t * hw * hw + hw * t * (t + 1) // 2


def genie_context():
    hw, inner, layers = 256, 512, 12
    macs = _genie_macs_per_token_block()
    rows = []
    for t in (1, 2, 4, 8, 15):
        n = t * hw
        full = n * (n + 1) // 2
        rows.append((t, _int(n), _int(_genie_pairs(t)), _int(full), f"{full / _genie_pairs(t):.1f}"))
    frows = []
    for t in (2, 4, 8, 15):
        whole = layers * (2 * t * hw * macs + 4 * inner * _genie_pairs(t))
        new = layers * (2 * hw * macs + 4 * inner * (hw * hw + hw * t))
        frows.append((t, f"{whole / 1e12:.3f}", f"{new / 1e12:.3f}", f"{whole / new:.2f}"))
    return "\n".join([
        "Token arithmetic of the dynamics, from the importer's constants: a 64 by 64 frame is cut into 4 by 4 patches, so a frame is "
        "16 by 16 = 256 tokens of 10 bits (a codebook of 1024), 2,560 bits against the 98,304 bits of the 8-bit frame. The position "
        "embedding has 4000 rows and is indexed by the position of a token in the whole clip of t frames, so t * 256 <= 4000, which is "
        "t <= 15 frames in one forward pass (3,840 positions).",
        "",
        "Attention pairs scored, per layer: spatial attention over each frame's 256 tokens (t * 256^2) plus causal temporal attention "
        "over each of the 256 locations (256 t (t + 1)/2), against causal attention over all t * 256 tokens (n (n + 1)/2):",
        "",
        _table(["frames t", "tokens n", "pairs, space then time", "pairs, one causal sequence", "ratio"], rows, "rrrrr"),
        "",
        "Per token and per block the linear parts are, in multiply-adds and for each of the two halves, a query matrix 519 to 512, a key-value matrix 519 to 1024, an output "
        "matrix 512 to 519, a feed-forward input 519 to 2 * 1384, an output 1384 to 519 (three feed-forward matrices of that size in all, counting the gate) and the PEG's 27 per channel "
        "(27 * 519). Pairs scored for the new frame alone are 256^2 within the frame plus 256 t to the t frames of the clip. "
        "TFLOPs of one forward pass of the 12-block dynamics over t frames of context (the linear layers and the PEG at 2 FLOPs per "
        "multiply-add, and attention at 4 * 512 FLOPs per pair scored: scores and weighted sum; the logits and the elementwise work are "
        "left out), the way brain runs it, with every frame recomputed; and the work that would remain if the keys, values and PEG "
        "inputs of earlier frames were kept, so that only the 256 new tokens were computed and attended against t frames (brain does "
        "not keep them):",
        "",
        _table(["frames t", "TFLOP, whole clip recomputed", "TFLOP, new frame only", "ratio"], frows, "rrrr"),
    ])


# ------------------------------------------------------------------------------------------------------------ the frame budget loop

K_DOWN, K_UP = 5, 30


def _pacer_run(fps, work_of_code, ticks):
    """The fixed-timestep loop with an adaptive quality controller, on a clock that is only a counter of milliseconds: work advances
    the counter, the sleep jumps it to the deadline. The model's quality code is 0 at level 0 and (-level) below, and the work of a
    tick is work_of_code(code) milliseconds. Returns the per-tick (code, work, overrun, period) list and the total milliseconds."""
    budget = 1000 // fps
    now = 0
    deadline = now + budget
    over = under = 0
    level = 0
    out = []
    for _ in range(ticks):
        code = 0 if level >= 0 else -level
        w = work_of_code(code)
        t0 = now
        now += w
        overrun = w > budget
        if deadline > now:
            now = deadline
        period = max(now - t0, 1)
        deadline = max(now, deadline) + budget
        delta = 0
        if overrun:
            over += 1
            under = 0
            if over >= K_DOWN:
                delta, over = -1, 0
        else:
            under += 1
            over = 0
            if under >= K_UP:
                delta, under = 1, 0
        out.append((code, w, overrun, period))
        level = max(-3, min(0, level + delta))
    return out, now


def pacing_arithmetic():
    brows = []
    for fps in (10, 15, 24, 30, 60):
        b = 1000 // fps
        brows.append((fps, b, f"{1000 / b:.2f}", f"{(1000 / b - fps) / fps:.3f}"))

    passes = {0: 3, 1: 2, 2: 1, 3: 1}
    crows = []
    for per_pass in (20, 22, 23, 30, 40):
        run, total = _pacer_run(15, lambda c, p=per_pass: passes[c] * p, 400)
        codes = [r[0] for r in run]
        late = sum(1 for r in run[-175:] if r[2])
        steady = [r for r in run[-175:]]
        seq = []
        for c in codes[-175:]:
            if not seq or seq[-1][0] != c:
                seq.append([c, 0])
            seq[-1][1] += 1
        pattern = ", ".join(f"code {c} x {n}" for c, n in seq[:4])
        rate = 1000 * 175 / sum(r[3] for r in steady)
        crows.append((per_pass, passes[0] * per_pass, f"{late}/175", pattern, f"{rate:.2f}"))
    return "\n".join([
        "The fixed-timestep loop gives a step a budget of 1000 // (frames a second) whole milliseconds (integer division), so the rate it holds is "
        "1000 / budget frames a second:",
        "",
        _table(["target frames a second", "budget (ms)", "rate held (frames a second)", "above the target (fraction)"], brows, "rrrr"),
        "",
        "The quality controller: a tick is an overrun when its work exceeds the budget (equal is not an overrun); after 5 overruns "
        "in a row the quality code goes down one level (code 0 = 3 evaluations, 1 = 2, 2 or more = 1), after 30 ticks in a row "
        "within budget it goes up one; a tick lasts from its start until the later of the end of its work and its deadline (so max(work, budget) when the loop has no slack in hand), and the next deadline is the later of that instant and the old deadline, plus the budget; an overrun resets the count of good ticks and a good tick the count of overruns. A "
        "hypothetical machine that takes w milliseconds for each evaluation, a target of 15 frames a second (budget 66 milliseconds), 400 ticks, "
        "counted over the last 175 (the first run listed is the tail of a longer one):",
        "",
        _table(["milliseconds an evaluation w", "milliseconds at code 0", "overruns in the last 175 ticks", "codes in the last 175 ticks (first four runs)",
                "rate over those ticks (frames a second)"], crows, "rrrlr"),
    ])


# ---------------------------------------------------------------------------------- a policy searched in the model, scored in the world

POLICY_H = 40
POLICY_GOAL = 8.0
POLICY_START = (1.0, 0.0)


def _policy_cost(step, theta):
    x, v = POLICY_START
    total = 0.0
    for _ in range(POLICY_H):
        a = 1 if x < theta else (-1 if v > 0 else 0)
        x, v = step(x, v, a)
        total += abs(x - POLICY_GOAL)
    return total / POLICY_H


def _policy_traj(step, theta):
    x, v = POLICY_START
    out = []
    for _ in range(POLICY_H):
        a = 1 if x < theta else (-1 if v > 0 else 0)
        x, v = step(x, v, a)
        out.append(x)
    return out


def _best_theta(step):
    best = None
    for i in range(101):
        th = i / 10.0
        c = _policy_cost(step, th)
        if best is None or c < best[0]:
            best = (c, th)
    return best


def dream_policy():
    _fit()
    cm, tm = _best_theta(_model_step)
    cr, tr = _best_theta(_track_step)
    rows = [
        ("searched in the model", f"{tm:.1f}", f"{cm:.4f}", f"{_policy_cost(_track_step, tm):.4f}"),
        ("searched in the real cart", f"{tr:.1f}", f"{_policy_cost(_model_step, tr):.4f}", f"{cr:.4f}"),
    ]
    real_traj = _policy_traj(_track_step, tm)
    dream_traj = _policy_traj(_model_step, tm)
    idx = (0, 3, 7, 11, 15, 39)
    traj = ", ".join(f"{real_traj[i]:.2f}" for i in idx)
    dtraj = ", ".join(f"{dream_traj[i]:.2f}" for i in idx)
    return "\n".join([
        "A policy with one number, theta: push right (a = +1) while x < theta; after that, push left (a = -1) while v > 0 and "
        "otherwise do nothing. The cart starts at (x, v) = (1, 0) and the cost of an episode is the mean of |x - 8| over 40 steps. "
        "theta is searched on the grid 0.0, 0.1, ..., 10.0 (101 values; the first minimum wins a tie) once in the learned model "
        "of the first block and once in the real cart. Each searched policy is then scored in both.",
        "",
        _table(["theta searched", "theta", "cost in the model", "cost in the real cart"], rows, "lrrr"),
        "",
        f"Under the model's choice theta = {tm:.1f}, the position after steps 1, 4, 8, 12, 16 and 40 is x = {dtraj} in the model and "
        f"x = {traj} in the real cart.",
    ])


def _share_plot_data():
    ns = list(range(1, 21))
    return ns, _share_curve(5.0, 0.8, ns), _share_curve(1.0, 0.8, ns)
