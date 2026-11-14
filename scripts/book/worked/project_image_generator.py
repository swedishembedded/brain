# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume VIII project "A Small Image Generator".

Plain Python 3, standard library only, written from the project's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: project_image_generator.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. What the project reuses from the chapters of the volume is
imported, never copied: the book's generator (`bookrng`), the stable-digits rule (`stable`), the Adam update (`first_models`), the
sinusoidal embedding of the noise fraction (`denoiser_architectures`) and the one-dimensional Euler and Heun steps of the sampler
chapter (`samplers_and_guidance`), against which the vector samplers written here are checked.

The world is a mixture of Gaussians that is known exactly:

* 6 x 6 pictures of three classes (a vertical line, a horizontal line, a 2 x 2 block), two positions each, pixel noise 0.25, so that
  the data are six equally likely Gaussians N(template, 0.0625 I) in 36 dimensions, and the class of a picture and its position are
  known for every draw;
* the exact velocity field of that mixture (class-conditional, unconditional and guided), which is the reference for every network;
* a pixel-space denoiser (two hidden layers of 32 tanh units, class embedding with a null row) with a hand-written backward pass, its
  gradient check, its flow-matching training with label dropout and the Euler sampler with classifier-free guidance;
* a variational autoencoder of the pictures (36-24-(4, 4)-24-36), its standardized latent, and the same denoiser trained on the
  latent codes, sampled, decoded and scored by the same measures against the known distribution;
* the cost of every stage as FLOPs and bytes, counted from the shapes.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. Trained
numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the nudges of
`stable.py`. The runs are trained in at most four worker processes.

Swedish Embedded AB implements solutions for conditional image generators whose training, sampling and evaluation are specified,
counted and made reproducible. If your team needs expertise in flow matching and latent image generation, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache

import bookrng
import denoiser_architectures as da
import diffusion
import first_models as fm
import samplers_and_guidance as sg
import stable

# ------------------------------------------------------------------ constants

SIDE, D = 6, 36                              # the pictures: 6 x 6 pixels, row-major
SIGMA_X = 0.25                               # pixel noise of the data (the standard deviation of the decoder, too)
CLASSES = ("column", "row", "block")
N_CLASS, N_POS, N_TEMPLATE = 3, 2, 6
NULL = 3                                     # the row of the class table that stands for "no class"
P_DROP = 0.15                                # probability that a training item loses its class
E_DIM = 4                                    # width of a class embedding
H = 48                                       # hidden width of the denoiser
TIME_DIM, TIME_BASE = 4, 4.0                 # the timestep features: da.embed(pi sigma, 4, base 4)
STEPS, BATCH, LR = 2000, 32, 1e-2            # denoiser training: Adam, LR * (1 - (i - 1) / STEPS) at step i
LOG_STEPS = (250, 500, 1000, 1500, 2000)
Z_DIM, AE_H = 4, 24                          # latent width and hidden width of the autoencoder
AE_STEPS, AE_BATCH, AE_LR = 2000, 32, 3e-3
STAT_N = 1200                                # pictures whose codes fix the latent's per-channel mean and variance
STAT_EPS = 1e-4
EVAL_N, SAMPLE_N = 600, 100                  # evaluation items of the loss tables; starting vectors of the sampling tables (the same 100 serve every class)
NET_STEPS = (4, 8, 16, 32)                   # evaluations of the field per sample in the solver table
SOLVER_N = 50                                # the first starting vectors, used for every class, of the solver table
REF_STEPS = 100                              # Runge-Kutta steps of the reference trajectories
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
G_VALUES = (0.0, 1.0, 2.0, 4.0)
G_STEPS = 16
VALID_RMS = 1.5 * SIGMA_X                    # a sample is "on the data" when its distance to the nearest template is below this, per pixel

DATA, RUN, EVAL, SAMPLE = ("project_image_generator." + k for k in ("data", "run", "evaluation", "sample"))
AE_RUN, AE_STAT, AE_EVAL = ("project_image_generator." + k for k in ("ae_run", "ae_stat", "ae_eval"))
LAT_RUN, DEC_NOISE, GRAD = ("project_image_generator." + k for k in ("latent_run", "decoder_noise", "gradcheck"))
AE_GRAD, REF = "project_image_generator.ae_gradcheck", "project_image_generator.reference"
LAT_START = "project_image_generator.latent_start"

_table, _seed, _f, _g = diffusion._table, diffusion._seed, diffusion._f, diffusion._g


def _int(n):
    return f"{n:,}"


def _fixed(d):
    return lambda x: f"{x:.{d}f}"


def _shared(vals, fmt):
    return stable.text(float(vals[0]), [float(v) for v in vals[1:]], fmt)


def _below(v, bound):
    return f"below {bound:.0e}" if v < bound else f"{v:.1e} (CHECK FAILED)"


def _dot(a, b):
    """The dot product of two equal-length lists: math.sumprod, which accumulates in extended precision; every printed digit of a trained
    number is stable under the nudges of `stable.py`, so no digit depends on the order or the extra precision of this sum."""
    return math.sumprod(a, b)


# ------------------------------------------------------------------ the world: templates, pictures, the exact mixture

def template(c, p):
    """The noise-free picture of class c at position p as 36 values (row-major), +1 where the shape is and -1 elsewhere.
    column: pixels of column 1 + 3 p in every row; row: pixels of row 1 + 3 p in every column; block: the 2 x 2 square whose top-left
    pixel is (1 + 2 p, 1 + 2 p)."""
    img = [-1.0] * D
    for r in range(SIDE):
        for q in range(SIDE):
            if c == 0:
                on = q == 1 + 3 * p
            elif c == 1:
                on = r == 1 + 3 * p
            else:
                on = 1 + 2 * p <= r < 3 + 2 * p and 1 + 2 * p <= q < 3 + 2 * p
            if on:
                img[r * SIDE + q] = 1.0
    return img


TEMPLATES = [template(c, p) for c in range(N_CLASS) for p in range(N_POS)]        # index 4 c + p
SQ = [sum(v * v for v in t) for t in TEMPLATES]


def draw_picture(gen):
    """One picture: c = integer(3), p = integer(2), then 36 pixel noises by normal() in row-major order; template + SIGMA_X normal()."""
    c = gen.integer(N_CLASS)
    p = gen.integer(N_POS)
    t = TEMPLATES[N_POS * c + p]
    return c, p, [v + SIGMA_X * gen.normal() for v in t]


def nearest_template(x):
    """Index of the template nearest to x in squared distance (the lowest index on a tie) and that distance divided by D, rooted."""
    best, bd = 0, math.inf
    for m, t in enumerate(TEMPLATES):
        d = sum((a - b) ** 2 for a, b in zip(x, t))
        if d < bd:
            best, bd = m, d
    return best, math.sqrt(bd / D)


def mixture_weights(x, a, v, members):
    """Posterior weights over the listed templates of the point x_sigma = x: proportional to exp(-|x - a t|^2 / (2 v)), a = 1 - sigma,
    v = a^2 SIGMA_X^2 + sigma^2, normalised after subtracting the largest exponent."""
    ex = []
    for m in members:
        t = TEMPLATES[m]
        ex.append(-sum((xi - a * ti) ** 2 for xi, ti in zip(x, t)) / (2.0 * v))
    top = max(ex)
    w = [math.exp(e - top) for e in ex]
    z = sum(w)
    return [q / z for q in w]


def members_of(c):
    return range(N_TEMPLATE) if c is None or c == NULL else range(N_POS * c, N_POS * c + N_POS)


def exact_guess(x, s, c):
    """E[x_0 | x_sigma = x, class c] for the mixture of templates (all six when c is None or NULL): the sum over the member
    templates t of w_t (t + a SIGMA_X^2 / v (x - a t)) with the weights of `mixture_weights`."""
    a = 1.0 - s
    v = a * a * SIGMA_X ** 2 + s * s
    mem = list(members_of(c))
    w = mixture_weights(x, a, v, mem)
    k = a * SIGMA_X ** 2 / v
    out = [0.0] * D
    for wm, m in zip(w, mem):
        t = TEMPLATES[m]
        for i in range(D):
            out[i] += wm * (t[i] + k * (x[i] - a * t[i]))
    return out


def exact_velocity(x, s, c):
    s = max(s, 1e-9)
    g = exact_guess(x, s, c)
    return [(xi - gi) / s for xi, gi in zip(x, g)]


def guided(field, g):
    """The classifier-free blend u_un + g (u_c - u_un) of two fields given as functions of (x, s, class): the class field of c and the
    field of NULL."""
    def f(x, s, c):
        un = field(x, s, NULL)
        if g == 0.0:
            return un
        cond = field(x, s, c)
        if g == 1.0:
            return cond
        return [u + g * (b - u) for u, b in zip(un, cond)]
    return f


# ------------------------------------------------------------------ samplers (vector form), checked against the chapter's scalars

def grid(n):
    return sg.uniform_grid(n)


def euler(x, gr, f):
    """Euler along the grid: x <- x + (s' - s) f(x, s), element by element (the sampler chapter's step, for a vector)."""
    for i in range(len(gr) - 1):
        s, t = gr[i], gr[i + 1]
        u = f(x, s)
        x = [xi + (t - s) * ui for xi, ui in zip(x, u)]
    return x


def heun(x, gr, f):
    """Heun's step on the velocity: an Euler step, the velocity at its end (at 1e-9 when the end is 0), the average of the two."""
    for i in range(len(gr) - 1):
        s, t = gr[i], gr[i + 1]
        k1 = f(x, s)
        xe = [xi + (t - s) * a for xi, a in zip(x, k1)]
        k2 = f(xe, max(t, 1e-9))
        x = [xi + (t - s) * 0.5 * (a + b) for xi, a, b in zip(x, k1, k2)]
    return x


def rk4(x, n, f):
    """Classical Runge-Kutta from sigma = 1 to 0 in n equal steps (the last stage of the last step is evaluated at 1e-9)."""
    h = 1.0 / n
    for i in range(n):
        s = 1.0 - i * h
        k1 = f(x, s)
        k2 = f([a - 0.5 * h * b for a, b in zip(x, k1)], s - 0.5 * h)
        k3 = f([a - 0.5 * h * b for a, b in zip(x, k2)], s - 0.5 * h)
        k4 = f([a - h * b for a, b in zip(x, k3)], max(s - h, 1e-9))
        x = [a - h / 6.0 * (p + 2.0 * q + 2.0 * r + w) for a, p, q, r, w in zip(x, k1, k2, k3, k4)]
    return x


@lru_cache(maxsize=None)
def start_noise():
    """SAMPLE_N starting vectors of 36 values each, drawn one after another by normal() from the stream `SAMPLE`."""
    gen = bookrng.generator(SAMPLE)
    return tuple(tuple(gen.normal() for _ in range(D)) for _ in range(SAMPLE_N))


def sampler_oracle():
    """Largest difference between the vector Euler and Heun and the scalar ones of `samplers_and_guidance`, on a separable guess."""
    guess1 = lambda x, s: 0.3 * x + 0.1 * s + 0.2            # a one-dimensional guess, used coordinatewise below

    def vec_u(x, s):
        return [(xi - guess1(xi, max(s, 1e-9))) / max(s, 1e-9) for xi in x]
    gen = bookrng.generator(REF)
    x0 = [gen.normal() for _ in range(5)]
    worst = 0.0
    for n in (4, 16):
        gr = grid(n)
        for ours, theirs in ((euler, sg.euler), (heun, sg.heun)):
            a = ours(list(x0), gr, vec_u)
            b = [theirs(xi, gr, guess1) for xi in x0]
            worst = max(worst, max(abs(p - q) for p, q in zip(a, b)))
    return worst


# ------------------------------------------------------------------ the measures against the known distribution

def log_density(x, c):
    """ln p(x | class c) of the known distribution: ln of the mean over the four templates of N(x; t, SIGMA_X^2 I)."""
    ex = [-sum((a - b) ** 2 for a, b in zip(x, TEMPLATES[m])) / (2.0 * SIGMA_X ** 2) for m in members_of(c)]
    top = max(ex)
    return top + math.log(sum(math.exp(e - top) for e in ex) / N_POS) - 0.5 * D * math.log(2.0 * math.pi * SIGMA_X ** 2)


def measures(samples, labels):
    """(class accuracy, share on the data, mean distance to the nearest template, mean ln p(x | class), position TV).
    A sample's class is that of its nearest template; 'on the data' is a distance below VALID_RMS per pixel; position TV is, per class,
    the total variation distance 0.5 sum |share - 1/4| of the positions of the samples requested in that class, averaged over classes."""
    hit = ok = 0
    dist = ld = 0.0
    pos = [[0] * N_POS for _ in range(N_CLASS)]
    cnt = [0] * N_CLASS
    for x, c in zip(samples, labels):
        m, d = nearest_template(x)
        hit += (m // N_POS == c)
        ok += (d < VALID_RMS)
        dist += d
        ld += log_density(x, c)
        pos[c][m % N_POS] += 1
        cnt[c] += 1
    n = len(samples)
    tv = sum(0.5 * sum(abs(q / cnt[c] - 1.0 / N_POS) for q in pos[c]) for c in range(N_CLASS) if cnt[c]) / sum(1 for q in cnt if q)
    return hit / n, ok / n, dist / n, ld / n, tv


@lru_cache(maxsize=None)
def data_balanced():
    """SAMPLE_N pictures of each class, class 0 first: for each, p = integer(2) and then 36 pixel noises by normal(), from the stream `DATA`
    (the same process as `draw_picture`, with the class given instead of drawn)."""
    gen = bookrng.generator(DATA)
    out = []
    for c in range(N_CLASS):
        for _ in range(SAMPLE_N):
            p = gen.integer(N_POS)
            out.append([v + SIGMA_X * gen.normal() for v in TEMPLATES[N_POS * c + p]])
    return out


def data_row():
    return measures(data_balanced(), class_labels(SAMPLE_N))


# ------------------------------------------------------------------ the denoiser

class Net:
    """The shapes of a denoiser for pictures of d values. Input: the picture x (d values) and the class embedding E[c] (4 values); the
    noise fraction enters by scale and shift (Chapter 8): its features tf (4 values, da.embed(pi sigma, 4, base 4)) give each hidden unit
    a gain 1 + gamma and a shift beta. Two hidden layers of H tanh units,
        h1 = tanh((1 + G1 tf + c1) * (W1 [x, E[c]] + b1) + B1 tf + e1),   h2 = tanh((1 + G2 tf + c2) * (W2 h1 + b2) + B2 tf + e2),
    and a linear output W3 h2 + b3 of d values. Flat order of the parameters: E (4 x 4: rows 0..2 the classes, row 3 the null class),
    W1 (H x (d + 4)), b1, G1 (H x 4), c1, B1 (H x 4), e1, W2 (H x H), b2, G2, c2, B2, e2, W3 (d x H), b3; matrices row-major."""

    def __init__(self, d):
        self.d = d
        self.nin = d + E_DIM
        self.shapes = [(4, E_DIM), (H, self.nin), (H,), (H, TIME_DIM), (H,), (H, TIME_DIM), (H,),
                       (H, H), (H,), (H, TIME_DIM), (H,), (H, TIME_DIM), (H,), (d, H), (d,)]
        self.n_params = sum(math.prod(sh) for sh in self.shapes)

    def init(self, gen):
        """E by normal(), W1 by normal() / sqrt(d + 4), W2 and W3 by normal() / sqrt(H), drawn in the order E, W1, W2, W3, each row-major;
        every bias and every modulation weight and bias 0, so that the gains start at 1 and the shifts at 0."""
        e = [gen.normal() for _ in range(4 * E_DIM)]
        w1 = [gen.normal() / math.sqrt(self.nin) for _ in range(H * self.nin)]
        w2 = [gen.normal() / math.sqrt(H) for _ in range(H * H)]
        w3 = [gen.normal() / math.sqrt(H) for _ in range(self.d * H)]
        mod = [0.0] * (2 * H * TIME_DIM + 2 * H)
        return e + w1 + [0.0] * H + mod + w2 + [0.0] * H + mod + w3 + [0.0] * self.d

    def unpack(self, P):
        """Row lists of the parameters (a copy), the form the forward and backward passes use."""
        o, out = 0, []
        for shape in self.shapes:
            if len(shape) == 2:
                rows, cols = shape
                out.append([P[o + r * cols:o + (r + 1) * cols] for r in range(rows)])
                o += rows * cols
            else:
                out.append(P[o:o + shape[0]])
                o += shape[0]
        return out

    def zeros_like(self):
        return [[[0.0] * len(r) for r in part] if isinstance(part[0], list) else [0.0] * len(part) for part in self.unpack([0.0] * self.n_params)]

    @staticmethod
    def flatten(parts):
        flat = []
        for part in parts:
            if isinstance(part[0], list):
                for r in part:
                    flat += r
            else:
                flat += part
        return flat

    def forward(self, W, x, s, c):
        """Output and the activations. Every sum runs over the input index in increasing order."""
        tf = da.embed(math.pi * s, TIME_DIM, TIME_BASE)
        a0 = list(x) + W[0][c]
        z1 = [b + _dot(r, a0) for r, b in zip(W[1], W[2])]
        g1 = [1.0 + c_ + _dot(r, tf) for r, c_ in zip(W[3], W[4])]
        h1 = [math.tanh(g * z + e + _dot(r, tf)) for g, z, r, e in zip(g1, z1, W[5], W[6])]
        z2 = [b + _dot(r, h1) for r, b in zip(W[7], W[8])]
        g2 = [1.0 + c_ + _dot(r, tf) for r, c_ in zip(W[9], W[10])]
        h2 = [math.tanh(g * z + e + _dot(r, tf)) for g, z, r, e in zip(g2, z2, W[11], W[12])]
        out = [b + _dot(r, h2) for r, b in zip(W[13], W[14])]
        return out, (a0, tf, z1, g1, h1, z2, g2, h2)

    def batch_grad(self, W, caches, classes, douts):
        """The gradient (the structure of `unpack`) of a loss that sums over the items of a batch, given the activations of every item,
        its class and dout = d loss / d out for it. Per item: the vectors that arrive at each pre-activation, by the chain rule of the
        formulas of the class docstring (du = dh (1 - h^2), dz = du g, dg = du z, dshift = du). Then every weight gradient is the sum
        over the items of a product of two of those vectors, taken as one dot product per weight."""
        WT = self.transposes(W)
        deltas = []
        for (a0, tf, z1, g1, h1, z2, g2, h2), dout in zip(caches, douts):
            dh2 = [_dot(col, dout) for col in WT[0]]
            du2 = [d * (1.0 - q * q) for d, q in zip(dh2, h2)]
            dz2 = [d * q for d, q in zip(du2, g2)]
            dg2 = [d * q for d, q in zip(du2, z2)]
            dh1 = [_dot(col, dz2) for col in WT[1]]
            du1 = [d * (1.0 - q * q) for d, q in zip(dh1, h1)]
            dz1 = [d * q for d, q in zip(du1, g1)]
            dg1 = [d * q for d, q in zip(du1, z1)]
            da0 = [_dot(col, dz1) for col in WT[2]]
            deltas.append((dout, dz2, dg2, du2, dz1, dg1, du1, da0))
        col = lambda k: [list(v) for v in zip(*[d[k] for d in deltas])]       # the k-th vector of every item, transposed
        act = lambda k: [list(v) for v in zip(*[cc[k] for cc in caches])]
        dout_t, dz2_t, dg2_t, du2_t, dz1_t, dg1_t, du1_t = (col(k) for k in range(7))
        a0_t, tf_t, h1_t, h2_t = act(0), act(1), act(4), act(7)
        G = self.zeros_like()
        G[13] = [[_dot(dk, hj) for hj in h2_t] for dk in dout_t]
        G[14] = [sum(dk) for dk in dout_t]
        for w, dz_t, dg_t, du_t, inp_t in ((7, dz2_t, dg2_t, du2_t, h1_t), (1, dz1_t, dg1_t, du1_t, a0_t)):
            G[w] = [[_dot(dj, q) for q in inp_t] for dj in dz_t]
            G[w + 1] = [sum(dj) for dj in dz_t]
            G[w + 2] = [[_dot(dj, q) for q in tf_t] for dj in dg_t]
            G[w + 3] = [sum(dj) for dj in dg_t]
            G[w + 4] = [[_dot(dj, q) for q in tf_t] for dj in du_t]
            G[w + 5] = [sum(dj) for dj in du_t]
        for c, d in zip(classes, deltas):
            G[0][c] = [g + q for g, q in zip(G[0][c], d[7][self.d:])]
        return G

    @staticmethod
    def transposes(W):
        return [list(zip(*W[13])), list(zip(*W[7])), list(zip(*W[1]))]

    def pass_flops(self):
        """FLOPs of one forward pass: 2 per weight multiplied (a multiply and an add, including the two 4-value dot products of the modulation
        of every hidden unit), 1 per bias added, 3 per tanh, and 5 per hidden unit of each layer for the modulation (add c, add 1, multiply
        by z, add e, add the shift). The timestep features and the class lookup are not counted."""
        macs = H * self.nin + H * H + self.d * H + 2 * 2 * H * TIME_DIM
        return 2 * macs + (2 * H + self.d) + 3 * 2 * H + 5 * 2 * H


PIXEL_NET, LATENT_NET = Net(D), Net(Z_DIM)


# ------------------------------------------------------------------ training items

def draw_training_item(gen, encode=None):
    """One item: sigma = uniform(); the class c = integer(3) and the position p = integer(2); drop = uniform() (the class becomes NULL when
    drop < P_DROP); the picture by 36 normal() in row-major order (template + SIGMA_X normal()); the data value x_0 (the picture, or
    `encode(picture)` for the latent system); and the noise by D normal() in order. Returns (sigma, x_sigma, target, class label)
    with x_sigma = (1 - sigma) x_0 + sigma noise and target = noise - x_0."""
    s = gen.uniform()
    c = gen.integer(N_CLASS)
    p = gen.integer(N_POS)
    drop = gen.uniform()
    t = TEMPLATES[N_POS * c + p]
    pic = [v + SIGMA_X * gen.normal() for v in t]
    x0 = encode(pic) if encode else pic
    eps = [gen.normal() for _ in x0]
    x = [(1.0 - s) * a + s * b for a, b in zip(x0, eps)]
    return s, x, [b - a for a, b in zip(x0, eps)], (NULL if drop < P_DROP else c)


def adam_flat(P, G, m, v, i, lr):
    fm.adamw_step(P, G, m, v, i, lr, 0.0)


def train_net(net, gen, P, encode=None, nudge=0.0, on_step=None):
    """Flow matching with label dropout: STEPS steps, each BATCH items drawn one after another by `draw_training_item` from `gen`; the loss
    of a batch is the mean over items of the mean over coordinates of (out - target)^2; Adam (beta1 0.9, beta2 0.999, epsilon 1e-8,
    no weight decay) with rate LR (1 - (i - 1) / STEPS) at step i. The first weight (E[0][0]) is moved by `nudge` before step 1."""
    P[0] += nudge
    m, v = [0.0] * net.n_params, [0.0] * net.n_params
    d = net.d
    for i in range(1, STEPS + 1):
        W = net.unpack(P)
        caches, classes, douts = [], [], []
        for _ in range(BATCH):
            s, x, y, c = draw_training_item(gen, encode)
            o, cache = net.forward(W, x, s, c)
            caches.append(cache)
            classes.append(c)
            douts.append([2.0 * (a - b) / (d * BATCH) for a, b in zip(o, y)])
        adam_flat(P, net.flatten(net.batch_grad(W, caches, classes, douts)), m, v, i, LR * (1.0 - (i - 1) / STEPS))
        if on_step:
            on_step(i, P)
    return P


# ------------------------------------------------------------------ the autoencoder

AE_SIZES = [AE_H * D, AE_H, 2 * Z_DIM * AE_H, 2 * Z_DIM, AE_H * Z_DIM, AE_H, D * AE_H, D]
AE_PARAMS = sum(AE_SIZES)
AE_SHAPES = ((AE_H, D), (AE_H,), (2 * Z_DIM, AE_H), (2 * Z_DIM,), (AE_H, Z_DIM), (AE_H,), (D, AE_H), (D,))


def ae_init(gen):
    """Weights drawn in the order encoder hidden (36 -> 24), encoder head (24 -> 8), decoder hidden (4 -> 24), decoder output (24 -> 36),
    each by normal() / sqrt(fan_in) in row-major order, biases 0. The flat vector is in the order of AE_SIZES: We1, be1, We2, be2,
    Wd1, bd1, Wd2, bd2. The head's rows 0..3 give the mean m and rows 4..7 the log-variance v."""
    we1 = [gen.normal() / math.sqrt(D) for _ in range(AE_H * D)]
    we2 = [gen.normal() / math.sqrt(AE_H) for _ in range(2 * Z_DIM * AE_H)]
    wd1 = [gen.normal() / math.sqrt(Z_DIM) for _ in range(AE_H * Z_DIM)]
    wd2 = [gen.normal() / math.sqrt(AE_H) for _ in range(D * AE_H)]
    return we1 + [0.0] * AE_H + we2 + [0.0] * (2 * Z_DIM) + wd1 + [0.0] * AE_H + wd2 + [0.0] * D


def ae_unpack(P):
    o, out = 0, []
    for shape in AE_SHAPES:
        if len(shape) == 2:
            rows, cols = shape
            out.append([P[o + r * cols:o + (r + 1) * cols] for r in range(rows)])
            o += rows * cols
        else:
            out.append(P[o:o + shape[0]])
            o += shape[0]
    return out


def ae_zeros():
    return [[[0.0] * len(r) for r in part] if isinstance(part[0], list) else [0.0] * len(part) for part in ae_unpack([0.0] * AE_PARAMS)]


def ae_encode(W, x):
    """(h, m, v): h = tanh(We1 x + be1); the head gives m (4) and the log-variance v (4)."""
    h = [math.tanh(b + _dot(r, x)) for r, b in zip(W[0], W[1])]
    o = [b + _dot(r, h) for r, b in zip(W[2], W[3])]
    return h, o[:Z_DIM], o[Z_DIM:]


def ae_decode(W, z):
    hd = [math.tanh(b + _dot(r, z)) for r, b in zip(W[4], W[5])]
    return hd, [b + _dot(r, hd) for r, b in zip(W[6], W[7])]


def ae_loss_grad(W, x, xi, G, weight):
    """The negative evidence lower bound of one picture, with z = m + exp(v / 2) xi:
        loss = |x - dec(z)|^2 / (2 SIGMA_X^2) + (1/2) sum (m^2 + exp(v) - 1 - v)
    (the constant (D/2) ln(2 pi SIGMA_X^2) is left out). Adds weight times its gradient to G and returns the loss."""
    h, m, v = ae_encode(W, x)
    sd = [math.exp(0.5 * q) for q in v]
    z = [a + b * c for a, b, c in zip(m, sd, xi)]
    hd, xhat = ae_decode(W, z)
    res = [a - b for a, b in zip(xhat, x)]
    rec = sum(r * r for r in res) / (2.0 * SIGMA_X ** 2)
    kl = 0.5 * sum(a * a + math.exp(q) - 1.0 - q for a, q in zip(m, v))
    # backward
    dxhat = [weight * r / SIGMA_X ** 2 for r in res]
    for k, dk in enumerate(dxhat):
        G[7][k] += dk
        G[6][k] = [g + dk * q for g, q in zip(G[6][k], hd)]
    dhd = [_dot(col, dxhat) for col in zip(*W[6])]
    d2 = [d * (1.0 - q * q) for d, q in zip(dhd, hd)]
    for j, dj in enumerate(d2):
        G[5][j] += dj
        G[4][j] = [g + dj * q for g, q in zip(G[4][j], z)]
    dz = [_dot(col, d2) for col in zip(*W[4])]
    dm = [a + weight * b for a, b in zip(dz, m)]
    dv = [a * 0.5 * s * x_ + weight * 0.5 * (math.exp(q) - 1.0) for a, s, x_, q in zip(dz, sd, xi, v)]
    do = dm + dv
    for k, dk in enumerate(do):
        G[3][k] += dk
        G[2][k] = [g + dk * q for g, q in zip(G[2][k], h)]
    dh = [_dot(col, do) for col in zip(*W[2])]
    d1 = [d * (1.0 - q * q) for d, q in zip(dh, h)]
    for j, dj in enumerate(d1):
        G[1][j] += dj
        G[0][j] = [g + dj * q for g, q in zip(G[0][j], x)]
    return rec + kl, rec, kl


def ae_draw_item(gen):
    """One picture (class, position, 36 pixel noises as in `draw_picture`) and then Z_DIM normal() draws for the reparameterization."""
    c, p, x = draw_picture(gen)
    return x, [gen.normal() for _ in range(Z_DIM)]


def train_ae(nudge=0.0):
    """AE_STEPS steps of AE_BATCH items drawn from the stream `AE_RUN` after the initialization; the loss of a batch is the mean over items
    of the negative bound of `ae_loss_grad`; Adam as for the denoiser with rate AE_LR (1 - (i - 1) / AE_STEPS). Returns the parameters."""
    gen = bookrng.generator(AE_RUN)
    P = ae_init(gen)
    P[0] += nudge
    m, v = [0.0] * AE_PARAMS, [0.0] * AE_PARAMS
    for i in range(1, AE_STEPS + 1):
        W = ae_unpack(P)
        G = ae_zeros()
        for _ in range(AE_BATCH):
            x, xi = ae_draw_item(gen)
            ae_loss_grad(W, x, xi, G, 1.0 / AE_BATCH)
        flat = [q for part in G for q in (sum(part, []) if isinstance(part[0], list) else part)]
        adam_flat(P, flat, m, v, i, AE_LR * (1.0 - (i - 1) / AE_STEPS))
    return P


# ------------------------------------------------------------------ the latent system

@lru_cache(maxsize=None)
def stat_pictures():
    gen = bookrng.generator(AE_STAT)
    return tuple(draw_picture(gen)[2] for _ in range(STAT_N))


def latent_stats(Wae):
    """Mean and variance (divisor n) of each of the Z_DIM channels of the encoder's mean m(x) over the STAT_N pictures of the stream
    `AE_STAT` (each drawn as in `draw_picture`)."""
    codes = [ae_encode(Wae, x)[1] for x in stat_pictures()]
    mean = [sum(c[k] for c in codes) / len(codes) for k in range(Z_DIM)]
    var = [sum((c[k] - mean[k]) ** 2 for c in codes) / len(codes) for k in range(Z_DIM)]
    return mean, var


def to_latent(Wae, stats, x):
    """The denoiser's data value: the encoder's mean, standardized per channel, (m - mean) / sqrt(var + STAT_EPS)."""
    mean, var = stats
    m = ae_encode(Wae, x)[1]
    return [(a - b) / math.sqrt(q + STAT_EPS) for a, b, q in zip(m, mean, var)]


def from_latent(Wae, stats, z):
    """The inverse of the standardization followed by the decoder's mean: z * sqrt(var + STAT_EPS) + mean, then ae_decode."""
    mean, var = stats
    return ae_decode(Wae, [a * math.sqrt(q + STAT_EPS) + b for a, b, q in zip(z, mean, var)])[1]


@lru_cache(maxsize=None)
def latent_start():
    """SAMPLE_N starting vectors of Z_DIM values each, by normal() in order from the stream `LAT_START`."""
    gen = bookrng.generator(LAT_START)
    return tuple(tuple(gen.normal() for _ in range(Z_DIM)) for _ in range(SAMPLE_N))


# ------------------------------------------------------------------ evaluation of a trained denoiser

@lru_cache(maxsize=None)
def eval_items():
    """EVAL_N training-format items from the stream `EVAL` (pixel space)."""
    gen = bookrng.generator(EVAL)
    return tuple(draw_training_item(gen) for _ in range(EVAL_N))


def pixel_losses(net, W, items):
    """(mean (net - target)^2, mean (exact - target)^2, mean (net - exact)^2), each averaged over the coordinates and then over the items;
    the exact field of an item's own label (the unconditional one for the null label)."""
    a = b = c_ = 0.0
    for s, x, y, c in items:
        o = net.forward(W, x, s, c)[0]
        u = exact_velocity(x, s, c)
        a += sum((p - q) ** 2 for p, q in zip(o, y)) / D
        b += sum((p - q) ** 2 for p, q in zip(u, y)) / D
        c_ += sum((p - q) ** 2 for p, q in zip(o, u)) / D
    n = len(items)
    return a / n, b / n, c_ / n


def net_field(net, W):
    return lambda x, s, c: net.forward(W, x, s, c)[0]


def sample_class(field, start, c, g, n):
    """End points of the guided Euler sampler with n steps for class c from the starting vectors `start`."""
    gf = guided(field, g)
    gr = grid(n)
    return [euler(list(e), gr, lambda x, s: gf(x, s, c)) for e in start]


def class_labels(n_per_class):
    return [c for c in range(N_CLASS) for _ in range(n_per_class)]


def guidance_rows(make_samples):
    """For each g of G_VALUES: the five measures of the samples `make_samples(g)` returns for class 0, then 1, then 2 (SAMPLE_N each)."""
    return {g: measures(make_samples(g), class_labels(SAMPLE_N)) for g in G_VALUES}


def pixel_samples(net, W, g, n=G_STEPS):
    start = start_noise()
    if g == 0.0:                                     # the unconditional field does not depend on the class: one run serves the three
        one = sample_class(net_field(net, W), start, NULL, 1.0, n)
        return one * N_CLASS
    out = []
    for c in range(N_CLASS):
        out += sample_class(net_field(net, W), start, c, g, n)
    return out


def exact_samples(g, n=G_STEPS):
    out = []
    for c in range(N_CLASS):
        out += sample_class(exact_velocity_field, start_noise(), c, g, n)
    return out


def exact_velocity_field(x, s, c):
    return exact_velocity(x, s, c)


@lru_cache(maxsize=None)
def reference_endpoints():
    """The exact class-conditional field carried from the SAMPLE_N starting vectors of each class to sigma = 0 by REF_STEPS Runge-Kutta steps."""
    out = []
    for c in range(N_CLASS):
        f = lambda x, s, c=c: exact_velocity(x, s, c)
        out += [rk4(list(e), REF_STEPS, f) for e in start_noise()]
    return out


def rms_distance(xs, ref):
    return sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(x, r)) / D) for x, r in zip(xs, ref)) / len(xs)


def solver_distances(field):
    """Mean root-mean-square distance (per pixel) to the reference end points, over the first SOLVER_N starting values of each class, of
    class-conditional Euler with n steps and Heun with n / 2 steps (n evaluations each), for n in NET_STEPS."""
    ref = reference_endpoints()
    out = {}
    for name, solver, per_step in (("euler", euler, 1), ("heun", heun, 2)):
        for n in NET_STEPS:
            xs, rs = [], []
            for c in range(N_CLASS):
                gr = grid(n // per_step)
                f = lambda x, s, c=c: field(x, s, c)
                xs += [solver(list(e), gr, f) for e in start_noise()[:SOLVER_N]]
                rs += ref[c * SAMPLE_N:c * SAMPLE_N + SOLVER_N]
            out[(name, n)] = rms_distance(xs, rs)
    return out


@lru_cache(maxsize=None)
def exact_solver_distances():
    return solver_distances(exact_velocity_field)


@lru_cache(maxsize=None)
def exact_guidance_rows():
    return guidance_rows(exact_samples)


# ------------------------------------------------------------------ the two systems as training jobs

def pixel_job(nudge):
    """Train the pixel denoiser (the stream `RUN`, the first weight moved by `nudge`) and evaluate it: the loss log at LOG_STEPS, the bands of
    sigma, the solver distances, the guidance rows and a few samples for the figures."""
    net = PIXEL_NET
    gen = bookrng.generator(RUN)
    P = net.init(gen)
    items = eval_items()
    log = {}

    def on_step(i, params):
        if i in LOG_STEPS:
            log[i] = pixel_losses(net, net.unpack(params), items)
    train_net(net, gen, P, nudge=nudge, on_step=on_step)
    W = net.unpack(P)
    bands = [pixel_losses(net, W, tuple(it for it in items if lo <= it[0] < hi)) + (sum(1 for it in items if lo <= it[0] < hi),)
             for lo, hi in BANDS]
    rows = guidance_rows(lambda g: pixel_samples(net, W, g))
    field = net_field(net, W)
    wrong_sign = []
    for c in range(N_CLASS):
        wrong_sign += sample_class(field, start_noise(), c, -1.0, G_STEPS)
    mistakes = {"one step": measures(pixel_samples(net, W, 1.0, 1), class_labels(SAMPLE_N)),
                "wrong sign": measures(wrong_sign, class_labels(SAMPLE_N))}
    gap = {}
    for g in (1.0, 2.0, 4.0):
        net_g, exact_g = guided(field, g), guided(exact_velocity_field, g)
        used = [it for it in items if it[3] != NULL]
        gap[g] = sum(sum((a - b) ** 2 for a, b in zip(net_g(x, s, c), exact_g(x, s, c))) / D for s, x, y, c in used) / len(used)
    return {"P": P, "log": log, "bands": bands, "solver": solver_distances(field), "guidance": rows, "mistakes": mistakes, "gap": gap,
            "pictures": {g: pixel_samples(net, W, g) for g in (1.0, 4.0)}}


def ae_metrics(Wae, stats):
    """Per-pixel mean squared reconstruction error of the AE_EVAL pictures from the mean code, the mean KL of their codes (nats per
    picture), the root-mean-square error of the six templates' round trip, and the per-channel mean and variance of the codes."""
    gen = bookrng.generator(AE_EVAL)
    pics = [draw_picture(gen)[2] for _ in range(EVAL_N)]
    mse = kl = 0.0
    for x in pics:
        h, m, v = ae_encode(Wae, x)
        mse += sum((a - b) ** 2 for a, b in zip(ae_decode(Wae, m)[1], x)) / D
        kl += 0.5 * sum(a * a + math.exp(q) - 1.0 - q for a, q in zip(m, v))
    rt = 0.0
    for t in TEMPLATES:
        z = ae_encode(Wae, t)[1]
        rt += math.sqrt(sum((a - b) ** 2 for a, b in zip(ae_decode(Wae, z)[1], t)) / D)
    return mse / len(pics), kl / len(pics), rt / N_TEMPLATE


def latent_samples(net, W, Wae, stats, g, n=G_STEPS, undo=True, add_noise=True):
    """Pictures of the latent system for the starting vectors of `LAT_START`: the guided Euler sampler in the latent, the decoder's mean
    of the un-standardized code (`undo`; with undo false the code goes to the decoder as it is), and SIGMA_X normal() on every pixel
    (`add_noise`), the draws taken in row-major order sample after sample from a fresh generator of `DEC_NOISE` (class 0's samples
    first, then class 1's, then class 2's)."""
    noise = bookrng.generator(DEC_NOISE)
    start = latent_start()
    if g == 0.0:
        zs = sample_class(net_field(net, W), start, NULL, 1.0, n) * N_CLASS
    else:
        zs = []
        for c in range(N_CLASS):
            zs += sample_class(net_field(net, W), start, c, g, n)
    plain = ([0.0] * Z_DIM, [1.0 - STAT_EPS] * Z_DIM)
    out = []
    for z in zs:
        mean = from_latent(Wae, stats if undo else plain, z)
        out.append([a + SIGMA_X * noise.normal() if add_noise else a for a in mean])
    return out


def latent_job(nudge):
    """Train the autoencoder (the stream `AE_RUN`) and then the latent denoiser on the codes of fresh pictures (the stream `LAT_RUN`), both with
    their first weight moved by `nudge`; evaluate the autoencoder, the guidance rows and the planted mistakes."""
    Wae_flat = train_ae(nudge)
    Wae = ae_unpack(Wae_flat)
    stats = latent_stats(Wae)
    encode = lambda x: to_latent(Wae, stats, x)
    net = LATENT_NET
    gen = bookrng.generator(LAT_RUN)
    P = net.init(gen)
    train_net(net, gen, P, encode=encode, nudge=nudge)
    W = net.unpack(P)
    rows = guidance_rows(lambda g: latent_samples(net, W, Wae, stats, g))
    labels = class_labels(SAMPLE_N)
    mistakes = {"no inverse standardization": measures(latent_samples(net, W, Wae, stats, 1.0, undo=False), labels),
                "no decoder noise": measures(latent_samples(net, W, Wae, stats, 1.0, add_noise=False), labels)}
    return {"P": P, "ae": ae_metrics(Wae, stats), "stats": stats, "guidance": rows, "mistakes": mistakes,
            "pictures": {g: latent_samples(net, W, Wae, stats, g) for g in (1.0, 4.0)}}


def _job(spec):
    kind, nudge = spec
    return pixel_job(nudge) if kind == "pixel" else latent_job(nudge)


@lru_cache(maxsize=None)
def all_runs():
    """Five pixel runs and five latent runs (the run and its four nudged copies), trained in four worker processes; returns
    (pixel_results, latent_results), each a list of five in the order of stable.NUDGES after the nominal run."""
    reference_endpoints()
    exact_solver_distances()
    exact_guidance_rows()
    specs = [(k, n) for n in (0.0, *stable.NUDGES) for k in ("latent", "pixel")]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        res = pool.map(_job, specs, chunksize=1)
    return [r for (k, _), r in zip(specs, res) if k == "pixel"], [r for (k, _), r in zip(specs, res) if k == "latent"]


@lru_cache(maxsize=None)
def nominal():
    """The pixel and the latent run without nudges, trained in two worker processes (for the figures); the same computation as the first
    entries of all_runs."""
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(2) as pool:
        a, b = pool.map(_job, [("pixel", 0.0), ("latent", 0.0)], chunksize=1)
    return a, b


# ================================================================== the tables of the chapter

MEASURE_HEAD = ["accuracy", "on data", "distance", "mean ln p", "TV"]


def _pct(x):
    return f"{100.0 * x:.1f}%"


def _mrow(m):
    return [f"{m[0]:.3f}", f"{m[1]:.3f}", f"{m[2]:.3f}", f"{m[3]:.2f}", f"{m[4]:.3f}"]


def _stable_measures(vals):
    """Five printed cells from the same measure tuple in the run and in its four nudged copies."""
    fmts = (_fixed(3), _fixed(3), _fixed(3), _fixed(2), _fixed(3))
    return [_shared([v[i] for v in vals], fmts[i]) for i in range(5)]


def _ascii(c, p):
    t = TEMPLATES[N_POS * c + p]
    return ["".join("#" if t[r * SIDE + q] > 0 else "." for q in range(SIDE)) for r in range(SIDE)]


def world():
    rows = []
    for r in range(SIDE):
        rows.append([f"`{_ascii(c, p)[r]}`" for c in range(N_CLASS) for p in range(N_POS)])
    head = [f"{CLASSES[c]} {p}" for c in range(N_CLASS) for p in range(N_POS)]
    gen = bookrng.generator(DATA)
    first = []
    for k in range(4):
        c, p, x = draw_picture(gen)
        m, d = nearest_template(x)
        first.append([k, f"{CLASSES[c]} {p}", f"{sum(x):.6f}", f"{x[0]:.6f}", f"{d:.6f}"])
    return ("The six templates of the world, 6 x 6 pixels, `#` is +1 and `.` is -1 (row 0 at the top). Class c is 0 column, 1 row, 2 block; "
            "position p is 0 or 1; template index 2 c + p.\n\n" + _table(head, rows, "l" * 6)
            + f"\n\nA picture is drawn from {_seed(DATA)}: c = integer(3), p = integer(2), then 36 values normal() in row-major order, "
              f"pixel = template + {SIGMA_X} normal(). The first four pictures of the stream, with the sum of their 36 pixels, their first "
              f"pixel and their root-mean-square distance to the nearest template per pixel:\n\n"
            + _table(["draw", "class, position", "sum of pixels", "first pixel", "distance"], first))


def opening():
    labels = class_labels(SAMPLE_N)
    data = data_row()
    alt = [TEMPLATES[N_POS * c + (i % N_POS)] for i, c in enumerate(labels)]
    first = [TEMPLATES[N_POS * c] for c in labels]
    # an average of the two positions of a class: the picture that minimizes the mean squared distance to the class
    mean = [[0.5 * (a + b) for a, b in zip(TEMPLATES[N_POS * c], TEMPLATES[N_POS * c + 1])] for c in labels]
    rows = [["the data, 300 pictures"] + _mrow(data), ["templates, positions alternating"] + _mrow(measures(alt, labels)),
            ["template of position 0 only"] + _mrow(measures(first, labels)),
            ["mean of the class's two templates"] + _mrow(measures(mean, labels))]
    ideal = -0.5 * D * (1.0 + math.log(2.0 * math.pi * SIGMA_X ** 2)) - math.log(N_POS)
    peak = -0.5 * D * math.log(2.0 * math.pi * SIGMA_X ** 2) - math.log(N_POS)
    return (f"Four ways to produce 300 pictures (100 requests for each of the three classes; the data are drawn class by class) and the five measures of the project. 'Accuracy' "
            f"is the share whose nearest template is of the requested class; 'on data' the share whose distance to that template, per pixel, "
            f"is below {VALID_RMS}; 'distance' the mean of that distance (the data's own is about {SIGMA_X}); the 'mean ln p' column the mean of "
            f"ln p(x given class) under the exact mixture; 'TV' the total variation distance of the positions among each class's pictures "
            f"from the even split (0.5 sum |share - 1/2|, averaged over the classes). The data row is drawn from {_seed(DATA)} (for each class in turn, 100 times p = integer(2) and then 36 normal()). The expected "
            f"value of the 'mean ln p' column for the data is -(D/2)(1 + ln(2 pi sigma_x^2)) - ln 2 = {ideal:.2f}, and for a picture that sits exactly "
            f"on a template -(D/2) ln(2 pi sigma_x^2) - ln 2 = {peak:.2f}.\n\n"
            + _table(["generator"] + MEASURE_HEAD, rows))


def exact_checks():
    gen = bookrng.generator(REF)
    worst_prior = worst_land = worst_post = worst_score = worst_sampler = 0.0
    pts = [[gen.normal() for _ in range(D)] for _ in range(4)]
    # the field at sigma = 1 is x minus the mean of the member templates
    for x in pts:
        for c in (0, 1, 2, NULL):
            mem = list(members_of(c))
            mean = [sum(TEMPLATES[m][i] for m in mem) / len(mem) for i in range(D)]
            u = exact_velocity(x, 1.0, c)
            worst_prior = max(worst_prior, max(abs(a - (xi - b)) for a, xi, b in zip(u, x, mean)))
    # one Euler step with the field from sigma to 0 lands on the guess
    for x in pts:
        for s in (0.3, 0.7):
            u = exact_velocity(x, s, 1)
            g = exact_guess(x, s, 1)
            worst_land = max(worst_land, max(abs(xi + (0.0 - s) * ui - gi) for xi, ui, gi in zip(x, u, g)))
    # the unconditional velocity is the posterior-weighted sum of the class velocities
    for x in pts:
        s = 0.5
        a = 1.0 - s
        v = a * a * SIGMA_X ** 2 + s * s
        w = mixture_weights(x, a, v, range(N_TEMPLATE))
        pc = [sum(w[N_POS * c:N_POS * c + N_POS]) for c in range(N_CLASS)]
        un = exact_velocity(x, s, NULL)
        mix = [sum(pc[c] * exact_velocity(x, s, c)[i] for c in range(N_CLASS)) for i in range(D)]
        worst_post = max(worst_post, max(abs(p - q) for p, q in zip(un, mix)))

    def logp(x, s, c):
        a = 1.0 - s
        v = a * a * SIGMA_X ** 2 + s * s
        ex = [-sum((xi - a * ti) ** 2 for xi, ti in zip(x, TEMPLATES[m])) / (2.0 * v) for m in members_of(c)]
        top = max(ex)
        return top + math.log(sum(math.exp(e - top) for e in ex))
    # the velocity equals -(x + sigma score)/(1 - sigma), with the score by central differences of the log density
    for x in pts[:2]:
        for s in (0.4, 0.8):
            u = exact_velocity(x, s, 1)
            for i in (0, 7, 20, 35):
                h = 1e-5
                xp, xm = list(x), list(x)
                xp[i] += h
                xm[i] -= h
                score = (logp(xp, s, 1) - logp(xm, s, 1)) / (2.0 * h)
                worst_score = max(worst_score, abs(u[i] + (x[i] + s * score) / (1.0 - s)))
    worst_sampler = sampler_oracle()
    crit = lambda v, b: f"below {b:.0e}" if v < b else f"{v:.1e} (CHECK FAILED)"
    return (f"Checks of the exact field on points drawn from {_seed(REF)} (each point is 36 values by normal() in order; four points, then "
            f"the five-value start of the sampler check):\n\n| identity | largest difference |\n|---|---:|\n"
            f"| at sigma = 1 the field of class c is x minus the mean of the class's templates (and of all six for the null class), 16 cases | {crit(worst_prior, 1e-12)} |\n"
            f"| one Euler step from sigma = 0.3 or 0.7 to 0 lands on the guess E[x_0 given x_sigma, class 1], 8 cases | {crit(worst_land, 1e-12)} |\n"
            f"| unconditional field = sum over classes of p(class given x) times the class field, sigma = 0.5, 4 cases | {crit(worst_post, 1e-12)} |\n"
            f"| class-1 field = -(x + sigma s)/(1 - sigma), s the central difference (step 1e-5) of the log density, sigma 0.4 and 0.8, 16 cases | {crit(worst_score, 1e-6)} |\n"
            f"| vector Euler and Heun against the scalar ones of Chapter 7 on a separable guess, 4 and 16 steps | {crit(worst_sampler, 1e-12)} |")


def exact_reference():
    labels = class_labels(SAMPLE_N)
    rows = [["the data, 300 pictures", "-"] + _mrow(data_row())]
    rows.append(["exact field, RK4", f"{REF_STEPS}, g = 1"] + _mrow(measures(reference_endpoints(), labels)))
    ex = exact_guidance_rows()
    for g in G_VALUES:
        rows.append(["exact field, Euler", f"{G_STEPS}, g = {g:g}"] + _mrow(ex[g]))
    return (f"The exact field as a generator: {SAMPLE_N} starting vectors in all (the first {SAMPLE_N} draws of 36 values each by normal() from "
            f"{_seed(SAMPLE)}, the same for every row and for the networks later), each carried to a picture of each of the three classes, so "
            f"that a row has {SAMPLE_N * N_CLASS} samples, class 0's first. They are carried from sigma = 1 to 0 by the exact class field (g = 1), by the guided field "
            f"u_un + g (u_c - u_un) and by 100 Runge-Kutta steps (the classical rule, with the field evaluated at max(sigma, 1e-9) in the last "
            f"stage of the last step, where it is 0/0 at sigma = 0). For g = 0 the three classes get the same pictures.\n\n"
            + _table(["sampler", "steps"] + MEASURE_HEAD, rows))


def _fmt_cells(vals, fmt):
    return _shared(vals, fmt)


def network():
    net = PIXEL_NET
    gen = bookrng.generator(RUN)
    P = net.init(gen)
    W = net.unpack(P)
    names = ["E", "W1", "b1", "G1", "c1", "B1", "e1", "W2", "b2", "G2", "c2", "B2", "e2", "W3", "b3"]
    rows = []
    for nm, sh in zip(names, net.shapes):
        rows.append([f"`{nm}`", " x ".join(str(v) for v in sh), _int(math.prod(sh))])
    x = list(start_noise()[0])
    out = net.forward(W, x, 0.5, 1)[0]
    gc = gradient_check(net, perturbed=True)
    ga = ae_gradient_check()
    return ("The pixel denoiser's tensors in their flat order (D = 36, H = 48, cond width 4, 4 timestep features).\n\n"
            + _table(["tensor", "shape", "values"], rows) + f"\n\nTotal: {_int(net.n_params)} values. Drawn from {_seed(RUN)} in the order E, W1, W2, W3 (E by normal(), W1 by "
              f"normal() / sqrt(40), W2 and W3 by normal() / sqrt(48), each row-major); the biases and the modulation weights are 0. A forward pass of the "
              f"initial network at x = the first starting vector of {_seed(SAMPLE)}, sigma = 0.5, class 1 gives the outputs "
              f"{', '.join(f'{v:.6f}' for v in out[:4])} for the first four pixels and a root-mean-square of {math.sqrt(sum(v * v for v in out) / D):.6f} "
              f"over all 36. One forward pass costs {_int(net.pass_flops())} FLOPs.\n\n"
              f"Gradient check ({_seed(GRAD)}: the parameters of the initialization plus 0.1 normal() each in flat order, then 3 items by "
              f"`draw_training_item`): every one of the {_int(net.n_params)} partial derivatives of the batch loss against the central difference of step 1e-5, "
              f"the largest difference divided by the largest partial derivative is {_below(gc, 1e-6)}. "
              f"The autoencoder's {_int(AE_PARAMS)} partial derivatives, on one picture and one draw of xi ({_seed(AE_GRAD)}), "
              f"the same: {_below(ga, 1e-6)}.")


def gradient_check(net, perturbed):
    gen = bookrng.generator(GRAD)
    P = net.init(gen)
    if perturbed:
        P = [v + 0.1 * gen.normal() for v in P]
    items = [draw_training_item(gen) for _ in range(3)]

    def loss(Q):
        W = net.unpack(Q)
        return sum(sum((a - b) ** 2 for a, b in zip(net.forward(W, x, s, c)[0], y)) / (net.d * len(items)) for s, x, y, c in items)
    W = net.unpack(P)
    caches, cs, ds = [], [], []
    for s, x, y, c in items:
        o, cache = net.forward(W, x, s, c)
        caches.append(cache)
        cs.append(c)
        ds.append([2.0 * (a - b) / (net.d * len(items)) for a, b in zip(o, y)])
    g = net.flatten(net.batch_grad(W, caches, cs, ds))
    top = max(abs(v) for v in g)
    worst = 0.0
    for i in range(net.n_params):
        Pp, Pm = list(P), list(P)
        Pp[i] += 1e-5
        Pm[i] -= 1e-5
        worst = max(worst, abs((loss(Pp) - loss(Pm)) / 2e-5 - g[i]))
    return worst / top


def ae_gradient_check():
    gen = bookrng.generator(AE_GRAD)
    P = [v + 0.1 * gen.normal() for v in ae_init(gen)]
    x, xi = ae_draw_item(gen)
    G = ae_zeros()
    ae_loss_grad(ae_unpack(P), x, xi, G, 1.0)
    g = [q for part in G for q in (sum(part, []) if isinstance(part[0], list) else part)]
    top = max(abs(v) for v in g)
    worst = 0.0
    for i in range(AE_PARAMS):
        Pp, Pm = list(P), list(P)
        Pp[i] += 1e-5
        Pm[i] -= 1e-5
        d = (ae_loss_grad(ae_unpack(Pp), x, xi, ae_zeros(), 1.0)[0] - ae_loss_grad(ae_unpack(Pm), x, xi, ae_zeros(), 1.0)[0]) / 2e-5
        worst = max(worst, abs(d - g[i]))
    return worst / top


NOTE = ("Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 "
        "(the stable-digit rule of Chapter 2); here all five agree on every digit shown.")


def training_table():
    px, _ = all_runs()
    rows = []
    for s in LOG_STEPS:
        rows.append([s] + [_shared([r["log"][s][i] for r in px], _fixed(4)) for i in range(3)])
    b = []
    for k, (lo, hi) in enumerate(BANDS):
        b.append([f"{lo:g} .. {hi:g}", px[0]["bands"][k][3]] + [_shared([r["bands"][k][i] for r in px], _fixed(4)) for i in (1, 0, 2)])
    return (f"The pixel denoiser trained {STEPS} steps of {BATCH} items from {_seed(RUN)} after its initialization, Adam (beta1 0.9, beta2 0.999, "
            f"epsilon 1e-8, no weight decay) at rate {LR:g} (1 - (i - 1)/{STEPS}) at step i. After each listed step the network and the exact field "
            f"are scored on {EVAL_N} fresh items ({_seed(EVAL)}, drawn as in training, so about 15% have the null label): the mean over items of the "
            f"mean over pixels of (output - target)^2 for the network and for the exact field, and of (network - exact)^2. {NOTE}\n\n"
            + _table(["step", "loss of the network", "loss of the exact field", "mean (network - exact)^2"], rows)
            + "\n\nThe same items after the last step, split by the noise fraction sigma of the item (lower end included, upper end excluded).\n\n"
            + _table(["sigma", "items", "loss of the exact field", "loss of the network", "mean (network - exact)^2"], b))


def solver_table():
    px, _ = all_runs()
    ex = exact_solver_distances()
    rows = []
    for n in NET_STEPS:
        rows.append([n, _fx3(ex[("euler", n)]), _shared([r["solver"][("euler", n)] for r in px], _fixed(4)),
                     _fx3(ex[("heun", n)]), _shared([r["solver"][("heun", n)] for r in px], _fixed(4))])
    return (f"Class-conditional sampling (g = 1) without the null pass. For each sampler and each count of evaluations of the field, the mean over "
            f"{SOLVER_N * N_CLASS} samples (the first {SOLVER_N} of the {SAMPLE_N} starting vectors of {_seed(SAMPLE)}, used for each of the three classes) of the root-mean-square distance per pixel between "
            f"the sampler's end point and the 100-step Runge-Kutta end point of the exact field from the same start. Euler takes n steps, Heun n/2 "
            f"(two evaluations each). The exact-field columns are plain arithmetic; the network columns share digits across the five runs.\n\n"
            + _table(["evaluations", "Euler, exact field", "Euler, network", "Heun, exact field", "Heun, network"], rows))


def _fx3(x):
    return f"{x:.4f}"


def guidance_table():
    px, _ = all_runs()
    ex = exact_guidance_rows()
    rows = []
    for g in G_VALUES:
        rows.append([f"network, g = {g:g}"] + _stable_measures([r["guidance"][g] for r in px]))
        rows.append([f"exact field, g = {g:g}"] + _mrow(ex[g]))
    return (f"Guided Euler sampling in pixel space, {G_STEPS} steps, the {SAMPLE_N} starting vectors of {_seed(SAMPLE)} used for each of the three classes, the same for the network "
            f"and the exact field. {NOTE}\n\n"
            + _table(["sampler"] + MEASURE_HEAD, rows))


def saturation_stats(pics):
    """(mean value of the pixels that lie on the nearest template's shape, the share of those above 1 + SIGMA_X, the share of the
    background pixels above 0)."""
    on, off = [], []
    for x in pics:
        t = TEMPLATES[nearest_template(x)[0]]
        for a, b in zip(x, t):
            (on if b > 0 else off).append(a)
    return sum(on) / len(on), sum(1 for v in on if v > 1.0 + SIGMA_X) / len(on), sum(1 for v in off if v > 0.0) / len(off)


def saturation_table():
    px, lt = all_runs()
    rows = [["the data, 300 pictures", "-"] + [f"{v:.3f}" for v in saturation_stats(data_balanced())]]
    for name, runs in (("pixel space", px), ("latent space", lt)):
        for g in (1.0, 4.0):
            st = [saturation_stats(r["pictures"][g]) for r in runs]
            rows.append([name, f"{g:g}"] + [_shared([q[i] for q in st], _fixed(3)) for i in range(3)])
    return (f"Saturation of the pictures of the guidance tables at g = 1 and g = 4 (the same 300 samples as there). For the pixels on the shape of "
            f"the nearest template: their mean and the share above 1 + sigma_x = {1.0 + SIGMA_X}; for the background pixels (the template's -1): the "
            f"share above 0. The data's own shares are those of a Gaussian: 0.159 above one standard deviation of the shape's value, and about 3e-5 "
            f"of the background above 0. {NOTE}\n\n"
            + _table(["system", "g", "mean on shape", "share on shape above 1.25", "share of background above 0"], rows))


def amplification_table():
    px, _ = all_runs()
    vals = {g: [r["gap"][g] for r in px] for g in (1.0, 2.0, 4.0)}
    base = vals[1.0][0]
    rows = [[f"{g:g}", _shared(vals[g], _fixed(4)), f"{vals[g][0] / base:.1f}"] for g in (1.0, 2.0, 4.0)]
    return (f"The error of the guided field of the trained pixel network against the guided exact field, on the {EVAL_N} evaluation items of "
            f"{_seed(EVAL)} that kept their class (the items with the null label are left out), each at its own class: the mean over items of "
            f"the mean over pixels of (guided network - guided exact)^2. The last column divides by the row of g = 1 (digits as in the other trained "
            f"tables, the ratio from the first run).\n\n" + _table(["g", "mean squared difference", "relative to g = 1"], rows))


def autoencoder_table():
    _, lt = all_runs()
    shapes = [["encoder hidden", f"{D} -> {AE_H}, tanh", _int(D * AE_H + AE_H)],
              ["encoder head (mean, log-variance)", f"{AE_H} -> {2 * Z_DIM}", _int(2 * Z_DIM * AE_H + 2 * Z_DIM)],
              ["decoder hidden", f"{Z_DIM} -> {AE_H}, tanh", _int(AE_H * Z_DIM + AE_H)],
              ["decoder output", f"{AE_H} -> {D}", _int(D * AE_H + D)]]
    mse = _shared([r["ae"][0] for r in lt], _fixed(4))
    kl = _shared([r["ae"][1] for r in lt], _fixed(2))
    rt = _shared([r["ae"][2] for r in lt], _fixed(3))
    mean = [_shared([r["stats"][0][k] for r in lt], _fixed(3)) for k in range(Z_DIM)]
    var = [_shared([r["stats"][1][k] for r in lt], _fixed(3)) for k in range(Z_DIM)]
    return ("The autoencoder's layers (flat order We1, be1, We2, be2, Wd1, bd1, Wd2, bd2; "
            f"{_int(AE_PARAMS)} values).\n\n" + _table(["layer", "map", "values"], shapes)
            + f"\n\nTrained {AE_STEPS} steps of {AE_BATCH} pictures from {_seed(AE_RUN)} (after the initialization; per picture c, p and 36 pixel noises as for "
              f"`draw_picture`, then {Z_DIM} values xi by normal()), Adam at rate {AE_LR:g} (1 - (i - 1)/{AE_STEPS}). On {EVAL_N} fresh pictures "
              f"({_seed(AE_EVAL)}) decoded from the mean code: mean squared error per pixel {mse} (the pixel noise alone has {SIGMA_X ** 2:.4f}), "
              f"mean KL {kl} nats per picture, and the six noise-free templates come back, each with its own root-mean-square error per pixel over its 36 pixels, "
              f"the mean of the six being {rt}. The mean KL of a picture is 0.5 the sum over the four channels of m^2 + exp(v) - 1 - v "
              f"for the mean m and log-variance v of its code. The {EVAL_N} pictures and the {STAT_N} pictures of the statistics are "
              f"drawn as `draw_picture` and nothing else (c, p and the 36 pixel noises per picture, no xi). "
              f"The mean and the variance (divisor {STAT_N}) of the four channels of the mean code over the {STAT_N} pictures of {_seed(AE_STAT)}:\n\n"
              + _table(["channel", "mean", "variance"], [[k, mean[k], var[k]] for k in range(Z_DIM)]) + f"\n\n{NOTE}")


def systems_table():
    px, lt = all_runs()
    rows = [["the data, 300 pictures", "-"] + _mrow(data_row())]
    for g in G_VALUES:
        rows.append(["pixel space", f"{g:g}"] + _stable_measures([r["guidance"][g] for r in px]))
    for g in G_VALUES:
        rows.append(["latent space", f"{g:g}"] + _stable_measures([r["guidance"][g] for r in lt]))
    return (f"The two systems on the same measures. Each row: {SAMPLE_N} starting vectors used for each class (so {SAMPLE_N * N_CLASS} samples), {G_STEPS} guided Euler steps. Pixel space starts from "
            f"{_seed(SAMPLE)}. Latent space starts from {_seed(LAT_START)} ({SAMPLE_N} vectors of 4 values by normal() in order, used for each class), "
            f"with a denoiser initialized and trained from {_seed(LAT_RUN)} like the pixel one from {_seed(RUN)} (the same items, with the "
            f"picture of each item replaced by its standardized code, and the same 2,000 steps); decodes the standardized code with the "
            f"autoencoder, and adds {SIGMA_X} normal() to every pixel from {_seed(DEC_NOISE)} (a fresh generator for every row, 36 draws per sample "
            f"in order). {NOTE}\n\n" + _table(["system", "g"] + MEASURE_HEAD, rows))


def mistakes_table():
    px, lt = all_runs()
    labels = ["pixel, one Euler step", "pixel, guidance g = -1", "latent, decoded without undoing the standardization", "latent, decoded without the pixel noise"]
    keys = [("p", "one step"), ("p", "wrong sign"), ("l", "no inverse standardization"), ("l", "no decoder noise")]
    rows = []
    for lab, (k, key) in zip(labels, keys):
        src = px if k == "p" else lt
        rows.append([lab] + _stable_measures([r["mistakes"][key] for r in src]))
    return ("Four mistakes planted in the systems of the last tables, each measured as a generator (g = 1 and 16 steps unless the row says "
            f"otherwise, the same starting values). {NOTE}\n\n" + _table(["mistake"] + MEASURE_HEAD, rows))


# ------------------------------------------------------------------ the ledger

def ae_flops():
    enc = 2 * D * AE_H + AE_H + 3 * AE_H + 2 * Z_DIM * AE_H + Z_DIM                   # hidden layer and the mean rows of the head
    dec = 2 * Z_DIM * AE_H + AE_H + 3 * AE_H + 2 * D * AE_H + D
    return enc, dec


def cost_ledger():
    pn, ln = PIXEL_NET, LATENT_NET
    enc, dec = ae_flops()
    fp, fl = pn.pass_flops(), ln.pass_flops()
    sample_p = G_STEPS * 2 * fp
    sample_l = G_STEPS * 2 * fl + dec
    train_p = STEPS * BATCH * 3 * fp
    ae_full = ae_flops()[0] + 2 * Z_DIM * AE_H + Z_DIM + dec                          # encoder with both heads, decoder
    train_ae_f = AE_STEPS * AE_BATCH * 3 * ae_full
    train_l = STEPS * BATCH * (3 * fl + enc)
    rows = [["pixel denoiser", _int(pn.n_params), _int(4 * pn.n_params), _int(fp), f"{fp / (4 * pn.n_params):.3f}"],
            ["latent denoiser", _int(ln.n_params), _int(4 * ln.n_params), _int(fl), f"{fl / (4 * ln.n_params):.3f}"],
            ["encoder (mean only)", _int(D * AE_H + AE_H + Z_DIM * AE_H + Z_DIM), _int(4 * (D * AE_H + AE_H + Z_DIM * AE_H + Z_DIM)), _int(enc),
             f"{enc / (4 * (D * AE_H + AE_H + Z_DIM * AE_H + Z_DIM)):.3f}"],
            ["decoder", _int(AE_H * Z_DIM + AE_H + D * AE_H + D), _int(4 * (AE_H * Z_DIM + AE_H + D * AE_H + D)), _int(dec),
             f"{dec / (4 * (AE_H * Z_DIM + AE_H + D * AE_H + D)):.3f}"]]
    rows2 = [["16-step guided sample, pixel space", f"{G_STEPS} x 2 x {_int(fp)}", _int(sample_p)],
             ["16-step guided sample, latent space", f"{G_STEPS} x 2 x {_int(fl)} + {_int(dec)}", _int(sample_l)],
             ["training the pixel denoiser", f"{_int(STEPS * BATCH)} items x 3 x {_int(fp)}", _int(train_p)],
             ["training the autoencoder", f"{_int(AE_STEPS * AE_BATCH)} pictures x 3 x {_int(ae_full)}", _int(train_ae_f)],
             ["training the latent denoiser (encoder in the loop)", f"{_int(STEPS * BATCH)} items x (3 x {_int(fl)} + {_int(enc)})", _int(train_l)],
             ["training the latent system in all", "autoencoder + latent denoiser", _int(train_ae_f + train_l)]]
    return ("FLOPs and bytes, counted from the shapes (a multiply and an add are 2 FLOPs, a tanh is 3 FLOPs, a pass reads every weight once as a 4-byte "
            "float, so the bytes of a pass are four times its values). The last column is FLOPs per byte read.\n\n"
            + _table(["stage", "values", "bytes of weights", "FLOPs per pass", "FLOPs per byte"], rows)
            + "\n\nA backward pass is counted as two forward passes, so a training item costs three.\n\n"
            + _table(["work", "count", "FLOPs"], rows2, "lrr")
            + f"\n\nA guided pixel sample costs {sample_p / sample_l:.2f} times the FLOPs of a guided latent sample; training the whole latent system "
              f"(autoencoder and denoiser) costs {(train_ae_f + train_l) / train_p:.2f} times the FLOPs of training the pixel denoiser.")


# ------------------------------------------------------------------ acceptance tests and figure data

def _cells(vals):
    return " / ".join(vals)


def acceptance_tests():
    px, lt = all_runs()
    net = PIXEL_NET
    W = net.unpack(net.init(bookrng.generator(RUN)))
    out = net.forward(W, list(start_noise()[0]), 0.5, 1)[0]
    ex = exact_guidance_rows()
    ref = measures(reference_endpoints(), class_labels(SAMPLE_N))
    d = data_row()
    first = draw_picture(bookrng.generator(DATA))
    enc, dec = ae_flops()
    fp, fl = net.pass_flops(), LATENT_NET.pass_flops()
    exs = exact_solver_distances()
    f4 = lambda vals: _shared(vals, _fixed(4))
    rows = [
        (1, "pixels on in the column, row and block templates", "6 / 6 / 4", "exact"),
        (1, "first picture of the stream `data`: class, position; sum of its pixels", f"{CLASSES[first[0]]}, {first[1]}; {sum(first[2]):.6f}", "exact; 1e-6"),
        (1, "the data, 300 pictures: accuracy / on the data / distance / ln p / TV", _cells(_mrow(d)), "exact; exact; 0.001; 0.01; 0.001"),
        (2, "100-step Runge-Kutta samples of the exact field: the same five measures", _cells(_mrow(ref)), "as above"),
        (2, "exact field, Euler 16 steps, g = 1 and g = 4: distance / ln p", f"{ex[1.0][2]:.3f} / {ex[1.0][3]:.2f}; {ex[4.0][2]:.3f} / {ex[4.0][3]:.2f}", "0.001; 0.01"),
        (2, "exact-field Euler against the reference at 4 and 32 steps; Heun at 4 and 32 evaluations", f"{exs[('euler', 4)]:.4f} / {exs[('euler', 32)]:.4f}; {exs[('heun', 4)]:.4f} / {exs[('heun', 32)]:.4f}", "1e-4"),
        (3, "values of the pixel denoiser, of the autoencoder; FLOPs of one pass", f"{_int(net.n_params)}; {_int(AE_PARAMS)}; {_int(fp)}", "exact"),
        (3, "initial network at the first starting vector, sigma 0.5, class 1: first four outputs", ", ".join(f"{v:.6f}" for v in out[:4]), "1e-6"),
        (4, "after 2,000 steps: loss of the network / of the exact field / (network - exact)^2",
         " / ".join(_shared([r["log"][STEPS][i] for r in px], _fixed(4)) for i in range(3)), "the digits shown"),
        (4, "distance to the reference, 16 evaluations: Euler / Heun, network",
         f"{f4([r['solver'][('euler', 16)] for r in px])} / {f4([r['solver'][('heun', 16)] for r in px])}", "0.001"),
        (4, "pixel network, g = 1, 16 steps: the five measures", _cells(_stable_measures([r["guidance"][1.0] for r in px])), "exact; exact; 0.001; 0.01; 0.001"),
        (4, "pixel network, g = 4, 16 steps: the five measures", _cells(_stable_measures([r["guidance"][4.0] for r in px])), "as above"),
        (5, "autoencoder on 600 pictures: squared error per pixel; KL; template round trip",
         " / ".join([_shared([r["ae"][0] for r in lt], _fixed(4)), _shared([r["ae"][1] for r in lt], _fixed(2)), _shared([r["ae"][2] for r in lt], _fixed(3))]),
         "1e-4; 0.01; 0.001"),
        (5, "latent system, g = 1, 16 steps: the five measures", _cells(_stable_measures([r["guidance"][1.0] for r in lt])), "exact; exact; 0.001; 0.01; 0.001"),
        (5, "latent system, g = 4, 16 steps: the five measures", _cells(_stable_measures([r["guidance"][4.0] for r in lt])), "as above"),
        (5, "latent system decoded without undoing the standardization: accuracy / on the data",
         " / ".join(_stable_measures([r["mistakes"]["no inverse standardization"] for r in lt])[:2]), "exact"),
        (6, "FLOPs of a 16-step guided sample: pixel space / latent space", f"{_int(G_STEPS * 2 * fp)} / {_int(G_STEPS * 2 * fl + dec)}", "exact"),
    ]
    return ("Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the "
            "last column, absolute. A cell of five numbers is the five measures of the project in the order accuracy, share on data, "
            "distance to the template, mean ln p, TV.\n\n"
            + _table(["milestone", "quantity", "value", "tolerance"], [list(r) for r in rows], "rlll"))


def pictures_data():
    """Rows of six 6 x 6 pictures (two per class) for the figure: the data, the exact field, the pixel network and the latent system."""
    pxr, ltr = nominal()
    cols = [c * SAMPLE_N + i for c in range(N_CLASS) for i in (0, 1)]
    ref = reference_endpoints()
    return {"data": [data_balanced()[k] for k in cols], "exact": [ref[k] for k in cols],
            "pixel1": [pxr["pictures"][1.0][k] for k in cols], "pixel4": [pxr["pictures"][4.0][k] for k in cols],
            "latent1": [ltr["pictures"][1.0][k] for k in cols], "latent4": [ltr["pictures"][4.0][k] for k in cols]}


def pipeline_data():
    """The numbers the pipeline diagram prints."""
    enc, dec = ae_flops()
    return {"fp": PIXEL_NET.pass_flops(), "fl": LATENT_NET.pass_flops(), "enc": enc, "dec": dec,
            "pn": PIXEL_NET.n_params, "ln": LATENT_NET.n_params, "ae": AE_PARAMS}
