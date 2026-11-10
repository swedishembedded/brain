# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "What It Means to Generate".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: generative_modeling.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

The world is a set of 2 by 2 black-and-white images, so that a distribution over all of them is a table of 16 numbers:

* the world itself (four bar patterns, a prior over them, a flip probability), its 16 probabilities and entropy;
* a dataset of 1000 images drawn by ancestral sampling from the world, counted;
* the table fitted by counting, an ablation showing that the counts maximize the training likelihood, the held-out
  log-loss of four models (counting table, smoothed table, independent pixels, latent-variable model) as the training set grows;
* the latent-variable model: the posterior of a hidden pattern, the pass-by-pass fit by expectation-maximization;
* ancestral sampling traced draw by draw, and checked by counting 10,000 draws; the chain rule as a pixel-by-pixel sampler;
* the density of a continuous pixel against the probability of its 8-bit bin, and bits per dimension.

Every printed number is a function of the stated inputs. The only randomness is the book's SplitMix64: the dataset, with seed
`seed_of("generative_modeling.dataset")`, and the check of the sampler, with `seed_of("generative_modeling.ancestral")`. The
fitted numbers are exact sums or the limit of a convergent iteration run for a stated number of passes, so none depends on
rounding noise at the printed digits.

Swedish Embedded AB implements solutions for generative models of images that can be trained, sampled and checked against data.
If your team needs expertise in density estimation and image generation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import information

# ----------------------------------------------------------------------------------------------------- the world

# Pixels are listed row by row: (top left, top right, bottom left, bottom right); 1 is a bright pixel.
PATTERNS = ((1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1))
PATTERN_NAMES = ("left bar", "right bar", "top bar", "bottom bar")
PRIOR = (0.4, 0.3, 0.2, 0.1)
FLIP = 0.1
NPIX = 4
N_IMAGES = 16
NPIX_28 = 28 * 28
N_DATA = 1000
N_SAMPLER_CHECK = 10_000
EM_START_PRIOR = (0.25, 0.25, 0.25, 0.25)
EM_START_FLIP = 0.2
EM_PASSES = 300
TRAIN_SIZES = (10, 20, 50, 200, 1000)
CURVE_SIZES = (10, 20, 50, 100, 200, 500, 1000)
DATASET = "generative_modeling.dataset"
ANCESTRAL = "generative_modeling.ancestral"


# ------------------------------------------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed}), used as the state before the first draw."


def _f(x, d=4):
    return f"{x:.{d}f}"


def _bits(x):
    return "infinite" if math.isinf(x) else f"{x:.4f}"


def image_index(x):
    """The index 0..15 of an image: its four pixels read as a binary number, top left most significant."""
    return x[0] * 8 + x[1] * 4 + x[2] * 2 + x[3]


def image_pixels(i):
    return ((i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1)


def _rows(i):
    x = image_pixels(i)
    return f"{x[0]}{x[1]}/{x[2]}{x[3]}"


def pixel_probability(x, k, flip):
    """p(x | pattern k): every pixel agrees with the pattern with probability 1 - flip, independently."""
    p = 1.0
    for j in range(NPIX):
        p *= (1.0 - flip) if x[j] == PATTERNS[k][j] else flip
    return p


def marginal(prior, flip):
    """The 16 probabilities p(x) = sum over patterns of prior[k] * p(x | k), each sum taken in pattern order 0..3."""
    out = []
    for i in range(N_IMAGES):
        x = image_pixels(i)
        s = 0.0
        for k in range(4):
            s += prior[k] * pixel_probability(x, k, flip)
        out.append(s)
    return out


def _draw(p, u):
    """The cumulative-table rule of Volume I: the first k whose running sum of p exceeds u (the last if rounding leaves none)."""
    acc = 0.0
    for k, pk in enumerate(p):
        acc += pk
        if u < acc:
            return k
    return len(p) - 1


def ancestral_sample(rng, trace=None):
    """One image by ancestral sampling: a pattern from the prior (one uniform), then the four pixels in order (one uniform each,
    flipped when the uniform is below FLIP). Returns the image index; `trace` collects (u_pattern, k, [(u, flipped)...])."""
    u_k = rng.uniform()
    k = _draw(PRIOR, u_k)
    x, steps = [], []
    for j in range(NPIX):
        u = rng.uniform()
        flipped = u < FLIP
        x.append(1 - PATTERNS[k][j] if flipped else PATTERNS[k][j])
        steps.append((u, flipped))
    if trace is not None:
        trace.append((u_k, k, steps))
    return image_index(x)


TRUE_P = marginal(PRIOR, FLIP)


def dataset():
    """The 1000 training images in drawing order; a training set of size N is the first N."""
    rng = bookrng.generator(DATASET)
    return [ancestral_sample(rng) for _ in range(N_DATA)]


DATA = dataset()


# --------------------------------------------------------------------------------------------------- the models

def counts(n):
    c = [0] * N_IMAGES
    for i in DATA[:n]:
        c[i] += 1
    return c


def table_model(n, alpha=0):
    """Counting table; alpha imaginary observations per image (alpha = 0 is the maximum-likelihood table)."""
    c = counts(n)
    return [(ci + alpha) / (n + N_IMAGES * alpha) for ci in c]


def independent_model(n):
    """Four independent pixels, each bright with its observed frequency; returns (16 probabilities, the four frequencies)."""
    m = [sum(image_pixels(i)[j] for i in DATA[:n]) / n for j in range(NPIX)]
    q = []
    for i in range(N_IMAGES):
        x = image_pixels(i)
        p = 1.0
        for j in range(NPIX):
            p *= m[j] if x[j] else 1.0 - m[j]
        q.append(p)
    return q, m


def _responsibilities(xs, prior, flip):
    resp = []
    for x in xs:
        w = [prior[k] * pixel_probability(x, k, flip) for k in range(4)]
        s = w[0] + w[1] + w[2] + w[3]
        resp.append([wk / s for wk in w])
    return resp


def mean_bits(xs, q):
    """Average of -log2 q(x) over the images xs (given as indices), summed in order."""
    total = 0.0
    for i in xs:
        total += -math.log2(q[i]) if q[i] > 0 else math.inf
    return total / len(xs)


def em_fit(n, passes=EM_PASSES, record=()):
    """Maximum likelihood for the latent-variable model by expectation-maximization on the first n training images.
    Returns (prior, flip, trace) where trace maps a requested pass number to (mean training bits, prior, flip) before that pass."""
    xs = [image_pixels(i) for i in DATA[:n]]
    prior, flip = list(EM_START_PRIOR), EM_START_FLIP
    trace = {}
    for it in range(passes + 1):
        if it in record:
            trace[it] = (mean_bits(DATA[:n], marginal(prior, flip)), tuple(prior), flip)
        if it == passes:
            break
        resp = _responsibilities(xs, prior, flip)
        prior = [sum(r[k] for r in resp) / n for k in range(4)]
        wrong = 0.0
        for r, x in zip(resp, xs):
            for k in range(4):
                miss = sum(1 for j in range(NPIX) if x[j] != PATTERNS[k][j])
                wrong += r[k] * miss
        flip = wrong / (NPIX * n)
    return prior, flip, trace


def heldout_bits(q):
    """Expected -log2 q(x) for x drawn from the world: sum over the 16 images in index order of p(x) * -log2 q(x)."""
    return information.cross_entropy(TRUE_P, q, base=2.0)


def models_at(n):
    qi, _ = independent_model(n)
    prior, flip, _ = em_fit(n)
    return {"table": table_model(n), "smoothed": table_model(n, 1), "independent": qi, "latent": marginal(prior, flip)}


# ------------------------------------------------------------------------------------------------ worked blocks

def image_counts():
    rows = []
    for name, h, w, c, levels in (("2 by 2, black and white", 2, 2, 1, 2), ("28 by 28, grey, 8 bits", 28, 28, 1, 256),
                                  ("64 by 64, colour, 8 bits", 64, 64, 3, 256), ("512 by 512, colour, 8 bits", 512, 512, 3, 256)):
        n = h * w * c
        exp = n * math.log10(levels)
        rows.append([name, n, levels, "16" if n == 4 else f"about 10^{exp:.1f}", n * (levels - 1)])
    return ("Number of images = (levels per value)^(values per image), with log10 = values x log10(levels); the last column is the "
            "number of free parameters of a model that treats the values as independent, one probability less than the "
            "number of levels per value.\n\n"
            + _table(["image type", "values per image", "levels", "number of different images", "independent-value parameters"],
                     rows))


def the_world():
    rows = []
    for i in range(N_IMAGES):
        x = image_pixels(i)
        rows.append([i, _rows(i), "".join(str(v) for v in x), _f(TRUE_P[i], 6)])
    ent = information.entropy(TRUE_P, 2.0)
    return ("Patterns (top left, top right, bottom left, bottom right): "
            + ", ".join(f"{n} = {''.join(map(str, p))}" for n, p in zip(PATTERN_NAMES, PATTERNS))
            + f". Prior {PRIOR}, flip probability {FLIP}. Index = 8 t_l + 4 t_r + 2 b_l + b_r.\n\n"
            + _table(["index", "image (top row / bottom row)", "pixels", "p(x)"], rows)
            + f"\n\nSum of the 16 probabilities {sum(TRUE_P):.6f}; entropy {ent:.4f} bits ({ent / NPIX:.4f} bits per pixel); "
              f"the uniform distribution on 16 images has 4 bits.")


def the_dataset():
    c20, c1000 = counts(20), counts(N_DATA)
    rows = [[i, _rows(i), c20[i], c1000[i], _f(c1000[i] / N_DATA)] for i in range(N_IMAGES)]
    return (_generator_line(DATASET)
            + " Each image is one call of the ancestral sampler of the chapter: `rng.uniform()` for the pattern, then one "
              "`rng.uniform()` per pixel in the order top left, top right, bottom left, bottom right. The training set of size N "
              f"is the first N of {N_DATA} images. The first 20 images (indices): {', '.join(str(i) for i in DATA[:20])}.\n\n"
            + _table(["index", "image", "count in first 20", f"count in {N_DATA}", f"frequency in {N_DATA}"], rows)
            + f"\n\nImages not seen among the first 20: {sum(1 for v in c20 if v == 0)} of 16; among the first 200: "
              f"{sum(1 for v in counts(200) if v == 0)}; among all {N_DATA}: {sum(1 for v in c1000 if v == 0)}.")


def counting_is_best():
    n = 20
    cand = {"uniform: 1/16 each": [1 / 16] * N_IMAGES, "the world's p(x)": TRUE_P, "counts / 20": table_model(n)}
    rows = [[k, _bits(mean_bits(DATA[:n], q))] for k, q in cand.items()]
    return (f"The first {n} training images. Mean of -log2 q(x) over them, in bits per image (lower means the data are more "
            "probable under q); sum taken in drawing order.\n\n" + _table(["candidate table q", "mean -log2 q(x), training"], rows))


def heldout_table():
    rows = []
    for n in TRAIN_SIZES:
        ms = models_at(n)
        rows.append([n, _bits(mean_bits(DATA[:n], ms["table"])), _bits(heldout_bits(ms["table"])),
                     _bits(heldout_bits(ms["smoothed"])), _bits(heldout_bits(ms["independent"])),
                     _bits(heldout_bits(ms["latent"]))])
    ent = information.entropy(TRUE_P, 2.0)
    return ("Held-out bits = the expected -log2 q(x) over images drawn from the world, computed exactly as the sum over the 16 "
            "images in index order of p(x) times -log2 q(x) (that is what an unlimited held-out set would give). Training "
            "sets are the first N images of the dataset. Smoothed = counts plus 1 imaginary observation per image. The "
            f"latent model is fitted by {EM_PASSES} passes of the update of the section on the latent model. The entropy of "
            f"the world, the best any model can do, is {ent:.4f} bits; the uniform table gives 4.0000.\n\n"
            + _table(["N", "counting table, training", "counting table, held-out", "smoothed table, held-out",
                      "independent pixels, held-out", "latent model, held-out"], rows))


def model_probabilities():
    n = N_DATA
    ms = models_at(n)
    _, m = independent_model(n)
    prior, flip, _ = em_fit(n)
    rows = []
    for i in range(N_IMAGES):
        rows.append([i, _rows(i), _f(TRUE_P[i]), _f(ms["table"][i]), _f(ms["independent"][i]), _f(ms["latent"][i])])
    rare = [i for i in range(N_IMAGES) if TRUE_P[i] < 0.01]
    return (f"Fitted on all {n} training images. Independent-pixel frequencies (bright share of top left, top right, bottom "
            f"left, bottom right): {', '.join(_f(v) for v in m)}. Latent model after {EM_PASSES} passes: prior "
            f"({', '.join(_f(v) for v in prior)}), flip probability {_f(flip)}.\n\n"
            + _table(["index", "image", "world", "counting table", "independent pixels", "latent model"], rows)
            + f"\n\nImages whose probability in the world is below 0.01: {', '.join(str(i) for i in rare)}; the mass they "
              f"hold is {_f(sum(TRUE_P[i] for i in rare))} in the world, {_f(sum(ms['independent'][i] for i in rare))} under "
              f"the independent-pixel model and {_f(sum(ms['latent'][i] for i in rare))} under the latent model.")


def parameter_counts():
    exp = NPIX_28 * math.log10(2)
    rows = [["counting table", "15 (16 probabilities that sum to 1)", f"2^784 - 1, about 10^{exp:.1f}"],
            ["independent pixels", "4", str(NPIX_28)],
            ["latent model with K given patterns", "K - 1 + 1 = 4 (prior and flip probability)", "K - 1 + 1 (and the patterns)"]]
    return _table(["model", "2 by 2 images", "28 by 28 black and white"], rows, "lll")


def posterior_example():
    x = (1, 0, 0, 0)
    joint = [PRIOR[k] * pixel_probability(x, k, FLIP) for k in range(4)]
    total = joint[0] + joint[1] + joint[2] + joint[3]
    rows = [[PATTERN_NAMES[k], "".join(map(str, PATTERNS[k])), _f(PRIOR[k], 1),
             sum(1 for j in range(NPIX) if x[j] != PATTERNS[k][j]), _f(pixel_probability(x, k, FLIP), 6),
             _f(joint[k], 6), _f(joint[k] / total)] for k in range(4)]
    return (f"The image x = 1000 (index {image_index(x)}, top left bright only). Terms are summed in pattern order; "
            f"p(x) = {total:.6f}, equal to the entry of the table of the world.\n\n"
            + _table(["pattern k", "pixels", "prior", "pixels that differ", "p(x \\| k)", "prior x p(x \\| k)",
                      "posterior p(k \\| x)"], rows))


def em_trace():
    n = 50
    record = (0, 1, 2, 3, 5, 10, 20, 50, EM_PASSES)
    _, _, tr = em_fit(n, record=record)
    rows = [[it, _bits(tr[it][0]), ", ".join(_f(v) for v in tr[it][1]), _f(tr[it][2])] for it in record]
    return (f"The first {n} training images. Start: prior {EM_START_PRIOR}, flip probability {EM_START_FLIP}. Each pass computes "
            "the 50 x 4 posteriors, then replaces the prior by their means and the flip probability by the posterior-weighted "
            "share of pixels that differ from the pattern (divisor 4 x 50). The row for pass t shows the state before pass t "
            "(pass 0 is the start); mean bits are over the 50 training images.\n\n"
            + _table(["pass", "mean training bits", "prior", "flip probability"], rows))


def ancestral_trace():
    rng = bookrng.generator(ANCESTRAL)
    tr = []
    for _ in range(3):
        ancestral_sample(rng, tr)
    rows = []
    for n, (u_k, k, steps) in enumerate(tr):
        x = [1 - PATTERNS[k][j] if steps[j][1] else PATTERNS[k][j] for j in range(NPIX)]
        rows.append([n, _f(u_k, 6), PATTERN_NAMES[k]] + [f"{_f(u, 6)} ({'flip' if fl else 'keep'})" for u, fl in steps]
                    + ["".join(map(str, x))])
    return (_generator_line(ANCESTRAL) + " Draw order for each image: one uniform for the pattern, then one for each pixel in "
            "the order top left, top right, bottom left, bottom right.\n\n"
            + _table(["image", "u (pattern)", "pattern", "u (top left)", "u (top right)", "u (bottom left)", "u (bottom right)",
                      "pixels"], rows, "lrlrrrrl"))


def ancestral_check():
    rng = bookrng.generator(ANCESTRAL)
    c = [0] * N_IMAGES
    for _ in range(N_SAMPLER_CHECK):
        c[ancestral_sample(rng)] += 1
    rows = [[i, _rows(i), _f(TRUE_P[i]), c[i], _f(c[i] / N_SAMPLER_CHECK)] for i in range(N_IMAGES)]
    worst = max(abs(c[i] / N_SAMPLER_CHECK - TRUE_P[i]) for i in range(N_IMAGES))
    return (_generator_line(ANCESTRAL) + f" The {N_SAMPLER_CHECK:,} images are drawn by the same sampler, continuing the stream "
            "of the trace above (its first three images are the first three here).\n\n"
            + _table(["index", "image", "p(x)", f"draws of {N_SAMPLER_CHECK:,}", "frequency"], rows)
            + f"\n\nLargest difference between a frequency and its probability: {worst:.4f}.")


def chain_rule():
    target = (1, 0, 1, 0)
    rows, prod = [], 1.0
    prefix_sum = lambda pre: sum(TRUE_P[i] for i in range(N_IMAGES) if image_pixels(i)[:len(pre)] == tuple(pre))
    for j in range(NPIX):
        pre = target[:j]
        num = prefix_sum(target[:j + 1])
        den = prefix_sum(pre) if j else 1.0
        cond = num / den
        prod *= cond
        cond_text = "p(top left = 1)" if j == 0 else "p(" + ["", "top right", "bottom left", "bottom right"][j] + " = " + str(target[j]) \
            + " given " + ", ".join(f"{['top left', 'top right', 'bottom left'][t]} = {pre[t]}" for t in range(j)) + ")"
        rows.append([j, cond_text, _f(num, 6), _f(den, 6), _f(cond)])
    return ("Probabilities are sums of the table of the world over the images that start with the given pixels. The factor for "
            "pixel j is the probability of the prefix through pixel j divided by the probability of the prefix before it.\n\n"
            + _table(["pixel j", "factor", "prefix probability", "previous prefix", "factor value"], rows, "lrrrr")
            + f"\n\nProduct of the four factors {prod:.6f}; p(x) of the table for image 1010 (index {image_index(target)}): "
              f"{TRUE_P[image_index(target)]:.6f}.")


def pixel_density():
    w = 1.0 / 255.0
    rows = []
    for sigma in (0.25, 0.05, 0.01, 0.002, 0.0005):
        dens = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        prob = math.erf(w / 2.0 / (sigma * math.sqrt(2.0)))
        rows.append([sigma, _f(dens, 3), _f(dens * w, 4), f"{prob:.4f}", f"{-math.log2(prob):.3f}"])
    return (f"A Gaussian pixel model centred on the code 128, so on the value 128/255. Bin = the interval of width 1/255 "
            "centred on that value. density = 1/(sigma sqrt(2 pi)) at the centre; bin probability = erf(w / (2 sigma sqrt 2)) "
            "with w = 1/255, using the library erf; bits = -log2 of the bin probability. The uniform model on 256 codes gives "
            "8 bits.\n\n"
            + _table(["sigma", "density at the centre", "density x 1/255", "probability of the bin", "bits for the code"], rows))


# ------------------------------------------------------------------------------------------------ figure data

def probability_rows():
    """For the figure: (index, pixels, world, independent pixels, latent model) at N = 1000, sorted by the world's probability."""
    ms = models_at(N_DATA)
    rows = [(i, image_pixels(i), TRUE_P[i], ms["independent"][i], ms["latent"][i]) for i in range(N_IMAGES)]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


def heldout_curve():
    """For the figure: sizes and, per model, the held-out bits (None where infinite)."""
    series = {"smoothed table": [], "independent pixels": [], "latent model": []}
    for n in CURVE_SIZES:
        ms = models_at(n)
        series["smoothed table"].append(heldout_bits(ms["smoothed"]))
        series["independent pixels"].append(heldout_bits(ms["independent"]))
        series["latent model"].append(heldout_bits(ms["latent"]))
    return list(CURVE_SIZES), series, information.entropy(TRUE_P, 2.0)
