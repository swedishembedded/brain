# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Denoiser Architectures: U-Nets and Diffusion Transformers".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: denoiser_architectures.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. The three clusters and their exact denoiser are those of
`flow_matching` (imported, not copied).

* a denoiser that is not told the noise level, against one that is, with exact posterior means (no training);
* the sinusoidal timestep embedding, the order of its two halves, and the scale that turns a noise fraction into a timestep;
* the SDXL U-Net derived from its configuration: the walk down and up with its stack of skips, every parameter and tensor counted,
  the multiply-adds of one pass split by kind and by resolution, and the parameters of the smoke-test variant of the same graph;
* the Z-Image diffusion transformer and the FLUX.2 double- and single-stream transformers derived from their configurations: the
  parameters of each block, the position ids of a joint sequence, the cost of a pass against the image side, and the places
  where conditioning enters;
* one layer of joint attention against one layer of cross-attention on five tokens;
* a small time-conditioned denoiser (a timestep MLP, Z-Image style adaptive modulation, joint attention over image and caption
  tokens, multi-axis rotary position, a final modulated layer) with every tensor's recipe, and the checks it must pass.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. The
small denoiser is evaluated, not trained, so its numbers are stable to the digits printed.

Swedish Embedded AB implements solutions for denoising networks of image generators that are specified, counted and checked against
exact references. If your team needs expertise in U-Nets and diffusion transformers, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import math

import bookrng
import diffusion
import flow_matching as fl

_table, _seed, _f, _g = diffusion._table, diffusion._seed, diffusion._f, diffusion._g

MODES, SD = fl.MODES, fl.SD


def _int(n):
    return f"{n:,}"


def _pct(x, d=1):
    return f"{100.0 * x:.{d}f}%"


def _gflop(x):
    return f"{x / 1e9:,.1f}"


# ------------------------------------------------------------------ a denoiser that is not told the noise level

BLIND_SIGMAS = (0.02, 0.1, 0.3, 0.5, 0.7, 0.9)  # the six noise fractions the blind denoiser is trained across, equally often
SIMPSON_N = 16000                              # panels of the Simpson rule over [-8, 8]


def _components(x, s):
    """For noise fraction s: the three (weight, posterior mean of x_0, posterior variance) triples at x, weights unnormalised:
    the clean value is N(m_k, SD^2), x = a x_0 + b noise with a = 1 - s, b = s, so x | k is N(a m_k, v) with v = a^2 SD^2 + b^2."""
    a, b = 1.0 - s, s
    v = a * a * SD * SD + b * b
    out = []
    for m in MODES:
        w = math.exp(-0.5 * (x - a * m) ** 2 / v) / math.sqrt(2.0 * math.pi * v)
        mu = m + a * SD * SD / v * (x - a * m)
        out.append((w, mu, SD * SD - a * a * SD ** 4 / v))
    return out


def aware(x, s):
    """The best guess of the clean value when the noise fraction s is known: the posterior mean."""
    c = _components(x, s)
    return sum(w * mu for w, mu, _ in c) / sum(w for w, _, _ in c)


def blind(x):
    """The best guess when the noise fraction is one of BLIND_SIGMAS with equal probability and is not told: the posterior mean over
    the 15 (noise fraction, cluster) pairs."""
    num = den = 0.0
    for s in BLIND_SIGMAS:
        for w, mu, _ in _components(x, s):
            num += w * mu
            den += w
    return num / den


def _mse(f, s):
    """E (f(x) - x_0)^2 at noise fraction s, by Simpson's rule over x in [-8, 8]: given x and the cluster k the clean value is
    N(mu_k(x), var_k), so the inner expectation is (f(x) - mu_k)^2 + var_k, weighted by the density of x under cluster k (weight 1/3)."""
    h = 16.0 / SIMPSON_N
    total = 0.0
    for i in range(SIMPSON_N + 1):
        x = -8.0 + i * h
        fx = f(x)
        g = sum(w / 3.0 * ((fx - mu) ** 2 + var) for w, mu, var in _components(x, s))
        total += g * (1 if i in (0, SIMPSON_N) else 4 if i % 2 else 2)
    return total * h / 3.0


def blind_denoiser():
    rows, raw_aware, raw_blind = [], [], []
    for s in BLIND_SIGMAS:
        e_aware = _mse(lambda x, s=s: aware(x, s), s)
        e_blind = _mse(blind, s)
        raw_aware.append(e_aware)
        raw_blind.append(e_blind)
        rows.append([f"{s:g}", _f(e_aware, 4), _f(e_blind, 4), f"{e_blind / e_aware:.1f}"])
    top = [[f"{x:.1f}"] + [_f(aware(x, s), 4) for s in BLIND_SIGMAS] + [_f(blind(x), 4)] for x in (0.5, 1.5, 2.0)]
    mean_aware = sum(raw_aware) / len(raw_aware)      # the mean of the unrounded errors, not of the printed cells
    mean_blind = sum(raw_blind) / len(raw_blind)
    return (
        "The data are the three clusters (centres -3, 0, 3, standard deviation 0.25, equal weights) and x = (1 - sigma) x_0 + sigma "
        "noise. Two exact denoisers, no training: the posterior mean E[x_0 | x, sigma] that is told sigma, and the posterior mean "
        "E[x_0 | x] when sigma is one of 0.02, 0.1, 0.3, 0.5, 0.7, 0.9 with equal probability and is not told. Squared error "
        f"E (guess - x_0)^2 at each sigma, by Simpson's rule with {SIMPSON_N} panels over x in [-8, 8] (the integrand is a sum of three "
        "Gaussians of width at least 0.24, so the rule is converged far below the digits printed).\n\n"
        + _table(["sigma", "told sigma", "not told", "ratio"], rows)
        + f"\n\nAveraged over the six levels the error is {mean_aware:.4f} when told and {mean_blind:.4f} when not told. The guesses at "
        "three values of x:\n\n"
        + _table(["x"] + [f"told, sigma = {s:g}" for s in BLIND_SIGMAS] + ["not told"], top)
    )


GAIN_MEAN = 3.0         # the cluster of the gain table: N(3, 0.25^2)


def gain_table():
    rows = []
    stats = []
    for s in BLIND_SIGMAS:
        a, b = 1.0 - s, s
        v = a * a * SD * SD + b * b
        k = a * SD * SD / v
        stats.append((s, k, v, SD * SD * b * b / v))
    w = sum(k * v for _, k, v, _ in stats) / sum(v for _, _, v, _ in stats)
    for s, k, v, best in stats:
        rows.append([f"{s:g}", _f(k, 4), _f(v, 4), _f(best, 4), _f((k - w) ** 2 * v, 4), "0.0000"])
    return (
        f"One cluster, N({GAIN_MEAN:g}, 0.25^2), and x = (1 - sigma) x_0 + sigma noise. The best guess of x_0 is "
        "m + k(sigma) (x - (1 - sigma) m) with the gain k = (1 - sigma) SD^2 / r and r = (1 - sigma)^2 SD^2 + sigma^2, the variance of x; "
        "its squared error is the posterior variance SD^2 sigma^2 / r. A layer of the form w x + delta(sigma) (the noise level enters "
        "as an added term) keeps one gain w for every sigma; the best w for the six levels together, weighting each by the "
        f"variance of x, is w = {w:.4f}. A layer of the form (1 + s(sigma)) x + delta(sigma) can set the gain at every sigma. "
        "The last two columns are the squared error above the best possible one, E (guess - best guess)^2 = (k - w)^2 r for the "
        "fixed gain, and 0 when the gain may depend on sigma.\n\n"
        + _table(["sigma", "gain k", "variance of x", "best possible error", "excess, added term", "excess, scale and shift"], rows)
    )


# ------------------------------------------------------------------ the timestep embedding

def embed(t, dim, base=10000.0, cos_first=True):
    """The sinusoidal embedding of the scalar t: half = dim // 2 rates w_k = base^(-k / half), k = 0 .. half - 1, angles t w_k, and the
    vector [cos(t w_0) .. cos(t w_(half-1)), sin(t w_0) .. sin(t w_(half-1))], or with the two halves exchanged. The rate is
    computed as exp(-ln(base) k / half) with the library log and exp, the angle as t * rate and cos, sin are the library functions."""
    half = dim // 2
    ang = [t * math.exp(-math.log(base) * k / half) for k in range(half)]
    c = [math.cos(a) for a in ang]
    s = [math.sin(a) for a in ang]
    return c + s if cos_first else s + c


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def embedding_table():
    rows = []
    for t in (0, 1, 2, 10, 100):
        e = embed(float(t), 8)
        rows.append([str(t)] + [_f(v, 4) for v in e])
    swapped = embed(1.0, 8, cos_first=False)
    rows.append(["1, halves exchanged"] + [_f(v, 4) for v in swapped])
    rates = [math.exp(-math.log(10000.0) * k / 4) for k in range(4)]
    head = ["t"] + [f"cos(t {r:g})" for r in rates] + [f"sin(t {r:g})" for r in rates]
    return (
        "The embedding of dimension 8 and base 10000: half = 4 rates w_k = 10000^(-k/4) = 1, 0.1, 0.01, 0.001, and the vector "
        "[cos(t w_0), .., cos(t w_3), sin(t w_0), .., sin(t w_3)] (library cos and sin, radians).\n\n"
        + _table(head, rows)
    )


def _scale_pairs():
    return ((0.50, 0.51), (0.50, 0.501), (0.10, 0.11))


def scale_table():
    rows = []
    for a, b in _scale_pairs():
        row = [f"{a:g} and {b:g}"]
        for scale in (1.0, 1000.0):
            row.append(_f(_dist(embed(a * scale, 256), embed(b * scale, 256)), 4))
        rows.append(row)
    # Gram: the inner product of two embeddings depends only on the difference of the arguments, and exchanging the halves is a
    # permutation of the coordinates, so it changes none of them.
    pairs = ((200.0, 500.0), (600.0, 900.0), (0.0, 300.0))
    g_rows = []
    worst_swap = 0.0
    for a, b in pairs:
        d1 = _dot(embed(a, 256), embed(b, 256))
        d2 = _dot(embed(a, 256, cos_first=False), embed(b, 256, cos_first=False))
        worst_swap = max(worst_swap, abs(d1 - d2))
        g_rows.append([f"{a:g}, {b:g}", f"{b - a:g}", _f(d1, 6), _f(d2, 6)])
    same = abs(_dot(embed(200.0, 256), embed(500.0, 256)) - _dot(embed(600.0, 256), embed(900.0, 256)))
    return (
        "Dimension 256, base 10000. The Euclidean distance between the embeddings of two noise fractions, when the embedding is fed "
        "the fraction itself (scale 1) and when it is fed 1000 times the fraction.\n\n"
        + _table(["sigma and sigma'", "distance, scale 1", "distance, scale 1000"], rows)
        + "\n\nThe inner product of two embeddings of dimension 256 with the cosine half first and with the halves exchanged, for "
        "arguments a and b (the arguments here are timesteps, already scaled).\n\n"
        + _table(["a, b", "b - a", "cos first", "sin first"], g_rows)
        + f"\n\nThe two orders give inner products that differ by less than {max(1e-12, math.ceil(worst_swap * 1e13) / 1e13):.0e}. "
        f"Pairs with the same difference b - a = 300 give inner products that differ by less than {max(1e-9, math.ceil(same * 1e10) / 1e10):.0e}."
    )


# ------------------------------------------------------------------ counting a network from its configuration

class Count:
    """Accumulates parameters, tensors and multiply-add FLOPs of a network by category and by resolution label. One multiply-add
    counts as two FLOPs; additions, normalisations, softmaxes and activations are not counted."""

    def __init__(self):
        self.params = {}
        self.flops = {}
        self.tensors = 0

    def add(self, cat, params=0, tensors=0, flops=0, where=""):
        self.params[cat] = self.params.get(cat, 0) + params
        self.tensors += tensors
        key = (cat, where)
        self.flops[key] = self.flops.get(key, 0) + flops

    def total_params(self):
        return sum(self.params.values())

    def total_flops(self):
        return sum(self.flops.values())

    def flops_by_cat(self):
        out = {}
        for (cat, _), v in self.flops.items():
            out[cat] = out.get(cat, 0) + v
        return out

    def flops_by_where(self):
        out = {}
        for (_, where), v in self.flops.items():
            out[where] = out.get(where, 0) + v
        return out


SDXL_UNET = dict(
    ch=(320, 640, 1280), down=("plain", "attn", "attn"), up=("attn", "attn", "plain"), layers=2, tf=(1, 2, 10), heads=(5, 10, 20),
    cross_dim=2048, time_dim=1280, add_time_dim=256, class_dim=2816, c_in=4, c_out=4, groups=32,
)
SDXL_TINY = dict(
    ch=(32, 64), down=("plain", "attn"), up=("attn", "plain"), layers=1, tf=(1, 2), heads=(2, 4),
    cross_dim=24, time_dim=128, add_time_dim=16, class_dim=8 + 6 * 16, c_in=4, c_out=4, groups=8,
)
ConvKinds = ("convolutions", "time projections", "norms", "self-attention", "cross-attention", "feed-forward", "token projections",
             "time and size embeddings", "self-attention scores", "cross-attention scores")


def unet_count(cfg, h, w, t_enc=77, through_mid=False):
    """Count the UNet2DConditionModel graph of `cfg` on a latent of h by w and t_enc text tokens, walking the stages in execution
    order: conv_in, the down blocks (layers resnets, each followed by a spatial transformer at an attention level, then a stride-2
    convolution between levels), the middle block, the up blocks (layers + 1 resnets that each read the hidden state concatenated
    with a skip popped from the stack, then a convolution after nearest-neighbour doubling between levels), conv_out. Returns the
    Count and the list of stage rows. With `through_mid` the walk stops after the middle block: the half of the network that a
    trainable copy duplicates (Chapter 11), with its conditioning embeddings, input convolution, down blocks and middle block."""
    n = Count()
    c0, te = cfg["ch"][0], cfg["time_dim"]
    levels = len(cfg["ch"])
    where = lambda hh, ww: f"{hh}x{ww}"  # noqa: E731
    rows = []

    def conv(cin, cout, k, hh, ww, name=""):
        n.add("convolutions", cin * cout * k * k + cout, 2, 2 * cin * cout * k * k * hh * ww, where(hh, ww))

    def norm(c, groups=True):
        n.add("norms", 2 * c, 2)

    def linear(tokens, i, o, bias, hh, ww, cat):
        n.add(cat, i * o + (o if bias else 0), 2 if bias else 1, 2 * tokens * i * o, where(hh, ww))

    def resnet(cin, cout, hh, ww):
        norm(cin)
        conv(cin, cout, 3, hh, ww)
        linear(1, te, cout, True, hh, ww, "time projections")
        norm(cout)
        conv(cout, cout, 3, hh, ww)
        if cin != cout:
            conv(cin, cout, 1, hh, ww)

    def transformer(c, heads, blocks, hh, ww):
        t = hh * ww
        norm(c)
        linear(t, c, c, True, hh, ww, "token projections")
        for _ in range(blocks):
            norm(c)
            linear(t, c, 3 * c, False, hh, ww, "self-attention")
            n.add("self-attention scores", 0, 0, 4 * t * t * c, where(hh, ww))      # scores q k^T and the weighted sum of values
            linear(t, c, c, True, hh, ww, "self-attention")
            norm(c)
            linear(t, c, c, False, hh, ww, "cross-attention")
            linear(t_enc, cfg["cross_dim"], 2 * c, False, hh, ww, "cross-attention")
            n.add("cross-attention scores", 0, 0, 4 * t * t_enc * c, where(hh, ww))
            linear(t, c, c, True, hh, ww, "cross-attention")
            norm(c)
            linear(t, c, 4 * c, True, hh, ww, "feed-forward")
            linear(t, c, 4 * c, True, hh, ww, "feed-forward")
            linear(t, 4 * c, c, True, hh, ww, "feed-forward")
        linear(t, c, c, True, hh, ww, "token projections")

    # conditioning: the timestep MLP (c0 -> te -> te) and the added-conditioning MLP (class_dim -> te -> te)
    for i, o in ((c0, te), (te, te), (cfg["class_dim"], te), (te, te)):
        n.add("time and size embeddings", i * o + o, 2, 2 * i * o, "-")
    conv(cfg["c_in"], c0, 3, h, w)
    stack = [(c0, h, w, "conv_in")]
    rows.append(["conv_in", f"{h}x{w}", f"{cfg['c_in']} -> {c0}", "0", "0", "-", "push 1"])
    hh, ww, prev = h, w, c0
    for i in range(levels):
        co = cfg["ch"][i]
        res = tf = 0
        for j in range(cfg["layers"]):
            resnet(prev if j == 0 else co, co, hh, ww)
            res += 1
            if cfg["down"][i] == "attn":
                transformer(co, cfg["heads"][i], cfg["tf"][i], hh, ww)
                tf += cfg["tf"][i]
            stack.append((co, hh, ww, f"down{i}.resnet{j}"))
        pushed = cfg["layers"]
        if i + 1 < levels:
            conv(co, co, 3, hh // 2, ww // 2)
            stack.append((co, hh // 2, ww // 2, f"down{i}.downsample"))
            pushed += 1
        rows.append([f"down {i}", f"{hh}x{ww}", f"{prev} -> {co}", str(res), str(tf), str(hh * ww) if tf else "-", f"push {pushed}"])
        if i + 1 < levels:
            hh, ww = hh // 2, ww // 2
        prev = co
    cm = cfg["ch"][-1]
    resnet(cm, cm, hh, ww)
    transformer(cm, cfg["heads"][-1], cfg["tf"][-1], hh, ww)
    resnet(cm, cm, hh, ww)
    rows.append(["middle", f"{hh}x{ww}", f"{cm} -> {cm}", "2", str(cfg["tf"][-1]), str(hh * ww), "-"])
    if through_mid:
        return n, rows
    prev_out = cm
    for i in range(levels):
        level = levels - 1 - i
        co = cfg["ch"][level]
        res = tf = 0
        popped = []
        for j in range(cfg["layers"] + 1):
            sc, sh, sw, _ = stack.pop()
            assert (sh, sw) == (hh, ww), "a skip must have the resolution of the hidden state it joins"
            cin = (prev_out if j == 0 else co) + sc
            resnet(cin, co, hh, ww)
            res += 1
            popped.append(str(cin))
            if cfg["up"][i] == "attn":
                transformer(co, cfg["heads"][level], cfg["tf"][level], hh, ww)
                tf += cfg["tf"][level]
        rows.append([f"up {i}", f"{hh}x{ww}", f"{prev_out} -> {co}", str(res), str(tf), str(hh * ww) if tf else "-",
                     f"pop {len(popped)} (resnet inputs " + ", ".join(popped) + ")"])
        if i + 1 < levels:
            hh, ww = hh * 2, ww * 2
            conv(co, co, 3, hh, ww)
        prev_out = co
    assert not stack, "the up path must consume the whole stack"
    norm(c0)
    conv(c0, cfg["c_out"], 3, hh, ww)
    rows.append(["conv_out", f"{hh}x{ww}", f"{c0} -> {cfg['c_out']}", "0", "0", "-", "-"])
    return n, rows


def _level_rows(cfg):
    """One row per level of the configuration: resolution divisor, channels, whether the level's blocks carry spatial transformers (the
    down and up path agree), transformer blocks per spatial transformer, heads and head width. The block count and the heads of a level
    without attention are not used."""
    rows = []
    for i, ch in enumerate(cfg["ch"]):
        attn = cfg["down"][i] == "attn"
        assert (cfg["up"][len(cfg["ch"]) - 1 - i] == "attn") == attn, "down and up path must agree on where the attention is"
        rows.append([str(i), f"1/{2 ** i}", str(ch), "yes" if attn else "no", str(cfg["tf"][i]) if attn else "-",
                     str(cfg["heads"][i]) if attn else "-", str(ch // cfg["heads"][i]) if attn else "-"])
    return rows


def _unet_config_text(cfg, name):
    """The configuration of one U-Net and the counting rules, so that every parameter, tensor and FLOP of the walk can be re-derived."""
    c0, te, ccross = cfg["ch"][0], cfg["time_dim"], cfg["cross_dim"]
    table = _table(["level", "resolution", "channels", "spatial transformers", "blocks per transformer", "heads", "head width"],
                   _level_rows(cfg), "llrlrrr")
    return (
        f"{name}: {len(cfg['ch'])} levels, resnets per down block {cfg['layers']}, per up block {cfg['layers'] + 1}, {cfg['c_in']} latent channels "
        f"in and {cfg['c_out']} out, {cfg['groups']} groups in every group normalisation, time dimension {te}, cross-attention (text) dimension "
        f"{ccross}. The middle block is a resnet, a spatial transformer of {cfg['tf'][-1]} blocks at {cfg['ch'][-1]} channels with {cfg['heads'][-1]} heads, and a resnet. "
        f"Per level:\n\n{table}\n\n"
        f"The conditioning vector comes from two networks, each a linear layer, an activation and a second linear layer, all with biases: "
        f"the timestep network {c0} -> {te} -> {te} (the sinusoid is {c0} wide, the width of the first level) and the added-conditioning network "
        f"{cfg['class_dim']} -> {te} -> {te}, whose input is the pooled text vector followed by six size values of {cfg['add_time_dim']} numbers each "
        f"({cfg['class_dim'] - 6 * cfg['add_time_dim']} + 6 x {cfg['add_time_dim']} = {cfg['class_dim']}); their outputs are added."
    )


_UNET_RULES = (
    "Counting rules. A convolution from ci to co channels with a k by k kernel has co ci k^2 weights and co biases. A group normalisation of c "
    "channels has a gain and a shift (2c). A linear layer from i to o numbers has i o weights and, where stated, o biases. A resnet from ci to co "
    "channels is: group normalisation of ci, 3 x 3 convolution ci -> co, a linear time projection (time dimension -> co, with bias) added to every "
    "pixel, group normalisation of co, 3 x 3 convolution co -> co, and a 1 x 1 shortcut convolution ci -> co only when ci differs from co. A spatial "
    "transformer at c channels is: group normalisation, a linear c -> c with bias (proj_in), then for each block a layer normalisation (2c), "
    "self-attention (query, key and value c -> c each without bias, 3c^2 in all; output c -> c with bias), a layer normalisation, cross-attention "
    "(query c -> c without bias, key and value text dimension -> c each without bias; output c -> c with bias), a layer normalisation, and the "
    "feed-forward (a gated projection c -> 4c computed twice, a gate and a value, each a weight and a bias, which is a single c -> 8c layer in "
    "the original checkpoint layout; the product of the activated gate and the value goes through 4c -> c with bias), and after the last block a "
    "linear c -> c with bias (proj_out). Between levels the down path has a stride-2 3 x 3 convolution at the channels of the level it leaves; the "
    "up path doubles the grid by nearest neighbour and applies a 3 x 3 convolution at the channels of the level it enters. The up resnets read the "
    "hidden state concatenated with the popped skip. The output is a group normalisation of the first level's channels, then a 3 x 3 convolution to "
    "the output channels. A tensor is one weight or one bias (a normalisation's gain and shift are two)."
)


def unet_walk():
    n, rows = unet_count(SDXL_UNET, 128, 128)
    stack_widths = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
    return (
        "The SDXL U-Net (320, 640, 1280 channels; resnets per down block 2; the configuration lists transformer blocks per level as 1, 2, 10 "
        "but attention is on the two coarser levels only, so the 1 of the first level is unused: 2 blocks per spatial transformer at 64 by 64, "
        "10 at 32 by 32 and in the middle block) on the 128 by 128 latent of a 1024 by 1024 image. A row is a stage: its resolution, "
        "channels in and out, resnets, spatial-transformer blocks, tokens for the self-attention of those blocks, and what it does to "
        "the stack of skips. The skip stack is pushed in the order conv_in, down 0 (two resnets, then its stride-2 convolution), "
        "down 1 (the same), down 2 (two resnets, no convolution): nine entries of widths "
        + ", ".join(str(c) for c in stack_widths) + ".\n\n"
        + _table(["stage", "resolution", "channels", "resnets", "transformer blocks", "tokens T", "skips"], rows, "lllrrrl")
    )


def _sdxl_totals():
    n, _ = unet_count(SDXL_UNET, 128, 128)
    return n


def unet_params():
    n = _sdxl_totals()
    tiny, tiny_rows = unet_count(SDXL_TINY, 16, 16, t_enc=8)
    rows = []
    for cat in ConvKinds:
        p = n.params.get(cat, 0)
        if p:
            rows.append([cat, _int(p), _pct(p / n.total_params())])
    rows.append(["total", _int(n.total_params()), "100.0%"])
    assert sum(n.params.values()) == n.total_params()
    gb32 = n.total_params() * 4 / 1e9
    gb16 = n.total_params() * 2 / 1e9
    # BasicTransformerBlocks and resnets of the walk
    tb = 0
    cfg = SDXL_UNET
    L = len(cfg["ch"])
    for i in range(L):
        if cfg["down"][i] == "attn":
            tb += cfg["layers"] * cfg["tf"][i]
    tb += cfg["tf"][-1]
    for i in range(L):
        if cfg["up"][i] == "attn":
            tb += (cfg["layers"] + 1) * cfg["tf"][L - 1 - i]
    res = cfg["layers"] * L + 2 + (cfg["layers"] + 1) * L
    # tensors: the counting rule counts a fused qkv / kv as one tensor and a split GEGLU projection as two, as in the manifest
    n_tensors_ckpt = n.tensors + tb      # per block: query, key, value are 3 tensors not 1 (+2), key, value 2 not 1 (+1), the gated projection 2 not 4 (-2)
    return (
        _unet_config_text(SDXL_UNET, "The SDXL U-Net") + "\n\n" + _UNET_RULES + "\n\n"
        "Every parameter of the SDXL U-Net derived from that configuration, by what the weights do (a convolution's bias, a "
        "normalisation's gain and shift and a linear layer's bias are counted with their layer).\n\n"
        + _table(["kind", "parameters", "share"], rows)
        + f"\n\nThe network has {res} resnets and {tb} spatial-transformer blocks, and {_int(n.total_params())} parameters: "
        f"{gb32:.2f} GB in 32-bit floats and {gb16:.2f} GB in 16-bit floats. Counting one tensor for each weight or bias, with a "
        f"fused query-key-value matrix and a fused key-value matrix of the cross-attention each as one tensor and the feed-forward's "
        f"gated projection as two (the way brain stores them), the network has {n.tensors} tensors; with the three separate "
        f"attention matrices and the one gated projection of the original layout it has {n_tensors_ckpt}.\n\n"
        "The same walk, with the same rules, on the variant that brain builds for its smoke tests gives "
        f"{_int(tiny.total_params())} parameters in {tiny.tensors} tensors. Its configuration:\n\n" + _unet_config_text(SDXL_TINY, "The smoke-test U-Net")
        + "\n\nand its walk on a 16 by 16 latent:\n\n"
        + _table(["stage", "resolution", "channels", "resnets", "transformer blocks", "tokens T", "skips"], tiny_rows, "lllrrrl")
    )


def unet_flops():
    n = _sdxl_totals()
    by_cat = n.flops_by_cat()
    tot = n.total_flops()
    rows = [[cat, _gflop(by_cat.get(cat, 0)), _pct(by_cat.get(cat, 0) / tot)] for cat in ConvKinds if by_cat.get(cat, 0)]
    rows.append(["total", _gflop(tot), "100.0%"])
    wh = n.flops_by_where()
    lv = [[k, _gflop(wh[k]), _pct(wh[k] / tot)] for k in ("128x128", "64x64", "32x32", "-") if k in wh]
    p_rows = []
    for l in range(3):
        c = SDXL_UNET["ch"][l]
        hts = SDXL_UNET["heads"][l]
        t = (128 >> l) ** 2
        p_rows.append([f"{128 >> l}x{128 >> l}", str(c), str(hts), _int(t), _int(hts * t * t * 4), "yes" if SDXL_UNET["down"][l] == "attn" else "no"])
    return (
        "One forward pass of the SDXL U-Net on the 128 by 128 grid of a 1024 by 1024 image with 77 text tokens, in GFLOP (a multiply-add is two operations; "
        "additions, normalisations, softmax and activations are not counted), by kind and by the resolution at which it is spent. Every layer of the "
        "counting rules above is counted, including the time projections of the resnets and the two conditioning networks (a matrix applied to a single "
        f"vector, {(by_cat.get('time projections', 0) + by_cat.get('time and size embeddings', 0)) / 1e6:.0f} million floating-point operations together). A convolution costs 2 ci co k^2 per output pixel, a linear layer 2 i o per token; the cross-attention key "
        "and value projections are applied to the 77 text tokens, 2 x 77 x 2048 x c each for a block of c channels; the scores q k^T and the weighted "
        "sum of the values cost 2 T^2 c each, 4 T^2 c together (T the tokens of the grid), and in cross-attention 4 T x 77 x c.\n\n"
        + _table(["kind", "GFLOP", "share"], rows)
        + "\n\n"
        + _table(["resolution", "GFLOP", "share"], lv)
        + "\n\nThe score matrix that a self-attention layer would build at each resolution (heads x T x T entries, 4 bytes each, if "
        "it were materialised), and whether the network has attention there.\n\n"
        + _table(["resolution", "channels", "heads", "tokens T", "score bytes (32-bit)", "attention"], p_rows)
    )


def unet_cost_data(sides=(256, 512, 768, 1024, 1536, 2048)):
    """(side, total GFLOP, share spent in the T x T terms of self-attention) of the SDXL U-Net for each image side in pixels."""
    out = []
    for s in sides:
        n, _ = unet_count(SDXL_UNET, s // 8, s // 8)
        quad = 0                                       # the T^2 term alone: 4 T^2 c per transformer block
        L = len(SDXL_UNET["ch"])
        cfg = SDXL_UNET
        for i in range(L):
            if cfg["down"][i] == "attn":
                quad += cfg["layers"] * cfg["tf"][i] * 4 * ((s // 8 >> i) ** 4) * cfg["ch"][i]
        quad += cfg["tf"][-1] * 4 * ((s // 8 >> (L - 1)) ** 4) * cfg["ch"][-1]
        for i in range(L):
            if cfg["up"][i] == "attn":
                lv = L - 1 - i
                quad += (cfg["layers"] + 1) * cfg["tf"][lv] * 4 * ((s // 8 >> lv) ** 4) * cfg["ch"][lv]
        out.append((s, n.total_flops() / 1e9, quad / n.total_flops()))
    return out


# ------------------------------------------------------------------ Z-Image

ZIMAGE = dict(dim=3840, layers=30, refiner=2, heads=30, cap_dim=2560, in_ch=16, patch=2, axes=(32, 48, 48), axes_len=(1536, 512, 512),
              theta=256.0, freq_dim=256, t_hidden=1024, cap_len=512)


def zimage_block_params(dim, heads, modulated):
    """The tensors of one block: four dim x dim attention matrices, a gain per head channel for the query and for the key, three
    feed-forward matrices of width dim*8//3, four dim-wide norm gains, and (modulated) a projection from the min(dim, 256)-wide
    conditioning vector to four dim-wide vectors (scale and gate for attention, scale and gate for the feed-forward) with bias."""
    hd = dim // heads
    hid = dim * 8 // 3
    cdim = min(dim, 256)
    parts = {
        "attention matrices": 4 * dim * dim,
        "query and key gains": 2 * hd,
        "feed-forward matrices": 3 * dim * hid,
        "norm gains": 4 * dim,
        "modulation projection": (4 * dim * cdim + 4 * dim) if modulated else 0,
    }
    return parts


def zimage_params_dict(c=ZIMAGE):
    d, hd = c["dim"], c["dim"] // c["heads"]
    cdim = min(d, 256)
    pd = c["in_ch"] * c["patch"] ** 2
    main = zimage_block_params(d, c["heads"], True)
    ctx = zimage_block_params(d, c["heads"], False)
    wrap = {
        "caption embedder": c["cap_dim"] + d * c["cap_dim"] + d,
        "pad tokens": 2 * d,
        "patch embedder": d * pd + d,
        "timestep MLP": c["t_hidden"] * c["freq_dim"] + c["t_hidden"] + cdim * c["t_hidden"] + cdim,
        "final layer": d * cdim + d + pd * d + pd,
    }
    return main, ctx, wrap


def zimage_params():
    c = ZIMAGE
    main, ctx, wrap = zimage_params_dict()
    bm, bc = sum(main.values()), sum(ctx.values())
    n_main, n_noise, n_ctx = c["layers"], c["refiner"], c["refiner"]
    total = n_main * bm + n_noise * bm + n_ctx * bc + sum(wrap.values())
    rows = [[k, _int(v)] for k, v in main.items()]
    rows.append(["one block with modulation", _int(bm)])
    rows.append(["one block without (context refiner)", _int(bc)])
    stack = [
        [f"{n_main} main blocks", _int(n_main * bm)],
        [f"{n_noise} noise-refiner blocks", _int(n_noise * bm)],
        [f"{n_ctx} context-refiner blocks", _int(n_ctx * bc)],
    ] + [[k, _int(v)] for k, v in wrap.items()] + [["total", _int(total)]]
    mod = (n_main + n_noise) * (4 * c["dim"] * min(c["dim"], 256) + 4 * c["dim"])
    tensors = 15 + (n_main + n_noise) * 15 + n_ctx * 13
    return (
        "Z-Image Turbo from its configuration: width 3840, 30 heads (head width 128), feed-forward width 3840*8//3 = 10240, "
        "conditioning width min(3840, 256) = 256, 30 main blocks and 2 blocks in each of the noise and context refiners, "
        "16 latent channels in 2 by 2 patches (64 values per token), caption features of width 2560. Outside the blocks: the caption embedder "
        "(a gain of 2560, a 2560 -> 3840 matrix and its bias), two learned pad tokens of width 3840, one for the image stream and one for the caption "
        "stream (2 x 3840 parameters), the patch embedder (64 -> 3840 with bias), the timestep network (256 -> 1024 -> 256, both with biases) and the "
        "final layer (a 256 -> 3840 modulation with bias, then 3840 -> 64 with bias).\n\n"
        + _table(["inside one block", "parameters"], rows)
        + "\n\n"
        + _table(["part of the network", "parameters"], stack)
        + f"\n\nThe {n_main + n_noise} modulated blocks spend {_int(mod)} parameters ({_pct(mod / total)} of the network) on turning the "
        f"256-wide timestep vector into scales and gates; the tensors are {tensors} in all (15 outside the blocks, 15 in a "
        "modulated block, 13 in a context-refiner block)."
    )


def token_ids():
    """The position ids of the joint sequence: the image tokens first, then the caption tokens; a caption token i has the triple
    (1 + i, 0, 0), an image token at frame f, patch row h, patch column w has (cap_len + 1 + f, h, w)."""
    cap_len, f_n, h_n, w_n = 2, 1, 2, 3
    rows = []
    idx = 0
    for f in range(f_n):
        for h in range(h_n):
            for w in range(w_n):
                rows.append([str(idx), "image patch", f"({cap_len + 1 + f}, {h}, {w})"])
                idx += 1
    for i in range(cap_len):
        rows.append([str(idx), "caption token", f"({1 + i}, 0, 0)"])
        idx += 1
    return (
        "A caption of 2 tokens and an image latent of 4 rows by 6 columns in 2 by 2 patches, which is a grid of 1 frame, 2 rows and 3 "
        "columns of patches: the position triples (axis 0, axis 1, axis 2) of the joint sequence, image first.\n\n"
        + _table(["index", "token", "ids"], rows, "llr")
    )


def rope_axes():
    c = ZIMAGE
    rows = []
    for name, d, ln in zip(("sequence and frame", "patch row", "patch column"), c["axes"], c["axes_len"]):
        pairs = d // 2
        slow = c["theta"] ** (-2.0 * (pairs - 1) / d)
        rows.append([name, str(d), str(pairs), "1", f"{slow:.5f}", f"{2 * math.pi / slow:.0f}", str(ln)])
    return (
        "The three axes of Z-Image's rotary position in a head of width 128: pair k of an axis of width d turns by (id on the axis) "
        "times 256^(-2k/d). The fastest rate of every axis is 1; the wavelength is 2 pi over the slowest rate.\n\n"
        + _table(["axis", "width d", "pairs", "fastest rate", "slowest rate", "slowest wavelength", "table length"], rows)
    )


def joint_counts(side, cap_len=512, patch_px=16):
    n_img = (side // patch_px) ** 2
    return n_img, n_img + cap_len


# ------------------------------------------------------------------ costs of transformers (multiply-adds, two FLOPs each)

def zimage_pass(side, cap_len=512, c=ZIMAGE):
    """FLOPs of one forward pass of Z-Image, split into the linear layers and the T x T terms of attention (2 T^2 d for the scores and
    2 T^2 d for the weighted sum, per block)."""
    d = c["dim"]
    hid = d * 8 // 3
    n_img, n_joint = joint_counts(side, cap_len)
    pd = c["in_ch"] * c["patch"] ** 2
    lin_block = lambda n: 2 * n * (4 * d * d + 3 * d * hid)      # noqa: E731
    att_block = lambda n: 4 * n * n * d                          # noqa: E731
    lin = c["layers"] * lin_block(n_joint) + c["refiner"] * lin_block(n_img) + c["refiner"] * lin_block(cap_len)
    att = c["layers"] * att_block(n_joint) + c["refiner"] * att_block(n_img) + c["refiner"] * att_block(cap_len)
    emb = 2 * n_img * pd * d + 2 * cap_len * c["cap_dim"] * d + 2 * n_joint * d * pd
    return lin + emb, att


FLUX2 = {
    "klein-4b": dict(d=3072, heads=24, double=5, single=20, mlp_ratio=3, in_ch=128, ctx=7680),
    "klein-9b": dict(d=4096, heads=32, double=8, single=24, mlp_ratio=3, in_ch=128, ctx=12288),
}


def zimage_vector_flops(c=ZIMAGE):
    """The multiply-adds, as FLOPs, of the matrices that act on the one conditioning vector and not on tokens: the modulation projection
    of every modulated block (256 -> 4 dim), the timestep network and the final layer's modulation (256 -> dim). `zimage_pass` leaves
    them out."""
    d, cdim = c["dim"], min(c["dim"], 256)
    return ((c["layers"] + c["refiner"]) * 2 * cdim * 4 * d + 2 * (c["t_hidden"] * c["freq_dim"] + cdim * c["t_hidden"]) + 2 * cdim * d)


def flux2_vector_flops(c):
    """The same for FLUX.2: the shared modulation projections (17 d^2 weights) and the timestep network (256 -> d -> d)."""
    d = c["d"]
    return 2 * 17 * d * d + 2 * (256 * d + d * d)


def flux2_params(c):
    d = c["d"]
    hd = d // c["heads"]
    mlp = d * c["mlp_ratio"]
    emb = d * c["in_ch"] + d * c["ctx"] + d * 256 + d * d
    mod = 6 * d * d + 6 * d * d + 3 * d * d + 2 * d * d
    dbl = c["double"] * 2 * (3 * d * d + d * d + 2 * hd + 2 * mlp * d + d * mlp)
    sgl = c["single"] * ((3 * d + 2 * mlp) * d + d * (d + mlp) + 2 * hd)
    out = c["in_ch"] * d
    return dict(embedders=emb, modulation=mod, double=dbl, single=sgl, output=out, total=emb + mod + dbl + sgl + out)


def flux2_tensors(c):
    return 4 + 3 + c["double"] * 2 * 6 + c["single"] * 4 + 2


def flux2_pass(c, side, txt=512):
    d = c["d"]
    mlp = d * c["mlp_ratio"]
    n_img, n = joint_counts(side, txt)
    dbl_lin = 2 * n_img * (4 * d * d + 3 * mlp * d) + 2 * txt * (4 * d * d + 3 * mlp * d)
    sgl_lin = 2 * n * ((3 * d + 2 * mlp) * d + d * (d + mlp))
    lin = c["double"] * dbl_lin + c["single"] * sgl_lin
    att = (c["double"] + c["single"]) * 4 * n * n * d
    pd = c["in_ch"]
    emb = 2 * n_img * pd * d + 2 * txt * c["ctx"] * d + 2 * n_img * d * pd
    return lin + emb, att


def architecture_table():
    sd, _ = unet_count(SDXL_UNET, 128, 128)
    sd_time = sd.params["time and size embeddings"] + sd.params["time projections"]
    main, ctx, wrap = zimage_params_dict()
    c = ZIMAGE
    z_total = (c["layers"] + c["refiner"]) * sum(main.values()) + c["refiner"] * sum(ctx.values()) + sum(wrap.values())
    z_mod = (c["layers"] + c["refiner"]) * main["modulation projection"] + wrap["timestep MLP"] + wrap["final layer"] - 0
    z_mod = (c["layers"] + c["refiner"]) * main["modulation projection"] + wrap["timestep MLP"] + (c["dim"] * min(c["dim"], 256) + c["dim"])
    zl, za = zimage_pass(1024)
    rows = [["SDXL U-Net", "-", _int(sd.total_params()), _int(sd_time), _pct(sd_time / sd.total_params()),
             _gflop(sd.total_flops()), "-"]]
    rows.append(["Z-Image Turbo", "single", _int(z_total), _int(z_mod), _pct(z_mod / z_total), _gflop(zl + za), _pct(za / (zl + za))])
    for name, cfg in FLUX2.items():
        p = flux2_params(cfg)
        fl_, fa = flux2_pass(cfg, 1024)
        m = p["modulation"] + cfg["d"] * 256 + cfg["d"] * cfg["d"]
        rows.append([f"FLUX.2 {name}", "double + single", _int(p["total"]), _int(m), _pct(m / p["total"]), _gflop(fl_ + fa),
                     _pct(fa / (fl_ + fa))])
    return (
        "Four denoisers at 1024 by 1024 pixels (SDXL: a 128 by 128 grid and 77 text tokens; the transformers: 4096 image tokens and 512 "
        "caption tokens, one token per 16 by 16 pixels). 'Time parameters' are the weights that turn the timestep (and for SDXL the "
        "size and pooled-text vector) into per-layer conditioning: SDXL's two embedding MLPs and the 17 per-resnet projections; "
        "Z-Image's timestep MLP, the per-block modulation projections and the final layer's; FLUX.2's timestep MLP, three shared "
        "modulation projections and the final layer's. GFLOP counts multiply-adds as two operations and leaves out norms, softmax and "
        "activations. The last column is the share spent in the T x T terms of attention. What is counted differs by what a pass applies to: for "
        "SDXL, every layer of the counting rules of the U-Net table, including the matrices that act on the one conditioning vector (about 50 million floating-point operations); "
        "for the transformers, the attention and feed-forward matrices applied to every token of every block (refiner blocks to their own tokens "
        "only), the patch embedder, the caption embedder and the final projection, and 4 T^2 d per block for the scores and the weighted sum "
        "(T the tokens that attend), but not the matrices that act on the one conditioning vector: "
        f"{zimage_vector_flops() / 1e6:.0f} million floating-point operations for Z-Image (the {ZIMAGE['layers'] + ZIMAGE['refiner']} modulation projections, the timestep network and "
        f"the final layer's modulation) and {flux2_vector_flops(FLUX2['klein-4b']) / 1e6:.0f} and {flux2_vector_flops(FLUX2['klein-9b']) / 1e6:.0f} million floating-point operations "
        "for the two FLUX.2 sizes (the shared modulation projections and the timestep network), which would move the last printed digit.\n\n"
        + _table(["network", "stream", "parameters", "time parameters", "share", "GFLOP per pass", "attention share"], rows,
                 "llrrrrr")
    )


def work_per_image():
    sd, _ = unet_count(SDXL_UNET, 128, 128)
    zl, za = zimage_pass(1024)
    rows = []

    def row(name, passes, params, flop, note):
        rows.append([name, str(passes), _gflop(flop), f"{flop / (2 * params):,.0f}", f"{passes * flop / 1e12:,.1f}", note])

    row("SDXL, 30 steps with guidance", 60, sd.total_params(), sd.total_flops(), "two passes a step")
    row("Z-Image Turbo, 8 steps", 8, zimage_total_params(), zl + za, "no guidance")
    for name, passes, note in (("klein-4b", 4, "distilled, no guidance"), ("klein-9b", 4, "distilled, no guidance")):
        c = FLUX2[name]
        fl_, fa = flux2_pass(c, 1024)
        row(f"FLUX.2 {name}, 4 steps", passes, flux2_params(c)["total"], fl_ + fa, note)
    c = FLUX2["klein-4b"]
    fl_, fa = flux2_pass(c, 1024)
    row("FLUX.2 base 4b, 50 steps with guidance", 100, flux2_params(c)["total"], fl_ + fa, "two passes a step")
    return (
        "The work of one 1024 by 1024 image at each pipeline's default pass count (the passes of Chapter 7's opening table). A "
        "pass reads every weight once; at 16 bits that is 2 bytes a parameter, so the operations per weight byte of a pass are its GFLOP "
        "over twice the parameters. TFLOP per image is passes times GFLOP per pass over 1000.\n\n"
        + _table(["pipeline", "passes", "GFLOP per pass", "operations per weight byte", "TFLOP per image", "note"], rows, "lrrrrl")
    )


def zimage_total_params():
    main, ctx, wrap = zimage_params_dict()
    c = ZIMAGE
    return (c["layers"] + c["refiner"]) * sum(main.values()) + c["refiner"] * sum(ctx.values()) + sum(wrap.values())


def flux2_table():
    rows = []
    for name, cfg in FLUX2.items():
        p = flux2_params(cfg)
        rows.append([name, str(cfg["d"]), f"{cfg['double']} + {cfg['single']}", _int(p["double"]), _int(p["single"]),
                     _int(p["modulation"]), _int(p["total"]), str(flux2_tensors(cfg))])
    return (
        "FLUX.2 from the configurations of its two sizes (hidden width d, SwiGLU width 3d, 128 latent channels per token, caption "
        "features of width 7680 and 12288). The total also holds what is outside the blocks, all of it without biases: the patch embedder (128 -> d), "
        "the caption embedder (caption width -> d), the timestep network (256 -> d -> d) and the final projection (d -> 128). A double-stream block holds two sets of attention, feed-forward and query-key gains, one "
        "for the image stream and one for the caption stream; a single-stream block holds one set with the attention and the "
        "feed-forward fused into two matrices. The shared modulation projections (6 d^2 for each double-stream stream, 3 d^2 for "
        "the single blocks, 2 d^2 for the final layer) serve every block of their kind.\n\n"
        + _table(["variant", "d", "double + single", "double blocks", "single blocks", "modulation", "total", "tensors"], rows,
                 "lrrrrrrr")
    )


def cost_vs_side_data(sides=(256, 512, 768, 1024, 1536, 2048)):
    """Share of the multiply-adds of one pass spent in attention's T x T terms against the image side, for the three families."""
    sd = [(s, share) for s, _, share in unet_cost_data(sides)]
    zi = [(s, zimage_pass(s)[1] / sum(zimage_pass(s))) for s in sides]
    fx = [(s, flux2_pass(FLUX2["klein-4b"], s)[1] / sum(flux2_pass(FLUX2["klein-4b"], s))) for s in sides]
    return sd, zi, fx


def cost_vs_side():
    sides = (512, 1024, 1536, 2048)
    rows = []
    for s in sides:
        n_img, n = joint_counts(s)
        lin, att = zimage_pass(s)
        c_scores = 30 * n * n * 4
        sdn, _ = unet_count(SDXL_UNET, s // 8, s // 8)
        rows.append([f"{s}", _int(n_img), _int(n), _gflop(lin + att), _pct(att / (lin + att)), f"{c_scores / 1e9:.1f}",
                     _gflop(sdn.total_flops())])
    return (
        "Z-Image with a 512-token caption at four image sides: the image tokens (one per 16 by 16 pixels), the joint sequence, the "
        "GFLOP of one pass and the share of them in attention's T x T terms, the bytes (in GB, 32-bit) of the score matrices of the 30 "
        "heads of ONE main block if they were stored, and, for comparison, the GFLOP of one SDXL U-Net pass at the same image side "
        "(grid side = image side / 8, 77 text tokens; counted with the rules of the U-Net table, so with the conditioning matrices, while the Z-Image column "
        "leaves out the matrices that act on the one conditioning vector, as in the table of the four denoisers). The score elements of the 30 heads, 30 T^2, still fit the 32-bit count of a dispatch for "
        f"T up to {math.isqrt((2 ** 32 - 1) // 30):,}.\n\n"
        + _table(["image side", "image tokens", "joint tokens N", "GFLOP, Z-Image", "attention share", "scores of one block (GB)",
                  "GFLOP, SDXL"], rows)
    )


# ------------------------------------------------------------------ the sign and the clock of Z-Image

def sign_convention():
    rows = []
    for x, s, s2 in ((1.0, 0.6, 0.4), (-2.2, 0.3, 0.2), (0.4, 0.9, 0.7)):
        m = fl.posterior_mean(x, 1.0 - s, s)
        u = (x - m) / s                     # the velocity d x / d sigma = noise - x_0 of the straight path, in expectation
        t_dit = 1.0 - s                     # the clock the transformer is given: 0 at noise, 1 at data
        arg = 1000.0 * t_dit                # the number the sinusoid sees
        out = -u                            # what the transformer is trained to output: E[x_0 - noise | x]
        step = x + (s2 - s) * (-out)
        rows.append([f"{x:g}", f"{s:g}", f"{s2:g}", _f(u, 6), _f(t_dit, 1), f"{arg:g}", _f(out, 6), _f(step, 6),
                     _f(x + (s2 - s) * u, 6)])
    return (
        "Three Euler steps of the straight path with the exact denoiser of the three clusters, written in the clock and the sign of "
        "Z-Image: velocity u = E[noise - x_0 | x] = (x - guess)/sigma; the transformer is given t = 1 - sigma, the sinusoid "
        "sees 1000 t, and the transformer's output is -u = E[x_0 - noise | x]; the sampler negates the output and steps "
        "x + (sigma' - sigma) u.\n\n"
        + _table(["x", "sigma", "sigma'", "u", "t", "1000 t", "output (-u)", "step from -output", "Euler step with u"], rows)
    )


# ------------------------------------------------------------------ one layer of attention over five tokens

def _round1(x):
    return math.floor(abs(x) * 10.0 + 0.5 + 1e-9) / 10.0 * (1.0 if x >= 0 else -1.0)


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


JOINT = "denoiser_architectures.joint"
JOINT_D, JOINT_IMG, JOINT_TXT = 4, 3, 2


def _joint_setup():
    g = bookrng.generator(JOINT)
    X = [[_round1(g.normal()) for _ in range(JOINT_D)] for _ in range(JOINT_IMG + JOINT_TXT)]
    W = [[[_round1(0.5 * g.normal()) for _ in range(JOINT_D)] for _ in range(JOINT_D)] for _ in range(3)]
    return X, W


def _joint_layer(X, W):
    """One single-head layer. Joint: every token (image rows 0 .. 2, caption rows 3 .. 4) makes a query, key and value, and each query
    takes one softmax over all five keys. Cross: only the image tokens make queries, and they take one softmax over the two caption
    keys; the caption tokens are not updated."""
    Q, K, V = (_matmul(X, w) for w in W)
    sc = 1.0 / math.sqrt(JOINT_D)
    n = len(X)
    joint = [_softmax([sc * sum(q * k for q, k in zip(Q[i], K[j])) for j in range(n)]) for i in range(n)]
    cross = [_softmax([sc * sum(q * k for q, k in zip(Q[i], K[j])) for j in range(JOINT_IMG, n)]) for i in range(JOINT_IMG)]
    out_j = _matmul(joint, V)
    out_c = [_matmul([cross[i]], V[JOINT_IMG:])[0] for i in range(JOINT_IMG)] + [X[j][:] for j in range(JOINT_IMG, n)]
    return joint, cross, out_j, out_c


def joint_attention():
    X, W = _joint_setup()
    joint, cross, out_j, out_c = _joint_layer(X, W)
    names = ["image 0", "image 1", "image 2", "caption 0", "caption 1"]
    rows = []
    for i in range(5):
        mi = sum(joint[i][:JOINT_IMG])
        mt = sum(joint[i][JOINT_IMG:])
        rows.append([names[i], " ".join(f"{p:.3f}" for p in joint[i]), f"{mi:.3f}", f"{mt:.3f}",
                     " ".join(f"{p:.3f}" for p in cross[i]) if i < JOINT_IMG else "(not updated)"])
    # information flow: add 0.5 to every entry of image token 0, see how the caption outputs move
    X2 = [r[:] for r in X]
    X2[0] = [v + 0.5 for v in X2[0]]
    _, _, oj2, oc2 = _joint_layer(X2, W)
    d_joint = max(abs(a - b) for i in range(JOINT_IMG, 5) for a, b in zip(out_j[i], oj2[i]))
    d_cross = max(abs(a - b) for i in range(JOINT_IMG, 5) for a, b in zip(out_c[i], oc2[i]))
    # and the other way: add 0.5 to every entry of caption token 0 and see the image tokens move
    X3 = [r[:] for r in X]
    X3[JOINT_IMG] = [v + 0.5 for v in X3[JOINT_IMG]]
    _, _, oj3, oc3 = _joint_layer(X3, W)
    e_joint = max(abs(a - b) for i in range(JOINT_IMG) for a, b in zip(out_j[i], oj3[i]))
    e_cross = max(abs(a - b) for i in range(JOINT_IMG) for a, b in zip(out_c[i], oc3[i]))
    wnames = ("W_Q", "W_K", "W_V")
    wt = "\n".join(
        "| " + " | ".join(f"({', '.join(f'{v + 0.0:g}' for v in W[m][r])})" for m in range(3)) + " |" for r in range(JOINT_D))
    return (
        f"Generator: the book's SplitMix64, seed 0x{bookrng.seed_of(JOINT):016x} (`seed_of` of the name `{JOINT}`). Draws, in this "
        "order: the 20 entries of X (5 tokens of 4 numbers, image tokens first), each `round(normal(), 1)`; then W_Q, W_K, W_V "
        "(4 by 4 each, row by row), each `round(0.5 * normal(), 1)` (`round` to one decimal, ties away from zero). One head of width "
        "4, scores divided by 2, no mask, no position information.\n\n"
        + _table(["token", "x (4 numbers)"], [[names[i], "(" + ", ".join(f"{v + 0.0:g}" for v in X[i]) + ")"] for i in range(5)], "ll")
        + "\n\n| " + " | ".join(wnames) + " |\n|---|---|---|\n" + wt
        + "\n\nThe attention weights of each token's query. Joint: one softmax over all five keys (image keys 0 to 2, caption keys 3 "
        "and 4). Cross: the image queries only, one softmax over the two caption keys; the caption tokens are not updated.\n\n"
        + _table(["token", "joint weights over keys 0 .. 4", "mass on image keys", "mass on caption keys", "cross weights over keys 3, 4"],
                 rows, "lllll")
        + f"\n\nAdd 0.5 to all four entries of image token 0 and look at the outputs of the two caption tokens: they move by at most "
        f"{d_joint:.4f} under joint attention and by {d_cross:.4f} under cross-attention. Add 0.5 to caption token 0 and look at the "
        f"image tokens: they move by at most {e_joint:.4f} under joint attention and by {e_cross:.4f} under cross-attention."
    )


# ------------------------------------------------------------------ the small time-conditioned denoiser

SMALL = "denoiser_architectures.small"
SMALL_INPUT = "denoiser_architectures.small_input"
S_DIM, S_HEADS, S_HID, S_FREQ, S_THID, S_CAPD, S_C, S_H, S_W, S_P, S_CAP, S_BLOCKS = 24, 2, 64, 16, 32, 5, 2, 4, 4, 2, 2, 2
S_HD = S_DIM // S_HEADS                 # 12
S_AXES = (4, 4, 4)
S_THETA = 256.0
S_EPS, S_EPS_FINAL = 1e-5, 1e-6
S_PD = S_C * S_P * S_P                  # 8 values per patch


def _linear_rows(X, W, b=None):
    """Y[r][o] = b[o] + sum_i X[r][i] W[o][i]; W is [out][in] and the sum runs over i in increasing order."""
    out = []
    for x in X:
        row = []
        for o, wr in enumerate(W):
            acc = b[o] if b is not None else 0.0
            for xi, wi in zip(x, wr):
                acc += xi * wi
            row.append(acc)
        out.append(row)
    return out


def _rms_rows(X, w, eps):
    out = []
    for x in X:
        inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
        out.append([wi * v * inv for wi, v in zip(w, x)])
    return out


def _ln_rows(X, eps):
    out = []
    for x in X:
        mean = sum(x) / len(x)
        var = sum((v - mean) ** 2 for v in x) / len(x)
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(v - mean) * inv for v in x])
    return out


def _silu(v):
    return v / (1.0 + math.exp(-v))


def small_spec():
    """(name, shape, kind) of every tensor in draw order; kind is 'w' (normal() / sqrt(fan_in)), 'b' (0.1 normal()) or 'g' (1 + 0.1 normal())."""
    d, hid = S_DIM, S_HID
    spec = [("cap_embedder.0.weight", (S_CAPD,), "g"), ("cap_embedder.1.weight", (d, S_CAPD), "w"), ("cap_embedder.1.bias", (d,), "b"),
            ("x_embedder.weight", (d, S_PD), "w"), ("x_embedder.bias", (d,), "b"),
            ("t_embedder.mlp.0.weight", (S_THID, S_FREQ), "w"), ("t_embedder.mlp.0.bias", (S_THID,), "b"),
            ("t_embedder.mlp.2.weight", (d, S_THID), "w"), ("t_embedder.mlp.2.bias", (d,), "b")]
    for l in range(S_BLOCKS):
        p = f"layers.{l}."
        spec += [(p + "attention.to_q.weight", (d, d), "w"), (p + "attention.to_k.weight", (d, d), "w"),
                 (p + "attention.to_v.weight", (d, d), "w"), (p + "attention.to_out.0.weight", (d, d), "w"),
                 (p + "attention.norm_q.weight", (S_HD,), "g"), (p + "attention.norm_k.weight", (S_HD,), "g"),
                 (p + "feed_forward.w1.weight", (hid, d), "w"), (p + "feed_forward.w2.weight", (d, hid), "w"),
                 (p + "feed_forward.w3.weight", (hid, d), "w"),
                 (p + "attention_norm1.weight", (d,), "g"), (p + "attention_norm2.weight", (d,), "g"),
                 (p + "ffn_norm1.weight", (d,), "g"), (p + "ffn_norm2.weight", (d,), "g"),
                 (p + "adaLN_modulation.0.weight", (4 * d, d), "w"), (p + "adaLN_modulation.0.bias", (4 * d,), "b")]
    spec += [("final_layer.adaLN_modulation.1.weight", (d, d), "w"), ("final_layer.adaLN_modulation.1.bias", (d,), "b"),
             ("final_layer.linear.weight", (S_PD, d), "w"), ("final_layer.linear.bias", (S_PD,), "b")]
    return spec


def small_weights():
    g = bookrng.generator(SMALL)
    out = {}
    for name, shape, kind in small_spec():
        if kind == "w":
            scale = 1.0 / math.sqrt(shape[1])
            out[name] = [[scale * g.normal() for _ in range(shape[1])] for _ in range(shape[0])]
        elif kind == "b":
            out[name] = [0.1 * g.normal() for _ in range(shape[0])]
        else:
            out[name] = [1.0 + 0.1 * g.normal() for _ in range(shape[0])]
    return out


def small_input():
    g = bookrng.generator(SMALL_INPUT)
    latent = [[[_round1(g.normal()) for _ in range(S_W)] for _ in range(S_H)] for _ in range(S_C)]
    cap = [[_round1(g.normal()) for _ in range(S_CAPD)] for _ in range(S_CAP)]
    return latent, cap


def _patchify(lat):
    """[C][H][W] to tokens in row-major patch order, each token [pH][pW][C] with the channel innermost."""
    toks = []
    for hi in range(S_H // S_P):
        for wi in range(S_W // S_P):
            t = []
            for b in range(S_P):
                for e in range(S_P):
                    for c in range(S_C):
                        t.append(lat[c][hi * S_P + b][wi * S_P + e])
            toks.append(t)
    return toks


def _unpatchify(toks):
    lat = [[[0.0] * S_W for _ in range(S_H)] for _ in range(S_C)]
    gw = S_W // S_P
    for i, t in enumerate(toks):
        hi, wi = divmod(i, gw)
        for b in range(S_P):
            for e in range(S_P):
                for c in range(S_C):
                    lat[c][hi * S_P + b][wi * S_P + e] = t[(b * S_P + e) * S_C + c]
    return lat


def _rope_tables(ids):
    """cos and sin per token for the pairs of a head: the three axes (width 4 each, two pairs each) in order, pair k of an axis of
    width d at rate theta^(-2k/d) times the token's id on that axis."""
    cos_t, sin_t = [], []
    for tid in ids:
        cr, sr = [], []
        for a, d in enumerate(S_AXES):
            for k in range(d // 2):
                ang = tid[a] * S_THETA ** (-2.0 * k / d)
                cr.append(math.cos(ang))
                sr.append(math.sin(ang))
        cos_t.append(cr)
        sin_t.append(sr)
    return cos_t, sin_t


def _rotate(vec, c, s):
    out = vec[:]
    for j in range(len(c)):
        e, o = vec[2 * j], vec[2 * j + 1]
        out[2 * j] = e * c[j] - o * s[j]
        out[2 * j + 1] = e * s[j] + o * c[j]
    return out


def _block(W, p, x, cvec, cos_t, sin_t, fold=True, rope=True, gate_zero=False):
    """One Z-Image block on the rows x. With fold=True the scales and gates multiply the norm gains (the way brain applies them);
    with fold=False the norms are applied with their own gains and the scale and gate multiply the results."""
    d = S_DIM
    m = [a + b for a, b in zip(_linear_rows([cvec], W[p + "adaLN_modulation.0.weight"])[0], W[p + "adaLN_modulation.0.bias"])]
    if gate_zero:
        m[d:2 * d] = [0.0] * d
        m[3 * d:4 * d] = [0.0] * d
    s_msa, g_msa, s_mlp, g_mlp = m[0:d], [math.tanh(v) for v in m[d:2 * d]], m[2 * d:3 * d], [math.tanh(v) for v in m[3 * d:4 * d]]
    an1, an2 = W[p + "attention_norm1.weight"], W[p + "attention_norm2.weight"]
    fn1, fn2 = W[p + "ffn_norm1.weight"], W[p + "ffn_norm2.weight"]
    if fold:
        n1 = _rms_rows(x, [w * (1.0 + s) for w, s in zip(an1, s_msa)], S_EPS)
    else:
        n1 = [[(1.0 + s) * v for s, v in zip(s_msa, row)] for row in _rms_rows(x, an1, S_EPS)]
    q = _linear_rows(n1, W[p + "attention.to_q.weight"])
    k = _linear_rows(n1, W[p + "attention.to_k.weight"])
    v = _linear_rows(n1, W[p + "attention.to_v.weight"])
    nq, nk = W[p + "attention.norm_q.weight"], W[p + "attention.norm_k.weight"]
    n = len(x)
    ctx = [[0.0] * d for _ in range(n)]
    for h in range(S_HEADS):
        sl = slice(h * S_HD, (h + 1) * S_HD)
        qh = _rms_rows([r[sl] for r in q], nq, S_EPS)
        kh = _rms_rows([r[sl] for r in k], nk, S_EPS)
        if rope:
            qh = [_rotate(r, cos_t[i], sin_t[i]) for i, r in enumerate(qh)]
            kh = [_rotate(r, cos_t[i], sin_t[i]) for i, r in enumerate(kh)]
        vh = [r[sl] for r in v]
        for i in range(n):
            wts = _softmax([sum(a * b for a, b in zip(qh[i], kh[j])) / math.sqrt(S_HD) for j in range(n)])
            for c in range(S_HD):
                ctx[i][h * S_HD + c] = sum(wts[j] * vh[j][c] for j in range(n))
    a = _linear_rows(ctx, W[p + "attention.to_out.0.weight"])
    if fold:
        post = _rms_rows(a, [w * g for w, g in zip(an2, g_msa)], S_EPS)
    else:
        post = [[g * v_ for g, v_ in zip(g_msa, row)] for row in _rms_rows(a, an2, S_EPS)]
    x1 = [[xi + pi for xi, pi in zip(xr, pr)] for xr, pr in zip(x, post)]
    if fold:
        f1 = _rms_rows(x1, [w * (1.0 + s) for w, s in zip(fn1, s_mlp)], S_EPS)
    else:
        f1 = [[(1.0 + s) * v_ for s, v_ in zip(s_mlp, row)] for row in _rms_rows(x1, fn1, S_EPS)]
    g1 = _linear_rows(f1, W[p + "feed_forward.w1.weight"])
    u1 = _linear_rows(f1, W[p + "feed_forward.w3.weight"])
    hid = [[_silu(a_) * b_ for a_, b_ in zip(ga, ua)] for ga, ua in zip(g1, u1)]
    ff = _linear_rows(hid, W[p + "feed_forward.w2.weight"])
    if fold:
        post2 = _rms_rows(ff, [w * g for w, g in zip(fn2, g_mlp)], S_EPS)
    else:
        post2 = [[g * v_ for g, v_ in zip(g_mlp, row)] for row in _rms_rows(ff, fn2, S_EPS)]
    return [[a_ + b_ for a_, b_ in zip(xr, pr)] for xr, pr in zip(x1, post2)]


def small_denoiser(W, latent, cap, sigma, fold=True, rope=True, gate_zero=False, return_stream=False):
    """The small denoiser: the latent [C][H][W] and the caption features [S_CAP][S_CAPD] at noise fraction sigma, returning the
    predicted latent (the output of the network, not negated). Steps:
    1. c = Linear(SiLU(Linear(embed(1000 (1 - sigma), 16))))  (the timestep MLP, 16 -> 32 -> 24);
    2. image tokens: patchify, Linear(8 -> 24); caption tokens: RMS norm with gain, Linear(5 -> 24); the joint sequence is the 4
       image tokens then the 2 caption tokens, with the position ids of `token_ids`;
    3. S_BLOCKS blocks, each reading all six tokens (see _block);
    4. the final layer on every token: parameter-free layer norm, times (1 + Linear(SiLU(c))), Linear(24 -> 8); the image tokens
       are unpatchified."""
    t_arg = 1000.0 * (1.0 - sigma)
    emb = embed(t_arg, S_FREQ)
    h0 = [_silu(v) for v in _linear_rows([emb], W["t_embedder.mlp.0.weight"], W["t_embedder.mlp.0.bias"])[0]]
    cvec = _linear_rows([h0], W["t_embedder.mlp.2.weight"], W["t_embedder.mlp.2.bias"])[0]
    img = _linear_rows(_patchify(latent), W["x_embedder.weight"], W["x_embedder.bias"])
    cn = _rms_rows(cap, W["cap_embedder.0.weight"], S_EPS)
    ct = _linear_rows(cn, W["cap_embedder.1.weight"], W["cap_embedder.1.bias"])
    ids = []
    gh, gw = S_H // S_P, S_W // S_P
    for hi in range(gh):
        for wi in range(gw):
            ids.append((S_CAP + 1, hi, wi))
    for i in range(S_CAP):
        ids.append((1 + i, 0, 0))
    cos_t, sin_t = _rope_tables(ids)
    x = img + ct
    start = [r[:] for r in x]
    for l in range(S_BLOCKS):
        x = _block(W, f"layers.{l}.", x, cvec, cos_t, sin_t, fold=fold, rope=rope, gate_zero=gate_zero)
    if return_stream:
        return start, x
    sc = [1.0 + v for v in _linear_rows([[_silu(v) for v in cvec]], W["final_layer.adaLN_modulation.1.weight"],
                                        W["final_layer.adaLN_modulation.1.bias"])[0]]
    normed = _ln_rows(x, S_EPS_FINAL)
    scaled = [[v * s for v, s in zip(row, sc)] for row in normed]
    outp = _linear_rows(scaled, W["final_layer.linear.weight"], W["final_layer.linear.bias"])
    return _unpatchify(outp[: gh * gw])


def small_setup():
    spec = small_spec()
    total = sum(math.prod(s) for _, s, _ in spec)
    rows = []
    d = S_DIM
    groups = [
        ("cap_embedder.0.weight", "gain: 1 + 0.1 normal()", f"[{S_CAPD}]"),
        ("cap_embedder.1.weight, .bias", "weight: normal()/sqrt(5); bias: 0.1 normal()", f"[{d}, {S_CAPD}], [{d}]"),
        ("x_embedder.weight, .bias", "weight: normal()/sqrt(8); bias: 0.1 normal()", f"[{d}, {S_PD}], [{d}]"),
        ("t_embedder.mlp.0.weight, .bias", "weight: normal()/sqrt(16); bias: 0.1 normal()", f"[{S_THID}, {S_FREQ}], [{S_THID}]"),
        ("t_embedder.mlp.2.weight, .bias", "weight: normal()/sqrt(32); bias: 0.1 normal()", f"[{d}, {S_THID}], [{d}]"),
        ("layers.l.attention.to_q, to_k, to_v, to_out.0 (weights)", "normal()/sqrt(24), in this order", f"4 x [{d}, {d}]"),
        ("layers.l.attention.norm_q, norm_k", "gain: 1 + 0.1 normal()", f"2 x [{S_HD}]"),
        ("layers.l.feed_forward.w1, w2, w3 (weights)", "normal()/sqrt(fan_in): 24, 64, 24", f"[{S_HID}, {d}], [{d}, {S_HID}], [{S_HID}, {d}]"),
        ("layers.l.attention_norm1, attention_norm2, ffn_norm1, ffn_norm2", "gain: 1 + 0.1 normal(), in this order", f"4 x [{d}]"),
        ("layers.l.adaLN_modulation.0.weight, .bias", "weight: normal()/sqrt(24); bias: 0.1 normal()", f"[{4 * d}, {d}], [{4 * d}]"),
        ("final_layer.adaLN_modulation.1.weight, .bias", "weight: normal()/sqrt(24); bias: 0.1 normal()", f"[{d}, {d}], [{d}]"),
        ("final_layer.linear.weight, .bias", "weight: normal()/sqrt(24); bias: 0.1 normal()", f"[{S_PD}, {d}], [{S_PD}]"),
    ]
    for a, b, c in groups:
        rows.append([a, b, c])
    latent, cap = small_input()
    lat_lines = []
    for c in range(S_C):
        for r in latent[c]:
            lat_lines.append("(" + ", ".join(f"{v + 0.0:g}" for v in r) + ")")
    return (
        f"Weights: the book's SplitMix64, seed 0x{bookrng.seed_of(SMALL):016x} (`seed_of` of the name `{SMALL}`), one stream for all "
        "tensors, drawn tensor by tensor in the order below (the two blocks l = 0 and l = 1 in turn, each in the order listed; a matrix row by row, so that its "
        "`[out, in]` entry (o, i) is draw number o*in + i of the matrix; `normal()` is the generator's Box-Muller draw). "
        # perf-number: hex prefix of a printed seed, not a multiplier
        f"Inputs: seed 0x{bookrng.seed_of(SMALL_INPUT):016x} (`{SMALL_INPUT}`): the 32 entries of the latent (2 channels of 4 rows of 4 "
        "columns, channel slowest), then the 10 caption features (2 tokens of 5), each `round(normal(), 1)`. Width 24, 2 heads of width "
        "12, feed-forward width 64, conditioning width 24, timestep embedding of dimension 16, 2 blocks, patches of 2 by 2 "
        f"(8 values), RMS-norm epsilon 1e-5 (final layer norm 1e-6). {len(spec)} tensors, {_int(total)} parameters.\n\n"
        + _table(["tensor", "recipe", "shape"], rows, "lll")
        + "\n\nLatent, channel 0 then channel 1, rows top to bottom: " + "; ".join(lat_lines[: S_H]) + " | " + "; ".join(lat_lines[S_H:])
        + ". Caption features: " + "; ".join("(" + ", ".join(f"{v + 0.0:g}" for v in r) + ")" for r in cap) + "."
    )


def _maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _flat(lat):
    return [[v for ch in lat for row in ch for v in row]]


def small_outputs():
    W = small_weights()
    latent, cap = small_input()
    rows = []
    outs = {}
    for s in (0.8, 0.2):
        o = small_denoiser(W, latent, cap, s)
        outs[s] = o
        for c in range(S_C):
            for r in range(S_H):
                rows.append([f"{s:g}", str(c), str(r)] + [_f(v, 6) for v in o[c][r]])
    return (
        "The output of the small denoiser at sigma = 0.8 and at sigma = 0.2 (time argument 1000 (1 - sigma) = 200 and 800), for the "
        "inputs above: the predicted latent, channel and row, four columns. Library cos, sin, exp, tanh; the values are those of "
        "double precision and are printed to 6 decimals.\n\n"
        + _table(["sigma", "channel", "row", "col 0", "col 1", "col 2", "col 3"], rows, "lllrrrr")
    )


def small_checks():
    W = small_weights()
    latent, cap = small_input()
    o_a = small_denoiser(W, latent, cap, 0.8)
    o_b = small_denoiser(W, latent, cap, 0.2)
    rows = []
    # 1. the gates at zero: every block is the identity on the joint sequence
    start, end = small_denoiser(W, latent, cap, 0.8, gate_zero=True, return_stream=True)
    ident = _maxdiff(start, end)
    rows.append(["both gates of every block set to 0", "largest change of the joint sequence over 2 blocks", f"{ident:g}"])
    # 2. the same blocks with real gates
    start, end = small_denoiser(W, latent, cap, 0.8, return_stream=True)
    rms = lambda M: math.sqrt(sum(v * v for r in M for v in r) / sum(len(r) for r in M))  # noqa: E731
    rel = rms([[a - b for a, b in zip(ra, rb)] for ra, rb in zip(start, end)]) / rms(start)
    rows.append(["trained-like gates (the weights above)", "rms change of the sequence / rms of the sequence", f"{rel:.4f}"])
    # 3. fold equals unfold
    unf = small_denoiser(W, latent, cap, 0.8, fold=False)
    fdiff = _maxdiff(_flat(o_a), _flat(unf))
    rows.append(["scale and gate folded into the norm gains vs applied after the norms", "largest difference of the output",
                 f"below {max(1e-12, math.ceil(fdiff * 1e13) / 1e13):.0e}"])
    # 4. the clock matters
    rows.append(["sigma = 0.8 vs sigma = 0.2", "largest difference of the output", f"{_maxdiff(_flat(o_a), _flat(o_b)):.4f}"])
    # 5. permuting the image tokens without rotary position
    s_, e_ = small_denoiser(W, latent, cap, 0.8, rope=False, return_stream=True)
    lat_p = [[row[:] for row in ch] for ch in latent]
    for ch in lat_p:                                         # swap the two patch columns of both patch rows
        for r in range(S_H):
            ch[r] = ch[r][2:4] + ch[r][0:2]
    s2, e2 = small_denoiser(W, lat_p, cap, 0.8, rope=False, return_stream=True)
    perm = [1, 0, 3, 2, 4, 5]
    pd = _maxdiff([e_[i] for i in perm], e2)
    rows.append(["no rotary position, the two patch columns exchanged", "largest difference between the permuted outputs",
                 f"below {max(1e-12, math.ceil(pd * 1e13) / 1e13):.0e}"])
    s3, e3 = small_denoiser(W, lat_p, cap, 0.8, rope=True, return_stream=True)
    s4, e4 = small_denoiser(W, latent, cap, 0.8, rope=True, return_stream=True)
    rows.append(["with rotary position, the same exchange", "largest difference between the permuted outputs",
                 f"{_maxdiff([e4[i] for i in perm], e3):.4f}"])
    # 6. information flow in the joint sequence
    cap2 = [r[:] for r in cap]
    cap2[0] = [v + 0.5 for v in cap2[0]]
    oc = small_denoiser(W, latent, cap2, 0.8)
    rows.append(["0.5 added to caption token 0", "largest change of the predicted latent", f"{_maxdiff(_flat(o_a), _flat(oc)):.4f}"])
    lat2 = [[row[:] for row in ch] for ch in latent]
    lat2[0][0][0] += 0.5
    _, e5 = small_denoiser(W, lat2, cap, 0.8, return_stream=True)
    _, e6 = small_denoiser(W, latent, cap, 0.8, return_stream=True)
    rows.append(["0.5 added to latent entry (0, 0, 0)", "largest change of the two caption tokens after the blocks",
                 f"{_maxdiff(e5[4:], e6[4:]):.4f}"])
    return (
        "Checks on the small denoiser (inputs and weights as above). The sequence is the 6 tokens of the joint stream, the 4 image tokens and then the 2 "
        "caption tokens. Every comparison of tokens ('after the blocks', the two exchange rows, the caption-token row) is on that 6-token sequence at "
        "sigma = 0.8 after the two blocks, before the final layer, not on the predicted latent; the exchange of the two patch columns permutes the "
        "image tokens 0 <-> 1 and 2 <-> 3 (the caption tokens stay), and the outputs are compared after permuting them the same way. The rows that "
        "say 'predicted latent' or 'output' compare the final layer's output.\n\n"
        + _table(["change", "measured", "result"], rows, "lll")
    )
