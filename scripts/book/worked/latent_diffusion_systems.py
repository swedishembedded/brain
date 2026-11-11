# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Latent Diffusion Systems End to End".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: latent_diffusion_systems.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures. The earlier worked modules are imported, not copied:
`text_conditioning` (the work of the text encoders and of FLUX.1's denoiser), `denoiser_architectures` (the FLUX.2 parameter and pass
counts), `flow_matching` (the sigma schedules) and `vae` (the latent packing and the decoder's parameters).

The first half is the accounting of brain's real pipelines, derived from their configurations:

* the multiply-adds of the autoencoders' decode and encode, level by level;
* the work of every stage of one image, with and without a reference image;
* the schedules of FLUX.1 and FLUX.2 and the compressed schedule of an image-to-image run;
* the bytes each part occupies on a device (the planner's figure and the layout actually uploaded), the arrangements that fit a card,
  and the bytes that move when weights are evicted or streamed;
* the windows of a tiled canvas and what they cost.

The second half is a toy pipeline, small enough to check by hand and exact (nothing in it is trained): a text encoder that is a table of
rows, a linear autoencoder with a null space, a latent packed and standardized as FLUX.2's is, an exact denoiser of a mixture of three
clusters with and without the prompt, guidance, an Euler sampler on FLUX.1's schedule rule, an image-to-image start, a masked blend in
latent space and a composite in pixel space. Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every
stream name is printed by the chapter. Nothing is trained, so every number is printed to the digits shown and another implementation
agrees up to the rounding of the library's exp, log and sqrt.

Swedish Embedded AB implements solutions for image generators whose stages are counted, placed on devices and checked one by one. If
your team needs expertise in latent diffusion systems and their memory budgets, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import denoiser_architectures as da
import diffusion
import flow_matching as fl
import text_conditioning as tc
import vae

_table, _seed = diffusion._table, diffusion._seed

GIB = 2 ** 30


def _int(n):
    return f"{n:,}"


def _gib(b, d=2):
    return f"{b / GIB:.{d}f}"


def _pct(x, d=1):
    """A share in per cent, as a plain number (the column header carries the unit)."""
    return f"{100.0 * x:.{d}f}"


# ------------------------------------------------------------------ the autoencoders, counted

CHANNELS = (128, 256, 512, 512)
VAES = {"FLUX.1 (16 channels)": dict(zc=16, quant=False), "FLUX.2 (32 channels)": dict(zc=32, quant=True)}


def _conv(ci, co, k, h, w):
    """Multiply-adds of a k x k convolution, counted as two operations, on an h x w output."""
    return 2 * ci * co * k * k * h * w


def _resnet(ci, co, h, w):
    return _conv(ci, co, 3, h, w) + _conv(co, co, 3, h, w) + (_conv(ci, co, 1, h, w) if ci != co else 0)


def _attn(c, h, w):
    """The mid-block attention: a 1 x 1 convolution to q, k, v, the scores and their average (4 T^2 c) and the 1 x 1 output."""
    t = h * w
    return _conv(c, 3 * c, 1, h, w) + 4 * t * t * c + _conv(c, c, 1, h, w)


def decode_flops(zc, quant, side, channels=CHANNELS):
    """(total, per level, mid-block attention) in FLOPs of the decoder at an output of side x side pixels: the optional 1 x 1
    mixing convolution, the 3 x 3 convolution in, the mid block (resnet, attention, resnet) at the latent size, four up blocks of three
    resnets each at 1, 2, 4 and 8 times the latent side and, after each but the last, a nearest-neighbour doubling and a 3 x 3
    convolution at the new size, and the 3 x 3 convolution to three channels."""
    h = w = side // 8
    rc = list(reversed(channels))
    total = _conv(zc, zc, 1, h, w) if quant else 0
    total += _conv(zc, rc[0], 3, h, w)
    mid = 2 * _resnet(rc[0], rc[0], h, w) + _attn(rc[0], h, w)
    total += mid
    prev, levels = rc[0], []
    for i, c in enumerate(rc):
        f = sum(_resnet(prev if r == 0 else c, c, h, w) for r in range(3))
        if i < len(rc) - 1:
            h, w = 2 * h, 2 * w
            f += _conv(c, c, 3, h, w)
        levels.append(f)
        total += f
        prev = c
    total += _conv(prev, 3, 3, h, w)
    return total, levels, _attn(rc[0], side // 8, side // 8)


def encode_flops(zc, quant, side, channels=CHANNELS):
    """The encoder of the same autoencoder on a side x side image: 3 x 3 convolution in, two resnets per level at 1, 1/2, 1/4, 1/8
    of the image side with a stride-2 3 x 3 convolution (counted on its output) after all but the last, the mid block, and the 3 x 3
    convolution to the moments (and a 1 x 1 on them if the autoencoder has one)."""
    h = w = side
    total = _conv(3, channels[0], 3, h, w)
    prev = channels[0]
    for i, c in enumerate(channels):
        for r in range(2):
            total += _resnet(prev if r == 0 else c, c, h, w)
            prev = c
        if i < len(channels) - 1:
            h, w = h // 2, w // 2
            total += _conv(c, c, 3, h, w)
    total += 2 * _resnet(prev, prev, h, w) + _attn(prev, h, w)
    total += _conv(prev, 2 * zc, 3, h, w)
    if quant:
        total += _conv(2 * zc, 2 * zc, 1, h, w)
    return total


def _cp(ci, co, k):
    return co * ci * k * k + co


def _rp(ci, co):
    return 2 * ci + _cp(ci, co, 3) + 2 * co + _cp(co, co, 3) + (_cp(ci, co, 1) if ci != co else 0)


def _ap(c):
    return 2 * c + 4 * _cp(c, c, 1)


def decoder_params(zc, quant, channels=CHANNELS):
    rc = list(reversed(channels))
    n = (_cp(zc, zc, 1) if quant else 0) + _cp(zc, rc[0], 3) + 2 * _rp(rc[0], rc[0]) + _ap(rc[0])
    prev = rc[0]
    for i, c in enumerate(rc):
        n += sum(_rp(prev if r == 0 else c, c) for r in range(3))
        if i < len(rc) - 1:
            n += _cp(c, c, 3)
        prev = c
    return n + 2 * prev + _cp(prev, 3, 3)


def encoder_params(zc, quant, channels=CHANNELS):
    n = _cp(3, channels[0], 3)
    prev = channels[0]
    for i, c in enumerate(channels):
        for r in range(2):
            n += _rp(prev if r == 0 else c, c)
            prev = c
        if i < len(channels) - 1:
            n += _cp(c, c, 3)
    n += 2 * _rp(prev, prev) + _ap(prev) + 2 * prev + _cp(prev, 2 * zc, 3)
    return n + (_cp(2 * zc, 2 * zc, 1) if quant else 0)


assert decoder_params(32, True) == vae.decoder_parameters()


def vae_work():
    rows = []
    for name, v in VAES.items():
        for side in (512, 1024, 2048):
            if side == 2048 and not v["quant"]:
                continue
            f, _, a = decode_flops(v["zc"], v["quant"], side)
            rows.append([name, f"decode {side} x {side}", _int(decoder_params(v["zc"], v["quant"])), f"{f / 1e12:.3f}", _pct(a / f)])
        for side in (512, 1024):
            f = encode_flops(v["zc"], v["quant"], side)
            rows.append([name, f"encode {side} x {side}", _int(encoder_params(v["zc"], v["quant"])), f"{f / 1e12:.3f}", ""])
    t1 = _table(["autoencoder", "direction and image", "parameters", "TFLOP", "mid-block attention, share (%)"], rows, "llrrr")
    _, lev, _ = decode_flops(32, True, 1024)
    tot, _, _ = decode_flops(32, True, 1024)
    side = 128
    lrows = []
    for i, f in enumerate(lev):
        lrows.append([f"up block {i}", f"{CHANNELS[::-1][i]}", f"{side * 2 ** i} x {side * 2 ** i}" if i == 0 else f"{side * 2 ** i} x {side * 2 ** i}",
                      f"{f / 1e12:.3f}", _pct(f / tot)])
    t2 = _table(["part of the 1024 x 1024 decode", "channels", "resnets at", "TFLOP", "share of the decode (%)"], lrows, "lrrrr")
    return (
        "The two autoencoders of brain from their channel schedule (128, 256, 512, 512), two residual blocks per level in the encoder and "
        "three in the decoder, a mid block with one attention layer, and a multiply-add counted as two operations; norms, activations and "
        "the nearest-neighbour doubling are left out. The FLUX.1 autoencoder has 16 latent channels and no 1 x 1 convolutions on the "
        "latent; FLUX.2's has 32 and both. The last column is the share of the decode spent in the mid-block attention over the "
        "(side/8)^2 positions of the latent.\n\n" + t1 +
        "\n\nWhere the 1024 x 1024 decode of FLUX.2's autoencoder spends its operations (up block i runs three resnets at the size "
        "shown; blocks 0 to 2 end with a doubling and a 3 x 3 convolution at the next size, counted in the block that ends with them; "
        "the rest of the total is the convolution in, the mid block and the convolution out).\n\n" + t2)


# ------------------------------------------------------------------ the work of each stage of one image

FLUX2 = da.FLUX2
D_HEAD = 128


def flux2_pass_tokens(c, n_img, txt=512):
    """FLOPs of one FLUX.2 forward pass over `txt` text rows and `n_img` image rows (the generated image and any reference images
    together, which join the image stream), by the rule of `denoiser_architectures.flux2_pass`."""
    d = c["d"]
    mlp = d * c["mlp_ratio"]
    n = n_img + txt
    dbl = 2 * n_img * (4 * d * d + 3 * mlp * d) + 2 * txt * (4 * d * d + 3 * mlp * d)
    sgl = 2 * n * ((3 * d + 2 * mlp) * d + d * (d + mlp))
    lin = c["double"] * dbl + c["single"] * sgl
    att = (c["double"] + c["single"]) * 4 * n * n * d
    emb = 2 * n_img * c["in_ch"] * d + 2 * txt * c["ctx"] * d + 2 * n_img * d * c["in_ch"]
    return lin + att + emb


for _k, _c in FLUX2.items():
    _l, _a = da.flux2_pass(_c, 1024)
    assert flux2_pass_tokens(_c, 4096) == _l + _a


def stage_data():
    """For FLUX.1 dev at 50 steps and FLUX.2 klein at 4: (name, encode FLOPs, passes, FLOPs per pass, decode FLOPs)."""
    work = {n: (fe, fd) for n, _, _, fe, fd in tc.encoder_work_data()}
    out = []
    d16, _, _ = decode_flops(16, False, 1024)
    d32, _, _ = decode_flops(32, True, 1024)
    fe, fd = work["FLUX.1 dev, 50 steps"]
    out.append(("FLUX.1 dev, 50 steps", fe, 50, fd / 50, d16))
    for k, name in (("klein-4b", "FLUX.2 klein-4b, 4 steps"), ("klein-9b", "FLUX.2 klein-9b, 4 steps")):
        fe, fd = work[name]
        out.append((name, fe, 4, fd / 4, d32))
    fe, fd = work["FLUX.2 klein-4b, 4 steps"]
    out.append(("FLUX.2 base-4b, 50 steps with guidance", fe, 100, fd / 4, d32))
    return out


def stage_work():
    rows = []
    for name, fe, passes, fp, fdec in stage_data():
        den = passes * fp
        tot = fe + den + fdec
        rows.append([name, f"{fe / 1e12:.2f}", f"{den / 1e12:.1f}", f"{fdec / 1e12:.2f}", f"{tot / 1e12:.1f}", _pct(fe / tot, 2),
                     _pct(den / tot, 2), _pct(fdec / tot, 2)])
    return (
        "The work of one 1024 x 1024 image, stage by stage, at each pipeline's default pass count: the text encoders (the table of "
        "Chapter 9, `text_conditioning.encoder_work_data`), all denoising passes (Chapter 8's pass counts; the base model runs two "
        "passes a step for guidance) and the decode of the autoencoder of this pipeline (`latent_diffusion_systems.vae_work`). TFLOP "
        "is 10^12 operations, a multiply-add counted as two.\n\n"
        + _table(["pipeline", "text, TFLOP", "denoise, TFLOP", "decode, TFLOP", "total, TFLOP", "text (%)", "denoise (%)", "decode (%)"], rows, "lrrrrrrr"))


def reference_cost():
    rows = []
    e32 = encode_flops(32, True, 1024)
    for k in ("klein-4b", "klein-9b"):
        c = FLUX2[k]
        plain = 4 * flux2_pass_tokens(c, 4096)
        ref = 4 * flux2_pass_tokens(c, 8192)
        dec, _, _ = decode_flops(32, True, 1024)
        for label, enc_n, den in (("text to image", 0, plain), ("edit: one 1024 x 1024 reference, encoded once", 1, ref),
                                  ("strength below 1: the same image encoded twice", 2, ref)):
            rows.append([k, label, f"{enc_n * e32 / 1e12:.2f}", f"{den / 1e12:.1f}", f"{dec / 1e12:.2f}",
                         f"{(enc_n * e32 + den + dec) / 1e12:.1f}"])
    return (
        "The denoising and autoencoding work of one 1024 x 1024 FLUX.2 image at four steps, with 512 text rows and no cost counted for "
        "the text encoder. A reference joins the image stream, so a 1024 x 1024 reference doubles the image rows from 4,096 to 8,192 in "
        "every pass. The encoder runs once per encoded image, at 1024 x 1024 (`latent_diffusion_systems.vae_work`).\n\n"
        + _table(["variant", "request", "VAE encode, TFLOP", "denoise, TFLOP", "VAE decode, TFLOP", "total, TFLOP"], rows, "llrrrr"))


# ------------------------------------------------------------------ schedules

def mu_flux1(tokens):
    """FLUX.1's mu: the straight line through (256 tokens, 0.5) and (4096 tokens, 1.15)."""
    m = (1.15 - 0.5) / (4096 - 256)
    return tokens * m + (0.5 - m * 256)


def sigmas_flux1(steps, tokens, shifted=True):
    base = [1.0 - i / steps for i in range(steps + 1)]
    if not shifted:
        return base
    mu = mu_flux1(tokens)
    return [fl.shift_exp(mu, s) for s in base]


def schedule_table():
    rows = []
    t = 4096
    cases = (
        ("FLUX.1 schnell, 4 steps (no shift)", sigmas_flux1(4, t, False)),
        ("FLUX.1 dev, 4 steps (mu = 1.15)", sigmas_flux1(4, t)),
        ("FLUX.2 klein, 4 steps (mu = %.4f)" % fl.empirical_mu(t, 4), fl.klein_sigmas(4, t)),
        ("klein, strength 0.6", [0.6 * s for s in fl.klein_sigmas(4, t)]),
        ("klein, strength 0.3", [0.3 * s for s in fl.klein_sigmas(4, t)]),
    )
    for name, s in cases:
        rows.append([name] + [f"{x:.4f}" for x in s])
    t1 = _table(["schedule at 4,096 tokens", "sigma 0", "sigma 1", "sigma 2", "sigma 3", "sigma 4"], rows, "lrrrrr")
    s50 = sigmas_flux1(50, t)
    idx = (0, 1, 5, 10, 20, 30, 40, 45, 49, 50)
    t2 = _table(["step i"] + [str(i) for i in idx], [["FLUX.1 dev, 50 steps: sigma"] + [f"{s50[i]:.4f}" for i in idx]], "l" + "r" * len(idx))
    return (
        "Noise fractions at 4,096 image tokens (a 1024 x 1024 canvas). FLUX.1: linspace(1, 0, steps + 1), shifted by "
        "e^mu / (e^mu + 1/sigma - 1) with mu the line through (256, 0.5) and (4096, 1.15), so mu = 1.15 here; schnell is not shifted. FLUX.2: "
        "`flow_matching.klein_sigmas`. An image-to-image run with strength s takes the FLUX.2 schedule times s.\n\n" + t1 +
        "\n\nThe 51 values of FLUX.1 dev's default schedule, at the steps listed.\n\n" + t2)


# ------------------------------------------------------------------ bytes on a device

def flux2_manifest(c, txt_len=512):
    d, mlp, hd = c["d"], c["d"] * c["mlp_ratio"], D_HEAD
    v = [("img_in.weight", (d, c["in_ch"])), ("txt_in.weight", (d, c["ctx"])), ("time_in.in_layer.weight", (d, 256)),
         ("time_in.out_layer.weight", (d, d)), ("double_stream_modulation_img.lin.weight", (6 * d, d)),
         ("double_stream_modulation_txt.lin.weight", (6 * d, d)), ("single_stream_modulation.lin.weight", (3 * d, d))]
    for n in range(c["double"]):
        for s in ("img", "txt"):
            p = f"double_blocks.{n}.{s}"
            v += [(p + "_attn.qkv.weight", (3 * d, d)), (p + "_attn.norm.query_norm.scale", (hd,)), (p + "_attn.norm.key_norm.scale", (hd,)),
                  (p + "_attn.proj.weight", (d, d)), (p + "_mlp.0.weight", (2 * mlp, d)), (p + "_mlp.2.weight", (d, mlp))]
    for n in range(c["single"]):
        p = f"single_blocks.{n}"
        v += [(p + ".linear1.weight", (3 * d + 2 * mlp, d)), (p + ".linear2.weight", (d, d + mlp)),
              (p + ".norm.query_norm.scale", (hd,)), (p + ".norm.key_norm.scale", (hd,))]
    v += [("final_layer.adaLN_modulation.1.weight", (2 * d, d)), ("final_layer.linear.weight", (c["in_ch"], d))]
    return v


HOST = {"time_in.in_layer.weight", "time_in.out_layer.weight", "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.lin.weight", "single_stream_modulation.lin.weight", "final_layer.adaLN_modulation.1.weight"}


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


assert sum(_numel(s) for _, s in flux2_manifest(FLUX2["klein-9b"])) == da.flux2_params(FLUX2["klein-9b"])["total"]


def flux2_weight_bytes(c, tier, planner):
    """Bytes of the DiT weights. `planner`: what `flux2::pipeline::dit_bytes` summed before commit 38e8b2b7a (every two-dimensional
    tensor at one byte under int8, every other tensor and every f32 tensor at four). Otherwise the layout the constructor uploads, which
    `dit_bytes` has priced since: the six tensors of the clock and the
    modulations stay on the host; under int8 the boundary linears, the double blocks' second feed-forward matrix and every vector stay
    f32, and each other matrix is 32 bytes of int8 plus one 32-bit scale per 32 weights (36 bytes per 32 weights)."""
    total = 0
    for name, shape in flux2_manifest(c):
        n = _numel(shape)
        if planner:
            total += n * (1 if tier == "int8" and len(shape) == 2 else 4)
            continue
        if name in HOST:
            continue
        keep = (name in ("img_in.weight", "txt_in.weight", "final_layer.linear.weight") or len(shape) == 1 or
                (name.startswith("double_blocks.") and name.endswith("_mlp.2.weight")))
        total += n * 4 if (tier == "f32" or keep) else n // 32 * 36
    return total


def flux2_scratch_bytes(c, tier, n_joint, txt=512, planner=False):
    """The activation scratch the planner prices for one joint sequence (`dit_scratch_bytes` of the crate): 16 words of width d,
    three of the feed-forward width, two latent buffers and one head width per row, the context, 17 d for the clock, and the
    attention's scores and probabilities (two words under int8, which runs fused, else 2 heads n^2); under int8 also one byte-packed
    copy of the hidden and the feed-forward row and a 32-bit scale per 32 values of the wider (before commit 38e8b2b7a, `planner`, one
    scale per row)."""
    d, mlp, heads = c["d"], c["d"] * c["mlp_ratio"], c["heads"]
    attn = 2 if tier == "int8" else 2 * heads * n_joint * n_joint
    words = n_joint * (16 * d + 3 * mlp + 2 * c["in_ch"] + D_HEAD) + txt * c["ctx"] + attn + 17 * d
    b = 4 * words
    if tier == "int8":
        b += n_joint * (d + mlp) + (4 * n_joint if planner else 4 * n_joint * (max(d, mlp) // 32))
    return b


QWEN = {"klein-4b": tc.QWEN3_4B, "klein-9b": tc.QWEN3_8B}


def te_bytes(k, tier, blocks=27):
    """Weights of the truncated Qwen3 text encoder: the token table, `blocks` blocks and nothing else. Norms and the table are f32; under
    int8 the seven matrices of a block are 36 bytes per 32 weights."""
    c = QWEN[k]
    d, hd = c["d"], c["head_dim"]
    hq, hkv, ff = c["heads"] * hd, c["kv_heads"] * hd, c["ff"]
    lin = d * hq + 2 * d * hkv + hq * d + 3 * d * ff
    other = 2 * d + 2 * hd
    table = tc.QWEN_VOCAB * d
    if tier == "f32":
        return 4 * (table + blocks * (lin + other))
    return 4 * (table + blocks * other) + blocks * lin // 32 * 36


def vae_bytes(direction, px):
    """`vae::decoder_device_bytes_for_pixels` and its encoder twin: the weights, a calibrated number of live buffers per resolution level
    at 928 bytes per output pixel for the level ladder (128, 256, 512, 512), and a fixed scratch of 768 MiB."""
    per_px = sum(c * 4 // 4 ** j for j, c in enumerate(CHANNELS))
    if direction == "decode":
        return 4 * decoder_params(32, True) + 11 * per_px * px + 768 * 2 ** 20
    return 4 * encoder_params(32, True) + 8 * per_px * px + 768 * 2 ** 20


def parts(k, tier, n_ref_tokens):
    c = FLUX2[k]
    n_joint = 512 + 4096 + n_ref_tokens
    p = {
        "dit": (flux2_weight_bytes(c, tier, True) + flux2_scratch_bytes(c, tier, n_joint, planner=True),
                flux2_weight_bytes(c, tier, False) + flux2_scratch_bytes(c, tier, n_joint)),
        "te": (te_bytes(k, tier), te_bytes(k, tier)),
        "vae_dec": (vae_bytes("decode", 1024 * 1024),) * 2,
    }
    if n_ref_tokens:
        p["vae_enc"] = (vae_bytes("encode", 1024 * 1024),) * 2
    return p


def arrangements(k, tier, n_ref_tokens, which):
    """Peak bytes per card of the four arrangements, by the planner's rule: a card is charged its permanent parts plus its heaviest phase.
    `which` picks the figures priced before commit 38e8b2b7a (0) or the uploaded layout's, priced since (1)."""
    p = {n: v[which] for n, v in parts(k, tier, n_ref_tokens).items()}
    ph1 = p["dit"] + p.get("vae_enc", 0)
    ph2 = p["vae_dec"]
    return {
        "all resident on one card": sum(p.values()),
        "one card, encoder permanent, denoiser and decoder in turn": p["te"] + max(ph1, ph2),
        "one card, encoder, denoiser and decoder each in turn": max(p["te"], ph1, ph2),
        "two cards: denoiser and decoder in turn | encoder": (max(ph1, ph2), p["te"]),
    }


def placement_table(card_gib=24):
    cap = card_gib * GIB
    blocks = []
    for k in ("klein-9b", "klein-4b"):
        for tier in ("f32", "int8"):
            for refs in (0, 4096):
                pr = parts(k, tier, refs)
                blocks.append((k, tier, refs, pr))
    rows = []
    for k, tier, refs, pr in blocks:
        if tier == "f32" and k == "klein-4b":
            continue
        label = f"{k} {tier}, {'one reference' if refs else 'no reference'}"
        rows.append([label, _gib(pr["dit"][0]), _gib(pr["dit"][1]), _gib(pr["te"][1]), _gib(pr["vae_dec"][1]),
                     _gib(pr["vae_enc"][1]) if refs else "-"])
    t1 = _table(["request (1024 x 1024)", "DiT, priced before", "DiT, uploaded", "encoder", "decode", "encode"], rows, "lrrrrr")
    rows2 = []
    for k, tier, refs in (("klein-9b", "int8", 0), ("klein-9b", "int8", 4096), ("klein-4b", "int8", 4096), ("klein-9b", "f32", 0)):
        for name in arrangements(k, tier, refs, 1):
            pl, up = arrangements(k, tier, refs, 0)[name], arrangements(k, tier, refs, 1)[name]

            def cell(v):
                if isinstance(v, tuple):
                    return f"{_gib(v[0])} and {_gib(v[1])}" + (", fits" if max(v) <= cap else ", does not fit")
                return f"{_gib(v)}" + (", fits" if v <= cap else ", does not fit")

            rows2.append([f"{k} {tier}{', one reference' if refs else ''}", name, cell(pl), cell(up)])
    t2 = _table(["request", "arrangement", f"priced before, GiB per card ({card_gib} GiB)", "uploaded layout, GiB per card"], rows2, "lllr")
    return (
        "Bytes (GiB = 2^30) that the parts of a FLUX.2 run hold on a device at 1024 x 1024: the denoiser's weights and its scratch for "
        "the joint sequence of 512 text rows, 4,096 generated rows and the reference's 4,096 if there is one (DiT, priced before: what "
        "`flux2::pipeline::dit_bytes` summed before commit 38e8b2b7a; DiT, uploaded: the layout the constructor writes, "
        "`flux2_weight_bytes`, which `dit_bytes` has summed since), the truncated text "
        "encoder's weights at the same tier, and the autoencoder's decode and encode graphs.\n\n" + t1 +
        f"\n\nThe four arrangements on cards of {card_gib} GiB, charged as the planner charges them: a card holds its permanent parts and "
        "the heaviest single phase (the denoiser, with the reference encoder if there is one, is phase 1; the decode is phase 2).\n\n" + t2)


def placement_data():
    """(label, bytes) for the timeline figure: klein-9b int8 with one reference, uploaded layout, over the stages."""
    pr = parts("klein-9b", "int8", 4096)
    p = {n: v[1] for n, v in pr.items()}
    return p


# ------------------------------------------------------------------ bytes that move

def traffic_table():
    rows = []
    c = FLUX2["klein-9b"]
    dit_up = flux2_weight_bytes(c, "int8", False)
    passes, fl_pass = 4, flux2_pass_tokens(c, 4096)
    rows.append(["FLUX.2 klein-9b, int8", "resident across images", "0", "0", "-"])
    rows.append(["FLUX.2 klein-9b, int8", "evicted for the decode, rebuilt for the next image", _gib(dit_up), _gib(dit_up),
                 f"{passes * fl_pass / dit_up:,.0f}"])
    rows.append(["FLUX.2 klein-9b, int8", "every weight uploaded again for every pass", _gib(passes * dit_up), _gib(dit_up),
                 f"{fl_pass / dit_up:,.0f}"])
    f32 = flux2_weight_bytes(c, "f32", False)
    rows.append(["FLUX.2 klein-9b, f32", "every weight uploaded again for every pass", _gib(passes * f32), _gib(f32), f"{fl_pass / f32:,.0f}"])
    d, mlp = 3072, 4 * 3072
    flux1_w = 4 * tc.flux1_params()
    t5 = 4 * sum(tc.t5_groups(tc.T5_XXL).values())
    clip = 4 * tc.CLIP_L_PARAMS
    f1 = tc.flux1_pass_flops(1024, 512)
    rows.append(["FLUX.1 dev, f32", "the encoders built and dropped for each prompt", _gib(t5 + clip), "-", "-"])
    rows.append(["FLUX.1 dev, f32", "resident across images", "0", "0", "-"])
    rows.append(["FLUX.1 dev, f32", "every weight uploaded again for every pass", _gib(50 * flux1_w), _gib(flux1_w), f"{f1 / flux1_w:,.0f}"])
    return (
        "Bytes that cross onto the device per image, from the layouts of `latent_diffusion_systems.placement_table` and the parameter counts "
        "of Chapters 8 and 9, and the operations a pass does for each byte it would have to upload (a pass of 1,024 x 1,024 with 512 text "
        "rows). The columns are the GiB uploaded per image and per pass, and the operations per uploaded byte. FLUX.1's weights are counted from its manifest in 32-bit floats; its encoders are T5-XXL and CLIP-L in 32-bit floats.\n\n"
        + _table(["pipeline", "scheme", "GiB per image", "GiB per pass", "FLOP per byte"], rows, "llrrr"))


# ------------------------------------------------------------------ windows over a large canvas

def tile_starts(total, tile, overlap):
    """Window starts along one axis in tokens (`flux2::pipeline::tile_starts`): as many windows as the overlap asks for, spread evenly."""
    if tile == 0 or total <= tile:
        return [0]
    step = max(tile - overlap, 1)
    n = max(-(-max(total - overlap, 0) // step), 2)
    starts = [i * (total - tile) // (n - 1) for i in range(n)]
    out = []
    for s in starts:
        if not out or out[-1] != s:
            out.append(s)
    return out


def tiling_table():
    c = FLUX2["klein-4b"]
    rows = []
    for side in (1024, 1536, 2048, 3072):
        t = side // 16
        xs = tile_starts(t, 64, 16)
        n_win = len(xs) ** 2
        shares = sorted({64 - (b - a) for a, b in zip(xs, xs[1:])})
        window = flux2_pass_tokens(c, 2 * 64 * 64) if n_win > 1 else flux2_pass_tokens(c, 2 * t * t)
        whole = flux2_pass_tokens(c, 2 * t * t)
        rows.append([f"{side} x {side}", _int(t * t), str(len(xs)) + " x " + str(len(xs)), "; ".join(str(16 * s) for s in shares) or "-",
                     _int(512 + 2 * (64 * 64 if n_win > 1 else t * t)), f"{n_win * window / 1e12:.1f}", f"{whole / 1e12:.1f}",
                     f"{whole / (n_win * window):.2f}"])
    return (
        "Windows of 1,024 x 1,024 pixels (64 x 64 tokens) with an overlap of 256 pixels (16 tokens), `Tiling::new(1024)`, over a larger "
        "canvas of FLUX.2 klein-4b, with the canvas-sized first reference that a windowed run always has (cropped to each window). "
        "Windows per axis come from `tile_starts`; the shared width is the number of pixels two neighbours have in common. TFLOP is one "
        "denoising step over all windows, and for comparison one step of a single forward over the whole canvas, with its generated and "
        "reference rows.\n\n"
        + _table(["canvas", "tokens", "windows", "shared pixels", "rows in one window's forward", "TFLOP, all windows",
                  "TFLOP, one forward", "one forward / windows"], rows, "lrrrrrrr"))


# ================================================================== the toy pipeline

TEXT, NOISE = "latent_diffusion_systems.text", "latent_diffusion_systems.noise"
CLASSES = ("bar", "box", "slash")
WORDS = ("<pad>", "<s>", "</s>") + CLASSES
ROWS_R, WIDTH, TAU, SPREAD, EPS = 6, 4, 0.25, 0.2, 1e-4
N_LAT = 32


def prototype(name):
    """An 8 x 8 image, rows as lists: every pair of rows equal. bar: columns 3 and 4; box: columns 2 to 5 in the row pairs 1 and 2;
    slash: row pair r has columns 2r + 1 and 2r + 2. White (+1) on black (-1)."""
    img = [[-1.0] * 8 for _ in range(8)]
    for y in range(8):
        r = y // 2
        for x in range(8):
            on = (x in (3, 4)) if name == "bar" else (r in (1, 2) and 2 <= x <= 5) if name == "box" else (x in (2 * r + 1, 2 * r + 2))
            img[y][x] = 1.0 if on else -1.0
    return img


def texture(amp=0.25):
    """A checkerboard of +-amp, (+) where x + y is even: it has mean zero and zero left-right difference in every 2 x 2 patch."""
    return [[amp if (x + y) % 2 == 0 else -amp for x in range(8)] for y in range(8)]


def source_image():
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(prototype("bar"), texture())]


def encode_image(img):
    """The autoencoder's encoder: per 2 x 2 patch, channel 0 the mean of the four pixels and channel 1 the left column minus the right
    column, halved (so (p00 + p10 - p01 - p11)/4). Returns the flat latent [2, 4, 4], channel slowest."""
    z = [0.0] * 32
    for i in range(4):
        for j in range(4):
            p00, p01, p10, p11 = img[2 * i][2 * j], img[2 * i][2 * j + 1], img[2 * i + 1][2 * j], img[2 * i + 1][2 * j + 1]
            z[i * 4 + j] = (p00 + p01 + p10 + p11) / 4
            z[16 + i * 4 + j] = ((p00 + p10) - (p01 + p11)) / 4
    return z


def decode_latent(z):
    """The decoder: left pixels of a patch a + d, right pixels a - d, for both rows."""
    img = [[0.0] * 8 for _ in range(8)]
    for i in range(4):
        for j in range(4):
            a, d = z[i * 4 + j], z[16 + i * 4 + j]
            for di in range(2):
                img[2 * i + di][2 * j] = a + d
                img[2 * i + di][2 * j + 1] = a - d
    return img


def _maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _rms(a, b):
    n = len(a) * len(a[0])
    return math.sqrt(sum((x - y) ** 2 for ra, rb in zip(a, b) for x, y in zip(ra, rb)) / n)


def packed_statistics():
    """The standardization of the toy's packed latent: for each of the 8 packed channels the mean and the variance (divisor n) of its
    values over the 4 tokens of the 3 prototypes."""
    raw = {n: vae.pack(encode_image(prototype(n)), 2, 4, 4, [0.0] * 8, [1.0] * 8, 0.0) for n in CLASSES}
    mean, var = [], []
    for oc in range(8):
        vals = [raw[n][oc * 4 + t] for n in CLASSES for t in range(4)]
        m = sum(vals) / len(vals)
        mean.append(m)
        var.append(sum((v - m) ** 2 for v in vals) / len(vals))
    return mean, var


MEAN, VAR = packed_statistics()


def to_tokens(img):
    """Image to the standardized, packed latent: encoder, 2 x 2 pixel-unshuffle, (z - mean)/sqrt(var + eps); the 32 values are 8 packed
    channels of 2 x 2 tokens, channel slowest (`vae.pack`)."""
    return vae.pack(encode_image(img), 2, 4, 4, MEAN, VAR, EPS)


def to_image(tokens):
    return decode_latent(vae.unpack_latent(tokens, 2, 4, 4, MEAN, VAR, EPS))


MEANS = {n: to_tokens(prototype(n)) for n in CLASSES}


def _table_rows():
    g = bookrng.generator(TEXT)
    return [[g.normal() for _ in range(WIDTH)] for _ in WORDS]


TABLE = _table_rows()


def text_stage(prompt):
    """(ids, rows, pooled, class weights): ids = [<s>] + the words + [</s>] padded with <pad> to 6; rows = the table's rows of the ids;
    pooled = the mean of the rows of the content words (none: no vector); class weights = softmax over the three classes of
    -|pooled - row of the class word|^2/(2 tau^2), and 1/3 each for the empty prompt (the denoiser's unconditional case)."""
    ids = [1] + [WORDS.index(w) for w in prompt.split()] + [2]
    ids += [0] * (ROWS_R - len(ids))
    rows = [TABLE[i] for i in ids]
    content = [TABLE[i] for i in ids if i >= 3]
    if not content:
        return ids, rows, None, [1.0 / 3] * 3
    pooled = [sum(r[k] for r in content) / len(content) for k in range(WIDTH)]
    lw = [-sum((pooled[k] - TABLE[3 + c][k]) ** 2 for k in range(WIDTH)) / (2 * TAU ** 2) for c in range(3)]
    top = max(lw)
    w = [math.exp(x - top) for x in lw]
    z = sum(w)
    return ids, rows, pooled, [x / z for x in w]


def class_weights(prompt):
    return text_stage(prompt)[3]


def field(x, sigma, pi):
    """The exact velocity field E[noise - x0 | x_sigma = x] for x0 ~ sum_c pi_c N(m_c, s^2 I): per class the Gaussian formula
    ((sigma - (1 - sigma)s^2)/V)(x - (1 - sigma)m_c) - m_c with V = (1 - sigma)^2 s^2 + sigma^2, weighted by the posterior
    pi_c N(x; (1 - sigma)m_c, V I) / sum. Returns (velocity, posterior weights)."""
    a = 1.0 - sigma
    v = a * a * SPREAD ** 2 + sigma ** 2
    lw = []
    for c, n in enumerate(CLASSES):
        d2 = sum((xi - a * mi) ** 2 for xi, mi in zip(x, MEANS[n]))
        lw.append(math.log(pi[c]) - 0.5 * d2 / v if pi[c] > 0.0 else -math.inf)
    top = max(lw)
    w = [math.exp(q - top) for q in lw]
    z = sum(w)
    w = [q / z for q in w]
    coef = (sigma - a * SPREAD ** 2) / v
    out = [0.0] * len(x)
    for c, n in enumerate(CLASSES):
        m = MEANS[n]
        for k in range(len(x)):
            out[k] += w[c] * (coef * (x[k] - a * m[k]) - m[k])
    return out, w


def toy_sigmas(steps):
    """FLUX.1's rule at the toy's 4 image tokens: mu = 4 m + b on the line through (256, 0.5) and (4096, 1.15), then
    e^mu / (e^mu + 1/s - 1) on linspace(1, 0, steps + 1)."""
    return sigmas_flux1(steps, 4)


def noise_draws(n):
    """n starting noises of 32 values, drawn in order from the stream NOISE."""
    g = bookrng.generator(NOISE)
    return [[g.normal() for _ in range(N_LAT)] for _ in range(n)]


def guided(x, sigma, pi, g):
    uc, _ = field(x, sigma, pi)
    uu, _ = field(x, sigma, [1.0 / 3] * 3)
    return [b + g * (a - b) for a, b in zip(uc, uu)]


def sample(eps, prompt, g, steps, sigmas=None, start=0.0):
    """Euler from the noise `eps`. With `start` = 0 the start is the noise itself; otherwise `start` is a latent x0 and the run is an image-to-image
    one with the schedule `sigmas` (whose first entry is the strength): x = (1 - sigma_0) x0 + sigma_0 eps."""
    pi = class_weights(prompt)
    sg = sigmas or toy_sigmas(steps)
    x = list(eps) if not isinstance(start, list) else [(1 - sg[0]) * a + sg[0] * e for a, e in zip(start, eps)]
    for i in range(len(sg) - 1):
        u = guided(x, sg[i], pi, g)
        x = [xi + (sg[i + 1] - sg[i]) * ui for xi, ui in zip(x, u)]
    return x


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest(x):
    d = [_dist(x, MEANS[n]) for n in CLASSES]
    k = min(range(3), key=lambda i: d[i])
    return k, d[k]


def toy_setup():
    mean_rows = [[oc, f"{MEAN[oc]:.4f}", f"{VAR[oc]:.4f}", f"{math.sqrt(VAR[oc] + EPS):.4f}"] for oc in range(8)]
    t1 = _table(["packed channel", "mean", "variance", "sqrt(variance + 1e-4)"], mean_rows)
    pic = []
    for n in CLASSES:
        img = prototype(n)
        pic.append([n] + ["`" + "".join("#" if v > 0 else "." for v in img[y]) + "`" for y in (0, 2, 4, 6)])
    t2 = _table(["prototype", "rows 0-1", "rows 2-3", "rows 4-5", "rows 6-7"], pic, "lllll")
    dist = [[f"{_dist(MEANS[a], MEANS[b]):.4f}" for b in CLASSES] for a in CLASSES]
    t3 = _table(["distance between class means"] + list(CLASSES), [[a] + r for a, r in zip(CLASSES, dist)])
    tab = [[f"`{WORDS[i]}`"] + [f"{v:.4f}" for v in TABLE[i]] for i in range(len(WORDS))]
    t4 = _table(["token"] + [f"e{k}" for k in range(WIDTH)], tab)
    return (
        "The toy pipeline. Images: 8 x 8 greys in [-1, 1], in which the two rows of every pair are equal for the three prototypes "
        "(# is +1, . is -1, each line standing for a row pair):\n\n" + t2 +
        "\n\nAutoencoder: for each 2 x 2 patch with pixels p00, p01 (top row) and p10, p11 the code is two channels, "
        "(p00 + p01 + p10 + p11)/4 and (p00 + p10 - p01 - p11)/4; the latent is 2 x 4 x 4. Decoder: the left pixels of a patch "
        "are a + d and the right a - d. The latent is packed by the 2 x 2 pixel-unshuffle of `vae.latent_packing` into 8 channels of "
        "2 x 2 tokens (packed channel 4c + 2 pi + pj), and each packed channel is standardized by its own statistics, the mean and "
        "the variance (divisor 12) over the 4 tokens of the three prototypes, with epsilon 1e-4:\n\n" + t1 +
        f"\n\nData: x0 given the class c is the standardized packed latent of prototype c plus noise of standard deviation {SPREAD} in each of the "
        "32 values; the three classes are equally likely. Distances between the class means (32 values):\n\n" + t3 +
        f"\n\nText: the table of 6 tokens by {WIDTH} numbers from {_seed(TEXT)}, `normal()` in row-major order:\n\n" + t4 +
        f"\n\nA prompt is [`<s>`] + words + [`</s>`] padded with `<pad>` to {ROWS_R} rows; the pooled vector is the mean of the rows of the words; "
        f"the class weights are the softmax over the classes of -|pooled - row of the class word|^2/(2 x {TAU}^2); the empty prompt has "
        "weights 1/3 (what a model trained with the prompt dropped has learnt).")


def toy_stages():
    rows = []
    for p in ("bar", "box", "slash", "bar box", ""):
        ids, _, pooled, pi = text_stage(p)
        rows.append([f"`{p}`" if p else "(empty)", " ".join(map(str, ids))] + [f"{x:.4f}" for x in pi])
    t1 = _table(["prompt", "ids", "w bar", "w box", "w slash"], rows, "llrrr")
    chk = []
    for n in CLASSES:
        img = prototype(n)
        rt = decode_latent(encode_image(img))
        chk.append([n, f"{_maxdiff(img, rt):.1e}", f"{_maxdiff(img, to_image(to_tokens(img))):.1e}"])
    src = source_image()
    rt = to_image(to_tokens(src))
    chk.append(["bar plus texture (0.25)", f"{_maxdiff(src, decode_latent(encode_image(src))):.4f}", f"{_maxdiff(src, rt):.4f}"])
    t2 = _table(["image", "codec alone", "with packing and statistics"], chk)
    x = noise_draws(1)[0]
    pi = class_weights("bar box")
    u, w = field(x, 1.0, pi)
    mbar = [sum(pi[c] * MEANS[CLASSES[c]][k] for c in range(3)) for k in range(N_LAT)]
    e1 = max(abs(ui - (xi - mi)) for ui, xi, mi in zip(u, x, mbar))
    sg = toy_sigmas(8)
    t3 = _table(["step i"] + [str(i) for i in range(9)], [["sigma_i"] + [f"{s:.4f}" for s in sg]], "l" + "r" * 9)
    return (
        "Stage by stage. The text stage for five prompts:\n\n" + t1 +
        "\n\nThe codec, and the codec with the packing and the standardization after it: the largest pixel error of the round trip of each image. "
        "The prototypes are in the codec's range, and the textured source is not; the texture (amplitude 0.25 on both diagonals of every "
        "patch) is exactly what the codec cannot keep, so the error is the texture's amplitude:\n\n" + t2 +
        f"\n\nThe denoiser at sigma = 1 for the prompt `bar box` and the first noise draw of {_seed(NOISE)}: the field must be x minus the mean of "
        f"the class means weighted by the prompt, and the largest difference is {e1:.1e}. The posterior weights there are the prompt's: "
        f"{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}.\n\nThe schedule of 8 steps on FLUX.1's rule at 4 tokens (mu = {mu_flux1(4):.4f}):\n\n" + t3)


def sampling_table(n=100):
    eps = noise_draws(n)
    rows = []
    for prompt in ("box", "bar box"):
        for g in (0.0, 1.0, 2.0, 4.0, 8.0):
            counts = [0, 0, 0]
            sd = []
            for e in eps:
                x = sample(e, prompt, g, 8)
                k, d = nearest(x)
                counts[k] += 1
                sd.append(d)
            rows.append([f"`{prompt}`", f"{g:g}", str(counts[0]), str(counts[1]), str(counts[2]), f"{sum(sd) / n:.4f}"])
    t1 = _table(["prompt", "g", "bar", "box", "slash", "mean distance to the nearest class mean"], rows, "lrrrrr")
    rows2 = []
    ref = [sample(e, "box", 2.0, 512) for e in eps[:20]]
    for steps in (1, 2, 4, 8, 16, 64, 512):
        xs = [sample(e, "box", 2.0, steps) for e in eps[:20]]
        dd = [_dist(x, r) for x, r in zip(xs, ref)]
        sp = [nearest(x)[1] for x in xs]
        rows2.append([str(steps), f"{sum(dd) / len(dd):.4f}", f"{max(dd):.4f}", f"{sum(sp) / len(sp):.4f}"])
    t2 = _table(["Euler steps", "mean distance to the 512-step result", "largest", "mean distance to the box mean"], rows2)
    return (
        f"{n} starting noises (the first {n} draws of 32 numbers from {_seed(NOISE)}, in order), the same for every row, sampled with 8 Euler "
        "steps of `toy_sigmas` and the guided field u_un + g (u_c - u_un). The columns count the samples whose nearest class mean is "
        "each class; the last is the mean distance of a sample to its nearest class mean (data have 0.2 sqrt(32) = 1.13 on average).\n\n"
        + t1 + "\n\nConvergence of the steps for the prompt `box` at g = 2 on the first 20 noises, against the same sampler at 512 steps.\n\n" + t2)


def strength_table(n=40):
    eps = noise_draws(n)
    src = source_image()
    x0 = to_tokens(src)
    bar_rt = to_image(x0)
    rows = []
    for s in (0.0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 1.0):
        counts = [0, 0, 0]
        errs = []
        sg = [s * v for v in toy_sigmas(8)]
        for e in eps:
            x = x0[:] if s == 0.0 else sample(e, "box", 2.0, 8, sigmas=sg, start=x0)
            k, _ = nearest(x)
            counts[k] += 1
            errs.append(_rms(to_image(x), bar_rt))
        rows.append([f"{s:g}", str(counts[0]), str(counts[1]), str(counts[2]), f"{sum(errs) / n:.4f}"])
    return (
        f"Image to image. The source is the bar with the texture; x0 is its standardized packed latent (the texture is already gone from "
        f"it). Each of {n} noises starts the run at x = (1 - s) x0 + s eps with the schedule s times `toy_sigmas(8)` (so the first "
        "sigma is s), the prompt is `box` and g = 2. Strength 0 skips the sampler. The columns count the final latents nearest to each "
        "class mean, and the last is the mean root-mean-square pixel distance of the decoded result from the decoded source x0.\n\n"
        + _table(["strength s", "bar", "box", "slash", "mean pixel distance to the codec's source"], rows, "lrrrr"))


def mask_table():
    eps = noise_draws(1)[0]
    src = source_image()
    x0 = to_tokens(src)
    x, gen = masked_run()
    kept = [oc * 4 + t for oc in range(8) for t in (0, 2)]
    err_lat = max(abs(x[k] - x0[k]) for k in kept)
    rt = to_image(x0)
    left_lat = [row[:4] for row in gen]
    # pixel composite: old + m (new - old), hard mask and a soft mask
    hard = [0.0] * 4 + [1.0] * 4
    soft = [0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]
    rows = []
    for name, m in (("hard (columns 4 to 7)", hard), ("soft (0, 0, 0, 0.25, 0.75, 1, 1, 1)", soft)):
        comp = [[s + m[xp] * (g - s) for xp, (s, g) in enumerate(zip(rs, rg))] for rs, rg in zip(src, gen)]
        outside = max(abs(comp[y][xp] - src[y][xp]) for y in range(8) for xp in range(8) if m[xp] == 0.0)
        near = max(abs(comp[y][xp] - src[y][xp]) for y in range(8) for xp in range(8) if 0.0 < m[xp] < 1.0) if any(0.0 < v < 1.0 for v in m) else 0.0
        has_band = any(0.0 < v < 1.0 for v in m)
        rows.append([name, f"{outside:.1e}", f"{near:.4f}" if has_band else "-"])
    t_pix = _table(["pixel mask", "largest change where the mask is 0", "largest change where the mask is between 0 and 1"], rows)
    left_err = max(abs(gen[y][xp] - src[y][xp]) for y in range(8) for xp in range(4))
    right_err = _rms([r[4:] for r in gen], [r[4:] for r in prototype("box")])
    hard_img = [[src[y][xp] + hard[xp] * (gen[y][xp] - src[y][xp]) for xp in range(8)] for y in range(8)]
    return (
        f"A masked run of the toy: the source with the texture, the prompt `box`, g = 2, 8 steps, the first noise of {_seed(NOISE)}, and the "
        "mask 0 on the two left tokens and 1 on the two right ones (tokens in raster order 0 to 3). After every Euler step the kept "
        "tokens are replaced by (1 - sigma) x0 + sigma eps at the sigma just reached (x0 itself when it is 0). The largest difference between "
        f"the final kept tokens and x0 is {err_lat:.1e}. Decoded, the left half of the picture differs from the *source* by up to "
        f"{left_err:.4f}, the amplitude of the texture, because the latent never held it; the left half equals the codec's round trip of the source to "
        f"{_maxdiff([r[:4] for r in gen], [r[:4] for r in rt]):.1e}; the right half is within {right_err:.4f} (root mean square per pixel) of the box prototype's right half. A composite in pixel space, source + m (generated - source) per pixel "
        "column, keeps the source:\n\n" + t_pix)


def masked_run():
    """The masked run of `mask_table`: (final tokens, decoded generated image)."""
    eps = noise_draws(1)[0]
    x0 = to_tokens(source_image())
    mask_tok = [0.0, 1.0, 0.0, 1.0]
    sg = toy_sigmas(8)
    pi = class_weights("box")
    x = list(eps)
    for i in range(8):
        u = guided(x, sg[i], pi, 2.0)
        x = [xi + (sg[i + 1] - sg[i]) * ui for xi, ui in zip(x, u)]
        for oc in range(8):
            for t in range(4):
                if mask_tok[t] <= 0.0:
                    k = oc * 4 + t
                    x[k] = x0[k] if sg[i + 1] <= 0.0 else (1 - sg[i + 1]) * x0[k] + sg[i + 1] * eps[k]
    return x, to_image(x)


def edit_figure_data():
    """Images for the figure: prototypes, the textured source and its codec round trip, three strengths, the masked run and its
    composite in pixel space."""
    eps = noise_draws(1)[0]
    src = source_image()
    x0 = to_tokens(src)
    out = {"bar": prototype("bar"), "box": prototype("box"), "slash": prototype("slash"), "source": src, "codec": to_image(x0)}
    sg = toy_sigmas(8)
    for s in (0.25, 0.5, 1.0):
        sgs = [s * v for v in sg]
        out[f"s{s}"] = to_image(sample(eps, "box", 2.0, 8, sigmas=sgs, start=x0))
    _, gen = masked_run()
    out["masked"] = gen
    hard = [0.0] * 4 + [1.0] * 4
    out["composite"] = [[src[y][x] + hard[x] * (gen[y][x] - src[y][x]) for x in range(8)] for y in range(8)]
    return out
