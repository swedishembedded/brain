# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Generating Video".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: video_generation.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree.

* the frame-to-frame difference curve of a clip and the blow-up ratio that summarises it, on constant-frame clips;
* the pairs of positions an attention layer scores for a clip read frame by frame, as space then time, and all at once;
* the arithmetic of one transformer block (linear layers, self-attention, text cross-attention) for brain's video models;
* the rotary angles of a latent token in LTX (physical positions, a band shared by three axes) and in Wan (three ladders);
* the token-count shift of the sampling schedule and the fixed distilled tables;
* a frozen token: initial mixture, per-token timestep, the conversion to a clean estimate and the blend that pins it;
* the window plan of a clip too long for one pass, and the seam of a toy motion carried by a still or by a latent tail;
* the audio latent that rides beside the video latent: token counts, time bounds, quanta of a seam;
* parameter counts of the models from their tensor shapes, the passes of a sampling step and the work of a clip.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, exp, log, cos) unless a block says otherwise. Sums
are accumulated left to right in index order. The only random numbers are the draws of the seam toy, from the book's SplitMix64
(`bookrng.generator`, stream name printed by the chapter). A FLOP is one multiply or one add, so a multiply-add is two.

Swedish Embedded AB implements solutions for generating video that stays coherent from the first frame to the last. If your team
needs expertise in video diffusion, conditioning and long-form generation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import images as im
import video as vid

_table = im._table


def _int(n):
    return f"{n:,}"


def _g(x, digits=4):
    return f"{x:.{digits}g}"


def _round_half_away(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ------------------------------------------------------------------------------------------------ the shapes of the models

# The LTX video grid: 32 pixels per latent cell, 8 pixel frames per latent frame after the first (chapter 1).
LTX_W, LTX_H, LTX_F = 768, 512, 121
LTX_LH, LTX_LW = LTX_H // 32, LTX_W // 32          # 16 x 24
LTX_PER_FRAME = LTX_LH * LTX_LW                    # 384 tokens in one latent frame
LTX_LAT = vid.latent_frames(LTX_F)                 # 16
LTX_TOKENS = LTX_LAT * LTX_PER_FRAME               # 6144

LTX = dict(d=4096, layers=48, ff=4, ctx=1024, heads=32)       # LtxDitConfig::ltx25_22b, context rows of the Gemma-4 frame
LTX_A = dict(d=2048, heads=32)                                # LtxAudioDitConfig::ltx25
WAN13 = dict(d=1536, ffn=8960, layers=30, heads=12, ctx=512, in_ch=16)
WAN14 = dict(d=5120, ffn=13824, layers=40, heads=40, ctx=512, in_ch=16)


def wan_tokens(frames, width, height):
    """Tokens of a Wan clip: latent (1 + (F-1)/4) x (H/8) x (W/8), then 1 x 2 x 2 patches."""
    assert (frames - 1) % 4 == 0 and height % 16 == 0 and width % 16 == 0
    return (1 + (frames - 1) // 4) * (height // 16) * (width // 16)


# ------------------------------------------------------------------------------------------------ the stability curve

PROBE_DISPLAY = 127.5      # a latent-free frame value a in [-1, 1] is shown as (a + 1) * 127.5


def frame_diffs(values):
    """Mean absolute difference of consecutive constant frames, in display units (the box-averaged probe of a constant frame is
    its value, so the difference of two constant frames is |a - b| * 127.5)."""
    return [abs(b - a) * PROBE_DISPLAY for a, b in zip(values, values[1:])]


def blowup(diffs):
    """The largest difference over the upper median (element len // 2 of the sorted list); 1 for fewer than two differences or a
    zero median."""
    if len(diffs) < 2:
        return 1.0
    s = sorted(diffs)
    med = s[len(s) // 2]
    return 1.0 if med <= 0 else s[-1] / med


def _clip_cases():
    steady = [round(-0.6 + 0.03 * i, 10) for i in range(16)]
    fast = [round(-0.9 + 0.1 * i, 10) for i in range(16)]
    collapse = [round(-0.6 + 0.03 * i, 10) for i in range(12)] + [0.7, -0.7, 0.7, -0.7]
    cut = [round(-0.6 + 0.03 * i, 10) for i in range(8)] + [round(0.2 + 0.03 * i, 10) for i in range(8)]
    return [("slow steady drift", steady), ("fast steady drift", fast), ("steady, then four thrashing frames", collapse),
            ("steady with one hard cut", cut)]


def clip_stability():
    rows = []
    for name, vals in _clip_cases():
        d = frame_diffs(vals)
        s = sorted(d)
        rows.append([name, len(d), f"{s[len(s) // 2]:.3f}", f"{s[-1]:.3f}", f"{blowup(d):.2f}"])
    return ("Four clips of 16 constant frames, a frame being the single number a in [-1, 1] that every pixel holds, shown as (a + 1) * 127.5. "
            "Clip 1: a_i = -0.6 + 0.03 i. Clip 2: a_i = -0.9 + 0.1 i. Clip 3: a_i = -0.6 + 0.03 i for i = 0 .. 11, then 0.7, -0.7, 0.7, -0.7. "
            "Clip 4: a_i = -0.6 + 0.03 i for i = 0 .. 7, then a_i = 0.2 + 0.03 (i - 8) for i = 8 .. 15. The difference d_i is |a_(i+1) - a_i| * 127.5 "
            "for i = 0 .. 14; the median is element 7 of the 15 sorted differences (the upper of the middle pair when the count is even); "
            "the ratio is the largest difference over that median.\n\n"
            + _table(["clip", "differences", "median", "largest", "ratio"], rows, "lrrrr"))


# ------------------------------------------------------------------------------------------------ pairs of positions

def pairs_for(frames_latent, per_frame):
    """(frame by frame, space then time, all at once) pairs of positions scored by one attention layer, per head."""
    f, s = frames_latent, per_frame
    return f * s * s, f * s * s + s * f * f, (f * s) ** 2


def attention_pairs_data(per_frame=LTX_PER_FRAME, frames=(1, 9, 25, 49, 97, 121, 241, 481)):
    rows = []
    for fr in frames:
        lf = vid.latent_frames(fr)
        a, b, c = pairs_for(lf, per_frame)
        rows.append((fr, lf, lf * per_frame, a, b, c))
    return rows


def attention_pairs():
    rows = []
    for fr, lf, t, a, b, c in attention_pairs_data():
        rows.append([fr, lf, _int(t), _int(a), _int(b), _int(c), f"{c / b:.1f}"])
    pic = LTX_PER_FRAME ** 2
    t121 = LTX_TOKENS ** 2
    return (f"The latent frame of a 768 x 512 clip is 16 x 24 = {LTX_PER_FRAME} tokens. For a clip of F pixel frames (F = 1 + 8k) with F' = 1 + k latent "
            f"frames and S = {LTX_PER_FRAME} tokens in each, one head of one attention layer scores: F' S^2 pairs when every latent frame attends only "
            f"within itself; F' S^2 + S F'^2 when each layer is either spatial (within a frame) or temporal (within a position, across frames); "
            f"(F' S)^2 when every token attends to every token.\n\n"
            + _table(["F", "F'", "tokens", "frame by frame", "space then time", "all at once", "all at once / space then time"], rows, "rrrrrrr")
            + f"\n\nOne picture (F = 1) costs {_int(pic)} pairs; the 121-frame clip costs {_int(t121)} with all-at-once attention, {t121 // pic} times as many "
            f"(16 frames, so 16 times the tokens and 16 squared the pairs).")


# ------------------------------------------------------------------------------------------------ one block, counted

def block_flops(model, tokens, ctx):
    """FLOPs (2 per multiply-add) of one block of the self-attention / text cross-attention / feed-forward stack, split into
    (linear layers on the tokens, text-side projections, self-attention scores + weighted sum, cross-attention scores + weighted sum).

    LTX: four d x d matrices in self-attention, two (q and o) in cross-attention applied to the T tokens and two (k and v) to the N
    context rows, a feed-forward of two d x 4d matrices. Wan: the same with a feed-forward of two d x f matrices."""
    d = model["d"]
    f = model.get("ffn", model.get("ff", 4) * d)
    lin = 2 * tokens * (4 * d * d) + 2 * tokens * (2 * d * d) + 2 * tokens * (2 * d * f)
    text = 2 * ctx * (2 * d * d)
    self_att = 4 * tokens * tokens * d
    cross_att = 4 * tokens * ctx * d
    return lin, text, self_att, cross_att


def crossover_tokens(model):
    """The token count at which the self-attention scores equal the token-side linear layers of a block: 4 T^2 d = c T, with c the
    per-token linear FLOPs, i.e. T = c / (4 d)."""
    d = model["d"]
    f = model.get("ffn", model.get("ff", 4) * d)
    c = 2 * (4 * d * d) + 2 * (2 * d * d) + 2 * (2 * d * f)
    return c / (4 * d)


def _models():
    return [("LTX-2.5 22B, 768 x 512, 121 frames", LTX, LTX_TOKENS),
            ("LTX-2.5 22B, 1536 x 1024, 121 frames", LTX, LTX_LAT * 32 * 48),
            ("Wan 1.3B, 832 x 480, 81 frames", WAN13, wan_tokens(81, 832, 480)),
            ("Wan 14B, 1280 x 720, 81 frames", WAN14, wan_tokens(81, 1280, 720))]


def forward_flops(model, tokens):
    lin, text, sa, ca = block_flops(model, tokens, model["ctx"])
    return model["layers"] * (lin + text + sa + ca), (lin, text, sa, ca)


def attention_split():
    rows = []
    for name, m, t in _models():
        total, (lin, text, sa, ca) = forward_flops(m, t)
        rows.append([name, _int(t), f"{lin / 1e12:.2f}", f"{sa / 1e12:.2f}", f"{(text + ca) / 1e12:.3f}", f"{sa / (lin + text + sa + ca):.2f}",
                     f"{total / 1e12:.1f}", _int(round(crossover_tokens(m)))])
    return ("FLOPs of one block (a multiply-add counted as two) in units of 10^12, from the four d x d matrices of self-attention, the q and o matrices of "
            "text cross-attention on the T tokens, a feed-forward of two matrices of width d by f (LTX: d = 4096, f = 4d; Wan 1.3B: d = 1536, f = 8960; Wan 14B: d = 5120, f = 13824), "
            "and the scores plus weighted sum of self-attention (4 T^2 d); the text side (k and v matrices on N = 1024 or 512 context rows, 4 N d^2, and "
            "the cross-attention scores 4 T N d) is shown separately. Left out: norms, rotation, gates, the embedding and output matrices, and the conditioning "
            "networks (each a small fraction of the block matrices). 'Attention share' is the self-attention scores over the whole block. 'Forward' multiplies the block by "
            "its depth (48, 30, 40). 'Crossover' is the T at which self-attention scores equal the token-side linear layers, T = c / (4 d) with c the per-token "
            "linear FLOPs.\n\n"
            + _table(["clip", "tokens T", "linear / block", "self-attention / block", "text side / block", "attention share", "forward (10^12)", "crossover T"],
                     rows, "lrrrrrrr"))


# ------------------------------------------------------------------------------------------------ rotary positions

LTX_THETA = 10000.0
LTX_MAX_POS = (20, 2048, 2048)       # seconds, pixels, pixels


def ltx_bands(inner_dim=4096, n_pos=3):
    """(bands L, angles per token, identity columns): L = inner_dim // (2 n_pos), n_pos L angles, the rest padded in front."""
    l = inner_dim // (2 * n_pos)
    return l, n_pos * l, inner_dim // 2 - n_pos * l


def ltx_rate(k, l, theta=LTX_THETA):
    """indices[k] = theta^(k / (L - 1)) * pi / 2 (pi / 2 when L = 1)."""
    return theta ** (k / (l - 1) if l > 1 else 0.0) * math.pi / 2


def latent_frame_bounds(k, fps):
    """The seconds [start, end) of latent frame k: pixel frames max(0, 8k - 7) .. max(0, 8k + 1), divided by the frame rate."""
    return max(0, 8 * k - 7) / fps, max(0, 8 * k + 1) / fps


def ltx_position_table():
    rows = []
    for fps in (24, 12):
        for k in range(4):
            s, e = latent_frame_bounds(k, fps)
            mid = (s + e) / 2
            rows.append([fps, k, f"{s:.4f}", f"{e:.4f}", f"{mid:.4f}", f"{mid / LTX_MAX_POS[0]:.6f}", f"{2 * mid / LTX_MAX_POS[0] - 1:+.6f}"])
    rows2 = []
    for row in range(3):
        s, e = row * 32, (row + 1) * 32
        mid = (s + e) / 2
        rows2.append([row, f"{s} .. {e}", f"{mid:g}", f"{mid / LTX_MAX_POS[1]:.6f}", f"{2 * mid / LTX_MAX_POS[1] - 1:+.6f}"])
    return ("The frame axis of the first four latent frames, in seconds: latent frame k spans the pixel frames max(0, 8k - 7) .. max(0, 8k + 1) "
            "(half-open), divided by the frame rate. The position that enters the rotation is the midpoint divided by the axis maximum "
            "(20 seconds), mapped to [-1, 1] by 2 mid / 20 - 1.\n\n"
            + _table(["fps", "k", "start (s)", "end (s)", "midpoint (s)", "mid / 20", "2 mid / 20 - 1"], rows, "rrrrrrr")
            + "\n\nThe height axis of latent rows 0, 1, 2 (32 pixels each, maximum 2048 pixels); the width axis is the same:\n\n"
            + _table(["row", "pixels", "midpoint", "mid / 2048", "2 mid / 2048 - 1"], rows2, "rlrrr"))


def ltx_angle_table():
    l, n_ang, pad = ltx_bands()
    ks = (0, 1, 340, 681)
    mid1 = sum(latent_frame_bounds(1, 24)) / 2
    mid2 = sum(latent_frame_bounds(2, 24)) / 2
    p1 = 2 * mid1 / LTX_MAX_POS[0] - 1
    p2 = 2 * mid2 / LTX_MAX_POS[0] - 1
    rows = []
    for k in ks:
        r = ltx_rate(k, l)
        rows.append([k, f"{r:.4f}", f"{r * p1:+.4f}", f"{r * p2:+.4f}", f"{r * (p2 - p1):.4f}"])
    return (f"inner_dim = 4096 gives L = 4096 // (2 * 3) = {l} bands, {n_ang} angles per token and {pad} identity columns in front of a table of 2048 columns. "
            f"Band k has rate indices[k] = 10000^(k / {l - 1}) * pi / 2, and the angle of a token on an axis is that rate times 2 mid / max - 1. "
            f"Latent frames 1 and 2 of a 24 frames per second clip have the positions {p1:+.6f} and {p2:+.6f} on the frame axis.\n\n"
            + _table(["band k", "rate", "angle, latent frame 1", "angle, latent frame 2", "turn between them"], rows, "rrrrr"))


def ltx_head_map(inner_dim=4096, heads=32, n_pos=3):
    """For head h: the (band, axis) of its first and last table column (None for an identity column)."""
    l, n_ang, pad = ltx_bands(inner_dim, n_pos)
    half = (inner_dim // 2) // heads

    def at(col):
        c = col - pad
        return None if c < 0 else (c // n_pos, c % n_pos)

    return [(h, h * half, (h + 1) * half - 1, at(h * half), at((h + 1) * half - 1)) for h in range(heads)], half


def ltx_head_table():
    info, half = ltx_head_map()
    rows = []
    for h in (0, 1, 2, 30, 31):
        _, c0, c1, a0, a1 = info[h]
        fmt = lambda x: "identity" if x is None else f"band {x[0]}, axis {x[1]}"  # noqa: E731
        rows.append([h, f"{c0} .. {c1}", fmt(a0), fmt(a1)])
    axes = "frame, height, width"
    return (f"The table of 2048 columns is cut into 32 consecutive chunks of {half} columns, one per head; column c >= 2 holds band (c - 2) // 3 and axis (c - 2) mod 3, "
            f"with the axes in the order {axes}.\n\n"
            + _table(["head", "columns", "first column holds", "last column holds"], rows, "rlll"))


WAN_THETA = 10000.0


def wan_axes(head_dim=128):
    """Pairs on (frame, height, width): c = head_dim // 2, hw = c // 3, frame takes the remainder."""
    c = head_dim // 2
    hw = c // 3
    return c - 2 * hw, hw, hw


def wan_rope_table():
    rows = []
    pairs = wan_axes()
    for name, p, ids in zip(("frame", "height", "width"), pairs, (21, 30, 52)):
        d = 2 * p
        fastest = 1.0
        slowest = WAN_THETA ** (-(2 * (p - 1)) / d)
        rows.append([name, p, d, f"{fastest:g}", f"{slowest:.6f}", _int(round(2 * math.pi / slowest)), ids])
    return ("Wan 1.3B has 12 heads of width 128, so 64 rotation pairs per head, divided as [c - 2 (c // 3), c // 3, c // 3] with c = 64: 22 pairs on the frame axis and 21 on each "
            "of the other two. Pair j of an axis of width d turns by (the integer position on that axis) * 10000^(-2 j / d). The last column is how many positions the "
            "default 832 x 480, 81-frame clip uses on the axis (21 latent frames, 480 / 16 = 30 rows, 832 / 16 = 52 columns of patches).\n\n"
            + _table(["axis", "pairs", "width d", "fastest rate", "slowest rate", "slowest wavelength", "positions used"], rows, "lrrrrrr"))


# ------------------------------------------------------------------------------------------------ the schedule

LTX_BASE_SHIFT, LTX_MAX_SHIFT = 0.95, 2.05
LTX_ANCHORS = (1024.0, 4096.0)
LTX_TERMINAL = 0.1
DISTILLED = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
DISTILLED_STAGE2 = (0.909375, 0.725, 0.421875, 0.0)


def ltx_mu(tokens, base=LTX_BASE_SHIFT, top=LTX_MAX_SHIFT):
    """mu(T): the straight line through (1024, base) and (4096, top), continued past both ends."""
    slope = (top - base) / (LTX_ANCHORS[1] - LTX_ANCHORS[0])
    return base + slope * (tokens - LTX_ANCHORS[0])


def ltx_sigmas(tokens, steps, stretch=True, terminal=LTX_TERMINAL):
    """linspace(1, 0, steps + 1), shifted by e^mu / (e^mu + 1/s - 1) (0 stays 0), then stretched so that the last non-zero value is `terminal`."""
    e = math.exp(ltx_mu(tokens))
    ramp = [1.0 - i / steps for i in range(steps + 1)]
    out = [0.0 if s == 0.0 else e / (e + (1.0 / s - 1.0)) for s in ramp]
    if stretch:
        nz = [i for i, v in enumerate(out) if v != 0.0]
        scale = (1.0 - out[nz[-1]]) / (1.0 - terminal)
        for i in nz:
            out[i] = 1.0 - (1.0 - out[i]) / scale
    return out


def schedule_shift_table():
    rows = []
    for name, t in (("one 768 x 512 latent frame", 384), ("1024 tokens", 1024), ("2048 tokens", 2048), ("a 1024 x 1024 picture", 4096),
                    ("768 x 512, 121 frames", 6144), ("1536 x 1024, 121 frames", 24576)):
        sg = ltx_sigmas(t, 4)
        rows.append([name, _int(t), f"{ltx_mu(t):.4f}", f"{math.exp(ltx_mu(t)):.3f}"] + [f"{v:.4f}" for v in sg[1:4]])
    return ("mu(T) is the straight line through (1024, 0.95) and (4096, 2.05), continued without a clamp. A 4-step schedule: s_i = 1 - i/4, shifted "
            "to e^mu / (e^mu + 1/s - 1), then stretched: with q the last non-zero shifted value, every non-zero value v becomes 1 - (1 - v) (1 - 0.1) / (1 - q), "
            "so the last non-zero value is exactly 0.1. The first value is 1 and the last is 0 in every row.\n\n"
            + _table(["clip", "tokens T", "mu", "e^mu", "sigma 1", "sigma 2", "sigma 3"], rows, "lrrrrrr"))


def log_snr(s):
    if s <= 0.0:
        return float("inf")
    if s >= 1.0:
        return float("-inf")
    return 2.0 * math.log((1.0 - s) / s)


def distilled_table():
    rows = []
    for i, s in enumerate(DISTILLED):
        nxt = DISTILLED[i + 1] if i + 1 < len(DISTILLED) else None
        lg = "-inf" if s >= 1.0 else ("+inf" if s <= 0 else f"{log_snr(s):+.3f}")
        rows.append([i, f"{s:.6f}", lg, "" if nxt is None else f"{s - nxt:.6f}", "stage 2 starts here" if s == DISTILLED_STAGE2[0] else ""])
    return ("The eight-step schedule of the distilled checkpoint is a fixed table (it is not the output of the shifted ramp above), and the second stage "
            "of a two-stage run takes its last four entries. log SNR = 2 ln((1 - s) / s).\n\n"
            + _table(["step", "sigma", "log SNR", "step size", "note"], rows, "rrrrl"))


def two_stage_work():
    """FLOPs of the forwards of a 1536 x 1024, 121-frame clip: 8 forwards at 768 x 512 and 3 at 1536 x 1024 (two stages) against
    8 forwards at 1536 x 1024."""
    t_half, t_full = LTX_TOKENS, LTX_LAT * 32 * 48
    f_half = forward_flops(LTX, t_half)[0]
    f_full = forward_flops(LTX, t_full)[0]
    two = 8 * f_half + 3 * f_full
    one = 8 * f_full
    rows = [["two stages", f"8 x {_int(t_half)} + 3 x {_int(t_full)}", f"{two / 1e15:.2f}"], ["one stage at full size", f"8 x {_int(t_full)}", f"{one / 1e15:.2f}"]]
    return ("The video-stream forward of the table above, 1536 x 1024 output, 121 frames: stage 1 runs 8 forwards on the half-size latent and stage 2 three on the full "
            "latent after a spatial upscale; the comparison is eight forwards at full size.\n\n"
            + _table(["plan", "forwards x tokens", "FLOPs (10^15)"], rows, "lrr")
            + f"\n\nThe two-stage plan does {one / two:.2f} times less arithmetic in the denoiser than the single stage at full size.")


# ------------------------------------------------------------------------------------------------ a frozen token

def conditioning_table():
    c, n = 10.0, -2.0
    rows = []
    for s in (1.0, 0.8, 0.5, 0.0):
        m = 1.0 - s
        init = (1.0 - m) * c + m * n
        rows.append([f"{s:g}", f"{m:g}", f"{init:+.2f}", f"{m * 1.0:g}", f"{m * DISTILLED[-2]:.6f}"])
    return ("A token whose clean content is c = 10, with initial noise draw n = -2, conditioned at strength s: the mask is m = 1 - s, the token starts at "
            "(1 - m) c + m n and is announced to the network at the timestep m * sigma. The last two columns are that timestep at the first sigma (1) "
            "and at the last non-zero sigma of the distilled table (0.421875).\n\n"
            + _table(["strength s", "mask m", "start value", "timestep at sigma = 1", "timestep at sigma = 0.421875"], rows, "rrrrr"))


def clean_estimate_table():
    sigma = DISTILLED[-2]
    x, v, c = 5.0, 1.0, 5.0
    rows = []
    for s in (1.0, 0.5, 0.0):
        m = 1.0 - s
        scalar = x - v * sigma
        per_token = x - v * m * sigma
        blended = m * per_token + (1.0 - m) * c
        rows.append([f"{s:g}", f"{scalar:.7f}", f"{per_token:.7f}", f"{blended:.7f}"])
    return ("One token at the terminal step of the distilled schedule, sigma = 0.421875, holding x = 5 with clean content c = 5, and a network that returns the velocity v = 1. "
            "The clean estimate is x - v t, with t the timestep: the scalar sigma for every token ('scalar'), or m sigma for a token with mask m ('per token'). "
            "The last column blends the per-token estimate with the clean content, m * estimate + (1 - m) * c, which is what the loop keeps.\n\n"
            + _table(["strength s", "scalar sigma", "per-token timestep", "after the blend"], rows, "rrrr"))


FROZEN_TOKENS = (("free", 1.0, 0.0, 2.0), ("half held (s = 0.5)", 0.5, 10.0, -2.0), ("pinned (s = 1)", 0.0, 5.0, 7.0))


def denoise_clip(x, mask, clean, sigmas, velocity):
    """The deterministic loop of the chapter's pseudocode on a clip of N tokens of one channel; returns the trace of x before each step and the end."""
    n_tok = len(x)
    x = [(1.0 - mask[n]) * clean[n] + mask[n] * x[n] for n in range(n_tok)]
    trace = [list(x)]
    for i in range(len(sigmas) - 1):
        t = [mask[n] * sigmas[i] for n in range(n_tok)]
        v = velocity(x, t)
        for n in range(n_tok):
            x0 = x[n] - t[n] * v[n]
            x0 = mask[n] * x0 + (1.0 - mask[n]) * clean[n]
            if sigmas[i + 1] == 0.0:
                x[n] = x0
            else:
                r = sigmas[i + 1] / sigmas[i]
                x[n] = r * x[n] + (1.0 - r) * x0
        trace.append(list(x))
    return trace


def frozen_loop():
    mask = [m for _, m, _, _ in FROZEN_TOKENS]
    clean = [c for _, _, c, _ in FROZEN_TOKENS]
    start = [x for _, _, _, x in FROZEN_TOKENS]
    trace = denoise_clip(start, mask, clean, list(DISTILLED), lambda x, t: [1.0] * len(x))
    rows = []
    for i, state in enumerate(trace):
        rows.append([i] + [f"{v:.6f}" for v in state])
    return ("Three tokens of one channel through the loop of the pseudocode on the eight-step distilled schedule, with a network that returns the velocity 1 for every token "
            "whatever it is given. The free token has mask 1 and starts at 2. The half-held token has mask 0.5, clean content 10 and starts from the noise draw -2 (the loop first "
            "replaces it by 0.5 * 10 + 0.5 * (-2) = 4). The pinned token has mask 0, clean content 5 and starts from the draw 7 (the loop replaces it by 5). Row i is the state "
            "before step i; the last row is the result.\n\n"
            + _table(["step", "free", "half held", "pinned"], rows, "rrrr")
            + "\n\nThe free token loses exactly the sum of the step sizes, 1, and ends at 1; the pinned token never moves.")


def anchor_cost():
    rows = []
    t = LTX_TOKENS
    for name, extra, free in (("text only", 0, t), ("first frame pinned into its slot (image to video)", 0, t - LTX_PER_FRAME),
                              ("a mid-clip still pinned into its slot", 0, t - LTX_PER_FRAME),
                              ("first still appended as a guide", LTX_PER_FRAME, t), ("first and last stills appended", 2 * LTX_PER_FRAME, t),
                              ("first, middle and last stills, middle pinned", 2 * LTX_PER_FRAME, t - LTX_PER_FRAME)):
        n = t + extra
        rows.append([name, _int(n), _int(n * n), f"{n * n / (t * t):.4f}", _int(free)])
    return (f"A 768 x 512, 121-frame clip is {_int(t)} tokens (16 latent frames of {LTX_PER_FRAME}). A pinned still overwrites a latent frame's tokens in place and adds nothing; "
            f"an appended guide adds {LTX_PER_FRAME} tokens to the sequence. 'Free' is the number of tokens the model is still generating.\n\n"
            + _table(["conditioning", "sequence length", "self-attention pairs", "pairs vs text only", "free tokens"], rows, "lrrrr"))


# ------------------------------------------------------------------------------------------------ windows

CONTEXT = 8
LONGFORM_MAX_TOKENS = 13200


def window_plan(frames, lh, lw, context, max_tokens, align=1):
    """The rolling-window plan of the chapter's pseudocode: a list of (context, new, first_frame)."""
    assert frames >= 1 and (frames - 1) % 8 == 0 and context >= 1
    max_lat = max_tokens // (lh * lw)
    k_total = (frames - 1) // 8
    if k_total < max_lat:
        return [(0, k_total + 1, 0)]
    assert max_lat >= context + align
    head_new = max_lat - (max_lat - context) % align
    cap_new = max_lat - context
    cap_aligned = align * (cap_new // align)
    remaining = k_total - (head_new - 1)
    n_cont = 1
    while (n_cont - 1) * cap_aligned + cap_new < remaining:
        n_cont += 1
    lo = align * max(-(-max(remaining - cap_new, 0) // align), n_cont - 1)
    hi = align * (min((n_cont - 1) * cap_aligned, remaining - 1) // align)
    want = align * ((remaining - remaining // n_cont) // align)
    constrained = min(max(want, lo), max(hi, lo))
    plan = []
    first = 0
    head = (0, head_new, first)
    first += 1 + 8 * (head_new - 1)
    plan.append(head)
    spread = n_cont - 1
    units = constrained // align
    base, rem = (0, 0) if spread == 0 else (units // spread, units % spread)
    for i in range(spread):
        new = align * (base + (1 if i < rem else 0))
        plan.append((context, new, first))
        first += 8 * new
    last = remaining - constrained
    assert 0 < last <= cap_new, "a plan this grid cannot express"
    plan.append((context, last, first))
    return plan


def emitted(w):
    ctx, new, _ = w
    return 1 + 8 * (new - 1) if ctx == 0 else 8 * new


def decoded(w):
    ctx, new, _ = w
    return 1 + 8 * (ctx + new - 1)


def plan_rows(frames, lh=LTX_LH, lw=LTX_LW, context=CONTEXT, max_tokens=LONGFORM_MAX_TOKENS, align=1):
    plan = window_plan(frames, lh, lw, context, max_tokens, align)
    rows = []
    for i, w in enumerate(plan):
        ctx, new, first = w
        rows.append((i, ctx, new, (ctx + new) * lh * lw, decoded(w), decoded(w) - emitted(w), emitted(w), first))
    return plan, rows


def window_table():
    out = []
    for frames in (121, 265, 601, 1201):
        plan, rows = plan_rows(frames)
        total = sum(r[6] for r in rows)
        assert total == frames
        body = [[r[0], r[1], r[2], _int(r[3]), r[4], r[5], r[6], r[7]] for r in rows]
        out.append(f"{frames} frames ({(frames - 1) // 8} groups of eight after the first):\n\n"
                   + _table(["window", "context", "new", "tokens", "decoded", "dropped", "emitted", "first emitted frame"], body, "rrrrrrrr")
                   + f"\n\nThe emitted frames add to {total}.")
    return ("The plan for a 768 x 512 clip (16 x 24 = 384 tokens per latent frame) under the ceiling of 13200 tokens per pass, so at most 34 latent frames, "
            "with a context of 8 latent frames. 'Decoded' is 1 + 8 (context + new - 1), 'dropped' the pixel frames of the carried context, 'emitted' "
            "1 + 8 (new - 1) for the first window and 8 new for the others.\n\n" + "\n\n".join(out))


def audio_quantum(fps):
    """The smallest n whose n latent frames are a whole number of audio tokens: 200 n divisible by fps."""
    return next(n for n in range(1, fps + 1) if (200 * n) % fps == 0)


def window_plan_sound_table():
    rows = []
    for fps in (24, 25, 30, 16):
        q = audio_quantum(fps)
        plan = window_plan(601, LTX_LH, LTX_LW, CONTEXT, LONGFORM_MAX_TOKENS, q)
        rows.append([fps, f"{200 // math.gcd(200, fps)} / {fps // math.gcd(200, fps)}", q, " + ".join(str(w[1]) for w in plan)])
    return ("For a clip with sound a window that is followed by another must advance by a whole number of audio tokens, n latent frames being 200 n / fps tokens "
            "(25 tokens a second, 8 frames per latent frame). The smallest n for which 200 n / fps is a whole number is the quantum q; the plan of the same 601-frame "
            "clip with every window but the last advancing by a multiple of q is below (new latent frames of each window).\n\n"
            + _table(["fps", "tokens per latent frame", "quantum q", "new latent frames per window"], rows, "rrrl"))


SEAM_NAME = "video_generation.seam"
SEAM_TRIALS = 2000
SEAM_VELOCITY = 1.5
SEAM_NOISE = 0.2
SEAM_PRIOR = 2.0
SEAM_CONTEXTS = (1, 2, 3, 4, 8)


def _slope(ys):
    """Least-squares slope of ys against 0, 1, ..., len(ys) - 1; every sum runs left to right."""
    n = len(ys)
    mx = 0.0
    for i in range(n):
        mx += i
    mx /= n
    my = 0.0
    for y in ys:
        my += y
    my /= n
    sxx = 0.0
    sxy = 0.0
    for i, y in enumerate(ys):
        sxx += (i - mx) * (i - mx)
        sxy += (i - mx) * (y - my)
    return sxy / sxx


def seam_trials():
    """Per trial, in this order: z_0 .. z_7 (the noise of the eight frames the first window made), then z' (the prior draw)."""
    g = bookrng.generator(SEAM_NAME)
    return [([g.normal() for _ in range(8)], g.normal()) for _ in range(SEAM_TRIALS)]


def seam_errors():
    """Mean |velocity the next window adopts - true velocity| for each context length c."""
    sums = {c: 0.0 for c in SEAM_CONTEXTS}
    for z, zp in seam_trials():
        y = [SEAM_VELOCITY * j + SEAM_NOISE * z[j] for j in range(8)]
        for c in SEAM_CONTEXTS:
            v = SEAM_PRIOR * zp if c == 1 else _slope(y[8 - c:])
            sums[c] += abs(v - SEAM_VELOCITY)
    return {c: sums[c] / SEAM_TRIALS for c in SEAM_CONTEXTS}


def folded_mean(mu, sd):
    """E|X| for X ~ N(mu, sd^2): sd sqrt(2/pi) exp(-mu^2 / (2 sd^2)) + mu erf(mu / (sd sqrt(2)))."""
    return sd * math.sqrt(2.0 / math.pi) * math.exp(-mu * mu / (2.0 * sd * sd)) + mu * math.erf(mu / (sd * math.sqrt(2.0)))


def seam_toy():
    err = seam_errors()
    rows = []
    for c in SEAM_CONTEXTS:
        if c == 1:
            pred = folded_mean(SEAM_VELOCITY, SEAM_PRIOR)
        else:
            pred = folded_mean(0.0, SEAM_NOISE * math.sqrt(12.0 / (c * (c * c - 1))))
        rows.append([c, f"{err[c]:.3f}", f"{pred:.3f}", f"{8 * err[c]:.2f}"])
    return (f"A dot moves {SEAM_VELOCITY} cells per latent frame. A window has produced eight latent frames, y_j = {SEAM_VELOCITY} j + {SEAM_NOISE} z_j for j = 0 .. 7. The next window "
            f"is given the last c of them and has to continue the motion. With c = 1 (a still) it sees a position and nothing else, so it draws a velocity {SEAM_PRIOR} z' from its prior; "
            f"with c >= 2 it adopts the slope of the least-squares line through the c frames. The error of a seam is |adopted velocity - {SEAM_VELOCITY}|. Draws from `{SEAM_NAME}` "
            f"(seed {bookrng.seed_of(SEAM_NAME):#018x}): for each of {SEAM_TRIALS} trials, in order, z_0 .. z_7 and then z', each one `normal()` of the book's generator; the means "
            f"add the {SEAM_TRIALS} errors left to right. The prediction is the mean of |X| for the exact distribution of the adopted velocity minus the truth: "
            f"X is normal with mean {SEAM_VELOCITY} and spread {SEAM_PRIOR} for c = 1 (the draw, sign aside), and for c >= 2 normal with mean 0 and spread "
            f"{SEAM_NOISE} sqrt(12 / (c (c^2 - 1))), the spread of a least-squares slope. Eight frames after the seam the dot is off by eight times the velocity error.\n\n"
            + _table(["context frames c", "mean velocity error", "predicted mean", "off after 8 frames"], rows, "rrrr"))


# ------------------------------------------------------------------------------------------------ sound beside the picture

LATENT_RATE = 25


def audio_tokens(frames, fps):
    """round(frames / fps * 25) (ties away from zero), at least 1."""
    return max(1, _round_half_away(frames / fps * LATENT_RATE))


def audio_bounds(i):
    """The seconds [start, end) of audio token i: mel frame max(0, 4 i - 3) times 160 samples over 16000 samples a second."""
    def sec(j):
        return max(0, 4 * j - 3) * 160 / 16000
    return sec(i), sec(i + 1)


def audio_grid_table():
    frames, fps = 25, 24
    ta = audio_tokens(frames, fps)
    rows = []
    for k in range(vid.latent_frames(frames)):
        s, e = latent_frame_bounds(k, fps)
        inside = [i for i in range(ta) if s <= sum(audio_bounds(i)) / 2 < e]
        rows.append([k, f"{s:.4f} .. {e:.4f}", len(inside), f"{inside[0]} .. {inside[-1]}"])
    samples = (4 * ta - 3) * 160
    return (f"A clip of {frames} frames at {fps} frames per second lasts {frames / fps:.4f} seconds and has ta = round({frames}/{fps} * 25) = {ta} audio tokens. Audio token i spans "
            f"max(0, 4 i - 3) / 100 to max(0, 4 (i + 1) - 3) / 100 seconds (token 0 spans 0 .. 0.01, token 1 spans 0.01 .. 0.05, token 2 spans 0.05 .. 0.09, and so on). "
            f"The table counts the audio tokens whose midpoint lies in each latent frame's span.\n\n"
            + _table(["latent frame", "span (s)", "audio tokens", "which"], rows, "rrrr")
            + f"\n\nThe sound decodes to (4 * {ta} - 3) = {4 * ta - 3} mel frames of 160 samples, {samples} samples, {samples / 16000:.4f} seconds at 16000 samples a second: "
            f"{frames / fps - samples / 16000:.4f} seconds shorter than the picture.")


def av_cross_cost():
    t, ta = LTX_TOKENS, audio_tokens(LTX_F, 24)
    dv, da = LTX["d"], LTX_A["d"]
    self_pairs = t * t
    cross_pairs = t * ta
    a2v = 2 * t * dv * da + 2 * (2 * ta * da * da) + 2 * t * da * dv + 4 * t * ta * da
    v2a = 2 * ta * da * da + 2 * (2 * t * dv * da) + 2 * ta * da * da + 4 * t * ta * da
    audio_self = 8 * ta * da * da + 4 * ta * ta * da
    audio_text = 4 * ta * da * da + 4 * LTX["ctx"] * da * da + 4 * ta * LTX["ctx"] * da
    audio_ff = 16 * ta * da * da
    audio_stream = audio_self + audio_text + audio_ff
    video = sum(block_flops(LTX, t, LTX["ctx"]))
    rows = [["video stream", _int(video)], ["audio stream (self, text, feed-forward)", _int(audio_stream)], ["audio to video cross-attention", _int(a2v)],
            ["video to audio cross-attention", _int(v2a)]]
    extra = audio_stream + a2v + v2a
    return (f"The 121-frame, 768 x 512 clip at 24 frames per second has T = {_int(t)} video tokens and ta = {ta} audio tokens (width 2048, 32 heads of 64). "
            f"The video stream's self-attention scores {_int(self_pairs)} pairs per head; each cross-attention direction scores T * ta = {_int(cross_pairs)}, "
            f"{cross_pairs / self_pairs:.3f} of that. FLOPs of one block (query, key, value and output matrices, scores and weighted sum; norms, gates and the "
            f"adaLN tables left out). The shapes are these. Video stream: the block of the attention-split table (width {dv}, feed-forward width 4 * {dv}, {LTX['ctx']} context rows of width {dv}). "
            f"Audio stream, width {da}: self-attention with four {da} x {da} matrices over the {ta} audio tokens and the scores 4 ta^2 {da}; text cross-attention whose q and o "
            f"matrices act on the {ta} audio tokens and whose k and v matrices act on the {LTX['ctx']} context rows, taken at width {da} like the video's (4 * {LTX['ctx']} * {da}^2), with scores 4 ta {LTX['ctx']} {da}; "
            f"a feed-forward of two {da} x 4 * {da} = {da} x {4 * da} matrices. Cross-attention in each direction works at width {da}, the audio width: with the query stream on the left, "
            f"its q matrix maps the query stream's width to {da}, its k and v matrices map the other stream's width to {da}, its o matrix maps {da} back to the query stream's width, and the scores and "
            f"weighted sum cost 4 T ta {da}. The columns below therefore count 2 T {dv} {da} (video q) + 2 (2 ta {da}^2) (audio k, v) + 2 T {da} {dv} (video o) + 4 T ta {da} for audio to video, and "
            f"2 ta {da}^2 (audio q) + 2 (2 T {dv} {da}) (video k, v) + 2 ta {da}^2 (audio o) + 4 T ta {da} for video to audio:\n\n"
            + _table(["part of the block", "FLOPs"], rows, "lr")
            + f"\n\nThe audio stream and both cross-attentions add {extra / video:.3f} of the video stream's FLOPs.")


# ------------------------------------------------------------------------------------------------ parameter counts

def _attn(q, kv, inner, heads=32):
    """The 12 tensors of one attention: q_norm, k_norm (inner each), gate logits (heads x q + heads), q, k, v, out (each with a bias)."""
    return 2 * inner + heads * q + heads + inner * q + inner + 2 * (inner * kv + inner) + q * inner + q


def _ff(dim, bias, mult=4):
    f = mult * dim
    return 2 * dim * f + ((f + dim) if bias else 0)


def _adaln(dim, rows):
    """An adaLN-single: two timestep matrices (256 -> dim, dim -> dim) with biases, then dim -> rows * dim with bias."""
    return 256 * dim + dim + dim * dim + dim + rows * dim * dim + rows * dim


def _connector(dim, layers=8, registers=128):
    return registers * dim + layers * (_attn(dim, dim, dim) + _ff(dim, True))


def ltx_params():
    dv, da, layers = LTX["d"], LTX_A["d"], LTX["layers"]
    video = 2 * _attn(dv, dv, dv) + _ff(dv, False) + 9 * dv + 2 * dv
    audio = 2 * _attn(da, da, da) + _ff(da, True) + 9 * da + 2 * da
    cross = _attn(dv, da, da) + _attn(da, dv, da) + 5 * dv + 5 * da
    top = (dv * 128 + dv) + (da * 128 + da) + (128 * dv + 128) + (128 * da + 128) + 2 * dv + 2 * da + dv
    ada = (_adaln(dv, 9) + _adaln(da, 9) + _adaln(dv, 2) + _adaln(da, 2) + _adaln(dv, 4) + _adaln(da, 4) + _adaln(dv, 1) + _adaln(da, 1))
    conn_v, conn_a = _connector(dv), _connector(da)
    tensors = 59 + layers * 84 + 2 * 129
    return dict(video=layers * video, audio=layers * audio, cross=layers * cross, top=top, ada=ada, conn_v=conn_v, conn_a=conn_a,
                per_block=(video, audio, cross), tensors=tensors,
                total=layers * (video + audio + cross) + top + ada + conn_v + conn_a)


def wan_params(m, in_ch=16, text_dim=4096, freq=256, patch=4, out=16):
    d, f, layers = m["d"], m["ffn"], m["layers"]
    n = d * in_ch * patch + d                       # patch embedding
    n += d * text_dim + d + d * d + d               # text embedding
    n += d * freq + d + d * d + d                   # time embedding
    n += 6 * d * d + 6 * d                          # time projection
    blk = 6 * d + 2 * d                              # modulation, cross-attention norm
    for _ in range(2):
        blk += 4 * (d * d + d) + 2 * d               # q k v o with biases, query and key norms
    blk += f * d + f + d * f + d                     # feed-forward
    n += layers * blk
    n += patch * out * d + patch * out + 2 * d       # head and its modulation
    return n, blk


def gemma_params():
    """Text tower of the Gemma-4 12B config: embeddings, 48 layers (40 sliding, 8 full), final norm; and the two aggregate heads."""
    vocab, h, inter, layers, heads = 262144, 3840, 15360, 48, 16
    total = vocab * h + h
    per_layer_nonembed = 0
    for l in range(layers):
        full = (l + 1) % 6 == 0
        hd = 512 if full else 256
        kvh = 1 if full else 8
        q, kv = heads * hd, kvh * hd
        p = 4 * h + 1 + q * h + kv * h + (0 if full else kv * h) + h * q + 2 * hd + 3 * h * inter
        total += p
        per_layer_nonembed += p
    agg_in = h * (layers + 1)
    heads_params = agg_in * 4096 + 4096 + agg_in * 2048 + 2048
    return total, per_layer_nonembed, heads_params, agg_in


def counts_table():
    ltx = ltx_params()
    g_total, g_layers, g_heads, agg_in = gemma_params()
    w13, b13 = wan_params(WAN13)
    w14, b14 = wan_params(WAN14)
    rows = [
        ["LTX-2.5 video stream (48 blocks)", _int(ltx["video"]), f"{ltx['video'] / ltx['total']:.3f}"],
        ["LTX-2.5 audio stream (48 blocks)", _int(ltx["audio"]), f"{ltx['audio'] / ltx['total']:.3f}"],
        ["LTX-2.5 audio-video cross-attention (48 blocks)", _int(ltx["cross"]), f"{ltx['cross'] / ltx['total']:.3f}"],
        ["LTX-2.5 two connectors, adaLN networks, input and output", _int(ltx["conn_v"] + ltx["conn_a"] + ltx["ada"] + ltx["top"]),
         f"{(ltx['conn_v'] + ltx['conn_a'] + ltx['ada'] + ltx['top']) / ltx['total']:.3f}"],
        ["LTX-2.5 transformer, all of it", _int(ltx["total"]), "1.000"],
        ["Gemma-4 text tower (embedding table + 48 layers)", _int(g_total), ""],
        ["Gemma-4 aggregate heads (188160 -> 4096 and -> 2048)", _int(g_heads), ""],
        ["Wan 1.3B transformer", _int(w13), ""],
        ["Wan 14B transformer", _int(w14), ""],
    ]
    return (f"Parameters counted from the tensor shapes of each model's checkpoint manifest (weights, biases, norm gains and tables; the {_int(ltx['tensors'])} tensors "
            f"of the LTX-2.5 transformer are 59 outside the blocks, 84 in each of 48 blocks, and 129 in each of two connectors). Per LTX block: video stream "
            f"{_int(ltx['per_block'][0])}, audio stream {_int(ltx['per_block'][1])}, cross-attention {_int(ltx['per_block'][2])}. Per Wan block: {_int(b13)} (1.3B) and {_int(b14)} (14B).\n\n"
            + _table(["part", "parameters", "share of the LTX transformer"], rows, "lrr")
            + "\n\n" + counts_inventory())



def counts_inventory():
    """The shape rules that reproduce every row of counts_table; kept next to the functions that apply them."""
    dv, da, layers = LTX["d"], LTX_A["d"], LTX["layers"]
    w = WAN13
    return (
        "The rows follow from these shapes. A *linear* layer of a inputs and b outputs has a b weights and b biases unless it is called bias-free. "
        f"An LTX attention with query width q, key-value width kv and inner width i (i = q for self-attention and text cross-attention; i = {da} for the audio-video attentions) has the gain vectors of its query and "
        f"key norms (2 i), a gate with {LTX['heads']} logits computed from the query stream (q {LTX['heads']} weights and {LTX['heads']} biases), and four linear layers with biases: q (q to i), k and v (kv to i) and the output (i to q). "
        f"A video block ({_int(ltx_params()['per_block'][0])}) holds two such attentions at width {dv} (self-attention, and cross-attention to the {LTX['ctx']} text rows of width {dv}), a bias-free feed-forward of two "
        f"{dv} x {4 * dv} matrices, a 9 x {dv} table and a 2 x {dv} table of modulation values. An audio block holds the same at width {da} with a feed-forward of two {da} x {4 * da} matrices that has biases "
        f"(one of {4 * da} and one of {da}), a 9 x {da} table and a 2 x {da} table. The audio-video cross-attention of a block is one attention with q = {dv}, kv = {da}, i = {da} and one with q = {da}, kv = {dv}, i = {da}, "
        f"plus 5 x {dv} and 5 x {da} modulation values. Outside the blocks: the input projections 128 to {dv} and 128 to {da} and the output projections {dv} to 128 and {da} to 128 (each with a bias), "
        f"the output tables of 2 x {dv}, 2 x {da} and one more row of {dv}, and eight adaLN networks (a timestep embedding 256 to d to d, each linear with a bias, then d to r d with a bias, "
        f"where d is the stream's width and r = 9, 2, 4 and 1 rows for the four networks of each stream). Each of the two connectors has 128 registers of its width and eight blocks, "
        "each an attention at that width (i = q = kv) and a feed-forward with biases of two matrices of width 4 times the width. "
        f"A Wan block (d = {w['d']} and f = {w['ffn']} for 1.3B; d = {WAN14['d']} and f = {WAN14['ffn']} for 14B) holds a 6 x d modulation table, the gain and bias of one more norm (2 d), two attentions "
        "(self and text cross-attention) of four d x d linear layers with biases and the gains of two norms (2 d), and a feed-forward d to f to d with biases; outside the blocks are a patch embedding (16 channels "
        "x 4 pixels of d weights and d biases), a text embedding (4096 to d to d, with biases), a timestep embedding (256 to d to d), a time projection (d to 6 d), and a head (d to 4 x 16 values) with a "
        "2 x d table. "
        "Gemma-4's text tower has 262,144 embedding rows of width 3840 and a final norm of 3840; each of its 48 layers has four norms of 3840, one scalar, "
        "bias-free q, k, v, o matrices (16 query heads; 8 key-value heads of width 256 in a sliding layer, and in a full layer, every sixth, 1 of width 512 with no v matrix), two norms of the head width (q and k) and a "
        "bias-free gated feed-forward of three 3840 x 15360 matrices; the two aggregate heads each have a bias. The tensor counts (4,349 for LTX, 825 for Wan 1.3B) are read from the manifests and are data, "
        "not derived here.")


def tower_work():
    g_total, g_layers, g_heads, agg_in = gemma_params()
    rows = []
    for n in (16, 64, 256, 1024):
        tower = 2 * g_layers * n
        proj_v = 2 * agg_in * 4096 * n
        proj_a = 2 * agg_in * 2048 * n
        rows.append([n, f"{tower / 1e12:.2f}", f"{proj_v / 1e12:.3f}", f"{proj_a / 1e12:.3f}"])
    fwd = forward_flops(LTX, LTX_TOKENS)[0]
    return ("FLOPs of encoding a prompt of n tokens (in units of 10^12): the 48 layers' matrices at 2 FLOPs per weight per token (embedding lookups are free; the attention "
            "among the prompt's own tokens is left out; the tower has 262,144 embedding rows of width 3840, and the layers' own weights are the tower's parameters minus 262,144 * 3840, "
            "so the column is 2 * (those weights) * n; layer shapes in the table of counts), then the aggregate projections 188160 -> 4096 (video) and 188160 -> 2048 (audio) at 2 * 188160 * width per token.\n\n"
            + _table(["prompt tokens n", "text tower", "video head", "audio head"], rows, "rrrr")
            + f"\n\nOne video-stream forward of the 121-frame clip is {fwd / 1e12:.1f} x 10^12 FLOPs, and the prompt is encoded once and reused by all of a clip's forwards.")


# ------------------------------------------------------------------------------------------------ passes and the work of a clip

def passes(cfg=1.0, stg=0.0, modality=1.0):
    """Forwards of one step: the conditional pass, plus one for each term that is switched on."""
    return 1 + (cfg != 1.0) + (stg != 0.0) + (modality != 1.0)


def clip_work_table():
    f_ltx = forward_flops(LTX, LTX_TOKENS)[0]
    f_wan = forward_flops(WAN13, wan_tokens(81, 832, 480))[0]
    rows = []
    for name, steps, ps, f, wbytes in (
            ("LTX-2.5 distilled, one stage, no guidance", 8, passes(), f_ltx, ltx_params()["video"] * 1),
            ("same, with a negative-prompt term (scale 3)", 8, passes(3.0), f_ltx, ltx_params()["video"] * 1),
            ("same, with the reference preset (cfg 3, stg 1, modality 3; needs sound)", 8, passes(3.0, 1.0, 3.0), f_ltx, ltx_params()["video"] * 1),
            ("Wan 1.3B, guidance 5", 50, passes(5.0), f_wan, wan_params(WAN13)[0] * 4)):
        forwards = steps * ps
        rows.append([name, steps, ps, forwards, f"{f / 1e12:.1f}", f"{forwards * f / 1e15:.2f}", f"{forwards * wbytes / 1e9:.0f}"])
    return ("The denoiser's forwards for a clip of about five seconds: LTX-2.5 at 768 x 512, 121 frames, 24 frames per second (6144 tokens); Wan 1.3B at 832 x 480, 81 frames, "
            "16 frames per second (32760 tokens). A step runs one forward for the conditional prediction plus one for each guidance term that is switched on (a negative-prompt term when "
            "the scale is not 1, an STG term when its scale is not 0, an isolated-modality term when its scale is not 1). The forward is the video-stream FLOPs of the table above; "
            "the sound stream and the VAE decode are not counted. The last column assumes every forward reads every weight of the video stream once, at one byte per weight for "
            "LTX-2.5 (8-bit) and four for Wan (32-bit, the default).\n\n"
            + _table(["setup", "steps", "passes per step", "forwards", "FLOPs per forward (10^12)", "FLOPs of the clip (10^15)", "weight bytes read (10^9)"], rows, "lrrrrrr")
            + f"\n\nThe Wan clip needs {rows[3][3] * f_wan / (rows[0][3] * f_ltx):.1f} times the arithmetic of the distilled LTX clip of the same length.")


def latent_budget_table():
    wfr, wh, ww = 81, 480, 832
    wl = 1 + (wfr - 1) // 4
    wvals = 16 * wl * (wh // 8) * (ww // 8)
    wpix = 3 * wfr * wh * ww
    lvals = 128 * LTX_LAT * LTX_PER_FRAME
    lpix = 3 * LTX_F * LTX_H * LTX_W
    rows = [["LTX-2.5", "8 x 32 x 32", 128, _int(LTX_TOKENS), _int(lvals), f"{lpix / lvals:.1f}"],
            ["Wan", "4 x 16 x 16", 64, _int(wan_tokens(81, 832, 480)), _int(wvals), f"{wpix / wvals:.1f}"]]
    return ("What one transformer token stands for in the default clip of each model. LTX-2.5: a token is one latent cell of 128 channels covering 8 frames x 32 x 32 pixels. "
            "Wan: a latent cell of 16 channels covers 4 frames x 8 x 8 pixels and a token is a 2 x 2 patch of cells, 64 values, covering 4 x 16 x 16 pixels. "
            "'RGB / latent' is the RGB values of the clip over its latent values.\n\n"
            + _table(["model", "pixels per token (frames x rows x columns)", "values per token", "tokens", "latent values", "RGB / latent"], rows, "llrrrr"))
