# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Music Generation".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: music.<name> -->` markers; the
functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the table beside
it cannot disagree. Earlier modules are imported, never copied: the table helper of `sound`, the layer and parameter formulas of
`model_cost` (the Qwen3 8B count of Volume VI) and `speech_synthesis` (the per-layer counts of the codec language model).

* the interfaces of a song generator, per second: what each carries as numbers and as bits;
* one autoregressive frame in samples, in latent steps, and how a frame count becomes a latent length;
* how long a song is: frames, codes, latents, chunks and denoiser evaluations;
* the geometry of the chunked denoiser: starts, carried prefixes, crops, and what the stitched song is as long as;
* the clock of the denoiser, closed form against the scheduler's arithmetic, and the blend that holds a carried prefix;
* a chunked flow small enough to follow by hand, run without and with the hold, on an exactly solvable velocity;
* the autoregressive sampler's two-stage restriction on a toy vocabulary, and the prompt contract (the lyrics rewrite and the
  unconditional prompt);
* the parameters of the five components, the multiply-adds of every stage per second of finished music, and the memory each stage
  asks of a card;
* the receptive field of the vocoder in latent steps, and the sizes of the three low-rank adapters.

Arithmetic is the language's 64-bit floating point. The only random numbers are the Gaussian draws of the chunked flow and the one
uniform draw of the sampler, from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.
Nothing is trained, so every digit printed is reproducible to the precision shown.

Swedish Embedded AB implements solutions for generating long audio with the cost of every stage counted before anything is run.
If your team needs expertise in music and long-form audio generation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import re

import bookrng
import model_cost as mc
import sound
import speech_synthesis as ss

_table = sound._table

# ----------------------------------------------------------------------------------------------------------- the constants
SR_AR = 24000            # rate at which the autoregressive stage's frame hop is defined, Hz
HOP_AR = 960             # samples of that rate per autoregressive frame
SR = 44100               # output rate, Hz
HOP_LAT = 512            # samples per latent step (the product of the vocoder's upsampling ratios 8 * 8 * 4 * 2)
FRAME_RATE = SR_AR // HOP_AR       # 25 frames a second
LAT_RATE = SR / HOP_LAT            # 86.1328125 latent steps a second

SEM_VOCAB, RES_VOCAB, N_RES = 16384, 1024, 7        # one semantic code of 14 bits and seven residual codes of 10 bits
N_HID = 1 + N_RES                                    # hidden vectors per frame: the language model's and one per depth step
CODES_PER_FRAME = 1 + N_RES

CHUNK_FRAMES, CHUNK_HOP, OVERLAP = 200, 100, 172     # frames per chunk, frame stride, latent prefix carried between chunks
CROP_L, CROP_R = 86, 258                             # latents cropped off the left of a non-first and the right of a non-last chunk
STEPS, GUIDANCE = 30, 1.7                            # Euler steps per chunk and the denoiser's guidance scale
AR_GUIDANCE, AR_TOPK = 1.5, 50

# the Global LLM: a Qwen3 8B decoder with a 200,000-id vocabulary
LM = dict(V=200000, D=4096, L=36, H=32, hkv=8, dh=128, F=12288)
# the RVQ depth decoder
DD = dict(h=4096, layers=4, inter=6144, heads=16, vocab=RES_VOCAB, books=N_RES + 1, max_pos=16)
# the condition encoder
CE = dict(hidden=4096, layers=8, out=2048, k=3)
# the flow-matching DiT
DIT = dict(cin=128, cdim=2048, layers=36, heads=32, dh=64, ff=8192, rot=32, four=256)
DIT["inner"] = DIT["heads"] * DIT["dh"]
DIT["concat"] = 2 * DIT["cin"] + DIT["cdim"]
# the vocoder
VOC = dict(latent=128, din=1024, dhid=1536, ratios=(8, 8, 4, 2))


def _i(n):
    return f"{int(round(n)):,}"


def _f(x, d=4):
    return f"{x:.{d}f}"


def _g(x, d=3):
    return f"{x:,.{d}f}"


def _seed(name):
    return f"`{name}` (seed 0x{bookrng.seed_of(name):016x})"


def _eng(x, unit, d=2):
    """x written with an SI prefix of `unit` (G, T, P)."""
    scale = {"G": 1e9, "T": 1e12, "P": 1e15}[unit]
    return f"{x / scale:,.{d}f}"


# ------------------------------------------------------------------------------------------ frames, latents and chunk geometry
def latent_length(n_frames):
    """Latent steps for n_frames autoregressive frames: float division at every step, one truncation at the end."""
    return max(1, int(n_frames * SR / SR_AR * HOP_AR / HOP_LAT))


def chunk_starts(n_frames):
    """Frame at which each chunk starts: one chunk for a song of at most 200 frames, else every 100 frames up to the last start
    that leaves more than 100 frames."""
    if n_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, n_frames - CHUNK_HOP, CHUNK_HOP))


def stitch_plan(n_frames):
    """One record per chunk: start, frames, latent length, kept latents [lo, hi) of the chunk, and where its kept part begins in
    the song's latent timeline."""
    starts = chunk_starts(n_frames)
    plan, out_pos = [], 0
    for i, s in enumerate(starts):
        frames = min(s + CHUNK_FRAMES, n_frames) - s
        length = latent_length(frames)
        lo = 0 if i == 0 else CROP_L
        hi = length - (0 if i == len(starts) - 1 else CROP_R)
        plan.append(dict(i=i, start=s, frames=frames, length=length, lo=lo, hi=hi, out=out_pos, kept=hi - lo))
        out_pos += hi - lo
    return plan


# ------------------------------------------------------------------------------------------------ the interfaces, per second
def interface_rates():
    rows = []

    def add(name, steps, per_step, bits, note=""):
        n = steps * per_step
        rows.append((name, f"{steps:,.7f}".rstrip("0").rstrip("."), _i(per_step), f"{bits:g}", _i(n), _i(n * bits)))

    add("waveform, stereo, 16-bit samples", SR, 2, 16)
    add("Flow-VAE latents, 32-bit floats", LAT_RATE, VOC["latent"], 32)
    add("semantic code (one of 16,384)", FRAME_RATE, 1, math.log2(SEM_VOCAB))
    add("residual codes (seven of 1,024 each)", FRAME_RATE, N_RES, math.log2(RES_VOCAB))
    add("hidden states of the language model and the depth steps, 32-bit floats", FRAME_RATE, N_HID * CE["hidden"], 32)
    add("condition for the denoiser, 32-bit floats", LAT_RATE, CE["out"], 32)
    codes = FRAME_RATE * (math.log2(SEM_VOCAB) + N_RES * math.log2(RES_VOCAB))
    return ("Per second of music: steps a second, numbers per step, bits per number, and the products. A code of a codebook of $K$ "
            "entries is $\\log_2 K$ bits; the latent rate is exactly $44100/512$ steps a second.\n\n"
            + _table(["representation", "steps per second", "numbers per step", "bits per number", "numbers per second",
                      "bits per second"], rows, "lrrrrr")
            + f"\n\nThe eight codes of a frame together are {_i(codes)} bits a second, {_i(FRAME_RATE * CODES_PER_FRAME)} integers "
              f"a second.")


# ---------------------------------------------------------------------------------------------------- one frame, in other units
def frame_arithmetic():
    n = list(range(1, 6)) + [200]
    rows = [(str(k), _i(k * HOP_AR), _i(k * HOP_AR * SR // SR_AR), _f(k * SR / SR_AR * HOP_AR / HOP_LAT, 4), str(latent_length(k)))
            for k in n]
    # the nearest-neighbour map from the 5 frames of the first rows to their 17 latent steps
    l_in, l_out = 5, latent_length(5)
    idx = [(i * l_in) // l_out for i in range(l_out)]
    return ("Frames $k$ of 40 milliseconds, in samples of the 24,000 Hz rate that defines the hop, in samples at 44,100 Hz, in "
            "latent steps before truncation, and the latent length after it (the product is formed left to right in 64-bit "
            "floating point, $k \\cdot 44100 / 24000 \\cdot 960 / 512$, and truncated once).\n\n"
            + _table(["frames", "samples at 24,000 Hz", "samples at 44,100 Hz", "latent steps, exact", "latent length"], rows, "rrrrr")
            + f"\n\nThe nearest-neighbour map of 5 frames onto their {l_out} latent steps, $\\lfloor i \\cdot 5 / {l_out} \\rfloor$ "
              f"for $i = 0 \\ldots {l_out - 1}$: {', '.join(str(j) for j in idx)}. Frame 0 is read {idx.count(0)} times, frame 1 "
              f"{idx.count(1)}, frame 2 {idx.count(2)}, frame 3 {idx.count(3)}, frame 4 {idx.count(4)}.")


def integer_division_trap():
    """The latent length with every division truncated on the way, set against the one truncation of the float product."""
    def chained(n):
        return (((n * SR) // SR_AR) * HOP_AR) // HOP_LAT

    rows = [(str(n), str(chained(n)), str(latent_length(n))) for n in range(1, 6)]
    top = 7500
    diff = [n for n in range(1, top + 1) if chained(n) != latent_length(n)]
    worst = max(latent_length(n) - chained(n) for n in range(1, top + 1))
    worst_n = max(range(1, top + 1), key=lambda n: (latent_length(n) - chained(n), -n))
    return ("The latent length of $k$ frames with every division truncated at once (`" "(((k * 44100) // 24000) * 960) // 512`" "), "
            "against the one truncation of the left-to-right float product.\n\n"
            + _table(["frames", "truncated at each division", "one truncation"], rows, "rrr")
            + f"\n\nOf the frame counts from 1 to {top:,}, {len(diff):,} give a different length, by at most {worst} steps "
              f"(first at {worst_n} frames: {chained(worst_n)} against {latent_length(worst_n)}).")


# ----------------------------------------------------------------------------------------------------------------- song length
def song_lengths():
    rows = []
    for secs in (10, 30, 60, 180, 240, 300):
        f = round(secs * FRAME_RATE)
        starts = chunk_starts(f)
        rows.append((str(secs), _i(f), _i(f * CODES_PER_FRAME), _i(secs * SR), _i(latent_length(f)), str(len(starts)),
                     _i(len(starts) * STEPS * 2)))
    return ("A song of a given length, counted as the model sees it: frames at 25 a second, integers the autoregressive stage "
            "writes (8 a frame), samples per channel at 44,100 Hz, latent steps, denoiser chunks, and denoiser evaluations "
            "(30 steps of 2 evaluations per chunk).\n\n"
            + _table(["seconds", "frames", "integers written", "samples per channel", "latent steps", "chunks", "evaluations"],
                     rows, "rrrrrrr"))


def chunk_rule():
    """The rule that turns a frame count into chunks, with the cases at its edges."""
    rows = []
    for f in (1, 200, 201, 300, 301, 450, 7500):
        plan = stitch_plan(f)
        last = plan[-1]
        rows.append((f"{f:,}", str(len(plan)), ", ".join(str(p["start"]) for p in plan[:3]) + (", ..." if len(plan) > 3 else ""),
                     str(last["frames"]), str(last["length"]), f"{last['lo']} to {last['hi']}"))
    return (f"A song of $F$ frames is one chunk when $F \\le {CHUNK_FRAMES}$, and otherwise "
            f"$\\lceil (F - {CHUNK_FRAMES}) / {CHUNK_HOP} \\rceil + 1$ chunks, starting at the frames $0, {CHUNK_HOP}, "
            f"{2 * CHUNK_HOP}, \\ldots$ up to the last start that is below $F - {CHUNK_HOP}$ (the starts are "
            f"`range(0, F - {CHUNK_HOP}, {CHUNK_HOP})`). Chunk $i$ has $\\min({CHUNK_FRAMES}, F - s_i)$ frames, so the last chunk has "
            f"more than {CHUNK_HOP} and at most {CHUNK_FRAMES} frames, and its latent length is the truncated product of the first "
            f"table. Every chunk but the first drops its first {CROP_L} latent steps and every chunk but the last drops its last "
            f"{CROP_R}; the last chunk therefore keeps steps {CROP_L} up to its own length. A song of {CHUNK_FRAMES} frames or "
            "fewer is a single chunk with no crop at all.\n\n"
            + _table(["frames $F$", "chunks", "first starts", "frames of the last chunk", "latents of the last chunk",
                      "kept of the last chunk"], rows, "rrlrrr"))


def chunk_geometry(n_frames=450):
    plan = stitch_plan(n_frames)
    rows = []
    for p in plan:
        carried = "-" if p["i"] == 0 else str(min(OVERLAP, p["length"]))
        rows.append((str(p["i"]), _i(p["start"]), _i(p["frames"]), _i(p["length"]), carried, _i(p["lo"]), _i(p["hi"]),
                     _i(p["kept"]), _i(p["out"] * HOP_LAT), _f(p["out"] * HOP_LAT / SR, 4)))
    total = sum(p["kept"] for p in plan)
    nominal = n_frames * HOP_AR * SR // SR_AR
    seams = ", ".join(_f(p["out"] * HOP_LAT / SR, 4) for p in plan[1:])
    joins = ", ".join(_f((p["out"] + min(OVERLAP, p["length"]) - p["lo"]) * HOP_LAT / SR, 4) for p in plan[1:])
    return (f"The chunks of a song of {n_frames} frames ({n_frames / FRAME_RATE:g} seconds). `carried` is the latent prefix "
            "the chunk takes over from the previous one, `kept` the span of its own latents that reaches the song, and the last "
            "two columns where that span starts in the song's samples and seconds.\n\n"
            + _table(["chunk", "start frame", "frames", "latents", "carried", "first kept", "end kept", "kept", "first sample",
                      "second"], rows, "rrrrrrrrrr")
            + f"\n\nThe kept spans add up to {_i(total)} latent steps, {_i(total * HOP_LAT)} samples per channel; "
              f"{n_frames} frames of 1,764 samples are {_i(nominal)}. The seams, where one chunk's kept stretch ends and the next "
              f"begins, are at {seams} seconds; the joins, where a chunk's prefix ends and its own generation begins, "
              f"{CROP_L} steps after the seams, are at {joins} seconds.")


def length_drift():
    rows = []
    for secs in (10, 60, 240, 300):
        f = round(secs * FRAME_RATE)
        plan = stitch_plan(f)
        kept = sum(p["kept"] for p in plan) * HOP_LAT
        nominal = f * HOP_AR * SR // SR_AR
        rows.append((str(secs), str(len(plan)), _i(kept), _i(nominal), f"{kept - nominal:+,}", _f((kept - nominal) / SR, 4)))
    mid = CROP_L, CROP_R
    kept_mid = latent_length(CHUNK_FRAMES) - sum(mid)
    return ("The stitched length against the frame count's nominal length, in samples per channel. A chunk that is neither "
            f"first nor last keeps {latent_length(CHUNK_FRAMES)} - {CROP_L} - {CROP_R} = {kept_mid} latent steps, "
            f"{kept_mid * HOP_LAT:,} samples, while its frame stride of {CHUNK_HOP} is {CHUNK_HOP * HOP_AR * SR // SR_AR:,}.\n\n"
            + _table(["seconds", "chunks", "samples stitched", "samples nominal", "difference", "difference in seconds"], rows,
                     "rrrrrr"))


def chunk_constants():
    full = latent_length(CHUNK_FRAMES)
    mid = full - CROP_L - CROP_R
    rows = [("chunk (on the frame clock)", f"{CHUNK_FRAMES} frames", f"{full}", CHUNK_FRAMES / FRAME_RATE),
            ("hop between chunk starts (on the frame clock)", f"{CHUNK_HOP} frames", f"{CHUNK_HOP * HOP_AR * SR / SR_AR / HOP_LAT:.4f}", CHUNK_HOP / FRAME_RATE),
            ("carried prefix", f"{OVERLAP * HOP_LAT / (HOP_AR * SR / SR_AR):.3f} frames", f"{OVERLAP}", OVERLAP / LAT_RATE),
            ("crop off the left of a later chunk", f"{CROP_L * HOP_LAT / (HOP_AR * SR / SR_AR):.3f} frames", f"{CROP_L}", CROP_L / LAT_RATE),
            ("crop off the right of an earlier chunk", f"{CROP_R * HOP_LAT / (HOP_AR * SR / SR_AR):.3f} frames", f"{CROP_R}", CROP_R / LAT_RATE),
            ("kept from a chunk with a neighbour on each side", f"{mid * HOP_LAT / (HOP_AR * SR / SR_AR):.3f} frames", f"{mid}", mid / LAT_RATE)]
    body = [(n, fr, lat, _f(sec, 4), _i(round(sec * SR))) for n, fr, lat, sec in rows]
    return ("The constants of the chunked denoiser in the units of the frame, the latent step and the clock. A latent step is "
            "$512/44100$ of a second; a frame is $1764/512 = 441/128 = 3.4453125$ latent steps. The chunk and the hop are "
            "defined in frames, and their seconds and samples are the frame clock's, nominal; the other four rows are defined in "
            f"latent steps, and their seconds and samples are the latent clock's. The {full} steps of a chunk are "
            f"{full * HOP_LAT:,} samples, {full * HOP_LAT / SR:.4f} seconds, {CHUNK_FRAMES * HOP_AR * SR // SR_AR - full * HOP_LAT} "
            "samples short of the nominal 200 frames, because the latent length is truncated; the hop's 344.5312 steps are "
            "fractional, and the carried slice is placed at a whole step.\n\n"
            + _table(["constant", "in frames", "in latent steps", "seconds", "samples at 44,100 Hz"], body, "lrrrr"))


def handoff_sizes():
    secs = 300
    f = secs * FRAME_RATE
    plan = stitch_plan(f)
    lat = sum(p["length"] for p in plan)
    rows = [("hidden vectors kept for the condition encoder", f * N_HID * CE["hidden"], 4),
            ("latents of every chunk, held until the vocoder runs", lat * VOC["latent"], 4),
            ("the finished waveform, 32-bit floats, two channels", secs * SR * 2, 4)]
    body = [(n, _i(c), str(w), _i(c * w), _g(c * w / 1e6, 1)) for n, c, w in rows]
    return (f"What the stages hand to one another for a song of {secs} seconds ({f:,} frames, {len(plan)} chunks): the count of "
            "numbers, bytes per number as the host holds them, and the total. The codes themselves are not on the list: the loop writes "
            "them, feeds them back and forgets them, and returns only the hidden vectors.\n\n"
            + _table(["what", "numbers", "bytes each", "bytes", "megabytes"], body, "lrrrr"))


# --------------------------------------------------------------------------------------------------------- the denoiser's clock
def denoise_clock(n=STEPS):
    s_in = [1.0 + (1.0 / n - 1.0) * i / (n - 1) for i in range(n)]      # linspace(1, 1/n, n)
    t = [1.0 - s for s in s_in]                                           # the inverted, unshifted schedule (shift = 1)
    t_all = t + [1.0]                                                     # the appended terminal value
    assert max(abs(t[i] - i / n) for i in range(n)) < 1e-15
    dts = [t_all[i + 1] - t_all[i] for i in range(n)]
    assert max(abs(d - 1 / n) for d in dts) < 1e-15
    show = [0, 1, 2, 3, n - 2, n - 1]
    rows = []
    for i in show:
        wn = 1.0 - (1.0 - 1e-6) * t[i]
        rows.append((str(i), _f(t[i], 6), _f(i / n, 6), _f(dts[i], 6), _f(wn, 6), _f(t[i], 6)))
    return (f"The clock of the denoiser for {n} steps: the input sigmas are linspace(1, 1/{n}, {n}), the shift is 1, every sigma "
            f"is replaced by 1 - sigma and the terminal value 1 is appended. The closed form is $t_i = i/{n}$; the largest "
            f"difference between the two over the {n} steps is below 1e-15 (floating-point rounding only). The last three columns are, at each step, the "
            f"step $\\Delta t = t_{{i+1}} - t_i$, and the weights of the noise and of the carried data in the blend "
            f"$(1 - (1 - 10^{{-6}}) t)\\,\\varepsilon + t\\,x_{{\\mathrm{{prev}}}}$ that holds a carried prefix.\n\n"
            + _table(["step i", "t_i", f"i / {n}", "Delta t", "weight of the noise", "weight of the carried data"], rows, "rrrrrr")
            + f"\n\nThe {n} steps all have $\\Delta t = 1/{n} = {1 / n:.6f}$ (the largest deviation is below 1e-15).")


# --------------------------------------------------------------------------------------------------------- the chunked flow toy
# A song of 40 latent steps in 2 channels, in chunks of L = 16 steps. Given the condition, the song is the melody m (known at every
# position) plus a texture z whose steps are correlated: each channel is a Gaussian process with covariance S[i][j] =
# SIGMA^2 RHO^|i - j|, the two channels independent. The flow is x_t = (1 - t) noise + t x_1, and the exact velocity of the chunk
# follows from the Gaussian posterior:  E[x_1 | x_t] = m + t S (t^2 S + (1 - t)^2 I)^-1 (x_t - t m),
# v(x_t, t) = (E[x_1 | x_t] - x_t) / (1 - t).
TOY = dict(C=2, L=16, n_song=40, steps=30, sigma=0.5, rho=0.9)
TOY_NAME = "music.chunk_flow_toy"


def _geometry(hop):
    """Prefix and crops of a toy chunk of L = 16 steps and a given hop, by the rules of the real constants (172 = (689 - 345) / 2,
    86 = 172 / 2, 258 = 2 x 172 - 86): the prefix is (L - hop) / 2, the left crop half of it, the right crop the rest of 2 prefixes."""
    pre = (TOY["L"] - hop) // 2
    return pre, pre // 2, 2 * pre - pre // 2


def _toy_starts(hop):
    """Start of every chunk that fits whole in the song (the real rule also admits a shorter last chunk)."""
    return list(range(0, TOY["n_song"] - TOY["L"] + 1, hop))


def _toy_melody():
    n = TOY["n_song"]
    return [[0.6 * math.sin(2 * math.pi * g / 10 + 0.5 * c) for g in range(n)] for c in range(TOY["C"])]


def _solve_inverse(A):
    """Gauss-Jordan inverse with partial pivoting (the matrix is symmetric positive definite, so no zero pivot arises)."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    return [row[n:] for row in M]


def _toy_filters():
    """W_i = t S (t^2 S + (1 - t)^2 I)^-1 for t = i / steps, i = 0 .. steps - 1: the smoother of the posterior mean."""
    L, n = TOY["L"], TOY["steps"]
    S = [[TOY["sigma"] ** 2 * TOY["rho"] ** abs(i - j) for j in range(L)] for i in range(L)]
    out = []
    for k in range(n):
        t = k / n
        A = [[t * t * S[i][j] + ((1 - t) ** 2 if i == j else 0.0) for j in range(L)] for i in range(L)]
        inv = _solve_inverse(A)
        out.append([[t * sum(S[i][q] * inv[q][j] for q in range(L)) for j in range(L)] for i in range(L)])
    return out


_FILTERS = []


def _toy_draw(gen, n_chunks):
    """n_chunks of 2 x 16 Gaussians, in chunk order, each channel-major."""
    return [[[gen.normal() for _ in range(TOY["L"])] for _ in range(TOY["C"])] for _ in range(n_chunks)]


def _toy_run(hold, hop, noises, clean=False):
    """Generate the chunks of the song in order, with or without the hold of the carried prefix (`clean` holds it at its carried
    values at every step instead of renoising it). Returns the final latents of each chunk, [C][L]."""
    if not _FILTERS:
        _FILTERS.extend(_toy_filters())
    C, L, n = TOY["C"], TOY["L"], TOY["steps"]
    pre, _, _ = _geometry(hop)
    m = _toy_melody()
    prev, out = None, []
    for k, start in enumerate(_toy_starts(hop)):
        x = [row[:] for row in noises[k]]
        noise_prefix = [row[:pre] for row in x]
        for i in range(n):
            t = i / n
            if hold and prev is not None:
                for c in range(C):
                    for j in range(pre):
                        x[c][j] = prev[c][j] if clean else (1 - (1 - 1e-6) * t) * noise_prefix[c][j] + t * prev[c][j]
            W = _FILTERS[i]
            for c in range(C):
                d = [x[c][j] - t * m[c][start + j] for j in range(L)]
                est = [m[c][start + j] + sum(W[j][q] * d[q] for q in range(L)) for j in range(L)]
                x[c] = [x[c][j] + (est[j] - x[c][j]) / (1 - t) / n for j in range(L)]
        if hold and prev is not None:
            for c in range(C):
                x[c][:pre] = prev[c][:]
        out.append(x)
        prev = [row[L - 2 * pre: L - pre] for row in x]
    return out


def toy_stitch(hold, hop=8, noises=None, clean=False):
    """The song of the toy: the kept span of every chunk, concatenated. Returns [C][n_song], and the positions of the seams (where
    two kept spans meet) and of the joins (where the free part of a chunk begins after its held prefix)."""
    C, L = TOY["C"], TOY["L"]
    starts = _toy_starts(hop)
    noises = noises or _toy_draw(bookrng.generator(TOY_NAME), len(starts))
    chunks = _toy_run(hold, hop, noises, clean)
    pre, cl, cr = _geometry(hop)
    song = [[] for _ in range(C)]
    seams, joins = [], []
    for k, ch in enumerate(chunks):
        lo = 0 if k == 0 else cl
        hi = L - (0 if k == len(chunks) - 1 else cr)
        if k > 0:
            seams.append(len(song[0]))
            joins.append(starts[k] + pre)
        for c in range(C):
            song[c].extend(ch[c][lo:hi])
    return song, seams, joins


def _residual_steps(song, seams, joins):
    """Mean squared step of the texture (song minus melody) at the boundary positions (a seam or a join: the step from the sample
    before to the position) and at all other positions, over both channels."""
    m = _toy_melody()
    bnd = set(seams) | set(joins)
    near, far = [], []
    for c in range(TOY["C"]):
        r = [song[c][g] - m[c][g] for g in range(len(song[c]))]
        for g in range(1, len(r)):
            (near if g in bnd else far).append((r[g] - r[g - 1]) ** 2)
    return sum(near) / len(near), sum(far) / len(far)


def chunk_flow_toy():
    hop = 8
    pre, cl, cr = _geometry(hop)
    m = _toy_melody()
    rows = []
    for hold in (False, True):
        song, seams, joins = toy_stitch(hold, hop)
        near, far = _residual_steps(song, seams, joins)
        r0 = [song[0][g] - m[0][g] for g in range(TOY["n_song"])]
        rows.append(("hold the carried prefix" if hold else "independent chunks",
                     *[_f(r0[g], 3) for g in range(seams[0] - 2, seams[0] + 4)], _f(math.sqrt(near), 3), _f(math.sqrt(far), 3)))
    seam1 = toy_stitch(False, hop)[1][0]
    data_step = TOY["sigma"] * math.sqrt(2 * (1 - TOY["rho"]))
    steps = {}
    for hold in (False, True):
        song, seams, joins = toy_stitch(hold, hop)
        r0 = [song[0][g] - m[0][g] for g in range(TOY["n_song"])]
        steps[hold] = ([abs(r0[g] - r0[g - 1]) for g in seams], [abs(r0[g] - r0[g - 1]) for g in joins])
    return (f"The chunked flow of the chapter's toy: a song of {TOY['n_song']} latent steps in {TOY['C']} channels, chunks of "
            f"{TOY['L']} steps, a hop of {hop} (chunks start at 0, 8, 16 and 24), a carried prefix of {pre}, crops of {cl} on "
            f"the left and {cr} on the right, and {TOY['steps']} Euler steps with $t_i = i/{TOY['steps']}$. Given the "
            "melody $m_c(g) = 0.6 \\sin(2\\pi g / 10 + 0.5c)$ of channel $c$, the texture $z = x_1 - m$ of each channel of a "
            f"chunk is a Gaussian process with covariance $\\Sigma_{{ij}} = {TOY['sigma']}^2\\, {TOY['rho']}^{{|i-j|}}$ (neighbouring "
            "steps are correlated), the two channels independent, and the velocity is the exact one of the posterior. The noise "
            f"is {_seed(TOY_NAME)}: four chunks in order, each 2 x 16 draws of `normal()`, channel 0's 16 first; both runs use "
            "the same draws. The table prints the texture of channel 0 at the six song positions "
            f"{seam1 - 2} to {seam1 + 3} around the first seam (the seam is at position {seam1}: the last kept step of "
            "chunk 0 is position "
            f"{seam1 - 1}, and the held prefix of chunk 1 ends, its join, at position {seam1 + pre - cl}), and the root mean square step of the texture (both channels) at the seams and at the joins against "
            f"all other positions. The texture of the data itself has a step of {data_step:.3f} (rms).\n\n"
            + _table(["run"] + [f"z at {seam1 - 2 + k}" for k in range(6)] + ["rms step at seams and joins", "rms step elsewhere"],
                     rows, "lrrrrrrrr")
            + "\n\nThe absolute step of channel 0 across the three seams, and across the three joins (the step into the first "
              "position that the chunk generated for itself), for each run:\n\n"
            + _table(["run", "seam 1", "seam 2", "seam 3", "join 1", "join 2", "join 3"],
                     [("independent chunks", *[_f(v, 3) for v in steps[False][0]], *[_f(v, 3) for v in steps[False][1]]),
                      ("hold the carried prefix", *[_f(v, 3) for v in steps[True][0]], *[_f(v, 3) for v in steps[True][1]])],
                     "lrrrrrr"))


def chunk_flow_trials(trials=100):
    rows = []
    for hop in (12, 8, 4):
        pre, _, _ = _geometry(hop)
        n_chunks = len(_toy_starts(hop))
        gen_name = f"{TOY_NAME}.trials.hop{hop}"
        ratios = []
        for hold, clean in ((False, False), (True, False), (True, True)):
            gen = bookrng.generator(gen_name)
            near = far = 0.0
            for _ in range(trials):
                song, seams, joins = toy_stitch(hold, hop, _toy_draw(gen, n_chunks), clean)
                a, b = _residual_steps(song, seams, joins)
                near += a
                far += b
            ratios.append(math.sqrt(near / trials) / math.sqrt(far / trials))
        rows.append((str(hop), str(pre), str(n_chunks), str(n_chunks * TOY["steps"]), *[_f(r, 2) for r in ratios]))
    return (f"The same toy over {trials} songs for three hops (and so three prefix lengths, $(16 - \\mathrm{{hop}})/2$). "
            "For each song the mean squared step of the texture at the seams and joins is set against the mean squared step "
            "elsewhere; the table gives the ratio of the square roots of their averages over the songs (1 would be a boundary "
            "no one could find). The third run holds the carried prefix at its carried values at every step instead of "
            "renoising it. The noise of each hop is its own stream, "
            f"`{TOY_NAME}.trials.hop<hop>` (seeds "
            + ", ".join(f"0x{bookrng.seed_of(f'{TOY_NAME}.trials.hop{h}'):016x}" for h in (12, 8, 4))
            + ", for hops 12, 8 and 4), songs one after the other, chunks in order, draws as above; the three runs of a hop "
              "replay the same draws. Cost counts denoiser steps for the whole song (one evaluation per step).\n\n"
            + _table(["hop", "prefix", "chunks", "steps", "independent chunks", "hold, renoised", "hold, clean"], rows,
                     "rrrrrrr"))


# -------------------------------------------------------------------------------------------------- the sampler on a toy vocabulary
def ar_sampling_toy():
    name = "music.ar_sampling_toy"
    cond = [2.0, 3.1, 0.2, 4.0, 3.9, 1.0, 2.6, 3.0]
    unc = [2.4, 2.0, 1.0, 3.0, 3.7, 1.2, 0.4, 3.4]
    k, g = 3, AR_GUIDANCE
    guided = [u + (c - u) * g for c, u in zip(cond, unc)]
    thr = sorted(cond, reverse=True)[k - 1]
    kept = [guided[i] if cond[i] >= thr else -math.inf for i in range(len(cond))]
    thr2 = sorted([v for v in kept if v > -math.inf], reverse=True)[k - 1]
    final = [v if v >= thr2 else -math.inf for v in kept]
    mx = max(v for v in final if v > -math.inf)
    e = [math.exp(v - mx) if v > -math.inf else 0.0 for v in final]
    s = sum(e)
    p = [x / s for x in e]
    u = bookrng.generator(name).uniform()
    cum, pick = 0.0, len(p) - 1
    for i, pi in enumerate(p):
        cum += pi
        if u < cum:
            pick = i
            break
    rows = []
    for i in range(len(cond)):
        rows.append((str(i), _f(cond[i], 2), _f(unc[i], 2), _f(guided[i], 2), "yes" if cond[i] >= thr else "no",
                     "-inf" if final[i] == -math.inf else _f(final[i], 2), _f(p[i], 4)))
    top_guided = sorted(range(len(guided)), key=lambda i: -guided[i])[:k]
    return (f"One draw of the autoregressive sampler on a vocabulary of 8 codes, with the top-$k$ of the real sampler shrunk to "
            f"$k = {k}$ (the code uses 50) and the guidance scale {g} of the code. Logits of the conditional and of the "
            f"unconditional prompt; guided = unconditional + {g} (conditional - unconditional); the conditional top-{k} threshold "
            f"is {thr:g}. The $k$ largest of the *unrestricted* guided logits would have been codes "
            f"{', '.join(str(i) for i in sorted(top_guided))}; after the conditional restriction the final top-$k$ keeps "
            f"{', '.join(str(i) for i in range(len(final)) if final[i] > -math.inf)}. The softmax of the survivors has no "
            f"temperature. The uniform draw is the first of {_seed(name)}, $u = {u:.6f}$, and the inverse CDF returns code "
            f"{pick}.\n\n"
            + _table(["code", "conditional", "unconditional", "guided", "in the conditional top-3", "kept after the second top-3",
                      "probability"], rows, "rrrrrrr"))


# --------------------------------------------------------------------------------------------------------- the prompt contract
_TAGS = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def normalize_lyrics(lyrics):
    """The lyrics rewrite: a line that begins with bracket tags keeps only those tags; then ']' + space and space + '[' become a
    newline-separated form, ' ^ ' a newline, the content of every bracket is lowercased, and the text opens with a start tag."""
    out = []
    for line in lyrics.split("\n"):
        m = _TAGS.match(line)
        out.append(m.group(1).strip() if m else line)
    text = "\n".join(out).replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda mo: "[" + mo.group(1).lower() + "]", text)
    return "[start]\n" + text


def _selfcheck():
    cases = {
        "": "[start]\n",
        "[verse]\nHello world": "[start]\n[verse]\nHello world",
        "[Verse 1] some lyrics here": "[start]\n[verse 1]",
        "[verse]   [chorus]  \ntext": "[start]\n[verse]\n \n[chorus]\ntext",
        "a ] b [ c ^ d": "[start]\na ]\nb\n[ c\nd",
        "[a][b][c] trailing text dropped": "[start]\n[a][b][c]",
    }
    for src, want in cases.items():
        assert normalize_lyrics(src) == want, (src, normalize_lyrics(src), want)


_selfcheck()


def prompt_contract():
    lyrics = "[Verse 1] Quiet morning light\nfading into you\n[Chorus]\nHold on to this feeling"
    caption = "warm acoustic ballad, 80 BPM"
    norm = normalize_lyrics(lyrics)
    tokens = ["<|im_start|>", "<|caption_start|>", "[caption text]", "<|caption_end|>", "<|lyrics_start|>", "[lyrics text]",
              "<|lyrics_end|>", "<|im_end|>", "<|audio_start|>"]
    cond_ids = len(tokens)
    rows = []
    for i, tk in enumerate(tokens):
        rows.append((str(i), tk, tk if i in (0, len(tokens) - 2, len(tokens) - 1) else "<audio-cfg id 151654>"))
    toks = [("[start]\\n[verse 1]\\n[chorus]\\nhold..."), ]
    return ("The lyrics `" + lyrics.replace("\n", "\\n") + "` after the rewrite (a line that opens with a tag keeps only its tags, "
            "so the words after `[Verse 1]` on that line are dropped; tags are lowercased; the text opens with `[start]`):\n\n"
            "```text\n" + norm + "\n```\n\n"
            "The conditional prompt is `<|im_start|><|caption_start|>`caption`<|caption_end|><|lyrics_start|>`lyrics`"
            "<|lyrics_end|><|im_end|><|audio_start|>`, and the unconditional prompt has the same length with every token except "
            "the first and the last two replaced by one placeholder id. With each part of the toy standing for one token "
            f"(a real prompt has as many tokens as its caption and lyrics tokenise to), the {cond_ids} positions are:\n\n"
            + _table(["position", "conditional prompt", "unconditional prompt"], rows, "rll"))


# ------------------------------------------------------------------------------------------------------- parameters and work
def _lm_layer_matrix_params():
    d, h, hk, dh, f = LM["D"], LM["H"], LM["hkv"], LM["dh"], LM["F"]
    return 2 * d * h * dh + 2 * d * hk * dh + 3 * d * f


def lm_params():
    return mc.parameters_formula(LM["V"], LM["D"], LM["L"], LM["H"], LM["hkv"], LM["dh"], LM["F"], untied=True, qk_norm=True)


def _prod(dims):
    n = 1
    for x in dims:
        n *= x
    return n


def _shape(dims):
    return " x ".join(f"{x:,}" for x in dims)


def lm_rows():
    """The tensors of the Global LLM as (name, dims, copies): the Qwen3 layout of Volume VI at a vocabulary of 200,000."""
    d, h, hk, dh, f, v, n = LM["D"], LM["H"], LM["hkv"], LM["dh"], LM["F"], LM["V"], LM["L"]
    return [("token embedding table", (v, d), 1), ("output head (untied)", (v, d), 1), ("final RMSNorm gain", (d,), 1),
            ("W_Q, per layer", (h * dh, d), n), ("W_K, per layer", (hk * dh, d), n), ("W_V, per layer", (hk * dh, d), n),
            ("W_O, per layer", (d, h * dh), n), ("query and key norm gains, per layer", (2, dh), n),
            ("W_gate, per layer", (f, d), n), ("W_up, per layer", (f, d), n), ("W_down, per layer", (d, f), n),
            ("two RMSNorm gains, per layer", (2, d), n)]


def dd_rows():
    """The tensors of the RVQ depth decoder as (name, dims, copies); every linear map is bias-free."""
    h, i, v, n = DD["h"], DD["inter"], DD["vocab"], DD["layers"]
    return [("audio embedding tables, one per residual codebook", (v, h), N_RES),
            ("output heads, one per residual codebook", (v, h), N_RES),
            ("input projection, one matrix applied to every input row", (h, h), 1),
            ("position table", (DD["max_pos"], h), 1), ("final RMSNorm gain", (h,), 1),
            ("attention W_q, W_k, W_v, W_out, per layer", (h, h), 4 * n),
            ("W_gate and W_up, per layer", (i, h), 2 * n), ("W_down, per layer", (h, i), n),
            ("two RMSNorm gains, per layer", (2, h), n)]


def ce_rows():
    """The condition encoder as (name, dims, copies)."""
    return [("softmax logits over the 8 hidden vectors of a frame", (CE["layers"],), 1), ("scale", (1,), 1),
            ("convolution weight, 3 taps along the frame axis", (CE["out"], CE["hidden"], CE["k"]), 1),
            ("convolution bias", (CE["out"],), 1)]


def dit_block_rows():
    """The tensors of one DiT block as (name, dims, copies)."""
    d, ff = DIT["inner"], DIT["ff"]
    return [("attention W_q, W_k, W_v, W_out (no bias)", (d, d), 4),
            ("two LayerNorms, each a gain and a bias", (2, 2, d), 1),
            ("up and gate projections, one stored matrix", (2 * ff, d), 1), ("their bias", (2 * ff,), 1),
            ("down projection", (d, ff), 1), ("its bias", (d,), 1)]


def dit_rows():
    """The tensors of the DiT as (name, dims, copies): the time embedding, the input path, 36 blocks, the output path."""
    d, c = DIT["inner"], DIT["concat"]
    rows = [("random-Fourier weights of the time embedding", (DIT["four"] // 2,), 1),
            ("time MLP, first layer weight (256 sin and cos inputs)", (d, DIT["four"]), 1), ("its bias", (d,), 1),
            ("time MLP, second layer weight", (d, d), 1), ("its bias", (d,), 1),
            ("input convolution, 1 tap, 2,304 to 2,304 (no bias)", (c, c), 1),
            ("input projection, 2,304 to 2,048 (no bias)", (d, c), 1)]
    rows += [(f"per block: {n}", dims, copies * DIT["layers"]) for n, dims, copies in dit_block_rows()]
    rows += [("output projection, 2,048 to 128 (no bias)", (DIT["cin"], d), 1),
             ("output convolution, 1 tap, 128 to 128 (no bias)", (DIT["cin"], DIT["cin"]), 1)]
    return rows


def _rows_sum(rows):
    return sum(_prod(dims) * copies for _, dims, copies in rows)


def lm_params():
    total = _rows_sum(lm_rows())
    assert total == mc.parameters_formula(LM["V"], LM["D"], LM["L"], LM["H"], LM["hkv"], LM["dh"], LM["F"], untied=True,
                                          qk_norm=True)
    return total


def dd_params():
    return _rows_sum(dd_rows())


def ce_params():
    return _rows_sum(ce_rows())


def dit_params():
    return _rows_sum(dit_rows())


def dd_macs_per_position():
    h, i = DD["h"], DD["inter"]
    return DD["layers"] * (4 * h * h + 3 * h * i)


def voc_layout():
    """(name, formula, parameters, multiply-adds per output position of the layer's own resolution, ratio of that resolution to the
    latent rate) for every layer of the decoder, one stereo stream (the two streams share the weights). Every convolution has
    a bias; a Snake has one alpha per channel; a residual unit is Snake, 7-tap convolution at its dilation, Snake, 1-tap
    convolution, and the sum with the unit's input."""
    layers = []
    half = VOC["latent"] // 2
    cur, up = VOC["dhid"], 1
    layers.append(("input projection (1x1)", f"{VOC['din']:,} x {half} + {VOC['din']:,}", half * VOC["din"] + VOC["din"],
                   half * VOC["din"], 1))
    layers.append(("first convolution (k=7)", f"{VOC['din']:,} x {cur:,} x 7 + {cur:,}", VOC["din"] * cur * 7 + cur,
                   VOC["din"] * cur * 7, 1))
    for b, s in enumerate(VOC["ratios"]):
        out = cur // 2
        layers.append((f"block {b}: Snake", f"{cur:,} alphas", cur, 0, up))
        layers.append((f"block {b}: transposed convolution (k={2 * s}, stride {s})",
                       f"{cur:,} x {out:,} x {2 * s} + {out:,}", cur * out * 2 * s + out, cur * out * 2 * s, up))
        up *= s
        for u, d in enumerate((1, 3, 9)):
            layers.append((f"block {b}, unit {u} (dilation {d})", f"2 x {out:,} + ({out:,}^2 x 7 + {out:,}) + ({out:,}^2 + {out:,})",
                           2 * out + out * out * 7 + out + out * out + out, out * out * 7 + out * out, up))
        cur = out
    layers.append(("final Snake and convolution (k=7)", f"{cur:,} alphas + {cur:,} x 7 + 1", cur + cur * 7 + 1, cur * 7, up))
    return layers


def voc_layers():
    return [(n, p, m, u) for n, _, p, m, u in voc_layout()]


def voc_params():
    return sum(p for _, p, _, _ in voc_layers())


def voc_macs(length):
    """Multiply-adds of one stereo pair (two streams) over `length` latent steps; the transposed convolution's cost is counted at
    its input positions (each input element meets every weight once), every other layer at its own output positions."""
    return 2 * sum(macs * length * up for _, _, macs, up in voc_layers())


def architecture_layouts():
    """The tensors behind every parameter count, so that each row of `parameter_counts` can be re-derived."""
    def table(rows):
        body = [(n, _shape(dims), f"{copies:,}", _i(_prod(dims) * copies)) for n, dims, copies in rows]
        body.append(("total", "", "", _i(_rows_sum(rows))))
        return _table(["tensor", "shape (outputs x inputs)", "copies", "parameters"], body, "llrr")

    voc = [(n, f, _i(p), _i(m), f"{u:,}") for n, f, p, m, u in voc_layout()]
    voc.append(("total, one stream", "", _i(voc_params()), "", ""))
    return ("The tensors of the five components, the Global LLM in the layout of Volume VI, with the counting conventions. A shape lists outputs before inputs, so a linear map from 4,096 to 2,048 "
            "numbers is 2,048 x 4,096; a convolution is outputs x inputs x taps. A count is the product of the shape and the "
            "copies, and the total of each table is the count of the next one. Where a bias is not listed the map has none.\n\n"
            "Global LLM (the layout of Volume VI's Qwen3 8B, a vocabulary of 200,000, 36 layers, 32 query and 8 key-value "
            "heads of 128, a feed-forward width of 12,288):\n\n"
            + table(lm_rows())
            + "\n\nRVQ depth decoder (4 layers of width 4,096, 16 heads, a feed-forward width of 6,144; no rotary "
              "embedding, no grouped heads, no bias; the table has 1,024 rows, the codebook size):\n\n"
            + table(dd_rows())
            + "\n\nCondition encoder (the 3-tap convolution runs along the frame axis and has a bias):\n\n"
            + table(ce_rows())
            + "\n\nFlow-matching DiT, one block listed once and then repeated 36 times in the full table:\n\n"
            + table([(n, dims, copies) for n, dims, copies in dit_block_rows()])
            + "\n\nThe whole denoiser, the block rows multiplied by 36:\n\n"
            + table(dit_rows())
            + "\n\nVocoder, one 64-channel stream (the two ears share the weights). The multiply-adds are per position of "
              "the layer's own resolution, and the last column is that resolution as a multiple of the latent rate; a "
              "transposed convolution is counted at its input positions:\n\n"
            + _table(["layer", "parameters, as a sum of tensors", "parameters", "multiply-adds a position", "positions per latent step"],
                     voc, "llrrr"))


def parameter_counts():
    lm, dd, ce, dt, vc = lm_params(), dd_params(), ce_params(), dit_params(), voc_params()
    rows = [("Global LLM (Qwen3 8B decoder, 200,000-id vocabulary, untied head)", lm),
            ("RVQ depth decoder (4 layers, width 4096)", dd),
            ("condition encoder (8 layer logits, one 3-tap convolution 4096 to 2048)", ce),
            ("flow-matching DiT (36 blocks of width 2048)", dt),
            ("vocoder (convolutional decoder with Snake activations, one 64-channel stream per ear)", vc)]
    total = sum(p for _, p in rows)
    body = [(n, _i(p), _g(p * 4 / 1e9, 2), f"{100 * p / total:.1f}") for n, p in rows]
    body.append(("all five", _i(total), _g(total * 4 / 1e9, 2), "100.0"))
    qwen8_151936 = mc.parameters_formula(151936, 4096, 36, 32, 8, 128, 12288)
    return ("Parameters of the five components, the totals of the tensor tables above (norm gains, biases and the DiT's time "
            "embedding included), and the bytes of a 32-bit copy in gigabytes ($10^9$ bytes). The "
            f"Global LLM is Volume VI's count of Qwen3 8B, {_i(qwen8_151936)} for a vocabulary of 151,936, with the two tables "
            f"enlarged to 200,000 ids: {_i(qwen8_151936)} + 2 x 4096 x (200,000 - 151,936) = {_i(lm)}.\n\n"
            + _table(["component", "parameters", "GB at 32 bits", "share (%)"], body, "lrrr"))


def kv_cache(prompt=200):
    per_pos = 2 * LM["L"] * LM["hkv"] * LM["dh"]
    rows = []
    for secs in (10, 60, 300):
        f = secs * FRAME_RATE
        pos = prompt + f
        rows.append((str(secs), _i(pos), _i(per_pos * pos), _g(per_pos * pos * 4 / 1e9, 3), _g(2 * per_pos * pos * 4 / 1e9, 3)))
    return (f"The attention cache of one branch of the Global LLM: 2 tensors x {LM['L']} layers x {LM['hkv']} key-value heads x "
            f"{LM['dh']} = {_i(per_pos)} numbers per position. One position per frame, after a prompt of {prompt} tokens (an "
            "assumed length: the real one is the length of the caption and the lyrics). Bytes are for 32-bit numbers; the "
            "last column is the two guidance branches.\n\n"
            + _table(["seconds", "positions", "numbers per branch", "GB per branch", "GB, two branches"], rows, "rrrrr"))


def column_vs_flat():
    rows = []
    for secs in (10, 60, 300):
        f = secs * FRAME_RATE
        flat = f * CODES_PER_FRAME
        rows.append((str(secs), _i(f), _i(f * (f + 1) // 2), _i(flat), _i(flat * (flat + 1) // 2),
                     f"{flat * (flat + 1) / (f * (f + 1)):.1f}"))
    return ("Positions of the causal language model for a song, with one position per frame (the eight codes of a frame summed "
            "into one row) and with one position per code, and the number of query-key pairs per head and layer, "
            "$n(n+1)/2$ for $n$ positions.\n\n"
            + _table(["seconds", "positions, one per frame", "pairs", "positions, one per code", "pairs", "ratio of pairs"], rows,
                     "rrrrrr"))


def lm_frame_macs():
    """Multiply-adds of the Global LLM for one frame of one branch: the matrices of 36 layers and the output head (the embedding
    lookup is a gather)."""
    return LM["L"] * _lm_layer_matrix_params() + LM["V"] * LM["D"]


def lm_attention_macs(prompt, n_frames):
    """Attention scores and weighted sums of one branch over the whole song: position p reads p keys, 2 products of the width
    heads x head_dim per layer."""
    width = LM["H"] * LM["dh"]
    return sum(2 * LM["L"] * width * (prompt + t) for t in range(1, n_frames + 1))


def lm_prompt_macs(prompt):
    """Multiply-adds of the prompt's rows through one branch of the Global LLM: the matrices of 36 layers for each of the `prompt`
    rows, and attention in which row p reads p keys. The output head is not applied: its one use at the last prompt row is the
    loop's extra first iteration, which the arithmetic leaves out."""
    width = LM["H"] * LM["dh"]
    return prompt * LM["L"] * _lm_layer_matrix_params() + sum(2 * LM["L"] * width * p for p in range(1, prompt + 1))


def dd_attention_macs_per_branch():
    """Attention of the depth decoder over the 8 positions of one frame: position p reads p keys, 2 products of the width."""
    return DD["layers"] * 2 * DD["h"] * sum(range(1, N_HID + 1))


def ar_frame_macs():
    """(Global LLM, depth decoder) multiply-adds of one frame for both guidance branches."""
    lm = 2 * lm_frame_macs()                                     # two guidance branches
    depth_pos = 2 * N_HID * dd_macs_per_position()               # 8 positions per frame, two branches
    heads = 2 * N_RES * DD["h"] * DD["vocab"]                    # seven residual heads, two branches
    # the input projection is applied to 9 rows: each branch's own hidden vector (2), then the semantic code's embedding and
    # the embeddings of the first six residual codes, which the two branches share because one code is drawn for both
    proj = (2 + 1 + (N_RES - 1)) * DD["h"] * DD["h"]
    return lm, depth_pos + heads + proj + 2 * dd_attention_macs_per_branch()


def dit_forward_macs(length):
    rows = length + 1
    d, ff, c = DIT["inner"], DIT["ff"], DIT["concat"]
    per_block = rows * (4 * d * d + 2 * ff * d + ff * d) + 2 * rows * rows * d
    fixed = length * (c * c + c * d + d * DIT["cin"] + DIT["cin"] ** 2) + DIT["four"] * d + d * d
    return DIT["layers"] * per_block + fixed


def dit_breakdown():
    length = latent_length(CHUNK_FRAMES)
    rows_ = length + 1
    d, ff, c, L = DIT["inner"], DIT["ff"], DIT["concat"], DIT["layers"]
    proj = L * rows_ * 4 * d * d
    ffn = L * rows_ * (2 * ff * d + ff * d)
    attn = L * 2 * rows_ * rows_ * d
    glue = length * (c * c + c * d + d * DIT["cin"] + DIT["cin"] ** 2) + DIT["four"] * d + d * d
    total = proj + ffn + attn + glue
    parts = [("attention projections (q, k, v, out), 36 blocks", proj), ("feed-forward (up, gate, down), 36 blocks", ffn),
             ("attention scores and weighted sums", attn),
             ("input and output glue (1x1 convolutions, projections, timestep embedding)", glue)]
    body = [(n, _eng(v, "G", 1), f"{100 * v / total:.2f}") for n, v in parts]
    body.append(("one evaluation", _eng(total, "G", 1), "100.00"))
    return (f"The multiply-adds of one DiT evaluation over a full chunk ({length} latent steps, {rows_} rows with the timestep "
            "token), in giga, split by kind. The blocks see all rows; the glue is applied to the latent steps only, as 1-tap "
            "convolutions 2,304 to 2,304, a projection 2,304 to 2,048, a projection 2,048 to 128 and a convolution 128 to 128, "
            "and the time embedding (256 to 2,048 to 2,048) is applied once.\n\n" + _table(["part", "multiply-adds (G)", "share (%)"], body, "lrr"))


def song_work(n_frames, prompt=200):
    plan = stitch_plan(n_frames)
    lm_mm, dd_mm = ar_frame_macs()
    ar_lm = n_frames * lm_mm + 2 * lm_attention_macs(prompt, n_frames) + 2 * lm_prompt_macs(prompt)
    ar_dd = n_frames * dd_mm
    dit = sum(dit_forward_macs(p["length"]) * STEPS * 2 for p in plan)
    voc = sum(voc_macs(p["length"]) for p in plan)
    return dict(lm=ar_lm, dd=ar_dd, dit=dit, voc=voc, plan=plan)


def work_layout(prompt=200, secs=240):
    """The counting rules of `work_per_stage`: every term as a formula in the shapes of `architecture_layouts` and its value."""
    L, D, h, hk, dh, f, V = LM["L"], LM["D"], LM["H"], LM["hkv"], LM["dh"], LM["F"], LM["V"]
    H, I = DD["h"], DD["inter"]
    d, ff, c, cin = DIT["inner"], DIT["ff"], DIT["concat"], DIT["cin"]
    n_frames = secs * FRAME_RATE
    length = latent_length(CHUNK_FRAMES)
    rows_ = length + 1
    lm_mm = _lm_layer_matrix_params() * L
    rows = [
        ("Global LLM", "matrices of one row, one branch: 36 x (2 x 4,096 x 4,096 + 2 x 4,096 x 1,024 + 3 x 4,096 x 12,288)", lm_mm),
        ("Global LLM", "output head, one row, one branch: 200,000 x 4,096", V * D),
        ("Global LLM", "one frame, both branches: 2 x (matrices + head)", ar_frame_macs()[0]),
        ("Global LLM", f"attention of the {n_frames:,} frames, both branches: 2 x 2 x 36 x 4,096 x the sum over t = 1 to {n_frames:,} of "
                       f"the {prompt} + t positions that frame t reads", 2 * lm_attention_macs(prompt, n_frames)),
        ("Global LLM", f"the {prompt} prompt rows, both branches: matrices and attention (row p reads p keys), no head",
         2 * lm_prompt_macs(prompt)),
        ("depth decoder", "one position, one branch: 4 x (4 x 4,096^2 + 3 x 4,096 x 6,144)", dd_macs_per_position()),
        ("depth decoder", "the 8 positions of a frame, both branches", 2 * N_HID * dd_macs_per_position()),
        ("depth decoder", "seven output heads, both branches: 2 x 7 x 4,096 x 1,024", 2 * N_RES * H * DD["vocab"]),
        ("depth decoder", "input projection, applied to 9 rows (each branch's hidden vector, the semantic code's embedding and six residual ones, the last seven shared by the branches): 9 x 4,096^2",
         9 * H * H),
        ("depth decoder", "attention over the 8 positions, both branches: 2 x (4 x 2 x 4,096 x (1 + 2 + ... + 8))",
         2 * dd_attention_macs_per_branch()),
        ("depth decoder", "one frame, the four terms above", ar_frame_macs()[1]),
        ("DiT", f"attention projections, one evaluation: 36 x {rows_} x 4 x 2,048^2", DIT["layers"] * rows_ * 4 * d * d),
        ("DiT", f"feed-forward, one evaluation: 36 x {rows_} x 3 x 8,192 x 2,048", DIT["layers"] * rows_ * 3 * ff * d),
        ("DiT", f"attention scores and weighted sums: 36 x 2 x {rows_}^2 x 2,048", DIT["layers"] * 2 * rows_ * rows_ * d),
        ("DiT", f"glue, on the {length} latent steps: {length} x (2,304^2 + 2,304 x 2,048 + 2,048 x 128 + 128^2), plus the time "
                f"embedding 256 x 2,048 + 2,048^2 once", length * (c * c + c * d + d * cin + cin * cin) + DIT["four"] * d + d * d),
        ("vocoder", f"one latent step, both streams: 2 x the sum over the layers of (multiply-adds a position x positions per step)",
         voc_macs(1)),
    ]
    body = [(a, b, _i(v)) for a, b, v in rows]
    return ("The terms of the count, as formulas in the shapes of the tables above. A product is counted once for every row it is "
            "applied to; norms, activations, embedding lookups and the sampling are left out. The output head is applied once per "
            "frame, to the frame's row, and so the prompt's rows and the loop's extra first iteration (the frame that the loop "
            "samples from the end of the prompt and does not keep) are counted as matrices and attention only, not as one more "
            "head and one more depth-decoder frame.\n\n"
            + _table(["stage", "term", "multiply-adds"], body, "llr"))


def work_per_stage():
    secs = 240
    f = secs * FRAME_RATE
    w = song_work(f)
    total = w["lm"] + w["dd"] + w["dit"] + w["voc"]
    names = [("Global LLM, two branches (matrices, head and attention)", w["lm"]),
             ("depth decoder, two branches", w["dd"]),
             ("DiT, 30 steps of 2 evaluations per chunk", w["dit"]),
             ("vocoder, every chunk decoded in full", w["voc"])]
    rows = [(n, _eng(v, "T", 1), _eng(v / secs, "T", 3), f"{100 * v / total:.2f}") for n, v in names]
    rows.append(("total", _eng(total, "T", 1), _eng(total / secs, "T", 3), "100.00"))
    one = dit_forward_macs(latent_length(CHUNK_FRAMES))
    lm_mm, dd_mm = ar_frame_macs()
    speech = ss._talker_frame_macs()
    speech_s = (speech[0] + speech[1] + speech[2] + speech[3]) * ss.FRAME_RATE + sum(r[2] for r in ss.ac._decoder_rows())
    return (f"Multiply-adds to make a song of {secs} seconds ({f:,} frames, {len(w['plan'])} chunks), counted as in the speech "
            "chapter and by the terms of the table above: every matrix product once per row, attention scores and weighted sums "
            "included, the 200 rows of the assumed prompt included, norms and elementwise functions left out. One DiT evaluation over a full chunk "
            f"(length {latent_length(CHUNK_FRAMES)}, {latent_length(CHUNK_FRAMES) + 1} rows with the timestep token) is "
            f"{_eng(one, 'T', 3)} tera-multiply-adds; one autoregressive frame costs {_eng(lm_mm, 'G', 2)} giga for the language "
            f"model's two branches (without attention) and {_eng(dd_mm, 'G', 2)} for the depth decoder's.\n\n"
            + _table(["stage", "multiply-adds (T)", "per second of music (T)", "share (%)"], rows, "lrrr")
            + f"\n\nOne second of the 16-codebook speech synthesiser of Chapter 6, its codec decoder included, costs "
              f"{_eng(speech_s, 'G', 1)} giga multiply-adds; {_eng(total / secs, 'T', 3)} tera is {total / secs / speech_s:,.0f} "
              f"times that. The {_eng(total, 'T', 0)} tera multiply-adds of the song are {_eng(2 * total, 'P', 2)} peta "
              "floating-point operations, a multiply-add being two.")


ROOF = 10.517e12      # the fp32 roof of the card of the crate's bench, operations a second (commit 3c5f30d77)


def roof_floor():
    total = sum(song_work(240 * FRAME_RATE)[k] for k in ("lm", "dd", "dit", "voc"))
    floor_s = 2 * total / ROOF
    return ("<!-- perf-number: historical measurement, commit 3c5f30d77 -->\n"
            f"Divided by the roof of the card of the crate's bench, {ROOF / 1e12:.3f} tera operations a second, the {_eng(2 * total, 'P', 2)} "
            f"peta operations of a four-minute song take at least {floor_s:,.0f} seconds, {floor_s / 60:.1f} minutes, and a real run "
            "is slower than its roof.")


def work_factors():
    one = dit_forward_macs(latent_length(CHUNK_FRAMES))
    per_hop = STEPS * 2 * one
    secs_per_hop = CHUNK_HOP / FRAME_RATE
    rows = [("one evaluation of one chunk", 1, one),
            ("30 steps", STEPS, STEPS * one),
            ("2 evaluations a step (guidance)", 2, 2 * STEPS * one),
            ("the chunk's 8 seconds cover 4 new ones, so per second of music: / 4", 0.25, per_hop / secs_per_hop)]
    out = [(r[0], f"{r[1]:g}", _eng(r[2], "T", 3)) for r in rows]
    return ("From one DiT evaluation to the denoiser's work per second of finished music (multiply-adds, in tera). The chunk of "
            "8 seconds is paid once for every 4 seconds of song, so each second is denoised in two chunks, and the second "
            "factor of 2 of the guidance is paid on top of the 30 steps.\n\n"
            + _table(["step of the count", "factor", "running total (T)"], out, "lrr"))


def vocoder_activations(length=None):
    """Floats of every buffer the decoder allocates for one chunk of `length` latent steps (one stream), in the order the forward
    records them: the whole tape is submitted at once, so all of them are alive together."""
    length = length or latent_length(CHUNK_FRAMES)
    rows = [("input projection and first convolution", 2, (VOC["din"] + VOC["dhid"]) * length)]
    cur, ln = VOC["dhid"], length
    for b, s_ in enumerate(VOC["ratios"]):
        out = cur // 2
        new_ln = ln * s_
        snake_in = cur * ln                       # the Snake before the upsampling, at the input resolution
        up = out * new_ln
        rows.append((f"block {b}: {out} channels at {new_ln:,} steps", 2 + 15, snake_in + up + 15 * up))
        cur, ln = out, new_ln
    rows.append(("final Snake and convolution", 2, cur * ln + ln))
    return rows


def vocoder_memory():
    rows = vocoder_activations()
    total = sum(f for _, _, f in rows)
    body = [(n, str(c), _i(f), _g(f * 2 * 4 / 2 ** 20, 1)) for n, c, f in rows]
    body.append(("all of them", "", _i(total), _g(total * 2 * 4 / 2 ** 20, 1)))
    return ("The buffers of the vocoder for one chunk of 689 latent steps: the decoder is recorded as one tape and submitted once, "
            "so every intermediate is alive until the end. Floats per stream, and mebibytes ($2^{20}$ bytes) for the two streams "
            "at 4 bytes. A block of $c$ channels in and $c/2$ out, from $L$ to $sL$ steps, holds the input Snake ($c L$ floats, at the "
            "block's input resolution), the transposed convolution's output ($c s L / 2$), and three residual units of five "
            "buffers each (Snake, dilated convolution, Snake, 1-tap convolution and the sum, each $c s L / 2$), 17 buffers in "
            "all. The first row is the projection ($1{,}024 L$) and the first convolution ($1{,}536 L$); the last is the "
            "Snake over 96 channels and the one-channel output.\n\n"
            + _table(["part", "buffers", "floats per stream", "MiB, two streams"], body, "lrrr")
            + f"\n\nThe crate's ledger records a measured peak of {12264:,} MiB for such a chunk "
              f"(`VOCODER_CHUNK_DEVICE_PEAK_BYTES`); the activations account for {total * 8 / 2 ** 20 / 12264 * 100:.1f} percent of it.")


# ----------------------------------------------------------------------------------------------------------------- memory
def stage_memory():
    gl = 6_950_000_000 + 2 * LM["V"] * LM["D"] * 4          # the measured int8 linears plus two fp32 tables
    dd_closed = (DD["layers"] * (4 * DD["h"] ** 2 + 3 * DD["h"] * DD["inter"]) + 2 * N_RES * DD["vocab"] * DD["h"]
                 + DD["h"] ** 2 + DD["max_pos"] * DD["h"]) * 4
    dit_blocks = DIT["layers"] * (4 * DIT["inner"] ** 2 + 2 * DIT["ff"] * DIT["inner"] + DIT["inner"] * DIT["ff"]) * 4
    dit = dit_blocks + (384 << 20)
    voc = 12264 << 20
    ar = gl + dd_closed
    rows = [("autoregressive: one Global LLM instance (int8 linears, fp32 tables) and the depth decoder", ar, "6.95 GB measured for the linears"),
            ("denoise: the DiT's block stack in fp32 plus a margin", dit, "margin of 384 MiB measured"),
            ("vocoder: one full 689-step chunk", voc, "12264 MiB measured")]
    body = [(n, _i(b), _g(b / 1e9, 2), note) for n, b, note in rows]
    two = 2 * ar
    lin_closed = LM["L"] * _lm_layer_matrix_params()
    parts = [("Global LLM linears, int8", gl - 2 * LM["V"] * LM["D"] * 4, "measured, 6.95 GB (three digits)"),
             ("Global LLM, two tables of 200,000 x 4,096 in fp32", 2 * LM["V"] * LM["D"] * 4, "2 x 200,000 x 4,096 x 4 bytes"),
             ("depth decoder, fp32, matrices and tables only", dd_closed,
              "4 x (4 x 4,096^2 + 3 x 4,096 x 6,144) + 2 x 7 x 1,024 x 4,096 + 4,096^2 + 16 x 4,096, times 4 bytes")]
    mem = [(n, _i(b), note) for n, b, note in parts] + [("autoregressive stage", _i(ar), "the three rows above")]
    gap = gl - 2 * LM["V"] * LM["D"] * 4 - lin_closed
    return ("What one generation asks of a card, stage by stage. The stages run one after the other and drop their device state "
            "between them, so the need is the largest of the three, not the sum. A stage's weights are the closed form of its "
            "shapes where one exists; the three figures marked measured are measurements of the crate's ledger and cannot be "
            "derived from a configuration.\n\n"
            + _table(["stage", "bytes", "GB", "source"], body, "lrrl")
            + "\n\nThe first row, term by term:\n\n" + _table(["part", "bytes", "how"], mem, "lrl")
            + f"\n\nThe closed form of the language model's int8 matrices, the 36 layers of "
              f"{_lm_layer_matrix_params():,} weights at one byte each, is {_i(lin_closed)} bytes; the measured 6.95 GB exceeds it by "
              f"{_i(gap)} bytes, {100 * gap / lin_closed:.2f} percent, which is inside the rounding of a figure given to three digits "
              "and so leaves no room to say where per-row scales or norm gains live. The depth decoder's closed form omits its "
              f"{_i(dd_params() - dd_closed // 4)} norm gains. The denoise row is the block stack without biases, 36 x (4 x 2,048^2 "
              f"+ 3 x 8,192 x 2,048) = {_i(dit_blocks // 4)} weights at 4 bytes, plus the margin of 384 MiB = {_i(384 << 20)} bytes.\n\n"
              f"The autoregressive stage runs two instances of the language model, one for each guidance branch; {_i(two)} bytes "
              f"({two / 1e9:.2f} GB) is what one card would have to hold if both lived on it, against the 24 GiB "
              f"({24 * 2 ** 30 / 1e9:.2f} GB) of the cards of the validation machine.")


# --------------------------------------------------------------------------------------------------- the vocoder's reach
def vocoder_reach():
    """Latent steps that influence one block of 512 output samples, found by walking the decoder backward as intervals."""
    q = 1000
    lo, hi = q * HOP_LAT, q * HOP_LAT + HOP_LAT - 1

    def conv(lo, hi, k, d):
        pad = d * (k - 1) // 2
        return lo - pad, hi + pad

    def convt(lo, hi, k, s):
        p = -(-s // 2)
        return -(-(lo + p - k + 1) // s), (hi + p) // s

    trace = []
    lo, hi = conv(lo, hi, 7, 1)
    trace.append(("final convolution (k=7)", lo, hi, 1))
    scale = 1
    for b in reversed(range(len(VOC["ratios"]))):
        s = VOC["ratios"][b]
        for d in (9, 3, 1):
            lo, hi = conv(lo, hi, 1, 1)
            lo, hi = conv(lo, hi, 7, d)
        lo, hi = convt(lo, hi, 2 * s, s)
        scale *= s
        trace.append((f"block {b}: three units and the upsampling by {s}", lo, hi, scale))
    lo, hi = conv(lo, hi, 7, 1)
    trace.append(("first convolution (k=7)", lo, hi, scale))
    rows = [(n, _i(a), _i(b), _i(b - a + 1)) for n, a, b, _ in trace]
    left, right = q - lo, hi - q
    return ("The interval of positions that can change one block of 512 output samples (those of latent step $q$ = 1000), found "
            "by walking the decoder backward from the output: each convolution widens the interval by its dilation times "
            "$(k-1)/2$ on each side, each transposed convolution maps it to the input positions that a window of its kernel "
            "touches. Positions are in the layer's own resolution. A residual unit is walked backward as its 1-tap convolution and then its "
            "7-tap convolution at the unit's dilation (a convolution pads each side by dilation x 3 so that the length is kept), "
            "and the three units of a block with dilations 9, 3 and 1 in that order (forward they are 1, 3 and 9, after the "
            "upsampling). A transposed convolution of stride $s$ has a kernel of $k$ taps, twice $s$, and pads $s/2$ on each side: its output "
            "position $o$ takes input $i$ through tap $j$ when $o = i s - s/2 + j$, so an output interval $[a, b]$ needs the "
            "inputs from $\\lceil (a + s/2 - k + 1)/s \\rceil$ to $\\lfloor (b + s/2)/s \\rfloor$.\n\n"
            + _table(["after walking back through", "first position", "last position", "positions"], rows, "lrrr")
            + f"\n\nAt the latent rate the 512 samples of step {q} depend on steps {lo} to {hi}: {left} before and {right} after. "
              f"The crops of the stitcher are {CROP_L} and {CROP_R} steps.")


# ----------------------------------------------------------------------------------------------------------------- adapters
def adapter_sizes():
    rows = []
    d = DIT["inner"]
    dit_shapes = [(d, d)] * 4 + [(2 * DIT["ff"], d), (d, DIT["ff"])]
    dd_shapes = [(DD["h"], DD["h"])] * 4 + [(DD["inter"], DD["h"]), (DD["inter"], DD["h"]), (DD["h"], DD["inter"])]
    vc = [(VOC["din"], VOC["latent"] // 2), (VOC["dhid"], VOC["din"] * 7)]
    cur = VOC["dhid"]
    for s in VOC["ratios"]:
        out = cur // 2
        vc.append((cur, out * 2 * s))
        vc += [(out, out * 7), (out, out)] * 3
        cur = out
    vc.append((1, cur * 7))
    sets = [("DiT", dit_shapes, DIT["layers"], dit_params()), ("depth decoder", dd_shapes, DD["layers"], dd_params()),
            ("vocoder", vc, 1, voc_params())]
    for name, shapes, reps, base in sets:
        per_rank = reps * sum(r + c for r, c in shapes)
        for rank in (4, 16):
            rows.append((name, str(len(shapes) * reps), str(rank), _i(per_rank * rank), f"{100 * per_rank * rank / base:.2f}"))
    return ("Parameters of the low-rank adapters of the three components that have them: the number of adapted matrices, and for "
            "a rank $r$ each matrix of $m \\times n$ weights gets $r(m + n)$ trainable numbers. A convolution is adapted as the "
            "matrix of its outputs by its inputs times taps ($m = c_{\\mathrm{out}}$, $n = c_{\\mathrm{in}} k$), and a transposed "
            "convolution as $m = c_{\\mathrm{in}}$, $n = c_{\\mathrm{out}} k$; $r(m + n)$ is the same either way. The DiT "
            "adapts the 6 maps of each of its 36 blocks (4 attention maps, the up-and-gate matrix and the down matrix: 216), the "
            "depth decoder the 7 maps of each of its 4 layers (28), and the vocoder 31 convolutions: the input projection, the "
            "first convolution, per block one transposed convolution and three units of two, and the final convolution. The "
            "percentage is against the component's own parameters.\n\n"
            + _table(["component", "adapted matrices", "rank", "adapter parameters", "% of the component"], rows, "lrrrr"))


# ------------------------------------------------------------------------------------------- the data of the figures
def seam_figure_data():
    """Stitched texture (song minus melody) of channel 0 of the toy, without and with the hold, with the seam and join positions."""
    m = _toy_melody()
    out = {}
    for hold in (False, True):
        song, seams, joins = toy_stitch(hold, 8)
        out[hold] = [song[0][g] - m[0][g] for g in range(TOY["n_song"])]
    return out, seams, joins


def geometry_figure_data(n_frames=450):
    return stitch_plan(n_frames)
