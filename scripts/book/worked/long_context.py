# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Long Context": how a model trained on short texts reads long ones.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds
between `<!-- worked: long_context.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a
picture and the table beside it cannot disagree.

* the rotation angles that a model trained on 4,096 positions never saw when it reads 32,768;
* the entropy of a softmax over random scores as the number of keys grows, and the temperature that cancels the growth;
* the frequency tables of position interpolation, NTK-aware scaling and YaRN (both ramps), which pairs change, and the pairs
  that still reach angles nobody trained;
* the YaRN table of a 128-wide head at base 1e6 (the pinned configuration of brain's tests);
* bytes of cache per token for stated configurations in four storage formats, and the error that 8-bit formats add;
* chunked prefill: activation memory against the weights read again for every chunk, and the equality of the chunked and the
  one-shot prefill on the tiny model of Volume V, Chapter 12 (`kv_cache`, imported, not copied);
* the bytes that offloading a sequence's cache moves, against streaming it every token and against recomputing it;
* a single-head needle: the attention weight on one matching key among n random ones.

The rotation rates and the YaRN table are those of Volume V, Chapter 8 (`positions.omegas`, `positions.yarn_table`, imported).
Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums are plain
left-to-right additions (never Python's `sum`, which may compensate). Library functions are the C library's `exp`, `log`, `pow`,
`sin` and `cos`.

Swedish Embedded AB implements solutions for serving language models at long context. If your team needs expertise in rotary
scaling, cache quantization and memory planning for long prompts, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import bisect
import math
from decimal import ROUND_HALF_UP, Decimal
import struct

import bookrng
import kv_cache as kc
import positions as pos

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _g(v, d=4):
    return f"{v:.{d}g}"


def _fx(v, d=3):
    """Fixed decimals, rounded half away from zero (the book's round rule) on the exact binary value, so only a true tie moves."""
    out = str(Decimal(v).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) if math.isfinite(v) else f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _gib(b):
    return b / 2 ** 30


def _acc(xs):
    """Left-to-right sum."""
    t = 0.0
    for x in xs:
        t += x
    return t


# ------------------------------------------------------------------------------------------------ the running configuration

D_HEAD, BASE, L_TRAIN, S, L_EXT = 64, 1e4, 4096, 8, 32768
TAU = 2 * math.pi


def attention_factor(s):
    return 0.1 * math.log(s) + 1.0


# ------------------------------------------------------------------------------------------------ angles nobody trained


def angle_reach_data(d=D_HEAD, base=BASE, trained=L_TRAIN, ext=L_EXT):
    """Per pair m: (largest angle at the last trained position, largest angle at the last extended position)."""
    w = pos.omegas(d, base)
    return [((trained - 1) * x, (ext - 1) * x) for x in w]


def unseen_angles():
    w = pos.omegas(D_HEAD, BASE)
    sel = (0, 8, 16, 20, 22, 23, 24, 28, 31)
    rows = []
    for m in sel:
        a, b = angle_reach_data()[m]
        lam = TAU / w[m]
        rows.append((m, _g(w[m], 4), _g(lam, 4), _g(a, 4), _g(b, 4), _g(a / TAU, 3), _g(b / TAU, 3),
                     "yes" if a < TAU else "no"))
    never = [m for m in range(D_HEAD // 2) if angle_reach_data()[m][0] < TAU]
    never_ext = [m for m in range(D_HEAD // 2) if angle_reach_data()[m][1] < TAU]
    return (f"Head width d = {D_HEAD} ({D_HEAD // 2} pairs), base {int(BASE)}, trained on positions 0 to {L_TRAIN - 1}, read at "
            f"positions up to {L_EXT - 1} (s = {S}). Pair m turns by w_m = base^(-2m/d) radians per position, so its wavelength is "
            "2 pi / w_m positions and its largest angle at position p is p w_m. 'Turns' columns divide the angle by 2 pi. "
            "A pair whose largest trained angle is below 2 pi never completed one turn in training.\n\n"
            + _table(["pair m", "w_m", "wavelength (positions)", "largest angle, trained", "largest angle, extended",
                      "turns, trained", "turns, extended", "never completed a turn"], rows, "lrrrrrrl")
            + f"\n\nPairs that never completed a turn in training: {len(never)} of {D_HEAD // 2} (m = {never[0]} to {never[-1]}). "
            f"Pairs that have not completed one at {L_EXT - 1} either: {len(never_ext)} (m = {never_ext[0]} to {never_ext[-1]}). "
            f"Wavelengths longer than the trained length {L_TRAIN}: pairs m >= "
            f"{next(m for m in range(D_HEAD // 2) if TAU / w[m] > L_TRAIN)}.")


# ------------------------------------------------------------------------------------------------ entropy against length

NAME_ENT = "long_context.entropy"
SIGMA = 2.0
N_ENT = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def _scores():
    g = bookrng.generator(NAME_ENT)
    return [SIGMA * g.normal() for _ in range(max(N_ENT))]


def _entropy(xs, lam=1.0):
    """Entropy in nats of softmax(lam x), with the maximum subtracted first, sums left to right."""
    m = max(lam * x for x in xs)
    e = [math.exp(lam * x - m) for x in xs]
    z = _acc(e)
    return -_acc(p / z * math.log(p / z) for p in e if p > 0.0)


def entropy_data():
    xs = _scores()
    return [(n, _entropy(xs[:n]), math.log(n), math.log(n) - SIGMA ** 2 / 2) for n in N_ENT]


def entropy_growth():
    rows = [(_int(n), _fx(h, 4), _fx(ln, 4), _fx(ap, 4), _fx(h - ap, 4)) for n, h, ln, ap in entropy_data()]
    return (f"One query, n keys, each score an independent draw of {SIGMA} x `normal()` (standard deviation {SIGMA}, variance "
            f"{SIGMA ** 2}) from the generator seeded {_seed(NAME_ENT)} (`seed_of` of the name `{NAME_ENT}`); {max(N_ENT):,} values "
            "are drawn once in order and the first n of them are the scores of the row with n keys. H is the entropy of "
            "softmax(scores) in nats (natural logarithm, the maximum subtracted first, sums left to right); the last two columns "
            "are the large-n approximation ln n - variance / 2 derived in the text and the difference.\n\n"
            + _table(["n keys", "H", "ln n", "ln n - 2", "H - (ln n - 2)"], rows))


def temperature_factor():
    xs = _scores()
    rows = []
    for s in (2, 4, 8, 16, 32, 64):
        t = attention_factor(s)
        lam = t * t
        implied = 2 * math.log(s) / (lam * lam - 1)
        rows.append((s, _fx(t, 4), _fx(lam, 4), _fx(lam * lam, 4), _fx(implied, 3)))
    s = L_EXT // L_TRAIN
    lam = attention_factor(s) ** 2
    h0, h1, h2 = _entropy(xs[:L_TRAIN]), _entropy(xs[:L_EXT]), _entropy(xs[:L_EXT], lam)
    return ("mu(s) = 0.1 ln s + 1 multiplies the cosine and the sine of every pair, so the rotated query and key are each mu(s) times "
            "larger and the score is multiplied by lambda = mu(s)^2. Keeping the entropy of the row at its value for n keys when "
            "there are s n of them needs lambda^2 = 1 + 2 ln(s) / variance (derived in the text); the last column solves "
            "that for the variance of the scores that mu(s) would be exactly right for.\n\n"
            + _table(["extension s", "mu(s)", "lambda = mu(s)^2", "lambda^2", "variance that makes mu exact"], rows)
            + f"\n\nThe scores of the previous block (variance {SIGMA ** 2}): entropy with n = {L_TRAIN} keys and lambda = 1: "
            f"{h0:.4f}; with n = {L_EXT} keys and lambda = 1: {h1:.4f}; with n = {L_EXT} keys and lambda = mu({s})^2 = {lam:.4f}: "
            f"{h2:.4f}.")


# ------------------------------------------------------------------------------------------------ the frequency tables


def pi_rates(w, s):
    return [x / s for x in w]


def ntk_base(d, base, s):
    return base * s ** (d / (d - 2))


def ntk_rates(d, base, s):
    b2 = ntk_base(d, base, s)
    return [b2 ** (-2 * m / d) for m in range(d // 2)]


def yarn_rates(d, base, s, trained, fast=32.0, slow=1.0):
    """The reference implementation's table: a linear ramp over the pair index (positions.yarn_table)."""
    return pos.yarn_table(d, base, s, trained, fast, slow)[0]


def yarn_paper_rates(d, base, s, trained, alpha=1.0, beta=32.0):
    """The paper's ramp: linear in the number of turns r = trained w_m / 2 pi between alpha and beta."""
    out = []
    for m, w in enumerate(pos.omegas(d, base)):
        r = trained * w / TAU
        gamma = 0.0 if r < alpha else 1.0 if r > beta else (r - alpha) / (beta - alpha)
        out.append((1 - gamma) * w / s + gamma * w)
    return out


def rate_ratio_data(d=D_HEAD, base=BASE, s=S, trained=L_TRAIN):
    """Per scheme, the ratio (scaled rate) / (original rate) for every pair."""
    w = pos.omegas(d, base)
    return {
        "interpolation": [1 / s] * (d // 2),
        "NTK-aware": [a / b for a, b in zip(ntk_rates(d, base, s), w)],
        "YaRN": [a / b for a, b in zip(yarn_rates(d, base, s, trained), w)],
        "YaRN, ramp in turns": [a / b for a, b in zip(yarn_paper_rates(d, base, s, trained), w)],
    }


def frequency_tables():
    d, base, s = D_HEAD, BASE, S
    w = pos.omegas(d, base)
    pi, ntk, ya, yp = pi_rates(w, s), ntk_rates(d, base, s), yarn_rates(d, base, s, L_TRAIN), yarn_paper_rates(d, base, s, L_TRAIN)
    sel = (0, 1, 4, 8, 10, 12, 16, 20, 23, 24, 27, 31)
    rows = [(m, _g(w[m], 4), _g(pi[m], 4), _g(ntk[m], 4), _g(ya[m], 4), _g(yp[m], 4)) for m in sel]
    same = lambda a: sum(1 for x, y in zip(a, w) if abs(x - y) <= 1e-12 * y)
    full = lambda a: sum(1 for x, y in zip(a, w) if abs(x - y / s) <= 1e-9 * y)
    pairs = d // 2
    cnt = [("original", pairs, 0), ("position interpolation", same(pi), full(pi)), ("NTK-aware", same(ntk), full(ntk)),
           ("YaRN, ramp over the pair index", same(ya), full(ya)), ("YaRN, ramp over the turns", same(yp), full(yp))]
    return (f"Head width d = {d}, base {int(base)}, trained length {L_TRAIN}, extension s = {s}. NTK-aware base: "
            f"base' = base x s^(d / (d - 2)) = {int(base)} x {s}^({d}/{d - 2}) = {ntk_base(d, base, s):.2f}. YaRN with bounds "
            "alpha = 1 and beta = 32 turns: the 'index' ramp rises linearly from 0 at the pair floor(c(32)) to 1 at the pair "
            "ceil(c(1)) (Volume V, Chapter 8), the 'turns' ramp is linear in the number of turns n_m = 4096 w_m / 2 pi between 1 and 32. "
            "Rates in radians per position, to 4 significant digits (the rates are computed in double precision with the library functions pow and exp; "
            "the base' of NTK-aware scaling is used unrounded).\n\n"
            + _table(["pair m", "original", "interpolation", "NTK-aware", "YaRN (index ramp)", "YaRN (turns ramp)"], rows)
            + "\n\nPairs whose rate equals the original, and pairs whose rate equals the original divided by s "
            f"({pairs} pairs). A library pow is not bit-exact, so 'equals' is a tolerance on the rate x of the scheme against the original rate w of the "
            "pair: |x - w| <= 1e-12 w for 'unchanged' and |x - w / s| <= 1e-9 w for 'divided by s'; a count differs in another implementation only if its rates "
            "are not within these tolerances.\n\n"
            + _table(["scheme", "unchanged", "divided by s exactly"], cnt))


def ramp_comparison():
    d, base, s = D_HEAD, BASE, S
    w = pos.omegas(d, base)
    ya, yp = yarn_rates(d, base, s, L_TRAIN), yarn_paper_rates(d, base, s, L_TRAIN)
    rows = []
    for m in range(8, 25, 2):
        turns = L_TRAIN * w[m] / TAU
        g_idx = (w[m] - ya[m]) / (w[m] - w[m] / s)
        g_pap = (w[m] - yp[m]) / (w[m] - w[m] / s)
        rows.append((m, _g(turns, 4), _fx(g_idx, 3), _fx(g_pap, 3)))
    return ("Weight of the interpolated rate for pairs 8 to 24 (every second pair), d = 64, base 10000, trained length 4096, "
            "alpha = 1, beta = 32: turns over the trained length, the weight under the ramp over the pair index and under the "
            "ramp over the turns.\n\n" + _table(["pair m", "turns n_m", "index ramp", "turns ramp"], rows))


def untrained_reach():
    """For each scheme, the pairs that never completed a turn in training and still reach beyond the trained angles."""
    d, base, s = D_HEAD, BASE, S
    w = pos.omegas(d, base)
    schemes = {"unscaled": w, "interpolation": pi_rates(w, s), "NTK-aware": ntk_rates(d, base, s),
               "YaRN": yarn_rates(d, base, s, L_TRAIN)}
    slow = [m for m in range(d // 2) if (L_TRAIN - 1) * w[m] < TAU]
    rows, counts = [], []
    for name, rates in schemes.items():
        # ratio of the largest extended angle to the trained length's angle w_m x 4096 (> 1: angles nobody trained)
        ex = {m: (L_EXT - 1) * rates[m] / (L_TRAIN * w[m]) for m in slow}
        bad = [m for m in slow if ex[m] > 1.0]
        counts.append((name, len(bad), _fx(max(ex.values()), 3)))
        rows.append((name, *[_fx(ex[m], 3) for m in (23, 25, 27, 29, 31)]))
    return ("For the 9 pairs that never completed a turn in training (m = 23 to 31), the largest angle reached at position 32767 "
            "divided by the angle w_m x 4096 that the trained window ends at; a value above 1 means angles no training example "
            "produced. d = 64, base 10000, s = 8.\n\n"
            + _table(["scheme", "m = 23", "m = 25", "m = 27", "m = 29", "m = 31"], rows)
            + "\n\nPairs among the 9 with a value above 1, and the largest value:\n\n"
            + _table(["scheme", "pairs above 1", "largest value"], counts))


def neighbour_resolution():
    d, base, s = D_HEAD, BASE, S
    rows = []
    for name, r in (("original", 1.0), ("interpolation", 1 / s), ("NTK-aware", rate_ratio_data()["NTK-aware"][0]),
                    ("YaRN", rate_ratio_data()["YaRN"][0])):
        rows.append((name, _fx(r, 4), _fx(1.0 * r, 4), _fx(2 * math.sin(r / 2), 4)))
    return ("Pair 0 turns by 1 radian per position originally. The rotation between two neighbouring tokens on that pair under "
            "each scheme, and the length of the chord between their two unit vectors, 2 sin(angle / 2):\n\n"
            + _table(["scheme", "rate ratio", "angle between neighbours (rad)", "chord"], rows))


def yarn_128():
    d, base, s, L = 128, 1e6, 4.0, 32768
    w = pos.omegas(d, base)
    ya = yarn_rates(d, base, s, L)
    low = max(0, math.floor(pos._correction_dim(32.0, d, base, L)))
    high = min(d // 2 - 1, math.ceil(pos._correction_dim(1.0, d, base, L)))
    head, tail = ya[:5], ya[59:]
    return (f"Head width d = 128 (64 pairs), base 1e6, trained length 32768, s = 4, bounds 32 and 1. Correction range "
            f"[{low}, {high}], attention factor 0.1 ln 4 + 1 = {attention_factor(s):.7f}. Pairs 0 to 4 of the scaled table: "
            + ", ".join(f"{x:.7g}" for x in head) + ". Pairs 59 to 63: " + ", ".join(f"{x:.7g}" for x in tail)
            + f". Pair 63 divided by its original rate: {ya[63] / w[63]:.7f}.")


# ------------------------------------------------------------------------------------------------ what eight times the text costs


def eightfold_counts():
    s = kc.DECODER_8B
    per_key = 4 * s["L"] * s["H"] * s["dh"]
    rows = []
    for n in (L_TRAIN, L_EXT):
        attn_decode = per_key * n
        attn_prefill = per_key * (n * (n + 1) // 2)
        rows.append((_int(n), _int(kc.cache_bytes_per_token(s) * n), _int(attn_decode), _int(attn_prefill)))
    r = (L_EXT * (L_EXT + 1) // 2) / (L_TRAIN * (L_TRAIN + 1) // 2)
    return ("The 7.5-billion-parameter decoder of Volume V, Chapter 12 (32 layers, 32 query heads and 8 key-value heads of width 128, bf16), "
            f"a sequence of n tokens. Cache bytes: 131,072 per token. Attention FLOPs: 4 L H d_h = {_int(per_key)} per visible key "
            "(scores and weighted sum, all layers); a decoded token at context n sees n keys, a prefill of n tokens sees "
            "n (n + 1) / 2 key-query pairs in all.\n\n"
            + _table(["n", "cache bytes", "attention FLOPs of one decoded token", "attention FLOPs of the prefill"], rows)
            + f"\n\nRatio of the prefill attention FLOPs, {L_EXT:,} against {L_TRAIN:,} tokens: {r:.4f}.")


def score_slab_limit():
    heads, limit = 16, 2047 * 2 ** 20
    t = next(t for t in range(1, 10 ** 5) if heads * t * t * 4 > limit)
    rows = []
    for n in (2048, 4096, t - 1, t, 8192, 16384, 32768):
        rows.append((_int(n), _int(heads * n * n * 4), _fx(heads * n * n * 4 / 2 ** 20, 2), "yes" if heads * n * n * 4 > limit else "no",
                     _int(heads * n * 4)))
    return (f"Training attention that keeps its scores: one buffer of H T^2 f32 numbers = 4 H T^2 bytes for H = {heads} query heads "
            f"(Qwen3-0.6B and 1.7B), against the largest single binding of {limit // 2 ** 20} MiB = {_int(limit)} bytes that the "
            f"device of knowledge #193 reported. The smallest T whose buffer exceeds it is {t}. The last column is the 4 H T bytes "
            "of the one log-sum-exp number per row and head that a flash backward keeps instead.\n\n"
            + _table(["T", "bytes of the score buffer", "MiB", "over the binding limit", "bytes of the log-sum-exp rows"], rows))


# ------------------------------------------------------------------------------------------------ cache bytes per token

TIERS = (("f32", 4.0), ("bf16", 2.0), ("int8", None), ("fp8", None))   # int8 and fp8: 1 byte + one f32 scale per (token, head) row
MODELS = (("7.5-billion-parameter decoder (8 key-value heads)", kc.DECODER_8B), ("70-billion-parameter decoder (8 key-value heads)", kc.DECODER_70B))
RIPPLE = {"L": 80, "global_every": 5, "kv_dim": 1024, "window": 1024}


def bytes_per_element(tier, dh):
    for name, b in TIERS:
        if name == tier:
            return b if b is not None else 1.0 + 4.0 / dh
    raise KeyError(tier)


def cache_bytes_per_token(s, tier):
    """2 (K and V) x layers x kv heads x head width x bytes per element of the tier."""
    return 2 * s["L"] * s["kvh"] * s["dh"] * bytes_per_element(tier, s["dh"])


def hybrid_bytes(n, tier="bf16", window=RIPPLE["window"]):
    """Cache bytes of one sequence of n tokens for the Ripple layer pattern of Chapter 3 (80 layers, every 5th global)."""
    r = RIPPLE
    n_global = r["L"] // r["global_every"]
    n_slide = r["L"] - n_global
    b = bytes_per_element(tier, 128)
    per = 2 * r["kv_dim"] * b
    return n_global * per * n + n_slide * per * min(n, window)


def cache_curve_data(ctxs):
    s8, s70 = kc.DECODER_8B, kc.DECODER_70B
    return {
        "7.5B, bf16": [cache_bytes_per_token(s8, "bf16") * c for c in ctxs],
        "7.5B, int8": [cache_bytes_per_token(s8, "int8") * c for c in ctxs],
        "70B, bf16": [cache_bytes_per_token(s70, "bf16") * c for c in ctxs],
        "70B, int8": [cache_bytes_per_token(s70, "int8") * c for c in ctxs],
        "80 layers, 4:1 windowed, int8": [hybrid_bytes(c, "int8") for c in ctxs],
    }


def cache_budget():
    n = 131072
    rows = []
    for name, s in MODELS:
        for tier, _ in TIERS:
            per = cache_bytes_per_token(s, tier)
            rows.append((name, tier, _fx(bytes_per_element(tier, s["dh"]), 5), _int(int(round(per))), _fx(_gib(per * n), 2),
                         _fx(_gib(per * 4096), 3)))
    rr = []
    for tier in ("bf16", "int8"):
        rr.append((tier, _fx(_gib(hybrid_bytes(n, tier)), 3), _fx(_gib(hybrid_bytes(4096, tier)), 3)))
    return ("Bytes per token = 2 (keys and values) x layers x key-value heads x head width x bytes per element. f32 4 bytes, bf16 2 "
            "bytes; int8 and fp8 (E4M3) 1 byte per element plus one f32 scale per (token, key-value head) row, so 1 + 4 / d_h "
            "bytes per element, 1.03125 at d_h = 128. GiB for ONE sequence.\n\n"
            + _table(["model", "tier", "bytes per element", "bytes per token", "GiB at 131,072 tokens", "GiB at 4,096 tokens"],
                     rows, "llrrrr")
            + "\n\nThe 80-layer decoder of Chapter 3 (every 5th layer global, the others a window of 1,024 keys, "
            "1,024 key and 1,024 value numbers per token and layer):\n\n"
            + _table(["tier", "GiB at 131,072 tokens", "GiB at 4,096 tokens"], rr))


# ------------------------------------------------------------------------------------------------ the error of 8-bit caches


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def round_bf16(x):
    """x rounded to float32, then to bfloat16 (round to nearest, ties to even on the 16 dropped bits)."""
    if x == 0.0:
        return 0.0
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    low = u & 0xFFFF
    u &= 0xFFFF0000
    if low > 0x8000 or (low == 0x8000 and (u >> 16) & 1):
        u += 0x10000
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _e4m3_table():
    """(magnitude, mantissa code) of every finite non-negative E4M3 value, ascending (the NaN code 0x7F is left out)."""
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            vals.append(((m * 2.0 ** -9) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7), m))
    return vals


_E4M3 = _e4m3_table()
_E4M3_MAG = [v for v, _ in _E4M3]


def round_e4m3(y):
    """The nearest finite E4M3 value to y (ties to the even mantissa code), saturating at 448."""
    a = min(abs(y), 448.0)
    i = bisect.bisect_left(_E4M3_MAG, a)
    if i == 0:
        v = _E4M3_MAG[0]
    elif i == len(_E4M3_MAG):
        v = _E4M3_MAG[-1]
    else:
        lo, hi = _E4M3[i - 1], _E4M3[i]
        dl, dh = a - lo[0], hi[0] - a
        v = lo[0] if dl < dh or (dl == dh and lo[1] % 2 == 0) else hi[0]
    return v if y >= 0 else -v


def quantize_row(row, tier):
    """Dequantized values of one (token, head) row stored in `tier`."""
    if tier == "bf16":
        return [round_bf16(x) for x in row]
    amax = max(abs(x) for x in row)
    if tier == "int8":
        scale = amax / 127.0 if amax > 0 else 1.0
        return [max(-127, min(127, round(x / scale))) * scale for x in row]
    if tier == "fp8":
        scale = amax / 448.0 if amax > 0 else 1.0
        return [round_e4m3(x / scale) * scale for x in row]
    raise KeyError(tier)


NAME_KV = "long_context.kv_error"
KV_HEADS, KV_DH, KV_N, KV_Q = 4, 64, 120, 16


def _kv_inputs():
    g = bookrng.generator(NAME_KV)
    def rows(count):
        out = []
        for _ in range(count):
            gain = math.exp(g.normal())
            out.append([gain * g.normal() for _ in range(KV_DH)])
        return out
    K = [rows(KV_N) for _ in range(KV_HEADS)]
    V = [rows(KV_N) for _ in range(KV_HEADS)]
    Q = [[[g.normal() for _ in range(KV_DH)] for _ in range(KV_Q)] for _ in range(KV_HEADS)]
    return K, V, Q


def _attend(q, K, V):
    sc = [_acc(a * b for a, b in zip(q, k)) / math.sqrt(KV_DH) for k in K]
    m = max(sc)
    e = [math.exp(x - m) for x in sc]
    z = _acc(e)
    return [_acc(e[j] / z * V[j][c] for j in range(len(K))) for c in range(KV_DH)]


def kv_error():
    K, V, Q = _kv_inputs()
    rows = []
    for tier in ("bf16", "int8", "fp8"):
        num = den = 0.0
        rowerr = rowtot = 0.0
        pred = 0.0
        for h in range(KV_HEADS):
            Kq = [quantize_row(r, tier) for r in K[h]]
            Vq = [quantize_row(r, tier) for r in V[h]]
            for r, rq in zip(K[h] + V[h], Kq + Vq):
                rowerr += _acc((a - b) ** 2 for a, b in zip(r, rq))
                rowtot += _acc(a * a for a in r)
                if tier == "int8":
                    amax = max(abs(a) for a in r)
                    pred += (amax / 127.0) ** 2 / 12.0 * KV_DH
            for q in Q[h]:
                ref, got = _attend(q, K[h], V[h]), _attend(q, Kq, Vq)
                num += _acc((a - b) ** 2 for a, b in zip(ref, got))
                den += _acc(a * a for a in ref)
        rows.append((tier, _g(math.sqrt(rowerr / rowtot), 3), _g(math.sqrt(pred / rowtot), 3) if tier == "int8" else "",
                     _g(math.sqrt(num / den), 3)))
    return (f"{KV_HEADS} key-value heads of width {KV_DH}, {KV_N} cached tokens, {KV_Q} queries per head, all random from the "
            f"generator seeded {_seed(NAME_KV)} (`seed_of` of the name `{NAME_KV}`). Draw order: the key rows of head 0 token by "
            "token, then the other heads, then the value rows in the same order, then the query vectors (head, query, channel). "
            "Each key or value row is exp(`normal()`) times a vector of 64 `normal()` draws (the gain is drawn first), so row "
            "magnitudes differ by a factor of several; queries are 64 `normal()` draws. The attention output of a query is "
            "softmax(q.k / 8) V in double precision from the cached values. 'bf16' rounds each value to float32 and then to the "
            "nearest bfloat16 (ties to even); 'int8' stores round(x / s) with s = absmax / 127 of the row, clamped to +-127; 'fp8' "
            "stores the nearest E4M3 value to x / s with s = absmax / 448 (ties to the even mantissa code). Row error: "
            "sqrt(sum of squared errors / sum of squares) over all 2 x 4 x 120 rows. Predicted (int8): each row's error is "
            "uniform in +-s/2, so its mean square is s^2 / 12 per element. Output error: the same ratio over the outputs of all "
            f"{KV_HEADS * KV_Q} queries.\n\n"
            + _table(["tier", "relative error of the stored rows", "predicted", "relative error of the attention output"], rows))


# ------------------------------------------------------------------------------------------------ chunked prefill

NAME_CHUNK = "long_context.chunk_prompt"


def chunked_prefill_memory():
    s = kc.DECODER_8B
    n, D, F, H = 131072, s["D"], s["F"], s["H"]
    w_bytes = kc.weight_params(s) * kc.BYTES
    rows = []
    for c in (131072, 16384, 2048, 512):
        mlp = 2 * c * F * kc.BYTES            # gate and up outputs of one layer, bf16
        resid = c * D * kc.BYTES
        slab = H * c * n * 4                  # scores of the last chunk against all keys, f32, if materialised
        chunks = -(-n // c)
        rows.append((_int(c), _int(chunks), _fx(_gib(mlp), 3), _fx(_gib(resid), 3), _fx(_gib(slab), 1),
                     _fx(_gib(chunks * w_bytes), 1)))
    return ("A prompt of n = 131,072 tokens through the 7.5-billion-parameter decoder of Volume V, Chapter 12 (D = 4096, F = 14336, 32 query "
            "heads of width 128, 8 key-value heads, bf16, weights 15,009,316,864 bytes), in chunks of c tokens. One layer's "
            "widest activation is the pair of outputs of the gate and up matrices, 2 c F x 2 bytes; the stream is c D x 2 bytes; "
            "a materialised score slab for the last chunk is H c n x 4 bytes (f32). Every chunk reads the weights once, so the "
            "weights are read ceil(n / c) times (the last column).\n\n"
            + _table(["chunk c", "chunks", "gate+up, GiB", "stream, GiB", "score slab, GiB", "weights read in all, GiB"], rows))


def chunk_equality():
    p, cfg = kc.model(), kc.CFG
    g = bookrng.generator(NAME_CHUNK)
    ids = [g.integer(cfg["V"]) for _ in range(12)]
    ref = kc.extend(p, cfg, kc.Cache(cfg["L"]), ids, kc.Meter(), all_logits=True)
    rows = []
    for chunk in (12, 5, 4, 1):
        cache = kc.Cache(cfg["L"])
        got = []
        for i in range(0, len(ids), chunk):
            got += kc.extend(p, cfg, cache, ids[i:i + chunk], kc.Meter(), all_logits=True)
        worst = max(kc._max_abs(a, b) for a, b in zip(ref, got))
        rows.append((chunk, -(-len(ids) // chunk), "below 1e-12" if worst < 1e-12 else f"{worst:.3e}"))
    return (f"The model of Volume V, Chapter 12 (`kv_cache.model`, seed {_seed(kc.NAME_MODEL)}) and a prompt of 12 token ids, each "
            f"`integer(12)` from the generator seeded {_seed(NAME_CHUNK)} (`seed_of` of the name `{NAME_CHUNK}`): {ids}. The prompt is "
            "run in one call and in chunks of c tokens on one cache; the logits of every position are compared. The difference is measured in double "
            "precision, and every row reads the same keys and values and adds them in the same order whatever the chunking, so the test is insensitive "
            "to the weights (any weights give the same result): it checks the bookkeeping of the cache (the positions written, the keys each row sees and "
            "the causal mask), not the numerical drift that a different summation order would cause, and not the quality of the model.\n\n"
            + _table(["chunk size c", "calls", "largest difference of any logit from the one-call prefill"], rows))


# ------------------------------------------------------------------------------------------------ offloading a sequence

BUS, DEVICE_BW, PEAK = 32e9, 800e9, 40000e9     # the illustrative GPU of Volume IV: bus, memory bandwidth, peak FLOP/s


def offload_traffic():
    s, n = kc.DECODER_8B, 131072
    per = cache_bytes_per_token(s, "int8")
    total = per * n
    flops = kc.flops_new(s, 0, n)
    k = 2048
    sel = per * k
    unit = total / DEVICE_BW                      # model time of one read of the whole cache from device memory
    rows = [
        ("read the whole cache from device memory for one decoded token", _int(int(total)), _g(total / DEVICE_BW / unit, 3)),
        ("stream the whole cache over the host bus for one decoded token", _int(int(total)), _g(total / BUS / unit, 3)),
        ("fetch only 2,048 selected positions over the host bus for one token", _int(int(sel)), _g(sel / BUS / unit, 3)),
        ("swap the sequence out (once)", _int(int(total)), _g(total / BUS / unit, 3)),
        ("swap the sequence out and back in (once each)", _int(int(2 * total)), _g(2 * total / BUS / unit, 3)),
        ("prefill the 131,072 tokens again (arithmetic only)", f"{_int(flops)} FLOPs", _g(flops / PEAK / unit, 4)),
    ]
    swap = 2 * total / BUS
    body = kc.weight_params(s) - s["V"] * s["D"]
    attn_per_key = 4 * s["L"] * s["H"] * s["dh"]
    head = 2 * s["D"] * s["V"]
    grow = []
    for m in (4096, 16384, 65536, 131072):
        w_term = 2 * m * body
        a_term = attn_per_key * (m * (m + 1) // 2)
        f_m = kc.flops_new(s, 0, m)
        assert f_m == w_term + a_term + head
        grow.append([_int(m), _g(w_term + head, 4), _g(a_term, 4), _g(f_m, 4), _g(f_m / PEAK / (2 * per * m / BUS), 3)])
    return (f"The 7.5-billion-parameter decoder (L = 32, 8 key-value heads of width 128) with an int8 cache: "
            f"{_int(int(per))} bytes per token, {_int(int(total))} bytes ({_fx(_gib(total), 2)} GiB) for 131,072 tokens. Model of "
            "Volume IV, Chapter 4: device memory 800 (GB/s), host bus 32 (GB/s), arithmetic 40,000 (GFLOP/s). The model time of an "
            "operation is its bytes over the bandwidth (or its FLOPs over the arithmetic rate); the last column divides it by the "
            "model time of the first row, one read of the whole cache from device memory (one unit).\n\n"
            + _table(["operation", "bytes or FLOPs", "model time (units)"], rows, "lrr")
            + f"\n\nThe re-prefill is a floor, since attention and the head do not reach the arithmetic rate. Model time of the "
            f"re-prefill over that of the swap out and back in: {_g(flops / PEAK / swap, 3)}.\n\n"
            "The FLOPs of the re-prefill of n tokens, from Chapter 12 of Volume V: F(n) = 2 n B + 4 L H d_h n (n + 1) / 2 + 2 D V, with "
            f"B = {_int(body)} the numbers in the weight matrices of the layers (everything but the embedding table, which is also the head), "
            f"L = {s['L']} layers, H = {s['H']} query heads of width d_h = {s['dh']} (so 4 L H d_h = {_int(attn_per_key)} for each visible key; "
            f"the sum over the n new tokens of the keys each sees is n (n + 1) / 2), and the head 2 D V = {_int(head)} once, for the last row. "
            "The first term, with the head, is linear in n; the second is quadratic. The first column of the next table is the weights term plus the head, "
            "the second the attention term, the third their sum F(n), and the last the model time of F(n) at the arithmetic rate of the model above, over the model time of "
            f"a swap out and back in (2 x {_int(int(per))} n bytes at the bus bandwidth of the model above).\n\n"
            + _table(["n tokens", "weights and head FLOPs", "attention FLOPs", "F(n)", "re-prefill time / swap time"], grow))


# ------------------------------------------------------------------------------------------------ a needle among random keys

NAME_NEEDLE = "long_context.needle"
NEEDLE_SCORE = 10.0
N_NEEDLE = (4096, 8192, 16384, 32768)


def needle_data():
    g = bookrng.generator(NAME_NEEDLE)
    xs = [SIGMA * g.normal() for _ in range(max(N_NEEDLE))]
    out = []
    for n in N_NEEDLE:
        s = n / L_TRAIN
        row = [n]
        for lam in (1.0, attention_factor(s) ** 2):
            m = max(NEEDLE_SCORE * lam, max(lam * x for x in xs[:n]))
            e = _acc(math.exp(lam * x - m) for x in xs[:n])
            nd = math.exp(NEEDLE_SCORE * lam - m)
            row.append(nd / (nd + e))
        out.append(tuple(row))
    return out


def needle_dilution():
    rows = []
    for n, w1, w2 in needle_data():
        s = n / L_TRAIN
        rows.append((_int(n), _g(s, 3), _fx(attention_factor(s) ** 2, 4), _fx(w1, 4), _fx(w2, 4)))
    return (f"One query. n noise keys each have a score of {SIGMA} x `normal()` (the first n of {max(N_NEEDLE):,} draws from the "
            f"generator seeded {_seed(NAME_NEEDLE)}, `seed_of` of the name `{NAME_NEEDLE}`) and one needle has the score "
            f"{NEEDLE_SCORE:g}, which is 5 standard deviations above their mean of 0. The table is the softmax weight of the needle. 'Unscaled': the "
            "scores as they are. 'YaRN factor': every score multiplied by lambda = (0.1 ln s + 1)^2 with s = n / 4096.\n\n"
            + _table(["n noise keys", "s", "lambda", "needle weight, unscaled", "needle weight, YaRN factor"], rows))
