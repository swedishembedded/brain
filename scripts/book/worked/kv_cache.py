# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The KV Cache": remembering the keys and values of the past instead of recomputing them.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The model is the small GPT of Chapter 10 (`training_gpt.init_gpt`: token embedding, learned
positions, pre-norm blocks of Chapter 9, a head tied to the embedding table), with random weights: cache equivalence is a
property of the computation and does not depend on training. What this module writes is the one function the chapter is
about, `extend`, which runs `n` new tokens through the model on top of a cache that already holds `c` of them:

* with an empty cache and the whole text it is the recomputation of the naive loop (and the prefill phase);
* with the cache of the text so far and one new token it is the decode phase;
* with a stand-in for the summation order of a parallel decode kernel it shows that cached and uncached logits agree to a
  tolerance and not to the last bit.

Everything that is counted is counted: `extend` adds the multiply-adds of every product it runs to a `Meter`, and the closed
forms of the chapter (`flops_new`, `generation_flops`) are asserted equal to those counts. The bytes and FLOPs of real
configurations are formulas of the shapes, evaluated, never measured. The reference the cached code is checked against is
`training_gpt.forward`, which uses the mask of Chapter 7 and no cache.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of
products are plain left-to-right additions, the order a C loop would use, except where a block says it changes the order.

Swedish Embedded AB implements solutions for serving language models without recomputing the past. If your team needs
expertise in key-value caches, inference memory budgets and decode performance, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import struct
from functools import lru_cache

import bookrng
import positions as pos_mod
import roofline
import self_attention as sa
import training_gpt as tg
import transformer_block as tb

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _bound(x):
    """The power of ten above x, written 1e-13: a difference is reported by the bound it stays under, not by its digits."""
    if x <= 0:
        return "0"
    return f"1e{math.ceil(math.log10(x * 1.0000001)):d}".replace("e-0", "e-")


def _gib(b):
    return b / 2 ** 30


# ------------------------------------------------------------------------------------------------ the model

NAME_MODEL = "kv_cache.model"
CFG = {"V": 12, "D": 16, "H": 2, "FF": 24, "L": 2, "T": 24}   # the tiny GPT of Chapter 10, a smaller vocabulary
STD = 1.0                                                        # standard deviation of every weight
PROMPT = (5, 2, 9)
N_SHORT = 4          # tokens generated in the first demonstration
N_LONG = 12          # tokens generated where a longer text is needed


@lru_cache(maxsize=None)
def model():
    return tg.init_gpt(NAME_MODEL, CFG, std=STD)


def _argmax(row):
    """Index of the largest entry, the first on a tie."""
    return max(range(len(row)), key=lambda j: (row[j], -j))


def _max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class Meter:
    """Multiply-adds executed by `extend`."""

    def __init__(self):
        self.macs = 0

    @property
    def flops(self):
        return 2 * self.macs


class Cache:
    """The keys and values of every layer: row j of `K[l]` is the key of token j in layer l, all heads side by side."""

    def __init__(self, layers):
        self.K = [[] for _ in range(layers)]
        self.V = [[] for _ in range(layers)]

    def length(self):
        return len(self.K[0])


def _mm(A, B, meter):
    meter.macs += len(A) * len(B) * len(B[0])
    return sa.matmul(A, B)


def _score(q, k, order, meter):
    """Dot product of a query and a key row. 'serial': left to right. 'lanes': the even and the odd entries summed apart, then added
    (the order of a kernel in which two threads share one dot product)."""
    meter.macs += len(q)
    if order == "serial":
        return sa._dot(q, k)
    even = odd = 0.0
    for t in range(0, len(q), 2):
        even += q[t] * k[t]
        odd += q[t + 1] * k[t + 1]
    return even + odd


def _weighted_sum(w, V, lo, dk, order, meter):
    """sum over j of w[j] * V[j][lo + d] for each d. 'serial': left to right over j. 'blocks': blocks of 4 keys summed apart,
    then the block sums added in order (the order of a kernel that gives 4 keys to each worker)."""
    meter.macs += len(w) * dk
    out = []
    for d in range(dk):
        if order == "serial":
            acc = 0.0
            for j in range(len(w)):
                acc += w[j] * V[j][lo + d]
            out.append(acc)
        else:
            total = None
            for s in range(0, len(w), 4):
                part = 0.0
                for j in range(s, min(s + 4, len(w))):
                    part += w[j] * V[j][lo + d]
                total = part if total is None else total + part
            out.append(total)
    return out


def extend(p, cfg, cache, ids, meter, positions=None, order="serial", all_logits=False):
    """Run the tokens `ids` through the model on top of `cache`, which holds the keys and values of the c tokens before them.

    The new tokens sit at positions c, c + 1, ... (or at `positions`, to show what a wrong position does). Their keys and values
    are appended to the cache, and new token i attends to cache rows 0 .. c + i, which is the causal mask. Returns the logits of
    the last new token (or of every new token with `all_logits`). `order` is the summation order of the attention sums."""
    H, D, F = cfg["H"], cfg["D"], cfg["FF"]
    dk = D // H
    scale = 1.0 / math.sqrt(dk)
    c = cache.length()
    n = len(ids)
    pos = list(range(c, c + n)) if positions is None else list(positions)
    x = [[a + b for a, b in zip(p["E"][t], p["P"][i])] for t, i in zip(ids, pos)]
    for l, blk in enumerate(p["blocks"]):
        n1, _ = tb.rms_forward(x, blk["g1"][0])
        q, k, v = _mm(n1, blk["Wq"], meter), _mm(n1, blk["Wk"], meter), _mm(n1, blk["Wv"], meter)
        cache.K[l].extend(k)
        cache.V[l].extend(v)
        attended = []
        for i in range(n):
            visible = c + i + 1
            row = []
            for h in range(H):
                lo = h * dk
                s = [_score(q[i][lo:lo + dk], cache.K[l][j][lo:lo + dk], order, meter) * scale for j in range(visible)]
                row += _weighted_sum(sa._softmax(s), cache.V[l], lo, dk, order, meter)
            attended.append(row)
        x1 = tb._add(x, _mm(attended, blk["Wo"], meter))
        n2, _ = tb.rms_forward(x1, blk["g2"][0])
        f, _ = tb.ffn_forward(blk, n2)
        meter.macs += 3 * n * D * F       # gate, up and down: the three products of the SwiGLU network
        x = tb._add(x1, f)
    rows = x if all_logits else x[-1:]
    xn, _ = tb.rms_forward(rows, p["gf"][0])
    logits = _mm(xn, sa.transpose(p["E"]), meter)
    return logits if all_logits else logits[0]


def prefill(p, cfg, prompt, meter, order="serial"):
    cache = Cache(cfg["L"])
    return cache, extend(p, cfg, cache, list(prompt), meter, order=order)


def decode_step(p, cfg, cache, token, meter, positions=None, order="serial"):
    return extend(p, cfg, cache, [token], meter, positions=positions, order=order)


def generate_naive(p, cfg, prompt, n_new):
    """Greedy generation that recomputes the whole text at every step. Returns (tokens, per step [prefix length, FLOPs, logits])."""
    ids, out, steps = list(prompt), [], []
    for _ in range(n_new):
        meter = Meter()
        _, logits = prefill(p, cfg, ids, meter)
        steps.append((len(ids), meter.flops, logits))
        tok = _argmax(logits)
        ids.append(tok)
        out.append(tok)
    return out, steps


def generate_cached(p, cfg, prompt, n_new, order="serial", positions_of=None):
    """Greedy generation with a cache: one prefill, then one decode step per new token. `positions_of(c, prompt_len)` overrides
    the position given to the token fed at cache length c. Returns (tokens, per step [new rows, cache length after, FLOPs, logits])."""
    meter = Meter()
    cache, logits = prefill(p, cfg, prompt, meter, order)
    steps = [(len(prompt), cache.length(), meter.flops, logits)]
    out = [_argmax(logits)]
    for _ in range(n_new - 1):
        meter = Meter()
        c = cache.length()
        pos = None if positions_of is None else [positions_of(c, len(prompt))]
        logits = decode_step(p, cfg, cache, out[-1], meter, positions=pos, order=order)
        steps.append((1, cache.length(), meter.flops, logits))
        out.append(_argmax(logits))
    return out, steps


# ------------------------------------------------------------------------------------------------ counting FLOPs

def shape(V, D, L, H, kv_heads, head_dim, F, mlp_matrices):
    return {"V": V, "D": D, "L": L, "H": H, "kvh": kv_heads, "dh": head_dim, "F": F, "mlp": mlp_matrices}


def weight_params(s):
    """Parameters that take part in a matrix product per token, the head included (Chapter 10's N_p, with the key and value
    projections as wide as the key-value heads: W_Q and W_O are D by H dh, W_K and W_V are D by kvh dh)."""
    per_block = 2 * s["D"] * s["H"] * s["dh"] + 2 * s["D"] * s["kvh"] * s["dh"] + s["mlp"] * s["D"] * s["F"]
    return s["L"] * per_block + s["V"] * s["D"]


def flops_new(s, start, n):
    """FLOPs of the matrix products of n new tokens at positions start .. start + n - 1 (one logits row, for the last): every
    weight matrix once per token (2 FLOPs per parameter, the head only once), plus 4 H dh per visible key per layer."""
    body = weight_params(s) - s["V"] * s["D"]
    keys = n * start + n * (n + 1) // 2         # sum over the new tokens of the number of keys each sees
    return 2 * n * body + 4 * s["L"] * s["H"] * s["dh"] * keys + 2 * s["D"] * s["V"]


def generation_flops(s, prompt, n_new):
    """(naive, cached) FLOPs to generate n_new tokens after a prompt of `prompt` tokens."""
    naive = sum(flops_new(s, 0, prompt + t) for t in range(n_new))
    cached = flops_new(s, 0, prompt) + sum(flops_new(s, prompt + t, 1) for t in range(n_new - 1))
    return naive, cached


def _tiny_shape():
    c = CFG
    return shape(c["V"], c["D"], c["L"], c["H"], c["H"], c["D"] // c["H"], c["FF"], 3)


GPT2_SMALL = shape(50257, 768, 12, 12, 12, 64, 3072, 2)
DECODER_8B = shape(128256, 4096, 32, 32, 8, 128, 14336, 3)      # eight key-value heads for 32 query heads
DECODER_8B_MHA = shape(128256, 4096, 32, 32, 32, 128, 14336, 3)  # the same model with one key-value head per query head
DECODER_70B = shape(128256, 8192, 80, 64, 8, 128, 28672, 3)
BYTES = 2    # bf16: bytes per stored weight or cache value


# ------------------------------------------------------------------------------------------------ the first demonstration

def tiny_model():
    p, c = model(), CFG
    n = tg.parameter_count(p)
    return (f"Generator: the book's SplitMix64, seed {_seed(NAME_MODEL)} (`seed_of` of the name `{NAME_MODEL}`). The model is the GPT of "
            f"Chapter 10 with V = {c['V']} token ids, D = {c['D']}, {c['H']} heads of width {c['D'] // c['H']}, L = {c['L']} pre-norm blocks "
            f"with a SwiGLU network of hidden width F = {c['FF']}, at most T = {c['T']} positions, learned absolute positions, RMS norms, a head "
            f"tied to the embedding table and no biases. Every weight is {STD} x `normal()`, drawn row by row in the order E [V, D], P [T, D], then "
            f"for each block W_Q, W_K, W_V, W_O (D by D), W_gate, W_up (D by F), W_down (F by D), with W_O and W_down multiplied by 1/sqrt(2 L); "
            f"all gains are 1. That is {_int(n)} parameters. The weights are random: nothing here is trained, and the prompt is the three token "
            f"ids {list(PROMPT)}. Greedy decoding takes the largest logit, the lowest id on a tie.")


def _check_against_masked_forward(p, ids):
    """The cached code path must equal Chapter 10's masked forward pass on every row; returns the largest difference."""
    ref, _ = tg.forward(p, list(ids), CFG, tg.causal)
    got = extend(p, CFG, Cache(CFG["L"]), list(ids), Meter(), all_logits=True)
    return max(_max_abs(a, b) for a, b in zip(ref, got))


def naive_generation():
    p = model()
    toks, steps = generate_naive(p, CFG, PROMPT, N_SHORT)
    s = _tiny_shape()
    rows = []
    total_rows = total = 0
    for t, (n, flops, _) in enumerate(steps):
        assert flops == flops_new(s, 0, n), "the counted FLOPs must equal the closed form"
        rows.append((t + 1, n, " ".join(map(str, list(PROMPT) + toks[:t])), toks[t], _int(flops)))
        total_rows += n
        total += flops
    ref = max(_max_abs(tg.forward(p, list(PROMPT) + toks[:t], CFG, tg.causal)[0][-1], steps[t][2]) for t in range(N_SHORT))
    return (f"The naive loop: at every step the whole text so far is run through the model from scratch (`extend` on an empty cache), and "
            f"the logits of its last position give the next token. Prompt {list(PROMPT)}, {N_SHORT} new tokens. FLOPs are counted by the model "
            f"run: a multiply-add is two FLOPs; the matrix products of every layer and the head (for the last position only) are counted, the "
            f"norms, the softmax and the SwiGLU gate are not (the rule of Chapter 10).\n\n"
            + _table(["step", "rows computed", "text fed in", "next token", "FLOPs"], rows)
            + f"\n\nIn all {total_rows} rows are computed and {_int(total)} FLOPs are spent, to produce {N_SHORT} tokens. The logits of the last "
            f"row agree with Chapter 10's masked forward pass (`training_gpt.forward`, no cache) to a largest difference of "
            f"{_bound(ref) if ref else '0 (identical)'} at every step.")


def mask_invariance():
    p = model()
    toks, _ = generate_naive(p, CFG, PROMPT, N_SHORT)
    ids = list(PROMPT) + toks[:3]          # six tokens
    short, long = ids[:3], ids[:6]
    unmasked = lambda i, j: True          # every position sees every other, as in Chapter 7 before the mask

    def kv(mask, upto):
        _, cache = tg.forward(p, upto, CFG, mask)
        out = []
        for l in range(CFG["L"]):
            heads = cache["caches"][l]["ca"]["heads"]
            K = [sum((hd["K"][i] for hd in heads), []) for i in range(len(upto))]
            V = [sum((hd["V"][i] for hd in heads), []) for i in range(len(upto))]
            out.append((K, V))
        return out

    rows = []
    for name, mask in (("causal, j <= i", tg.causal), ("no mask, every j", unmasked)):
        a, b = kv(mask, short), kv(mask, long)
        for l in range(CFG["L"]):
            for i in range(3):
                dk = max(_max_abs(a[l][0][i], b[l][0][i]), _max_abs(a[l][1][i], b[l][1][i]))
                rows.append((name if (l, i) == (0, 0) else "", l, i, "0 (identical)" if dk == 0 else f"{dk:.3g}"))
    return (f"The text of six tokens {long} (the prompt and the first three tokens the naive loop chose), run twice: once as its first three "
            f"tokens alone and once whole. For each layer l and each of the first three positions, the largest difference over the key and "
            f"value entries (all heads) between the two runs, for the causal mask and for no mask at all.\n\n"
            + _table(["mask", "layer", "position", "largest difference of key and value"], rows, "lrrr"))


def cached_generation():
    p = model()
    toks_n, steps_n = generate_naive(p, CFG, PROMPT, N_SHORT)
    toks_c, steps_c = generate_cached(p, CFG, PROMPT, N_SHORT)
    s = _tiny_shape()
    rows = []
    total_c = 0
    for t, (n_new, clen, flops, logits) in enumerate(steps_c):
        expect = flops_new(s, clen - n_new, n_new)
        assert flops == expect, "the counted FLOPs must equal the closed form"
        diff = _max_abs(logits, steps_n[t][2])
        rows.append((t + 1, "prefill" if t == 0 else "decode", n_new, clen, toks_c[t], _int(flops), "0" if diff == 0 else _bound(diff)))
        total_c += flops
    total_n = sum(f for _, f, _ in steps_n)
    rows_n = sum(n for n, _, _ in steps_n)
    rows_c = sum(r[2] for r in rows if isinstance(r[2], int))
    assert toks_c == toks_n
    return (f"The same model, prompt and {N_SHORT} tokens with a cache: step 1 runs the prompt through the model once (the prefill phase) and "
            f"stores the keys and values of its {len(PROMPT)} tokens in every layer; each later step feeds only the token chosen before, at position "
            f"equal to the number of tokens already in the cache, and appends one key and one value row per layer (the decode phase). 'Largest "
            f"difference' is over the {CFG['V']} logits against the naive loop's logits at the same step.\n\n"
            + _table(["step", "phase", "rows computed", "cache length after", "next token", "FLOPs", "largest difference from naive logits"], rows)
            + f"\n\nThe tokens are {toks_c}, the same as the naive loop's {toks_n}. {rows_c} rows are computed instead of {rows_n} and {_int(total_c)} "
            f"FLOPs are spent instead of {_int(total_n)}: {_fx(total_n / total_c, 2)} times fewer. Each cache holds {CFG['L']} layers of "
            f"{len(PROMPT) + N_SHORT - 1} rows of key and of value (the last token chosen is never fed back).")


def flops_growth():
    s = GPT2_SMALL
    prompt = 64
    rows = []
    for n_new in (4, 16, 64, 256, 512):
        naive, cached = generation_flops(s, prompt, n_new)
        rows.append((n_new, _int(naive), _int(cached), _fx(naive / cached, 1)))
    t = _tiny_shape()
    for n_new in (4, 12):         # the closed forms are checked against the counted runs of the tiny model
        want = generation_flops(t, len(PROMPT), n_new)
        _, sn = generate_naive(model(), CFG, PROMPT, n_new)
        _, sc = generate_cached(model(), CFG, PROMPT, n_new)
        assert (sum(f for _, f, _ in sn), sum(f for _, _, f, _ in sc)) == want
    body = weight_params(s) - s["V"] * s["D"]
    return (f"GPT-2 small (V = {_int(s['V'])}, D = {s['D']}, L = {s['L']}, 12 heads of width {s['dh']}, F = {_int(s['F'])}, a network of two "
            f"matrices) with a prompt of {prompt} tokens, FLOPs of the matrix products by the formula of the text: 2 x {_int(body)} per token for the "
            f"weights of the blocks, 4 L H d_h = {4 * s['L'] * s['H'] * s['dh']:,} per token per visible key, and the head (2 D V = {_int(2 * s['D'] * s['V'])}) "
            f"once per step. The formulas are asserted equal to the counted runs of the tiny model for 4 and for 12 new tokens.\n\n"
            + _table(["new tokens N", "naive FLOPs", "cached FLOPs", "naive / cached"], rows))


# ------------------------------------------------------------------------------------------------ equivalence and summation order

def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _sum32(xs, order):
    """Sum of float32 numbers, every addition rounded to float32 (the exact sum of two float32 numbers fits in a double, so one
    rounding of it is the IEEE float32 addition)."""
    def add(a, b):
        return _f32(a + b)

    def serial(v):
        acc = 0.0
        for x in v:
            acc = add(acc, x)
        return acc

    if order == "left to right":
        return serial(xs)
    if order == "right to left":
        return serial(xs[::-1])
    if order == "two lanes":
        return add(serial(xs[0::2]), serial(xs[1::2]))
    if order == "eight blocks of eight":
        acc = 0.0
        for b in range(0, len(xs), 8):
            acc = add(acc, serial(xs[b:b + 8]))
        return acc
    v = list(xs)
    while len(v) > 1:
        v = [add(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    return v[0]


def summation_order():
    name = "kv_cache.sums"
    g = bookrng.generator(name)
    xs = [_f32(g.normal()) for _ in range(64)]
    exact = math.fsum(xs)
    rows = []
    for order in ("left to right", "right to left", "two lanes", "eight blocks of eight", "pairwise tree"):
        v = _sum32(xs, order)
        rows.append((order, f"{v:.9g}", f"{v - exact:.2e}"))
    vals = [_sum32(xs, o) for o in ("left to right", "right to left", "two lanes", "eight blocks of eight", "pairwise tree")]
    return (f"Generator: the book's SplitMix64, seed {_seed(name)} (`seed_of` of the name `{name}`). Draws: 64 values `normal()`, each rounded to "
            f"float32. They are added in float32, every addition rounded to float32, in five orders: left to right; right to left; the even-numbered "
            f"entries and the odd-numbered entries apart, then the two sums added; eight blocks of eight consecutive entries, each summed left to "
            f"right, the block sums added left to right; and a pairwise tree (entries 0 and 1, 2 and 3, ... then the pair sums in pairs). The exact "
            f"sum of the 64 float32 values is {exact:.9g} (`math.fsum`).\n\n"
            + _table(["order", "float32 sum", "sum minus exact"], rows)
            + f"\n\nThe five sums take {len(set(vals))} different values; the largest gap between two of them is {max(vals) - min(vals):.2e}, "
            f"about {(max(vals) - min(vals)) / abs(exact):.1e} of the sum. The largest error against the exact sum is "
            f"{max(abs(v - exact) for v in vals):.2e}, which is {max(abs(v - exact) for v in vals) / abs(exact):.1e} of the sum "
            f"(relative error), {max(abs(v - exact) for v in vals) / abs(exact) / 2.0 ** -24:.1f} unit round-offs of 2^-24.")


def _generate_cached_split(p, cfg, prompt, n_new):
    """Prefill in the serial order, then decode steps whose attention sums use the lane and block orders of `extend`."""
    meter = Meter()
    cache, logits = prefill(p, cfg, prompt, meter, "serial")
    out, all_logits = [_argmax(logits)], [logits]
    for _ in range(n_new - 1):
        logits = decode_step(p, cfg, cache, out[-1], Meter(), order="split")
        out.append(_argmax(logits))
        all_logits.append(logits)
    return out, all_logits


def cache_equivalence():
    p = model()
    toks_n, steps_n = generate_naive(p, CFG, PROMPT, N_LONG)
    toks_c, steps_c = generate_cached(p, CFG, PROMPT, N_LONG)
    toks_s, logits_s = _generate_cached_split(p, CFG, PROMPT, N_LONG)
    d_serial = max(_max_abs(sc[3], sn[2]) for sc, sn in zip(steps_c, steps_n))
    d_split = max(_max_abs(a, sn[2]) for a, sn in zip(logits_s, steps_n))
    tol = 1e-12
    assert d_split < tol
    rows = [("naive loop (recompute)", "-", "-", " ".join(map(str, toks_n))),
            ("cached, summation order of the naive loop", "0 (identical)" if d_serial == 0 else f"below {_bound(d_serial)}",
             "yes" if toks_c == toks_n else "no", " ".join(map(str, toks_c))),
            ("cached, decode steps summed in lanes of 2 and blocks of 4", f"below {_bound(d_split)}", "yes" if toks_s == toks_n else "no",
             " ".join(map(str, toks_s)))]
    return (f"The same model and prompt as before, {N_LONG} greedy tokens. 'Largest difference' is over all {N_LONG} steps and all {CFG['V']} "
            f"logits of a step against the naive loop's logits at the same step. In the third row the prefill keeps the serial order but every "
            f"decode step computes each dot product of a query and a key as two partial sums (even and odd entries, then added) and each weighted "
            f"sum of values as blocks of 4 keys summed apart and then added in order: the same mathematics in the order a parallel kernel might "
            f"use. Tolerance of the check: {tol:g} on every logit (the unit round-off of double precision is 1.1e-16).\n\n"
            + _table(["generation", "largest difference of logits", "same tokens", "tokens"], rows, "lrrl")
            + f"\n\nThe largest absolute logit along the naive run is {_fx(max(max(abs(x) for x in st[2]) for st in steps_n), 2)}, and the smallest gap between "
            f"the best and the second-best logit of a step is {_fx(min(sorted(st[2])[-1] - sorted(st[2])[-2] for st in steps_n), 4)}.")


# ------------------------------------------------------------------------------------------------ the position of the new token

VARIANTS = (("correct: the position of the new token is the cache length", lambda c, plen: c),
            ("one too small: the last position that is already in the cache", lambda c, plen: c - 1),
            ("never advanced: the length of the prompt for every decode step", lambda c, plen: plen),
            ("restarted: position 0, the index of the row inside the call", lambda c, plen: 0))


def _teacher_forced(p, cfg, prompt, fed, positions_of):
    """Logits of the decode steps that feed the tokens `fed` (the reference tokens) with the position `positions_of` gives each."""
    cache, _ = prefill(p, cfg, prompt, Meter())
    out = []
    for tok in fed:
        c = cache.length()
        out.append(decode_step(p, cfg, cache, tok, Meter(), positions=[positions_of(c, len(prompt))]))
    return out


def position_bugs():
    p = model()
    toks_n, steps_n = generate_naive(p, CFG, PROMPT, N_LONG)
    fed = toks_n[:-1]                      # the reference tokens fed back at decode steps 2 .. N_LONG
    rows = []
    for label, fn in VARIANTS:
        logits = _teacher_forced(p, CFG, PROMPT, fed, fn)
        diffs = [_max_abs(lg, steps_n[t + 1][2]) for t, lg in enumerate(logits)]
        flips = sum(1 for t, lg in enumerate(logits) if _argmax(lg) != _argmax(steps_n[t + 1][2]))
        toks, _ = generate_cached(p, CFG, PROMPT, N_LONG, positions_of=fn)
        first = next((i + 1 for i, (a, b) in enumerate(zip(toks, toks_n)) if a != b), None)
        rows.append((label, "0 (identical)" if max(diffs) == 0 else _fx(max(diffs), 2), flips, "none" if first is None else first))
    return (f"The cached generation of the last block ({N_LONG} tokens), with the position given to the token fed at each decode step changed. "
            f"The prefill is always right (positions 0, 1, 2) and the new keys and values are always appended at the end of the cache; only "
            f"the number added to the token's embedding differs. Two measurements. Teacher forced: every variant is fed the naive loop's own tokens "
            f"({' '.join(map(str, toks_n[:-1]))}) at decode steps 2 to {N_LONG}, so any difference in the logits comes from the position alone; the "
            f"table gives the largest difference over those {N_LONG - 1} steps and the number of steps at which the largest logit moves to another "
            f"token. Free running: the step at which the variant's own generated token first differs from the naive loop's.\n\n"
            + _table(["position given to the fed token", "largest logit difference", f"steps with another top token, of {N_LONG - 1}",
                      "first wrong token when run freely"], rows, "lrrr"))


NAME_ROPE = "kv_cache.rope"


def rope_cache():
    d, n_keys = 8, 5
    g = bookrng.generator(NAME_ROPE)
    q = [round(g.normal(), 1) for _ in range(d)]
    keys = [[round(g.normal(), 1) for _ in range(d)] for _ in range(n_keys)]
    ws = pos_mod.omegas(d)
    qpos = n_keys
    dot = pos_mod._dot
    q_rot = pos_mod.rotate(q, qpos, ws)
    rotated = [pos_mod.rotate(k, j, ws) for j, k in enumerate(keys)]       # rotated once, when the token is cached
    ref = [dot(q_rot, kr) for kr in rotated]
    cases = [
        ("recomputed: every key rotated by its own position", ref),
        ("cached after rotation, read later", [dot(q_rot, kr) for kr in rotated]),
        ("cached before rotation, only the query rotated", [dot(q_rot, k) for k in keys]),
        ("cached after rotation, rotated again when read", [dot(q_rot, pos_mod.rotate(kr, j, ws)) for j, kr in enumerate(rotated)]),
        ("query rotated by position 4 instead of 5", [dot(pos_mod.rotate(q, qpos - 1, ws), kr) for kr in rotated]),
    ]
    rows = []
    for label, sc in cases:
        wts = sa._softmax(sc)
        rows.append((label, " ".join(_fx(v, 3) for v in sc), _fx(max(abs(a - b) for a, b in zip(sc, ref)), 3), _argmax(wts)))
    return (f"Generator: the book's SplitMix64, seed {_seed(NAME_ROPE)} (`seed_of` of the name `{NAME_ROPE}`). Draws: the 8 entries of the query q, then "
            f"the 8 entries of each of 5 keys k_0 .. k_4, each `round(normal(), 1)`. One head of width 8, adjacent pairs, base 10000 (Chapter 8, "
            f"`positions.rotate`). The query is the token at position 5 and attends to the five cached keys; each score is the dot product of the "
            f"rotated query and the key as stored, without the scale 1/sqrt(8).\n\n"
            + _table(["how the keys are cached and read", "scores for keys 0 .. 4", "largest difference from recomputed", "key with most weight"],
                     rows, "llrr"))


# ------------------------------------------------------------------------------------------------ bytes, intensity and memory

def cache_bytes_per_token(s, b=BYTES):
    """2 (keys and values) x layers x key-value heads x head width x bytes per value."""
    return 2 * s["L"] * s["kvh"] * s["dh"] * b


def decode_work(s, ctx, batch=1, b=BYTES):
    """(FLOPs, weight bytes, cache bytes read) of one decode step for `batch` sequences that each hold ctx tokens: the weights
    are read once for the step, every sequence reads its own cache."""
    flops = batch * flops_new(s, ctx, 1)
    return flops, weight_params(s) * b, batch * cache_bytes_per_token(s, b) * ctx


def prefill_work(s, n, b=BYTES):
    """(FLOPs, weight bytes, cache bytes written) of a prefill of n tokens into an empty cache."""
    return flops_new(s, 0, n), weight_params(s) * b, cache_bytes_per_token(s, b) * n


def bytes_split_data(s, ctxs, b=BYTES):
    """GB per decoded token, batch 1, for each context length: (weights, cache)."""
    w = weight_params(s) * b / 1e9
    return [(w, cache_bytes_per_token(s, b) * c / 1e9) for c in ctxs]


def roofline_points(s=DECODER_8B):
    """((label, intensity) for decode at batch 1, 8, 64 at ctx 4096 and prefill of 16, 128, 1024, 8192 tokens), FLOP per byte."""
    out = []
    for batch in (1, 8, 64):
        f, wb, cb = decode_work(s, 4096, batch)
        out.append((f"decode, batch {batch}", f / (wb + cb)))
    for n in (16, 128, 1024, 8192):
        f, wb, cb = prefill_work(s, n)
        out.append((f"prefill, {n} tokens", f / (wb + cb)))
    return out


def decode_and_prefill():
    s = DECODER_8B
    peak_g, bw_g = roofline.MACHINES["GPU"]
    ridge = peak_g / bw_g
    n_params = weight_params(s)
    rows = []
    for ctx in (1024, 8192, 32768, 131072):
        f, wb, cb = decode_work(s, ctx)
        i = f / (wb + cb)
        rows.append(("decode, one token", f"context {_int(ctx)}", _int(f), _int(wb), _int(cb), _fx(i, 2), "bandwidth" if i < ridge else "compute"))
    for n in (1, 16, 128, 1024, 8192):
        f, wb, cb = prefill_work(s, n)
        i = f / (wb + cb)
        rows.append(("prefill", f"{_int(n)} token" + ("" if n == 1 else "s"), _int(f), _int(wb), _int(cb), _fx(i, 2), "bandwidth" if i < ridge else "compute"))
    batch_rows = []
    for batch in (1, 8, 64, 256):
        f, wb, cb = decode_work(s, 4096, batch)
        batch_rows.append((batch, _int(f), _int(wb), _int(cb), _fx(f / (wb + cb), 2), _fx(min(1.0, f / (wb + cb) / ridge), 3)))
    n_star = next(n for n in range(1, 100000) if prefill_work(s, n)[0] / (prefill_work(s, n)[1] + prefill_work(s, n)[2]) >= ridge)
    f1, w1, c1 = decode_work(s, 4096)
    limit = f1 / c1
    return (f"A decoder with D = {s['D']}, L = {s['L']}, {s['H']} query heads and {s['kvh']} key-value heads of width {s['dh']}, F = {_int(s['F'])} (SwiGLU), "
            f"V = {_int(s['V'])}, values stored in bf16 ({BYTES} bytes). Its {_int(n_params)} parameters in matrix products are {_int(n_params * BYTES)} bytes. "
            f"FLOPs: the formula of the text, {2 * (n_params - s['V'] * s['D']):,} per token for the blocks, 4 L H d_h = "
            f"{4 * s['L'] * s['H'] * s['dh']:,} per visible key, and the head once. Bytes: the weights once (the embedding lookup reads one row and is "
            f"left out) plus the cache rows read (decode) or written (prefill) at {_int(cache_bytes_per_token(s))} bytes per token; activations are not counted. "
            f"A decode step at context c counts the c stored rows as bytes read, while its attention FLOPs count c + 1 keys: the new token attends to its own "
            f"key and value too, which it has just computed, so they are not read from the cache and their write is left out of the bytes. "
            f"The decoder's head is a matrix of its own (D by V), not tied to the embedding table, and counted once per step. "
            f"The device is the illustrative GPU of Volume IV (F_pk = {peak_g:,.0f} (GFLOP/s), BW = {bw_g:,.0f} (GB/s)), whose ridge is {ridge:g} FLOP per byte.\n\n"
            + _table(["phase", "size", "FLOPs", "weight bytes", "cache bytes", "FLOP per byte", "limited by"], rows, "llrrrrl")
            + "\n\nDecoding at context 4,096 for several sequences at once: the weights are read once for the step and every sequence reads its own cache.\n\n"
            + _table(["sequences", "FLOPs", "weight bytes", "cache bytes", "FLOP per byte", "fraction of peak attainable"], batch_rows)
            + f"\n\nThe shortest prompt whose prefill reaches the ridge ({ridge:g} FLOP per byte) is {n_star} tokens. With ever more sequences at context 4,096 "
            f"the weights are shared by all of them and the intensity tends to the FLOPs of one sequence's step over its own cache bytes, "
            f"{_int(f1)} / {_int(c1)} = {_fx(limit, 2)} FLOP per byte.")


def cache_memory():
    ctxs = (1024, 4096, 32768, 131072)
    configs = (("GPT-2 small", GPT2_SMALL), ("7.5-billion decoder, 8 key-value heads", DECODER_8B),
               ("same, 32 key-value heads", DECODER_8B_MHA), ("70-billion decoder, 8 key-value heads", DECODER_70B))
    rows = []
    for name, s in configs:
        per = cache_bytes_per_token(s)
        rows.append((name, f"2 x {s['L']} x {s['kvh']} x {s['dh']} x {BYTES}", _int(per)) + tuple(f"{_gib(per * c):.2f}" for c in ctxs)
                    + (f"{_gib(weight_params(s) * BYTES):.1f}",))
    arch = []
    for name, s in configs:
        arch.append((name, _int(s["V"]), _int(s["D"]), s["L"], s["H"], s["kvh"], s["dh"], _int(s["F"]), s["mlp"], _int(weight_params(s))))
    device = 80.0
    fit = []
    for name, s in configs[1:3]:
        free = device - _gib(weight_params(s) * BYTES)
        per_seq = _gib(cache_bytes_per_token(s) * 32768)
        fit.append((name, f"{free:.1f}", f"{per_seq:.2f}", max(0, int(free // per_seq))))
    return ("The four architectures, with every number the weights need. The weights counted are the parameters that take part in a matrix product "
            "per token, as in the text: per block W_Q and W_O (D by H d_h each), W_K and W_V (D by h_kv d_h each) and the feed-forward network "
            "(m matrices D by F, m = 2 for GPT-2 small's GELU network and 3 for the SwiGLU network of the others), times L, plus one head D by V "
            "(a matrix of its own, not tied to the embedding table, whose lookup is not a product); norm gains and position tables are left out. "
            "'Same, 32 key-value heads' is the 7.5-billion decoder with W_K and W_V widened from D by 1,024 to D by 4,096, which is why its weights "
            "are larger.\n\n"
            + _table(["model", "V", "D", "L", "query heads H", "key-value heads h_kv", "head width d_h", "F", "m", "parameters in matrix products"],
                     arch, "lrrrrrrrrr")
            + "\n\nCache bytes per token = 2 (keys and values) x layers x key-value heads x head width x bytes per value, in bf16 (2 bytes). The context "
            "columns are GiB of cache for ONE sequence of that many tokens; the last column is the GiB of the weights in bf16 (2 bytes per "
            "parameter of the table above), for comparison.\n\n"
            + _table(["model", "2 x L x kv heads x d_h x bytes", "bytes per token"] + [f"GiB at {_int(c)}" for c in ctxs] + ["weights, GiB"], rows, "llrrrrrr")
            + f"\n\nSessions of 32,768 tokens that fit beside the weights on a device of {device:.0f} GiB (the cache is the only other memory counted):\n\n"
            + _table(["model", "GiB free after the weights", "GiB per session", "sessions that fit"], fit))
