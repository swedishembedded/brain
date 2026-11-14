# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Benchmarking and Performance Claims".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: benchmarking.<name> -->` markers. Helpers with a leading
underscore are shared with the chapter's figures, so a picture and the table
beside it cannot disagree.

Nothing here measures a machine. The served engine is a model: a decoder of
1.5e9 parameters on the model GPU unit of Volume IV, whose every iteration
costs the host's fixed share plus the roofline time of the FLOPs and bytes it
does, at stated engine rates. Times are consequences of those counts and
rates, in model milliseconds. Randomness comes from the book's own generator
(`bookrng.generator("benchmarking.<example>")`), is printed with its seed,
and is drawn in an order the chapter states.

Swedish Embedded AB implements solutions for benchmarking machine learning
systems so that a performance claim survives review. If your team needs
expertise in end-to-end benchmarks of inference servers and in honest
performance reporting, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import kernel_selection as ks
import profiling as pr

# ---------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    return f"{v:,}"


def _f(v, d=1):
    return f"{v:,.{d}f}"


def _seed_line(name):
    return (f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`, "
            "defined on the pseudocode page).")


def _percentile(a, q):
    """The pseudocode page's percentile: linear interpolation between neighbouring order statistics."""
    v = sorted(a)
    m = len(v)
    pos = q * (m - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, m - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def _mean(a):
    return sum(a) / len(a)


def _sd(a):
    mu = _mean(a)
    return math.sqrt(sum((x - mu) ** 2 for x in a) / (len(a) - 1))


# ---------------------------------------------------------------- the served engine (a model)

N_PARAM = 1.5e9            # parameters of the model decoder
W_BYTES = 3.0e9            # its weights in 16-bit numbers: 2 bytes each
KV_TOKEN = 32768.0         # bytes of keys and values one token of context holds (and a decode step reads)
F_ENG = 20000.0e9          # FLOP/s the engine's kernels reach on the model GPU unit (0.58 of its measured arithmetic roof, `pr.F_MS`)
BW_ENG = 520.0e9           # bytes/s the engine's kernels reach (0.80 of its measured bandwidth roof, `pr.BW_MS`)
T_HOST = 0.4               # model milliseconds of host work per iteration: scheduling, sampling, the launch
T_COLD = 300.0             # model milliseconds added to the first iteration of the engine's life (pipelines are built)
B_MAX = 16                 # sequences that fit in the key-value budget at once
N_MEASURED = 64            # requests a run measures ...
N_WARM = 4                 # ... after this many warm-up requests, which no statistic sees
IN_LO, IN_SPAN = 64, 257   # prompt tokens: 64 + integer(257), so 64 .. 320
OUT_LO, OUT_SPAN = 32, 97  # output tokens: 32 + integer(97), so 32 .. 128
SLO_FIRST = 600.0          # model milliseconds allowed from submission to the first token
SLO_GAP = 60.0             # model milliseconds allowed between any two consecutive tokens of a request


def _iter_ms(flops, nbytes, host=T_HOST, f_eng=F_ENG, bw_eng=BW_ENG):
    """One iteration of the engine: the host's share plus the roofline time of its FLOPs and bytes, in milliseconds."""
    return host + 1000.0 * max(flops / f_eng, nbytes / bw_eng)


def _prefill_ms(n_in, **kw):
    return _iter_ms(2.0 * N_PARAM * n_in, W_BYTES + n_in * KV_TOKEN, **kw)


def _decode_ms(contexts, w_bytes=W_BYTES, **kw):
    return _iter_ms(2.0 * N_PARAM * len(contexts), w_bytes + KV_TOKEN * sum(contexts), **kw)


def _lengths(gen, n):
    """n requests, each (prompt tokens, output tokens), drawn in request order, the prompt first."""
    out = []
    for _ in range(n):
        n_in = IN_LO + gen.integer(IN_SPAN)
        n_out = OUT_LO + gen.integer(OUT_SPAN)
        out.append((n_in, n_out))
    return out


def _stream(name, n=N_MEASURED + N_WARM):
    return _lengths(bookrng.generator(name), n)


def _poisson_arrivals(gen, n, per_s):
    """Arrival times in model milliseconds: the first after one exponential gap, each gap -ln(1 - u) / rate."""
    t, out = 0.0, []
    for _ in range(n):
        t += -math.log(1.0 - gen.uniform()) / per_s * 1000.0
        out.append(t)
    return out


class _Req:
    __slots__ = ("i", "n_in", "n_out", "warm", "submit", "admit", "first", "done", "arts", "made")

    def __init__(self, i, n_in, n_out, warm, submit):
        self.i, self.n_in, self.n_out, self.warm, self.submit = i, n_in, n_out, warm, submit
        self.admit = self.first = self.done = None
        self.arts = []
        self.made = 0


def _serve(reqs, mode, conc=1, arrivals=None, host=T_HOST, w_bytes=W_BYTES, warm=N_WARM, split=None):
    """Run the model engine over `reqs` (list of (prompt, output) tokens) and return every request's record.

    mode 'saturated': every request is submitted at time 0. 'closed': `conc` requests are in flight at once and a new
    one is submitted at the moment one finishes. 'rate': request j is submitted at arrivals[j]. The scheduler: when a
    request waits and fewer than B_MAX run, and the previous iteration was not a prefill (or nothing runs), the oldest
    waiting request is admitted and its prompt is prefilled in one iteration, which also yields its first token;
    otherwise one decode step gives every running sequence one token."""
    n = len(reqs)
    recs, waiting, running = [], [], []
    t, nxt, in_flight, cold, last_prefill = 0.0, 0, 0, True, False

    def submit(at):
        nonlocal nxt, in_flight
        r = _Req(nxt, reqs[nxt][0], reqs[nxt][1], nxt < warm, at)
        recs.append(r)
        waiting.append(r)
        nxt += 1
        in_flight += 1

    def finish(r):
        nonlocal in_flight
        r.done = t
        in_flight -= 1

    if mode == "saturated":
        while nxt < n:
            submit(0.0)
    while True:
        if mode == "closed":
            while nxt < n and in_flight < conc:
                submit(t)
        elif mode == "rate":
            while nxt < n and arrivals[nxt] <= t:
                submit(arrivals[nxt])
        if not waiting and not running:
            if nxt >= n:
                break
            if mode == "rate":
                t = arrivals[nxt]
                continue
        extra = T_COLD if cold else 0.0
        cold = False
        if waiting and len(running) < B_MAX and (not last_prefill or not running):
            r = waiting.pop(0)
            r.admit = t
            cost = _prefill_ms(r.n_in, host=host)
            if split is not None:
                split["host"] += host
                split["prefill"] += cost - host
            t += cost + extra
            r.first = t
            r.arts.append(t)
            r.made = 1
            if r.made == r.n_out:
                finish(r)
            else:
                running.append(r)
            last_prefill = True
        else:
            ctx = [r.n_in + r.made for r in running]
            cost = _decode_ms(ctx, w_bytes=w_bytes, host=host)
            if split is not None:
                split["host"] += host
                split["decode"] += cost - host
            t += cost + extra
            still = []
            for r in running:
                r.made += 1
                r.arts.append(t)
                if r.made == r.n_out:
                    finish(r)
                else:
                    still.append(r)
            running = still
            last_prefill = False
    return recs


def _meets(r):
    if r.done is None:
        return False
    if r.first - r.submit > SLO_FIRST:
        return False
    return all(b - a <= SLO_GAP for a, b in zip(r.arts, r.arts[1:]))


def _summary(recs):
    """The statistics of the measured (non-warm-up) records, every rule of the text in one place."""
    m = [r for r in recs if not r.warm]
    t0 = min(r.submit for r in m)
    t1 = max(r.done for r in m)
    wall_s = (t1 - t0) / 1000.0
    out_tok = sum(r.made for r in m)
    good = sum(r.made for r in m if _meets(r))
    ttfa = [r.first - r.submit for r in m]
    gaps = [b - a for r in m for a, b in zip(r.arts, r.arts[1:])]
    e2e = [r.done - r.submit for r in m]
    return dict(n=len(m), wall_s=wall_s, req_s=len(m) / wall_s, tok_s=out_tok / wall_s, good_s=good / wall_s,
                attain=sum(1 for r in m if _meets(r)) / len(m), out_tok=out_tok,
                ttfa50=_percentile(ttfa, 0.5), ttfa99=_percentile(ttfa, 0.99),
                gap50=_percentile(gaps, 0.5), gap99=_percentile(gaps, 0.99), gapmax=max(gaps),
                e2e50=_percentile(e2e, 0.5))


ENGINE_LINE = (
    "The model engine of this chapter, every parameter labelled, chosen by the author and none a measurement. A decoder of 1.5e9 parameters held "
    "in 16-bit numbers (3.0e9 bytes of weights) on the model GPU unit of Volume IV, whose kernels reach 20,000 (GFLOP/s) "
    "and 520 (GB/s) (0.58 and 0.80 of its measured roofline, 34,469 and 646). An iteration costs 0.4 model milliseconds of "
    "host work plus max(FLOPs / 20,000 (GFLOP/s), bytes / 520 (GB/s)). A prefill of a prompt of n tokens does 2 x 1.5e9 x n "
    "FLOPs and moves 3.0e9 + 32,768 n bytes and yields the request's first token; a decode step over b running sequences "
    "does 2 x 1.5e9 x b FLOPs and moves 3.0e9 bytes plus 32,768 bytes for every token of context each sequence holds "
    "(its prompt and the tokens it has produced); the arithmetic of attention is left out of the count. At most 16 "
    "sequences run at once; a waiting request is admitted (oldest first) when a place is free and the previous iteration "
    "was not a prefill, so at most one prefill falls between two decode steps. The first iteration of the engine's life "
    "costs 300 model milliseconds more (pipelines are built). Each run serves 4 warm-up requests and then 64 measured "
    "ones; a request's prompt is 64 + integer(257) tokens and its output 32 + integer(97), drawn in request order with the "
    "prompt first. Statistics use the measured requests only, over the window from the first measured submission to the "
    "last measured completion; percentiles interpolate as the pseudocode page's `percentile`; gaps are pooled over all "
    "measured requests. A request meets its service-level objective when its first token comes within 600 model "
    "milliseconds of its submission and no gap between two of its tokens exceeds 60."
)


# ---------------------------------------------------------------- the opening and the arrival processes

def _row_summary(label, s):
    return (label, _f(s["tok_s"]), _f(s["good_s"]), f"{s['attain']:.3f}", _f(s["ttfa50"]), _f(s["ttfa99"]),
            _f(s["gap99"]))


def opening_claims():
    reqs = _stream("benchmarking.opening")
    sat = _summary(_serve(reqs, "saturated"))
    c4 = _summary(_serve(reqs, "closed", 4))
    rows = [_row_summary("all 68 requests submitted at once", sat),
            _row_summary("4 requests in flight at a time", c4)]
    return ("The same engine and the same 68 requests (4 warm-up, 64 measured), served two ways. The engine is a model whose rates the author chose, and no number here is a measurement. "
            + _seed_line("benchmarking.opening")
            + " Draw order: the 68 requests in order, the prompt length and then the output length of each. "
            "Rates in tokens per second and times in milliseconds are of the model engine that the block "
            "`benchmarking.request_timeline` specifies in full; 'goodput' counts only the tokens of requests that met the service-level objective.\n\n"
            + _table(["how the requests arrive", "output tokens per second", "goodput (tokens per second)",
                      "share meeting the objective", "time to first token, median (milliseconds)",
                      "time to first token, 99th percentile", "gap between tokens, 99th percentile"],
                     rows, "lrrrrrr"))


def request_timeline():
    reqs = _stream("benchmarking.opening")
    recs = _serve(reqs, "closed", 4)
    rows = []
    for r in recs[:10]:
        gaps = [b - a for a, b in zip(r.arts, r.arts[1:])]
        rows.append((r.i, "warm-up" if r.warm else "measured", r.n_in, r.n_out, _f(r.submit), _f(r.admit), _f(r.first),
                     _f(r.done), _f(r.first - r.submit), _f(max(gaps)), _f(r.done - r.submit),
                     "yes" if _meets(r) else "no"))
    return (ENGINE_LINE + "\n\nThe first 10 requests of the run with 4 in flight of `benchmarking.opening_claims` (the same "
            "68 requests, the same seed). Requests are numbered from 0; times are model milliseconds from the start of "
            "the run.\n\n"
            + _table(["request", "kind", "prompt tokens", "output tokens", "submitted", "admitted", "first token",
                      "done", "time to first token", "longest gap", "end to end", "meets the objective"], rows,
                     "rlrrrrrrrrrl"))


def arrival_processes():
    reqs = _stream("benchmarking.opening")
    gen = bookrng.generator("benchmarking.arrivals")
    arr8 = _poisson_arrivals(gen, len(reqs), 8.0)
    arr20 = _poisson_arrivals(gen, len(reqs), 20.0)
    cases = [("saturated: all at once", _serve(reqs, "saturated")),
             ("closed loop, 1 in flight", _serve(reqs, "closed", 1)),
             ("closed loop, 8 in flight", _serve(reqs, "closed", 8)),
             ("closed loop, 16 in flight", _serve(reqs, "closed", 16)),
             ("open loop, 8 requests per second", _serve(reqs, "rate", arrivals=arr8)),
             ("open loop, 20 requests per second", _serve(reqs, "rate", arrivals=arr20))]
    rows = []
    for label, recs in cases:
        s = _summary(recs)
        rows.append((label, _f(s["req_s"], 2), _f(s["tok_s"]), _f(s["good_s"]), f"{s['attain']:.3f}", _f(s["ttfa50"]),
                     _f(s["ttfa99"]), _f(s["gap99"]), _f(s["e2e50"])))
    return ("The 68 requests of `benchmarking.opening_claims` under six arrival processes. The open-loop rows draw "
            "their arrival times from a second generator: " + _seed_line("benchmarking.arrivals")
            + " Draw order: 68 gaps for the rate of 8 per second, then 68 for the rate of 20; a gap is "
            "-ln(1 - u) / rate seconds with u = rng.uniform() and ln the library natural logarithm, and request j "
            "arrives at the sum of the first j + 1 gaps. Rates per second and times in milliseconds of the model engine, whose rates the author chose (no number here is a measurement).\n\n"
            + _table(["arrival process", "requests per second", "output tokens per second", "goodput (tokens per second)",
                      "share meeting the objective", "first token, median (milliseconds)", "first token, 99th percentile",
                      "gap, 99th percentile", "end to end, median"], rows, "lrrrrrrrr"))


SWEEP_LEVELS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48)


def _sweep(reqs=None):
    reqs = reqs or _stream("benchmarking.opening")
    return [(c, _summary(_serve(reqs, "closed", c))) for c in SWEEP_LEVELS]


def concurrency_sweep():
    pts = _sweep()
    rows = [(c, _f(s["tok_s"]), _f(s["good_s"]), f"{s['attain']:.3f}", _f(s["ttfa99"]), _f(s["gap99"])) for c, s in pts]
    peak = max(pts, key=lambda p: p[1]["tok_s"])
    held = [p for p in pts if p[1]["attain"] >= 0.99]
    best = max(held, key=lambda p: p[1]["good_s"]) if held else None
    tail = (f"\n\nPeak output rate: {_f(peak[1]['tok_s'])} tokens per second at {peak[0]} in flight. "
            + (f"Highest goodput among the levels where at least 0.99 of the requests met the objective: "
               f"{_f(best[1]['good_s'])} tokens per second at {best[0]} in flight." if best else
               "No level held the objective for 0.99 of its requests."))
    return ("A closed-loop ladder over the same 68 requests of `benchmarking.opening_claims` (the same seed, the same "
            "requests at every level, the engine reset between levels so that each level's first iteration pays the "
            "cold start, which falls on a warm-up request). Rates per second and times in milliseconds of the model engine, whose rates the author chose (no number here is a measurement).\n\n"
            + _table(["requests in flight", "output tokens per second", "goodput (tokens per second)",
                      "share meeting the objective", "first token, 99th percentile (milliseconds)",
                      "gap, 99th percentile (milliseconds)"], rows, "rrrrrr") + tail)


# ---------------------------------------------------------------- Amdahl at the scale of a request

def _split(reqs, conc, w_bytes=W_BYTES):
    s = dict(host=0.0, prefill=0.0, decode=0.0)
    recs = _serve(reqs, "closed", conc, w_bytes=w_bytes, split=s)
    return s, recs


def amdahl_request():
    reqs = _stream("benchmarking.opening")
    rows = []
    for c in (1, 16):
        tot, recs = _split(reqs, c)
        tot2, recs2 = _split(reqs, c, W_BYTES / 2.0)
        busy = tot["host"] + tot["prefill"] + tot["decode"]
        p_dec = tot["decode"] / busy
        g = tot["decode"] / tot2["decode"]
        ratio = _summary(recs2)["tok_s"] / _summary(recs)["tok_s"]
        rows.append((c, f"{tot['host'] / busy:.3f}", f"{tot['prefill'] / busy:.3f}", f"{p_dec:.3f}", f"{g:.3f}",
                     f"{pr.amdahl(p_dec, g):.3f}", f"{ratio:.3f}", f"{1.0 / (1.0 - p_dec):.3f}"))
    return ("The engine's busy time (the model engine of `benchmarking.request_timeline`, whose rates the author chose; no number here is a measurement) for all 68 requests split into the host's 0.4 model milliseconds per iteration, the "
            "prefill work and the decode work (the cold start left out), for the closed-loop runs with 1 and with 16 in "
            "flight of `benchmarking.concurrency_sweep`. The change: weights stored in 8-bit numbers and nothing else, so a "
            "decode step reads 1.5e9 bytes of weights in place of 3.0e9 (a prefill is compute-bound and its count is "
            "unchanged). g is the decode work before over after, from a rerun of the changed engine; A(p, g) = "
            "1 / ((1 - p) + p / g) with p the decode share; 'rerun' is the output rate of the changed engine over the "
            "original's, measured over each run's own window.\n\n"
            + _table(["requests in flight", "host share", "prefill share", "decode share p", "g on decode",
                      "Amdahl A(p, g)", "rerun: new output rate / old", "ceiling 1 / (1 - p)"], rows, "rrrrrrrr"))


# ---------------------------------------------------------------- fast and wrong: the output check

FID_V, FID_H = 16, 8          # vocabulary and width of the toy decoder
FID_PROMPT, FID_NEW = 3, 24   # prompt tokens and tokens generated per probe
FID_PROBES = 3
FID_EARLY = 16                # the early-stopping variant produces this many tokens


def _fid_weights():
    """E [V, H], A [H, H], U [V, H], each entry (integer(512) - 256) / 256, drawn E, A, U in row-major order; then the
    prompts, FID_PROBES of FID_PROMPT tokens, each integer(FID_V)."""
    gen = bookrng.generator("benchmarking.fidelity")
    draw = lambda r, c: [[(gen.integer(512) - 256) / 256.0 for _ in range(c)] for _ in range(r)]  # noqa: E731
    E, A, U = draw(FID_V, FID_H), draw(FID_H, FID_H), draw(FID_V, FID_H)
    prompts = [[gen.integer(FID_V) for _ in range(FID_PROMPT)] for _ in range(FID_PROBES)]
    return E, A, U, prompts


def _q8(M):
    """Each entry rounded to a multiple of 1/64, halves away from zero: it fits one signed byte with scale 1/64."""
    def r(w):
        s = w * 64.0
        return (math.floor(abs(s) + 0.5) * (1 if s >= 0 else -1)) / 64.0
    return [[r(w) for w in row] for row in M]


def _argmax(z):
    best = 0
    for i in range(1, len(z)):
        if z[i] > z[best]:
            best = i
    return best


def _decode_toy(E, A, U, prompt, n_new, recurrent=True, reverse=False):
    """h starts at zero; every token x (prompt, then each generated token) updates h = clamp(A h + E[x], -1, 1), each
    entry summed over j ascending (descending when reverse) and E[x] added last; after the last prompt token and after
    each generated one, the next token is argmax(U h), U h summed over j ascending, ties to the lowest id."""
    h = [0.0] * FID_H
    out = []

    def step(x):
        nonlocal h
        new = []
        for i in range(FID_H):
            s = 0.0
            if recurrent:
                order = range(FID_H - 1, -1, -1) if reverse else range(FID_H)
                for j in order:
                    s += A[i][j] * h[j]
            s += E[x][i]
            new.append(min(max(s, -1.0), 1.0))
        h = new

    for x in prompt:
        step(x)
    for _ in range(n_new):
        z = []
        for v in range(FID_V):
            s = 0.0
            for j in range(FID_H):
                s += U[v][j] * h[j]
            z.append(s)
        y = _argmax(z)
        out.append(y)
        step(y)
    return out


def _agreement(cand, ref):
    """Position-by-position agreement over max(len) of each pair; a missing position disagrees; zero positions fail."""
    compared = agree = 0
    first = None
    for k, (c, r) in enumerate(zip(cand, ref)):
        for p in range(max(len(c), len(r))):
            compared += 1
            if p < len(c) and p < len(r) and c[p] == r[p]:
                agree += 1
            elif first is None:
                first = (k, p)
    passed = compared > 0 and agree == compared and len(cand) == len(ref)
    return agree, compared, first, passed


def _prefix_score(cand, ref):
    """The lax scorer the text warns about: agreement over the common prefix of each pair only."""
    agree = compared = 0
    for c, r in zip(cand, ref):
        for p in range(min(len(c), len(r))):
            compared += 1
            agree += c[p] == r[p]
    return agree / compared if compared else 1.0


def fidelity_gate():
    E, A, U, prompts = _fid_weights()
    ref = [_decode_toy(E, A, U, p, FID_NEW) for p in prompts]
    E8, A8, U8 = _q8(E), _q8(A), _q8(U)
    variants = [
        ("the reference itself", [_decode_toy(E, A, U, p, FID_NEW) for p in prompts], 2, True),
        ("sums over j in descending order", [_decode_toy(E, A, U, p, FID_NEW, reverse=True) for p in prompts], 2, True),
        ("weights rounded to 8 bits", [_decode_toy(E8, A8, U8, p, FID_NEW) for p in prompts], 1, True),
        ("recurrent term dropped", [_decode_toy(E, A, U, p, FID_NEW, recurrent=False) for p in prompts], 2, False),
        ("stops after 16 tokens", [_decode_toy(E, A, U, p, FID_EARLY) for p in prompts], 2, True),
    ]
    rows = []
    for name, cand, bpw, rec in variants:
        n_w = FID_H + (FID_H * FID_H if rec else 0) + FID_V * FID_H
        flops = (2 * FID_H * FID_H if rec else 0) + FID_H + 2 * FID_V * FID_H
        agree, compared, first, passed = _agreement(cand, ref)
        rows.append((name, _n(n_w * bpw), _n(flops), f"{agree} / {compared}",
                     "-" if first is None else f"probe {first[0]}, token {first[1]}", "pass" if passed else "fail",
                     f"{_prefix_score(cand, ref):.3f}"))
    listing = "; ".join(f"probe {k}: prompt {prompts[k]}, reference {ref[k]}" for k in range(FID_PROBES))
    return ("A toy greedy decoder with random weights (a model, not a measurement), the same work in five forms, each checked against the reference output. Vocabulary 16, "
            "width 8: E [16, 8], A [8, 8], U [16, 8], every entry (integer(512) - 256) / 256, drawn E, then A, then U, "
            "each in row-major order, then the 3 prompts of 3 tokens, each integer(16). "
            + _seed_line("benchmarking.fidelity")
            + " The state h starts at zero; each token x (the prompt's, then each generated one) sets h = clamp(A h + "
            "E[x], -1, 1), the sum over j taken in ascending order and E[x] added last; after the prompt and after each "
            "generated token the next token is argmax(U h) (sum over j ascending, ties to the lowest id). 24 tokens per "
            "probe, arithmetic in double precision, one operation at a time, no fused multiply-add. The 8-bit form "
            "rounds every weight to a multiple of 1/64, halves away from zero. Weights read per generated token: one "
            "row of E (8), A (64) and U (128); stored in 2 bytes each (they are multiples of 1/256 in [-1, 1)) or 1 byte "
            "in the 8-bit form. FLOPs per generated token, a multiply-add counted as 2 and the clamp's comparisons not "
            "counted: 128 for A h, 8 for adding E[x], 256 for U h. Agreement counts positions over the longer of each "
            "pair of streams; the gate passes only on exact agreement with at least one position compared. 'Common "
            "prefix' scores only the positions both streams have.\n\n"
            + _table(["form", "bytes of weights per token", "FLOPs per token", "tokens agreeing / positions compared",
                      "first divergence", "exact gate", "common-prefix score"], rows, "lrrrlll")
            + "\n\nThe reference streams: " + listing + ".")


# ---------------------------------------------------------------- peaks on paper and on the device

DEV_X = dict(name="X", lanes=10240, clock_ghz=1.953125, bus_bits=256, pin_gbps=25.0,
             f_meas=pr.F_MS, bw_meas=pr.BW_MS, i8_ratio=4, i8_meas=128000.0, cache_meas=2400.0)
DEV_Y = dict(name="Y", lanes=6144, clock_ghz=1.953125, bus_bits=384, pin_gbps=20.0,
             f_meas=22800.0, bw_meas=590.0, i8_ratio=4, i8_meas=None, cache_meas=None)


def _nominal(dev):
    f = dev["lanes"] * 2 * dev["clock_ghz"]                # GFLOP/s: lanes x 2 FLOPs per FMA x the boost clock
    bw = dev["bus_bits"] / 8.0 * dev["pin_gbps"]           # GB/s: bus bytes x the per-pin data rate
    return f, bw, f * dev["i8_ratio"]


GEMM = 2 * 4096 ** 3   # a 4096 x 4096 x 4096 product: 2 m n k operations


def _peak_kernels():
    """(kernel, device, work label, work, unit, model time in microseconds, class, wrong-roof label or None)."""
    return [
        ("decode step, weights streamed", DEV_X, "bytes", 3.0e9, 6000.0, "bandwidth", None),
        ("decode step, weights streamed", DEV_Y, "bytes", 3.0e9, 3.0e9 / 540.0e9 * 1e6, "bandwidth", None),
        ("fp32 matrix product 4096^3", DEV_X, "FLOPs", GEMM, GEMM / 22000.0e9 * 1e6, "fp32", None),
        ("int8 matrix product 4096^3", DEV_X, "integer operations", GEMM, GEMM / 12000.0e9 * 1e6, "int8", "fp32"),
        ("reduction over a 4 MiB working set, 600 passes", DEV_X, "bytes", 4 * 2 ** 20 * 600,
         4 * 2 ** 20 * 600 / 1500.0e9 * 1e6, "cache", "dram"),
    ]


def peak_denominators():
    drows = []
    for d in (DEV_X, DEV_Y):
        f, bw, i8 = _nominal(d)
        drows.append((d["name"], _n(d["lanes"]), f"{d['clock_ghz']}", _n(round(f)), _n(round(d["f_meas"])),
                      f"{d['f_meas'] / f:.3f}", f"{d['bus_bits']} x {d['pin_gbps']:g}", _n(round(bw)),
                      _n(round(d["bw_meas"])), f"{d['bw_meas'] / bw:.3f}"))
    krows = []
    for name, d, wl, work, t_us, cls, wrong in _peak_kernels():
        f, bw, i8 = _nominal(d)
        rate = work / (t_us * 1e-6) / 1e9                      # G per second of the work's own unit
        nominal = {"bandwidth": bw, "fp32": f, "int8": i8, "cache": None}[cls]
        measured = {"bandwidth": d["bw_meas"], "fp32": d["f_meas"], "int8": d["i8_meas"], "cache": d["cache_meas"]}[cls]
        wrong_val = {"fp32": d["f_meas"], "dram": d["bw_meas"]}.get(wrong) if wrong else None
        krows.append((name, d["name"], f"{work:.3e} {wl}", _f(t_us), _n(round(rate)),
                      "-" if nominal is None else f"{rate / nominal:.3f}", f"{rate / measured:.3f}",
                      "-" if wrong is None else f"{rate / wrong_val:.3f} (measured {'fp32' if wrong == 'fp32' else 'main-memory'} roof)"))
    return ("Two model devices, X (the model GPU unit of Volume IV) and Y, none of it a measurement. The nominal "
            "arithmetic peak is lanes x 2 FLOPs per fused multiply-add x the boost clock in GHz; the nominal bandwidth is "
            "the bus width in bytes x the data rate per pin in Gb/s; the nominal 8-bit integer rate is 4 times the "
            "nominal fp32 rate (four 8-bit products per lane and cycle). The measured rates are the probes of Volume IV, "
            "Chapter 10: X's are that chapter's calibration, Y's are parameters of this example. X also has a measured "
            "8-bit rate of 128,000 (GOP/s) and a measured cache-resident bandwidth of 2,400 (GB/s); nothing nominal is "
            "published for the cache.\n\n"
            + _table(["device", "lanes", "boost clock (GHz)", "nominal fp32 (GFLOP/s)", "measured fp32", "measured / nominal",
                      "bus bits x Gb/s per pin", "nominal bandwidth (GB/s)", "measured bandwidth", "measured / nominal"],
                     drows, "lrrrrrlrrr")
            + "\n\nFive kernels, their counted work and model times. The rate is the work over the time, in G per second "
            "of the work's own unit (bytes, FLOPs or integer operations). Each fraction divides by the roof of the "
            "kernel's own class (main-memory bandwidth, fp32 arithmetic, 8-bit integer arithmetic, cache bandwidth); "
            "the last column divides by the roof that a report with only fp32 and main-memory roofs would use.\n\n"
            + _table(["kernel", "device", "counted work", "time (microseconds)", "rate (G per second)",
                      "fraction of nominal peak", "fraction of measured roof", "against a roof of the wrong class"],
                     krows, "llrrrrrl"))


# ---------------------------------------------------------------- order, heat and drift

DRIFT_B = (1, 2, 4, 8)
DRIFT_TRUE = (1.000, 1.020, 1.035, 1.044)      # the steady-state rate of each batch size, relative to B = 1
DRIFT_HOT = 0.72                                # the clock factor a device settles at under continuous load
DRIFT_TAU = 150.0                               # model seconds of continuous load per e-fold of the clock's fall
DRIFT_LEVEL = 60.0                              # model seconds each level is measured for
DRIFT_NOISE = 0.005                             # each measurement times (1 + 0.005 normal())
DRIFT_PREHEAT = 900.0                           # model seconds of load before the pre-heated ladder


def _clock_mean(t0, t1):
    """The mean of m(t) = hot + (1 - hot) exp(-t / tau) over [t0, t1]."""
    a = DRIFT_HOT
    return a + (1.0 - a) * DRIFT_TAU / (t1 - t0) * (math.exp(-t0 / DRIFT_TAU) - math.exp(-t1 / DRIFT_TAU))


def _drift_orders():
    gen = bookrng.generator("benchmarking.order")
    out = {}
    for label in ("ascending", "descending", "rest", "preheated"):
        seq = list(range(4)) if label != "descending" else [3, 2, 1, 0]
        vals = [None] * 4
        t = DRIFT_PREHEAT if label == "preheated" else 0.0
        for k in seq:
            t0 = 0.0 if label == "rest" else t
            v = DRIFT_TRUE[k] * _clock_mean(t0, t0 + DRIFT_LEVEL) * (1.0 + DRIFT_NOISE * gen.normal())
            vals[k] = v
            t = t0 + DRIFT_LEVEL
        out[label] = [v / vals[0] for v in vals]
    return out


def order_drift():
    o = _drift_orders()
    rows = []
    for k, b in enumerate(DRIFT_B):
        rows.append((b, f"{DRIFT_TRUE[k]:.3f}", f"{o['ascending'][k]:.3f}", f"{o['descending'][k]:.3f}",
                     f"{o['rest'][k]:.3f}", f"{o['preheated'][k]:.3f}"))
    return ("A batch-size ladder on a device that slows as it heats, a model and not a measurement. The steady rate of "
            "batch size B relative to B = 1 is 1.000, 1.020, 1.035 and 1.044 for B = 1, 2, 4, 8. Under continuous load "
            "the clock factor is m(t) = 0.72 + 0.28 exp(-t / 150), t the model seconds of load since the device was "
            "cold, exp the library function; a level is measured for 60 model seconds and reads its steady rate times "
            "the mean of m over its interval, 0.72 + 0.28 (150 / 60) (exp(-t0 / 150) - exp(-t1 / 150)), times "
            "(1 + 0.005 z) with z = rng.normal(). " + _seed_line("benchmarking.order")
            + " Draw order: the columns from left to right, and within a column the levels in the order the column "
            "measures them, one normal each. 'Ascending' runs B = 1, 2, 4, 8 back to back from cold; 'descending' runs "
            "8, 4, 2, 1; 'rest' lets the device cool fully before each level; 'pre-heated' runs 900 model seconds of load "
            "first and then the ascending ladder. Each column is divided by its own B = 1 reading.\n\n"
            + _table(["batch size B", "steady rate (truth)", "ascending from cold", "descending from cold",
                      "cooled before each level", "pre-heated, ascending"], rows, "rrrrrr"))


# ---------------------------------------------------------------- one request stream is one sample

STREAMS = 8
STREAM_CONC = 8
HOST_B = 0.3     # variant B: 0.3 model milliseconds of host work per iteration in place of 0.4


def _stream_runs():
    rows = []
    for k in range(STREAMS):
        name = f"benchmarking.streams.{k}"
        reqs = _stream(name)
        a = _summary(_serve(reqs, "closed", STREAM_CONC))["tok_s"]
        b = _summary(_serve(reqs, "closed", STREAM_CONC, host=HOST_B))["tok_s"]
        rows.append((k, bookrng.seed_of(name), a, b))
    return rows


def stream_seeds():
    runs = _stream_runs()
    rows = []
    for i, (k, seed, a, b) in enumerate(runs):
        nb = runs[(i + 1) % STREAMS][3]
        rows.append((k, f"{seed:#018x}", _f(a), _f(b), f"{b / a - 1:+.4f}", f"{nb / a - 1:+.4f}"))
    paired = [b / a - 1 for _, _, a, b in runs]
    unpaired = [runs[(i + 1) % STREAMS][3] / runs[i][2] - 1 for i in range(STREAMS)]
    a_vals = [a for _, _, a, _ in runs]
    return ("Variant A is the engine of `benchmarking.opening_claims`; variant B is the same engine with 0.3 model "
            "milliseconds of host work per iteration in place of 0.4, nothing else changed. Both serve 8 in flight, "
            "closed loop. Stream k (k = 0 .. 7) is 68 requests drawn as before from its own generator, seeded with "
            "`seed_of` of the name `benchmarking.streams.k`, the seed printed in each row. 'Paired' compares B and A on "
            "the same stream; 'unpaired' compares B on stream k + 1 (stream 0 after stream 7) with A on stream k. Output "
            "rates in tokens per second of the model engine, whose rates the author chose (no number here is a measurement).\n\n"
            + _table(["stream k", "seed", "A: output tokens per second", "B: output tokens per second",
                      "paired B / A - 1", "unpaired B(k + 1) / A(k) - 1"], rows, "rlrrrr")
            + f"\n\nPaired: mean {_mean(paired):+.4f}, sample standard deviation {_sd(paired):.4f}. Unpaired: mean "
            f"{_mean(unpaired):+.4f}, sample standard deviation {_sd(unpaired):.4f}. Spread of A over the 8 streams, "
            f"(largest - smallest) / largest: {(max(a_vals) - min(a_vals)) / max(a_vals):.4f}. (Divisor n - 1 = 7.)")


# ---------------------------------------------------------------- a floor that does not flap

GATE_FLOORS = (0.95, 0.90, 0.85)
GATE_REGRESSIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
GATE_N = 3
GATE_TRIALS = 2000
GATE_SHARED = 0.15     # on the shared machine a run is also slowed by (1 + 0.15 u3): another tenant's share of it


def gate_floors():
    gen = bookrng.generator("benchmarking.gate")
    rows = []
    for d in GATE_REGRESSIONS:
        fails = {"quiet": [0] * len(GATE_FLOORS), "shared": [0] * len(GATE_FLOORS)}
        for _ in range(GATE_TRIALS):
            runs = []
            for _ in range(2 * GATE_N):
                f = pr._factor(gen)
                runs.append((f, f * (1.0 + GATE_SHARED * gen.uniform())))
            for kind, col in (("quiet", 0), ("shared", 1)):
                base = max(1.0 / r[col] for r in runs[:GATE_N])
                cand = max((1.0 - d) / r[col] for r in runs[GATE_N:])
                for j, fl in enumerate(GATE_FLOORS):
                    if cand < fl * base:
                        fails[kind][j] += 1
        rows.append((f"{d:.2f}",) + tuple(f"{x / GATE_TRIALS:.4f}" for x in fails["quiet"] + fails["shared"]))
    return ("A throughput gate: the candidate passes when its best of 3 runs is at least the floor times the baseline's "
            "best of 3. On the quiet machine a run's throughput is its true rate over the noise factor of Volume IV, "
            "Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1), which only ever slows a run; on the "
            "shared machine the same factor is multiplied by (1 + 0.15 u3), the share of the run another tenant took. "
            "The baseline's true rate is 1 and the candidate's is 1 minus the regression; the noise rates are chosen by the author and this is a simulation, not a measurement. "
            + _seed_line("benchmarking.gate")
            + " Draw order: for each regression in the order of the table, 2,000 trials; a trial draws the baseline's 3 "
            "runs and then the candidate's 3, four uniforms per run (u0, u1, u2, u3); both machines and every floor are "
            "judged on the same draws (the quiet machine ignores u3). Each cell is the share of the 2,000 trials in "
            "which the gate failed.\n\n"
            + _table(["true regression", "quiet: floor 0.95", "quiet: 0.90", "quiet: 0.85", "shared: floor 0.95",
                      "shared: 0.90", "shared: 0.85"], rows, "rrrrrrr"))


# ---------------------------------------------------------------- a task that defeats the shortcut

MQ_HALF = 8                   # keys are content ids 0 .. 7, values 8 .. 15 (printed here without the two markers)
MQ_TRAIN, MQ_VAL = 6000, 200


def _mq_corpus(gen, n, fixed=None):
    """n sequences of 2 bindings and 2 queries: keys (k1, k2) distinct, values drawn per binding (or fixed[k])."""
    out = []
    for _ in range(n):
        i1 = gen.integer(MQ_HALF)
        j = gen.integer(MQ_HALF - 1)
        i2 = j if j < i1 else j + 1
        keys = (i1, i2)
        if fixed is None:
            vals = (MQ_HALF + gen.integer(MQ_HALF), MQ_HALF + gen.integer(MQ_HALF))
        else:
            vals = (fixed[i1], fixed[i2])
        first = gen.integer(2)
        order = (first, 1 - first)
        out.append((keys, vals, [(keys[q], vals[q]) for q in order]))
    return out


def _mq_scores(train, val):
    counts = {}
    for _, _, qa in train:
        for q, a in qa:
            counts.setdefault(q, [0] * (2 * MQ_HALF))[a] += 1
    overall = [0] * (2 * MQ_HALF)
    for _, _, qa in train:
        for _, a in qa:
            overall[a] += 1
    uni = _argmax(overall)
    no_ctx = hit_uni = hit_ctx = total = 0
    for keys, vals, qa in val:
        for q, a in qa:
            total += 1
            guess = _argmax(counts[q]) if q in counts else uni
            no_ctx += guess == a
            hit_uni += uni == a
            hit_ctx += vals[keys.index(q)] == a
    return hit_uni / total, no_ctx / total, hit_ctx / total, total


def recall_shortcut():
    gen = bookrng.generator("benchmarking.recall")
    tr = _mq_corpus(gen, MQ_TRAIN)
    va = _mq_corpus(gen, MQ_VAL)
    fixed = {k: MQ_HALF + gen.integer(MQ_HALF) for k in range(MQ_HALF)}
    tr2 = _mq_corpus(gen, MQ_TRAIN, fixed)
    va2 = _mq_corpus(gen, MQ_VAL, fixed)
    rows = []
    for label, t, v in (("values drawn afresh in every sequence", tr, va),
                        ("each key always bound to the same value", tr2, va2)):
        u, nc, ctx, total = _mq_scores(t, v)
        rows.append((label, total, f"{1 / MQ_HALF:.3f}", f"{u:.3f}", f"{nc:.3f}", f"{ctx:.3f}"))
    return ("An associative-recall task in two designs and three predictors that are not trained networks, on synthetic sequences (a model, not a measurement). A sequence "
            "binds two distinct keys to values and then queries both keys, k1 v1 k2 v2 SEP q1 a1 q2 a2; keys are ids "
            "0 .. 7 and values ids 8 .. 15. " + _seed_line("benchmarking.recall")
            + " Draw order: 6,000 training then 200 held-out sequences of the first design, then the fixed table "
            "(one integer(8) per key 0 .. 7), then 6,000 and 200 sequences of the second design. A sequence draws its "
            "first key integer(8), its second key from integer(7) skipping the first (an index j at or above the first "
            "key becomes j + 1), its two values integer(8) each in the first design (offset by 8; the second design "
            "reads the table and draws none), and which key is queried first, integer(2). 'Most common value' always "
            "answers the value most frequent at the answer positions of training; 'value seen most often after this "
            "key' answers, for a queried key, the value that followed it most often at the answer positions of "
            "training (ties to the lowest id); 'look the key up in this sequence' copies the value bound to the key "
            "earlier in the same sequence. Accuracy is over the 400 answer positions of the held-out sequences.\n\n"
            + _table(["design", "answer positions", "chance, 1 / 8", "most common value",
                      "value seen most often after this key", "look the key up in this sequence"], rows, "lrrrrr"))
