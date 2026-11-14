# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Batching and Scheduling Requests".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: batching.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's figures, so a
picture and the table beside it cannot disagree.

Every number is integer arithmetic or an exact fraction (the cost of a forward pass divides its FLOPs by the ridge point), and
every schedule is a deterministic simulation whose rules the chapter states, so another implementation reproduces every
printed digit. No generator is used.

The decoder is the 7.5-billion-parameter one of Volume V, Chapter 12, and its FLOP and byte counts are imported from that
chapter's module (`kv_cache`), so the two chapters cannot count the same step differently.

Swedish Embedded AB implements solutions for serving language models to many users at once on the hardware they already
have. If your team needs expertise in continuous batching, request scheduling and admission control then you can procure
our services by sending an email to info@swedishembedded.com.
"""
from fractions import Fraction

import kv_cache as kc
import roofline

_table = kc._table
_int = kc._int

MODEL = kc.DECODER_8B
WEIGHT_BYTES = kc.weight_params(MODEL) * kc.BYTES          # B_w: every weight once, bf16
KV_BYTES = kc.cache_bytes_per_token(MODEL)                  # B_kv: keys and values of one token in every layer
HEAD_FLOPS = 2 * MODEL["D"] * MODEL["V"]                    # the output head applied to one row
_PEAK, _BW = roofline.MACHINES["GPU"]
RIDGE = int(_PEAK) // int(_BW)                              # 50 FLOP per byte, the illustrative GPU of Volume IV
assert RIDGE * int(_BW) == int(_PEAK)
GIGA, TERA = 10 ** 9, 10 ** 12


def _round(x, d):
    """x (an int or Fraction) rounded half away from zero to d decimals, with thousands separators."""
    q = Fraction(x) * 10 ** d
    n = int(abs(q) + Fraction(1, 2))
    n = -n if q < 0 else n
    whole, frac = divmod(abs(n), 10 ** d)
    sign = "-" if n < 0 else ""
    return f"{sign}{whole:,}" + (f".{frac:0{d}d}" if d else "")


# ------------------------------------------------------------------------------------------------ the cost of one forward

def forward_work(ctx, slices):
    """(FLOPs, bytes) of ONE forward pass that carries a decode row for every entry of `ctx` (the number of positions
    already in that sequence's cache) and a prompt slice for every (start, rows, last) of `slices`.

    FLOPs: a decode row is kv_cache.flops_new(c, 1) (its weights, its attention over c + 1 keys, the head); a slice is
    kv_cache.flops_new(start, rows) without the head unless it is the last slice of its prompt.
    Bytes: the weights once for the whole forward, every decode row's c cached positions read, every slice's `start`
    earlier positions read and its `rows` new positions written (a decode row's own new position is not counted).
    """
    if not ctx and not slices:
        return 0, 0
    flops, nbytes = 0, WEIGHT_BYTES
    for c in ctx:
        flops += kc.flops_new(MODEL, c, 1)
        nbytes += KV_BYTES * c
    for s, n, last in slices:
        flops += kc.flops_new(MODEL, s, n) - (0 if last else HEAD_FLOPS)
        nbytes += KV_BYTES * (s + n)
    return flops, nbytes


def forward_cost50(ctx, slices):
    """50 B_eq = max(50 bytes, FLOPs) of one forward, an integer (0 for a forward that carries nothing): the cost in
    equivalent bytes times the ridge, so that every sum of costs stays exact."""
    flops, nbytes = forward_work(ctx, slices)
    return max(RIDGE * nbytes, flops)


def forward_cost(ctx, slices):
    """B_eq of one forward as an exact Fraction."""
    return Fraction(forward_cost50(ctx, slices), RIDGE)


def free_rows(batch, ctx):
    """The most prompt rows (one slice from position 0, not the prompt's last) that a decode step of `batch` sequences at
    context `ctx` can carry while its FLOPs stay at or below 50 times its bytes."""
    n = 0
    while True:
        flops, nbytes = forward_work([ctx] * batch, [(0, n + 1, False)])
        if flops > RIDGE * nbytes:
            return n
        n += 1


def cost_curve(batch, ctx, rows, separate=False):
    """[(n, B_eq in 10^9 equivalent bytes)] of a decode step of `batch` rows at `ctx` with an n-row slice from position 0,
    carried in the same forward, or (`separate`) run as a forward of its own; floats, for the figure."""
    out = []
    for n in rows:
        sl = [(0, n, False)] if n else []
        c50 = forward_cost50([ctx] * batch, []) + forward_cost50([], sl) if separate else forward_cost50([ctx] * batch, sl)
        out.append((n, c50 / RIDGE / GIGA))
    return out


# ------------------------------------------------------------------------------------------------ static against continuous

SLOTS = 4
LENGTHS = (40, 6, 9, 30, 5, 12, 8, 25, 7, 10, 6, 20)        # tokens each request produces, in arrival order


def static_slots(lengths, slots):
    """[(slot, first, last)] of each request when the first `slots` waiting requests form a batch only once the previous
    batch has finished, and the batch runs until its longest member is done."""
    out, t = [], 1
    for b in range(0, len(lengths), slots):
        batch = lengths[b:b + slots]
        for j, n in enumerate(batch):
            out.append((j, t, t + n - 1))
        t += max(batch)
    return out


def continuous_slots(lengths, slots):
    """[(slot, first, last)] of each request when at the start of every iteration each free slot takes the oldest waiting
    request, the lowest-numbered free slot first: a request admitted at iteration a produces a token at each of a .. a + n - 1
    and its slot is free again from a + n."""
    free_at = [1] * slots
    out = []
    for n in lengths:
        j = min(range(slots), key=lambda s: (free_at[s], s))
        out.append((j, free_at[j], free_at[j] + n - 1))
        free_at[j] += n
    return out


def static_vs_continuous():
    st, co = static_slots(LENGTHS, SLOTS), continuous_slots(LENGTHS, SLOTS)
    rows = [(i, n, st[i][1], st[i][2], co[i][1], co[i][2]) for i, n in enumerate(LENGTHS)]
    useful = sum(LENGTHS)
    summary = []
    for name, sched in (("static", st), ("continuous", co)):
        end = max(last for _, _, last in sched)
        wait = Fraction(sum(first - 1 for _, first, _ in sched), len(sched))
        summary.append((name, end, SLOTS * end, useful, _round(Fraction(100 * useful, SLOTS * end), 2), _round(wait, 2)))
    return (f"Twelve requests arrive together, numbered 0 to 11 in arrival order, and are served on {SLOTS} slots (a slot is a row of "
            f"the batched decode step). Request i produces the number of tokens in the column 'tokens', one in every iteration it holds "
            f"a slot; its prompt and first token are folded into its first iteration. Iterations are numbered from 1. Static: the first "
            f"{SLOTS} waiting requests form a batch, which runs until its longest member is done; only then is the next batch formed. "
            f"Continuous: at the start of every iteration each free slot takes the oldest waiting request, the lowest-numbered free "
            f"slot first; a request that produces its last token in iteration t frees its slot for iteration t + 1.\n\n"
            + _table(["request", "tokens", "static: first", "static: last", "continuous: first", "continuous: last"], rows)
            + "\n\nSlot-iterations are slots times iterations; the useful ones are the tokens produced. The mean wait is the mean, over "
            "the 12 requests, of the first iteration minus 1; both percentages and means are rounded half away from zero to two "
            "decimals.\n\n"
            + _table(["policy", "iterations", "slot-iterations", "useful", "useful (%)", "mean wait (iterations)"], summary, "lrrrrr"))


# ------------------------------------------------------------------------------------------------ the cost of a step

def step_costs():
    rows = []

    def row(label, ctx, sl):
        f, b = forward_work(ctx, sl)
        rows.append((label, _int(f), _int(b), _round(Fraction(f, b), 2), _round(forward_cost(ctx, sl) / GIGA, 3)))

    for b in (1, 16, 64):
        row(f"decode, {b} sequence{'s' if b > 1 else ''} at context 2,048", [2048] * b, [])
    row("prefill, 2,048 tokens in one pass", [], [(0, 2048, True)])
    row("prefill slice, tokens 0 to 255", [], [(0, 256, False)])
    row("prefill slice, tokens 1,792 to 2,047 (the last)", [], [(1792, 256, True)])
    row("decode of 16 at 2,048 with the slice 0 to 255", [2048] * 16, [(0, 256, False)])
    knee = [(b, free_rows(b, 2048)) for b in (1, 4, 16, 64)]
    body = kc.weight_params(MODEL) - MODEL["V"] * MODEL["D"]
    return (f"The 7.5-billion decoder of Volume V, Chapter 12 ({_int(WEIGHT_BYTES)} bytes of bf16 weights, {_int(KV_BYTES)} bytes of "
            f"keys and values per token) on the illustrative GPU of Volume IV, whose ridge is {RIDGE} FLOP per byte. FLOPs are counted "
            f"as in that chapter (a multiply-add is 2, only matrix products count): a decode row at context c costs F(c, 1), the head "
            f"included; a prompt slice of n rows starting at position s costs F(s, n), without the head unless it is the last slice of "
            f"its prompt. Bytes: the weights once per forward, the c cached positions every decode row reads, the s earlier positions "
            f"a slice reads and the n it writes. The cost of a forward is B_eq = max(bytes, FLOPs / {RIDGE}) in equivalent bytes "
            f"(Volume IV, Chapter 8), given in units of 10^9 and rounded half away from zero to three decimals; FLOP per byte to two.\n\n"
            + _table(["forward", "FLOPs", "bytes", "FLOP per byte", "B_eq (10^9)"], rows, "lrrrr")
            + "\n\nThe most prompt rows (a slice from position 0, not the prompt's last) that a decode step at context 2,048 can carry "
            "while its FLOPs stay at or below 50 times its bytes:\n\n"
            + _table(["sequences decoding", "free prompt rows"], knee)
            + f"\n\nPast that point each further prompt row adds {_round(Fraction(2 * body, RIDGE) / GIGA, 3)} x 10^9 equivalent bytes "
            f"for its weights (2 x {_int(body)} FLOPs over 50), plus its attention.")


# ------------------------------------------------------------------------------------------------ a burst of long prompts

BURST_PROMPT = 2048          # tokens in each new prompt
BURST_TOKENS = 32            # tokens each new request produces
BURST_N = 8                  # new requests, arriving together
LONG_PROMPT = 512            # the sequence already decoding: its prompt
CHUNK = 256                  # prompt rows per iteration under chunking
POLICIES = ("whole prompts, no budget", "one prompt an iteration", "chunks of 256, own pass", "chunks of 256, mixed pass")


def run_burst(policy, chunk=CHUNK):
    """(cost50 of every iteration, first-token iteration and last iteration of every request (entries 1 .. 8; entry 0 is
    unused), the iterations in which request 0 got a token), with prompt slices of `chunk` rows under the two chunked policies. Request 0 is the sequence already decoding; a sequence is [id, cached, made, wants]."""
    running = [[0, LONG_PROMPT, 1, 10 ** 9]]
    waiting = list(range(1, BURST_N + 1))
    pre, pre_done = -1, 0                 # the prompt being read in under chunking (-1: none) and its rows already in
    costs, first, last, long_iters = [], [0] * (BURST_N + 1), [0] * (BURST_N + 1), []
    t = 0
    while waiting or pre >= 0 or any(r[0] != 0 for r in running):
        t += 1
        cost, started, sl = 0, [], []
        if policy == POLICIES[0]:
            for i in waiting:
                cost += forward_cost50([], [(0, BURST_PROMPT, True)])
                started.append(i)
            waiting = []
        elif policy == POLICIES[1]:
            if waiting:
                cost += forward_cost50([], [(0, BURST_PROMPT, True)])
                started.append(waiting[0])
                waiting = waiting[1:]
        else:
            if pre < 0 and waiting:
                pre, pre_done = waiting[0], 0
                waiting = waiting[1:]
            if pre >= 0:
                n = min(chunk, BURST_PROMPT - pre_done)
                sl = [(pre_done, n, pre_done + n == BURST_PROMPT)]
                pre_done += n
                if pre_done == BURST_PROMPT:
                    started.append(pre)
                    pre = -1
            if policy == POLICIES[2]:
                cost += forward_cost50([], sl)
                sl = []
        if policy != POLICIES[3]:
            for i in started:
                running.append([i, BURST_PROMPT, 1, BURST_TOKENS])
                first[i] = t
        act = [j for j in range(len(running)) if running[j][2] < running[j][3]]
        cost += forward_cost50([running[j][1] for j in act], sl)
        for j in act:
            running[j][1] += 1
            running[j][2] += 1
            if running[j][0] == 0:
                long_iters.append(t)
            elif running[j][2] == running[j][3]:
                last[running[j][0]] = t
        if policy == POLICIES[3]:
            for i in started:
                running.append([i, BURST_PROMPT, 1, BURST_TOKENS])
                first[i] = t
        running = [r for r in running if r[2] < r[3]]
        costs.append(cost)
    return costs, first, last, long_iters


def burst_timeline(policy):
    """[(iteration, cumulative cost in 10^12 equivalent bytes)] for the figure, starting at (0, 0)."""
    costs = run_burst(policy)[0]
    out, acc = [(0, 0.0)], 0
    for k, c in enumerate(costs, 1):
        acc += c
        out.append((k, acc / RIDGE / TERA))
    return out


def burst():
    rows = []
    for p in POLICIES:
        costs, first, last, long_iters = run_burst(p)
        cum = [0]
        for c in costs:
            cum.append(cum[-1] + c)
        firsts = [cum[k] for k in first[1:]]
        to_t = lambda c50: Fraction(c50, RIDGE * TERA)
        worst = max(costs[k - 1] for k in long_iters)
        rows.append((p, len(costs), _round(to_t(firsts[0]), 3), _round(to_t(sum(firsts)) / BURST_N, 3), _round(to_t(firsts[-1]), 3),
                     _round(to_t(max(cum[k] for k in last[1:])), 3), _round(Fraction(worst, RIDGE * GIGA), 1), len(long_iters)))
    return (f"Request 0 is decoding (a prompt of {LONG_PROMPT} tokens, {LONG_PROMPT} positions cached, one token produced) when "
            f"{BURST_N} requests, 1 to {BURST_N}, arrive together, each with a prompt of {_int(BURST_PROMPT)} tokens and wanting "
            f"{BURST_TOKENS} tokens (the first from its prompt's last pass, the others from decode steps). The batch and the cache "
            f"have room for all of them. An iteration does the prompt work of the policy and then one decode row for every sequence "
            f"that still wants tokens, a new sequence included from the iteration its prompt finished, except under the mixed pass, "
            f"where it joins in the next one. Whole prompts, no budget: every waiting prompt is prefilled whole, each in a forward of "
            f"its own, in the first iteration. One prompt an iteration: the oldest waiting prompt is prefilled whole. Chunks of 256, "
            f"own pass: 256 rows of the one prompt being read in, oldest first, in a forward of their own before the decode forward; "
            f"the next prompt starts in the iteration after the previous one is in. Chunks of 256, mixed pass: the same rows carried "
            f"in the decode forward. Every forward costs B_eq of the cost table; an iteration costs the sum of its forwards, and a "
            f"token exists at the cumulative cost at the end of the iteration that computes it. The run ends when requests 1 to "
            f"{BURST_N} have finished. The first new token is the earliest cumulative cost at which any of requests 1 to {BURST_N} has "
            f"its first token; the longest gap of request 0 is the largest difference between the cumulative costs at the ends of "
            f"two consecutive iterations in which it decodes, the first measured from cost 0 because its first token exists at "
            f"time 0; its tokens are the number of those iterations. Costs in 10^12 and in 10^9 equivalent bytes, rounded half away from zero.\n\n"
            + _table(["policy", "iterations", "first new token (10^12)", "mean first token (10^12)", "last first token (10^12)",
                      "all new done (10^12)", "longest gap of request 0 (10^9)", "tokens of request 0"], rows, "lrrrrrrr"))


# ------------------------------------------------------------------------------------------------ admission against the pool

POOL_BLOCKS = 48
BLOCK = 16                   # tokens per block
MAX_RUN = 8
POOL_PROMPT = 96
MAX_NEW = 400
POOL_LENGTHS = (40, 380, 70, 20, 250, 120, 60, 300)      # tokens each request produces before it stops
POOL_POLICIES = ("reserve the ceiling", "prompt only, recompute", "prompt only, no preemption")


def blocks(tokens):
    return -(-tokens // BLOCK)


def run_pool(policy, pool=POOL_BLOCKS):
    """(dry, iterations, decode rows, prompt tokens prefilled, preemptions, work50) of the eight requests against a pool of `pool` blocks; a
    dry run (the third policy) returns (True, the iteration at which the pool cannot back a decode step, the sequences then
    decoding, 0, 0, 0). A waiting request is [id, ctx, made],
    a running sequence [id, cached, held, made]."""
    reserve = policy == POOL_POLICIES[0]

    def grow(r):
        return 0 if reserve else blocks(r[1] + 1) - r[2]

    waiting = [[i, POOL_PROMPT, 0] for i in range(len(POOL_LENGTHS))]
    running = []
    free, t, work50, prefilled, preempted, rows = pool, 0, 0, 0, 0, 0
    while waiting or running:
        t += 1
        # 1. Admission, oldest first; a request that does not fit holds the line.
        while waiting and len(running) < MAX_RUN:
            w = waiting[0]
            want = blocks(POOL_PROMPT + MAX_NEW - 1) if reserve else blocks(w[1] + 1) + sum(grow(r) for r in running)
            if free < want:
                break
            waiting = waiting[1:]
            held = blocks(POOL_PROMPT + MAX_NEW - 1) if reserve else blocks(w[1])
            free -= held
            work50 += forward_cost50([], [(0, w[1], True)])
            prefilled += w[1]
            running.append([w[0], w[1], held, max(w[2], 1)])     # a fresh prompt's last pass gives the first token
        # 2. Room for this iteration's decode step.
        act = [j for j in range(len(running)) if running[j][3] < POOL_LENGTHS[running[j][0]]]
        need = sum(grow(running[j]) for j in act)
        if policy == POOL_POLICIES[1]:
            while free < need and len(act) > 1:
                j = act[-1]
                act = act[:-1]
                v = running[j]
                need -= grow(v)
                free += v[2]
                preempted += 1
                running = running[:j] + running[j + 1:]
                waiting = [[v[0], POOL_PROMPT + v[3] - 1, v[3]]] + waiting
        elif policy == POOL_POLICIES[2] and free < need:
            return True, t, len(act), 0, 0, 0
        # 3. One decode step for every sequence that still wants tokens.
        if act:
            work50 += forward_cost50([running[j][1] for j in act], [])
            rows += len(act)
        for j in act:
            g = grow(running[j])
            free -= g
            running[j][2] += g
            running[j][1] += 1
            running[j][3] += 1
        # 4. The finished ones give their blocks back.
        for r in running:
            if r[3] >= POOL_LENGTHS[r[0]]:
                free += r[2]
        running = [r for r in running if r[3] < POOL_LENGTHS[r[0]]]
        assert free >= 0
    return False, t, rows, prefilled, preempted, work50


def _pool_rows(pool):
    """One table row per policy for a pool of `pool` blocks."""
    out = []
    for p in POOL_POLICIES:
        dry, t, rows, prefilled, preempted, work50 = run_pool(p, pool)
        if dry:
            out.append((p, f"no free block at iteration {t}, {rows} sequences decoding", "", "", "", ""))
        else:
            out.append((p, _int(t), _round(Fraction(rows, t), 2), _int(prefilled), preempted, _round(Fraction(work50, RIDGE * TERA), 3)))
    return out


def pool_admission():
    out = _pool_rows(POOL_BLOCKS)
    return (f"A pool of {POOL_BLOCKS} blocks of {BLOCK} tokens ({POOL_BLOCKS * BLOCK} tokens), at most {MAX_RUN} sequences in flight, "
            f"and eight requests, 0 to 7, arriving together, each with a prompt of {POOL_PROMPT} tokens and a ceiling of {MAX_NEW} new "
            f"tokens; they produce {', '.join(str(n) for n in POOL_LENGTHS)} tokens and then stop ({_int(sum(POOL_LENGTHS))} in all). "
            f"A sequence holding c positions holds ceil(c / {BLOCK}) blocks. Every iteration: (1) admit waiting requests oldest first "
            f"while fewer than {MAX_RUN} sequences run and the free blocks cover what the policy asks, stopping at the first that does "
            f"not fit; an admitted request is prefilled whole in a forward of its own, and a fresh one gets its first token from it; "
            f"(2) make room for the decode step; (3) decode one row for every sequence that still wants tokens, each adding one "
            f"position and taking a block when that position starts one; (4) the finished ones give their blocks back. Reserve the "
            f"ceiling: admission asks for the blocks of prompt + ceiling - 1 positions and holds them to the end. Prompt only: "
            f"admission asks for the blocks of the prompt and its first token, plus the block (if any) that each running sequence's "
            f"next position needs, and a sequence grows a block at a time. Recompute: while the free blocks cannot back every "
            f"sequence's next position and more than one sequence decodes, the newest gives back all its blocks and goes to the head "
            f"of the queue, to be prefilled again over its prompt and every token it produced except the last. Work is the sum of "
            f"B_eq over every forward (the rules of the cost table), in 10^12 equivalent bytes, rounded half away from zero to three "
            f"decimals; the mean batch is decode rows over iterations, to two.\n\n"
            + _table(["policy", "iterations", "mean batch", "prompt tokens prefilled", "preemptions", "work (10^12)"], out, "lrrrrr"))


# ------------------------------------------------------------------------------------------------ refusing what cannot be served in time

ARRIVALS = 300
SERVICE = 6                  # units of device work per request
DEADLINE = 60                # units a request may wait to start
GAPS = (12, 6, 4, 3)         # units between arrivals
DEPTH = 4
ADMISSION = ("unbounded queue", f"queue depth below {DEPTH}", "deadline-aware")


def run_overload(gap, policy):
    """(in time, late, refused) for ARRIVALS requests arriving every `gap` units at a device that serves one request at a time,
    each for SERVICE units, first come first served."""
    starts = []
    busy_until = 0
    in_time = late = refused = 0
    for k in range(ARRIVALS):
        t = k * gap
        q = sum(1 for s in starts if s > t)
        wait = max(t, busy_until) - t
        if policy == ADMISSION[0]:
            ok = True
        elif policy == ADMISSION[1]:
            ok = q < DEPTH
        else:
            ok = wait <= DEADLINE
        if not ok:
            refused += 1
            continue
        starts.append(t + wait)
        busy_until = t + wait + SERVICE
        if wait <= DEADLINE:
            in_time += 1
        else:
            late += 1
    return in_time, late, refused


def overload():
    rows = []
    for gap in GAPS:
        for p in ADMISSION:
            a, b, c = run_overload(gap, p)
            rows.append((_round(Fraction(SERVICE, gap), 2), p, a, b, c, _int(b * SERVICE)))
    return (f"{ARRIVALS} requests arrive, request k (from 0) at k x g units of device work, for g = {', '.join(str(g) for g in GAPS)}; "
            f"the device serves one request at a time, first come first served, each for {SERVICE} units, so the offered load "
            f"{SERVICE} / g is 0.5, 1, 1.5 and 2. A request is useful when it starts at most {DEADLINE} units after it arrives. At "
            f"its arrival a policy sees q, the admitted requests that have not started (one whose start equals the arrival has "
            f"started), and the wait the request would have, the work still ahead of it. The unbounded queue admits every request; "
            f"the depth rule admits when q < {DEPTH}; the deadline-aware rule admits when the wait is at most {DEADLINE}. A refused "
            f"request is answered at once and costs no work.\n\n"
            + _table(["offered load", "policy", "in time", "late", "refused", "work on late answers (units)"], rows, "llrrrr"))


# ------------------------------------------------------------------------------------------------ the answers' extra runs

POOL_HEADERS = ["policy", "iterations", "mean batch", "prompt tokens prefilled", "preemptions", "work (10^12)"]


def variations():
    """The runs that the check-your-understanding and exercise answers quote: the burst under the mixed pass with slices of 64
    rows, the twelve requests served shortest first, and the pool of 64 blocks."""
    burst_rows = []
    for chunk in (CHUNK, 64):
        costs, first, last, long_iters = run_burst(POLICIES[3], chunk)
        cum = [0]
        for c in costs:
            cum.append(cum[-1] + c)
        worst = max(costs[k - 1] for k in long_iters)
        burst_rows.append((chunk, len(costs), _round(Fraction(worst, RIDGE * GIGA), 1), _round(Fraction(cum[first[BURST_N]], RIDGE * TERA), 3),
                           _round(Fraction(max(cum[k] for k in last[1:]), RIDGE * TERA), 3)))
    shortest_first = sorted(range(len(LENGTHS)), key=lambda i: (LENGTHS[i], i))
    order_rows = []
    for name, lengths in (("arrival order", LENGTHS), ("shortest first", tuple(LENGTHS[i] for i in shortest_first))):
        sched = continuous_slots(lengths, SLOTS)
        order_rows.append((name, max(last for _, _, last in sched), _round(Fraction(sum(first - 1 for _, first, _ in sched), len(sched)), 2)))
    return ("The burst of `batching.burst` under the mixed pass with the slice of 256 rows and with a slice of 64 rows (the last first token is "
            "the cumulative cost at which the last of requests 1 to 8 has its first token; costs as in that block):\n\n"
            + _table(["slice rows", "iterations", "longest gap of request 0 (10^9)", "last first token (10^12)", "all new done (10^12)"],
                     burst_rows, "rrrrr")
            + "\n\nThe twelve requests of `batching.static_vs_continuous` on 4 slots under the continuous rule, in arrival order and sorted "
            "by tokens produced, shortest first, ties by request number:\n\n"
            + _table(["order", "iterations", "mean wait (iterations)"], order_rows, "lrr")
            + f"\n\nThe pool of `batching.pool_admission` made {64} blocks instead of {POOL_BLOCKS}, with every other rule unchanged:\n\n"
            + _table(POOL_HEADERS, _pool_rows(64), "lrrrrr"))
