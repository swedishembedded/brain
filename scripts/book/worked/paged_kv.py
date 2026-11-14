# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Managing the KV Cache".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: paged_kv.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared with the
chapter's figures so a picture and the table beside it cannot disagree.

The allocator, the prefix cache and the scheduler are integer state machines, so another implementation reproduces every
printed digit. The draws come from the book's SplitMix64 (`bookrng.generator`) with integer draws only, except in the
quantization example, which draws normals and rounds to nearest with ties to even.

Swedish Embedded AB implements solutions for serving many concurrent conversations from the memory of one device for its
clients. If your team needs expertise in KV cache management, paging, prefix reuse and cache quantization then you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import kv_cache as kc
import long_context as lc
import number_formats as nf

_table = nf._table

GIB, MIB = 1 << 30, 1 << 20
S8 = kc.DECODER_8B                         # the 7.5-billion-parameter decoder of Volume V, Chapter 12
BF16_TOKEN = int(lc.cache_bytes_per_token(S8, "bf16"))      # 131,072 bytes per token of context
CTX = 32768                                # the context a reservation must cover
POOL_BYTES = 64 * GIB                      # the device bytes granted to the caches
POOL_TOKENS = POOL_BYTES // BF16_TOKEN     # 524,288 token rows in bf16


def _int(n):
    return f"{n:,}"


def _pct(x, d=1):
    return f"{100.0 * x:.{d}f}"


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _ceil_div(a, b):
    return -(-a // b)


# ------------------------------------------------------------------------------------------------ the conversations

NAME_CONV = "paged_kv.reservation"
N_CONV = 256


def conversation_lengths():
    """Final lengths in tokens of the 256 conversations, in arrival order: for each, k = 1 + integer(15) and then
    n = 2^(k-1) + integer(2^(k-1)), two draws per conversation, k first. Every octave from 1 to 32,767 is equally likely."""
    g = bookrng.generator(NAME_CONV)
    out = []
    for _ in range(N_CONV):
        k = 1 + g.integer(15)
        out.append((1 << (k - 1)) + g.integer(1 << (k - 1)))
    return out


def admit_in_order(lengths, slots_of, capacity):
    """Admit conversations in arrival order while their slots fit; stop at the first that does not.
    Returns (held, tokens stored, slots allocated)."""
    used = held = stored = 0
    for n in lengths:
        s = slots_of(n)
        if used + s > capacity:
            break
        used += s
        held += 1
        stored += n
    return held, stored, used


def paging_curve(sizes):
    """(block size, conversations held, share of allocated slots holding a row) for each block size."""
    ls = conversation_lengths()
    out = []
    for b in sizes:
        held, stored, used = admit_in_order(ls, lambda n: _ceil_div(n, b) * b, POOL_TOKENS)
        out.append((b, held, stored / used))
    return out


def reservation():
    ls = conversation_lengths()
    rows = []
    held, stored, used = admit_in_order(ls, lambda n: CTX, POOL_TOKENS)
    rows.append(("reserve 32,768 tokens each", _int(held), _int(stored), _int(used), _pct(stored / used, 2)))
    for b in (1024, 256, 64, 16, 1):
        held, stored, used = admit_in_order(ls, lambda n, b=b: _ceil_div(n, b) * b, POOL_TOKENS)
        rows.append((f"blocks of {b:,} token{'s' if b > 1 else ''}", _int(held), _int(stored), _int(used), _pct(stored / used, 2)))
    first = ", ".join(_int(n) for n in ls[:8])
    mean = sum(ls) / len(ls)
    return (f"The pool holds 64 GiB = {_int(POOL_BYTES)} bytes, {_int(POOL_TOKENS)} token rows at {_int(BF16_TOKEN)} bytes per "
            "token (the 7.5-billion-parameter decoder of Volume V, Chapter 12, in bf16). The 256 conversations come from the "
            f"generator `paged_kv.reservation` (seed {_seed(NAME_CONV)}): for each in turn, k = 1 + `integer(15)` and then "
            "n = 2^(k - 1) + `integer(2^(k - 1))` tokens, k drawn first, so every octave from 1 to 32,767 tokens is "
            f"equally likely. The first eight are {first}; the shortest is {_int(min(ls))}, the longest {_int(max(ls))}, "
            f"the sum {_int(sum(ls))} and the mean {mean:,.2f} tokens. Conversations are admitted in arrival order at "
            "their final length until the next one does not fit (it and all after it are left out). A slot is a token "
            "row of the pool; a conversation of n tokens takes 32,768 slots under the reservation and ceil(n / b) x b "
            "slots in blocks of b tokens. The last column is the tokens stored over the slots allocated, rounded half "
            "away from zero.\n\n"
            + _table(["policy", "conversations held", "tokens stored", "slots allocated", "% of allocated slots holding a row"],
                     rows))


# ------------------------------------------------------------------------------------------------ the allocator

class Allocator:
    """The chapter's block pool: a stack of free ids (the top is the most recently freed) and a reference count per block."""

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(num_blocks - 1, -1, -1))     # bottom .. top; the top is popped first
        self.ref = [0] * num_blocks

    def alloc(self):
        if not self.free:
            return None
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def incref(self, b):
        self.ref[b] += 1

    def decref(self, b):
        assert self.ref[b] > 0, "double free"
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)


class Table:
    """One sequence's block table: physical block ids in logical order and the number of tokens stored."""

    def __init__(self):
        self.blocks = []
        self.len = 0

    def append(self, a):
        off = self.len % a.block_size
        if off == 0:
            b = a.alloc()
            if b is None:
                raise MemoryError("pool exhausted")
            self.blocks.append(b)
        self.len += 1
        return self.blocks[-1], off

    def reserve(self, n, a):
        start = self.len
        need = _ceil_div(self.len + n, a.block_size)
        while len(self.blocks) < need:
            b = a.alloc()
            if b is None:
                raise MemoryError("pool exhausted")
            self.blocks.append(b)
        self.len += n
        return start

    def release(self, a):
        for b in self.blocks:
            a.decref(b)
        self.blocks, self.len = [], 0

    def fork(self, a):
        t = Table()
        for b in self.blocks:
            a.incref(b)
        t.blocks, t.len = list(self.blocks), self.len
        return t

    def unshare_tail(self, a):
        last = self.blocks[-1]
        if a.ref[last] <= 1:
            return None
        fresh = a.alloc()
        self.blocks[-1] = fresh
        a.decref(last)
        return last, fresh

    def locate(self, tok, bs):
        return self.blocks[tok // bs], tok % bs


def _state(a, tables):
    tabs = "; ".join(f"{n} [{', '.join(str(b) for b in t.blocks)}] len {t.len}" for n, t in tables.items() if t.blocks)
    shared = ", ".join(f"{b}:{r}" for b, r in enumerate(a.ref) if r > 1) or "none"
    return tabs, shared, "[" + ", ".join(str(b) for b in a.free) + "]"


def block_trace_state():
    """Run the trace; returns the list of (event, tables, shared, free) and the final allocator and tables."""
    a = Allocator(8, 4)
    T = {"A": Table(), "B": Table()}
    rows = []

    def log(ev):
        rows.append((ev,) + _state(a, T))

    T["A"].reserve(6, a)
    log("A reserves 6 rows (its prompt)")
    T["B"].reserve(3, a)
    log("B reserves 3 rows")
    blk, off = T["A"].append(a)
    log(f"A appends one row: block {blk}, offset {off}")
    for _ in range(2):
        blk, off = T["B"].append(a)
    log(f"B appends two rows, the second at block {blk}, offset {off}")
    T["D"] = T["A"].fork(a)
    log("D forks A (shares blocks 0 and 1)")
    old, new = T["D"].unshare_tail(a)
    blk, off = T["D"].append(a)
    log(f"D copies its tail block {old} to {new}, then appends at block {blk}, offset {off}")
    T["B"].release(a)
    log("B finishes and releases its blocks")
    T["C"] = Table()
    T["C"].reserve(9, a)
    log("C reserves 9 rows")
    return rows, a, T


def block_trace():
    rows, a, T = block_trace_state()
    blk, off = T["C"].locate(5, 4)
    return ("A pool of 8 blocks of 4 token rows, ids 0 to 7. The free list is a stack printed bottom to top: an allocation "
            "takes the top, a freed block is put on the top. A table is printed as its physical blocks in logical order and "
            "its length in rows; 'shared' lists block:count for every block with more than one reference. Events run in "
            "the order listed.\n\n"
            + _table(["event", "tables after the event", "shared", "free list (bottom to top)"], rows, "llll")
            + f"\n\nRow 5 of C (counted from 0) is at block C[5 // 4] = {blk}, offset 5 mod 4 = {off}.")


# ------------------------------------------------------------------------------------------------ the prefix cache

ROOT = "root"


class PrefixCache:
    """Full blocks keyed by (parent physical block or ROOT, the block's token ids); one reference per indexed block."""

    def __init__(self):
        self.map = {}          # key -> [block, last_use]
        self.by_block = {}     # block -> key
        self.tick = 0

    def lookup(self, prompt, bs, max_tokens, touch=True):
        parent, out = ROOT, []
        for i in range(len(prompt) // bs):
            if (len(out) + 1) * bs > max_tokens:
                break
            key = (parent, tuple(prompt[i * bs:(i + 1) * bs]))
            if key not in self.map:
                break
            e = self.map[key]
            if touch:
                self.tick += 1
                e[1] = self.tick
            parent = e[0]
            out.append(e[0])
        return out

    def insert_chain(self, prompt, blocks, start, a):
        bs = a.block_size
        full = min(len(prompt) // bs, len(blocks))
        parent = ROOT if start == 0 else blocks[start - 1]
        added = []
        for i in range(start, full):
            key = (parent, tuple(prompt[i * bs:(i + 1) * bs]))
            if key in self.map:
                parent = self.map[key][0]
                continue
            b = blocks[i]
            a.incref(b)
            self.tick += 1
            self.map[key] = [b, self.tick]
            self.by_block[b] = key
            added.append(key)
            parent = b
        return added

    def children(self):
        ch = {}
        for parent, _ in self.map:
            ch[parent] = ch.get(parent, 0) + 1
        return ch

    def evict_leaves(self, want, keep, a):
        """Free up to `want` cache-only leaf entries, least recently used first; never a block in `keep`."""
        freed = 0
        while freed < want:
            ch = self.children()
            cands = [(e[1], e[0]) for e in self.map.values() if a.ref[e[0]] == 1 and e[0] not in keep and e[0] not in ch]
            if not cands:
                break
            _, b = min(cands)
            del self.map[self.by_block.pop(b)]
            a.decref(b)
            freed += 1
        return freed

    def evict_plain(self, want, a):
        """Free up to `want` cache-only entries, least recently used first, leaf or not."""
        freed = 0
        while freed < want:
            cands = [(e[1], e[0]) for e in self.map.values() if a.ref[e[0]] == 1]
            if not cands:
                break
            _, b = min(cands)
            del self.map[self.by_block.pop(b)]
            a.decref(b)
            freed += 1
        return freed


def run_prompt(cache, a, prompt):
    """A prompt arrives, adopts the longest cached chain (at most len - 1 tokens), computes the rest, indexes its new full
    blocks and, once it has finished, releases its table. Returns (blocks adopted, rows computed, keys added)."""
    bs = a.block_size
    hits = cache.lookup(prompt, bs, len(prompt) - 1)
    t = Table()
    for b in hits:
        a.incref(b)
    t.blocks, t.len = list(hits), len(hits) * bs
    t.reserve(len(prompt) - t.len, a)
    added = cache.insert_chain(prompt, t.blocks, len(hits), a)
    computed = len(prompt) - len(hits) * bs
    t.release(a)
    return hits, computed, added


SYSTEM = [11, 12, 13, 14, 15, 16, 17, 18]
PROMPTS = (
    ("P1", SYSTEM + [21, 22]),
    ("P2", SYSTEM + [31, 32, 33]),
    ("P3", [15, 16, 17, 18, 11, 12, 13, 14, 41]),
    ("P4", list(SYSTEM)),
)


def _key(k):
    return f"({k[0]}, ({', '.join(str(t) for t in k[1])}))"


def prefix_trace():
    a = Allocator(12, 4)
    cache = PrefixCache()
    rows = []
    for name, p in PROMPTS:
        hits, computed, added = run_prompt(cache, a, p)
        hit_text = ", ".join(str(b) for b in hits) or "none"
        added_text = "; ".join(f"{_key(k)} -> {cache.map[k][0]}" for k in added) or "none"
        rows.append((name, " ".join(str(t) for t in p), _int(len(p)), hit_text, _int(computed), added_text))
    cached = sorted((v[0], k) for k, v in cache.map.items())
    refs = ", ".join(f"{b}:{a.ref[b]}" for b, _ in cached)
    return ("Blocks of 4 tokens in a pool of 12 blocks (ids 0 to 11, the stack free list of the previous example). Each prompt "
            "arrives after the previous one has finished: it adopts the longest cached chain of full blocks covering at most "
            "its length minus one token, computes the rest, indexes its new full blocks under (parent, tokens), with the "
            "parent `root` for a first block, and releases its own references. The cache keeps one reference on every block "
            "it indexes.\n\n"
            + _table(["prompt", "token ids", "tokens", "blocks adopted", "rows computed", "keys added -> block"], rows, "llrlrl")
            + f"\n\nAfter the four prompts the cache holds {len(cached)} blocks, each with this reference count "
            f"(block:count): {refs}.")


def _two_chains():
    """X (4 full blocks, older) and Y (4 full blocks, newer), both finished: 8 cache-only blocks."""
    a = Allocator(16, 4)
    cache = PrefixCache()
    X = list(range(1, 17))
    Y = list(range(101, 117))
    tx = Table()
    tx.reserve(16, a)
    cache.insert_chain(X, tx.blocks, 0, a)
    xb = list(tx.blocks)
    tx.release(a)
    ty = Table()
    ty.reserve(16, a)
    cache.insert_chain(Y, ty.blocks, 0, a)
    ty.release(a)
    return a, cache, X, Y, xb


def eviction_orders():
    rows = []
    for policy in ("least recently used, any entry", "least recently used, leaves only",
                   "leaves only, sparing X's chain"):
        a, cache, X, Y, xb = _two_chains()
        if policy.startswith("least recently used, any"):
            freed = cache.evict_plain(2, a)
        elif policy.endswith("leaves only"):
            freed = cache.evict_leaves(2, [], a)
        else:
            freed = cache.evict_leaves(2, xb, a)
        rx = len(cache.lookup(X, 4, 16, touch=False)) * 4
        ry = len(cache.lookup(Y, 4, 16, touch=False)) * 4
        reach = set(cache.lookup(X, 4, 16, touch=False)) | set(cache.lookup(Y, 4, 16, touch=False))
        stranded = len(cache.map) - len(reach)
        rows.append((policy, _int(freed), _int(len(cache.map)), _int(rx), _int(ry), _int(stranded)))
    return ("Blocks of 4 tokens. Conversation X (tokens 1 to 16) was cached first, its four blocks stamped 1 to 4 in chain "
            "order, then conversation Y (tokens 101 to 116), stamped 5 to 8; both have finished, so all 8 blocks are "
            "held by the cache alone. The pool asks for 2 blocks. 'Reachable' is the cached tokens a lookup of the "
            "conversation's own 16 tokens finds; a stranded block is still indexed and occupies the pool but no lookup "
            "can reach it, because a block before it in its chain is gone.\n\n"
            + _table(["eviction rule", "blocks freed", "blocks still cached", "reachable tokens of X",
                      "reachable tokens of Y", "stranded blocks"], rows, "lrrrrr"))


SHARED, OWN, BS = 1000, 200, 16


def prefix_savings():
    bpb = BS * BF16_TOKEN
    total = SHARED + OWN
    adopt = SHARED // BS
    own_blocks = _ceil_div(total, BS) - adopt
    start = adopt * BS
    cold = kc.flops_new(S8, 0, total)
    warm = kc.flops_new(S8, start, total - start)
    rows = []
    for n in (1, 8, 64):
        f_cold = n * cold
        f_warm = cold + (n - 1) * warm
        b_cold = n * _ceil_div(total, BS)
        b_warm = adopt + n * own_blocks
        rows.append((_int(n), f"{f_cold:.4e}", f"{f_warm:.4e}", _pct(f_warm / f_cold), _int(b_cold), _int(b_warm),
                     f"{b_warm * bpb / GIB:.3f}"))
    return (f"Every request is a shared system prompt of {SHARED:,} tokens followed by {OWN} tokens of its own, "
            f"{total:,} in all, in blocks of {BS} tokens for the 7.5-billion-parameter decoder. A cold prefill computes all "
            f"{total:,} rows: {cold:.4e} FLOPs (`flops_new` of Volume V, Chapter 12, from position 0). A request that "
            f"finds the system prompt cached adopts {SHARED:,} // {BS} = {adopt} full blocks ({start:,} tokens; the "
            f"last {SHARED - start} system tokens share a block with its own tokens) and computes {total - start} rows "
            f"from position {start:,}: {warm:.4e} FLOPs. The first request is always cold. Blocks are those held while "
            f"all requests are alive at once: {_ceil_div(total, BS)} per request without sharing, {adopt} shared plus "
            f"{own_blocks} per request with it; a block is {BS} x {BF16_TOKEN:,} = {bpb:,} bytes in bf16.\n\n"
            + _table(["requests", "prefill FLOPs, no cache", "prefill FLOPs, cached prefix", "% of the cold FLOPs",
                      "blocks, no sharing", "blocks, shared prefix", "GiB, shared prefix"], rows))


# ------------------------------------------------------------------------------------------------ the formats in a pool

FORMATS = ("f32", "bf16", "int8", "fp8")


def block_bytes(fmt, bs=BS, s=S8):
    """Bytes of one block of bs token rows across every layer, K and V, with the per-row scales of the 8-bit formats."""
    return int(round(lc.cache_bytes_per_token(s, fmt) * bs))


def pool_formats():
    ls = conversation_lengths()
    rows = []
    for fmt in FORMATS:
        bb = block_bytes(fmt)
        blocks = POOL_BYTES // bb
        held, stored, used = admit_in_order(ls, lambda n: _ceil_div(n, BS), blocks)
        rows.append((fmt, f"{lc.bytes_per_element(fmt, S8['dh']):.5f}", _int(bb), _int(blocks), _int(blocks * BS), _int(held)))
    return ("Blocks of 16 tokens of the 7.5-billion-parameter decoder (32 layers, 8 key-value heads of width 128): a block "
            "holds 16 x 2 x 32 x 8 x 128 elements, and in int8 and fp8 also one 4-byte scale for every (token, layer, "
            "K or V, key-value head) row of 128 elements, so 1 + 4/128 bytes per element. The pool is the 64 GiB of the "
            "first example, its blocks the whole blocks that fit (rounded down). Conversations are those of the first "
            "example, admitted in arrival order at their final length until the next does not fit.\n\n"
            + _table(["format", "bytes per element", "bytes per block", "blocks in 64 GiB", "token rows",
                      "conversations held"], rows))


# ------------------------------------------------------------------------------------------------ the clip ceiling

NAME_CLIP = "paged_kv.clip_tradeoff"
D_ROW, N_ROWS = 64, 400
OUTLIERS = (3.0, 5.0, 10.0, 20.0)
CLIPS = (None, 4.0, 8.0)


def quantize_row_clipped(row, clip):
    """Symmetric int8 of one row: scale = min(absmax, clip) / 127 (1 when it is 0), codes round(x / scale) with ties to
    even, clamped to -127 .. 127; returns the decoded row."""
    amax = max(abs(x) for x in row)
    capped = amax if clip is None else min(amax, clip)
    s = capped / 127.0
    if s == 0.0:
        s = 1.0
    return [max(-127, min(127, round(x / s))) * s for x in row]


def _clip_rows():
    g = bookrng.generator(NAME_CLIP)
    out = {}
    for m in OUTLIERS:
        rows = []
        for _ in range(N_ROWS):
            r = [g.normal() for _ in range(D_ROW - 1)]
            sign = 1.0 if g.uniform() < 0.5 else -1.0
            rows.append([sign * m] + r)
        out[m] = rows
    return out


def clip_tradeoff():
    data = _clip_rows()
    rows = []
    for m in OUTLIERS:
        for c in CLIPS:
            e_out = e_rest = tot = 0.0
            for r in data[m]:
                q = quantize_row_clipped(r, c)
                e_out += (q[0] - r[0]) ** 2
                for x, y in zip(r[1:], q[1:]):
                    e_rest += (y - x) ** 2
                tot += sum(x * x for x in r)
            rows.append((f"{m:g}", "absmax (no ceiling)" if c is None else f"{c:g}",
                         f"{e_out / N_ROWS:.3g}", f"{e_rest / N_ROWS:.3g}", f"{math.sqrt((e_out + e_rest) / tot):.3g}"))
    return (f"{N_ROWS} rows of {D_ROW} elements for each outlier size m, from the generator `{NAME_CLIP}` (seed "
            f"{_seed(NAME_CLIP)}), one generator across the whole table: for m = 3, 5, 10, 20 in turn and each row in turn, "
            "63 `normal()` draws fill elements 1 to 63, then one `uniform()` gives the sign of element 0, + when it is below "
            "0.5, and element 0 is that sign times m. Each row is stored as int8 with scale = min(absmax, ceiling) / 127 "
            "and codes round(x / scale) (to nearest, ties to even) clamped to -127 .. 127. Squared errors are summed per "
            "row and averaged over the rows; the last column is sqrt(total squared error / total sum of squares) over "
            "all rows of that m. Three significant digits, which the draws fix.\n\n"
            + _table(["outlier m", "ceiling", "squared error of the outlier, per row", "squared error of the other 63, per row",
                      "relative error of the rows"], rows))


# ------------------------------------------------------------------------------------------------ preemption

SCHED = {"bs": 16, "pool": 24, "cached": 3, "prompts": (("A", 60, 20), ("B", 70, 60), ("C", 80, 50), ("D", 90, 40))}
TRACE_HOST = 7
COST_HOSTS = (0, 7, 16)


def _blocks_for(n, bs):
    return _ceil_div(n, bs)


def simulate(host_blocks, p=SCHED):
    """Run the chapter's scheduler model; returns (log, summary)."""
    bs = p["bs"]
    seqs = {n: {"len": L, "held": _blocks_for(L, bs), "gen": 0, "want": g} for n, L, g in p["prompts"]}
    running = [n for n, _, _ in p["prompts"]]
    swapped, waiting = [], []
    cache = p["cached"]
    free = p["pool"] - cache - sum(s["held"] for s in seqs.values())
    host_used = 0
    out_b = in_b = 0
    recomputed = []
    log = []
    step = 0

    def next_need(n):
        s = seqs[n]
        return _blocks_for(s["len"] + 1, bs) - s["held"]

    def reclaim(k):
        nonlocal cache, free
        r = min(k, cache)
        cache -= r
        free += r
        return r

    finished = []
    while running or swapped or waiting:
        step += 1
        events = []
        promoted = set()
        # 0. resume swapped sequences, oldest demotion first
        while swapped:
            n = swapped[0]
            slack = sum(next_need(r) for r in running) + 1 if running else 0
            need = seqs[n]["held"] + slack
            if free < need:
                got = reclaim(need - free)
                if got:
                    events.append(f"cache gives back {got}")
                if free < need:
                    break
            swapped.pop(0)
            free -= seqs[n]["held"]
            host_used -= seqs[n]["held"]
            in_b += seqs[n]["held"]
            running.append(n)
            promoted.add(n)
            events.append(f"{n} swapped in ({seqs[n]['held']} blocks)")
        # 1. re-admit sequences preempted by recompute, only while nothing is swapped out
        while not swapped and waiting:
            n = waiting[0]
            want = _blocks_for(seqs[n]["len"] + 1, bs) + sum(next_need(r) for r in running)
            if free < want:
                got = reclaim(want - free)
                if got:
                    events.append(f"cache gives back {got}")
                if free < want:
                    break
            waiting.pop(0)
            seqs[n]["held"] = _blocks_for(seqs[n]["len"], bs)
            free -= seqs[n]["held"]
            recomputed.append(seqs[n]["len"])
            running.append(n)
            events.append(f"{n} prefilled again ({seqs[n]['len']} rows)")
        # 2. make room for one more row of every running sequence
        need = sum(next_need(r) for r in running)
        if free < need:
            got = reclaim(need - free)
            if got:
                events.append(f"cache gives back {got}")
        while free < need and len(running) > 1:
            cands = [r for r in running if r not in promoted] or running
            v = cands[-1]
            running.remove(v)
            need -= next_need(v)
            k = seqs[v]["held"]
            free += k
            seqs[v]["held"] = 0
            if host_blocks - host_used >= k:
                host_used += k
                seqs[v]["held"] = k          # the count the swap-in must bring back
                out_b += k
                swapped.append(v)
                events.append(f"{v} swapped out ({k} blocks)")
            else:
                waiting.insert(0, v)
                events.append(f"{v} dropped for recompute ({k} blocks)")
        # 3. decode one row for every running sequence
        for r in running:
            s = seqs[r]
            if s["len"] % bs == 0:
                free -= 1
                s["held"] += 1
            s["len"] += 1
            s["gen"] += 1
        # 4. reap
        for r in list(running):
            if seqs[r]["gen"] == seqs[r]["want"]:
                running.remove(r)
                free += seqs[r]["held"]
                seqs[r]["held"] = 0
                finished.append(r)
                events.append(f"{r} finishes")
        assert free >= 0, "the model let the pool run dry"
        if events:
            log.append((step, "; ".join(events), free, host_used))
    summary = {"steps": step, "out": out_b, "in": in_b, "recomputed": recomputed, "order": finished}
    return log, summary


def preemption_trace():
    p = SCHED
    log, _ = simulate(TRACE_HOST)
    rows = [(_int(s), e, _int(f), _int(h)) for s, e, f, h in log]
    seqs = ", ".join(f"{n} with {L} rows cached and {g} to append" for n, L, g in p["prompts"])
    return (f"Blocks of {p['bs']} rows, a device pool of {p['pool']} blocks of which {p['cached']} are held by the prefix "
            f"cache alone, and a host pool of {TRACE_HOST} blocks. Four sequences are running at step 0, in admission order: "
            f"{seqs}. Each appends one row per step and finishes when it has appended its count. A step runs, in order: "
            "(0) swap sequences back in, oldest swap-out first, while the free blocks cover the sequence's blocks plus one "
            "row's growth of every running sequence plus 1 (no slack when nothing runs), and stop at the first that does "
            "not fit; (1) while nothing is swapped out, prefill again the sequences dropped for recompute, front of the "
            "queue first, when the free blocks cover ceil((rows + 1) / 16) blocks plus the running sequences' growth; "
            "(2) if the free blocks do not cover one row's growth of every running sequence, take cache-only blocks first, "
            "then remove the newest running sequence (not one swapped in this step, unless no other is left) until they "
            "do or one sequence is left: it is swapped out when the host pool has room for all its blocks and dropped for "
            "recompute otherwise, its blocks freed either way; (3) every running sequence appends a row, taking a block "
            "when its last is full; (4) finished sequences free their blocks. Steps (0) and (1) also take cache-only "
            "blocks when they are short. A newly swapped-in or re-prefilled sequence joins the end of the running list. "
            "Only the steps on which something other than a plain decode happens are listed.\n\n"
            + _table(["step", "events, in order", "free device blocks after the step", "host blocks in use"], rows, "lllr"))


def preemption_costs():
    rows = []
    bb = block_bytes("int8")
    _, sm7 = simulate(TRACE_HOST)
    last = sm7["recomputed"][-1]
    nb = _blocks_for(last, SCHED["bs"])
    fl_last = kc.flops_new(S8, 0, last)
    swap_bytes = 2 * nb * bb
    for host in COST_HOSTS:
        _, sm = simulate(host)
        bus = (sm["out"] + sm["in"]) * bb
        fl = sum(kc.flops_new(S8, 0, n) for n in sm["recomputed"])
        rec = " + ".join(_int(n) for n in sm["recomputed"]) or "none"
        rows.append((_int(host), _int(sm["steps"]), "".join(sm["order"]), _int(sm["out"]), _int(sm["in"]), rec,
                     f"{bus:,}", "0" if fl == 0 else f"{fl:.4e}"))
    return ("The trace run with host pools of 0, 7 and 16 blocks, and the cost if the blocks were those of the 7.5-billion-parameter decoder in "
            f"int8 ({bb:,} bytes per block of 16 tokens): bytes over the host bus are the blocks copied out plus the blocks "
            "copied back, times the block's bytes; FLOPs of the prefills run again are `flops_new` of Volume V, Chapter 12, "
            "from position 0, summed over the re-prefills.\n\n"
            + _table(["host blocks", "steps until all finish", "finishing order", "blocks out", "blocks in",
                      "rows prefilled again", "bytes over the bus", "FLOPs prefilled again"], rows, "rrlrrlrr")
            + f"\n\nThe second preemption of D, priced both ways: prefilling its {last:,} rows again is {fl_last:.4e} FLOPs, "
            f"and swapping its {nb} blocks out and back is {swap_bytes:,} bytes, so the recompute spends "
            f"{fl_last / swap_bytes:,.0f} FLOPs (rounded to a whole number) for each byte the swap would move.")
