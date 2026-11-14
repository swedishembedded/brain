# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Memory Budgets and Model Residency".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: residency.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared with the
chapter's figures so a picture and the table beside it cannot disagree.

All sizes are whole GiB (1 GiB = 2^30 bytes) or integer counts. The one stream of random requests comes from the book's
SplitMix64 generator, and its draws are integers, so another implementation reproduces every printed digit. The one place that
divides (the cost-aware score) compares real numbers by order only.

Swedish Embedded AB implements memory-budgeted, multi-model serving for teams whose models do not all fit resident at once.
If your team needs expertise in model residency, eviction policy and placement across several devices then you can procure our
services by sending an email to info@swedishembedded.com.
"""
import bookrng
import number_formats as nf

_table = nf._table

GIB = 1 << 30


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


# ---------------------------------------------------------------------------------------------------------------------
# A device's budget and a pool of devices that share memory
# ---------------------------------------------------------------------------------------------------------------------

class _Budget:
    """total, reserved headroom (never above total) and the bytes charged so far."""

    def __init__(self, total, reserved):
        self.total, self.reserved, self.used = total, min(reserved, total), 0

    def free(self):
        return max(0, self.total - self.reserved - self.used)

    def fits(self, n):
        return n <= self.free()


def budgets_and_pools():
    b = _Budget(24, 2)
    rows = [("a device of 24 GiB with 2 GiB reserved", "-", b.total - b.reserved, b.used, b.free(), "-")]
    for label, n in (("charge 10 GiB", 10), ("charge 10 GiB", 10)):
        assert b.fits(n)
        b.used += n
        rows.append((label, n, b.total - b.reserved, b.used, b.free(), "yes"))
    rows.append(("ask for 10 GiB more", 10, b.total - b.reserved, b.used, b.free(), "yes" if b.fits(10) else "no"))
    rows.append(("ask for 2 GiB more", 2, b.total - b.reserved, b.used, b.free(), "yes" if b.fits(2) else "no"))
    t1 = _table(["step", "bytes asked (GiB)", "usable (GiB)", "used (GiB)", "free (GiB)", "fits"], rows, "lrrrrr")

    # Two devices of 24 GiB each that share one pool of 20 GiB: the free bytes of a device are the smaller of its own and the pool's.
    pool_total = 20
    rows = []
    for label, charge in (("nothing charged", 0), ("charge 8 GiB to the gpu", 8), ("release it", 0)):
        gpu_free = min(24 - charge, pool_total - charge)
        cpu_free = min(24, pool_total - charge)
        rows.append((label, charge, gpu_free, cpu_free))
    t2 = _table(["step", "pool used (GiB)", "free on the gpu (GiB)", "free on the cpu (GiB)"], rows, "lrrr")
    return (
        "A device's budget: total capacity 24 GiB, reserved headroom 2 GiB, used bytes starting at 0. free = max(0, total - "
        "reserved - used); a request of n GiB fits when n <= free; a charge adds n to used and is made only after a fit. "
        "Sizes are whole GiB.\n\n" + t1 +
        "\n\nTwo devices that physically share memory. Each has its own budget of 24 GiB with no reserve, and both are declared "
        "into one pool of 20 GiB with no reserve. A charge on either device is added to the device's own used bytes and to the "
        "pool's. The free bytes of a device are the smaller of its own free bytes and the pool's free bytes.\n\n" + t2)


# ---------------------------------------------------------------------------------------------------------------------
# One device, many models: the resident table and the two eviction policies
# ---------------------------------------------------------------------------------------------------------------------

class _Entry:
    def __init__(self, size, last_use):
        self.size, self.last_use, self.uses = size, last_use, 1


def _score_lru(e, now):
    return e.last_use


def _score_cost(e, now):
    # uses x bytes / (age + 1): evict the lowest score first.
    return e.uses * (e.size * GIB) / (now - e.last_use + 1)


class _Device:
    """One device's budget and resident table, driven by claim then release of one model at a time.

    The logical clock advances by one at every claim (a hit or an insertion) and by one at every release. A hit adds 1 to the
    entry's use count; the release only refreshes its last-use stamp. Victims are taken in ascending order of score (ties in
    ascending order of last use) until the bytes freed cover the deficit."""

    def __init__(self, usable, score):
        self.usable, self.score, self.tick, self.entries = usable, score, 0, {}
        self.loads = self.loaded_gib = self.hits = 0

    def used(self):
        return sum(e.size for e in self.entries.values())

    def claim(self, name, size):
        """Returns (hit, victims)."""
        if name in self.entries:
            self.tick += 1
            e = self.entries[name]
            e.last_use, e.uses = self.tick, e.uses + 1
            self.hits += 1
            self._release(name)
            return True, []
        victims = []
        deficit = size - (self.usable - self.used())
        if deficit > 0:
            order = sorted(self.entries.items(), key=lambda kv: kv[1].last_use)
            order.sort(key=lambda kv: self.score(kv[1], self.tick))      # stable: equal scores keep last-use order
            for k, e in order:
                victims.append(k)
                if e.size >= deficit:
                    deficit = 0
                    break
                deficit -= e.size
            assert deficit == 0, "a model that fits an empty device always has enough victims"
            for k in victims:
                del self.entries[k]
        self.tick += 1
        self.entries[name] = _Entry(size, self.tick)
        self.loads += 1
        self.loaded_gib += size
        self._release(name)
        return False, victims

    def _release(self, name):
        self.tick += 1
        self.entries[name].last_use = self.tick


def lru_swap():
    d = _Device(22, _score_lru)
    seq = ["a", "b", "c", "b", "a", "c"]
    rows = []
    for i, n in enumerate(seq):
        hit, vic = d.claim(n, 10)
        order = sorted(d.entries, key=lambda k: d.entries[k].last_use)
        rows.append((i, n, "hit" if hit else "load", ", ".join(vic) or "-", ", ".join(order), 22 - d.used()))
    t = _table(["step", "claim", "outcome", "evicted", "resident, oldest use first", "free (GiB)"], rows, "rllllr")
    return (
        "One card of 24 GiB with 2 GiB reserved (usable 22 GiB) and three models a, b and c of 10 GiB each, claimed in the order "
        "a, b, c, b, a, c. Policy: least recently used first. The clock advances by 1 at every claim and by 1 at every release, "
        "and a claim finds its model resident (a hit, no bytes moved) or loads it, evicting the least recently used models "
        "until the deficit is covered. Steps are counted from 0.\n\n" + t +
        f"\n\nLoads: {d.loads}, bytes loaded: {d.loaded_gib} GiB, hits: {d.hits} of {len(seq)} claims.")


_SCENARIO = [  # name, size in GiB, request weight
    ("m1", 6, 60), ("m2", 12, 30), ("m3", 2, 20), ("m4", 8, 15), ("m5", 4, 12), ("m6", 1, 10)]
NAME_STREAM = "residency.eviction_stream"
N_STREAM = 2000
USABLE_STREAM = 22


def _stream():
    g = bookrng.generator(NAME_STREAM)
    total = sum(w for _, _, w in _SCENARIO)
    out = []
    for _ in range(N_STREAM):
        r = g.integer(total)
        for name, size, w in _SCENARIO:
            if r < w:
                out.append((name, size))
                break
            r -= w
    return out


def _run(score, stream):
    d = _Device(USABLE_STREAM, score)
    for name, size in stream:
        d.claim(name, size)
    return d


def eviction_stream():
    stream = _stream()
    rows = []
    for label, score in (("least recently used", _score_lru), ("cost-aware: uses x bytes / age", _score_cost)):
        d = _run(score, stream)
        rows.append((label, d.hits, N_STREAM - d.hits, d.loaded_gib, f"{100 * d.hits / N_STREAM:.1f}"))
    t = _table(["policy", "hits", "loads", "GiB loaded", "hit rate (%)"], rows, "lrrrr")
    inv = _table(["model", "size (GiB)", "request weight", "share of requests (%)"],
                 [(n, s, w, f"{100 * w / sum(x for _, _, x in _SCENARIO):.1f}") for n, s, w in _SCENARIO], "lrrr")
    return (
        f"Six models share one device of 22 GiB usable. A stream of {N_STREAM} requests, one claim and release each: request "
        f"k asks for model m with probability proportional to its weight. Generator: the book's SplitMix64, seed "
        f"{_seed(NAME_STREAM)} (`seed_of` of the name `{NAME_STREAM}`), one draw `integer(W)` per request with W = "
        f"{sum(w for _, _, w in _SCENARIO)} the sum of the weights, mapped by walking the models in the order listed and "
        "subtracting each weight from the draw until it is smaller than the next weight. One generator runs once through the "
        "stream and both policies replay the same stream. Everything else is the rule of the previous example: the clock "
        "advances at every claim and every release, a hit adds 1 to the use count, a load starts it at 1, and the score of "
        "the cost-aware policy at clock value now is uses x bytes / (now - last use + 1) with the lowest score evicted first, where "
        "now is the clock value when the claim arrives, before the claim's own tick (the value left by the previous request's release).\n\n"
        + inv + "\n\n" + t)


def pinned_scenario():
    """The scenario that motivated the cost-aware policy: a hot 4 GiB model and a cold 200 MiB one that was touched last."""
    now = 53
    rows = []
    for label, score in (("least recently used", _score_lru), ("cost-aware", _score_cost)):
        hot, cold = _Entry(4.0, 52), _Entry(200 / 1024, 53)
        hot.uses = 51
        ranked = sorted([("hot 4 GiB", hot), ("cold 200 MiB", cold)], key=lambda kv: score(kv[1], now))
        (v, ev), (k, ek) = ranked
        rows.append((label, v, f"{score(ev, now):.6g}", k, f"{score(ek, now):.6g}"))
    return (
        "A hot model of 4 GiB (51 uses, last used at clock 52) and a cold one of 200 MiB (1 use, last used at clock 53, the "
        "most recent), clock now 53 (both inserted, the hot one touched 50 times, the cold one touched last). The score of the least-recently-used policy is the last-use clock value; the cost-aware "
        "score is uses x bytes / (now - last use + 1), bytes in bytes (1 GiB = 2^30). The model with the lower score is "
        "evicted first.\n\n"
        + _table(["policy", "evicted first", "its score", "kept", "its score"], rows, "lllll"))


# ---------------------------------------------------------------------------------------------------------------------
# Deciding what a claim does: place, evict, spill to the host, refuse
# ---------------------------------------------------------------------------------------------------------------------

def _decide(cost, gpus, host, residents, peak):
    """gpus: list of [name, total, reserved, used]; host: (total, reserved, used) or None; residents: per gpu list of
    (name, size) from least to most recently used; peak: per gpu the largest total it has had. Returns a text."""
    def free(g):
        return max(0, g[1] - g[2] - g[3])
    fit = [g for g in gpus if free(g) >= cost]
    if fit:
        best = max(fit, key=lambda g: (free(g), -gpus.index(g)))
        return f"place on {best[0]}", ""
    host_usable = (host[0] - host[1]) if host else 0
    ever = any(peak[g[0]] - g[2] >= cost for g in gpus) or host_usable >= cost
    if not ever:
        return "refuse: too large", "no device has ever held it"
    empty_now = any(g[1] - g[2] >= cost for g in gpus) or host_usable >= cost
    if not empty_now:
        return "refuse: starved", "an empty device would not hold it now"
    best = None                                       # the plan that evicts the fewest bytes; the first card on a tie
    for g in gpus:
        if g[1] - g[2] < cost:
            continue
        deficit = cost - free(g)
        vic, got = [], 0
        for name, size, pinned in residents.get(g[0], []):
            if pinned:
                continue                              # a pinned resident is never a victim
            vic.append(name)
            got += size
            if got >= deficit:
                if best is None or got < best[1]:
                    best = (f"evict {', '.join(vic)} on {g[0]}, then place", got)
                break
    if best:
        return best[0], f"frees {best[1]} GiB"
    if host and host_usable >= cost:
        return "place on the host", "no card can ever hold it"
    return "refuse: no capacity", "nothing evictable frees enough"


def claim_outcomes():
    cards = lambda used0, used1: [["gpu0", 24, 2, used0], ["gpu1", 24, 2, used1]]
    host = (128, 8, 0)
    peak = {"gpu0": 24, "gpu1": 24}
    cases = [
        ("6 GiB; gpu0 holds 15 GiB, gpu1 is empty", 6, cards(15, 0), host, {}, peak),
        ("13 GiB; both cards hold 20 GiB; gpu0 holds one pinned 20 GiB resident, on gpu1 the older resident is 8 GiB, the newer 12 GiB", 13, cards(20, 20), host,
         {"gpu0": [("pinned", 20, True)], "gpu1": [("old", 8, False), ("new", 12, False)]}, peak),
        ("23 GiB; cards empty", 23, cards(0, 0), host, {}, peak),
        ("121 GiB; cards empty", 121, cards(0, 0), host, {}, peak),
        ("20 GiB; one card of 24 GiB, no host budget, empty", 20, [["gpu0", 24, 2, 0]], None, {}, {"gpu0": 24}),
        ("20 GiB; the same card now measures 14 GiB total", 20, [["gpu0", 14, 2, 0]], None, {}, {"gpu0": 24}),
        ("25 GiB; the same card at its best 24 GiB", 25, [["gpu0", 24, 2, 0]], None, {}, {"gpu0": 24}),
    ]
    rows = []
    for label, cost, gpus, h, res, pk in cases:
        out, why = _decide(cost, gpus, h, res, pk)
        rows.append((label, out, why or "-"))
    return (
        "A model that holds only device bytes (cost c GiB) claims a device. Cards: 24 GiB total with 2 GiB reserved each; host: "
        "128 GiB total with 8 GiB reserved (usable 120 GiB) where a host is listed. The rule, in order: (1) among the cards "
        "whose free bytes are at least c, take the one with the most free bytes (the lower index on a tie); (2) otherwise, if "
        "every card's best usable bytes (the largest total it has had, less its reserve) and the host's usable bytes are all "
        "below c, refuse as too large; (3) otherwise, if every card's current usable bytes and the host's usable bytes are "
        "all below c, refuse as starved; (4) otherwise, for each card whose current usable bytes are at "
        "least c, plan to evict its unpinned residents, oldest use first, until the deficit is covered (a pinned resident is never a "
        "victim, so a card whose unpinned residents cannot cover the deficit has no plan), and carry out the plan that evicts the "
        "fewest bytes, the lower index on a tie, as `plan_eviction_with` does; (5) otherwise place on the host if "
        "its free bytes allow, else refuse for lack of capacity. Free bytes of a card: total - reserved - used.\n\n"
        + _table(["case", "outcome", "note"], rows, "lll"))


# ---------------------------------------------------------------------------------------------------------------------
# Parts that are live in different phases
# ---------------------------------------------------------------------------------------------------------------------

def _charge(parts):
    """parts: list of (name, gib, phase) with phase None for permanent. Returns (sum, peak)."""
    perm = sum(g for _, g, p in parts if p is None)
    by = {}
    for _, g, p in parts:
        if p is not None:
            by[p] = by.get(p, 0) + g
    return sum(g for _, g, _ in parts), perm + max(by.values(), default=0)


def phased_fit():
    cases = [
        ("denoiser 16 (phase 1), decoder 16 (phase 2)", [("dit", 16, 1), ("vae", 16, 2)]),
        ("denoiser 16 (phase 1), encoder 16 (phase 1)", [("dit", 16, 1), ("enc", 16, 1)]),
        ("text encoder 10 (permanent), denoiser 16 (phase 1)", [("te", 10, None), ("dit", 16, 1)]),
        ("text encoder 6 (permanent), denoiser 16 (phase 1), decoder 16 (phase 2)",
         [("te", 6, None), ("dit", 16, 1), ("vae", 16, 2)]),
        ("denoiser 13 (phase 1), vae encode 8 (phase 1), vae decode 11 (phase 2)",
         [("dit", 13, 1), ("vae_enc", 8, 1), ("vae_dec", 11, 2)]),
    ]
    rows = []
    for label, parts in cases:
        s, pk = _charge(parts)
        cap = 22 if "vae encode" in label else 24
        rows.append((label, cap, s, pk, "yes" if pk <= cap else "no", "yes" if s <= cap else "no"))
    t = _table(["parts (GiB)", "card (GiB)", "sum", "peak", "fits by the peak", "fits by the sum"], rows, "lrrrrr")

    # Order of placement: permanent bytes first, against the largest-first order.
    def place(order):
        cards = {"gpu0": 23, "gpu1": 8}
        out = {}
        for name, g in order:
            c = max((k for k in cards if cards[k] >= g), key=lambda k: cards[k], default=None)
            if c is None:
                out[name] = "host"
            else:
                cards[c] -= g
                out[name] = c
        return out
    largest_first = place([("te", 18), ("dit", 14)])
    permanent_first = place([("dit", 14), ("te", 18)])
    t2 = _table(["order of placement", "te (18 GiB, live in one phase)", "dit (14 GiB, permanent)"],
                [("largest first: te, then dit", largest_first["te"], largest_first["dit"]),
                 ("permanent first: dit, then te", permanent_first["te"], permanent_first["dit"])], "lll")
    return (
        "The charge a card pays for a set of parts that share it is the bytes of its permanent parts plus the largest sum, "
        "over the phases, of the parts live in one phase; the sum of all parts is shown for comparison. All sizes are GiB "
        "and every card has no reserve (the card is 24 GiB, except in the last row, 22 GiB).\n\n" + t +
        "\n\nPlacement order. Two cards of 23 GiB and 8 GiB with no reserve, and a host that has room. Each part in turn goes to "
        "the card with the most free bytes among those that fit it, else to the host; a card is charged the part's size. "
        "The text encoder `te` is live in one phase only, so the cards are charged for it only at that time, but here the "
        "order alone decides who gets the large card.\n\n" + t2)


# ---------------------------------------------------------------------------------------------------------------------
# A window of weight groups over a schedule that is known in advance
# ---------------------------------------------------------------------------------------------------------------------

def _scan(n_groups, passes, budget, plan, lookahead=1):
    """Replays `passes` passes over groups 0 .. n_groups - 1 through a window of `budget` slots. Returns the number of loads
    (an advance that finds its group in no slot). plan: "scan" pins the first budget - reserve groups and evicts the
    unpinned resident whose next use is furthest (the last such in slot order on a tie, and a group never used again counts
    as infinitely far); "lru" pins nothing and evicts the resident whose last use is oldest (a group never used counts as
    older than any)."""
    order = [g for _ in range(passes) for g in range(n_groups)]
    slots = [None] * budget
    pinned = set()
    if plan == "scan":
        reserve = 0 if budget >= n_groups else min(max(lookahead, 1), budget)
        pin_n = min(max(budget - reserve, 0), n_groups)
        for i in range(pin_n):
            slots[i] = i
        pinned = set(range(pin_n))
    loads = 0
    for c, g in enumerate(order):
        if g in slots:
            continue
        loads += 1
        if None in slots:
            slots[slots.index(None)] = g
            continue
        res = [s for s in slots if s is not None and s not in pinned]
        if plan == "scan":
            def nxt(x):
                return next((i for i in range(c, len(order)) if order[i] == x), 1 << 60)
            victim = res[0]
            for x in res:
                if nxt(x) >= nxt(victim):
                    victim = x
        else:
            def last(x):
                return max((i for i in range(c) if order[i] == x), default=-1)
            victim = min(res, key=last)
        slots[slots.index(victim)] = g
    return loads


WINDOW_GROUPS, WINDOW_PASSES, GROUP_MIB = 32, 4, 400


def window_series(budgets):
    return ([_scan(WINDOW_GROUPS, WINDOW_PASSES, b, "scan") for b in budgets],
            [_scan(WINDOW_GROUPS, WINDOW_PASSES, b, "lru") for b in budgets])


def weight_window():
    budgets = [4, 8, 10, 16, 24, 31, 32, 40]
    scan, lru = window_series(budgets)
    rows = []
    for b, s, l in zip(budgets, scan, lru):
        bound = 0 if b >= WINDOW_GROUPS else WINDOW_PASSES * (WINDOW_GROUPS - (b - 1))
        rows.append((b, s, bound, l, f"{s * GROUP_MIB / 1024:.2f}", f"{l * GROUP_MIB / 1024:.2f}"))
    return (
        f"A model of {WINDOW_GROUPS} weight groups (one per block) of {GROUP_MIB} MiB each is run {WINDOW_PASSES} times over "
        "its groups in order 0, 1, ..., 31, a schedule known before the first step. The window has `slots` slots, each holding "
        "one group; a slot that is empty is filled first. A load is a group that is in no slot when its turn comes. scan: "
        "reserve 1 slot when the window is narrower than the model, pin groups 0 to slots - 2 (in slot order) and never "
        "evict them, and on a load into a full window evict the unpinned resident whose next use is furthest ahead; with "
        "slots at least 32 it pins every group. lru: pin nothing and evict the resident whose last use is oldest. The "
        "closed form for scan is 4 x (32 - (slots - 1)) loads below 32 slots and 0 from 32 slots up. GiB moved = loads x "
        f"{GROUP_MIB} MiB / 1,024.\n\n"
        + _table(["slots", "loads, scan", "closed form", "loads, lru", "GiB moved, scan", "GiB moved, lru"], rows, "rrrrrr"))


# ---------------------------------------------------------------------------------------------------------------------
# The memory authority: a ledger of grants against a live view of the pool
# ---------------------------------------------------------------------------------------------------------------------

def authority_ledger():
    total, reserved = 30, 4
    avail, charged = 30, 0
    rows = []

    def headroom():
        return max(0, min(avail, max(0, total - charged)) - reserved)

    def row(event, result):
        rows.append((event, avail, charged, headroom(), result))

    def request(n):
        nonlocal charged
        usable = max(0, total - reserved)
        if n > usable:
            return "never fits"
        if n > headroom():
            return "would exceed the pool now"
        charged += n
        return "granted"

    row("start: pool of 30 GiB, 4 GiB reserved, probe sees all of it", "-")
    r = request(14)
    row("request 14 GiB on the gpu (same pool as the cpu)", r)
    r = request(12)
    row("request 12 GiB on the cpu", r)
    avail = 8
    row("another process takes memory: the probe now reports 8 GiB available", "-")
    r = request(8)
    row("request 8 GiB", r)
    charged -= 12
    row("the 12 GiB grant is dropped", "released")
    r = request(30)
    row("request 30 GiB", r)
    r = request(5)
    row("request 5 GiB", r)
    r = request(27)
    row("request 27 GiB", r)
    return (
        "One pool of 30 GiB, shared by the gpu and the cpu, with 4 GiB reserved. The authority's ledger counts the bytes of "
        "its live grants (charged). The probe reports what the machine says is available. headroom = max(0, min(available, "
        "30 - charged) - 4). A request of n GiB is refused as never fitting when n exceeds 30 - 4 = 26 GiB, refused as "
        "exceeding the pool when n exceeds the headroom, and otherwise granted and added to charged; a refused request "
        "charges nothing. All values in GiB.\n\n"
        + _table(["event", "probe available", "charged", "headroom", "result"], rows, "lrrrl"))


# ---------------------------------------------------------------------------------------------------------------------
# The window policy against the best any policy could do on a known schedule
# ---------------------------------------------------------------------------------------------------------------------

def _minimum_loads(n_groups, passes, slots):
    """The fewest loads any eviction rule needs for the schedule, found by exhaustive search over every choice of victim.
    The window starts empty and every first touch counts as a load."""
    from functools import lru_cache
    order = tuple(g for _ in range(passes) for g in range(n_groups))

    @lru_cache(None)
    def rec(c, cache):
        if c == len(order):
            return 0
        g = order[c]
        if g in cache:
            return rec(c + 1, cache)
        if len(cache) < slots:
            return 1 + rec(c + 1, tuple(sorted(cache + (g,))))
        return 1 + min(rec(c + 1, tuple(sorted(tuple(x for x in cache if x != v) + (g,)))) for v in cache)
    return rec(0, ())


def window_versus_optimum():
    cases = [(6, 3, 1), (6, 3, 2), (6, 3, 3), (8, 4, 2), (8, 4, 4), (10, 4, 3), (12, 5, 4)]
    rows = []
    for n, s, p in cases:
        pre = min(s - 1, n)
        reloads = _scan(n, p, s, "scan")
        best = _minimum_loads(n, p, s)
        rows.append((n, s, p, pre, reloads, pre + reloads, best, pre + reloads - best))
    return (
        "Small schedules, so that every sequence of eviction choices can be tried. groups x passes touches in the order "
        "0, ..., groups - 1 repeated; the window starts empty. scan: the policy of the previous example (it preloads "
        "slots - 1 groups, which are loads too, and then reloads on a miss). minimum: the fewest loads, first touches "
        "included, over every choice of victim at every miss, found by exhaustive search with no pinning rule.\n\n"
        + _table(["groups", "slots", "passes", "preloaded by scan", "reloads by scan", "loads by scan, in all",
                  "minimum loads", "scan minus minimum"], rows, "rrrrrrrr"))
