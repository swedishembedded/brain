# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume XIV project "A Multi-Client Inference Server".

Plain Python 3, standard library only, written from the project's own description. The model and the routine that runs tokens
through it on top of a cache are imported from the KV cache chapter of Volume V (`kv_cache.model`, `kv_cache.extend`,
`kv_cache.Meter`), never copied: the tiny GPT of 5,008 numbers drawn from `seed_of('kv_cache.model')`, greedy decoding with the
lowest id on a tie. What this module writes is the server around that routine:

* a paged cache: a fixed pool of blocks of `BLK` positions, a stack of free blocks and a block table per sequence, behind an
  object that `kv_cache.extend` reads and writes as if it were a dense cache;
* a step of continuous batching: arrivals, admission against the free blocks and the blocks promised to the running sequences,
  preemption by recompute of the newest sequence when the pool cannot back the next token, one decode row per running sequence;
* the validation of a request message at the boundary, before anything is queued: size, declared length, token range,
  token budget, capacity and the per-client and total bounds of the waiting queue;
* planted mistakes, each a one-line change of the specified server, and the comparisons that catch them.

Every number is either an integer count or comes from the 64-bit arithmetic of `kv_cache.extend`. Nothing is trained and
nothing is chaotic: a printed token is an argmax, and the smallest gap between the best and the second-best logit of the
reference replies is printed beside them, so that another implementation whose summation order differs in the last bits
still picks the same tokens. A build with a planted mistake prints only which replies differ and how its run ends.

Swedish Embedded AB implements solutions for serving language models to many clients from a fixed amount of device memory.
If your team needs expertise in paged caches, continuous batching and hardened inference endpoints then you can procure our
services by sending an email to info@swedishembedded.com.
"""
from functools import lru_cache

import bookrng
import kv_cache as kc

_table = kc._table

# ------------------------------------------------------------------ the model and the server's constants

CFG = kc.CFG                    # V = 12 ids, D = 16, H = 2, F = 24, L = 2 layers, T = 24 positions
V, D, L, T = CFG["V"], CFG["D"], CFG["L"], CFG["T"]
STOP = 0                        # the id that ends a reply (it is part of the reply)
BLK = 4                        # positions per block
NBLK = 6                        # blocks of the pool
ROWS = 4                        # most sequences one step carries
Q_CLIENT = 4                    # most waiting requests of one client
Q_TOTAL = 16                    # most waiting requests in all
WORD = 4                        # bytes of one field of a message (a signed 32-bit integer)
MAX_MSG = WORD * (2 + T)        # the largest message: max_new, n and T tokens
FLOAT = 8                       # bytes of one stored number (64-bit)
N_REQ = 10                      # honest requests of the workload
FLOOD = 40                      # requests of the flooding client

NAMES = {k: f"project_inference_server.{k}" for k in ("workload", "flood")}


def seed(name):
    return bookrng.seed_of(NAMES[name])


def model():
    return kc.model()


def blocks_for(n):
    """Blocks that hold n positions: ceil(n / BLK)."""
    return -(-n // BLK)


def weight_numbers():
    """Numbers that take part in a matrix product per row: W_Q, W_K, W_V, W_O, W_gate, W_up, W_down of each layer and the head."""
    return kc.weight_params(kc._tiny_shape())


def all_numbers():
    import training_gpt as tg
    return tg.parameter_count(model())


# ------------------------------------------------------------------ the workload


def _draw(g, i, at, client):
    n = 3 + g.integer(6)
    prompt = tuple(g.integer(V) for _ in range(n))
    max_new = 6 + g.integer(11)
    return {"id": i, "arrival": at, "client": client, "prompt": prompt, "max_new": max_new}


@lru_cache(maxsize=None)
def workload():
    """N_REQ requests from one generator, request by request: the gap to the previous arrival integer(3) (none for the
    first, which arrives at step 0), the client integer(3), the prompt length 3 + integer(6), the prompt's tokens integer(12)
    each, max_new 6 + integer(11)."""
    g = bookrng.generator(NAMES["workload"])
    out, at = [], 0
    for i in range(N_REQ):
        if i:
            at += g.integer(3)
        client = g.integer(3)
        out.append(_draw(g, i, at, client))
    return tuple(out)


@lru_cache(maxsize=None)
def flood():
    """FLOOD requests of client 3, all arriving at step 0 ahead of the honest ones, drawn as the honest prompts are."""
    g = bookrng.generator(NAMES["flood"])
    return tuple(_draw(g, 100 + i, 0, 3) for i in range(FLOOD))


def message(prompt, max_new):
    """The message of a request: max_new, n, then the n tokens, each a 32-bit field."""
    return [max_new, len(prompt), *prompt]


# ------------------------------------------------------------------ the reference: one request alone, dense cache


def cut(tokens, max_new):
    """The reply: the first max_new tokens, ended after the first STOP."""
    out = []
    for t in tokens[:max_new]:
        out.append(t)
        if t == STOP:
            break
    return tuple(out)


def _top2_gap(logits):
    s = sorted(logits, reverse=True)
    return s[0] - s[1]


@lru_cache(maxsize=None)
def solo(prompt, max_new):
    """(reply, smallest top-1 gap) of the request alone on the dense cache of Volume V."""
    toks, steps = kc.generate_cached(model(), CFG, list(prompt), max_new)
    reply = cut(toks, max_new)
    gap = min(_top2_gap(s[3]) for s in steps[:len(reply)])
    return reply, gap


# ------------------------------------------------------------------ the paged cache


class PoolExhausted(Exception):
    pass


class PositionOverflow(Exception):
    pass


class EmptyPrefill(Exception):
    pass


class Pool:
    """NBLK blocks; block p holds, for every layer and for keys and values, BLK rows of D numbers (zero at the start). The free
    blocks are a stack whose top is the end of the list; it starts as [NBLK - 1, ..., 1, 0], so block 0 is handed out first, and
    a released block is pushed on top, so the block freed last is handed out next."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.free = list(range(blocks - 1, -1, -1))
        self.K = [[[[0.0] * D for _ in range(BLK)] for _ in range(blocks)] for _ in range(L)]
        self.V = [[[[0.0] * D for _ in range(BLK)] for _ in range(blocks)] for _ in range(L)]

    def alloc(self):
        if not self.free:
            raise PoolExhausted()
        return self.free.pop()

    def release(self, blocks):
        for p in blocks:
            self.free.append(p)


class _Rows:
    """The rows of one layer's keys (or values) of one sequence, as `kv_cache.extend` reads and writes them: row j is slot
    j mod BLK of block table[j div BLK]. The blocks must be in the table before a row is written."""

    def __init__(self, seq, store, layer, mistake):
        self.seq, self.store, self.layer, self.mistake = seq, store, layer, mistake
        self.n = 0

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        if j >= self.n:
            raise IndexError(j)
        return self.store[self.layer][self.seq.table[j // BLK]][j % BLK]

    def extend(self, rows):
        for row in rows:
            j = self.n
            self.n += 1
            # the specified write takes the slot of the position before the count advances
            off = self.n % BLK if self.mistake == "offset after the count" else j % BLK
            self.store[self.layer][self.seq.table[j // BLK]][off] = list(row)


class _Cache:
    """What `kv_cache.extend` sees: K[l] and V[l] of every layer and length()."""

    def __init__(self, seq, pool, mistake):
        self.K = [_Rows(seq, pool.K, l, mistake) for l in range(L)]
        self.V = [_Rows(seq, pool.V, l, mistake) for l in range(L)]

    def length(self):
        return len(self.K[0])


class Seq:
    """One request inside the server."""

    def __init__(self, req):
        self.req = req
        self.id = req["id"]
        self.table = []          # block ids, logical block i first
        self.generated = []
        self.logits = []         # the logits each generated token was chosen from
        self.next = None         # the token to feed next (the last one generated)
        self.cache = None
        self.resumed = False
        self.admitted_at = None
        self.first_at = None
        self.finished_at = None
        self.preemptions = 0

    def context(self):
        """What a prefill must put into the cache: the prompt, then every generated token but the last."""
        return list(self.req["prompt"]) + self.generated[:-1]

    def length(self):
        return self.cache.length() if self.cache is not None else 0

    def done(self):
        g = self.generated
        return bool(g) and (g[-1] == STOP or len(g) == self.req["max_new"])


def run_rows(seq, ids, meter):
    """kv_cache.extend on the sequence's paged cache; the positions must stay inside the model's T."""
    if not ids:
        raise EmptyPrefill()
    if seq.length() + len(ids) > T:
        raise PositionOverflow()
    return kc.extend(model(), CFG, seq.cache, list(ids), meter)


def argmax(logits):
    return kc._argmax(logits)


# ------------------------------------------------------------------ validation at the boundary

REFUSALS = ("too large", "malformed", "length mismatch", "empty prompt", "token out of range", "max_new below 1",
            "exceeds capacity", "client queue full", "queue full")


def capacity(pool_blocks):
    """The most positions one sequence may hold: the model's T, or the whole pool if that is smaller."""
    return min(T, pool_blocks * BLK)


def validate(msg, pool_blocks, waiting_of_client, waiting_total, checks):
    """The first refusal of a message (a list of integers, each one 32-bit field), or None. `checks` names the checks that
    are on; the specified server has all of them, in the order of REFUSALS."""
    on = lambda c: c in checks
    nbytes = WORD * len(msg)
    if on("too large") and nbytes > MAX_MSG:
        return "too large"
    if on("malformed") and len(msg) < 2:
        return "malformed"
    max_new, n, toks = msg[0], msg[1], msg[2:]
    if on("length mismatch") and n != len(toks):
        return "length mismatch"
    if on("empty prompt") and n < 1:
        return "empty prompt"
    if on("token out of range") and any(t < 0 or t >= V for t in toks):
        return "token out of range"
    if on("max_new below 1") and max_new < 1:
        return "max_new below 1"
    if on("exceeds capacity") and n + max_new - 1 > capacity(pool_blocks):
        return "exceeds capacity"
    if on("client queue full") and waiting_of_client >= Q_CLIENT:
        return "client queue full"
    if on("queue full") and waiting_total >= Q_TOTAL:
        return "queue full"
    return None


ALL_CHECKS = frozenset(REFUSALS)

CRASHES = {"PoolExhausted": "no free block for a decode row", "PositionOverflow": "a position past the model's T",
           "IndexError": "a token id outside the embedding table", "EmptyPrefill": "a prefill of no rows"}


# ------------------------------------------------------------------ the server


class Server:
    """One process with one pool. `admission` is 'promise' (the specified rule: the prompt, its first new position and the
    next position of every running sequence must be free) or 'reserve' (every sequence reserves blocks for its whole budget
    when it is admitted, and nothing is preempted). `mistake` names one planted change, or None."""

    def __init__(self, pool_blocks=NBLK, rows=ROWS, admission="promise", mistake=None, checks=ALL_CHECKS, victim="newest"):
        self.pool = Pool(pool_blocks)
        self.rows, self.admission, self.mistake, self.checks, self.victim = rows, admission, mistake, checks, victim
        self.waiting, self.running, self.finished, self.finished_step = [], [], {}, {}
        self.refused = {}
        self.step_no = 0
        self.log = []
        self.flops = {"prefill": 0, "decode": 0, "recompute": 0}
        self.forwards = 0
        self.peak_blocks = 0
        self.peak_waiting = 0
        self.preemptions = 0
        self.preempt_log = []
        self.ledger_broken = None
        self.crash = None

    # -- bookkeeping

    def free(self):
        return len(self.pool.free)

    def need_next(self, s):
        """Blocks s must gain before it can store one more position."""
        return max(0, blocks_for(s.length() + 1) - len(s.table))

    def committed(self):
        return sum(self.need_next(s) for s in self.running)

    def _check_ledger(self):
        held = [p for s in self.running for p in s.table]
        if sorted(held + self.pool.free) != list(range(self.pool.blocks)) and self.ledger_broken is None:
            self.ledger_broken = self.step_no

    def _release(self, s):
        self.pool.release(s.table)
        if self.mistake != "table kept after preemption" or s.done():
            s.table = []

    # -- arrivals

    def submit(self, req, msg=None):
        msg = message(req["prompt"], req["max_new"]) if msg is None else msg
        mine = sum(1 for s in self.waiting if s.req["client"] == req["client"])
        why = validate(msg, self.pool.blocks, mine, len(self.waiting), self.checks)
        if why is not None:
            self.refused[req["id"]] = why
            return why
        s = Seq(req)
        s.arrived_at = self.step_no
        self.waiting.append(s)
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))
        return None

    # -- one step

    def _want(self, s):
        if self.admission == "reserve":
            return blocks_for(len(s.req["prompt"]) + s.req["max_new"] - 1)
        if self.mistake == "admit on the prompt alone":
            return blocks_for(len(s.context()))
        if self.admission == "own":
            return blocks_for(len(s.context()) + 1)
        return blocks_for(len(s.context()) + 1) + self.committed()

    def _admit(self, report, meter_of):
        while self.waiting and len(self.running) < self.rows:
            s = self.waiting[0]
            want = self._want(s)
            if self.free() < want:
                break
            self.waiting.pop(0)
            ctx = s.context()
            if self.mistake == "rebuilt from the prompt alone" and s.resumed:
                ctx = list(s.req["prompt"])
            if self.mistake == "rebuilt with the pending token" and s.resumed:
                ctx = list(s.req["prompt"]) + s.generated
            s.cache = _Cache(s, self.pool, self.mistake)
            take = want if self.admission == "reserve" else blocks_for(len(ctx))
            # the table is empty here; under the planted mistake a resumed sequence still lists its old block ids first
            for _ in range(take):
                s.table.append(self.pool.alloc())
            kind = "recompute" if s.resumed else "prefill"
            logits = run_rows(s, ctx, meter_of(kind))
            report["rows"] += len(ctx)
            if s.admitted_at is None:
                s.admitted_at = self.step_no
            if not s.resumed:
                s.logits.append(logits)
                s.generated.append(argmax(logits))
                s.next = s.generated[-1]
                s.first_at = self.step_no
                report["admitted"].append(s.id)
            else:
                report["resumed"].append(s.id)
            s.resumed = False
            if s.done():
                self._finish(s, report)
            else:
                self.running.append(s)

    def _ensure_room(self, report):
        if self.admission == "reserve" or self.mistake == "admit on the prompt alone":
            return
        need = self.committed()
        while self.free() < need and len(self.running) > 1:
            v = self.running.pop() if self.victim == "newest" else self.running.pop(0)
            need -= self.need_next(v)
            self._release(v)
            v.cache = None
            v.resumed = True
            v.preemptions += 1
            self.preemptions += 1
            self.preempt_log.append((self.step_no, v.id, len(v.generated)))
            self.waiting.insert(0, v)
            report["preempted"].append(v.id)

    def _decode(self, report, meter_of):
        for s in self.running:
            if self.need_next(s):
                s.table.append(self.pool.alloc())
            logits = run_rows(s, [s.next], meter_of("decode"))
            report["rows"] += 1
            s.logits.append(logits)
            s.generated.append(argmax(logits))
            s.next = s.generated[-1]

    def _finish(self, s, report):
        s.finished_at = self.step_no
        self.finished[s.id] = tuple(s.generated)
        self.finished_step[s.id] = self.step_no
        self._last_logits = s.logits
        self._release(s)
        report["finished"].append(s.id)

    def step(self, arrivals=()):
        report = {"step": self.step_no, "arrived": [], "refused": [], "admitted": [], "resumed": [], "preempted": [],
                  "finished": [], "rows": 0}
        for req, msg in arrivals:
            (report["refused"] if self.submit(req, msg) else report["arrived"]).append(req["id"])
        meters = {}

        def meter_of(kind):
            return meters.setdefault(kind, kc.Meter())
        try:
            self._admit(report, meter_of)
            self._ensure_room(report)
            self._decode(report, meter_of)
        except (PoolExhausted, PositionOverflow, EmptyPrefill, IndexError) as e:
            self.crash = (self.step_no, CRASHES[type(e).__name__])
            self.log.append(report)
            return False
        for s in [s for s in self.running if s.done()]:
            self.running.remove(s)
            self._finish(s, report)
        for kind, m in meters.items():
            self.flops[kind] += m.flops
        if report["rows"]:
            self.forwards += 1
        held = sum(len(s.table) for s in self.running)
        self.peak_blocks = max(self.peak_blocks, held)
        report["running"] = [s.id for s in self.running]
        report["free"] = self.free()
        report["tables"] = {s.id: list(s.table) for s in self.running}
        self._check_ledger()
        self.log.append(report)
        self.step_no += 1
        return True

    def busy(self):
        return bool(self.waiting or self.running)


def serve(requests, max_steps=400, **kw):
    """Run a server over a list of (request, message or None) until every request is answered or refused, it crashes, or
    `max_steps` steps pass. Returns the server."""
    srv = Server(**kw)
    pending = sorted(requests, key=lambda rm: (rm[0]["arrival"], rm[0]["id"]))
    k = 0
    while k < len(pending) or srv.busy():
        if srv.step_no >= max_steps:
            break
        now = []
        while k < len(pending) and pending[k][0]["arrival"] == srv.step_no:
            now.append(pending[k])
            k += 1
        if not srv.step(now):
            break
    return srv


def honest():
    return [(r, None) for r in workload()]


def hostile_messages():
    """Client 4's messages: (label, message, arrival step)."""
    ok = [8, 3, 7, 7, 7]
    return (
        ("a token id past the vocabulary", [8, 3, 7, 7, 12], 1),
        ("a negative token id", [8, 3, 7, -1, 7], 1),
        ("a budget of no tokens", [0, 3, 7, 7, 7], 2),
        ("a budget of a billion tokens", [10 ** 9, 3, 7, 7, 7], 2),
        ("a declared length of a million", [8, 10 ** 6, 7, 7, 7], 3),
        ("a prompt of no tokens", [8, 0], 3),
        ("one field", [8], 4),
        ("25 tokens", [8, 25] + [2] * 25, 4),
        ("20 tokens and a budget of 8", [8, 20] + [2] * 20, 5),
        ("a well-formed request", ok, 5),
    )


def _hostile_requests():
    out = []
    for k, (label, msg, at) in enumerate(hostile_messages()):
        toks = tuple(msg[2:]) if len(msg) >= 2 else ()
        req = {"id": 200 + k, "arrival": at, "client": 4, "prompt": toks, "max_new": msg[0] if msg else 0}
        out.append((req, list(msg)))
    return out


# ------------------------------------------------------------------ the worked blocks

_int = kc._int


def _toks(t):
    return " ".join(str(x) for x in t)


def _equal(srv, reqs=None):
    reqs = workload() if reqs is None else reqs
    return [r["id"] for r in reqs if srv.finished.get(r["id"]) == solo(r["prompt"], r["max_new"])[0]]


def _ledger(srv):
    return "yes" if srv.ledger_broken is None else f"no, from step {srv.ledger_broken}"


def _ending(srv):
    if srv.crash:
        return f"stopped in step {srv.crash[0]}: {srv.crash[1]}"
    if srv.busy():
        return f"still busy after step {srv.step_no - 1}"
    return f"every request answered or refused after {srv.step_no} steps"


MISTAKES = {
    "offset after the count": "a new key and value written at the slot of the count after it advances",
    "table kept after preemption": "a preempted sequence keeps its old block table",
    "admit on the prompt alone": "admission counts only the prompt's blocks and nothing is preempted",
    "rebuilt from the prompt alone": "a preempted sequence is rebuilt from its prompt alone",
    "rebuilt with the pending token": "a preempted sequence is rebuilt with its pending token",
}


@lru_cache(maxsize=None)
def _run(**kw):
    return serve(honest(), **kw)


def _conventions():
    return (f"The server has a pool of {NBLK} blocks of {BLK} positions, carries at most {ROWS} sequences in a step, ends a reply "
            f"after the token {STOP} or after max_new tokens, and admits by the rule of the text. ")


def opening():
    rows = []
    for label, kw in (("the specified server", {}), (MISTAKES["table kept after preemption"], {"mistake": "table kept after preemption"}),
                      (MISTAKES["admit on the prompt alone"], {"mistake": "admit on the prompt alone"})):
        srv = _run(**kw)
        rows.append([label, f"{len(srv.finished)} of {N_REQ}", f"{len(_equal(srv))} of {N_REQ}", _ledger(srv), _ending(srv)])
    return (f"The ten requests of the workload below, served three ways. " + _conventions() +
            "'Equal to alone' counts the replies that are token for token the reply of the same request served by itself on "
            "the dense cache of Volume V; 'ledger' is whether, after every step, the free blocks and the blocks in the tables of "
            "the running sequences are together each of the pool's blocks exactly once.\n\n"
            + _table(["build", "replies returned", "equal to alone", "ledger holds", "how the run ends"], rows, "lrrll"))


def requests():
    rows = []
    gaps = []
    for r in workload():
        reply, gap = solo(r["prompt"], r["max_new"])
        gaps.append(gap)
        rows.append([r["id"], r["arrival"], r["client"], _toks(r["prompt"]), r["max_new"], _toks(reply), f"{gap:.3f}"])
    return (f"Generator: the book's SplitMix64, seed {bookrng.seed_of(NAMES['workload'])} (`seed_of('{NAMES['workload']}')`), "
            f"one generator for the whole workload, never restarted. Draws, request by request for requests 0 to {N_REQ - 1}: "
            f"the gap to the arrival of the previous request, integer(3) (not drawn for request 0, which arrives at step 0); the "
            f"client, integer(3); the prompt length n = 3 + integer(6); the n tokens, integer(12) each, in order; the budget "
            f"max_new = 6 + integer(11). 'Reply alone' is the greedy reply of the request served by itself with the dense cache of "
            f"Volume V (`generate` of its Chapter 12 on the model of seed `seed_of('kv_cache.model')`), cut after the first token "
            f"{STOP} or after max_new tokens. 'Gap' is the smallest difference, over the tokens of that reply, between the largest "
            f"and the second-largest logit, printed to 3 decimals; the smallest over the workload is {min(gaps):.3f}.\n\n"
            + _table(["request", "arrives at step", "client", "prompt", "max_new", "reply alone", "gap"], rows, "rrrlrlr"))


def memory_bill():
    per_pos = 2 * L * D * FLOAT
    rows = [
        ["weights", _int(all_numbers()), _int(all_numbers() * FLOAT), "every number of the model, 8 bytes each"],
        ["one position of the cache", _int(2 * L * D), _int(per_pos), "a key and a value of D numbers in each of L layers"],
        ["one block", _int(2 * L * D * BLK), _int(per_pos * BLK), f"{BLK} positions"],
        ["the pool", _int(2 * L * D * BLK * NBLK), _int(per_pos * BLK * NBLK), f"{NBLK} blocks"],
        ["block tables", _int(ROWS * blocks_for(T)), _int(ROWS * blocks_for(T) * WORD),
         f"at most {blocks_for(T)} block ids for each of {ROWS} running sequences, 4 bytes each"],
        ["waiting messages", _int(Q_TOTAL * MAX_MSG // WORD), _int(Q_TOTAL * MAX_MSG),
         f"at most {Q_TOTAL} messages of at most {MAX_MSG} bytes"],
    ]
    return ("What the server holds, counted in numbers and in bytes. Cache and weights are 64-bit floats (8 bytes); block ids, "
            "tokens and the fields of a message are 32-bit integers (4 bytes). Replies being built and the bookkeeping of "
            "each sequence (its counters, its budget) are left out: they are bounded by the same limits, T tokens per sequence "
            f"and {ROWS} running sequences.\n\n"
            + _table(["what", "numbers", "bytes", "rule"], rows, "lrrl"))


def _alone(r, **kw):
    srv = Server(**kw)
    srv.step([(r, None)])
    while srv.busy() and srv.crash is None and srv.step_no < 100:
        srv.step()
    return srv


def paged_alone():
    rows = []
    for label, mistake in (("the specified block table", None), (MISTAKES["offset after the count"], "offset after the count")):
        equal, diff, worst = [], [], 0.0
        for r in workload():
            srv = _alone(r, mistake=mistake)
            ref, _ = solo(r["prompt"], r["max_new"])
            got = srv.finished.get(r["id"])
            _, steps = kc.generate_cached(model(), CFG, list(r["prompt"]), r["max_new"])
            s_logits = [st[3] for st in steps]
            seq_logits = srv._last_logits
            k = 0
            while k < min(len(ref), len(got)) and ref[k] == got[k]:
                worst = max(worst, max(abs(a - b) for a, b in zip(s_logits[k], seq_logits[k])))
                k += 1
            (equal if got == ref else diff).append(r["id"])
        rows.append([label, f"{len(equal)} of {N_REQ}", ", ".join(map(str, diff)) or "none",
                     "0 (the same bits)" if worst == 0.0 else f"{worst:.3g}"])
    return ("Each request of the workload served by itself on the paged cache (a fresh server, nothing else in it), against "
            "its reply on the dense cache. The largest difference is over every logit of every token up to the first token "
            "on which the two replies differ (all of them when the replies are equal).\n\n"
            + _table(["block table", "replies equal to the dense cache", "requests that differ",
                      "largest difference of a logit"], rows, "lrll"))


def _ids(xs):
    return " ".join(str(x) for x in xs) or "-"


def trace():
    srv = _run()
    rows = []
    for rep in srv.log:
        rows.append([rep["step"], _ids(rep["arrived"]), _ids(rep["admitted"]), _ids(rep["resumed"]), _ids(rep["preempted"]),
                     _ids(rep["finished"]), rep["rows"], _ids(rep["running"]), rep["free"]])
    return ("The specified server on the workload, step by step. In a step, in this order: the requests of the step arrive "
            "and are validated; waiting requests are admitted from the head of the queue while the rule allows, each prefilled "
            "at once ('first token' for a new request, 'rebuilt' for a preempted one, whose prefill only refills its cache); "
            "the newest running sequences are preempted until the pool can back one more position for every running sequence; "
            "every running sequence decodes one token; finished sequences give back their blocks. 'Rows' is the number of token "
            "rows the step runs through the model (prompt and rebuilt rows plus one per decoding sequence); 'running' and "
            "'free blocks' are after the step.\n\n"
            + _table(["step", "arrive", "first token", "rebuilt", "preempted", "finished", "rows", "running", "free blocks"],
                     rows, "rllllllll"))


def tables_by_step():
    """For the figure: [(step, {block: request id})] after every step of the specified server."""
    srv = _run()
    return [(rep["step"], {p: sid for sid, t in rep["tables"].items() for p in t}) for rep in srv.log]


def preemption():
    srv = _run()
    rebuilt = {sid: rep["step"] for rep in srv.log for sid in rep["resumed"]}
    rows1 = []
    for step, sid, n in srv.preempt_log:
        r = workload()[sid]
        rows1.append([sid, step, n, rebuilt[sid], f"{len(r['prompt'])} + {n - 1} = {len(r['prompt']) + n - 1}"])
    rows2 = []
    for label, kw in (("the specified rebuild", {}), (MISTAKES["rebuilt from the prompt alone"], {"mistake": "rebuilt from the prompt alone"}),
                      (MISTAKES["rebuilt with the pending token"], {"mistake": "rebuilt with the pending token"})):
        s = _run(**kw)
        bad = [r for r in workload() if s.finished.get(r["id"]) != solo(r["prompt"], r["max_new"])[0]]
        rows2.append([label, s.preemptions, f"{N_REQ - len(bad)} of {N_REQ}",
                      "; ".join(f"{r['id']}: {_toks(s.finished[r['id']])}" for r in bad) or "none"])
    return ("The preemptions of the specified server ('tokens' counts the tokens the sequence had generated when it was "
            "preempted; the rebuild prefills the prompt and all of them but the last, which is fed by the next decode step), and "
            "three ways of rebuilding a preempted sequence, each on the whole workload. 'Replies that differ' prints the reply "
            "the build returned; the replies alone are in the workload table.\n\n"
            + _table(["request", "preempted in step", "tokens", "rebuilt in step", "rows rebuilt"], rows1, "rrrrl")
            + "\n\n" + _table(["rebuild", "preemptions", "equal to alone", "replies that differ"], rows2, "lrrl"))


def _mean_wait(srv, reqs):
    """Mean over the requests of the steps from arrival to the step of the last token (both counted from 0)."""
    vals = []
    for r in reqs:
        f = srv.finished_step.get(r["id"])
        if f is None:
            return None
        vals.append(f - r["arrival"])
    return sum(vals) / len(vals)


def admission_rules():
    w = weight_numbers() * FLOAT
    rows = []
    for label, kw in (("one sequence at a time", {"rows": 1}),
                      ("reserve the whole budget at admission", {"admission": "reserve"}),
                      ("the prompt and its first position, preempt the newest", {"admission": "own"}),
                      ("promise the next position, preempt the oldest", {"victim": "oldest"}),
                      ("promise the next position, preempt the newest", {})):
        srv = _run(**kw)
        f = srv.flops
        rows.append([label, srv.forwards, _int(srv.forwards * w), _int(f["prefill"] + f["decode"]), _int(f["recompute"]),
                     srv.preemptions, srv.peak_blocks, f"{_mean_wait(srv, workload()):.1f}"])
    return (f"Five admission rules on the workload, each with the same pool of {NBLK} blocks. A forward pass is a step that runs "
            f"at least one row; it reads the {_int(weight_numbers())} numbers of the matrix products once, {_int(w)} bytes, "
            "whatever its number of rows (the rows of a step are one batch). FLOPs are counted as in Volume V, Chapter 12: two "
            "per multiply-add of every matrix product, the attention's included, and the head for the last row of each prefill "
            "and for each decode row; norms, softmax and the gate are not counted. 'Rebuilt' is the FLOPs of the prefills of "
            "preempted sequences. 'Wait' is the mean over the ten requests of the steps from arrival to the step of the last "
            "token.\n\n"
            + _table(["admission", "forward passes", "weight bytes", "FLOPs, first pass", "FLOPs rebuilt", "preemptions",
                      "peak blocks", "wait (steps)"], rows, "lrrrrrrr"))


def _fields(msg):
    if len(msg) < 2:
        return ", ".join(str(x) for x in msg)
    toks = msg[2:]
    if len(toks) > 4 and len(set(toks)) == 1:
        shown = f"{len(toks)} times {toks[0]}"
    else:
        shown = " ".join(str(t) for t in toks) or "none"
    return f"{msg[0]:,}; {msg[1]:,}; {shown}"


def messages():
    rows = []
    srv = Server()
    for label, msg, _ in hostile_messages():
        why = validate(msg, NBLK, 0, 0, ALL_CHECKS)
        rows.append([label, _fields(msg), WORD * len(msg), why or "accepted"])
    return (f"Client 4's messages, each checked by the specified server with an empty queue. A message is a list of 32-bit "
            f"fields: max_new, the declared prompt length n, then the tokens (printed as 'max_new; n; tokens'). The checks run "
            f"in this order and the first that fails is the answer: the message is at most {MAX_MSG} bytes (2 + T fields); it "
            f"has at least 2 fields; n equals the number of tokens; n is at least 1; every token is in 0 .. {V - 1}; max_new is "
            f"at least 1; n + max_new - 1 is at most the capacity, min(T, blocks x {BLK}) = {capacity(NBLK)} positions; the "
            f"client has fewer than {Q_CLIENT} requests waiting; fewer than {Q_TOTAL} requests are waiting in all.\n\n"
            + _table(["message", "fields", "bytes", "answer"], rows, "llrl"))


def _hostile(label):
    for req, msg in _hostile_requests():
        if hostile_messages()[req["id"] - 200][0] == label:
            return req, msg
    raise KeyError(label)


def unguarded():
    cases = (("token out of range", "a token id past the vocabulary"),
             ("token out of range", "a negative token id"),
             ("max_new below 1", "a budget of no tokens"),
             ("exceeds capacity", "a budget of a billion tokens"),
             ("exceeds capacity", "20 tokens and a budget of 8"),
             ("empty prompt", "a prompt of no tokens"))
    rows = []
    for check, label in cases:
        req, msg = _hostile(label)
        srv = serve(honest() + [(req, msg)], checks=ALL_CHECKS - {check})
        got = srv.finished.get(req["id"])
        what = _ending(srv) if srv.crash else f"answered: {_toks(got)}" if got is not None else _ending(srv)
        rows.append([check, label, f"{len(_equal(srv))} of {N_REQ}", what])
    stand_in = tuple(V - 1 if t == -1 else t for t in _hostile("a negative token id")[0]["prompt"])
    srv = serve(honest() + [_hostile("a negative token id")], checks=ALL_CHECKS - {"token out of range"})
    neg = srv.finished.get(200 + 1)
    same = neg == solo(stand_in, 8)[0]
    return ("The honest workload with one of client 4's messages added at its step, on the specified server with one check "
            "removed. 'Honest replies' counts the replies of requests 0 to 9 that are equal to their replies alone. A server "
            "that stops takes every reply in flight with it. The reply to the negative id is "
            f"{'the' if same else 'not the'} reply to the prompt {_toks(stand_in)}: this module is written in Python, whose "
            "list read at index -1 is the last entry, so the row read is row 11 of the embedding table; an implementation in a "
            "language without that rule reads memory before the table.\n\n"
            + _table(["check removed", "message", "honest replies", "what happens"], rows, "llrl"))


def amplification():
    per_pos = 2 * L * D * FLOAT
    rows = []
    for label, msg, _ in hostile_messages():
        if label in ("a declared length of a million", "a budget of a billion tokens"):
            n_pos = msg[1] if label.startswith("a declared") else msg[1] + msg[0] - 1
            rows.append([label, WORD * len(msg), _int(n_pos), _int(blocks_for(n_pos)), _int(n_pos * per_pos)])
    return (f"What a server would set aside if it sized a sequence's cache from the fields of the message instead of checking "
            f"them: the declared positions (n, or n + max_new - 1 for a budget), the blocks of {BLK} positions they need and "
            f"their bytes at {per_pos} bytes per position, against the bytes of the message.\n\n"
            + _table(["message", "message bytes", "positions", "blocks", "cache bytes"], rows, "lrrrr"))


def _flood_run(checks):
    return serve(honest() + [(r, None) for r in flood()], checks=checks, max_steps=2000)


def flood_table():
    rows = []
    for label, checks, with_flood in (("no flood", ALL_CHECKS, False),
                                      ("flood, both queue bounds", ALL_CHECKS, True),
                                      ("flood, the total bound alone", ALL_CHECKS - {"client queue full"}, True),
                                      ("flood, no queue bound", ALL_CHECKS - {"client queue full", "queue full"}, True)):
        srv = _flood_run(checks) if with_flood else _run()
        hon = [r for r in workload() if r["id"] in srv.finished]
        refused_h = sum(1 for r in workload() if r["id"] in srv.refused)
        fl_in = sum(1 for r in flood() if r["id"] not in srv.refused) if with_flood else 0
        wait = _mean_wait(srv, hon) if hon else None
        rows.append([label, f"{len(_equal(srv))} of {N_REQ}", refused_h, f"{fl_in} of {FLOOD}" if with_flood else "-",
                     srv.peak_waiting, "-" if wait is None else f"{wait:.1f}", srv.step_no])
    fr = flood()
    return (f"Client 3 sends {FLOOD} well-formed requests at step 0, just after request 0 (generator: the book's SplitMix64, "
            f"seed {bookrng.seed_of(NAMES['flood'])}, `seed_of('{NAMES['flood']}')`, one generator, never restarted; each request "
            f"drawn as an honest one is, the prompt length 3 + integer(6), its tokens integer(12), max_new 6 + integer(11); the "
            f"first is {_toks(fr[0]['prompt'])} with max_new {fr[0]['max_new']}). 'Honest replies' counts the replies of requests 0 to 9 "
            f"equal to their replies alone, 'honest refused' those refused at arrival, 'peak waiting' the most requests waiting "
            f"at once and 'wait' the mean over the answered honest requests of the steps from arrival to the last token.\n\n"
            + _table(["queue", "honest replies", "honest refused", "flood accepted", "peak waiting", "wait (steps)", "steps"],
                     rows, "lrrrrrr"))


def _whole(**kw):
    reqs = honest() + [(r, None) for r in flood()] + _hostile_requests()
    return serve(reqs, max_steps=2000, **kw)


def whole_run():
    ref = _whole()
    expect = dict(ref.refused)
    rows = []
    builds = [("the specified server", {})] + [(MISTAKES[k], {"mistake": k}) for k in
                                               ("offset after the count", "table kept after preemption", "admit on the prompt alone",
                                                "rebuilt from the prompt alone", "rebuilt with the pending token")]
    for label, kw in builds:
        srv = _whole(**kw)
        same_refusals = sum(1 for k, v in expect.items() if srv.refused.get(k) == v)
        rows.append([label, f"{len(_equal(srv))} of {N_REQ}", f"{same_refusals} of {len(expect)}", _ledger(srv), _ending(srv)])
    counts = {}
    for v in expect.values():
        counts[v] = counts.get(v, 0) + 1
    tally = "; ".join(f"{k} {n}" for k, n in sorted(counts.items(), key=lambda kv: REFUSALS.index(kv[0])))
    flood_answered = sum(1 for r in flood() if r["id"] in ref.finished)
    host_answered = [k for k in range(200, 200 + len(hostile_messages())) if k in ref.finished]
    return (f"Everything at once: the ten honest requests, client 3's flood of {FLOOD} at step 0 and client 4's ten messages "
            f"at their steps (the message table gives the step of each), on the specified server and with each planted "
            f"mistake. The specified server refuses {len(expect)} messages ({tally}), answers {flood_answered} requests of the "
            f"flood and {len(host_answered)} of client 4's, and its peak of waiting requests is {ref.peak_waiting}. 'Refusals "
            f"as specified' counts the refusals that a build gives with the same reason as the specified server.\n\n"
            + _table(["build", "honest replies", "refusals as specified", "ledger holds", "how the run ends"], rows, "lrrll"))
