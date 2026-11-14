# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Checkpoints, Determinism and Recovery".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: checkpoints_and_recovery.<name> -->` markers. Helpers with a leading underscore, or that take arguments,
are shared with the chapter's figures so a picture and the table beside it cannot disagree.

The two roundings, the counter-based draw and the epoch order are imported from `mixed_precision` and `training_data`
(never copied). Every number printed with full digits comes from IEEE double arithmetic restricted to addition,
subtraction, multiplication, division and square root (each correctly rounded) on draws that are exact (`uniform()` of the
book's generator, integer arithmetic of `mix32`), so another implementation reproduces it digit for digit. The numbers
that use the library `exp` are printed to four significant digits.

Swedish Embedded AB implements solutions for crash-safe, resumable training runs for its clients. If your team needs
expertise in checkpointing and exact resume of large training jobs then you can procure our services by sending an email
to info@swedishembedded.com.
"""
import hashlib
import itertools
import json
import math
import struct

import bookrng
import mixed_precision as mp
import number_formats as nf
import training_data as td

_table = nf._table


def _g(x, d=4):
    """d significant digits, plain for moderate sizes and scientific otherwise."""
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e6:
        return f"{float(f'{x:.{d}g}'):g}"
    return f"{x:.{d - 1}e}"


# ------------------------------------------------------------------ the arithmetic of losing a run to a failure

DEVICE_STEPS = 204_800     # mean device-steps between failures of one device (an assumption of the example)
RUN_STEPS = 100_000        # steps of the run
SAVE_COST = 8              # one checkpoint write costs as much as this many steps of work
GROUPS = (1, 64, 1024)     # devices


def mean_gap(devices):
    """M: mean steps between failures of a group of `devices` devices (failures independent, constant rate)."""
    return DEVICE_STEPS / devices


def from_scratch_steps(run_steps, gap):
    """Expected steps of work to finish a run of `run_steps` steps that restarts from step 0 after each failure: M (e^(S/M) - 1)."""
    return gap * (math.exp(run_steps / gap) - 1.0)


def slowdown(tau, cost, gap):
    """Expected steps of work per step of progress when a checkpoint costing `cost` steps is written every `tau` steps and
    a failure sends the run back to its last checkpoint: M (e^((tau + cost)/M) - 1) / tau."""
    return gap * (math.exp((tau + cost) / gap) - 1.0) / tau


def best_interval(cost, gap):
    """tau* = sqrt(2 c M), the minimum of the first-order waste c / tau + tau / (2 M)."""
    return math.sqrt(2.0 * cost * gap)


def failure_budget():
    rows = []
    for d in GROUPS:
        m = mean_gap(d)
        s = RUN_STEPS
        tau = round(best_interval(SAVE_COST, m))
        rows.append([f"{d:,}", _g(m, 6), _g(s / m, 4), _g(math.exp(-s / m), 4), _g(from_scratch_steps(s, m), 4), f"{tau:,}",
                     _g(math.sqrt(2.0 * SAVE_COST / m), 4), _g(slowdown(tau, SAVE_COST, m), 4)])
    head = (f"Assumptions of the example, not measurements: one device fails on average once in {DEVICE_STEPS:,} device-steps, "
            f"failures of different devices are independent and arrive at a constant rate, a run is S = {RUN_STEPS:,} steps, and one "
            f"checkpoint write costs c = {SAVE_COST} steps of work. For a group of D devices M = {DEVICE_STEPS:,} / D steps. "
            "The expected number of failures is S / M; the chance of none is e^(-S/M); a run that restarts from step 0 needs "
            "M (e^(S/M) - 1) steps on average; the interval tau* = sqrt(2 c M) (rounded to a whole step) minimises the first-order "
            "waste sqrt(2 c / M); the last column is the exact slowdown M (e^((tau + c)/M) - 1) / tau at that interval, in steps of "
            "work per step of progress. exp is the library function; numbers are to four significant digits.\n\n")
    heads = ["devices D", "M (steps)", "expected failures", "chance of none", "steps with no checkpoint", "tau* (steps)",
             "first-order waste", "slowdown at tau*"]
    return head + _table(heads, rows, "rrrrrrrr")


def _curve(devices, taus):
    m = mean_gap(devices)
    return [slowdown(t, SAVE_COST, m) for t in taus]


# ------------------------------------------------------------------ crash states of four ways to write a file

# An operation list is run against a small disk model. A name operation (create, rename) changes the directory; a data
# operation (write) fills one part of a file; fsync(f) makes the earlier writes of file f durable; syncdir makes the earlier
# name operations durable. After a crash, the disk holds:
#   process crash: every operation issued so far;
#   power loss: the name operations as a prefix of those issued (a journalling file system keeps their order), at least those
#   before the last completed syncdir, and any subset of the issued writes that contains those before a completed fsync.

def _states(ops, power_loss):
    """Every disk a crash can leave: a list of (ns, contents) built by replaying the persisted operations on the starting disk."""
    out = []
    for i in range(len(ops) + 1):
        issued = ops[:i]
        names = [o for o in issued if o[0] in ("create", "rename")]
        writes = [o for o in issued if o[0] == "write"]
        if not power_loss:
            name_counts = [len(names)]
            write_sets = [writes]
        else:
            last_sync = max([k for k, o in enumerate(issued) if o[0] == "syncdir"], default=None)
            floor = len([o for o in issued[:last_sync] if o[0] in ("create", "rename")]) if last_sync is not None else 0
            name_counts = range(floor, len(names) + 1)
            forced = set()
            for k, o in enumerate(issued):
                if o[0] == "fsync":
                    forced |= {w for w in writes if w[1] == o[1] and issued.index(w) < k}
            free = [w for w in writes if w not in forced]
            write_sets = [sorted(forced | set(c), key=writes.index) for n in range(len(free) + 1) for c in itertools.combinations(free, n)]
        for nc in name_counts:
            for ws in write_sets:
                out.append((nc, ws, names))
    return out


def _replay(start_ns, start_contents, names, nc, ws):
    ns = dict(start_ns)
    contents = {f: list(p) for f, p in start_contents.items()}
    for o in names[:nc]:
        if o[0] == "create":
            _, name, fid, parts = o
            ns[name] = fid
            contents[fid] = [None] * parts
        else:
            _, src, dst = o
            if src in ns:
                ns[dst] = ns.pop(src)
    for _, fid, k, label in ws:
        if fid in contents:
            contents[fid][k] = label
    return ns, contents


def _naive_reader(ns, contents):
    """A reader that trusts the file named T: its parts are all the old content, all the new, or a mixture."""
    if "T" not in ns:
        return "no file"
    parts = contents[ns["T"]]
    if all(p == "A" for p in parts):
        return "old"
    if all(p == "B" for p in parts):
        return "new"
    return "torn, accepted"


def _generation_reader(ns, contents):
    """A reader that takes the newest generation whose three parts are all present and equal to what its meta part would hash."""
    def good(name, tag):
        return name in ns and contents[ns[name]] == [tag + "_V", tag + "_S", tag + "_M"]
    if good("gen-2", "B"):
        return "new"
    if good("gen-1", "A"):
        return "old, damage found" if "gen-2" in ns else "old"
    return "no file"


_PROTOCOLS = [
    ("overwrite in place", "naive",
     [("create", "T", 1, 3), ("write", 1, 0, "B"), ("write", 1, 1, "B"), ("write", 1, 2, "B")]),
    ("temporary file, rename", "naive",
     [("create", "tmp", 1, 3), ("write", 1, 0, "B"), ("write", 1, 1, "B"), ("write", 1, 2, "B"), ("rename", "tmp", "T")]),
    ("temporary file, fsync, rename, directory fsync", "naive",
     [("create", "tmp", 1, 3), ("write", 1, 0, "B"), ("write", 1, 1, "B"), ("write", 1, 2, "B"), ("fsync", 1), ("rename", "tmp", "T"),
      ("syncdir",)]),
    ("generation, parts fsynced, meta last", "generation",
     [("create", "tmp", 1, 3), ("write", 1, 0, "B_V"), ("fsync", 1), ("write", 1, 1, "B_S"), ("fsync", 1), ("write", 1, 2, "B_M"),
      ("fsync", 1), ("syncdir",), ("rename", "tmp", "gen-2"), ("syncdir",)]),
    ("the same generation, no fsync", "generation",
     [("create", "tmp", 1, 3), ("write", 1, 0, "B_V"), ("write", 1, 1, "B_S"), ("write", 1, 2, "B_M"), ("rename", "tmp", "gen-2")]),
]
_OUTCOMES = ["old", "new", "old, damage found", "torn, accepted", "no file"]


def _tally(ops, reader, power_loss):
    if reader == "naive":
        start_ns, start_contents = {"T": 0}, {0: ["A", "A", "A"]}
        read = _naive_reader
    else:
        start_ns, start_contents = {"gen-1": 0}, {0: ["A_V", "A_S", "A_M"]}
        read = _generation_reader
    counts = {o: 0 for o in _OUTCOMES}
    total = 0
    for nc, ws, names in _states(ops, power_loss):
        ns, contents = _replay(start_ns, start_contents, names, nc, ws)
        counts[read(ns, contents)] += 1
        total += 1
    return total, counts


def _op_text(o):
    if o[0] == "create":
        return f"create {o[1]} (a name operation; for T it creates or truncates it)" if o[1] == "T" else f"create {o[1]}"
    if o[0] == "write":
        return f"write {o[3].split('_')[-1] if '_' in o[3] else 'part ' + str(o[2])}"
    if o[0] == "fsync":
        return "fsync the file"
    if o[0] == "rename":
        return f"rename {o[1]} to {o[2]}"
    return "syncdir"


def crash_states():
    rows = []
    lost = 0
    for name, reader, ops in _PROTOCOLS:
        for power in (False, True):
            total, c = _tally(ops, reader, power)
            lost += c["no file"]
            rows.append([name, "power loss" if power else "process crash", total] + [c[o] for o in _OUTCOMES[:4]])
    head = ("Each protocol replaces an old file (content A, three parts) by a new one (content B, three parts; a generation has "
            "parts V, S and M, with M, the meta part that carries the digests, last). The model: a name operation (create, rename) "
            "changes the directory, a write fills one part, `fsync` makes the earlier writes of its file durable and `syncdir` the "
            "earlier name operations. A crash can come after any prefix of the operations. In a process crash everything issued "
            "survives; in a power loss the name operations survive as a prefix in issue order, at least up to the last completed "
            "`syncdir`, and any subset of the writes survives that includes those before a completed `fsync` (a write that did "
            "not survive leaves its part empty). The reader of the first three protocols trusts the file `T`; the reader of the "
            "generations takes the newest whose three parts are all present, else the one before it. Every crash state of every "
            f"prefix is counted once, including the empty prefix and the full list. The disk before the first operation holds the old "
            "file A under the name `T` (first three protocols) or the old generation `gen-1` (last two); creating a name gives a file "
            "whose three parts are empty. The operations are, in issue order: "
            + "; ".join(f"{name}: " + ", ".join(_op_text(o) for o in ops) for name, _, ops in _PROTOCOLS)
            + f". The number of states in which the reader finds no file at all is {lost} in every row.\n\n")
    heads = ["protocol", "crash", "states", "old", "new", "old, damage found", "torn, accepted"]
    return head + _table(heads, rows, "llrrrrr")


# ------------------------------------------------------------------ a digest that notices a changed byte

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
M64 = (1 << 64) - 1


def fnv1a64(data):
    """FNV-1a, 64 bits: h = offset; for each byte b: h = ((h xor b) * prime) mod 2^64."""
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & M64
    return h


PAYLOAD_BYTES = 1000


def _payload():
    g = bookrng.generator("checkpoints_and_recovery.payload")
    return bytes(g.integer(256) for _ in range(PAYLOAD_BYTES))


def digest():
    vectors = [b"", b"a", b"foobar"]
    rows = [[repr(v.decode()), f"{fnv1a64(v):016x}"] for v in vectors]
    first = _table(["input", "FNV-1a 64"], rows, "ll")
    data = _payload()
    ref = fnv1a64(data)
    flips = same = 0
    for i in range(len(data)):
        for bit in range(8):
            flips += 1
            mutated = bytearray(data)
            mutated[i] ^= 1 << bit
            same += fnv1a64(mutated) == ref
    swaps = swap_same = 0
    for i in range(len(data) - 1):
        if data[i] != data[i + 1]:
            swaps += 1
            mutated = bytearray(data)
            mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
            swap_same += fnv1a64(mutated) == ref
    cuts = [(k, fnv1a64(data[:-k]) == ref) for k in range(1, 17)]
    zeros = fnv1a64(data + b"\x00") == ref
    rows = [["one bit flipped", flips, same, "no"],
            ["two different neighbouring bytes swapped", swaps, swap_same, "no"],
            ["the last k bytes cut off, k = 1 .. 16", len(cuts), sum(1 for _, s in cuts if s), "yes"],
            ["one zero byte appended", 1, int(zeros), "yes"]]
    seed = bookrng.seed_of("checkpoints_and_recovery.payload")
    sha = hashlib.sha256(b"weights-v1").hexdigest()
    head = (f"\n\nA payload of {PAYLOAD_BYTES} bytes, each `integer(256)` of the generator seeded {seed} (seed_of "
            "'checkpoints_and_recovery.payload'), in order; its FNV-1a digest is "
            f"{ref:016x}. Each change below is applied to the payload and its digest is compared with the original. 'Length "
            "changes' says whether the recorded length alone would have caught it.\n\n")
    second = _table(["change", "cases", "same digest", "length changes"], rows, "lrrl")
    tail = (f"\n\nFor comparison, the SHA-256 digest of the 10 bytes `weights-v1` (hashlib.sha256) is `{sha}`.")
    return ("Test vectors of the 64-bit FNV-1a hash (offset 14695981039346656037 = 0xcbf29ce484222325, prime 1099511628211 = "
            "0x100000001b3, bytes in order):\n\n" + first + head + second + tail)


# ------------------------------------------------------------------ the order of the keys in a header

_HEADER_KEYS = [("brain.config", '{"d_model":8}'), ("card.family", "toy"), ("card.id", "brain/toy"), ("card.kind", "lora")]


def key_order():
    seen = {}
    for order in itertools.permutations(_HEADER_KEYS):
        body = json.dumps({"__metadata__": dict(order)}, separators=(",", ":")).encode()
        seen.setdefault(hashlib.sha256(body).hexdigest(), body)
    canon = json.dumps({"__metadata__": dict(sorted(_HEADER_KEYS))}, separators=(",", ":")).encode()
    rows = [["each of the 4! = 24 orders of the keys", len(list(itertools.permutations(_HEADER_KEYS))), len(seen)],
            ["keys sorted by name", 1, 1]]
    head = ("Four metadata keys (`brain.config`, `card.family`, `card.id`, `card.kind`, with the values `{\"d_model\":8}`, `toy`, "
            "`brain/toy`, `lora`) are written as the JSON text `{\"__metadata__\": {...}}` without spaces, in every order, and "
            f"hashed with SHA-256 (hashlib). The sorted form is `{canon.decode()}`.\n\n")
    return head + _table(["header written", "headers", "distinct SHA-256 digests"], rows, "lrr")


# ------------------------------------------------------------------ a run, its state, and what resuming without a part does

DIM, N_EX, BATCH, STEPS, STOP = 4, 8, 2, 12, 5
W_STAR = (1.0, -2.0, 0.5, 3.0)
LR0, BETA1, BETA2, EPS, WD = 0.25, 0.9, 0.999, 1e-8, 0.01
M32 = 0xFFFFFFFF


def _data():
    g = bookrng.generator("checkpoints_and_recovery.data")
    X = [[2.0 * g.uniform() - 1.0 for _ in range(DIM)] for _ in range(N_EX)]
    y = []
    for x in X:
        s = 0.0
        for j in range(DIM):
            s += x[j] * W_STAR[j]
        y.append(s)
    return X, y


def _order_seed():
    return bookrng.seed_of("checkpoints_and_recovery.order")


def _rounding_seed32():
    return bookrng.seed_of("checkpoints_and_recovery.rounding") & M32


def _power(base, t):
    p = 1.0
    for _ in range(t):
        p *= base
    return p


def _rounding_key(seed32, t):
    """The key of the draws of update t: mix32(seed32 xor mix32(t))."""
    return mp.mix32((seed32 ^ mp.mix32(t)) & M32)


class _Run:
    """The state a step reads and the steps themselves. rounding is 'counter' (a draw per (seed, t, element)) or 'stream'
    (the next uniform() of one generator, in element order)."""

    def __init__(self, rounding="counter"):
        self.X, self.y = _data()
        self.rounding = rounding
        self.w = [0.0] * DIM
        self.m = [0.0] * DIM
        self.v = [0.0] * DIM
        self.t = 0          # updates applied
        self.draws = 0      # examples drawn so far
        self.seed32 = _rounding_seed32()
        self.gen = bookrng.generator("checkpoints_and_recovery.stream")

    def state(self):
        return {"w": list(self.w), "m": list(self.m), "v": list(self.v), "t": self.t, "draws": self.draws, "seed32": self.seed32,
                "gen": self.gen.state}

    def load(self, st):
        self.w, self.m, self.v = list(st["w"]), list(st["m"]), list(st["v"])
        self.t, self.draws, self.seed32 = st["t"], st["draws"], st["seed32"]
        self.gen = bookrng.SplitMix64(st["gen"])

    def _batch(self):
        seed = _order_seed()
        idx = [td.epoch_order(seed, N_EX, (self.draws + k) // N_EX)[(self.draws + k) % N_EX] for k in range(BATCH)]
        self.draws += BATCH
        return idx

    def step(self):
        idx = self._batch()
        err = []
        for i in idx:
            pred = 0.0
            for j in range(DIM):
                pred += self.X[i][j] * self.w[j]
            err.append(pred - self.y[i])
        g = []
        for j in range(DIM):
            s = 0.0
            for b, i in enumerate(idx):
                s += err[b] * self.X[i][j]
            g.append((2.0 / BATCH) * s)
        self.t += 1
        lr = LR0 * ((STEPS - (self.t - 1)) / STEPS)
        b1t, b2t = _power(BETA1, self.t), _power(BETA2, self.t)
        key = _rounding_key(self.seed32, self.t)
        for j in range(DIM):
            self.m[j] = BETA1 * self.m[j] + (1.0 - BETA1) * g[j]
            self.v[j] = BETA2 * self.v[j] + (1.0 - BETA2) * (g[j] * g[j])
            mhat = self.m[j] / (1.0 - b1t)
            vhat = self.v[j] / (1.0 - b2t)
            cand = (self.w[j] - (lr * WD) * self.w[j]) - (lr * mhat) / (math.sqrt(vhat) + EPS)
            u = mp.counter_uniform(key, j) if self.rounding == "counter" else self.gen.uniform()
            self.w[j] = mp.sr_bf16(cand, u)

    def run(self, n):
        for _ in range(n):
            self.step()
        return self

    def loss(self):
        total = 0.0
        for i in range(N_EX):
            pred = 0.0
            for j in range(DIM):
                pred += self.X[i][j] * self.w[j]
            total += (pred - self.y[i]) ** 2
        return total / N_EX


def _straight(rounding="counter"):
    return _Run(rounding).run(STEPS)


def _resume(rounding, edit):
    first = _Run(rounding).run(STOP)
    st = first.state()
    edit(st)
    second = _Run(rounding)
    second.load(st)
    return second.run(STEPS - STOP)


def _max_dev(a, b):
    return max(abs(x - y) for x, y in zip(a.w, b.w))


def _wtext(w):
    return "(" + ", ".join(repr(x) for x in w) + ")"


def _zero(*keys):
    def edit(st):
        for k in keys:
            st[k] = [0.0] * DIM if isinstance(st[k], list) else 0
    return edit


def _other_seed(st):
    st["seed32"] = bookrng.seed_of("checkpoints_and_recovery.other") & M32


def omitted_state():
    ref = _straight()
    variants = [
        ("everything restored", lambda st: None),
        ("first and second moments set to zero", _zero("m", "v")),
        ("update count set to 0 (schedule, bias correction and rounding key restart)", _zero("t")),
        ("examples drawn set to 0 (the data order restarts)", _zero("draws")),
        ("rounding seed replaced by another", _other_seed),
        ("weights only (moments, count and draws all 0)", _zero("m", "v", "t", "draws")),
    ]
    rows = []
    for name, edit in variants:
        res = _resume("counter", edit)
        same = res.w == ref.w
        rows.append([name, "yes" if same else "no", _g(_max_dev(res, ref), 4), _g(res.loss(), 4)])
    st5 = _Run("counter").run(STOP)
    head = (f"The run: {DIM} weights, all starting at 0, held as bf16 numbers; {N_EX} examples x_i (each coordinate 2u - 1, u the next "
            f"uniform() of the generator seeded {bookrng.seed_of('checkpoints_and_recovery.data')}, seed_of "
            "'checkpoints_and_recovery.data', examples in order, coordinates in order) with target y_i = x_i . w* "
            f"(w* = {W_STAR}, sum in coordinate order from 0.0). Each step draws {BATCH} examples (draw d is entry d mod {N_EX} of "
            f"`epoch_order(seed, {N_EX}, d // {N_EX})`, seed = {_order_seed():#x}, seed_of 'checkpoints_and_recovery.order'), "
            f"forms g = (2 / {BATCH}) sum_b (x_b . w - y_b) x_b (sum over the batch in draw order), and applies AdamW with "
            f"beta1 = {BETA1}, beta2 = {BETA2}, eps = {EPS}, weight decay {WD}, and rate lr_t = {LR0} ((12 - (t - 1)) / 12) "
            "after t applied updates: m = beta1 m + (1 - beta1) g, v = beta2 v + (1 - beta2)(g g), mhat = m / (1 - beta1^t), vhat "
            "= v / (1 - beta2^t) (the powers by repeated multiplication from 1.0), candidate = (w - (lr wd) w) - (lr mhat) / "
            "(sqrt(vhat) + eps). The candidate is rounded to bf16 stochastically with the draw counter_uniform(key, j) of "
            f"coordinate j, key = mix32(seed32 xor mix32(t)), seed32 = {_rounding_seed32():#x} (the low 32 bits of seed_of "
            "'checkpoints_and_recovery.rounding'; the row that replaces the seed uses the low 32 bits of seed_of "
            f"'checkpoints_and_recovery.other', {bookrng.seed_of('checkpoints_and_recovery.other') & M32:#x}). The run has 12 updates; it is stopped after "
            f"{STOP}, its state is saved, and a second process loads the saved state, with the part named in the first column "
            f"changed, and does the remaining {STEPS - STOP}. After 5 updates w = {_wtext(st5.w)}; after 12 in the uninterrupted "
            f"run w = {_wtext(ref.w)} with mean squared error {_g(ref.loss(), 4)}, the plain mean over the 8 examples of (x_i . w - y_i)^2, without a factor 1/2. 'Largest deviation' is the "
            "largest difference of a weight from the uninterrupted run after 12 updates.\n\n")
    return head + _table(["what the second process is given", "same weights as the uninterrupted run", "largest deviation",
                          "mean squared error"], rows, "lrrr")


# ------------------------------------------------------------------ the draw of a stochastic rounding and the resume

def _json_state(st):
    """The generator state written as a JSON number and read by a reader that parses numbers as 64-bit floats."""
    st["gen"] = int(float(st["gen"]))


def _fresh_stream(st):
    st["gen"] = bookrng.seed_of("checkpoints_and_recovery.stream")


def rounding_resume():
    rows = []
    cases = [
        ("counter draw keyed by (seed, t, element)", "counter", lambda st: None, "the seed and the update count"),
        ("stream draw, generator state restored", "stream", lambda st: None, "the generator's state, a u64"),
        ("stream draw, generator restarted from its seed", "stream", _fresh_stream, "nothing"),
        ("stream draw, state saved as a JSON number", "stream", _json_state, "the state rounded to 53 bits"),
    ]
    for name, mode, edit, kept in cases:
        ref = _straight(mode)
        res = _resume(mode, edit)
        rows.append([name, kept, "yes" if res.w == ref.w else "no", _g(_max_dev(res, ref), 4)])
    st = _Run("stream").run(STOP).state()
    s = st["gen"]
    head = (f"The run of the previous example, with the draw of the stochastic rounding either the counter draw there or the next "
            f"uniform() of one generator seeded {bookrng.seed_of('checkpoints_and_recovery.stream')} (seed_of "
            f"'checkpoints_and_recovery.stream'), drawn in coordinate order, {DIM} per update. In each row the run is stopped "
            f"after {STOP} updates and resumed with the saved state changed as the first column says; the run it is compared "
            "with is the uninterrupted run that rounds the same way. After 5 updates the generator's state is the u64 "
            f"{s}; read through a 64-bit float it comes back as {int(float(s))}.\n\n")
    return head + _table(["rounding draw and what the resume has", "kept from the saved state", "same weights as the uninterrupted run",
                          "largest deviation"], rows, "llrr")


# ------------------------------------------------------------------ a rate written as text is not the rate

def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_not_decimals():
    rows = []
    for text in ("0.5", "0.001", "0.0003", "0.1", "1e-5"):
        x = _f32(float(text))
        shortest = next(f"{x:.{d}g}" for d in range(1, 10) if _f32(float(f"{x:.{d}g}")) == x)
        rows.append([text, f"{_f32_bits(x):#010x}", repr(x), repr(float(shortest)), "yes" if float(shortest) == x else "no"])
    head = ("Each rate is rounded to the nearest 32-bit float (struct 'f'); its bit pattern is printed, then the 32-bit value as an "
            "exact 64-bit float, then the decimal text of the fewest digits that read back to the same 32-bit float, parsed as a "
            "64-bit float (what a JSON reader gives). The last column asks whether the two 64-bit numbers are equal.\n\n")
    return head + _table(["rate as typed", "float32 bits", "the float32 as a float64", "its shortest decimal text, as a float64",
                          "equal"], rows, "lllll")
