# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Running and Observing an Engine".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: operations.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's figures, so a
picture and the table beside it cannot disagree.

Everything here is integer arithmetic, string matching and the stepping of small state machines over scripted inputs, so
another implementation reproduces every printed value. No generator is used.

Swedish Embedded AB implements solutions for observing, controlling and cleanly stopping long-running inference servers for its
clients. If your team needs expertise in the operation of machine learning services then you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math

import number_formats as nf

_table = nf._table


def _n(v):
    return f"{v:,}"


# ------------------------------------------------------------------ how much a trace writes

LEVELS = ["error", "warn", "info", "debug", "trace"]   # severity 1 .. 5; a filter at level L admits severities 1 .. L
LINE_BYTES = 120                                       # assumed bytes of one rendered line, an assumption of the example

VOLUME_RUN = {"requests": 8, "tokens": 512}


def _volume_events(requests, tokens):
    """Events of the scripted run at each severity (1 = error ... 5 = trace): one batch of `requests` conversations decoded
    together for `tokens` steps. Per request two info lines (admitted, finished); per decode step one debug line (the batch
    the step ran); per token of each request one trace line (the token sampled); one warn line for the whole run."""
    return {1: 0, 2: 1, 3: 2 * requests, 4: tokens, 5: requests * tokens}


def trace_volume():
    r, t = VOLUME_RUN["requests"], VOLUME_RUN["tokens"]
    ev = _volume_events(r, t)
    rows = []
    generated = r * t
    for level in range(0, 6):
        lines = sum(ev[s] for s in range(1, level + 1))
        name = "off" if level == 0 else LEVELS[level - 1]
        rows.append([level, name, _n(lines), _n(lines * LINE_BYTES), f"{lines / generated:.4f}"])
    head = (f"A scripted run: {r} conversations decoded together as one batch for {t} decode steps, {_n(generated)} tokens in "
            f"all. It emits, by severity: {ev[1]} error lines; {ev[2]} warn line (one fallback for the whole run); "
            f"{ev[3]} info lines (each conversation admitted and finished); {ev[4]} debug lines (one per decode step, "
            f"naming the batch it ran); {_n(ev[5])} trace lines (one per token of each conversation). A filter at level L "
            f"writes every line of severity 1 to L. Bytes assume {LINE_BYTES} bytes per rendered line, an assumption of the "
            "example; the last column divides the lines written by the tokens generated, to four decimals.\n\n")
    return head + _table(["level", "name", "lines written", "bytes written", "lines per token"], rows)


# ------------------------------------------------------------------ families, flags and the filter

FAMILIES = [("gpu", ["gpu_core", "backend_wgpu", "backend_vulkan", "vulkan"]),
            ("ltxv", ["ltxv", "gemma4"])]

EVENTS = [("E1", "ltxv::pipeline", 1, "boom"),
          ("E2", "ltxv::pipeline", 2, "degraded"),
          ("E3", "ltxv::pipeline", 3, "started"),
          ("E4", "ltxv::dit", 4, "cache miss"),
          ("E5", "ltxv::dit", 5, "per-iteration detail"),
          ("E6", "gpu_core::devices", 1, "adapter enumeration failed"),
          ("E7", "backend_vulkan", 4, "queue submitted"),
          ("E8", "backend_cuda::exec", 3, "graph launched")]


class TraceError(Exception):
    pass


def _family(name):
    for f, targets in FAMILIES:
        if f == name:
            return targets
    raise TraceError(f"unknown family {name!r}")


def _level(s):
    if not s.isdigit() or int(s) > 5:
        raise TraceError(f"level {s!r} is not 0-5")
    return int(s)


def strip_trace_args(argv, env=None):
    """The flag grammar of the chapter: returns (pairs, rest). `--trace-<family> L`, `--trace <family>=L`, `--trace-format`
    and `--trace-output` take the next token as their value; every other token is handed back in order. When no flag named
    a family, the comma-separated pairs of the environment variable are used instead."""
    pairs, rest, from_flags, i = [], [], False, 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--trace-format", "--trace-output", "--trace") or tok.startswith("--trace-"):
            if i + 1 >= len(argv):
                raise TraceError(f"{tok} needs a value")
            val = argv[i + 1]
            if tok == "--trace-format":
                if val not in ("text", "json"):
                    raise TraceError("format must be text or json")
            elif tok == "--trace-output":
                pass
            elif tok == "--trace":
                if "=" not in val:
                    raise TraceError("--trace expects <family>=<level>")
                name, lv = val.split("=", 1)
                _family(name)
                pairs.append((name, _level(lv)))
                from_flags = True
            else:
                name = tok[len("--trace-"):]
                _family(name)
                pairs.append((name, _level(val)))
                from_flags = True
            i += 2
        else:
            rest.append(tok)
            i += 1
    if not from_flags and env:
        for part in [p.strip() for p in env.split(",") if p.strip()]:
            if "=" not in part:
                raise TraceError("BRAIN_TRACE expects <family>=<level>")
            name, lv = part.split("=", 1)
            _family(name)
            pairs.append((name, _level(lv)))
    return pairs, rest


def level_of(pairs, name):
    for f, lv in reversed(pairs):
        if f == name:
            return lv
    return 0


def directives(pairs):
    out = []
    for name, targets in FAMILIES:
        lv = level_of(pairs, name)
        if lv > 0:
            out += [(t, lv) for t in targets]
    return out


def trace_admits(directive_list, target, severity):
    """The filter: everything is off unless a directive turns it on. Of the directives whose target is a prefix of the
    event's target (a prefix of the text, character by character), the one with the longest target decides; the event is
    written when its severity is at most that directive's level."""
    best, best_len = 0, -1
    for t, lv in directive_list:
        if target.startswith(t) and len(t) > best_len:
            best, best_len = lv, len(t)
    return severity <= best


TRACE_CASES = [([], None),
               (["--trace-ltxv", "3"], None),
               (["--trace-ltxv", "1", "--trace", "ltxv=5"], None),
               (["--trace-gpu", "2", "--trace-format", "json"], None),
               (["--trace-ltxv", "4", "--trace-gpu", "1"], None),
               ([], "gpu=5,ltxv=1"),
               (["--trace-ltxv", "3"], "gpu=5"),
               (["--trace-ltxv", "0"], None),
               (["--trace-ltxv", "9"], None),
               (["--trace-quen", "5"], None),
               (["--trace", "ltxv"], None)]


def trace_filter():
    ev_rows = [[e[0], f"`{e[1]}`", f"{e[2]} ({LEVELS[e[2] - 1]})", e[3]] for e in EVENTS]
    fam = "; ".join(f"`{n}` covers " + ", ".join(f"`{t}`" for t in ts) for n, ts in FAMILIES)
    head = (f"The families: {fam}. The eight events below are emitted once each, in this order; a target is the module "
            "path of the code that emitted it. The configurations are the flags of one command line (after the program "
            "name) and the value of BRAIN_TRACE, if set. A level of 0 for every family installs no subscriber; a "
            "refused command line is an error before anything runs.\n\n")
    t1 = _table(["event", "target", "severity", "message"], ev_rows, "llll")
    rows = []
    for argv, env in TRACE_CASES:
        args = " ".join(argv) if argv else "(none)"
        try:
            pairs, _ = strip_trace_args(["brain", "run"] + argv, env)
        except TraceError as e:
            rows.append([f"`{args}`", f"`{env}`" if env else "unset", f"refused: {e}", "", ""])
            continue
        lv = ", ".join(f"{n}={level_of(pairs, n)}" for n, _ in FAMILIES)
        ds = directives(pairs)
        if not ds:
            rows.append([f"`{args}`", f"`{env}`" if env else "unset", lv, "none: no subscriber", "none"])
            continue
        shown = [e[0] for e in EVENTS if trace_admits(ds, e[1], e[2])]
        dtext = ", ".join(f"{t}={LEVELS[l - 1]}" for t, l in ds)
        rows.append([f"`{args}`", f"`{env}`" if env else "unset", lv, dtext, " ".join(shown) if shown else "none"])
    t2 = _table(["flags", "BRAIN_TRACE", "levels", "directives", "events written"], rows, "lllll")
    return head + t1 + "\n\n" + t2


# ------------------------------------------------------------------ counters of a scheduler

COUNTER_RUN = {"max_batch": 2, "steps_per_batch": 3, "arrivals": {0: 1, 1: 3, 4: 1, 8: 2}, "steps": 14}


def counter_series():
    """Step a toy scheduler with the executor's counter rules and return one snapshot per step. One lane runs at most one
    batch at a time. At step t, in this order: (1) a batch admitted at step t - d completes: jobs += its size, batches += 1;
    (2) the jobs arriving at step t join the back of the queue, each raising queue_peak to the queue's length if that is
    larger; (3) an idle lane admits up to max_batch jobs from the front: admitted += their number. queue_depth is the
    queue's length at the end of the step, when the snapshot is taken."""
    b, d = COUNTER_RUN["max_batch"], COUNTER_RUN["steps_per_batch"]
    queue, running = 0, None          # running = (size, step it completes at)
    s = {"admitted": 0, "jobs": 0, "batches": 0, "queue_peak": 0, "queue_depth": 0, "max_batch": 0}
    out = []
    for t in range(COUNTER_RUN["steps"]):
        if running and running[1] == t:
            s["jobs"] += running[0]
            s["batches"] += 1
            s["max_batch"] = max(s["max_batch"], running[0])
            running = None
        arrive = COUNTER_RUN["arrivals"].get(t, 0)
        for _ in range(arrive):
            queue += 1
            s["queue_peak"] = max(s["queue_peak"], queue)
        if running is None and queue > 0:
            n = min(b, queue)
            queue -= n
            s["admitted"] += n
            running = (n, t + d)
        s["queue_depth"] = queue
        out.append(dict(s, step=t, arrive=arrive, running=running[0] if running else 0))
    return out


def counters():
    run = COUNTER_RUN
    arr = ", ".join(f"{n} at step {t}" for t, n in sorted(run["arrivals"].items()))
    head = (f"One lane, batches of at most {run['max_batch']} jobs of the same model, each batch running for "
            f"{run['steps_per_batch']} steps; jobs arrive {arr}. Steps are counted from 0. Within a step: a batch that has "
            "run its steps completes first (jobs and batches count it), then the step's arrivals join the queue (queue_peak "
            "is raised by each arrival), then an idle lane admits from the front of the queue (admitted counts the jobs). "
            "The snapshot is taken at the end of the step; running is the size of the batch on the lane, and "
            "admitted - jobs equals it in every row.\n\n")
    rows = [[r["step"], r["arrive"], r["queue_depth"], r["queue_peak"], r["admitted"], r["jobs"], r["batches"], r["running"]]
            for r in counter_series()]
    return head + _table(["step", "arrivals", "queue_depth", "queue_peak", "admitted", "jobs", "batches", "running"], rows)


# ------------------------------------------------------------------ flattening a snapshot for scripts

def _flatten_value(prefix, v, out):
    if isinstance(v, dict):
        for k in sorted(v):
            _flatten_value(f"{prefix}.{k}", v[k], out)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _flatten_value(f"{prefix}.{i}", x, out)
    elif isinstance(v, bool):
        out.append(f"{prefix}={'true' if v else 'false'}")
    elif v is None:
        out.append(f"{prefix}=null")
    else:
        out.append(f"{prefix}={v}")


def flatten_snapshot(snap):
    """The flattening rule of the chapter: collections keyed by their id, typed fields in a fixed order, every `extra` map
    flattened by sorted key with nested objects dotted and arrays indexed from 0."""
    out = [f"schema={snap['schema']}"]
    for a in snap["accelerators"]:
        p = f"accelerator.{a['id']}"
        out += [f"{p}.kind={a['kind']}", f"{p}.mem_total={a['mem_total']}", f"{p}.mem_used={a['mem_used']}"]
        for k in sorted(a.get("extra", {})):
            _flatten_value(f"{p}.extra.{k}", a["extra"][k], out)
    for m in snap["models"]:
        p = f"model.{m['id']}"
        out.append(f"{p}.resident={'true' if m['instances'] else 'false'}")
        for inst in m["instances"]:
            ip = f"{p}.instances.{inst['device']}"
            out += [f"{ip}.tier={inst['tier']}", f"{ip}.mem={inst['mem']}"]
            for k in sorted(inst.get("extra", {})):
                _flatten_value(f"{ip}.extra.{k}", inst["extra"][k], out)
    for k in sorted(snap.get("extra", {})):
        _flatten_value(f"extra.{k}", snap["extra"][k], out)
    return out


GIB = 1 << 30

SNAP_A = {"schema": 1,
          "accelerators": [{"id": "cpu", "kind": "cpu", "mem_total": 16 * GIB, "mem_used": 2 * GIB},
                           {"id": "gpu0", "kind": "gpu", "mem_total": 24 * GIB, "mem_used": 10 * GIB,
                            "extra": {"gpu_freq_mhz": 1650, "gpu_throttled_thermal": False}}],
          "models": [{"id": "qwen", "instances": [{"device": "gpu0", "tier": "hot", "mem": 8 * GIB,
                                                    "extra": {"replica": 0, "queued": 3, "running": 2}}]},
                     {"id": "tts", "instances": []}],
          "extra": {"build": {"rev": "abc"}, "lanes": [2, 1]}}


def flatten():
    snap_b = dict(SNAP_A, models=list(reversed(SNAP_A["models"])), accelerators=list(reversed(SNAP_A["accelerators"])))
    a, b = flatten_snapshot(SNAP_A), flatten_snapshot(snap_b)
    same = sorted(a) == sorted(b)
    head = ("A snapshot with two accelerators, a resident model `qwen` with one instance and a cold model `tts`, an `extra` "
            "map on the GPU, on the instance and at the top (one nested object and one array). The second snapshot holds the "
            "same data with both collections listed in the opposite order. GiB = 2^30 bytes; memory is printed in bytes.\n\n")
    body = "```text\n" + "\n".join(a) + "\n```\n\n"
    tail = (f"The first snapshot gives {len(a)} lines and the reordered one {len(b)}; as sets they are "
            f"{'equal' if same else 'different'}, and only the order of the lines differs.")
    return head + body + tail


# ------------------------------------------------------------------ percentiles, a tail and fairness

WAITS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 40]


def nearest_rank(sorted_xs, q):
    n = len(sorted_xs)
    rank = min(max(math.ceil(q * n), 1), n)
    return sorted_xs[rank - 1]


def _r3(x):
    """Rounding to 3 decimals, half away from zero."""
    s = -1 if x < 0 else 1
    return s * math.floor(abs(x) * 1000 + 0.5) / 1000


def _summary(xs):
    s = sorted(xs)
    return {"n": len(s), "p50": nearest_rank(s, 0.50), "p95": nearest_rank(s, 0.95), "p99": nearest_rank(s, 0.99),
            "p999": nearest_rank(s, 0.999), "mean": _r3(sum(s) / len(s)), "min": s[0], "max": s[-1]}


def jain(xs):
    sq = sum(x * x for x in xs)
    if sq <= 0:
        return 1.0
    return _r3(sum(xs) ** 2 / (len(xs) * sq))


def percentiles():
    samples = [("the 20 waits", WAITS),
               ("999 waits of 1, one of 10,000", [1] * 999 + [10000]),
               ("990 waits of 1, ten of 10,000", [1] * 990 + [10000] * 10)]
    rows = []
    for name, xs in samples:
        s = _summary(xs)
        rows.append([name, s["n"], s["p50"], s["p95"], s["p99"], s["p999"], s["mean"], s["min"], s["max"],
                     f"{sum(1 for x in xs if x <= 10) / len(xs):g}"])
    head = ("Waits are counted in scheduler steps, the number of steps a request spent queued. The 20 waits in the order "
            f"recorded: {', '.join(str(w) for w in WAITS)}; sorted: {', '.join(str(w) for w in sorted(WAITS))}. The "
            "percentile at q is the nearest-rank value: the sample of rank ceil(q n) in ascending order, ranks counted from "
            "1 and clamped to 1 .. n. The mean divides the sum by n and is rounded to 3 decimals, half away from zero. The "
            "last column is the fraction of the waits that are at most 10 steps.\n\n")
    t1 = _table(["sample", "n", "p50", "p95", "p99", "p999", "mean", "min", "max", "share <= 10"], rows)
    fair = [[", ".join(_n(x) for x in xs), jain(xs)] for xs in ([5, 5, 5], [6, 3, 1], [10, 0, 0], [7, 7, 0, 0])]
    t2 = ("\n\nJain's index $(\\sum x)^2 / (n \\sum x^2)$ of the tokens served to each class, rounded to 3 decimals half away "
          "from zero:\n\n" + _table(["tokens served per class", "fairness index"], fair))
    return head + t1 + t2


# ------------------------------------------------------------------ the controller's state machine

PARENT = {"Root": None, "Operational": "Root", "Faulted": "Root",
          "Idle": "Operational", "Chatting": "Operational", "Detecting": "Operational", "Synthesizing": "Operational",
          "Forecasting": "Operational", "Backtesting": "Operational", "Cancelled": "Operational"}


def path_to_root(s):
    out = [s]
    while PARENT[out[-1]] is not None:
        out.append(PARENT[out[-1]])
    return out


def lca(a, b):
    anc = path_to_root(a)
    for c in path_to_root(b):
        if c in anc:
            return c
    return None


class Controller:
    """The controller of the chapter over the engine of the chapter. `script` is the text the stand-in language model
    produces, one character per token, then its end token. `fixed` chooses how a detection completes: True requests the
    return to Idle, False requests another Tick (the defect of the history the chapter tells)."""

    def __init__(self, script="hi", fixed=True, infer=True):
        self.state, self.queue, self.log, self.out = "Idle", [], [], []
        self.script, self.fixed, self.infer = script, fixed, infer
        self.pump = None
        self.seq = 0
        self.pending_tick = self.want_idle = False

    # --- the machine's behaviour
    def dispatch(self, s, ev):
        if s == "Idle":
            if ev == "user_text":
                return ("tran", "Chatting")
            if ev == "camera_frame":
                return ("tran", "Detecting")
            return ("unhandled",)
        if s == "Chatting":
            if ev == "Tick":
                self.pump_one_token()
                return ("handled",)
            if ev == "GoIdle":
                return ("tran", "Idle")
            if ev in ("user_text", "camera_frame"):
                return ("handled",)
            return ("unhandled",)
        if s in ("Detecting", "Cancelled"):
            if ev == "GoIdle":
                return ("tran", "Idle")
            if ev == "Tick":
                return ("handled",)
            return ("unhandled",)
        if s == "Operational":
            if ev == "cancel":
                return ("tran", "Cancelled")
            if ev == "error":
                return ("tran", "Faulted")
            return ("unhandled",)
        if s == "Faulted":
            return ("handled",)
        return ("unhandled",)

    def on_entry(self, s):
        if s == "Chatting":
            self.pump = list(self.script) if self.infer else None
            self.seq = 0
            self.pending_tick = True
        elif s == "Detecting":
            self.out.append("object_detected")
            if self.fixed:
                self.want_idle = True
            else:
                self.pending_tick = True
        elif s == "Cancelled":
            self.out.append("cancelled")
            self.want_idle = True
        elif s == "Faulted":
            self.out.append("error")

    def on_exit(self, s):
        if s == "Chatting":
            self.pump = None

    def pump_one_token(self):
        if self.pump:
            ch = self.pump.pop(0)
            self.out.append(f"chunk '{ch}' seq {self.seq}")
            self.seq += 1
            self.pending_tick = True
        else:
            self.out.append(f"chunk done seq {self.seq}")
            self.want_idle = True

    # --- the engine
    def step(self, ev):
        before = len(self.out)
        source = self.state
        while True:
            d = self.dispatch(source, ev)
            if d[0] == "handled":
                self.log.append((ev, source, "", "", "", self.out[before:], self.state))
                return
            if d[0] == "unhandled":
                if PARENT[source] is None:
                    self.log.append((ev, "nobody (ignored)", "", "", "", self.out[before:], self.state))
                    return
                source = PARENT[source]
                continue
            target = d[1]
            break
        top = lca(source, target)
        exited, s = [], self.state
        while s != top:
            self.on_exit(s)
            exited.append(s)
            if PARENT[s] is None:
                break
            s = PARENT[s]
        path, t = [], target
        while t != top:
            path.append(t)
            if PARENT[t] is None:
                break
            t = PARENT[t]
        entered = []
        for st in reversed(path):
            self.on_entry(st)
            entered.append(st)
        self.state = target
        self.log.append((ev, source, f"{target} (LCA {top})", ", ".join(exited), ", ".join(entered), self.out[before:],
                         self.state))

    def run(self):
        while self.queue:
            self.step(self.queue.pop(0))

    def turn(self, ev, control=None):
        """One external event, run to completion: the loop of `controller_turn` in the chapter."""
        self.queue.append(ev)
        for _ in range(100000):
            self.run()
            if self.want_idle:
                self.want_idle = self.pending_tick = False
                self.queue.append("GoIdle")
                continue
            if self.pending_tick:
                self.pending_tick = False
                c = control() if control else None
                if c is not None:
                    self.queue.append(c)
                    continue
                self.queue.append("Tick")
                continue
            break


def _cancel_after(n):
    left = [n]

    def poll():
        if left[0] == 0:
            return "cancel"
        left[0] -= 1
        return None
    return poll


HSM_TURNS = [("user_text", None, "none"),
             ("camera_frame", None, "none"),
             ("user_text", 1, "cancel after 1 chunk"),
             ("cancel", None, "none"),
             ("error", None, "none"),
             ("user_text", None, "none")]


def hsm_trace():
    c = Controller(script="hi")
    rows = []
    for k, (ev, n, ctl_name) in enumerate(HSM_TURNS, 1):
        c.log = []
        c.turn(ev, _cancel_after(n) if n is not None else None)
        for i, (e, by, tran, ex, en, em, after) in enumerate(c.log):
            rows.append([k if i == 0 else "", f"`{e}`", by, tran, ex, en, "; ".join(em), after])
    head = ("The controller starts in Idle (its entry chain Root, Operational, Idle has run). The language model is a "
            "stand-in that produces the text `hi`, one character per token, and then its end token; the detector returns "
            "one box. Each turn posts one external event and runs to completion; the third turn polls a control source "
            "before every Tick that answers nothing once and `cancel` after that. One row per event the engine dispatches, "
            "in order: the state whose handler took it (after bubbling up from the active state), the transition and its "
            "least common ancestor, the states exited and entered in order, and what was emitted.\n\n")
    return head + _table(["turn", "event", "handled by", "transition to", "exited", "entered", "emitted", "state after"],
                         rows, "llllllll")


def stuck_detector():
    rows = []
    for name, fixed in (("completion asks for a Tick", False), ("completion asks for GoIdle", True)):
        c = Controller(fixed=fixed)
        for f in (1, 2, 3):
            c.log, before = [], len(c.out)
            c.turn("camera_frame")
            rows.append([name if f == 1 else "", f, "; ".join(c.out[before:]) or "nothing", len(c.log), c.state])
    head = ("Three camera frames in a row, each its own turn, under the two ways a detection can complete. The detection "
            "itself runs in the entry action of Detecting. 'Dispatches' counts the events the engine took from its queue "
            "during the turn.\n\n")
    return head + _table(["completion", "frame", "emitted", "dispatches", "state after"], rows, "lrlrl")


# ------------------------------------------------------------------ the drain after a shutdown signal

DRAIN_BOUND = 3            # seconds: the HTTP surface's drain bound in the example (the code's SHUTDOWN_DRAIN)
REQUESTS = [("A", "generation, 120 decode steps of 0.1 left", 120, 0.1, True),
            ("B", "generation, 30 decode steps of 0.4 left", 30, 0.4, True),
            ("C", "cold model build, 20 left, cannot be interrupted", 1, 20.0, False),
            ("D", "idle keep-alive connection", 0, 0.0, True)]


def _end(req, cancel, bound):
    _, _, steps, step, interruptible = req
    finish = round(steps * step, 6)
    if cancel and interruptible:
        end, how = min(step, finish), ("cancelled" if steps else "closed")
    else:
        end, how = finish, ("finished" if steps else "closed")
    if bound is not None and end > bound:
        return bound, "abandoned"
    return end, how


def drain():
    policies = [("wait, no cancel", False, None), ("cancel, no bound", True, None),
                ("bound only", False, DRAIN_BOUND), ("cancel and bound", True, DRAIN_BOUND)]
    rows = []
    for rid, desc, *_ in REQUESTS:
        row = [rid, desc]
        for _, cancel, bound in policies:
            end, how = _end(next(r for r in REQUESTS if r[0] == rid), cancel, bound)
            row.append(f"{how} at {end:g}")
        rows.append(row)
    total = ["surface", "drained at"]
    for _, cancel, bound in policies:
        total.append(f"{max(_end(r, cancel, bound)[0] for r in REQUESTS):g}")
    rows.append(total)
    head = ("Four connections are open when the shutdown signal fires, at time 0 (times in seconds; the step lengths are "
            "an example, not a measurement). The listener stops accepting at once. A cancelled generation ends at the end of "
            "the decode step it is in, where it next looks at its cancellation flag; an idle connection is closed at once; "
            f"a step that cannot be interrupted runs to its end. With a bound, whatever is still running at {DRAIN_BOUND} "
            "is abandoned. The last row is the time the surface has drained: the latest end.\n\n")
    return head + _table(["", "open at the signal"] + [p[0] for p in policies], rows, "llllll")


# ------------------------------------------------------------------ the readiness marker and its probe

def ready_probe():
    events = ["the previous run's marker is on disk", "construction removes a stale marker", "writability probe: create",
              "writability probe: delete", "surface 1 bound (1 of 2)", "surface 2 bound (2 of 2): marker created"]

    def exists(k, probe_on_path):
        if k == 0:
            return True
        if k == 2:
            return probe_on_path
        return k == 5

    rows = []
    for k, e in enumerate(events):
        bound = 0 if k < 4 else k - 3
        old, new = exists(k, True), exists(k, False)
        mark = lambda x: ("yes" + (" (stale)" if k == 0 else "") + (" (false)" if x and bound < 2 and k > 0 else "")) if x else "no"
        rows.append([k, e, bound, mark(old), mark(new)])
    head = ("A server with two surfaces starts with a readiness marker left behind by a previous run. A waiter checks "
            "whether the marker path exists after each step of the start; 'false' marks a yes while fewer than both "
            "surfaces are bound. The writability probe either creates and deletes the marker path itself or a sibling "
            "name beside it.\n\n")
    return head + _table(["after", "step of the start", "surfaces bound", "path exists, probe on the path",
                          "path exists, probe on a sibling"], rows, "rlrll")

