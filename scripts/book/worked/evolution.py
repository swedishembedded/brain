# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "How brain Grew".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: evolution.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's figures so
a picture and the table beside it cannot disagree.

Every number here is a closed-form expectation or a date difference: nothing is drawn at random, so nothing needs a
seed. Models and periods are counted from 1, as the chapter counts them.

Swedish Embedded AB implements solutions for keeping long-lived machine learning engines coherent as models, devices
and formats are added to them for its clients. If your team needs expertise in the evolution and maintenance of
machine learning systems then you can procure our services by sending an email to info@swedishembedded.com.
"""
import datetime


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _g(x, d=4):
    """d significant digits with trailing zeros kept (Python's correctly rounded decimal conversion), so that a value
    just below 1 prints as 1.000 and not as an exact 1."""
    if x == 0:
        return "0"
    return f"{x:#.{d}g}"


# ---------------------------------------------------------------------------------------------------------------------
# The order in which the path was built
# ---------------------------------------------------------------------------------------------------------------------

SPLIT_DAY = datetime.date(2026, 6, 24)

# (stage of the end-to-end path, what arrived, commit, author date). Each row is a commit the chapter cites.
FIRST_APPEARANCES = [
    ("kernels, device, model, optimizer, checkpoint", "the workspace split: WGSL kernels as the source of truth, "
     "the device layer, parameter store, AdamW, the weights container, two models", "d9bfa70d4",
     datetime.date(2026, 6, 24)),
    ("data", "tokenizers, dataset generators, loaders", "00121704e", datetime.date(2026, 6, 24)),
    ("checking", "the finite-difference gradient check", "afbbae41d", datetime.date(2026, 6, 24)),
    ("device", "the backend traits under the device facade", "7c8c97045", datetime.date(2026, 7, 14)),
    ("serving", "one capability interface for every model's actions", "da5bdf257", datetime.date(2026, 7, 26)),
    ("serving", "the HTTP layer", "28fd28203", datetime.date(2026, 7, 31)),
    ("shared answers", "the architecture registry", "e2879ae14", datetime.date(2026, 8, 13)),
    ("shared answers", "the served-model catalog as a library", "859427cf0", datetime.date(2026, 8, 25)),
    ("library", "the public library facade", "c44076117", datetime.date(2026, 9, 15)),
    ("serving", "serving under a memory budget, from a library", "b03687583", datetime.date(2026, 10, 2)),
    ("serving", "one continuous-batching loop for every language model", "0bd516395", datetime.date(2026, 10, 16)),
    ("quantization", "a measured mixed-precision weight search", "e68da7686", datetime.date(2026, 11, 4)),
]


def _path_days():
    """(stage, what, commit, date, days after the split) for every row, in date order."""
    return [(s, w, c, d, (d - SPLIT_DAY).days) for s, w, c, d in FIRST_APPEARANCES]


def path_order():
    rows = [(s, w, f"`{c}`", d.isoformat(), n) for s, w, c, d, n in _path_days()]
    head = ("Each row is the first commit of one part of the path, with its author date; the last column is the "
            "number of calendar days after 24 June 2026, the date of the workspace split (date difference, so the "
            "split itself is day 0).")
    return head + "\n\n" + _table(["stage of the path", "what arrived", "commit", "date", "day"], rows, "llllr")


# ---------------------------------------------------------------------------------------------------------------------
# Dense and sparse evaluation of the experts
# ---------------------------------------------------------------------------------------------------------------------

FIRST_ROUTER_CAPACITY = 64   # experts the first router kernel's scratch arrays held

# (configuration, where its numbers are read, d = model width, h = expert hidden width, E = experts, k = per token)
EXPERT_CONFIGS = [
    ("first model", "the training default of `crates/toymoe`", 64, 128, 4, 2),
    ("128 experts", "`thinker_defaults` of `crates/qwen3omnimoe`", 2048, 768, 128, 8),
    ("256 experts", "`GlmConfig::glm5_2` of `crates/glmdsa`", 6144, 2048, 256, 8),
]


def _expert_flops(d, h):
    """FLOPs of one expert on one token: three matrices of d x h multiply-adds (gate, up, down), 2 FLOPs each."""
    return 2 * 3 * d * h


def _sci(n):
    """An exact integer, printed whole below a million and to four significant digits above."""
    return f"{n:,}" if n < 10 ** 6 else f"{n:.3e}"


def expert_work():
    rows = []
    for name, _src, d, h, e, k in EXPERT_CONFIGS:
        one = _expert_flops(d, h)
        rows.append((name, d, h, e, k, _sci(e * one), _sci(k * one), f"{e / k:g}",
                     "yes" if e <= FIRST_ROUTER_CAPACITY else "no"))
    sources = "; ".join(f"{c[0]}: {c[1]}" for c in EXPERT_CONFIGS)
    head = ("Counting rule: one expert is a gated feed-forward block with three matrices of d x h weights (gate and up "
            "projections from width d to h, down projection back), so it costs 3 d h multiply-adds per token, counted "
            "as 2 FLOPs each; the router, the gate's activation and product and the weighted sum of the experts are "
            "not counted. Dense evaluation runs all E experts on every token and multiplies the unselected ones by a "
            "zero gate; sparse dispatch runs the k selected ones. The configurations are read from the code ("
            + sources + "). The last column asks whether E fits the "
            f"{FIRST_ROUTER_CAPACITY} slots of the first router kernel's scratch arrays. Counts above a million are "
            "printed to four significant digits.")
    return head + "\n\n" + _table(
        ["configuration", "d", "h", "E", "k", "dense FLOPs per token", "sparse FLOPs per token", "dense / sparse",
         f"E <= {FIRST_ROUTER_CAPACITY}"], rows, "lrrrrrrrl")


# ---------------------------------------------------------------------------------------------------------------------
# Copies, fixes and the hoist
# ---------------------------------------------------------------------------------------------------------------------

N_MODELS = 40      # models added, one per period, periods 1 .. N_MODELS
FIXES = 1          # fixes to the shared component that land in every period, after that period's model arrives
MISS = 0.05        # chance that one fix is forgotten in one copy, independently for every (fix, copy)
HOISTS = (2, 3, 5, 10, 20)


def _drifted_at(k, f=FIXES, mu=MISS):
    """Expected copies that have missed at least one fix when model k has just arrived (before period k's fixes).

    Copy i (made when model i arrived, from an up-to-date copy) has seen the fixes of periods i .. k-1, that is
    (k - i) f of them, and has missed none with probability (1 - mu)^((k - i) f).
    """
    return sum(1.0 - (1.0 - mu) ** ((k - i) * f) for i in range(1, k + 1))


def _drifted_after(n, f=FIXES, mu=MISS):
    """Expected drifted copies after the fixes of period n, when nothing is ever hoisted."""
    return sum(1.0 - (1.0 - mu) ** ((n - i + 1) * f) for i in range(1, n + 1))


def _fix_edits(k, n=N_MODELS, f=FIXES):
    """Edits that apply the fixes over n periods when the copies are hoisted on the arrival of model k (k = None: never).

    Before the hoist a fix in period t is applied to the t copies that exist; from period k on it is one edit.
    """
    if k is None:
        return sum(f * t for t in range(1, n + 1))
    return sum(f * t for t in range(1, k)) + f * (n - k + 1)


def hoist_timing():
    rows = []
    for k in HOISTS:
        fix = _fix_edits(k)
        rows.append((f"on the arrival of model {k}", fix, k, fix + k, _g(_drifted_at(k)), 0))
    fix = _fix_edits(None)
    rows.append(("never", fix, 0, fix, "-", _g(_drifted_after(N_MODELS))))
    head = (f"Assumptions of the example, not measurements: {N_MODELS} models arrive one per period (periods 1 to "
            f"{N_MODELS}); in each period the model arrives first and then {FIXES} fix of the shared component lands. "
            f"Before the hoist every model has its own copy, made from an up-to-date copy, and a fix must be applied to "
            f"every copy that exists; each application is forgotten with probability {MISS}, independently. A copy "
            f"that has missed any fix is drifted. The hoist happens when the stated model has arrived and before that "
            f"period's fix: one copy becomes the shared implementation, every model is pointed at it (one edit per "
            f"model so far), and every drifted copy must be reconciled with it, which changes that model's numbers. "
            f"After the hoist a fix is one edit to the shared implementation. Expected values, printed to four "
            f"significant digits.")
    return head + "\n\n" + _table(
        ["hoist", "edits for fixes", "edits to migrate", "total edits", "drifted copies to reconcile",
         f"drifted copies after model {N_MODELS}"], rows)


# ---------------------------------------------------------------------------------------------------------------------
# Closing a defect class, once or per copy
# ---------------------------------------------------------------------------------------------------------------------

Q_ESC = 0.1        # chance that one new model meets the class in one copy and the existing tests miss it
COPIES = 4         # copies of the code that carries the class
MODEL_COUNTS = (1, 5, 10, 20, 40, 80)


def _q_shared(q=Q_ESC, kc=COPIES):
    """One implementation serving the uses the copies served: a model meets it if it meets any of the kc uses."""
    return 1.0 - (1.0 - q) ** kc


def _escapes(m, regime, q=Q_ESC, kc=COPIES):
    """Expected escaped defects after m new models.

    'instance': every escape is fixed for that one case only, the class stays open in every copy: kc m q.
    'class per copy': a copy's class is closed by its first escape: kc (1 - (1 - q)^m).
    'class shared': one implementation, closed by the first escape anywhere: 1 - (1 - q_shared)^m.
    """
    if regime == "instance":
        return kc * m * q
    if regime == "class per copy":
        return kc * (1.0 - (1.0 - q) ** m)
    if regime == "class shared":
        return 1.0 - (1.0 - _q_shared(q, kc)) ** m
    raise ValueError(regime)


def class_closing():
    rows = [(m, _g(_escapes(m, "instance")), _g(_escapes(m, "class per copy")), _g(_escapes(m, "class shared")))
            for m in MODEL_COUNTS]
    head = (f"Assumptions of the example, not measurements: the code that carries a defect class exists in "
            f"{COPIES} copies; each new model meets the class in each copy with probability {Q_ESC}, independently, "
            f"and the existing tests miss it, so the defect escapes. An instance fix repairs the case that escaped "
            f"and leaves the class open; a class fix closes it for good in the code it is applied to. With one shared "
            f"implementation a model meets the class with probability 1 - (1 - {Q_ESC})^{COPIES} = "
            f"{_g(_q_shared())}. Expected escaped defects after M new models, four significant digits.")
    return head + "\n\n" + _table(
        ["new models M", f"{COPIES} copies, instance fixes", f"{COPIES} copies, class fix in each",
         "one implementation, class fix"], rows)
