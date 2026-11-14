# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Architecture of a Machine Learning Engine".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: engine_architecture.<name> -->` markers. Helpers with a leading underscore, and the data tables below, are
shared with the chapter's figures so a picture and the table beside it cannot disagree.

Everything here is counting, exact string comparison and reachability over small graphs, so another implementation
reproduces every printed value. No generator is used.

Swedish Embedded AB implements solutions for structuring large machine learning engines so that they can grow to hundreds
of models without collapsing for its clients. If your team needs expertise in software architecture for machine learning
systems then you can procure our services by sending an email to info@swedishembedded.com.
"""
from math import comb

import number_formats as nf

_table = nf._table


# ------------------------------------------------------------------ what a new model costs

def growth_cost():
    fronts, tables = 5, 4
    rows = []
    for n in (10, 100, 300):
        rows.append([f"{n:,}", f"{n * fronts:,}", f"{n + fronts:,}", f"{comb(tables, 2) * n:,}", f"{n:,}"])
    head = ("Counting rule: an engine serves N architectures through F = 5 front ends, and k = 4 separate tables each "
            "name every architecture. Without a shared interface every pair (architecture, front end) needs its own "
            "adapter, N x F of them; with one interface every architecture implements it once and every front end "
            "calls it once, N + F pieces. Four separate tables agree only when every one of the C(4, 2) = 6 pairs of "
            "tables agrees on every architecture, 6 N name pairs; one registry that all four read has N rows and "
            "nothing to keep in agreement.\n\n")
    return head + _table(["architectures N", "adapters, one per model and front end", "pieces through one interface",
                          "name pairs that must agree, four tables", "rows of one registry"], rows)


# ------------------------------------------------------------------ routing a configuration's class name

# The substring scan that once routed a checkpoint's configuration to an importer: the lowercased class name is tested
# for each family string in this order and the first one it contains wins.
SCAN_ORDER = ["omni", "gpt", "glm", "qwen", "lfm"]
SCAN_ORDER_OMNI_LAST = ["gpt", "glm", "qwen", "lfm", "omni"]

# An excerpt of the registry: (id, exact class names and model_type spellings, config probe). A probe (key, present)
# tells apart rows that share a class: the row applies when the configuration's having `key` equals `present`.
CLASS_ROWS = [
    ("gpt2", ["GPT2LMHeadModel"], None),
    ("qwen3", ["Qwen3ForCausalLM", "qwen3"], None),
    ("llama", ["LlamaForCausalLM", "llama"], None),
    ("qwen2", ["Qwen2ForCausalLM", "qwen2"], None),
    ("glmdsa", ["GlmMoeDsaForCausalLM", "glm_moe_dsa"], None),
    ("lfm2", ["Lfm2ForCausalLM"], None),
    ("qwen3omnimoe", ["Qwen3OmniMoeForConditionalGeneration"], None),
    ("qwen3vl", ["Qwen3VLForConditionalGeneration"], None),
    ("fastvlm", ["LlavaQwen2ForCausalLM"], None),
    ("qwen3tts", ["Qwen3TTSForConditionalGeneration"], None),
    ("deepseekvl", ["MultiModalityCausalLM", "multi_modality"], ("gen_head_config", False)),
    ("januspro", ["MultiModalityCausalLM", "multi_modality"], ("gen_head_config", True)),
]

# (class name, keys at the top level of its configuration)
CLASS_QUERIES = [
    ("GPT2LMHeadModel", []),
    ("Qwen3ForCausalLM", []),
    ("Qwen3OmniMoeForConditionalGeneration", []),
    ("Qwen3VLForConditionalGeneration", []),
    ("LlavaQwen2ForCausalLM", []),
    ("Qwen3TTSForConditionalGeneration", []),
    ("Qwen2ForCausalLM", []),
    ("LlamaForCausalLM", []),
    ("GlmMoeDsaForCausalLM", []),
    ("Lfm2ForCausalLM", []),
    ("MultiModalityCausalLM", []),
    ("MultiModalityCausalLM", ["gen_head_config"]),
    ("MistralForCausalLM", []),
]


def route_by_substring(name, order):
    """The first family string of `order` that the lowercased class name contains, or None."""
    lower = name.lower()
    for fam in order:
        if fam in lower:
            return fam
    return None


def by_hf(rows, cls):
    """The one row listing exactly `cls`; None when no row or more than one row lists it."""
    found = [r for r in rows if cls in r[1]]
    return found[0][0] if len(found) == 1 else None


def by_hf_config(rows, cls, keys):
    """The one row listing exactly `cls` whose probe, if any, agrees with the configuration's keys."""
    found = [r for r in rows if cls in r[1] and (r[2] is None or ((r[2][0] in keys) == r[2][1]))]
    return found[0][0] if len(found) == 1 else None


def _show(x):
    return "none" if x is None else f"`{x}`"


def class_routing():
    excerpt = "; ".join(
        f"`{i}`: " + ", ".join(f"`{c}`" for c in cls) + (f" (probe `{p[0]}` {'present' if p[1] else 'absent'})" if p else "")
        for i, cls, p in CLASS_ROWS)
    head = ("The scan lowercases the class name and returns the first of the family strings "
            + ", ".join(f"`{f}`" for f in SCAN_ORDER) + " (in this order) that it contains; the second scan column "
            "tries them in the order " + ", ".join(f"`{f}`" for f in SCAN_ORDER_OMNI_LAST) + ". A family string names "
            "one importer. The exact lookups read this excerpt of the registry, one row per id with its class names: "
            + excerpt + ". The class lookup answers only when exactly one row lists the class; the class and "
            "configuration lookup also keeps only the rows whose probe agrees with the configuration's keys, and "
            "answers when exactly one remains. Text comparison is exact and case-sensitive.\n\n")
    rows = []
    for cls, keys in CLASS_QUERIES:
        name = f"`{cls}`" + (" with " + ", ".join(f"`{k}`" for k in keys) if keys else "")
        rows.append([name, _show(route_by_substring(cls, SCAN_ORDER)), _show(route_by_substring(cls, SCAN_ORDER_OMNI_LAST)),
                     _show(by_hf(CLASS_ROWS, cls)), _show(by_hf_config(CLASS_ROWS, cls, keys))])
    return head + _table(["class name (and configuration key)", "scan", "scan, omni last", "exact class",
                          "class and configuration"], rows, "lllll")


# ------------------------------------------------------------------ the other lookups of a row

# (id, package, file-format spelling or None, recipe family aliases)
ID_ROWS = [
    ("qwen3", "brain-qwen3", None, ["qwen"]),
    ("llama", "brain-qwen3", None, []),
    ("qwen2", "brain-qwen3", None, []),
    ("qwen35", "brain-qwen35", "qwen35", []),
    ("edgerock", "brain-qwen35", None, []),
    ("deepseek2ocr", "brain-deepseek2ocr", "deepseek2-ocr", []),
    ("s3dit", "brain-s3dit", None, ["zimage"]),
    ("autoencoderkl", "brain-vae", None, []),
    ("wan", "brain-wan", None, []),
]


def by_id(rows, i):
    for r in rows:
        if r[0] == i:
            return r
    return None


def by_gguf(rows, spelling):
    for r in rows:
        if (r[2] if r[2] is not None else r[0]) == spelling:
            return r[0]
    return None


def by_family(rows, fam):
    r = by_id(rows, fam)
    if r is not None:
        return r[0]
    for r in rows:
        if fam in r[3]:
            return r[0]
    return None


def implementation(rows, i):
    """The row whose id is the row's package name without its `brain-` prefix, or the row itself when no row has
    that id."""
    row = by_id(rows, i)
    crate = row[1][len("brain-"):] if row[1].startswith("brain-") else row[1]
    impl = by_id(rows, crate)
    return impl[0] if impl is not None else row[0]


def lookups():
    excerpt = "; ".join(f"`{i}`: package `{p}`" + (f", file spelling `{g}`" if g else "")
                        + (", family alias " + ", ".join(f"`{a}`" for a in fams) if fams else "")
                        for i, p, g, fams in ID_ROWS)
    head = ("The registry excerpt: " + excerpt + ". A row with no file spelling uses its id as its spelling. The "
            "family lookup tries the id first and then the alias lists in row order. The implementation of a row is "
            "the row whose id is the row's package name without the `brain-` prefix, and the row itself when no row "
            "has that id.\n\n")
    queries = [
        ("id", "qwen3", _show((by_id(ID_ROWS, "qwen3") or [None])[0])),
        ("id", "Qwen3", _show((by_id(ID_ROWS, "Qwen3") or [None])[0])),
        ("file spelling", "deepseek2-ocr", _show(by_gguf(ID_ROWS, "deepseek2-ocr"))),
        ("file spelling", "deepseek2ocr", _show(by_gguf(ID_ROWS, "deepseek2ocr"))),
        ("file spelling", "wan", _show(by_gguf(ID_ROWS, "wan"))),
        ("family", "zimage", _show(by_family(ID_ROWS, "zimage"))),
        ("family", "qwen", _show(by_family(ID_ROWS, "qwen"))),
        ("family", "s3dit", _show(by_family(ID_ROWS, "s3dit"))),
        ("implementation", "llama", _show(implementation(ID_ROWS, "llama"))),
        ("implementation", "edgerock", _show(implementation(ID_ROWS, "edgerock"))),
        ("implementation", "qwen3", _show(implementation(ID_ROWS, "qwen3"))),
        ("implementation", "autoencoderkl", _show(implementation(ID_ROWS, "autoencoderkl"))),
    ]
    return head + _table(["lookup", "query", "answer"], [[k, f"`{q}`", a] for k, q, a in queries], "lll")


# ------------------------------------------------------------------ layers and the direction of edges

LAYER = {
    "arch": 1, "backend-api": 1, "gpu-core": 2, "data": 3, "model": 3, "promote": 3, "qwen3": 4,
    "bench": 5, "rl": 5, "loader": 6, "catalog": 6, "brain": 6, "cli": 6,
}
CRATES = list(LAYER)

# Normal dependencies (optional ones included) among these crates, as their manifests declare them.
DEPS_NOW = {
    "arch": [],
    "backend-api": [],
    "gpu-core": ["backend-api"],
    "data": [],
    "model": ["data", "gpu-core"],
    "promote": ["data"],
    "qwen3": ["arch", "data", "gpu-core", "model", "promote"],
    "bench": ["data", "gpu-core", "model", "promote", "qwen3"],
    "rl": ["bench", "data", "model", "promote", "qwen3"],
    "loader": ["arch", "gpu-core"],
    "catalog": ["data", "gpu-core", "loader", "model", "qwen3"],
    "brain": ["arch", "catalog", "data", "gpu-core", "loader", "model", "promote", "qwen3", "rl"],
    "cli": ["arch", "bench", "brain", "catalog", "data", "gpu-core", "loader", "model", "qwen3", "rl"],
}


def deps_gate_in_rl():
    """The same workspace with the promote/reject gate back inside rl and bench (no promote crate), plus the one edge
    a model crate needs to reach the gate: qwen3 -> rl."""
    out = {}
    for c, ds in DEPS_NOW.items():
        if c == "promote":
            continue
        out[c] = [d for d in ds if d != "promote"]
    out["qwen3"] = sorted(out["qwen3"] + ["rl"])
    return out


def upward_edges(layer, deps):
    return [(c, d) for c in deps for d in deps[c] if layer[d] > layer[c]]


def closure(deps, c):
    """Every crate that c depends on, directly or through others, c excluded unless it lies on a cycle."""
    seen, stack = [], list(deps[c])
    while stack:
        d = stack.pop(0)
        if d in seen:
            continue
        seen.append(d)
        stack.extend(deps[d])
    return seen


def reach(deps, c):
    """Every crate whose closure contains c: what must be rebuilt when c changes."""
    return [x for x in deps if x != c and c in closure(deps, x)]


def components(deps):
    """Groups of two or more crates that each reach all the others (the crates that lie on a cycle)."""
    out, placed = [], []
    for c in deps:
        if c in placed or c not in closure(deps, c):
            continue
        group = [c] + [d for d in deps if d != c and d in closure(deps, c) and c in closure(deps, d)]
        placed.extend(group)
        out.append(group)
    return out


def _edges_text(deps):
    return "; ".join(f"`{c}` -> " + ", ".join(f"`{d}`" for d in ds) for c, ds in deps.items() if ds)


def layer_check():
    layers = "; ".join(f"layer {n}: " + ", ".join(f"`{c}`" for c in CRATES if LAYER[c] == n) for n in range(1, 7))
    head = ("Thirteen crates and their layers (" + layers + "; `brain` is the SDK's package name). Today's edges, "
            "\"depends on\" written as an arrow: " + _edges_text(DEPS_NOW) + ". The layers are those of "
            "`scripts/gates/crate-layers.map` and the edges are read from the `[dependencies]` tables of the thirteen "
            "crates' current `Cargo.toml` files (normal dependencies, optional ones included, development-only ones "
            "excluded). The second graph is a constructed counterfactual, not the dependency graph of any past "
            "commit: it is today's graph with the gate put back inside `rl` and `bench` (no `promote` crate and none "
            "of its five edges) and the edge `qwen3` -> `rl` that a model crate needs to reach it. An edge is upward when its target's "
            "layer number is larger than its source's; a cycle group is a set of two or more crates that each "
            "depend, directly or through others, on all the rest.\n\n")
    rows = []
    for name, deps in [("today's edges", DEPS_NOW), ("gate in rl, plus qwen3 -> rl", deps_gate_in_rl())]:
        up = upward_edges(LAYER, deps)
        comps = components(deps)
        rows.append([name, str(len(deps)), str(sum(len(v) for v in deps.values())),
                     ", ".join(f"`{a}` -> `{b}`" for a, b in up) or "none",
                     "; ".join("{" + ", ".join(f"`{c}`" for c in g) + "}" for g in comps) or "none"])
    return head + _table(["graph", "crates", "edges", "upward edges", "cycle groups"], rows, "lrrll")


def change_reach():
    head = ("For today's thirteen-crate graph of the previous table. A crate's closure is every crate it depends on, "
            "directly or through others; its reach is every crate whose closure contains it. The work of a change "
            "is counted in crates rebuilt and retested: the changed crate and its reach. The mean is taken over the "
            "13 crates, each counted once (divisor 13), and printed to three decimals.\n\n")
    rows, total = [], 0
    for c in sorted(CRATES, key=lambda x: (LAYER[x], x)):
        n_need, n_reach = len(closure(DEPS_NOW, c)), len(reach(DEPS_NOW, c))
        total += 1 + n_reach
        rows.append([f"`{c}`", str(LAYER[c]), str(n_need), str(n_reach), str(1 + n_reach)])
    out = _table(["crate", "layer", "crates in its closure", "crates in its reach", "crates rebuilt by a change"],
                 rows, "lrrrr")
    return head + out + f"\n\nMean crates rebuilt by a change to one crate: {total} / 13 = {total / 13:.3f}."


# ------------------------------------------------------------------ the SDK's feature vocabulary

# Excerpts of the SDK's feature table before and after the change that split the image type out. Each feature lists
# the features it turns on and, after "dep:", the crates it links. `full` is cut to the three surfaces shown.
FEATURES_BEFORE = {
    "default": ["full"],
    "full": ["image", "forecast", "vision"],
    "image": ["resolve", "dep:brain-flux2", "dep:brain-s3dit", "dep:brain-imaging", "dep:brain-rrdbnet",
              "dep:brain-codeformer"],
    "forecast": ["resolve", "dep:brain-forecast", "dep:brain-kronos", "dep:brain-timesfm3"],
    "vision": ["resolve", "dep:brain-clip", "dep:brain-yolov8", "dep:brain-sam2"],
    "device": ["dep:brain-gpu-core"],
    "resolve": ["device", "dep:brain-loader", "dep:brain-model"],
}
FEATURES_AFTER = dict(FEATURES_BEFORE)
FEATURES_AFTER.update({
    "image": ["resolve", "imagetype", "dep:brain-flux2", "dep:brain-s3dit", "dep:brain-rrdbnet",
              "dep:brain-codeformer"],
    "vision": ["resolve", "imagetype", "dep:brain-clip", "dep:brain-yolov8", "dep:brain-sam2", "dep:brain-zipdepth"],
    "imagetype": ["dep:brain-imaging"],
})

# The feature each public item is compiled under, before and after, and the items each surface's code uses.
GATE_BEFORE = {"Image": "image", "DetectionPipeline": "vision", "ForecastPipeline": "forecast",
               "ImagePipeline": "image"}
GATE_AFTER = {"Image": "imagetype", "DetectionPipeline": "vision", "ForecastPipeline": "forecast",
              "ImagePipeline": "image"}
USES = {"DetectionPipeline": ["Image"], "ImagePipeline": ["Image"], "ForecastPipeline": [], "Image": []}


def feature_closure(table, chosen, defaults):
    """Every feature a build turns on: the named ones, `default` unless switched off, and everything they list."""
    todo = list(chosen) + (["default"] if defaults else [])
    on, crates = [], []
    while todo:
        f = todo.pop(0)
        if f in on:
            continue
        on.append(f)
        for x in table[f]:
            if x.startswith("dep:"):
                if x[4:] not in crates:
                    crates.append(x[4:])
            else:
                todo.append(x)
    return on, crates


def compiles(gate, on):
    """True when every item compiled in this build finds every item it uses compiled too."""
    present = [i for i, f in gate.items() if f in on]
    missing = [(i, u) for i in present for u in USES[i] if u not in present]
    return missing


def feature_builds():
    def fmt_table(t):
        return "; ".join(f"`{k}` = [" + ", ".join(f"`{x}`" for x in v) + "]" for k, v in t.items())
    head = ("Excerpts of the feature table of `crates/sdk/Cargo.toml` (`full` is cut to the three surfaces shown; the "
            "real one lists more). Before: " + fmt_table(FEATURES_BEFORE) + ". After, three lines differ (the same "
            "commit also added `dep:brain-zipdepth` to `vision`): " +
            fmt_table({k: FEATURES_AFTER[k] for k in ("image", "vision", "imagetype")}) +
            ". Items are compiled under one feature each, before: " +
            ", ".join(f"`{i}` under `{f}`" for i, f in GATE_BEFORE.items()) + "; after, `Image` is compiled under "
            "`imagetype`. `DetectionPipeline` and `ImagePipeline` use `Image`. A build turns on the features it "
            "names, `default` unless defaults are switched off, and every feature those list, repeatedly; it "
            "compiles when every compiled item finds the items it uses compiled too. The features turned on are listed in alphabetical order; crates are counted once each.\n\n")
    rows = []
    for label, table, gate in [("before", FEATURES_BEFORE, GATE_BEFORE), ("after", FEATURES_AFTER, GATE_AFTER)]:
        for chosen, defaults in [(["vision"], True), (["vision"], False), (["forecast"], False), (["image"], False)]:
            on, crates = feature_closure(table, chosen, defaults)
            missing = compiles(gate, on)
            rows.append([label, f"`{chosen[0]}`" + ("" if defaults else ", no defaults"), ", ".join(f"`{f}`" for f in sorted(on)),
                         str(len(crates)), "yes" if not missing else "no: " + ", ".join(f"`{i}` needs `{u}`" for i, u in missing)])
    return head + _table(["table", "build names", "features turned on", "crates linked", "compiles"], rows, "lllrl")
