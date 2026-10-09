#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements instruction-caption generation for
# image-editing LoRA fine-tuning for its clients. If your team needs
# expertise in measured, instruction-conditioned training data for
# diffusion models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Instruction captions for the RGB-to-IR translator, from measured contrast.

Every caption opens with the neutral instruction. A caption then states, per
object (or part) class of the tile, only what `rir_regions` measured on the REAL
IR: warmer than, cooler than, or about the same temperature as the surroundings,
the last when |c| < tau. Nothing is inferred from the class name and no
counterfactual target is ever synthesised: the instruction describes the IR
that sits next to it.

Grammar. A clause is `<subject> <predicate>.` with the subject drawn from
SUBJECTS ("The car", "The visible car"; a part is "The bonnet of the car") and
the predicate from the polarity's TRAIN_PREDICATES. A caption names one clause
or up to three, one per class, from a seeded draw. A class whose instances
disagree in one tile (one warm car, one cool car) is not stated, and the
conflict is counted. Two predicates, one warmer and one cooler, are HELD_OUT:
they are never written into captions.yaml and appear only in
heldout-captions.jsonl, to test whether the adapter follows an instruction in
words it did not train on.

Neutral share. `neutral_share` of all tiles (default 0.3) keep the neutral
caption alone, so the adapter also learns to translate without being told what
to do. Tiles with no measured statement are neutral by necessity and count
toward that share; when there are more of them than the share asks for, the
share is higher and the report says so.

Balance. For each class the report counts the tiles that state warmer, cooler
and same, and flags a class whose rarest polarity is under 10 percent of its
statements as "not controllable from natural data": the adapter cannot learn
an instruction whose outcome it almost never saw.

Paraphraser hook. `paraphrase(statement_text) -> str` may rewrite the
statements of a training caption (an LLM, say). The result is used only when it
is a single line that still names every object of the statement; otherwise the
template text stays. The hook never sees the neutral instruction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from typing import Callable

import numpy as np

import rir_tiles as T

NEUTRAL_CAPTION = T.NEUTRAL_CAPTION
POLARITIES = ("warmer", "cooler", "same")
MAX_OBJECTS = 3
FLAG_BELOW = 0.10
FLAG_TEXT = "not controllable from natural data"

SUBJECTS = ("The {name}", "The visible {name}")
TRAIN_PREDICATES = {
    "warmer": ("is warmer than its surroundings", "appears warmer than the background",
               "is hotter than the area around it", "stands out as warmer than its surroundings"),
    "cooler": ("is cooler than its surroundings", "appears colder than the background",
               "is colder than the area around it", "stands out as cooler than its surroundings"),
    "same": ("is about the same temperature as its surroundings", "has nearly the same temperature as the background",
             "is no warmer or cooler than the area around it", "blends in thermally with its surroundings"),
}
HELD_OUT = {
    "warmer": "reads as a heat source against its cooler surroundings",
    "cooler": "reads as a cold patch against its warmer surroundings",
}


def subject_name(cls: str, part: str | None) -> str:
    cls = cls.replace("_", " ")
    return f"{part.replace('_', ' ')} of the {cls}" if part else cls


def clause(name: str, polarity: str, rng: np.random.Generator | None, held_out: bool = False) -> str:
    """One sentence. `rng=None` takes the first subject and predicate (the evaluation wording)."""
    if held_out and polarity in HELD_OUT:
        predicate = HELD_OUT[polarity]
    else:
        options = TRAIN_PREDICATES[polarity]
        predicate = options[0 if rng is None else int(rng.integers(len(options)))]
    subject = SUBJECTS[0 if rng is None else int(rng.integers(len(SUBJECTS)))]
    return f"{subject.format(name=name)} {predicate}."


def has_polarity(caption: str, cls: str, polarity: str, part: str | None = None) -> bool:
    """Whether `caption` states `polarity` for the class (training or held-out wording)."""
    name = subject_name(cls, part)
    predicates = TRAIN_PREDICATES[polarity] + ((HELD_OUT[polarity],) if polarity in HELD_OUT else ())
    return any(f"{s.format(name=name)} {p}." in caption for s in SUBJECTS for p in predicates)


def agreed_statements(objects: list[dict]) -> tuple[dict[tuple[str, str | None], tuple[str, int]], int]:
    """({(class, part): (polarity, mask pixels)}, conflicts): instances of one class agree or the class is omitted."""
    seen: dict[tuple[str, str | None], list[dict]] = {}
    for o in objects:
        if o.get("polarity") is not None:
            seen.setdefault((o["class"], o.get("part")), []).append(o)
    agreed, conflicts = {}, 0
    for key, group in seen.items():
        if len({o["polarity"] for o in group}) == 1:
            agreed[key] = (group[0]["polarity"], sum(o.get("mask_px", 0) for o in group))
        else:
            conflicts += 1
    return agreed, conflicts


def _ranked(statements: dict) -> list[tuple[tuple[str, str | None], str]]:
    """Directional statements first, then by mask size, then by name: a stable order for the seeded draw."""
    return [(k, p) for k, (p, px) in sorted(statements.items(), key=lambda kv: (kv[1][0] == "same", -kv[1][1], kv[0][0], kv[0][1] or ""))]


def pick_statements(statements: dict, rng: np.random.Generator) -> list[tuple[tuple[str, str | None], str]]:
    """One statement half of the time, otherwise 2..3 (never more than MAX_OBJECTS), drawn from the ranked ones."""
    ranked = _ranked(statements)
    if not ranked:
        return []
    top = ranked[: MAX_OBJECTS + 2]
    count = 1 if len(top) == 1 or rng.random() < 0.5 else int(rng.integers(2, min(MAX_OBJECTS, len(top)) + 1))
    chosen = sorted(rng.choice(len(top), count, replace=False).tolist())
    return [top[i] for i in chosen]


def _names(chosen) -> list[str]:
    return [subject_name(cls, part) for (cls, part), _ in chosen]


def render_statements(chosen, rng: np.random.Generator, paraphrase: Callable[[str], str] | None) -> str:
    text = " ".join(clause(subject_name(cls, part), pol, rng) for (cls, part), pol in chosen)
    if paraphrase is not None:
        rewritten = paraphrase(text)
        if (isinstance(rewritten, str) and rewritten.strip() and "\n" not in rewritten
                and all(n.lower() in rewritten.lower() for n in _names(chosen))):
            return rewritten.strip()
    return text


def _load_regions(tiles_dir: str, name: str) -> list[dict]:
    path = os.path.join(tiles_dir, "regions", f"{name}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return json.load(fh)["objects"]


def _rank_key(seed: int, name: str) -> int:
    return zlib.crc32(f"{seed}/neutral/{name}".encode())


def build_captions(tiles_dir: str, seed: int = 1, neutral_share: float = 0.3,
                   paraphrase: Callable[[str], str] | None = None) -> dict:
    """Write captions.yaml (training), heldout-captions.jsonl (evaluation only) and captions-report.json."""
    tiles = [e for e in T.read_index(tiles_dir) if e["accepted"]]
    per_tile, conflicts = {}, 0
    for e in tiles:
        statements, c = agreed_statements(_load_regions(tiles_dir, e["name"]))
        per_tile[e["name"]], conflicts = statements, conflicts + c
    stated = [e["name"] for e in tiles if per_tile[e["name"]]]
    extra_neutral = max(0, round(neutral_share * len(tiles)) - (len(tiles) - len(stated)))
    neutral = set(sorted(stated, key=lambda n: _rank_key(seed, n))[:extra_neutral])

    captions, heldout, template_use, named = {}, [], {}, {}
    for e in tiles:
        name, statements = e["name"], per_tile[e["name"]]
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if statements and name not in neutral:
            chosen = pick_statements(statements, rng)
            captions[e["ir"]] = f"{NEUTRAL_CAPTION} {render_statements(chosen, rng, paraphrase)}"
            named[len(chosen)] = named.get(len(chosen), 0) + 1
            for _, pol in chosen:
                template_use[pol] = template_use.get(pol, 0) + 1
        else:
            captions[e["ir"]] = NEUTRAL_CAPTION
        directional = [(k, p) for k, p in _ranked(statements) if p in HELD_OUT]
        if directional:
            shown = _ranked(statements)[:MAX_OBJECTS]
            text = " ".join(clause(subject_name(*k), p, None, held_out=True) for k, p in shown)
            heldout.append({"name": e["ir"], "caption": f"{NEUTRAL_CAPTION} {text}",
                            "statements": [[subject_name(*k), p] for k, p in shown]})

    T.write_flat_yaml(os.path.join(tiles_dir, "captions.yaml"), captions)
    with open(os.path.join(tiles_dir, "heldout-captions.jsonl"), "w") as fh:
        for row in heldout:
            fh.write(json.dumps(row) + "\n")
    report = {"tiles": len(tiles), "tiles_with_statement": len(stated),
              "neutral_share": sum(c == NEUTRAL_CAPTION for c in captions.values()) / max(1, len(tiles)),
              "objects_named": {str(k): v for k, v in sorted(named.items())}, "statements_by_polarity": template_use,
              "conflicting_classes": conflicts, "heldout_tiles": len(heldout),
              "classes": balance(per_tile.values())}
    with open(os.path.join(tiles_dir, "captions-report.json"), "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    return report


def balance(per_tile_statements) -> dict:
    """Per class: tiles stating each polarity, their shares, the rarest polarity and the controllability flag."""
    counts: dict[str, dict[str, int]] = {}
    for statements in per_tile_statements:
        for (cls, part), (pol, _) in statements.items():
            counts.setdefault(subject_name(cls, part), dict.fromkeys(POLARITIES, 0))[pol] += 1
    out = {}
    for name, c in sorted(counts.items()):
        n = sum(c.values())
        share = {p: c[p] / n for p in POLARITIES}
        minority = min(POLARITIES, key=lambda p: (share[p], p))
        out[name] = {**c, "n": n, "share": share, "minority": minority,
                     "flag": FLAG_TEXT if share[minority] < FLAG_BELOW else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write instruction captions for a measured tile folder.")
    ap.add_argument("tiles", help="tile folder with regions/ from rir_regions.py")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--neutral-share", type=float, default=0.3)
    a = ap.parse_args(argv)
    report = build_captions(a.tiles, a.seed, a.neutral_share)
    print(json.dumps({k: v for k, v in report.items() if k != "classes"}), file=sys.stderr)
    for name, c in report["classes"].items():
        flag = f"  [{c['flag']}]" if c["flag"] else ""
        print(f"{name}: warmer {c['warmer']} cooler {c['cooler']} same {c['same']} (n={c['n']}){flag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
