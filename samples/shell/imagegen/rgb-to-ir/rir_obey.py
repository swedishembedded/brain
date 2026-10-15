#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements instruction-following measurement for
# image-to-image translators for its clients. If your team needs expertise
# in controllable generation or in testing whether a generator does what it
# is told, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Does the instruction-conditioned translator do what the instruction says?

The translator is told that an object is warmer than, cooler than or about the
same temperature as its surroundings. This module measures the IR tile it
generated, with the same contrast the instructions were derived from
(`rir_regions.object_contrast`: median in the eroded mask less median in a ring
that excludes other objects, over the frame's MAD):

    c_gen             contrast of the generated tile in the region (mask from the source RGB tile)
    direction         the instruction is obeyed when polarity(c_gen, tau) equals it, where polarity
                      is rir_regions.polarity and tau the sample's noise floor: `ObeyPair.tau`
                      if given, else measured on the real IR tile (`real_ir`), else on the
                      generated tile. "same" means |c_gen| < tau.
    controllability   s * (c_gen - c_cf), the generation with the instruction against a
                      counterfactual generation of the SAME tile with the flipped instruction;
                      s is +1 for warmer and -1 for cooler, so a positive value is obedience.
                      "same" has no flipped twin. Absent (None) without counterfactuals, never 0.
    luminance shortcut partial correlation of c_gen with the contrast of the same region in the
                      RGB luma (the same function on the source tile), holding the instruction
                      fixed (residuals of the within-instruction means). Near 1 means the
                      generator reads brightness instead of the instruction.
    within_noise      the share of renders whose |c_gen| is under the noise floor tau: the instruction left
                      no measurable trace on the object, whichever way it pointed.
    neutral shift     s * (c_gen - c_neutral) against the render of the same tile with the neutral caption
                      (no statement), per instruction: how far each polarity moves the object from where the
                      translator puts it unprompted, which controllability alone cannot split (a generator
                      that only ever obeys "warmer" has a controllability and a zero cooler shift).
    fidelity          |c_gen - c_real|, c_real from `real_ir` or given; how far the generated
                      contrast is from the real one. `oracle_fidelity` keeps the renders whose instruction
                      is what the real IR shows (polarity of c_real against tau): the quality of the rendering
                      when told the truth. `neutral_fidelity` is the same distance for the neutral render. With an oracle instruction (measured on
                      the real IR) this is the quality of the rendering; with a prior it also
                      shows where the prior was wrong.

Every number is a mean over pairs with a 95 percent percentile interval from a bootstrap that
resamples SEQUENCES (rir_bootstrap.resample_weights), per instruction source and region type.
Oracle instructions (measured from the real IR), prior instructions (class priors,
`class_priors`) and forced instructions (chosen by the experimenter whatever the real IR shows, with the training
wording as `forced` and the held-out wording as `forced_heldout`) are reported apart under `sources`. Pairs that
carry a `stratum` (day or night, say) are also reported per stratum under `strata` of each region type. A region too small to measure is counted in
`n_skipped` and is in no mean. `effect` is the number rir_decide's K5 reads: the controllability
when counterfactuals exist, else the direction margin (accuracy less the 0.5 chance level over
warmer / cooler instructions).

    rir_obey.py PAIRS.jsonl --out obedience.json [--resamples 2000] --seed N

PAIRS.jsonl has one pair per line, paths relative to the file: sequence, region_type,
instruction, source ("oracle" | "prior"), generated, mask (PNG, non-zero is the object), and
optionally rgb, other_masks [paths], counterfactual, neutral, real_ir, stratum, tau, c_real, key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

import rir_bootstrap as B
import rir_captions as C
import rir_data as D
import rir_regions as R

INSTRUCTIONS = C.POLARITIES
SOURCES = ("oracle", "prior", "forced", "forced_heldout")
FLIP = {"warmer": "cooler", "cooler": "warmer"}
SIGN = {"warmer": 1.0, "cooler": -1.0}
CHANCE = 0.5  # direction accuracy of a generator that guesses warmer or cooler


@dataclass
class ObeyPair:
    sequence: str
    region_type: str
    instruction: str
    source: str
    generated: np.ndarray  # single-channel uint8 IR tile
    mask: np.ndarray  # bool, on the source RGB tile
    rgb: np.ndarray | None = None  # BGR source tile, for the luminance shortcut
    other_masks: list = field(default_factory=list)
    counterfactual: np.ndarray | None = None  # same tile, flipped instruction
    neutral: np.ndarray | None = None  # same tile, neutral caption
    stratum: str | None = None  # e.g. day / night: also reported apart
    real_ir: np.ndarray | None = None
    tau: float | None = None
    c_real: float | None = None
    key: str = ""


def class_priors(captions_report: dict) -> dict[str, str]:
    """Class -> its most frequent polarity in the caption report of split T (ties: warmer, cooler, same)."""
    return {name: max(INSTRUCTIONS, key=lambda p: (c[p], -INSTRUCTIONS.index(p)))
            for name, c in captions_report["classes"].items()}


# ---------------------------------------------------------------- measurement


def _contrast(ir: np.ndarray, pair: ObeyPair, params: R.RegionParams, seed: int) -> float | None:
    c = R.object_contrast(R.dequantize(ir, seed), pair.mask, pair.other_masks, params)
    return None if c is None else c["contrast"]


def _noise_floor(ir: np.ndarray, pair: ObeyPair, params: R.RegionParams, seed: int) -> float | None:
    f = R.dequantize(ir, seed)
    inside = R.object_contrast(f, pair.mask, pair.other_masks, params)
    if inside is None:
        return None
    return R.noise_floor(f, [pair.mask, *pair.other_masks], inside["inside_px"], np.random.default_rng([seed, 0x70A5]), params)


def measure_pair(pair: ObeyPair, params: R.RegionParams, seed: int) -> dict | str:
    """Per-pair numbers, or the reason the region cannot be measured."""
    if pair.instruction not in INSTRUCTIONS:
        raise ValueError(f"{pair.key}: instruction must be one of {INSTRUCTIONS}, got {pair.instruction!r}")
    if pair.source not in SOURCES:
        raise ValueError(f"{pair.key}: source must be one of {SOURCES}, got {pair.source!r}")
    seed = R.tile_seed(seed, pair.key)
    c_gen = _contrast(pair.generated, pair, params, seed)
    if c_gen is None:
        return "mask_or_ring_too_small"
    tau = pair.tau
    if tau is None:
        tau = _noise_floor(pair.real_ir if pair.real_ir is not None else pair.generated, pair, params, seed)
    c_cf = None
    if pair.counterfactual is not None and pair.instruction in FLIP:
        c_cf = _contrast(pair.counterfactual, pair, params, seed)
    c_neutral = None
    if pair.neutral is not None:
        c_neutral = _contrast(pair.neutral, pair, params, seed)
    c_real = pair.c_real
    if c_real is None and pair.real_ir is not None:
        c_real = _contrast(pair.real_ir, pair, params, seed)
    return {
        "c_gen": c_gen,
        "correct": None if tau is None else R.polarity(c_gen, tau) == pair.instruction,
        "delta": None if c_cf is None else SIGN[pair.instruction] * (c_gen - c_cf),
        "neutral_delta": None if c_neutral is None or pair.instruction not in SIGN
        else SIGN[pair.instruction] * (c_gen - c_neutral),
        "neutral_fidelity": None if c_neutral is None or c_real is None else abs(c_neutral - c_real),
        "real_polarity": None if c_real is None else R.polarity(c_real, tau),
        "within_noise": None if tau is None else abs(c_gen) < tau,
        "c_rgb": None if pair.rgb is None else _contrast(D.to_gray(pair.rgb), pair, params, seed),
        "fidelity": None if c_real is None else abs(c_gen - c_real),
    }


# ----------------------------------------------------------------- statistics


def partial_correlation(x: np.ndarray, y: np.ndarray, groups: np.ndarray, weights: np.ndarray) -> float:
    """Weighted correlation of x and y after removing each group's (weighted) mean: NaN when undefined."""
    rx, ry = np.zeros(len(x)), np.zeros(len(y))
    for g in np.unique(groups):
        m = groups == g
        total = weights[m].sum()
        if total > 0:
            rx[m] = x[m] - (weights[m] * x[m]).sum() / total
            ry[m] = y[m] - (weights[m] * y[m]).sum() / total
    den = np.sqrt((weights * rx * rx).sum() * (weights * ry * ry).sum())
    return float((weights * rx * ry).sum() / den) if den > 0 else float("nan")


def _summary(estimate: float, replicates: np.ndarray, n: int) -> dict | None:
    replicates = replicates[~np.isnan(replicates)]
    if n == 0 or np.isnan(estimate) or len(replicates) == 0:
        return None
    lo, hi = B.percentile_ci(replicates)
    return {"estimate": float(estimate), "ci": [lo, hi], "n": n}


def mean_with_ci(values: list, sequences: list, n_resamples: int, seed: int) -> dict | None:
    """Mean and 95 percent interval of per-pair values, resampling whole sequences; None for no values."""
    if not values:
        return None
    v = np.asarray(values, float)
    w = B.resample_weights(sequences, n_resamples, seed)
    return _summary(float(v.mean()), (w @ v) / w.sum(axis=1), len(v))


def _shortcut(rows: list, n_resamples: int, seed: int) -> dict | None:
    rows = [r for r in rows if r["c_rgb"] is not None]
    if len(rows) < 3:
        return None
    x, y = np.array([r["c_gen"] for r in rows]), np.array([r["c_rgb"] for r in rows])
    groups = np.array([r["instruction"] for r in rows])
    w = B.resample_weights([r["sequence"] for r in rows], n_resamples, seed)
    reps = np.array([partial_correlation(x, y, groups, wb) for wb in w])
    return _summary(partial_correlation(x, y, groups, np.ones(len(x))), reps, len(rows))


def _neutral_shift(stat) -> dict | None:
    shift = {i: stat("neutral_delta", lambda r, i=i: r["instruction"] == i) for i in FLIP}
    return shift if any(shift.values()) else None


def _region_report(rows: list, n_resamples: int, seed: int) -> dict:
    def stat(key, keep=lambda r: True, value=None):
        value = value or (lambda r: r[key])
        sel = [r for r in rows if r[key] is not None and keep(r)]
        return mean_with_ci([value(r) for r in sel], [r["sequence"] for r in sel], n_resamples, seed)

    directional = lambda r: r["instruction"] in FLIP
    out = {"n": len(rows),
           "direction_accuracy": stat("correct", value=lambda r: float(r["correct"])),
           "controllability": stat("delta"),
           "within_noise_floor": stat("within_noise", value=lambda r: float(r["within_noise"])),
           "neutral_shift": _neutral_shift(stat),
           "luminance_shortcut": _shortcut(rows, n_resamples, seed),
           "fidelity": stat("fidelity"),
           "oracle_fidelity": stat("fidelity", lambda r: r["real_polarity"] == r["instruction"]),
           "neutral_fidelity": stat("neutral_fidelity")}
    margin = stat("correct", directional, lambda r: float(r["correct"]) - CHANCE)
    out["effect"] = ({"measure": "controllability", **out["controllability"]} if out["controllability"]
                     else {"measure": "direction", **margin} if margin else None)
    return out


def _typed_report(rows: list, n_resamples: int, seed: int) -> dict:
    """The region report of all rows and, when rows carry a stratum, of each stratum apart."""
    out = _region_report(rows, n_resamples, seed)
    strata = sorted({r["stratum"] for r in rows if r["stratum"] is not None})
    if strata:
        out["strata"] = {s: _region_report([r for r in rows if r["stratum"] == s], n_resamples, seed) for s in strata}
    return out


def evaluate(pairs: list[ObeyPair], params: R.RegionParams = R.RegionParams(), n_resamples: int = B.DEFAULT_RESAMPLES,
             seed: int = 1) -> dict:
    """The obedience report (plain JSON types), see the module docstring."""
    rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped: dict[str, Counter] = defaultdict(Counter)
    for pair in pairs:
        m = measure_pair(pair, params, seed)
        if isinstance(m, str):
            skipped[pair.source][m] += 1
        else:
            rows[pair.source][pair.region_type].append({**m, "instruction": pair.instruction, "sequence": pair.sequence,
                                                       "stratum": pair.stratum})
    sources = {}
    for src in sorted(set(rows) | set(skipped)):
        sources[src] = {"n_pairs": sum(len(r) for r in rows[src].values()), "n_skipped": dict(skipped[src]),
                        "region_types": {t: _typed_report(r, n_resamples, seed) for t, r in sorted(rows[src].items())}}
    return {"n_resamples": n_resamples, "seed": seed, "sources": sources}


# ------------------------------------------------------------------------ CLI


def _read(base: str, name: str, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(os.path.join(base, name), flags)
    if img is None:
        raise FileNotFoundError(os.path.join(base, name))
    return img


def load_pairs(path: str) -> list[ObeyPair]:
    base = os.path.dirname(os.path.abspath(path))
    pairs = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("sequence", "region_type", "instruction", "source", "generated", "mask") if k not in row]
            if missing:
                raise ValueError(f"line {n}: missing {missing}")
            ir = lambda name: None if name not in row else D.to_gray(_read(base, row[name]))
            pairs.append(ObeyPair(
                sequence=row["sequence"], region_type=row["region_type"], instruction=row["instruction"], source=row["source"],
                generated=ir("generated"), mask=D.to_gray(_read(base, row["mask"])) > 0,
                rgb=None if "rgb" not in row else _read(base, row["rgb"], cv2.IMREAD_COLOR),
                other_masks=[D.to_gray(_read(base, p)) > 0 for p in row.get("other_masks", [])],
                counterfactual=ir("counterfactual"), neutral=ir("neutral"), real_ir=ir("real_ir"), stratum=row.get("stratum"), tau=row.get("tau"), c_real=row.get("c_real"),
                key=row.get("key", f"line{n}")))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure whether generated IR tiles obey their instructions.")
    ap.add_argument("pairs", help="pairs jsonl, see the module docstring")
    ap.add_argument("--out", required=True, help="obedience.json to write (the file rir_decide reads for K5)")
    ap.add_argument("--resamples", type=int, default=B.DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    report = evaluate(load_pairs(a.pairs), n_resamples=a.resamples, seed=a.seed)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    for src, s in report["sources"].items():
        print(f"{src}: {s['n_pairs']} pairs measured, skipped {s['n_skipped']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
