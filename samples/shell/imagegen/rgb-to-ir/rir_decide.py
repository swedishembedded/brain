#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Pre-registered decision rules of the RGB-to-IR study: results in, verdict out.

    rir_decide.py RESULTS --config decision-config.json --out DIR   # DIR/decision.md and decision.json

RESULTS holds one directory per arm and in it one `brain yolov8 eval
--dump-preds` file per training seed (`RESULTS/<arm>/<anything>.jsonl`), all
scored on the SAME packed held-out set; a run on another image set is refused.
The decision config (JSON, paths relative to it):

    sequences         sequences.json of the packed evaluation set (rir_pack.py)
    roles             {role: arm directory name}; roles A1 A2 B3 C2 REAL_K REAL_K_PLUS_SYNTHETIC
    resamples, seed   bootstrap size (2000) and seed (1)
    alpha             0.05
    nc                number of classes of the detector head (default: from the data)
    instruction_model true when the translator is the instruction-conditioned one (enables K5)
    gate_statistics   JSON of the geometry-gate rejections of C2 (rir_gates.py), for K4
    obedience         JSON report of rir_obey.py, for K5

Roles: A1 trained on RGB, A2 on real IR (upper bound), B3 on the trivial
transform (grayscale, CLAHE, sensor model), C2 on the learned translator's
output, REAL_K a small real-IR training set alone, REAL_K_PLUS_SYNTHETIC the
same plus the translated images. The primary metric is mAP@0.5:0.95; mAP@0.5
is reported beside it. The statistics are in rir_bootstrap.py.

Pre-registered comparisons (one Holm family):
    P1  C2 - B3                          the learned translator against the trivial transform
    P2  REAL_K_PLUS_SYNTHETIC - REAL_K   synthetic images on top of a few real ones
    P3  g = (C2 - A1) / (A2 - A1)        the share of the real-IR gain C2 recovers, with a bootstrap interval
A comparison "favours" the first arm / is "positive" when it counts (rir_bootstrap)
and its difference is positive. g >= 0.5 and g < 0.25 are judged on the point estimate.

Kill criteria (K1 to K5 are triggered, not triggered, or not evaluable):
    K1  A2 - A1 < 3 mAP points (the premise fails: real IR does not help)
    K2  g < 0.25 and P2 is not positive
    K3  C2 is not better than B3 (P1 does not favour C2)
    K4  more than 30 percent of C2's images fail the geometry gates
    K5  (instruction model only) every region type's controllability interval includes 0

Verdict, by the rules and nothing else:
    "reliable transform"        P1 favours C2 AND g >= 0.5 AND P2 is positive
    "useful but not reliable"   P2 is positive AND g < 0.5
    "no reliable transform"     otherwise, naming the triggered kill criteria
    "not evaluable"             a missing arm or an unmeasured seed noise leaves the answer open
An arm without predictions is NOT measured: it does not appear as 0, and every rule that
needs it is not evaluable. Kill criteria that fire next to a "reliable transform" or a
"useful but not reliable" verdict are reported beside it, they do not change it.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

import rir_bootstrap as B
import rir_eval as E

ROLES = ("A1", "A2", "B3", "C2", "REAL_K", "REAL_K_PLUS_SYNTHETIC")
PRIMARY = 1  # index of mAP@0.5:0.95 in a (mAP@0.5, mAP@0.5:0.95) pair
K1_MIN_GAIN = 0.03
K2_MAX_G = 0.25
RELIABLE_MIN_G = 0.5
K4_MAX_FAIL_FRACTION = 0.30
CONFIG_KEYS = ("sequences", "roles", "resamples", "seed", "alpha", "nc", "instruction_model", "gate_statistics", "obedience")


class DecisionError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    sequences: str
    roles: dict
    resamples: int = B.DEFAULT_RESAMPLES
    seed: int = 1
    alpha: float = 0.05
    nc: int | None = None
    instruction_model: bool = False
    gate_statistics: str | None = None
    obedience: str | None = None


def load_config(path: str) -> Config:
    with open(path) as fh:
        doc = json.load(fh)
    unknown = sorted(set(doc) - set(CONFIG_KEYS))
    if unknown:
        raise DecisionError(f"{path}: unknown keys {unknown}; expected a subset of {list(CONFIG_KEYS)}")
    if "sequences" not in doc or "roles" not in doc:
        raise DecisionError(f"{path}: 'sequences' and 'roles' are required")
    bad = sorted(set(doc["roles"]) - set(ROLES))
    if bad:
        raise DecisionError(f"{path}: unknown roles {bad}; expected a subset of {list(ROLES)}")
    base = os.path.dirname(os.path.abspath(path))
    resolve = lambda p: None if p is None else p if os.path.isabs(p) else os.path.join(base, p)
    return Config(**{**doc, "sequences": resolve(doc["sequences"]), "gate_statistics": resolve(doc.get("gate_statistics")),
                     "obedience": resolve(doc.get("obedience"))})


# ------------------------------------------------------------------- results


def load_results(results_dir: str) -> dict[str, dict[str, list[E.Image]]]:
    """arm -> {dump file name -> images}, for the arms that have at least one image; others are absent."""
    arms = {}
    for arm_dir in sorted(glob.glob(os.path.join(results_dir, "*", ""))):
        arm = os.path.basename(os.path.dirname(arm_dir))
        runs = {os.path.basename(f): E.read_jsonl(f) for f in sorted(glob.glob(os.path.join(arm_dir, "*.jsonl")))}
        runs = {name: images for name, images in runs.items() if images}
        if runs:
            arms[arm] = runs
    return arms


def _check_same_test_set(arms: dict) -> list[E.Image]:
    """Every run must be scored on the same images with the same ground truth, else the arms are not paired."""
    reference, ref_name = None, ""
    for arm, runs in arms.items():
        for name, images in runs.items():
            if reference is None:
                reference, ref_name = images, f"{arm}/{name}"
                continue
            same = len(images) == len(reference) and all(
                a.index == b.index and np.array_equal(a.gt_class, b.gt_class) and np.array_equal(a.gt_box, b.gt_box)
                for a, b in zip(images, reference))
            if not same:
                raise DecisionError(f"{arm}/{name} was not scored on the same images and ground truth as {ref_name}; "
                                    "all arms and seeds must be evaluated on one packed held-out set")
    return reference or []


# ------------------------------------------------------------------ decision


@dataclass(frozen=True)
class Estimate:
    estimate: float
    ci: tuple[float, float]
    seed_sd: float | None


@dataclass(frozen=True)
class ArmReport:
    seeds: int
    map50: Estimate
    map50_95: Estimate


@dataclass(frozen=True)
class Kill:
    rule: str
    triggered: bool | None  # None: not evaluable, or not applicable
    applicable: bool
    detail: str


@dataclass
class Decision:
    verdict: str
    verdict_detail: str
    arms: dict
    not_measured: list
    comparisons: dict  # P1, P2, P3 -> rir_bootstrap.Judgement
    gap_closure: B.Contrast
    kills: dict
    method: dict
    roles: dict = field(default_factory=dict)


def _estimate(arm: B.ArmScore, metric: int, alpha: float) -> Estimate:
    sd = arm.seed_sd[metric]
    return Estimate(float(arm.estimate[metric]), B.percentile_ci(arm.replicates[:, metric], alpha),
                    None if np.isnan(sd) else float(sd))


def _unavailable(*missing: str) -> B.Contrast:
    names = ", ".join(repr(m) for m in missing)
    return B.Contrast(float("nan"), np.zeros(0), None, False, f"not evaluable: arm {names} not measured")


def _and3(a, b):
    """Kleene conjunction over True / False / None (unknown)."""
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


def _not3(a):
    return None if a is None else not a


def _favours_first(j: B.Judgement):
    """True: the comparison counts and the first arm is ahead; False: it does not; None: cannot be told."""
    return None if j.counts is None else bool(j.counts and j.direction > 0)


def _load_json(path: str | None):
    if path is None:
        return None
    with open(path) as fh:
        return json.load(fh)


def _kill_k1(arms, role):
    rule = "K1: A2 - A1 < 3 mAP points (the premise fails)"
    a1, a2 = role("A1"), role("A2")
    if a1 not in arms or a2 not in arms:
        return Kill(rule, None, True, "A1 or A2 not measured")
    gain = arms[a2].map50_95.estimate - arms[a1].map50_95.estimate
    return Kill(rule, gain < K1_MIN_GAIN, True, f"A2 - A1 = {gain * 100:+.1f} points")


def _kill_k4(stats):
    rule = f"K4: more than {K4_MAX_FAIL_FRACTION:.0%} of C2 images fail the geometry gates"
    if stats is None:
        return Kill(rule, None, True, "no gate statistics given")
    if not stats.get("n_images"):
        return Kill(rule, None, True, "gate statistics cover no images")
    fraction = stats["n_failed"] / stats["n_images"]
    return Kill(rule, fraction > K4_MAX_FAIL_FRACTION, True,
                f"{stats['n_failed']} of {stats['n_images']} images fail ({fraction:.1%})")


def _kill_k5(config: Config, report):
    rule = "K5: controllability interval includes 0 for every region type (instruction model only)"
    if not config.instruction_model:
        return Kill(rule, None, False, "not an instruction model")
    types = (report or {}).get("region_types") or {}
    effects = {t: v["effect"] for t, v in types.items() if v.get("effect")}
    if not effects:
        return Kill(rule, None, True, "no obedience measurement given")
    includes_zero = {t: e["ci"][0] <= 0 <= e["ci"][1] for t, e in effects.items()}
    detail = ", ".join(f"{t}: {'includes 0' if z else 'excludes 0'}" for t, z in sorted(includes_zero.items()))
    return Kill(rule, all(includes_zero.values()), True, detail)


def _verdict(p1, p2, g, premise_fails):
    """Kleene evaluation of the three verdict rules. p1/p2: favours/positive; g: point estimate or None."""
    if premise_fails is True:
        g_ge, g_lt = False, False  # no gain to recover: g is undefined and neither rule can hold
    else:
        g_ge = None if g is None else g >= RELIABLE_MIN_G
        g_lt = None if g is None else g < RELIABLE_MIN_G
    reliable = _and3(_and3(p1, g_ge), p2)
    useful = _and3(p2, g_lt)
    if reliable is True:
        return "reliable transform"
    if useful is True:
        return "useful but not reliable"
    if reliable is False and useful is False:
        return "no reliable transform"
    return "not evaluable"


def decide(results_dir: str, config: Config) -> Decision:
    arms_images = load_results(results_dir)
    reference = _check_same_test_set(arms_images)
    role = lambda r: config.roles.get(r)
    on_disk = sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))
    wanted = sorted({a for a in config.roles.values()} | set(on_disk))
    not_measured = [a for a in wanted if a not in arms_images]

    scores, arm_reports, method = {}, {}, {}
    if arms_images:
        sequences = E.sequence_ids(reference, E.load_sequences(config.sequences))
        weights = B.resample_weights(sequences, config.resamples, config.seed)
        for arm, runs in arms_images.items():
            scores[arm] = B.score_arm(arm, [E.prepare(images) for images in runs.values()], weights, config.nc)
            arm_reports[arm] = ArmReport(len(runs), _estimate(scores[arm], 0, config.alpha),
                                         _estimate(scores[arm], PRIMARY, config.alpha))
        method = {"unit": "sequence", "n_sequences": len(set(sequences)), "n_images": len(reference),
                  "resamples": config.resamples, "seed": config.seed, "alpha": config.alpha,
                  "seeds_per_arm": {a: r.seeds for a, r in arm_reports.items()}}

    def need(*roles):
        arms = [role(r) for r in roles]
        missing = [a if a else r for a, r in zip(arms, roles) if a not in scores]
        return arms, missing

    contrasts = {}
    for name, (first, second) in {"P1": ("C2", "B3"), "P2": ("REAL_K_PLUS_SYNTHETIC", "REAL_K")}.items():
        (a, b), missing = need(first, second)
        contrasts[name] = _unavailable(*missing) if missing else B.difference(scores[a], scores[b], PRIMARY)
    (c2, a1, a2), missing = need("C2", "A1", "A2")
    g = _unavailable(*missing) if missing else B.gap_closure(scores[c2], scores[a1], scores[a2], PRIMARY)
    contrasts["P3"] = g
    comparisons = B.judge(contrasts, config.alpha)

    kills = {"K1": _kill_k1(arm_reports, role)}
    g_value = g.estimate if g.evaluable else None
    p1, p2 = _favours_first(comparisons["P1"]), _favours_first(comparisons["P2"])
    k2_rule = "K2: g < 0.25 and P2 is not positive"
    g_small = None if g_value is None else g_value < K2_MAX_G
    k2 = _and3(g_small, _not3(p2))
    kills["K2"] = Kill(k2_rule, k2, True, f"g = {g_value:.2f}, P2 {'positive' if p2 else 'not positive' if p2 is False else 'not evaluable'}"
                       if g_value is not None else "g not evaluable")
    kills["K3"] = Kill("K3: C2 is not better than B3 (P1 does not favour C2)", _not3(p1), True,
                       comparisons["P1"].reason or ("P1 favours C2" if p1 else "P1 does not favour C2"))
    kills["K4"] = _kill_k4(_load_json(config.gate_statistics))
    kills["K5"] = _kill_k5(config, _load_json(config.obedience))

    verdict = _verdict(p1, p2, g_value, kills["K1"].triggered)
    detail = _verdict_detail(verdict, comparisons, g, kills, not_measured)
    return Decision(verdict, detail, arm_reports, not_measured, comparisons, g, kills, method, dict(config.roles))


def _verdict_detail(verdict, comparisons, g, kills, not_measured) -> str:
    triggered = [k for k, v in kills.items() if v.triggered]
    if verdict == "no reliable transform":
        unmet = [f"{name}: {j.reason}" for name, j in comparisons.items() if name != "P3" and j.counts is False and j.reason]
        return ("kill criteria triggered: " + ", ".join(triggered)) if triggered else \
            "no kill criterion triggered; unmet: " + "; ".join(unmet or ["g < 0.5 with P2 not positive"])
    if verdict == "not evaluable":
        open_items = [f"{n}: {j.reason}" for n, j in comparisons.items() if j.counts is None and j.reason]
        if not g.evaluable and g.why_not and not any(g.why_not in o for o in open_items):
            open_items.append(f"g: {g.why_not}")
        return "; ".join(open_items) + (f"; arms not measured: {', '.join(not_measured)}" if not_measured else "")
    note = f"; kill criteria triggered beside it: {', '.join(triggered)}" if triggered else ""
    return f"g = {g.estimate:.2f}" + note


# -------------------------------------------------------------------- report


def _points(x: float) -> str:
    return f"{100 * x:.1f}"


def _est(e: Estimate) -> str:
    return f"{_points(e.estimate)} [{_points(e.ci[0])}, {_points(e.ci[1])}]"


def render_markdown(d: Decision) -> str:
    out = [f"# RGB-to-IR decision\n", f"**Verdict: {d.verdict}**\n", f"{d.verdict_detail}\n"]
    out.append("## Arms\n")
    out.append("mAP in points on the held-out real IR frames; the interval is the 95 percent cluster-bootstrap interval of the "
               "mean over seeds. Seed std is across independent trainings.\n")
    out.append("| arm | seeds | mAP@0.5:0.95 | seed std | mAP@0.5 | seed std |\n|---|---|---|---|---|---|")
    sd = lambda e: "n/a" if e.seed_sd is None else _points(e.seed_sd)
    for arm, r in sorted(d.arms.items()):
        out.append(f"| {arm} | {r.seeds} | {_est(r.map50_95)} | {sd(r.map50_95)} | {_est(r.map50)} | {sd(r.map50)} |")
    for arm in d.not_measured:
        out.append(f"| {arm} | 0 | not measured | | not measured | |")
    arm_of = lambda role: d.roles.get(role, role)
    out.append("\n## Pre-registered comparisons\n")
    out.append("Differences in mAP@0.5:0.95 points (P3 is a ratio). A comparison counts only if its interval excludes 0, "
               "it exceeds 2 x the seed std of the difference and its Holm-adjusted p (family of 3) is below 0.05.\n")
    out.append("| comparison | estimate | 95% CI | seed std | p | p (Holm) | counts |\n|---|---|---|---|---|---|---|")
    titles = {"P1": f"P1 {arm_of('C2')} - {arm_of('B3')}", "P2": f"P2 {arm_of('REAL_K_PLUS_SYNTHETIC')} - {arm_of('REAL_K')}",
              "P3": f"P3 g = ({arm_of('C2')} - {arm_of('A1')}) / ({arm_of('A2')} - {arm_of('A1')})"}
    for name, j in d.comparisons.items():
        if j.estimate is None:
            out.append(f"| {titles[name]} | not evaluable | | | | | {j.reason} |")
            continue
        scale = (lambda x: f"{x:.2f}") if name == "P3" else _points
        counts = {True: "yes", False: "no", None: "not evaluable"}[j.counts] + (f" ({j.reason})" if j.reason else "")
        out.append(f"| {titles[name]} | {scale(j.estimate)} | [{scale(j.ci[0])}, {scale(j.ci[1])}] | "
                   f"{'n/a' if j.seed_sd is None else scale(j.seed_sd)} | {j.p_raw:.4f} | {j.p_holm:.4f} | {counts} |")
    out.append("\n## Kill criteria\n")
    out.append("| criterion | status | detail |\n|---|---|---|")
    for k in d.kills.values():
        status = "not applicable" if not k.applicable else {True: "TRIGGERED", False: "not triggered", None: "not evaluable"}[k.triggered]
        out.append(f"| {k.rule} | {status} | {k.detail} |")
    if d.method:
        m = d.method
        out.append(f"\n## Method\n\nResampling unit: sequence ({m['n_sequences']} sequences, {m['n_images']} images); "
                   f"{m['resamples']} resamples, seed {m['seed']}; one set of resamples scores every run of every arm "
                   "(paired); an arm's value on a resample is the mean of its seeds.")
    return "\n".join(out) + "\n"


def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x) if f.name != "replicates"}
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    return None if isinstance(x, float) and x != x else x  # JSON has no NaN: an unmeasured number is null


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apply the pre-registered decision rules to a results directory.")
    ap.add_argument("results", help="directory with one subdirectory of seed prediction dumps per arm")
    ap.add_argument("--config", required=True, help="decision config JSON")
    ap.add_argument("--out", required=True, help="directory for decision.md and decision.json")
    a = ap.parse_args(argv)
    try:
        d = decide(a.results, load_config(a.config))
    except (DecisionError, FileNotFoundError, ValueError) as e:
        print(f"rir_decide: {e}", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "decision.md"), "w") as fh:
        fh.write(render_markdown(d))
    with open(os.path.join(a.out, "decision.json"), "w") as fh:
        json.dump(_jsonable(d), fh, indent=1)
    print(f"verdict: {d.verdict} ({d.verdict_detail})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
