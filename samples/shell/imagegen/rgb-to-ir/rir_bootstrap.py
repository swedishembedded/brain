# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Paired cluster bootstrap of mAP differences between detector arms.

The unit of resampling is the SEQUENCE. Frames of one video are near-duplicates,
so a frame-level bootstrap would count the same evidence many times and give an
interval that is too narrow; and a leak of one extra frame per sequence would
shrink it further without adding any information. A resample draws as many
sequences as the test set has, with replacement, and every image of a drawn
sequence is counted as often as its sequence was drawn (`resample_weights`).

Seeds. An arm is trained from several seeds and each seed gives its own
prediction dump. ONE set of resamples is drawn and EVERY run of EVERY arm is
scored on it; an arm's value on a resample is the mean of its seeds' mAP. The
arms are therefore paired (what makes a sequence hard hits all of them alike
and cancels in the difference), and the interval carries the test-set
sampling uncertainty of the seed average. Seed-to-seed variation is a
different quantity - how much one training differs from the next - and is
reported apart, as the sample standard deviation (n - 1) of the seeds' mAP on
the full test set, per arm.

The percentile interval is the 2.5th to 97.5th percentile of the resampled
difference. A difference COUNTS only if
  1. its interval excludes 0,
  2. its size exceeds twice the seed noise of a difference of two single
     trainings, 2 * sqrt(sd_a^2 + sd_b^2) (not evaluable with one seed in
     either arm: the noise has not been measured), and
  3. its two-sided bootstrap p-value survives Holm's step-down correction over
     the family of pre-registered comparisons.
The p-value is 2 * min(share of resamples <= 0, share >= 0), each share with
one added to numerator and denominator so it cannot be 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import rir_eval as E

DEFAULT_RESAMPLES = 2000
SEED_NOISE_FACTOR = 2.0
MAX_NONPOSITIVE_GAP_SHARE = 0.05  # share of resamples whose denominator may be non-positive


def resample_weights(sequences: list[str], n_resamples: int = DEFAULT_RESAMPLES, seed: int = 1) -> np.ndarray:
    """(n_resamples, n_images): how often each image is counted in each resample of whole sequences."""
    ids, cluster = np.unique(np.asarray(sequences), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
    counts = np.stack([np.bincount(d, minlength=len(ids)) for d in draws])
    return counts[:, cluster].astype(np.float64)


@dataclass(frozen=True)
class ArmScore:
    """One arm: its seeds' runs scored on the full test set and on every resample."""
    name: str
    per_seed: np.ndarray  # (n_seeds, 2): mAP@0.5, mAP@0.5:0.95 on the full test set
    replicates: np.ndarray  # (n_resamples, 2): the seeds' mean on each resample

    @property
    def estimate(self) -> np.ndarray:
        return self.per_seed.mean(axis=0)

    @property
    def seed_sd(self) -> np.ndarray:
        """Sample standard deviation over seeds, per metric; NaN with fewer than two seeds."""
        if len(self.per_seed) < 2:
            return np.full(2, np.nan)
        return self.per_seed.std(axis=0, ddof=1)


def score_arm(name: str, runs: list[E.Prepared], weights: np.ndarray, nc: int | None = None) -> ArmScore:
    if not runs:
        raise ValueError(f"arm {name!r} has no runs")
    ones = np.ones(runs[0].n_images)
    per_seed = np.array([run.maps(ones, nc) for run in runs])
    replicates = np.mean([[run.maps(w, nc) for w in weights] for run in runs], axis=0)
    return ArmScore(name, per_seed, replicates)


@dataclass(frozen=True)
class Contrast:
    """A statistic of the resamples (a difference of arms, or a ratio of such)."""
    estimate: float
    replicates: np.ndarray
    seed_sd: float | None  # noise of the statistic from retraining; None when not measured
    evaluable: bool = True
    why_not: str = ""

    @property
    def ci(self) -> tuple[float, float]:
        return percentile_ci(self.replicates)


def difference(a: ArmScore, b: ArmScore, metric: int) -> Contrast:
    """a - b for one metric (0: mAP@0.5, 1: mAP@0.5:0.95), on the shared resamples."""
    if a.replicates.shape != b.replicates.shape:
        raise ValueError("arms were scored on different resamples")
    sd = float(np.hypot(a.seed_sd[metric], b.seed_sd[metric]))
    return Contrast(float(a.estimate[metric] - b.estimate[metric]), a.replicates[:, metric] - b.replicates[:, metric],
                    None if np.isnan(sd) else sd)


def gap_closure(method: ArmScore, floor: ArmScore, ceiling: ArmScore, metric: int) -> Contrast:
    """g = (method - floor) / (ceiling - floor): the share of the gain real IR brings that `method` recovers.

    Evaluable only when ceiling - floor is positive in the full sample and in
    all but a few resamples; otherwise the ratio has no meaning and says so.
    Its seed noise is that of the numerator method - floor, in units of g.
    """
    num, den = difference(method, floor, metric), difference(ceiling, floor, metric)
    positive = den.replicates > 0
    if den.estimate <= 0 or (1 - positive.mean()) > MAX_NONPOSITIVE_GAP_SHARE:
        return Contrast(float("nan"), np.zeros(0), None, False,
                        f"A2 - A1 = {den.estimate:+.4f} is not reliably positive "
                        f"({(1 - positive.mean()):.0%} of resamples are not): the gap to close is not established")
    return Contrast(num.estimate / den.estimate, num.replicates[positive] / den.replicates[positive],
                    None if num.seed_sd is None else num.seed_sd / den.estimate)


def percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_p(replicates: np.ndarray) -> float:
    n = len(replicates)
    below, above = (np.sum(replicates <= 0) + 1) / (n + 1), (np.sum(replicates >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjusted p-values: the k-th smallest of m is multiplied by m - k + 1, made monotone."""
    adjusted, running = {}, 0.0
    for rank, name in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (len(p) - rank) * p[name]))
        adjusted[name] = running
    return adjusted


@dataclass(frozen=True)
class Judgement:
    estimate: float | None
    ci: tuple[float, float] | None
    p_raw: float | None
    p_holm: float | None
    seed_sd: float | None
    counts: bool | None  # True: a difference; False: not shown; None: not evaluable
    direction: int  # sign of the estimate when it counts, else 0
    reason: str = field(default="")


def judge(contrasts: dict[str, Contrast], alpha: float = 0.05) -> dict[str, Judgement]:
    """Apply the three conditions of the module docstring to a pre-registered family.

    Holm is taken over the whole family: a member that cannot be tested
    (an arm is missing) is an untested hypothesis with p = 1, so it still
    counts toward the size of the family.
    """
    p_raw = {k: bootstrap_p(c.replicates) if c.evaluable else 1.0 for k, c in contrasts.items()}
    p_holm = holm(p_raw)
    out = {}
    for name, c in contrasts.items():
        if not c.evaluable:
            out[name] = Judgement(None, None, None, None, None, None, 0, c.why_not)
            continue
        lo, hi = c.ci
        excludes_zero = lo > 0 or hi < 0
        survives = p_holm[name] < alpha
        if c.seed_sd is None:
            counts, reason = None, "seed noise not measured: needs at least two seeds in each arm"
        else:
            beyond_noise = abs(c.estimate) > SEED_NOISE_FACTOR * c.seed_sd
            counts = bool(excludes_zero and beyond_noise and survives)
            failed = [text for ok, text in ((excludes_zero, "interval includes 0"),
                                            (beyond_noise, f"not beyond {SEED_NOISE_FACTOR:g} x seed std"),
                                            (survives, "does not survive Holm correction")) if not ok]
            reason = "; ".join(failed)
        out[name] = Judgement(c.estimate, (lo, hi), p_raw[name], p_holm[name], c.seed_sd, counts,
                              (1 if c.estimate > 0 else -1) if counts else 0, reason)
    return out
