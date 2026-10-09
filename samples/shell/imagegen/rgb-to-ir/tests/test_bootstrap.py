# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_bootstrap: the paired cluster bootstrap over sequences,
the seed-noise rule and the Holm correction."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import predsynth as P  # noqa: E402
import rir_bootstrap as B  # noqa: E402
import rir_eval as E  # noqa: E402

PRIMARY = 1  # mAP@0.5:0.95


def astuple_tail(im):
    return im.gt_class, im.gt_box, im.pred_class, im.pred_score, im.pred_box


def contrast(ds, skill_a, skill_b, n_resamples=300, seed=7, salt_a=1, salt_b=2):
    """arm A minus arm B on one dataset, three seeds each."""
    weights = B.resample_weights(ds.sequence_of_image, n_resamples, seed)
    a = B.score_arm("A", P.arm(ds, skill_a, salt=salt_a), weights)
    b = B.score_arm("B", P.arm(ds, skill_b, salt=salt_b), weights)
    return B.difference(a, b, PRIMARY)


class Resampling(unittest.TestCase):
    def test_every_image_of_a_sequence_gets_the_same_weight_and_weights_sum_to_the_set(self):
        seqs = ["a", "a", "a", "b", "b", "c"]
        w = B.resample_weights(seqs, 50, seed=1)
        self.assertEqual(w.shape, (50, 6))
        self.assertTrue((w[:, 0] == w[:, 1]).all() and (w[:, 1] == w[:, 2]).all() and (w[:, 3] == w[:, 4]).all())
        # three sequences are drawn per resample, so the sequence weights sum to 3
        self.assertTrue(((w[:, 0] + w[:, 3] + w[:, 5]) == 3).all())

    def test_seeded_and_different_across_seeds(self):
        seqs = [f"s{i // 2}" for i in range(20)]
        self.assertTrue((B.resample_weights(seqs, 20, 5) == B.resample_weights(seqs, 20, 5)).all())
        self.assertFalse((B.resample_weights(seqs, 20, 5) == B.resample_weights(seqs, 20, 6)).all())


class PlantedEffect(unittest.TestCase):
    def test_a_true_improvement_is_detected(self):
        c = contrast(P.Dataset(40, 3, seed=11), skill_a=2.5, skill_b=0.0)
        r = B.judge({"P": c})["P"]
        self.assertGreater(r.estimate, 0.05)
        self.assertGreater(r.ci[0], 0.0)
        self.assertTrue(r.counts)
        self.assertGreater(r.estimate, 2 * c.seed_sd)

    def test_equal_arms_are_not_a_difference(self):
        c = contrast(P.Dataset(40, 3, seed=11), skill_a=1.0, skill_b=1.0)
        r = B.judge({"P": c})["P"]
        self.assertFalse(r.counts)

    def test_identical_arms_have_an_exactly_zero_interval(self):
        # Both arms are scored on the very same resamples, so a shared run cancels completely.
        ds = P.Dataset(20, 2, seed=3)
        weights = B.resample_weights(ds.sequence_of_image, 100, 1)
        runs = P.arm(ds, 1.0)
        c = B.difference(B.score_arm("A", runs, weights), B.score_arm("B", runs, weights), PRIMARY)
        self.assertEqual(B.percentile_ci(c.replicates), (0.0, 0.0))

    def test_ci_covers_the_truth_about_95_percent_of_the_time_on_null_data(self):
        covered, sims = 0, 120
        for k in range(sims):
            ds = P.Dataset(40, 2, seed=100 + k)
            weights = B.resample_weights(ds.sequence_of_image, 200, k)
            a = B.score_arm("A", P.arm(ds, 1.0, seeds=(1,), salt=2 * k + 1), weights)
            b = B.score_arm("B", P.arm(ds, 1.0, seeds=(1,), salt=2 * k + 2), weights)
            lo, hi = B.percentile_ci(B.difference(a, b, PRIMARY).replicates)
            covered += lo <= 0.0 <= hi
        # The simulations are seeded, so this is not flaky; the band is about 3 standard errors (2 points) around 95.
        self.assertGreaterEqual(covered / sims, 0.88, covered)
        self.assertLessEqual(covered / sims, 0.99, covered)


class ClusterUnit(unittest.TestCase):
    def test_a_leak_of_one_frame_per_sequence_does_not_shrink_the_interval(self):
        # Every frame gets an exact near-duplicate in its own sequence: no new
        # information, so the interval must not move. Frame-level resampling
        # takes the copies for evidence and narrows it.
        ds = P.Dataset(40, 1, seed=5)

        def width(copies, unit):
            runs = [[im for im in images for _ in range(copies)] for images in
                    (P.detect(ds, 1.5, 1), P.detect(ds, 0.5, 2))]
            runs = [[E.Image(i, *astuple_tail(im)) for i, im in enumerate(r)] for r in runs]
            sequences = [s for s in ds.sequence_of_image for _ in range(copies)]
            clusters = sequences if unit == "sequence" else [str(i) for i in range(len(sequences))]
            weights = B.resample_weights(clusters, 400, 3)
            a, b = (B.score_arm(n, [E.prepare(r)], weights) for n, r in zip("AB", runs))
            lo, hi = B.percentile_ci(B.difference(a, b, PRIMARY).replicates)
            return hi - lo

        clean, leaked = width(1, "sequence"), width(2, "sequence")
        self.assertLess(abs(np.log(leaked / clean)), np.log(1.05), (clean, leaked))
        self.assertLess(width(2, "frame"), 0.85 * clean)

    def test_frames_of_a_sequence_cannot_be_split_by_a_resample(self):
        ds = P.Dataset(10, 4, seed=2)
        w = B.resample_weights(ds.sequence_of_image, 30, 9).reshape(30, 10, 4)
        self.assertTrue((w == w[:, :, :1]).all())


class SeedNoise(unittest.TestCase):
    def test_seed_std_is_reported_apart_from_the_interval(self):
        ds = P.Dataset(30, 2, seed=1)
        weights = B.resample_weights(ds.sequence_of_image, 100, 1)
        a = B.score_arm("A", P.arm(ds, 2.0, salt=1), weights)
        self.assertEqual(a.per_seed.shape, (3, 2))
        self.assertAlmostEqual(a.seed_sd[PRIMARY], float(np.std(a.per_seed[:, PRIMARY], ddof=1)), places=12)
        self.assertAlmostEqual(a.estimate[PRIMARY], float(a.per_seed[:, PRIMARY].mean()), places=12)

    def test_a_difference_within_twice_the_seed_std_does_not_count(self):
        # Interval clear of zero, but smaller than two seed standard deviations.
        reps = np.random.default_rng(0).normal(0.02, 0.003, 500)
        small = B.Contrast(0.02, reps, seed_sd=0.02)
        large = B.Contrast(0.02, reps, seed_sd=0.005)
        self.assertFalse(B.judge({"P": small})["P"].counts)
        self.assertTrue(B.judge({"P": large})["P"].counts)

    def test_one_seed_leaves_the_rule_not_evaluable_not_passed(self):
        reps = np.random.default_rng(0).normal(0.05, 0.003, 500)
        r = B.judge({"P": B.Contrast(0.05, reps, seed_sd=None)})["P"]
        self.assertIsNone(r.counts)
        self.assertIn("seed", r.reason)
        self.assertGreater(r.ci[0], 0.0)  # the interval is still reported


class Holm(unittest.TestCase):
    def test_adjusted_p_values_follow_holm_step_down(self):
        adj = B.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)  # 0.04 * 1, raised to the previous adjusted value

    def test_adjusted_p_never_exceeds_one(self):
        self.assertEqual(B.holm({"a": 0.6, "b": 0.7}), {"a": 1.0, "b": 1.0})  # 2 * 0.6 is capped, and 0.7 follows it

    def test_the_family_is_corrected_before_deciding(self):
        # Each alone has p = 0.04 and counts; three tested together do not.
        reps = np.concatenate([np.full(10, -0.001), np.full(490, 0.05)])  # 2% of replicates at or below zero
        family = {k: B.Contrast(0.05, reps, seed_sd=0.001) for k in ("P1", "P2", "P3")}
        alone = B.judge({"P1": family["P1"]})["P1"]
        self.assertTrue(alone.counts)
        together = B.judge(family)
        self.assertAlmostEqual(together["P1"].p_holm, 3 * alone.p_raw)
        self.assertFalse(any(r.counts for r in together.values()))


    def test_a_member_that_cannot_be_tested_still_counts_in_the_family(self):
        # The family is pre-registered: an untestable member is an untested hypothesis, not a smaller family.
        reps = np.concatenate([np.full(5, -0.001), np.full(495, 0.05)])
        gone = B.Contrast(float("nan"), np.zeros(0), None, False, "arm not measured")
        two = B.judge({"P1": B.Contrast(0.05, reps, 0.001), "P2": gone})
        self.assertAlmostEqual(two["P1"].p_holm, 2 * two["P1"].p_raw)
        self.assertIsNone(two["P2"].counts)
        self.assertEqual(two["P2"].reason, "arm not measured")


class GapClosure(unittest.TestCase):
    def test_gap_closure_is_the_share_of_the_real_ir_gain_a_method_recovers(self):
        ds = P.Dataset(40, 3, seed=21)
        weights = B.resample_weights(ds.sequence_of_image, 300, 4)
        score = lambda skill, salt: B.score_arm("x", P.arm(ds, skill, salt=salt), weights)
        a1, a2, c2 = score(0.0, 1), score(3.0, 2), score(1.5, 3)
        g = B.gap_closure(c2, a1, a2, PRIMARY)
        want = (c2.estimate[PRIMARY] - a1.estimate[PRIMARY]) / (a2.estimate[PRIMARY] - a1.estimate[PRIMARY])
        self.assertAlmostEqual(g.estimate, want, places=12)
        self.assertTrue(0.2 < g.estimate < 0.8, g.estimate)
        # seed noise in the units of g: the numerator's, over the gap
        num = B.difference(c2, a1, PRIMARY)
        self.assertAlmostEqual(g.seed_sd, num.seed_sd / (a2.estimate[PRIMARY] - a1.estimate[PRIMARY]), places=12)
        self.assertLess(g.ci[0], g.estimate)
        self.assertGreater(g.ci[1], g.estimate)

    def test_a_denominator_that_may_be_zero_makes_the_ratio_not_evaluable(self):
        ds = P.Dataset(30, 2, seed=2)
        weights = B.resample_weights(ds.sequence_of_image, 300, 4)
        score = lambda skill, salt: B.score_arm("x", P.arm(ds, skill, salt=salt), weights)
        g = B.gap_closure(score(1.0, 3), score(1.0, 1), score(1.0, 2), PRIMARY)
        self.assertFalse(g.evaluable)
        self.assertIn("A2 - A1", g.why_not)


if __name__ == "__main__":
    unittest.main()
