# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_decide: the pre-registered comparisons, the kill
criteria and every branch of the verdict, on synthetic detector outputs with
planted quality (tests/predsynth.py)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import predsynth as P  # noqa: E402
import rir_decide as D  # noqa: E402

DS = P.Dataset(40, 2, seed=31)
ROLE_ARM = {"A1": "a1", "A2": "a2", "B3": "b3", "C2": "c2", "REAL_K": "real-k", "REAL_K_PLUS_SYNTHETIC": "real-k+syn"}
SALT = {role: i + 1 for i, role in enumerate(ROLE_ARM)}
_dumps = {}


def dump_of(role, skill, seed):
    if (role, skill, seed) not in _dumps:
        _dumps[(role, skill, seed)] = P.detect(DS, skill, 1000 * SALT[role] + seed)
    return _dumps[(role, skill, seed)]


# skill (logit) of each arm; a role mapped to None is not run
RELIABLE = {"A1": 0.0, "A2": 4.0, "B3": 0.2, "C2": 2.6, "REAL_K": 1.0, "REAL_K_PLUS_SYNTHETIC": 2.4}


class Study:
    def __init__(self, skills, seeds=(1, 2, 3), instruction_model=False, gates=None, obedience=None):
        self.tmp = tempfile.mkdtemp()
        self.results = os.path.join(self.tmp, "results")
        for role, skill in skills.items():
            if skill is None:
                continue
            arm_dir = os.path.join(self.results, ROLE_ARM[role])
            os.makedirs(arm_dir)
            for seed in seeds:
                P.write_dump(os.path.join(arm_dir, f"seed{seed}.jsonl"), dump_of(role, skill, seed))
        self.config = {"sequences": "sequences.json", "roles": ROLE_ARM, "resamples": 200, "seed": 1, "nc": 1,
                       "instruction_model": instruction_model}
        with open(os.path.join(self.tmp, "sequences.json"), "w") as fh:
            json.dump(DS.sequence_of_image, fh)
        for key, doc in (("gate_statistics", gates), ("obedience", obedience)):
            if doc is not None:
                with open(os.path.join(self.tmp, f"{key}.json"), "w") as fh:
                    json.dump(doc, fh)
                self.config[key] = f"{key}.json"
        self.config_path = os.path.join(self.tmp, "decision-config.json")
        with open(self.config_path, "w") as fh:
            json.dump(self.config, fh)

    def decide(self):
        return D.decide(self.results, D.load_config(self.config_path))


def gates(n_failed, n=100):
    return {"n_images": n, "n_failed": n_failed, "by_reason": {}}


def obedience(*cis, source="oracle"):
    """The shape rir_obey.evaluate writes: instruction source -> region type -> effect."""
    return {"sources": {source: {"region_types": {
        f"type{i}": {"effect": {"measure": "controllability", "estimate": 0.0, "ci": list(ci), "n": 30}}
        for i, ci in enumerate(cis)}}}}


class Verdicts(unittest.TestCase):
    def test_reliable_transform_needs_p1_p2_and_half_the_gap(self):
        d = Study(RELIABLE).decide()
        self.assertEqual(d.verdict, "reliable transform", d.verdict_detail)
        self.assertTrue(d.comparisons["P1"].counts and d.comparisons["P1"].direction > 0)
        self.assertTrue(d.comparisons["P2"].counts and d.comparisons["P2"].direction > 0)
        self.assertGreaterEqual(d.gap_closure.estimate, 0.5)
        self.assertFalse(any(k.triggered for k in d.kills.values()), d.kills)

    def test_useful_but_not_reliable_when_p2_is_positive_and_the_gap_stays_open(self):
        d = Study({**RELIABLE, "C2": 0.8}).decide()
        self.assertLess(d.gap_closure.estimate, 0.5)
        self.assertTrue(d.comparisons["P2"].counts)
        self.assertEqual(d.verdict, "useful but not reliable")

    def test_p1_must_favour_c2_even_when_the_gap_and_p2_are_fine(self):
        # B3 as good as the learned translator: the trivial transform does the job.
        d = Study({**RELIABLE, "B3": 2.6}).decide()
        self.assertGreaterEqual(d.gap_closure.estimate, 0.5)
        self.assertTrue(d.comparisons["P2"].counts)
        self.assertFalse(d.comparisons["P1"].counts)
        self.assertEqual(d.verdict, "no reliable transform")
        self.assertTrue(d.kills["K3"].triggered)

    def test_no_reliable_transform_names_the_triggered_kill_criteria(self):
        d = Study({"A1": 0.0, "A2": 4.0, "B3": 0.5, "C2": 0.5, "REAL_K": 1.5, "REAL_K_PLUS_SYNTHETIC": 1.5}).decide()
        self.assertEqual(d.verdict, "no reliable transform")
        self.assertTrue(d.kills["K2"].triggered and d.kills["K3"].triggered)
        self.assertFalse(d.kills["K1"].triggered)
        md = D.render_markdown(d)
        self.assertRegex(md, r"(?s)no reliable transform.*K2.*K3")

    def test_a_negative_p2_is_not_positive(self):
        d = Study({**RELIABLE, "REAL_K_PLUS_SYNTHETIC": 0.2}).decide()  # synthetic data hurts
        self.assertLess(d.comparisons["P2"].estimate, 0)
        self.assertFalse(d.comparisons["P2"].direction > 0)
        self.assertEqual(d.verdict, "no reliable transform")


class KillCriteria(unittest.TestCase):
    def test_k1_premise_fails_when_real_ir_gains_under_three_points(self):
        d = Study({**RELIABLE, "A2": 0.05}).decide()
        self.assertTrue(d.kills["K1"].triggered)
        self.assertLess(d.arms["a2"].map50_95.estimate - d.arms["a1"].map50_95.estimate, 0.03)
        self.assertFalse(d.gap_closure.evaluable)  # no gap to close

    def test_k1_is_quiet_when_the_gap_is_large(self):
        self.assertFalse(Study(RELIABLE).decide().kills["K1"].triggered)

    def test_k2_needs_both_a_small_g_and_a_negative_p2(self):
        small_g_p2_positive = Study({**RELIABLE, "C2": 0.3}).decide()
        self.assertFalse(small_g_p2_positive.kills["K2"].triggered)
        small_g_p2_flat = Study({**RELIABLE, "C2": 0.3, "REAL_K_PLUS_SYNTHETIC": 1.0}).decide()
        self.assertTrue(small_g_p2_flat.kills["K2"].triggered)

    def test_k4_fires_above_thirty_percent_of_c2_images_failing_the_gates(self):
        over = Study(RELIABLE, gates=gates(31)).decide()
        at = Study(RELIABLE, gates=gates(30)).decide()
        self.assertTrue(over.kills["K4"].triggered)
        self.assertFalse(at.kills["K4"].triggered)
        # the rules are iff: the verdict word is unchanged, the kill is reported beside it
        self.assertEqual(over.verdict, "reliable transform")
        self.assertIn("K4", D.render_markdown(over))

    def test_k4_without_gate_statistics_is_not_evaluable_rather_than_passed(self):
        k4 = Study(RELIABLE).decide().kills["K4"]
        self.assertIsNone(k4.triggered)

    def test_k5_fires_only_when_every_region_type_includes_zero(self):
        every = Study(RELIABLE, instruction_model=True, obedience=obedience([-0.2, 0.3], [-0.1, 0.4])).decide()
        one_clear = Study(RELIABLE, instruction_model=True, obedience=obedience([-0.2, 0.3], [0.2, 0.9])).decide()
        self.assertTrue(every.kills["K5"].triggered)
        self.assertFalse(one_clear.kills["K5"].triggered)

    def test_k5_reads_the_oracle_instructions_unless_told_otherwise(self):
        prior_only = Study(RELIABLE, instruction_model=True, obedience=obedience([-0.2, 0.3], source="prior")).decide()
        self.assertIsNone(prior_only.kills["K5"].triggered)  # nothing measured with oracle instructions
        study = Study(RELIABLE, instruction_model=True, obedience=obedience([-0.2, 0.3], source="prior"))
        study.config["obedience_source"] = "prior"
        with open(study.config_path, "w") as fh:
            json.dump(study.config, fh)
        self.assertTrue(study.decide().kills["K5"].triggered)

    def test_k5_applies_to_the_instruction_model_only(self):
        d = Study(RELIABLE, instruction_model=False, obedience=obedience([-0.2, 0.3])).decide()
        self.assertFalse(d.kills["K5"].applicable)
        self.assertIsNone(d.kills["K5"].triggered)
        unmeasured = Study(RELIABLE, instruction_model=True).decide().kills["K5"]
        self.assertTrue(unmeasured.applicable)
        self.assertIsNone(unmeasured.triggered)  # no obedience measurement yet


class Unmeasured(unittest.TestCase):
    def test_an_arm_without_predictions_is_absent_not_zero(self):
        d = Study({**RELIABLE, "C2": None}).decide()
        self.assertNotIn("c2", d.arms)
        self.assertIn("c2", d.not_measured)
        md = D.render_markdown(d)
        self.assertRegex(md, r"c2.*not measured")
        self.assertEqual(d.comparisons["P1"].reason, "not evaluable: arm 'c2' not measured")
        self.assertIsNone(d.comparisons["P1"].counts)
        self.assertFalse(d.gap_closure.evaluable)
        self.assertIsNone(d.kills["K3"].triggered)

    def test_an_empty_dump_is_not_measured_either(self):
        study = Study({**RELIABLE, "C2": None})
        os.makedirs(os.path.join(study.results, "c2"))
        open(os.path.join(study.results, "c2", "seed1.jsonl"), "w").close()
        self.assertIn("c2", study.decide().not_measured)

    def test_a_detector_that_found_nothing_is_measured_and_scores_zero(self):
        study = Study({**RELIABLE, "C2": None})
        os.makedirs(os.path.join(study.results, "c2"))
        for seed in (1, 2, 3):
            silent = [P.E.Image(im.index, im.gt_class, im.gt_box, im.pred_class[:0], im.pred_score[:0], im.pred_box[:0])
                      for im in dump_of("C2", 0.0, seed)]
            P.write_dump(os.path.join(study.results, "c2", f"seed{seed}.jsonl"), silent)
        d = study.decide()
        self.assertEqual(d.arms["c2"].map50_95.estimate, 0.0)
        self.assertNotIn("c2", d.not_measured)

    def test_the_verdict_is_not_evaluable_when_a_missing_arm_could_change_it(self):
        d = Study({**RELIABLE, "C2": None}).decide()
        self.assertEqual(d.verdict, "not evaluable")
        self.assertIn("c2", d.verdict_detail)

    def test_the_verdict_still_stands_when_the_missing_arm_cannot_change_it(self):
        # P2 positive and g < 0.5 are known without B3, and that already decides "useful but not reliable".
        d = Study({**RELIABLE, "C2": 0.8, "B3": None}).decide()
        self.assertEqual(d.verdict, "useful but not reliable")
        self.assertIsNone(d.comparisons["P1"].counts)

    def test_p2_missing_while_the_gap_is_small_is_not_evaluable(self):
        d = Study({**RELIABLE, "C2": 0.3, "REAL_K": None, "REAL_K_PLUS_SYNTHETIC": None}).decide()
        self.assertEqual(d.verdict, "not evaluable")
        self.assertIsNone(d.kills["K2"].triggered)


class Reporting(unittest.TestCase):
    def test_the_report_carries_arm_intervals_comparisons_and_kills(self):
        d = Study(RELIABLE, gates=gates(10)).decide()
        md = D.render_markdown(d)
        for needle in ("mAP@0.5:0.95", "mAP@0.5", "P1", "P2", "P3", "K1", "K2", "K3", "K4", "K5", "Holm", "Verdict: reliable transform"):
            self.assertIn(needle, md)
        for arm in ("a1", "a2", "b3", "c2", "real-k", "real-k+syn"):
            self.assertIn(arm, md)
        ci = d.arms["c2"].map50_95.ci
        self.assertLessEqual(ci[0], d.arms["c2"].map50_95.estimate)
        self.assertGreaterEqual(ci[1], d.arms["c2"].map50_95.estimate)

    def test_holm_adjusted_p_values_are_not_below_the_raw_ones(self):
        for c in Study(RELIABLE).decide().comparisons.values():
            self.assertGreaterEqual(c.p_holm, c.p_raw)

    def test_main_writes_decision_md_and_json(self):
        study = Study(RELIABLE)
        out = os.path.join(study.tmp, "out")
        self.assertEqual(D.main([study.results, "--config", study.config_path, "--out", out]), 0)
        with open(os.path.join(out, "decision.json")) as fh:
            self.assertEqual(json.load(fh)["verdict"], "reliable transform")
        with open(os.path.join(out, "decision.md")) as fh:
            self.assertIn("reliable transform", fh.read())

    def test_runs_scored_on_different_test_sets_are_refused(self):
        study = Study(RELIABLE)
        path = os.path.join(study.results, "a1", "seed1.jsonl")
        with open(path) as fh:
            lines = fh.readlines()
        with open(path, "w") as fh:
            fh.writelines(lines[:-1])
        with self.assertRaisesRegex(D.DecisionError, "a1.*seed1"):
            study.decide()


if __name__ == "__main__":
    unittest.main()
