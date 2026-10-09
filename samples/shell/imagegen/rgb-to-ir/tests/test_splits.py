# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters and leakage-safe evaluation
# protocols for multimodal perception pipelines for its clients. If your team
# needs expertise in paired RGB / thermal-IR data, detector training sets or
# sensor-domain adaptation, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Spec tests for rir_splits: sequence-only splitting, no leakage, determinism."""
import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
import rir_data  # noqa: E402
import rir_readers as R  # noqa: E402
import rir_splits as S  # noqa: E402

PARAMS = S.SplitParams(default_s_cap=8, s_cap={"b": 12, "c": 20})


def records(dataset, args):
    recs, _ = R.build_records({"dataset": dataset, **args})
    return recs


def layout_a(root):
    """Videos with an official test list; ids 12-13 have a shrunk RGB."""
    args = synth.write_voc_pairs(
        root,
        train_videos=[
            {"ids": range(10, 20), "scene": 1, "level": (100, 200), "padded": {12, 13}},
            {"ids": range(20, 30), "scene": 2, "scenes": {n: 3 for n in range(25, 30)}, "level": (100, 200)},
            {"ids": list(range(40, 45)) + list(range(50, 55)), "scene": 4, "level": (100, 200)},
            {"ids": range(60, 65), "scene": 5, "level": (5, 50)},
            {"ids": range(70, 75), "scene": 6, "level": (150, 250)},
        ],
        test_videos=[{"ids": range(200, 205), "scene": 7, "level": (100, 200)}],
    )
    return records("a", args)


def layout_b(root):
    return records("b", synth.write_coco_rgbir(root, {f"{i:02d}": 10 + i for i in range(1, 11)}, {"19": 31, "20": 32}))


def layout_c(root, hint=False):
    """Interleaved ids of the same footage, cut into blocks of 100 with a guard of 10; no official hint."""
    args = synth.write_yolo_alpha(
        root,
        train={i: i // 100 + 1 for i in range(0, 400, 3)},
        val={i: i // 100 + 1 for i in range(1, 400, 3)},
    )
    args = {k: v for k, v in args.items() if k != "official_split_regex"}
    return records("c", {**args, "sequence_rule": "block:100", "block_guard": 10})


def rows_of(doc, **match):
    return [r for r in doc["frames"] if all(r[k] == v for k, v in match.items())]


class SplitInvariants(unittest.TestCase):
    def check_invariants(self, doc):
        seq_splits = {}
        for r in doc["frames"]:
            if r["usable"]:
                seq_splits.setdefault((r["dataset"], r["sequence_id"]), set()).add(r["split"])
        for key, splits in seq_splits.items():
            self.assertEqual(len(splits), 1, f"sequence {key} appears in {splits}")
        for r in doc["frames"]:
            self.assertEqual(r["split"] is not None, r["usable"], r)
            self.assertTrue(r["usable"] or r["reason"], r)

    def build_all(self, seed=1, params=PARAMS):
        recs = layout_a(tempfile.mkdtemp()) + layout_b(tempfile.mkdtemp()) + layout_c(tempfile.mkdtemp())
        doc, _ = S.build_splits(recs, seed=seed, params=params)
        return doc

    def test_no_sequence_in_two_splits_and_every_row_explained(self):
        self.check_invariants(self.build_all())

    def test_official_hints_make_test_and_only_test(self):
        doc = self.build_all()
        for r in rows_of(doc, dataset="a"):
            self.assertEqual(r["split"] == "Test", r["id"] >= "v00200" and r["usable"], r)
        for r in rows_of(doc, dataset="b"):
            self.assertEqual(r["split"] == "Test", r["id"][:2] in ("19", "20"), r)

    def test_without_a_hint_whole_sequences_are_held_out_by_seed(self):
        doc = self.build_all()
        c = rows_of(doc, dataset="c")
        self.assertTrue({r["split"] for r in c if r["usable"]} >= {"Test", "T"})
        test_seqs = {r["sequence_id"] for r in c if r["split"] == "Test"}
        self.assertEqual(len(test_seqs), 1)  # ceil(0.2 * 4 blocks)

    def test_forced_test_sequences_override_a_dataset_without_hint(self):
        recs = layout_c(tempfile.mkdtemp())
        doc, _ = S.build_splits(recs, params=S.SplitParams(test_sequences=("block00002",), default_s_cap=20))
        self.assertEqual({r["split"] for r in rows_of(doc, sequence_id="block00002") if r["usable"]}, {"Test"})
        self.assertNotIn("Test", {r["split"] for r in doc["frames"] if r["sequence_id"] != "block00002"})

    def test_deterministic_for_a_seed_and_seed_dependent(self):
        def plan(seed):  # paths differ between temp dirs; the assignment must not
            return [(r["dataset"], r["id"], r["sequence_id"], r["split"], r["reason"]) for r in self.build_all(seed)["frames"]]

        self.assertEqual(plan(3), plan(3))
        self.assertTrue(any(plan(s) != plan(3) for s in (4, 5, 6, 7)))


class SequenceSplitRules(unittest.TestCase):
    def setUp(self):
        self.doc, _ = S.build_splits(layout_a(tempfile.mkdtemp()), seed=1, params=PARAMS)

    def test_shrunk_rgb_frames_are_dropped_with_a_reason(self):
        for n in (12, 13):
            r = rows_of(self.doc, id=f"v{n:05d}")[0]
            self.assertFalse(r["usable"])
            self.assertIsNone(r["split"])
            self.assertEqual(r["reason"], "rgb_inset_padding")
        self.assertTrue(rows_of(self.doc, id="v00011")[0]["usable"])

    def test_day_night_follows_sequence_luminance_when_the_record_has_none(self):
        self.assertEqual(rows_of(self.doc, id="v00060")[0]["day_night"], "night")
        self.assertEqual(rows_of(self.doc, id="v00070")[0]["day_night"], "day")

    def test_a_recorded_day_night_wins(self):
        recs = layout_a(tempfile.mkdtemp())
        for r in recs:
            r["day_night"] = "night"
        doc, _ = S.build_splits(recs, params=PARAMS)
        self.assertEqual({r["day_night"] for r in doc["frames"] if r["usable"]}, {"night"})

    def test_s_respects_its_frame_cap_and_v_is_a_tenth_of_sequences(self):
        self.assertTrue(0 < len(rows_of(self.doc, split="S")) <= 8)
        self.assertEqual(len({r["sequence_id"] for r in rows_of(self.doc, split="V")}), 1)
        self.assertTrue(rows_of(self.doc, split="T"))

    def test_a_record_marked_unusable_keeps_its_reason(self):
        recs = layout_a(tempfile.mkdtemp())
        recs[0]["unusable"] = "operator_flag"
        doc, _ = S.build_splits(recs, params=PARAMS)
        r = rows_of(doc, id=recs[0]["id"])[0]
        self.assertEqual((r["usable"], r["reason"], r["split"]), (False, "operator_flag", None))


class LeakDetector(unittest.TestCase):
    def probes(self, items):
        return {("x", key): rir_data.probe_rgb(synth.rgb_from_gray(synth.scene_gray(seed, k))) for key, (seed, k) in items.items()}

    def doc(self, splits):
        return {"frames": [{"dataset": "x", "id": k, "split": s, "usable": True} for k, s in splits.items()]}

    def test_flags_a_planted_near_duplicate_across_the_boundary(self):
        items = {"a": (1, 0), "b": (2, 0), "leak": (1, 1), "clean": (9, 0)}
        splits = {"a": "T", "b": "S", "leak": "Test", "clean": "V"}
        leaks = S.find_split_leaks(self.doc(splits), self.probes(items), min_corr=0.95, max_hamming=6)
        self.assertEqual([(l["held_out"], l["train"]) for l in leaks], [("leak", "a")])

    def test_clean_split_reports_nothing(self):
        items = {"a": (1, 0), "b": (2, 0), "c": (9, 0), "d": (13, 0)}
        splits = {"a": "T", "b": "S", "c": "V", "d": "Test"}
        self.assertEqual(S.find_split_leaks(self.doc(splits), self.probes(items), 0.95, 6), [])


class Cli(unittest.TestCase):
    def test_writes_splits_json_and_leak_audit_from_a_manifest(self):
        import json
        import rir_manifest as M
        tmp = tempfile.mkdtemp()
        manifest = os.path.join(tmp, "pairs.jsonl")
        M.write_records(manifest, layout_b(tempfile.mkdtemp()))
        out, leaks = os.path.join(tmp, "splits.json"), os.path.join(tmp, "leaks.json")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(S.main(["--manifest", manifest, "--out", out, "--leak-audit", leaks, "--workers", "1"]), 0)
        with open(out) as fh:
            doc = json.load(fh)
        self.assertEqual(doc["summary"]["b"]["splits"]["Test"]["sequences"], 2)


if __name__ == "__main__":
    unittest.main()
