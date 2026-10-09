# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters and leakage-safe evaluation
# protocols for multimodal perception pipelines for its clients. If your team
# needs expertise in paired RGB / thermal-IR data, detector training sets or
# sensor-domain adaptation, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Spec tests for the generic readers, IR knobs and frame checks."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
import cv2  # noqa: E402
import rir_data as D  # noqa: E402
import rir_manifest as M  # noqa: E402
import rir_readers as R  # noqa: E402


def build(args, dataset="d"):
    records, report = R.build_records({"dataset": dataset, **args})
    errs = [e for rec in records for e in M.validate_record(rec, "")]
    assert not errs, errs
    return {r["id"]: r for r in records}, report


def voc_fixture(**over):
    root = tempfile.mkdtemp()
    args = synth.write_voc_pairs(
        root,
        train_videos=[
            {"ids": range(10, 20), "scene": 1, "padded": {12, 13}},
            {"ids": range(20, 30), "scene": 2, "scenes": {n: 3 for n in range(25, 30)}},
            {"ids": list(range(40, 45)) + list(range(50, 55)), "scene": 4},
        ],
        test_videos=[{"ids": range(200, 204), "scene": 7}],
    )
    return root, {**args, **over}


class VocPairs(unittest.TestCase):
    def test_pairs_by_key_boxes_and_official_hint(self):
        _, args = voc_fixture()
        recs, report = build(args)
        self.assertEqual(report["unpaired_rgb"], 0)
        r = recs["v00010"]
        self.assertEqual({b["class"] for b in r["boxes"]}, {"person", "car", "dog"})
        self.assertEqual((r["width"], r["height"]), (64, 48))
        self.assertEqual(r["official_split"], "train")
        self.assertEqual(recs["v00200"]["official_split"], "test")

    def test_frames_without_a_partner_are_counted_and_skipped(self):
        root, args = voc_fixture()
        os.remove(os.path.join(root, "images", "v00011_IR.png"))
        recs, report = build(args)
        self.assertNotIn("v00011", recs)
        self.assertEqual(report["unpaired_rgb"], 1)

    def test_class_map_renames_classes(self):
        _, args = voc_fixture(class_map="car=vehicle,dog=pet")
        recs, _ = build(args)
        self.assertEqual({b["class"] for b in recs["v00010"]["boxes"]}, {"person", "vehicle", "pet"})

    def test_similarity_rule_breaks_on_gaps_and_cuts_but_not_on_a_padded_frame(self):
        _, args = voc_fixture()
        recs, _ = build(args)
        seq = lambda n: recs[f"v{n:05d}"]["sequence_id"]
        self.assertEqual(seq(10), seq(19))  # frames 12-13 have a shrunk RGB and must not cut the video
        self.assertNotEqual(seq(19), seq(20))  # consecutive numbers, new scene
        self.assertNotEqual(seq(24), seq(25))  # scene cut inside a run
        self.assertEqual(seq(40), seq(44))
        self.assertNotEqual(seq(44), seq(50))  # same scene, gap of 6 > 3
        self.assertNotEqual(seq(200), seq(54))  # official test frames never join a train sequence

    def test_segment_by_similarity_documented_rule(self):
        t = lambda *v: np.array(v, dtype=np.float32) / np.linalg.norm(v)
        thumbs = [t(1, 0), t(1, 0.1), t(0, 1), t(0, 1), t(0, 1)]
        self.assertEqual(R.segment_by_similarity([1, 2, 3, 4, 9], R.thumb_similarity(thumbs), 0.5, 3), [0, 0, 1, 1, 2])

    def test_pair_key_collision_is_an_error_with_advice(self):
        root, args = voc_fixture(pair_regex=r"^v(\d)\d+.*$")
        with self.assertRaisesRegex(R.ReaderError, "share the pair key"):
            build(args)


class YoloAlpha(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.args = synth.write_yolo_alpha(self.root, train={i: 1 for i in range(0, 120, 7)}, val={i: 2 for i in range(3, 120, 7)})

    def test_ir_is_the_alpha_channel_rgb_the_colour_channels_and_labels_are_pixels(self):
        recs, _ = build(self.args)
        r = recs["00000"]
        bgra = cv2.imread(r["rgb"], cv2.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(D.read_ir(r), bgra[..., 3]) or r["ir_read"]["invert"])
        self.assertTrue(np.array_equal(D.read_rgb(r), bgra[..., :3]))
        person = next(b for b in r["boxes"] if b["class"] == "people")
        self.assertEqual((person["x1"], person["x2"]), (24.0, 40.0))  # 0.5 +- 0.125 of 64
        self.assertEqual(recs["00003"]["official_split"], "test")  # a `val` path word counts as test

    def test_block_rule_cuts_numbers_into_blocks_and_guards_the_edges(self):
        args = {k: v for k, v in self.args.items() if k != "official_split_regex"}
        recs, _ = build({**args, "sequence_rule": "block:50", "block_guard": 5})
        self.assertEqual(recs["00003"]["sequence_id"], recs["00049"]["sequence_id"])
        self.assertNotEqual(recs["00049"]["sequence_id"], recs["00052"]["sequence_id"])
        guarded = {i for i, r in recs.items() if r.get("unusable") == "block_gap"}
        self.assertTrue(all(int(i) % 50 < 5 or int(i) % 50 >= 45 for i in guarded) and guarded)
        self.assertNotIn("official_split", recs["00003"])

    def test_missing_alpha_is_a_clear_error(self):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "x"))
        cv2.imwrite(os.path.join(root, "x", "a.png"), np.zeros((8, 8, 3), np.uint8))
        rec = {"id": "a", "rgb": os.path.join(root, "x", "a.png"), "ir": os.path.join(root, "x", "a.png"),
               "ir_read": {"channel": "alpha"}}
        with self.assertRaisesRegex(ValueError, "no alpha channel"):
            D.read_ir(rec)


class CocoReplicated(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.args = synth.write_coco_rgbir(self.root, {f"{i:02d}": 10 + i for i in range(1, 5)}, {"09": 30})

    def test_coco_boxes_regex_sequences_and_three_channel_ir(self):
        recs, _ = build(self.args)
        r = recs["010001"]
        self.assertEqual(r["boxes"], [{"class": "person", "x1": 10, "y1": 8, "x2": 20, "y2": 30}])
        self.assertEqual({v["sequence_id"] for k, v in recs.items() if k.startswith("01")}, {"01"})
        self.assertEqual(len({v["sequence_id"] for v in recs.values()}), 5)
        ir = D.read_ir(r)
        self.assertEqual(ir.ndim, 2)
        self.assertEqual(recs["090001"]["official_split"], "test")

    def test_dir_rule_uses_the_directory_name(self):
        recs, _ = build({**self.args, "sequence_rule": "dir"})
        self.assertEqual({v["sequence_id"] for v in recs.values()}, {"train", "test"})

    def test_bad_arguments_fail_with_readable_errors(self):
        with self.assertRaisesRegex(R.ReaderError, "matched no files"):
            build({**self.args, "rgb_glob": os.path.join(self.root, "nothing", "*.jpg")})
        with self.assertRaisesRegex(R.ReaderError, "needs --yolo-names"):
            build({**self.args, "label_format": "yolo", "labels_dir": self.root})
        with self.assertRaisesRegex(R.ReaderError, "unknown --sequence-rule"):
            build({**self.args, "sequence_rule": "magic"})


class IrKnobs(unittest.TestCase):
    def pair(self, ir_img, boxes_hot, **extra):
        root = tempfile.mkdtemp()
        recs = []
        for i in range(6):
            gray = synth.scene_gray(i, 0, (60, 100))
            ir = gray.copy()
            ir[8:30, 10:20] = ir_img  # the "person" region
            cv2.imwrite(os.path.join(root, f"f{i}_rgb.png"), synth.rgb_from_gray(gray))
            cv2.imwrite(os.path.join(root, f"f{i}_ir.png"), ir if "wide" not in extra else ir.astype(np.uint16) * 256)
        args = {"rgb_glob": os.path.join(root, "*_rgb.png"), "ir_glob": os.path.join(root, "*_ir.png"),
                "pair_regex": r"^(f\d)(?:_rgb|_ir)?\.[^.]+$", "sequence_rule": "dir"}
        return root, args

    def with_boxes(self, args):
        # a one-box label per frame, through the voc reader
        root = os.path.dirname(args["rgb_glob"])
        ann = os.path.join(root, "ann")
        os.makedirs(ann, exist_ok=True)
        for i in range(6):
            with open(os.path.join(ann, f"f{i}.xml"), "w") as fh:
                fh.write(synth.VOC_XML)
        return {**args, "labels_dir": ann, "label_format": "voc"}

    def test_auto_polarity_keeps_a_white_hot_source(self):
        _, args = self.pair(250, True)
        recs, report = build(self.with_boxes(args))
        self.assertEqual(report["polarity"]["decision"], "white-hot")
        self.assertFalse(any(r["ir_read"]["invert"] for r in recs.values()))

    def test_auto_polarity_inverts_a_black_hot_source_so_reads_are_white_hot(self):
        _, args = self.pair(5, False)
        recs, report = build(self.with_boxes(args))
        self.assertEqual(report["polarity"]["decision"], "black-hot")
        r = recs["f0"]
        raw = cv2.imread(r["ir"], cv2.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(D.read_ir(r), 255 - raw))
        self.assertGreater(D.read_ir(r)[10:25, 12:18].mean(), np.median(D.read_ir(r)))

    def test_polarity_override_wins_over_the_measurement(self):
        _, args = self.pair(250, True)
        recs, report = build({**self.with_boxes(args), "ir_polarity": "black-hot"})
        self.assertTrue(all(r["ir_read"]["invert"] for r in recs.values()))
        self.assertNotIn("polarity", report)

    def test_16_bit_ir_is_reduced_to_8_bit(self):
        _, args = self.pair(250, True, wide=True)
        recs, _ = build(args)
        r = recs["f0"]
        raw = cv2.imread(r["ir"], cv2.IMREAD_UNCHANGED)
        self.assertEqual(raw.dtype, np.uint16)
        out = D.read_ir(r)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.array_equal(out, (raw >> 8).astype(np.uint8)))


class FrameChecks(unittest.TestCase):
    def test_valid_fraction_flags_a_shrunk_rgb(self):
        full = synth.scene_gray(1)
        self.assertGreater(D.valid_fraction(full), 0.99)
        self.assertLess(D.valid_fraction(synth.pad_into_subrect(full)), 0.6)

    def test_edge_alignment_measures_a_known_misregistration(self):
        rng = np.random.default_rng(0)
        base = cv2.GaussianBlur(rng.random((160, 200)).astype(np.float32), (0, 0), 2.0)
        rgb = (base * 255 / base.max()).astype(np.uint8)
        aligned = D.edge_alignment(rgb, rgb)
        shifted = D.edge_alignment(rgb, np.roll(rgb, (2, 4), axis=(0, 1)))
        self.assertGreater(aligned[0], 0.99)
        self.assertEqual(aligned[2], (0, 0))
        self.assertLess(shifted[0], aligned[0] - 0.1)
        self.assertGreater(shifted[1], shifted[0])
        self.assertEqual(shifted[2], (4, 2))


if __name__ == "__main__":
    unittest.main()
