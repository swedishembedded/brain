# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_eval: offline COCO-style scoring of a `brain yolov8 eval
--dump-preds` file, matched against the Rust implementation."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import rir_eval as E  # noqa: E402


def det(cls, box, score):
    return {"class": cls, "score": score, "xyxy": list(box)}


def gt(cls, box):
    return {"class": cls, "xyxy": list(box)}


def image(index, gts, preds):
    return {"image": index, "gts": gts, "preds": preds}


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class Scoring(unittest.TestCase):
    """Goldens that the Rust test-suite of eval::detection_report asserts as well."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def score(self, rows, nc=None):
        path = os.path.join(self.tmp, "preds.jsonl")
        write_jsonl(path, rows)
        return E.score(E.read_jsonl(path), nc)

    @staticmethod
    def two_images(shift):
        boxes = [(0, [10, 10, 110, 110]), (1, [20, 30, 120, 130])]
        return [image(i, [gt(c, b)], [det(c, [b[0] + shift, b[1], b[2] + shift, b[3]], 0.9)])
                for i, (c, b) in enumerate(boxes)]

    def test_perfect_predictions_score_one(self):
        r = self.score(self.two_images(0), 2)
        self.assertAlmostEqual(r.map50, 1.0, places=6)
        self.assertAlmostEqual(r.map50_95, 1.0, places=6)
        self.assertEqual(sorted(r.per_class), [0, 1])
        self.assertEqual((r.n_images, r.n_preds, r.n_gts), (2, 2, 2))

    def test_shift_keeps_map50_and_loses_the_strict_thresholds(self):
        # IoU = 85 / 115 = 0.739: a hit at 0.50 .. 0.70 (5 of 10 thresholds).
        r = self.score(self.two_images(15), 2)
        self.assertAlmostEqual(r.map50, 1.0, places=6)
        self.assertAlmostEqual(r.map50_95, 0.5, places=6)
        for ap in r.per_class.values():
            self.assertAlmostEqual(ap.ap50_95, 0.5, places=6)

    def test_images_never_match_each_other_even_with_equal_coordinates(self):
        b = [0, 0, 50, 50]
        r = self.score([image(0, [gt(0, b)], []), image(1, [gt(0, b)], [det(0, b, 0.8)])], 1)
        self.assertAlmostEqual(r.map50, 0.5, places=6)
        self.assertAlmostEqual(r.recall50, 0.5, places=6)

    def test_ap_is_the_area_under_the_precision_envelope(self):
        # ranked TP, FP, TP over 2 ground truths: p = 1, 1/2, 2/3 at r = 1/2, 1/2, 1.
        # Envelope 1, 2/3, 2/3: AP = 1/2 * 1 + 1/2 * 2/3.
        a, b = [0, 0, 10, 10], [50, 50, 60, 60]
        rows = [image(0, [gt(0, a), gt(0, b)], [det(0, a, 0.9), det(0, [20, 20, 30, 30], 0.8), det(0, b, 0.7)])]
        self.assertAlmostEqual(self.score(rows, 1).map50, 1 / 2 + 1 / 3, places=6)

    def test_ranking_is_global_across_images(self):
        a = [0, 0, 10, 10]
        # image 1's false positive outranks image 0's true positive.
        rows = [image(0, [gt(0, a)], [det(0, a, 0.5)]), image(1, [], [det(0, a, 0.9)])]
        self.assertAlmostEqual(self.score(rows, 1).map50, 0.5, places=6)

    def test_a_ground_truth_matches_once_and_a_duplicate_is_a_false_positive(self):
        a = [0, 0, 10, 10]
        r = self.score([image(0, [gt(0, a)], [det(0, a, 0.9), det(0, a, 0.8)])], 1)
        self.assertAlmostEqual(r.map50, 1.0, places=6)
        self.assertAlmostEqual(r.precision50, 0.5, places=6)

    def test_a_hit_at_exactly_the_threshold_counts(self):
        r = self.score([image(0, [gt(0, [0, 0, 10, 10])], [det(0, [0, 0, 10, 5], 0.9)])], 1)
        self.assertAlmostEqual(r.map50, 1.0, places=6)
        self.assertAlmostEqual(r.map50_95, 0.1, places=6)

    def test_matching_is_class_aware(self):
        b = [0, 0, 10, 10]
        self.assertAlmostEqual(self.score([image(0, [gt(0, b)], [det(1, b, 0.9)])], 2).map50, 0.0, places=6)

    def test_a_class_without_ground_truth_is_excluded_and_one_without_hits_counts_zero(self):
        b, c = [0, 0, 10, 10], [30, 30, 40, 40]
        rows = [image(0, [gt(0, b), gt(2, c)], [det(0, b, 0.9), det(1, b, 0.9)])]
        r = self.score(rows, 3)
        self.assertEqual(sorted(r.per_class), [0, 2])  # class 1 has predictions but no ground truth
        self.assertAlmostEqual(r.map50, 0.5, places=6)  # (1 + 0) / 2
        self.assertAlmostEqual(r.per_class[2].ap50, 0.0, places=6)

    def test_nothing_to_score_is_zero_not_an_error(self):
        r = self.score([image(0, [], [det(0, [0, 0, 5, 5], 0.9)])], 1)
        self.assertEqual((r.map50, r.map50_95, r.per_class), (0.0, 0.0, {}))

    def test_a_malformed_line_is_reported_with_its_number(self):
        path = os.path.join(self.tmp, "bad.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps(image(0, [], [])) + "\n\n" + '{"image": 1, "gts": [], "preds": [{"class": 0}]}\n')
        with self.assertRaisesRegex(ValueError, "line 3"):
            E.read_jsonl(path)


class Weighted(unittest.TestCase):
    """The cluster bootstrap re-scores the same matches under image multiplicities."""

    def rows(self):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(12):
            gts, preds = [], []
            for k in range(int(rng.integers(0, 4))):
                x, y = (float(v) for v in rng.integers(0, 80, 2))
                box = [x, y, x + 20, y + 20]
                cls = int(rng.integers(0, 2))
                gts.append(gt(cls, box))
                jitter = rng.normal(0, 3, 4)
                preds.append(det(cls, [float(v) for v in np.array(box) + jitter], float(rng.random())))
            preds.append(det(int(rng.integers(0, 2)), [5, 5, 30, 30], float(rng.random())))
            rows.append(image(i, gts, preds))
        return rows

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        path = os.path.join(self.tmp, "p.jsonl")
        write_jsonl(path, self.rows())
        self.images = E.read_jsonl(path)

    def test_unit_weights_reproduce_the_score(self):
        prepared = E.prepare(self.images)
        ref = E.score(self.images)
        m50, m5095 = prepared.maps(np.ones(len(self.images)))
        self.assertAlmostEqual(m50, ref.map50, places=6)
        self.assertAlmostEqual(m5095, ref.map50_95, places=6)

    def test_a_multiplicity_equals_repeating_the_image(self):
        weights = np.array([2, 0, 1, 3, 0, 1, 1, 0, 2, 1, 1, 1], float)
        repeated = [im for im, w in zip(self.images, weights) for _ in range(int(w))]
        self.assertEqual(len(repeated), int(weights.sum()))
        m50, m5095 = E.prepare(self.images).maps(weights)
        ref = E.score(repeated)
        # equal up to the order of exactly tied scores, which the random scores here do not have
        self.assertAlmostEqual(m50, ref.map50, places=6)
        self.assertAlmostEqual(m5095, ref.map50_95, places=6)

    def test_removing_every_image_scores_zero(self):
        self.assertEqual(E.prepare(self.images).maps(np.zeros(len(self.images))), (0.0, 0.0))


class Sequences(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_image_index_leads_to_its_packed_sequence(self):
        path = os.path.join(self.tmp, "sequences.json")
        with open(path, "w") as fh:
            json.dump(["d:a", "d:b", "d:a"], fh)
        images = [E.Image.empty(2), E.Image.empty(0)]  # a dump may be partial and in any order
        self.assertEqual(E.sequence_ids(images, E.load_sequences(path)), ["d:a", "d:a"])

    def test_an_index_outside_the_pack_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "index 5"):
            E.sequence_ids([E.Image.empty(5)], ["d:a"])

    def test_a_pack_without_sequences_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "sequences.json"):
            E.load_sequences(os.path.join(self.tmp, "nope", "sequences.json"))


def find_brain():
    """$BRAIN_BIN, `brain` on PATH, or a built target/release/brain in an ancestor of this checkout."""
    env = os.environ.get("BRAIN_BIN")
    if env:
        return env if os.path.isfile(env) else None
    on_path = shutil.which("brain")
    if on_path:
        return on_path
    here = os.path.abspath(os.path.dirname(__file__))
    while True:
        candidate = os.path.join(here, "target", "release", "brain")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


BRAIN = find_brain()
PARITY_STEPS = "150"
PRINTED = 0.5e-4 + 1e-6  # half a unit of the fourth decimal


@unittest.skipIf(BRAIN is None, "no brain binary: set BRAIN_BIN, put brain on PATH, or build target/release/brain")
class RustParity(unittest.TestCase):
    """Score the dump the real `brain yolov8 eval` wrote and compare with the table it printed.

    The table has four decimals, so agreement is asserted to half a unit of the
    last printed digit (plus float32 rounding of the binary's own sums).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        env = {**os.environ, "BRAIN_DEVICE": "cpu"}
        data, weights = (os.path.join(cls.tmp, n) for n in ("data", "w.safetensors"))

        def brain(*args):
            done = subprocess.run([BRAIN, *args], env=env, capture_output=True, text=True, timeout=600)
            assert done.returncode == 0, f"brain {args}: {done.stderr}"
            return done.stdout

        brain("data", "gen", "detect", "--out", data, "--n", "24", "--seed", "7")
        brain("yolov8", "train", data, "--out", weights, "--steps", PARITY_STEPS, "--batch", "4", "--seed", "7")
        cls.dumps = {}
        for conf in ("0.001", "0.05"):
            dump = os.path.join(cls.tmp, f"preds-{conf}.jsonl")
            printed = brain("yolov8", "eval", "--weights", weights, "--data", data, "--split", "all",
                            "--conf", conf, "--dump-preds", dump)
            cls.dumps[conf] = (dump, printed)

    def parse(self, printed):
        num = lambda label: float(re.search(rf"^{re.escape(label)}\s+([0-9.]+)", printed, re.M).group(1))
        per_class = {int(c): (float(a), float(b))
                     for c, a, b in re.findall(r"^(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$", printed, re.M)}
        counts = re.search(r"preds (\d+)\s+gts (\d+)\s+\(images (\d+)\)", printed)
        return {"map50": num("mAP@0.5"), "map50_95": num("mAP@0.5:0.95"), "precision50": num("precision@0.5"),
                "recall50": num("recall@0.5")}, per_class, tuple(int(g) for g in counts.groups())

    def test_every_number_the_binary_prints_is_reproduced(self):
        for conf, (dump, printed) in self.dumps.items():
            with self.subTest(conf=conf):
                top, per_class, counts = self.parse(printed)
                images = E.read_jsonl(dump)
                ours = E.score(images, nc=3)
                self.assertEqual((ours.n_preds, ours.n_gts, ours.n_images), counts)
                for name, value in top.items():
                    self.assertAlmostEqual(getattr(ours, name), value, delta=PRINTED, msg=name)
                self.assertEqual(sorted(ours.per_class), sorted(per_class))
                for c, (ap50, ap5095) in per_class.items():
                    self.assertAlmostEqual(ours.per_class[c].ap50, ap50, delta=PRINTED, msg=f"class {c} AP@0.5")
                    self.assertAlmostEqual(ours.per_class[c].ap50_95, ap5095, delta=PRINTED, msg=f"class {c} AP@0.5:0.95")

    def test_the_dump_is_a_meaningful_comparison(self):
        # Zeros everywhere would agree with any implementation.
        dump, _ = self.dumps["0.001"]
        self.assertGreater(E.score(E.read_jsonl(dump), nc=3).map50, 0.0)


if __name__ == "__main__":
    unittest.main()
