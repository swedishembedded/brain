# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements label-preservation checks for synthetic
# training images for its clients. If your team needs expertise in
# detector-training data validation or image-to-image translation quality
# control, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_gates: model-free label-preservation gates on a
synthetic IR image, with planted shifts, planted objects and a planted
mismatching texture."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import rir_gates as G  # noqa: E402

H, W = 192, 256
BOXES = [(0, 30, 30, 110, 120), (1, 140, 60, 230, 160)]  # (class, x1, y1, x2, y2)


def blocks(seed, h=H, w=W, n=40):
    """Piecewise-constant scene with plenty of sharp edges, in [0, 255]."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 100.0, np.float32)
    for _ in range(n):
        x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        img[y0:y0 + int(rng.integers(8, 50)), x0:x0 + int(rng.integers(8, 60))] = rng.uniform(20, 235)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def to_rgb(gray):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return np.dstack([g, np.clip(g.astype(int) + 8, 0, 255).astype(np.uint8), g])


def to_ir(gray, invert=False, noise=2.0, seed=0, shift=(0, 0)):
    """A synthetic IR derived from the scene: contrast changed, blurred, noisy, optionally shifted."""
    img = 255 - gray if invert else gray
    img = cv2.GaussianBlur(img, (0, 0), 1.5) * 0.6 + 40
    img = np.roll(img, shift, axis=(0, 1))  # (dy, dx)
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_pairs(n=12):
    """Real-pair stand-ins: IR is a related but imperfect rendering of the RGB scene."""
    for k in range(n):
        gray = blocks(100 + k)
        yield to_rgb(gray), to_ir(gray, invert=bool(k % 2), noise=6.0, seed=k), BOXES


CFG_ARGS = dict(edge_threshold=0.2)


class EdgeGate(unittest.TestCase):
    def test_an_aligned_synthetic_ir_correlates_inside_every_box_and_a_foreign_texture_does_not(self):
        gray = blocks(1)
        rgb = to_rgb(gray)
        same = G.box_edge_correlations(rgb, to_ir(gray), BOXES)
        foreign = G.box_edge_correlations(rgb, to_ir(blocks(2)), BOXES)
        self.assertTrue(all(c > 0.5 for c in same), same)
        self.assertTrue(all(abs(c) < 0.25 for c in foreign), foreign)

    def test_the_contrast_polarity_of_the_ir_does_not_matter(self):
        gray = blocks(1)
        a = G.box_edge_correlations(to_rgb(gray), to_ir(gray), BOXES)
        b = G.box_edge_correlations(to_rgb(gray), to_ir(gray, invert=True), BOXES)
        np.testing.assert_allclose(a, b, atol=0.1)

    def test_a_box_too_small_to_correlate_is_not_evaluated(self):
        gray = blocks(1)
        c = G.box_edge_correlations(to_rgb(gray), to_ir(gray), [(0, 10, 10, 14, 14), BOXES[0]], min_box_px=8)
        self.assertIsNone(c[0])
        self.assertIsNotNone(c[1])

    def test_the_threshold_is_the_tenth_percentile_of_real_pairs(self):
        ref = G.reference_distribution(real_pairs())
        self.assertEqual(ref.n, 12 * len(BOXES))
        self.assertAlmostEqual(ref.threshold, float(np.percentile(ref.values, 10)))
        self.assertEqual(ref.percentile, 10)
        # about a tenth of the real pairs fall below their own threshold
        below = float(np.mean(np.array(ref.values) < ref.threshold))
        self.assertLessEqual(below, 0.15)

    def test_the_reference_survives_a_round_trip_through_json(self):
        ref = G.reference_distribution(real_pairs(4))
        back = G.Reference.from_dict(json.loads(json.dumps(ref.to_dict())))
        self.assertEqual(back.threshold, ref.threshold)
        self.assertEqual(back.n, ref.n)

    def test_an_empty_reference_is_an_error_not_a_threshold_of_zero(self):
        with self.assertRaisesRegex(ValueError, "no boxes"):
            G.reference_distribution(iter([]))


class ShiftGate(unittest.TestCase):
    def shift_of(self, dy, dx):
        gray = blocks(3)
        return G.global_shift(to_rgb(gray), to_ir(gray, shift=(dy, dx)))

    def test_an_aligned_pair_has_no_shift(self):
        dx, dy = self.shift_of(0, 0)
        self.assertLess(np.hypot(dx, dy), 0.5)

    def test_a_planted_shift_is_measured(self):
        dx, dy = self.shift_of(0, 5)
        self.assertAlmostEqual(abs(dx), 5, delta=0.7)
        self.assertLess(abs(dy), 0.7)

    def test_a_featureless_ir_has_no_measurable_shift(self):
        self.assertIsNone(G.global_shift(to_rgb(blocks(3)), np.full((H, W), 90, np.uint8)))


def offset_pairs(dx_values, n_per=2):
    """Real-pair stand-ins registered with a constant offset of a few pixels, as the study's real pairs are."""
    for k, dx in enumerate(dx_values * n_per):
        gray = blocks(300 + k)
        yield to_rgb(gray), to_ir(gray, noise=4.0, seed=k, shift=(0, dx)), BOXES


class ShiftCalibration(unittest.TestCase):
    """The limit is what real pairs do: no more misaligned than they are."""

    def setUp(self):
        self.ref = G.real_reference(offset_pairs([3, 3, 4, 3, 2, 3], 3))
        self.gray = blocks(7)

    def synthetic(self, dx):
        return G.check_image(to_rgb(self.gray), to_ir(self.gray, shift=(0, dx)), BOXES,
                             G.GateConfig(edge_threshold=-1.0, **self.ref.shift.config_args()))

    def test_the_reference_is_the_median_offset_and_the_spread_of_the_real_pairs(self):
        sh = self.ref.shift
        self.assertEqual(sh.n, 18)
        self.assertAlmostEqual(abs(sh.median[0]), 3.0, delta=0.7)
        self.assertLess(abs(sh.median[1]), 0.7)
        self.assertGreater(sh.p95, 0.0)
        self.assertLess(sh.p95, 2.0)
        self.assertEqual(sh.floor, G.SHIFT_FLOOR_PX)
        self.assertAlmostEqual(sh.limit, sh.p95 + sh.floor)
        json.dumps(sh.to_dict())

    def test_a_synthetic_image_shifted_by_the_real_median_offset_passes(self):
        r = self.synthetic(3)
        self.assertNotIn("global_shift", r.reasons)

    def test_one_shifted_six_pixels_further_fails(self):
        self.assertIn("global_shift", self.synthetic(9).reasons)

    def test_a_fixed_limit_is_centred_on_zero_so_it_fails_the_real_offset(self):
        r = G.check_image(to_rgb(self.gray), to_ir(self.gray, shift=(0, 3)), BOXES, G.GateConfig(edge_threshold=-1.0))
        self.assertIn("global_shift", r.reasons)

    def test_the_edge_reference_reports_its_percentiles(self):
        e = self.ref.edge
        self.assertEqual(e.n, 18 * len(BOXES))
        pct = e.percentiles()
        self.assertEqual(sorted(pct), ["10", "25", "5", "50", "75", "90", "95"])
        self.assertAlmostEqual(pct["10"], e.threshold)

    def test_pairs_without_a_measurable_shift_do_not_count_toward_it(self):
        flat = (np.dstack([np.full((H, W), 90, np.uint8)] * 3), np.full((H, W), 90, np.uint8), BOXES)
        ref = G.real_reference(list(offset_pairs([3], 2)) + [flat])
        self.assertEqual(ref.shift.n, 2)


class Hallucinations(unittest.TestCase):
    def test_a_confident_unmatched_detection_is_flagged(self):
        planted = G.Detection(1, 0.9, (150, 10, 200, 40))
        found = G.find_hallucinations([planted], BOXES)
        self.assertEqual(found, [planted])

    def test_detections_on_labelled_objects_are_not_hallucinations(self):
        on_gt = [G.Detection(0, 0.95, (32, 28, 108, 122)), G.Detection(1, 0.8, (138, 62, 228, 158))]
        self.assertEqual(G.find_hallucinations(on_gt, BOXES), [])

    def test_a_detection_of_the_wrong_class_on_an_object_is_unmatched(self):
        wrong = G.Detection(1, 0.9, (30, 30, 110, 120))
        self.assertEqual(G.find_hallucinations([wrong], BOXES), [wrong])

    def test_below_the_confidence_floor_it_is_noise(self):
        faint = G.Detection(0, 0.3, (150, 10, 200, 40))
        self.assertEqual(G.find_hallucinations([faint], BOXES, conf_min=0.5), [])

    def test_a_row_of_brain_yolov8_detect_is_a_detection(self):
        self.assertEqual(G.Detection.from_row([1.0, 2.0, 30.0, 40.0, 0.75, 2]), G.Detection(2, 0.75, (1.0, 2.0, 30.0, 40.0)))


def config():
    return G.GateConfig(**CFG_ARGS)


class WholeImage(unittest.TestCase):
    def setUp(self):
        self.gray = blocks(7)
        self.rgb = to_rgb(self.gray)

    def test_a_faithful_synthetic_image_passes_every_gate(self):
        r = G.check_image(self.rgb, to_ir(self.gray), BOXES, config(), detections=[G.Detection(0, 0.9, (31, 31, 109, 119))],
                          mask_iou=lambda rgb, ir, box: 0.8)
        self.assertTrue(r.passed, r.reasons)
        self.assertEqual(r.reasons, [])
        self.assertEqual(len(r.boxes), 2)

    def test_a_shifted_image_fails_the_global_alignment_gate(self):
        r = G.check_image(self.rgb, to_ir(self.gray, shift=(0, 6)), BOXES, config())
        self.assertFalse(r.passed)
        self.assertIn("global_shift", r.reasons)

    def test_a_shift_within_two_pixels_passes(self):
        r = G.check_image(self.rgb, to_ir(self.gray, shift=(0, 1)), BOXES, config())
        self.assertNotIn("global_shift", r.reasons)

    def test_a_mismatching_texture_inside_one_box_fails_the_edge_gate_for_that_box_only(self):
        ir = to_ir(self.gray)
        x1, y1, x2, y2 = BOXES[1][1:]
        ir[y1:y2, x1:x2] = to_ir(blocks(99))[y1:y2, x1:x2]
        r = G.check_image(self.rgb, ir, BOXES, config())
        self.assertIn("box_edge_correlation", r.reasons)
        self.assertEqual([b.passed for b in r.boxes], [True, False])

    def test_a_planted_object_fails_the_hallucination_gate(self):
        r = G.check_image(self.rgb, to_ir(self.gray), BOXES, config(), detections=[G.Detection(0, 0.9, (150, 10, 200, 40))])
        self.assertIn("hallucination", r.reasons)
        self.assertEqual(len(r.hallucinations), 1)

    def test_the_mask_gate_is_a_pluggable_callable_given_the_images_and_the_box(self):
        seen = []

        def mask_iou(rgb, ir, box):
            seen.append((rgb.shape, ir.shape, box))
            return 0.2 if box[0] == 30 else 0.9

        r = G.check_image(self.rgb, to_ir(self.gray), BOXES, config(), mask_iou=mask_iou)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], ((H, W, 3), (H, W), (30, 30, 110, 120)))
        self.assertEqual(r.reasons, ["box_mask_iou"])
        self.assertEqual([b.mask_iou for b in r.boxes], [0.2, 0.9])

    def test_the_mask_gate_runs_only_on_boxes_that_passed_the_model_free_gates(self):
        ir = to_ir(self.gray)
        x1, y1, x2, y2 = BOXES[1][1:]
        ir[y1:y2, x1:x2] = to_ir(blocks(99))[y1:y2, x1:x2]
        seen = []
        r = G.check_image(self.rgb, ir, BOXES, config(), mask_iou=lambda rgb, ir, box: seen.append(box) or 0.9)
        self.assertEqual(seen, [(30, 30, 110, 120)])
        self.assertEqual([b.mask_iou for b in r.boxes], [0.9, None])
        self.assertEqual(r.reasons, ["box_edge_correlation"])

    def test_the_mask_gate_is_not_run_on_an_image_that_already_fails_the_shift_gate(self):
        seen = []
        r = G.check_image(self.rgb, to_ir(self.gray, shift=(0, 6)), BOXES, config(),
                          mask_iou=lambda rgb, ir, box: seen.append(box) or 0.9)
        self.assertEqual(seen, [])
        self.assertEqual(r.reasons, ["global_shift"])

    def test_a_mask_gate_that_cannot_decide_is_not_a_failure_and_is_counted(self):
        r = G.check_image(self.rgb, to_ir(self.gray), BOXES, config(), mask_iou=lambda *a: None)
        self.assertTrue(r.passed)
        self.assertEqual([b.mask_iou for b in r.boxes], [None, None])

    def test_without_a_detector_or_segmenter_those_gates_are_simply_absent(self):
        r = G.check_image(self.rgb, to_ir(self.gray), BOXES, config())
        self.assertTrue(r.passed)
        self.assertFalse(r.hallucinations_checked)


class FakeSegmenter:
    """Masks the box on an RGB image; on an IR image (identical channels) only the left quarter of it."""

    def __init__(self):
        self.calls = []

    def segment(self, bgr, boxes):
        self.calls.append(boxes)
        is_ir = bool(np.array_equal(bgr[..., 0], bgr[..., 1]))
        masks = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(bgr.shape[:2], bool)
            m[y1:y2, x1:(x1 + (x2 - x1) // 4 if is_ir else x2)] = True
            masks.append(m)
        return masks


class SegmenterMaskGate(unittest.TestCase):
    def test_the_iou_of_the_masks_segmented_on_the_rgb_and_on_the_ir_is_the_gate_value(self):
        seg = FakeSegmenter()
        gray = blocks(7)
        iou = G.segmenter_mask_iou(seg)(to_rgb(gray), to_ir(gray), (30, 30, 110, 120))
        self.assertAlmostEqual(iou, 0.25)
        self.assertEqual(seg.calls, [[(30, 30, 110, 120)]] * 2)

    def test_a_segmenter_finding_nothing_on_either_image_cannot_decide(self):
        class Empty:
            def segment(self, bgr, boxes):
                return [np.zeros(bgr.shape[:2], bool) for _ in boxes]

        gray = blocks(7)
        self.assertIsNone(G.segmenter_mask_iou(Empty())(to_rgb(gray), to_ir(gray), (30, 30, 110, 120)))

    def test_wired_into_the_gate_a_collapsed_ir_mask_fails_the_box(self):
        gray = blocks(7)
        r = G.check_image(to_rgb(gray), to_ir(gray), BOXES, config(), mask_iou=G.segmenter_mask_iou(FakeSegmenter()))
        self.assertEqual(r.reasons, ["box_mask_iou"])


class Statistics(unittest.TestCase):
    def test_rejections_are_counted_per_reason_and_per_image(self):
        gray = blocks(7)
        rgb = to_rgb(gray)
        stats = G.RejectionStats("c2")
        good = G.check_image(rgb, to_ir(gray), BOXES, config())
        shifted = G.check_image(rgb, to_ir(gray, shift=(0, 7)), BOXES, config())
        halluc = G.check_image(rgb, to_ir(gray), BOXES, config(), detections=[G.Detection(0, 0.9, (150, 10, 200, 40))])
        for r in (good, good, shifted, halluc):
            stats.add(r)
        d = stats.to_dict()
        self.assertEqual((d["arm"], d["n_images"], d["n_failed"]), ("c2", 4, 2))
        self.assertAlmostEqual(d["fail_fraction"], 0.5)
        self.assertEqual(d["by_reason"]["global_shift"], 1)
        self.assertEqual(d["by_reason"]["hallucination"], 1)
        self.assertEqual(d["by_reason"]["box_mask_iou"], 0)
        self.assertEqual(d["n_boxes"], 8)
        json.dumps(d)  # plain JSON: this is the gate-statistics file rir_decide reads

    def test_an_image_failing_two_gates_counts_once_as_failed_and_once_per_reason(self):
        gray = blocks(7)
        rgb = to_rgb(gray)
        both = G.check_image(rgb, to_ir(gray, shift=(0, 7)), BOXES, config(), detections=[G.Detection(0, 0.9, (150, 10, 200, 40))])
        stats = G.RejectionStats("c2")
        stats.add(both)
        d = stats.to_dict()
        self.assertEqual(d["n_failed"], 1)
        self.assertGreaterEqual(sum(1 for v in d["by_reason"].values() if v), 2)

    def test_no_images_has_no_fail_fraction(self):
        self.assertIsNone(G.RejectionStats("c2").to_dict()["fail_fraction"])


if __name__ == "__main__":
    unittest.main()
