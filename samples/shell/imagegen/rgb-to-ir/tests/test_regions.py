# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements region-level thermal measurement for
# paired RGB / thermal-IR datasets for its clients. If your team needs
# expertise in robust radiometric contrast statistics or promptable
# segmentation pipelines, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Spec tests for rir_regions: object masks from box prompts, the robust
within-frame contrast of an object against its ring, and the noise floor tau."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import rir_regions as R  # noqa: E402

H, W = 160, 200
SIGMA = 3.0


class BoxSegmenter:
    """A perfect segmenter for rectangular objects: the mask is the prompt box."""

    def segment(self, bgr, boxes):
        masks = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(bgr.shape[:2], bool)
            m[int(y1):int(y2), int(x1):int(x2)] = True
            masks.append(m)
        return masks


def noisy_frame(seed=0, level=100.0, sigma=SIGMA):
    rng = np.random.default_rng(seed)
    return np.clip(np.round(level + rng.normal(0, sigma, (H, W))), 0, 255).astype(np.uint8)


def plant(frame, box, delta):
    x1, y1, x2, y2 = box
    out = frame.astype(np.float32)
    out[y1:y2, x1:x2] += delta
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_of(box):
    x1, y1, x2, y2 = box
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


class Contrast(unittest.TestCase):
    BOX = (40, 40, 80, 90)

    def test_a_planted_warm_blob_reads_as_delta_over_the_frame_mad(self):
        ir = plant(noisy_frame(1), self.BOX, +30)
        c = R.object_contrast(ir, mask_of(self.BOX), [], R.RegionParams())
        self.assertAlmostEqual(c["contrast"], 30 / SIGMA, delta=1.5)
        self.assertGreater(c["mad"], 0)

    def test_a_cool_blob_is_negative_and_a_flat_one_is_near_zero(self):
        cool = R.object_contrast(plant(noisy_frame(2), self.BOX, -30), mask_of(self.BOX), [], R.RegionParams())
        flat = R.object_contrast(noisy_frame(3), mask_of(self.BOX), [], R.RegionParams())
        self.assertLess(cool["contrast"], -6)
        self.assertLess(abs(flat["contrast"]), 0.7)

    def test_the_ring_excludes_other_objects(self):
        other = (82, 40, 120, 90)  # touches the ring of BOX
        base = plant(noisy_frame(4), self.BOX, +30)
        crowded = plant(base, other, +120)
        with_other = R.object_contrast(crowded, mask_of(self.BOX), [mask_of(other)], R.RegionParams())
        naive = R.object_contrast(crowded, mask_of(self.BOX), [], R.RegionParams())
        self.assertAlmostEqual(with_other["ring_median"], 100, delta=1.0)
        self.assertLess(with_other["ring_px"], naive["ring_px"] - 500, "the neighbour's pixels are not in the ring")
        self.assertEqual(base.shape, crowded.shape)

    def test_boundary_bleed_is_eroded_away(self):
        # A blurred blob: the plateau is +30, its skirt is not part of the object.
        ir = cv2.GaussianBlur(plant(noisy_frame(5, sigma=0.5), self.BOX, +30), (0, 0), 3)
        c = R.object_contrast(ir, mask_of(self.BOX), [], R.RegionParams())
        mad = c["mad"]
        self.assertGreater(c["contrast"] * mad, 25)

    def test_a_mask_too_small_for_a_stable_median_is_not_measured(self):
        tiny = (10, 10, 14, 14)
        self.assertIsNone(R.object_contrast(noisy_frame(6), mask_of(tiny), [], R.RegionParams()))


class NoiseFloor(unittest.TestCase):
    def test_tau_of_pure_noise_matches_the_order_statistics_of_patch_medians(self):
        # Two independent medians of n pixels differ with sd sqrt(2) * 1.2533 * sigma / sqrt(n);
        # the 75th percentile of |difference| is 1.1503 sd. Contrast is in units of the frame MAD (= sigma).
        side = 20
        box = (30, 30, 30 + side, 30 + side)
        ir = R.dequantize(noisy_frame(7), 0)
        tau = R.noise_floor(ir, [mask_of(box)], side * side, np.random.default_rng(0), R.RegionParams(n_pairs=400))
        expected = 1.1503 * np.sqrt(2) * 1.2533 / side
        self.assertAlmostEqual(tau, expected, delta=0.3 * expected)

    def test_a_planted_object_does_not_inflate_tau(self):
        box = (60, 50, 120, 120)
        plain = noisy_frame(8)
        planted = R.dequantize(plant(plain, box, +80), 0)
        plain = R.dequantize(plain, 0)
        side = 12
        kw = dict(n_pairs=300)
        a = R.noise_floor(plain, [mask_of(box)], side * side, np.random.default_rng(1), R.RegionParams(**kw))
        b = R.noise_floor(planted, [mask_of(box)], side * side, np.random.default_rng(1), R.RegionParams(**kw))
        self.assertAlmostEqual(a, b, delta=0.25 * a)  # only the frame MAD differs; a patch on the object would be ~25x

    def test_pairs_are_local_by_default_and_frame_wide_on_request(self):
        ramp = np.tile(np.linspace(40, 200, W, dtype=np.float32), (H, 1))
        ir = R.dequantize(np.clip(ramp, 0, 255).astype(np.uint8), 0)
        local = R.noise_floor(ir, [], 144, np.random.default_rng(3), R.RegionParams())
        wide = R.noise_floor(ir, [], 144, np.random.default_rng(3), R.RegionParams(pair_reach=None))
        self.assertGreater(wide, 2 * local)

    def test_no_room_for_background_patches_gives_no_tau(self):
        full = np.ones((H, W), bool)
        self.assertIsNone(R.noise_floor(R.dequantize(noisy_frame(0), 0), [full], 144, np.random.default_rng(0), R.RegionParams()))


class Polarity(unittest.TestCase):
    def test_a_statement_needs_the_contrast_to_reach_tau(self):
        self.assertEqual(R.polarity(2.0, 0.5), "warmer")
        self.assertEqual(R.polarity(-2.0, 0.5), "cooler")
        self.assertEqual(R.polarity(0.4, 0.5), "same")
        self.assertEqual(R.polarity(-0.4, 0.5), "same")
        self.assertEqual(R.polarity(0.5, 0.5), "warmer")
        self.assertIsNone(R.polarity(2.0, None), "no noise floor, no statement")
        self.assertIsNone(R.polarity(None, 0.5))


class StubGrounder:
    """Returns one part box per request: the upper half of the object's box."""

    def ground(self, bgr, box, parts):
        x1, y1, x2, y2 = box
        return {p: (x1, y1, x2, (y1 + y2) / 2) for p in parts}


class MeasureTile(unittest.TestCase):
    def tile(self):
        ir = noisy_frame(11)
        ir = plant(ir, (30, 40, 70, 80), +40)  # a warm car
        ir = plant(ir, (130, 40, 150, 100), -40)  # a cool person
        ir = plant(ir, (30, 40, 70, 60), +40)  # a hotter bonnet on the car (upper half)
        rgb = np.dstack([ir] * 3)
        boxes = [{"class": "car", "x1": 30, "y1": 40, "x2": 70, "y2": 80},
                 {"class": "person", "x1": 130, "y1": 40, "x2": 150, "y2": 100}]
        return rgb, ir, boxes

    def test_objects_get_a_contrast_a_tau_and_a_polarity(self):
        rgb, ir, boxes = self.tile()
        out = R.measure_tile(rgb, ir, boxes, BoxSegmenter(), seed=1)
        pol = {o["class"]: o["polarity"] for o in out["objects"]}
        self.assertEqual(pol, {"car": "warmer", "person": "cooler"})
        for o in out["objects"]:
            self.assertIsNotNone(o["tau"])
            self.assertGreaterEqual(abs(o["contrast"]), o["tau"])
        self.assertNotIn("parts", out["objects"][0])

    def test_measurement_is_deterministic(self):
        rgb, ir, boxes = self.tile()
        a = R.measure_tile(rgb, ir, boxes, BoxSegmenter(), seed=1)
        b = R.measure_tile(rgb, ir, boxes, BoxSegmenter(), seed=1)
        self.assertEqual(json.dumps(a), json.dumps(b))

    def test_parts_are_measured_only_through_a_grounder_and_a_part_list(self):
        rgb, ir, boxes = self.tile()
        out = R.measure_tile(rgb, ir, boxes, BoxSegmenter(), seed=1, grounder=StubGrounder(),
                             parts_by_class={"car": ["bonnet"]})
        car, person = out["objects"]
        self.assertEqual([p["part"] for p in car["parts"]], ["bonnet"])
        self.assertEqual(car["parts"][0]["polarity"], "warmer")
        self.assertNotIn("parts", person, "a class without a part list is measured at object level only")

    def test_asking_for_parts_without_a_grounder_is_an_error_not_a_guess(self):
        rgb, ir, boxes = self.tile()
        with self.assertRaises(ValueError):
            R.measure_tile(rgb, ir, boxes, BoxSegmenter(), seed=1, parts_by_class={"car": ["bonnet"]})

    def test_a_mask_that_does_not_fit_its_box_is_skipped_with_a_reason(self):
        class Leaky:
            def segment(self, bgr, boxes):
                return [np.ones(bgr.shape[:2], bool) for _ in boxes]

        rgb, ir, boxes = self.tile()
        out = R.measure_tile(rgb, ir, boxes, Leaky(), seed=1)
        self.assertTrue(all(o["polarity"] is None and o["skipped"] == "mask_does_not_fit_box" for o in out["objects"]))


class SetRun(unittest.TestCase):
    def test_a_tile_set_gets_one_json_per_accepted_tile_and_a_throughput_summary(self):
        import rir_tiles as T

        with tempfile.TemporaryDirectory() as d:
            ir = plant(noisy_frame(12), (30, 30, 90, 110), +40)
            index = []
            for k in range(2):
                name = f"t{k}"
                cv2.imwrite(os.path.join(d, f"{name}_rgb.png"), np.dstack([ir] * 3))
                cv2.imwrite(os.path.join(d, f"{name}_ir.png"), np.dstack([ir] * 3))
                index.append({"name": name, "accepted": True, "rgb": f"{name}_rgb.png", "ir": f"{name}_ir.png",
                              "boxes": [{"class": "car", "x1": 30, "y1": 30, "x2": 90, "y2": 110}]})
            index.append({"name": "rej", "accepted": False, "reason": "edge_misaligned", "boxes": []})
            T.write_index(d, index)
            summary = R.measure_set(d, BoxSegmenter(), seed=1)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "regions"))), ["summary.json", "t0.json", "t1.json"])
            self.assertEqual(summary["tiles"], 2)
            self.assertEqual(summary["boxes"], 2)
            self.assertIn("seconds_per_box", summary)
            again = R.measure_set(d, BoxSegmenter(), seed=1)
            self.assertEqual(again["tiles"], 0, "tiles already measured are not measured again")


if __name__ == "__main__":
    unittest.main()
