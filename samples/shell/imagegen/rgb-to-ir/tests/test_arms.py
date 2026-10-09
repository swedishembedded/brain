# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements synthetic thermal-image rendering and sensor
# modelling for its clients. If your team needs expertise in thermal-IR
# simulation or detector training-data synthesis, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Spec tests for rir_arms: the generative-model-free training-image arms."""
import dataclasses
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
import cv2  # noqa: E402
import rir_arms as A  # noqa: E402
import rir_data  # noqa: E402
import rir_pack  # noqa: E402
import rir_readers  # noqa: E402
import rir_splits  # noqa: E402

H, W = 160, 224


def manifest_rows(out, arm):
    with open(os.path.join(out, arm, "manifest.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def rect_scene(seed, h=H, w=W):
    """Piecewise-constant scene: sharp edges, flat interiors."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90.0, np.float32)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w - 30)), int(rng.integers(0, h - 30))
        img[y0:y0 + int(rng.integers(15, 60)), x0:x0 + int(rng.integers(15, 80))] = rng.uniform(40, 220)
    return img


def sensor_forward(scene, blur, noise, stripe, seed):
    """Independent forward model for the fit tests: blur, then noise, then
    per-column offsets. Deliberately not rir_arms code."""
    rng = np.random.default_rng(seed)
    out = cv2.GaussianBlur(scene, (0, 0), blur) if blur > 0 else scene.copy()
    out = out + rng.normal(0, noise, out.shape) + rng.normal(0, stripe, (1, out.shape[1]))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class GrayArms(unittest.TestCase):
    def test_b2_is_the_exact_inverse_of_b1(self):
        rgb = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        b1, b2 = A.render_b1(rgb), A.render_b2(rgb)
        self.assertEqual(b1.ndim, 2)
        self.assertTrue(np.array_equal(b2, 255 - b1))

    def test_b1_is_luma_of_the_rgb(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[..., 2] = 255  # pure red in BGR order
        self.assertEqual(int(A.render_b1(rgb)[0, 0]), int(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[0, 0]))


class SensorFit(unittest.TestCase):
    """Tolerances: noise sigma +-15%, stripe amplitude +-30% (or 0.5 level),
    blur sigma +-0.4 px, contrast statistics +-4 levels."""

    TRUE = dict(blur=2.0, noise=3.0, stripe=2.5)

    def pairs(self, n=12):
        for i in range(n):
            scene = rect_scene(i)
            ir = sensor_forward(scene, self.TRUE["blur"], self.TRUE["noise"], self.TRUE["stripe"], 100 + i)
            yield np.dstack([np.round(scene).astype(np.uint8)] * 3), ir

    def test_recovers_known_blur_noise_and_stripe_parameters(self):
        m = A.fit_sensor_model(self.pairs())
        self.assertAlmostEqual(m.noise_sigma, self.TRUE["noise"], delta=0.15 * self.TRUE["noise"])
        self.assertAlmostEqual(m.stripe_amplitude, self.TRUE["stripe"], delta=max(0.5, 0.3 * self.TRUE["stripe"]))
        self.assertAlmostEqual(m.blur_sigma, self.TRUE["blur"], delta=0.4)

    def test_recovers_contrast_statistics_of_the_ir(self):
        pairs = list(self.pairs())
        m = A.fit_sensor_model(iter(pairs))
        means = [float(ir.mean()) for _, ir in pairs]
        stds = [float(ir.std()) for _, ir in pairs]
        self.assertAlmostEqual(m.mean_mu, float(np.median(means)), delta=4)
        self.assertAlmostEqual(m.std_mu, float(np.median(stds)), delta=4)

    def test_sharp_noise_free_ir_fits_to_no_added_degradation(self):
        scene = rect_scene(3)
        gray = np.round(scene).astype(np.uint8)
        m = A.fit_sensor_model(iter([(np.dstack([gray] * 3), gray)] * 3))
        self.assertLess(m.blur_sigma, 0.3)
        self.assertLess(m.noise_sigma, 0.5)
        self.assertLess(m.stripe_amplitude, 0.5)

    def test_fit_from_splits_reads_only_split_t_frames(self):
        rows = [{"dataset": "a", "id": f"{sp}{i}", "split": sp, "usable": True}
                for sp in ("T", "S", "V", "Test") for i in range(4)]
        rows.append({"dataset": "a", "id": "dropped", "split": None, "usable": False})
        rows.append({"dataset": "other", "id": "otherds", "split": "T", "usable": True})
        seen = []

        def read_pair(row):
            seen.append((row["id"], row["split"]))
            gray = np.round(rect_scene(len(seen))).astype(np.uint8)
            return np.dstack([gray] * 3), gray

        A.fit_from_splits({"frames": rows}, "a", read_pair, max_frames=3, seed=0)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(sp == "T" for _, sp in seen), seen)

    def test_fit_from_splits_refuses_when_there_is_no_t_frame(self):
        rows = [{"dataset": "a", "id": "a", "split": "Test", "usable": True}]
        with self.assertRaises(ValueError):
            A.fit_from_splits({"frames": rows}, "a", lambda r: None, max_frames=3, seed=0)

    def test_model_json_round_trip(self):
        m = A.SensorModel(blur_sigma=1.5, noise_sigma=2.0, stripe_amplitude=1.0, mean_mu=100, mean_sd=10, std_mu=40, std_sd=5)
        self.assertEqual(A.SensorModel.from_dict(json.loads(json.dumps(m.to_dict()))), m)


MODEL = A.SensorModel(blur_sigma=1.2, noise_sigma=1.5, stripe_amplitude=1.0, mean_mu=110.0, mean_sd=8.0, std_mu=45.0, std_sd=4.0)


class B3(unittest.TestCase):
    def rgb(self):
        return np.dstack([np.round(rect_scene(7)).astype(np.uint8)] * 3)

    def test_deterministic_for_a_seeded_rng_and_single_channel_uint8(self):
        a = A.render_b3(self.rgb(), MODEL, np.random.default_rng(4))
        b = A.render_b3(self.rgb(), MODEL, np.random.default_rng(4))
        self.assertEqual((a.dtype, a.ndim), (np.uint8, 2))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, A.render_b3(self.rgb(), MODEL, np.random.default_rng(5))))

    def test_output_contrast_follows_the_fitted_statistics(self):
        outs = [A.render_b3(self.rgb(), MODEL, np.random.default_rng(s)) for s in range(30)]
        self.assertAlmostEqual(float(np.mean([o.mean() for o in outs])), MODEL.mean_mu, delta=6)
        self.assertAlmostEqual(float(np.mean([o.std() for o in outs])), MODEL.std_mu, delta=6)


class B4(unittest.TestCase):
    CLEAN = dataclasses.replace(MODEL, noise_sigma=0.0, stripe_amplitude=0.0, blur_sigma=0.8)

    def test_box_regions_are_brighter_than_their_surroundings(self):
        for cls in (0, 1, 2):
            img = A.render_b4(240, 320, [(cls, 140, 150, 180, 220)], self.CLEAN, np.random.default_rng(0))
            inside = img[165:205, 150:170].mean()
            ring = np.concatenate([img[150:220, 120:130].ravel(), img[150:220, 190:200].ravel()]).mean()
            self.assertGreater(inside, ring + 25, f"class {cls}")

    def test_person_is_hotter_than_a_car(self):
        p = A.render_b4(240, 320, [(0, 100, 150, 160, 220)], self.CLEAN, np.random.default_rng(0))
        c = A.render_b4(240, 320, [(1, 100, 150, 160, 220)], self.CLEAN, np.random.default_rng(0))
        self.assertGreater(p[170:200, 120:140].mean(), c[170:200, 120:140].mean())

    def test_cold_sky_is_darker_than_ground(self):
        img = A.render_b4(240, 320, [], self.CLEAN, np.random.default_rng(0))
        self.assertLess(img[:24].mean() + 20, img[-24:].mean())
        self.assertEqual((img.dtype, img.shape), (np.uint8, (240, 320)))


class RenderEndToEnd(unittest.TestCase):
    """splits -> fit -> render S -> manifest -> pack, on a synthetic pairs layout."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        args = synth.write_voc_pairs(
            self.root,
            train_videos=[{"ids": range(10 * i, 10 * i + 6), "scene": i + 1, "level": (60, 200)} for i in range(1, 8)],
            test_videos=[{"ids": range(200, 204), "scene": 30, "level": (60, 200)}],
        )
        records, _ = rir_readers.build_records({"dataset": "a", **args})
        self.doc, _ = rir_splits.build_splits(records, seed=1, params=rir_splits.SplitParams(default_s_cap=12))
        self.out = tempfile.mkdtemp()

    def test_render_writes_images_manifest_and_a_packable_dataset(self):
        model = A.fit_dataset_models(self.doc, ("a",), max_frames=6, seed=1)
        s_rows = [r for r in self.doc["frames"] if r["split"] == "S"]
        self.assertTrue(s_rows)
        A.render_arms(self.doc, model, self.out, A.ARMS, "S", datasets=("a",), limit=0, seed=1)
        for arm in A.ARMS:
            rows = manifest_rows(self.out, arm)
            self.assertEqual(len(rows), len(s_rows))
            first = cv2.imread(os.path.join(self.out, arm, rows[0]["image"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(first.ndim, 3 if arm == "a1" else 2, arm)
            self.assertEqual({b[0] for b in rows[0]["boxes"]}, {0, 1})  # dog dropped
        out = os.path.join(self.out, "pack")
        rir_pack.main(["--manifest", os.path.join(self.out, "b4", "manifest.jsonl"), "--out", out, "--size", "64", "--seed", "1"])
        with open(os.path.join(out, "meta.json")) as fh:
            self.assertEqual(json.load(fh)["n"], len(s_rows))

    def test_arms_a2_is_the_real_ir_and_a1_the_rgb(self):
        model = A.fit_dataset_models(self.doc, ("a",), max_frames=6, seed=1)
        A.render_arms(self.doc, model, self.out, ("a1", "a2"), "S", datasets=("a",), limit=2, seed=1)
        for arm, reader in (("a1", rir_data.read_rgb), ("a2", rir_data.read_ir)):
            row = manifest_rows(self.out, arm)[0]
            source = next(r for r in self.doc["frames"] if r["id"] == row["id"])
            written = cv2.imread(os.path.join(self.out, arm, row["image"]), cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.array_equal(written, reader(source)), arm)

    def test_limit_selects_the_same_seeded_frames_for_every_arm(self):
        model = A.fit_dataset_models(self.doc, ("a",), max_frames=6, seed=1)
        A.render_arms(self.doc, model, self.out, A.ARMS, "S", datasets=("a",), limit=3, seed=9)
        picks = [[r["id"] for r in manifest_rows(self.out, arm)] for arm in A.ARMS]
        self.assertEqual(len(picks[0]), 3)
        self.assertTrue(all(p == picks[0] for p in picks))


if __name__ == "__main__":
    unittest.main()
