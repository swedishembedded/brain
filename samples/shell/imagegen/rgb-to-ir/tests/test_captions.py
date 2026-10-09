# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements instruction-caption generation for
# image-editing LoRA fine-tuning for its clients. If your team needs
# expertise in measured, instruction-conditioned training data for
# diffusion models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_captions: instruction text that states only what the real
IR measured, held-out phrasing that never trains, and an honest polarity balance."""
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
import rir_captions as C  # noqa: E402
import rir_regions as R  # noqa: E402
import rir_tiles as T  # noqa: E402
from test_regions import BoxSegmenter, noisy_frame, plant  # noqa: E402


def obj(cls, polarity, mask_px=1000, part=None):
    entry = {"class": cls, "polarity": polarity, "mask_px": mask_px, "contrast": 1.0, "tau": 0.5}
    if part:
        entry["part"] = part
    return entry


def write_tiles(d, specs):
    """specs: list of lists of objects (a regions document each). Returns the tile names."""
    os.makedirs(os.path.join(d, "regions"), exist_ok=True)
    index = []
    for i, objects in enumerate(specs):
        name = f"t{i:04d}"
        index.append({"name": name, "accepted": True, "rgb": f"{name}_rgb.png", "ir": f"{name}_ir.png",
                      "boxes": [{"class": o["class"], "x1": 0, "y1": 0, "x2": 9, "y2": 9} for o in objects]})
        if objects:
            with open(os.path.join(d, "regions", f"{name}.json"), "w") as fh:
                json.dump({"tile": name, "objects": objects}, fh)
    T.write_index(d, index)
    return [e["ir"] for e in index]


def read_captions(d):
    out = {}
    with open(os.path.join(d, "captions.yaml")) as fh:
        for line in fh:
            key, value = line.rstrip("\n").split(": ", 1)
            out[key] = json.loads(value)
    return out


def mixed_specs(n=200):
    rng = np.random.default_rng(0)
    pols = ["warmer", "cooler", "same"]
    return [[obj(c, pols[int(rng.integers(0, 3))]) for c in rng.choice(["car", "person", "bicycle", "dog"], int(rng.integers(0, 6)))]
            for _ in range(n)]


class PlantedBlobs(unittest.TestCase):
    def test_a_warm_car_and_a_cool_person_are_captioned_as_measured(self):
        with tempfile.TemporaryDirectory() as d:
            ir = noisy_frame(21)
            ir = plant(ir, (30, 40, 70, 80), +40)
            ir = plant(ir, (130, 40, 150, 100), -40)
            specs = []
            index = []
            for k in range(20):
                name = f"t{k:02d}"
                cv2.imwrite(os.path.join(d, f"{name}_rgb.png"), np.dstack([ir] * 3))
                cv2.imwrite(os.path.join(d, f"{name}_ir.png"), np.dstack([ir] * 3))
                index.append({"name": name, "accepted": True, "rgb": f"{name}_rgb.png", "ir": f"{name}_ir.png",
                              "boxes": [{"class": "car", "x1": 30, "y1": 40, "x2": 70, "y2": 80},
                                        {"class": "person", "x1": 130, "y1": 40, "x2": 150, "y2": 100}]})
            T.write_index(d, index)
            R.measure_set(d, BoxSegmenter(), seed=1)
            C.build_captions(d, seed=1, neutral_share=0.0)
            caps = read_captions(d)
            self.assertEqual(len(caps), 20)
            for text in caps.values():
                self.assertNotIn("car is cooler", text)
                self.assertNotIn("person is warmer", text)
                self.assertNotIn("car is about the same", text)
            self.assertTrue(any(C.has_polarity(t, "car", "warmer") for t in caps.values()))
            self.assertTrue(any(C.has_polarity(t, "person", "cooler") for t in caps.values()))


class Training(unittest.TestCase):
    def build(self, specs, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        write_tiles(d.name, specs)
        report = C.build_captions(d.name, seed=kw.pop("seed", 1), **kw)
        return d.name, read_captions(d.name), report

    def test_held_out_templates_never_appear_in_training_captions(self):
        _, caps, _ = self.build(mixed_specs(300))
        self.assertEqual(len(C.HELD_OUT), 2)
        for text in caps.values():
            for held in C.HELD_OUT.values():
                self.assertNotIn(held, text)
        self.assertTrue(set(C.HELD_OUT.values()).isdisjoint(set(C.TRAIN_PREDICATES["warmer"] + C.TRAIN_PREDICATES["cooler"])))

    def test_held_out_phrasing_is_written_apart_for_evaluation(self):
        d, _, _ = self.build(mixed_specs(60))
        with open(os.path.join(d, "heldout-captions.jsonl")) as fh:
            rows = [json.loads(line) for line in fh]
        self.assertTrue(rows)
        self.assertTrue(all(any(h in r["caption"] for h in C.HELD_OUT.values()) for r in rows))

    def test_thirty_percent_of_tiles_get_the_neutral_caption_only(self):
        _, caps, report = self.build(mixed_specs(200))
        neutral = [t for t in caps.values() if t == C.NEUTRAL_CAPTION]
        self.assertAlmostEqual(len(neutral) / len(caps), 0.3, delta=0.015)
        self.assertEqual(report["neutral_share"], len(neutral) / len(caps))
        self.assertTrue(all(t.startswith(C.NEUTRAL_CAPTION) for t in caps.values()),
                        "every caption opens with the neutral instruction")

    def test_a_tile_without_a_measured_statement_is_neutral_and_pushes_the_share_up(self):
        _, caps, _ = self.build([[]] * 50 + [[obj("car", "warmer")]] * 50)
        self.assertGreaterEqual(sum(t == C.NEUTRAL_CAPTION for t in caps.values()), 50)

    def test_at_most_three_objects_are_named(self):
        specs = [[obj("car", "warmer"), obj("person", "cooler"), obj("bicycle", "same"), obj("dog", "warmer")]] * 80
        _, caps, _ = self.build(specs, neutral_share=0.0)
        for text in caps.values():
            self.assertLessEqual(sum(w in text for w in ("car", "person", "bicycle", "dog")), 3)
        self.assertTrue(any(sum(w in t for w in ("car", "person", "bicycle", "dog")) >= 2 for t in caps.values()),
                        "the multi-object variant is used")

    def test_disagreeing_instances_of_one_class_state_nothing_about_it(self):
        _, caps, _ = self.build([[obj("car", "warmer"), obj("car", "cooler")]] * 10, neutral_share=0.0)
        self.assertEqual(set(caps.values()), {C.NEUTRAL_CAPTION})

    def test_a_clear_majority_of_instances_is_stated_and_a_split_is_not(self):
        three_one = [obj("car", "warmer")] * 3 + [obj("car", "same")]
        _, caps, report = self.build([three_one] * 20, neutral_share=0.0)
        self.assertTrue(all(C.has_polarity(t, "car", "warmer") for t in caps.values()))
        self.assertEqual(report["conflicting_classes"], 0)
        _, caps, report = self.build([[obj("car", "warmer")] * 2 + [obj("car", "same")] * 2] * 20, neutral_share=0.0)
        self.assertEqual(set(caps.values()), {C.NEUTRAL_CAPTION})
        self.assertEqual(report["conflicting_classes"], 20)

    def test_a_part_gets_its_own_subject(self):
        _, caps, _ = self.build([[obj("car", None), obj("car", "warmer", part="bonnet")]] * 30, neutral_share=0.0)
        self.assertTrue(any("bonnet of the car" in t for t in caps.values()))

    def test_output_is_deterministic_and_seed_dependent(self):
        specs = mixed_specs(80)
        a, caps_a, _ = self.build(specs, seed=1)
        b, caps_b, _ = self.build(specs, seed=1)
        _, caps_c, _ = self.build(specs, seed=2)
        self.assertEqual(caps_a, caps_b)
        self.assertNotEqual(caps_a, caps_c)
        for name in ("captions.yaml", "captions-report.json", "heldout-captions.jsonl"):
            with open(os.path.join(a, name), "rb") as x, open(os.path.join(b, name), "rb") as y:
                self.assertEqual(x.read(), y.read(), name)

    def test_the_paraphraser_hook_rewrites_statements_but_cannot_drop_an_object(self):
        def shout(text):
            return text.upper()

        def forget(text):
            return "Something changed."

        _, caps, _ = self.build([[obj("car", "warmer")]] * 20, neutral_share=0.0, paraphrase=shout)
        self.assertTrue(all("CAR" in t and "car" not in t.split(".", 1)[1] for t in caps.values()))
        _, caps, _ = self.build([[obj("car", "warmer")]] * 20, neutral_share=0.0, paraphrase=forget)
        self.assertTrue(all("car" in t for t in caps.values()), "a paraphrase that loses the object falls back to the template")


class Balance(unittest.TestCase):
    def test_classes_with_a_polarity_under_ten_percent_are_flagged_not_controllable(self):
        specs = ([[obj("person", "warmer")]] * 95 + [[obj("person", "cooler")]] * 3 + [[obj("person", "same")]] * 40
                 + [[obj("car", "warmer")]] * 30 + [[obj("car", "cooler")]] * 30 + [[obj("car", "same")]] * 30)
        with tempfile.TemporaryDirectory() as d:
            write_tiles(d, specs)
            report = C.build_captions(d, seed=1)
        person, car = report["classes"]["person"], report["classes"]["car"]
        self.assertEqual((person["warmer"], person["cooler"], person["same"]), (95, 3, 40))
        self.assertAlmostEqual(person["share"]["cooler"], 3 / 138)
        self.assertEqual(person["minority"], "cooler")
        self.assertEqual(person["flag"], "not controllable from natural data")
        self.assertIsNone(car["flag"])


if __name__ == "__main__":
    unittest.main()
