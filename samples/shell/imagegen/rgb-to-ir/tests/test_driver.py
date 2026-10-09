# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset preparation pipelines for
# multimodal detector studies for its clients. If your team needs expertise
# in RGB / thermal-IR detector training data, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Spec tests for the shell driver: every setting is an explicit flag, the
environment configures nothing."""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import test_gates as TG  # noqa: E402
import test_synth as TS  # noqa: E402

DRIVER = os.path.join(os.path.dirname(__file__), "..", "rgb_to_ir.sh")

# A detector evaluation that records how it was called and dumps one perfect prediction.
FAKE_BRAIN = """#!/usr/bin/env python3
import json, sys
argv = sys.argv[1:]
with open(%(log)r, "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
box = [2.0, 2.0, 10.0, 12.0]
with open(argv[argv.index("--dump-preds") + 1], "w") as fh:
    fh.write(json.dumps({"image": 0, "gts": [{"class": 0, "xyxy": box}],
                         "preds": [{"class": 0, "score": 0.9, "xyxy": box}]}) + "\\n")
"""


def run_driver(*args, env=None):
    return subprocess.run(["bash", DRIVER, *args], capture_output=True, text=True,
                          env={"PATH": os.environ["PATH"], **(env or {})})


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")

    def test_a_stage_without_a_work_directory_is_refused_and_the_environment_does_not_supply_one(self):
        done = run_driver("validate", env={"RIR_WORK": self.work})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--work", done.stderr)
        self.assertFalse(os.path.exists(self.work))

    def test_the_seed_flag_seeds_the_packing_order_and_the_environment_does_not(self):
        arm = os.path.join(self.work, "arms", "b1")
        os.makedirs(arm)
        keys = [f"f{k}" for k in range(8)]
        with open(os.path.join(arm, "manifest.jsonl"), "w") as fh:
            for k in keys:
                cv2.imwrite(os.path.join(arm, f"{k}.png"), np.full((8, 8), 10 * len(k), np.uint8))
                fh.write(json.dumps({"dataset": "d", "id": k, "image": f"{k}.png", "boxes": []}) + "\n")

        def order(seed_args, env=None):
            out = run_driver("--work", self.work, *seed_args, "pack", "b1", "--size", "16", env=env)
            self.assertEqual(out.returncode, 0, out.stderr)
            with open(os.path.join(self.work, "packed", "b1", "order.json")) as fh:
                return json.load(fh)

        expected = [f"d:{keys[i]}" for i in np.random.default_rng(5).permutation(len(keys))]
        self.assertEqual(order(["--seed", "5"]), expected)
        self.assertEqual(order([], env={"SEED": "5"}), [f"d:{keys[i]}" for i in np.random.default_rng(1).permutation(len(keys))])

    def test_the_brain_binary_and_the_device_are_flags_placed_before_the_verb(self):
        packed = os.path.join(self.work, "packed", "eval-Test")
        os.makedirs(packed)
        with open(os.path.join(packed, "sequences.json"), "w") as fh:
            json.dump(["s"], fh)
        log = os.path.join(self.tmp.name, "log")
        brain = os.path.join(self.tmp.name, "brain")
        with open(brain, "w") as fh:
            fh.write(FAKE_BRAIN % {"log": log})
        os.chmod(brain, os.stat(brain).st_mode | stat.S_IXUSR)
        done = run_driver("--work", self.work, "--brain", brain, "--device", "gpu1", "evaluate", "a2", "1", "w.safetensors",
                          env={"BRAIN": "/nonexistent/brain"})
        self.assertEqual(done.returncode, 0, done.stderr)
        with open(log) as fh:
            call = json.loads(fh.readline())
        self.assertEqual(call[:4], ["--device", "gpu1", "yolov8", "eval"])
        self.assertTrue(os.path.isfile(os.path.join(self.work, "results", "Test", "a2", "seed1.score.json")))


class SyntheticArmChain(unittest.TestCase):
    """generate -> ingest -> gate -> pack, each a driver stage, over a fake brain."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        os.makedirs(self.work)
        frames = []
        for k in range(8):
            gray = TG.blocks(100 + k)
            for kind, img in (("rgb", TG.to_rgb(gray)), ("ir", TG.to_ir(gray, noise=6.0, seed=k))):
                cv2.imwrite(os.path.join(self.work, f"t{k}_{kind}.png"), img)
        for k in range(8):
            frames.append(self.row(f"t{k}", "T", f"t{k}_rgb.png", f"t{k}_ir.png"))
        for k in range(2):
            cv2.imwrite(os.path.join(self.work, f"s{k}_rgb.png"), TG.to_rgb(TG.blocks(7 + k)))
            cv2.imwrite(os.path.join(self.work, f"s{k}_ir.png"), TG.to_ir(TG.blocks(7 + k)))
            frames.append(self.row(f"s{k}", "S", f"s{k}_rgb.png", f"s{k}_ir.png"))
        with open(os.path.join(self.work, "splits.json"), "w") as fh:
            json.dump({"frames": frames}, fh)
        self.log, self.plan = os.path.join(self.tmp.name, "log"), os.path.join(self.tmp.name, "plan.json")
        with open(self.plan, "w") as fh:
            json.dump([], fh)
        self.brain = os.path.join(self.tmp.name, "brain")
        with open(self.brain, "w") as fh:
            fh.write(TS.FAKE_BRAIN % {"log": self.log, "plan": self.plan})
        os.chmod(self.brain, os.stat(self.brain).st_mode | stat.S_IXUSR)
        with open(os.path.join(self.work, "synth.json"), "w") as fh:
            json.dump({"dit": "d", "vae": "v", "text_encoder": "t", "tokenizer": "k", "strength": 1.0, "seed": 4}, fh)

    def row(self, fid, split, rgb, ir):
        boxes = [{"class": "person", "x1": 30.0, "y1": 30.0, "x2": 110.0, "y2": 120.0}]
        return {"id": fid, "dataset": "d", "rgb": os.path.join(self.work, rgb), "ir": os.path.join(self.work, ir), "boxes": boxes,
                "sequence_id": fid, "split": split, "usable": True, "day_night": "day", "width": TG.W, "height": TG.H}

    def drive(self, *args):
        done = run_driver("--work", self.work, "--brain", self.brain, "--device", "gpu1", *args)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_four_stages_chain_into_a_packed_arm(self):
        cfg = os.path.join(self.work, "synth.json")
        self.drive("generate", "run1", "--config", cfg, "--modes", "neutral,vae-roundtrip")
        with open(self.log) as fh:
            calls = [json.loads(line) for line in fh]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][:5], ["--device", "gpu1", "flux2", "generate", "--variant"])
        self.assertEqual(calls[0][calls[0].index("--seed") + 1], "4")
        self.drive("ingest", "c2", "run1/neutral", "--no-sensor")
        self.drive("ingest", "v0", "run1/vae-roundtrip", "--no-sensor")
        self.drive("gate", "c2", "--keep-rejected")
        self.drive("pack", "c2", "--size", "64")
        self.assertTrue(os.path.isfile(os.path.join(self.work, "packed", "c2", "images.f32")))
        self.assertTrue(os.path.isfile(os.path.join(self.work, "arms", "c2", "gate-stats.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.work, "arms", "v0", "ingest-stats.json")))

    def test_a_stage_that_needs_an_earlier_one_says_which(self):
        done = run_driver("--work", self.work, "ingest", "c2", "run1/neutral", "--no-sensor")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("generate", done.stderr)
        done = run_driver("--work", self.work, "gate", "c2")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("ingest", done.stderr)


if __name__ == "__main__":
    unittest.main()
