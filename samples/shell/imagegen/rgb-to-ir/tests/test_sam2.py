# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements promptable-segmentation and visual-grounding
# integrations for dataset annotation pipelines for its clients. If your team
# needs expertise in SAM 2 or Florence-2 based annotation workflows, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Plumbing tests for rir_sam2 against a fake `brain` executable: the command
lines, the mask decoding and the grounder's coordinate mapping. The real
models are exercised by running the stage, not here."""
import os
import stat
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import rir_sam2 as S  # noqa: E402

FAKE_BRAIN = """#!/usr/bin/env python3
LOG_PATH = %(log)r
import json, os, sys
import cv2, numpy as np
argv = sys.argv[1:]
with open(LOG_PATH, "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
opt = lambda name: argv[argv.index(name) + 1]
image = cv2.imread(opt("--in").split("=", 1)[1])
if "segment" in argv:
    x1, y1, x2, y2 = (int(float(v)) for v in opt("--box").split(","))
    mask = np.zeros(image.shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    cv2.imwrite(opt("--out").split("=", 1)[1], mask)
    print(json.dumps({"area": (x2 - x1) * (y2 - y1)}))
else:
    print("florence2: loading")
    print(json.dumps({"found": opt("--target") != "the nothing", "boxes": [{"phrase": opt("--target"), "bbox": [0.25, 0.5, 0.75, 1.0]}]}))
"""


class CliPlumbing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.brain = os.path.join(self.tmp.name, "brain")
        self.log = os.path.join(self.tmp.name, "log")
        with open(self.brain, "w") as fh:
            fh.write(FAKE_BRAIN % {"log": self.log})
        os.chmod(self.brain, os.stat(self.brain).st_mode | stat.S_IXUSR)

    def calls(self):
        import json

        with open(self.log) as fh:
            return [json.loads(line) for line in fh]

    def test_one_process_per_box_with_the_global_flags_before_the_verb(self):
        seg = S.CliSegmenter(self.brain, ["--device", "gpu1"], "tiny")
        masks = seg.segment(np.zeros((40, 60, 3), np.uint8), [(5, 6, 20, 30), (30, 10, 50, 20)])
        self.assertEqual([m.sum() for m in masks], [15 * 24, 20 * 10])
        self.assertEqual(masks[0].shape, (40, 60))
        first = self.calls()[0]
        self.assertEqual(first[:5], ["--device", "gpu1", "sam2", "segment", "--variant"])
        self.assertIn("--json", first)
        self.assertEqual(first[first.index("--box") + 1], "5.0,6.0,20.0,30.0")
        self.assertEqual(len(self.calls()), 2)

    def test_a_failing_process_is_an_error_carrying_its_stderr(self):
        with open(self.brain, "w") as fh:
            fh.write("#!/bin/sh\necho 'out of memory' >&2\nexit 3\n")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            S.CliSegmenter(self.brain).segment(np.zeros((8, 8, 3), np.uint8), [(0, 0, 4, 4)])

    def test_grounder_maps_normalised_crop_boxes_back_to_image_pixels(self):
        grounder = S.CliGrounder(self.brain, pad=0.0)
        found = grounder.ground(np.zeros((100, 200, 3), np.uint8), (40, 20, 140, 80), ["bonnet"])
        # crop = x 40..140, y 20..80 (100 x 60); bbox [0.25, 0.5, 0.75, 1.0] of it
        self.assertEqual(found["bonnet"], (65.0, 50.0, 115.0, 80.0))

    def test_a_part_the_grounder_does_not_find_is_none(self):
        found = S.CliGrounder(self.brain, pad=0.0).ground(np.zeros((100, 200, 3), np.uint8), (40, 20, 140, 80),
                                                          ["nothing"])
        self.assertEqual(found, {"nothing": None})


class GroundParsing(unittest.TestCase):
    def test_boxes_may_arrive_as_a_json_string(self):
        out = 'log line\n{"found": true, "boxes": "[{\\"phrase\\": \\"x\\", \\"bbox\\": [0, 0, 1, 1]}]"}\n'
        self.assertEqual(S.parse_ground_boxes(out), [[0.0, 0.0, 1.0, 1.0]])

    def test_output_without_json_is_an_error(self):
        with self.assertRaises(ValueError):
            S.parse_ground_boxes("nothing here")


if __name__ == "__main__":
    unittest.main()
