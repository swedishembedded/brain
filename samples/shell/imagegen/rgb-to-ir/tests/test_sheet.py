# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset inspection tooling for paired
# RGB / thermal-IR training sets for its clients. If your team needs
# expertise in auditing instruction-conditioned image-editing data, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Spec test for rir_sheet: a sheet of the asked size, drawn from measured tiles only."""
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
import rir_sheet as Sheet  # noqa: E402
import rir_tiles as T  # noqa: E402
from test_regions import BoxSegmenter, noisy_frame, plant  # noqa: E402


class ContactSheet(unittest.TestCase):
    def test_sheet_has_the_requested_grid_and_names_only_measured_tiles(self):
        with tempfile.TemporaryDirectory() as d:
            ir = plant(noisy_frame(31), (30, 40, 70, 80), +40)
            index = []
            for k in range(7):
                name = f"t{k}"
                cv2.imwrite(os.path.join(d, f"{name}_rgb.png"), np.dstack([ir] * 3))
                cv2.imwrite(os.path.join(d, f"{name}_ir.png"), np.dstack([ir] * 3))
                boxes = [{"class": "car", "x1": 30, "y1": 40, "x2": 70, "y2": 80}] if k < 5 else []
                index.append({"name": name, "accepted": True, "rgb": f"{name}_rgb.png", "ir": f"{name}_ir.png", "boxes": boxes})
            T.write_index(d, index)
            R.measure_set(d, BoxSegmenter(), seed=1)
            C.build_captions(d, seed=1)
            out = os.path.join(d, "sheet.png")
            names = Sheet.contact_sheet(d, out, BoxSegmenter(), count=4, columns=2, panel=64, seed=1)
            self.assertEqual(len(names), 4)
            self.assertTrue(set(names) <= {f"t{k}" for k in range(5)}, "tiles without a measurement are not sampled")
            self.assertEqual(names, Sheet.contact_sheet(d, out, BoxSegmenter(), count=4, columns=2, panel=64, seed=1))
            img = cv2.imread(out)
            self.assertEqual(img.shape[1], 2 * 2 * 64)
            self.assertEqual(img.shape[0], 2 * (64 + 70))
            self.assertGreater(img.std(), 5)


if __name__ == "__main__":
    unittest.main()
