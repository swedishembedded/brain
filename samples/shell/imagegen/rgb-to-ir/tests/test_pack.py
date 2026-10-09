# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements detector training-set packaging for its
# clients. If your team needs expertise in object-detection data pipelines or
# edge-AI model fine-tuning, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Spec tests for rir_pack: letterbox box transform, byte layout, determinism."""
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import rir_pack as P  # noqa: E402

SIZE = 64


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def bread(path):
    with open(path, "rb") as fh:
        return fh.read()


def write_png(path, value, shape=(40, 80), channels=3):
    img = np.full(shape if channels == 1 else (*shape, 3), value, np.uint8)
    cv2.imwrite(path, img)


def read_pack(out):
    meta = jload(os.path.join(out, "meta.json"))
    imgs = np.fromfile(os.path.join(out, "images.f32"), dtype="<f4")
    raw = bread(os.path.join(out, "boxes.bin"))
    boxes, off = [], 0
    for _ in range(meta["n"]):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        rec = []
        for _ in range(n):
            rec.append(struct.unpack_from("<Iffff", raw, off))
            off += 20
        boxes.append(rec)
    assert off == len(raw), "trailing bytes in boxes.bin"
    return meta, imgs.reshape(meta["n"], 3, meta["h"], meta["w"]), boxes


class Letterbox(unittest.TestCase):
    def test_box_follows_the_content_through_the_letterbox(self):
        # A wide 200x100 frame with a bright 40x20 patch at x 60..100, y 30..50.
        img = np.full((100, 200, 3), 20, np.uint8)
        img[30:50, 60:100] = 250
        canvas, scale, px, py = P.letterbox(img, SIZE)
        self.assertEqual(canvas.shape, (SIZE, SIZE, 3))
        (cls, x1, y1, x2, y2), = P.letterbox_boxes([(1, 60, 30, 100, 50)], scale, px, py, SIZE)
        inside = canvas[int(y1) + 1:int(y2) - 1, int(x1) + 1:int(x2) - 1]
        self.assertGreater(inside.mean(), 200)
        outside = canvas[int(y1) + 1:int(y2) - 1, int(x2) + 3:int(x2) + 6]
        self.assertLess(outside.mean(), 60)
        # and the transform inverts back to the source box
        self.assertAlmostEqual((x1 - px) / scale, 60, delta=1.0)
        self.assertAlmostEqual((y2 - py) / scale, 50, delta=1.0)

    def test_padding_is_grey_and_centred(self):
        canvas, scale, px, py = P.letterbox(np.zeros((100, 200, 3), np.uint8), SIZE)
        self.assertEqual((scale, px, py), (0.32, 0, 16))
        self.assertTrue((canvas[0] == 114).all())

    def test_boxes_are_clipped_and_degenerate_ones_dropped(self):
        out = P.letterbox_boxes([(0, -10, -10, 20, 20), (2, 500, 500, 600, 600), (0, 10, 10, 10.2, 30)], 1.0, 0, 0, SIZE)
        self.assertEqual(out, [(0, 0.0, 0.0, 20.0, 20.0)])


class Layout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def items(self, n=3):
        items = []
        for i in range(n):
            path = os.path.join(self.tmp, f"{i}.png")
            write_png(path, 10 + 30 * i)
            items.append(P.PackItem(key=f"f{i}", image=path, boxes=[(i % 3, 8, 8, 40, 24)]))
        return items

    def test_byte_layout_matches_brains_flat_yolo_format(self):
        out = os.path.join(self.tmp, "pack")
        P.pack(self.items(3), out, SIZE, nc=3, seed=0)
        self.assertEqual(os.path.getsize(os.path.join(out, "images.f32")), 3 * 3 * SIZE * SIZE * 4)
        meta, imgs, boxes = read_pack(out)
        self.assertEqual(meta, {"n": 3, "c": 3, "h": SIZE, "w": SIZE, "nc": 3})
        self.assertTrue(0.0 <= imgs.min() and imgs.max() <= 1.0)
        for rec in boxes:
            self.assertEqual(len(rec), 1)
            _, cx, cy, w, h = rec[0]
            self.assertTrue(0 < cx < 1 and 0 < cy < 1 and 0 < w <= 1 and 0 < h <= 1)

    def test_single_channel_ir_is_replicated_to_three_channels(self):
        path = os.path.join(self.tmp, "ir.png")
        write_png(path, 200, channels=1)
        out = os.path.join(self.tmp, "pack")
        P.pack([P.PackItem("ir", path, [])], out, SIZE, nc=3, seed=0)
        _, imgs, boxes = read_pack(out)
        self.assertTrue(np.array_equal(imgs[0, 0], imgs[0, 1]) and np.array_equal(imgs[0, 1], imgs[0, 2]))
        self.assertAlmostEqual(float(imgs[0, 0, SIZE // 2, SIZE // 2]), 200 / 255, places=2)
        self.assertEqual(boxes, [[]])

    def test_deterministic_for_a_seed_and_shuffled(self):
        a, b, c = (os.path.join(self.tmp, n) for n in "abc")
        P.pack(self.items(8), a, SIZE, 3, seed=5)
        P.pack(self.items(8), b, SIZE, 3, seed=5)
        P.pack(self.items(8), c, SIZE, 3, seed=6)
        for name in ("images.f32", "boxes.bin", "meta.json", "order.json"):
            self.assertEqual(bread(os.path.join(a, name)), bread(os.path.join(b, name)))
        order_a = jload(os.path.join(a, "order.json"))
        self.assertEqual(sorted(order_a), [f"f{i}" for i in range(8)])
        self.assertNotEqual(order_a, [f"f{i}" for i in range(8)])
        self.assertNotEqual(order_a, jload(os.path.join(c, "order.json")))

    def test_order_json_names_the_item_stored_at_each_index(self):
        out = os.path.join(self.tmp, "pack")
        P.pack(self.items(5), out, SIZE, 3, seed=2)
        _, imgs, _ = read_pack(out)
        for idx, key in enumerate(jload(os.path.join(out, "order.json"))):
            i = int(key[1:])
            self.assertAlmostEqual(float(imgs[idx, 0, SIZE // 2, SIZE // 2]), (10 + 30 * i) / 255, places=2)

    def test_class_outside_nc_is_rejected(self):
        path = os.path.join(self.tmp, "x.png")
        write_png(path, 5)
        with self.assertRaises(ValueError):
            P.pack([P.PackItem("x", path, [(3, 1, 1, 20, 20)])], os.path.join(self.tmp, "o"), SIZE, nc=3, seed=0)


if __name__ == "__main__":
    unittest.main()
