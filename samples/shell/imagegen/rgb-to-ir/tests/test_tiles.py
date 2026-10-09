# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements paired-image training-set preparation for
# image-editing LoRA fine-tuning for its clients. If your team needs
# expertise in reference-to-target dataset pipelines for diffusion
# fine-tuning, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_tiles: aligned square tiles of split-T pairs only, in the
folder format `brain flux2 finetune` reads."""
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
import rir_tiles as T  # noqa: E402

FRAME_W, FRAME_H, SIZE = 160, 96, 96


def rect_scene(seed, h=FRAME_H, w=FRAME_W):
    """Piecewise-constant gray scene with sharp edges everywhere."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.uint8)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w - 12)), int(rng.integers(0, h - 12))
        img[y0:y0 + int(rng.integers(8, 30)), x0:x0 + int(rng.integers(8, 40))] = int(rng.integers(30, 230))
    return img


def write_pair(root, name, seed, ir_shift=0, rgb_inset=False):
    gray = rect_scene(seed)
    rgb = np.dstack([gray, gray, gray])
    if rgb_inset:
        padded = np.zeros_like(rgb)
        padded[24:72, 40:120] = cv2.resize(rgb, (80, 48))
        rgb = padded
    ir = np.roll(gray, ir_shift, axis=1) if ir_shift else gray
    cv2.imwrite(os.path.join(root, f"{name}_rgb.png"), rgb)
    cv2.imwrite(os.path.join(root, f"{name}_ir.png"), ir)
    return os.path.join(root, f"{name}_rgb.png"), os.path.join(root, f"{name}_ir.png")


def make_doc(root, frames):
    """frames: (id, split, kwargs for write_pair, usable)."""
    rows = []
    for i, (fid, split, kw, usable) in enumerate(frames):
        rgb, ir = write_pair(root, fid, i + 1, **kw)
        rows.append({"id": fid, "dataset": "d", "rgb": rgb, "ir": ir, "sequence_id": f"s{i}", "split": split,
                     "usable": usable, "reason": None if usable else "x", "width": FRAME_W, "height": FRAME_H,
                     "boxes": [{"class": "car", "x1": 10, "y1": 10, "x2": 50, "y2": 40},
                               {"class": "person", "x1": 150, "y1": 80, "x2": 159, "y2": 95}]})
    return {"frames": rows}


def read_index(out):
    with open(os.path.join(out, "tiles.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def read_flat_yaml(path):
    """The two manifests are flat `key: value` mappings, one per line."""
    out = {}
    with open(path) as fh:
        for line in fh:
            key, value = line.rstrip("\n").split(": ", 1)
            out[key] = json.loads(value) if value.startswith('"') else value
    return out


class Grid(unittest.TestCase):
    def test_a_640x512_frame_gets_a_left_and_a_right_square(self):
        self.assertEqual(T.tile_grid(640, 512, 512), [(0, 0), (128, 0)])

    def test_every_pixel_of_any_aspect_is_covered_by_exact_squares(self):
        for w, h, size in [(640, 512, 512), (1280, 1024, 512), (100, 100, 64), (97, 301, 40), (64, 64, 64)]:
            covered = np.zeros((h, w), bool)
            for x0, y0 in T.tile_grid(w, h, size):
                self.assertTrue(0 <= x0 <= w - size and 0 <= y0 <= h - size, (w, h, size, x0, y0))
                covered[y0:y0 + size, x0:x0 + size] = True
            self.assertTrue(covered.all(), (w, h, size))

    def test_a_frame_smaller_than_the_tile_is_an_error_not_a_resize(self):
        with self.assertRaises(ValueError):
            T.tile_grid(300, 200, 256)


class Cutting(unittest.TestCase):
    def cut(self, root, doc, out, **kw):
        return T.cut_tiles(doc, out, size=SIZE, seed=1, **kw)

    def test_only_split_t_pairs_are_tiled_never_s_v_or_test(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [("t1", "T", {}, True), ("s1", "S", {}, True), ("v1", "V", {}, True),
                               ("x1", "Test", {}, True), ("u1", "T", {}, False)])
            out = os.path.join(d, "out")
            self.cut(d, doc, out)
            self.assertEqual({r["id"] for r in read_index(out)}, {"t1"})
            self.assertEqual({r["id"] for r in read_index(out) if r["accepted"]}, {"t1"})

    def test_both_tiles_share_one_crop_box_so_alignment_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [("t1", "T", {}, True)])
            out = os.path.join(d, "out")
            self.cut(d, doc, out)
            tiles = [r for r in read_index(out) if r["accepted"]]
            self.assertEqual(len(tiles), 2)
            full = cv2.imread(doc["frames"][0]["rgb"], cv2.IMREAD_UNCHANGED)
            for t in tiles:
                rgb = cv2.imread(os.path.join(out, t["rgb"]), cv2.IMREAD_UNCHANGED)
                ir = cv2.imread(os.path.join(out, t["ir"]), cv2.IMREAD_UNCHANGED)
                self.assertEqual(rgb.shape, (SIZE, SIZE, 3))
                self.assertEqual(ir.shape, (SIZE, SIZE, 3), "the IR target is written as 3 channels")
                np.testing.assert_array_equal(rgb, full[t["y0"]:t["y0"] + SIZE, t["x0"]:t["x0"] + SIZE])
                np.testing.assert_array_equal(ir[..., 0], rgb[..., 0])  # synthetic IR equals the RGB gray

    def test_pairs_and_captions_are_in_the_format_brain_finetune_reads(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [("t1", "T", {}, True), ("t2", "T", {}, True)])
            out = os.path.join(d, "out")
            self.cut(d, doc, out)
            tiles = [r for r in read_index(out) if r["accepted"]]
            pairs = read_flat_yaml(os.path.join(out, "pairs.yaml"))
            caps = read_flat_yaml(os.path.join(out, "captions.yaml"))
            self.assertEqual(pairs, {t["ir"]: t["rgb"] for t in tiles})
            self.assertEqual(set(caps), {t["ir"] for t in tiles}, "captions cover targets only, never references")
            self.assertEqual(set(caps.values()), {T.NEUTRAL_CAPTION})
            self.assertEqual(T.NEUTRAL_CAPTION, "Convert to thermal infrared, white-hot.")
            for name in list(pairs) + list(pairs.values()):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)

    def test_boxes_are_clipped_into_tile_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [("t1", "T", {}, True)])
            out = os.path.join(d, "out")
            self.cut(d, doc, out)
            by_x = {r["x0"]: r for r in read_index(out) if r["accepted"]}
            left, right = by_x[0], by_x[FRAME_W - SIZE]
            self.assertEqual([(b["class"], b["x1"], b["x2"]) for b in left["boxes"]], [("car", 10, 50)])
            self.assertEqual([(b["class"], b["x1"], b["x2"]) for b in right["boxes"]],
                             [("person", 150 - (FRAME_W - SIZE), 159 - (FRAME_W - SIZE))])

    def test_a_black_padded_rgb_and_a_misregistered_ir_are_rejected_with_a_reason(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [("ok", "T", {}, True), ("pad", "T", {"rgb_inset": True}, True),
                               ("shift", "T", {"ir_shift": 6}, True)])
            out = os.path.join(d, "out")
            self.cut(d, doc, out)
            rows = read_index(out)
            accepted = {r["id"] for r in rows if r["accepted"]}
            self.assertEqual(accepted, {"ok"})
            reasons = {r["id"]: r["reason"] for r in rows if not r["accepted"]}
            self.assertEqual(reasons["pad"], "rgb_black_padding")
            self.assertEqual(reasons["shift"], "edge_misaligned")
            files = set(os.listdir(out))
            self.assertFalse([f for f in files if f.startswith(("d_pad", "d_shift"))], "rejected tiles are not written")

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            doc = make_doc(d, [(f"t{i}", "T", {}, True) for i in range(5)])
            outs = [os.path.join(d, f"out{k}") for k in range(2)]
            for o in outs:
                self.cut(d, doc, o, limit=3)
            self.assertEqual(sorted(os.listdir(outs[0])), sorted(os.listdir(outs[1])))
            for name in os.listdir(outs[0]):
                with open(os.path.join(outs[0], name), "rb") as a, open(os.path.join(outs[1], name), "rb") as b:
                    self.assertEqual(a.read(), b.read(), name)
            self.assertEqual(len({r["id"] for r in read_index(outs[0])}), 3)


if __name__ == "__main__":
    unittest.main()
