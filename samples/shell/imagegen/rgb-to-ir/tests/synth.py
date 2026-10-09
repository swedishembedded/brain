# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters and leakage-safe evaluation
# protocols for multimodal perception pipelines for its clients. If your team
# needs expertise in paired RGB / thermal-IR data, detector training sets or
# sensor-domain adaptation, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Tiny synthetic RGB / IR pairs written in three common on-disk layouts.

A "scene" is a smooth random pattern fixed by its seed; frame k of a scene is
that pattern shifted by k pixels plus faint noise. Two frames of one scene are
therefore strongly correlated and two scenes are not - exactly the property the
sequence-recovery and near-duplicate logic rely on.

Layouts (each writer takes a temp dir and returns the reader arguments that
read it back):

  voc_pairs    <root>/images/NAME_RGB.jpg + NAME_IR.png (1-channel), Pascal VOC
               <root>/ann/NAME_IR.xml, ids NAME = "v<number>", a test list file
  yolo_alpha   <root>/{train,val}/images/NNNNN.tiff, BGRA with the IR in the
               alpha channel, <root>/{train,val}/labels/NNNNN.txt (YOLO)
  coco_rgbir   <root>/visible/<split>/SSNNNN.jpg + <root>/infrared/<split>/SSNNNN.jpg
               (3-channel replicated IR), one COCO json
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # synthetic 4-channel TIFFs trigger libtiff warnings

import cv2
import numpy as np

H, W = 48, 64


def scene_gray(seed: int, k: int = 0, level: tuple[int, int] = (30, 220)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.random((6, 8)).astype(np.float32), (W + 16, H), interpolation=cv2.INTER_CUBIC)
    img = np.roll(base, k, axis=1)[:, :W]
    noise = np.random.default_rng(seed * 1000 + k).normal(0, 0.01, img.shape)
    img = np.clip(img + noise, 0, 1)
    return (level[0] + img * (level[1] - level[0])).astype(np.uint8)


def rgb_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.dstack([gray, np.clip(gray.astype(int) + 5, 0, 255).astype(np.uint8), gray])


def pad_into_subrect(img: np.ndarray, frac: float = 0.7) -> np.ndarray:
    """RGB occupying a centred sub-rectangle of an otherwise black frame."""
    h, w = img.shape[:2]
    nh, nw = int(h * frac), int(w * frac)
    out = np.zeros_like(img)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return out


# One person, one car; the dog is outside any study vocabulary and is dropped by consumers.
VOC_XML = """<annotation><size><width>64</width><height>48</height></size>
<object><name>person</name><bndbox><xmin>10</xmin><ymin>8</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
<object><name>car</name><bndbox><xmin>30</xmin><ymin>20</ymin><xmax>60</xmax><ymax>40</ymax></bndbox></object>
<object><name>dog</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def write_voc_pairs(root: str, train_videos: list[dict], test_videos: list[dict]) -> dict:
    """Each video: {"ids": numbers, "scene": seed, "level": (lo, hi), "padded": set of numbers whose RGB is
    shrunk, "scenes": {number: seed} to switch scene mid-run}. Returns reader args (paths absolute)."""
    for sub in ("images", "ann"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    test_names = []
    for videos, is_test in ((train_videos, False), (test_videos, True)):
        for v in videos:
            for k, num in enumerate(v["ids"]):
                name = f"v{num:05d}"
                gray = scene_gray(v.get("scenes", {}).get(num, v["scene"]), k, v.get("level", (30, 220)))
                rgb = rgb_from_gray(gray)
                if num in v.get("padded", ()):
                    rgb = pad_into_subrect(rgb)
                cv2.imwrite(os.path.join(root, "images", f"{name}_RGB.jpg"), rgb)
                cv2.imwrite(os.path.join(root, "images", f"{name}_IR.png"), gray)
                with open(os.path.join(root, "ann", f"{name}_IR.xml"), "w") as fh:
                    fh.write(VOC_XML)
                if is_test:
                    test_names.append(name)
    with open(os.path.join(root, "test_list.txt"), "w") as fh:
        fh.write("\n".join(test_names) + "\n")
    return {"rgb_glob": os.path.join(root, "images", "*_RGB.jpg"), "ir_glob": os.path.join(root, "images", "*_IR.png"),
            "labels_dir": os.path.join(root, "ann"), "label_format": "voc",
            "pair_regex": r"^(.+?)(?:_RGB|_IR)?\.[^.]+$", "numeric_id_regex": r"v(\d+)",
            "test_list": os.path.join(root, "test_list.txt"), "sequence_rule": "similarity"}


def write_yolo_alpha(root: str, train: dict[int, int], val: dict[int, int]) -> dict:
    """Frame number -> scene seed, for the `train` and `val` folders. Returns reader args."""
    for sub, frames in (("train", train), ("val", val)):
        os.makedirs(os.path.join(root, sub, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, sub, "labels"), exist_ok=True)
        for num, seed in frames.items():
            gray = scene_gray(seed, num % 7, (20, 90))
            bgra = np.dstack([rgb_from_gray(gray), np.clip(255 - gray, 0, 255).astype(np.uint8)])
            cv2.imwrite(os.path.join(root, sub, "images", f"{num:05d}.tiff"), bgra)
            with open(os.path.join(root, sub, "labels", f"{num:05d}.txt"), "w") as fh:
                fh.write("0 0.5 0.5 0.25 0.5\n1 0.25 0.25 0.1 0.1\n")
    return {"rgb_glob": os.path.join(root, "*", "images", "*.tiff"), "ir_channel": "alpha",
            "labels_dir": root, "label_format": "yolo", "yolo_names": "people,car",
            "numeric_id_regex": r"(\d+)", "official_split_regex": r"/(train|val)/images/"}


def write_coco_rgbir(root: str, train_scenes: dict[str, int], test_scenes: dict[str, int], frames: int = 6,
                     level: dict[str, tuple[int, int]] | None = None) -> dict:
    """Scene prefix (2 digits) -> seed. File names are prefix + 4-digit index. Returns reader args."""
    images, annotations = [], []
    for split, scenes in (("train", train_scenes), ("test", test_scenes)):
        for kind in ("visible", "infrared"):
            os.makedirs(os.path.join(root, kind, split), exist_ok=True)
        for prefix, seed in scenes.items():
            for k in range(frames):
                gray = scene_gray(seed, k, (level or {}).get(prefix, (20, 120)))
                name = f"{prefix}{k + 1:04d}.jpg"
                cv2.imwrite(os.path.join(root, "visible", split, name), rgb_from_gray(gray))
                cv2.imwrite(os.path.join(root, "infrared", split, name), np.dstack([gray] * 3))
                images.append({"id": len(images), "file_name": f"{split}/{name}", "width": W, "height": H})
                annotations.append({"id": len(annotations), "image_id": images[-1]["id"], "category_id": 1,
                                    "bbox": [10, 8, 10, 22]})
    with open(os.path.join(root, "coco.json"), "w") as fh:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "pedestrian"}]}, fh)
    return {"rgb_glob": os.path.join(root, "visible", "*", "*.jpg"), "ir_glob": os.path.join(root, "infrared", "*", "*.jpg"),
            "label_format": "coco", "coco_json": os.path.join(root, "coco.json"), "class_map": "pedestrian=person",
            "sequence_rule": r"regex:/(\d\d)\d{4}\.jpg$", "official_split_regex": r"/visible/(train|test)/"}
