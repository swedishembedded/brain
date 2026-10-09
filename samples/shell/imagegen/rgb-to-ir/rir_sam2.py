# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements promptable-segmentation and visual-grounding
# integrations for dataset annotation pipelines for its clients. If your team
# needs expertise in SAM 2 or Florence-2 based annotation workflows, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Segmenters and the part grounder behind rir_regions, over brain.

`CliSegmenter` runs one `brain sam2 segment` process per box. Every process
loads the checkpoint and initialises the device again, which dominated the
cost when measured (tens of seconds per box on a GPU), so it is the simple,
dependency-free path and not the one for a tile set.

`DbusSegmenter` talks to ONE resident `brain serve --dbus` through brain-py:
the checkpoint is loaded once, the image encoding is cached per image, and the
boxes of a tile after the first cost only the mask decoder. It needs the
`jeepney` package (brain-py's D-Bus client) and finds brain-py at the
repository root, or at the directory given as `brain_py` / `--brain-py`.

`CliGrounder` boxes named parts of an object with `brain florence2 ground`
(image + a phrase in, normalised [x0,y0,x1,y1] boxes out). The object is
cropped first so the grounder works at the object's own scale.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def _box_arg(box) -> str:
    return ",".join(f"{v:.1f}" for v in box)


def _run(cmd: list[str], timeout: float) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({done.returncode}): {done.stderr.strip()[-400:]}")
    return done.stdout


class CliSegmenter:
    def __init__(self, brain: str = "brain", global_args: list[str] | None = None, variant: str = "tiny",
                 timeout: float = 600.0):
        self.brain, self.global_args, self.variant, self.timeout = brain, list(global_args or []), variant, timeout

    def segment(self, bgr: np.ndarray, boxes) -> list[np.ndarray]:
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "image.png")
            cv2.imwrite(image, bgr)
            masks = []
            for k, box in enumerate(boxes):
                out = os.path.join(tmp, f"mask{k}.png")
                _run([self.brain, *self.global_args, "sam2", "segment", "--variant", self.variant,
                      "--in", f"image={image}", "--box", _box_arg(box), "--out", f"mask={out}", "--json"], self.timeout)
                mask = cv2.imread(out, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise RuntimeError(f"brain sam2 segment wrote no readable mask for box {_box_arg(box)}")
                masks.append((mask if mask.ndim == 2 else mask.max(axis=2)) > 127)
            return masks


def _brain_py_path(override: str | None = None) -> str:
    """The brain-py directory: the explicit `override`, else the one in this repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    return override or os.path.normpath(os.path.join(here, "..", "..", "..", "..", "brain-py"))


class DbusSegmenter:
    """SAM 2 resident in `brain serve --dbus`; `address` is the bus address (or SESSION / SYSTEM)."""

    def __init__(self, address: str = "SESSION", variant: str = "tiny", brain_py: str | None = None):
        sys.path.insert(0, _brain_py_path(brain_py))
        try:
            from brain_py.dbus import BrainDBus
        except ImportError as e:
            raise RuntimeError(f"the D-Bus backend needs brain-py ({_brain_py_path(brain_py)}, or pass --brain-py) and its "
                               f"`jeepney` dependency: {e}") from e
        self._brain = BrainDBus(bus=address)
        self.variant = variant

    def close(self) -> None:
        self._brain.close()

    def segment(self, bgr: np.ndarray, boxes) -> list[np.ndarray]:
        h, w = bgr.shape[:2]
        image = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("<f4") / 255.0).tobytes()
        meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}
        masks = []
        for box in boxes:
            out = self._brain.run("brain/sam2", "segment", {"box": _box_arg(box), "variant": self.variant},
                                  blobs={"image": image}, meta=meta)
            masks.append(np.frombuffer(out.blobs["mask"], "<f4").reshape(h, w) > 0.5)
        return masks


def parse_ground_boxes(stdout: str) -> list[list[float]]:
    """Normalised boxes from the last JSON line of `brain florence2 ground --json`: `{found, boxes: [{phrase, bbox}]}`."""
    lines = [ln for ln in stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise ValueError(f"no JSON in the grounder output: {stdout[-200:]!r}")
    doc = json.loads(lines[-1])
    boxes = doc.get("boxes", [])
    if isinstance(boxes, str):
        boxes = json.loads(boxes)
    return [list(map(float, b["bbox"])) for b in boxes if doc.get("found", True)]


class CliGrounder:
    def __init__(self, brain: str = "brain", global_args: list[str] | None = None, pad: float = 0.1, timeout: float = 600.0):
        self.brain, self.global_args, self.pad, self.timeout = brain, list(global_args or []), pad, timeout

    def ground(self, bgr: np.ndarray, box, parts: list[str]) -> dict:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = self.pad * (x2 - x1), self.pad * (y2 - y1)
        cx1, cy1, cx2, cy2 = max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py))
        crop = bgr[cy1:cy2, cx1:cx2]
        found = {}
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "crop.png")
            cv2.imwrite(image, crop)
            for part in parts:
                out = _run([self.brain, *self.global_args, "florence2", "ground", "--in", f"image={image}",
                            "--target", f"the {part}", "--json"], self.timeout)
                boxes = parse_ground_boxes(out)
                found[part] = None if not boxes else (cx1 + boxes[0][0] * (cx2 - cx1), cy1 + boxes[0][1] * (cy2 - cy1),
                                                      cx1 + boxes[0][2] * (cx2 - cx1), cy1 + boxes[0][3] * (cy2 - cy1))
        return found
