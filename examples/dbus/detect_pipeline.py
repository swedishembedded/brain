#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Full multi-model pipeline over brain's D-Bus surface: **generate → detect → draw**.

Generates an image with z-image, runs YOLOv8 object detection on it, and saves the
image with labeled boxes drawn over the detections — every step over D-Bus, exchanging
the image as a file descriptor. Both models are served and scheduled by the same
residency Executor.

    dbus-run-session -- bash -c '
        BRAIN_ZIMAGE_*=... BRAIN_YOLO=... brain serve --dbus &
        sleep 1
        python3 examples/dbus/detect_pipeline.py'
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import to_pil  # noqa: E402

OUT = Path(os.environ.get("OUT", "/tmp"))
SIZE = int(os.environ.get("SIZE", "512"))
STEPS = int(os.environ.get("STEPS", "8"))
PROMPT = os.environ.get("PROMPT", "two dogs sitting side by side on grass, full body, realistic wildlife photograph")


def generate(brain: BrainDBus) -> bytes:
    """Stream a z-image text2image generation; return the HWC-f32 image bytes."""
    params = {"prompt": PROMPT, "width": SIZE, "height": SIZE, "steps": STEPS, "seed": 7}

    def on_progress(step: int, total: int, message: str) -> None:
        print(f"  [{step}/{total}] {message}")

    out = brain.subscribe("z-image", "text2image", params, on_progress=on_progress)
    image = out.blobs.get("image")
    if not image:
        raise RuntimeError("no image produced")
    return image


def detect(brain: BrainDBus, image: bytes) -> list[dict]:
    """Run yolo detection on the image (passed as an input blob)."""
    meta = {"image": {"media": "image", "w": SIZE, "h": SIZE, "c": 3}}
    out = brain.run("yolo", "detect", {"conf": 0.25}, blobs={"image": image}, meta=meta)
    return out.outputs.get("detections", [])


def annotate(image: bytes, detections: list[dict]):
    """Return a PIL image with labeled boxes drawn over the detections."""
    from PIL import ImageDraw

    img = to_pil(image, SIZE, SIZE)
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
        draw.text((x1 + 3, y1 + 3), f"{d['label']} {d['conf']:.2f}", fill=(255, 255, 0))
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with BrainDBus() as brain:
        models = brain.models()
        print("models:", models)
        for need in ("z-image", "yolo"):
            if need not in models:
                skip(f"'{need}' not served (set its weights env)")

        # Step 1 — generate. Save the raw z-image output.
        print(f"[1/3 generate] z-image {SIZE}x{SIZE}: {PROMPT!r}")
        t = time.perf_counter()
        image = generate(brain)
        gen_s = time.perf_counter() - t
        gen_path = OUT / "step1_generated.png"
        to_pil(image, SIZE, SIZE).save(gen_path)
        print(f"[1/3 generate] {gen_s:.1f}s -> {gen_path}")

        # Step 2 — detect (over D-Bus, image passed as an fd).
        t = time.perf_counter()
        detections = detect(brain, image)
        det_s = time.perf_counter() - t
        summary = ", ".join(f"{d['label']}({d['conf']:.2f})" for d in detections)
        print(f"[2/3 detect] {len(detections)} objects in {det_s:.2f}s: {summary}")

        # Step 3 — annotate. Save the final labeled image.
        box_path = OUT / "step3_boxes.png"
        annotate(image, detections).save(box_path)
        print(f"[3/3 draw] labeled boxes -> {box_path}")
        print("stats:", brain.stats())
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
