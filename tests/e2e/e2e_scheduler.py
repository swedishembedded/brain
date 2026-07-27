#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""End-to-end scheduler + multi-model validation over brain's D-Bus surface.

Drives ``brain serve --dbus`` through the residency Executor and validates, with real
numbers:

1. **generate** — z-image ``text2image`` "two dogs" (streaming), timed + saved;
2. **detect**   — yolo ``detect`` on that image (when ``BRAIN_YOLO`` is configured),
   boxes drawn into the image buffer, saved;
3. **batch**    — several ``text2image`` requests fired concurrently must coalesce
   into one scheduler group (``Stats.max_batch >= 2``);
4. **evict**    — requesting more distinct instance sizes than fit forces an LRU
   eviction (``Stats.evictions >= 1``).

Exits 0 iff batching and eviction both pass; prints a PROFILE block. Run under
``brain serve --dbus`` on a session bus — see ``scheduler.bats``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Dogfood the shipped D-Bus client + image helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "dbus"))
from brain_dbus_client import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_image import draw_boxes, save_ppm  # noqa: E402

OUT = Path(os.environ.get("OUT", "/tmp/brain_e2e"))
SIZE = int(os.environ.get("SIZE", "256"))
STEPS = int(os.environ.get("STEPS", "8"))
BATCH_N = int(os.environ.get("BATCH_N", "4"))


@dataclass
class Profile:
    times: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        yield
        self.times[name] = round(time.perf_counter() - start, 2)


def generate_dogs(brain: BrainDBus, prof: Profile) -> bytes:
    """Stream a text2image generation; return the image bytes."""
    image: bytes | None = None
    params = {"prompt": "two dogs sitting side by side on grass, photo", "width": SIZE, "height": SIZE, "steps": STEPS, "seed": 7}
    with prof.measure("generate_s"):
        for frame, fds in brain.subscribe("z-image", "text2image", params):
            if frame["type"] == "blob" and fds:
                image = read_fd(fds[0])
            elif frame["type"] == "error":
                raise RuntimeError(frame["message"])
    if image is None:
        raise RuntimeError("z-image produced no image")
    save_ppm(OUT / "dogs.ppm", image, SIZE, SIZE)
    print(f"[generate] {SIZE}x{SIZE} in {prof.times['generate_s']}s -> {OUT / 'dogs.ppm'}")
    return image


def detect_and_annotate(brain: BrainDBus, image: bytes, prof: Profile) -> None:
    """Run yolo detection over dbus and draw the boxes into the image buffer."""
    meta = {"image": {"media": "image", "w": SIZE, "h": SIZE, "c": 3}}
    with prof.measure("detect_s"):
        out = brain.run("yolo", "detect", {"conf": 0.25}, in_fds={"image": sealed_memfd(image)}, in_meta=meta)
    detections = out.result.get("detections", [])
    labels = [f"{d['label']}({d['conf']:.2f})" for d in detections]
    print(f"[detect] {out.result.get('count', 0)} objects in {prof.times['detect_s']}s: {labels}")
    annotated = draw_boxes(image, SIZE, SIZE, [d["bbox"] for d in detections])
    save_ppm(OUT / "dogs_boxes.ppm", annotated, SIZE, SIZE)
    print(f"[detect] boxes drawn -> {OUT / 'dogs_boxes.ppm'}")


def fire_concurrent(count: int) -> None:
    """Submit `count` same-size text2image requests at once (each on its own connection)."""

    def one(i: int) -> None:
        with BrainDBus() as brain:
            brain.run("z-image", "text2image", {"prompt": f"two dogs, variation {i}", "width": SIZE, "height": SIZE, "steps": STEPS, "seed": 100 + i})

    threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prof = Profile()
    with BrainDBus() as brain:
        models = brain.models()
        print("models:", models)
        if "z-image" not in models:
            print("FATAL: z-image not served (set BRAIN_ZIMAGE_*)", file=sys.stderr)
            return 2

        image = generate_dogs(brain, prof)

        if "yolo" in models:
            detect_and_annotate(brain, image, prof)
        else:
            print("[detect] SKIPPED — no yolo model (set BRAIN_YOLO to a brain YOLOv8 checkpoint)")

        # batching: concurrent same-size requests should coalesce into one group.
        with prof.measure("batch_wall_s"):
            fire_concurrent(BATCH_N)
        after = brain.stats()
        print(f"[batch] {BATCH_N} concurrent requests in {prof.times['batch_wall_s']}s; "
              f"max_batch={after['max_batch']} batches={after['batches']} builds={after['builds']}")

        # eviction: more distinct instance sizes than fit → an LRU swap.
        with prof.measure("evict_builds_s"):
            for side in (SIZE + 32, SIZE + 64):
                brain.run("z-image", "text2image", {"prompt": "two dogs", "width": side, "height": side, "steps": STEPS, "seed": 5})
        final = brain.stats()
        print(f"[evict] +2 sizes in {prof.times['evict_builds_s']}s; "
              f"evictions={final['evictions']} builds={final['builds']} resident={final['resident']}")

    print("\n=== PROFILE ===")
    for name, seconds in prof.times.items():
        print(f"  {name:16} {seconds}s")
    print("=== FINAL STATS ===\n ", json.dumps(final))

    ok_batch = after["max_batch"] >= 2
    ok_evict = final["evictions"] >= 1
    print(f"\nVALIDATION: batching={'PASS' if ok_batch else 'FAIL'} (max_batch={after['max_batch']}) | "
          f"eviction={'PASS' if ok_evict else 'FAIL'} (evictions={final['evictions']})")
    return 0 if ok_batch and ok_evict else 1


if __name__ == "__main__":
    sys.exit(main())
