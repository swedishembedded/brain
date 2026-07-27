#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Example client for brain's D-Bus control surface (com.swedishembedded.Brain1).

Discovers the served models, gets a real image back over a file descriptor, and
streams a z-image generation — using the reusable client in ``brain_dbus_client``.
Run under a private session bus so it needs no system config:

    dbus-run-session -- bash -c 'brain serve --dbus & sleep 1; python3 examples/dbus/brain_dbus.py'

With brain's z-image weights exported (``BRAIN_ZIMAGE_*``) it also generates an image
over D-Bus; otherwise it runs the no-GPU ``imageops`` path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Use the reusable client from the brain-py package (run straight from the repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd  # noqa: E402
from brain_py.image import save_ppm  # noqa: E402

OUT = Path(os.environ.get("OUT", "/tmp"))


def demo_image_over_fd(brain: BrainDBus) -> None:
    """imageops.gradient → a real image returned as a memfd → PPM."""
    out = brain.run("imageops", "gradient", {"width": 128, "height": 128, "style": "aurora"})
    meta = out.meta["image"]
    data = read_fd(out.fds["image"])
    dims = meta.get("meta") or {}
    w, h, c = dims.get("w", 128), dims.get("h", 128), dims.get("c", 3)
    path = OUT / "brain_dbus_gradient.ppm"
    save_ppm(path, data, w, h, c)
    print(f"imageops.gradient -> {len(data)} bytes ({meta['transport']}, {w}x{h}x{c}) -> {path}")


def demo_streaming_generation(brain: BrainDBus) -> None:
    """z-image text2image over Subscribe: progress frames + the image via SCM_RIGHTS."""
    print("z-image text2image (streaming over dbus)...")
    params = {"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 8}
    for frame, fds in brain.subscribe("z-image", "text2image", params):
        kind = frame["type"]
        if kind == "progress":
            print(f"  [{frame['step']}/{frame['total']}] {frame['message']}")
        elif kind == "blob" and fds:
            data = read_fd(fds[0])
            path = OUT / "brain_dbus_image.ppm"
            save_ppm(path, data, 256, 256)
            print(f"  saved image -> {path}")
        elif kind == "done":
            print("  done:", frame.get("result"))
        elif kind == "error":
            print("  error:", frame["message"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with BrainDBus() as brain:
        models = brain.models()
        print("models:", models)

        if "imageops" in models:
            demo_image_over_fd(brain)

        if "z-image" in models and os.environ.get("BRAIN_ZIMAGE_DIT"):
            demo_streaming_generation(brain)
        else:
            print("z-image streaming demo skipped (export BRAIN_ZIMAGE_* to enable)")

        print("scheduler stats:", brain.stats())
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
