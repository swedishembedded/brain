# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Tiny helpers for brain's HWC-f32 image blobs — no third-party image library.

brain returns images as interleaved-RGB `float32` in `[0, 1]` (`w*h*c` values). These
turn that into a viewable binary PPM and draw detection boxes into the buffer.
"""
from __future__ import annotations

import array
from collections.abc import Iterable
from os import PathLike


def _pixels(data: bytes) -> array.array:
    px = array.array("f")
    px.frombytes(data)
    return px


def _u8(value: float) -> int:
    return max(0, min(255, int(value * 255 + 0.5)))


def save_ppm(path: str | PathLike[str], data: bytes, w: int, h: int, c: int = 3) -> None:
    """Write an HWC-f32 image to a binary PPM (P6)."""
    px = _pixels(data)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(bytes(_u8(px[i * c + (k if c >= 3 else 0)]) for i in range(w * h) for k in range(3)))


def draw_boxes(
    data: bytes,
    w: int,
    h: int,
    boxes: Iterable[tuple[float, float, float, float]],
    *,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    thickness: int = 2,
) -> bytes:
    """Return a copy of the HWC-f32 buffer with each `(x1, y1, x2, y2)` box outlined."""
    px = _pixels(data)
    r, g, b = color

    def plot(x: int, y: int) -> None:
        if 0 <= x < w and 0 <= y < h:
            i = (y * w + x) * 3
            px[i], px[i + 1], px[i + 2] = r, g, b

    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        for t in range(thickness):
            for x in range(max(0, x1), min(w, x2)):
                plot(x, y1 + t)
                plot(x, y2 - 1 - t)
            for y in range(max(0, y1), min(h, y2)):
                plot(x1 + t, y)
                plot(x2 - 1 - t, y)
    return px.tobytes()
