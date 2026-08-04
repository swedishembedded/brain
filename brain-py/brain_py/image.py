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


def to_pil(data: bytes, w: int, h: int, c: int = 3):
    """Convert an HWC-f32 blob to a PIL ``Image`` (RGB). Requires pillow."""
    from PIL import Image

    px = _pixels(data)
    buf = bytes(_u8(px[i * c + (k if c >= 3 else 0)]) for i in range(w * h) for k in range(3))
    return Image.frombytes("RGB", (w, h), buf)


def _u8(value: float) -> int:
    return max(0, min(255, int(value * 255 + 0.5)))


def load_ppm(path: str | PathLike[str]) -> tuple[bytes, int, int]:
    """Read a binary PPM (P6) into `(hwc_f32_bytes, w, h)` — the inverse of
    :func:`save_ppm`, and the format every example uses to send an image INTO
    brain (`in_meta = {"media": "image", "w": w, "h": h, "c": 3}`).

    One implementation, here, so no example grows its own P6 parser.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"P6"):
        raise ValueError(f"{path}: not a binary PPM (P6)")
    # header: P6, width, height, maxval — whitespace-separated, '#' comments.
    fields, i = [], 2
    while len(fields) < 3:
        while i < len(raw) and raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while i < len(raw) and raw[i : i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j : j + 1].isspace():
            j += 1
        fields.append(int(raw[i:j]))
        i = j
    w, h, maxval = fields
    if maxval != 255:
        raise ValueError(f"{path}: only 8-bit PPMs are supported (maxval {maxval})")
    px = raw[i + 1 : i + 1 + w * h * 3]
    if len(px) != w * h * 3:
        raise ValueError(f"{path}: truncated ({len(px)} of {w * h * 3} bytes)")
    out = array.array("f", [b / 255.0 for b in px])
    return out.tobytes(), w, h


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
