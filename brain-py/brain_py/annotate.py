# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Draw detection boxes + class labels onto a PIL image."""

from __future__ import annotations

from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

# A small, distinct palette for class colors (cycled by class id).
_PALETTE = [
    (255, 64, 64),
    (64, 220, 64),
    (80, 120, 255),
    (240, 210, 60),
    (230, 90, 230),
    (60, 220, 220),
]


def _font() -> Optional[ImageFont.ImageFont]:
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover
        return None


def annotate(image: Image.Image, detections: Iterable, width: int = 2) -> Image.Image:
    """Return a copy of ``image`` with each detection drawn as a labeled box.

    ``detections`` is an iterable of :class:`brain_py.client.Detection` (or any
    object exposing ``x1,y1,x2,y2,conf,cls,label``).
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()
    for d in detections:
        color = _PALETTE[int(d.cls) % len(_PALETTE)]
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{d.label or d.cls} {d.conf:.2f}"
        # Label background for readability.
        if font is not None:
            try:
                tb = draw.textbbox((0, 0), text, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = 8 * len(text), 11
        else:
            tw, th = 8 * len(text), 11
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        draw.text((x1 + 2, ty + 1), text, fill=(0, 0, 0), font=font)
    return out
