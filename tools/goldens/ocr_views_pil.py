# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements document-OCR pipelines that read pages the way
# their models were trained to. If your team needs expertise in reproducing a
# vision model's reference preprocessing exactly then you can procure our
# services by sending an email to info@swedishembedded.com.

"""Pillow goldens for crates/imaging/src/views.rs.

Prints the FNV-1a 64 hashes the `views_match_pillow_byte_for_byte` test pins:
ImageOps.pad for a landscape image, and a Pillow mosaic (both in the regime where
Pillow resamples horizontally first, which is the order the Rust code uses). The test
image is the same deterministic LCG field `image(w, h)` builds in Rust.

    python3 -I tools/goldens/ocr_views_pil.py     # needs Pillow
"""
from PIL import Image, ImageOps


def image(w, h):
    s = 12345
    px = bytearray()
    for i in range(w * h * 3):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        x, y = (i // 3) % w, (i // 3) // w
        px.append(((x * 255 // w + y * 255 // h + (i % 3) * 40) // 2 + (s >> 28)) & 0xFF)
    return Image.frombytes("RGB", (w, h), bytes(px))


def fnv(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    print("PAD_LANDSCAPE", fnv(ImageOps.pad(image(173, 101), (64, 64), color=(127, 127, 127)).tobytes()))
    big = image(150, 90).resize((96, 64), Image.Resampling.BICUBIC)
    tiles = [big.crop((c * 32, r * 32, c * 32 + 32, r * 32 + 32)) for r in range(2) for c in range(3)]
    print("MOSAIC", [fnv(t.tobytes()) for t in tiles])


if __name__ == "__main__":
    main()
