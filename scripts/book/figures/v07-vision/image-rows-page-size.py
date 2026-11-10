# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Rows of the image block against the height of a portrait page (`ocr.rows_series`), for the first generation (tiles of 640 pixels, at most
# 9), the second (tiles of 768 pixels, at most 6) and the dynamic resolution of the previous chapter at the served limit.
#
# Swedish Embedded AB implements solutions for document-reading models whose token budget is counted before it is paid. If your team
# needs expertise in token budgets for document OCR, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ocr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    h, v1, v2, dyn = ocr.rows_series()
    p = Plot((300, 4000), (40, 1500), width=640, height=380, xlabel="height of a portrait page of aspect 1 : sqrt 2 (pixels)",
             ylabel="rows in the decoder's sequence", title="Rows of the image block against page size", xlog=True, ylog=True)
    for i, (label, ys) in enumerate((("first generation (640 px tiles, at most 9)", v1), ("second generation (768 px tiles, at most 6)", v2),
                                     ("dynamic resolution, 1 Mpx limit (previous chapter)", dyn))):
        p.line(h, ys, color=i, label=label)
    return p.svg()
