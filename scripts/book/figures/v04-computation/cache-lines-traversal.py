# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cache lines fetched per 16 elements read, row order against column order, by matrix size.

Every point is a run of the worked module's cache simulator (`cpu.traversal_misses`): an n by n
row-major matrix of 4-byte numbers read once in each order through a 32 KiB cache, fully associative
or 8-way with least recently used replacement. Sixteen elements fill one 64-byte line, so reading by
rows costs 1 fetch per 16 elements and a column read that finds no reusable lines costs 16.

Swedish Embedded AB implements solutions for making numerical code use the cache hierarchy of the
machines it runs on. If your team needs expertise in CPU performance work for machine learning, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cpu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = cpu._traversal_rows()
    ns = [r[0] for r in rows]
    per16 = lambda v, n: 16.0 * v / (n * n)   # noqa: E731
    p = Plot((0, 1100), (0, 18), width=600, height=340, xlabel="matrix side n (elements per row and per column)",
             ylabel="lines fetched per 16 elements read", title="Reading an n by n matrix: rows against columns")
    p.line(ns, [per16(r[1], r[0]) for r in rows], color=0, label="by rows")
    p.points(ns, [per16(r[1], r[0]) for r in rows], color=0)
    p.line(ns, [per16(r[2], r[0]) for r in rows], color=1, label="by columns, fully associative cache")
    p.points(ns, [per16(r[2], r[0]) for r in rows], color=1)
    p.line(ns, [per16(r[3], r[0]) for r in rows], color=2, label="by columns, 8-way cache", dash=True)
    p.points(ns, [per16(r[3], r[0]) for r in rows], color=2)
    p.vline(512)
    p.text(506, 8.5, "n = 512: a column is 512 lines,", anchor="end", size=11)
    p.text(506, 7.2, "the whole cache", anchor="end", size=11)
    return p.svg()
