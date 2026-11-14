# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One step of the tiled load of the chapter "Loading Models": a tile of packed bytes and its scales is read from the mapped file, held
# in a leased staging buffer, optionally repacked or converted, written to the sink, and its file pages are released. The tile length
# is the worked module's plan for the bf16 to nf4 load of a 256 x 32768 tensor under a 128 KiB tile cap.
#
# Swedish Embedded AB implements solutions for loading multi-gigabyte model files quickly and within a fixed memory budget for its
# clients. If your team needs expertise in memory-bounded model loading then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import loading as ld  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402

W, H = 1000, 330


def render() -> str:
    tile, tiles, staged = ld.plan_tiles([256, 32768], "bf16", "nf4 block 32", "Convert", 128 * ld.KIB, 512 * ld.KIB)
    d = Diagram(W, H)
    fil = d.box(110, 150, 170, 80, ["mapped file", "bf16 tensor,", "16 MiB"], "plain")
    lease = d.box(330, 54, 230, 56, ["budget: one lease for the", "whole load, sized for one tile"], "result")
    read = d.box(330, 150, 190, 80, ["read tile k", f"{tile:,} elements", f"of {tiles} tiles"], "accent")
    conv = d.box(580, 150, 170, 80, ["convert", "decode to f32,", "encode to nf4"], "model")
    sink = d.box(810, 150, 170, 80, ["sink", "write tile k,", "finish or abort"], "plain")
    drop = d.box(810, 260, 190, 56, ["release tile k's", "file pages"], "plain")
    d.arrow(fil["r"], read["l"])
    d.arrow(lease["b"], read["t"], f"{staged:,} bytes", side="right")
    d.arrow(read["r"], conv["l"])
    d.arrow(conv["r"], sink["l"])
    d.arrow(sink["b"], drop["t"])
    d.path([drop["l"], (110, 260), fil["b"]], color=INK)
    d.text(470, 300, ["tile k + 1 reuses the same buffers; the lease is returned when the load ends, on failure too"], size=12, color=MUTED)
    return d.svg()
