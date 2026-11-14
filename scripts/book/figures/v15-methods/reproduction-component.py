# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The component of the project "Reproduce, Verify, Optimize, Defend": a pre-norm gated feed-forward block, with the six
# taps the golden stores (n, gate, up, h, f, y) as ovals between the operations. The shapes are the worked module's
# constants for the printable component (D = 6, F = 10).
#
# Swedish Embedded AB implements solutions for reproducing published model components and proving them correct. If your
# team needs expertise in porting and layer-by-layer parity then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_reproduction as pr  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 980, 300
TOP, MID, BOT = 70, 150, 230


def render() -> str:
    D, F = pr.D0, pr.F0
    d = Diagram(W, H)
    x = d.oval(60, MID, 80, 40, ["x", f"[{D}]"], "plain")
    norm = d.box(175, MID, 110, 56, ["RMS norm", "gain g"], "model")
    n = d.oval(290, MID, 76, 40, ["n", f"[{D}]"], "result")
    wg = d.box(405, TOP, 110, 48, ["W_gate", f"{F} x {D}"], "model")
    wu = d.box(405, BOT, 110, 48, ["W_up", f"{F} x {D}"], "model")
    gate = d.oval(520, TOP, 76, 40, ["gate", f"[{F}]"], "result")
    up = d.oval(520, BOT, 76, 40, ["up", f"[{F}]"], "result")
    mix = d.box(620, MID, 100, 56, ["SiLU(gate)", "times up"], "model")
    h = d.oval(722, MID, 64, 40, ["h", f"[{F}]"], "result")
    wd = d.box(818, MID, 84, 56, ["W_down", f"{D} x {F}"], "model")
    f = d.oval(818, BOT + 30, 64, 40, ["f", f"[{D}]"], "result")
    add = d.box(920, BOT + 30, 60, 40, ["+ x"], "model")
    y = d.oval(920, TOP, 64, 40, ["y", f"[{D}]"], "result")
    d.arrow(x["r"], norm["l"])
    d.arrow(norm["r"], n["l"])
    d.path([n["t"], (290, TOP), wg["l"]])
    d.path([n["b"], (290, BOT), wu["l"]])
    d.arrow(wg["r"], gate["l"])
    d.arrow(wu["r"], up["l"])
    d.path([gate["r"], (620, TOP), mix["t"]])
    d.path([up["r"], (620, BOT), mix["b"]])
    d.arrow(mix["r"], h["l"])
    d.arrow(h["r"], wd["l"])
    d.arrow(wd["b"], f["t"])
    d.arrow(f["r"], add["l"])
    d.arrow(add["t"], y["b"])
    d.path([x["b"], (60, 288), (920, 288), add["b"]], color="#4a5568", dash="5 3")
    d.text(470, 276, ["the residual: x is added to f"], size=12, color="#4a5568")
    d.text(160, 24, ["shaded ovals: the six taps of the golden"], size=12, color="#4a5568")
    return d.svg()
