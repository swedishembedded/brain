# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path of one dispatch, host to device and back, for y = scale x + shift over n = 10 numbers: the host buffers x and
# the parameter block go up over the bus, the kernel runs over a grid of ceil(n / 4) workgroups on the device, and y
# comes back, a step that waits for the device. The counts (bytes, workgroups) come from the worked module's
# `dispatch_figure_numbers`, so the picture and the tables of the chapter cannot disagree.
#
# Swedish Embedded AB implements solutions for moving data to a GPU and back without waiting on it more than needed. If
# your team needs expertise in GPU host programs for machine learning, you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import compute_kernels as ck  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 780, 388
X_COLS = (130, 340, 650)          # columns: x / uniform block, params, y
BOX_W, BOX_H = 190, 50


def render() -> str:
    f = ck.dispatch_figure_numbers()
    d = Diagram(W, H)
    d.panel(20, 14, W - 40, 100, "Host memory")
    d.panel(20, 150, W - 40, 226, "Device memory and compute")
    hx = d.box(X_COLS[0], 76, BOX_W, BOX_H, [f"x: {f['n']} f32", f"{f['vec_bytes']} bytes"], "model")
    hp = d.box(X_COLS[1], 76, BOX_W, BOX_H, ["scale, shift, n", f"padded to {f['params_bytes']} bytes"], "model")
    hy = d.box(X_COLS[2], 76, BOX_W, BOX_H, [f"y: {f['n']} f32", f"{f['vec_bytes']} bytes"], "result")
    dx = d.box(X_COLS[0], 214, BOX_W, BOX_H, ["storage buffer x", "read-only"], "model")
    dp = d.box(X_COLS[1], 214, BOX_W, BOX_H, ["uniform buffer", "scale, shift, n"], "model")
    dy = d.box(X_COLS[2], 214, BOX_W, BOX_H, ["storage buffer y", "read-write"], "result")
    kx = 235
    k = d.box(kx, 330, 330, 56, [f"kernel over a grid of {f['groups']} workgroups", f"of {f['wg']} invocations (N = {f['n']}, "
                                 f"{f['groups'] * f['wg']} launched)"], "accent")
    d.arrow(hx["b"], dx["t"], f"upload {f['vec_bytes']} B", side="right")
    d.arrow(hp["b"], dp["t"], f"upload {f['params_bytes']} B", side="right")
    d.arrow(dy["t"], hy["b"], ["read back", f"{f['vec_bytes']} B", "waits"], side="right")
    d.arrow(dx["b"], (dx["b"][0], k["t"][1]), None)
    d.arrow(dp["b"], (dp["b"][0], k["t"][1]), None)
    d.path([k["r"], (dy["b"][0], k["r"][1]), dy["b"]], radius=8)
    d.text((k["r"][0] + dy["b"][0]) / 2, k["r"][1] - 13, ["writes y"], size=12, color="#4a5568")
    return d.svg()
