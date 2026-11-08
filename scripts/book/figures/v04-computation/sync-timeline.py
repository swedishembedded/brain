# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The host and the device on one time axis for one dispatch (the addition of 2^20 floats on the model unit of the chapter's
# worked module): without a wait the clock stops 5 microseconds in, while the kernel is still running; with a wait it stops
# after the device is done and the host has noticed. The widths are to scale (9 pixels per microsecond) and the numbers are the
# worked module's `timeline_numbers`, the noise-free times of the model.
#
# Swedish Embedded AB implements solutions for timing GPU work so that the number read is the number the device spent. If your
# team needs expertise in benchmarking and profiling of machine learning kernels, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import profiling as pr  # noqa: E402
from svgdiagram import Diagram, MUTED  # noqa: E402

W, H = 800, 452
X0, SCALE = 118, 9.0          # pixel of time 0, pixels per microsecond
PANEL_X, PANEL_W, PANEL_H = 20, W - 40, 204
BOX_H = 38


def _x(t):
    return X0 + SCALE * t


def _span(d, t0, t1, cy, lines, kind):
    """A box covering the time span [t0, t1] on the lane at height cy."""
    return d.box((_x(t0) + _x(t1)) / 2, cy, _x(t1) - _x(t0), BOX_H, lines, kind)


def _panel(d, top, title, host_cy, dev_cy):
    d.panel(PANEL_X, top, PANEL_W, PANEL_H, title)
    d.text(66, host_cy, ["host"], weight="bold")
    d.text(66, dev_cy, ["device"], weight="bold")


def render() -> str:
    n = pr.timeline_numbers()
    enq, dev, wake = n["enq"], n["dev"], n["wake"]
    t_start, t_end = enq, enq + dev
    clock_sync = t_end + wake
    d = Diagram(W, H)

    # Panel 1: the clock stops when the call returns.
    top = 14
    host, devc = top + 84, top + 150
    _panel(d, top, "Clock stopped when the call returns", host, devc)
    _span(d, 0, enq, host, [], "model")
    d.text(_x(enq / 2), host - 31, [f"enqueue {enq:.1f}"], size=12, color=MUTED)
    _span(d, t_start, t_end, devc, ["kernel runs", f"{dev:.1f} microseconds"], "accent")
    d.line((_x(enq), host - BOX_H / 2 - 4), (_x(enq), devc + BOX_H / 2 + 8), dash="5 3")
    d.text(_x(enq) + 12, host, [f"the clock stops here and reads {enq:.1f} microseconds,", "while the kernel has only just started"],
           size=12, anchor="start")

    # Panel 2: the clock stops after the wait.
    top = 14 + PANEL_H + 16
    host, devc = top + 84, top + 150
    _panel(d, top, "Clock stopped after waiting for the device", host, devc)
    _span(d, 0, enq, host, [], "model")
    d.text(_x(enq / 2), host - 31, [f"enqueue {enq:.1f}"], size=12, color=MUTED)
    _span(d, t_start, t_end, host, ["blocked until", "the device is done"], "plain")
    _span(d, t_end, clock_sync, host, ["the wait itself", f"{wake:.1f} microseconds"], "result")
    _span(d, t_start, t_end, devc, ["kernel runs", f"{dev:.1f} microseconds"], "accent")
    d.text(_x(t_start + dev / 2), devc + BOX_H / 2 + 16, [f"device stamps read {dev:.1f}"], size=12, color=MUTED)
    d.line((_x(clock_sync), host - BOX_H / 2 - 4), (_x(clock_sync), devc + BOX_H / 2 + 8), dash="5 3")
    d.text(_x(clock_sync) - 8, host - 31, [f"the clock stops here and reads {clock_sync:.1f} microseconds"], size=12, anchor="end")
    return d.svg()
