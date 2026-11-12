# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loop of the project: a panel with an event process is cut into blocks, forecasters with heads are fitted and judged on the
# test block, repaired on the calibration block, and turned into orders and alerts that are scored by the cost they incurred. The
# three planted mistakes are marked where they enter.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty and are judged honestly. If your team needs
# expertise in calibrated forecasting systems, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import RED, Diagram, text_width  # noqa: E402


def render() -> str:
    top = [
        (["40 series of 300 bars:", "season, regime change,", "stock-outs"], "plain"),
        (["blocks: fit, calibration,", "test, then deployment;", "purge and embargo"], "model"),
        (["naive, seasonal naive,", "seasonal AR(1) and a", "pooled ridge map, each", "with a Gaussian head"], "model"),
        (["scaled quantile loss,", "skill and block", "bootstrap, coverage,", "random-walk control"], "result"),
    ]
    bottom = [
        (["quantile levels repaired", "by the probability integral", "transform; isotonic fit of", "the event probability"], "accent"),
        (["orders at the 0.9 quantile,", "alerts at the cost", "threshold 1 / (1 + 5)"], "model"),
        (["cost actually incurred", "in the test block and", "in the deployment block"], "result"),
    ]
    gap, pad, h = 40, 24, 86
    wt = [int(text_width(t[0]) + 16) for t in top]
    wb = [int(text_width(b[0]) + 16) for b in bottom]
    widths = [max(wt[0], wb[0] if len(wb) > 0 else 0), max(wt[1], wb[1]), max(wt[2], wb[2]), wt[3]]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y_top, y_bot = pad + h / 2, pad + h + 96 + h / 2
    d = Diagram(width, int(y_bot + h / 2 + pad + 30))
    tb = [d.box(xs[i], y_top, widths[i], h, top[i][0], top[i][1]) for i in range(4)]
    bb = [d.box(xs[i], y_bot, widths[i], h, bottom[i][0], bottom[i][1]) for i in range(3)]
    for i in range(3):
        d.arrow(tb[i]["r"], tb[i + 1]["l"])
        d.arrow(bb[i]["r"], bb[i + 1]["l"]) if i < 2 else None
    mid = (y_top + h / 2 + y_bot - h / 2) / 2
    d.path([tb[3]["b"], (xs[3], mid), (xs[0], mid), bb[0]["t"]])
    d.text((xs[0] + xs[3]) / 2, mid - 12, ["the forecasts of the test block"], size=12, color="#4a5568")
    d.text(xs[1], y_top + h / 2 + 16, ["mistake 1: a random split"], size=12, color=RED)
    d.text(xs[2], y_top + h / 2 + 16, ["mistake 2: a scaler that sees the future"], size=12, color=RED)
    d.text(xs[0], y_bot + h / 2 + 16, ["mistake 3: fitted on the test block"], size=12, color=RED)
    return d.svg()
