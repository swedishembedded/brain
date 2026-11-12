# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The closed loop of the project: twelve observed frames and their tracks become cameras, points, a floor and ball detections;
# the Gaussians are fitted through the exported cameras; a dynamics model of each ball is run free for six frames and the
# result is rendered and scored against the held-out frames. The three planted mistakes are marked where they enter.
#
# Swedish Embedded AB implements solutions for reconstructing a scene from footage and predicting how it moves. If your team needs
# expertise in structure from motion, Gaussian splatting and learned world models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import RED, Diagram, text_width  # noqa: E402


def render() -> str:
    top = [
        (["12 observed frames,", "43 feature tracks"], "plain"),
        (["structure from motion:", "cameras, 40 points,", "floor, ball detections"], "model"),
        (["export the cameras", "to the renderer"], "model"),
        (["fit 10 + 3 Gaussians", "to the 12 frames"], "accent"),
    ]
    bottom = [
        (["dynamics of each ball:", "p' = c + a p + b q"], "model"),
        (["free run for 6 frames,", "clipped to the clamp box"], "model"),
        (["render the Gaussians at", "each held-out camera"], "model"),
        (["squared error against", "the held-out frame"], "result"),
    ]
    gap, pad, h = 40, 24, 74
    widths = [int(max(text_width(t[0]), text_width(b[0])) + 8 + 8) for t, b in zip(top, bottom)]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y_top, y_bot = pad + h / 2, pad + h + 96 + h / 2
    d = Diagram(width, int(y_bot + h / 2 + pad + 30))
    tb = [d.box(xs[i], y_top, widths[i], h, top[i][0], top[i][1]) for i in range(4)]
    bb = [d.box(xs[i], y_bot, widths[i], h, bottom[i][0], bottom[i][1]) for i in range(4)]
    for i in range(3):
        d.arrow(tb[i]["r"], tb[i + 1]["l"])
        d.arrow(bb[i]["r"], bb[i + 1]["l"])
    mid = (y_top + h / 2 + y_bot - h / 2) / 2
    d.path([tb[3]["b"], (xs[3], mid), (xs[0], mid), bb[0]["t"]])
    d.text((xs[0] + xs[3]) / 2, mid - 12, ["the ball positions of the 12 frames"], size=12, color="#4a5568")
    d.text(xs[2], y_top + h / 2 + 16, ["mistakes 1 and 2"], size=12, color=RED)
    d.text(xs[1], y_bot + h / 2 + 16, ["mistake 3"], size=12, color=RED)
    return d.svg()
