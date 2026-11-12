# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The tiled splat renderer and its backward pass as two rows of stages: forward (Gaussians to image) above, backward (image loss to
# the gradients of the Gaussians) below, each backward stage under the forward stage whose work it reuses.
#
# Swedish Embedded AB implements differentiable splat renderers whose forward and backward passes share one sorted work list. If your
# team needs expertise in Gaussian splatting, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import RED, Diagram, text_width  # noqa: E402


def render() -> str:
    cols = [
        (["Gaussians", "14 numbers each"], "plain", ["gradients", "14 numbers each"], "plain"),
        (["project", "mean, conic,", "depth, radius"], "model", ["projection", "backward"], "model"),
        (["count, scan,", "emit keys", "(tile | depth)"], "model", ["sum each Gaussian's", "instances"], "model"),
        (["radix sort,", "tile ranges"], "model", ["reduce over the", "tile's pixels"], "model"),
        (["composite", "per pixel, front", "to back"], "model", ["walk each pixel", "again, in reverse"], "model"),
        (["image"], "result", ["image loss,", "dL / d pixel"], "result"),
    ]
    gap, pad, h = 40, 24, 74
    widths = [int(max(text_width(c[0]), text_width(c[2])) + 8 + 8) for c in cols]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y_top, y_bot = pad + h / 2, pad + h + 56 + h / 2
    d = Diagram(width, int(y_bot + h / 2 + pad))
    top, bot = [], []
    for i, (fl, fk, bl, bk) in enumerate(cols):
        top.append(d.box(xs[i], y_top, widths[i], h, fl, fk))
        bot.append(d.box(xs[i], y_bot, widths[i], h, bl, bk))
    for i in range(len(cols) - 1):
        d.arrow(top[i]["r"], top[i + 1]["l"])
        d.arrow(bot[i + 1]["l"], bot[i]["r"], color=RED)
    d.arrow(top[-1]["b"], bot[-1]["t"], color=RED, label="loss", side="right")
    return d.svg()
