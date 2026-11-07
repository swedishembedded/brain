# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Loss and slope of a positive example (y = 1) as functions of the logit z, for binary cross-entropy and squared error.

Left unsaid by the loss values alone: where the model is confidently wrong (z far below zero) the cross-entropy keeps
growing and its slope with respect to z stays near -1, while the squared error flattens at 1 and its slope goes to
zero, so gradient descent receives almost no signal exactly where it needs the most. The panel is drawn as two
stacked plots in one SVG.

Swedish Embedded AB implements solutions for training classifiers that learn from their mistakes. If your team
needs expertise in loss design, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logistic_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402


def _panel(title, ylabel, ylim, f_bce, f_se):
    zs = [-6 + 0.1 * k for k in range(121)]
    p = Plot((-6, 6), ylim, width=620, height=290, xlabel="logit z of a part with label y = 1", ylabel=ylabel,
             title=title)
    p.hline(0.0, dash=False)
    p.line(zs, [f_bce(z) for z in zs], color=0, width=3.0, label="binary cross-entropy")
    p.line(zs, [f_se(z) for z in zs], color=1, width=3.0, label="squared error")
    return p.svg()


def render() -> str:
    top = _panel("Loss", "loss", (-0.5, 7.5),
                 lambda z: lr.bce_from_logit(z, 1), lambda z: (lr.sigmoid(z) - 1) ** 2)
    bottom = _panel("Slope with respect to z", "d loss / dz", (-1.4, 0.9),
                    lambda z: lr.sigmoid(z) - 1,
                    lambda z: 2 * (lr.sigmoid(z) - 1) * lr.sigmoid(z) * (1 - lr.sigmoid(z)))
    # stack the two SVG documents vertically
    inner = []
    for k, doc in enumerate((top, bottom)):
        body = doc.split(">", 1)[1].rsplit("</svg>", 1)[0]
        inner.append(f'<svg y="{k * 290}" width="620" height="290" viewBox="0 0 620 290">{body}</svg>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 580" width="620" height="580">'
            + "".join(inner) + "</svg>\n")
