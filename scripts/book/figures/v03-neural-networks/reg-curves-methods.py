# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Mean validation loss over the 8 seeds, without regularization and with weight decay, dropout and noise augmentation.

Computed by worked module `regularization_nn` (the settings of the comparison table; each curve is the mean over the
8 seeds of the loss recorded every 10 steps). The vertical axis is cut at 0.5.

Swedish Embedded AB implements solutions for models that keep working on data they have not seen. If your team needs
expertise in regularizing neural networks, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgplot import Plot  # noqa: E402


def mean_curve(runs, column):
    n = len(runs)
    return [sum(r["history"][i][column] for r in runs) / n for i in range(len(runs[0]["history"]))]


def render() -> str:
    wd, dp, ag = rg._chosen()
    data = {name: runs for name, _, runs in rg.comparison_data()}
    spec = [("none", "no regularization", 0), ("weight decay", f"weight decay, lambda {wd:g}", 2),
            ("dropout", f"dropout, p {dp:g}", 1), ("augmentation", f"input noise, sd {ag:g}", 4)]
    steps = [row[0] for row in data["none"][0]["history"]]
    p = Plot((0, rg.STEPS), (0.08, 0.5), width=640, height=380, xlabel="gradient step", ylabel="validation loss (mean of 8 seeds)",
             title="Validation loss with each method")
    p.hline(rg.NOISE_SD ** 2)
    for name, label, color in spec:
        p.line(steps, [min(0.5, v) for v in mean_curve(data[name], 2)], color=color, label=label)
    return p.svg()
